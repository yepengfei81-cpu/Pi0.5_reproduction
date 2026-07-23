#!/usr/bin/env python3
"""pack v2: 零转码 + 增量打包(遥操作/双臂 -> 双臂 20D 统一 LeRobot 数据集)。

与 pack_lerobot.py 的区别(为什么快 ~20x):
  - 视频【字节级复制】源 mp4(源与目标编码参数完全一致: h264/640x480/yuv420p/10fps),
    不再"解码->重编码"。画质=采集原始一代压缩, 严格优于旧脚本的二代压缩。
  - 状态转换复用 pack_lerobot 的同一套数学(_arm_state_action/REST10), 逐元素一致。
  - manifest 增量: pack_manifest.json 记录每条源 episode 的打包去向, --append 只处理新增。
  - 单臂样本的 wrist_image_1 黑视频用 ffmpeg lavfi 生成(按长度缓存复用), 不逐帧编码。

限制: 只支持双臂 20D schema(现行主线); UMI/单臂10D 旧格式请继续用 pack_lerobot.py。

用法:
  # 全量(格式模板默认取现有打包集 cotrain_dualarm2 的 schema)
  python umi/pack_lerobot_fast.py --teleop-dir ... --teleop-gripper ... \
      --dualarm-dir ... --dualarm-gripper get parallel \
      --repo-id cotrain_dualarm2 -o /home/ypf/pi_data/cotrain_dualarm2_fast --verify
  # 增量追加(新采的数据目录直接加进已有打包集, 秒级/条)
  python umi/pack_lerobot_fast.py --append ... -o <已有打包集>
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pack_lerobot import (  # noqa: E402  复用同一套数学/工具, 保证与旧打包逐元素一致
    REST10, _arm_state_action, load_source_tasks, verify_dataset,
    make_server_compatible, OUT_W, OUT_H, TARGET_FPS,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gripper_geom"))
from gripper_params import get_params  # noqa: E402

CHUNK = 1000                     # LeRobot chunks_size(与模板一致)
VIDEO_KEYS = ("image", "wrist_image", "wrist_image_1")
N_IMG_STAT_FRAMES = 10           # 图像 stats 采样帧数(仅元数据, 训练不使用图像stats)


# ---------------------------------------------------------------------------
# 源 episode -> 20D 数值(不碰视频像素)
# ---------------------------------------------------------------------------
def teleop_rows(pqf, gp):
    """单臂遥操作 parquet -> (state20, actions20, arm1_mask=0)。数学同 process_teleop_episode。"""
    import pyarrow.parquet as pq
    t = pq.read_table(pqf).to_pydict()
    if "state_eef" not in t:
        return None, None
    s, a = _arm_state_action(t["state_eef"], t.get("actions_eef"),
                             gp["close"], gp["open"], gp["tcp_offset"])
    n = len(s)
    rest = np.tile(REST10, (n, 1))
    state = np.concatenate([s, rest], axis=1).astype(np.float32)
    actions = np.concatenate([a, rest], axis=1).astype(np.float32)
    ti = int(t["task_index"][0]) if "task_index" in t else None
    return {"state": state, "actions": actions, "arm1": 0.0, "task_index": ti}, n


def dualarm_rows(pqf, gp0, gp1):
    """双臂 parquet -> (state20, actions20, arm1_mask=1)。数学同 process_dualarm_episode。"""
    import pyarrow.parquet as pq
    t = pq.read_table(pqf).to_pydict()
    if "state_eef_0" not in t or "state_eef_1" not in t:
        return None, None
    s0, a0 = _arm_state_action(t["state_eef_0"], t.get("actions_eef_0"),
                               gp0["close"], gp0["open"], gp0["tcp_offset"])
    s1, a1 = _arm_state_action(t["state_eef_1"], t.get("actions_eef_1"),
                               gp1["close"], gp1["open"], gp1["tcp_offset"])
    n = min(len(s0), len(s1))
    state = np.concatenate([s0[:n], s1[:n]], axis=1).astype(np.float32)
    actions = np.concatenate([a0[:n], a1[:n]], axis=1).astype(np.float32)
    ti = int(t["task_index"][0]) if "task_index" in t else None
    return {"state": state, "actions": actions, "arm1": 1.0, "task_index": ti}, n


# ---------------------------------------------------------------------------
# 视频: 复制 / 黑视频生成 / 帧数与统计
# ---------------------------------------------------------------------------
def video_nframes(mp4):
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    return n


def sample_image_stats(mp4, n_frames):
    """抽 N_IMG_STAT_FRAMES 帧算 0-1 归一化的逐通道 stats(仅元数据用)。"""
    cap = cv2.VideoCapture(str(mp4))
    idxs = np.linspace(0, max(n_frames - 1, 0), min(N_IMG_STAT_FRAMES, max(n_frames, 1))).astype(int)
    pix = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            pix.append(bgr[:, :, ::-1].astype(np.float32) / 255.0)   # -> RGB, 0-1
    cap.release()
    if not pix:
        pix = [np.zeros((OUT_H, OUT_W, 3), np.float32)]
    arr = np.stack(pix)                                   # (k,H,W,3)
    def f(v):  # (3,) -> [[ [c] ] x3] 与 LeRobot 格式一致
        return [[[float(x)]] for x in v]
    return {"min": f(arr.min((0, 1, 2))), "max": f(arr.max((0, 1, 2))),
            "mean": f(arr.mean((0, 1, 2))), "std": f(arr.std((0, 1, 2))),
            "count": [int(len(pix))]}


_Z3 = [[[0.0]], [[0.0]], [[0.0]]]   # 黑视频的逐通道零统计
_BLACK_STATS = {"min": _Z3, "max": _Z3, "mean": _Z3, "std": _Z3,
                "count": [N_IMG_STAT_FRAMES]}


def make_black_video(n, cache_dir, cache):
    """n 帧黑视频(ffmpeg lavfi, 极快); 同长度缓存复用。返回缓存文件路径。"""
    if n in cache:
        return cache[n]
    out = cache_dir / f"black_{n}.mp4"
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"color=black:s={OUT_W}x{OUT_H}:r={TARGET_FPS}",
             "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
            check=True)
    cache[n] = out
    return out


# ---------------------------------------------------------------------------
# 数值 stats(精确) + parquet 写出(schema 从模板 parquet 复制, 含 HF 元数据/Sequence 兼容)
# ---------------------------------------------------------------------------
def vec_stats(x):
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "count": [int(len(x))]}


def write_parquet(path, tmpl_schema, ep_idx, task_idx, start_index, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(rows["state"])
    cols = {
        "state": rows["state"], "actions": rows["actions"],
        "env_mask": np.ones(n, np.float32),
        "arm1_mask": np.full(n, rows["arm1"], np.float32),
        "gripper_id_0": np.full(n, rows["gid0"], np.int64),
        "gripper_id_1": np.full(n, rows["gid1"], np.int64),
        "timestamp": (np.arange(n) / TARGET_FPS).astype(np.float32),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, ep_idx, np.int64),
        "index": np.arange(start_index, start_index + n, dtype=np.int64),
        "task_index": np.full(n, task_idx, np.int64),
    }
    arrays = []
    for f in tmpl_schema:
        v = cols[f.name]
        if pa.types.is_fixed_size_list(f.type):
            flat = pa.array(np.asarray(v, np.float32).reshape(-1), type=f.type.value_type)
            arrays.append(pa.FixedSizeListArray.from_arrays(flat, f.type.list_size))
        else:
            arrays.append(pa.array(v))
    table = pa.Table.from_arrays(arrays, names=tmpl_schema.names).cast(tmpl_schema)
    pq.write_table(table, path)
    return cols


def episode_stats_row(ep_idx, cols, img_stats):
    st = {k: img_stats[k] for k in VIDEO_KEYS}
    st["state"] = vec_stats(cols["state"]); st["actions"] = vec_stats(cols["actions"])
    for k in ("env_mask", "arm1_mask", "gripper_id_0", "gripper_id_1",
              "timestamp", "frame_index", "episode_index", "index", "task_index"):
        st[k] = vec_stats(cols[k])
    return {"episode_index": ep_idx, "stats": st}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="pack v2: 零转码+增量 (双臂 20D)")
    ap.add_argument("--teleop-dir", nargs="+", default=[])
    ap.add_argument("--teleop-gripper", nargs="+", default=["parallel"])
    ap.add_argument("--teleop-task", default=None, help="覆盖 prompt; 默认读各源 tasks.jsonl")
    ap.add_argument("--dualarm-dir", nargs="+", default=[])
    ap.add_argument("--dualarm-gripper", nargs=2, default=["get", "parallel"])
    ap.add_argument("--dualarm-task", default=None)
    ap.add_argument("--dualarm-repeat", type=int, default=1)
    ap.add_argument("--gripper-names", nargs="+", default=["parallel", "get"])
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--repo-id", default="cotrain_dualarm2")
    ap.add_argument("--template", default="/home/ypf/pi_data/cotrain_dualarm2",
                    help="schema 模板(现有同构打包集); --append 时自动用输出集自身")
    ap.add_argument("--append", action="store_true", help="增量: 跳过 manifest 里已打包的源 episode")
    ap.add_argument("--link", action="store_true", help="硬链接代替复制(同分区, 零拷贝)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    out = Path(args.output).expanduser().resolve()
    tmpl_root = out if args.append else Path(args.template).expanduser().resolve()
    tmpl_pq = sorted((tmpl_root / "data").rglob("episode_*.parquet"))
    if not tmpl_pq:
        sys.exit(f"模板 {tmpl_root} 里找不到 parquet(需要一个现成的双臂20D打包集当 schema 模板)")
    tmpl_schema = pq.read_schema(tmpl_pq[0])
    tmpl_info = json.loads((tmpl_root / "meta" / "info.json").read_text(encoding="utf-8"))

    name_to_id = {n: i for i, n in enumerate(args.gripper_names)}

    # ---- manifest / 既有状态 ----
    mf_path = out / "pack_manifest.json"
    if args.append:
        if not mf_path.exists():
            sys.exit("--append 需要输出目录里有 pack_manifest.json(即之前由 v2 打包)")
        manifest = json.loads(mf_path.read_text(encoding="utf-8"))
        tasks = {}     # task 串 -> index
        for line in (out / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line); tasks[d["task"]] = int(d["task_index"])
        ep_idx = manifest["next_episode"]
        g_index = manifest["next_index"]
        ep_lines, stat_lines = [], []          # 追加写
    else:
        if (out / "meta").exists():
            sys.exit(f"{out} 已存在; 全量重建请换目录或先删除, 增量请用 --append")
        for sub in ("data", "meta"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        manifest = {"version": 1, "entries": {}, "next_episode": 0, "next_index": 0}
        tasks = {}
        ep_idx = 0; g_index = 0
        ep_lines, stat_lines = [], []

    done = set(manifest["entries"].keys())
    black_cache_dir = out / "_black_cache"   # 放 videos/ 外, 不污染数据集视频计数
    black_cache_dir.mkdir(parents=True, exist_ok=True)
    black_cache = {int(p.stem.split("_")[1]): p for p in black_cache_dir.glob("black_*.mp4")}

    def task_id(t):
        if t not in tasks:
            tasks[t] = len(tasks)
        return tasks[t]

    def put_video(src_mp4, key, idx):
        dst_dir = out / "videos" / f"chunk-{idx // CHUNK:03d}" / key
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"episode_{idx:06d}.mp4"
        if dst.exists():
            dst.unlink()
        if args.link:
            try:
                import os; os.link(src_mp4, dst); return dst
            except OSError:
                pass
        shutil.copy2(src_mp4, dst)
        return dst

    def emit(rows, n, task, gid0, gid1, vids, key_id):
        """写一条打包 episode: parquet + 3 路视频复制/生成 + stats + manifest。"""
        nonlocal ep_idx, g_index
        rows["gid0"], rows["gid1"] = gid0, gid1
        rows["state"] = rows["state"][:n]; rows["actions"] = rows["actions"][:n]
        pq_dir = out / "data" / f"chunk-{ep_idx // CHUNK:03d}"
        pq_dir.mkdir(parents=True, exist_ok=True)
        cols = write_parquet(pq_dir / f"episode_{ep_idx:06d}.parquet",
                             tmpl_schema, ep_idx, task_id(task), g_index, rows)
        img_stats = {}
        for key in VIDEO_KEYS:
            if vids.get(key) is None:                     # 单臂: 臂1 黑视频
                src = make_black_video(n, black_cache_dir, black_cache)
                put_video(src, key, ep_idx)
                img_stats[key] = dict(_BLACK_STATS)
            else:
                put_video(vids[key], key, ep_idx)
                img_stats[key] = sample_image_stats(vids[key], n)
        ep_lines.append(json.dumps({"episode_index": ep_idx, "tasks": [task], "length": n},
                                   ensure_ascii=False))
        stat_lines.append(json.dumps(episode_stats_row(ep_idx, cols, img_stats),
                                     ensure_ascii=False))
        manifest["entries"][key_id] = {"out_index": ep_idx, "task": task, "n": n}
        ep_idx += 1; g_index += n

    n_new = n_skip = 0

    # ---- 单臂遥操作 ----
    tg = args.teleop_gripper if len(args.teleop_gripper) != 1 else args.teleop_gripper * len(args.teleop_dir)
    for d, gname in zip(args.teleop_dir, tg):
        troot = Path(d).expanduser().resolve()
        gp = get_params(gname); gid = name_to_id[gname]
        src_tasks = load_source_tasks(troot)
        pqs = sorted((troot / "data").rglob("episode_*.parquet"))[: args.limit or None]
        print(f"[单臂 {troot.name}] {len(pqs)} 条 爪={gname}")
        for pqf in pqs:
            key_id = f"{troot}::{pqf.stem}::0"
            if key_id in done:
                n_skip += 1; continue
            sub = pqf.parent.name
            vids = {"image": troot / "videos" / sub / "image" / f"{pqf.stem}.mp4",
                    "wrist_image": troot / "videos" / sub / "wrist_image" / f"{pqf.stem}.mp4",
                    "wrist_image_1": None}
            if not (vids["image"].exists() and vids["wrist_image"].exists()):
                print(f"  ✗ {pqf.stem}: 缺视频, 跳过"); continue
            rows, n = teleop_rows(pqf, gp)
            if rows is None:
                print(f"  ✗ {pqf.stem}: 无 state_eef, 跳过"); continue
            nv = min(video_nframes(vids["image"]), video_nframes(vids["wrist_image"]))
            if nv < n:
                print(f"  ⚠ {pqf.stem}: 视频帧({nv})<状态行({n}), 截断到 {nv}"); n = nv
            task = args.teleop_task or src_tasks.get(rows["task_index"])
            if task is None:
                print(f"  ✗ {pqf.stem}: 无 task, 跳过"); continue
            emit(rows, n, task, gid, 0, vids, key_id)
            n_new += 1

    # ---- 双臂 ----
    gp0, gp1 = get_params(args.dualarm_gripper[0]), get_params(args.dualarm_gripper[1])
    gid0, gid1 = name_to_id[args.dualarm_gripper[0]], name_to_id[args.dualarm_gripper[1]]
    for d in args.dualarm_dir:
        droot = Path(d).expanduser().resolve()
        src_tasks = load_source_tasks(droot)
        pqs = sorted((droot / "data").rglob("episode_*.parquet"))[: args.limit or None]
        print(f"[双臂 {droot.name}] {len(pqs)} 条")
        for pqf in pqs:
            sub = pqf.parent.name
            vids = {k: droot / "videos" / sub / k / f"{pqf.stem}.mp4" for k in VIDEO_KEYS}
            if not all(v.exists() for v in vids.values()):
                print(f"  ✗ {pqf.stem}: 缺视频, 跳过"); continue
            rows, n = dualarm_rows(pqf, gp0, gp1)
            if rows is None:
                print(f"  ✗ {pqf.stem}: 缺 state_eef_0/1, 跳过"); continue
            nv = min(video_nframes(v) for v in vids.values())
            if nv < n:
                print(f"  ⚠ {pqf.stem}: 视频帧({nv})<状态行({n}), 截断到 {nv}"); n = nv
            task = args.dualarm_task or src_tasks.get(rows["task_index"])
            if task is None:
                print(f"  ✗ {pqf.stem}: 无 task, 跳过"); continue
            for r in range(max(1, args.dualarm_repeat)):
                key_id = f"{droot}::{pqf.stem}::{r}"
                if key_id in done:
                    n_skip += 1; continue
                emit({**rows, "state": rows["state"].copy(), "actions": rows["actions"].copy()},
                     n, task, gid0, gid1, vids, key_id)
                n_new += 1

    # ---- meta 落盘 ----
    meta = out / "meta"
    mode = "a" if args.append else "w"
    with open(meta / "episodes.jsonl", mode, encoding="utf-8") as f:
        for line in ep_lines: f.write(line + "\n")
    with open(meta / "episodes_stats.jsonl", mode, encoding="utf-8") as f:
        for line in stat_lines: f.write(line + "\n")
    with open(meta / "tasks.jsonl", "w", encoding="utf-8") as f:
        for t, i in sorted(tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n")

    info = dict(tmpl_info)
    info.update(total_episodes=ep_idx, total_frames=g_index, total_tasks=len(tasks),
                total_videos=ep_idx * 3, total_chunks=(ep_idx + CHUNK - 1) // CHUNK,
                splits={"train": f"0:{ep_idx}"})
    (meta / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")

    manifest["next_episode"] = ep_idx; manifest["next_index"] = g_index
    mf_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n===== pack v2 完成 =====")
    print(f"  新打包 {n_new} 条(跳过已有 {n_skip}), 总计 {ep_idx} 条 / {g_index} 帧")
    print(f"  输出: {out}")

    # 服务器兼容: parquet 元数据 List->Sequence(datasets 3.x 才能读)。
    # 注意模板若来自未跑完的旧打包会遗留 "List", 这里兜底保证产物永远是修补过的。
    make_server_compatible(out)

    if args.verify:
        verify_dataset(out, args.repo_id)


if __name__ == "__main__":
    main()
