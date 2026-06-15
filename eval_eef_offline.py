#!/usr/bin/env python3
"""
离线开环评测（teacher-forcing）：把 cotrain_eef 里某一条真实 episode 的「真实观测」
逐帧喂给训练好的 pi05_cotrain_eef 模型，对比模型预测的动作块 vs parquet 里的真值动作。

目的：绕开真机，直接量「模型到底有没有学会这条数据」。
  - flow-matching 的训练 loss 低 ≠ 采样动作准。这里直接算采样动作和真值的误差。
  - 若预测≈真值 → 模型学会了，真机差是「部署时分布漂移」(相机/W'/起始位姿 OOD)。
  - 若预测就错（或退化成“几乎不动”/UMI 式大平移）→ 模型没学好这条任务
    （协同训练干扰 / 归一化把精细动作压没了）。

只用 pyarrow + cv2 解码，不走 LeRobot 的 dataloader（避开 datasets/torchcodec 版本坑）。

用法（服务器上）:
  python eval_eef_offline.py \
    --checkpoint /path/to/checkpoints/pi05_cotrain_eef/<exp>/17000 \
    --dataset    /path/to/cotrain_eef \
    --task-filter "block"          # 选第一条积木(遥操作)的 episode；擦黑板用 "wipe"
    --plot eval_block.png

  # 不给 --task-filter / --episode-index 时，会先列出所有 episode 供你挑，然后默认评第 0 条
"""
import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_ROOT / "airbot"))
sys.path.insert(0, str(_ROOT / "umi"))

from openpi.policies import policy_config as _policy_config   # noqa: E402
from openpi.training import config as _config                 # noqa: E402
from umi_to_lerobot import rot6d_to_rot                        # noqa: E402


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def geodesic_deg(r6_a, r6_b):
    """两个 6D 旋转之间的测地角(度)。"""
    Ra, Rb = rot6d_to_rot(np.asarray(r6_a)), rot6d_to_rot(np.asarray(r6_b))
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def load_tasks(root):
    tasks = {}
    f = root / "meta" / "tasks.jsonl"
    for line in f.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            tasks[int(d["task_index"])] = d["task"]
    return tasks


def list_episodes(root):
    return sorted((root / "data").rglob("episode_*.parquet"))


def read_parquet(pqf):
    t = pq.read_table(pqf).to_pydict()
    state = np.array(t["state"], dtype=np.float32)        # (N,10) W'
    actions = np.array(t["actions"], dtype=np.float32)    # (N,10) = 下一帧 state
    env_mask = np.array(t["env_mask"], dtype=np.float32).reshape(-1)  # (N,)
    task_index = int(t["task_index"][0])
    return state, actions, env_mask, task_index


def video_path(root, pqf, feat):
    return root / "videos" / pqf.parent.name / feat / f"{pqf.stem}.mp4"


def decode_video(mp4):
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr[:, :, ::-1].copy())   # BGR->RGB (与打包时一致)
    cap.release()
    return frames


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="离线开环评测 EEF 模型（预测 vs 真值）")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, help="cotrain_eef 数据集根目录")
    ap.add_argument("--config-name", default="pi05_cotrain_eef")
    ap.add_argument("--episode-index", type=int, default=None, help="第几条 parquet(按排序)")
    ap.add_argument("--task-filter", default=None, help="按任务串选 episode, 如 'block' / 'wipe'")
    ap.add_argument("--stride", type=int, default=3, help="每隔几帧评一次(省时间)")
    ap.add_argument("--max-frames", type=int, default=None, help="最多评多少个采样点")
    ap.add_argument("--plot", default=None, help="保存预测vs真值对比图(png)")
    args = ap.parse_args()

    root = pathlib.Path(args.dataset).expanduser().resolve()
    tasks = load_tasks(root)
    eps = list_episodes(root)
    if not eps:
        print(f"✗ 没找到 episode: {root}/data"); return

    # ---- 列出所有 episode（便于挑选）----
    print(f"\n=== {len(eps)} 条 episode ===")
    metas = []
    for i, pqf in enumerate(eps):
        s, a, em, ti = read_parquet(pqf)
        metas.append((i, pqf, em.mean(), tasks.get(ti, "?"), len(s)))
    for i, pqf, emm, tk, n in metas[:200]:
        kind = "遥操作" if emm > 0.5 else "UMI"
        print(f"  [{i:3d}] {pqf.name}  env_mask={emm:.2f}({kind})  n={n:4d}  task='{tk}'")

    # ---- 选定一条 ----
    if args.episode_index is not None:
        idx = args.episode_index
    elif args.task_filter:
        cand = [m for m in metas if args.task_filter.lower() in m[3].lower()]
        if not cand:
            print(f"✗ 没有任务含 '{args.task_filter}' 的 episode"); return
        idx = cand[0][0]
    else:
        idx = 0
    _, pqf, emm, task, _ = metas[idx]
    print(f"\n>>> 评测 episode [{idx}] {pqf.name}  task='{task}'  "
          f"env_mask={emm:.2f}({'遥操作' if emm > 0.5 else 'UMI'})")

    state, actions, env_mask, _ = read_parquet(pqf)
    env_frames = decode_video(video_path(root, pqf, "image"))
    wrist_frames = decode_video(video_path(root, pqf, "wrist_image"))
    n = min(len(state), len(env_frames), len(wrist_frames))
    print(f">>> 帧数 state={len(state)} env={len(env_frames)} wrist={len(wrist_frames)} -> 用 {n}")

    # ---- 加载模型 ----
    print(">>> 加载模型...", flush=True)
    cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        cfg, pathlib.Path(args.checkpoint), default_prompt=task)
    print(">>> 模型就绪", flush=True)

    # ---- 逐帧 teacher-forcing 推理 ----
    rows = list(range(0, n - 1, args.stride))
    if args.max_frames:
        rows = rows[:args.max_frames]

    # 误差累积：模型 vs 真值；以及 "原地不动" 基线 vs 真值
    H_errs = {"pos": [], "rot": [], "grip": []}        # 仅 horizon-0
    H_errs_full = {"pos": [], "rot": [], "grip": []}   # 整段 horizon 平均
    stay_errs = {"pos": [], "rot": [], "grip": []}     # 基线: pred=当前 state
    traj_pred0, traj_gt0 = [], []                      # 画图: 每帧 horizon-0 的下一位姿

    for j, t in enumerate(rows):
        obs = {
            "observation/image": env_frames[t].astype(np.uint8),
            "observation/wrist_image": wrist_frames[t].astype(np.uint8),
            "observation/state": state[t],
            "observation/env_mask": np.float32(env_mask[t]),
            "prompt": task,
        }
        chunk = policy.infer(obs)["actions"]      # (H,10) W'
        H = min(len(chunk), n - t - 1)
        if H <= 0:
            continue
        gt = actions[t:t + H]                     # 真值动作块

        # horizon-0（最关键：下一步该去哪）
        p0, r0, g0 = chunk[0, :3], chunk[0, 3:9], chunk[0, 9]
        gp0, gr0, gg0 = gt[0, :3], gt[0, 3:9], gt[0, 9]
        H_errs["pos"].append(np.linalg.norm(p0 - gp0) * 1000)
        H_errs["rot"].append(geodesic_deg(r0, gr0))
        H_errs["grip"].append(abs(float(g0) - float(gg0)))

        # 整段 horizon 平均
        H_errs_full["pos"].append(np.mean(np.linalg.norm(chunk[:H, :3] - gt[:, :3], axis=1)) * 1000)
        H_errs_full["rot"].append(np.mean([geodesic_deg(chunk[k, 3:9], gt[k, 3:9]) for k in range(H)]))
        H_errs_full["grip"].append(np.mean(np.abs(chunk[:H, 9] - gt[:, 9])))

        # 基线：原地不动 (pred = 当前 state[t])
        stay_errs["pos"].append(np.linalg.norm(state[t, :3] - gp0) * 1000)
        stay_errs["rot"].append(geodesic_deg(state[t, 3:9], gr0))
        stay_errs["grip"].append(abs(float(state[t, 9]) - float(gg0)))

        traj_pred0.append(chunk[0])
        traj_gt0.append(gt[0])

        if j < 3:   # 头几帧打印细节
            print(f"\n  [t={t}] state pos={state[t,:3].round(3).tolist()} grip={state[t,9]:.2f}")
            print(f"        pred[0] pos={p0.round(3).tolist()} grip={float(g0):.2f}")
            print(f"        gt  [0] pos={gp0.round(3).tolist()} grip={float(gg0):.2f}")

    def stat(d):
        a = np.array(d)
        return f"mean={a.mean():.2f} median={np.median(a):.2f} p90={np.percentile(a,90):.2f} max={a.max():.2f}"

    print("\n" + "=" * 64)
    print(f"评测点数: {len(H_errs['pos'])}")
    print("\n--- 模型 vs 真值  (horizon-0, 下一步) ---")
    print(f"  位置误差(mm): {stat(H_errs['pos'])}")
    print(f"  旋转误差(°) : {stat(H_errs['rot'])}")
    print(f"  夹爪误差    : {stat(H_errs['grip'])}")
    print("\n--- 模型 vs 真值  (整段 horizon 平均) ---")
    print(f"  位置误差(mm): {stat(H_errs_full['pos'])}")
    print(f"  旋转误差(°) : {stat(H_errs_full['rot'])}")
    print(f"  夹爪误差    : {stat(H_errs_full['grip'])}")
    print("\n--- 基线: 原地不动 vs 真值  (模型必须明显优于它) ---")
    print(f"  位置误差(mm): {stat(stay_errs['pos'])}")
    print(f"  旋转误差(°) : {stat(stay_errs['rot'])}")
    print(f"  夹爪误差    : {stat(stay_errs['grip'])}")
    mp = np.mean(H_errs['pos']); sp = np.mean(stay_errs['pos'])
    print(f"\n  >> 位置: 模型 {mp:.1f}mm vs 原地不动 {sp:.1f}mm  "
          f"({'模型更好✓' if mp < sp*0.8 else '模型≈不动, 没学到运动✗'})")
    print("=" * 64)

    # ---- 画图 ----
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pred = np.array(traj_pred0); gt = np.array(traj_gt0)
        x = np.array(rows[:len(pred)])
        fig, ax = plt.subplots(2, 2, figsize=(13, 7))
        for k, name in enumerate(["x", "y", "z"]):
            ax[0, 0].plot(x, gt[:, k], "-", label=f"gt {name}")
            ax[0, 0].plot(x, pred[:, k], "--", label=f"pred {name}")
        ax[0, 0].set_title("W' position (one-step pred vs gt)"); ax[0, 0].legend(fontsize=7)
        ax[0, 1].plot(x, gt[:, 9], "-", label="gt grip")
        ax[0, 1].plot(x, pred[:, 9], "--", label="pred grip")
        ax[0, 1].set_title("gripper"); ax[0, 1].legend(fontsize=7)
        ax[1, 0].plot(x, H_errs["pos"], label="model")
        ax[1, 0].plot(x, stay_errs["pos"], label="stay-baseline")
        ax[1, 0].set_title("pos error (mm)"); ax[1, 0].legend(fontsize=7)
        ax[1, 1].plot(x, H_errs["rot"], label="model")
        ax[1, 1].plot(x, stay_errs["rot"], label="stay-baseline")
        ax[1, 1].set_title("rot error (deg)"); ax[1, 1].legend(fontsize=7)
        fig.suptitle(f"episode[{idx}] task='{task}'")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f">>> 图已保存: {args.plot}")


if __name__ == "__main__":
    main()
