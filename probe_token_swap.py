#!/usr/bin/env python3
"""离线 token-swap 探针: 量化"模型输出对夹爪点云到底有多敏感", 不动机械臂。

原理: 流匹配每次推理从随机噪声积分, 同一观测跑两次本身就有差异——这是【噪声地板】。
把"换爪点云"造成的输出差异与地板相比:
    ratio ≈ 1     -> 换点云的效应淹没在采样噪声里(通道对爪身份不敏感/半死)
    ratio >= ~2-3 -> 模型确实在读点云(通道活着)
另配两个极端探针: 全零点云 / 随机点云——连这两个都拉不开地板, 就是"死透"的强证据。

每帧 5 次推理:
    A1, A2 = 正确点云 x2(-> 地板)   B = 换成另一把爪   Z = 全零   R = 随机

用法(约 3-5 分钟, 首次推理含 XLA 编译会慢):
    python probe_token_swap.py \
        --checkpoint checkpoints/pi05_cotrain_dualarm_region/region_v1/33999 \
        --config-name pi05_cotrain_dualarm_region
"""
import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_ROOT / "umi"))
sys.path.insert(0, str(_ROOT / "gripper_geom"))
from pack_lerobot_fast import REST10, _arm_state_action  # noqa: E402  与训练打包同一套数学
from gripper_params import get_params  # noqa: E402

# 探针数据源: (目录, 单臂爪名 或 双臂(爪0,爪1))。切泥最关键——训练里该任务与 get 点云
# 完全绑定, 通道活着的话这里对 swap 应最敏感。
SOURCES = [
    ("/home/ypf/pi_data/GET/pick_block_task/get_block", "get"),
    ("/home/ypf/pi_data/parallel/pick_block_task/airbot_block_data", "parallel"),
    ("/home/ypf/pi_data/GET/cut_dough_task/cut_dough_get_v2", ("get", "parallel")),
]
FRAME_FRACS = (0.15, 0.3, 0.5, 0.7, 0.85)   # 每条 episode 抽帧位置
OTHER = {"get": "parallel", "parallel": "get"}


def pick_cloud(z, name, P, region, rng):
    """与 local_inference_eef._pick 一致: 规范未增强点云, region->(3,P,3)。"""
    pc = np.asarray(z[f"{name}_points"], np.float32)
    if region:
        reg = np.asarray(z[f"{name}_region"])
        return np.stack([
            pc[rng.choice(np.flatnonzero(reg == r), P, replace=int((reg == r).sum()) < P)]
            for r in range(3)])
    return pc[rng.choice(len(pc), P, replace=len(pc) < P)]


def load_tasks(root):
    tasks = {}
    f = root / "meta" / "tasks.jsonl"
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line); tasks[int(d["task_index"])] = d["task"]
    return tasks


def read_frame(mp4, idx):
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, bgr = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f"读帧失败: {mp4} @ {idx}")
    return bgr[:, :, ::-1].copy().astype(np.uint8)          # -> RGB


def episode_obs(root, pqf, grip, n_frames_wanted):
    """一条源 episode -> [(obs_dict 不含点云, 描述串), ...]。单臂/双臂自动识别。"""
    import pyarrow.parquet as pq
    t = pq.read_table(pqf).to_pydict()
    tasks = load_tasks(root)
    dual = isinstance(grip, tuple)
    sub = pqf.parent.name
    vid = lambda key: root / "videos" / sub / key / f"{pqf.stem}.mp4"  # noqa: E731
    if dual:
        gp0, gp1 = get_params(grip[0]), get_params(grip[1])
        s0, _ = _arm_state_action(t["state_eef_0"], t.get("actions_eef_0"),
                                  gp0["close"], gp0["open"], gp0["tcp_offset"])
        s1, _ = _arm_state_action(t["state_eef_1"], t.get("actions_eef_1"),
                                  gp1["close"], gp1["open"], gp1["tcp_offset"])
        n = min(len(s0), len(s1))
        state = np.concatenate([s0[:n], s1[:n]], axis=1)
        arm1_mask = 1.0
    else:
        gp = get_params(grip)
        s, _ = _arm_state_action(t["state_eef"], t.get("actions_eef"),
                                 gp["close"], gp["open"], gp["tcp_offset"])
        n = len(s)
        state = np.concatenate([s, np.tile(REST10, (n, 1))], axis=1)
        arm1_mask = 0.0
    task = tasks[int(t["task_index"][0])]
    out = []
    for f in FRAME_FRACS[:n_frames_wanted]:
        i = int(f * (n - 1))
        obs = {
            "observation/image": read_frame(vid("image"), i),
            "observation/wrist_image": read_frame(vid("wrist_image"), i),
            "observation/wrist_image_1": (read_frame(vid("wrist_image_1"), i) if dual
                                          else np.zeros((480, 640, 3), np.uint8)),
            "observation/state": state[i].astype(np.float32),
            "observation/env_mask": np.float32(1.0),
            "observation/arm1_mask": np.float32(arm1_mask),
            "prompt": task,
        }
        out.append((obs, f"{root.name}/{pqf.stem}@{i}"))
    return out


def dgroups(a, b):
    """两个动作 chunk (H,20) 的差异, 按维组拆分。"""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return {"pos0": d[:, 0:3].mean(), "rot0": d[:, 3:9].mean(),
            "grip0": d[:, 9].mean(), "arm1": d[:, 10:20].mean(),
            "all": d[:, :20].mean()}


def main():
    ap = argparse.ArgumentParser(description="离线 token-swap 探针")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--grippers-npz", default=str(_ROOT / "gripper_geom" / "grippers.npz"))
    ap.add_argument("--episodes-per-dir", type=int, default=2)
    ap.add_argument("--frames-per-ep", type=int, default=3)
    ap.add_argument("--nan-check", action="store_true",
                    help="连通性检查: 点云灌 NaN, 输出必被污染=点云确实进图; 干净=管线断了")
    args = ap.parse_args()

    cfg = _config.get_config(args.config_name)
    assert getattr(cfg.model, "gripper_token", False), "该 config 没开 gripper_token, 无从探针"
    region = bool(getattr(cfg.model, "region_tokens", False))
    P = int(getattr(cfg.model, "num_gripper_points", 512))
    z = np.load(args.grippers_npz, allow_pickle=True)
    clouds = {n: pick_cloud(z, n, P, region, np.random.default_rng(0))
              for n in ("parallel", "get")}
    lo = min(c.min() for c in clouds.values()); hi = max(c.max() for c in clouds.values())
    rr = np.random.default_rng(1)
    cloud_zero = np.zeros_like(clouds["get"])
    cloud_rand = rr.uniform(lo, hi, clouds["get"].shape).astype(np.float32)

    print(f">>> 加载 policy: {args.config_name} @ {args.checkpoint}", flush=True)
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint)

    if args.nan_check:
        root = pathlib.Path(SOURCES[0][0])
        pqf = sorted((root / "data").rglob("episode_*.parquet"))[0]
        obs, tag = episode_obs(root, pqf, SOURCES[0][1], 1)[0]
        obs["observation/gripper_pc"] = np.full_like(clouds["get"], np.nan)
        obs["observation/gripper_pc_1"] = clouds["get"]
        acts = np.asarray(policy.infer(obs)["actions"])
        poisoned = bool(np.isnan(acts).any())
        print(f"\nNaN 连通性检查 @ {tag}: 输出{'被污染(NaN)' if poisoned else '干净'}")
        print("  -> " + ("点云确实进入计算图, 探针结果可信: 是模型学会了无视它"
                         if poisoned else
                         "⚠ 点云没进图! 推理管线在某处丢弃了它, 探针的 1.0x 是假象, 要先修管线"))
        return

    rows = []   # (来源标签, arm0爪名, {cond: 动作chunk})
    for src, grip in SOURCES:
        root = pathlib.Path(src)
        if not root.exists():
            print(f"  ⚠ 跳过(不存在): {src}"); continue
        pqs = sorted((root / "data").rglob("episode_*.parquet"))[: args.episodes_per_dir]
        g0 = grip[0] if isinstance(grip, tuple) else grip
        g1 = grip[1] if isinstance(grip, tuple) else grip   # 单臂: 臂1点云无所谓(mask=0), 喂同名
        for pqf in pqs:
            for obs, tag in episode_obs(root, pqf, grip, args.frames_per_ep):
                def infer(c0):
                    o = dict(obs)
                    o["observation/gripper_pc"] = c0
                    o["observation/gripper_pc_1"] = clouds[g1]
                    return np.asarray(policy.infer(o)["actions"])[:, :20]
                acts = {"A1": infer(clouds[g0]), "A2": infer(clouds[g0]),
                        "B": infer(clouds[OTHER[g0]]),
                        "Z": infer(cloud_zero), "R": infer(cloud_rand)}
                rows.append((tag, g0, acts))
                print(f"  ✓ {tag} (爪={g0})", flush=True)

    # ---- 汇总: 各条件 vs 噪声地板 ----
    conds = [("A1 vs A2 (噪声地板)", "A2"), ("swap 另一把爪", "B"),
             ("全零点云", "Z"), ("随机点云", "R")]
    print("\n===== 输出敏感度(动作 chunk 平均绝对差, 及其相对噪声地板的倍数) =====")
    for scope, keep in [("全部帧", None), ("仅切泥(任务与get点云训练绑定)", "cut_dough")]:
        sel = [r for r in rows if keep is None or keep in r[0]]
        if not sel:
            continue
        print(f"\n--- {scope} ({len(sel)} 帧) ---")
        floor = np.mean([dgroups(r[2]["A1"], r[2]["A2"])["all"] for r in sel])
        print(f"{'条件':<28}{'all':>9}{'pos0':>9}{'rot0':>9}{'grip0':>9}{'倍数':>7}")
        for label, k in conds:
            ds = [dgroups(r[2]["A1"], r[2][k]) for r in sel]
            m = {kk: np.mean([d[kk] for d in ds]) for kk in ds[0]}
            print(f"{label:<28}{m['all']:>9.4f}{m['pos0']:>9.4f}{m['rot0']:>9.4f}"
                  f"{m['grip0']:>9.4f}{m['all'] / max(floor, 1e-9):>6.1f}x")
    print("\n判读: swap倍数≈1 -> 通道对爪身份不敏感; >=2-3 -> 在读点云; "
          "连 全零/随机 都≈1 -> 死透(FiLM 动机成立)")


if __name__ == "__main__":
    main()
