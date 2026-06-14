#!/usr/bin/env python3
"""
阶段 4：把 UMI mcap + 遥操作 LeRobot 打包成统一的「联合训练」LeRobot 数据集。

两个来源 -> 同一套 schema（都无根 W' 相对系 + 6D 旋转 + 夹爪归一化 + video）：
  state   = [pos(3), rot6d(6), gripper(1)]  = 10D   (W' 系当前指尖位姿)
  actions = 同 10D，定义为「下一帧位姿」(action[t]=state[t+1], 末帧重复)
  image       = 环境相机 (video)：遥操作=真实D405环境图；UMI=零图(无环境相机)
  wrist_image = 手腕相机 (video)：遥操作=D405手眼；UMI=去畸变中间鱼眼(79°)
  env_mask    = (1,) 环境相机是否有效：遥操作=1.0；UMI=0.0
              （供策略 image_mask 用：UMI 样本忽略环境相机槽）

坐标统一：
  - 遥操作 get_end_pose 已是机械臂工具系(base 系) -> 直接建 W'。
  - UMI eef_pose 是相机本体系(world 系) -> 右乘 Trans(tcp_offset) 到指尖、
    右乘 R_align 转成工具系约定 -> 再建 W'。两边落到同一"工具系/W'"表示。

时序：统一重采到 10Hz（遥操作本就是 10Hz；UMI VIO 30Hz 下采样）。

用法:
    # 先小规模本地测试（每个来源只取前 3 条，快速验证管线）
    python pack_lerobot.py --umi-dir /home/ypf/gen_robotics/data/output/clean_board \
        --teleop-dir /home/ypf/pi_data/airbot_play_data \
        -o /home/ypf/pi_data/cotrain_eef --limit 3 --verify

    # 全量打包
    python pack_lerobot.py --umi-dir /home/ypf/gen_robotics/data/output/clean_board \
        --teleop-dir /home/ypf/pi_data/airbot_play_data \
        -o /home/ypf/pi_data/cotrain_eef --verify
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from umi_to_lerobot import (  # noqa: E402
    _iter_decoded, read_mcap_cameras, decode_h264, build_undistort_maps,
    read_pose_stream, quat_to_rot, rot_to_6d, nearest_idx,
    DEFAULT_TCP_OFFSET, DEFAULT_ROBOT_GRIPPER_MAX, POSE_TOPIC, GRIPPER_TOPIC,
)
from replay_check import rpy_to_rot  # noqa: E402

import tempfile

DEFAULT_ALIGN_RPY = (-1.0, -16.8, 0.0)   # UMI body -> AIRBOT 工具系 (calib_align 标定)
TARGET_HFOV = 79.0                        # 去畸变目标视场，对齐 D405
TARGET_FPS = 10                           # 统一帧率（遥操作 10Hz）
OUT_W, OUT_H = 640, 480
UMI_MAIN_CAM = "camera0"                  # UMI 中间主相机


# ---------------------------------------------------------------------------
# 共享：世界/base 系工具位姿 -> W' 相对系 (pos3 + rot6d6)
# ---------------------------------------------------------------------------
def build_wprime(tip_pos, R_tool):
    """tip_pos (N,3), R_tool (N,3,3) 在重力对齐(z向上)的参考系里
    -> W'(原点=首帧指尖, z=重力, 首帧 yaw 归零) 下的 pos(N,3)+rot6d(N,6)。"""
    fwd0 = R_tool[0][:, 0]                       # 工具前向(x)在参考系的方向
    yaw0 = np.arctan2(fwd0[1], fwd0[0])
    c, s = np.cos(yaw0), np.sin(yaw0)
    Rw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])   # ref <- W'
    t0 = tip_pos[0]
    pos = (tip_pos - t0) @ Rw                    # = Rw^T (p - t0)
    R_loc = np.einsum("ji,njk->nik", Rw, R_tool)
    rot6d = np.stack([rot_to_6d(R) for R in R_loc])
    return pos.astype(np.float32), rot6d.astype(np.float32)


def make_state_action(pos, rot6d, grip01):
    """组装 state(N,10) 和 actions(N,10)=下一帧 state。"""
    state = np.concatenate([pos, rot6d, grip01[:, None]], axis=1).astype(np.float32)
    action = np.roll(state, -1, axis=0)
    action[-1] = state[-1]                       # 末帧无下一帧，重复
    return state, action


# ---------------------------------------------------------------------------
# UMI：mcap -> 帧序列
# ---------------------------------------------------------------------------
def read_wrist_frames(mcap_path, cam_name, target_hfov):
    """读 UMI 主相机：去畸变(79°)后的帧 + 每帧时间戳。返回 (frames[HxWx3 RGB], ts(ns))。"""
    cams = read_mcap_cameras(mcap_path)
    cam = next((c for c in cams.values() if c["name"] == cam_name), None)
    if cam is None or cam["calib"] is None or not cam["h264"]:
        return None, None
    # 帧时间戳：扫一遍该相机的 compressed 消息 log_time
    ts = []
    for _s, ch, msg, _d in _iter_decoded(mcap_path, [cam["img_topic"]]):
        ts.append(msg.log_time)
    ts = np.array(ts)
    with tempfile.TemporaryDirectory() as td:
        jpgs = decode_h264(cam["h264"], 30, Path(td), cam_name)
        if not jpgs:
            return None, None
        first = cv2.imread(str(jpgs[0]))
        native = (first.shape[1], first.shape[0])
        m1, m2, _K, (w, h) = build_undistort_maps(
            cam["calib"], 0.0, 1.0, target_hfov, src_size=native,
            out_size=(OUT_W, OUT_H))
        frames = []
        for jpg in jpgs:
            img = cv2.imread(str(jpg))
            if (img.shape[1], img.shape[0]) != native:
                img = cv2.resize(img, native)
            und = cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
            frames.append(und[:, :, ::-1].copy())   # BGR->RGB
    n = min(len(frames), len(ts))
    return frames[:n], ts[:n]


def process_umi_episode(mcap_path, tcp_offset, R_align, robot_gripper_max,
                        target_hfov):
    """UMI mcap -> {wrist, env, state, actions, env_mask}（10Hz, W' 系）。"""
    wrist_frames, cam_ts = read_wrist_frames(mcap_path, UMI_MAIN_CAM, target_hfov)
    if wrist_frames is None:
        print(f"  ✗ {mcap_path.name}: 无主相机帧, 跳过")
        return None
    pts, cam_pos_w, quat, gts, gv = read_pose_stream(mcap_path, POSE_TOPIC, GRIPPER_TOPIC)
    if len(pts) == 0:
        print(f"  ✗ {mcap_path.name}: 无位姿, 跳过")
        return None

    R_body = np.stack([quat_to_rot(*q) for q in quat])
    tip_pos = cam_pos_w + np.einsum("nij,j->ni", R_body, np.asarray(tcp_offset))
    R_tool = np.einsum("nij,jk->nik", R_body, R_align)        # 本体系 -> 工具系约定

    # 统一 10Hz 时间网格（取位姿与相机的重叠时段）
    t0 = max(pts[0], cam_ts[0]); t1 = min(pts[-1], cam_ts[-1])
    if t1 <= t0:
        print(f"  ✗ {mcap_path.name}: 位姿/相机时间不重叠, 跳过")
        return None
    grid = np.arange(t0, t1, 1e9 / TARGET_FPS)
    pi = [nearest_idx(pts, t) for t in grid]
    ci = [nearest_idx(cam_ts, t) for t in grid]
    gi = [nearest_idx(gts, t) for t in grid] if len(gts) else None

    pos, rot6d = build_wprime(tip_pos[pi], R_tool[pi])
    grip_m = gv[gi] if gi is not None else np.zeros(len(grid))
    grip01 = np.clip(grip_m / robot_gripper_max, 0.0, 1.0).astype(np.float32)
    state, action = make_state_action(pos, rot6d, grip01)
    wrist = [wrist_frames[i] for i in ci]
    env = [np.zeros((OUT_H, OUT_W, 3), np.uint8)] * len(grid)
    return {"wrist": wrist, "env": env, "state": state, "actions": action,
            "env_mask": np.zeros(len(grid), np.float32)}


# ---------------------------------------------------------------------------
# 遥操作：LeRobot episode -> 帧序列
# ---------------------------------------------------------------------------
def read_video_frames(mp4_path):
    """mp4 -> list of HxWx3 RGB。"""
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr[:, :, ::-1].copy())     # BGR->RGB
    cap.release()
    return frames


def process_teleop_episode(parquet, img_mp4, wrist_mp4, robot_gripper_max):
    """遥操作 episode -> {wrist, env, state, actions, env_mask}（W' 系）。"""
    import pyarrow.parquet as pq
    t = pq.read_table(parquet).to_pydict()
    if "state_eef" not in t:
        print(f"  ✗ {parquet.name}: 无 state_eef（旧格式?）, 跳过")
        return None
    se = np.array(t["state_eef"], dtype=np.float64)   # (N,8) pos3+quat4(xyzw)+grip(m)
    pos_base = se[:, :3]
    quat = se[:, 3:7]
    grip_m = se[:, 7]
    R_tool = np.stack([quat_to_rot(*q) for q in quat])    # 已是工具系(base)

    pos, rot6d = build_wprime(pos_base, R_tool)
    grip01 = np.clip(grip_m / robot_gripper_max, 0.0, 1.0).astype(np.float32)
    state, action = make_state_action(pos, rot6d, grip01)

    env = read_video_frames(img_mp4)
    wrist = read_video_frames(wrist_mp4)
    n = min(len(state), len(env), len(wrist))
    if n < 5:
        print(f"  ✗ {parquet.name}: 帧数不足({n}), 跳过")
        return None
    return {"wrist": [cv2.resize(w, (OUT_W, OUT_H)) for w in wrist[:n]],
            "env": [cv2.resize(e, (OUT_W, OUT_H)) for e in env[:n]],
            "state": state[:n], "actions": action[:n],
            "env_mask": np.ones(n, np.float32)}


# ---------------------------------------------------------------------------
# 打包
# ---------------------------------------------------------------------------
def create_dataset(out_dir, repo_id):
    import lerobot.common.datasets.lerobot_dataset as lr_ds
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    if not getattr(lr_ds, "_h264_patched", False):
        _orig = lr_ds.encode_video_frames
        def _h264(*a, **k):
            k.setdefault("vcodec", "h264"); return _orig(*a, **k)
        lr_ds.encode_video_frames = _h264
        lr_ds._h264_patched = True
    return LeRobotDataset.create(
        repo_id=repo_id, robot_type="airbot_play", fps=TARGET_FPS, root=out_dir,
        features={
            "image":       {"dtype": "video", "shape": (OUT_H, OUT_W, 3),
                            "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "video", "shape": (OUT_H, OUT_W, 3),
                            "names": ["height", "width", "channel"]},
            "state":   {"dtype": "float32", "shape": (10,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (10,), "names": ["actions"]},
            "env_mask": {"dtype": "float32", "shape": (1,), "names": ["env_mask"]},
        },
        image_writer_threads=4, image_writer_processes=2,
    )


def write_episode(dataset, ep, task):
    for i in range(len(ep["state"])):
        dataset.add_frame({
            "image": ep["env"][i],
            "wrist_image": ep["wrist"][i],
            "state": ep["state"][i],
            "actions": ep["actions"][i],
            "env_mask": np.array([ep["env_mask"][i]], np.float32),
            "task": task,
        })
    dataset.save_episode()


def main():
    ap = argparse.ArgumentParser(description="阶段4: UMI + 遥操作 -> 统一联合训练 LeRobot")
    ap.add_argument("--umi-dir", required=True, help="UMI VIO mcap 目录")
    ap.add_argument("--teleop-dir", required=True, help="遥操作 LeRobot 数据集根目录")
    ap.add_argument("-o", "--output", required=True, help="输出合并数据集目录")
    ap.add_argument("--repo-id", default="cotrain_eef")
    ap.add_argument("--umi-task", default="wipe the blackboard")
    ap.add_argument("--teleop-task", default="pick up the block and place it in the bowl")
    ap.add_argument("--tcp-offset", nargs=3, type=float, default=list(DEFAULT_TCP_OFFSET))
    ap.add_argument("--align-rpy", nargs=3, type=float, default=list(DEFAULT_ALIGN_RPY))
    ap.add_argument("--robot-gripper-max", type=float, default=DEFAULT_ROBOT_GRIPPER_MAX)
    ap.add_argument("--target-hfov", type=float, default=TARGET_HFOV)
    ap.add_argument("--limit", type=int, default=None, help="每来源最多处理几条(调试)")
    ap.add_argument("--verify", action="store_true", help="打包后重载抽检")
    args = ap.parse_args()

    R_align = rpy_to_rot(*args.align_rpy)
    out_dir = Path(args.output).expanduser().resolve()
    dataset = create_dataset(out_dir, args.repo_id)

    stats = {"umi": [0, 0], "teleop": [0, 0]}   # [episodes, frames]

    # ---- UMI ----
    umi_mcaps = sorted(Path(args.umi_dir).expanduser().glob("*.mcap"))
    if args.limit:
        umi_mcaps = umi_mcaps[:args.limit]
    print(f"\n=== UMI: {len(umi_mcaps)} 条 ===")
    for m in umi_mcaps:
        ep = process_umi_episode(m, args.tcp_offset, R_align,
                                 args.robot_gripper_max, args.target_hfov)
        if ep is None:
            continue
        write_episode(dataset, ep, args.umi_task)
        stats["umi"][0] += 1; stats["umi"][1] += len(ep["state"])
        print(f"  ✓ {m.name}: {len(ep['state'])} 帧")

    # ---- 遥操作 ----
    troot = Path(args.teleop_dir).expanduser()
    pqs = sorted((troot / "data").rglob("episode_*.parquet"))
    if args.limit:
        pqs = pqs[:args.limit]
    print(f"\n=== 遥操作: {len(pqs)} 条 ===")
    for pqf in pqs:
        stem = pqf.stem
        img_mp4 = troot / "videos" / pqf.parent.name / "image" / f"{stem}.mp4"
        wrist_mp4 = troot / "videos" / pqf.parent.name / "wrist_image" / f"{stem}.mp4"
        if not img_mp4.exists() or not wrist_mp4.exists():
            print(f"  ✗ {stem}: 缺视频, 跳过"); continue
        ep = process_teleop_episode(pqf, img_mp4, wrist_mp4, args.robot_gripper_max)
        if ep is None:
            continue
        write_episode(dataset, ep, args.teleop_task)
        stats["teleop"][0] += 1; stats["teleop"][1] += len(ep["state"])
        print(f"  ✓ {stem}: {len(ep['state'])} 帧")

    uf, tf = stats["umi"][1], stats["teleop"][1]
    tot = uf + tf
    print(f"\n===== 打包完成 =====")
    print(f"  UMI:    {stats['umi'][0]} 条 / {uf} 帧 ({uf/max(tot,1)*100:.0f}%)")
    print(f"  遥操作: {stats['teleop'][0]} 条 / {tf} 帧 ({tf/max(tot,1)*100:.0f}%)")
    print(f"  真实混合比例(按帧) UMI:遥操作 = {uf}:{tf} ≈ {uf/max(tf,1):.2f}:1")
    print(f"  输出: {out_dir}")

    if args.verify:
        verify_dataset(out_dir, args.repo_id)


def verify_dataset(out_dir, repo_id):
    print("\n===== 抽检（重载）=====")
    import pyarrow.parquet as pq
    pqs = sorted((out_dir / "data").rglob("episode_*.parquet"))
    mp4s = sorted((out_dir / "videos").rglob("*.mp4"))
    print(f"  parquet: {len(pqs)}  视频: {len(mp4s)}")
    t = pq.read_table(pqs[0]).to_pydict()
    st = np.array(t["state"]); ac = np.array(t["actions"]); em = np.array(t["env_mask"])
    print(f"  state shape={st.shape}  actions shape={ac.shape}  env_mask[0]={em[0]}")
    print(f"  state[0]={np.round(st[0],3).tolist()}")
    # 解码一帧确认视频可读
    for sub in ("wrist_image", "image"):
        v = sorted((out_dir / "videos").rglob(f"{sub}/*.mp4"))
        if v:
            cap = cv2.VideoCapture(str(v[0])); ok, fr = cap.read(); cap.release()
            print(f"  {sub}: {'可解码' if ok else '✗解码失败'} "
                  f"{fr.shape if ok else ''}")
    print("  ✓ 抽检完成")


if __name__ == "__main__":
    main()
