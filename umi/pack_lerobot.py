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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gripper_geom"))
from gripper_params import get_params  # noqa: E402  每爪开合范围 + 指尖偏移

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


# 单臂 10D 的"静止/rest"值: pos=0 + 单位旋转 rot6d[1,0,0,0,1,0] + grip=0。
# 单臂样本的臂1 用它填充并 arm1_mask=0(loss/相机/token 都会屏蔽), 只是占位。
REST10 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], np.float32)


def to_dualarm_from_single(ep):
    """把单臂 episode(10D) 包成双臂统一 schema:
    臂0=真值, 臂1=rest 占位 + arm1_mask=0, 臂1 手眼=空图。就地改 ep 并返回。"""
    n = len(ep["state"])
    rest = np.tile(REST10, (n, 1))
    ep["state"] = np.concatenate([np.asarray(ep["state"], np.float32), rest], axis=1)      # (n,20)
    ep["actions"] = np.concatenate([np.asarray(ep["actions"], np.float32), rest], axis=1)  # (n,20)
    ep["wrist_1"] = [np.zeros((OUT_H, OUT_W, 3), np.uint8)] * n
    ep["arm1_mask"] = np.zeros(n, np.float32)
    return ep


# ---------------------------------------------------------------------------
# UMI：mcap -> 帧序列
# ---------------------------------------------------------------------------
def read_wrist_frames(mcap_path, cam_name, target_hfov, undistort=True):
    """读 UMI 主相机帧 + 每帧时间戳。返回 (frames[HxWx3 RGB 640x480], ts(ns))。
    undistort=True: 去畸变(target_hfov, 对齐 D405); False: 原生鱼眼直接 resize(保留全广角)。"""
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
        m1 = m2 = None
        if undistort:
            m1, m2, _K, _wh = build_undistort_maps(
                cam["calib"], 0.0, 1.0, target_hfov, src_size=native,
                out_size=(OUT_W, OUT_H))
        frames = []
        for jpg in jpgs:
            img = cv2.imread(str(jpg))
            if (img.shape[1], img.shape[0]) != native:
                img = cv2.resize(img, native)
            out = (cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
                   if undistort else cv2.resize(img, (OUT_W, OUT_H)))
            frames.append(out[:, :, ::-1].copy())   # BGR->RGB
    n = min(len(frames), len(ts))
    return frames[:n], ts[:n]


def process_umi_episode(mcap_path, tcp_offset, R_align, robot_gripper_max,
                        target_hfov, undistort=True):
    """UMI mcap -> {wrist, env, state, actions, env_mask}（10Hz, W' 系）。"""
    wrist_frames, cam_ts = read_wrist_frames(mcap_path, UMI_MAIN_CAM, target_hfov, undistort)
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
def load_source_tasks(troot):
    """读遥操作源数据集的 meta/tasks.jsonl -> {task_index: task串}。"""
    tasks = {}
    f = troot / "meta" / "tasks.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line); tasks[int(d["task_index"])] = d["task"]
    return tasks


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


def process_teleop_episode(parquet, img_mp4, wrist_mp4, gclose, gopen, tcp_offset):
    """遥操作 episode -> {wrist, env, state, actions, env_mask}（W' 系）。
    夹爪按该爪 [gclose, gopen] 归一化; tcp_offset 把位姿锚到指尖(parallel=0 即原状)。"""
    import pyarrow.parquet as pq
    t = pq.read_table(parquet).to_pydict()
    if "state_eef" not in t:
        print(f"  ✗ {parquet.name}: 无 state_eef（旧格式?）, 跳过")
        return None
    task_index = int(t["task_index"][0]) if "task_index" in t else None
    se = np.array(t["state_eef"], dtype=np.float64)   # (N,8) pos3+quat4(xyzw)+follow(实际)grip
    pos_base = se[:, :3]
    quat = se[:, 3:7]
    grip_follow = se[:, 7]
    R_tool = np.stack([quat_to_rot(*q) for q in quat])    # 已是工具系(base)
    off = np.asarray(tcp_offset, float)
    pos_tip = pos_base + np.einsum("nij,j->ni", R_tool, off)   # EE点 -> 指尖(off=0 即不变)
    span = (gopen - gclose) or 1.0

    pos, rot6d = build_wprime(pos_tip, R_tool)
    grip_state = np.clip((grip_follow - gclose) / span, 0.0, 1.0).astype(np.float32)
    state, action = make_state_action(pos, rot6d, grip_state)
    # 关键修正: 动作夹爪改用 lead(遥操作指令)夹爪。follow(实际)抓取时只能闭到积木宽度、
    # 表达不出"夹紧力", 模型学了就会"贴着积木宽度但不夹住"。lead 抓取时→~0, 才会真夹紧。
    if "actions_eef" in t:
        ae = np.array(t["actions_eef"], dtype=np.float64)   # (N,8) lead pose + lead(指令)grip
        grip_lead = np.clip((ae[:, 7] - gclose) / span, 0.0, 1.0).astype(np.float32)
        m = min(len(action), len(grip_lead))
        action[:m, 9] = grip_lead[:m]                       # 只换动作的夹爪维, 位姿仍用 next-state(已验证)
    else:
        print(f"  ⚠ {parquet.name}: 无 actions_eef, 动作夹爪退回 follow(可能夹不紧)")

    env = read_video_frames(img_mp4)
    wrist = read_video_frames(wrist_mp4)
    n = min(len(state), len(env), len(wrist))
    if n < 5:
        print(f"  ✗ {parquet.name}: 帧数不足({n}), 跳过")
        return None
    return {"wrist": [cv2.resize(w, (OUT_W, OUT_H)) for w in wrist[:n]],
            "env": [cv2.resize(e, (OUT_W, OUT_H)) for e in env[:n]],
            "state": state[:n], "actions": action[:n],
            "env_mask": np.ones(n, np.float32), "task_index": task_index}


def _arm_state_action(se_arm, ae_arm, gclose, gopen, tcp_offset):
    """单条臂的 (N,8)=pos3+quat4(xyzw)+grip -> 10D state/actions(W' 系, 该爪归一化)。
    与 process_teleop_episode 的单臂逻辑一致(含动作夹爪改用 lead 指令夹爪)。"""
    se = np.asarray(se_arm, np.float64)
    R_tool = np.stack([quat_to_rot(*q) for q in se[:, 3:7]])
    off = np.asarray(tcp_offset, float)
    pos_tip = se[:, :3] + np.einsum("nij,j->ni", R_tool, off)
    span = (gopen - gclose) or 1.0
    pos, rot6d = build_wprime(pos_tip, R_tool)
    grip_state = np.clip((se[:, 7] - gclose) / span, 0.0, 1.0).astype(np.float32)
    state, action = make_state_action(pos, rot6d, grip_state)
    if ae_arm is not None:                                   # 动作夹爪用 lead(指令)夹爪
        ae = np.asarray(ae_arm, np.float64)
        grip_lead = np.clip((ae[:, 7] - gclose) / span, 0.0, 1.0).astype(np.float32)
        m = min(len(action), len(grip_lead)); action[:m, 9] = grip_lead[:m]
    return state, action


def process_dualarm_episode(parquet, env_mp4, wrist0_mp4, wrist1_mp4, gp0, gp1):
    """双臂 episode(collect_data 双臂格式) -> 双臂统一 schema。
    需要 parquet 含 state_eef_0/1(各 N×8)+ actions_eef_0/1; 三路视频(env + 两手眼)。
    臂0/臂1 各自建 W'、各自爪归一化, 拼成 20D; arm1_mask=1。"""
    import pyarrow.parquet as pq
    t = pq.read_table(parquet).to_pydict()
    if "state_eef_0" not in t or "state_eef_1" not in t:
        print(f"  ✗ {parquet.name}: 缺 state_eef_0/1(不是双臂格式?), 跳过")
        return None
    task_index = int(t["task_index"][0]) if "task_index" in t else None
    s0, a0 = _arm_state_action(t["state_eef_0"], t.get("actions_eef_0"),
                               gp0["close"], gp0["open"], gp0["tcp_offset"])
    s1, a1 = _arm_state_action(t["state_eef_1"], t.get("actions_eef_1"),
                               gp1["close"], gp1["open"], gp1["tcp_offset"])
    n = min(len(s0), len(s1))
    state = np.concatenate([s0[:n], s1[:n]], axis=1)         # (n,20)
    actions = np.concatenate([a0[:n], a1[:n]], axis=1)
    env = read_video_frames(env_mp4)
    wrist0 = read_video_frames(wrist0_mp4)
    wrist1 = read_video_frames(wrist1_mp4)
    n = min(n, len(env), len(wrist0), len(wrist1))
    if n < 5:
        print(f"  ✗ {parquet.name}: 帧数不足({n}), 跳过")
        return None
    return {"env": [cv2.resize(e, (OUT_W, OUT_H)) for e in env[:n]],
            "wrist": [cv2.resize(w, (OUT_W, OUT_H)) for w in wrist0[:n]],
            "wrist_1": [cv2.resize(w, (OUT_W, OUT_H)) for w in wrist1[:n]],
            "state": state[:n], "actions": actions[:n],
            "env_mask": np.ones(n, np.float32), "arm1_mask": np.ones(n, np.float32),
            "task_index": task_index}


# ---------------------------------------------------------------------------
# 打包
# ---------------------------------------------------------------------------
def create_dataset(out_dir, repo_id, dual=False):
    """dual=False: 单臂 10D 传统 schema(兼容现有 config); dual=True: 双臂 20D schema。"""
    import lerobot.common.datasets.lerobot_dataset as lr_ds
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    if not getattr(lr_ds, "_h264_patched", False):
        _orig = lr_ds.encode_video_frames
        def _h264(*a, **k):
            k.setdefault("vcodec", "h264"); return _orig(*a, **k)
        lr_ds.encode_video_frames = _h264
        lr_ds._h264_patched = True

    vid = {"dtype": "video", "shape": (OUT_H, OUT_W, 3), "names": ["height", "width", "channel"]}
    if dual:
        features = {
            "image": vid, "wrist_image": vid,     # 臂0 手眼
            "wrist_image_1": vid,                 # 臂1 手眼(单臂=空图)
            # 20D = 臂0(pos3+rot6d6+grip1) + 臂1(同10, 各自 W' 系)。单臂样本臂1=rest。
            "state":   {"dtype": "float32", "shape": (20,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (20,), "names": ["actions"]},
            "env_mask":  {"dtype": "float32", "shape": (1,), "names": ["env_mask"]},
            "arm1_mask": {"dtype": "float32", "shape": (1,), "names": ["arm1_mask"]},  # 1=双臂/0=单臂
            "gripper_id_0": {"dtype": "int64", "shape": (1,), "names": ["gripper_id_0"]},
            "gripper_id_1": {"dtype": "int64", "shape": (1,), "names": ["gripper_id_1"]},
        }
    else:
        features = {
            "image": vid, "wrist_image": vid,
            "state":   {"dtype": "float32", "shape": (10,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (10,), "names": ["actions"]},
            "env_mask": {"dtype": "float32", "shape": (1,), "names": ["env_mask"]},
            # 方案C: 该帧用哪把爪(0=parallel, 1=get, ...)。与 grippers.npz names / config 一致。
            "gripper_id": {"dtype": "int64", "shape": (1,), "names": ["gripper_id"]},
        }
    return LeRobotDataset.create(
        repo_id=repo_id, robot_type="airbot_play", fps=TARGET_FPS, root=out_dir,
        features=features, image_writer_threads=4, image_writer_processes=2,
    )


def make_server_compatible(out_dir):
    """把本地 datasets>=4.0 写的特征类型 'List' 降级为 'Sequence'，
    让服务器(datasets 3.x / lerobot 0.1.0)能直接读。打包末尾自动调用——
    保持“一个脚本出可直接训练的数据集”，不另开脚本。只改 schema 元数据，不动数据。"""
    import pyarrow.parquet as pq
    n = 0
    for pf in sorted((out_dir / "data").rglob("*.parquet")):
        t = pq.read_table(pf)
        md = dict(t.schema.metadata or {})
        ch = False
        for k, v in list(md.items()):
            if b'"List"' in v:
                md[k] = v.replace(b'"List"', b'"Sequence"'); ch = True
        if ch:
            pq.write_table(t.replace_schema_metadata(md), pf); n += 1
    info = out_dir / "meta" / "info.json"
    if info.exists():
        s = info.read_text(encoding="utf-8")
        if '"List"' in s:
            info.write_text(s.replace('"List"', '"Sequence"'), encoding="utf-8")
    if n:
        print(f"  ✓ 兼容处理: {n} 个 parquet 特征类型 List->Sequence (服务器 datasets 3.x 可读)")


def write_episode(dataset, ep, task, gid0=0, gid1=0, dual=False):
    """写入一条 episode。dual=True(20D): 单臂来源须先经 to_dualarm_from_single();
    dual=False(10D): 直接写 ep 的 10D state/actions + gripper_id=gid0。"""
    for i in range(len(ep["state"])):
        if dual:
            frame = {
                "image": ep["env"][i], "wrist_image": ep["wrist"][i], "wrist_image_1": ep["wrist_1"][i],
                "state": ep["state"][i], "actions": ep["actions"][i],
                "env_mask": np.array([ep["env_mask"][i]], np.float32),
                "arm1_mask": np.array([ep["arm1_mask"][i]], np.float32),
                "gripper_id_0": np.array([gid0], np.int64),
                "gripper_id_1": np.array([gid1], np.int64),
                "task": task,
            }
        else:
            frame = {
                "image": ep["env"][i], "wrist_image": ep["wrist"][i],
                "state": ep["state"][i], "actions": ep["actions"][i],
                "env_mask": np.array([ep["env_mask"][i]], np.float32),
                "gripper_id": np.array([gid0], np.int64),
                "task": task,
            }
        dataset.add_frame(frame)
    dataset.save_episode()


def main():
    ap = argparse.ArgumentParser(description="阶段4: UMI + 遥操作 -> 统一联合训练 LeRobot")
    ap.add_argument("--umi-dir", nargs="+", default=None,
                    help="UMI VIO mcap 目录(可多个; 建议一个任务一个目录)。--skip-umi 时可省")
    ap.add_argument("--teleop-dir", required=True, nargs="+",
                    help="遥操作 LeRobot 数据集根目录(可传多个, 如积木目录 擦黑板目录)")
    ap.add_argument("--skip-umi", action="store_true",
                    help="只打包遥操作（测试③：纯遥操作 EEF，头部相机全程有效）")
    ap.add_argument("--dual-arm", action="store_true",
                    help="输出双臂 20D schema(state20 + wrist_image_1 + arm1_mask + gripper_id_0/1)。"
                         "不给=单臂 10D 传统 schema(state10+gripper_id, 兼容 pi05_cotrain_eef_grip)。"
                         "--dualarm-dir 必须配 --dual-arm")
    ap.add_argument("-o", "--output", required=True, help="输出合并数据集目录")
    ap.add_argument("--repo-id", default="cotrain_eef")
    ap.add_argument("--umi-task", nargs="+", default=["wipe the blackboard"],
                    help="每个 --umi-dir 的 prompt: 给1个=全部广播, 或与 --umi-dir 数等长(多任务)")
    # 方案C: 夹爪与"采集方式无关", 按来源显式指定爪名。名字须在 --gripper-names 里,
    # 其下标即存进数据的 gripper_id(须与 grippers.npz names / config gripper_names 一致)。
    #   --gripper-names   定义爪名->id 的顺序 (默认 parallel get)
    #   --umi-gripper     UMI 这批用哪把爪 (默认 get; 未来 UMI 换爪改这里)
    #   --teleop-gripper  每个 --teleop-dir 用哪把爪: 给1个=全部广播, 或与目录数等长
    #                     (默认 parallel; 未来在臂上装 GET 遥操作就传 get)
    ap.add_argument("--gripper-names", nargs="+", default=["parallel", "get"])
    ap.add_argument("--umi-gripper", nargs="+", default=["get"],
                    help="每个 --umi-dir 的爪: 给1个=广播, 或与 --umi-dir 数等长")
    ap.add_argument("--teleop-gripper", nargs="+", default=["parallel"])
    ap.add_argument("--teleop-task", default=None,
                    help="覆盖所有 teleop 的 prompt; 默认 None=从各源 tasks.jsonl 按 task_index 自动读")
    # 双臂来源(collect_data 双臂格式: parquet 含 state_eef_0/1 + actions_eef_0/1, 三路视频
    # image / wrist_image / wrist_image_1)。单臂来源会自动补臂1=rest+mask。
    ap.add_argument("--dualarm-dir", nargs="+", default=None,
                    help="双臂 LeRobot 数据集根目录(可多个)")
    ap.add_argument("--dualarm-gripper", nargs=2, default=["get", "parallel"],
                    metavar=("ARM0", "ARM1"),
                    help="双臂: 臂0(主/持刀) 臂1(按压/拉开) 各自爪名; 应用到所有 --dualarm-dir")
    ap.add_argument("--dualarm-task", default=None,
                    help="覆盖双臂 prompt; 默认从各源 tasks.jsonl 按 task_index 读")
    ap.add_argument("--tcp-offset", nargs=3, type=float, default=list(DEFAULT_TCP_OFFSET))
    ap.add_argument("--align-rpy", nargs=3, type=float, default=list(DEFAULT_ALIGN_RPY))
    ap.add_argument("--robot-gripper-max", type=float, default=DEFAULT_ROBOT_GRIPPER_MAX)
    ap.add_argument("--target-hfov", type=float, default=TARGET_HFOV)
    ap.add_argument("--no-undistort", action="store_true",
                    help="UMI 不去畸变: 用原生鱼眼 resize 到 640x480(保留全广角), "
                         "先用鱼眼原生画面训练时开此项")
    ap.add_argument("--limit", type=int, default=None, help="每来源最多处理几条(调试)")
    ap.add_argument("--verify", action="store_true", help="打包后重载抽检")
    args = ap.parse_args()

    # 夹爪名 -> id (下标), 与 grippers.npz / config.gripper_names 一致
    name_to_id = {n: i for i, n in enumerate(args.gripper_names)}

    def gid_of(name):
        if name not in name_to_id:
            sys.exit(f"未知夹爪 '{name}', 须在 --gripper-names {args.gripper_names} 中")
        return name_to_id[name]

    def broadcast(lst, n, what):
        if len(lst) == 1:
            return lst * n
        if len(lst) != n:
            sys.exit(f"{what} 须为1个(广播)或与目录数({n})等长, 当前 {len(lst)} 个")
        return lst

    # 遥操作: 每目录的爪(prompt 由各目录 tasks.jsonl 自带)
    teleop_gids = [gid_of(n) for n in broadcast(args.teleop_gripper, len(args.teleop_dir), "--teleop-gripper")]
    # UMI: 每目录的 prompt + 爪(mcap 无 prompt, 在此按目录指定; 多任务=多目录)
    umi_dirs = [] if (args.skip_umi or not args.umi_dir) else list(args.umi_dir)
    umi_tasks = broadcast(args.umi_task, len(umi_dirs), "--umi-task") if umi_dirs else []
    umi_gids = [gid_of(n) for n in broadcast(args.umi_gripper, len(umi_dirs), "--umi-gripper")] if umi_dirs else []
    print(f"夹爪映射: names={args.gripper_names}")
    print(f"  UMI    -> {list(zip(umi_dirs, [args.gripper_names[g] for g in umi_gids], umi_tasks))}")
    print(f"  teleop -> {list(zip(args.teleop_dir, [args.gripper_names[g] for g in teleop_gids]))}")

    dual = args.dual_arm
    if args.dualarm_dir and not dual:
        sys.exit("--dualarm-dir 需要 --dual-arm(20D schema); 单臂 10D 不能含双臂数据")
    print(f"输出 schema: {'双臂 20D' if dual else '单臂 10D(兼容 pi05_cotrain_eef_grip)'}")

    R_align = rpy_to_rot(*args.align_rpy)
    out_dir = Path(args.output).expanduser().resolve()
    dataset = create_dataset(out_dir, args.repo_id, dual)

    stats = {"umi": [0, 0], "teleop": [0, 0], "dual": [0, 0]}   # [episodes, frames]

    # ---- UMI (每目录: 自己的 prompt + 爪; 多任务=多目录) ----
    if not umi_dirs:
        print("\n=== UMI: 跳过 (--skip-umi) ===")
    print(f"\n=== UMI: {len(umi_dirs)} 个目录 ===") if umi_dirs else None
    for udir, utask, ugid in zip(umi_dirs, umi_tasks, umi_gids):
        mcaps = sorted(Path(udir).expanduser().glob("*.mcap"))
        if args.limit:
            mcaps = mcaps[:args.limit]
        print(f"  [{Path(udir).name}] {len(mcaps)} 条  task='{utask}'  爪={args.gripper_names[ugid]}")
        for m in mcaps:
            ep = process_umi_episode(m, args.tcp_offset, R_align,
                                     args.robot_gripper_max, args.target_hfov,
                                     undistort=not args.no_undistort)
            if ep is None:
                continue
            write_episode(dataset, to_dualarm_from_single(ep) if dual else ep,
                          utask, gid0=ugid, gid1=0, dual=dual)
            stats["umi"][0] += 1; stats["umi"][1] += len(ep["state"])
            print(f"    ✓ {m.name}: {len(ep['state'])} 帧")

    # ---- 遥操作(可多个目录; prompt 默认从各源 tasks.jsonl 按 task_index 自动读) ----
    troots = [Path(d).expanduser() for d in args.teleop_dir]
    print(f"\n=== 遥操作: {len(troots)} 个目录 ===")
    for ti_dir, troot in enumerate(troots):
        teleop_gid = teleop_gids[ti_dir]
        tgp = get_params(args.gripper_names[teleop_gid])   # 该目录爪的开合范围+指尖偏移
        src_tasks = load_source_tasks(troot)
        pqs = sorted((troot / "data").rglob("episode_*.parquet"))
        if args.limit:
            pqs = pqs[:args.limit]
        print(f"  [{troot.name}] {len(pqs)} 条  源 tasks={list(src_tasks.values())}")
        for pqf in pqs:
            stem = pqf.stem
            img_mp4 = troot / "videos" / pqf.parent.name / "image" / f"{stem}.mp4"
            wrist_mp4 = troot / "videos" / pqf.parent.name / "wrist_image" / f"{stem}.mp4"
            if not img_mp4.exists() or not wrist_mp4.exists():
                print(f"    ✗ {stem}: 缺视频, 跳过"); continue
            ep = process_teleop_episode(pqf, img_mp4, wrist_mp4,
                                        tgp["close"], tgp["open"], tgp["tcp_offset"])
            if ep is None:
                continue
            # prompt: --teleop-task 覆盖优先, 否则按本条 task_index 从源 tasks.jsonl 读
            task = args.teleop_task
            if task is None:
                ti = ep.get("task_index")
                task = src_tasks.get(ti) if ti is not None else None
                if task is None:
                    print(f"    ✗ {stem}: 源无 task(index={ti}), 跳过"); continue
            write_episode(dataset, to_dualarm_from_single(ep) if dual else ep,
                          task, gid0=teleop_gid, gid1=0, dual=dual)
            stats["teleop"][0] += 1; stats["teleop"][1] += len(ep["state"])
            print(f"    ✓ {stem}: {len(ep['state'])} 帧  task='{task}'")

    # ---- 双臂(collect_data 双臂格式: state_eef_0/1 + 两手眼视频) ----
    if args.dualarm_dir:
        gp0 = get_params(args.dualarm_gripper[0]); gp1 = get_params(args.dualarm_gripper[1])
        gid0 = gid_of(args.dualarm_gripper[0]); gid1 = gid_of(args.dualarm_gripper[1])
        droots = [Path(d).expanduser() for d in args.dualarm_dir]
        print(f"\n=== 双臂: {len(droots)} 个目录  臂0={args.dualarm_gripper[0]} 臂1={args.dualarm_gripper[1]} ===")
        for droot in droots:
            src_tasks = load_source_tasks(droot)
            pqs = sorted((droot / "data").rglob("episode_*.parquet"))
            if args.limit:
                pqs = pqs[:args.limit]
            print(f"  [{droot.name}] {len(pqs)} 条")
            for pqf in pqs:
                stem = pqf.stem; sub = pqf.parent.name
                env_mp4 = droot / "videos" / sub / "image" / f"{stem}.mp4"
                w0 = droot / "videos" / sub / "wrist_image" / f"{stem}.mp4"
                w1 = droot / "videos" / sub / "wrist_image_1" / f"{stem}.mp4"
                if not (env_mp4.exists() and w0.exists() and w1.exists()):
                    print(f"    ✗ {stem}: 缺视频(需 image/wrist_image/wrist_image_1), 跳过"); continue
                ep = process_dualarm_episode(pqf, env_mp4, w0, w1, gp0, gp1)
                if ep is None:
                    continue
                task = args.dualarm_task
                if task is None:
                    ti = ep.get("task_index")
                    task = src_tasks.get(ti) if ti is not None else None
                    if task is None:
                        print(f"    ✗ {stem}: 源无 task, 跳过"); continue
                write_episode(dataset, ep, task, gid0=gid0, gid1=gid1, dual=True)
                stats["dual"][0] += 1; stats["dual"][1] += len(ep["state"])
                print(f"    ✓ {stem}: {len(ep['state'])} 帧  task='{task}'")

    uf, tf, df = stats["umi"][1], stats["teleop"][1], stats["dual"][1]
    tot = uf + tf + df
    print(f"\n===== 打包完成 =====")
    print(f"  UMI:      {stats['umi'][0]} 条 / {uf} 帧 ({uf/max(tot,1)*100:.0f}%)")
    print(f"  单臂遥操作: {stats['teleop'][0]} 条 / {tf} 帧 ({tf/max(tot,1)*100:.0f}%)")
    print(f"  双臂:      {stats['dual'][0]} 条 / {df} 帧 ({df/max(tot,1)*100:.0f}%)")
    print(f"  输出: {out_dir}")

    make_server_compatible(out_dir)   # 自动降级 List->Sequence, 服务器可直接训练

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
    if st.shape[1] >= 20:                     # 双臂 20D
        a1 = np.array(t.get("arm1_mask", [[0]]))
        g0 = np.array(t.get("gripper_id_0", [[-1]])); g1 = np.array(t.get("gripper_id_1", [[-1]]))
        print(f"  arm1_mask[0]={a1[0]}  gripper_id_0[0]={g0[0]}  gripper_id_1[0]={g1[0]}")
        print(f"  臂0 state[0]={np.round(st[0][:10],3).tolist()}")
        print(f"  臂1 state[0]={np.round(st[0][10:],3).tolist()}  (单臂应=rest[0,0,0,1,0,0,0,1,0,0])")
    else:                                     # 单臂 10D
        gid = np.array(t.get("gripper_id", [[-1]]))
        print(f"  gripper_id[0]={gid[0]}  state[0]={np.round(st[0],3).tolist()}")
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
