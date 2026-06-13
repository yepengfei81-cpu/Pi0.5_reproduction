#!/usr/bin/env python3
"""
UMI mcap 同步可视化：左侧 3D 末端轨迹 + 右侧三路相机画面，按时间同步播放。

- 3D 轨迹来自 /robot0/vio/eef_pose（绝对位姿，world 系），含末端朝向坐标三轴。
- 三路相机来自 foxglove.CompressedImage(h264)，用 ffmpeg 解成 jpg 帧后读入。
- 用位姿时间戳做主时钟，每一帧取最接近时刻的相机画面与夹爪开合。

依赖:
    pip install mcap mcap-protobuf-support imageio matplotlib numpy
    系统需安装 ffmpeg

用法:
    # 交互播放（需要图形界面）
    python visualize_episode.py xxx_vio.mcap

    # 无显示环境/想存成视频
    python visualize_episode.py xxx_vio.mcap --save out.mp4

    # 调小画面分辨率省内存
    python visualize_episode.py xxx_vio.mcap --width 240 --save out.mp4
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def quat_to_rot(x, y, z, w):
    """四元数 (x,y,z,w) -> 3x3 旋转矩阵。"""
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def read_mcap(mcap_path: Path, pose_topic: str, gripper_topic: str):
    """读出相机话题、相机 H.264 流+时间戳、位姿、夹爪。"""
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
    cam_topics = sorted(
        ch.topic for ch in summary.channels.values()
        if (s := summary.schemas.get(ch.schema_id)) and s.name == "foxglove.CompressedImage"
    )

    cam_bytes = {t: bytearray() for t in cam_topics}
    cam_times = {t: [] for t in cam_topics}
    pose_t, pose_xyz, pose_quat = [], [], []
    grip_t, grip_v = [], []

    topics = cam_topics + [pose_topic, gripper_topic]
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _s, ch, msg, dec in reader.iter_decoded_messages(topics=topics):
            if ch.topic in cam_bytes:
                cam_bytes[ch.topic].extend(bytes(dec.data))
                cam_times[ch.topic].append(msg.log_time)
            elif ch.topic == pose_topic:
                p, o = dec.pose.position, dec.pose.orientation
                pose_t.append(msg.log_time)
                pose_xyz.append([p.x, p.y, p.z])
                pose_quat.append([o.x, o.y, o.z, o.w])
            elif ch.topic == gripper_topic:
                grip_t.append(msg.log_time)
                grip_v.append(dec.value)

    return {
        "cam_topics": cam_topics,
        "cam_bytes": cam_bytes,
        "cam_times": {t: np.array(v) for t, v in cam_times.items()},
        "pose_t": np.array(pose_t),
        "pose_xyz": np.array(pose_xyz, dtype=float),
        "pose_quat": np.array(pose_quat, dtype=float),
        "grip_t": np.array(grip_t),
        "grip_v": np.array(grip_v, dtype=float),
    }


def decode_frames(h264: bytearray, width: int, fps: int, tmp: Path, tag: str):
    """H.264 裸流 -> jpg 帧序列 -> 读成 numpy 列表。"""
    raw = tmp / f"{tag}.h264"
    raw.write_bytes(h264)
    frame_dir = tmp / tag
    frame_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-r", str(fps), "-i", str(raw),
         "-vf", f"scale={width}:-2", "-q:v", "3", str(frame_dir / "%05d.jpg")],
        check=True,
    )
    return [imageio.imread(p) for p in sorted(frame_dir.glob("*.jpg"))]


def nearest_idx(times: np.ndarray, t: int) -> int:
    if len(times) == 0:
        return 0
    i = int(np.searchsorted(times, t))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i - 1 if abs(times[i - 1] - t) <= abs(times[i] - t) else i


def main():
    ap = argparse.ArgumentParser(description="UMI 轨迹 + 三路相机 同步可视化")
    ap.add_argument("input", type=str, help="mcap 文件")
    ap.add_argument("--save", type=str, default=None, help="存成 mp4（不弹窗）")
    ap.add_argument("--width", type=int, default=320, help="单路画面宽度 (默认 320)")
    ap.add_argument("--fps", type=int, default=30, help="解码/播放帧率 (默认 30)")
    ap.add_argument("--pose-topic", type=str, default="/robot0/vio/eef_pose")
    ap.add_argument("--gripper-topic", type=str, default="/robot0/sensor/magnetic_encoder")
    args = ap.parse_args()

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    mcap_path = Path(args.input).expanduser().resolve()
    if not mcap_path.is_file():
        sys.exit(f"文件不存在: {mcap_path}")

    print("读取 mcap ...")
    d = read_mcap(mcap_path, args.pose_topic, args.gripper_topic)
    cam_topics = d["cam_topics"]
    if not cam_topics:
        sys.exit("未找到相机话题")
    if len(d["pose_t"]) == 0:
        sys.exit(f"未找到位姿话题 {args.pose_topic}")
    print(f"  相机: {cam_topics}")
    print(f"  位姿: {len(d['pose_t'])} 帧, 夹爪: {len(d['grip_t'])} 帧")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("ffmpeg 解码相机帧 ...")
        cam_frames = {}
        for i, t in enumerate(cam_topics):
            frames = decode_frames(d["cam_bytes"][t], args.width, args.fps, tmp, f"cam{i}")
            n = min(len(frames), len(d["cam_times"][t]))
            cam_frames[t] = (frames[:n], d["cam_times"][t][:n])
            print(f"  {t}: {n} 帧")

        xyz = d["pose_xyz"]
        pose_t = d["pose_t"]
        # 朝向三轴长度 = 轨迹包围盒对角线的 8%
        span = np.linalg.norm(xyz.max(0) - xyz.min(0)) or 1.0
        axis_len = 0.08 * span

        # ---- 画布布局: 左侧大 3D, 右侧三路画面竖排 ----
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(len(cam_topics), 2, width_ratios=[1.4, 1])
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-", color="lightgray", lw=1)
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
        ax3d.set_title("EEF trajectory (world)")
        trail, = ax3d.plot([], [], [], "-", color="tab:blue", lw=2)
        dot, = ax3d.plot([], [], [], "o", color="red", ms=6)
        triad = [ax3d.plot([], [], [], c=c, lw=2)[0] for c in ("r", "g", "b")]

        img_axes, img_handles = [], []
        for i, t in enumerate(cam_topics):
            ax = fig.add_subplot(gs[i, 1])
            ax.axis("off")
            cam = t.strip("/").split("/")[-2]
            ax.set_title(cam, fontsize=9)
            h = ax.imshow(cam_frames[t][0][0])
            img_axes.append(ax); img_handles.append(h)

        sup = fig.suptitle("")

        def update(k):
            t = pose_t[k]
            # 轨迹
            trail.set_data(xyz[:k + 1, 0], xyz[:k + 1, 1])
            trail.set_3d_properties(xyz[:k + 1, 2])
            dot.set_data([xyz[k, 0]], [xyz[k, 1]])
            dot.set_3d_properties([xyz[k, 2]])
            # 朝向三轴
            R = quat_to_rot(*d["pose_quat"][k])
            o = xyz[k]
            for a in range(3):
                e = o + R[:, a] * axis_len
                triad[a].set_data([o[0], e[0]], [o[1], e[1]])
                triad[a].set_3d_properties([o[2], e[2]])
            # 相机画面（取最接近时刻）
            for h, tp in zip(img_handles, cam_topics):
                frames, times = cam_frames[tp]
                h.set_array(frames[nearest_idx(times, t)])
            # 夹爪
            g = d["grip_v"][nearest_idx(d["grip_t"], t)] if len(d["grip_t"]) else float("nan")
            sup.set_text(f"t={ (t-pose_t[0])/1e9:5.2f}s   gripper={g*1000:5.1f} mm   "
                         f"frame {k+1}/{len(pose_t)}")
            return [trail, dot, *triad, *img_handles, sup]

        anim = FuncAnimation(fig, update, frames=len(pose_t),
                             interval=1000 / args.fps, blit=False)

        if args.save:
            out = Path(args.save).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"保存到 {out} ...")
            anim.save(str(out), fps=args.fps, dpi=100)
            print("完成")
        else:
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
