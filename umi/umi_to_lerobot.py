#!/usr/bin/env python3
"""
UMI mcap -> openpi/LeRobot 训练数据转换。

最终目标：把商用 UMI 导出的 VIO mcap（三路鱼眼 H.264 + VIO 末端位姿 + 夹爪
磁编码器）转换成可与遥操作数据联合训练 pi05 的 LeRobot 数据集。

转换链各阶段（逐步实现，当前进度见标注）：
  [✓] 1. 相机去鱼眼：主相机（默认 camera0，即 UMI 中间那路；camera1/2 是
         左右对，内参几乎相同）equidistant 鱼眼 -> 针孔图像，与机械臂 D405
         手腕相机在同一个相机槽里风格对齐。
         目标视场用 umi/compare_fov.py 与 D405 实拍对比确定（实测 79°）。
  [✓] 2. 位姿提取（--poses）：
         VIO eef_pose（原点=camera0 光心，world 系 z=重力反方向）
           -> 右乘 Trans(tcp_offset) 重表达到指尖 TCP
              （tcp_offset 用 umi/pivot_calib.py 标定，本套设备实测
               [0.1274, -0.0008, -0.0174]，残差 RMS 4.5mm）
           -> 构造 W' 系（原点=首帧指尖位置, z=重力反方向, 首帧 yaw 归零；
              pitch/roll 保持重力参考的绝对量, 位置/yaw 相对）
           -> pos(3) + rot6d(6) + gripper(1) 序列存 npz + 诊断图
         内置自检：6D 旋转往返误差、首帧位置/yaw 归零、重力方向语义。
  [ ] 3. 夹爪：磁编码器读数(米) -> 物理开口标定 -> 共享参考归一化 [0,1]。
  [ ] 4. 打包 LeRobot 数据集（base_0_rgb 置零 + image_mask=False，
         wrist 槽放去畸变后的主相机画面）。

当前用法（阶段 1：导出去畸变视频，确认畸变去对了）:
    # 标准用法：去畸变并把视场裁到与 D405 一致（本套 D405 实测 HFOV=78.6°，
    # 用 compare_fov.py --d405-capture 测得）。默认输出到 umi/output/
    python umi_to_lerobot.py xxx_vio.mcap --target-hfov 79

    # 输出"原图|去畸变"左右对比，方便肉眼确认
    python umi_to_lerobot.py xxx_vio.mcap --target-hfov 87 -o ./out --side-by-side

    # 不指定目标视场时用 balance/fov-scale 控制（调试用）
    python umi_to_lerobot.py xxx_vio.mcap --fov-scale 0.5

    # 处理全部三路相机
    python umi_to_lerobot.py xxx_vio.mcap --cameras all

阶段 2 用法（位姿提取，输出 <stem>_poses.npz + 诊断图）:
    python umi_to_lerobot.py xxx_vio.mcap --poses
    # 换 TCP 偏移（重新标定后）
    python umi_to_lerobot.py xxx_vio.mcap --poses --tcp-offset 0.13 0.0 -0.02

依赖（在装了 openpi + 机械臂 SDK 的 conda 环境里就有）:
    pip install mcap mcap-protobuf-support opencv-python numpy
    系统需安装 ffmpeg
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader, NonSeekingReader
from mcap_protobuf.decoder import DecoderFactory

CAMERA_SCHEMA = "foxglove.CompressedImage"
CALIB_SCHEMA = "foxglove.CameraCalibration"


def _iter_decoded(mcap_path: Path, topics):
    """优先用带索引的 make_reader；若读不出消息（部分原始 mcap 没写 summary
    索引会静默返回 0 条），自动回退到线性扫描的 NonSeekingReader。"""
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        got_any = False
        for item in reader.iter_decoded_messages(topics=topics):
            got_any = True
            yield item
        if got_any:
            return
    # fallback: 线性扫描
    with open(mcap_path, "rb") as f:
        reader = NonSeekingReader(f, decoder_factories=[DecoderFactory()])
        tset = set(topics) if topics else None
        for sch, ch, msg, dec in reader.iter_decoded_messages():
            if tset is None or ch.topic in tset:
                yield sch, ch, msg, dec


def camera_prefix(topic: str) -> str:
    """/robot0/sensor/camera0/compressed -> /robot0/sensor/camera0"""
    return topic.rsplit("/", 1)[0]


def read_mcap_cameras(mcap_path: Path):
    """返回 {prefix: {"name", "img_topic", "calib", "h264"(bytearray)}}。"""
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
    img_topics, calib_topics = [], []
    for ch in summary.channels.values():
        sch = summary.schemas.get(ch.schema_id)
        if not sch:
            continue
        if sch.name == CAMERA_SCHEMA:
            img_topics.append(ch.topic)
        elif sch.name == CALIB_SCHEMA and ch.topic.endswith("/camera_info"):
            calib_topics.append(ch.topic)

    cams = {}
    for it in sorted(img_topics):
        pre = camera_prefix(it)
        cams[pre] = {
            "name": pre.rsplit("/", 1)[-1],
            "img_topic": it,
            "calib_topic": pre + "/camera_info",
            "calib": None,
            "h264": bytearray(),
        }

    want = [c["img_topic"] for c in cams.values()] + \
           [c["calib_topic"] for c in cams.values()]
    calib_by_topic = {}
    for _s, ch, _m, dec in _iter_decoded(mcap_path, want):
        pre = camera_prefix(ch.topic)
        if ch.topic.endswith("/camera_info"):
            if ch.topic not in calib_by_topic:   # 内参只取第一条
                calib_by_topic[ch.topic] = {
                    "model": str(dec.distortion_model),
                    "w": int(dec.width), "h": int(dec.height),
                    "K": np.array(dec.K, dtype=np.float64).reshape(3, 3),
                    "D": np.array(dec.D, dtype=np.float64).reshape(-1, 1),
                }
        elif pre in cams:
            cams[pre]["h264"].extend(bytes(dec.data))

    for pre, c in cams.items():
        c["calib"] = calib_by_topic.get(c["calib_topic"])
    return cams


def decode_h264(h264: bytearray, fps: int, tmp: Path, tag: str) -> list[Path]:
    """H.264 裸流 -> 原生分辨率 jpg 帧文件列表。"""
    raw = tmp / f"{tag}.h264"
    raw.write_bytes(h264)
    frame_dir = tmp / tag
    frame_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-r", str(fps),
         "-i", str(raw), "-q:v", "2", str(frame_dir / "%05d.jpg")],
        check=True,
    )
    return sorted(frame_dir.glob("*.jpg"))


def build_undistort_maps(calib: dict, balance: float, fov_scale: float,
                         target_hfov: float | None = None,
                         src_size: tuple[int, int] | None = None,
                         out_size: tuple[int, int] | None = None):
    """按畸变模型生成 remap 表，返回 (map1, map2, Knew, out_size)。

    target_hfov: 若给定（度），直接构造水平视场为该值的针孔新内参，
                 使输出视场与部署相机（D405）一致；否则用 balance/fov_scale 估计。
    src_size:    实际输入帧的分辨率。H.264 原生帧（如 1600x1300）比标定分辨率
                 （640x480）高时传原生尺寸：K 会按比例缩放到原生分辨率，
                 remap 直接从高清原图采样，输出清晰度远好于先缩小再去畸变。
                 （鱼眼畸变系数 D 作用在归一化坐标上，与分辨率无关，无需改。）
    out_size:    输出图像尺寸，默认与标定分辨率一致（640x480，对齐 D405）。
    """
    K, D = calib["K"].copy(), calib["D"]
    calib_size = (calib["w"], calib["h"])
    src_size = src_size or calib_size
    out_size = out_size or calib_size
    if src_size != calib_size:  # 把 K 缩放到实际输入帧的分辨率
        sx, sy = src_size[0] / calib_size[0], src_size[1] / calib_size[1]
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
    model = calib["model"].lower()
    eye = np.eye(3)
    if model in ("equidistant", "fisheye", "kannala_brandt"):
        Dk = D[:4].reshape(4, 1)
        if target_hfov is not None:
            fx = out_size[0] / (2 * math.tan(math.radians(target_hfov) / 2))
            Knew = np.array([[fx, 0, out_size[0] / 2],
                             [0, fx, out_size[1] / 2],
                             [0, 0, 1.0]])
        else:
            Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, Dk, src_size, eye, balance=balance,
                new_size=out_size, fov_scale=fov_scale)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            K, Dk, eye, Knew, out_size, cv2.CV_16SC2)
    else:  # plumb_bob / rational / 默认按普通针孔畸变处理
        Knew, _ = cv2.getOptimalNewCameraMatrix(K, D.reshape(-1), src_size, balance,
                                                out_size)
        m1, m2 = cv2.initUndistortRectifyMap(
            K, D.reshape(-1), eye, Knew, out_size, cv2.CV_16SC2)
    return m1, m2, Knew, out_size


# ---------------------------------------------------------------------------
# 阶段 2：位姿提取（TCP 重表达 + W' 系 + 6D 旋转）
# ---------------------------------------------------------------------------

# 本套设备 pivot 标定结果（pivot_calib.py, 残差 RMS 4.5mm）：
# VIO eef_pose 原点(=camera0 光心)到指尖的偏移，body 系 (x前 y左 z上)
DEFAULT_TCP_OFFSET = (0.1274, -0.0008, -0.0174)

POSE_TOPIC = "/robot0/vio/eef_pose"
GRIPPER_TOPIC = "/robot0/sensor/magnetic_encoder"


def quat_to_rot(x, y, z, w) -> np.ndarray:
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def rot_to_6d(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 6D 表示（前两列，连续无奇异，适合网络学习）。"""
    return np.concatenate([R[:, 0], R[:, 1]])


def rot6d_to_rot(d6: np.ndarray) -> np.ndarray:
    """6D -> 旋转矩阵（Gram-Schmidt 正交化）。"""
    a, b = d6[:3], d6[3:]
    x = a / np.linalg.norm(a)
    b = b - x * (x @ b)
    y = b / np.linalg.norm(b)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def euler_zyx_deg(R: np.ndarray) -> tuple[float, float, float]:
    """诊断用 zyx 欧拉角 (roll, pitch, yaw)，度。"""
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    pitch = math.degrees(math.asin(-np.clip(R[2, 0], -1, 1)))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    return roll, pitch, yaw


def nearest_idx(times: np.ndarray, t) -> int:
    i = int(np.searchsorted(times, t))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i - 1 if abs(times[i - 1] - t) <= abs(times[i] - t) else i


def read_pose_stream(mcap_path: Path, pose_topic: str, gripper_topic: str):
    ts, xyz, quat, gts, gv = [], [], [], [], []
    for _s, ch, msg, dec in _iter_decoded(mcap_path, [pose_topic, gripper_topic]):
        if ch.topic == pose_topic:
            p, o = dec.pose.position, dec.pose.orientation
            ts.append(msg.log_time)
            xyz.append([p.x, p.y, p.z])
            quat.append([o.x, o.y, o.z, o.w])
        else:
            gts.append(msg.log_time)
            gv.append(dec.value)
    return (np.array(ts), np.array(xyz, float), np.array(quat, float),
            np.array(gts), np.array(gv, float))


def extract_episode(mcap_path: Path, tcp_offset, pose_topic: str,
                    gripper_topic: str) -> dict:
    """mcap -> W' 系下指尖位姿序列 (pos3 + rot6d6 + gripper1) + 自检。"""
    ts, cam_pos_w, quat, gts, gv = read_pose_stream(mcap_path, pose_topic, gripper_topic)
    if len(ts) == 0:
        raise RuntimeError(f"未找到位姿话题 {pose_topic}")
    p_tcp = np.asarray(tcp_offset, float)
    R_all = np.stack([quat_to_rot(*q) for q in quat])              # (N,3,3)

    # 1) TCP 重表达：指尖位置 = 相机位置 + R·p（朝向不变，刚体上同一姿态）
    tip_pos_w = cam_pos_w + np.einsum("nij,j->ni", R_all, p_tcp)

    # 2) 构造 W'：原点=首帧指尖，z=world z（重力反方向），首帧 yaw 归零
    fwd0 = R_all[0][:, 0]                       # 首帧夹爪前向(body x)在 world 的方向
    if np.linalg.norm(fwd0[:2]) < 0.15:
        print("  ⚠ 首帧夹爪接近竖直，yaw 归零方向不可靠（建议开录时大致水平）")
    yaw0 = math.atan2(fwd0[1], fwd0[0])
    c, s = math.cos(yaw0), math.sin(yaw0)
    R_w = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])           # world <- W'
    t0 = tip_pos_w[0]

    pos = (tip_pos_w - t0) @ R_w                                   # = (R_w^T (p-t0))^T
    R_loc = np.einsum("ji,njk->nik", R_w, R_all)                   # R_w^T @ R_n
    rot6d = np.stack([rot_to_6d(R) for R in R_loc])
    euler = np.array([euler_zyx_deg(R) for R in R_loc])            # (N,3) roll,pitch,yaw

    # 3) 夹爪按位姿时间戳最近邻对齐（保留原始米值，归一化留给阶段3）
    if len(gts):
        grip = np.array([gv[nearest_idx(gts, t)] for t in ts])
    else:
        print("  ⚠ 无夹爪话题，gripper 置 NaN")
        grip = np.full(len(ts), np.nan)

    # ---- 自检 ----
    rt_err = max(
        math.degrees(math.acos(np.clip(
            (np.trace(rot6d_to_rot(rot_to_6d(R)).T @ R) - 1) / 2, -1, 1)))
        for R in R_loc[:: max(1, len(R_loc) // 100)])
    checks = {
        "首帧位置 |p0| (应=0)": float(np.linalg.norm(pos[0])),
        "首帧 yaw (应=0°)": float(euler[0, 2]),
        "6D旋转往返最大误差(°)": rt_err,
        "首帧 roll/pitch(°, 保留绝对值)": (round(float(euler[0, 0]), 2),
                                          round(float(euler[0, 1]), 2)),
    }
    return {
        "ts": ts, "pos": pos, "rot6d": rot6d, "euler_deg": euler,
        "gripper_m": grip, "cam_pos_w": cam_pos_w, "tip_pos_w": tip_pos_w,
        "quat_w": quat, "tcp_offset": p_tcp, "yaw0_rad": yaw0, "t0_w": t0,
        "checks": checks,
    }


def plot_episode(ep: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = (ep["ts"] - ep["ts"][0]) / 1e9
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(131, projection="3d")
    ax.plot(*ep["cam_pos_w"].T, color="lightgray", lw=1, label="camera origin (raw VIO)")
    ax.plot(*ep["tip_pos_w"].T, color="tab:blue", lw=1.5, label="fingertip (TCP)")
    ax.scatter(*ep["tip_pos_w"][0], color="red", s=40, label="start")
    ax.set_title("world frame"); ax.legend(fontsize=7)

    ax2 = fig.add_subplot(132)
    for i, lbl in enumerate("xyz"):
        ax2.plot(t, ep["pos"][:, i], label=f"tip {lbl} (W')")
    if not np.isnan(ep["gripper_m"]).all():
        ax2.plot(t, ep["gripper_m"], "--", color="gray", label="gripper (m)")
    ax2.legend(fontsize=8); ax2.set_xlabel("t (s)"); ax2.set_ylabel("m")
    ax2.grid(alpha=0.3); ax2.set_title("fingertip pos in W' + gripper")

    ax3 = fig.add_subplot(133)
    for i, lbl in enumerate(["roll", "pitch", "yaw"]):
        ax3.plot(t, ep["euler_deg"][:, i], label=lbl)
    ax3.legend(fontsize=8); ax3.set_xlabel("t (s)"); ax3.set_ylabel("deg")
    ax3.grid(alpha=0.3); ax3.set_title("orientation in W' (yaw starts at 0)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def process_poses(mcap_path: Path, out_dir: Path, tcp_offset,
                  pose_topic: str, gripper_topic: str):
    print(f"\n=== {mcap_path.name} (位姿提取) ===")
    ep = extract_episode(mcap_path, tcp_offset, pose_topic, gripper_topic)
    dur = (ep["ts"][-1] - ep["ts"][0]) / 1e9
    span = ep["pos"].max(0) - ep["pos"].min(0)
    print(f"  {len(ep['ts'])} 帧 / {dur:.1f}s   W'内运动范围 "
          f"x={span[0]:.3f} y={span[1]:.3f} z={span[2]:.3f} m")
    for k, v in ep["checks"].items():
        print(f"  自检  {k}: {v}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = mcap_path.stem
    npz = out_dir / f"{stem}_poses.npz"
    np.savez_compressed(
        npz, ts=ep["ts"], pos=ep["pos"], rot6d=ep["rot6d"],
        euler_deg=ep["euler_deg"], gripper_m=ep["gripper_m"],
        cam_pos_w=ep["cam_pos_w"], tip_pos_w=ep["tip_pos_w"], quat_w=ep["quat_w"],
        tcp_offset=ep["tcp_offset"], yaw0_rad=ep["yaw0_rad"], t0_w=ep["t0_w"])
    png = out_dir / f"{stem}_poses.png"
    plot_episode(ep, png)
    print(f"  ✓ {npz}\n  ✓ 诊断图 {png}")


def process_mcap(mcap_path: Path, out_dir: Path, fps: int, balance: float,
                 fov_scale: float, side_by_side: bool, only: set[str] | None,
                 target_hfov: float | None = None):
    print(f"\n=== {mcap_path.name} ===")
    cams = read_mcap_cameras(mcap_path)
    if not cams:
        print("  未找到相机话题 (foxglove.CompressedImage)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(out_dir, os.W_OK):
        sys.exit(f"  ✗ 输出目录无写权限: {out_dir}\n"
                 f"    请用 -o 指定一个可写目录, 例如 -o ~/umi_videos_test")
    stem = mcap_path.stem
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for pre, c in cams.items():
            name = c["name"]
            if only and name not in only:
                continue
            if c["calib"] is None:
                print(f"  ! {name}: 缺少 camera_info，跳过")
                continue
            if not c["h264"]:
                print(f"  ! {name}: 无图像数据，跳过")
                continue

            cal = c["calib"]
            print(f"  {name}: model={cal['model']} {cal['w']}x{cal['h']} "
                  f"D={[round(x,4) for x in cal['D'].reshape(-1)[:4]]}")
            frames = decode_h264(c["h264"], fps, tmp, name)
            if not frames:
                print(f"  ! {name}: ffmpeg 解码 0 帧，跳过")
                continue

            # 用原生分辨率帧做 remap 源，输出 640x480：清晰度远好于先缩小再去畸变
            first = cv2.imread(str(frames[0]))
            native = (first.shape[1], first.shape[0])
            m1, m2, Knew, (w, h) = build_undistort_maps(cal, balance, fov_scale,
                                                        target_hfov, src_size=native)
            hfov = 2 * math.degrees(math.atan(w / (2 * Knew[0, 0])))
            vfov = 2 * math.degrees(math.atan(h / (2 * Knew[1, 1])))
            print(f"    源 {native[0]}x{native[1]} -> 输出 {w}x{h}  "
                  f"视场: HFOV={hfov:.1f}°  VFOV={vfov:.1f}°")
            out_w = w * 2 if side_by_side else w
            out_mp4 = out_dir / f"{stem}_{name}_undist.mp4"
            vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (out_w, h))
            if not vw.isOpened():
                print(f"  ✗ 无法创建输出文件: {out_mp4}\n"
                      f"    （目录无写权限？请用 -o 指定可写目录）")
                continue
            for jpg in frames:
                img = cv2.imread(str(jpg))
                if (img.shape[1], img.shape[0]) != native:
                    img = cv2.resize(img, native)
                und = cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
                if side_by_side:
                    vw.write(np.hstack([cv2.resize(img, (w, h)), und]))
                else:
                    vw.write(und)
            vw.release()
            print(f"  ✓ {out_mp4}  ({len(frames)} 帧)")


def main():
    ap = argparse.ArgumentParser(description="UMI mcap -> openpi/LeRobot 转换（阶段1: 主相机去鱼眼）")
    ap.add_argument("input", type=str, help="VIO mcap 文件或包含 mcap 的目录")
    ap.add_argument("-o", "--output-dir", type=str, default=None,
                    help="输出目录（默认 umi/output/，不写原始数据目录）")
    ap.add_argument("--fps", type=int, default=30, help="解码/封装帧率 (默认 30)")
    ap.add_argument("--balance", type=float, default=0.0,
                    help="去畸变保留视场: 0=裁到全有效像素, 1=尽量保留源像素(默认 0.0)")
    ap.add_argument("--target-hfov", type=float, default=None,
                    help="目标水平视场角(度), 与部署相机 D405 对齐时用, 如 87; "
                         "给定后忽略 --balance/--fov-scale")
    ap.add_argument("--fov-scale", type=float, default=1.0,
                    help="输出视场缩放, >1 看到更广但更小 (默认 1.0)")
    ap.add_argument("--side-by-side", action="store_true",
                    help="输出 原图|去畸变 左右对比, 便于肉眼确认")
    ap.add_argument("--cameras", nargs="*", default=["camera0"],
                    help="要处理的相机 (默认只有主相机 camera0，即中间那路)；传 all 处理全部三路")
    ap.add_argument("--poses", action="store_true",
                    help="阶段2: 提取 W' 系指尖位姿序列 (npz+诊断图)，不导出视频")
    ap.add_argument("--tcp-offset", nargs=3, type=float, default=list(DEFAULT_TCP_OFFSET),
                    metavar=("X", "Y", "Z"),
                    help=f"camera0 光心->指尖偏移 (body系, m), 默认 {DEFAULT_TCP_OFFSET} "
                         f"(pivot_calib.py 标定)")
    ap.add_argument("--pose-topic", type=str, default=POSE_TOPIC)
    ap.add_argument("--gripper-topic", type=str, default=GRIPPER_TOPIC)
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if in_path.is_dir():
        mcaps = sorted(in_path.glob("*.mcap"))
    elif in_path.is_file():
        mcaps = [in_path]
    else:
        sys.exit(f"输入不存在: {in_path}")
    if not mcaps:
        sys.exit(f"未找到 .mcap 文件: {in_path}")

    only = None if "all" in args.cameras else set(args.cameras)
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:  # 默认输出到本脚本旁的 output/，绝不写进原始数据目录
        out_dir = Path(__file__).resolve().parent / "output"
    for m in mcaps:
        if args.poses:
            process_poses(m, out_dir, args.tcp_offset,
                          args.pose_topic, args.gripper_topic)
        else:
            process_mcap(m, out_dir, args.fps, args.balance, args.fov_scale,
                         args.side_by_side, only, args.target_hfov)


if __name__ == "__main__":
    main()
