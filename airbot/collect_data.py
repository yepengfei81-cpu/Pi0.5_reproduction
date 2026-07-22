"""
openpi 数据收集脚本：遥操作 + 双相机录制 → LeRobot 格式
硬件：Lead (Replay, can1:50051) + Follow (Play, can0:50050) + 2x D405

记录内容（每帧）:
  - 关节空间: state=follow 6关节+夹爪, actions=lead 6关节+夹爪 (7D, 老格式)
  - 任务空间: state_eef/actions_eef = base 系 EEF(在线厂家FK get_end_pose)
              pos3 + quat4(xyzw) + gripper1 = 8D
    -> 用于和 UMI 数据联合训练；W'相对系/6D旋转/夹爪归一化的统一转换
       由阶段4打包脚本完成（与 umi_to_lerobot 同套表示）。

相机存储: video(H.264)，体积比逐帧 PNG 小 10~50 倍，且与 UMI 数据同格式
          （便于阶段4合并联合训练）。openpi/LeRobot 加载时透明解码。

用法:
    python collect_data.py --task "pick up the block and place it in the bowl"

流程:
    1. 启动后自动连接双臂 + 双相机
    2. 按 Enter 开始录制一条 episode
    3. 遥操作完成后按 Enter 保存，按 d 丢弃
    4. 按 q 结束收集，自动保存为 LeRobot 格式
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gripper_geom"))
from gripper_params import get_params  # noqa: E402  每爪开合范围

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

# ======================== 配置 ========================
LEAD_PORT = 50051       # Replay 无动力臂
FOLLOW_PORT = 50050     # Play 有动力臂
CONTROL_FREQ = 100      # 遥操作控制频率 Hz
RECORD_FREQ = 10        # 数据记录频率 Hz
HEAD_CAMERA_SERIAL = "230422271972"        # 环境(头)相机
WRIST_CAMERA_SERIAL = "230422271433"       # 右臂(arm0)手眼
LEFT_WRIST_CAMERA_SERIAL = "218622271178"  # 左臂(arm1)手眼(fw 5.17, 新接入的第3个)
HOME_JOINT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 零位关节角

# 夹爪量程对齐(固定修正)：厂家把 lead(Replay)和 follow(Play)夹爪的全开宽度设计得
# 不一致(lead 更小)——本应一样,属于设计 bug。这里按一次性标定的全开值线性映射
# lead->follow 修正它,让 follow 能张到自己的最大(否则夹不起板擦等宽物)。固定即可,无需每次标定。
LEAD_GRIPPER_MAX = 0.049     # lead(Replay) 完全张开时 get_eef_pos()[0] (标定固定值)
# follow(Play)上装的爪的 [刚好闭合, 完全张开] 电机读数。main 里按 --follow-gripper
# 从 gripper_params.py 覆盖。平行爪=[0, 0.073]; GET 用 [c_get, o_get](避免过挤压)。
FOLLOW_GRIPPER_MIN = 0.0
FOLLOW_GRIPPER_MAX = 0.073


def lead_grip_to_follow(lead_g):
    """lead 夹爪值 -> follow 当前爪的 [MIN, MAX] 量程(全闭→MIN, 全开→MAX)。
    映射值同时(a)下发给 follow,(b)存进 action —— 与 state(follow) 同单位;
    打包时按每爪 [close, open] 归一化。GET 的 MIN=c_get>0, 防止滑轨压到 0 把 GET 过挤压。"""
    frac = float(np.clip(lead_g / LEAD_GRIPPER_MAX, 0.0, 1.0))
    return float(FOLLOW_GRIPPER_MIN + frac * (FOLLOW_GRIPPER_MAX - FOLLOW_GRIPPER_MIN))


def parse_args():
    parser = argparse.ArgumentParser(description="openpi 数据收集")
    parser.add_argument("--task", type=str, required=True,
                        help="任务描述，如 'pick up the block and place it in the bowl'")
    parser.add_argument("--repo-id", type=str, default="airbot_play_data",
                        help="LeRobot 数据集名称")
    parser.add_argument("--record-freq", type=int, default=RECORD_FREQ,
                        help=f"数据记录频率 (默认 {RECORD_FREQ} Hz)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认 HF_LEROBOT_HOME)")
    parser.add_argument("--no-display", action="store_true",
                        help="禁用相机可视化窗口")
    parser.add_argument("--follow-gripper", type=str, default="parallel",
                        help="[单臂] follow 臂上装的爪名(读 gripper_params.py 的开合范围); 默认 parallel")
    parser.add_argument("--telemetry-hz", type=int, default=100,
                        help="遥测 sidecar 记录频率(Hz): 关节力矩(电流换算)/关节角/速度/指令角, "
                             "存 <数据集>/telemetry/episode_XXXXXX.npz, 供 NeuralActuator 力估计等使用; "
                             "0=关闭。若遥操作跟手性变差, 降到 50")
    # ===== 双臂模式 =====
    parser.add_argument("--dual-arm", action="store_true",
                        help="双臂采集: 两对 lead-follow + 环境相机 + 两个手眼; 输出 state_eef_0/1 + wrist_image_1")
    parser.add_argument("--lead-port1", type=int, default=50053, help="[双臂] 臂1 lead 端口")
    parser.add_argument("--follow-port1", type=int, default=50052, help="[双臂] 臂1 follow 端口")
    parser.add_argument("--wrist-serial1", type=str, default=LEFT_WRIST_CAMERA_SERIAL,
                        help="[双臂] 臂1 手眼相机 SN; 默认使用 LEFT_WRIST_CAMERA_SERIAL 常量")
    parser.add_argument("--arm0-gripper", type=str, default="get",
                        help="[双臂] 臂0(主/持刀)爪名; 默认 get")
    parser.add_argument("--arm1-gripper", type=str, default="parallel",
                        help="[双臂] 臂1(按压/拉开)爪名; 默认 parallel")
    return parser.parse_args()


class DualCamera:
    """管理两个 D405 相机"""

    def __init__(self, head_serial=HEAD_CAMERA_SERIAL, wrist_serial=WRIST_CAMERA_SERIAL,
                 wrist1_serial=None):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipelines = {}
        self.aligns = {}
        self._last = {}          # 每个相机上一帧(取帧超时时复用, 避免整条采集崩)

        ctx = rs.context()
        devices = ctx.query_devices()
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        logger.info(f"检测到 {len(serials)} 个 RealSense 设备: {serials}")

        # wrist1 仅双臂模式给; None 则不启动(单臂行为不变)
        cam_list = [("head", head_serial), ("wrist", wrist_serial)]
        if wrist1_serial is not None:
            cam_list.append(("wrist1", wrist1_serial))

        # 3 台并行启动: pipeline.start(USB协商+初始化)每台要好几秒, 串行3台=3倍慢。
        # 各台独立 pipeline 线程安全; 每台打印耗时, 便于定位哪台慢(USB口/带宽问题)。
        import threading
        lock = threading.Lock()

        def _start_one(name, serial):
            t0 = time.monotonic()
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            align = rs.align(rs.stream.color)
            for _ in range(8):        # warmup: 8 帧足够自动曝光初步稳定
                pipeline.wait_for_frames()
            with lock:
                self.aligns[name] = align
                self.pipelines[name] = pipeline
            logger.info(f"{name} 相机已就绪 (SN: {serial}, 耗时 {time.monotonic()-t0:.1f}s)")

        threads = []
        for name, serial in cam_list:
            if serial is None:
                continue
            if serial not in serials:
                logger.warning(f"{name} 相机 SN={serial} 未找到，跳过")
                continue
            th = threading.Thread(target=_start_one, args=(name, serial), daemon=True)
            th.start(); threads.append(th)
        for th in threads:
            th.join()

    def _grab(self, name, pipeline, timeout_ms=2000):
        """取一帧; 超时/无帧则打印是哪个相机 + 复用上一帧, 不让整条采集崩。"""
        z = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            frameset = pipeline.wait_for_frames(timeout_ms=timeout_ms)
            color = self.aligns[name].process(frameset).get_color_frame()
            if color:
                img = np.asanyarray(color.get_data())
                self._last[name] = img
                return img
            logger.warning(f"⚠ 相机 '{name}' 无 color frame, 复用上一帧")
        except RuntimeError as e:
            logger.warning(f"⚠ 相机 '{name}' 取帧超时({e}), 复用上一帧 (USB 带宽/供电?)")
        return self._last.get(name, z)

    def get_frames(self):
        """返回 head_bgr, wrist_bgr (numpy BGR uint8, H×W×3)"""
        f = {n: self._grab(n, p) for n, p in self.pipelines.items()}
        z = np.zeros((480, 640, 3), dtype=np.uint8)
        return f.get("head", z), f.get("wrist", z)

    def get_frames_dual(self):
        """双臂: 返回 head(env), wrist0(臂0), wrist1(臂1) 三路 BGR。"""
        f = {n: self._grab(n, p) for n, p in self.pipelines.items()}
        z = np.zeros((480, 640, 3), dtype=np.uint8)
        return f.get("head", z), f.get("wrist", z), f.get("wrist1", z)

    def stop(self):
        for pipeline in self.pipelines.values():
            pipeline.stop()
        logger.info("相机已关闭")


def show_camera_view(head_bgr, wrist_bgr, episode_num, frame_count, record_freq):
    """在一个窗口中并排显示两个相机画面"""
    # 缩放到一半大小以节省屏幕空间
    h_small = cv2.resize(head_bgr, (320, 240))
    w_small = cv2.resize(wrist_bgr, (320, 240))

    # 添加标签
    cv2.putText(h_small, "Head Camera", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(w_small, "Wrist Camera", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 状态信息
    info = f"Ep:{episode_num}  Frame:{frame_count}  Time:{frame_count/record_freq:.1f}s"
    cv2.putText(h_small, info, (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 拼接
    combined = np.hstack([h_small, w_small])
    cv2.imshow("Data Collection", combined)
    cv2.waitKey(1)


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


_NAN6 = [float("nan")] * 6   # 遥测某次读取失败时的占位(整行不丢, 离线可插值/剔除)


def save_telemetry(save_dir, ep_idx, telem):
    """遥测 sidecar 落盘: <数据集>/telemetry/episode_{ep:06d}.npz (ep 与 LeRobot episode 索引一致, 0起)。
    与 10Hz 主记录靠 frame_idx 列对齐(每行遥测记录"当时已录多少主帧")。
    用途: NeuralActuator 电流→力标定/离线打标 + 接触/握法分析; 不动 LeRobot 主格式。"""
    if not telem or not telem.get("t"):
        return
    tdir = Path(save_dir) / "telemetry"
    tdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tdir / f"episode_{ep_idx:06d}.npz",
                        **{k: np.asarray(v, dtype=np.float32) for k, v in telem.items()})
    readme = tdir / "README.txt"
    if not readme.exists():
        readme.write_text(
            "telemetry sidecar (每 episode 一个 npz, float32; 双臂字段带 _0/_1 后缀):\n"
            "  t          [N]    episode 内时间(s)\n"
            "  frame_idx  [N]    该行时刻已录的 10Hz 主记录帧数(对齐用)\n"
            "  cmd_q      [N,6]  指令关节角(lead, rad)     cmd_grip [N]   指令夹爪(m, follow量程)\n"
            "  q          [N,6]  实际关节角(follow, rad)   dq       [N,6] 关节速度(rad/s)\n"
            "  tau        [N,6]  关节力矩(Nm, 电流换算)    grip_q   [N]   夹爪开度(m)\n"
            "  grip_tau   [N]    夹爪力矩(Nm)\n", encoding="utf-8")
    logger.info(f"  遥测 sidecar: {len(telem['t'])} 行 -> telemetry/episode_{ep_idx:06d}.npz")


def reset_to_home(follow):
    """将 Follow 臂回到零位"""
    from airbot_py.arm import RobotMode
    logger.info("Follow 臂回零位...")
    follow.switch_mode(RobotMode.PLANNING_POS)
    follow.move_to_joint_pos(HOME_JOINT, blocking=True)
    follow.move_eef_pos(1.0)  # 张开夹爪
    time.sleep(0.3)
    logger.info("已回零位")


def collect_episode(lead, follow, cameras, record_freq, control_freq,
                    episode_num, show_display, telemetry_hz=0):
    """
    录制一条 episode。
    返回: (frames, telem) 或 None（丢弃）。frames=[{state, action, ...}];
    telem = 遥测 sidecar 字典(telemetry_hz>0) 或 None, 由 save_telemetry 落盘。
    """
    import select
    import sys

    from airbot_py.arm import RobotMode, SpeedProfile

    # 对齐位置
    lead_joints = lead.get_joint_pos()
    follow_joints = follow.get_joint_pos()
    diff = sum(abs(a - b) for a, b in zip(lead_joints, follow_joints))
    if diff > 0.1:
        logger.info("对齐 Follow 臂到 Lead 臂位置...")
        follow.switch_mode(RobotMode.PLANNING_POS)
        follow.move_to_joint_pos(lead_joints, blocking=True)
        time.sleep(0.5)

    # 切换到伺服模式
    follow.switch_mode(RobotMode.SERVO_JOINT_POS)
    follow.set_speed_profile(SpeedProfile.FAST)

    # 录制前先显示一下相机画面，让用户确认视角
    if show_display:
        head_bgr, wrist_bgr = cameras.get_frames()
        show_camera_view(head_bgr, wrist_bgr, episode_num, 0, record_freq)

    logger.info(">>> 按 Enter 开始录制（移动 Lead 臂开始演示）...")
    input()

    frames = []
    control_dt = 1.0 / control_freq
    record_interval = control_freq // record_freq
    step = 0
    recording = True
    # 遥测 sidecar(默认100Hz): 力矩/角度/速度/指令角, 供 NeuralActuator 力估计等离线使用
    telem_interval = max(1, control_freq // telemetry_hz) if telemetry_hz > 0 else 0
    telem = ({k: [] for k in ("t", "frame_idx", "cmd_q", "cmd_grip",
                              "q", "dq", "tau", "grip_q", "grip_tau")}
             if telem_interval else None)
    t_ep0 = time.monotonic()

    logger.info("录制中... 按 Enter 结束并保存")

    try:
        while recording:
            t0 = time.monotonic()

            # 读 Lead 臂状态
            lead_joints = lead.get_joint_pos()
            lead_eef = lead.get_eef_pos()

            # lead 夹爪 -> follow 量程(否则 follow 开不够宽, 夹不起宽物体)
            grip_follow = (lead_grip_to_follow(lead_eef[0])
                           if lead_eef is not None and len(lead_eef) > 0 else 0.0)

            # 发送到 Follow 臂
            if lead_joints is not None:
                follow.servo_joint_pos(lead_joints)
            if lead_eef is not None and len(lead_eef) > 0:
                follow.servo_eef_pos([grip_follow])

            # 遥测 sidecar 采样(默认每个控制周期=100Hz)
            if telem is not None and step % telem_interval == 0:
                _q = follow.get_joint_pos(); _dq = follow.get_joint_vel()
                _tau = follow.get_joint_eff()
                _ge = follow.get_eef_pos(); _gt = follow.get_eef_eff()
                telem["t"].append(time.monotonic() - t_ep0)
                telem["frame_idx"].append(len(frames))
                telem["cmd_q"].append(list(lead_joints) if lead_joints is not None else _NAN6)
                telem["cmd_grip"].append(grip_follow)
                telem["q"].append(list(_q) if _q else _NAN6)
                telem["dq"].append(list(_dq) if _dq else _NAN6)
                telem["tau"].append(list(_tau) if _tau else _NAN6)
                telem["grip_q"].append(_ge[0] if _ge else float("nan"))
                telem["grip_tau"].append(_gt[0] if _gt else float("nan"))

            # 按记录频率采样数据
            if step % record_interval == 0:
                follow_joints = follow.get_joint_pos()
                follow_eef = follow.get_eef_pos()
                # 在线厂家 FK：base 系末端位姿 [[x,y,z],[qx,qy,qz,qw]]
                follow_pose = follow.get_end_pose()   # state = 当前(follow)末端
                lead_pose = lead.get_end_pose()        # action = 目标(lead)末端

                if (follow_joints is not None and follow_eef is not None
                        and follow_pose is not None and lead_pose is not None):
                    state = np.array(follow_joints + follow_eef, dtype=np.float32)

                    # 存映射后的 follow 单位夹爪(与下发一致、与 state 单位一致)
                    gripper_val = [grip_follow]
                    action = np.array(
                        list(lead_joints) + list(gripper_val), dtype=np.float32
                    )

                    # EEF (base 系, 原始): pos3 + quat4(xyzw) + gripper1 = 8D
                    # W'/6D/夹爪归一化的统一转换交给阶段4打包脚本(与 UMI 同套代码)
                    state_eef = np.array(
                        list(follow_pose[0]) + list(follow_pose[1])
                        + list(follow_eef[:1]), dtype=np.float32)
                    action_eef = np.array(
                        list(lead_pose[0]) + list(lead_pose[1])
                        + list(gripper_val[:1]), dtype=np.float32)

                    head_bgr, wrist_bgr = cameras.get_frames()

                    frames.append({
                        "state": state,
                        "action": action,
                        "state_eef": state_eef,
                        "action_eef": action_eef,
                        "head_rgb": bgr_to_rgb(head_bgr),
                        "wrist_rgb": bgr_to_rgb(wrist_bgr),
                    })

                    # 可视化
                    if show_display:
                        show_camera_view(head_bgr, wrist_bgr, episode_num,
                                         len(frames), record_freq)

                    if len(frames) % record_freq == 0:
                        logger.info(f"  已录制 {len(frames)} 帧 ({len(frames)/record_freq:.1f}s)")

            step += 1

            # 非阻塞检查回车
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                recording = False

            elapsed = time.monotonic() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

    except KeyboardInterrupt:
        logger.info("录制被中断，丢弃本条")
        follow.switch_mode(RobotMode.PLANNING_POS)
        follow.set_speed_profile(SpeedProfile.DEFAULT)
        return None

    # 恢复安全模式
    follow.switch_mode(RobotMode.PLANNING_POS)
    follow.set_speed_profile(SpeedProfile.DEFAULT)

    logger.info(f"录制结束：共 {len(frames)} 帧 ({len(frames)/record_freq:.1f}s)")

    if len(frames) < 5:
        logger.warning("帧数太少，丢弃")
        return None

    resp = input("保存本条? [Enter=保存, d=丢弃]: ").strip().lower()
    if resp == "d":
        logger.info("已丢弃")
        return None

    return frames, telem

def create_lerobot_dataset(repo_id, record_freq, output_dir=None):
    """开头创建一次空数据集；之后每条 episode 立即写盘（低内存、抗崩溃）。"""
    import lerobot.common.datasets.lerobot_dataset as lr_ds
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    # 相机用 video(H.264) 存储：体积比逐帧 PNG 小 10~50 倍，且与 UMI 数据同格式。
    # LeRobot 默认编码器是 libsvtav1（本机 pyav 无此编码器），改用 h264(=libx264)。
    if not getattr(lr_ds, "_h264_patched", False):
        _orig_encode = lr_ds.encode_video_frames
        def _encode_h264(*a, **k):
            k.setdefault("vcodec", "h264")
            return _orig_encode(*a, **k)
        lr_ds.encode_video_frames = _encode_h264
        lr_ds._h264_patched = True

    base = Path(output_dir) if output_dir else HF_LEROBOT_HOME
    save_dir = base / repo_id
    if (save_dir / "meta" / "info.json").exists():
        # 已有同名数据集，加时间戳后缀避免冲突（分多次采集会得到多个数据集，
        # 阶段4打包时合并）
        import datetime
        repo_id = f"{repo_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_dir = base / repo_id
        logger.info(f"已有数据集存在，新建: {repo_id}")

    logger.info(f"创建数据集: {repo_id} (fps={record_freq})")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="airbot_play",
        fps=record_freq,
        root=save_dir,
        features={
            "image": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            # base 系原始 EEF: pos3 + quat4(xyzw) + gripper1。阶段4打包脚本
            # 把它和 UMI 一起转成统一的 W'/6D/归一化训练表示。
            "state_eef": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state_eef"],
            },
            "actions_eef": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions_eef"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )
    return dataset, save_dir


def write_episode(dataset, frames, task):
    """把一条 episode 立即写入数据集（add_frame + save_episode）。"""
    for frame in frames:
        dataset.add_frame({
            "image": frame["head_rgb"],
            "wrist_image": frame["wrist_rgb"],
            "state": frame["state"],
            "actions": frame["action"],
            "state_eef": frame["state_eef"],
            "actions_eef": frame["action_eef"],
            "task": task,
        })
    dataset.save_episode()


# ======================== 双臂采集 ========================
def lead_grip_to_follow_r(lead_g, fmin, fmax):
    """按给定该臂 follow 爪的 [fmin,fmax] 量程映射 lead 夹爪(每臂各自范围)。"""
    frac = float(np.clip(lead_g / LEAD_GRIPPER_MAX, 0.0, 1.0))
    return float(fmin + frac * (fmax - fmin))


def show_camera_view_dual(head, w0, w1, ep, fc, freq):
    hs = cv2.resize(head, (320, 240)); a = cv2.resize(w0, (320, 240)); b = cv2.resize(w1, (320, 240))
    cv2.putText(hs, "Env", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(a, "Wrist0 (arm0)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(b, "Wrist1 (arm1)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(hs, f"Ep:{ep} F:{fc} {fc/freq:.1f}s", (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("Data Collection", np.hstack([hs, a, b])); cv2.waitKey(1)


def collect_episode_dual(arms, cameras, record_freq, control_freq, episode_num, show_display,
                         telemetry_hz=0):
    """双臂录制。arms=[(lead,follow,fmin,fmax), ...]。返回 (frames, telem) 或 None。
    每帧: head_rgb + wrist_rgb_0/1 + state_eef_0/1(follow实际) + action_eef_0/1(lead指令)。
    telem = 遥测 sidecar(字段带 _0/_1 后缀) 或 None。"""
    import select
    import sys
    from airbot_py.arm import RobotMode, SpeedProfile

    for lead, follow, _fmin, _fmax in arms:               # 对齐 + 伺服模式
        lj = lead.get_joint_pos(); fj = follow.get_joint_pos()
        if lj is not None and fj is not None and sum(abs(a - b) for a, b in zip(lj, fj)) > 0.1:
            follow.switch_mode(RobotMode.PLANNING_POS)
            follow.move_to_joint_pos(lj, blocking=True); time.sleep(0.5)
        follow.switch_mode(RobotMode.SERVO_JOINT_POS); follow.set_speed_profile(SpeedProfile.FAST)

    if show_display:
        h, w0, w1 = cameras.get_frames_dual(); show_camera_view_dual(h, w0, w1, episode_num, 0, record_freq)
    logger.info(">>> 按 Enter 开始录制（双手移动两个 Lead 臂）...")
    input()

    frames = []; control_dt = 1.0 / control_freq; record_interval = control_freq // record_freq
    step = 0; recording = True
    telem_interval = max(1, control_freq // telemetry_hz) if telemetry_hz > 0 else 0
    telem = ({k: [] for k in ["t", "frame_idx"]
              + [f"{n}_{ai}" for ai in range(len(arms))
                 for n in ("cmd_q", "cmd_grip", "q", "dq", "tau", "grip_q", "grip_tau")]}
             if telem_interval else None)
    t_ep0 = time.monotonic()
    logger.info("录制中... 按 Enter 结束并保存")
    try:
        while recording:
            t0 = time.monotonic()
            grips = []; cmd_qs = []
            for lead, follow, fmin, fmax in arms:          # 两臂各自 lead->follow
                lj = lead.get_joint_pos(); le = lead.get_eef_pos()
                g = lead_grip_to_follow_r(le[0], fmin, fmax) if le is not None and len(le) > 0 else 0.0
                grips.append(g); cmd_qs.append(lj)
                if lj is not None:
                    follow.servo_joint_pos(lj)
                if le is not None and len(le) > 0:
                    follow.servo_eef_pos([g])

            # 遥测 sidecar 采样(默认每个控制周期=100Hz, 每臂一组字段)
            if telem is not None and step % telem_interval == 0:
                telem["t"].append(time.monotonic() - t_ep0)
                telem["frame_idx"].append(len(frames))
                for ai, (_lead, follow, _fm, _fx) in enumerate(arms):
                    _q = follow.get_joint_pos(); _dq = follow.get_joint_vel()
                    _tau = follow.get_joint_eff()
                    _ge = follow.get_eef_pos(); _gt = follow.get_eef_eff()
                    telem[f"cmd_q_{ai}"].append(list(cmd_qs[ai]) if cmd_qs[ai] is not None else _NAN6)
                    telem[f"cmd_grip_{ai}"].append(grips[ai])
                    telem[f"q_{ai}"].append(list(_q) if _q else _NAN6)
                    telem[f"dq_{ai}"].append(list(_dq) if _dq else _NAN6)
                    telem[f"tau_{ai}"].append(list(_tau) if _tau else _NAN6)
                    telem[f"grip_q_{ai}"].append(_ge[0] if _ge else float("nan"))
                    telem[f"grip_tau_{ai}"].append(_gt[0] if _gt else float("nan"))

            if step % record_interval == 0:
                recs = []; ok = True
                for ai, (lead, follow, _fmin, _fmax) in enumerate(arms):
                    fe = follow.get_eef_pos(); fp = follow.get_end_pose(); lp = lead.get_end_pose()
                    if fe is None or fp is None or lp is None:
                        ok = False; break
                    se = np.array(list(fp[0]) + list(fp[1]) + list(fe[:1]), dtype=np.float32)   # follow实际
                    ae = np.array(list(lp[0]) + list(lp[1]) + [grips[ai]], dtype=np.float32)    # lead指令
                    recs.append((se, ae))
                if ok:
                    h, w0, w1 = cameras.get_frames_dual()
                    fr = {"head_rgb": bgr_to_rgb(h), "wrist_rgb_0": bgr_to_rgb(w0), "wrist_rgb_1": bgr_to_rgb(w1)}
                    for ai, (se, ae) in enumerate(recs):
                        fr[f"state_eef_{ai}"] = se; fr[f"action_eef_{ai}"] = ae
                    frames.append(fr)
                    if show_display:
                        show_camera_view_dual(h, w0, w1, episode_num, len(frames), record_freq)
                    if len(frames) % record_freq == 0:
                        logger.info(f"  已录制 {len(frames)} 帧 ({len(frames)/record_freq:.1f}s)")
            step += 1
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline(); recording = False
            el = time.monotonic() - t0
            if el < control_dt:
                time.sleep(control_dt - el)
    except KeyboardInterrupt:
        logger.info("录制被中断，丢弃本条")
        for _l, follow, _a, _b in arms:
            follow.switch_mode(RobotMode.PLANNING_POS); follow.set_speed_profile(SpeedProfile.DEFAULT)
        return None

    for _l, follow, _a, _b in arms:
        follow.switch_mode(RobotMode.PLANNING_POS); follow.set_speed_profile(SpeedProfile.DEFAULT)
    logger.info(f"录制结束：共 {len(frames)} 帧 ({len(frames)/record_freq:.1f}s)")
    if len(frames) < 5:
        logger.warning("帧数太少，丢弃"); return None
    if input("保存本条? [Enter=保存, d=丢弃]: ").strip().lower() == "d":
        logger.info("已丢弃"); return None
    return frames, telem


def create_lerobot_dataset_dual(repo_id, record_freq, output_dir=None):
    """双臂数据集: image + wrist_image + wrist_image_1 + state_eef_0/1 + actions_eef_0/1(各8D)。
    与 pack_lerobot.py 的双臂分支对应。"""
    import lerobot.common.datasets.lerobot_dataset as lr_ds
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    if not getattr(lr_ds, "_h264_patched", False):
        _orig = lr_ds.encode_video_frames
        def _h264(*a, **k):
            k.setdefault("vcodec", "h264"); return _orig(*a, **k)
        lr_ds.encode_video_frames = _h264
        lr_ds._h264_patched = True

    base = Path(output_dir) if output_dir else HF_LEROBOT_HOME
    save_dir = base / repo_id
    if (save_dir / "meta" / "info.json").exists():
        import datetime
        repo_id = f"{repo_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_dir = base / repo_id
        logger.info(f"已有数据集存在，新建: {repo_id}")

    logger.info(f"创建双臂数据集: {repo_id} (fps={record_freq})")
    vid = lambda: {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]}
    eef = lambda nm: {"dtype": "float32", "shape": (8,), "names": [nm]}
    dataset = LeRobotDataset.create(
        repo_id=repo_id, robot_type="airbot_play", fps=record_freq, root=save_dir,
        features={
            "image": vid(), "wrist_image": vid(), "wrist_image_1": vid(),
            "state_eef_0": eef("state_eef_0"), "actions_eef_0": eef("actions_eef_0"),
            "state_eef_1": eef("state_eef_1"), "actions_eef_1": eef("actions_eef_1"),
        },
        image_writer_threads=6, image_writer_processes=2,
    )
    return dataset, save_dir


def write_episode_dual(dataset, frames, task):
    for fr in frames:
        dataset.add_frame({
            "image": fr["head_rgb"], "wrist_image": fr["wrist_rgb_0"], "wrist_image_1": fr["wrist_rgb_1"],
            "state_eef_0": fr["state_eef_0"], "actions_eef_0": fr["action_eef_0"],
            "state_eef_1": fr["state_eef_1"], "actions_eef_1": fr["action_eef_1"],
            "task": task,
        })
    dataset.save_episode()


def run_dual(args):
    """双臂采集主流程(两对 lead-follow + env + 两手眼)。"""
    if args.wrist_serial1 is None:
        sys.exit("--dual-arm 需要 --wrist-serial1 (臂1 手眼相机 SN)")
    gp0 = get_params(args.arm0_gripper); gp1 = get_params(args.arm1_gripper)
    logger.info(f"臂0 爪={args.arm0_gripper} [{gp0['close']:.4f},{gp0['open']:.4f}]  "
                f"臂1 爪={args.arm1_gripper} [{gp1['close']:.4f},{gp1['open']:.4f}]")
    from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile

    lead0 = AIRBOTArm(url="localhost", port=LEAD_PORT)
    follow0 = AIRBOTArm(url="localhost", port=FOLLOW_PORT)
    lead1 = AIRBOTArm(url="localhost", port=args.lead_port1)
    follow1 = AIRBOTArm(url="localhost", port=args.follow_port1)
    for a, nm in [(lead0, f"lead0:{LEAD_PORT}"), (follow0, f"follow0:{FOLLOW_PORT}"),
                  (lead1, f"lead1:{args.lead_port1}"), (follow1, f"follow1:{args.follow_port1}")]:
        if not a.connect():
            sys.exit(f"无法连接 {nm}")
        logger.info(f"{nm} 已连接")
    arms = [(lead0, follow0, float(gp0["close"]), float(gp0["open"])),
            (lead1, follow1, float(gp1["close"]), float(gp1["open"]))]

    cameras = DualCamera(wrist1_serial=args.wrist_serial1)   # env(head) + wrist(arm0) + wrist1(arm1)
    show_display = not args.no_display
    if show_display:
        cv2.namedWindow("Data Collection", cv2.WINDOW_AUTOSIZE)

    dataset = None; save_dir = None; num_saved = 0
    logger.info("=" * 60); logger.info(f"[双臂] 任务: {args.task}"); logger.info("=" * 60)
    try:
        for _l, follow, _a, _b in arms:
            reset_to_home(follow)
        while True:
            logger.info(f"\n--- Episode {num_saved + 1} --- 按 Enter 录制, q+Enter 结束")
            if show_display:
                import select
                import sys as _sys
                while True:
                    h, w0, w1 = cameras.get_frames_dual()
                    show_camera_view_dual(h, w0, w1, num_saved + 1, 0, args.record_freq)
                    if select.select([_sys.stdin], [], [], 0.03)[0]:
                        cmd = _sys.stdin.readline().strip().lower(); break
            else:
                cmd = input().strip().lower()
            if cmd == "q":
                break
            ep = collect_episode_dual(arms, cameras, args.record_freq, CONTROL_FREQ,
                                      num_saved + 1, show_display,
                                      telemetry_hz=args.telemetry_hz)
            if ep is not None:
                frames, telem = ep
                if dataset is None:
                    dataset, save_dir = create_lerobot_dataset_dual(
                        args.repo_id, args.record_freq, args.output_dir)
                write_episode_dual(dataset, frames, args.task)
                save_telemetry(save_dir, num_saved, telem)   # 本条 LeRobot episode 索引 = num_saved(0起)
                num_saved += 1; del frames, ep
                logger.info(f"已保存 {num_saved} 条 -> {save_dir}")
            for _l, follow, _a, _b in arms:
                reset_to_home(follow)
    except KeyboardInterrupt:
        logger.info("\n收集被中断（已保存的不受影响）")
    finally:
        for _l, follow, _a, _b in arms:
            try:
                follow.switch_mode(RobotMode.PLANNING_POS)
                follow.set_speed_profile(SpeedProfile.DEFAULT); follow.disconnect()
            except Exception:
                pass
        for lead in (lead0, lead1):
            try:
                lead.disconnect()
            except Exception:
                pass
        cameras.stop()
        if show_display:
            cv2.destroyAllWindows()
    logger.info(f"\n完成：共 {num_saved} 条 -> {save_dir}" if num_saved else "没有收集到数据")


def main():
    args = parse_args()
    if args.dual_arm:                       # 双臂走独立流程; 单臂保持原样不变
        return run_dual(args)

    # follow 臂当前爪的开合范围(覆盖默认平行爪), 防止对 GET 过挤压
    global FOLLOW_GRIPPER_MIN, FOLLOW_GRIPPER_MAX
    _gp = get_params(args.follow_gripper)
    FOLLOW_GRIPPER_MIN, FOLLOW_GRIPPER_MAX = float(_gp["close"]), float(_gp["open"])
    logger.info(f"follow 爪='{args.follow_gripper}' 开合范围 "
                f"[{FOLLOW_GRIPPER_MIN:.4f}, {FOLLOW_GRIPPER_MAX:.4f}]")

    from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile

    # 连接双臂
    lead = AIRBOTArm(url="localhost", port=LEAD_PORT)
    follow = AIRBOTArm(url="localhost", port=FOLLOW_PORT)

    if not lead.connect():
        raise RuntimeError(f"无法连接 Lead 臂 (port={LEAD_PORT})")
    logger.info("Lead 臂 (Replay) 已连接")

    if not follow.connect():
        lead.disconnect()
        raise RuntimeError(f"无法连接 Follow 臂 (port={FOLLOW_PORT})")
    logger.info("Follow 臂 (Play) 已连接")

    # 连接相机
    cameras = DualCamera()

    show_display = not args.no_display
    if show_display:
        cv2.namedWindow("Data Collection", cv2.WINDOW_AUTOSIZE)

    # 开头创建数据集（懒创建：第一条 episode 确认保存时才建），之后每条立即写盘
    dataset = None
    save_dir = None
    num_saved = 0
    logger.info("=" * 60)
    logger.info(f"任务: {args.task}")
    logger.info(f"记录频率: {args.record_freq} Hz")
    logger.info(f"可视化: {'开启' if show_display else '关闭'}")
    logger.info("=" * 60)

    try:
        # 初始回零
        reset_to_home(follow)

        while True:
            logger.info(f"\n--- Episode {num_saved + 1} ---")
            logger.info("按 Enter 准备录制，按 q+Enter 结束收集")

            # 等待输入时持续刷新相机画面
            if show_display:
                import select, sys
                while True:
                    head_bgr, wrist_bgr = cameras.get_frames()
                    show_camera_view(head_bgr, wrist_bgr,
                                     num_saved + 1, 0, args.record_freq)
                    if select.select([sys.stdin], [], [], 0.03)[0]:
                        cmd = sys.stdin.readline().strip().lower()
                        break
            else:
                cmd = input().strip().lower()

            if cmd == "q":
                break

            episode = collect_episode(
                lead, follow, cameras,
                record_freq=args.record_freq,
                control_freq=CONTROL_FREQ,
                episode_num=num_saved + 1,
                show_display=show_display,
                telemetry_hz=args.telemetry_hz,
            )
            if episode is not None:
                frames, telem = episode
                # 懒创建数据集 + 立即写盘（低内存、崩溃最多只丢当前这条）
                if dataset is None:
                    dataset, save_dir = create_lerobot_dataset(
                        args.repo_id, args.record_freq, args.output_dir)
                write_episode(dataset, frames, args.task)
                save_telemetry(save_dir, num_saved, telem)   # 本条 LeRobot episode 索引 = num_saved(0起)
                num_saved += 1
                del frames, episode   # 立刻释放该条的图像内存
                logger.info(f"已保存 {num_saved} 条 episode -> {save_dir}")

            # 每条 episode 结束后回零位
            reset_to_home(follow)

    except KeyboardInterrupt:
        logger.info("\n收集被中断（已保存的 episode 不受影响）")

    finally:
        follow.switch_mode(RobotMode.PLANNING_POS)
        follow.set_speed_profile(SpeedProfile.DEFAULT)
        follow.disconnect()
        lead.disconnect()
        cameras.stop()
        if show_display:
            cv2.destroyAllWindows()

    if num_saved > 0:
        logger.info(f"\n完成：共保存 {num_saved} 条 episode 到 {save_dir}")
    else:
        logger.info("没有收集到数据")


if __name__ == "__main__":
    main()