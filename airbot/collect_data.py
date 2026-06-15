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
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

# ======================== 配置 ========================
LEAD_PORT = 50051       # Replay 无动力臂
FOLLOW_PORT = 50050     # Play 有动力臂
CONTROL_FREQ = 100      # 遥操作控制频率 Hz
RECORD_FREQ = 10        # 数据记录频率 Hz
HEAD_CAMERA_SERIAL = "230422271972"
WRIST_CAMERA_SERIAL = "230422271433"
HOME_JOINT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 零位关节角

# 夹爪量程对齐：lead(Replay 无动力臂)全开 < follow(Play)全开。
# 不映射就会导致 follow 开不够宽、夹不起宽物体(如板擦)。
# 先用 --print-grip 把 lead 完全张开/闭合各读一次, 把全开值填到 LEAD_GRIPPER_MAX。
LEAD_GRIPPER_MAX = 0.05      # ←★测量后填: lead 完全张开时 get_eef_pos()[0]
FOLLOW_GRIPPER_MAX = 0.073   # Play 夹爪完全张开值(全流程一致)


def lead_grip_to_follow(lead_g, lead_max=None, follow_max=None):
    """把 lead 夹爪值线性映射到 follow 量程(两端对齐: 全闭→0, 全开→follow_max)。
    映射后的值同时用于(a)下发给 follow, (b)存进 action/action_eef —— 保证训练里
    夹爪单位 = follow 物理单位, 与 state(follow) 一致, 打包 ÷0.073 归一化才正确。"""
    lm = lead_max if lead_max is not None else LEAD_GRIPPER_MAX
    fm = follow_max if follow_max is not None else FOLLOW_GRIPPER_MAX
    return float(np.clip(lead_g * fm / max(lm, 1e-6), 0.0, fm))


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
    parser.add_argument("--lead-gripper-max", type=float, default=LEAD_GRIPPER_MAX,
                        help=f"lead 夹爪完全张开时 get_eef_pos()[0] (默认 {LEAD_GRIPPER_MAX}); 用 --print-grip 测")
    parser.add_argument("--print-grip", action="store_true",
                        help="标定模式: 持续打印 lead/follow 夹爪读数, 手动全开/全闭 lead 读出量程后退出")
    return parser.parse_args()


class DualCamera:
    """管理两个 D405 相机"""

    def __init__(self, head_serial=HEAD_CAMERA_SERIAL, wrist_serial=WRIST_CAMERA_SERIAL):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipelines = {}
        self.aligns = {}

        ctx = rs.context()
        devices = ctx.query_devices()
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        logger.info(f"检测到 {len(serials)} 个 RealSense 设备: {serials}")

        for name, serial in [("head", head_serial), ("wrist", wrist_serial)]:
            if serial is None:
                continue
            if serial not in serials:
                logger.warning(f"{name} 相机 SN={serial} 未找到，跳过")
                continue
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipeline.start(config)
            self.aligns[name] = rs.align(rs.stream.color)
            self.pipelines[name] = pipeline
            for _ in range(30):
                pipeline.wait_for_frames()
            logger.info(f"{name} 相机已就绪 (SN: {serial})")

    def get_frames(self):
        """返回 head_bgr, wrist_bgr (numpy BGR uint8, H×W×3)"""
        frames = {}
        for name, pipeline in self.pipelines.items():
            frameset = pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.aligns[name].process(frameset)
            color = aligned.get_color_frame()
            if color:
                frames[name] = np.asanyarray(color.get_data())

        head_bgr = frames.get("head", np.zeros((480, 640, 3), dtype=np.uint8))
        wrist_bgr = frames.get("wrist", np.zeros((480, 640, 3), dtype=np.uint8))
        return head_bgr, wrist_bgr

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
                    episode_num, show_display):
    """
    录制一条 episode。
    返回: frames 列表 [{state, action, head_rgb, wrist_rgb}, ...] 或 None（丢弃）
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

    return frames

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


def main():
    args = parse_args()

    global LEAD_GRIPPER_MAX
    LEAD_GRIPPER_MAX = args.lead_gripper_max   # 命令行覆盖, 供 lead_grip_to_follow 使用

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

    # 标定模式: 手动把 lead 完全张开/闭合, 读出量程, 填到 --lead-gripper-max
    if args.print_grip:
        logger.info("标定: 手动完全张开/闭合 lead 夹爪, 观察 lead 全开值 -> 填 --lead-gripper-max。Ctrl+C 退出")
        try:
            while True:
                le = lead.get_eef_pos(); fe = follow.get_eef_pos()
                lv = le[0] if le else float("nan"); fv = fe[0] if fe else float("nan")
                print(f"  lead grip={lv:.4f}   follow grip={fv:.4f}   "
                      f"(当前映射: lead全开{args.lead_gripper_max}->follow {FOLLOW_GRIPPER_MAX})",
                      end="\r", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print()
        finally:
            lead.disconnect(); follow.disconnect()
        return

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
            )
            if episode is not None:
                # 懒创建数据集 + 立即写盘（低内存、崩溃最多只丢当前这条）
                if dataset is None:
                    dataset, save_dir = create_lerobot_dataset(
                        args.repo_id, args.record_freq, args.output_dir)
                write_episode(dataset, episode, args.task)
                num_saved += 1
                del episode   # 立刻释放该条的图像内存
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