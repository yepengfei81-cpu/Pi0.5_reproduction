"""
openpi 数据收集脚本：遥操作 + 双相机录制 → LeRobot 格式
硬件：Lead (Replay, can1:50051) + Follow (Play, can0:50050) + 2x D405

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

            # 发送到 Follow 臂
            if lead_joints is not None:
                follow.servo_joint_pos(lead_joints)
            if lead_eef is not None and len(lead_eef) > 0:
                follow.servo_eef_pos(lead_eef)

            # 按记录频率采样数据
            if step % record_interval == 0:
                follow_joints = follow.get_joint_pos()
                follow_eef = follow.get_eef_pos()

                if follow_joints is not None and follow_eef is not None:
                    state = np.array(follow_joints + follow_eef, dtype=np.float32)

                    gripper_val = lead_eef if lead_eef and len(lead_eef) > 0 else [0.0]
                    action = np.array(
                        list(lead_joints) + list(gripper_val), dtype=np.float32
                    )

                    head_bgr, wrist_bgr = cameras.get_frames()

                    frames.append({
                        "state": state,
                        "action": action,
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

def save_to_lerobot(all_episodes, task, repo_id, record_freq, output_dir=None):
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    if output_dir:
        save_dir = Path(output_dir) / repo_id
    else:
        save_dir = HF_LEROBOT_HOME / repo_id

    info_file = save_dir / "meta" / "info.json"

    if info_file.exists():
        # 已有数据集，加一个时间戳后缀避免冲突
        import datetime
        suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_repo_id = f"{repo_id}_{suffix}"
        if output_dir:
            save_dir = Path(output_dir) / new_repo_id
        else:
            save_dir = HF_LEROBOT_HOME / new_repo_id
        logger.info(f"已有数据集存在，新建: {new_repo_id}")
        repo_id = new_repo_id

    logger.info(f"创建数据集: {repo_id} (fps={record_freq})")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="airbot_play",
        fps=record_freq,
        root=save_dir,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
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
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    for ep_idx, frames in enumerate(all_episodes):
        for frame in frames:
            dataset.add_frame({
                "image": frame["head_rgb"],
                "wrist_image": frame["wrist_rgb"],
                "state": frame["state"],
                "actions": frame["action"],
                "task": task,
            })
        dataset.save_episode()
        logger.info(f"  保存 Episode {ep_idx + 1}/{len(all_episodes)} ({len(frames)} 帧)")

    logger.info(f"数据集共 {dataset.num_episodes} 条 episode")
    logger.info(f"保存在: {save_dir}")


def main():
    args = parse_args()

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

    all_episodes = []
    logger.info("=" * 60)
    logger.info(f"任务: {args.task}")
    logger.info(f"记录频率: {args.record_freq} Hz")
    logger.info(f"可视化: {'开启' if show_display else '关闭'}")
    logger.info("=" * 60)

    try:
        # 初始回零
        reset_to_home(follow)

        while True:
            logger.info(f"\n--- Episode {len(all_episodes) + 1} ---")
            logger.info("按 Enter 准备录制，按 q+Enter 结束收集")

            # 等待输入时持续刷新相机画面
            if show_display:
                import select, sys
                while True:
                    head_bgr, wrist_bgr = cameras.get_frames()
                    show_camera_view(head_bgr, wrist_bgr,
                                     len(all_episodes) + 1, 0, args.record_freq)
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
                episode_num=len(all_episodes) + 1,
                show_display=show_display,
            )
            if episode is not None:
                all_episodes.append(episode)
                logger.info(f"已收集 {len(all_episodes)} 条 episode")

            # 每条 episode 结束后回零位
            reset_to_home(follow)

    except KeyboardInterrupt:
        logger.info("\n收集被中断")

    finally:
        follow.switch_mode(RobotMode.PLANNING_POS)
        follow.set_speed_profile(SpeedProfile.DEFAULT)
        follow.disconnect()
        lead.disconnect()
        cameras.stop()
        if show_display:
            cv2.destroyAllWindows()

    if all_episodes:
        logger.info(f"\n开始保存 {len(all_episodes)} 条 episode 到 LeRobot 格式...")
        save_to_lerobot(
            all_episodes, args.task, args.repo_id,
            args.record_freq, args.output_dir,
        )
    else:
        logger.info("没有收集到数据")


if __name__ == "__main__":
    main()