#!/usr/bin/env python3
"""
AirBot Play 本地实时推理脚本

用法：
  python local_inference.py \
    --checkpoint ./checkpoints/pi05_airbot_play/my_experiment/16000 \
    --task "pick up the block and place it in the bowl"

急停：按 q 或 Esc，机械臂立即停止并回零点
Debug：  python local_inference.py --checkpoint ... --dry-run
"""

import argparse
import logging
import pathlib
import threading
import time
from typing import Optional

import cv2
import numpy as np
import json

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

import sys
# play_sdk.py 与本脚本一起放在仓库的 airbot/ 子目录下
sys.path.insert(0, str(pathlib.Path(__file__).parent / "airbot"))
from play_sdk import PlayRealRobot, RobotMode, SpeedProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOME_JOINT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _fmt(arr) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in arr) + "]"


class AirbotPlayInferenceRunner:
    def __init__(
        self,
        checkpoint_dir: pathlib.Path,
        port: int = 50050,
        task_prompt: Optional[str] = None,
        inference_freq: int = 10,
        head_camera_serial: Optional[str] = None,
        wrist_camera_serial: Optional[str] = None,
        dry_run: bool = False,
        debug: bool = False,
        no_display: bool = False,
        min_z_height = None,
        chunk_execute: int = 5,
    ):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.task_prompt = task_prompt or "do something"
        self.inference_freq = inference_freq
        self.inference_interval = 1.0 / inference_freq
        self.dry_run = dry_run
        self.debug = debug
        self.show_display = not no_display

        self._stop_requested = False   # 急停标志
        self._vis_head_bgr = None
        self._vis_wrist_bgr = None

        logger.info("加载模型...")
        self.train_config = _config.get_config("pi05_airbot_play")
        self.policy = _policy_config.create_trained_policy(
            self.train_config,
            self.checkpoint_dir,
            default_prompt=self.task_prompt,
        )

        logger.info("初始化机械臂...")
        self.robot = PlayRealRobot(
            port=port,
            enable_cameras=True,
            head_camera_serial=head_camera_serial,
            left_wrist_camera_serial=wrist_camera_serial,
        )

        self.last_joint_pos = None
        self.last_gripper_cmd = 0.0
        self.last_sent_gripper = None
        self.min_gripper_delta = 0.001
        self.min_z_height = min_z_height
        self.chunk_execute = chunk_execute

        eef = self.robot.left._robot.get_eef_pos()
        if eef is not None and len(eef) > 0:
            self.last_gripper_cmd = float(eef[0])
            self.last_sent_gripper = self.last_gripper_cmd
            logger.info(f"home 后夹爪状态同步: {self.last_gripper_cmd:.4f}")

        if self.show_display:
            cv2.namedWindow("Inference [q/Esc=急停]", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Inference [q/Esc=急停]", 960, 360)

        if dry_run:
            logger.info("*** DRY RUN 模式：只打印不发出运动指令 ***")
        else:
            logger.info("提示：按 q 或 Esc 急停并回零点")
        logger.info("推理系统初始化完成")

    # ------------------------------------------------------------------ #
    #  急停监听（无显示窗口时用后台线程监听 stdin 输入 q）
    # ------------------------------------------------------------------ #
    def _start_stdin_stop_listener(self):
        def _listen():
            try:
                while not self._stop_requested:
                    line = sys.stdin.readline().strip().lower()
                    if line in ("q", "quit", "stop"):
                        logger.warning("收到急停指令 (stdin)")
                        self._stop_requested = True
                        break
            except Exception:
                pass
        t = threading.Thread(target=_listen, daemon=True)
        t.start()

    # ------------------------------------------------------------------ #
    #  机械臂控制
    # ------------------------------------------------------------------ #
    def _servo_gripper(self, position: float) -> None:
        self.robot.left._robot.servo_eef_pos([position])

    def _move_to_home(self):
        logger.info("移动到起始位置（零点）...")
        try:
            self.robot.left._robot.switch_mode(RobotMode.PLANNING_POS)
            self.robot.left._robot.move_to_joint_pos(HOME_JOINT, blocking=True)
            self.robot.left._robot.move_eef_pos(1.0)
            time.sleep(0.5)
            logger.info("已到达起始位置")
        except Exception as e:
            logger.error(f"移动到起始位置失败: {e}")

    # ------------------------------------------------------------------ #
    #  可视化
    # ------------------------------------------------------------------ #
    def _update_display(self, step: int, current_state: np.ndarray,
                        target_joints: np.ndarray, target_gripper: float) -> bool:
        """
        更新显示窗口。
        返回 True 正常；返回 False 表示用户按了急停键。
        """
        if not self.show_display or self._vis_head_bgr is None:
            return True

        h = cv2.resize(self._vis_head_bgr, (480, 360))
        w = cv2.resize(self._vis_wrist_bgr, (480, 360))

        cv2.putText(h, f"HEAD  step={step}  [q=STOP]", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(w, f"WRIST step={step}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        for i, v in enumerate(current_state[:6]):
            cv2.putText(h, f"J{i}:{v:+.3f}", (8, 52 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        for i, v in enumerate(target_joints):
            cv2.putText(w, f"T{i}:{v:+.3f}", (8, 52 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

        cv2.putText(h, f"Grip:{current_state[6]:.3f}", (8, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        cv2.putText(w, f"T-Grip:{target_gripper:.3f}", (8, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

        cv2.imshow("Inference [q/Esc=急停]", np.hstack([h, w]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q 或 Esc
            logger.warning("检测到急停键 (q/Esc)")
            self._stop_requested = True
            return False
        return True

    # ------------------------------------------------------------------ #
    #  观测 & 推理
    # ------------------------------------------------------------------ #
    def get_observation(self) -> dict:
        head_bgr, _ = next(self.robot.head_camera)
        wrist_bgr, _ = next(self.robot.left_wrist_camera)

        self._vis_head_bgr = head_bgr
        self._vis_wrist_bgr = wrist_bgr

        # BGR → RGB（与 collect_data.py 的 bgr_to_rgb 一致）
        head_rgb = head_bgr[:, :, ::-1].copy()
        wrist_rgb = wrist_bgr[:, :, ::-1].copy()

        joint_q = self.robot.get_joint_q()
        if joint_q is None:
            logger.warning("无法获取关节状态")
            return None

        state = np.array(list(joint_q) + [self.last_gripper_cmd], dtype=np.float32)
        obs = {
            "observation/image": head_rgb.astype(np.uint8),
            "observation/wrist_image": wrist_rgb.astype(np.uint8),
            "observation/state": state,
            "prompt": self.task_prompt,
        }
        self.last_joint_pos = joint_q
        return obs

    def _should_send_gripper(self, target_gripper: float) -> bool:
        if self.last_sent_gripper is None:
            return True
        return abs(target_gripper - self.last_sent_gripper) >= self.min_gripper_delta

    def _execute_action(self, exec_step: int, action_row: np.ndarray) -> bool:
        """执行 chunk 中的单行动作。返回 False 表示急停。"""
        target_joints = action_row[:6]
        target_gripper = float(action_row[6])

        # 获取当前关节状态（用于 Z 保护和显示）
        joint_q = self.robot.get_joint_q()
        current_state = np.array(
            list(joint_q) + [self.last_gripper_cmd], dtype=np.float32
        ) if joint_q is not None else np.zeros(7, dtype=np.float32)

        # Z轴高度保护
        if self.min_z_height is not None:
            end_pose = self.robot.left._robot.get_end_pose()
            if end_pose is not None and end_pose[0][2] < self.min_z_height:
                logger.warning(
                    f"Z轴保护[exec]: Z={end_pose[0][2]:.4f} < {self.min_z_height:.4f}，仅发夹爪"
                )
                if self._should_send_gripper(target_gripper):
                    self._servo_gripper(target_gripper)
                    self.last_sent_gripper = target_gripper
                    self.last_gripper_cmd = target_gripper
                return self._update_display(exec_step, current_state, target_joints, target_gripper)

        self.robot.servo_joint_pos(target_joints.tolist())
        if self._should_send_gripper(target_gripper):
            self._servo_gripper(target_gripper)
            self.last_sent_gripper = target_gripper
            self.last_gripper_cmd = target_gripper

        return self._update_display(exec_step, current_state, target_joints, target_gripper)

    def run_single_step(self, step: int) -> bool:
        try:
            obs = self.get_observation()
            if obs is None:
                return False

            current_state = obs["observation/state"]

            t0 = time.monotonic()
            outputs = self.policy.infer(obs)
            infer_time = time.monotonic() - t0

            # policy.infer() 输出已是绝对关节角（AbsoluteActions 已还原）
            actions = outputs["actions"]  # (10, 7)

            print(f"\n{'='*60}")
            print(f"Step {step:4d}  推理耗时: {infer_time*1000:.1f}ms")
            print(f"  当前关节 : {_fmt(current_state[:6])}  夹爪={current_state[6]:.4f}")
            print(f"  预测动作块 (前3步):")
            for i in range(min(3, len(actions))):
                diff = actions[i, :6] - current_state[:6]
                print(f"    step+{i}: {_fmt(actions[i, :6])}  夹爪={actions[i,6]:.4f}"
                      f"  Δ={_fmt(diff)}")
            if self.debug:
                print(f"  完整动作块 ({len(actions)} steps):")
                for i, a in enumerate(actions):
                    print(f"    [{i:2d}]: joints={_fmt(a[:6])}  gripper={a[6]:.4f}")

            target_joints = actions[0, :6]
            target_gripper = float(actions[0, 6])

            # ---- Z轴高度保护 ----
            if self.min_z_height is not None and not self.dry_run:
                end_pose = self.robot.left._robot.get_end_pose()
                if end_pose is not None:
                    current_z = end_pose[0][2]
                    if current_z < self.min_z_height:
                        logger.warning(f"Z轴保护: 当前Z={current_z:.4f} < 限制{self.min_z_height:.4f}，仅发夹爪指令")
                        # 只发夹爪，不发关节（防止继续下压）
                        if not self.dry_run and self._should_send_gripper(target_gripper):
                            self._servo_gripper(target_gripper)
                            self.last_sent_gripper = target_gripper
                            self.last_gripper_cmd = target_gripper
                        return True

            if not self.dry_run:
                self.robot.servo_joint_pos(target_joints.tolist())
                if self._should_send_gripper(target_gripper):
                    self._servo_gripper(target_gripper)
                    self.last_sent_gripper = target_gripper
                    self.last_gripper_cmd = target_gripper
            else:
                print("  [DRY RUN] 未发出运动指令")

            return True

        except Exception as e:
            logger.error(f"推理步骤失败: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------ #
    #  主循环
    # ------------------------------------------------------------------ #
    def run_continuous(self, num_steps: Optional[int] = None, timeout: Optional[float] = None):
        logger.info("=" * 60)
        logger.info(f"开始推理  |  task='{self.task_prompt}'  |  freq={self.inference_freq}Hz"
                    f"  |  chunk_execute={self.chunk_execute}")
        logger.info("急停：按 q 或 Esc（有窗口时），或在终端输入 q 回车")
        logger.info("=" * 60)

        if not self.show_display and not self.dry_run:
            self._start_stdin_stop_listener()

        exec_step = 0
        start_time = time.monotonic()
        chunk = None
        chunk_idx = 0

        try:
            if not self.dry_run:
                self._move_to_home()

            obs = self.get_observation()
            if obs is None:
                logger.error("无法获取初始观测")
                return
            logger.info(f"起始状态: {_fmt(obs['observation/state'][:6])}")

            if not self.dry_run:
                self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
                time.sleep(0.1)

            while not self._stop_requested:
                if num_steps is not None and exec_step >= num_steps:
                    logger.info(f"完成 {num_steps} 步")
                    break
                if timeout is not None and (time.monotonic() - start_time) > timeout:
                    logger.info(f"超时 {timeout}s")
                    break

                # ---- 需要重新推理 ----
                if chunk is None or chunk_idx >= self.chunk_execute:
                    obs = self.get_observation()
                    if obs is None:
                        break
                    current_state = obs["observation/state"]

                    t0 = time.monotonic()
                    outputs = self.policy.infer(obs)
                    infer_time = time.monotonic() - t0

                    chunk = outputs["actions"]  # (10, 7)
                    chunk_idx = 0

                    print(f"\n{'='*60}")
                    print(f"[Infer]  exec_step={exec_step}  耗时={infer_time*1000:.1f}ms")
                    print(f"  当前关节 : {_fmt(current_state[:6])}  夹爪={current_state[6]:.4f}")
                    print(f"  即将执行 chunk[0..{self.chunk_execute-1}]:")
                    for i in range(min(self.chunk_execute, len(chunk))):
                        diff = chunk[i, :6] - current_state[:6]
                        print(f"    [{i}]: {_fmt(chunk[i, :6])}  夹爪={chunk[i,6]:.4f}"
                            f"  Δ={_fmt(diff)}")

                # ---- 执行 chunk 中当前子步 ----
                step_start = time.monotonic()

                if not self.dry_run:
                    if not self._execute_action(exec_step, chunk[chunk_idx]):
                        self._stop_requested = True
                        break
                else:
                    print(f"  [DRY RUN] chunk[{chunk_idx}]: {_fmt(chunk[chunk_idx, :6])}"
                        f"  夹爪={chunk[chunk_idx, 6]:.4f}")

                chunk_idx += 1
                exec_step += 1

                elapsed = time.monotonic() - step_start
                sleep_time = self.inference_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("\nCtrl+C 中断")
        finally:
            self.shutdown()

    # ------------------------------------------------------------------ #
    #  清理
    # ------------------------------------------------------------------ #
    def _return_to_home(self):
        logger.info("回起始位置...")
        try:
            self.robot.left._robot.switch_mode(RobotMode.PLANNING_POS)
            self.robot.left._robot.move_to_joint_pos(HOME_JOINT, blocking=True)
            self.robot.left._robot.move_eef_pos(1.0)
            time.sleep(0.3)
            logger.info("已回起始位置")
        except Exception as e:
            logger.error(f"回起始位置失败: {e}")

    def shutdown(self):
        logger.info("关闭系统...")
        if not self.dry_run:
            self._return_to_home()
        try:
            self.robot.shutdown()
        except Exception as e:
            logger.error(f"关闭机械臂失败: {e}")
        if self.show_display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="AirBot Play 本地推理")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--port", type=int, default=50050)
    parser.add_argument("--task", type=str, default="do something")
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--play-config", type=str,
                        default=str(pathlib.Path(__file__).parent / "airbot" / "play_config.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印推理结果，不发出运动指令")
    parser.add_argument("--debug", action="store_true",
                        help="打印完整的 10 步动作块")
    parser.add_argument("--no-display", action="store_true",
                        help="禁用相机可视化窗口（用 stdin q 急停）")
    parser.add_argument("--min-z", type=float, default=-0.065,
                        help="末端Z轴最低高度(米)，低于此值只发夹爪不下压，如 -0.02")    
    parser.add_argument("--chunk", type=int, default=5,
                        help="每次推理后执行几步动作再重新推理 (default: 5)")    
    args = parser.parse_args()

    with open(args.play_config, "r", encoding="utf-8") as f:
        play_cfg = json.load(f)

    camera_cfg = play_cfg.get("cameras", {})

    runner = AirbotPlayInferenceRunner(
        checkpoint_dir=args.checkpoint,
        port=args.port,
        task_prompt=args.task,
        inference_freq=args.freq,
        head_camera_serial=camera_cfg.get("head_serial"),
        wrist_camera_serial=camera_cfg.get("wrist_serial"),
        dry_run=args.dry_run,
        debug=args.debug,
        no_display=args.no_display,
        min_z_height=args.min_z,
    )
    runner.run_continuous(num_steps=args.steps, timeout=args.timeout)


if __name__ == "__main__":
    main()