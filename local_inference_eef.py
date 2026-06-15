#!/usr/bin/env python3
"""
AirBot Play 本地实时推理（任务空间 EEF 版，用于 UMI+遥操作联合训练的 pi05_cotrain_eef）。

与 local_inference.py(关节空间) 的区别：
  - 模型 state/action 是任务空间 10D = pos(3) + rot6d(6) + gripper(1)，且在 W' 相对系。
  - 部署时每步把 get_end_pose(base) 反算成 W' 喂给模型；模型输出 W' 位姿再转回
    base，用高频 SERVO_CART_POSE 下发（chunk 间插值到 servo_hz，仿 replay_check
    验证过的方式）。
  - 单相机任务(擦黑板)用 env_mask=0 忽略环境相机槽。

W' 系：episode 开头读一次当前末端位姿，定义 W'(原点=当前指尖, z=重力, 当前 yaw 归零)，
之后整段都相对这个固定的 W'。等价于回放里的 start_pos/start_yaw，只是这里由真机
当前位姿决定，且 R_align=I（模型输出本就在工具系约定下）。

⚠️ 部署 ≠ 回放：模型输出可能跳变，speed-profile 先用 default、手放急停旁，
   确认平稳再考虑 fast。先 --dry-run（只推理+打印，不动）验证坐标链。

用法：
  # 干跑：只推理、打印 W' 状态和换算出的 base 指令，不发运动
  python local_inference_eef.py --checkpoint <ckpt> --task "wipe the blackboard" --dry-run

  # 真机（默认 default 档、250Hz、speed-scale 0.5）
  python local_inference_eef.py --checkpoint <ckpt> --task "wipe the blackboard"
急停：q / Esc（有窗口）或终端输入 q。
"""
import argparse
import json
import logging
import pathlib
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_ROOT / "airbot"))
sys.path.insert(0, str(_ROOT / "umi"))
from play_sdk import PlayRealRobot, RobotMode  # noqa: E402
from umi_to_lerobot import quat_to_rot, rot6d_to_rot, rot_to_6d  # noqa: E402
from replay_check import (  # noqa: E402
    rot_to_quat_xyzw, to_base_frame, base_to_wprime,
    load_home_joint, load_workspace,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRIPPER_MAX = 0.073          # 与训练归一化一致
EYE3 = np.eye(3)
ZERO_JOINT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 退出时回到的“零点”(与 local_inference.py 一致)


def nlerp(q0, q1, a):
    """四元数归一化线性插值（小步够用，避免依赖 scipy slerp）。"""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - a) * q0 + a * q1
    return q / np.linalg.norm(q)


class EEFInferenceRunner:
    def __init__(self, checkpoint_dir, port, task, head_serial, wrist_serial,
                 servo_hz, speed_profile, speed_scale, chunk_execute,
                 env_mask, min_z, max_step_mm, dry_run, no_display, config_path,
                 start_joint=None):
        self.task = task or "do something"
        self.servo_hz = servo_hz
        self.speed_profile = speed_profile
        self.speed_scale = speed_scale
        self.chunk_execute = chunk_execute
        self.env_mask = float(env_mask)
        self.min_z = min_z
        self.max_step_m = max_step_mm / 1000.0
        self.dry_run = dry_run
        self.show = not no_display
        self.config_path = pathlib.Path(config_path)
        self.start_joint = start_joint
        self._stop = False
        self._vis = None

        print(">>> [1/4] 加载模型 (pi05_cotrain_eef)...", flush=True)
        self.cfg = _config.get_config("pi05_cotrain_eef")
        self.policy = _policy_config.create_trained_policy(
            self.cfg, pathlib.Path(checkpoint_dir), default_prompt=self.task)
        print(">>> [2/4] 模型加载完成", flush=True)

        print(">>> [3/4] 连接机械臂 + 相机...", flush=True)
        self.robot = PlayRealRobot(port=port, enable_cameras=not dry_run or self.show,
                                   head_camera_serial=head_serial,
                                   left_wrist_camera_serial=wrist_serial)
        print(">>> [4/4] 机械臂 + 相机连接完成", flush=True)
        self.home_joint = load_home_joint(self.config_path)
        # W' 参数（episode 开头确定）
        self.start_pos = None
        self.start_yaw = None
        if self.show:
            cv2.namedWindow("EEF Inference [q/Esc=stop]", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("EEF Inference [q/Esc=stop]", 960, 360)

    # ---- W' 建立 + 状态/动作变换 ----
    def establish_wprime(self):
        print(">>> 读取 get_end_pose 建立 W'（若卡在这说明位姿反馈阻塞）...", flush=True)
        p, q = self.robot.get_end_pose()
        print(f">>> get_end_pose 返回: pos={np.round(p,4).tolist()} quat={np.round(q,4).tolist()}", flush=True)
        R = quat_to_rot(*q)
        fwd = R[:, 0]
        self.start_pos = np.asarray(p, float)
        self.start_yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
        print(f">>> W' 建立: 原点={self.start_pos.round(3).tolist()} "
              f"yaw0={self.start_yaw:.1f}°", flush=True)

    def get_state_wprime(self):
        p, q = self.robot.get_end_pose()
        R = quat_to_rot(*q)
        pw, Rw = base_to_wprime(p, R, self.start_pos, self.start_yaw, EYE3, [0, 0, 0])
        grip = self.robot.get_gripper_state() or 0.0
        g01 = float(np.clip(grip / GRIPPER_MAX, 0.0, 1.0))
        return np.concatenate([pw, rot_to_6d(Rw), [g01]]).astype(np.float32)

    def get_observation(self):
        head_bgr, _ = next(self.robot.head_camera)
        wrist_bgr, _ = next(self.robot.left_wrist_camera)
        self._vis = (head_bgr, wrist_bgr)
        return {
            "observation/image": head_bgr[:, :, ::-1].copy().astype(np.uint8),
            "observation/wrist_image": wrist_bgr[:, :, ::-1].copy().astype(np.uint8),
            "observation/state": self.get_state_wprime(),
            "observation/env_mask": np.float32(self.env_mask),
            "prompt": self.task,
        }

    def chunk_to_base(self, chunk):
        """模型输出 (H,10) W' 位姿 -> base 系 (cmd_pos, quats, grip01)。"""
        pos_w = chunk[:, :3]
        R_w = [rot6d_to_rot(a[3:9]) for a in chunk]
        cmd_pos, cmd_R = to_base_frame(pos_w, R_w, self.start_pos, self.start_yaw,
                                       EYE3, [0, 0, 0])
        quats = np.stack([rot_to_quat_xyzw(R) for R in cmd_R])
        grip = np.clip(chunk[:, 9], 0.0, 1.0)
        return cmd_pos, quats, grip

    # ---- 显示 / 急停 ----
    def _poll_stop(self):
        if not self.show or self._vis is None:
            return
        h = cv2.resize(self._vis[0], (480, 360))
        w = cv2.resize(self._vis[1], (480, 360))
        cv2.putText(h, "ENV [q=STOP]", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(w, "WRIST", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("EEF Inference [q/Esc=stop]", np.hstack([h, w]))
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            self._stop = True

    def _stdin_stop(self):
        def _l():
            while not self._stop:
                if sys.stdin.readline().strip().lower() in ("q", "stop"):
                    self._stop = True; break
        threading.Thread(target=_l, daemon=True).start()

    # ---- chunk 执行（servo 高频流式 + 插值）----
    def execute_chunk(self, cmd_pos, quats, grip):
        dt = 1.0 / self.servo_hz
        step_dt = (1.0 / 10.0) / self.speed_scale     # 训练 10Hz，每 chunk 步的真实时长
        k_exec = min(self.chunk_execute, len(cmd_pos) - 1)
        for k in range(k_exec):
            p0, p1 = cmd_pos[k], cmd_pos[k + 1]
            q0, q1 = quats[k], quats[k + 1]
            g0, g1 = float(grip[k]), float(grip[k + 1])
            # 跳变保护：单 chunk 步位移过大说明模型抖了，跳过该步
            if np.linalg.norm(p1 - p0) > self.max_step_m * (step_dt * 10):
                logger.warning(f"chunk step {k} 位移过大({np.linalg.norm(p1-p0)*1000:.0f}mm)，跳过")
                continue
            nsub = max(1, int(self.servo_hz * step_dt))
            for s in range(nsub):
                t0 = time.monotonic()
                a = (s + 1) / nsub
                p = (1 - a) * p0 + a * p1
                q = nlerp(q0, q1, a)
                if p[2] >= self.min_z:
                    self.robot.servo_cart_pose(p.tolist(), q.tolist())
                self.robot.left._robot.servo_eef_pos([(1 - a) * g0 + a * g1])
                self._poll_stop()
                if self._stop:
                    return
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))

    # ---- 主循环 ----
    def run(self, num_steps=None, timeout=None):
        print(">>> 进入 run()", flush=True)
        if not self.show and not self.dry_run:
            self._stdin_stop()
        try:
            if not self.dry_run and self.start_joint is not None:
                print(f">>> 移动到指定起始关节构型 {self.start_joint}", flush=True)
                self.robot.set_joint_positions(list(self.start_joint), blocking=True)
                time.sleep(0.3)
            else:
                print(">>> 在当前位姿建立 W'（不自动移动；请预先把机械臂摆到"
                      "与该任务训练开局相似的姿态）", flush=True)
            self.establish_wprime()

            if not self.dry_run:
                self.robot.set_speed_profile(self.speed_profile)
                self.robot.switch_mode(RobotMode.SERVO_CART_POSE)
                time.sleep(0.2)
                p, q = self.robot.get_end_pose()
                self.robot.servo_cart_pose(list(p), list(q))   # 播种保持点
                time.sleep(0.3)

            t_start = time.monotonic()
            step = 0
            first = True
            print(f">>> 进入推理循环（dry-run={self.dry_run}）", flush=True)
            while not self._stop:
                if num_steps and step >= num_steps:
                    break
                if timeout and time.monotonic() - t_start > timeout:
                    break
                obs = self.get_observation()
                if first:
                    print(">>> 首次推理 JIT 编译中，可能 1-2 分钟，请耐心等待...", flush=True)
                    first = False
                ti = time.monotonic()
                chunk = self.policy.infer(obs)["actions"]      # (H,10) W'
                cmd_pos, quats, grip = self.chunk_to_base(chunk)
                print(f">>> [step {step}] 推理 {(time.monotonic()-ti)*1000:.0f}ms  "
                      f"W'state pos={obs['observation/state'][:3].round(3).tolist()} "
                      f"rot6d={obs['observation/state'][3:9].round(2).tolist()} "
                      f"grip={obs['observation/state'][9]:.2f}\n"
                      f"    chunk[0] base pos={cmd_pos[0].round(3).tolist()} "
                      f"grip={grip[0]:.2f}  chunk[-1] base pos={cmd_pos[-1].round(3).tolist()}",
                      flush=True)
                if self.dry_run:
                    self._poll_stop()
                    time.sleep(0.1)
                else:
                    self.execute_chunk(cmd_pos, quats, grip)
                step += 1
        except KeyboardInterrupt:
            logger.info("Ctrl+C 中断")
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("关闭...")
        try:
            if not self.dry_run:
                self.robot.left._robot.switch_mode(RobotMode.PLANNING_POS)
                self.robot.set_joint_positions(ZERO_JOINT, blocking=True)   # 回零点(非 config 弯曲 home)
                try:
                    self.robot.left._robot.move_eef_pos(1.0)               # 张开夹爪
                except Exception:
                    pass
            self.robot.shutdown()
        except Exception as e:
            logger.error(f"关闭失败: {e}")
        if self.show:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="AirBot Play EEF(任务空间) 推理")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="wipe the blackboard")
    ap.add_argument("--port", type=int, default=50050)
    ap.add_argument("--play-config", default=str(_ROOT / "airbot" / "play_config.json"))
    ap.add_argument("--servo-hz", type=float, default=250.0)
    ap.add_argument("--speed-profile", choices=["slow", "default", "fast"], default="default",
                    help="部署先用 default；确认平稳再考虑 fast")
    ap.add_argument("--speed-scale", type=float, default=0.5,
                    help="chunk 执行速度倍率, 0.5=按演示一半速度执行(更安全)")
    ap.add_argument("--chunk", type=int, default=5, help="每次推理执行 chunk 前几步再重推")
    ap.add_argument("--env-mask", type=float, default=0.0,
                    help="环境相机是否有效: 擦黑板(单相机)=0; 双相机任务=1")
    ap.add_argument("--min-z", type=float, default=-0.06, help="base 系最低 z(m) 防撞桌")
    ap.add_argument("--max-step-mm", type=float, default=30.0, help="单步位移上限,超过判为模型抖动跳过")
    ap.add_argument("--start-joint", type=float, nargs=6, default=None,
                    metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
                    help="起始关节构型(6值)；不给则在当前位姿建 W'(请手动摆到训练开局姿态)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="只推理+打印,不发运动")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    cam = json.load(open(args.play_config, encoding="utf-8")).get("cameras", {})
    runner = EEFInferenceRunner(
        checkpoint_dir=args.checkpoint, port=args.port, task=args.task,
        head_serial=cam.get("head_serial"), wrist_serial=cam.get("wrist_serial"),
        servo_hz=args.servo_hz, speed_profile=args.speed_profile,
        speed_scale=args.speed_scale, chunk_execute=args.chunk,
        env_mask=args.env_mask, min_z=args.min_z, max_step_mm=args.max_step_mm,
        dry_run=args.dry_run, no_display=args.no_display, config_path=args.play_config,
        start_joint=args.start_joint,
    )
    runner.run(num_steps=args.steps, timeout=args.timeout)


if __name__ == "__main__":
    main()
