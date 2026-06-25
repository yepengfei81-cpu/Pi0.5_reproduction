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
sys.path.insert(0, str(_ROOT / "gripper_geom"))
from gripper_params import get_params  # noqa: E402  每爪开合范围 + 指尖偏移
from play_sdk import PlayRealRobot, RobotMode  # noqa: E402
from umi_to_lerobot import quat_to_rot, rot6d_to_rot, rot_to_6d  # noqa: E402
from replay_check import (  # noqa: E402
    rot_to_quat_xyzw, to_base_frame, base_to_wprime,
    load_home_joint, load_workspace,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRIPPER_MAX = 0.073          # 与训练归一化一致
WIPE_GLIDE_DEPTH = 0.286     # 训练里擦黑板"贴板滑动"的典型下压落差(W'z, 由遥操作数据统计)
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
                 start_joint=None, config_name="pi05_cotrain_eef", ensemble_m=0.1,
                 start_z_offset=0.0, board_z=None, press=0.005, gripper_pc_path=None,
                 gripper_name="parallel"):
        # 每爪开合范围(get_eef_pos) + 指尖偏移(EE系); parallel 默认 = 现状(offset0/0~0.073)
        gp = get_params(gripper_name)
        self.g_close = float(gp["close"]); self.g_open = float(gp["open"])
        self.tcp_offset = np.asarray(gp["tcp_offset"], float)
        self.task = task or "do something"
        self.config_name = config_name
        self.ensemble_m = ensemble_m   # temporal ensembling 指数权重衰减(越大=越偏最新)
        self.start_z_offset = start_z_offset   # 额外微调: 建 W' 时再抬/降一点(默认0)
        self.board_z = board_z   # 示教测得的黑板面 base z(给了就自动配准起点高度)
        self.press = press       # 想让板擦压进黑板多少(m), 默认 5mm 轻压
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

        print(f">>> [1/4] 加载模型 ({self.config_name})...", flush=True)
        self.cfg = _config.get_config(self.config_name)
        self.policy = _policy_config.create_trained_policy(
            self.cfg, pathlib.Path(checkpoint_dir), default_prompt=self.task)
        print(">>> [2/4] 模型加载完成", flush=True)

        # 夹爪几何 token: 给了 --gripper-pc 且模型开了 gripper_token, 就每帧注入当前夹爪
        # 的点云描述符(可以是训练见过的, 也可以是未见爪的 CAD -> 零样本)。
        self.gripper_pc = None
        if gripper_pc_path:
            if getattr(self.cfg.model, "gripper_token", False):
                d = np.load(gripper_pc_path, allow_pickle=True).item()
                pc = np.asarray(d["points"], np.float32)
                P = int(getattr(self.cfg.model, "num_gripper_points", 512))
                idx = np.random.default_rng(0).choice(len(pc), P, replace=len(pc) < P)
                self.gripper_pc = pc[idx]
                print(f">>> 夹爪 token: 载入 {gripper_pc_path} -> {self.gripper_pc.shape}", flush=True)
            else:
                print(">>> ⚠ 当前 config 未开 gripper_token, --gripper-pc 被忽略", flush=True)

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

    # ---- 指尖(TCP)偏移: get_end_pose 报的 EE 点 <-> 实际夹爪指尖 (offset=0 时为恒等) ----
    def _ee_to_tip(self, p, q):
        return np.asarray(p, float) + quat_to_rot(*q) @ self.tcp_offset

    def _tip_to_ee(self, p_tip, q):
        return np.asarray(p_tip, float) - quat_to_rot(*q) @ self.tcp_offset

    # ---- W' 建立 + 状态/动作变换 ----
    def establish_wprime(self):
        print(">>> 读取 get_end_pose 建立 W'（若卡在这说明位姿反馈阻塞）...", flush=True)
        p, q = self.robot.get_end_pose()
        print(f">>> get_end_pose 返回: pos={np.round(p,4).tolist()} quat={np.round(q,4).tolist()}", flush=True)
        R = quat_to_rot(*q)
        fwd = R[:, 0]
        self.start_pos = self._ee_to_tip(p, q)   # W' 原点锚到夹爪指尖(offset=0 即 EE 点)
        # 起点高度配准(接触任务): 给了 --board-z 就自动把 W' 原点 z 设成
        # board_z + 训练下压落差 - 想要的压入量, 让模型的下压正好落在"轻压黑板"处;
        # 否则用真机当前高度。再叠加 start_z_offset 做手动微调。只影响这次运行。
        if self.board_z is not None:
            self.start_pos[2] = self.board_z + WIPE_GLIDE_DEPTH - self.press
            print(f">>> 起点 z 按黑板自动配准: board_z={self.board_z:+.3f} press={self.press:.3f} "
                  f"-> 原点 z={self.start_pos[2]:.3f}", flush=True)
        self.start_pos[2] += self.start_z_offset
        self.start_yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
        print(f">>> W' 建立: 原点={self.start_pos.round(3).tolist()} "
              f"yaw0={self.start_yaw:.1f}° (start_z_offset={self.start_z_offset:+.3f})", flush=True)

    def get_state_wprime(self):
        p, q = self.robot.get_end_pose()
        R = quat_to_rot(*q)
        p_tip = self._ee_to_tip(p, q)
        pw, Rw = base_to_wprime(p_tip, R, self.start_pos, self.start_yaw, EYE3, [0, 0, 0])
        grip = self.robot.get_gripper_state() or 0.0
        g01 = float(np.clip((grip - self.g_close) / (self.g_open - self.g_close), 0.0, 1.0))
        return np.concatenate([pw, rot_to_6d(Rw), [g01]]).astype(np.float32)

    def get_observation(self):
        head_bgr, _ = next(self.robot.head_camera)
        wrist_bgr, _ = next(self.robot.left_wrist_camera)
        self._vis = (head_bgr, wrist_bgr)
        obs = {
            "observation/image": head_bgr[:, :, ::-1].copy().astype(np.uint8),
            "observation/wrist_image": wrist_bgr[:, :, ::-1].copy().astype(np.uint8),
            "observation/state": self.get_state_wprime(),
            "observation/env_mask": np.float32(self.env_mask),
            "prompt": self.task,
        }
        if self.gripper_pc is not None:
            obs["observation/gripper_pc"] = self.gripper_pc
        return obs

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

    # ---- temporal ensembling: 单动作换算 + 重叠 chunk 加权平均 ----
    def _action_to_base(self, a_w):
        """单个 W' 动作 (10,) -> (base pos(3), quat(xyzw), grip[0,1])。"""
        cmd_pos, quats, grip = self.chunk_to_base(a_w[None, :])
        return cmd_pos[0], quats[0], float(grip[0])

    def _ensemble(self, chunks, macro):
        """对覆盖当前 macro 时刻的所有 chunk 预测做指数加权平均(ACT 式 temporal ensembling)。

        chunks: [(start_macro, chunk_w (H,10))]。每个 chunk 在 start_macro 时刻推理得到，
        覆盖未来 [start, start+H)。当前 macro 若被多个 chunk 覆盖，就把它们对该时刻的
        预测加权平均 —— 抹平"相邻推理给出相反方向"造成的过冲/回退抖动。
        rot6d 直接线性平均(转旋转时会重新正交化)，pos/grip 也线性平均。
        """
        preds, weights = [], []
        n = len(chunks)
        for idx, (start, cw) in enumerate(chunks):
            off = macro - start
            if 0 <= off < len(cw):
                preds.append(cw[off])
                age = (n - 1) - idx          # 0=最新 chunk
                weights.append(float(np.exp(-self.ensemble_m * age)))
        if not preds:
            return None
        w = np.asarray(weights)
        w /= w.sum()
        return (w[:, None] * np.asarray(preds, dtype=np.float32)).sum(0).astype(np.float32)

    def _servo_macro(self, last_p, last_q, cmd_pos, cmd_q, g, macro_dt):
        """从 last 位姿插值到 cmd(单个 macro 时刻的集成动作)，servo_hz 高频下发。"""
        dt = 1.0 / self.servo_hz
        nsub = max(1, int(self.servo_hz * macro_dt))
        for s in range(nsub):
            t0 = time.monotonic()
            a = (s + 1) / nsub
            p = (1 - a) * last_p + a * cmd_pos
            q = nlerp(last_q, cmd_q, a)
            if p[2] >= self.min_z:                       # p 是指尖位姿; 下发前转回 EE 点
                ee = self._tip_to_ee(p, q)
                self.robot.servo_cart_pose(ee.tolist(), q.tolist())
            # 夹爪反归一化: g01 -> 该爪 [close, open] 的真实电机位置(避免对 GET 过挤压)
            g_cmd = self.g_close + g * (self.g_open - self.g_close)
            self.robot.left._robot.servo_eef_pos([g_cmd])
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
            macro = 0                 # 绝对 macro 时刻(=训练 10Hz 的一拍)
            chunks = []               # [(start_macro, chunk_w (H,10))] 最近若干个用于集成
            macro_dt = (1.0 / 10.0) / self.speed_scale
            if not self.dry_run:
                ee_p, last_q = self.robot.get_end_pose()
                last_p = self._ee_to_tip(ee_p, last_q)   # 插值起点用指尖位姿
            else:
                last_p, last_q = self.start_pos, [0.0, 0.0, 0.0, 1.0]
            last_p = np.asarray(last_p, float)
            last_q = np.asarray(last_q, float)
            first = True
            print(f">>> 进入推理循环（dry-run={self.dry_run}, ensemble_m={self.ensemble_m}, "
                  f"重推间隔={self.chunk_execute} macro）", flush=True)
            while not self._stop:
                if num_steps and macro >= num_steps:
                    break
                if timeout and time.monotonic() - t_start > timeout:
                    break

                # 每 chunk_execute 个 macro 重推理一次；新 chunk 与旧 chunk 在重叠区做集成
                if macro % self.chunk_execute == 0:
                    obs = self.get_observation()
                    if first:
                        print(">>> 首次推理 JIT 编译中，可能 1-2 分钟，请耐心等待...", flush=True)
                        first = False
                    ti = time.monotonic()
                    cw = np.asarray(self.policy.infer(obs)["actions"], dtype=np.float32)  # (H,10) W'
                    chunks.append((macro, cw))
                    # 保留足以覆盖重叠的最近几个 chunk: ceil(H / 重推间隔)
                    keep = max(2, -(-cw.shape[0] // self.chunk_execute))
                    chunks = chunks[-keep:]
                    st = obs["observation/state"]
                    print(f">>> [macro {macro}] 推理 {(time.monotonic()-ti)*1000:.0f}ms  "
                          f"W'pos={st[:3].round(3).tolist()} rot6d={st[3:9].round(2).tolist()} "
                          f"grip={st[9]:.2f}  (集成 {len(chunks)} 个 chunk)", flush=True)

                a_w = self._ensemble(chunks, macro)
                if a_w is None:
                    break
                cmd_pos, cmd_q, g = self._action_to_base(a_w)
                # 单 macro 位移软钳: 防野跳(集成后通常已很小)
                d = cmd_pos - last_p
                nd = float(np.linalg.norm(d))
                if nd > self.max_step_m:
                    cmd_pos = last_p + d / nd * self.max_step_m

                if self.dry_run:
                    if macro % self.chunk_execute == 0:
                        print(f"    -> 集成 base pos={np.round(cmd_pos,3).tolist()} grip={g:.2f}", flush=True)
                    self._poll_stop()
                    time.sleep(0.05)
                else:
                    self._servo_macro(last_p, last_q, cmd_pos, cmd_q, g, macro_dt)
                    last_p, last_q = cmd_pos, cmd_q
                macro += 1
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
    ap.add_argument("--chunk", type=int, default=5,
                    help="每隔几个 macro 重推理一次; 越小=重叠越多=集成越平滑(但推理更频繁)")
    ap.add_argument("--ensemble-m", type=float, default=0.1,
                    help="temporal ensembling 权重衰减: 0=各重叠 chunk 等权(最平滑); "
                         "越大越偏最新 chunk(更跟手但更抖); 关掉集成可设很大如 10")
    ap.add_argument("--env-mask", type=float, default=0.0,
                    help="环境相机是否有效: 擦黑板(单相机)=0; 双相机任务=1")
    ap.add_argument("--min-z", type=float, default=None,
                    help="base 系最低 z(m) 防撞桌; 不给时: 给了 --board-z 则自动= board_z-0.012, "
                         "否则 -0.06")
    ap.add_argument("--board-z", type=float, default=None,
                    help="【接触任务用】示教测得的黑板面 base z(m)。给了它脚本每次自动把起点"
                         "抬到'板擦轻压黑板'的高度, 不用你手算。配 --press 调压入量")
    ap.add_argument("--press", type=float, default=0.005,
                    help="板擦想压进黑板多少(m), 默认 0.005(5mm 轻压); 悬空调大, 堵转调小")
    ap.add_argument("--start-z-offset", type=float, default=0.0,
                    help="额外手动微调起点 base z(m), 叠加在 --board-z 配准之上; 默认 0")
    ap.add_argument("--max-step-mm", type=float, default=30.0, help="单步位移上限,超过判为模型抖动跳过")
    ap.add_argument("--start-joint", type=float, nargs=6, default=None,
                    metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
                    help="起始关节构型(6值)；不给则在当前位姿建 W'(请手动摆到训练开局姿态)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--config-name", default="pi05_cotrain_eef",
                    help="训练配置名: 协同=pi05_cotrain_eef; 测试③纯遥操作=pi05_teleop_eef")
    ap.add_argument("--gripper-pc", default=None,
                    help="当前夹爪的几何描述符 .npy(gripper_geom/*.npy)。模型开了 gripper_token "
                         "时部署必传, 告诉模型现在是哪把爪; 未见爪直接传它的 CAD npy 即可零样本")
    ap.add_argument("--gripper-name", default="parallel",
                    help="当前装的爪名(读 gripper_params.py 的开合范围+指尖偏移); 默认 parallel=现状")
    ap.add_argument("--dry-run", action="store_true", help="只推理+打印,不发运动")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    # min-z 解析: 给了黑板深度就自动放到板面下方 1.2cm 当安全底, 否则沿用 -0.06
    min_z = args.min_z
    if min_z is None:
        min_z = (args.board_z - 0.012) if args.board_z is not None else -0.06

    cam = json.load(open(args.play_config, encoding="utf-8")).get("cameras", {})
    runner = EEFInferenceRunner(
        checkpoint_dir=args.checkpoint, port=args.port, task=args.task,
        head_serial=cam.get("head_serial"), wrist_serial=cam.get("wrist_serial"),
        servo_hz=args.servo_hz, speed_profile=args.speed_profile,
        speed_scale=args.speed_scale, chunk_execute=args.chunk,
        env_mask=args.env_mask, min_z=min_z, max_step_mm=args.max_step_mm,
        dry_run=args.dry_run, no_display=args.no_display, config_path=args.play_config,
        start_joint=args.start_joint, config_name=args.config_name,
        ensemble_m=args.ensemble_m, start_z_offset=args.start_z_offset,
        board_z=args.board_z, press=args.press, gripper_pc_path=args.gripper_pc,
        gripper_name=args.gripper_name,
    )
    runner.run(num_steps=args.steps, timeout=args.timeout)


if __name__ == "__main__":
    main()
