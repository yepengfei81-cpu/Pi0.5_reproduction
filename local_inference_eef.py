#!/usr/bin/env python3
"""
AirBot Play 任务空间(EEF) 实时推理 —— 单臂/双臂统一版。

一个 Runner 覆盖四种组合:
  模型维度(由 --config-name 对应 config 的 data.dual 自动判断, 不用手选):
    10D 单臂模型(如 pi05_cotrain_eef_grip_aug) / 20D 双臂模型(如 pi05_cotrain_dualarm)
  驱动硬件(--dual-arm 决定):
    只驱动臂0(右) / 驱动臂0+臂1(左)。20D 模型只驱动臂0时, 臂1 自动按训练里单臂样本
    的编码喂给模型(state=REST10, arm1_mask=0, 左腕黑图) —— 即原 --arm1-rest, 已自动化。

W' 系: 每条臂在开局位姿各建一个 W'(原点=指尖, yaw 归零), 模型输入输出均在 W' 相对系。
--start-from-lead: rollout 前把在场的 follow 臂对齐到各自 lead(遥操作臂)的当前关节角
——与采集时的对齐一致, 保证 IK 关节分支与示教相同(防撞限位)。单/双臂都支持。

用法示例:
  # 旧 10D 权重 + 单臂 + 从 lead 姿态开始
  python local_inference_eef.py --checkpoint <ckpt10d> --config-name pi05_cotrain_eef_grip_aug \
      --task "stack the bricks into a pyramid" --arm0-gripper parallel --start-from-lead
  # 20D 权重跑单臂任务(臂1 自动 rest)
  python local_inference_eef.py --checkpoint <ckpt20d> --config-name pi05_cotrain_dualarm \
      --task "stack the bricks into a pyramid" --arm0-gripper parallel --start-from-lead
  # 20D 双臂(切橡皮泥)
  python local_inference_eef.py --checkpoint <ckpt20d> --config-name pi05_cotrain_dualarm \
      --task "cut the play-dough with the knife" --dual-arm \
      --arm0-gripper get --arm1-gripper parallel --start-from-lead
  # 擦黑板(接触任务): 加 --env-mask 0 --board-z-auto

先 --dry-run(只推理+打印, 不动)验证坐标链。急停: q / Esc(有窗口)或终端输入 q。
退出后停在原地 + 张开夹爪; 按回车可选自动回零。
"""
import argparse
import logging
import pathlib
import sys
import threading
import time

import cv2
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_ROOT / "airbot"))
sys.path.insert(0, str(_ROOT / "umi"))
sys.path.insert(0, str(_ROOT / "gripper_geom"))
from collect_data import DualCamera  # noqa: E402  color-only 相机(并行启动, 带宽减半)
from gripper_params import get_params  # noqa: E402  每爪开合范围 + 指尖偏移 + board_z
from play_sdk import PlayRealRobot, RobotMode  # noqa: E402
from umi_to_lerobot import quat_to_rot, rot6d_to_rot, rot_to_6d  # noqa: E402
from replay_check import rot_to_quat_xyzw, to_base_frame, base_to_wprime  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WIPE_GLIDE_DEPTH = 0.286   # 擦黑板示教里"贴板滑动"的典型下压落差(W'z), board_z 配准用
EYE3 = np.eye(3)
ZERO_JOINT = [0.0] * 6
# 20D 双臂模型里单臂样本的臂1占位状态(pos0 + rot6d单位 + grip0), 与 pack_lerobot REST10 一致
REST10 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)


def nlerp(q0, q1, a):
    """四元数归一化线性插值(小步够用, 避免依赖 scipy slerp)。"""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - a) * q0 + a * q1
    return q / np.linalg.norm(q)


class _ArmCtx:
    """单条臂的部署上下文: 句柄 + 该爪开合范围/指尖偏移 + 点云 token + W' 帧 + servo 插值状态。"""

    def __init__(self, handle, gripper_name, gripper_cloud):
        self.h = handle
        gp = get_params(gripper_name)
        self.g_close = float(gp["close"]); self.g_open = float(gp["open"])
        self.tcp_offset = np.asarray(gp["tcp_offset"], float)
        self.name = gripper_name
        self.pc = gripper_cloud            # (P,3) 已子采样; None=不注入
        self.start_pos = None; self.start_yaw = None
        self.last_p = None; self.last_q = None

    def _ee_to_tip(self, p, q):
        return np.asarray(p, float) + quat_to_rot(*q) @ self.tcp_offset

    def _tip_to_ee(self, p_tip, q):
        return np.asarray(p_tip, float) - quat_to_rot(*q) @ self.tcp_offset

    def establish_wprime(self, board_z=None, press=0.005, z_offset=0.0):
        """在当前位姿建 W'。board_z(擦黑板等接触任务): 把原点 z 配准到
        board_z + 示教下压落差 - 压入量, 让模型的下压正好落在"轻压板面"处。"""
        p, q = self.h.get_end_pose()
        R = quat_to_rot(*q); fwd = R[:, 0]
        self.start_pos = self._ee_to_tip(p, q)   # W' 原点锚到指尖
        if board_z is not None:
            self.start_pos[2] = board_z + WIPE_GLIDE_DEPTH - press
            print(f">>>   [{self.name}] 起点 z 按 board_z 配准: {board_z:+.3f} press={press:.3f} "
                  f"-> 原点 z={self.start_pos[2]:.3f}", flush=True)
        self.start_pos[2] += z_offset
        self.start_yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
        print(f">>>   [{self.name}] W' 原点={self.start_pos.round(3).tolist()} "
              f"yaw0={self.start_yaw:.1f}°", flush=True)

    def state_wprime(self):
        """当前 base 位姿 -> 本臂 W' 系 10D = pos3 + rot6d6 + grip1(按该爪开合范围归一化)。"""
        p, q = self.h.get_end_pose()
        R = quat_to_rot(*q)
        p_tip = self._ee_to_tip(p, q)
        pw, Rw = base_to_wprime(p_tip, R, self.start_pos, self.start_yaw, EYE3, [0, 0, 0])
        eef = self.h.get_eef_pos()
        g = eef[0] if eef and len(eef) > 0 else 0.0
        g01 = float(np.clip((g - self.g_close) / (self.g_open - self.g_close), 0.0, 1.0))
        return np.concatenate([pw, rot_to_6d(Rw), [g01]]).astype(np.float32)

    def action_to_base(self, a_w):
        """本臂 W' 动作 (10,) -> (base 指尖 pos(3), quat(xyzw), grip01)。"""
        cmd_pos, cmd_R = to_base_frame(a_w[None, :3], [rot6d_to_rot(a_w[3:9])],
                                       self.start_pos, self.start_yaw, EYE3, [0, 0, 0])
        return cmd_pos[0], rot_to_quat_xyzw(cmd_R[0]), float(np.clip(a_w[9], 0.0, 1.0))

    def seed_last(self, dry_run):
        if dry_run:
            self.last_p = np.asarray(self.start_pos, float)
            self.last_q = np.asarray([0.0, 0.0, 0.0, 1.0], float)
        else:
            ee_p, q = self.h.get_end_pose()
            self.last_p = np.asarray(self._ee_to_tip(ee_p, q), float)
            self.last_q = np.asarray(q, float)


class EEFRunner:
    """统一推理 Runner: 模型 10D/20D 自动判断(config.data.dual), 硬件单/双臂由 --dual-arm 决定。"""

    def __init__(self, a, board_z, min_z):
        self.task = a.task
        self.servo_hz = a.servo_hz
        self.speed_profile = a.speed_profile
        self.speed_scale = a.speed_scale
        self.chunk_execute = a.chunk
        self.ensemble_m = a.ensemble_m
        self.env_mask = float(a.env_mask)
        self.min_z = min_z
        self.max_step_m = a.max_step_mm / 1000.0
        self.dry_run = a.dry_run
        self.show = not a.no_display
        self.start_from_lead = a.start_from_lead
        self.lead_ports = (a.lead_port0, a.lead_port1)
        self.board_z = board_z; self.press = a.press; self.z_offset = a.start_z_offset
        self._stop = False
        self._vis = None

        print(f">>> [1/4] 加载模型 ({a.config_name})...", flush=True)
        self.cfg = _config.get_config(a.config_name)
        self.policy = _policy_config.create_trained_policy(
            self.cfg, pathlib.Path(a.checkpoint), default_prompt=self.task)
        # 模型维度: 20D 双臂模型(data.dual=True) or 10D 单臂模型
        self.model_dual = bool(getattr(self.cfg.data, "dual", False))
        if a.dual_arm and not self.model_dual:
            sys.exit(f"--dual-arm 需要 20D 双臂模型, 但 {a.config_name} 是单臂 10D 配置")
        print(f">>> [2/4] 模型加载完成 (模型={'20D双臂' if self.model_dual else '10D单臂'}, "
              f"驱动={'双臂' if a.dual_arm else '仅臂0'})", flush=True)

        # 夹爪几何 token: 统一从 grippers.npz 按爪名取(与训练同源), 子采样到 num_gripper_points
        cloud0 = cloud1 = None
        if getattr(self.cfg.model, "gripper_token", False):
            z = np.load(a.grippers_npz, allow_pickle=True)
            P = int(getattr(self.cfg.model, "num_gripper_points", 512))
            rng = np.random.default_rng(0)

            def _pick(name):
                pc = np.asarray(z[f"{name}_points"], np.float32)
                idx = rng.choice(len(pc), P, replace=len(pc) < P)
                return pc[idx]
            cloud0 = _pick(a.arm0_gripper)
            cloud1 = _pick(a.arm1_gripper) if self.model_dual else None
            print(f">>> 夹爪 token: 臂0={a.arm0_gripper}{cloud0.shape}"
                  + (f" 臂1={a.arm1_gripper}{cloud1.shape}" if cloud1 is not None else ""), flush=True)
        else:
            print(">>> ⚠ 当前 config 未开 gripper_token, 不注入点云", flush=True)

        # 臂(play_sdk) + 相机(collect_data.DualCamera, color-only)
        if a.dual_arm:
            print(">>> [3/4] 连接双臂 + 3 相机...", flush=True)
            self.robot = PlayRealRobot(left_port=a.port1, right_port=a.port, enable_cameras=False)
            arm0_handle, arm1_handle = self.robot.right, self.robot.left
            wrist1_serial = a.arm1_wrist_serial
        else:
            print(">>> [3/4] 连接单臂(臂0) + 2 相机...", flush=True)
            self.robot = PlayRealRobot(port=a.port, enable_cameras=False)
            arm0_handle, arm1_handle = self.robot.left, None   # play_sdk 单臂句柄挂在 .left
            wrist1_serial = None                               # 不启动 -> 取帧自动黑图
        self.cams = DualCamera(head_serial=a.head_serial,
                               wrist_serial=a.arm0_wrist_serial,
                               wrist1_serial=wrist1_serial)
        print(">>> [4/4] 连接完成", flush=True)

        self.arm0 = _ArmCtx(arm0_handle, a.arm0_gripper, cloud0)
        self.arm1 = _ArmCtx(arm1_handle, a.arm1_gripper, cloud1) if arm1_handle else None
        self.pc1 = cloud1   # 20D 模型只驱动臂0时也照喂(被 arm1_mask=0 屏蔽)

        if self.show:
            cv2.namedWindow("EEF Inference [q/Esc=stop]", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("EEF Inference [q/Esc=stop]", 1200 if a.dual_arm else 960, 300)

    def _arms(self):
        return (self.arm0,) if self.arm1 is None else (self.arm0, self.arm1)

    # ---- 开局对齐: follow 移到各自 lead 的当前关节角(同采集, 关节分支=示教) ----
    def _align_to_lead(self):
        print(">>> --start-from-lead: follow 对齐到 lead 的关节角(会移动!)", flush=True)
        from airbot_py.arm import AIRBOTArm  # noqa: PLC0415  与 collect_data 同款直连
        for ctx, port in zip(self._arms(), self.lead_ports):
            lead = AIRBOTArm(url="localhost", port=port)
            if not lead.connect():
                print(f">>> ⚠ [{ctx.name}] 连不上 lead(:{port}), 跳过该臂对齐", flush=True)
                continue
            lj = lead.get_joint_pos()
            lead.disconnect()
            if lj is None:
                print(f">>> ⚠ [{ctx.name}] lead(:{port}) 读不到关节角, 跳过", flush=True)
                continue
            print(f">>> [{ctx.name}] 对齐到 lead(:{port}): {[f'{v:.3f}' for v in lj]}", flush=True)
            ctx.h.set_joint_positions(list(lj), blocking=True)   # PLANNING 模式阻塞到位
        time.sleep(0.3)

    # ---- 观测: 按模型维度组装(20D 缺臂1时按训练的单臂编码: REST10 + mask0 + 黑图) ----
    def get_observation(self):
        head_bgr, w0_bgr, w1_bgr = self.cams.get_frames_dual()
        self._vis = (head_bgr, w0_bgr, w1_bgr) if self.model_dual else (head_bgr, w0_bgr)
        st0 = self.arm0.state_wprime()
        obs = {
            "observation/image": head_bgr[:, :, ::-1].copy().astype(np.uint8),
            "observation/wrist_image": w0_bgr[:, :, ::-1].copy().astype(np.uint8),
            "observation/env_mask": np.float32(self.env_mask),
            "prompt": self.task,
        }
        if self.model_dual:
            st1 = self.arm1.state_wprime() if self.arm1 is not None else REST10
            obs["observation/state"] = np.concatenate([st0, st1]).astype(np.float32)
            obs["observation/wrist_image_1"] = w1_bgr[:, :, ::-1].copy().astype(np.uint8)
            obs["observation/arm1_mask"] = np.float32(1.0 if self.arm1 is not None else 0.0)
            if self.arm0.pc is not None:
                obs["observation/gripper_pc"] = self.arm0.pc
                obs["observation/gripper_pc_1"] = self.pc1
        else:
            obs["observation/state"] = st0
            if self.arm0.pc is not None:
                obs["observation/gripper_pc"] = self.arm0.pc
        return obs

    # ---- 显示 / 急停 ----
    def _poll_stop(self):
        if not self.show or self._vis is None:
            return
        labels = ("ENV [q=STOP]", "ARM0 WRIST(R)", "ARM1 WRIST(L)")
        imgs = [cv2.resize(v, (400, 300)) for v in self._vis]
        for im, lab in zip(imgs, labels):
            cv2.putText(im, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("EEF Inference [q/Esc=stop]", np.hstack(imgs))
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            self._stop = True

    def _stdin_stop(self):
        def _l():
            while not self._stop:
                if sys.stdin.readline().strip().lower() in ("q", "stop"):
                    self._stop = True; break
        threading.Thread(target=_l, daemon=True).start()

    # ---- temporal ensembling: 对整条动作向量(10D/20D)加权平均, 再拆臂 ----
    def _ensemble(self, chunks, macro):
        preds, weights = [], []
        n = len(chunks)
        for idx, (start, cw) in enumerate(chunks):
            off = macro - start
            if 0 <= off < len(cw):
                preds.append(cw[off])
                weights.append(float(np.exp(-self.ensemble_m * ((n - 1) - idx))))
        if not preds:
            return None
        w = np.asarray(weights); w /= w.sum()
        return (w[:, None] * np.asarray(preds, dtype=np.float32)).sum(0).astype(np.float32)

    def _clamp_step(self, arm, cmd_pos):
        d = cmd_pos - arm.last_p
        nd = float(np.linalg.norm(d))
        if nd > self.max_step_m:
            cmd_pos = arm.last_p + d / nd * self.max_step_m
        return cmd_pos

    def _servo_macro(self, tgt0, tgt1, macro_dt):
        """在场的臂同时从各自 last 插值到目标, servo_hz 高频下发。tgt=(pos,quat,g); tgt1=None 只动臂0。"""
        dt = 1.0 / self.servo_hz
        nsub = max(1, int(self.servo_hz * macro_dt))
        moves = [(self.arm0, self.arm0.last_p, self.arm0.last_q, *tgt0)]
        if tgt1 is not None:
            moves.append((self.arm1, self.arm1.last_p, self.arm1.last_q, *tgt1))
        for s in range(nsub):
            t0 = time.monotonic(); a = (s + 1) / nsub
            for arm, lp, lq, pt, qt, g in moves:
                p = (1 - a) * lp + a * pt
                q = nlerp(lq, qt, a)
                if p[2] >= self.min_z:                       # 指尖位姿; 下发前转回 EE 点
                    ee = arm._tip_to_ee(p, q)
                    arm.h.servo_cart_pose(ee.tolist(), q.tolist())
                arm.h._robot.servo_eef_pos([arm.g_close + g * (arm.g_open - arm.g_close)])
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
            if self.start_from_lead and not self.dry_run:
                self._align_to_lead()
            else:
                print(">>> 在当前位姿建立 W'(不自动移动; 请先把臂摆到与训练开局相似的姿态)", flush=True)
            # W': 臂0 带 board_z/press/z_offset 配准(接触任务); 臂1 用原始位姿
            self.arm0.establish_wprime(self.board_z, self.press, self.z_offset)
            if self.arm1 is not None:
                self.arm1.establish_wprime()

            if not self.dry_run:
                self.robot.set_speed_profile(self.speed_profile)   # 对在场的臂都设速度档
                for arm in self._arms():
                    arm.h.switch_mode(RobotMode.SERVO_CART_POSE)
                time.sleep(0.2)
                for arm in self._arms():        # 播种保持点
                    p, q = arm.h.get_end_pose()
                    arm.h.servo_cart_pose(list(p), list(q))
                time.sleep(0.3)
            for arm in self._arms():
                arm.seed_last(self.dry_run)

            t_start = time.monotonic()
            macro = 0                 # 绝对 macro 时刻(=训练 10Hz 的一拍)
            chunks = []               # [(start_macro, chunk (H, 10/20))] 用于时序集成
            macro_dt = (1.0 / 10.0) / self.speed_scale
            first = True
            print(f">>> 进入推理循环(dry-run={self.dry_run}, ensemble_m={self.ensemble_m}, "
                  f"重推间隔={self.chunk_execute}, speed_scale={self.speed_scale})", flush=True)
            while not self._stop:
                if num_steps and macro >= num_steps:
                    break
                if timeout and time.monotonic() - t_start > timeout:
                    break

                if macro % self.chunk_execute == 0:
                    obs = self.get_observation()
                    if first:
                        print(">>> 首次推理 JIT 编译中, 可能 1-2 分钟...", flush=True)
                        first = False
                    ti = time.monotonic()
                    cw = np.asarray(self.policy.infer(obs)["actions"], dtype=np.float32)
                    chunks.append((macro, cw))
                    keep = max(2, -(-cw.shape[0] // self.chunk_execute))
                    chunks = chunks[-keep:]
                    st = obs["observation/state"]
                    msg = (f">>> [macro {macro}] 推理 {(time.monotonic()-ti)*1000:.0f}ms  "
                           f"臂0 pos={st[:3].round(3).tolist()} grip={st[9]:.2f} "
                           f"爪chunk[{cw[0,9]:.2f}→{cw[-1,9]:.2f}]")   # 首→末: 全程持平=不动点, 末端抬升=打算张开
                    if self.model_dual and self.arm1 is not None:
                        msg += f" | 臂1 pos={st[10:13].round(3).tolist()} grip={st[19]:.2f}"
                    print(msg + f"  (集成 {len(chunks)})", flush=True)

                a_w = self._ensemble(chunks, macro)
                if a_w is None:
                    break
                p0, q0, g0 = self.arm0.action_to_base(a_w[:10])
                p0 = self._clamp_step(self.arm0, p0)
                tgt1 = None
                if self.arm1 is not None:
                    p1, q1, g1 = self.arm1.action_to_base(a_w[10:20])
                    tgt1 = (self._clamp_step(self.arm1, p1), q1, g1)

                if self.dry_run:
                    if macro % self.chunk_execute == 0:
                        msg = f"    -> 臂0 base pos={np.round(p0,3).tolist()} g={g0:.2f}"
                        if tgt1 is not None:
                            msg += f" | 臂1 base pos={np.round(tgt1[0],3).tolist()} g={tgt1[2]:.2f}"
                        print(msg, flush=True)
                    self._poll_stop()
                    time.sleep(0.05)
                else:
                    self._servo_macro((p0, q0, g0), tgt1, macro_dt)
                    self.arm0.last_p, self.arm0.last_q = p0, q0
                    if tgt1 is not None:
                        self.arm1.last_p, self.arm1.last_q = tgt1[0], tgt1[1]
                macro += 1
        except KeyboardInterrupt:
            logger.info("Ctrl+C 中断")
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("关闭...")
        try:
            if not self.dry_run:
                # 停在原地: 切 PLANNING 保持当前位姿, 夹爪张到最大(松开持物)
                for arm in self._arms():
                    try:
                        arm.h._robot.switch_mode(RobotMode.PLANNING_POS)
                    except Exception:
                        pass
                    try:
                        arm.h._robot.move_eef_pos(1.0)
                    except Exception:
                        pass
                print(">>> 已停在原地并张开夹爪。", flush=True)
                # 可选回零: 回车=回零; n/Ctrl+C/无stdin = 保持原地
                do_home = False
                try:
                    ans = input(">>> 按【回车】自动回零点, 或输入 n 后回车保持原地: ").strip().lower()
                    do_home = (ans == "")
                except (EOFError, KeyboardInterrupt):
                    print("\n>>> 跳过自动回零, 保持原地", flush=True)
                if do_home:
                    for arm in self._arms():
                        try:
                            arm.h._robot.switch_mode(RobotMode.PLANNING_POS)
                            arm.h.set_joint_positions(list(ZERO_JOINT), blocking=True)
                            print(f">>> [{arm.name}] 已回零点", flush=True)
                        except Exception as e:
                            print(f">>> [{arm.name}] 回零失败: {e}", flush=True)
                else:
                    print(">>> 保持原地; 需要时用 test_play.py 手动归零", flush=True)
            try:
                self.cams.stop()
            except Exception:
                pass
            self.robot.shutdown()
        except Exception as e:
            logger.error(f"关闭失败: {e}")
        if self.show:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="AirBot Play EEF(任务空间) 推理 — 单/双臂统一")
    # 模型
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-name", required=True,
                    help="训练配置名(决定模型维度): 10D=pi05_cotrain_eef_grip_aug 等; "
                         "20D=pi05_cotrain_dualarm")
    ap.add_argument("--task", required=True, help="prompt, 必须与训练 tasks.jsonl 逐字一致")
    # 硬件
    ap.add_argument("--dual-arm", action="store_true",
                    help="驱动两条臂(需 20D 模型)。不加=只驱动臂0(右); 20D 模型时臂1自动按 rest 编码")
    ap.add_argument("--port", type=int, default=50050, help="臂0(右)follow 端口")
    ap.add_argument("--port1", type=int, default=50052, help="臂1(左)follow 端口")
    ap.add_argument("--start-from-lead", action="store_true",
                    help="rollout 前 follow 对齐到 lead(遥操作臂)当前关节角(同采集, 防撞限位)。"
                         "先把 lead 摆到示教开局姿态! 单/双臂都支持")
    ap.add_argument("--lead-port0", type=int, default=50051, help="臂0 lead 端口(同 collect_data)")
    ap.add_argument("--lead-port1", type=int, default=50053, help="臂1 lead 端口(同 collect_data)")
    # 夹爪(与臂号解耦, 装什么写什么)
    ap.add_argument("--arm0-gripper", default="parallel",
                    help="臂0(右)实际装的爪 parallel/get; 决定点云 token+开合范围+TCP偏移")
    ap.add_argument("--arm1-gripper", default="parallel", help="臂1(左)实际装的爪")
    ap.add_argument("--grippers-npz", default=str(_ROOT / "gripper_geom" / "grippers.npz"),
                    help="多爪点云库, 按爪名取(与训练同源)")
    # 相机
    ap.add_argument("--head-serial", default="230422271972", help="环境(头)相机 SN")
    ap.add_argument("--arm0-wrist-serial", default="230422271433", help="臂0(右)手眼相机 SN")
    ap.add_argument("--arm1-wrist-serial", default="218622271178", help="臂1(左)手眼相机 SN")
    # 任务相关
    ap.add_argument("--env-mask", type=float, default=1.0,
                    help="环境(头)相机有效性: 默认 1; 擦黑板(单相机训练)传 0")
    ap.add_argument("--board-z", type=float, default=None,
                    help="[接触任务] 示教测得的板面 base z(m), 自动配准起点高度(臂0)")
    ap.add_argument("--board-z-auto", action="store_true",
                    help="[接触任务] 读 gripper_params.py 里臂0爪的 board_z(擦黑板专用)")
    ap.add_argument("--press", type=float, default=0.005, help="压入板面量(m), 配 board_z")
    ap.add_argument("--start-z-offset", type=float, default=0.0, help="臂0 W' 原点 z 手动微调(m)")
    ap.add_argument("--min-z", type=float, default=None,
                    help="base 系最低 z(m) 防撞桌; 缺省: board_z-0.012, 无 board_z 时 -0.12")
    # 执行
    ap.add_argument("--servo-hz", type=float, default=250.0)
    ap.add_argument("--speed-profile", choices=["slow", "default", "fast"], default="fast")
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="执行速度倍率; 1.0=按示教原速(勿超过 1.0)")
    ap.add_argument("--chunk", type=int, default=10,
                    help="每隔几拍重推理; 10=整段开环执行(离散事件如松爪不被截断, 推荐); "
                         "动态场景需快速反应时调小(如5)")
    ap.add_argument("--ensemble-m", type=float, default=0.3,
                    help="时序集成权重衰减(仅 chunk<horizon 有重叠时生效)")
    ap.add_argument("--max-step-mm", type=float, default=30.0, help="单拍位移上限(防野跳)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="只推理+打印, 不发运动")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    # board_z: 显式 > auto(读臂0爪的 gripper_params) > 无
    board_z = args.board_z
    if board_z is None and args.board_z_auto:
        board_z = get_params(args.arm0_gripper).get("board_z")
        if board_z is None:
            sys.exit(f"--board-z-auto: gripper_params 里 '{args.arm0_gripper}' 没有 board_z")
        print(f">>> --board-z-auto: 用 {args.arm0_gripper} 的 board_z={board_z:+.3f}", flush=True)
    min_z = args.min_z if args.min_z is not None else (
        board_z - 0.012 if board_z is not None else -0.12)

    runner = EEFRunner(args, board_z=board_z, min_z=min_z)
    runner.run(num_steps=args.steps, timeout=args.timeout)


if __name__ == "__main__":
    main()
