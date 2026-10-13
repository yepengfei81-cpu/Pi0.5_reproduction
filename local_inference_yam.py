#!/usr/bin/env python3
"""YAM(i2rt) + GET 真机推理: 20D 双臂模型只驱动臂0(YAM), 臂1 按训练里单臂样本的 rest 编码。

与 local_inference_eef.py 的区别只在硬件层: YAM 没有笛卡尔伺服, 这里 观测 = FK(关节角) -> W' 状态,
动作 = W' -> base 位姿 -> 数值 IK(collect_data_yam.yam_ik) -> 关节角, 100Hz 关节插值下发。
坐标约定、ARP、夹爪归一化与采集脚本 collect_data_yam.py 完全一致(--tool-offset / --grip-* 必须同值)。

用法:
  python local_inference_yam.py --checkpoint <ckpt> --config-name pi05_single_yam_sweep \
      --task "sweep the objects into the dustpan" --gripper get_yam \
      --tool-offset 0.0XX --grip-open <raw> --grip-closed <raw> [--start-from-gello --gello-config <yaml>] --dry-run
  --fake: 假臂假相机, 只验证 模型->IK->指令 链路(不接硬件)。
急停: q / Esc(有窗口)或终端 q。退出后停在原地并张开夹爪。
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
for sub in ("airbot", "umi", "gripper_geom"):
    sys.path.insert(0, str(_ROOT / sub))
from collect_data_yam import (GelloLeader, YamFollower, YAM_JOINT_LIMITS, load_gello_yaml,  # noqa: E402
                              yam_fk_R, yam_ik)
from joint_guard import JointGuard  # noqa: E402
from umi_to_lerobot import rot6d_to_rot, rot_to_6d  # noqa: E402
from replay_check import to_base_frame, base_to_wprime  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
EYE3 = np.eye(3)
REST10 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)


class FakeFollower:
    def __init__(self):
        self.q = np.array([0.0, 1.2, 1.0, 0.0, 0.3, 0.0]); self.g = 0.8; self.gc, self.go = 0.0, 1.0
    def read(self): return self.q.copy(), self.g
    def grip01(self, raw): return float(raw)
    def command(self, q6, g01): self.q = np.asarray(q6, float).copy(); self.g = float(g01)
    def move_to(self, q6, g01, seconds=2.0): self.command(q6, g01)
    def close(self): pass


class FakeCams:
    def get_frames(self): z = np.zeros((480, 640, 3), np.uint8); return z, z
    def stop(self): pass


class YamRunner:
    def __init__(self, a):
        self.a = a; self._stop = False; self._vis = None
        print(f">>> [1/4] 加载模型 ({a.config_name})...", flush=True)
        self.cfg = _config.get_config(a.config_name)
        parts = pathlib.Path(a.checkpoint).resolve().parts
        if len(parts) >= 3 and parts[-3] != a.config_name:
            print(f">>> ⚠ checkpoint 路径里的配置名是 '{parts[-3]}', --config-name 是 '{a.config_name}'", flush=True)
        self.policy = _policy_config.create_trained_policy(self.cfg, pathlib.Path(a.checkpoint), default_prompt=a.task)
        if not bool(getattr(self.cfg.data, "dual", False)):
            sys.exit("本脚本按 20D 双臂模型写(臂1 rest 编码); 10D 单臂配置不支持")
        self.arm_frame = bool(getattr(self.cfg.data, "arm_frame", False))
        npz = a.grippers_npz or str(_ROOT / (getattr(self.cfg.data, "grippers_npz_path", None) or "gripper_geom/grippers.npz"))
        print(f">>> [2/4] 模型加载完成; arm_frame={self.arm_frame}; 点云={npz}", flush=True)
        self.pc0 = self.pc1 = None
        if getattr(self.cfg.model, "gripper_token", False):
            z = np.load(npz, allow_pickle=True); P = int(getattr(self.cfg.model, "num_gripper_points", 512))
            region = bool(getattr(self.cfg.model, "region_tokens", False)); rng = np.random.default_rng(0)
            def _pick(name):
                pc = np.asarray(z[f"{name}_points"], np.float32)
                if region:
                    reg = np.asarray(z[f"{name}_region"])
                    return np.stack([pc[rng.choice(np.flatnonzero(reg == r), P, replace=int((reg == r).sum()) < P)] for r in range(3)])
                return pc[rng.choice(len(pc), P, replace=len(pc) < P)]
            self.pc0, self.pc1 = _pick(a.gripper), _pick(a.arm1_cloud)
            print(f">>> 夹爪 token: 臂0={a.gripper}{self.pc0.shape} 臂1(rest)={a.arm1_cloud}", flush=True)
        print(">>> [3/4] 连接 YAM + 2 相机..." if not a.fake else ">>> [3/4] 假臂假相机", flush=True)
        if a.fake:
            self.follower, self.cams, self.leader = FakeFollower(), FakeCams(), None
        else:
            from collect_data import DualCamera
            self.cams = DualCamera(head_serial=a.head_serial, wrist_serial=a.wrist_serial)
            self.follower = YamFollower(a.follow_can, a.follow_gripper_type, a.grip_closed, a.grip_open)
            self.leader = None
            if a.start_from_gello:
                cfg = load_gello_yaml(a.gello_config) if a.gello_config else {}
                self.leader = GelloLeader(a.gello_port or cfg["port"], a.gello_ids or cfg["joint_ids"], a.gello_offsets or cfg["joint_offsets"],
                                          a.gello_signs or cfg["joint_signs"], a.gello_gripper or cfg.get("gripper_config"))
        print(">>> [4/4] 连接完成", flush=True)
        self.guard = JointGuard("yam", rate_hz=10, limits=YAM_JOINT_LIMITS, log=lambda m: print(m, flush=True))
        self.start_pos = self.start_yaw = None; self.last_p = None; self.last_q = None
        if not a.no_display:
            cv2.namedWindow("YAM Inference [q/Esc=stop]", cv2.WINDOW_NORMAL); cv2.resizeWindow("YAM Inference [q/Esc=stop]", 960, 300)

    # ---- 观测 ----
    def _fk(self, q):
        return yam_fk_R(q, self.a.tool_offset, self.a.tool_roll_deg)

    def state_wprime(self):
        q, graw = self.follower.read(); p, R = self._fk(q)
        pw, Rw = base_to_wprime(p, R, self.start_pos, self.start_yaw, EYE3, [0, 0, 0])
        return np.concatenate([pw, rot_to_6d(Rw), [self.follower.grip01(graw)]]).astype(np.float32), q, p

    def get_observation(self):
        head, wrist = self.cams.get_frames(); self._vis = (head, wrist)
        st0, q, p = self.state_wprime()
        obs = {"observation/image": head[:, :, ::-1].copy(), "observation/wrist_image": wrist[:, :, ::-1].copy(),
               "observation/wrist_image_1": np.zeros_like(head), "observation/env_mask": np.float32(1.0),
               "observation/arm1_mask": np.float32(0.0), "observation/state": np.concatenate([st0, REST10]).astype(np.float32),
               "prompt": self.a.task}
        if self.pc0 is not None:
            obs["observation/gripper_pc"], obs["observation/gripper_pc_1"] = self.pc0, self.pc1
        return obs, q, p

    # ---- 工具 ----
    def _poll_stop(self):
        if self.a.no_display or self._vis is None:
            return
        imgs = [cv2.resize(v, (400, 300)) for v in self._vis]
        for im, lab in zip(imgs, ("ENV [q=STOP]", "WRIST")):
            cv2.putText(im, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YAM Inference [q/Esc=stop]", np.hstack(imgs))
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            self._stop = True

    def _ensemble(self, chunks, macro):
        preds, w = [], []; n = len(chunks)
        for idx, (start, cw) in enumerate(chunks):
            off = macro - start
            if 0 <= off < len(cw):
                preds.append(cw[off]); w.append(float(np.exp(-self.a.ensemble_m * ((n - 1) - idx))))
        if not preds:
            return None
        w = np.asarray(w); w /= w.sum()
        return (w[:, None] * np.asarray(preds, np.float32)).sum(0).astype(np.float32)

    # ---- 主循环 ----
    def run(self):
        a = self.a
        if not a.no_display and not a.dry_run and not a.fake:
            pass
        else:
            threading.Thread(target=lambda: [setattr(self, "_stop", True) for line in iter(sys.stdin.readline, "") if line.strip().lower() in ("q", "stop")], daemon=True).start() if not a.fake else None
        try:
            if self.leader is not None and not a.dry_run:
                ql, gl = self.leader.read(); print(f">>> --start-from-gello: 移到 GELLO 当前关节 {np.round(np.degrees(ql)).tolist()}(会移动!)", flush=True)
                self.follower.move_to(ql, gl, seconds=2.5); time.sleep(0.3)
            q, graw = self.follower.read(); p, R = self._fk(q)
            self.start_pos = p.copy(); self.start_yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            self.last_p, self.last_q = p.copy(), q.copy()
            print(f">>> W' 原点={np.round(p, 3).tolist()} yaw0={self.start_yaw:.1f}°  关节(deg)={np.round(np.degrees(q)).tolist()}", flush=True)
            macro = 0; chunks = []; macro_dt = 0.1 / a.speed_scale; t_start = time.monotonic(); first = True
            print(f">>> 进入推理循环(dry-run={a.dry_run}, fake={a.fake}, chunk={a.chunk})", flush=True)
            while not self._stop:
                if a.steps and macro >= a.steps:
                    break
                if a.timeout and time.monotonic() - t_start > a.timeout:
                    break
                if macro % a.chunk == 0:
                    obs, q_now, p_now = self.get_observation()
                    if first:
                        print(">>> 首次推理 JIT 编译中, 可能 1-2 分钟...", flush=True); first = False
                    ti = time.monotonic(); cw = np.asarray(self.policy.infer(obs)["actions"], np.float32)
                    chunks.append((macro, cw)); chunks = chunks[-max(2, -(-cw.shape[0] // a.chunk)):]
                    st = obs["observation/state"]
                    print(f">>> [macro {macro}] 推理 {(time.monotonic()-ti)*1000:.0f}ms  W'pos={st[:3].round(3).tolist()} grip={st[9]:.2f} 爪chunk[{cw[0,9]:.2f}→{cw[-1,9]:.2f}]", flush=True)
                a_w = self._ensemble(chunks, macro)
                if a_w is None:
                    break
                cmd_pos, cmd_R = to_base_frame(a_w[None, :3], [rot6d_to_rot(a_w[3:9])], self.start_pos, self.start_yaw, EYE3, [0, 0, 0])
                p_t, R_t, g_t = cmd_pos[0], cmd_R[0], float(np.clip(a_w[9], 0.0, 1.0))
                d = p_t - self.last_p; nd = float(np.linalg.norm(d))
                if nd > a.max_step_mm / 1000.0:
                    p_t = self.last_p + d / nd * a.max_step_mm / 1000.0
                if a.min_z is not None and p_t[2] < a.min_z:
                    p_t = p_t.copy(); p_t[2] = a.min_z
                q_t, ep, er, ok = yam_ik(p_t, R_t, self.last_q, a.tool_offset, a.tool_roll_deg)
                if not ok and ep > 0.005:
                    print(f"    ⚠ IK 未收敛(残差 {ep*1000:.1f}mm/{er:.1f}°), 本拍保持", flush=True); q_t = self.last_q.copy()
                self.guard.update(q_t)
                if a.dry_run or a.fake:
                    if macro % a.chunk == 0:
                        print(f"    -> base 目标 {np.round(p_t, 3).tolist()} g={g_t:.2f}  关节(deg) {np.round(np.degrees(q_t)).tolist()}  IK {ep*1000:.2f}mm", flush=True)
                    if a.fake:
                        self.follower.command(q_t, g_t)
                    self._poll_stop(); time.sleep(0.02 if a.fake else 0.05)
                else:
                    nsub = max(1, int(a.servo_hz * macro_dt)); dt = 1.0 / a.servo_hz
                    for s in range(nsub):
                        t0 = time.monotonic(); al = (s + 1) / nsub
                        self.follower.command((1 - al) * self.last_q + al * q_t, g_t)
                        self._poll_stop()
                        if self._stop:
                            break
                        time.sleep(max(0.0, dt - (time.monotonic() - t0)))
                self.last_p, self.last_q = p_t, q_t; macro += 1
        except KeyboardInterrupt:
            logger.info("Ctrl+C 中断")
        finally:
            print(">>> " + self.guard.summary(), flush=True); self.shutdown()

    def shutdown(self):
        try:
            if not self.a.dry_run and not self.a.fake:
                q, _ = self.follower.read(); self.follower.command(q, 1.0)      # 停在原地, 张开夹爪
                print(">>> 已停在原地并张开夹爪(保持 PD 力矩; Ctrl+C 退出后 i2rt 会关闭)", flush=True)
            self.cams.stop(); self.follower.close()
        except Exception as e:
            logger.error(f"关闭失败: {e}")
        if not self.a.no_display:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="YAM + GET 真机推理(20D 模型驱动臂0)")
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--config-name", required=True); ap.add_argument("--task", required=True)
    ap.add_argument("--gripper", default="get_yam", help="臂0 点云名(npz 里的键)"); ap.add_argument("--arm1-cloud", default="parallel")
    ap.add_argument("--grippers-npz", default=None)
    ap.add_argument("--follow-can", default="can0"); ap.add_argument("--follow-gripper-type", default="linear_4310")
    ap.add_argument("--grip-closed", type=float, default=0.0); ap.add_argument("--grip-open", type=float, default=1.0)
    ap.add_argument("--tool-offset", type=float, default=0.0, help="与采集时同值"); ap.add_argument("--tool-roll-deg", type=float, default=0.0)
    ap.add_argument("--start-from-gello", action="store_true"); ap.add_argument("--gello-config", default=None)
    ap.add_argument("--gello-port", default=None); ap.add_argument("--gello-ids", type=int, nargs="*", default=None)
    ap.add_argument("--gello-offsets", type=float, nargs="*", default=None); ap.add_argument("--gello-signs", type=float, nargs="*", default=None)
    ap.add_argument("--gello-gripper", type=float, nargs=3, default=None)
    ap.add_argument("--head-serial", default="230422271972"); ap.add_argument("--wrist-serial", default="230422271433")
    ap.add_argument("--servo-hz", type=float, default=100.0); ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=10); ap.add_argument("--ensemble-m", type=float, default=0.3)
    ap.add_argument("--max-step-mm", type=float, default=30.0); ap.add_argument("--min-z", type=float, default=None, help="base 系最低 z(m), 防压桌; 按桌面实测设")
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--fake", action="store_true"); ap.add_argument("--no-display", action="store_true")
    a = ap.parse_args()
    if a.tool_offset == 0.0 and not a.fake:
        print(">>> ⚠ --tool-offset 为 0, 必须与采集时相同的值", flush=True)
    YamRunner(a).run()


if __name__ == "__main__":
    main()
