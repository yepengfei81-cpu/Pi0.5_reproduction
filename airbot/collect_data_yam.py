#!/usr/bin/env python3
"""YAM(i2rt) 机械臂 + GELLO/第二台 YAM 主臂 遥操作采集 -> 与 collect_data.py 完全同格式的 LeRobot 数据集。

输出可直接喂 umi/pack_lerobot_fast.py(--teleop-dir ... --teleop-gripper get_yam), 和 AIRBOT 数据共训:
  image / wrist_image           640x480 视频(环境相机 + 装在 GET 上的手眼)
  state   7 = follower 6 关节(rad) + 夹爪(归一化 0=闭 1=开)
  actions 7 = leader  6 关节(rad) + 夹爪指令(归一化)
  state_eef   8 = FK(follower 关节) 的 base 系【机械臂参考点 ARP】位姿 pos3+quat4(xyzw) + 夹爪
  actions_eef 8 = FK(leader 关节)  同上 + 夹爪指令
  telemetry/episode_XXXXXX.npz  100Hz 关节角/速度/力矩(来自 i2rt get_observations)

坐标约定(必须和 AIRBOT 数据一致, 否则 W' 和夹爪点云都会错):
  工具系 +X = 接近方向(指尖朝向), +Y = 夹爪开合轴, +Z = 上。YAM URDF 的 gripper 系是 +Z 前/+X 右/+Y 下,
  脚本里已换算。ARP(机械臂参考点) = AIRBOT 出厂平行爪指尖所在的点 = 安装面前方 73.5 mm; GET 指尖在 ARP 前方
  17 mm(gripper_params 的 tcp_offset)。所以 --tool-offset 应设为: 【GET 指尖到 YAM gripper 系原点沿接近方向
  的距离 − 0.017】。装好 GET 后用卡尺量一次(gripper 系原点在 joint6 法兰面, 见 URDF)。
  --tool-roll-deg: GET 的开合轴若不与 YAM gripper 系的 X 轴(右)对齐, 绕接近方向补转。

主臂两种接法:
  --leader gello   : GELLO(Dynamixel) 经 gello_software 的 DynamixelRobot 读, 参数来自
                     scripts/generate_yam_config.py 生成的 yaml(--gello-config), 也可用 --gello-* 覆盖
  --leader yam     : 第二台 YAM / 示教手柄经 CAN 读(--leader-can can1), 零重力模式手拖

夹爪: follower 的 i2rt 读数单位不确定(命令空间由 gripper_limits 定义), 脚本启动会打印当前原始读数;
  先手动全开/全闭各读一次, 填 --grip-open / --grip-closed, 存盘统一归一化到 0=闭 1=开。
  leader 的 GELLO 夹爪由 gello 归一化为 0=开 1=闭(脚本已翻转); leader=yam 时同 follower 的量程参数。

用法:
  python airbot/collect_data_yam.py --fk-check                       # 不连硬件: 验证内嵌 FK
  python airbot/collect_data_yam.py --fake --task x --repo-id t -o /tmp/yamtest   # 不连硬件: 假臂走通写盘
  python airbot/collect_data_yam.py --dry-run --leader gello --gello-config ~/gello_software/configs/yam_auto_generated.yaml
  python airbot/collect_data_yam.py --leader gello --gello-config <yaml> \
      --task "sweep the objects into the dustpan" --repo-id yam_get_sweep --output-dir ~/pi_data/YAM/sweep_task \
      --tool-offset 0.0XX --grip-open <raw> --grip-closed <raw>
流程同 collect_data.py: 回车开录, 回车结束, 回车保存/d 丢弃, q 退出。
"""
import argparse
import logging
import math
import select
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_guard import JointGuard  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

CONTROL_FREQ = 100
RECORD_FREQ = 10
HEAD_CAMERA_SERIAL = "230422271972"
WRIST_CAMERA_SERIAL = "230422271433"   # 装在 GET 上的手眼, 随夹爪一起搬到 YAM

# ---- YAM v1 运动学(i2rt/robot_models/arm/yam/v1/yam.urdf, base -> gripper 系; 与官方 PoE 模型一致到 4e-3 mm) ----
YAM_CHAIN = [  # (xyz, rpy, axis)
    ((0.0, 0.0, 0.068), (0.0, 1.5708, 3.1416), (-1.0, 0.0, 0.0)),
    ((-0.0455, -0.0339, -0.02), (0.0, 0.0, -3.14159), (0.0, 1.0, 0.0)),
    ((0.0, -0.0688, 0.264), (0.0, 0.0, -3.14159), (0.0, 1.0, 0.0)),
    ((-0.0600003, -0.0688, -0.244999), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.0405003, 0.0338995, -0.0739989), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ((0.0404996, 0.0, -0.0356), (3.1415926535897927, 0.0, 1.5707963267949736), (0.0, 0.0, 1.0)),
]
YAM_JOINT_LIMITS = ((-2.61799, 3.14159), (0.0, 3.66519), (0.0, 3.14159),
                    (-1.69297, 1.5708), (-1.5708, 1.5708), (-2.0944, 2.0944))
# YAM gripper 系(+Z 前, +X 右, +Y 下) -> 我们的工具系(+X 前, +Y 开合, +Z 上): 列 = 我们的 x,y,z 在 YAM 系的坐标
YAM_TO_TOOL = np.array([[0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [1.0, 0.0, 0.0]])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axang(a, t):
    a = np.asarray(a, float) / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def yam_fk_R(q, tool_offset=0.0, tool_roll_deg=0.0):
    """6 关节角(rad) -> (pos(3), R(3x3)) : base 系下 ARP 位姿, 工具系按我们的约定(+X 前, +Z 上)。"""
    T = np.eye(4)
    for (xyz, rpy, axis), qi in zip(YAM_CHAIN, q):
        A = np.eye(4); A[:3, :3] = _rpy(*rpy); A[:3, 3] = xyz
        B = np.eye(4); B[:3, :3] = _axang(axis, qi)
        T = T @ A @ B
    R = T[:3, :3] @ YAM_TO_TOOL @ _axang((1.0, 0.0, 0.0), math.radians(tool_roll_deg))
    p = T[:3, 3] + R[:, 0] * tool_offset          # 沿接近方向前移到 ARP
    return p, R


def yam_fk(q, tool_offset=0.0, tool_roll_deg=0.0):
    """同 yam_fk_R, 但姿态给四元数 xyzw(采集存盘用)。"""
    p, R = yam_fk_R(q, tool_offset, tool_roll_deg)
    return p, rot_to_quat_xyzw(R)


def _rotvec(R):
    """旋转矩阵 -> 旋转向量(轴*角)。"""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2.0 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def yam_ik(p_t, R_t, q0, tool_offset=0.0, tool_roll_deg=0.0, iters=60, damping=0.05,
           tol_pos=3e-4, tol_rot_deg=0.2, max_step_rad=0.3):
    """阻尼最小二乘数值 IK(有限差分雅可比), 从 q0 出发求到达 (p_t, R_t) 的关节角。
    返回 (q, 位置残差 m, 姿态残差 deg, 是否收敛)。每拍目标离上一拍很近, 从上一拍的解起步 2-5 次迭代即收敛。"""
    q = np.asarray(q0, float).copy(); lo = np.array([l for l, _ in YAM_JOINT_LIMITS]); hi = np.array([h for _, h in YAM_JOINT_LIMITS])
    tol_rot = math.radians(tol_rot_deg); eps = 1e-6
    for _ in range(iters):
        p, R = yam_fk_R(q, tool_offset, tool_roll_deg)
        e = np.concatenate([p_t - p, _rotvec(R_t @ R.T)])
        if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
            return q, float(np.linalg.norm(e[:3])), math.degrees(np.linalg.norm(e[3:])), True
        J = np.zeros((6, 6))
        for j in range(6):
            dq = np.zeros(6); dq[j] = eps
            p2, R2 = yam_fk_R(q + dq, tool_offset, tool_roll_deg)
            J[:3, j] = (p2 - p) / eps; J[3:, j] = _rotvec(R2 @ R.T) / eps
        step = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), e)
        n = np.linalg.norm(step)
        if n > max_step_rad:
            step *= max_step_rad / n
        q = np.clip(q + step, lo, hi)
    p, R = yam_fk_R(q, tool_offset, tool_roll_deg); e = np.concatenate([p_t - p, _rotvec(R_t @ R.T)])
    return q, float(np.linalg.norm(e[:3])), math.degrees(np.linalg.norm(e[3:])), False


def rot_to_quat_xyzw(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([x, y, z, w], np.float32)


def fk_check():
    p, q = yam_fk(np.zeros(6))
    print("home 位姿 ARP(未加 tool_offset):", np.round(p, 6), " 官方 M 位置 (0.110598, 0, 0.173501)")
    R = np.eye(3)  # 校验工具系: home 时 YAM gripper +Z = base +X -> 我们的 +X(接近) 应 = base +X
    pq = yam_fk(np.zeros(6))[1]; x, y, z, w = pq
    Rt = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    print("home 时工具系各轴在 base 系: +X(接近)=", np.round(Rt[:, 0], 3), " +Y(开合)=", np.round(Rt[:, 1], 3), " +Z(上)=", np.round(Rt[:, 2], 3))
    ok = np.allclose(p, [0.110598, 0, 0.173501], atol=1e-4) and np.allclose(Rt[:, 0], [1, 0, 0], atol=1e-3) and np.allclose(Rt[:, 2], [0, 0, 1], atol=1e-3)
    print("FK 自检:", "✓ 通过" if ok else "✗ 失败")
    return ok


# ---------------------------------------------------------------- 硬件接口 ----
class YamFollower:
    """i2rt YAM: get_joint_pos() = 6 关节 + 夹爪(命令空间); command_joint_pos(7)。"""

    def __init__(self, channel, gripper_type, grip_closed, grip_open):
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType
        self.robot = get_yam_robot(channel=channel, gripper_type=getattr(GripperType, gripper_type.upper()),
                                   zero_gravity_mode=False)
        self.gc, self.go = float(grip_closed), float(grip_open)
        n = int(self.robot.num_dofs())
        if n != 7:
            raise RuntimeError(f"YAM 报 {n} 个自由度, 期望 7(6 关节 + 夹爪); 检查 gripper_type")

    def read(self):
        q = np.asarray(self.robot.get_joint_pos(), float)
        return q[:6], float(q[6])

    def grip01(self, raw):           # 原始读数 -> 0=闭 1=开
        return float(np.clip((raw - self.gc) / ((self.go - self.gc) or 1.0), 0.0, 1.0))

    def grip_raw(self, g01):         # 0=闭 1=开 -> 原始命令
        return self.gc + float(np.clip(g01, 0.0, 1.0)) * (self.go - self.gc)

    def command(self, q6, g01):
        self.robot.command_joint_pos(np.concatenate([np.asarray(q6, float), [self.grip_raw(g01)]]))

    def telemetry(self):
        o = self.robot.get_observations()
        return (np.asarray(o.get("joint_pos", np.full(6, np.nan)), float)[:6],
                np.asarray(o.get("joint_vel", np.full(6, np.nan)), float)[:6],
                np.asarray(o.get("joint_eff", np.full(6, np.nan)), float)[:6],
                float(np.asarray(o.get("gripper_pos", [np.nan])).reshape(-1)[0]),
                float(np.asarray(o.get("gripper_eff", [np.nan])).reshape(-1)[0]))

    def move_to(self, q6, g01, seconds=2.0):
        self.robot.move_joints(np.concatenate([np.asarray(q6, float), [self.grip_raw(g01)]]), time_interval_s=seconds)

    def close(self):
        try:
            self.robot.close()
        except Exception:
            pass


class GelloLeader:
    """GELLO(Dynamixel) 经 gello_software: get_joint_state() = 6 关节(rad, 已扣 offset/sign) + 夹爪(0=开 1=闭)。"""

    def __init__(self, port, joint_ids, joint_offsets, joint_signs, gripper_config):
        from gello.robots.dynamixel import DynamixelRobot
        self.robot = DynamixelRobot(joint_ids=list(joint_ids), joint_offsets=list(joint_offsets),
                                    joint_signs=list(joint_signs), real=True, port=port,
                                    gripper_config=tuple(gripper_config) if gripper_config else None)

    def read(self):
        s = np.asarray(self.robot.get_joint_state(), float)
        g01 = 1.0 - float(s[6]) if len(s) > 6 else 1.0      # gello: 0=开 1=闭 -> 我们: 0=闭 1=开
        return s[:6], g01


class YamLeader:
    """第二台 YAM(带夹爪)或示教手柄, 零重力模式手拖。夹爪量程参数同 follower 的约定。"""

    def __init__(self, channel, gripper_type, grip_closed, grip_open):
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType
        self.robot = get_yam_robot(channel=channel, gripper_type=getattr(GripperType, gripper_type.upper()),
                                   zero_gravity_mode=True)
        self.gc, self.go = float(grip_closed), float(grip_open)

    def read(self):
        q = np.asarray(self.robot.get_joint_pos(), float)
        g01 = float(np.clip((q[6] - self.gc) / ((self.go - self.gc) or 1.0), 0, 1)) if len(q) > 6 else 1.0
        return q[:6], g01


class FakeArm:
    """无硬件自检: 关节走一条平滑曲线, leader 比 follower 超前 2 帧。"""

    def __init__(self, lead=False):
        self.t0 = time.monotonic(); self.lead = lead; self.gc, self.go = 0.0, 1.0

    def _q(self, t):
        return np.array([0.3 * np.sin(t), 1.2 + 0.2 * np.sin(0.7 * t), 1.0 + 0.2 * np.cos(0.5 * t), 0.1 * np.sin(t), 0.2 * np.cos(t), 0.0])

    def read(self):
        t = time.monotonic() - self.t0 + (0.2 if self.lead else 0.0)
        return self._q(t), 0.5 + 0.5 * np.sin(t)

    def grip01(self, raw): return raw
    def command(self, q6, g01): pass
    def telemetry(self): q, g = self.read(); return q, np.zeros(6), np.zeros(6), g, 0.0
    def move_to(self, q6, g01, seconds=2.0): pass
    def close(self): pass


class FakeCams:
    def get_frames(self):
        z = np.zeros((480, 640, 3), np.uint8); return z, z

    def stop(self): pass


# ---------------------------------------------------------------- 采集 ----
def collect_episode(leader, follower, cams, a, episode_num):
    from collect_data import bgr_to_rgb, show_camera_view
    q_l, g_l = leader.read()
    q_f, _ = follower.read()
    if np.abs(q_l - q_f).max() > 0.1:
        logger.info("对齐 follower 到 leader 关节角(会移动)...")
        follower.move_to(q_l, g_l, seconds=2.0); time.sleep(0.3)
    if a.show:
        h, w = cams.get_frames(); show_camera_view(h, w, episode_num, 0, RECORD_FREQ)
    if not a.fake:
        logger.info(">>> 按 Enter 开始录制(移动主臂开始演示)..."); input()

    frames = []; telem = {k: [] for k in ("t", "frame_idx", "cmd_q", "cmd_grip", "q", "dq", "tau", "grip_q", "grip_tau")}
    guard = JointGuard("yam", margin_deg=a.joint_margin, rate_hz=RECORD_FREQ, limits=YAM_JOINT_LIMITS, log=logger.warning) if a.joint_margin > 0 else None
    dt = 1.0 / CONTROL_FREQ; rec_every = CONTROL_FREQ // RECORD_FREQ; step = 0; t0 = time.monotonic()
    logger.info("录制中... 按 Enter 结束并保存")
    try:
        while True:
            ts = time.monotonic()
            q_l, g_l = leader.read()
            follower.command(q_l, g_l)
            q, dq, tau, gq, gt = follower.telemetry()
            telem["t"].append(time.monotonic() - t0); telem["frame_idx"].append(len(frames))
            telem["cmd_q"].append(list(q_l)); telem["cmd_grip"].append(g_l)
            telem["q"].append(list(q)); telem["dq"].append(list(dq)); telem["tau"].append(list(tau))
            telem["grip_q"].append(gq); telem["grip_tau"].append(gt)
            if step % rec_every == 0:
                q_f, g_raw = follower.read(); g_f = follower.grip01(g_raw)
                if guard is not None:
                    guard.update(q_l, q_f)
                p_f, quat_f = yam_fk(q_f, a.tool_offset, a.tool_roll_deg)
                p_l, quat_l = yam_fk(q_l, a.tool_offset, a.tool_roll_deg)
                head, wrist = cams.get_frames()
                frames.append({
                    "state": np.array(list(q_f) + [g_f], np.float32),
                    "action": np.array(list(q_l) + [g_l], np.float32),
                    "state_eef": np.array(list(p_f) + list(quat_f) + [g_f], np.float32),
                    "action_eef": np.array(list(p_l) + list(quat_l) + [g_l], np.float32),
                    "head_rgb": bgr_to_rgb(head), "wrist_rgb": bgr_to_rgb(wrist),
                })
                if a.show:
                    show_camera_view(head, wrist, episode_num, len(frames), RECORD_FREQ)
                if len(frames) % RECORD_FREQ == 0:
                    logger.info(f"  已录制 {len(frames)} 帧 ({len(frames)/RECORD_FREQ:.1f}s)  ARP={np.round(p_f, 3).tolist()} grip={g_f:.2f}")
            step += 1
            if a.fake:
                if len(frames) >= a.fake_frames:
                    break
            elif select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline(); break
            el = time.monotonic() - ts
            if el < dt:
                time.sleep(dt - el)
    except KeyboardInterrupt:
        logger.info("录制被中断, 丢弃本条"); return None
    logger.info(f"录制结束: 共 {len(frames)} 帧 ({len(frames)/RECORD_FREQ:.1f}s)")
    if guard is not None:
        logger.info(guard.summary())
    if len(frames) < 5:
        logger.warning("帧数太少, 丢弃"); return None
    if not a.fake and input("保存本条? [Enter=保存, d=丢弃]: ").strip().lower() == "d":
        logger.info("已丢弃"); return None
    return frames, telem


def load_gello_yaml(path):
    """从 generate_yam_config.py 的 yaml 里找 port/joint_ids/joint_offsets/joint_signs/gripper_config(任意嵌套)。"""
    import yaml
    d = yaml.safe_load(open(path)); found = {}
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("port", "joint_ids", "joint_offsets", "joint_signs", "gripper_config") and k not in found:
                    found[k] = v
                walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(d); return found


def main():
    ap = argparse.ArgumentParser(description="YAM + GELLO 遥操作采集(与 collect_data.py 同格式)")
    ap.add_argument("--task", default="sweep the objects into the dustpan"); ap.add_argument("--repo-id", default="yam_get_sweep")
    ap.add_argument("--output-dir", "-o", default=None); ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--follow-can", default="can0"); ap.add_argument("--follow-gripper-type", default="linear_4310", help="i2rt GripperType 名(小写)")
    ap.add_argument("--grip-closed", type=float, default=0.0, help="follower 夹爪全闭时 get_joint_pos()[6] 原始读数")
    ap.add_argument("--grip-open", type=float, default=1.0, help="follower 夹爪全开时的原始读数")
    ap.add_argument("--leader", choices=["gello", "yam"], default="gello")
    ap.add_argument("--gello-config", default=None, help="gello_software/configs/yam_auto_generated.yaml")
    ap.add_argument("--gello-port", default=None); ap.add_argument("--gello-ids", type=int, nargs="*", default=None)
    ap.add_argument("--gello-offsets", type=float, nargs="*", default=None); ap.add_argument("--gello-signs", type=float, nargs="*", default=None)
    ap.add_argument("--gello-gripper", type=float, nargs=3, default=None, metavar=("ID", "OPEN_DEG", "CLOSED_DEG"))
    ap.add_argument("--leader-can", default="can1"); ap.add_argument("--leader-gripper-type", default="linear_4310")
    ap.add_argument("--tool-offset", type=float, default=0.0, help="YAM gripper 系原点 -> ARP 沿接近方向的距离(m) = GET 指尖距 − 0.017 (必填, 卡尺量)")
    ap.add_argument("--tool-roll-deg", type=float, default=0.0, help="GET 开合轴相对 YAM gripper +X 绕接近方向的转角")
    ap.add_argument("--joint-margin", type=float, default=15.0)
    ap.add_argument("--head-serial", default=HEAD_CAMERA_SERIAL); ap.add_argument("--wrist-serial", default=WRIST_CAMERA_SERIAL)
    ap.add_argument("--dry-run", action="store_true", help="连硬件, 只打印 leader/follower 读数与 FK, 不驱动不录")
    ap.add_argument("--fk-check", action="store_true"); ap.add_argument("--fake", action="store_true", help="假臂假相机走通写盘")
    ap.add_argument("--fake-frames", type=int, default=30); ap.add_argument("--fake-episodes", type=int, default=2)
    a = ap.parse_args(); a.show = not a.no_display and not a.fake

    if a.fk_check:
        sys.exit(0 if fk_check() else 1)
    if a.tool_offset == 0.0 and not a.fake:
        logger.warning("⚠ --tool-offset 为 0: ARP 落在 YAM 法兰上, 与 AIRBOT 数据的参考点不一致; 装好 GET 后量一次再采")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import collect_data as cd
    if a.fake:
        follower, leader, cams = FakeArm(), FakeArm(lead=True), FakeCams()
    else:
        cams = cd.DualCamera(head_serial=a.head_serial, wrist_serial=a.wrist_serial)   # 相机先起(同 collect_data 理由)
        follower = YamFollower(a.follow_can, a.follow_gripper_type, a.grip_closed, a.grip_open)
        if a.leader == "gello":
            cfg = load_gello_yaml(a.gello_config) if a.gello_config else {}
            port = a.gello_port or cfg.get("port"); ids = a.gello_ids or cfg.get("joint_ids")
            offs = a.gello_offsets or cfg.get("joint_offsets"); signs = a.gello_signs or cfg.get("joint_signs")
            grip = a.gello_gripper or cfg.get("gripper_config")
            if not (port and ids and offs and signs):
                sys.exit(f"GELLO 参数不全(port/ids/offsets/signs): 从 yaml 读到 {cfg}; 用 --gello-* 补")
            logger.info(f"GELLO: port={port} ids={ids} offsets={np.round(offs, 3).tolist()} signs={signs} gripper={grip}")
            leader = GelloLeader(port, ids, offs, signs, grip)
        else:
            leader = YamLeader(a.leader_can, a.leader_gripper_type, a.grip_closed, a.grip_open)
        q, g = follower.read(); logger.info(f"follower 关节(deg)={np.round(np.degrees(q)).tolist()} 夹爪原始读数={g:.4f} -> 归一化 {follower.grip01(g):.2f}")
        ql, gl = leader.read(); logger.info(f"leader   关节(deg)={np.round(np.degrees(ql)).tolist()} 夹爪(0闭1开)={gl:.2f}")
        p, _ = yam_fk(q, a.tool_offset, a.tool_roll_deg); logger.info(f"follower ARP(base 系)={np.round(p, 3).tolist()}")

    if a.dry_run:
        logger.info("dry-run: 每 0.5s 打印一次, Ctrl+C 退出")
        try:
            while True:
                q, g = follower.read(); ql, gl = leader.read(); p, _ = yam_fk(q, a.tool_offset, a.tool_roll_deg)
                print(f"follower deg={np.round(np.degrees(q)).tolist()} grip={follower.grip01(g):.2f} ARP={np.round(p,3).tolist()} | leader deg={np.round(np.degrees(ql)).tolist()} grip={gl:.2f} | Δ关节 max={np.degrees(np.abs(q-ql).max()):.1f}°", flush=True)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        follower.close(); cams.stop(); return

    if a.show:
        import cv2; cv2.namedWindow("Data Collection", cv2.WINDOW_AUTOSIZE)
    dataset = None; save_dir = None; num_saved = 0; attempts = 0
    logger.info("=" * 60); logger.info(f"[YAM] 任务: {a.task}"); logger.info("=" * 60)
    try:
        while True:
            attempts += 1
            if a.fake and (num_saved >= a.fake_episodes or attempts > 2 * a.fake_episodes):
                break
            if not a.fake:
                logger.info(f"\n--- Episode {num_saved + 1} --- 按 Enter 录制, q+Enter 结束")
                if input().strip().lower() == "q":
                    break
            ep = collect_episode(leader, follower, cams, a, num_saved + 1)
            if ep is not None:
                frames, telem = ep
                if dataset is None:
                    dataset, save_dir = cd.create_lerobot_dataset(a.repo_id, RECORD_FREQ, a.output_dir)
                cd.write_episode(dataset, frames, a.task)
                cd.save_telemetry(save_dir, num_saved, telem)
                num_saved += 1; logger.info(f"✓ 已保存 episode {num_saved} -> {save_dir}")
    except KeyboardInterrupt:
        logger.info("中断")
    finally:
        follower.close(); cams.stop()
        if a.show:
            import cv2; cv2.destroyAllWindows()
        logger.info(f"共保存 {num_saved} 条 -> {save_dir}")


if __name__ == "__main__":
    main()
