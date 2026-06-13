#!/usr/bin/env python3
"""
UMI 轨迹真机回放验证：umi_to_lerobot.py --poses 的 npz -> AIRBOT Play 开环复现。

这是坐标转换正确性的金标准验证：机械臂把人用 UMI 演示的动作"演"一遍，
形状/升降方向/转腕方向全部一致才算转换链通过。

W' -> 机械臂 base 系的映射（回放时补上的"最后一公里"）:
    T_base_tip(t) = Trans(start_pos) · Rz(start_yaw) · T_W'(t) · R_align
  - start_pos / start_yaw: 自由参数，把轨迹摆进工作空间（姿态不可选，
    pitch/roll 由数据决定 —— 重力锚定）；
  - R_align: UMI body 系 -> AIRBOT 工具系的固定轴向对齐旋转（--align-rpy）；
  - 机械臂 TCP: get_end_pose 参考点 -> 指尖的偏移（--robot-tip-offset），
    指令位置 = p_tip - R·offset。

四种模式（按顺序使用）:
  1) --probe       连真机只读不动: 打印 get_end_pose/关节角/产品信息，确认
                   姿态格式与零位约定。本套实测: 零位四元数≈单位阵(xyzw)，
                   base=工具系 x前/y左/z上，夹爪零位水平向前正立。
  2) --calib-align 从"UMI 摆成机械臂零位同姿态"的静止 mcap 求 R_align
  3) --dry-run     纯离线: base 系指令序列 + 工作空间/步距检查 + 3D 预览图
  4) (默认)        真机回放: slow 速度 -> 规划到起始位姿 -> 伺服跟踪

用法:
    python replay_check.py --probe --port 50050
    python replay_check.py --calib-align umi_flat_vio.mcap
    python replay_check.py xxx_poses.npz --dry-run --auto-place
    python replay_check.py xxx_poses.npz --auto-place --align-rpy R P Y --speed-scale 0.5
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from umi_to_lerobot import (  # noqa: E402
    rot6d_to_rot, quat_to_rot, euler_zyx_deg, read_pose_stream,
    POSE_TOPIC, GRIPPER_TOPIC,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "airbot" / "play_config.json"


def load_config(config_path: Path) -> dict:
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_workspace(config_path: Path):
    """从 play_config.json 读工作空间边界，没有则返回 None。"""
    ws = load_config(config_path).get("workspace")
    if ws and "bounds_min" in ws and "bounds_max" in ws:
        return {"lo": np.array(ws["bounds_min"], float),
                "hi": np.array(ws["bounds_max"], float)}
    return None


def load_home_joint(config_path: Path):
    """读弯曲的 ready 关节构型(避开零位奇异)，没有则 None。"""
    return load_config(config_path).get("arm", {}).get("home_joint")


# ---------------------------------------------------------------------------
# 旋转工具
# ---------------------------------------------------------------------------
def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 四元数 (x,y,z,w)。"""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            x, w = 0.25 * s, (R[2, 1] - R[1, 2]) / s
            y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            y, w = 0.25 * s, (R[0, 2] - R[2, 0]) / s
            x, z = (R[0, 1] + R[1, 0]) / s, (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            z, w = 0.25 * s, (R[1, 0] - R[0, 1]) / s
            x, y = (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s
    return np.array([x, y, z, w])


def rpy_to_rot(roll, pitch, yaw):
    """zyx 顺序欧拉角（度）-> 旋转矩阵。"""
    r, p, y = map(math.radians, (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# 轨迹准备
# ---------------------------------------------------------------------------
def load_and_resample(npz_path: Path, rate: float, speed_scale: float):
    """npz -> 均匀时间步的 (pos(N,3), R(N,3,3), grip(N))，W' 系。"""
    d = np.load(npz_path)
    ts = (d["ts"] - d["ts"][0]) / 1e9 / speed_scale     # 减速 = 拉长时间轴
    pos, rot6d, grip = d["pos"], d["rot6d"], d["gripper_m"]

    t_new = np.arange(0.0, ts[-1], 1.0 / rate)
    pos_r = np.stack([np.interp(t_new, ts, pos[:, i]) for i in range(3)], axis=1)
    d6_r = np.stack([np.interp(t_new, ts, rot6d[:, i]) for i in range(6)], axis=1)
    R_r = np.stack([rot6d_to_rot(d6) for d6 in d6_r])   # 插值后重新正交化
    grip_r = np.interp(t_new, ts, np.nan_to_num(grip, nan=np.nanmax(grip)))
    return t_new, pos_r, R_r, grip_r


def to_base_frame(pos_w, R_w, start_pos, start_yaw_deg, R_align,
                  robot_tip_offset):
    """W' 轨迹 -> base 系指令序列 (cmd_pos, cmd_R)。

    T_base_tip = Trans(start_pos)·Rz(yaw)·T_W'·R_align
    指令参考点 = 指尖 - R_tool·robot_tip_offset
    """
    yaw = math.radians(start_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    p0 = np.asarray(start_pos, float)
    off = np.asarray(robot_tip_offset, float)

    cmd_pos, cmd_R = [], []
    for p, R in zip(pos_w, R_w):
        R_tool = Rz @ R @ R_align
        p_tip = p0 + Rz @ p
        cmd_pos.append(p_tip - R_tool @ off)
        cmd_R.append(R_tool)
    return np.array(cmd_pos), np.array(cmd_R)


def base_to_wprime(p_base, R_base, start_pos, start_yaw_deg, R_align, tip_offset):
    """to_base_frame 的逆：base 系实测位姿 -> UMI W' 系。
    正向: p_base = start_pos + Rz·p_w' - R_base·off;  R_base = Rz·R_w'·R_align
    """
    yaw = math.radians(start_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    RzT = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])      # Rz(yaw)ᵀ
    p0 = np.asarray(start_pos, float)
    off = np.asarray(tip_offset, float)
    p_tip = np.asarray(p_base, float) + R_base @ off          # 还原指尖位置
    p_w = RzT @ (p_tip - p0)
    R_w = RzT @ R_base @ R_align.T
    return p_w, R_w


def auto_place(pos_w, start_yaw_deg, ws, z_margin=0.05):
    """把轨迹包围盒摆进工作空间：xy 居中，z 让最低点离下界 z_margin。
    z_margin 调大 = 整条轨迹抬高（起点构型更舒展、远离桌面）。"""
    yaw = math.radians(start_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    pw = pos_w @ Rz.T                       # 先转 yaw 再算包围盒
    lo, hi = pw.min(0), pw.max(0)
    size = hi - lo
    ws_size = ws["hi"] - ws["lo"]
    for i, ax in enumerate("xyz"):
        if size[i] > ws_size[i]:
            print(f"  ⚠ 轨迹 {ax} 向跨度 {size[i]:.3f}m 超过工作空间 {ws_size[i]:.3f}m，放不下")
    start = np.zeros(3)
    ctr = (ws["lo"] + ws["hi"]) / 2
    start[:2] = ctr[:2] - (lo[:2] + hi[:2]) / 2          # xy 居中
    start[2] = ws["lo"][2] + z_margin - lo[2]            # 最低点贴着下界+margin
    return start


def _quat_angle_deg(q1, q2) -> float:
    """两个四元数(xyzw)之间的夹角，度。"""
    d = abs(float(np.dot(np.asarray(q1), np.asarray(q2))))
    return math.degrees(2 * math.acos(min(1.0, d)))


def wait_until_stopped(robot, final_pos=None, final_quat=None, record=False,
                       timeout=120.0, window=0.7, pos_eps=0.0015, rot_eps=0.8,
                       goal_pos_tol=0.01, goal_rot_tol=5.0, backup_still=3.0,
                       rate=30.0):
    """轮询 get_end_pose 判停，靠"末端位姿不再变化"而非关节速度。

    关键：判停需同时满足 (a)已停稳 且 (b)已到达轨迹终点附近 —— 这样起步慢、
    规划器分段间的短暂慢速(离终点还远)都不会误判为结束。
    兜底：若规划器终点和目标对不齐，停稳持续 backup_still 秒也退出。
    moving 标志确保规划阶段(尚未运动)的静止不被当成结束。

    final_pos/final_quat: 轨迹最后一帧的 base 系目标(用于"到终点"判据)。
    返回采集到的 [(pos, quat_xyzw), ...]。
    """
    from collections import deque
    poses, win = [], deque()        # win: (t, pos(np), quat(np))
    dt = 1.0 / rate
    t_start = time.monotonic()
    moving = False
    still_since = None
    fp = np.asarray(final_pos, float) if final_pos is not None else None
    while time.monotonic() - t_start < timeout:
        ts = time.monotonic()
        p, q = robot.get_end_pose()
        p, q = np.asarray(p, float), np.asarray(q, float)
        if record:
            poses.append((p.tolist(), q.tolist()))
        win.append((ts, p, q))
        while win and ts - win[0][0] > window:   # 只保留最近 window 秒
            win.popleft()
        if ts - win[0][0] >= window * 0.9 and len(win) >= 3:
            ps = np.array([w[1] for w in win])
            pos_span = float(np.linalg.norm(ps.max(0) - ps.min(0)))
            rot_span = max(_quat_angle_deg(win[0][2], w[2]) for w in win)
            still = pos_span < pos_eps and rot_span < rot_eps
            if not still:
                moving = True
                still_since = None
            else:
                if still_since is None:
                    still_since = ts
                if moving:                       # 必须先动过
                    at_goal = True
                    if fp is not None:
                        at_goal = (np.linalg.norm(p - fp) < goal_pos_tol and
                                   _quat_angle_deg(q, final_quat) < goal_rot_tol)
                    # 到终点附近且停稳 → 结束；或停稳很久(兜底)
                    if at_goal or ts - still_since > backup_still:
                        break
        time.sleep(max(0.0, dt - (time.monotonic() - ts)))
    return poses


def resample_by_arclength(pos, extra, n=200):
    """按位置弧长归一化重采样，去掉时序差异。
    pos:(M,3), extra:(M,k) 同步量(如 rpy)。返回 (grid, pos_s, extra_s) 或 None。"""
    pos = np.asarray(pos)
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    d = np.concatenate([[0], np.cumsum(seg)])
    if d[-1] < 1e-6:
        return None
    s = d / d[-1]
    grid = np.linspace(0, 1, n)
    pos_s = np.stack([np.interp(grid, s, pos[:, i]) for i in range(pos.shape[1])], axis=1)
    extra = np.asarray(extra)
    extra_s = np.stack([np.interp(grid, s, extra[:, i]) for i in range(extra.shape[1])], axis=1)
    return grid, pos_s, extra_s


def plot_plan_compare(ref_pos, ref_rpy, act_pos, act_rpy, out_png,
                      label_ref="UMI npz", label_act="robot actual",
                      title="plan mode: UMI npz vs robot actual (W' frame, arclength-normalized)"):
    """按路径弧长归一化叠画两条轨迹(时序不同, 按路径进度对齐成同始同终)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rc = resample_by_arclength(ref_pos, ref_rpy)
    ra = resample_by_arclength(act_pos, act_rpy)
    if rc is None or ra is None:
        print("  (path too short, skip arclength compare)")
        return
    g, cps, crs = rc
    _, aps, ars = ra
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for i, lbl in enumerate("xyz"):
        ax = axes[0, i]
        ax.plot(g, cps[:, i], label=label_ref, lw=1.5)
        ax.plot(g, aps[:, i], "--", label=label_act, lw=1.2)
        err = np.abs(cps[:, i] - aps[:, i])
        ax.set_title(f"pos {lbl}  err mean={err.mean()*1000:.1f}mm max={err.max()*1000:.1f}mm")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlabel("path progress"); ax.set_ylabel("m")
    for i, lbl in enumerate(["roll", "pitch", "yaw"]):
        ax = axes[1, i]
        ax.plot(g, crs[:, i], label=label_ref, lw=1.5)
        ax.plot(g, ars[:, i], "--", label=label_act, lw=1.2)
        err = np.abs(crs[:, i] - ars[:, i])
        ax.set_title(f"{lbl}  err mean={err.mean():.1f}deg max={err.max():.1f}deg")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlabel("path progress"); ax.set_ylabel("deg")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def plot_fk_compare(t_cmd, cmd_pos, cmd_rpy, t_act, act_pos, act_rpy, out_png):
    """指令位姿 vs 机械臂实测 FK，按各自时间轴叠画(指令与实测采样率可不同)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmd_pos, cmd_rpy = np.array(cmd_pos), np.array(cmd_rpy)
    act_pos, act_rpy = np.array(act_pos), np.array(act_rpy)
    t_cmd, t_act = np.asarray(t_cmd), np.asarray(t_act)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for i, lbl in enumerate("xyz"):
        ax = axes[0, i]
        ax.plot(t_cmd, cmd_pos[:, i], label="cmd", lw=1.5)
        ax.plot(t_act, act_pos[:, i], "--", label="actual FK", lw=1.2)
        act_on_cmd = np.interp(t_cmd, t_act, act_pos[:, i])   # 对齐到指令时间算误差
        err = np.abs(cmd_pos[:, i] - act_on_cmd)
        ax.set_title(f"pos {lbl}  err mean={err.mean()*1000:.1f}mm max={err.max()*1000:.1f}mm")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlabel("t(s)"); ax.set_ylabel("m")
    for i, lbl in enumerate(["roll", "pitch", "yaw"]):
        ax = axes[1, i]
        ax.plot(t_cmd, cmd_rpy[:, i], label="cmd", lw=1.5)
        ax.plot(t_act, act_rpy[:, i], "--", label="actual FK", lw=1.2)
        act_on_cmd = np.interp(t_cmd, t_act, act_rpy[:, i])
        err = np.abs(cmd_rpy[:, i] - act_on_cmd)
        ax.set_title(f"{lbl}  err mean={err.mean():.1f}deg max={err.max():.1f}deg")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlabel("t(s)"); ax.set_ylabel("deg")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def safety_report(t, cmd_pos, rate, min_z, ws=None):
    span_lo, span_hi = cmd_pos.min(0), cmd_pos.max(0)
    step = np.linalg.norm(np.diff(cmd_pos, axis=0), axis=1)
    print(f"  指令帧数: {len(cmd_pos)}  时长 {t[-1]:.1f}s @ {rate}Hz")
    print(f"  base 系包围盒: x[{span_lo[0]:+.3f},{span_hi[0]:+.3f}] "
          f"y[{span_lo[1]:+.3f},{span_hi[1]:+.3f}] z[{span_lo[2]:+.3f},{span_hi[2]:+.3f}] m")
    print(f"  相邻步距: mean={step.mean()*1000:.1f}mm max={step.max()*1000:.1f}mm "
          f"(峰值速度≈{step.max()*rate:.2f}m/s)")
    ok = True
    if ws is not None:
        out_lo = span_lo < ws["lo"]
        out_hi = span_hi > ws["hi"]
        if out_lo.any() or out_hi.any():
            bad = [ax for i, ax in enumerate("xyz") if out_lo[i] or out_hi[i]]
            print(f"  ⚠ 轨迹超出工作空间 ({','.join(bad)} 向)，"
                  f"边界 {ws['lo'].tolist()} ~ {ws['hi'].tolist()}；"
                  f"调 --start-pos/--start-yaw 或用 --auto-place")
            ok = False
        else:
            print(f"  ✓ 全程在工作空间内")
    if span_lo[2] < min_z:
        print(f"  ⚠ 轨迹最低点 z={span_lo[2]:.3f} < min-z {min_z}，会触发保护/撞桌面")
        ok = False
    if step.max() * rate > 0.5:
        print(f"  ⚠ 峰值速度 >0.5m/s，建议加 --speed-scale 调慢")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# 模式实现
# ---------------------------------------------------------------------------
def calib_align(umi_mcap: Path, robot_quat_xyzw):
    """从"UMI 摆成机械臂零位同姿态"的静止 mcap 求 R_align。

    R_align = R_umiᵀ · R_robot （同一物理姿态下两边本体系之差，与姿态无关）。
    机械臂零位 ≈ 单位阵时 R_align ≈ R_umiᵀ。yaw 分量会被回放的 --start-yaw
    吸收，真正需要准的是 roll/pitch（由重力决定）。
    """
    ts, xyz, quat, _gts, _gv = read_pose_stream(umi_mcap, POSE_TOPIC, GRIPPER_TOPIC)
    if len(quat) < 5:
        sys.exit(f"位姿太少 ({len(quat)} 帧)")
    Rs = np.stack([quat_to_rot(*q) for q in quat])
    # 平均旋转（SVD 正交化），并报告静止程度
    Rbar = Rs.mean(0)
    U, _s, Vt = np.linalg.svd(Rbar)
    R_umi = U @ Vt
    if np.linalg.det(R_umi) < 0:
        U[:, -1] *= -1
        R_umi = U @ Vt
    ang_spread = max(
        math.degrees(math.acos(np.clip((np.trace(R_umi.T @ R) - 1) / 2, -1, 1)))
        for R in Rs)
    print(f"读取 {len(quat)} 帧；姿态抖动(相对均值)max={ang_spread:.2f}° "
          f"{'(够稳)' if ang_spread < 3 else '⚠ 偏大，尽量录更稳'}")
    r_u, p_u, y_u = euler_zyx_deg(R_umi)
    print(f"UMI 本体系姿态(world): roll={r_u:+.2f}° pitch={p_u:+.2f}° yaw={y_u:+.2f}° "
          f"(pitch 反映相机光轴下倾)")

    R_robot = quat_to_rot(*robot_quat_xyzw)
    R_align = R_umi.T @ R_robot
    roll, pitch, yaw = euler_zyx_deg(R_align)
    print(f"\n===== R_align (zyx 欧拉角, 度) =====")
    print(f"  roll={roll:+.2f}  pitch={pitch:+.2f}  yaw={yaw:+.2f}")
    print(f"\n回放时加参数:\n  --align-rpy {roll:.2f} {pitch:.2f} {yaw:.2f}")
    print("(yaw 不准没关系, 会被 --start-yaw 吸收; 重点看 roll/pitch)")


def test_cart(port: int):
    """诊断 move_to_cart_pose 的四元数约定 + 基本可达性。

    依次规划移动到: 当前位姿(原样回发) -> 当前-Z3cm -> 当前+X3cm -> 当前低头15°。
    "当前位姿"成功 => SET 接受 xyzw(与 get_end_pose 同约定)；失败 => 约定不同/接口问题。
    """
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "airbot"))
    from play_sdk import PlayRealRobot
    robot = PlayRealRobot(port=port, enable_cameras=False)
    try:
        robot.set_speed_profile("slow")
        pos0, q0 = robot.get_end_pose()
        print(f"当前位姿 pos={[round(v,4) for v in pos0]} quat(xyzw)={[round(v,4) for v in q0]}")
        # 当前姿态低头 15°（绕 base y 轴）
        R_down = rpy_to_rot(0, 15, 0) @ quat_to_rot(*q0)
        q_down = rot_to_quat_xyzw(R_down)
        tests = [
            ("① 当前位姿原样回发(验证约定)", list(pos0), list(q0)),
            ("② 当前 -Z 3cm", [pos0[0], pos0[1], pos0[2] - 0.03], list(q0)),
            ("③ 当前 +X 3cm", [pos0[0] + 0.03, pos0[1], pos0[2]], list(q0)),
            ("④ 当前姿态低头15°", list(pos0), q_down.tolist()),
        ]
        for name, p, q in tests:
            ok = robot.set_end_pose(p, q, blocking=True)
            print(f"  {name}: {'✓ 成功' if ok else '✗ 失败'}")
            time.sleep(1.0)
        print("\n判读: ① 成功=约定为 xyzw, 可达性正常; ① 失败=SET 约定可能是 wxyz")
    finally:
        robot.shutdown()


def probe(port: int):
    """连真机只读不动，打印所有约定相关信息。"""
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "airbot"))
    from play_sdk import PlayRealRobot
    robot = PlayRealRobot(port=port, enable_cameras=False)
    try:
        print("\n===== 产品信息 =====")
        print(robot.get_product_info())
        print("\n===== 关节角 =====")
        print(robot.get_joint_q())
        print("\n===== get_end_pose() 原始返回 =====")
        pose = robot.get_end_pose()
        print(pose)
        if pose:
            p, o = pose
            print(f"  position 长度 {len(p)}: {p}")
            print(f"  orientation 长度 {len(o)}: {o}")
            if len(o) == 4:
                which = "xyzw" if abs(o[3]) >= abs(o[0]) else "wxyz"
                print(f"  -> 四元数；若当前接近零位，模长最大的分量是 w："
                      f"看起来像 {which}（请结合当前姿态判断）")
            elif len(o) == 3:
                print("  -> 看起来是 RPY 欧拉角(弧度)")
        print("\n===== 夹爪 =====")
        print(robot.get_gripper_state())
        print("\n请把以上输出发回，用于确定姿态格式/参考点约定。")
    finally:
        robot.shutdown()


def replay(args):
    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.is_file():
        sys.exit(f"npz 不存在: {npz_path}")

    print(f"=== 载入 {npz_path.name} ===")
    t, pos_w, R_w, grip = load_and_resample(npz_path, args.rate, args.speed_scale)
    R_align = rpy_to_rot(*args.align_rpy)

    ws = load_workspace(Path(args.config))
    if ws is None:
        print(f"  (未读到工作空间配置 {args.config}，跳过边界检查)")
    min_z = args.min_z if args.min_z is not None else \
        (float(ws["lo"][2]) + 0.01 if ws else 0.05)

    start_pos = args.start_pos
    if args.auto_place:
        if ws is None:
            sys.exit("--auto-place 需要 play_config.json 里有 workspace 配置")
        start_pos = auto_place(pos_w, args.start_yaw, ws, args.z_margin).tolist()
        print(f"  自动摆放起点: {[round(v,4) for v in start_pos]} "
              f"(yaw={args.start_yaw}° z-margin={args.z_margin})")

    cmd_pos, cmd_R = to_base_frame(pos_w, R_w, start_pos, args.start_yaw,
                                   R_align, args.robot_tip_offset)
    ok = safety_report(t, cmd_pos, args.rate, min_z, ws)
    quats = np.stack([rot_to_quat_xyzw(R) for R in cmd_R])
    cmd_rpy = np.array([euler_zyx_deg(R) for R in cmd_R])

    # 夹爪: 米 -> [0,1]（粗略线性，正式归一化在阶段3做）
    g01 = np.clip(grip / args.gripper_max, 0.0, 1.0)

    if args.dry_run:
        out_png = npz_path.parent / f"{npz_path.stem}_replay_preview.png"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121, projection="3d")
        ax.plot(*cmd_pos.T, lw=1.2)
        ax.scatter(*cmd_pos[0], color="green", s=50, label="start")
        ax.scatter(*cmd_pos[-1], color="red", s=50, label="end")
        ax.set_title("command trajectory (robot base frame)")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.legend()
        ax2 = fig.add_subplot(122)
        for i, lbl in enumerate("xyz"):
            ax2.plot(t, cmd_pos[:, i], label=lbl)
        ax2.plot(t, g01 * 0.1, "--", color="gray", label="gripper(0-1)x0.1")
        ax2.axhline(min_z, color="red", lw=0.8, ls=":", label="min-z")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        ax2.set_xlabel("t (s)"); ax2.set_ylabel("m")
        fig.tight_layout(); fig.savefig(out_png, dpi=110)
        print(f"  ✓ [DRY RUN] 预览图: {out_png}")
        print(f"  起始指令位姿: pos={cmd_pos[0].round(4).tolist()} "
              f"quat(xyzw)={quats[0].round(4).tolist()}")
        return

    if not ok and not args.force:
        sys.exit("安全检查未通过；修正参数后重试，或加 --force 强行继续")

    # ---- 真机 ----
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "airbot"))
    from play_sdk import PlayRealRobot, RobotMode
    robot = PlayRealRobot(port=args.port, enable_cameras=False)
    try:
        robot.set_speed_profile("slow")
        # 先到弯曲的 ready 构型，避开零位伸直奇异（否则笛卡尔规划到起点常失败）
        home_joint = load_home_joint(Path(args.config))
        if home_joint:
            print(f"先移动到 ready 关节构型(避开零位奇异): {[round(v,3) for v in home_joint]}")
            if not robot.set_joint_positions(home_joint, blocking=True):
                sys.exit("✗ 移动到 ready 构型失败")
            time.sleep(0.3)
        cur = robot.get_end_pose()
        print(f"\n当前末端位姿: {cur}")
        print(f"即将规划移动到起始位姿:\n  pos={cmd_pos[0].round(4).tolist()}\n"
              f"  quat(xyzw)={quats[0].round(4).tolist()}")
        if input("确认移动到起始位姿? (yes/no) > ").strip().lower() != "yes":
            print("取消"); return
        if not robot.set_end_pose(cmd_pos[0].tolist(), quats[0].tolist(), blocking=True):
            sys.exit("✗ 规划到起始位姿失败（可能不可达），换 --start-pos/--start-yaw 重试")
        robot.set_gripper(position=float(g01[0]))
        time.sleep(0.5)

        if args.plan:
            print(f"\n就位。即将用路点规划复现 {len(cmd_pos)} 帧轨迹"
                  f"(每 {max(1, args.plan_step)} 帧取 1 路点)，由下位机规划执行")
        else:
            print(f"\n就位。即将以 {args.servo_hz:.0f}Hz 伺服跟踪 "
                  f"({t[-1]:.1f}s)，Ctrl+C 随时中断")
        if input("开始回放? (yes/no) > ").strip().lower() != "yes":
            print("取消"); return

        # ---- 路点规划模式：绕开 servo 带宽不足，几何忠实复现 ----
        if args.plan:
            step = max(1, args.plan_step)
            idx = list(range(0, len(cmd_pos), step))
            if idx[-1] != len(cmd_pos) - 1:
                idx.append(len(cmd_pos) - 1)
            waypoints = [[cmd_pos[k].tolist(), quats[k].tolist()] for k in idx]
            print(f"路点规划: {len(waypoints)} 个路点 (每 {step} 帧取 1)，"
                  f"由下位机解 IK+规划执行；夹爪开 (规划模式不同步夹爪)")
            robot.set_gripper(position=1.0)
            # 非阻塞下发 + 轮询: 既能等它真正停稳, 又能(可选)采集实测 FK
            ok2 = robot.set_end_pose_waypoints(waypoints, blocking=False)
            if not ok2:
                print("✗ 路点规划失败(可能某路点不可达)"); return
            actual = wait_until_stopped(robot, final_pos=cmd_pos[-1],
                                        final_quat=quats[-1], record=args.log_fk)
            print("✓ 路点规划执行完成(已停稳)")
            if args.log_fk and len(actual) > 5:
                # 实测 base 位姿反变换回 UMI W' 系，直接对比 npz 原始轨迹
                npz_pos = pos_w
                npz_rpy = np.array([euler_zyx_deg(R) for R in R_w])
                act_pos_w, act_rpy_w = [], []
                for ap, aq in actual:
                    R_base = quat_to_rot(*aq)
                    pw, Rw = base_to_wprime(ap, R_base, start_pos, args.start_yaw,
                                            R_align, args.robot_tip_offset)
                    act_pos_w.append(pw); act_rpy_w.append(euler_zyx_deg(Rw))
                out_png = npz_path.parent / f"{npz_path.stem}_plan_compare.png"
                plot_plan_compare(npz_pos, npz_rpy, np.array(act_pos_w),
                                  np.array(act_rpy_w), out_png)
                print(f"  ✓ 对比图(UMI npz vs 机械臂实测, W'系): {out_png}")
            return

        servo_mode = getattr(RobotMode, "SERVO_CART_POSE", None)
        if servo_mode is None:
            sys.exit("✗ airbot_py 没有 SERVO_CART_POSE 模式，请告知实际枚举名")

        # 关键：servo 跟踪要切到更快的速度档。'slow' 把 moveit_servo 旋转/线性
        # scale 限到 0.05，旋转跟随被压到 5%，导致姿态(尤其大角度 pitch)严重滞后
        # —— 即使把轨迹放到极慢也跟不满。'default'(0.2)/'fast'(10) 给足旋转增益。
        # 接近起点仍用 slow(安全)，这里才切。
        print(f"切换速度档位 slow -> '{args.speed_profile}' 供 servo 跟踪")
        robot.set_speed_profile(args.speed_profile)
        time.sleep(0.1)

        # 高频密集重采样供 servo：SDK servo 控制器内部 ~250Hz，必须高频(默认100Hz)
        # 喂目标否则严重滞后(10Hz 那种)。仿 dynamic_pose_trajectory.py 验证过的方式。
        ts_s, pos_w_s, R_w_s, grip_s = load_and_resample(
            npz_path, args.servo_hz, args.speed_scale)
        cmd_pos_s, cmd_R_s = to_base_frame(pos_w_s, R_w_s, start_pos,
                                           args.start_yaw, R_align, args.robot_tip_offset)
        quats_s = np.stack([rot_to_quat_xyzw(R) for R in cmd_R_s])
        cmd_rpy_s = np.array([euler_zyx_deg(R) for R in cmd_R_s])
        g01_s = np.clip(grip_s / args.gripper_max, 0.0, 1.0)
        n = len(cmd_pos_s)
        step_mm = np.linalg.norm(np.diff(cmd_pos_s, axis=0), axis=1) * 1000
        print(f"伺服: {n} 个目标 @ {args.servo_hz:.0f}Hz, "
              f"每步 mean={step_mm.mean():.2f}mm max={step_mm.max():.2f}mm")

        robot.switch_mode(servo_mode)
        time.sleep(0.2)
        # 先播种初始目标，让 servo 控制器有保持点再开始流式
        robot.servo_cart_pose(cmd_pos_s[0].tolist(), quats_s[0].tolist())
        time.sleep(0.3)

        dt = 1.0 / args.servo_hz
        fk_every = max(1, round(args.servo_hz / 50.0))   # FK 采样限到 ~50Hz 不拖慢主循环
        last_g = None
        act_t, act_pos, act_rpy = [], [], []
        t_start = time.monotonic()
        next_t = t_start
        for k in range(n):
            if args.log_fk and (k % fk_every == 0):
                ap, aq = robot.get_end_pose()
                act_t.append(time.monotonic() - t_start)
                act_pos.append(list(ap))
                act_rpy.append(euler_zyx_deg(quat_to_rot(*aq)))
            if cmd_pos_s[k][2] >= min_z:         # z 保护
                robot.servo_cart_pose(cmd_pos_s[k].tolist(), quats_s[k].tolist())
            g = float(g01_s[k])
            if last_g is None or abs(g - last_g) > 0.01:
                robot.left._robot.servo_eef_pos([g])
                last_g = g
            next_t += dt                         # 绝对调度，避免累积 sleep 漂移
            pad = next_t - time.monotonic()
            if pad > 0:
                time.sleep(pad)
        print("✓ 回放完成")
        if args.log_fk and len(act_pos) > 5:
            out_png = npz_path.parent / f"{npz_path.stem}_fk_compare.png"
            plot_fk_compare(ts_s, cmd_pos_s, cmd_rpy_s,
                            np.array(act_t), act_pos, act_rpy, out_png)
            print(f"  ✓ FK 对比图: {out_png}")
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        robot.shutdown()


def main():
    ap = argparse.ArgumentParser(description="UMI 轨迹真机回放验证")
    ap.add_argument("npz", nargs="?", type=str,
                    help="umi_to_lerobot.py --poses 输出的 _poses.npz")
    ap.add_argument("--probe", action="store_true", help="连真机只读不动，确认约定")
    ap.add_argument("--test-cart", action="store_true",
                    help="诊断 move_to_cart_pose 的四元数约定+可达性(会小幅移动)")
    ap.add_argument("--calib-align", type=str, default=None, metavar="UMI_MCAP",
                    help="从'UMI 摆成机械臂零位同姿态'的静止 mcap 求 R_align")
    ap.add_argument("--robot-quat", nargs=4, type=float,
                    default=[0.0, 0.0, 0.0, 1.0], metavar=("X", "Y", "Z", "W"),
                    help="标定时机械臂姿态四元数(xyzw)，默认零位单位阵")
    ap.add_argument("--dry-run", action="store_true", help="离线计算+预览图，不连真机")
    ap.add_argument("--port", type=int, default=50050)
    ap.add_argument("--start-pos", nargs=3, type=float, default=[0.30, 0.0, 0.20],
                    metavar=("X", "Y", "Z"),
                    help="起始指尖位置 (base系, m)，默认 0.30 0 0.20")
    ap.add_argument("--auto-place", action="store_true",
                    help="按 play_config.json 的 workspace 自动选起点 "
                         "(xy 居中, 轨迹最低点 = 工作空间下界 + z-margin)")
    ap.add_argument("--z-margin", type=float, default=0.05,
                    help="auto-place 时轨迹最低点离工作空间下界的高度(m)，"
                         "调大整条轨迹抬高(默认 0.05；起点难规划时加到 0.12~0.18)")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                    help=f"play_config.json 路径 (默认 {DEFAULT_CONFIG})")
    ap.add_argument("--start-yaw", type=float, default=0.0,
                    help="W' 在 base 系中的 yaw (度)，默认 0=轨迹前向对准 base x")
    ap.add_argument("--align-rpy", nargs=3, type=float, default=[-1.0, -16.8, 0.0],
                    metavar=("R", "P", "Y"),
                    help="UMI body -> AIRBOT 工具系的轴向对齐旋转 (zyx 欧拉角, 度)。"
                         "默认 [-1.0, -16.8, 0] 由 --calib-align 标定(test5/6, ±0.6°)；"
                         "pitch≈-17° 即 UMI 相机光轴下倾。yaw 设 0(由 --start-yaw 吸收)")
    ap.add_argument("--robot-tip-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                    metavar=("X", "Y", "Z"),
                    help="get_end_pose 参考点->机械臂指尖偏移 (工具系, m)，待标定")
    ap.add_argument("--rate", type=float, default=10.0,
                    help="离线重采样/安全检查/plan 模式的帧率 Hz (默认 10)")
    ap.add_argument("--servo-hz", type=float, default=100.0,
                    help="SERVO_CART_POSE 流式更新频率 Hz (默认 100；SDK 内部"
                         "伺服 ~250Hz，喂太慢会严重滞后)")
    ap.add_argument("--speed-profile", type=str, default="default",
                    choices=["slow", "default", "fast"],
                    help="servo 跟踪阶段的速度档(接近起点始终用 slow)。"
                         "'slow' 旋转增益仅 0.05 会严重拖累姿态跟踪；"
                         "默认 'default'(0.2)；姿态仍滞后可试 'fast'")
    ap.add_argument("--speed-scale", type=float, default=0.5,
                    help="回放速度倍率, 0.5=半速 (默认 0.5)")
    ap.add_argument("--gripper-max", type=float, default=0.073,
                    help="夹爪归一化共享参考 = AIRBOT 全开(m)；UMI 开口(米)÷此值 clip 到[0,1]。"
                         "默认 0.073(AIRBOT 73mm)，与训练归一化(umi_to_lerobot)保持一致")
    ap.add_argument("--min-z", type=float, default=None,
                    help="base 系最低允许 z (m)；默认取 workspace 下界+1cm")
    ap.add_argument("--plan", action="store_true",
                    help="用笛卡尔路点规划(move_with_cart_waypoints)代替 servo 流式，"
                         "下位机内部解IK+规划, 精确到达每个路点, 绕开 servo 带宽不足")
    ap.add_argument("--plan-step", type=int, default=5,
                    help="--plan 时每隔几帧取一个路点 (默认 5，降采样减负担)")
    ap.add_argument("--log-fk", action="store_true",
                    help="记录机械臂实测 FK 末端位姿并出'指令 vs 实测'对比图。"
                         "servo 模式按时间对比；plan 模式按路径弧长对比(时序无关)")
    ap.add_argument("--force", action="store_true", help="忽略安全检查警告")
    args = ap.parse_args()

    if args.probe:
        probe(args.port)
        return
    if args.test_cart:
        test_cart(args.port)
        return
    if args.calib_align:
        mcap = Path(args.calib_align).expanduser().resolve()
        if not mcap.is_file():
            sys.exit(f"mcap 不存在: {mcap}")
        calib_align(mcap, args.robot_quat)
        return
    if not args.npz:
        sys.exit("需要 npz 文件（或用 --probe）")
    replay(args)


if __name__ == "__main__":
    main()
