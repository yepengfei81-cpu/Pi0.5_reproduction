#!/usr/bin/env python3
"""
UMI eef_pose 原点（TCP）枢轴标定。

用途（一次性标定）：确定 VIO 输出的 /robot0/vio/eef_pose 的位姿原点
到底在夹爪的哪个位置（指尖中心？相机？基座？），为 umi_to_lerobot.py
的位姿转换提供固定的 TCP 偏移参数。

数据要求：录一条"指尖顶住固定点、绕指尖朝各方向旋转"的 mcap
（pitch/roll/yaw 都扫到，幅度越大越好；指尖滑动 <1cm 没关系，会进残差）。

原理（标准 pivot calibration）：指尖在 body 系中的坐标 p 恒定且在世界系中
不动: R_t·p + t_t = c (对所有 t)。堆叠成线性方程组
    [R_t  -I3] [p; c] = -t_t
最小二乘解出 p（指尖在 eef body 系的偏移 = TCP 偏移）和 c（桌上固定点）。

判读：
  - |p| < ~2cm   → eef_pose 原点就在指尖，转换无需偏移；
  - |p| 较大     → 原点在别处，阶段 2 转换时用 T'_t = T_t·Trans(p)
                   把位姿重新表达到指尖（脚本会打印该参数）。
  - 残差 RMS    ≈ 录制时指尖滑动量，1cm 级别属正常。
  - 条件数/转角覆盖：若只绕单轴转，p 沿该轴分量不可观——脚本会警告。

用法:
    python pivot_calib.py test4_xxx_vio.mcap
    python pivot_calib.py test4_xxx_vio.mcap --plot pivot.png   # 附验证图
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def quat_to_rot(x, y, z, w):
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def read_poses(mcap_path: Path, topic: str):
    ts, xyz, quat = [], [], []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _s, _ch, msg, dec in reader.iter_decoded_messages(topics=[topic]):
            p, o = dec.pose.position, dec.pose.orientation
            ts.append(msg.log_time)
            xyz.append([p.x, p.y, p.z])
            quat.append([o.x, o.y, o.z, o.w])
    return np.array(ts), np.array(xyz, float), np.array(quat, float)


def rotation_coverage_deg(quats: np.ndarray) -> np.ndarray:
    """相对第一帧的旋转矢量（axis-angle）在三个轴上的覆盖范围（度）。"""
    R0 = quat_to_rot(*quats[0])
    rotvecs = []
    for q in quats:
        R = R0.T @ quat_to_rot(*q)
        # 旋转矩阵 -> 旋转矢量
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-8:
            rotvecs.append(np.zeros(3))
            continue
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / (2 * np.sin(angle))
        rotvecs.append(axis * angle)
    rv = np.degrees(np.array(rotvecs))
    return rv.max(0) - rv.min(0)   # 每轴覆盖范围


def main():
    ap = argparse.ArgumentParser(description="UMI eef_pose TCP 枢轴标定")
    ap.add_argument("input", type=str, help="绕指尖定点旋转的 VIO mcap")
    ap.add_argument("--pose-topic", type=str, default="/robot0/vio/eef_pose")
    ap.add_argument("--plot", type=str, default=None, help="输出验证图 png")
    args = ap.parse_args()

    mcap_path = Path(args.input).expanduser().resolve()
    if not mcap_path.is_file():
        sys.exit(f"文件不存在: {mcap_path}")

    ts, xyz, quat = read_poses(mcap_path, args.pose_topic)
    if len(xyz) < 30:
        sys.exit(f"位姿太少 ({len(xyz)} 帧)，无法标定")
    dur = (ts[-1] - ts[0]) / 1e9
    print(f"读取 {len(xyz)} 帧位姿, 时长 {dur:.1f}s")

    # 转角覆盖（决定可观性）
    cov = rotation_coverage_deg(quat)
    print(f"旋转覆盖 (绕 x/y/z, 度): {cov.round(1)}")
    if (cov < 20).sum() >= 2:
        print("⚠ 警告: 至少两个轴的旋转覆盖 <20°，p 的解可能病态，建议补录更大幅度的摆动")

    # 位置 "原始抖动"（标定前基准）：若原点本来就在指尖，这个数就很小
    spread0 = np.linalg.norm(xyz - xyz.mean(0), axis=1)
    print(f"\n位姿原点的位置散布(标定前): RMS={spread0.std():.4f}m  "
          f"mean|·|={spread0.mean():.4f}m  max={spread0.max():.4f}m")

    # ---- 最小二乘: [R_t  -I][p;c] = -t_t ----
    N = len(xyz)
    A = np.zeros((3 * N, 6))
    b = np.zeros(3 * N)
    for i in range(N):
        R = quat_to_rot(*quat[i])
        A[3 * i:3 * i + 3, :3] = R
        A[3 * i:3 * i + 3, 3:] = -np.eye(3)
        b[3 * i:3 * i + 3] = -xyz[i]
    sol, _res, _rank, svals = np.linalg.lstsq(A, b, rcond=None)
    p, c = sol[:3], sol[3:]
    cond = svals[0] / svals[-1]

    # 残差: 指尖重建点在世界系中的散布
    tips = np.array([quat_to_rot(*quat[i]) @ p + xyz[i] for i in range(N)])
    resid = np.linalg.norm(tips - c, axis=1)

    print(f"\n===== 标定结果 =====")
    print(f"TCP 偏移 p (指尖在 eef body 系中的坐标, m): "
          f"[{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]   |p|={np.linalg.norm(p):.4f}m")
    print(f"固定点 c (world): [{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}]")
    print(f"残差: RMS={np.sqrt((resid**2).mean()):.4f}m  max={resid.max():.4f}m  "
          f"(≈录制时指尖的滑动量)")
    print(f"最小二乘条件数: {cond:.1f}")

    if np.linalg.norm(p) < 0.02:
        print("\n结论: |p| < 2cm —— eef_pose 原点基本就在指尖，转换可不加偏移。")
    else:
        print(f"\n结论: eef_pose 原点不在指尖，距指尖 {np.linalg.norm(p)*100:.1f}cm。")
        print(f"umi_to_lerobot.py 阶段2 转换时用 --tcp-offset "
              f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 把位姿重表达到指尖。")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.plot(*xyz.T, lw=0.8, label="eef origin")
        ax1.scatter(*c, color="red", s=40, label="pivot c")
        ax1.set_title("eef_pose origin path")
        ax1.legend(fontsize=8)
        ax2 = fig.add_subplot(132, projection="3d")
        ax2.plot(*tips.T, lw=0.8, color="green")
        ax2.scatter(*c, color="red", s=40)
        ax2.set_title("reconstructed tip = R p + t\n(should be a tight blob)")
        # 让两个 3D 图同尺度，肉眼可比
        span = max(xyz.max(0) - xyz.min(0)) / 2
        mid = xyz.mean(0)
        for ax in (ax1, ax2):
            ax.set_xlim(mid[0]-span, mid[0]+span)
            ax.set_ylim(mid[1]-span, mid[1]+span)
            ax.set_zlim(mid[2]-span, mid[2]+span)
        ax3 = fig.add_subplot(133)
        ax3.plot((ts - ts[0]) / 1e9, resid * 1000)
        ax3.set_xlabel("t (s)"); ax3.set_ylabel("residual (mm)")
        ax3.set_title("pivot residual over time")
        out = Path(args.plot).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        print(f"✓ 验证图: {out}")


if __name__ == "__main__":
    main()
