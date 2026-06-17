#!/usr/bin/env python3
"""
方案C 第1把爪：把 AIRBOT 平行夹爪(left+right, 可选 link6)按 URDF 关节变换装配到
end_effector(TCP)系, 采成点云 + 功能区锚点, 存成描述符 parallel.npy。

URDF 关键(airbot_play.urdf)：
  endleft : link6 -> left   origin xyz(0, 0.04,0) rpy(π,-π,0)  prismatic axis(0,1,0) q∈[0,0.04]
  endright: link6 -> right  origin xyz(0,-0.04,0) rpy(π,-π,0)  prismatic axis(0,1,0) q∈[-0.04,0]
  joint_end(fixed): link6 -> end_effector  origin xyz(0,0,0.02) rpy(0,-π/2,0)   <- TCP
  => 点云在 end_effector 系下(原点=TCP)。⚠️ 若 get_end_pose 用的不是 end_effector 而是
     link6 法兰, 用 --frame link6 切换。

用法:
  python gripper_geom/build_parallel_descriptor.py --model-dir /home/ypf/airbot_play_model \
      --opening 0.0 -o gripper_geom/parallel.npy --plot gripper_geom/parallel.png
"""
import argparse
import pathlib

import numpy as np


def rpy_to_R(roll, pitch, yaw):
    """URDF rpy(固定轴 XYZ): R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M


def apply(Tm, pts):
    return (pts @ Tm[:3, :3].T) + Tm[:3, 3]


def sample(path, n):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    pts, _ = trimesh.sample.sample_surface(m, n)
    return np.asarray(pts, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--frame", choices=["end_effector", "link6"], default="end_effector",
                    help="点云落在哪个系; 默认 end_effector(TCP)。若 get_end_pose 是 link6 法兰则改 link6")
    ap.add_argument("--opening", type=float, default=0.0, help="夹爪开度 q(米, 0~0.04), 单边")
    ap.add_argument("--tcp-rpy", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                    metavar=("ROLL", "PITCH", "YAW"),
                    help="装配后再转到【统一 get_end_pose 约定系】(+X 伸出/+Z 上)的额外旋转(deg)。"
                         "本平行爪(URDF-EE 指沿 -X)用: 0 0 180")
    ap.add_argument("--include-link6", action="store_true", help="把 link6(整条腕节)也并进描述符")
    ap.add_argument("--n", type=int, default=2500, help="每个 mesh 采样点数")
    ap.add_argument("-o", "--out", default="gripper_geom/parallel.npy")
    ap.add_argument("--plot", default="gripper_geom/parallel.png")
    args = ap.parse_args()

    mdir = pathlib.Path(args.model_dir) / "meshes"
    q = float(np.clip(args.opening, 0.0, 0.04))

    # link6 系下的关节变换
    T_l6_left = T(rpy_to_R(3.14159, -3.1416, 0), [0, 0.04, 0]) @ T(np.eye(3), [0, q, 0])
    T_l6_right = T(rpy_to_R(3.14159, -3.1416, 0), [0, -0.04, 0]) @ T(np.eye(3), [0, -q, 0])
    T_l6_ee = T(rpy_to_R(0, -1.5708, 0), [0, 0, 0.02])
    T_ee_l6 = np.linalg.inv(T_l6_ee)

    # 采点(各自局部系) -> link6 系
    pL = apply(T_l6_left, sample(mdir / "left.obj", args.n))
    pR = apply(T_l6_right, sample(mdir / "right.obj", args.n))
    parts = [("left", pL), ("right", pR)]
    if args.include_link6:
        parts.insert(0, ("link6", sample(mdir / "link6.obj", args.n * 2)))

    # link6 系 -> 目标系
    if args.frame == "end_effector":
        parts = [(name, apply(T_ee_l6, p)) for name, p in parts]

    # 额外朝向修正: 把描述符转到【统一 get_end_pose 约定系】(所有爪共用 +X 伸出/+Z 上)。
    # 注意: 每把爪的这个旋转量【不同】(取决于各自 CAD/URDF 原始朝向), 但目标系约定【相同】。
    if any(abs(a) > 1e-6 for a in args.tcp_rpy):
        Rextra = rpy_to_R(*np.radians(args.tcp_rpy))
        parts = [(name, p @ Rextra.T) for name, p in parts]

    pts = np.vstack([p for _, p in parts])
    region = np.concatenate([np.full(len(p), i) for i, (_, p) in enumerate(parts)]).astype(int)
    names = [name for name, _ in parts]

    # 功能区锚点: 每个指的"指尖"= 离整体质心最远的点(snap 在表面上, 因为就是采样点本身)
    base = pts.mean(0)
    anchors = {}
    for i, (name, p) in enumerate(parts):
        if name in ("left", "right"):
            tip = p[int(np.argmax(np.linalg.norm(p - base, axis=1)))]
            anchors[f"fingertip_{name}"] = tip.tolist()

    np.save(args.out, {"points": pts.astype(np.float32), "region": region,
                       "part_names": names, "anchors": anchors, "frame": args.frame,
                       "opening": q}, allow_pickle=True)
    print(f">>> 已存描述符: {args.out}")
    print(f"    点云 {pts.shape}  parts={names}  frame={args.frame}  opening={q}")
    print(f"    包围盒(m) min={pts.min(0).round(3).tolist()} max={pts.max(0).round(3).tolist()}")
    for k, v in anchors.items():
        print(f"    {k}: {np.round(v,3).tolist()}")

    # 可视化
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#888888", "#1f77b4", "#d62728"]
    fig = plt.figure(figsize=(11, 5.5))
    ax1 = fig.add_subplot(121, projection="3d")
    off = 1 if args.include_link6 else 0
    for i, (name, p) in enumerate(parts):
        ax1.scatter(p[:, 0], p[:, 1], p[:, 2], s=2, c=cols[i % 3], label=name)
    ax1.set_title(f"parallel gripper in {args.frame} frame"); ax1.legend(fontsize=8)
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="#cccccc")
    for k, v in anchors.items():
        v = np.asarray(v)
        ax2.scatter([v[0]], [v[1]], [v[2]], s=140, marker="*", edgecolor="k", label=k)
    # 画 TCP 原点
    ax2.scatter([0], [0], [0], s=80, marker="o", c="k", label="TCP origin")
    ax2.set_title("+ fingertip anchors + TCP"); ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        lim = np.array([pts.min(0), pts.max(0)]); ctr = lim.mean(0)
        r = (lim[1] - lim[0]).max() / 2 + 1e-6
        ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r); ax.set_zlim(ctr[2]-r, ctr[2]+r)
    fig.tight_layout(); fig.savefig(args.plot, dpi=110)
    print(f">>> 图已保存: {args.plot}")


if __name__ == "__main__":
    main()
