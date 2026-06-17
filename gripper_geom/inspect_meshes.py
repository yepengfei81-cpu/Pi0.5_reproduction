#!/usr/bin/env python3
"""
临时检视脚本：把 URDF 里指定的几个 mesh(OBJ/STL)画出来, 确认哪些是夹爪。

⚠️ 注意：这里画的是各 mesh 的【局部坐标】, 没有按 URDF 关节变换装配——只用来确认
"哪个 mesh 长什么样/是不是夹爪指"。确认后, 装配到 TCP 系会用 URDF 的关节 origin 来做。

用法:
  python gripper_geom/inspect_meshes.py --model-dir /home/ypf/airbot_play_model \
      --meshes link6 left right -o gripper_geom/inspect.png
"""
import argparse
import pathlib

import numpy as np


def load_pts(path, n=4000):
    import trimesh
    m = trimesh.load(str(path), force="mesh")
    pts, _ = trimesh.sample.sample_surface(m, n)
    return np.asarray(pts), m.bounds, len(m.vertices), len(m.faces)


def main():
    ap = argparse.ArgumentParser(description="检视 URDF mesh, 确认夹爪")
    ap.add_argument("--model-dir", required=True, help="机械臂模型根目录(含 meshes/)")
    ap.add_argument("--meshes", nargs="+", default=["link6", "left", "right"],
                    help="要看的 mesh 名(不含扩展名)")
    ap.add_argument("--ext", default="obj")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("-o", "--out", default="gripper_geom/inspect.png")
    args = ap.parse_args()

    mdir = pathlib.Path(args.model_dir) / "meshes"
    data = []
    for name in args.meshes:
        p = mdir / f"{name}.{args.ext}"
        if not p.exists():
            print(f"  ✗ 找不到 {p}"); continue
        pts, bounds, nv, nf = load_pts(p, args.n)
        size = (bounds[1] - bounds[0])
        print(f"  ✓ {name}: 顶点={nv} 面={nf}  "
              f"包围盒尺寸(m)={np.round(size,3).tolist()}  "
              f"中心(m)={np.round(bounds.mean(0),3).tolist()}")
        data.append((name, pts))

    if not data:
        print("没有可显示的 mesh"); return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    ncol = len(data) + 1
    fig = plt.figure(figsize=(4.2 * ncol, 4.4))

    # 每个 mesh 单独一图(局部坐标)
    for i, (name, pts) in enumerate(data):
        ax = fig.add_subplot(1, ncol, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=cols[i % len(cols)])
        ax.set_title(f"{name}  (local frame)")
        _equal(ax, pts)

    # 叠加图(各自局部坐标, 仅看相对大小/形状)
    axo = fig.add_subplot(1, ncol, ncol, projection="3d")
    allp = []
    for i, (name, pts) in enumerate(data):
        axo.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2,
                    c=cols[i % len(cols)], label=name)
        allp.append(pts)
    axo.set_title("overlay (local frames, NOT URDF-assembled)")
    axo.legend(fontsize=8)
    _equal(axo, np.vstack(allp))

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f">>> 图已保存: {args.out}")


def _equal(ax, pts):
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    lim = np.array([pts.min(0), pts.max(0)])
    ctr = lim.mean(0); r = (lim[1] - lim[0]).max() / 2 + 1e-6
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)


if __name__ == "__main__":
    main()
