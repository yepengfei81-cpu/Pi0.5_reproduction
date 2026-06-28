#!/usr/bin/env python3
"""交互式查看夹爪:点云(按功能区上色)+ 原始 CAD mesh 并排、同朝向,一窗对照。

- 选夹爪: 传名字(get / parallel)或描述符 npy 路径。
- 左边: 区域点云(红=指尖 tip / 橙=中部 middle / 灰=后部 rear), TCP 坐标轴(X红/Y绿/Z蓝)。
- 右边: 同坐标系的原始 CAD mesh(浅蓝, 同朝向, 沿开合轴平移开),由 build 脚本导出的 <npy>_mesh.ply。
- 优先 open3d(鼠标拖动旋转/缩放, 满意后系统截图); 没装则退回 matplotlib 交互窗口。

用法:
  python gripper_geom/view_descriptor.py get
  python gripper_geom/view_descriptor.py parallel
  python gripper_geom/view_descriptor.py gripper_geom/get.npy --tip-frac 0.3 --rear-frac 0.3
  python gripper_geom/view_descriptor.py get --no-mesh        # 只看点云
"""
import argparse
import pathlib
import numpy as np


def region_by_x(x, tip_frac, rear_frac):
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    span = float(hi - lo) or 1.0
    reg = np.ones(len(x), int)
    reg[x >= hi - tip_frac * span] = 0      # tip
    reg[x <= lo + rear_frac * span] = 2     # rear
    return reg


def main():
    ap = argparse.ArgumentParser(description="点云 + CAD mesh 并排查看(拖动旋转+截图)")
    ap.add_argument("gripper", help="夹爪名(get/parallel)或描述符 .npy 路径")
    ap.add_argument("--mesh", default=None, help="对齐 mesh 路径; 默认 <npy同名>_mesh.ply")
    ap.add_argument("--no-mesh", action="store_true", help="只看点云, 不显示 mesh")
    ap.add_argument("--tip-frac", type=float, default=0.33)
    ap.add_argument("--rear-frac", type=float, default=0.33)
    args = ap.parse_args()

    # 解析 npy: 支持传名字(get/parallel)或路径
    p = pathlib.Path(args.gripper)
    if not p.exists():
        p = pathlib.Path(__file__).parent / f"{args.gripper}.npy"
    if not p.exists():
        raise SystemExit(f"找不到描述符: {args.gripper} (也试了 {p})")
    d = np.load(p, allow_pickle=True).item()
    pts = np.asarray(d["points"], float)
    reg = region_by_x(pts[:, 0], args.tip_frac, args.rear_frac)
    palette = np.array([[0.84, 0.15, 0.16], [1.0, 0.50, 0.05], [0.55, 0.55, 0.55]])
    colors = palette[reg]

    mesh_path = pathlib.Path(args.mesh) if args.mesh else p.with_name(p.stem + "_mesh.ply")
    has_mesh = (not args.no_mesh) and mesh_path.exists()
    if not args.no_mesh and not mesh_path.exists():
        print(f"⚠ 未找到对齐 mesh {mesh_path}(用 build 脚本重新生成会一并导出),只显示点云。")
    off_y = float(np.ptp(pts[:, 1])) * 1.6 + 0.03   # mesh 沿开合轴(Y)平移到一边

    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector(colors)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=[0, 0, 0])
        geoms = [pc, frame]
        if has_mesh:
            m = o3d.io.read_triangle_mesh(str(mesh_path))
            m.compute_vertex_normals()
            m.paint_uniform_color([0.65, 0.72, 0.85])   # 浅蓝 CAD
            m.translate([0.0, off_y, 0.0])               # 平移到点云旁, 朝向不变
            geoms.append(m)
        print(">>> open3d: 左键拖动旋转 / 滚轮缩放 / 右键平移; 满意后用系统截图。关窗退出。")
        print("    左=区域点云(红指尖/橙中部/灰后部) + TCP轴(X红Y绿Z蓝)  右=原始CAD(同朝向)")
        o3d.visualization.draw_geometries(geoms, window_name=f"{p.name}  point cloud  +  CAD")
    except ImportError:
        print(">>> 未装 open3d, 退回 matplotlib 交互窗口(也能拖)。装 open3d 体验更好: pip install open3d")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=colors)
        ax.scatter([0], [0], [0], s=120, marker="*", c="k")
        L = 0.04
        ax.quiver(0, 0, 0, L, 0, 0, color="r"); ax.quiver(0, 0, 0, 0, L, 0, color="g"); ax.quiver(0, 0, 0, 0, 0, L, color="b")
        if has_mesh:
            import trimesh
            mv = np.asarray(trimesh.load(str(mesh_path), force="mesh").vertices, float).copy()
            mv[:, 1] += off_y
            ax.scatter(mv[:, 0], mv[:, 1], mv[:, 2], s=2, c="#a6c8e0")   # CAD 顶点(浅蓝)
        allp = np.vstack([pts, (mv if has_mesh else pts)])
        m_ = np.ptp(allp, 0).max() / 2; c_ = allp.mean(0)
        ax.set_xlim(c_[0]-m_, c_[0]+m_); ax.set_ylim(c_[1]-m_, c_[1]+m_); ax.set_zlim(c_[2]-m_, c_[2]+m_)
        ax.set_title("left: region point cloud (red tip/orange mid/gray rear)   right: CAD")
        plt.show()


if __name__ == "__main__":
    main()
