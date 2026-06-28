#!/usr/bin/env python3
"""
通用夹爪几何描述符构建(方案C, 任意夹爪)。

把一把夹爪的若干 mesh(STL/OBJ)装配到统一的 "工具系公共约定" 下, 采成点云 +
功能区标签, 存成描述符 npy, 并出验证图。所有夹爪都落到同一套约定, PointNet 才能
横向比较几何:

  公共约定(与 parallel.npy 对齐):
    - 原点 = TCP(指尖/抓取点)
    - +X = 接近方向(夹爪体沿 +X 延伸, 指尖在原点附近)
    - +Y = 开合轴(两指分布在 ±Y)
    - +Z = 上/厚度

------------------------------------------------------------------------------
GET 爪(本套, 两个独立 STL, 各自局部系):
  每个 finger 局部系(经 inspect 实测):
    - 原点在指尖背面; 接触面在局部 +Z = +0.0075m(=7.5mm, TCP 沿 +Z 偏 7.5mm)
    - 指尖在 Y=0, 指体往 -Y 延伸  -> 接近方向 = 局部 +Y
    - 局部 +Z = 接触面法线(闭合方向); 局部 +X = 横向
  装配: 两指接触面相向, 沿闭合轴(局部 Z)分开 opening(默认 0.085m), 翻转其中一指。
  装配系: 接近=+Y, 开合=±Z, 横向=X, TCP=原点(尖端 Y=0 / 间隙中心 Z=0 / X=0)。
  再旋到公共约定: 局部(接近Y,开合Z,横向X) -> 公共(接近X,开合Y,上Z)。

用法:
  python build_gripper_descriptor.py get \
    --left  cad/GET/left_gripper.stl \
    --right cad/GET/right_gripper.stl \
    --opening 0.085 -o gripper_geom/get.npy --plot gripper_geom/get.png
  # 看图纠偏: 接触面尖端中点是否在原点、指尖是否朝 +X、+Z 是否朝上、开口是否沿 Y、
  #          宽指/窄指不对称是否保留; 不对就调 --flip / --opening 重跑。
"""
import argparse
import pathlib

import numpy as np
import trimesh


FINGER_TCP_Z = 0.0075   # 每个 finger 局部系: 接触面相对原点(背面)沿 +Z 的偏移(m)


def rot_x(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def load_points(path, n, units):
    m = trimesh.load(str(path), force="mesh")
    pts, _ = trimesh.sample.sample_surface(m, n)
    pts = np.asarray(pts, float)
    if units == "auto":
        units = "mm" if np.ptp(pts, axis=0).max() > 1.0 else "m"
    if units == "mm":
        pts = pts / 1000.0
    return pts


def load_aligned_mesh(path, units, M):
    """加载原始 mesh -> 缩放到米 -> 应用同一坐标重映射 M, 得到与点云同一公共系的 mesh。"""
    m = trimesh.load(str(path), force="mesh")
    if units == "auto":
        units = "mm" if (m.bounds[1] - m.bounds[0]).max() > 1.0 else "m"
    if units == "mm":
        m.apply_scale(0.001)
    T = np.eye(4); T[:3, :3] = np.asarray(M, float)
    m.apply_transform(T)
    return m


def assemble_get(left_pts, right_pts, opening, flip):
    """两指局部系 -> 装配系(接近=+Y, 开合=±Z, 横向=X, TCP=原点)。

    右指: 接触面法线本就是 +Z, 放到 -Z 侧、法线指向中心(+Z)。
    左指: 翻 180°(绕接近轴 Y, +Z->-Z)放到 +Z 侧、法线指向中心(-Z)。
    两接触面间隙 = opening。
    flip 控制翻哪一指(默认 left), 看图不对就换。
    """
    half = opening / 2.0
    out = []
    for name, pts in (("left", left_pts), ("right", right_pts)):
        p = pts.copy()
        if name == flip:
            p = p @ rot_y(180).T              # 翻转使接触面相向
            # 翻转后接触面在局部 -Z; 放到 +half 侧, 接触面落到 +half
            p[:, 2] += half + FINGER_TCP_Z
        else:
            # 接触面在 +Z; 放到 -half 侧, 接触面落到 -half
            p[:, 2] += -half - FINGER_TCP_Z
        out.append((name, p))
    return out   # 装配系: 接近=+Y, 开合=Z, 横向=X, 尖端在 Y=0, 间隙中心 Z=0


def to_common(pts):
    """装配系(接近Y, 开合Z, 横向X) -> 公共约定(接近X, 开合Y, 上Z)。
    真实机械臂工具系: +X = 接近方向, 指尖朝 +X(指尖在 +X 端 X≈0, 夹爪体在 -X)。
    映射: 公共x=装配y(指尖Y=0->X≈0, 指体-Y->-X), 公共y=装配z, 公共z=装配x。"""
    P = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]], float)
    return pts @ P.T


def region_by_x(x, tip_frac=0.33, rear_frac=0.33):
    """按接近轴 X 切 3 个功能区(指尖在 +X 端): 0=指尖, 1=中部, 2=后部(近 -X 端)。"""
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    span = float(hi - lo) or 1.0
    reg = np.ones(len(x), np.int8)              # 1 = 中部 middle
    reg[x >= hi - tip_frac * span] = 0          # 0 = 指尖 tip(近 +X)
    reg[x <= lo + rear_frac * span] = 2         # 2 = 后部 rear(近 -X)
    return reg


def _axis_vec(s):
    d = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    neg = s.startswith("-")
    v = np.array(d[s.lstrip("-")], float)
    return -v if neg else v


def assembly_remap(approach: str, openax: str) -> np.ndarray:
    """装配体坐标系 -> 公共约定(接近=+X, 开合=+Y, 上=+Z)的旋转。
    行 = 公共系各轴在装配体坐标里的方向; p_common = p_stl @ M.T。
    上轴由 接近×开合 自动取(保证右手系/正交)。"""
    ax = _axis_vec(approach)
    op = _axis_vec(openax)
    up = np.cross(ax, op)
    M = np.stack([ax, op, up])
    assert abs(np.linalg.det(M) - 1.0) < 1e-6, "approach/open 不正交, 检查轴设置"
    return M


def main():
    ap = argparse.ArgumentParser(description="通用夹爪几何描述符构建")
    ap.add_argument("name", help="夹爪名(存进描述符, 如 get)")
    # 方式A(推荐): 单个已装配 STL(指尖已在原点, 姿态=装机时), 只需坐标重映射
    ap.add_argument("--assembly", default=None,
                    help="已装配 STL 路径(指尖在原点)。给了就走装配体模式, 忽略 --left/--right")
    ap.add_argument("--asm-approach", default="-x", choices=["x", "-x", "y", "-y", "z", "-z"],
                    help="装配体里'接近方向(指尖朝向)'是哪个轴 (GET: -x)")
    ap.add_argument("--asm-open", default="z", choices=["x", "-x", "y", "-y", "z", "-z"],
                    help="装配体里'开合轴'是哪个轴 (GET: z)")
    # 方式B: 两个独立 finger STL, 按 opening 装配(旧 GET 用)
    ap.add_argument("--left", default=None)
    ap.add_argument("--right", default=None)
    ap.add_argument("--opening", type=float, default=0.085, help="两接触面间隙(m), GET 最大 0.085")
    ap.add_argument("--flip", choices=["left", "right"], default="left",
                    help="翻 180° 让接触面相向的那一指(看图不对就换)")
    ap.add_argument("--units", choices=["auto", "mm", "m"], default="auto")
    ap.add_argument("--n", type=int, default=2500, help="每个 finger 采样点数")
    ap.add_argument("--tip-frac", type=float, default=0.33,
                    help="近 +X 端这一比例标为指尖区(tip)")
    ap.add_argument("--rear-frac", type=float, default=0.33,
                    help="近 -X 端这一比例标为后部区(rear); 中间为 middle")
    ap.add_argument("-o", "--out", default="gripper_geom/get.npy")
    ap.add_argument("--plot", default="gripper_geom/get.png")
    args = ap.parse_args()

    if args.assembly:                                # 方式A: 已装配 STL + 坐标重映射
        M = assembly_remap(args.asm_approach, args.asm_open)
        raw = load_points(args.assembly, args.n * 2, args.units)
        print(f"assembly 采样 {raw.shape}  原始bbox尺寸={np.round(np.ptp(raw,0),4).tolist()}")
        print(f"重映射 M(行=公共轴在装配体坐标):\n{M.astype(int)}")
        pts = (raw @ M.T).astype(np.float32)
        # finger_id 按开合轴(公共 +Y/-Y)分两指; 哪个是 left 看图定
        finger_id = (pts[:, 1] < 0).astype(np.int8)
        meta_extra = {"opening": "as-built", "asm_approach": args.asm_approach,
                      "asm_open": args.asm_open}
        aligned_mesh = load_aligned_mesh(args.assembly, args.units, M)   # 同公共系的 mesh
    else:                                            # 方式B: 两个 finger STL 按 opening 装配
        lp = load_points(args.left, args.n, args.units)
        rp = load_points(args.right, args.n, args.units)
        print(f"left  采样 {lp.shape}  bbox尺寸={np.round(np.ptp(lp,0),4).tolist()}")
        print(f"right 采样 {rp.shape}  bbox尺寸={np.round(np.ptp(rp,0),4).tolist()}")
        parts = assemble_get(lp, rp, args.opening, args.flip)
        pts_list, finger_id = [], []
        for i, (name, p) in enumerate(parts):
            pc = to_common(p)
            pts_list.append(pc)
            finger_id += [0 if name == "left" else 1] * len(pc)
        pts = np.concatenate(pts_list).astype(np.float32)
        finger_id = np.array(finger_id, np.int8)        # 0=left, 1=right
        meta_extra = {"opening": args.opening}
        aligned_mesh = None   # 两指模式暂不导出对齐 mesh

    # 区域: 公共系接近=+X, 指尖在 +X 端。3 段: 0=指尖 1=中部 2=后部
    region = region_by_x(pts[:, 0], args.tip_frac, args.rear_frac)

    anchors = {
        "tcp": [0.0, 0.0, 0.0],
        "fingertip_left": pts[(finger_id == 0) & (region == 0)].mean(0).tolist()
        if ((finger_id == 0) & (region == 0)).any() else None,
        "fingertip_right": pts[(finger_id == 1) & (region == 0)].mean(0).tolist()
        if ((finger_id == 1) & (region == 0)).any() else None,
    }

    out = pathlib.Path(args.out)
    np.save(out, {"points": pts, "region": region, "finger_id": finger_id,
                  "part_names": ["left", "right"], "anchors": anchors,
                  "frame": "tcp_common", "name": args.name, **meta_extra},
            allow_pickle=True)
    print(f"\n✓ {out}  点云 {pts.shape}  ({meta_extra})")
    print(f"  公共系 bbox: min={np.round(pts.min(0),4).tolist()} "
          f"max={np.round(pts.max(0),4).tolist()}")
    print(f"  anchors: {anchors}")
    if aligned_mesh is not None:                      # 导出与点云同坐标系的 mesh, 供 view_descriptor 并排显示
        mp = out.with_name(out.stem + "_mesh.ply")
        aligned_mesh.export(mp)
        print(f"  ✓ 对齐 mesh: {mp}")

    # ---- 验证图: 左=本爪(按区域上色)+TCP+坐标轴; 右=和 parallel.npy 并排 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = np.array(["#d62728", "#ff7f0e", "#7f7f7f"])   # 0=指尖红 1=中部橙 2=后部灰

    def draw(ax, P, fid, reg, title):
        col = palette[np.asarray(reg).astype(int)]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, c=col)
        ax.scatter([0], [0], [0], s=120, marker="*", c="k")       # TCP
        L = 0.04
        ax.quiver(0, 0, 0, L, 0, 0, color="r"); ax.text(L, 0, 0, "X approach", color="r")
        ax.quiver(0, 0, 0, 0, L, 0, color="g"); ax.text(0, L, 0, "Y open", color="g")
        ax.quiver(0, 0, 0, 0, 0, L, color="b"); ax.text(0, 0, L, "Z up", color="b")
        ax.set_title(title); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        m = np.ptp(P, 0).max() / 2
        c = P.mean(0)
        for setlim, ci in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setlim(c[ci] - m, c[ci] + m)
        # 视角: 从夹爪根部(-X)看向指尖(+X), +X 朝里、Z 朝上(=坐在机械臂后方看夹爪)
        ax.view_init(elev=18, azim=180)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    draw(ax1, pts, finger_id, region, f"{args.name} (red=tip, orange=middle, gray=rear, *=TCP)")
    ax2 = fig.add_subplot(122, projection="3d")
    par = pathlib.Path("gripper_geom/parallel.npy")
    if par.exists():
        d = np.load(par, allow_pickle=True).item()
        pp = np.asarray(d["points"])
        rr = region_by_x(pp[:, 0], args.tip_frac, args.rear_frac)   # 同规则重算, 才能和 GET 对比
        draw(ax2, pp, None, rr, "parallel.npy (compare orientation)")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=110)
    print(f"✓ 验证图: {args.plot}")


if __name__ == "__main__":
    main()
