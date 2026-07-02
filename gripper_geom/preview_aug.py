#!/usr/bin/env python3
"""预览夹爪点云增强: 每把爪画"原始 + K 个增强样本", 按功能区上色, 存 png 供肉眼筛查。

坐标范围按"原始点云"固定 -> 一旦某次增强飞出/缩放离谱, 会明显超出框, 一眼可见。
用的是和训练同一个 augment_cloud(改这里的范围 = 改训练范围, 看满意再锁定)。

用法:
  python gripper_geom/preview_aug.py get
  python gripper_geom/preview_aug.py get parallel --k 5 -o gripper_geom/aug_preview.png
  python gripper_geom/preview_aug.py get --open-shift 0.006 --rot-deg 2   # 试不同范围
"""
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gripper_aug import augment_cloud  # noqa: E402

PAL = np.array(["#d62728", "#ff7f0e", "#7f7f7f"])   # tip / middle / rear


def region_by_x(x, tf=0.33, rf=0.33):
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    span = float(hi - lo) or 1.0
    r = np.ones(len(x), int)
    r[x >= hi - tf * span] = 0
    r[x <= lo + rf * span] = 2
    return r


def load(name):
    p = pathlib.Path(name)
    if not p.exists():
        p = pathlib.Path(__file__).parent / f"{name}.npy"
    d = np.load(p, allow_pickle=True).item()
    fid = d.get("finger_id")
    return p.stem, np.asarray(d["points"], float), (np.asarray(fid) if fid is not None else None)


def main():
    ap = argparse.ArgumentParser(description="夹爪点云增强预览")
    ap.add_argument("grippers", nargs="+", help="爪名(get/parallel)或 npy 路径, 可多个")
    ap.add_argument("--k", type=int, default=5, help="每把爪画几个增强样本")
    ap.add_argument("-o", "--out", default="gripper_geom/aug_preview.png")
    # 范围(与训练共用默认; 想试别的在这调)
    ap.add_argument("--jitter", type=float, default=0.0015)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--open-shift", type=float, default=0.010)
    ap.add_argument("--close-shift", type=float, default=0.008,
                    help="可闭合幅度(m), 只对本就张开的爪(非GET)生效; GET 恒为 0(只开不闭)")
    ap.add_argument("--scale", type=float, nargs=2, default=[0.95, 1.05])
    ap.add_argument("--rot-deg", type=float, default=2.5)
    args = ap.parse_args()

    def _ann(p):
        """把这次实际用的参数写成英文标注(对照 original = 无改动)。"""
        sh = p["shift"] * 1000.0
        op = f"open +{sh:.0f}mm" if sh > 0.5 else (f"close {sh:.0f}mm" if sh < -0.5 else "open 0mm")
        r = p["rot_deg"]
        return (f"scale x{p['scale']:.2f}  rot {r[0]:+.0f}/{r[1]:+.0f}/{r[2]:+.0f}deg\n"
                f"{op}  jitter {p['jitter']*1000:.1f}mm  drop {p['n0']-p['n']}->{p['n']}")

    rows = [load(g) for g in args.grippers]
    ncol = args.k + 1
    fig = plt.figure(figsize=(2.7 * ncol, 3.1 * len(rows)))
    for ri, (name, pts, fid) in enumerate(rows):
        is_get = "get" in name.lower()
        close = 0.0 if is_get else args.close_shift     # GET 本就闭合 -> 只开不闭
        kw = dict(jitter_std=args.jitter, dropout=args.dropout, open_shift_max=args.open_shift,
                  close_shift_max=close, scale_range=tuple(args.scale), rot_deg=args.rot_deg)
        c = pts.mean(0)
        m = np.ptp(pts, 0).max() / 2 * 1.35             # 框按原始固定, 飞出即可见
        print(f"[{name}] {'open-only (本就闭合)' if is_get else f'open+close (close<= {close*1000:.0f}mm)'}"
              f"  original bbox={np.round(np.ptp(pts, 0), 4).tolist()} n={len(pts)}")
        for ci in range(ncol):
            ax = fig.add_subplot(len(rows), ncol, ri * ncol + ci + 1, projection="3d")
            if ci == 0:
                P = pts
                title = f"{name} ORIGINAL\n(reference, no aug)"
            else:
                P, prm = augment_cloud(pts, fid, rng=ci, return_params=True, **kw)
                title = f"aug #{ci}\n{_ann(prm)}"
                print(f"   aug#{ci} bbox={np.round(np.ptp(P, 0), 4).tolist()} "
                      f"shift={prm['shift']*1000:+.0f}mm scale={prm['scale']:.2f} n={prm['n']}")
            reg = region_by_x(P[:, 0])
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, c=PAL[reg])
            ax.scatter([0], [0], [0], s=40, marker="*", c="k")   # TCP
            ax.set_title(title, fontsize=6.5)
            ax.set_xlim(c[0]-m, c[0]+m); ax.set_ylim(c[1]-m, c[1]+m); ax.set_zlim(c[2]-m, c[2]+m)
            ax.view_init(elev=18, azim=180)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout(); fig.savefig(args.out, dpi=110)
    print(f"✓ 预览图: {args.out}")


if __name__ == "__main__":
    main()
