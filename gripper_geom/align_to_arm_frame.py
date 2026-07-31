#!/usr/bin/env python3
"""把各爪点云统一到【机械臂参考点(ARP)系】——去几何标定训练(v3)的前置步骤。

背景: 机械臂只知道第7个电机在位, 报的 get_end_pose 永远是【出厂平行爪指尖】那个点,
与实际装哪把爪无关 -> 它是一个"与夹爪无关的固定参考点"(ARP), 正是我们要的公共原点。

问题: 现有 grippers.npz 里两把爪各自以【自己的指尖】为原点(parallel frame=end_effector,
get frame=tcp_common), 于是 "GET 比平行爪多伸 17mm" 这个信息被归一化掉了——点云里
根本不存在。去掉 state 的 TCP 补偿后模型无从得知指尖在哪, 问题会变成 ill-posed。

本脚本做三件事(只动几何, 不动语义):
  1) 平移对齐: 每把爪沿 +X 平移, 使其指尖落在该爪 tcp_offset 处
     (parallel -> x=0 即 ARP 本身; get -> x=+0.017)。此后原点=ARP, 两爪共用。
  2) 法兰面裁剪: 丢弃 x < -MOUNT_LEN 的点。parallel 的描述子来自 URDF, 含安装面
     【后方】21.6mm 的结构(占 41%), GET 的来自 CAD 只到安装面——这是建模差异造成的
     "伪身份线索"(模型可以靠"有没有后方结构"认爪, 却学不到真几何, 且第三把爪必然
     没有该结构 -> 系统性错判)。裁剪后两爪都只描述"安装面往前的部分", 与未来新爪
     的建模方式一致。
  3) 区域重标: 裁剪改变了 x 跨度, 按与原脚本相同的规则(x 方向等分三段)重算
     tip/mid/rear, 保证两爪区域定义同构。

验收(脚本会自动核对): 对齐+裁剪后各爪长度应等于卡尺实测的"安装面->指尖"
  parallel 73.5mm / get 90.5mm; GET 后端应恰好落在法兰面 -MOUNT_LEN。

用法:
    python gripper_geom/align_to_arm_frame.py                 # 写 grippers_armframe.npz
    python gripper_geom/align_to_arm_frame.py --plot side.png  # 附侧视图对照
"""
import argparse
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from gripper_params import GRIPPER_PARAMS  # noqa: E402

# ARP(=出厂平行爪指尖) 到安装面的距离(m)。卡尺: 平行爪 安装面->指尖 73.5mm,
# 而 ARP 就落在平行爪指尖 -> 安装面在 ARP 后方 73.5mm 处。
MOUNT_LEN = 0.0735
# 各爪"安装面->指尖"卡尺实测(m), 仅用于验收核对
CALIPER_LEN = {"parallel": 0.0735, "get": 0.0905}


def region_by_x(x, tip_frac=0.33, rear_frac=0.33):
    """与 build_gripper_descriptor.py 同规则: 沿 +X 等分三段。0=tip(近+X) 1=mid 2=rear。"""
    lo, hi = float(x.min()), float(x.max())
    span = max(hi - lo, 1e-9)
    reg = np.ones(len(x), np.int8)
    reg[x >= hi - tip_frac * span] = 0
    reg[x <= lo + rear_frac * span] = 2
    return reg


def main():
    ap = argparse.ArgumentParser(description="点云统一到机械臂参考点(ARP)系")
    ap.add_argument("--src", default=str(_HERE / "grippers.npz"))
    ap.add_argument("--out", default=str(_HERE / "grippers_armframe.npz"))
    ap.add_argument("--mount-len", type=float, default=MOUNT_LEN)
    ap.add_argument("--no-trim", action="store_true", help="不做法兰面裁剪(保留伪线索, 仅调试用)")
    ap.add_argument("--plot", default=None, help="存侧视图对照 png")
    args = ap.parse_args()

    z = np.load(args.src, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    out = {"names": np.array(names)}
    report = []

    for n in names:
        pts = z[f"{n}_points"].astype(np.float64)
        reg_old = z[f"{n}_region"]
        fid = z[f"{n}_finger"]
        off_x = float(GRIPPER_PARAMS[n]["tcp_offset"][0])

        # 1) 平移: 指尖(最靠 +X 的点) -> 该爪 tcp_offset
        tip_x = float(pts[:, 0].max())
        shift = off_x - tip_x
        pts = pts.copy()
        pts[:, 0] += shift

        # 2) 法兰面裁剪
        if args.no_trim:
            keep = np.ones(len(pts), bool)
        else:
            keep = pts[:, 0] >= -args.mount_len
        pts, fid = pts[keep], fid[keep]

        # 3) 区域重标(裁剪后按同一规则等分)
        reg = region_by_x(pts[:, 0])

        out[f"{n}_points"] = pts.astype(np.float32)
        out[f"{n}_region"] = reg.astype(np.int8)
        out[f"{n}_finger"] = np.asarray(fid, np.int8)
        report.append((n, tip_x, shift, keep.sum(), len(keep) - keep.sum(), pts, reg, reg_old))

    print(f"源: {args.src}\n{'爪':10s}{'原指尖x':>10s}{'平移':>9s}"
          f"{'保留点':>8s}{'裁掉':>7s}{'前端':>9s}{'后端':>9s}{'长度':>9s}{'卡尺':>8s}")
    ok = True
    for n, tip_x, shift, nkeep, ncut, pts, reg, _ in report:
        lo, hi = pts[:, 0].min(), pts[:, 0].max()
        length, cal = hi - lo, CALIPER_LEN.get(n)
        mark = ""
        if cal is not None:
            good = abs(length - cal) < 0.002
            ok &= good
            mark = f"{cal * 1000:.1f}mm {'✓' if good else '✗'}"
        print(f"{n:10s}{tip_x:+10.4f}{shift:+9.4f}{nkeep:>8d}{ncut:>7d}"
              f"{hi:+9.4f}{lo:+9.4f}{length * 1000:>8.1f}mm  {mark}")

    tips = {r[0]: float(r[5][:, 0].max()) for r in report}
    if "parallel" in tips and "get" in tips:
        gap = (tips["get"] - tips["parallel"]) * 1000
        print(f"\n两爪指尖前后差 = {gap:.1f}mm (应 = GET 比平行爪多伸的 17.0mm) "
              f"{'✓' if abs(gap - 17.0) < 0.5 else '✗'}")
    print(f"法兰面 x = {-args.mount_len:+.4f}: 各爪后端都应落在此处或其前方")

    print("\n区域点数(裁剪+重标后):")
    for r in report:
        n, reg, reg_old = r[0], r[6], r[7]
        cnt = {k: int((reg == v).sum()) for k, v in (("tip", 0), ("mid", 1), ("rear", 2))}
        print(f"  {n:10s}{cnt}  (原 rear={int((reg_old == 2).sum())})")

    np.savez(args.out, **out)
    print(f"\n写出: {args.out}  {'(全部验收通过 ✓)' if ok else '(⚠ 有验收未通过, 请核对)'}")
    print("注意: npz 被 gitignore, 上集群训练前需手动 rsync 这个新文件。")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for ax, use_new in zip(axes, (False, True)):
            for n, color in zip(names, ("tab:blue", "tab:red")):
                p = (out[f"{n}_points"] if use_new else z[f"{n}_points"]).astype(float)
                ax.scatter(p[:, 0], p[:, 1], s=1, alpha=0.25, c=color, label=n)
            ax.axvline(0, color="k", lw=1.2, ls="--")
            ax.text(0, ax.get_ylim()[1] * 0.9, " ARP(臂报点)", fontsize=8)
            if use_new:
                ax.axvline(-args.mount_len, color="g", lw=1.2, ls=":")
                ax.text(-args.mount_len, ax.get_ylim()[1] * 0.9, " 法兰面", fontsize=8, color="g")
            ax.set_title("AFTER: arm-reference frame (tips separated by 17mm)" if use_new
                         else "BEFORE: each cloud anchored at its own tip (17mm invisible)")
            ax.set_ylabel("y (m)"); ax.legend(markerscale=8, fontsize=8); ax.grid(alpha=0.3)
        axes[1].set_xlabel("x (m), +X = approach direction")
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print(f"侧视图: {args.plot}")


if __name__ == "__main__":
    main()
