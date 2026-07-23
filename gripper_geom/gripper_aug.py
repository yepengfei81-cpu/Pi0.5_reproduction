#!/usr/bin/env python3
"""共享的夹爪点云增强(训练 & 离线可视化共用同一套)。

设计原则:只做"真实会发生"的变化,且必须保住两样东西——
  ① 规范坐标系: 原点=TCP, +X 接近 / +Y 开合 / +Z 上;
  ② 夹爪真实身份与尺度。
因此: 不做自由 SO(3) 旋转、不整体平移离开 TCP、不镜像翻转、不大幅/各向异性缩放、
不超物理开合。所有增强以"真值"为中心、范围 clamp,保证干净的部署点云仍在分布内。

⚠ 只在训练时调用;部署/评测请喂未增强的规范点云。
"""
import numpy as np


def _rot_small(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def augment_cloud(points, finger_id=None, rng=None, *,
                  jitter_std=0.0015,        # per-point 高斯噪声(m), ~1.5mm 传感噪声
                  dropout=0.15,             # 随机丢点比例(模拟遮挡), 随机不按区域
                  open_shift_max=0.010,     # 开合抖动: 每指沿 ±Y "外移"(更开)上限(m)
                  close_shift_max=0.008,    # 每指 "内移"(更闭)上限(m); 实际按两指内距 clamp
                                            # -> GET 指尖相触(内距≈0)几乎不能闭, 平行爪可闭
                  scale_range=(0.95, 1.05), # 尺度抖动, 以 TCP 原点为中心
                  rot_deg=2.5,              # 微旋转上限(deg), 模拟标定误差
                  labels=None,              # (N,) 逐点标签(如 region), dropout 时同步保留对应行
                  return_params=False):     # True 则同时返回这次实际用了哪些参数(给可视化标注)
    """(N,3) TCP-common 点云 -> 增强后 (M,3)。

    finger_id (N,): 给了就做"开合抖动"——两指刚体沿各自 ±Y 移动。
        shift>0 = 外移(更开); shift<0 = 内移(更闭, 自动 clamp 防两指穿过中线)。
        本就张开的(平行爪)可给 close_shift_max>0 允许闭合; 本就闭合的(GET)留 0 只开。
        非 0/1 的点(如 -1=link6/body)不动。
    labels (N,): 给了就返回 (P, labels_kept[, params]) —— 几何变换不重排点序, 只有
        dropout 改行数, 对标签应用同一 keep 掩码, 保证逐点对应(区域化 token 拆分用)。
    rng: np.random.Generator / int 种子 / None。
    """
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    P = np.asarray(points, np.float64).copy()
    n0 = len(P)

    # 1) 开合抖动: 带符号 shift(+更开 / -更闭)。finger_id 没给就按 sign(Y) 分两指
    #    (夹爪点云无中间体, 可靠); 闭合按两指内侧间距 clamp 防穿模。
    shift = 0.0
    if open_shift_max > 0 or close_shift_max > 0:
        y = P[:, 1]
        if finger_id is not None:
            fid = np.asarray(finger_id)
            side = np.where(fid == 0, 1.0, np.where(fid == 1, -1.0, 0.0))
        else:
            side = np.sign(y)
        shift = float(rng.uniform(-close_shift_max, open_shift_max))
        if shift < 0:                                   # 闭合: 按内侧间距 clamp(GET 内距≈0 -> 几乎不闭)
            pos, neg = y[side > 0], y[side < 0]
            if len(pos) and len(neg):
                inner = float(pos.min() - neg.max())
                shift = -min(-shift, max(0.0, inner / 2 * 0.9))
        P[:, 1] += side * shift

    # 2) 尺度抖动(以 TCP 原点为中心; 指尖锚在原点附近, 保身份)
    s = float(rng.uniform(*scale_range))
    P *= s

    # 3) 微旋转(绕 TCP 原点, ±rot_deg, 仅模拟小标定误差)
    a_deg = rng.uniform(-rot_deg, rot_deg, 3) if rot_deg > 0 else np.zeros(3)
    if rot_deg > 0:
        P = P @ _rot_small(*np.radians(a_deg)).T

    # 4) per-point 抖动(传感噪声)
    if jitter_std > 0:
        P += rng.normal(0.0, jitter_std, P.shape)

    # 5) 随机 dropout(模拟遮挡; 随机, 不系统性丢某区域; 别丢太狠)
    keep = None
    if dropout > 0:
        k = rng.random(len(P)) >= dropout
        if k.sum() >= max(8, int(0.3 * len(P))):
            P = P[k]; keep = k

    P = P.astype(np.float32)
    out = [P]
    if labels is not None:
        lab = np.asarray(labels)
        out.append(lab[keep] if keep is not None else lab)
    if return_params:
        out.append({"scale": s, "rot_deg": a_deg.tolist(), "shift": shift,
                    "jitter": jitter_std, "n": len(P), "n0": n0})
    return out[0] if len(out) == 1 else tuple(out)
