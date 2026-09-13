"""关节限位守卫: 采集/推理时, 关节逼近限位或 follow 顶住跟不上 lead 就在终端提示。

按关节分级(依据 700 条示教的实测统计, 见 MODEL_ITERATIONS.md "已知部署问题"):
  - J4/J6 手腕滚转关节: 最危险。贴限位时笛卡尔 IK 可能换解(J4-180°, J5 取反, J6+180°),
    部署时表现为突然一跳。平行爪抽屉示教 31/50 条 J4 顶到上限。提示 <margin(15°), 危险 <5°;
  - J1/J5: 提示 <10°, 危险 <3°;
  - J2/J3: 常规任务里 J2 频繁顶上限止挡(积木/金字塔/擦黑板几乎每条), 只限制够远、部署无害,
    不实时刷屏, 只在本条小结里给出顶止挡时长。
跟随卡住: 关节已在提示区内, 且 |lead 指令 − follow 实际| > 12° 持续 0.3s —— 顶住的直接证据
(区外的大误差多是手腕快速转动的正常滞后, 不报)。
"""
import math

# AIRBOT Play 关节限位(rad), 取自 ~/airbot_play_model/urdf/airbot_play.urdf(2026-05)。
# 实测 follow 固件软限位比这里略靠内(J4 上限实际停在 +171.9°, URDF +172.6°)。
PLAY_JOINT_LIMITS = ((-3.151, 2.089), (-2.963, 0.181), (-0.094, 3.161),
                     (-3.012, 3.012), (-1.859, 1.859), (-3.017, 3.017))
WRIST = (3, 5)      # J4, J6
STOPS = (1, 2)      # J2, J3


class JointGuard:
    def __init__(self, name, margin_deg=15.0, track_deg=12.0, track_frames=3, rate_hz=10,
                 limits=PLAY_JOINT_LIMITS, log=print):
        m = float(margin_deg); s = min(m, 10.0)
        self.warn = (s, None, None, m, s, m)                       # None = 只进小结
        self.danger = (min(3.0, s), None, None, min(5.0, m), min(3.0, s), min(5.0, m))
        self.stop_deg = 1.0
        self.name, self.track_deg, self.track_frames = name, track_deg, track_frames
        self.rate_hz, self.limits, self.log = rate_hz, limits, log
        self.reset()

    def reset(self):
        n = len(self.limits)
        self.n = 0
        self.level = [0] * n                 # 0=安全 1=已报接近 2=已报危险
        self.min_margin = [math.inf] * n
        self.min_t = [0.0] * n
        self.min_side = [""] * n
        self.danger_frames = [0] * n
        self.stop_frames = [0] * n
        self.track_run = [0] * n
        self.track_events = []               # (关节号, 偏差°, 时刻s)

    def _margin(self, i, q):
        lo, hi = self.limits[i]
        dlo, dhi = math.degrees(q - lo), math.degrees(hi - q)
        return (dlo, "下限", lo) if dlo < dhi else (dhi, "上限", hi)

    def update(self, cmd_q, actual_q=None):
        """每个记录帧调用一次。cmd_q: lead 指令(推理时传实际角); actual_q: follow 实际, 可省。单位 rad。"""
        t = self.n / self.rate_hz
        self.n += 1
        if cmd_q is None:
            return
        nj = min(len(cmd_q), len(self.limits))
        in_zone = [False] * nj
        for i in range(nj):
            q = cmd_q[i]
            m, side, lim = self._margin(i, q)
            if m < self.min_margin[i]:
                self.min_margin[i], self.min_t[i], self.min_side[i] = m, t, side
            if self.warn[i] is None:
                if m < self.stop_deg:
                    self.stop_frames[i] += 1
                continue
            if m < self.warn[i]:
                in_zone[i] = True
                if m < self.danger[i]:
                    self.danger_frames[i] += 1
                lvl = 2 if m < self.danger[i] else 1
                if lvl > self.level[i]:
                    self.level[i] = lvl
                    what = f"已超出{side} {-m:.1f}°" if m < 0 else f"离{side}仅 {m:.1f}°"
                    tag = "\a⛔ 危险" if lvl == 2 else "⚠ 接近限位"
                    hint = " (手腕滚转关节: 部署时 IK 易换解抽动)" if (lvl == 2 and i in WRIST) else ""
                    self.log(f"    {tag} [{self.name}] J{i+1} {what} "
                             f"(角度 {math.degrees(q):+.0f}°, 限位 {math.degrees(lim):+.0f}°) @ {t:.1f}s{hint}")
            elif m > self.warn[i] + 3.0:
                self.level[i] = 0

        if actual_q is None:
            return
        for i in range(min(nj, len(actual_q))):
            e = abs(math.degrees(cmd_q[i] - actual_q[i]))
            if in_zone[i] and e > self.track_deg:
                self.track_run[i] += 1
                if self.track_run[i] == self.track_frames:
                    self.track_events.append((i + 1, e, t))
                    self.log(f"    \a⛔ [{self.name}] J{i+1} 顶住限位: follow 跟不上 lead 偏差 {e:.0f}° "
                             f"持续 {self.track_frames / self.rate_hz:.1f}s @ {t:.1f}s")
            else:
                self.track_run[i] = 0

    def summary(self):
        """本条小结(一行), 录完问"保存本条?"之前打印。"""
        live = [i for i in range(len(self.limits)) if self.warn[i] is not None]
        if self.n == 0 or all(math.isinf(self.min_margin[i]) for i in live):
            return f"[{self.name}] 关节守卫: 无数据"
        j = min(live, key=lambda k: self.min_margin[k])
        parts = [f"最贴限位 J{j+1} 离{self.min_side[j]} {self.min_margin[j]:.1f}° @ {self.min_t[j]:.1f}s"]
        parts += [f"J{i+1} 贴限位 {self.danger_frames[i] / self.rate_hz:.1f}s" for i in live if self.danger_frames[i]]
        parts += [f"J{i+1} 顶止挡 {self.stop_frames[i] / self.rate_hz:.1f}s(一般无害)" for i in STOPS if self.stop_frames[i]]
        if self.track_events:
            parts.append("顶住: " + ", ".join(f"J{k} {e:.0f}°@{tt:.1f}s" for k, e, tt in self.track_events))
        wrist_hit = any(self.min_margin[i] < self.danger[i] for i in WRIST) or \
            any(k - 1 in WRIST for k, _, _ in self.track_events)
        if wrist_hit:
            verdict = "⛔ 手腕关节顶限位: 部署时易抽动, 考虑换 J4 分支或调整摆位"
        elif self.track_events:
            verdict = "⛔ 关节顶住限位, 建议回放检查"
        elif any(self.min_margin[i] < self.danger[i] for i in live):
            verdict = "⚠ 贴近限位"
        elif any(self.min_margin[i] < self.warn[i] for i in live):
            verdict = "⚠ 接近限位"
        else:
            verdict = "✓ 余量充足"
        return f"[{self.name}] 关节守卫: " + "; ".join(parts) + "  " + verdict
