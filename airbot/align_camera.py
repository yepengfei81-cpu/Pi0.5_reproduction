#!/usr/bin/env python3
"""环境相机复位工具: 把数据集里的参考画面叠在实时画面上, 手动把相机调回采集时的位置。

为什么需要: 模型的 state 是 W' 相对位姿, 环境相机是唯一的全局定位线索, 相机挪过就是 OOD。
参考画面取自采集时的 mp4(与训练输入完全同源: 同相机、同 640x480 流配置)。

用法:
  python airbot/align_camera.py                                    # 默认: 平行爪扫地数据集 episode 0 首帧
  python airbot/align_camera.py --dataset <lerobot目录> --episode 3 --frame 0
  python airbot/align_camera.py --median 20                        # 用 20 条首帧的逐像素中值当参考(物体被抹掉, 只剩静态背景)
  python airbot/align_camera.py --live-video <另一条的 image mp4>  # 无相机调试: 用视频当实时源
  python airbot/align_camera.py --once /tmp/a.png                  # 渲染一帧存图退出(不开窗口)

窗口按键:
  m  切换显示模式: blend(半透明叠加) -> edges(参考边缘绿/实时边缘红, 重合=黄) -> split(左参考右实时)
                 -> flicker(每 0.5s 交替) -> diff(灰度差)
  a / d  叠加透明度 -/+        s  存当前实时帧+叠加图到 --save-dir        q / Esc  退出
屏幕上的 edge match = 参考画面的边缘里有多少比例在实时画面 2px 内也有边缘; 对齐时应明显上升
(两次画面物体不同, 到不了 100%; 看静态结构: 垫板边、桌沿、簸箕/杯架底座)。
"""
import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np

HEAD_CAMERA_SERIAL = "230422271972"
DEFAULT_DATASET = "/home/ypf/pi_data/parallel/sweep_task/sweep_objects_20260824_003716"
MODES = ("blend", "edges", "split", "flicker", "diff")


def episode_video(root: pathlib.Path, ep: int, key: str) -> pathlib.Path:
    info = json.loads((root / "meta" / "info.json").read_text())
    chunk = ep // int(info.get("chunks_size", 1000))
    p = root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep:06d}.mp4"
    if not p.exists():
        sys.exit(f"找不到 {p}")
    return p


def read_frame(mp4: pathlib.Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    if idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read(); cap.release()
    if not ok:
        sys.exit(f"读不到 {mp4} 第 {idx} 帧")
    return f


def load_reference(root: pathlib.Path, key: str, ep: int, frame: int, median_n: int):
    if median_n <= 1:
        return read_frame(episode_video(root, ep, key), frame), f"ep{ep} f{frame}"
    n_eps = json.loads((root / "meta" / "info.json").read_text())["total_episodes"]
    frames = [read_frame(episode_video(root, e, key), frame) for e in range(min(median_n, n_eps))]
    return np.median(np.stack(frames), axis=0).astype(np.uint8), f"median of {len(frames)} eps f{frame}"


def edges(img: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    return cv2.Canny(g, 60, 140)


def edge_match(ref_e: np.ndarray, live_e: np.ndarray, tol_px: int = 2) -> float:
    """参考边缘像素中, 实时边缘在 tol_px 内也存在的比例。"""
    k = np.ones((2 * tol_px + 1,) * 2, np.uint8)
    near = cv2.dilate(live_e, k) > 0
    n = int((ref_e > 0).sum())
    return float(((ref_e > 0) & near).sum() / n) if n else 0.0


def render(live, ref, mode, alpha, t):
    ref_e, live_e = edges(ref), edges(live)
    score = edge_match(ref_e, live_e)
    if mode == "blend":
        out = cv2.addWeighted(live, 1 - alpha, ref, alpha, 0)
    elif mode == "edges":
        out = live.copy()
        out[live_e > 0] = (0, 0, 255)                      # 实时边缘: 红
        out[ref_e > 0] = (0, 255, 0)                       # 参考边缘: 绿
        out[(ref_e > 0) & (live_e > 0)] = (0, 255, 255)    # 重合: 黄
    elif mode == "split":
        out = live.copy(); w = out.shape[1] // 2
        out[:, :w] = ref[:, :w]
        cv2.line(out, (w, 0), (w, out.shape[0]), (255, 255, 0), 1)
    elif mode == "flicker":
        out = ref.copy() if int(t * 2) % 2 == 0 else live.copy()
    else:  # diff
        d = cv2.absdiff(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
        out = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    return out, score


class LiveSource:
    def __init__(self, serial: str | None, video: str | None):
        self.cap = None; self.pipe = None
        if video:
            self.cap = cv2.VideoCapture(video)
            return
        import pyrealsense2 as rs
        cfg = rs.config(); cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)   # 与 collect_data.DualCamera 一致
        self.pipe = rs.pipeline(); self.pipe.start(cfg)
        for _ in range(8):
            self.pipe.wait_for_frames()

    def read(self) -> np.ndarray:
        if self.cap is not None:
            ok, f = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, f = self.cap.read()
            return f
        fs = self.pipe.wait_for_frames(timeout_ms=2000)
        return np.asanyarray(fs.get_color_frame().get_data())

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.pipe is not None:
            self.pipe.stop()


def main():
    ap = argparse.ArgumentParser(description="环境相机复位: 参考画面叠加实时画面")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="LeRobot 数据集目录(采集原始目录)")
    ap.add_argument("--key", default="image", help="视频流名: image=环境相机")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--median", type=int, default=1, help=">1: 取前 N 条该帧的逐像素中值作参考")
    ap.add_argument("--serial", default=HEAD_CAMERA_SERIAL, help="实时相机 SN")
    ap.add_argument("--live-video", default=None, help="调试: 用 mp4 当实时源, 不开相机")
    ap.add_argument("--mode", choices=MODES, default="edges")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--once", default=None, help="渲染一帧存到该路径后退出(不开窗口)")
    ap.add_argument("--save-dir", default="/tmp/align_camera")
    a = ap.parse_args()

    root = pathlib.Path(a.dataset).expanduser()
    ref, ref_tag = load_reference(root, a.key, a.episode, a.frame, a.median)
    print(f">>> 参考: {root.name} {a.key} {ref_tag}  {ref.shape[1]}x{ref.shape[0]}", flush=True)
    src = LiveSource(None if a.live_video else a.serial, a.live_video)
    mode, alpha = a.mode, a.alpha
    try:
        if a.once:
            out, score = render(src.read(), ref, mode, alpha, 0.0)
            cv2.imwrite(a.once, out); print(f">>> 已存 {a.once}  edge match {score*100:.1f}%"); return
        win = "align camera [m=mode a/d=alpha s=save q=quit]"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1280, 960)
        pathlib.Path(a.save_dir).mkdir(parents=True, exist_ok=True)
        hist = []
        t0 = time.monotonic()
        while True:
            live = src.read()
            out, score = render(live, ref, mode, alpha, time.monotonic() - t0)
            hist.append(score); hist = hist[-15:]
            txt = f"{mode}  alpha={alpha:.1f}  edge match {np.mean(hist)*100:5.1f}%   ref: {root.name} {ref_tag}"
            cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow(win, out)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("m"):
                mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
            elif k == ord("a"):
                alpha = max(0.0, alpha - 0.1)
            elif k == ord("d"):
                alpha = min(1.0, alpha + 0.1)
            elif k == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{a.save_dir}/live_{ts}.png", live); cv2.imwrite(f"{a.save_dir}/overlay_{ts}.png", out)
                print(f">>> 已存 {a.save_dir}/live_{ts}.png (+overlay)  edge match {score*100:.1f}%", flush=True)
    finally:
        src.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
