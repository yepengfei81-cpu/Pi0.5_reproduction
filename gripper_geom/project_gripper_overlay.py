#!/usr/bin/env python3
"""
真机验证：把 TCP 系下的夹爪点云(parallel.npy)投影到手眼 D405 实时画面上, 拖滑条调
T_cam<-cloud(6 自由度: 平移 mm + 旋转 deg)直到投影点与真实夹爪重合。

重合即同时验证：① 点云几何对; ② URDF→真机朝向对; ③ 相机外参被标出来了。
这个 6-DOF 变换吸收了"URDF-EE 朝向 vs 真机 get_end_pose 朝向"和"TCP->相机外参"两件事。

按键: s=打印当前变换(记下来), q/Esc=退出。

用法:
  python gripper_geom/project_gripper_overlay.py --npy gripper_geom/parallel.npy \
      --serial 230422271433
"""
import argparse
import json
import pathlib

import cv2
import numpy as np


def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r); cp, sp = np.cos(p), np.sin(p); cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default="gripper_geom/parallel.npy")
    ap.add_argument("--serial", default="230422271433", help="手眼 D405 序列号")
    ap.add_argument("--max-pts", type=int, default=1500)
    args = ap.parse_args()

    d = np.load(args.npy, allow_pickle=True).item()
    pts = d["points"].astype(np.float64)
    if len(pts) > args.max_pts:
        pts = pts[np.random.default_rng(0).choice(len(pts), args.max_pts, replace=False)]
    anchors = d.get("anchors", {})
    print(f">>> 点云 {pts.shape}  frame={d.get('frame')}  anchors={list(anchors)}")

    import pyrealsense2 as rs
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    print(f">>> 手眼内参 fx={intr.fx:.1f} fy={intr.fy:.1f} ppx={intr.ppx:.1f} ppy={intr.ppy:.1f}")

    win = "gripper overlay [s=print, q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 960, 720)
    # 滑条: 平移 ±300mm(偏移300), 旋转 0..360deg(偏移180)
    for n, v in [("tx", 300), ("ty", 300), ("tz", 300)]:
        cv2.createTrackbar(n, win, v, 600, lambda x: None)
    for n in ("rx", "ry", "rz"):
        cv2.createTrackbar(n, win, 180, 360, lambda x: None)

    def get_T():
        tx = (cv2.getTrackbarPos("tx", win) - 300) / 1000.0
        ty = (cv2.getTrackbarPos("ty", win) - 300) / 1000.0
        tz = (cv2.getTrackbarPos("tz", win) - 300) / 1000.0
        rx = np.radians(cv2.getTrackbarPos("rx", win) - 180)
        ry = np.radians(cv2.getTrackbarPos("ry", win) - 180)
        rz = np.radians(cv2.getTrackbarPos("rz", win) - 180)
        return rpy_to_R(rx, ry, rz), np.array([tx, ty, tz])

    try:
        while True:
            fr = pipe.wait_for_frames(); c = fr.get_color_frame()
            if not c:
                continue
            img = np.asanyarray(c.get_data()).copy()
            R, t = get_T()
            Pc = pts @ R.T + t                       # cloud -> camera
            z = Pc[:, 2]
            m = z > 1e-3
            uv = (K @ Pc[m].T).T
            uv = uv[:, :2] / uv[:, 2:3]
            zz = z[m]
            zr = (zz - zz.min()) / (zz.ptp() + 1e-6)
            for (u, v), s in zip(uv, zr):
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    col = (int(255 * (1 - s)), 80, int(255 * s))   # 近红远蓝
                    cv2.circle(img, (int(u), int(v)), 1, col, -1)
            # 投影锚点(大星)
            for name, a in anchors.items():
                ac = np.asarray(a) @ R.T + t
                if ac[2] > 1e-3:
                    p = K @ ac; p = p[:2] / p[2]
                    if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
                        cv2.drawMarker(img, (int(p[0]), int(p[1])), (0, 255, 0),
                                       cv2.MARKER_STAR, 16, 2)
            cv2.imshow(win, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                rx = cv2.getTrackbarPos("rx", win) - 180
                ry = cv2.getTrackbarPos("ry", win) - 180
                rz = cv2.getTrackbarPos("rz", win) - 180
                print(f">>> T_cam<-cloud: t(mm)=[{(cv2.getTrackbarPos('tx',win)-300)},"
                      f"{(cv2.getTrackbarPos('ty',win)-300)},{(cv2.getTrackbarPos('tz',win)-300)}]  "
                      f"rpy(deg)=[{rx},{ry},{rz}]", flush=True)
    finally:
        pipe.stop(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
