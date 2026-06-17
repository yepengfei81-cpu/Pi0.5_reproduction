#!/usr/bin/env python3
"""
运动学轴向验证：确认 get_end_pose 的工具系轴向是不是 +X=伸出、+Y=开合、+Z=上,
从而把描述符的 --tcp-rpy 约定【数值钉死】(肉眼投影只能粗看, 这里是确定性的)。

两种用法:
  (1) 只读(安全, 默认): 打印当前 get_end_pose 的工具系三轴在 base 系下的方向。
      对照你已知的 base 朝向(AIRBOT: +X 前/远离底座, +Z 上)即可判断工具 +X/+Z 指哪。
  (2) --move: 让末端沿【工具系】+X / +Z 各点动 5cm 再回来, 你直接眼看夹爪往哪动
      (最不含糊)。手放急停旁。

用法:
  python gripper_geom/axis_check.py                 # 只读
  python gripper_geom/axis_check.py --move          # 带点动(手放急停旁)
"""
import argparse
import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "airbot"))
sys.path.insert(0, str(_ROOT / "umi"))
from play_sdk import PlayRealRobot, RobotMode      # noqa: E402
from umi_to_lerobot import quat_to_rot             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=50050)
    ap.add_argument("--move", action="store_true", help="沿工具系 +X/+Z 各点动 5cm 验证(手放急停旁)")
    ap.add_argument("--d", type=float, default=0.05, help="点动距离(米)")
    args = ap.parse_args()

    robot = PlayRealRobot(port=args.port, enable_cameras=False)
    p, q = robot.get_end_pose()
    p = np.asarray(p, float)
    R = quat_to_rot(*q)        # tool->base, 列 = 工具系三轴在 base 下的方向
    print(f">>> 当前 TCP 位置(base, m): {p.round(3).tolist()}")
    print(">>> 工具系三轴在 base 系下的方向(单位向量):")
    print(f"      工具 +X = {R[:,0].round(3).tolist()}   <- 若指向夹爪伸出方向, 则 +X=伸出")
    print(f"      工具 +Y = {R[:,1].round(3).tolist()}   <- 应是开合方向")
    print(f"      工具 +Z = {R[:,2].round(3).tolist()}   <- 若朝上(base z 分量为正), 则 +Z=上")
    print(">>> 提醒: AIRBOT base 通常 +X=前/远离底座, +Z=上。据此判断工具轴指向。")

    if args.move:
        print("\n>>> 点动模式: 将沿【工具系】+X、+Z 各走 %.0fcm 再回来。手放急停旁, 回车开始..." % (args.d*100))
        input()
        robot.set_speed_profile("default")
        robot.switch_mode(RobotMode.SERVO_CART_POSE)
        time.sleep(0.2)
        robot.servo_cart_pose(list(p), list(q)); time.sleep(0.3)   # 播种保持点

        def stream(a_from, a_to, n=60):
            # servo 模式需高频连续喂目标; 单发到不了, 这里密集插值
            for a in np.linspace(0.0, 1.0, n):
                cur = (1 - a) * a_from + a * a_to
                robot.servo_cart_pose(cur.tolist(), list(q))
                time.sleep(0.02)

        def jog(axis_name, vec):
            tgt = p + R @ np.asarray(vec) * args.d                 # 沿工具系方向, base 目标
            print(f"    -> 沿工具 {axis_name} 走 {args.d*100:.0f}cm, 看夹爪往哪动...")
            stream(p, tgt)                                          # 去
            time.sleep(0.4)
            stream(tgt, p)                                          # 回原位
            input(f"    观察完 {axis_name} 了吗? 回车继续...")

        jog("+X", [1, 0, 0])
        jog("+Z", [0, 0, 1])
        robot.switch_mode(RobotMode.PLANNING_POS)

    robot.shutdown()
    print(">>> 完成。把工具 +X / +Z 的实际指向告诉我, 即可确定 --tcp-rpy 是否就是 0 0 180。")


if __name__ == "__main__":
    main()
