#!/usr/bin/env python3
"""临时夹爪开合标定脚本（键盘控制版）。

机械臂示教模式不能手拖, 所以用键盘小步命令夹爪开/合, 屏幕显示实际电机读数
get_eef_pos(米)。用来量 GET 夹爪的 close/open:

  - 按 'o' 让夹爪完全张开, 读 get_eef_pos -> open
  - 反复按 '-' 一点点闭合, 肉眼看 GET 指尖【刚好接触/闭合】时, 读 get_eef_pos -> close
    (别再往下压, GET 会过挤压堵转)
  把两个数填进 ../gripper_params.py 的 "get": {"close": ..., "open": ...}

命令(每条回车):  +=开大一步  -=闭合一步  数字=直接到该宽度(米)  o=全开  q=退出

⚠️ PLANNING_POS 模式下只命令夹爪, 手臂不动。靠近闭合时小步走、盯着指尖。

用法:
  python teach_probe.py            # 默认 follow 臂 port 50050, 步长 2mm
  python teach_probe.py --step 0.001
"""
import argparse
import time

FOLLOW_PORT = 50050
GRIP_MAX_SAFE = 0.085   # 命令宽度上限(m), 防越界


def main():
    ap = argparse.ArgumentParser(description="GET 夹爪开合标定(键盘控制)")
    ap.add_argument("--url", default="localhost")
    ap.add_argument("--port", type=int, default=FOLLOW_PORT)
    ap.add_argument("--step", type=float, default=0.002, help="每次 +/- 的开合步长(m)")
    args = ap.parse_args()

    from airbot_py.arm import AIRBOTArm, RobotMode
    arm = AIRBOTArm(url=args.url, port=args.port)
    if not arm.connect():
        raise SystemExit(f"无法连接 follow 臂 (port={args.port})")
    arm.switch_mode(RobotMode.PLANNING_POS)

    def read_grip():
        eef = arm.get_eef_pos()
        return float(eef[0]) if eef else float("nan")

    target = read_grip()
    if target != target or target <= 0:
        target = 0.05
    print(">>> PLANNING_POS, 键盘命令夹爪(手臂不动)。")
    print(">>> 'o'全开读 open; 反复 '-' 闭到 GET 指尖刚接触, 读 get_eef_pos = close")
    print(">>> 命令: +  -  数字(米)  o  q\n")
    try:
        while True:
            cmd = input(f"[target={target:.4f}m] +/-/数字/o/q: ").strip().lower()
            if cmd in ("q", "quit"):
                break
            elif cmd == "o":
                target = GRIP_MAX_SAFE
            elif cmd in ("+", ""):
                target += args.step
            elif cmd == "-":
                target -= args.step
            else:
                try:
                    target = float(cmd)
                except ValueError:
                    print("  无法识别, 用 + / - / 数字 / o / q"); continue
            target = float(min(max(target, 0.0), GRIP_MAX_SAFE))
            arm.move_eef_pos([target], blocking=True)
            time.sleep(0.2)
            g = read_grip()
            print(f"  下发 target={target:.4f}m  ->  实读 get_eef_pos={g:+.4f}m "
                  f"(GET 刚闭合时记下这个数=close; 全开时=open)")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
