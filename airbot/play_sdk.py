"""
AirBot Play SDK (based on airbot_py 5.x)
Supports single-arm and dual-arm (two independent AirBot Play) configurations.
"""
import logging
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Optional, Dict, List, Tuple

from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile, State

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLAY_ZERO_JOINT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class RealsenseCamera:
    """
    Realsense camera with iterator interface.
    Returns: (color_bgr, depth_u16)
    """

    def __init__(self, serial_no=None, width=640, height=480, fps=30):
        self.pipeline = None
        self.align = None
        self._intrinsics = None

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            logger.error("No Realsense device detected")
            return

        for i, dev in enumerate(devices):
            sn = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            logger.info(f"Found device {i}: {name} (SN: {sn})")

        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_no:
            config.enable_device(serial_no)

        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        self._intrinsics = {
            'fx': intr.fx, 'fy': intr.fy,
            'cx': intr.ppx, 'cy': intr.ppy,
            'width': intr.width, 'height': intr.height,
            'coeffs': list(intr.coeffs),
        }

        logger.info("Waiting for Realsense to stabilize...")
        for _ in range(60):
            self.pipeline.wait_for_frames()
        logger.info(f"Realsense ready (fx={intr.fx:.1f}, fy={intr.fy:.1f})")

    def __iter__(self):
        return self

    def __next__(self):
        if self.pipeline is None:
            return None, None
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self.align.process(frames)
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        color = np.asanyarray(cf.get_data()) if cf else None
        depth = np.asanyarray(df.get_data()) if df else None
        return color, depth

    def get_intrinsics(self):
        return self._intrinsics

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
            logger.info("Realsense stopped")


class _ArmHandle:
    """
    Wrapper around a single AIRBOTArm instance.
    """

    def __init__(self, robot: AIRBOTArm, name: str):
        self._robot = robot
        self.name = name

    # -- Joint control --
    def set_joint_positions(self, joint_q: List[float], blocking=True) -> bool:
        self._robot.switch_mode(RobotMode.PLANNING_POS)
        result = self._robot.move_to_joint_pos(joint_q, blocking=blocking)
        logger.info(f"[{self.name}] Joint set: {[f'{q:.3f}' for q in joint_q]}")
        return result

    def set_joint_waypoints(self, waypoints: List[List[float]], blocking=True) -> bool:
        self._robot.switch_mode(RobotMode.PLANNING_WAYPOINTS)
        return self._robot.move_with_joint_waypoints(waypoints, blocking=blocking)

    def set_end_pose(self, position: List[float], orientation: List[float],
                     blocking=True) -> bool:
        self._robot.switch_mode(RobotMode.PLANNING_POS)
        result = self._robot.move_to_cart_pose(
            [list(position), list(orientation)], blocking=blocking
        )
        logger.info(f"[{self.name}] End pose: pos={[f'{p:.4f}' for p in position]}")
        return result

    def set_end_pose_waypoints(self, waypoints: List[List[List[float]]],
                               blocking=True) -> bool:
        self._robot.switch_mode(RobotMode.PLANNING_WAYPOINTS_PATH)
        return self._robot.move_with_cart_waypoints(waypoints, blocking=blocking)

    # -- Servo mode --
    def servo_joint_pos(self, joint_q: List[float]) -> None:
        self._robot.servo_joint_pos(joint_q)

    def servo_cart_pose(self, position: List[float], orientation: List[float]) -> None:
        self._robot.servo_cart_pose([list(position), list(orientation)])

    def servo_cart_twist(self, linear: List[float], angular: List[float]) -> None:
        self._robot.servo_cart_twist([list(linear), list(angular)])

    def switch_mode(self, mode: RobotMode) -> bool:
        return self._robot.switch_mode(mode)

    # -- Gripper --
    def set_gripper(self, position: float = 1.0) -> bool:
        result = self._robot.move_eef_pos(position)
        logger.info(f"[{self.name}] Gripper: {position:.2f}")
        return result

    def open_gripper(self) -> bool:
        return self.set_gripper(1.0)

    def close_gripper(self) -> bool:
        return self.set_gripper(0.0)

    # -- State --
    def get_joint_pos(self) -> Optional[List[float]]:
        return self._robot.get_joint_pos()

    def get_joint_vel(self) -> Optional[List[float]]:
        return self._robot.get_joint_vel()

    def get_joint_eff(self) -> Optional[List[float]]:
        return self._robot.get_joint_eff()

    def get_end_pose(self) -> Optional[List[List[float]]]:
        return self._robot.get_end_pose()

    def get_eef_pos(self) -> Optional[List[float]]:
        return self._robot.get_eef_pos()

    def get_state(self) -> State:
        return self._robot.get_state()

    def get_control_mode(self) -> Optional[RobotMode]:
        return self._robot.get_control_mode()

    def get_product_info(self) -> Optional[Dict]:
        return self._robot.get_product_info()

    def reset_to_zero(self) -> bool:
        return self.set_joint_positions(PLAY_ZERO_JOINT, blocking=True)


class PlayRealRobot:
    """
    AirBot Play robot controller.

    Single-arm:
        robot = PlayRealRobot(port=50051)

    Dual-arm (two independent AirBot Play):
        robot = PlayRealRobot(left_port=50051, right_port=50052)
        robot.left.open_gripper()
        robot.right.close_gripper()

    With cameras:
        robot = PlayRealRobot(
            left_port=50051, right_port=50052,
            head_camera_serial="xxx",
            left_wrist_camera_serial="yyy",
            right_wrist_camera_serial="zzz",
        )
        color, depth = next(robot.head_camera)
        color, depth = next(robot.left_wrist_camera)
    """

    def __init__(self, url: str = "localhost",
                 port: int = None,
                 left_port: int = None,
                 right_port: int = None,
                 head_camera_serial: str = None,
                 left_wrist_camera_serial: str = None,
                 right_wrist_camera_serial: str = None,
                 camera_width: int = 640,
                 camera_height: int = 480,
                 camera_fps: int = 30,
                 enable_cameras: bool = True):
        """
        Args:
            url: gRPC server address
            port: single-arm gRPC port
            left_port: left arm gRPC port (dual-arm mode)
            right_port: right arm gRPC port (dual-arm mode)
            head_camera_serial: fixed head Realsense serial number
            left_wrist_camera_serial: left wrist Realsense serial number
            right_wrist_camera_serial: right wrist Realsense serial number
        """
        self.dual_arm = (left_port is not None and right_port is not None)
        self.left: Optional[_ArmHandle] = None
        self.right: Optional[_ArmHandle] = None

        if self.dual_arm:
            left_robot = AIRBOTArm(url=url, port=left_port)
            if not left_robot.connect():
                raise RuntimeError(f"Cannot connect to left arm ({url}:{left_port})")
            left_robot.switch_mode(RobotMode.PLANNING_POS)
            self.left = _ArmHandle(left_robot, "left")
            logger.info(f"Left arm connected ({url}:{left_port})")

            right_robot = AIRBOTArm(url=url, port=right_port)
            if not right_robot.connect():
                raise RuntimeError(f"Cannot connect to right arm ({url}:{right_port})")
            right_robot.switch_mode(RobotMode.PLANNING_POS)
            self.right = _ArmHandle(right_robot, "right")
            logger.info(f"Right arm connected ({url}:{right_port})")
        else:
            arm_port = port or 50051
            robot = AIRBOTArm(url=url, port=arm_port)
            if not robot.connect():
                raise RuntimeError(f"Cannot connect to arm ({url}:{arm_port})")
            robot.switch_mode(RobotMode.PLANNING_POS)
            self.left = _ArmHandle(robot, "arm")
            logger.info(f"Arm connected ({url}:{arm_port})")

        # Cameras: head (fixed), left wrist, right wrist
        self.head_camera: Optional[RealsenseCamera] = None
        self.left_wrist_camera: Optional[RealsenseCamera] = None
        self.right_wrist_camera: Optional[RealsenseCamera] = None

        if enable_cameras:
            if head_camera_serial:
                self.head_camera = RealsenseCamera(
                    serial_no=head_camera_serial,
                    width=camera_width, height=camera_height, fps=camera_fps,
                )
            if left_wrist_camera_serial:
                self.left_wrist_camera = RealsenseCamera(
                    serial_no=left_wrist_camera_serial,
                    width=camera_width, height=camera_height, fps=camera_fps,
                )
            if right_wrist_camera_serial:
                self.right_wrist_camera = RealsenseCamera(
                    serial_no=right_wrist_camera_serial,
                    width=camera_width, height=camera_height, fps=camera_fps,
                )

    # ================================================================
    #  Convenience methods (delegate to left/single arm)
    # ================================================================
    def set_joint_positions(self, joint_q: List[float], blocking=True) -> bool:
        return self.left.set_joint_positions(joint_q, blocking=blocking)

    def set_joint_waypoints(self, waypoints: List[List[float]], blocking=True) -> bool:
        return self.left.set_joint_waypoints(waypoints, blocking=blocking)

    def set_end_pose(self, position: List[float], orientation: List[float],
                     blocking=True) -> bool:
        return self.left.set_end_pose(position, orientation, blocking=blocking)

    def set_end_pose_waypoints(self, waypoints, blocking=True) -> bool:
        return self.left.set_end_pose_waypoints(waypoints, blocking=blocking)

    def servo_joint_pos(self, joint_q: List[float]) -> None:
        self.left.servo_joint_pos(joint_q)

    def servo_cart_pose(self, position: List[float], orientation: List[float]) -> None:
        self.left.servo_cart_pose(position, orientation)

    def servo_cart_twist(self, linear: List[float], angular: List[float]) -> None:
        self.left.servo_cart_twist(linear, angular)

    def switch_mode(self, mode: RobotMode) -> bool:
        return self.left.switch_mode(mode)

    # ================================================================
    #  Gripper
    # ================================================================
    def set_gripper(self, left_pos: float = None, right_pos: float = None,
                    position: float = None) -> bool:
        """
        Single-arm: set_gripper(position=1.0)
        Dual-arm:   set_gripper(left_pos=1.0, right_pos=0.0)
        """
        result = True
        if position is not None:
            result = self.left.set_gripper(position)
        else:
            if left_pos is not None and self.left:
                result = self.left.set_gripper(left_pos) and result
            if right_pos is not None and self.right:
                result = self.right.set_gripper(right_pos) and result
        return result

    def open_gripper(self, left=True, right=True) -> bool:
        result = True
        if left and self.left:
            result = self.left.open_gripper() and result
        if right and self.right:
            result = self.right.open_gripper() and result
        return result

    def close_gripper(self, left=True, right=True) -> bool:
        result = True
        if left and self.left:
            result = self.left.close_gripper() and result
        if right and self.right:
            result = self.right.close_gripper() and result
        return result

    # ================================================================
    #  State
    # ================================================================
    def get_joint_positions(self) -> Optional[Tuple[List[str], List[float]]]:
        if self.dual_arm:
            lq = self.left.get_joint_pos() or []
            rq = self.right.get_joint_pos() or []
            names = [f"left_joint_{i}" for i in range(len(lq))] + \
                    [f"right_joint_{i}" for i in range(len(rq))]
            return names, list(lq) + list(rq)
        else:
            q = self.left.get_joint_pos()
            if q is None:
                return None
            names = [f"joint_{i}" for i in range(len(q))]
            return names, list(q)

    def get_joint_q(self) -> Optional[List[float]]:
        return self.left.get_joint_pos()

    def get_joint_velocities(self) -> Optional[List[float]]:
        return self.left.get_joint_vel()

    def get_joint_efforts(self) -> Optional[Dict[str, float]]:
        if self.dual_arm:
            result = {}
            lt = self.left.get_joint_eff() or []
            rt = self.right.get_joint_eff() or []
            for i, v in enumerate(lt):
                result[f"left_joint_{i}"] = v
            for i, v in enumerate(rt):
                result[f"right_joint_{i}"] = v
            return result if result else None
        else:
            t = self.left.get_joint_eff()
            if t is None:
                return None
            return {f"joint_{i}": v for i, v in enumerate(t)}

    def get_arm_end_poses(self) -> Optional[Dict[str, Dict]]:
        """
        Single-arm: {'left_arm': {'position': [...], 'orientation': [...]}}
        Dual-arm:   {'left_arm': {...}, 'right_arm': {...}}
        """
        result = {}
        lp = self.left.get_end_pose()
        if lp:
            result['left_arm'] = {'position': lp[0], 'orientation': lp[1]}
        if self.dual_arm and self.right:
            rp = self.right.get_end_pose()
            if rp:
                result['right_arm'] = {'position': rp[0], 'orientation': rp[1]}
        return result if result else None

    def get_end_pose(self) -> Optional[Tuple[List[float], List[float]]]:
        pose = self.left.get_end_pose()
        if pose is None:
            return None
        return pose[0], pose[1]

    def get_gripper_state(self, arm: str = "left") -> Optional[float]:
        handle = self.left if arm == "left" else self.right
        if handle is None:
            return None
        eef = handle.get_eef_pos()
        if eef is None or len(eef) == 0:
            return None
        return eef[0]

    def get_robot_state(self) -> Optional[Dict]:
        state = {
            'left': {
                'joint_q': self.left.get_joint_pos(),
                'joint_v': self.left.get_joint_vel(),
                'joint_t': self.left.get_joint_eff(),
                'pose': self.left.get_end_pose(),
                'eef': self.left.get_eef_pos(),
                'state': self.left.get_state(),
                'mode': self.left.get_control_mode(),
            },
        }
        if self.dual_arm and self.right:
            state['right'] = {
                'joint_q': self.right.get_joint_pos(),
                'joint_v': self.right.get_joint_vel(),
                'joint_t': self.right.get_joint_eff(),
                'pose': self.right.get_end_pose(),
                'eef': self.right.get_eef_pos(),
                'state': self.right.get_state(),
                'mode': self.right.get_control_mode(),
            }
        return state

    def get_product_info(self, arm: str = "left") -> Optional[Dict]:
        handle = self.left if arm == "left" else self.right
        if handle is None:
            return None
        return handle.get_product_info()

    # ================================================================
    #  Speed profile
    # ================================================================
    def set_speed_profile(self, profile: str = "default") -> None:
        mapping = {
            'default': SpeedProfile.DEFAULT,
            'slow': SpeedProfile.SLOW,
            'fast': SpeedProfile.FAST,
        }
        if profile.lower() not in mapping:
            logger.error(f"Unknown speed profile: {profile}, options: default/slow/fast")
            return
        sp = mapping[profile.lower()]
        self.left._robot.set_speed_profile(sp)
        if self.dual_arm and self.right:
            self.right._robot.set_speed_profile(sp)
        logger.info(f"Speed profile: {profile}")

    # ================================================================
    #  Preset poses
    # ================================================================
    def reset_to_zero(self) -> bool:
        result = self.left.reset_to_zero()
        if self.dual_arm and self.right:
            result = self.right.reset_to_zero() and result
        return result

    def reset_to_start(self) -> bool:
        return self.reset_to_zero()

    # ================================================================
    #  Cleanup
    # ================================================================
    def shutdown(self):
        if self.head_camera:
            self.head_camera.stop()
        if self.left_wrist_camera:
            self.left_wrist_camera.stop()
        if self.right_wrist_camera:
            self.right_wrist_camera.stop()
        if self.left:
            self.left._robot.disconnect()
        if self.right:
            self.right._robot.disconnect()
        logger.info("PlayRealRobot shutdown")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# ========== Test ==========
if __name__ == "__main__":
    # Single-arm:
    #   with PlayRealRobot(port=50051) as robot:
    #
    # Dual-arm:
    #   with PlayRealRobot(left_port=50051, right_port=50052,
    #                      head_camera_serial="xxx",
    #                      left_wrist_camera_serial="yyy",
    #                      right_wrist_camera_serial="zzz") as robot:

    with PlayRealRobot(port=50051, enable_cameras=False) as robot:
        print("=" * 50)
        info = robot.get_product_info()
        if info:
            for k, v in info.items():
                print(f"  {k}: {v}")

        print("\n" + "=" * 50)
        state = robot.get_robot_state()
        if state:
            left = state['left']
            print(f"Joint pos: {left['joint_q']}")
            print(f"Joint eff: {left['joint_t']}")
            print(f"End pose:  {left['pose']}")
            print(f"EEF:       {left['eef']}")
            print(f"State:     {left['state']}")
            print(f"Mode:      {left['mode']}")

        print("\n" + "=" * 50)
        poses = robot.get_arm_end_poses()
        if poses:
            print(f"Position:    {poses['left_arm']['position']}")
            print(f"Orientation: {poses['left_arm']['orientation']}")