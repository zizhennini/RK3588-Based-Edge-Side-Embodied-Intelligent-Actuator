"""SO-ARM101 控制封装 — LeRobot 官方 RobotKinematics + Feetech 串口"""
import struct
import serial
import numpy as np


class ArmController:
    """使用 LeRobot 官方 IK + Feetech 串口控制"""

    JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 1000000,
        urdf_path: str = "./models/so101_urdf/so101_new_calib.urdf",
    ):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.ik = self._init_ik(urdf_path)

    def _init_ik(self, urdf_path: str):
        try:
            from .kinematics import RobotKinematics
            kin = RobotKinematics(urdf_path, target_frame_name="gripper_frame_link")
            print(f"  IK: 官方 placo (URDF={urdf_path})")
            return kin
        except Exception as e:
            print(f"  IK: 官方不可用({e})，降级为几何 IK")

        # 降级：几何 IK
        class GeoIK:
            def inverse_kinematics(self, current_joint_pos, desired_ee_pose):
                x, y, z = desired_ee_pose[:3, 3]
                L1, L2, L3 = 0.120, 0.150, 0.180
                theta1 = np.arctan2(y, x)
                r = np.sqrt(x ** 2 + y ** 2)
                d = np.sqrt((r - 0.025) ** 2 + (z - L1) ** 2)
                cos_t3 = (d ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
                cos_t3 = np.clip(cos_t3, -1.0, 1.0)
                theta3 = np.arccos(cos_t3)
                theta2 = np.arctan2(z - L1, r - 0.025) - np.arctan2(
                    L3 * np.sin(theta3), L2 + L3 * np.cos(theta3)
                )
                theta4, theta5, theta6 = 0.0, -theta2 - theta3 + np.pi / 2, 0.0
                q = np.deg2rad([theta1, theta2, theta3, theta4, theta5, theta6])
                if len(current_joint_pos) > 6:
                    result = np.zeros_like(current_joint_pos)
                    result[:6] = np.rad2deg(q)
                    result[6:] = current_joint_pos[6:]
                    return result
                return np.rad2deg(q)

            def forward_kinematics(self, joint_pos_deg):
                return np.eye(4)

        return GeoIK()

    def move_to(self, x: float, y: float, z: float):
        current = self._read_current_pos()
        t_des = np.eye(4)
        t_des[:3, 3] = [x, y, z]
        angles_deg = self.ik.inverse_kinematics(current, t_des)
        for sid in range(1, 7):
            self._write_angle(sid, float(np.deg2rad(angles_deg[sid - 1])))

    def _read_current_pos(self) -> np.ndarray:
        return np.zeros(len(self.JOINT_NAMES))

    def _write_angle(self, sid: int, rad: float):
        pulse = int(500 + rad / np.pi * 500)
        pulse = max(0, min(1000, pulse))
        cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, sid, 5, 0x03, 0x2A, pulse)
        checksum = (~sum(cmd[2:]) & 0xFF)
        self.ser.write(cmd + struct.pack("<B", checksum))

    def gripper(self, open: bool):
        pulse = 400 if open else 200
        cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, 6, 5, 0x03, 0x2A, pulse)
        cks = (~sum(cmd[2:]) & 0xFF)
        self.ser.write(cmd + struct.pack("<B", cks))

    def close(self):
        self.ser.close()
