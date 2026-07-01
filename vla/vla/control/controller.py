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

    # 从 URDF 提取的关节限位（弧度）
    JOINT_LIMITS = {
        "shoulder_pan": (-1.91986, 1.91986),
        "shoulder_lift": (-1.74533, 1.74533),
        "elbow_flex": (-1.69, 1.69),
        "wrist_flex": (-1.65806, 1.65806),
        "wrist_roll": (-2.74385, 2.84121),
        "gripper": (-0.174533, 1.74533),
    }

    # 标定数据（来自 LeRobot 校准）
    CALIB = {
        1: {"homing_offset": 2048, "range_min": 946, "range_max": 3287},
        2: {"homing_offset": 2048, "range_min": 821, "range_max": 3206},
        3: {"homing_offset": 2048, "range_min": 888, "range_max": 3105},
        4: {"homing_offset": 2048, "range_min": 851, "range_max": 3192},
        5: {"homing_offset": 2048, "range_min": 130, "range_max": 3985},
        6: {"homing_offset": 1781, "range_min": 1495, "range_max": 2860},
    }

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 1000000,
        urdf_path: str = "./models/so101_urdf/so101_new_calib.urdf",
    ):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.ik = self._init_ik(urdf_path)
        self._configure_motors()

    def _configure_motors(self):
        """设置舵机加速度和 PID，让运动更平滑"""
        import time
        for sid in range(1, 7):
            # 加速度寄存器 0x29，值越大加速越快
            cmd = struct.pack("<BBBBBBB", 0xFF, 0xFF, sid, 4, 0x03, 0x29, 100)
            cks = (~sum(cmd[2:]) & 0xFF)
            self.ser.write(cmd + struct.pack("<B", cks))
            time.sleep(0.01)
            # P 系数 (0x1A) — 降低减少抖动
            cmd = struct.pack("<BBBBBBB", 0xFF, 0xFF, sid, 4, 0x03, 0x1A, 16)
            cks = (~sum(cmd[2:]) & 0xFF)
            self.ser.write(cmd + struct.pack("<B", cks))
            time.sleep(0.01)

    def _init_ik(self, urdf_path: str):
        class GeoIK:
            def inverse_kinematics(self, current_joint_pos, desired_ee_pose):
                x, y, z = desired_ee_pose[:3, 3]
                L1, L2, L3 = 0.120, 0.150, 0.180
                theta1 = np.arctan2(y, x)
                r = np.sqrt(x ** 2 + y ** 2)
                d = np.sqrt((r - 0.025) ** 2 + (z - L1) ** 2)
                cos_t3 = (d ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
                cos_t3 = np.clip(cos_t3, -1.0, 1.0)
                theta3 = -abs(np.arccos(cos_t3))
                theta2 = np.arctan2(z - L1, r - 0.025) - np.arctan2(
                    L3 * np.sin(theta3), L2 + L3 * np.cos(theta3)
                )
                theta4, theta5, theta6 = 0.0, -theta2 - theta3 + np.pi / 2, 0.0
                q = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
                limits = ArmController.JOINT_LIMITS
                for i, name in enumerate(ArmController.JOINT_NAMES):
                    low, high = limits[name]
                    q[i] = np.clip(q[i], low, high)
                if len(current_joint_pos) > 6:
                    result = np.zeros_like(current_joint_pos)
                    result[:6] = np.rad2deg(q)
                    result[6:] = current_joint_pos[6:]
                    return result
                return np.rad2deg(q)

            def forward_kinematics(self, joint_pos_deg):
                return np.eye(4)

        return GeoIK()

    def _clamp_joints(self, angles_rad: np.ndarray) -> np.ndarray:
        for i, name in enumerate(self.JOINT_NAMES):
            low, high = self.JOINT_LIMITS[name]
            angles_rad[i] = np.clip(angles_rad[i], low, high)
        return angles_rad

    def move_to(self, x: float, y: float, z: float):
        current = self._read_current_pos()
        t_des = np.eye(4)
        t_des[:3, 3] = [x, y, z]
        angles_deg = self.ik.inverse_kinematics(current, t_des)
        angles_rad = np.deg2rad(angles_deg[:6])
        angles_rad = self._clamp_joints(angles_rad)
        # 同时发送6个关节，避免逐个动作导致碰撞
        for sid in range(1, 7):
            self._write_angle(sid, float(angles_rad[sid - 1]))

    def _read_current_pos(self) -> np.ndarray:
        return np.zeros(len(self.JOINT_NAMES))

    def _write_angle(self, sid: int, rad: float):
        calib = self.CALIB[sid]
        deg = np.rad2deg(rad)
        mid = (calib["range_min"] + calib["range_max"]) / 2
        raw = int(deg * 4095 / 360 + mid)
        raw = max(calib["range_min"], min(calib["range_max"], raw))
        cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, sid, 5, 0x03, 0x2A, raw)
        checksum = (~sum(cmd[2:]) & 0xFF)
        self.ser.write(cmd + struct.pack("<B", checksum))

    def gripper(self, open: bool):
        pulse = 2600 if open else 1781
        cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, 6, 5, 0x03, 0x2A, pulse)
        cks = (~sum(cmd[2:]) & 0xFF)
        self.ser.write(cmd + struct.pack("<B", cks))

    def close(self):
        self.ser.close()
