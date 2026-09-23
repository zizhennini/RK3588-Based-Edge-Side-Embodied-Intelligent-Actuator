"""SO-ARM101 完整 6DOF 运动学（XLeRobot 偏移补偿）

参考: XLeRobot SO101Robot.py 的 IK 偏移补偿方案
URDF: models/so101_urdf/so101_new_calib.urdf
"""
import math
import numpy as np
from typing import Optional


class Kinematics:
    """SO-ARM101 完整 6DOF 运动学"""

    # URDF 精确参数（来自 XLeRobot + URDF joint origin）
    L1 = 0.1159   # 上臂长度 (elbow_flex origin.x = 0.11257, 取 XLeRobot 值)
    L2 = 0.1350   # 前臂长度 (wrist_flex origin.x = 0.1349, 取 XLeRobot 值)
    BASE_HEIGHT = 0.0624  # 基座高度 (shoulder_pan origin.z)

    # 机械偏移补偿（来自 URDF joint origin 的 y/z 偏移）
    THETA1_OFFSET = math.atan2(0.028, 0.11257)    # ~14.0 deg 肩部偏移
    THETA2_OFFSET = math.atan2(0.0052, 0.1349) + math.atan2(0.028, 0.11257)  # ~16.2 deg 肘部偏移

    # 关节限位 (rad)
    JOINT_LIMITS = {
        1: (-math.pi, math.pi),        # shoulder_pan
        2: (-0.1, 3.45),               # shoulder_lift
        3: (-0.2, math.pi),            # elbow_flex
        4: (-math.pi, math.pi),        # wrist_flex
        5: (-math.pi, math.pi),        # wrist_roll
        6: (0.0, 1.0),                 # gripper (归一化)
    }

    def inverse_kinematics(
        self,
        target_xyz: np.ndarray,
        current_angles: np.ndarray,
        wrist_roll_rad: Optional[float] = None,
    ) -> np.ndarray:
        """完整 6DOF IK，返回 6 维关节角度 (rad)

        Args:
            target_xyz: 目标笛卡尔坐标 (x, y, z)，单位米
            current_angles: 当前 6 维关节角度 (rad)
            wrist_roll_rad: 可选的 wrist_roll 角度 (rad)，None 则保持当前值

        Returns:
            6 维关节角度 np.ndarray (rad)，已钳制到限位
        """
        x, y, z = target_xyz[0], target_xyz[1], target_xyz[2]

        # Joint 1: shoulder_pan = atan2(y, x)
        j1 = math.atan2(y, x)

        # 平面距离和高度（相对于基座）
        r = math.sqrt(x**2 + y**2)
        h = z - self.BASE_HEIGHT

        # Joint 2-3: 2DOF 平面 IK + 偏移补偿（返回舵机角度 deg）
        j2_deg, j3_deg = self._ik_2dof(r, h)

        # 舵机角度 → 数学角度 (rad)
        # 从 xlerobot_fk 反推: j2_rad = radians(90 - j2_deg) = theta1 + offset
        # 即 theta1 = pi/2 - radians(j2_deg)
        theta1 = math.pi / 2 - math.radians(j2_deg)
        # j3_rad = radians(j3_deg + 90) = theta2 + offset
        # 即 theta2 = radians(j3_deg + 90) - pi  (肘部弯曲角，减去 pi 保持与 URDF 一致)
        theta2 = math.radians(j3_deg + 90)

        # Joint 4 (wrist_flex): 使用指定值或保持当前
        j4 = wrist_roll_rad if wrist_roll_rad is not None else current_angles[3]

        # Joint 5 (wrist_roll): 保持当前值
        j5 = current_angles[4]

        # Joint 6 (gripper): 保持当前值
        j6 = current_angles[5]

        result = np.array([j1, theta1, theta2, j4, j5, j6])
        return self.clamp_joint(result)

    def _ik_2dof(self, r: float, h: float) -> tuple[float, float]:
        """2DOF 平面 IK + 偏移补偿（从 test_xlerobot_ik.py 移植）

        Args:
            r: 水平距离 (m)
            h: 垂直高度 (m)，相对于基座顶部

        Returns:
            (j2_deg, j3_deg): 舵机角度 (deg)
        """
        # 工作空间钳制
        r_max = self.L1 + self.L2
        d = math.sqrt(r**2 + h**2)
        if d > r_max:
            scale = r_max / d
            r *= scale
            h *= scale
            d = r_max
        r_min = abs(self.L1 - self.L2)
        if d < r_min and d > 0:
            scale = r_min / d
            r *= scale
            h *= scale
            d = r_min

        # 余弦定理求 elbow（参考 test_xlerobot_ik.py 精确公式）
        cos_t2 = -(r**2 + h**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_t2 = max(-1.0, min(1.0, cos_t2))
        theta2 = math.pi - math.acos(cos_t2)  # 肘部弯曲

        # 肩部角度
        beta = math.atan2(h, r)
        gamma = math.atan2(
            self.L2 * math.sin(theta2),
            self.L1 + self.L2 * math.cos(theta2),
        )
        theta1 = beta + gamma

        # 应用偏移补偿
        joint2 = theta1 + self.THETA1_OFFSET
        joint3 = theta2 + self.THETA2_OFFSET

        # 关节限位钳制 (rad)
        joint2 = max(-0.1, min(3.45, joint2))
        joint3 = max(-0.2, min(math.pi, joint3))

        # 转换为舵机度数输出（与舵机角度约定一致）
        j2_deg = 90 - math.degrees(joint2)
        j3_deg = math.degrees(joint3) - 90

        return j2_deg, j3_deg

    def forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """正运动学：给定 6 维关节角度 (rad)，返回 (x, y, z) 笛卡尔坐标

        使用变换矩阵链式计算，与 2D FK (xlerobot_fk) 保持一致。
        """
        j = angles

        # 去除偏移补偿，得到数学平面角度
        # 从 IK 关系: theta1 = pi/2 - radians(j2_deg), 且 j2_rad = j[1] + offset
        theta1 = j[1] + self.THETA1_OFFSET
        theta2 = j[2] + self.THETA2_OFFSET

        # --- 变换矩阵链 ---
        # T01: shoulder_pan 绕 Z 轴旋转
        c0, s0 = math.cos(j[0]), math.sin(j[0])
        T01 = np.array([
            [c0, -s0, 0, 0],
            [s0,  c0, 0, 0],
            [0,   0,  1, self.BASE_HEIGHT],
            [0,   0,  0, 1],
        ])

        # T12: shoulder_lift 绕 Y 轴旋转（平面内）
        c1, s1 = math.cos(theta1), math.sin(theta1)
        T12 = np.array([
            [c1, 0, s1, 0],
            [0,  1, 0,  0],
            [-s1, 0, c1, 0],
            [0,  0, 0,  1],
        ])

        # T23: elbow_flex 绕 Y 轴旋转 + L1 平移
        c2, s2 = math.cos(theta2), math.sin(theta2)
        T23 = np.array([
            [c2, 0, s2, self.L1],
            [0,  1, 0,  0],
            [-s2, 0, c2, 0],
            [0,  0, 0,  1],
        ])

        # T34: wrist_flex 绕 Y 轴旋转 + L2 平移
        c3, s3 = math.cos(j[3]), math.sin(j[3])
        T34 = np.array([
            [c3, 0, s3, self.L2],
            [0,  1, 0,  0],
            [-s3, 0, c3, 0],
            [0,  0, 0,  1],
        ])

        # 链式乘法
        T = T01 @ T12 @ T23 @ T34
        return T[:3, 3]

    def clamp_workspace(self, xyz: np.ndarray) -> np.ndarray:
        """工作空间钳制，确保目标在可达范围内

        - 径向距离钳制到 [r_min, r_max]
        - 高度钳制到 [BASE_HEIGHT, BASE_HEIGHT + r_max]
        """
        xyz = xyz.copy()
        r_max = self.L1 + self.L2
        r_min = abs(self.L1 - self.L2)

        x, y, z = xyz[0], xyz[1], xyz[2]
        r = math.sqrt(x**2 + y**2)

        # 径向钳制
        if r > r_max:
            scale = r_max / r
            xyz[0] *= scale
            xyz[1] *= scale
        elif r < r_min and r > 0:
            scale = r_min / r
            xyz[0] *= scale
            xyz[1] *= scale

        # 高度钳制
        xyz[2] = max(self.BASE_HEIGHT, min(self.BASE_HEIGHT + r_max, xyz[2]))

        return xyz

    def clamp_joint(self, angles: np.ndarray) -> np.ndarray:
        """关节限位钳制"""
        result = angles.copy()
        for i in range(6):
            low, high = self.JOINT_LIMITS[i + 1]
            result[i] = np.clip(result[i], low, high)
        return result
