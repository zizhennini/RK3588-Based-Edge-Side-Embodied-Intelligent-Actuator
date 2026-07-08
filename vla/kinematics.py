#!/usr/bin/env python3
"""IK 运动学控制层 — 基于 SO-101 URDF 参数的正解+逆解，纯 numpy"""

import numpy as np


# ── SO-101 URDF 连杆参数 ──
# 从 so101_new_calib.urdf 提取
L1 = 0.120    # 基座到肩部高度
L2 = 0.150    # 上臂长度 (shoulder_lift 到 elbow_flex)
L3 = 0.120    # 前臂长度 (elbow_flex 到 wrist_flex)
L4 = 0.060    # 腕部到夹爪长度

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# 关节限位（弧度，从 URDF 提取）
JOINT_LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),
}

# 工作空间 [x_min, x_max, y_min, y_max, z_min, z_max]
WORKSPACE = np.array([0.03, 0.45, -0.30, 0.45, 0.01, 0.40])

# 归零位/初始位
HOME_POSE = np.array([-0.0054, -1.8052, 1.6794, 0.7925, 0.0284, -0.992])


def clamp_workspace(xyz: np.ndarray) -> np.ndarray:
    """钳制末端坐标到工作空间"""
    ws = WORKSPACE
    xyz[0] = np.clip(xyz[0], ws[0], ws[1])
    xyz[1] = np.clip(xyz[1], ws[2], ws[3])
    xyz[2] = np.clip(xyz[2], ws[4], ws[5])
    return xyz


def clamp_joints(angles: np.ndarray) -> np.ndarray:
    """钳制关节角度到限位"""
    for i, name in enumerate(JOINT_NAMES):
        low, high = JOINT_LIMITS[name]
        angles[i] = np.clip(angles[i], low, high)
    return angles


def forward_kinematics(joints_rad: np.ndarray) -> np.ndarray:
    """正运动学：关节角度 -> 末端 4x4 变换矩阵"""
    if len(joints_rad) < 5:
        return np.eye(4)
    
    j = joints_rad  # [pan, lift, elbow, wrist_flex, wrist_roll]
    
    # 肩部旋转
    T01 = np.array([
        [np.cos(j[0]), -np.sin(j[0]), 0, 0],
        [np.sin(j[0]),  np.cos(j[0]), 0, 0],
        [0, 0, 1, L1],
        [0, 0, 0, 1]
    ])
    
    # 肩部抬升 -> 上臂
    T12 = np.array([
        [np.cos(j[1]), 0, np.sin(j[1]), 0],
        [0, 1, 0, 0],
        [-np.sin(j[1]), 0, np.cos(j[1]), 0],
        [0, 0, 0, 1]
    ])
    
    # 肘部弯曲 -> 前臂
    T23 = np.array([
        [np.cos(j[2]), 0, np.sin(j[2]), L2],
        [0, 1, 0, 0],
        [-np.sin(j[2]), 0, np.cos(j[2]), 0],
        [0, 0, 0, 1]
    ])
    
    # 腕部弯曲
    T34 = np.array([
        [np.cos(j[3]), 0, np.sin(j[3]), L3],
        [0, 1, 0, 0],
        [-np.sin(j[3]), 0, np.cos(j[3]), 0],
        [0, 0, 0, 1]
    ])
    
    # 腕部旋转 + 夹爪长度
    T45 = np.array([
        [np.cos(j[4]), -np.sin(j[4]), 0, 0],
        [np.sin(j[4]),  np.cos(j[4]), 0, 0],
        [0, 0, 1, L4],
        [0, 0, 0, 1]
    ])
    
    T05 = T01 @ T12 @ T23 @ T34 @ T45
    return T05


def inverse_kinematics(target_xyz, current_joints=None) -> np.ndarray:
    """逆运动学：目标坐标 (x,y,z) -> 6 关节角度（弧度）"""
    x, y, z = target_xyz[:3]
    
    # 肩部旋转角 (shoulder_pan)
    theta1 = np.arctan2(y, x)
    
    # 水平距离
    r = np.sqrt(x**2 + y**2)
    
    # 从肩部到目标点的距离
    d = np.sqrt(r**2 + (z - L1)**2)
    
    # 肘部弯曲 (elbow_flex) - 余弦定理解算
    cos_t3 = (d**2 - L2**2 - L3**2) / (2 * L2 * L3)
    cos_t3 = np.clip(cos_t3, -1.0, 1.0)
    theta3 = -abs(np.arccos(cos_t3))  # 肘部始终向上弯曲
    
    # 肩部抬升 (shoulder_lift)
    phi = np.arctan2(z - L1, r)
    psi = np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
    theta2 = phi - psi
    
    # 腕部弯曲 (wrist_flex) - 保持末端水平
    theta4 = -(theta2 + theta3) + np.pi / 2
    
    # 腕部旋转 (wrist_roll)
    if current_joints is not None and len(current_joints) > 4:
        theta5 = current_joints[4]  # 保持当前方向
    else:
        theta5 = 0.0
    
    # 夹爪
    if current_joints is not None and len(current_joints) > 5:
        theta6 = current_joints[5]
    else:
        theta6 = 0.0
    
    q = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    return clamp_joints(q)


class Kinematics:
    """SO-101 运动学封装"""

    @staticmethod
    def fk(joints_rad):
        return forward_kinematics(joints_rad)

    @staticmethod
    def ik(xyz, current=None):
        return inverse_kinematics(xyz, current)

    @staticmethod
    def get_ee_position(joints_rad):
        """从关节角度获取末端位置 (x, y, z)"""
        T = forward_kinematics(joints_rad)
        return T[:3, 3]

    @staticmethod
    def clamp(xyz):
        return clamp_workspace(xyz.copy())

    @staticmethod
    def clamp_j(joints):
        return clamp_joints(joints.copy())
