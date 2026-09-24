# vla/kinematics.py — 向后兼容桩 (Deprecated)
#
# 警告: 此模块已废弃，新代码请直接使用 policy.kinematics.Kinematics。
#
# 重构方案 v9 第 10 节 / 第 4 节 Bug 1:
#   - 旧 vla/kinematics.py 的 NumericalIK 类不存在（致命导入错误）、无偏移补偿
#     （连杆参数与 URDF 偏差 37mm）
#   - 替代实现: policy/kinematics.py（完整 6DOF 解析 IK + XLeRobot 偏移补偿，
#     FK/IK 严格互逆，tests/test_kinematics.py 6/6 通过）
#   - 此桩文件为 scripts/calib_vlm.py、scripts/vlm_grasp.py 提供旧 API 兼容
"""SO-ARM101 运动学 — 向后兼容包装 (Deprecated)

旧 API:
    from vla.kinematics import Kinematics, WORKSPACE, forward_kinematics
    T = forward_kinematics(joints_rad_6)   # → 4x4 齐次矩阵

全部委托 policy.kinematics.Kinematics（单一事实来源）。
"""
import warnings

import numpy as np

warnings.warn(
    "vla.kinematics 已废弃，请使用 policy.kinematics.Kinematics",
    DeprecationWarning, stacklevel=2,
)

from policy.kinematics import Kinematics

__all__ = ["Kinematics", "WORKSPACE", "forward_kinematics", "inverse_kinematics"]

# 工作空间限位 [x_min, x_max, y_min, y_max, z_min, z_max] (米)
# 与 hardware/arm.py WORKSPACE 保持一致（单一事实来源为 URDF + 实测）
WORKSPACE = np.array([0.03, 0.45, -0.30, 0.45, 0.01, 0.40])

# 共享实例（无状态，可安全复用）
_shared_kin = Kinematics()


def forward_kinematics(angles) -> np.ndarray:
    """旧 API: 6 维关节角度 (rad) → 4x4 齐次变换矩阵（基座坐标系）

    位置列 T[:3, 3] 与 IK 严格互逆；姿态部分为近似合成（见
    policy.kinematics.Kinematics.forward_kinematics_matrix 文档）。

    Args:
        angles: (6,) 关节角度 (rad)，舵机约定（含机械偏移）

    Returns:
        (4, 4) 齐次变换矩阵
    """
    return _shared_kin.forward_kinematics_matrix(np.asarray(angles, dtype=float))


def inverse_kinematics(target_xyz, current_angles, wrist_roll_rad=None) -> np.ndarray:
    """旧 API 兼容: 委托 Kinematics.inverse_kinematics

    Args:
        target_xyz: 目标笛卡尔坐标 (x, y, z)，米
        current_angles: 当前 6 维关节角度 (rad)
        wrist_roll_rad: 可选腕部旋转角

    Returns:
        (6,) 关节角度 (rad)，已钳制到限位
    """
    return _shared_kin.inverse_kinematics(
        np.asarray(target_xyz, dtype=float),
        np.asarray(current_angles, dtype=float),
        wrist_roll_rad=wrist_roll_rad,
    )