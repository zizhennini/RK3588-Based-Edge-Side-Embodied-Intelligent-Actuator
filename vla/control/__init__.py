"""vla.control — 废弃模块，新代码请使用 hardware.arm.SO101Arm

refactor_plan_v9.md 第 1.4 节：此模块为旧版脚本提供向后兼容。
"""
import warnings
warnings.warn(
    "vla.control 已废弃，请使用 hardware.arm.SO101Arm",
    DeprecationWarning, stacklevel=2,
)

from .controller import ArmController
