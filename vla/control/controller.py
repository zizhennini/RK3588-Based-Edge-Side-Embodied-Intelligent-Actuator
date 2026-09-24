# vla/control/controller.py — 向后兼容桩 (Deprecated)
#
# 警告: 此文件仅为旧版脚本提供向后兼容，新代码请直接使用 hardware.arm.SO101Arm。
#
# 重构方案 v9 第 1.4 节决策:
#   - 旧 ArmController 的问题: 手工 pyserial 协议包、无 SYNC_READ、缺方法、外参不一致
#   - 替代实现: hardware/arm.py (scservo_sdk 封装, SYNC_READ/WRITE)
#   - 此桩文件保留仅使 scripts/replay_traj.py、vla/pipe/pipeline.py 等旧代码可继续运行
#
# 设计说明:
#   - 完全自包含 (pyserial)，不绑定 SO101Arm 实例，避免同一串口双句柄冲突
#   - IK 委托 policy.kinematics.Kinematics (XLeRobot 偏移补偿, 与 SO101Arm 同源)
#   - 外参统一读取 config/settings.py 实测标定值 [0.182, -0.129, 0.47] (v9 第 5.3 节)
"""SO-ARM101 机械臂控制 — 向后兼容包装 (Deprecated)

保留旧版 pyserial 协议实现，供 scripts/replay_traj.py 等旧脚本使用。
IK / 外参 / 标定与新架构 (hardware.arm.SO101Arm) 保持同源一致。

废弃计划: 旧脚本迁移到 SO101Arm 后移除。
"""
import time
import logging
from typing import Optional

import numpy as np

try:
    import serial
except ImportError:  # pragma: no cover - Windows 开发机可能未装 pyserial
    serial = None

from policy.kinematics import Kinematics

logger = logging.getLogger(__name__)

# ── 舵机协议常量 (Feetech STS/SCS 半双工协议) ─────────────────────────────────
_WRITE = 0x03               # WRITE 指令
_GOAL_POSITION = 0x2A       # Goal Position 地址 (2 bytes)
_PRESENT_POSITION = 0x38    # Present Position 地址 (2 bytes)
_TORQUE_ENABLE = 0x28       # 扭矩使能地址

# ── 关节定义 (与 hardware/arm.py 一致) ────────────────────────────────────────
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# 归零位（弧度，与 hardware/arm.py HOME_POSE 一致）
HOME_POSE = np.array([-0.0054, -1.8052, 1.6794, 0.7925, 0.0284, -0.992])

# 夹爪脉冲
GRIPPER_OPEN_PULSE = 2600
GRIPPER_CLOSE_PULSE = 1781

# 默认标定参数（与 hardware/arm.py DEFAULT_CALIBRATION 一致）
DEFAULT_CALIBRATION = {
    "1": {"homing_offset": 2048, "range_min": 946, "range_max": 3287},
    "2": {"homing_offset": 2048, "range_min": 821, "range_max": 3206},
    "3": {"homing_offset": 2048, "range_min": 888, "range_max": 3105},
    "4": {"homing_offset": 2048, "range_min": 851, "range_max": 3192},
    "5": {"homing_offset": 2048, "range_min": 130, "range_max": 3985},
    "6": {"homing_offset": 1781, "range_min": 1495, "range_max": 2860},
}


def _load_calibration(path: str = "./config/calibration.json") -> dict:
    """从 JSON 加载标定参数，失败时使用默认值（与 SO101Arm._load_calibration 行为一致）"""
    import json
    from pathlib import Path
    calib_path = Path(path)
    if calib_path.exists():
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid in range(1, 7):
                if str(sid) not in data:
                    data[str(sid)] = DEFAULT_CALIBRATION[str(sid)]
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("标定加载失败 (%s)，使用默认值", e)
    return dict(DEFAULT_CALIBRATION)


class ArmController:
    """SO-ARM101 机械臂控制 — 旧版兼容接口 (Deprecated)

    完全自包含的 pyserial 实现，不与 SO101Arm 共享串口句柄。
    IK 使用 policy.kinematics.Kinematics（与 SO101Arm.move_to 同源）。

    Args:
        port: 串口设备路径 (默认 /dev/ttyACM0, 统一自 config/settings.py)
        baud: 波特率 (默认 1000000, STS3215 出厂值)
        calibration_path: 标定 JSON 路径
    """

    # 归零位（弧度）— 旧脚本通过 arm.HOME_POSE 访问
    HOME_POSE: np.ndarray = HOME_POSE

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 1000000,
                 calibration_path: str = "./config/calibration.json"):
        if serial is None:
            raise RuntimeError("pyserial 未安装，无法使用 ArmController")

        self.port = port
        self.baud = baud
        self.ser = serial.Serial(port, baud, timeout=0.02, write_timeout=0.02)
        self.ser.reset_input_buffer()

        self.calibration = _load_calibration(calibration_path)
        self._kin = Kinematics()
        self._last_cmd_angles: Optional[np.ndarray] = None

        # 相机外参 — 统一读取 settings.py 实测值 (v9 第 5.3 节)
        try:
            from config.settings import CAMERA_POSITION
            self.camera_position = np.asarray(CAMERA_POSITION, dtype=float)
        except ImportError:
            self.camera_position = np.array([0.182, -0.129, 0.47])

        logger.info("ArmController(deprecated) %s @ %d baud — 新代码请使用 hardware.arm.SO101Arm",
                    port, baud)

    # ────────────────────────────────────────────────────────────────────────
    # 底层写入（旧脚本 replay_traj.py 直接调用）
    # ────────────────────────────────────────────────────────────────────────

    def _angle_to_raw(self, sid: int, angle_rad: float) -> int:
        """弧度 → 脉冲值（含标定偏移和限位钳制）"""
        calib = self.calibration[str(sid)]
        mid = calib["homing_offset"]
        deg = float(np.rad2deg(angle_rad))
        raw = int(round(deg * 4095.0 / 360.0 + mid))
        return max(calib["range_min"], min(calib["range_max"], raw))

    def _write_raw_pos(self, sid: int, raw_pos: int) -> None:
        """写入原始脉冲值（Feetech WRITE 协议包）"""
        pkt = bytearray([
            0xFF, 0xFF, sid, 5, _WRITE,
            _GOAL_POSITION,
            raw_pos & 0xFF,
            (raw_pos >> 8) & 0xFF,
        ])
        cks = (~sum(pkt[2:]) & 0xFF)
        self.ser.write(pkt + bytearray([cks]))

    def _write_angle(self, sid: int, angle_rad: float) -> None:
        """写入单个舵机目标角度（弧度）

        Args:
            sid: 舵机 ID (1-6)
            angle_rad: 目标角度（弧度）
        """
        self._write_raw_pos(sid, self._angle_to_raw(sid, angle_rad))

    def write_angles(self, angles: np.ndarray) -> None:
        """批量写入 6 关节角度（弧度），逐个发送 WRITE 包"""
        for sid in range(1, 7):
            self._write_angle(sid, float(angles[sid - 1]))
        self._last_cmd_angles = np.asarray(angles, dtype=float).copy()

    def read_positions(self) -> np.ndarray:
        """批量读取 6 关节角度（弧度）。

        注意: 旧协议为逐舵机读取（6 次串口交互），无 SYNC_READ。
        新代码请使用 SO101Arm.read_positions()（SYNC_READ 单次批量读取）。
        """
        angles = np.zeros(6)
        for sid in range(1, 7):
            try:
                self.ser.reset_input_buffer()
                pkt = bytearray([0xFF, 0xFF, sid, 4, 0x02, _PRESENT_POSITION, 0x02])
                cks = (~sum(pkt[2:]) & 0xFF)
                self.ser.write(pkt + bytearray([cks]))
                time.sleep(0.003)
                resp = self.ser.read(10)
                if len(resp) >= 9 and resp[0] == 0xFF and resp[1] == 0xFF:
                    raw = int.from_bytes(resp[7:9], "little")
                    mid = self.calibration[str(sid)]["homing_offset"]
                    deg = (raw - mid) * 360.0 / 4095.0
                    angles[sid - 1] = float(np.deg2rad(deg))
            except Exception as e:
                logger.debug("读取舵机 %d 失败: %s", sid, e)
        return angles

    # 旧 API 别名（scripts/calibrate_quick.py 使用）
    _read_current_pos = read_positions

    # ────────────────────────────────────────────────────────────────────────
    # 运动控制（IK 委托 policy.kinematics — 与 SO101Arm 同源）
    # ────────────────────────────────────────────────────────────────────────

    def move_to(self, x: float, y: float, z: float,
                wrist_roll_rad: Optional[float] = None) -> None:
        """笛卡尔 IK 运动

        Args:
            x, y, z: 机器人基座坐标系目标位置（米）
            wrist_roll_rad: 腕部旋转角（弧度），None 保持当前
        """
        xyz = self._kin.clamp_workspace(np.array([x, y, z], dtype=float))
        current = self._last_cmd_angles if self._last_cmd_angles is not None \
            else self.read_positions()
        angles = self._kin.inverse_kinematics(xyz, current, wrist_roll_rad)
        self.write_angles(angles)

    def camera_to_robot(self, cam_x: float, cam_y: float,
                        cam_z: float) -> np.ndarray:
        """相机坐标 → 机器人基座坐标（平移外参, 与 SO101Arm 一致）"""
        return np.array([
            cam_x + self.camera_position[0],
            cam_y + self.camera_position[1],
            cam_z + self.camera_position[2],
        ])

    def move_to_camera(self, cam_x: float, cam_y: float, cam_z: float) -> None:
        """相机坐标系目标 → IK 运动"""
        xyz = self.camera_to_robot(cam_x, cam_y, cam_z)
        self.move_to(xyz[0], xyz[1], xyz[2])

    def move_to_camera_with_angle(self, cam_x: float, cam_y: float,
                                  cam_z: float, angle: float) -> None:
        """相机坐标 + 腕部角度 IK 运动"""
        xyz = self.camera_to_robot(cam_x, cam_y, cam_z)
        self.move_to(xyz[0], xyz[1], xyz[2], wrist_roll_rad=angle)

    def gripper(self, open: bool) -> None:
        """夹爪开/关"""
        pulse = GRIPPER_OPEN_PULSE if open else GRIPPER_CLOSE_PULSE
        self._write_raw_pos(6, pulse)

    def gripper_width(self, width_m: float) -> None:
        """自适应夹爪宽度（米 → 脉冲线性映射, 与 SO101Arm.gripper_width 一致）"""
        calib = self.calibration["6"]
        max_width = 0.08
        width_m = max(0.0, min(max_width, width_m))
        ratio = width_m / max_width
        pulse = int(calib["range_min"] + ratio * (calib["range_max"] - calib["range_min"]))
        pulse = max(calib["range_min"], min(calib["range_max"], pulse))
        self._write_raw_pos(6, pulse)

    def home(self, steps: int = 50, delay_s: float = 0.02) -> None:
        """归零（插值平滑）

        Args:
            steps: 插值步数
            delay_s: 每步间隔（秒）
        """
        current = self.read_positions()
        target = HOME_POSE.copy()
        for i in range(1, steps + 1):
            t = i / steps
            self.write_angles(current * (1 - t) + target * t)
            time.sleep(delay_s)
        self._last_cmd_angles = target

    def emergency_stop(self) -> None:
        """急停 — 关闭全部舵机扭矩"""
        for sid in range(1, 7):
            pkt = bytearray([0xFF, 0xFF, sid, 4, _WRITE, _TORQUE_ENABLE, 0x00])
            cks = (~sum(pkt[2:]) & 0xFF)
            try:
                self.ser.write(pkt + bytearray([cks]))
            except Exception:
                pass
        logger.warning("ArmController emergency_stop 已触发")

    # ────────────────────────────────────────────────────────────────────────
    # 资源释放
    # ────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """释放串口资源"""
        try:
            self.ser.close()
        except Exception:
            pass
        logger.info("ArmController 已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class NumericalIK:
    """旧版数值 IK 兼容类 (Deprecated)

    scripts/calibrate_quick.py 使用:
        ik = NumericalIK()
        T = ik.forward_kinematics(np.rad2deg(joints[:5]))  # 5 关节角度(deg) → 4x4

    现委托 policy.kinematics.Kinematics 解析解（单一事实来源）。
    旧实现为数值迭代解，已随原 controller.py 一并废弃。
    """

    def __init__(self):
        self._kin = Kinematics()

    def forward_kinematics(self, joints_deg) -> np.ndarray:
        """正运动学 — 旧 API: 关节角度(度, 5 或 6 维) → 4x4 齐次矩阵

        Args:
            joints_deg: 5 或 6 维关节角度（度）。5 维时第 6 关节按 0 处理。

        Returns:
            (4, 4) 齐次变换矩阵（基座坐标系），位置列与 IK 严格互逆。
        """
        deg = np.asarray(joints_deg, dtype=float).ravel()
        rad = np.deg2rad(deg)
        angles = np.zeros(6)
        n = min(len(rad), 6)
        angles[:n] = rad[:n]
        return self._kin.forward_kinematics_matrix(angles)

    def inverse_kinematics(self, target_xyz, current_angles=None) -> np.ndarray:
        """逆运动学 — 委托解析 IK

        Args:
            target_xyz: 目标笛卡尔坐标 (x, y, z)，米
            current_angles: 当前 6 维关节角度 (rad)，None 时取 HOME_POSE

        Returns:
            (6,) 关节角度 (rad)，已钳制到限位
        """
        cur = HOME_POSE.copy() if current_angles is None \
            else np.asarray(current_angles, dtype=float)
        return self._kin.inverse_kinematics(
            np.asarray(target_xyz, dtype=float), cur)