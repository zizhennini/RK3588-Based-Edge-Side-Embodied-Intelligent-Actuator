# hardware/arm.py
"""SO-ARM101 机械臂控制 -- 基于 scservo_sdk 封装"""
import time
import threading
import json
import logging
import math
import numpy as np
from pathlib import Path
from typing import Optional

import scservo_sdk as scs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monkey-patch: 修复 feetech-servo-sdk v1.0.0 的超时计算 Bug
# 参考: https://gitee.com/ftservo/SCServoSDK/issues/IBY2S6
# ---------------------------------------------------------------------------
def _patch_setPacketTimeout(self, packet_length):  # noqa: N802
    """修复 feetech-servo-sdk v1.0.0 的超时计算 Bug"""
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + \
                          (self.tx_time_per_byte * 3.0) + 50


# ---------------------------------------------------------------------------
# SO-ARM101 关节定义
# ---------------------------------------------------------------------------
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
MOTOR_IDS = {name: i + 1 for i, name in enumerate(JOINT_NAMES)}

# 默认标定参数（来自 controller.py CALIB）
DEFAULT_CALIBRATION = {
    "1": {"homing_offset": 2048, "range_min": 946, "range_max": 3287},
    "2": {"homing_offset": 2048, "range_min": 821, "range_max": 3206},
    "3": {"homing_offset": 2048, "range_min": 888, "range_max": 3105},
    "4": {"homing_offset": 2048, "range_min": 851, "range_max": 3192},
    "5": {"homing_offset": 2048, "range_min": 130, "range_max": 3985},
    "6": {"homing_offset": 1781, "range_min": 1495, "range_max": 2860},
}

# 关节限位（弧度）
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

# 归零位
HOME_POSE = np.array([-0.0054, -1.8052, 1.6794, 0.7925, 0.0284, -0.992])

# 夹爪脉冲
GRIPPER_OPEN_PULSE = 2600
GRIPPER_CLOSE_PULSE = 1781


class SO101Arm:
    """SO-ARM101 机械臂控制 -- 基于 scservo_sdk

    线程安全单例模式，防止多实例争抢串口。
    所有串口写入操作经 _safe_write 包裹，支持自动重试和串口恢复。
    """

    _instance: Optional["SO101Arm"] = None
    _lock = threading.Lock()

    # ------------------------------------------------------------------
    # 单例管理
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, **kwargs) -> "SO101Arm":
        """线程安全单例获取"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 1000000,
                 calibration_path: str = "./config/calibration.json"):
        # 串口
        self.port = port
        self.baud = baud
        self.port_handler = scs.PortHandler(port)
        self.packet_handler = scs.PacketHandler(0)  # Protocol 0 (STS/SCS)

        # monkey-patch 修复超时计算 Bug
        self.port_handler.setPacketTimeout = \
            _patch_setPacketTimeout.__get__(self.port_handler, type(self.port_handler))

        # SYNC 读写器
        self.sync_writer = scs.GroupSyncWrite(
            self.port_handler, self.packet_handler, 0x2A, 2)  # Goal_Position
        self.sync_reader = scs.GroupSyncRead(
            self.port_handler, self.packet_handler, 0x38, 2)  # Present_Position

        # 标定
        self.calibration = self._load_calibration(calibration_path)

        # 外参（统一使用 settings.py 实测值）
        self.camera_position = np.array([0.182, -0.129, 0.47])

        # 状态
        self._last_cmd_angles: Optional[np.ndarray] = None
        self._connected = False

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """打开串口并配置舵机"""
        if self._connected:
            logger.warning("Already connected")
            return
        if not self.port_handler.openPort():
            raise IOError(f"Failed to open serial port {self.port}")
        if not self.port_handler.setBaudRate(self.baud):
            self.port_handler.closePort()
            raise IOError(f"Failed to set baud rate {self.baud}")
        logger.info("Serial port %s opened at %d baud", self.port, self.baud)
        self._configure_motors()
        self._connected = True

    def disconnect(self) -> None:
        """禁用扭矩并关闭串口"""
        if not self._connected:
            return
        try:
            self.emergency_stop()
        except Exception:
            pass
        try:
            self.port_handler.closePort()
        except Exception:
            pass
        self._connected = False
        logger.info("Serial port closed")

    def _configure_motors(self) -> None:
        """配置所有 6 个舵机参数（降低 Return_Delay_Time，提高加速度）"""
        ph = self.port_handler
        for sid in range(1, 7):
            # Return_Delay_Time (addr 0x29) = 100 → 减少响应延迟
            self._safe_write(
                self.packet_handler.write1ByteTxRx,
                ph, sid, 0x29, 100
            )
            # Acceleration (addr 0x1A) = 16 → 平滑加速
            self._safe_write(
                self.packet_handler.write1ByteTxRx,
                ph, sid, 0x1A, 16
            )
        logger.debug("Motors configured (IDs 1-6)")

    def _reset_serial(self) -> None:
        """串口异常恢复：关闭 → 等待 0.5s → 重开"""
        logger.warning("Resetting serial port %s", self.port)
        try:
            self.port_handler.closePort()
        except Exception:
            pass
        time.sleep(0.5)
        if not self.port_handler.openPort():
            raise IOError(f"Failed to reopen serial port {self.port}")
        if not self.port_handler.setBaudRate(self.baud):
            raise IOError(f"Failed to set baud rate after reset")
        self._configure_motors()
        logger.info("Serial port reset complete")

    def _force_reset(self) -> None:
        """强制重置串口状态"""
        try:
            self.port_handler.closePort()
        except Exception:
            pass
        self._connected = False

    def _safe_write(self, func, *args, max_retries: int = 3):
        """带重试和串口恢复的安全写入

        scservo_sdk 的 write*TxRx 返回 (comm_result, hw_error) 元组，
        通信失败时 comm_result != COMM_SUCCESS。此方法同时处理异常和错误码。

        Args:
            func: packet_handler 的写入方法
            *args: 传递给 func 的参数
            max_retries: 最大重试次数
        Returns:
            func 的返回值，失败返回 None
        """
        for attempt in range(max_retries):
            try:
                result = func(*args)
                # 检查 scservo_sdk 通信结果（返回元组时）
                if isinstance(result, tuple) and len(result) >= 1:
                    comm_result = result[0]
                    if comm_result != 0 and comm_result != scs.COMM_SUCCESS:
                        logger.warning(
                            "Comm error on attempt %d/%d: %s",
                            attempt + 1, max_retries,
                            self.packet_handler.getTxRxResult(comm_result)
                        )
                        if attempt < max_retries - 1:
                            self._reset_serial()
                        continue
                return result
            except Exception as e:
                logger.warning("Write attempt %d/%d failed: %s",
                               attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    try:
                        self._reset_serial()
                    except Exception as reset_err:
                        logger.error("Serial reset failed: %s", reset_err)
        logger.error("Write failed after %d retries", max_retries)
        return None

    # ------------------------------------------------------------------
    # 位置读写
    # ------------------------------------------------------------------
    def read_positions(self) -> np.ndarray:
        """SYNC_READ 批量读取 6 个舵机位置，返回弧度数组 (6,)"""
        angles = np.zeros(6)

        # 添加所有舵机到 SYNC_READ（地址和长度已在构造时指定）
        for sid in range(1, 7):
            self.sync_reader.addParam(sid)

        # 执行 SYNC_READ 通信
        self.sync_reader.txRxPacket()

        for sid in range(1, 7):
            try:
                raw_pos = self.sync_reader.getData(sid, 0x38, 2)
                calib = self.calibration[str(sid)]
                mid = calib["homing_offset"]
                angle_deg = (raw_pos - mid) * 360.0 / 4095.0
                angle_rad = np.deg2rad(angle_deg)
                angles[sid - 1] = angle_rad
            except Exception as e:
                logger.debug("Failed to read servo %d: %s", sid, e)
                angles[sid - 1] = 0.0

        # 清除参数以便下次读取
        self.sync_reader.clearParam()
        return angles

    def write_positions(self, angles_rad: np.ndarray) -> None:
        """SYNC_WRITE 批量写入 6 个关节角度（弧度）"""
        # 清除之前的参数
        self.sync_writer.clearParam()

        for sid in range(1, 7):
            calib = self.calibration[str(sid)]
            mid = calib["homing_offset"]
            deg = np.rad2deg(angles_rad[sid - 1])
            raw = int(deg * 4095.0 / 360.0 + mid)
            raw = max(calib["range_min"], min(calib["range_max"], raw))
            # 添加参数: [SID, LOBYTE, HIBYTE]
            self.sync_writer.addParam(sid, [scs.SCS_LOBYTE(raw), scs.SCS_HIBYTE(raw)])

        self.sync_writer.txRxPacket()
        self._last_cmd_angles = angles_rad.copy()

    # ------------------------------------------------------------------
    # 运动控制
    # ------------------------------------------------------------------
    def move_to(self, x: float, y: float, z: float,
                wrist_roll_rad: Optional[float] = None) -> None:
        """笛卡尔 IK 运动

        Args:
            x, y, z: 目标笛卡尔坐标（米）
            wrist_roll_rad: 腕部旋转角度（弧度），None 则保持当前值
        """
        from policy.kinematics import Kinematics

        kin = Kinematics()
        xyz = kin.clamp_workspace(np.array([x, y, z]))
        current = self._last_cmd_angles if self._last_cmd_angles is not None \
            else self.read_positions()
        angles_rad = kin.inverse_kinematics(xyz, current, wrist_roll_rad)
        self._last_cmd_angles = angles_rad.copy()

        # 写入前 5 个关节（不含夹爪）
        self.sync_writer.clearParam()
        for sid in range(1, 6):
            calib = self.calibration[str(sid)]
            mid = calib["homing_offset"]
            deg = np.rad2deg(angles_rad[sid - 1])
            raw = int(deg * 4095.0 / 360.0 + mid)
            raw = max(calib["range_min"], min(calib["range_max"], raw))
            self.sync_writer.addParam(sid, [scs.SCS_LOBYTE(raw), scs.SCS_HIBYTE(raw)])
        self.sync_writer.txRxPacket()

    def camera_to_robot(self, cam_x: float, cam_y: float,
                        cam_z: float) -> np.ndarray:
        """相机坐标转机器人基座坐标

        Args:
            cam_x, cam_y, cam_z: 相机坐标系下的位置（米）
        Returns:
            机器人基座坐标系下的 [x, y, z]
        """
        robot_x = cam_x + self.camera_position[0]
        robot_y = cam_y + self.camera_position[1]
        robot_z = cam_z + self.camera_position[2]
        return np.array([robot_x, robot_y, robot_z])

    def move_to_camera_with_angle(self, cam_x: float, cam_y: float,
                                  cam_z: float, angle: float) -> None:
        """相机坐标 + 角度 IK 运动

        Args:
            cam_x, cam_y, cam_z: 相机坐标系下的目标位置
            angle: 腕部旋转角度（弧度）
        """
        robot_xyz = self.camera_to_robot(cam_x, cam_y, cam_z)
        self.move_to(robot_xyz[0], robot_xyz[1], robot_xyz[2],
                     wrist_roll_rad=angle)

    def gripper(self, open: bool) -> None:
        """夹爪开/关

        Args:
            open: True=打开, False=关闭
        """
        ph = self.port_handler
        pulse = GRIPPER_OPEN_PULSE if open else GRIPPER_CLOSE_PULSE
        self._safe_write(
            self.packet_handler.write2ByteTxRx,
            ph, 6, 0x2A, pulse
        )

    def gripper_width(self, width_m: float) -> None:
        """自适应夹爪宽度（米制转脉冲）

        Args:
            width_m: 夹爪开合宽度（米），范围约 [0.0, 0.08]
        """
        calib = self.calibration["6"]
        # 将米制宽度线性映射到脉冲范围
        # 假设全开 ~0.08m 对应 range_max, 全闭 0m 对应 range_min
        max_width = 0.08  # 最大开合宽度（米）
        width_m = max(0.0, min(max_width, width_m))
        ratio = width_m / max_width
        pulse = int(calib["range_min"] + ratio * (calib["range_max"] - calib["range_min"]))
        pulse = max(calib["range_min"], min(calib["range_max"], pulse))
        self._safe_write(
            self.packet_handler.write2ByteTxRx,
            self.port_handler, 6, 0x2A, pulse
        )

    def home(self, steps: int = 50, delay_s: float = 0.02) -> None:
        """归零（插值平滑）

        Args:
            steps: 插值步数
            delay_s: 每步间隔（秒）
        """
        current = self.read_positions()
        target = HOME_POSE

        for i in range(1, steps + 1):
            t = i / steps
            angles = current * (1 - t) + target * t

            # 写入前 5 个关节
            self.sync_writer.clearParam()
            for sid in range(1, 6):
                calib = self.calibration[str(sid)]
                mid = calib["homing_offset"]
                deg = np.rad2deg(angles[sid - 1])
                raw = int(deg * 4095.0 / 360.0 + mid)
                raw = max(calib["range_min"], min(calib["range_max"], raw))
                self.sync_writer.addParam(sid, [scs.SCS_LOBYTE(raw), scs.SCS_HIBYTE(raw)])
            self.sync_writer.txRxPacket()

            # 夹爪插值
            calib_g = self.calibration["6"]
            g_pulse = int(np.interp(
                angles[5],
                [JOINT_LIMITS["gripper"][0], JOINT_LIMITS["gripper"][1]],
                [calib_g["range_min"], calib_g["range_max"]]
            ))
            g_pulse = max(calib_g["range_min"], min(calib_g["range_max"], g_pulse))
            self._safe_write(
                self.packet_handler.write2ByteTxRx,
                self.port_handler, 6, 0x2A, g_pulse
            )
            time.sleep(delay_s)

        self._last_cmd_angles = target.copy()

    def emergency_stop(self) -> None:
        """急停 — 禁用所有舵机扭矩"""
        ph = self.port_handler
        for sid in range(1, 7):
            self._safe_write(
                self.packet_handler.write1ByteTxRx,
                ph, sid, 0x28, 0  # Torque_Enable = 0
            )
        self._connected = False
        logger.warning("Emergency stop triggered")

    # ------------------------------------------------------------------
    # 标定管理
    # ------------------------------------------------------------------
    def _load_calibration(self, path: str) -> dict:
        """从 JSON 加载标定，不存在则用默认值"""
        calib_path = Path(path)
        if calib_path.exists():
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 验证标定数据完整性
                for sid in range(1, 7):
                    key = str(sid)
                    if key not in data:
                        logger.warning("Calibration missing for servo %d, using default", sid)
                        data[key] = DEFAULT_CALIBRATION[key]
                logger.info("Calibration loaded from %s", calib_path)
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load calibration: %s, using defaults", e)
        else:
            logger.info("No calibration file at %s, using defaults", calib_path)
        return dict(DEFAULT_CALIBRATION)

    def save_calibration(self, path: str) -> None:
        """保存标定到 JSON"""
        calib_path = Path(path)
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(self.calibration, f, indent=2, ensure_ascii=False)
        logger.info("Calibration saved to %s", calib_path)

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------
    def close(self) -> None:
        """释放资源 + 清除单例（允许重建）"""
        try:
            self.emergency_stop()
        except Exception:
            pass
        try:
            self._force_reset()
        except Exception:
            pass
        with SO101Arm._lock:
            SO101Arm._instance = None
        logger.info("SO101Arm instance released")

    # ------------------------------------------------------------------
    # 上下文管理器支持
    # ------------------------------------------------------------------
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass
