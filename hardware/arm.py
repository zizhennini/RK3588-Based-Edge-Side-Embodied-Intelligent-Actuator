# hardware/arm.py
"""SO-ARM101 机械臂控制 -- 基于 scservo_sdk 封装

协议层委托 hardware.feetech_bus.FeetechBus（控制表驱动 + 握手校验 + STS3215
硬件坑位修复），本模块聚焦臂语义：标定换算、IK 运动、归零、夹爪、安全限幅、
线程安全单例。公开 API 与重构前完全兼容（main.py / grasp_pipeline / safety
等调用点零改动）。
"""
import time
import threading
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from hardware.interfaces import HardwareModule, Observation
from hardware.feetech_bus import (FeetechBus, GRIPPER_MOTOR_ID, angle_zero,
                                  rad_to_raw, raw_to_rad)

logger = logging.getLogger(__name__)


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


class SO101Arm(HardwareModule):
    """SO-ARM101 机械臂控制 -- 协议层 FeetechBus，臂语义本类

    线程安全单例模式，防止多实例争抢串口。
    串口写入统一经 FeetechBus（自动重试 + 串口恢复）。

    实现 HardwareModule 接口 (refactor_plan_v9 §4.2):
      - setup/start/stop/is_available/on_failure: 生命周期
      - execute(Action): 关节空间执行（write_positions + gripper），
        可选 G3 帧间突变限幅（max_relative_step_deg）
      - get_observation(): 返回仅含 state 的 Observation（rgb/depth 由 System 从相机合并）

    kinematics 由组合根 (main.py) 注入，硬件层不反向依赖策略层（修复依赖倒置）。
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
                 calibration_path: str = "./config/calibration.json",
                 kinematics=None,
                 max_relative_step_deg: Optional[float] = None):
        # 串口协议层（延迟连接；scservo_sdk 缺失时此处即报 ImportError）
        self.port = port
        self.baud = baud
        self.bus = FeetechBus(port, baud=baud, name="arm")

        # 标定
        self.calibration = self._load_calibration(calibration_path)

        # 外参：读取单一事实来源 config/settings.py（修复配置硬编码债）
        # 标定脚本 calibrate_extrinsics.py / calibrate_camera.py 只回写 settings.py，
        # 故此处必须引用而非复制字面量，否则标定值会静默漂移。
        try:
            from config import settings as _settings
            self.camera_position = np.asarray(_settings.CAMERA_POSITION, dtype=float)
        except Exception:
            logger.warning("无法读取 config.settings.CAMERA_POSITION，回退默认外参")
            self.camera_position = np.array([0.182, -0.129, 0.47])

        # kinematics：由组合根 (main.py) 注入；硬件层不 import 策略层（修复依赖倒置债）
        self._kinematics = kinematics

        # 状态
        self._last_cmd_angles: Optional[np.ndarray] = None
        self._connected = False

        # G3: execute() 帧间突变限幅（度/帧）。None=关闭（默认，保持历史行为）；
        # 策略流（ACT 30Hz）建议 15~30，遥操作跟随由 teleop 层自行限幅。
        self.max_relative_step_deg = max_relative_step_deg

    # ------------------------------------------------------------------
    # HardwareModule 接口 (refactor_plan_v9 §4.2)
    # ------------------------------------------------------------------
    def set_kinematics(self, kinematics) -> None:
        """注入运动学解算器（由组合根提供，避免硬件层反向依赖策略层）"""
        self._kinematics = kinematics

    def setup(self, config: dict) -> None:
        """配置模块。支持 config 键: kinematics, calibration_path,
        max_relative_step_deg"""
        if not config:
            return
        if config.get("kinematics") is not None:
            self._kinematics = config["kinematics"]
        calib = config.get("calibration_path")
        if calib:
            self.calibration = self._load_calibration(calib)
        if config.get("max_relative_step_deg") is not None:
            self.max_relative_step_deg = float(config["max_relative_step_deg"])

    def start(self) -> None:
        """启动模块（= 连接串口）"""
        self.connect()

    def stop(self) -> None:
        """停止模块（= 断开串口，禁用扭矩）"""
        self.disconnect()

    @property
    def is_available(self) -> bool:
        """机械臂是否已连接可用"""
        return self._connected

    def on_failure(self) -> str:
        """机械臂为硬性依赖，失败即中止（与 main.py init_hardware 语义一致）"""
        return "abort"

    def execute(self, action) -> bool:
        """执行动作：关节空间位置写入 + 夹爪开合

        G3 防线：max_relative_step_deg 配置时，对上一帧指令的突变做截断
        （仅策略流路径生效；move_to/home 为受控插值不限幅）。

        Args:
            action: Action(positions=(6,) rad, gripper=[0=全闭,1=全开], ...)
        Returns:
            是否成功
        """
        try:
            positions = np.asarray(action.positions, dtype=float)
            step = self.max_relative_step_deg
            if step is not None and self._last_cmd_angles is not None:
                max_delta = float(np.deg2rad(step))
                delta = positions - self._last_cmd_angles
                clipped = np.clip(delta, -max_delta, max_delta)
                if not np.allclose(delta, clipped):
                    logger.warning(
                        "execute 目标突变被限幅（G3）: max|Δ|=%.1f° > %.1f°",
                        float(np.rad2deg(np.abs(delta).max())), step)
                    positions = self._last_cmd_angles + clipped
            self.write_positions(positions)
            # gripper [0=全闭, 1=全开] → 米制宽度（量程约 0~0.08m）
            self.gripper_width(float(action.gripper) * 0.08)
            return True
        except Exception as e:
            logger.error("SO101Arm.execute 失败: %s", e)
            return False

    def get_observation(self) -> Observation:
        """返回仅含关节状态的 Observation

        机械臂无视觉传感器，rgb/depth 置 None，由 System 层与相机观测合并。
        """
        return Observation(
            rgb=None, depth=None,
            state=self.read_positions(), timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    def connect(self, handshake: bool = True) -> None:
        """打开串口、握手校验（G1）并写入推荐配置（G2/G4/G8）

        Args:
            handshake: True（默认）时逐 ID ping + 型号码校验，设备缺失立即
                       抛出 ConnectionError（fail-fast，on_failure=abort）。
        """
        if self._connected:
            logger.warning("Already connected")
            return
        self.bus.connect(handshake=handshake)
        # STS3215 推荐配置：Phase bit4 防溢出、POSITION 模式、夹爪防烧、
        # Return_Delay=0、Acceleration=16（修复原地址对调 Bug，见 feetech_bus.configure）
        self.bus.configure(gripper_id=MOTOR_IDS["gripper"])
        self._connected = True

    def disconnect(self) -> None:
        """禁用扭矩并关闭串口"""
        if not self._connected:
            return
        try:
            self.bus.disconnect(disable_torque=True)
        except Exception:
            pass
        self._connected = False
        logger.info("Serial port closed")

    # ------------------------------------------------------------------
    # 位置读写
    # ------------------------------------------------------------------
    def _calib_mid(self, sid: int) -> float:
        """关节角度零点（raw 值域）

        体关节(1-5) 用行程中点 (range_min+range_max)/2 —— lerobot DEGREES 官方语义；
        夹爪(6) 沿用标定中位点 homing_offset（0 rad = 夹爪闭合位，本实现既有语义）。
        """
        return angle_zero(self.calibration[str(sid)],
                          use_range_midpoint=(sid != GRIPPER_MOTOR_ID))

    def _raw_to_rad(self, sid: int, raw: float) -> float:
        return raw_to_rad(raw, self.calibration[str(sid)],
                          use_range_midpoint=(sid != GRIPPER_MOTOR_ID))

    def _rad_to_raw(self, sid: int, rad: float) -> int:
        return rad_to_raw(rad, self.calibration[str(sid)],
                          use_range_midpoint=(sid != GRIPPER_MOTOR_ID))

    def read_positions(self) -> np.ndarray:
        """SYNC_READ 批量读取 6 个舵机位置，返回弧度数组 (6,)

        个别舵机无响应时该关节回退 0.0（保持历史容错行为）。
        """
        angles = np.zeros(6)
        raw_map = self.bus.sync_read("Present_Position", num_retry=1)
        for sid in range(1, 7):
            raw = raw_map.get(sid)
            if raw is None:
                logger.debug("Failed to read servo %d", sid)
                continue
            angles[sid - 1] = self._raw_to_rad(sid, raw)
        return angles

    def write_positions(self, angles_rad: np.ndarray) -> None:
        """SYNC_WRITE 批量写入 6 个关节角度（弧度，标定限位 clamp）"""
        raws = {sid: self._rad_to_raw(sid, angles_rad[sid - 1])
                for sid in range(1, 7)}
        self.bus.sync_write("Goal_Position", raws)
        self._last_cmd_angles = np.asarray(angles_rad, dtype=float).copy()

    # ------------------------------------------------------------------
    # 运动控制
    # ------------------------------------------------------------------
    def move_to(self, x: float, y: float, z: float,
                wrist_roll_rad: Optional[float] = None,
                wait: bool = False, wait_timeout_s: float = 5.0) -> None:
        """笛卡尔 IK 运动

        Args:
            x, y, z: 目标笛卡尔坐标（米）
            wrist_roll_rad: 腕部旋转角度（弧度），None 则保持当前值
            wait: True 时插值下发后轮询等待各关节到位（Moving=0 且进入
                  容差），超时仅告警不抛出（move_sync 语义）
            wait_timeout_s: 等待超时（秒）

        Note:
            需先注入 kinematics（构造参数或 set_kinematics）。硬件层不再
            import policy.kinematics，规避 hardware→policy 依赖倒置。
        """
        kin = self._kinematics
        if kin is None:
            raise RuntimeError(
                "SO101Arm.move_to 需要先注入 Kinematics 实例"
                "（SO101Arm(kinematics=...) 或 set_kinematics()）；"
                "硬件层不再反向依赖 policy.kinematics。"
            )
        xyz = kin.clamp_workspace(np.array([x, y, z]))
        current = self._last_cmd_angles if self._last_cmd_angles is not None \
            else self.read_positions()
        angles_rad = kin.inverse_kinematics(xyz, current, wrist_roll_rad)
        self._last_cmd_angles = angles_rad.copy()

        # 写入前 5 个关节（不含夹爪）
        raws = {sid: self._rad_to_raw(sid, angles_rad[sid - 1])
                for sid in range(1, 6)}
        self.bus.sync_write("Goal_Position", raws)
        if wait and not self.bus.wait_until_stopped(raws, timeout_s=wait_timeout_s):
            logger.warning("move_to 等待到位超时（%.1fs），部分关节可能未达目标",
                           wait_timeout_s)

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
        pulse = GRIPPER_OPEN_PULSE if open else GRIPPER_CLOSE_PULSE
        self.bus.write("Goal_Position", MOTOR_IDS["gripper"], pulse)

    def gripper_width(self, width_m: float) -> None:
        """自适应夹爪宽度（米制转脉冲）

        Args:
            width_m: 夹爪开合宽度（米），范围约 [0.0, 0.08]
        """
        calib = self.calibration[str(MOTOR_IDS["gripper"])]
        # 将米制宽度线性映射到脉冲范围
        # 假设全开 ~0.08m 对应 range_max, 全闭 0m 对应 range_min
        max_width = 0.08  # 最大开合宽度（米）
        width_m = max(0.0, min(max_width, width_m))
        ratio = width_m / max_width
        pulse = int(calib["range_min"] + ratio * (calib["range_max"] - calib["range_min"]))
        pulse = max(calib["range_min"], min(calib["range_max"], pulse))
        self.bus.write("Goal_Position", MOTOR_IDS["gripper"], pulse)

    def gripper_current(self):
        """夹爪电流与负载（抓取闭环判据：夹到物体后电流/负载上升）

        Returns:
            (current_mA, load_percent) 元组；读取失败返回 None
        """
        gid = MOTOR_IDS["gripper"]
        diag = self.bus.read_diagnostics([gid]).get(gid)
        if not diag or diag["current_mA"] is None or diag["load"] is None:
            return None
        return diag["current_mA"], diag["load"][0]

    def diagnostics(self) -> dict:
        """全臂健康诊断透传（错误标志/温度/电压/电流/负载/运动状态）"""
        return self.bus.read_diagnostics()

    def home(self, steps: int = 50, delay_s: float = 0.02) -> None:
        """归零（插值平滑）

        Args:
            steps: 插值步数
            delay_s: 每步间隔（秒）
        """
        current = self.read_positions()
        target = HOME_POSE
        gripper_id = MOTOR_IDS["gripper"]
        calib_g = self.calibration[str(gripper_id)]

        for i in range(1, steps + 1):
            t = i / steps
            angles = current * (1 - t) + target * t

            # 写入前 5 个关节
            raws = {sid: self._rad_to_raw(sid, angles[sid - 1])
                    for sid in range(1, 6)}
            self.bus.sync_write("Goal_Position", raws)

            # 夹爪插值
            g_pulse = int(np.interp(
                angles[5],
                [JOINT_LIMITS["gripper"][0], JOINT_LIMITS["gripper"][1]],
                [calib_g["range_min"], calib_g["range_max"]]
            ))
            g_pulse = max(calib_g["range_min"], min(calib_g["range_max"], g_pulse))
            self.bus.write("Goal_Position", gripper_id, g_pulse)
            time.sleep(delay_s)

        self._last_cmd_angles = target.copy()

    def emergency_stop(self) -> None:
        """急停 — 禁用所有舵机扭矩"""
        try:
            self.bus.disable_torque()
        except Exception as e:
            logger.error("急停禁扭矩失败: %s", e)
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
    def _force_reset(self) -> None:
        """强制重置串口状态"""
        try:
            self.bus.port_handler.closePort()
        except Exception:
            pass
        self.bus._connected = False
        self._connected = False

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
