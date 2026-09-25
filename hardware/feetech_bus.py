# hardware/feetech_bus.py
"""Feetech STS3215 串行总线协议层（自研，单型号精简设计）

设计依据 docs/pc_board_feetech_plan.md 阶段 A：
- 控制表驱动（data_name → (addr, length)），消灭魔数（G7）
  地址核对自 Feetech STS/SMS 系列手册与 lerobot (Apache-2.0) motors/feetech/tables.py
- connect(handshake=True)：逐 ID ping + 型号码校验，fail-fast（G1）
- configure()：STS3215 硬件坑位一次修齐——Phase bit4 清除防角度反馈溢出（G2）、
  Operating_Mode 显式 POSITION（G8）、Return_Delay_Time=0、夹爪防烧三参数（G4）
- write（带响应回执，配置用）与 sync_write（无回执，高频回路用）分工明确
- clamp_relative_goal：目标突变限幅（G3），策略流第二道防线
- 串口异常自动恢复（关闭→重开），继承自 SO101Arm._safe_write 的强于 lerobot 的设计
- 延迟导入 scservo_sdk：模块级仅 stdlib+numpy，PC 端无硬件亦可导入与单测

依赖: feetech-servo-sdk（提供 scservo_sdk）+ pyserial —— 两端均已部署实测。
"""
import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# STS3215 控制表（Feetech STS/SMS 系列，Protocol 0）
# data_name: (address, size_byte)
# EPROM 区（addr <= 39）写入需先 Torque_Enable=0 且 Lock=0
# ---------------------------------------------------------------------------
STS3215_TABLE: Dict[str, Tuple[int, int]] = {
    # EPROM（只读）
    "Firmware_Major_Version": (0, 1),
    "Firmware_Minor_Version": (1, 1),
    "Model_Number": (3, 2),
    # EPROM（可写，需解锁）
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Protection_Current": (28, 2),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Overload_Torque": (36, 1),
    # SRAM
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Velocity": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Velocity": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
    "Present_Current": (69, 2),
    # Factory 区
    "Maximum_Acceleration": (85, 1),
}

#: EPROM 可写寄存器（addr <= 39 且非只读）——写入前需 Torque off + Lock=0
EPROM_WRITABLE = frozenset(
    name for name, (addr, _) in STS3215_TABLE.items()
    if addr <= 39 and name not in
    ("Firmware_Major_Version", "Firmware_Minor_Version", "Model_Number")
)

#: sign-magnitude 编码位（bit 索引）
SIGN_BITS = {
    "Homing_Offset": 11,
    "Present_Load": 10,
    "Goal_Position": 15,
    "Goal_Velocity": 15,
    "Present_Position": 15,
    "Present_Velocity": 15,
}

RESOLUTION = 4096                 # STS3215 12-bit 编码器
MODEL_NUMBER_STS3215 = 777        # 型号码（握手校验用）
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_MOTOR_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

#: 出厂扫描波特率序列（低频→高频）
SCAN_BAUDRATES = [9_600, 19_200, 38_400, 57_600, 115_200,
                  128_000, 250_000, 500_000, 1_000_000]
#: Baud_Rate 寄存器值映射
BAUDRATE_TABLE = {
    1_000_000: 0, 500_000: 1, 250_000: 2, 128_000: 3,
    115_200: 4, 57_600: 5, 38_400: 6, 19_200: 7,
}

#: 运行模式（Operating_Mode 寄存器）
MODE_POSITION = 0
MODE_VELOCITY = 1
MODE_PWM = 2
MODE_STEP = 3

#: 夹爪防烧默认参数（lerobot so_follower.configure 同款取值）
GRIPPER_PROTECTION = {
    "Max_Torque_Limit": 500,     # 50% 最大扭矩
    "Protection_Current": 250,   # 50% 最大电流
    "Overload_Torque": 25,       # 过载时 25% 扭矩
}

#: Status 寄存器(65) 错误标志位（STS3215 手册；与社区实现 commanderfun/STS3215 一致）
STATUS_ERROR_FLAGS = {0: "Voltage", 1: "Sensor", 2: "Temperature",
                      3: "Current", 5: "Overload"}
#: Present_Current 单位: 6.5 mA/step（STS3215）
CURRENT_MA_PER_STEP = 6.5
#: Present_Voltage 单位: 0.1 V/step
VOLTAGE_V_PER_STEP = 0.1


# ---------------------------------------------------------------------------
# sign-magnitude 编解码（独立实现，逻辑为通用位运算）
# ---------------------------------------------------------------------------
def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    """有符号整数 → sign-magnitude 编码（bit[sign_bit] 为符号位）"""
    max_magnitude = (1 << sign_bit) - 1
    magnitude = abs(value)
    if magnitude > max_magnitude:
        raise ValueError(
            f"幅值 {magnitude} 超出 sign_bit={sign_bit} 上限 {max_magnitude}")
    direction = 1 if value < 0 else 0
    return (direction << sign_bit) | magnitude


def decode_sign_magnitude(encoded: int, sign_bit: int) -> int:
    """sign-magnitude 编码 → 有符号整数"""
    direction = (encoded >> sign_bit) & 1
    magnitude = encoded & ((1 << sign_bit) - 1)
    return -magnitude if direction else magnitude


def decode_status_flags(status_raw: int) -> List[str]:
    """Status 寄存器错误标志解码 → 错误名列表（空 = 健康）"""
    return [name for bit, name in STATUS_ERROR_FLAGS.items()
            if status_raw & (1 << bit)]


def decode_load(load_raw: int) -> Tuple[float, str]:
    """Present_Load 原始值（未经 sign 解码）→ (负载百分比 0-100, 方向 CW/CCW)

    bit10 = 方向（1=CW, 0=CCW），低 10 位 = 幅值（0-1000 → 0-100%）。
    抓取闭环判据: 夹到物体后负载/电流上升。
    """
    magnitude = load_raw & 0x3FF
    direction = "CW" if load_raw & 0x400 else "CCW"
    return magnitude / 10.0, direction


# ---------------------------------------------------------------------------
# SDK 超时 Bug 修复（feetech-servo-sdk v1.0.0，gitee IBY2S6）
# 与原 SO101Arm / lerobot 同源修复；官方新 SDK (FTServo_Python) 已修但未上 PyPI
# ---------------------------------------------------------------------------
def _patch_setPacketTimeout(self, packet_length):  # noqa: N802
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + \
                          (self.tx_time_per_byte * 3.0) + 50


class FeetechBus:
    """STS3215 串行总线封装（Protocol 0）

    用法::

        bus = FeetechBus("/dev/ttyACM0")
        bus.connect()                 # 含握手校验（G1）
        bus.configure(gripper_id=6)   # Phase/防烧/模式一次修齐（G2/G4/G8）
        pos = bus.sync_read("Present_Position")        # {id: raw}
        bus.sync_write("Goal_Position", {1: 2048, ...})  # 高频无回执
        bus.disconnect()              # 默认先禁扭矩

    高频控制回路用 sync_read/sync_write（无逐包回执）；
    配置/标定用 read/write（带回执 + 自动重试 + 串口恢复）。
    """

    def __init__(self, port: str,
                 motor_ids: Sequence[int] = DEFAULT_MOTOR_IDS,
                 baud: int = DEFAULT_BAUDRATE,
                 name: str = "feetech"):
        # 延迟导入：PC 端无 scservo_sdk 也能 import 本模块做纯逻辑单测
        try:
            import scservo_sdk as scs
        except ImportError as e:
            raise ImportError(
                "FeetechBus 需要 feetech-servo-sdk: pip install feetech-servo-sdk"
            ) from e
        self._scs = scs
        self.port = port
        self.baud = baud
        self.name = name
        self.motor_ids: Tuple[int, ...] = tuple(motor_ids)

        self.port_handler = scs.PortHandler(port)
        # monkey-patch 修复 SDK 超时计算 Bug
        self.port_handler.setPacketTimeout = \
            _patch_setPacketTimeout.__get__(self.port_handler, type(self.port_handler))
        self.packet_handler = scs.PacketHandler(0)  # Protocol 0 (STS/SCS)
        # SYNC 读写器：地址/长度在每次调用时动态设置（_setup_sync_*）
        self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)

        self._connected = False

    # ------------------------------------------------------------------
    # 地址解析（控制表驱动）
    # ------------------------------------------------------------------
    @staticmethod
    def addr(data_name: str) -> Tuple[int, int]:
        """data_name → (address, length)，未知名立即 KeyError（防魔数笔误）"""
        return STS3215_TABLE[data_name]

    # ------------------------------------------------------------------
    # 连接与握手
    # ------------------------------------------------------------------
    def connect(self, handshake: bool = True) -> None:
        """打开串口并配置波特率

        Args:
            handshake: True 时逐 ID ping + 型号码校验（G1 fail-fast）。
                       设备缺失/接线错误立即抛出，避免盲写串口。
        """
        if self._connected:
            logger.warning("[%s] 已连接，跳过", self.name)
            return
        if not self.port_handler.openPort():
            raise IOError(f"[{self.name}] 无法打开串口 {self.port}")
        if not self.port_handler.setBaudRate(self.baud):
            self.port_handler.closePort()
            raise IOError(f"[{self.name}] 无法设置波特率 {self.baud}")
        self._connected = True
        self._online: Optional[Dict[int, int]] = None
        logger.info("[%s] 串口 %s @ %d 已打开", self.name, self.port, self.baud)
        if handshake:
            self._online = self.do_handshake()

    @property
    def online_devices(self) -> Optional[Dict[int, int]]:
        """握手结果 {id: model_number}（未握手为 None）"""
        return self._online

    def do_handshake(self) -> Dict[int, int]:
        """逐 ID ping + 型号码校验（G1）

        Returns:
            {id: model_number} 在线设备表
        Raises:
            ConnectionError: 有 ID 无响应或型号码不符（报错列表化，指明缺失/错型）
        """
        found: Dict[int, int] = {}
        for mid in self.motor_ids:
            model = self.ping(mid, num_retry=2)
            if model is not None:
                found[mid] = model
        missing = [mid for mid in self.motor_ids if mid not in found]
        wrong_model = {mid: mv for mid, mv in found.items()
                       if mv != MODEL_NUMBER_STS3215}
        if missing or wrong_model:
            lines = [f"[{self.name}] 握手失败 @ {self.port}:"]
            if missing:
                lines.append(f"  无响应 ID: {missing}（检查接线/供电/ID 配置）")
            if wrong_model:
                lines.append(f"  型号码不符（期望 {MODEL_NUMBER_STS3215}=STS3215）: "
                             f"{wrong_model}")
            lines.append(f"  在线设备: {found}")
            raise ConnectionError("\n".join(lines))
        logger.info("[%s] 握手通过: %d 个 STS3215 在线 %s",
                    self.name, len(found), sorted(found))
        # G12: 固件版本一致性（混批只警告不中止——行为差异可由标定吸收）
        versions = self.firmware_versions(list(found))
        unique = set(versions.values())
        if len(unique) > 1:
            logger.warning("[%s] 舵机固件版本不一致（混批，行为可能有差异）: %s",
                           self.name, versions)
        else:
            logger.info("[%s] 固件版本: %s", self.name,
                        next(iter(unique)) if unique else "?")
        return found

    def disconnect(self, disable_torque: bool = True) -> None:
        """断开（默认先禁扭矩，防止断开后舵机持续堵转发热）"""
        if not self._connected:
            return
        if disable_torque:
            try:
                self.disable_torque()
            except Exception as e:
                logger.debug("[%s] 断开时禁扭矩失败（忽略）: %s", self.name, e)
        try:
            self.port_handler.closePort()
        except Exception:
            pass
        self._connected = False
        logger.info("[%s] 串口已关闭", self.name)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def ping(self, motor_id: int, num_retry: int = 0) -> Optional[int]:
        """ping 单个舵机，返回型号码或 None"""
        scs = self._scs
        for attempt in range(1 + num_retry):
            model, comm, error = self.packet_handler.ping(self.port_handler, motor_id)
            if comm == scs.COMM_SUCCESS and error == 0:
                return model
            logger.debug("[%s] ping id=%d 失败 (%d/%d): %s", self.name, motor_id,
                         attempt + 1, num_retry + 1,
                         self.packet_handler.getTxRxResult(comm))
        return None

    # ------------------------------------------------------------------
    # 串口异常恢复（继承 SO101Arm 强于 lerobot 的设计）
    # ------------------------------------------------------------------
    def reset_serial(self) -> None:
        """串口异常恢复：关闭 → 0.5s → 重开 + 重设波特率"""
        logger.warning("[%s] 串口重置 %s", self.name, self.port)
        try:
            self.port_handler.closePort()
        except Exception:
            pass
        time.sleep(0.5)
        if not self.port_handler.openPort():
            raise IOError(f"[{self.name}] 串口重开失败 {self.port}")
        if not self.port_handler.setBaudRate(self.baud):
            raise IOError(f"[{self.name}] 波特率重设失败")

    def _comm_ok(self, comm) -> bool:
        return comm == self._scs.COMM_SUCCESS

    # ------------------------------------------------------------------
    # 底层读写（控制表驱动，带重试 + 串口恢复）
    # ------------------------------------------------------------------
    def read(self, data_name: str, motor_id: int,
             num_retry: int = 2, decode_sign: bool = True) -> int:
        """单寄存器读取（带响应回执，配置/标定用）"""
        scs = self._scs
        addr, length = self.addr(data_name)
        fn = {1: self.packet_handler.read1ByteTxRx,
              2: self.packet_handler.read2ByteTxRx,
              4: self.packet_handler.read4ByteTxRx}[length]
        last_err = "unknown"
        for attempt in range(1 + num_retry):
            try:
                value, comm, error = fn(self.port_handler, motor_id, addr)
                if self._comm_ok(comm):
                    if decode_sign and data_name in SIGN_BITS:
                        value = decode_sign_magnitude(value, SIGN_BITS[data_name])
                    return value
                last_err = self.packet_handler.getTxRxResult(comm)
            except Exception as e:
                last_err = str(e)
            if attempt < num_retry:
                try:
                    self.reset_serial()
                except Exception as re_err:
                    logger.error("[%s] 串口恢复失败: %s", self.name, re_err)
        raise ConnectionError(
            f"[{self.name}] 读取 {data_name}(id={motor_id}) 失败: {last_err}")

    def write(self, data_name: str, motor_id: int, value: int,
              num_retry: int = 2, encode_sign: bool = True,
              unlock: bool = False) -> bool:
        """单寄存器写入（带响应回执确认写入成功，配置/标定用）

        Args:
            unlock: EEPROM 寄存器（addr<=39）需 Torque off + Lock=0；
                    True 时自动临时解锁并回锁（不改变扭矩状态语义由调用方保证）
        """
        scs = self._scs
        addr, length = self.addr(data_name)
        if encode_sign and data_name in SIGN_BITS and value < 0:
            value = encode_sign_magnitude(value, SIGN_BITS[data_name])
        if value < 0:
            raise ValueError(f"{data_name} 不支持负值 {value}（非 sign-magnitude 寄存器）")
        data = self._split_bytes(value, length)

        need_unlock = unlock or (data_name in EPROM_WRITABLE)
        locked_back = False
        if need_unlock:
            # EEPROM 写入需 Lock=0（且扭矩关闭由调用流程保证）
            self._write_raw(*self.addr("Lock"), motor_id, [0])
            locked_back = True
        try:
            last_err = "unknown"
            for attempt in range(1 + num_retry):
                try:
                    comm, error = self.packet_handler.writeTxRx(
                        self.port_handler, motor_id, addr, length, data)
                    if self._comm_ok(comm):
                        return True
                    last_err = self.packet_handler.getTxRxResult(comm)
                except Exception as e:
                    last_err = str(e)
                if attempt < num_retry:
                    try:
                        self.reset_serial()
                    except Exception as re_err:
                        logger.error("[%s] 串口恢复失败: %s", self.name, re_err)
            logger.error("[%s] 写入 %s(id=%d)=%s 失败: %s",
                         self.name, data_name, motor_id, value, last_err)
            return False
        finally:
            if locked_back:
                try:
                    self._write_raw(*self.addr("Lock"), motor_id, [1])
                except Exception:
                    pass

    def _write_raw(self, addr: int, length: int, motor_id: int,
                   data: List[int]) -> Tuple:
        return self.packet_handler.writeTxRx(
            self.port_handler, motor_id, addr, length, data)

    def _split_bytes(self, value: int, length: int) -> List[int]:
        scs = self._scs
        if length == 1:
            return [value & 0xFF]
        if length == 2:
            return [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
        if length == 4:
            return [scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
                    scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
                    scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
                    scs.SCS_HIBYTE(scs.SCS_HIWORD(value))]
        raise ValueError(f"不支持的寄存器长度: {length}")

    # ------------------------------------------------------------------
    # SYNC 批量读写（高频回路用）
    # ------------------------------------------------------------------
    def sync_read(self, data_name: str,
                  motor_ids: Optional[Sequence[int]] = None,
                  num_retry: int = 1,
                  raise_on_error: bool = False) -> Dict[int, int]:
        """SYNC_READ 批量读取同一寄存器

        Returns:
            {motor_id: raw_value}；个别舵机无响应时缺省该键
            （raise_on_error=False 时部分成功也返回）
        """
        scs = self._scs
        ids = list(motor_ids) if motor_ids else list(self.motor_ids)
        addr, length = self.addr(data_name)
        self.sync_reader.clearParam()
        self.sync_reader.start_address = addr
        self.sync_reader.data_length = length
        for mid in ids:
            self.sync_reader.addParam(mid)

        comm = None
        for attempt in range(1 + num_retry):
            comm = self.sync_reader.txRxPacket()
            if self._comm_ok(comm):
                break
            logger.debug("[%s] sync_read %s 失败 (%d/%d)", self.name,
                         data_name, attempt + 1, num_retry + 1)
        if not self._comm_ok(comm):
            if raise_on_error:
                raise ConnectionError(
                    f"[{self.name}] sync_read {data_name} 失败: "
                    f"{self.packet_handler.getTxRxResult(comm)}")
            self.sync_reader.clearParam()
            return {}

        result: Dict[int, int] = {}
        for mid in ids:
            try:
                if self.sync_reader.isAvailable(mid, addr, length):
                    result[mid] = self.sync_reader.getData(mid, addr, length)
            except Exception:
                continue
        self.sync_reader.clearParam()
        return result

    def sync_write(self, data_name: str, values: Dict[int, int],
                   num_retry: int = 1) -> bool:
        """SYNC_WRITE 批量写入（无回执，高频回路用；丢包由下帧覆盖）"""
        scs = self._scs
        addr, length = self.addr(data_name)
        self.sync_writer.clearParam()
        self.sync_writer.start_address = addr
        self.sync_writer.data_length = length
        for mid, value in values.items():
            self.sync_writer.addParam(mid, self._split_bytes(int(value), length))
        for attempt in range(1 + num_retry):
            comm = self.sync_writer.txPacket()
            if self._comm_ok(comm):
                return True
            logger.debug("[%s] sync_write %s 失败 (%d/%d)", self.name,
                         data_name, attempt + 1, num_retry + 1)
        logger.error("[%s] sync_write %s 最终失败", self.name, data_name)
        return False

    # ------------------------------------------------------------------
    # 扭矩管理
    # ------------------------------------------------------------------
    def enable_torque(self, motor_ids: Optional[Sequence[int]] = None) -> None:
        """恢复扭矩（防突跳：先把 Goal_Position 对齐当前 Present_Position）

        禁扭矩期间臂可能被手搬（标定场景），Goal 停留在旧值——若不直接
        对齐，恢复扭矩瞬间舵机会冲向旧目标。Present/Goal 同为 raw 编码域，
        直接透传无需换算。
        """
        ids = list(motor_ids) if motor_ids else list(self.motor_ids)
        present = self.sync_read("Present_Position", motor_ids=ids, num_retry=1)
        if present:
            self.sync_write("Goal_Position", present, num_retry=1)
        for mid in ids:
            self.write("Torque_Enable", mid, 1, num_retry=1)
            self.write("Lock", mid, 1, num_retry=1)

    def disable_torque(self, motor_ids: Optional[Sequence[int]] = None) -> None:
        """禁扭矩 + 解锁 EPROM（Lock=0），可手搬关节/写标定"""
        ids = motor_ids or self.motor_ids
        for mid in ids:
            self.write("Torque_Enable", mid, 0, num_retry=1)
            self.write("Lock", mid, 0, num_retry=1)

    @contextmanager
    def torque_disabled(self, motor_ids: Optional[Sequence[int]] = None):
        """上下文管理器：保证退出时恢复扭矩（配置/标定安全包裹）"""
        self.disable_torque(motor_ids)
        try:
            yield
        finally:
            self.enable_torque(motor_ids)

    # ------------------------------------------------------------------
    # 推荐配置（G2/G4/G8 + Return_Delay 地址 Bug 修复）
    # ------------------------------------------------------------------
    def configure(self, gripper_id: Optional[int] = 6,
                  pid: Optional[Dict[str, int]] = None,
                  acceleration: int = 16,
                  return_delay: int = 0) -> None:
        """一次性写入推荐配置（在禁扭矩+解锁上下文中执行）

        修复项:
          - G2: STS3215 Phase 寄存器 bit4 清除 → 角度反馈强制 [0,4095]，防溢出为负
          - G8: Operating_Mode 显式写 POSITION（防外部工具改过模式）
          - G4: 夹爪防烧三参数（Max_Torque_Limit/Protection_Current/Overload_Torque）
          - 修复原 arm.py 地址对调 Bug：Return_Delay_Time 实为 addr 7（原误写 0x29=
            Acceleration），Acceleration 实为 addr 41（原误写 0x1A=CW_Dead_Zone）

        Args:
            gripper_id: 夹爪舵机 ID（None 跳过防烧参数）
            pid: 可选 {"P_Coefficient": p, "I_Coefficient": i, "D_Coefficient": d}
            acceleration: 加速度（原 arm.py 意图值 16 = 平滑加速）
            return_delay: 响应延迟（0 = 最小 2µs，lerobot 同款）
        """
        with self.torque_disabled():
            for mid in self.motor_ids:
                self.write("Return_Delay_Time", mid, return_delay, num_retry=1)
                # G2: STS3215 Phase bit4 → 位置读数落在 [0, resolution-1]
                phase = self.read("Phase", mid, num_retry=1)
                if phase & 0x10:
                    self.write("Phase", mid, phase & ~0x10, num_retry=1)
                    logger.info("[%s] id=%d Phase bit4 已清除（防角度反馈溢出）",
                                self.name, mid)
                self.write("Operating_Mode", mid, MODE_POSITION, num_retry=1)
                self.write("Acceleration", mid, acceleration, num_retry=1)
                if pid:
                    for key, val in pid.items():
                        self.write(key, mid, val, num_retry=1)
            if gripper_id is not None and gripper_id in self.motor_ids:
                for key, val in GRIPPER_PROTECTION.items():
                    self.write(key, gripper_id, val, num_retry=1)
                logger.info("[%s] 夹爪 id=%d 防烧参数已写入 %s",
                            self.name, gripper_id, GRIPPER_PROTECTION)

    # ------------------------------------------------------------------
    # 标定读写（G5/G6，阶段 B tools/calibrate_arm.py 使用）
    # ------------------------------------------------------------------
    def read_calibration(self) -> Dict[int, Dict[str, int]]:
        """从舵机 EEPROM 读回标定 {id: {homing_offset, range_min, range_max}}"""
        calib = {}
        for mid in self.motor_ids:
            calib[mid] = {
                "homing_offset": self.read("Homing_Offset", mid, num_retry=1),
                "range_min": self.read("Min_Position_Limit", mid, num_retry=1),
                "range_max": self.read("Max_Position_Limit", mid, num_retry=1),
            }
        return calib

    def write_calibration(self, calib: Dict[int, Dict[str, int]]) -> None:
        """标定写入舵机 EEPROM（需先禁扭矩；Homing_Offset 走 sign-magnitude bit11）"""
        with self.torque_disabled():
            for mid, c in calib.items():
                self.write("Homing_Offset", mid, int(c["homing_offset"]), num_retry=1)
                self.write("Min_Position_Limit", mid, int(c["range_min"]), num_retry=1)
                self.write("Max_Position_Limit", mid, int(c["range_max"]), num_retry=1)

    # ------------------------------------------------------------------
    # 安全限幅（G3）
    # ------------------------------------------------------------------
    @staticmethod
    def clamp_relative_goal(goal: Dict[int, float],
                            present: Dict[int, float],
                            max_delta: float) -> Dict[int, float]:
        """目标突变限幅：单帧变化超过 max_delta 时截断到 present ± max_delta

        纯函数（无 IO），供策略流（execute/teleop）做总线级第二道防线。
        present 缺失的 ID 不限幅（首帧场景）。
        """
        if max_delta is None or max_delta <= 0:
            return dict(goal)
        clamped = {}
        for mid, target in goal.items():
            if mid in present:
                cur = present[mid]
                delta = target - cur
                if abs(delta) > max_delta:
                    clamped[mid] = cur + max_delta if delta > 0 else cur - max_delta
                    continue
            clamped[mid] = target
        return clamped

    # ------------------------------------------------------------------
    # 诊断与健康监测（对照社区实现 commanderfun/STS3215 Servo 类）
    # ------------------------------------------------------------------
    def firmware_versions(self, motor_ids: Optional[Sequence[int]] = None
                          ) -> Dict[int, str]:
        """读各 ID 固件版本 {id: "major.minor"}（G12）"""
        ids = list(motor_ids) if motor_ids else list(self.motor_ids)
        versions: Dict[int, str] = {}
        for mid in ids:
            try:
                major = self.read("Firmware_Major_Version", mid, num_retry=1)
                minor = self.read("Firmware_Minor_Version", mid, num_retry=1)
                versions[mid] = f"{major}.{minor}"
            except ConnectionError:
                versions[mid] = "?"
        return versions

    def read_diagnostics(self, motor_ids: Optional[Sequence[int]] = None
                         ) -> Dict[int, dict]:
        """读取总线健康诊断（只读，不驱动任何运动）

        Returns:
            {id: {"errors": [错误名], "temperature": °C|None,
                  "voltage": V|None, "current_mA": float|None,
                  "load": (百分比, "CW"/"CCW")|None, "moving": bool}}
        """
        ids = list(motor_ids) if motor_ids else list(self.motor_ids)
        status = self.sync_read("Status", motor_ids=ids, num_retry=1)
        temp = self.sync_read("Present_Temperature", motor_ids=ids, num_retry=1)
        volt = self.sync_read("Present_Voltage", motor_ids=ids, num_retry=1)
        cur = self.sync_read("Present_Current", motor_ids=ids, num_retry=1)
        load = self.sync_read("Present_Load", motor_ids=ids, num_retry=1)
        moving = self.sync_read("Moving", motor_ids=ids, num_retry=1)
        diag: Dict[int, dict] = {}
        for mid in ids:
            diag[mid] = {
                "errors": decode_status_flags(status.get(mid, 0)),
                "temperature": temp.get(mid),
                "voltage": round(volt[mid] * VOLTAGE_V_PER_STEP, 1)
                           if mid in volt else None,
                "current_mA": round(cur[mid] * CURRENT_MA_PER_STEP, 1)
                              if mid in cur else None,
                "load": decode_load(load[mid]) if mid in load else None,
                "moving": bool(moving.get(mid, 0)),
            }
        return diag

    def wait_until_stopped(self, targets: Dict[int, int],
                           tolerance_raw: int = 10,
                           timeout_s: float = 5.0,
                           poll_hz: float = 50) -> bool:
        """轮询等待到位（move_sync 语义）: 各舵机 Moving=0 且位置进入容差

        Args:
            targets: {id: 目标 raw}
            tolerance_raw: 到位容差（编码器步）
        Returns:
            True=全部到位, False=超时（调用方决定降级策略）
        """
        ids = list(targets)
        deadline = time.perf_counter() + timeout_s
        period = 1.0 / poll_hz
        while time.perf_counter() < deadline:
            pos = self.sync_read("Present_Position", motor_ids=ids, num_retry=0)
            moving = self.sync_read("Moving", motor_ids=ids, num_retry=0)
            if pos and all(
                (not moving.get(mid, 0))
                and abs(pos.get(mid, targets[mid]) - targets[mid]) <= tolerance_raw
                for mid in ids
            ):
                return True
            time.sleep(period)
        return False

    # ------------------------------------------------------------------
    # 广播 ping 与扫描（阶段 D 工具用；裸协议解析独立实现）
    # ------------------------------------------------------------------
    def broadcast_ping(self) -> Dict[int, int]:
        """广播 ping：返回 {id: error_status}（Feetech Protocol 0 裸包解析）"""
        scs = self._scs
        data_list: Dict[int, int] = {}
        status_length = 6
        wait_length = status_length * scs.MAX_ID
        tx_time_per_byte = (1000.0 / self.port_handler.getBaudRate()) * 10.0

        txpacket = [0] * 6
        txpacket[scs.PKT_ID] = scs.BROADCAST_ID
        txpacket[scs.PKT_LENGTH] = 2
        txpacket[scs.PKT_INSTRUCTION] = scs.INST_PING
        if self.packet_handler.txPacket(self.port_handler, txpacket) != scs.COMM_SUCCESS:
            self.port_handler.is_using = False
            return data_list
        self.port_handler.setPacketTimeoutMillis(
            (wait_length * tx_time_per_byte) + (3.0 * scs.MAX_ID) + 16.0)

        rxpacket: List[int] = []
        rx_length = 0
        while not self.port_handler.isPacketTimeout() and rx_length < wait_length:
            rxpacket += self.port_handler.readPort(wait_length - rx_length)
            rx_length = len(rxpacket)
        self.port_handler.is_using = False
        if rx_length == 0:
            return data_list

        while rx_length >= status_length:
            # 定位包头 0xFF 0xFF
            idx = 0
            while idx < rx_length - 1 and not (rxpacket[idx] == 0xFF
                                               and rxpacket[idx + 1] == 0xFF):
                idx += 1
            if idx >= rx_length - 1:
                break
            if idx > 0:
                del rxpacket[:idx]
                rx_length = len(rxpacket)
                continue
            checksum = ~sum(rxpacket[2:status_length - 1]) & 0xFF
            if rxpacket[status_length - 1] == checksum:
                data_list[rxpacket[scs.PKT_ID]] = rxpacket[scs.PKT_ERROR]
                del rxpacket[:status_length]
                rx_length = len(rxpacket)
            else:
                del rxpacket[:2]
                rx_length = len(rxpacket)
        return data_list

    def scan_baudrates(self, baudrates: Optional[Sequence[int]] = None
                       ) -> Dict[int, Dict[int, int]]:
        """多波特率扫描：{baudrate: {id: model_number}}（出厂调试用）"""
        result: Dict[int, Dict[int, int]] = {}
        for baud in (baudrates or SCAN_BAUDRATES):
            try:
                self.port_handler.setBaudRate(baud)
            except Exception:
                continue
            ids = self.broadcast_ping()
            if ids:
                models = {}
                for mid in ids:
                    mv, comm, _ = self.packet_handler.ping(self.port_handler, mid)
                    if self._comm_ok(comm):
                        models[mid] = mv
                result[baud] = models or dict(ids)
        # 恢复目标波特率
        try:
            self.port_handler.setBaudRate(self.baud)
        except Exception:
            pass
        return result

    def setup_motor(self, initial_id: int, target_id: int,
                    target_baud: int = DEFAULT_BAUDRATE) -> bool:
        """单电机出厂初始化：改 ID + 波特率（需单独连接该电机）"""
        with self.torque_disabled([initial_id]):
            ok = self.write("ID", initial_id, target_id, num_retry=1)
            if ok and target_baud in BAUDRATE_TABLE:
                ok = self.write("Baud_Rate", target_id,
                                BAUDRATE_TABLE[target_baud], num_retry=1)
        return ok

    # ------------------------------------------------------------------
    # 换算纯函数（与 SO101Arm 历史公式保持一致，回归防护见单测）
    # ------------------------------------------------------------------
    @staticmethod
    def raw_to_rad(raw: float, mid: float) -> float:
        """编码器 raw → 弧度（(raw-mid) * 2π / 4095）"""
        return float(np.deg2rad((raw - mid) * 360.0 / 4095.0))

    @staticmethod
    def rad_to_raw(rad: float, mid: float,
                   range_min: int = 0, range_max: int = 4095) -> int:
        """弧度 → 编码器 raw（clamp 到 [range_min, range_max]）"""
        raw = int(np.rad2deg(rad) * 4095.0 / 360.0 + mid)
        return max(range_min, min(range_max, raw))

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.disconnect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
