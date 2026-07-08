"""SO-ARM101 控制封装 — URDF 运动学 + Feetech 串口"""
import struct
import time
import serial
import numpy as np
from config.cpu_affinity import bind_current_thread, LITTLE_CORES
from vla.kinematics import Kinematics, JOINT_NAMES, JOINT_LIMITS, WORKSPACE, HOME_POSE


class ArmController:
    """使用几何 IK + Feetech 串口控制 SO-ARM101"""

    JOINT_NAMES = JOINT_NAMES
    JOINT_LIMITS = JOINT_LIMITS
    WORKSPACE = WORKSPACE
    HOME_POSE = HOME_POSE

    CALIB = {
        1: {"homing_offset": 2048, "range_min": 946, "range_max": 3287},
        2: {"homing_offset": 2048, "range_min": 821, "range_max": 3206},
        3: {"homing_offset": 2048, "range_min": 888, "range_max": 3105},
        4: {"homing_offset": 2048, "range_min": 851, "range_max": 3192},
        5: {"homing_offset": 2048, "range_min": 130, "range_max": 3985},
        6: {"homing_offset": 1781, "range_min": 1495, "range_max": 2860},
    }

    # 相机 → 机械臂基座外参（用于将相机坐标转为机械臂坐标）
    CAMERA_POSITION = np.array([0.0, 0.21, 0.40], dtype=float)

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud: int = 1000000,
        urdf_path: str = "./models/so101_urdf/so101_new_calib.urdf",
        bind_little: bool = True,
    ):
        bind_current_thread(LITTLE_CORES)
        self.ser = serial.Serial(port, baud, timeout=0.005)
        self._init_ik(urdf_path)
        self._configure_motors()

    def _init_ik(self, urdf_path: str):
        """初始化运动学求解器"""
        self.kin = Kinematics()

    def _configure_motors(self):
        for sid in range(1, 7):
            cmd = struct.pack("<BBBBBBB", 0xFF, 0xFF, sid, 4, 0x03, 0x29, 100)
            cks = (~sum(cmd[2:]) & 0xFF)
            self.ser.write(cmd + struct.pack("<B", cks))
            time.sleep(0.01)
            cmd = struct.pack("<BBBBBBB", 0xFF, 0xFF, sid, 4, 0x03, 0x1A, 16)
            cks = (~sum(cmd[2:]) & 0xFF)
            self.ser.write(cmd + struct.pack("<B", cks))
            time.sleep(0.01)

    # ── 关节限位/工作空间 ──

    def _clamp_joints(self, angles_rad: np.ndarray) -> np.ndarray:
        for i, name in enumerate(self.JOINT_NAMES):
            low, high = self.JOINT_LIMITS[name]
            angles_rad[i] = np.clip(angles_rad[i], low, high)
        return angles_rad

    def clamp_workspace(self, xyz: np.ndarray) -> np.ndarray:
        ws = self.WORKSPACE
        xyz[0] = np.clip(xyz[0], ws[0], ws[1])
        xyz[1] = np.clip(xyz[1], ws[2], ws[3])
        xyz[2] = np.clip(xyz[2], ws[4], ws[5])
        return xyz

    # ── 运动控制 ──

    def move_to(self, x: float, y: float, z: float, wrist_roll_rad: float | None = None, use_current=True):
        xyz = Kinematics.clamp(np.array([x, y, z]))
        if hasattr(self, '_last_cmd_angles') and not use_current:
            current = self._last_cmd_angles
        else:
            current = self._read_current_pos()
        angles_rad = Kinematics.ik(xyz, current)
        if wrist_roll_rad is not None:
            angles_rad[4] = np.clip(wrist_roll_rad,
                JOINT_LIMITS["wrist_roll"][0],
                JOINT_LIMITS["wrist_roll"][1])
        self._last_cmd_angles = angles_rad.copy()
        for sid in range(1, 6):
            self._write_angle(sid, float(angles_rad[sid - 1]))

    def home(self, steps: int = 50, delay_s: float = 0.02):
        current = np.zeros(6)
        for sid in range(1, 7):
            self.ser.reset_input_buffer()
            pkt = bytearray([0xFF, 0xFF, sid, 4, 0x02, 0x38, 0x02])
            ck = (~sum(pkt[2:]) & 0xFF)
            self.ser.write(pkt + bytearray([ck]))
            time.sleep(0.005)
            resp = self.ser.read(16)
            if len(resp) >= 7 and resp[0] == 0xFF and resp[1] == 0xFF:
                raw = int.from_bytes(resp[5:7], "little")
                if sid == 6:
                    calib = self.CALIB[6]
                    mid = (calib["range_min"] + calib["range_max"]) / 2
                    current[5] = np.deg2rad((raw - mid) * 360 / 4095)
                else:
                    calib = self.CALIB[sid]
                    mid = (calib["range_min"] + calib["range_max"]) / 2
                    current[sid - 1] = np.deg2rad((raw - mid) * 360 / 4095)

        target = self.HOME_POSE
        for i in range(1, steps + 1):
            t = i / steps
            angles = current * (1 - t) + target * t
            for sid in range(1, 6):
                self._write_angle(sid, float(angles[sid - 1]))
            g_pulse = int(np.interp(angles[5],

                         [self.JOINT_LIMITS["gripper"][0], self.JOINT_LIMITS["gripper"][1]],
                         [self.CALIB[6]["range_min"], self.CALIB[6]["range_max"]]))
            g_pulse = max(self.CALIB[6]["range_min"], min(self.CALIB[6]["range_max"], g_pulse))
            g_cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, 6, 5, 0x03, 0x2A, g_pulse)
            g_ck = (~sum(g_cmd[2:]) & 0xFF)
            self.ser.write(g_cmd + struct.pack("<B", g_ck))
            time.sleep(delay_s)

    # ── 串口读写 ──

    def _read_current_pos(self) -> np.ndarray:
        result = np.zeros(6)
        for sid in range(1, 7):
            try:
                self.ser.reset_input_buffer()
                pkt = bytearray([0xFF, 0xFF, sid, 4, 0x02, 0x38, 0x02])
                ck = (~sum(pkt[2:]) & 0xFF)
                self.ser.write(pkt + bytearray([ck]))
                time.sleep(0.005)
                resp = self.ser.read(16)
                if len(resp) >= 7 and resp[0] == 0xFF and resp[1] == 0xFF:
                    raw = int.from_bytes(resp[5:7], "little")
                    calib = self.CALIB[sid]
                    mid = (calib["range_min"] + calib["range_max"]) / 2.0
                    result[sid - 1] = np.deg2rad((raw - mid) * 360.0 / 4095.0)
            except Exception:
                result[sid - 1] = 0.0
        return result

    def _write_angle(self, sid: int, rad: float):
        calib = self.CALIB[sid]
        deg = np.rad2deg(rad)
        mid = (calib["range_min"] + calib["range_max"]) / 2
        raw = int(deg * 4095 / 360 + mid)
        raw = max(calib["range_min"], min(calib["range_max"], raw))
        cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, sid, 5, 0x03, 0x2A, raw)
        checksum = (~sum(cmd[2:]) & 0xFF)
        self.ser.write(cmd + struct.pack("<B", checksum))

    def _sync_write_angles(self, angles_rad):
        """SYNC_WRITE — 一次串口包写入多个关节角度（5个关节，不含夹爪）"""
        addr = 0x2A  # 目标位置寄存器地址
        packet = bytearray([0xFF, 0xFF, 0xFE, 4 + 5 * 3, 0x83, addr & 0xFF, (addr >> 8) & 0xFF])
        for sid in range(1, 6):
            calib = self.CALIB[sid]
            mid = (calib["range_min"] + calib["range_max"]) / 2
            deg = np.rad2deg(angles_rad[sid - 1])
            raw = int(deg * 4095 / 360 + mid)
            raw = max(calib["range_min"], min(calib["range_max"], raw))
            packet += struct.pack("<BH", sid, raw)
        cks = (~sum(packet[2:]) & 0xFF)
        self.ser.write(packet + struct.pack("<B", cks))

    def gripper(self, open: bool):
        pulse = 2600 if open else 1781
        cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, 6, 5, 0x03, 0x2A, pulse)
        cks = (~sum(cmd[2:]) & 0xFF)
        self.ser.write(cmd + struct.pack("<B", cks))

    def emergency_stop(self):
        try:
            self.ser.reset_output_buffer()
            self.ser.reset_input_buffer()
            for sid in range(1, 7):
                cmd = struct.pack("<BBBBBBB", 0xFF, 0xFF, sid, 4, 0x03, 0x28, 0)
                cks = (~sum(cmd[2:]) & 0xFF)
                self.ser.write(cmd + struct.pack("<B", cks))
                time.sleep(0.002)
        except Exception:
            pass

    def write_angles(self, angles_rad):
        for sid in range(1, 6):
            self._write_angle(sid, float(angles_rad[sid - 1]))
        g_pulse = int(np.interp(angles_rad[5],
                     [self.JOINT_LIMITS["gripper"][0], self.JOINT_LIMITS["gripper"][1]],
                     [self.CALIB[6]["range_min"], self.CALIB[6]["range_max"]]))
        g_pulse = max(self.CALIB[6]["range_min"], min(self.CALIB[6]["range_max"], g_pulse))
        g_cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, 6, 5, 0x03, 0x2A, g_pulse)
        g_ck = (~sum(g_cmd[2:]) & 0xFF)
        self.ser.write(g_cmd + struct.pack("<B", g_ck))

    def close(self):
        self.ser.close()
