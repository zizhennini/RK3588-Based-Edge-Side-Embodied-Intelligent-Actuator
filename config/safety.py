"""安全防护模块 — 深度视觉避障 + 电流柔顺防撞"""

from __future__ import annotations
import time
import struct
import threading
import numpy as np
from pathlib import Path


# ── 默认安全参数 ──
DEFAULT_PARAMS = {
    # 深度避障（米）
    "depth_warn_dist": 0.25,    # 警告距离：减速
    "depth_stop_dist": 0.12,    # 停止距离：急停
    "depth_roi_margin": 0.15,   # ROI 相对于画面宽高的比例
    "depth_skip_frames": 2,     # 隔帧检测（降低CPU/带宽）
    # 电流柔顺
    "current_warn": 600,        # 警告电流（mA）
    "current_stop": 900,        # 停止电流（mA）
    "current_sample_hz": 10,    # 电流采样频率
    # 响应
    "backoff_duration": 0.5,    # 回退时长（s）
    "backoff_distance": 0.02,   # 回退距离（m）
}


class DepthObstacleDetector:
    """基于 D435i 深度图的障碍物检测"""

    def __init__(self, camera, params: dict = None):
        self.cam = camera
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self._frame_count = 0

    def check(self) -> tuple[str, float]:
        """
        检测前方障碍物
        返回: (状态, 最小距离)
          "safe"  - 安全
          "warn"  - 接近，需减速
          "stop"  - 危险，需急停
        """
        self._frame_count += 1
        if self._frame_count % (self.p["depth_skip_frames"] + 1) != 0:
            return "safe", 99.0

        _, depth = self.cam.read()
        if depth is None:
            return "safe", 99.0

        h, w = depth.shape[:2]
        margin_x = int(w * self.p["depth_roi_margin"])
        margin_y = int(h * self.p["depth_roi_margin"])
        roi = depth[margin_y:h-margin_y, margin_x:w-margin_x]

        valid = roi[(roi > 0.01) & (~np.isnan(roi))]
        if len(valid) == 0:
            return "safe", 99.0

        min_dist = float(np.min(valid))
        warn = self.p["depth_warn_dist"]
        stop = self.p["depth_stop_dist"]

        if min_dist < stop:
            return "stop", min_dist
        if min_dist < warn:
            return "warn", min_dist
        return "safe", min_dist


class CurrentMonitor:
    """ST3215 舵机电流监测（通过串口读取）"""

    def __init__(self, ser, num_motors: int = 6, params: dict = None):
        self.ser = ser
        self.n = num_motors
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self._running = False
        self._thread: threading.Thread | None = None
        self._currents: list[float] = []
        self._lock = threading.Lock()

    def read_current(self, sid: int) -> float | None:
        """读取单个舵机电流（mA）"""
        try:
            self.ser.reset_input_buffer()
            pkt = bytearray([0xFF, 0xFF, sid, 4, 0x02, 0x3E, 0x02])
            ck = (~sum(pkt[2:]) & 0xFF)
            self.ser.write(pkt + bytearray([ck]))
            time.sleep(0.003)
            resp = self.ser.read(16)
            if len(resp) >= 9 and resp[0] == 0xFF and resp[1] == 0xFF:
                raw = int.from_bytes(resp[7:9], "little", signed=True)
                return float(abs(raw) * 3.36)  # ST3215: 3.36mA/bit
        except Exception:
            pass
        return None

    def start(self):
        """启动后台监控线程"""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            currents = []
            for sid in range(1, self.n + 1):
                val = self.read_current(sid)
                currents.append(val if val is not None else 0)
            with self._lock:
                self._currents = currents
            time.sleep(1.0 / self.p["current_sample_hz"])

    @property
    def max_current(self) -> float:
        with self._lock:
            return max(self._currents) if self._currents else 0

    def check(self) -> tuple[str, float]:
        """返回 (状态, 最大电流)"""
        mc = self.max_current
        if mc > self.p["current_stop"]:
            return "stop", mc
        if mc > self.p["current_warn"]:
            return "warn", mc
        return "safe", mc


class SafetyController:
    """安全防护控制器 — 整合深度避障 + 电流防撞"""

    def __init__(self, camera=None, ser=None, params: dict = None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.depth = DepthObstacleDetector(camera, params) if camera else None
        self.current = CurrentMonitor(ser, params=params) if ser else None
        self._backoff_active = False

    def start(self):
        if self.current:
            self.current.start()

    def stop(self):
        if self.current:
            self.current.stop()

    def check_all(self, arm=None) -> tuple[str, str, float]:
        """
        全面安全检查
        返回: (action, source, value)
          action: "safe"/"warn"/"stop"/"backoff"
          source: "depth"/"current"
          value: 距离或电流值
        """
        # 深度检查
        if self.depth:
            status, dist = self.depth.check()
            if status != "safe":
                return ("stop" if status == "stop" else "warn"), "depth", dist

        # 电流检查
        if self.current:
            status, mc = self.current.check()
            if status != "safe":
                if status == "stop":
                    if arm and hasattr(arm, "emergency_stop"):
                        arm.emergency_stop()
                    return "stop", "current", mc
                return "warn", "current", mc

        # 回退恢复
        if self._backoff_active:
            self._backoff_active = False

        return "safe", "", 0.0

    def backoff(self, arm):
        """碰撞后向上回退"""
        if not arm or not hasattr(arm, "move_to"):
            return
        try:
            # 读取关节3(elbow_flex)角度估算当前Z高度
            arm.ser.reset_input_buffer()
            pkt = bytearray([0xFF, 0xFF, 3, 4, 0x02, 0x38, 0x02])
            ck = (~sum(pkt[2:]) & 0xFF)
            arm.ser.write(pkt + bytearray([ck]))
            time.sleep(0.005)
            resp = arm.ser.read(16)
            cur_z = 0.15
            if len(resp) >= 9 and resp[0] == 0xFF:
                raw = int.from_bytes(resp[7:9], "little")
                deg = (raw - 2048) * 360.0 / 4095.0
                cur_z = 0.12 + 0.18 * np.sin(np.deg2rad(abs(deg))) * 0.5
            new_z = min(cur_z + self.p["backoff_distance"], 0.40)
            arm.move_to(0.25, 0.0, new_z)
        except Exception:
            pass
        self._backoff_active = True
