# hardware/teleop.py
"""遥操作对（阶段 C / G9）: LeaderArm 只读主臂 + TeleopPair 跟随/录制

自研轻量实现，替代已删除的 vendored lerobot 依赖（docs/pc_board_feetech_plan.md）。
- LeaderArm: 主臂（人手搬动，全程禁扭矩，只读 Present_Position）
- TeleopPair: 30Hz 跟随环 leader→follower（G3 帧间限幅）+ JSON 录制
  （输出格式与 scripts/lerobot-record-lite 完全兼容，PC 端可用
  scripts/json_to_lerobot.py 转 LeRobot dataset / npz）

板端依赖: scservo_sdk + pyserial（已装）；PC 端 USB 直通亦可运行。
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from hardware.feetech_bus import FeetechBus
from hardware.arm import SO101Arm, DEFAULT_CALIBRATION, JOINT_NAMES

logger = logging.getLogger(__name__)


class LeaderArm:
    """主臂（遥操作输入端）: 禁扭矩只读"""

    def __init__(self, port: str = "/dev/ttyACM1", baud: int = 1_000_000,
                 calibration: Optional[dict] = None):
        self.bus = FeetechBus(port, baud=baud, name="leader")
        self.calibration = calibration or dict(DEFAULT_CALIBRATION)
        self._connected = False

    def connect(self, handshake: bool = True) -> None:
        self.bus.connect(handshake=handshake)
        # 主臂全程禁扭矩（人手自由搬动），无需 configure 防烧/模式写入
        self.bus.disable_torque()
        self._connected = True
        logger.info("LeaderArm 已连接（扭矩禁用，可手搬）")

    def disconnect(self) -> None:
        if self._connected:
            self.bus.disconnect(disable_torque=False)
            self._connected = False

    def _raw_to_rad(self, sid: int, raw: float) -> float:
        mid = self.calibration[str(sid)]["homing_offset"]
        return float(np.deg2rad((raw - mid) * 360.0 / 4095.0))

    def read_raw(self) -> Dict[int, int]:
        """SYNC_READ 6 关节原始值 {id: raw}（个别缺响应则该键缺失）"""
        return self.bus.sync_read("Present_Position", num_retry=1)

    def read_joints(self) -> Optional[np.ndarray]:
        """6 关节弧度 (6,)；任一关节无响应返回 None（丢弃该帧）"""
        raw_map = self.read_raw()
        if len(raw_map) < 6:
            return None
        return np.array([self._raw_to_rad(sid, raw_map[sid])
                         for sid in range(1, 7)], dtype=float)


class TeleopPair:
    """主从遥操作对: 30Hz leader→follower 跟随 + JSON 录制

    录制格式（与 lerobot-record-lite 兼容）::

        {"fps": 30, "total_frames": N, "duration_s": T,
         "frames": [{"J1": deg, ..., "J6": deg, "t": sec}, ...]}
    """

    def __init__(self, leader_port: str = "/dev/ttyACM1",
                 follower_port: str = "/dev/ttyACM0",
                 fps: int = 30,
                 leader_calibration: Optional[dict] = None,
                 follower_calibration_path: str = "./config/calibration.json",
                 max_relative_step_deg: float = 15.0,
                 handshake: bool = True):
        self.fps = fps
        self.max_step_rad = float(np.deg2rad(max_relative_step_deg)) \
            if max_relative_step_deg else None
        self.handshake = handshake
        self.leader = LeaderArm(leader_port, calibration=leader_calibration)
        self.follower = SO101Arm(
            port=follower_port,
            calibration_path=follower_calibration_path,
            max_relative_step_deg=max_relative_step_deg,
        )
        self._last_cmd: Optional[np.ndarray] = None

    def start(self) -> None:
        self.follower.connect(handshake=self.handshake)
        self.leader.connect(handshake=self.handshake)
        # 首帧对齐: 从臂插值平滑移动到主臂当前姿态（防起步突跳）
        joints = self._read_leader_filtered()
        if joints is not None:
            self._smooth_goto(joints)
            self._last_cmd = joints.copy()
        logger.info("TeleopPair 就绪: leader→follower @ %dHz, 限幅 %.1f°/帧",
                    self.fps,
                    float(np.rad2deg(self.max_step_rad)) if self.max_step_rad else -1)

    def _smooth_goto(self, target: np.ndarray, steps: int = 30,
                     delay_s: float = 0.02) -> None:
        """从臂从当前姿态插值平滑过渡到目标（起步对齐用）"""
        current = self.follower.read_positions()
        for i in range(1, steps + 1):
            t = i / steps
            self.follower.write_positions(current * (1 - t) + target * t)
            time.sleep(delay_s)

    def _read_leader_filtered(self, timeout_s: float = 2.0) -> Optional[np.ndarray]:
        """带超时的主臂读取（连续丢帧放弃）"""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            joints = self.leader.read_joints()
            if joints is not None:
                return joints
            time.sleep(0.01)
        return None

    def step(self) -> Optional[np.ndarray]:
        """单帧: leader 读 → G3 限幅 → follower 写。返回实际下发角度或 None"""
        joints = self.leader.read_joints()
        if joints is None:
            return None
        if self.max_step_rad is not None and self._last_cmd is not None:
            delta = np.clip(joints - self._last_cmd,
                            -self.max_step_rad, self.max_step_rad)
            joints = self._last_cmd + delta
        self.follower.write_positions(joints)
        self._last_cmd = joints.copy()
        return joints

    def run(self, duration_s: float, out_path: Optional[str] = None,
            follow: bool = True) -> List[dict]:
        """运行遥操作环 duration_s 秒并录制

        Args:
            duration_s: 录制时长
            out_path: JSON 输出路径（None 不保存）
            follow: False 时只录不跟随（从臂不动）
        Returns:
            frames 列表
        """
        frames: List[dict] = []
        period = 1.0 / self.fps
        print(f"遥操作录制 {duration_s:.0f}s @ {self.fps}fps"
              f"（follow={follow}）... Ctrl-C 提前结束并保存")
        t0 = time.perf_counter()
        try:
            while True:
                elapsed = time.perf_counter() - t0
                if elapsed >= duration_s:
                    break
                loop_t = time.perf_counter()
                if follow:
                    joints = self.step()
                else:
                    joints = self.leader.read_joints()
                if joints is not None:
                    frame = {f"J{i + 1}": round(float(np.rad2deg(joints[i])), 1)
                             for i in range(6)}
                    frame["t"] = round(elapsed, 3)
                    frames.append(frame)
                # 恒定帧率
                sleep_left = period - (time.perf_counter() - loop_t)
                if sleep_left > 0:
                    time.sleep(sleep_left)
        except KeyboardInterrupt:
            print("\n提前结束（已保存已录帧）")
        if out_path:
            self.save(frames, out_path, duration_s=min(
                duration_s, time.perf_counter() - t0))
        return frames

    def save(self, frames: List[dict], out_path: str,
             duration_s: Optional[float] = None) -> None:
        """保存为 record-lite 兼容 JSON"""
        data = {
            "fps": self.fps,
            "total_frames": len(frames),
            "duration_s": round(duration_s, 3) if duration_s else
                          (frames[-1]["t"] if frames else 0),
            "frames": frames,
        }
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("录制已保存: %s (%d 帧)", p, len(frames))

    def stop(self) -> None:
        try:
            self.leader.disconnect()
        except Exception:
            pass
        try:
            self.follower.close()
        except Exception:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
