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

from hardware.feetech_bus import (FeetechBus, GRIPPER_MOTOR_ID, angle_zero,
                                  raw_to_rad)
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
        # leader 与 follower 必须同配置（lerobot SOLeader.configure 同款）：
        # Return_Delay_Time=0 / Maximum_Acceleration / Acceleration / PID /
        # Operating_Mode=POSITION / Phase bit4+bit6 对齐。之后保持禁扭矩供人手搬动。
        self.bus.configure(gripper_id=None)
        self.bus.disable_torque()
        self._connected = True
        logger.info("LeaderArm 已连接（configure 完成，扭矩禁用，可手搬）")

    def disconnect(self) -> None:
        if self._connected:
            self.bus.disconnect(disable_torque=False)
            self._connected = False

    def _calib_mid(self, sid: int) -> float:
        """角度零点：体关节用行程中点（官方 DEGREES 语义），夹爪用标定中位点"""
        return angle_zero(self.calibration[str(sid)],
                          use_range_midpoint=(sid != GRIPPER_MOTOR_ID))

    def _raw_to_rad(self, sid: int, raw: float) -> float:
        return raw_to_rad(raw, self.calibration[str(sid)],
                          use_range_midpoint=(sid != GRIPPER_MOTOR_ID))

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
        try:
            self.follower.connect(handshake=self.handshake)
            self.leader.connect(handshake=self.handshake)
            # 首帧对齐: 从臂插值平滑移动到主臂当前姿态（防起步突跳）
            joints = self._read_leader_filtered()
            if joints is not None:
                self._smooth_goto(joints)
                self._last_cmd = joints.copy()
        except Exception:
            # 半连接失败必须清理：否则从臂会停在扭矩开启状态（进程退出后舵机仍锁死）
            logger.error("TeleopPair 启动失败，正在关闭两侧并释放扭矩")
            self.stop()
            raise
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

    def run(self, duration_s: float = 0, out_path: Optional[str] = None,
            follow: bool = True) -> List[dict]:
        """运行遥操作环并录制

        Args:
            duration_s: 录制时长（秒）；<=0 或 None = **无限时**，Ctrl-C 结束并保存
            out_path: JSON 输出路径（None 不保存）
            follow: False 时只录不跟随（从臂不动）
        Returns:
            frames 列表
        """
        frames: List[dict] = []
        period = 1.0 / self.fps
        unlimited = duration_s is None or duration_s <= 0
        if unlimited:
            print(f"无限时跟随 @ {self.fps}fps（follow={follow}）... "
                  f"Ctrl-C 结束并保存")
        else:
            print(f"遥操作录制 {duration_s:.0f}s @ {self.fps}fps"
                  f"（follow={follow}）... Ctrl-C 提前结束并保存")
        leader_fail_frames = 0   # 连续丢帧计数（无限时自恢复用）
        recovered = 0
        t0 = time.perf_counter()
        try:
            while True:
                elapsed = time.perf_counter() - t0
                if not unlimited and elapsed >= duration_s:
                    break
                loop_t = time.perf_counter()
                if follow:
                    joints = self.step()
                else:
                    joints = self.leader.read_joints()
                if joints is not None:
                    leader_fail_frames = 0
                    frame = {f"J{i + 1}": round(float(np.rad2deg(joints[i])), 1)
                             for i in range(6)}
                    frame["t"] = round(elapsed, 3)
                    frames.append(frame)
                else:
                    # 无限时场景自恢复: leader 连续丢帧 1 秒 → 尝试串口重置
                    leader_fail_frames += 1
                    if leader_fail_frames >= self.fps:
                        recovered += 1
                        logger.warning(
                            "leader 连续丢帧 %d 帧，尝试串口恢复（第 %d 次）",
                            leader_fail_frames, recovered)
                        try:
                            self.leader.bus.reset_serial()
                        except Exception as e:
                            logger.error("leader 串口恢复失败: %s", e)
                        leader_fail_frames = 0
                # 恒定帧率
                sleep_left = period - (time.perf_counter() - loop_t)
                if sleep_left > 0:
                    time.sleep(sleep_left)
        except KeyboardInterrupt:
            print("\n结束（保存已录帧）")
        elapsed_total = time.perf_counter() - t0
        measured_fps = len(frames) / elapsed_total if elapsed_total > 0 else 0.0
        print(f"录制结束: {len(frames)} 帧, {elapsed_total:.1f}s, "
              f"实测 {measured_fps:.1f} fps（目标 {self.fps}）"
              + (f", 串口自恢复 {recovered} 次" if recovered else ""))
        if out_path:
            self.save(frames, out_path, duration_s=elapsed_total)
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
