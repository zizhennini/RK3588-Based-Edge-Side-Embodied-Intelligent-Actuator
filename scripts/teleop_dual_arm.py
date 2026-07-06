#!/usr/bin/env python3
"""双臂遥操作 — 北通蝙蝠4 (BD4A) 手柄控制双臂 SO-ARM101"""

import sys
import time
import threading
import numpy as np

sys.path.insert(0, ".")

from evdev import InputDevice, list_devices, ecodes
from vla.control import ArmController


# ── 按键/轴常量 ──
BTN_A = 304; BTN_B = 305; BTN_X = 307; BTN_Y = 308
BTN_C = 306; BTN_Z = 309
BTN_LB = 310; BTN_RB = 311
BTN_SELECT = 314; BTN_START = 315; BTN_HOME = 316
BTN_L3 = 317; BTN_R3 = 318

AX_LX = 0; AX_LY = 1   # 左摇杆/十字键

# ── 参数 ──
LOOP_HZ = 50
DT = 1.0 / LOOP_HZ
VELOCITY = 0.08        # m/s 最大移动速度
GRIPPER_STEP = 0.3     # 每次切换夹爪开合角度增量


class GamepadReader:
    """北通蝙蝠4 手柄读取器 — 轮询状态"""

    def __init__(self, device_path: str | None = None):
        self._dev = None
        if device_path:
            self._dev = InputDevice(device_path)
        else:
            for p in list_devices():
                d = InputDevice(p)
                if "betop" in d.name.lower():
                    self._dev = d
                    break
        if not self._dev:
            raise RuntimeError("未检测到北通蝙蝠4手柄")

        self._dev.grab()

        self.buttons = {c: 0 for c in (
            304, 305, 306, 307, 308, 309, 310, 311, 312, 313,
            314, 315, 316, 317, 318, 398)}
        self.axes = {c: 128 for c in (0, 1, 2, 5, 9, 10)}

    def poll(self):
        """非阻塞轮询，更新最新状态"""
        while True:
            ev = self._dev.read_one()
            if ev is None:
                break
            if ev.type == 1 and ev.code in self.buttons:
                self.buttons[ev.code] = ev.value
            elif ev.type == 3 and ev.code in self.axes:
                self.axes[ev.code] = ev.value

    def __del__(self):
        try:
            self._dev.ungrab()
        except Exception:
            pass

    def axis_norm(self, code: int) -> float:
        """将轴值归一化为 [-1, 1]"""
        v = self.axes.get(code, 128)
        return (v - 128) / 128.0

    def btn(self, code: int) -> bool:
        return self.buttons.get(code, 0) == 1


class DualArmTeleop:
    """双臂遥操作控制器"""

    def __init__(self, arm1_port: str, arm2_port: str, baud: int = 1000000):
        self.arm1 = ArmController(arm1_port, baud, bind_little=True)
        self.arm2 = ArmController(arm2_port, baud, bind_little=True)
        self.pad = GamepadReader()

        # 目标位姿（双臂末端笛卡尔坐标）
        self.target = {
            "arm1": np.array([0.25, 0.0, 0.15], dtype=float),
            "arm2": np.array([0.25, 0.0, 0.15], dtype=float),
        }
        self.gripper_state = {"arm1": True, "arm2": True}  # True=开
        self.running = False

    def _deadzone(self, val: float, zone: float = 0.15) -> float:
        if abs(val) < zone:
            return 0.0
        return (val - np.sign(val) * zone) / (1.0 - zone)

    def _map_controls(self):
        """将手柄输入映射为双臂控制指令"""
        p = self.pad

        # 左摇杆 → 臂1 XY 平移
        lx = self._deadzone(p.axis_norm(AX_LX))
        ly = self._deadzone(p.axis_norm(AX_LY))

        # 右摇杆 → 臂2 XY 平移
        rx = self._deadzone(p.axis_norm(2))
        ry = self._deadzone(p.axis_norm(5))

        # D-pad 上下 → 臂1 Z
        ly_pure = p.axes.get(AX_LY, 128)
        dz1 = 0

        # LT/RT → 臂2 Z
        lt = p.axis_norm(10)
        rt = p.axis_norm(9)

        # 更新臂1 目标位置
        self.target["arm1"][0] += lx * VELOCITY * DT
        self.target["arm1"][1] -= ly * VELOCITY * DT

        # 更新臂1 Z (十字键/左摇杆极限值)
        if ly_pure == 0:
            dz1 = VELOCITY * DT
        elif ly_pure == 255:
            dz1 = -VELOCITY * DT
        self.target["arm1"][2] += dz1

        # 更新臂2 目标位置
        self.target["arm2"][0] += rx * VELOCITY * DT
        self.target["arm2"][1] -= ry * VELOCITY * DT
        self.target["arm2"][2] += (rt - lt) * VELOCITY * DT

        # 夹爪控制（边缘触发）
        if p.btn(BTN_LB):
            self.gripper_state["arm1"] = not self.gripper_state["arm1"]
            time.sleep(0.2)
        if p.btn(BTN_RB):
            self.gripper_state["arm2"] = not self.gripper_state["arm2"]
            time.sleep(0.2)

    def _execute(self):
        """发送控制指令到双臂"""
        for name, arm, target_key in [
            ("arm1", self.arm1, "arm1"),
            ("arm2", self.arm2, "arm2"),
        ]:
            pos = self.target[target_key]
            arm.move_to(float(pos[0]), float(pos[1]), float(pos[2]))
            arm.gripper(self.gripper_state[target_key])

    def run(self):
        """主控制循环"""
        self.running = True
        print("[Teleop] 双臂遥控制启动")
        print("  左摇杆 → 臂1 XY   |   右摇杆  → 臂2 XY")
        print("  十字键 ↑↓ → 臂1 Z  |   LT/RT   → 臂2 Z")
        print("  LB → 臂1夹爪       |   RB      → 臂2夹爪")
        print("  Start → 归零       |   Select  → 急停")
        print("  Home → 退出")
        print()

        try:
            while self.running:
                self.pad.poll()

                # 功能键
                if self.pad.btn(BTN_HOME):
                    print("[Teleop] 退出")
                    break
                if self.pad.btn(BTN_SELECT):
                    print("[Teleop] 紧急停止")
                    self.arm1.emergency_stop()
                    self.arm2.emergency_stop()
                    break
                if self.pad.btn(BTN_START):
                    print("[Teleop] 归零位")
                    self.target["arm1"][:] = [0.25, 0.0, 0.15]
                    self.target["arm2"][:] = [0.25, 0.0, 0.15]

                self._map_controls()
                self._execute()

                if not self.pad.btn(BTN_LB) and not self.pad.btn(BTN_RB):
                    time.sleep(DT)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """安全停止"""
        self.running = False
        self.arm1.emergency_stop()
        self.arm2.emergency_stop()
        self.arm1.close()
        self.arm2.close()
        print("[Teleop] 已停止")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="蝙蝠4 双臂遥操作")
    parser.add_argument("--arm1-port", default="/dev/ttyACM1", help="臂1(主) 串口")
    parser.add_argument("--arm2-port", default="/dev/ttyACM1", help="臂2(从) 串口")
    parser.add_argument("--baud", type=int, default=1000000, help="串口波特率")
    args = parser.parse_args()

    teleop = DualArmTeleop(args.arm1_port, args.arm2_port, args.baud)
    teleop.run()


if __name__ == "__main__":
    main()
