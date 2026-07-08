#!/usr/bin/env python3
"""双臂遥操作 — 北通蝙蝠4 手柄同步控制主臂+从臂"""

import sys, os, time, threading, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evdev import InputDevice, list_devices
from vla.control import ArmController


BTN_LB=310; BTN_RB=311
BTN_A=304; BTN_B=305; BTN_X=307; BTN_Y=308
BTN_SEL=314; BTN_START=315; BTN_HOME=316
AX_LX=0; AX_LY=1; AX_RX=2; AX_RY=5; AX_LT=10; AX_RT=9

HOME_XYZ = [0.25, 0.0, 0.15]
VEL = 0.08


class Gamepad:
    """北通蝙蝠4 — 基于 evdev read_loop 的事件驱动读取"""
    def __init__(self):
        self._dev = None
        for p in list_devices():
            d = InputDevice(p)
            if "betop" in d.name.lower() and "Mouse" not in d.name:
                self._dev = d; break
        if not self._dev:
            raise RuntimeError("未检测到手柄，请确认已开机")
        print(f"  手柄: {self._dev.path} {self._dev.name}")

        # 读取初始状态
        caps = self._dev.capabilities()
        self.a = {}  # code -> value
        self.b = {}  # code -> value (0/1)
        from evdev import ecodes as _e
        if _e.EV_ABS in caps:
            for code, info in caps[_e.EV_ABS]:
                self.a[code] = info.value
        if _e.EV_KEY in caps:
            for code in caps[_e.EV_KEY]:
                self.b[code] = 0

        import threading as _t
        self._lock = _t.Lock()
        self._running = True
        _t.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        """后台线程：持续读取事件"""
        try:
            for ev in self._dev.read_loop():
                if not self._running: break
                if ev.type == 1 and ev.code in self.b:
                    with self._lock: self.b[ev.code] = ev.value
                elif ev.type == 3 and ev.code in self.a:
                    with self._lock: self.a[ev.code] = ev.value
        except:
            pass

    def ax(self, c):
        with self._lock:
            return (self.a.get(c, 128) - 128) / 128.0

    def btn(self, c):
        with self._lock:
            return self.b.get(c, 0) == 1

    def stop(self):
        self._running = False


def dz(v, z=0.12):
    if abs(v) < z: return 0.0
    return (v - np.sign(v)*z)/(1.0-z)


class Teleop:
    def __init__(self, p1="/dev/ttyACM1", p2="/dev/ttyACM0", baud=1000000):
        print("连接主臂...", end="", flush=True)
        self.a1 = ArmController(p1, baud, bind_little=True); print("OK")
        print("连接从臂...", end="", flush=True)
        self.a2 = ArmController(p2, baud, bind_little=True); print("OK")
        self.pad = Gamepad()
        self.x1 = np.array(HOME_XYZ, dtype=float)
        self.x2 = np.array(HOME_XYZ, dtype=float)
        self.g1 = True; self.g2 = True
        self._l1 = True; self._l2 = True
        self.lock = threading.Lock()

    def run(self):
        print("\n左摇杆+十字键→主臂  LB→夹爪   右摇杆+LT/RT→从臂  RB→夹爪")
        print("Start=归零  Select=急停  Home=退出\n")
        try:
            while True:
                p = self.pad
                if p.btn(BTN_HOME): break
                if p.btn(BTN_SEL):
                    self.a1.emergency_stop(); self.a2.emergency_stop(); break
                if p.btn(BTN_START):
                    self.a1.home(steps=40); self.a2.home(steps=40)
                    with self.lock: self.x1[:]=HOME_XYZ; self.x2[:]=HOME_XYZ

                lx = dz(p.ax(AX_LX)); ly = dz(p.ax(AX_LY))
                rx = dz(p.ax(AX_RX)); ry = dz(p.ax(AX_RY))
                lt = dz(p.ax(AX_LT)); rt = dz(p.ax(AX_RT))
                lr = p.a.get(AX_LY, 128)

                with self.lock:
                    self.x1[0] += lx * VEL * 0.02
                    self.x1[1] -= ly * VEL * 0.02
                    if lr == 0: self.x1[2] += VEL * 0.02
                    elif lr == 255: self.x1[2] -= VEL * 0.02
                    self.x2[0] += rx * VEL * 0.02
                    self.x2[1] -= ry * VEL * 0.02
                    self.x2[2] += (rt - lt) * VEL * 0.02
                    self.x1 = self.a1.clamp_workspace(self.x1)
                    self.x2 = self.a2.clamp_workspace(self.x2)

                self.a1.move_to(*self.x1, use_current=False)
                self.a2.move_to(*self.x2, use_current=False)
                if p.btn(BTN_LB): self.g1 = not self.g1; self.a1.gripper(self.g1); time.sleep(0.2)
                if p.btn(BTN_RB): self.g2 = not self.g2; self.a2.gripper(self.g2); time.sleep(0.2)

                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        finally:
            self.pad.stop()
            self.a1.emergency_stop(); self.a2.emergency_stop()
            self.a1.close(); self.a2.close()
            print("已停止")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--arm1", default="/dev/ttyACM1")
    p.add_argument("--arm2", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=1000000)
    a = p.parse_args()
    Teleop(a.arm1, a.arm2, a.baud).run()
