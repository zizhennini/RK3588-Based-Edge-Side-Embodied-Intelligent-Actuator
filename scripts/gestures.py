"""机械臂动作序列 — 运动插补（每次动作前读取当前角度）"""
import sys, os, json, time, struct, serial
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vla.control import ArmController
from config.settings import SERIAL_PORT, SERIAL_BAUD

POSES_FILE = os.path.join(os.path.dirname(__file__), "gesture_poses.json")
with open(POSES_FILE) as f:
    POSES = json.load(f)

calib = [(946,3287),(821,3206),(888,3105),(851,3192),(130,3985),(1495,2860)]


def read_angles(ser) -> list:
    angles = [0.0] * 6
    for sid in range(1, 7):
        c = struct.pack("<BBBBBBB", 0xFF, 0xFF, sid, 4, 0x02, 0x38, 2)
        cks = (~sum(c[2:]) & 0xFF)
        ser.write(c + struct.pack("<B", cks))
        time.sleep(0.01)
        resp = ser.read(15)
        idx = resp.find(bytes([0xFF, 0xFF, sid, 0x04]))
        if idx >= 0:
            pos = int.from_bytes(resp[idx+5:idx+7], "little")
            lo, hi = calib[sid-1]
            mid = (lo + hi) / 2
            angles[sid-1] = (pos - mid) * 360 / 4095
    return angles


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def _interpolate(current, target, duration, fps=20):
    steps = max(int(duration * fps), 3)
    for step in range(1, steps + 1):
        t = _smoothstep(step / steps)
        yield [c + (trg - c) * t for c, trg in zip(current, target)]


MAX_J2 = 2  # J2每帧最大变化（度）


class Gestures:
    def __init__(self, arm):
        self.arm = arm
        self._prev_j2 = 0.0

    def _send(self, angles: list):
        diff = angles[1] - self._prev_j2
        if abs(diff) > MAX_J2:
            angles[1] = self._prev_j2 + (MAX_J2 if diff > 0 else -MAX_J2)
        self._prev_j2 = angles[1]
        for sid in range(1, 7):
            self.arm._write_angle(sid, float(np.deg2rad(angles[sid - 1])))

    def _to_list(self, pose_name_or_dict) -> list:
        if isinstance(pose_name_or_dict, str):
            d = POSES[pose_name_or_dict]
        else:
            d = pose_name_or_dict
        return [d.get(f"J{i}", d.get(i, 0.0)) for i in range(1, 7)]

    def _go(self, current: list, target: list, duration: float) -> list:
        for frame in _interpolate(current, target, duration):
            self._send(frame)
            time.sleep(1 / 20)
        return target

    def home(self):
        cur = read_angles(self.arm.ser)
        self._prev_j2 = cur[1]
        self._go(cur, self._to_list("home"), 2.0)

    def greeting(self):
        cur = read_angles(self.arm.ser)
        self._prev_j2 = cur[1]
        cur = self._go(cur, self._to_list("hello_up"), 2.5)
        for j1 in [-30, 30, 0]:
            tgt = cur.copy()
            tgt[0] = j1
            cur = self._go(cur, tgt, 0.8)
        for j5 in [71, -37, -30, 5]:
            tgt = cur.copy()
            tgt[4] = j5
            cur = self._go(cur, tgt, 0.4)
        time.sleep(0.5)
        self._go(cur, self._to_list("home"), 2.5)

    def wave(self):
        cur = read_angles(self.arm.ser)
        self._prev_j2 = cur[1]
        cur = self._go(cur, self._to_list("hello_up"), 1.0)

    def nod(self):
        cur = read_angles(self.arm.ser)
        self._prev_j2 = cur[1]
        cur = self._go(cur, self._to_list("hello_up"), 1.0)
        for j1 in [-30, 30, 0]:
            tgt = cur.copy()
            tgt[0] = j1
            cur = self._go(cur, tgt, 0.8)
        self._go(cur, self._to_list("home"), 2.0)

    def nod(self):
        cur = read_angles(self.arm.ser)
        self._prev_j2 = cur[1]
        for j5 in [71, -37, -30, 5]:
            tgt = cur.copy()
            tgt[4] = j5
            cur = self._go(cur, tgt, 0.4)
        time.sleep(0.5)
        self._go(cur, self._to_list("home"), 2.5)


if __name__ == "__main__":
    arm = ArmController(SERIAL_PORT, SERIAL_BAUD)
    g = Gestures(arm)

    print("交互动作（运动插补）:")
    print("  g 打招呼     w 挥手")
    print("  n 点头       h 归位")
    print("  q 退出")

    while True:
        cmd = input("\n选择: ").strip().lower()
        if cmd == "q":
            break
        if cmd == "g":
            g.greeting()
        elif cmd == "w":
            g.wave()
        elif cmd == "n":
            g.nod()
        elif cmd == "h":
            g.home()

    arm.close()
