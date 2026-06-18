"""测试官方 v0.4.4 角度→脉冲映射公式"""
import struct, serial, time, sys
sys.path.insert(0, "..")

PORT = "/dev/ttyACM0"
BAUD = 1000000

# 标定数据
CALIB = {
    1: {"name": "shoulder_pan", "min": 946, "max": 3287, "homing": 2048},
    2: {"name": "shoulder_lift", "min": 821, "max": 3206, "homing": 2048},
    3: {"name": "elbow_flex",    "min": 888, "max": 3105, "homing": 2048},
    4: {"name": "wrist_flex",    "min": 851, "max": 3192, "homing": 2048},
    5: {"name": "wrist_roll",    "min": 130, "max": 3985, "homing": 2048},
    6: {"name": "gripper",       "min": 1495,"max": 2860, "homing": 1781},
}

def deg_to_pulse_v1(sid, deg):
    """旧公式: offset + deg/lim * (range_max - offset)"""
    c = CALIB[sid]
    lim = 110  # shoulder_pan 的 URDF 限位
    low_deg = -110
    if deg >= 0:
        return int(c["homing"] + deg / lim * (c["max"] - c["homing"]))
    else:
        return int(c["homing"] + deg / low_deg * (c["homing"] - c["min"]))

def deg_to_pulse_v2(sid, deg):
    """官方 v0.4.4 公式: deg * 4095/360 + (min+max)/2"""
    c = CALIB[sid]
    mid = (c["min"] + c["max"]) / 2
    return int(deg * 4095 / 360 + mid)

ser = serial.Serial(PORT, BAUD, timeout=0.3)

def read_pos(sid):
    cmd_r = struct.pack("<BBBBBBB", 0xFF, 0xFF, sid, 4, 0x02, 0x38, 2)
    cks_r = (~sum(cmd_r[2:]) & 0xFF)
    ser.write(cmd_r + struct.pack("<B", cks_r))
    time.sleep(0.08)
    resp = ser.read(20)
    idx = resp.find(bytes([0xFF, 0xFF, sid, 0x04]))
    if idx >= 0:
        return int.from_bytes(resp[idx+5:idx+7], "little")
    return None

def write_pulse(sid, pulse):
    cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, sid, 5, 0x03, 0x2A, pulse)
    cks = (~sum(cmd[2:]) & 0xFF)
    ser.write(cmd + struct.pack("<B", cks))

print("=" * 60)
print("  测试关节1 (shoulder_pan) 角度→脉冲映射")
print("=" * 60)

for deg in [0, 10, 30, 45, -10, -30]:
    p1 = deg_to_pulse_v1(1, deg)
    p2 = deg_to_pulse_v2(1, deg)

    write_pulse(1, p2)
    time.sleep(2)
    actual = read_pos(1)

    # 计算实际角度
    mid = (CALIB[1]["min"] + CALIB[1]["max"]) / 2
    actual_deg = (actual - mid) * 360 / 4095 if actual else -999

    print(f"\n{deg:>4}°  |  V1脉冲={p1:>5}  |  V2脉冲={p2:>5}  →  实际脉冲={actual or -999:>5}  →  {actual_deg:>5.1f}°")
    print(f"     {'':>10}偏差={'目标:'+str(p2):>6}  实际-目标={(actual or 0)-p2:>4}")

# 回中位(2117)
print("\n回中位 (mid=2117)...")
write_pulse(1, 2117)
time.sleep(2)
actual = read_pos(1)
print(f"中位实际脉冲: {actual}  (目标2117, 偏差{(actual or 0)-2117})")

ser.close()
print("\n测试完成")
