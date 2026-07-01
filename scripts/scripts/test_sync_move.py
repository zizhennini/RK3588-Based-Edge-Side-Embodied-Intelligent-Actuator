"""测试6关节同时移动 vs 逐个移动"""
import struct, serial, time, sys

PORT = "/dev/ttyACM0"
BAUD = 1000000

CALIB = {
    1: {"min": 946, "max": 3287},
    2: {"min": 821, "max": 3206},
    3: {"min": 888, "max": 3105},
    4: {"min": 851, "max": 3192},
    5: {"min": 130, "max": 3985},
    6: {"min": 1495,"max": 2860},
}

def deg_to_pulse(sid, deg):
    mid = (CALIB[sid]["min"] + CALIB[sid]["max"]) / 2
    return int(deg * 4095 / 360 + mid)

def make_write_packet(sid, pulse):
    cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, sid, 5, 0x03, 0x2A, pulse)
    cks = (~sum(cmd[2:]) & 0xFF)
    return cmd + struct.pack("<B", cks)

ser = serial.Serial(PORT, BAUD, timeout=0.3)
time.sleep(0.2)

# 先回中位
print("回中位...")
for sid in range(1, 7):
    mid = (CALIB[sid]["min"] + CALIB[sid]["max"]) / 2
    ser.write(make_write_packet(sid, int(mid)))
time.sleep(3)

# —— 逐个发（当前方式）——
targets = {1: 30, 2: 20, 3: -20, 4: 15, 5: 45, 6: 40}
print("\n=== 逐个发送 ===")
t0 = time.perf_counter()
for sid in range(1, 7):
    p = deg_to_pulse(sid, targets[sid])
    print(f"  关节{sid}: {targets[sid]:>3}° → pulse={p}")
    ser.write(make_write_packet(sid, p))
    time.sleep(0.3)  # 模仿当前代码的逐个延迟
t_seq = time.perf_counter() - t0
print(f"  耗时: {t_seq*1000:.0f}ms")
time.sleep(3)

# 回中位
print("\n回中位...")
for sid in range(1, 7):
    mid = (CALIB[sid]["min"] + CALIB[sid]["max"]) / 2
    ser.write(make_write_packet(sid, int(mid)))
time.sleep(3)

# —— 同时发（sync_write 模拟）——
print("\n=== 同时发送 ===")
t0 = time.perf_counter()
for sid in range(1, 7):
    p = deg_to_pulse(sid, targets[sid])
    print(f"  关节{sid}: {targets[sid]:>3}° → pulse={p}")
    ser.write(make_write_packet(sid, p))  # 连续写，不等待
t_sync = time.perf_counter() - t0
print(f"  耗时: {t_sync*1000:.0f}ms (6个包在串口缓冲区排队发出)")
time.sleep(3)

# 回中位
print("\n回中位...")
for sid in range(1, 7):
    mid = (CALIB[sid]["min"] + CALIB[sid]["max"]) / 2
    ser.write(make_write_packet(sid, int(mid)))
time.sleep(2)

ser.close()
print("\n测试完成 - 请观察两种方式的动作是否一样流畅")
print("逐个发: 关节依次动（卡顿）")
print("同时发: 6个关节同时启动（流畅）")
