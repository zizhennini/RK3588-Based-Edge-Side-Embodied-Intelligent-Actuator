"""记录机械臂动作姿势 — 掰到安全位置后记录"""
import sys, os, json, struct, serial, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RECORD_FILE = os.path.join(os.path.dirname(__file__), "gesture_poses.json")

poses = {}

while True:
    name = input("姿势名称（如 wave_l，q 退出，s 保存）: ").strip()
    if name == "q":
        break
    if name == "s":
        with open(RECORD_FILE, "w") as f:
            json.dump(poses, f, ensure_ascii=False, indent=2)
        print(f"已保存到 {RECORD_FILE}")
        continue
    if not name:
        continue

    ser = serial.Serial("/dev/ttyACM0", 1000000, timeout=0.3)
    angles = {}
    for sid in range(1, 7):
        c = struct.pack("<BBBBBBB", 0xFF, 0xFF, sid, 4, 0x02, 0x38, 2)
        cks = (~sum(c[2:]) & 0xFF)
        ser.write(c + struct.pack("<B", cks))
        time.sleep(0.05)
        resp = ser.read(15)
        idx = resp.find(bytes([0xFF, 0xFF, sid, 0x04]))
        if idx >= 0:
            pos = int.from_bytes(resp[idx+5:idx+7], "little")
            # 脉冲 → 角度
            calib = {
                1: (946, 3287), 2: (821, 3206), 3: (888, 3105),
                4: (851, 3192), 5: (130, 3985), 6: (1495, 2860),
            }
            lo, hi = calib[sid]
            mid = (lo + hi) / 2
            deg = (pos - mid) * 360 / 4095
            angles[f"J{sid}"] = round(deg, 1)
    ser.close()

    poses[name] = angles
    print(f"  已记录 {name}: {angles}")
