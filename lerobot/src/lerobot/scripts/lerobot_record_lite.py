"""精简版：遥操作录制关节角度到JSON（无外部依赖）"""
import sys, os, json, time, datetime, struct, serial

SERIAL_BAUD = 1000000
calib = [(946,3287),(821,3206),(888,3105),(851,3192),(0,4095),(1495,2860)]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def read_follower(ser) -> dict:
    cmd = struct.pack("<BBBBBBB", 0xFF, 0xFF, 0xFE, 4, 0x92, 0x38, 2)
    cks = (~sum(cmd[2:]) & 0xFF)
    ser.write(cmd + struct.pack("<B", cks))
    time.sleep(0.003)
    resp = ser.read(60)
    angles = {}
    for sid in range(1, 7):
        idx = resp.find(bytes([0xFF, 0xFF, sid]))
        if idx >= 0 and idx + 7 < len(resp):
            pos = int.from_bytes(resp[idx+5:idx+7], "little")
            lo, hi = calib[sid - 1]
            mid = (lo + hi) / 2
            angles[f"J{sid}"] = round((pos - mid) * 360 / 4095, 1)
    return angles


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot.port", default="/dev/ttyACM0")
    parser.add_argument("--teleop.port", default="/dev/ttyACM1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time_s", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or "record_{}.json".format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    ser = serial.Serial(args.teleop.port, SERIAL_BAUD, timeout=0.05)

    print("请在另一终端运行遥操作，然后按 Enter 开始录制")
    print("  lerobot-teleoperate \\")
    print("    --robot.type=so101_follower --robot.port={} ...".format(args.robot.port))
    print("    --teleop.type=so101_leader --teleop.port={} ...".format(args.teleop.port))
    input("按 Enter 开始录制 {} 秒...".format(args.episode_time_s))

    frames = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.episode_time_s:
        loop_t = time.perf_counter()
        angles = read_follower(ser)
        if angles:
            angles["t"] = round(time.perf_counter() - t0, 3)
            frames.append(angles)
        dt = time.perf_counter() - loop_t
        time.sleep(max(1 / args.fps - dt, 0))

    ser.close()

    data = {
        "fps": args.fps,
        "total_frames": len(frames),
        "duration_s": args.episode_time_s,
        "frames": frames,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("  {} 帧, 已保存 {}".format(len(frames), out_path))


if __name__ == "__main__":
    main()
