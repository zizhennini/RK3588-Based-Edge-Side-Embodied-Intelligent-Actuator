"""示教轨迹回放 — 加载 JSON → 逐帧发送"""
import sys, os, json, time, struct, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vla.control import ArmController
from config.settings import SERIAL_PORT, SERIAL_BAUD

JOINT_KEYS = [
    ("J1", "shoulder_pan"), ("J2", "shoulder_lift"), ("J3", "elbow_flex"),
    ("J4", "wrist_flex"), ("J5", "wrist_roll"), ("J6", "gripper"),
]

# gripper 校准（0-100 → 脉冲）
G_MIN, G_MAX = 1495, 2860


def get_val(frame, k1, k2):
    return float(frame.get(k1, frame.get(k2, 0.0)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--home", action="store_true", help="回放结束后自动归零")
    parser.add_argument("--initial", action="store_true", help="回放结束后回到初始态(等待位)")
    args = parser.parse_args()

    with open(args.traj_file) as f:
        data = json.load(f)

    frames = data["frames"]
    print("轨迹: {}帧, {}FPS".format(len(frames), data.get("fps", "?")))

    arm = ArmController(args.port, SERIAL_BAUD)
    print("回放中 ({}FPS)...".format(args.fps))
    t0 = time.perf_counter()

    def send_frame(f):
        for sid, (k1, k2) in enumerate(JOINT_KEYS, 1):
            val = get_val(f, k1, k2)
            if sid == 6:
                raw = int((val / 100) * (G_MAX - G_MIN) + G_MIN)
                raw = max(G_MIN, min(G_MAX, raw))
                cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, 6, 5, 0x03, 0x2A, raw)
                cks = (~sum(cmd[2:]) & 0xFF)
                arm.ser.write(cmd + struct.pack("<B", cks))
            else:
                arm._write_angle(sid, float(np.deg2rad(val)))

    for f_idx in range(len(frames) - 1):
        f0, f1 = frames[f_idx], frames[f_idx + 1]
        loop_t = time.perf_counter()

        # 帧间插值：每帧拆成 3 小步
        for step in range(1, 4):
            t = step / 3.0
            for sid, (k1, k2) in enumerate(JOINT_KEYS, 1):
                v0 = get_val(f0, k1, k2)
                v1 = get_val(f1, k1, k2)
                val = v0 + (v1 - v0) * t
                if sid == 6:
                    raw = int((val / 100) * (G_MAX - G_MIN) + G_MIN)
                    raw = max(G_MIN, min(G_MAX, raw))
                    cmd = struct.pack("<BBBBBBH", 0xFF, 0xFF, 6, 5, 0x03, 0x2A, raw)
                    cks = (~sum(cmd[2:]) & 0xFF)
                    arm.ser.write(cmd + struct.pack("<B", cks))
                else:
                    arm._write_angle(sid, float(np.deg2rad(val)))

        dt = time.perf_counter() - loop_t
        time.sleep(max(1 / args.fps - dt, 0))
        if (f_idx + 1) % 30 == 0:
            print("  {}/{}".format(f_idx + 1, len(frames)))

    # 发送最后一帧
    send_frame(frames[-1])
    print("完成: {:.1f}s".format(time.perf_counter() - t0))
    if args.home or args.initial:
        print("归零中...")
        arm.write_angles(arm.HOME_POSE)
        time.sleep(2)
        print("归零中...")
        arm.home(steps=30, delay_s=0.03)
    arm.close()


if __name__ == "__main__":
    main()

