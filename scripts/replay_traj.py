"""示教轨迹回放 — 加载 JSON → 逐帧发送"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vla.control import ArmController
from config.settings import SERIAL_PORT, SERIAL_BAUD

# 支持两种命名格式
JOINT_KEYS = [
    ("J1", "shoulder_pan"), ("J2", "shoulder_lift"), ("J3", "elbow_flex"),
    ("J4", "wrist_flex"), ("J5", "wrist_roll"), ("J6", "gripper"),
]


def get_deg(frame: dict, key1: str, key2: str) -> float:
    return float(frame.get(key1, frame.get(key2, 0.0)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--port", default=SERIAL_PORT)
    args = parser.parse_args()

    with open(args.traj_file) as f:
        data = json.load(f)

    frames = data["frames"]
    print("轨迹: {}帧, {}FPS".format(len(frames), data.get("fps", "?")))

    arm = ArmController(args.port, SERIAL_BAUD)
    print("回放中 ({}FPS)...".format(args.fps))
    t0 = time.perf_counter()

    for i, frame in enumerate(frames):
        loop_t = time.perf_counter()
        for sid, (k1, k2) in enumerate(JOINT_KEYS, 1):
            deg = get_deg(frame, k1, k2)
            arm._write_angle(sid, float(np.deg2rad(deg)))

        dt = time.perf_counter() - loop_t
        time.sleep(max(1 / args.fps - dt, 0))

        if (i + 1) % 30 == 0:
            print("  {}/{}".format(i + 1, len(frames)))

    elapsed = time.perf_counter() - t0
    print("完成: {:.1f}s, {:.1f}FPS".format(elapsed, len(frames) / elapsed))
    arm.close()


if __name__ == "__main__":
    main()
