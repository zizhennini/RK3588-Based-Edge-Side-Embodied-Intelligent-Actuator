#!/usr/bin/env python3
"""一键遥操作录制"""
import sys, os, json, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="一键遥操作录制")
    parser.add_argument("--follower_port", default="/dev/ttyACM0")
    parser.add_argument("--leader_port", default="/dev/ttyACM1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time_s", type=int, default=15)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from lerobot.robots.so_follower.so_follower import SO100Follower
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SOLeader
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    out_path = args.out or "teleop_record_{}.json".format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    cfg_f = SO101FollowerConfig(port=args.follower_port, use_degrees=True, id="my_awesome_follower_arm")
    follower = SO100Follower(cfg_f)
    follower.connect()
    print("从臂已连接")

    cfg_l = SO101LeaderConfig(port=args.leader_port, use_degrees=True, id="my_awesome_leader_arm")
    leader = SOLeader(cfg_l)
    leader.connect()
    print("主臂已连接 - 开始录制 ({}秒)".format(args.episode_time_s))

    frames = []
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < args.episode_time_s:
            loop_t = time.perf_counter()
            obs = follower.get_observation()
            action = leader.get_action()
            follower.send_action(action)

            frame = {"t": round(time.perf_counter() - t0, 3)}
            for name in JOINT_NAMES:
                val = obs.get("{}.pos".format(name), 0.0)
                frame[name] = round(float(val), 1)
            frames.append(frame)

            dt = time.perf_counter() - loop_t
            time.sleep(max(1 / args.fps - dt, 0))
    finally:
        follower.disconnect()
        leader.disconnect()

    data = {"fps": args.fps, "total_frames": len(frames), "duration_s": args.episode_time_s, "frames": frames}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("{} 帧, 已保存 {}".format(len(frames), out_path))

if __name__ == "__main__":
    main()
