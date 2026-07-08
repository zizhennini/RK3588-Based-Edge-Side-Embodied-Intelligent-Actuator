#!/usr/bin/env python3
"""一键遥操作录制（含相机校准数据）"""
import sys, os, json, time, datetime, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CALIB_DIR = "/tmp/calib_capture"


def find_marker(img):
    """检测红色标记物中心像素"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([179, 255, 255]))
    mask = m1 | m2
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 50: return None
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="一键遥操作录制（相机校准版）")
    parser.add_argument("--follower_port", default="/dev/ttyACM0")
    parser.add_argument("--leader_port", default="/dev/ttyACM1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time_s", type=int, default=30)
    parser.add_argument("--out", default=None)
    parser.add_argument("--calib", action="store_true", help="同时录制相机校准数据")
    args = parser.parse_args()

    from lerobot.robots.so_follower.so_follower import SO100Follower
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SOLeader
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    out_path = args.out or "teleop_record_{}.json".format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # 如果启用校准，启动相机
    cam = None
    calib_data = []
    if args.calib:
        from camera import D435iCamera
        os.makedirs(CALIB_DIR, exist_ok=True)
        cam = D435iCamera()
        cam.connect()
        print("校准模式: 请在从臂夹爪上贴红色标记物")

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
    last_cap = 0
    try:
        while time.perf_counter() - t0 < args.episode_time_s:
            loop_t = time.perf_counter()
            obs = follower.get_observation()
            action = leader.get_action()
            follower.send_action(action)

            now = time.perf_counter() - t0
            frame = {"t": round(now, 3)}
            for name in JOINT_NAMES:
                val = obs.get("{}.pos".format(name), 0.0)
                frame[name] = round(float(val), 1)
            frames.append(frame)

            # 校准数据采集（每 1 秒采一帧）
            if args.calib and cam and now - last_cap > 1.0:
                rgb, depth = cam.read()
                px = find_marker(rgb)
                if px:
                    joints = [frame.get(n, 0) for n in JOINT_NAMES]
                    calib_data.append({"pixel": list(px), "joints": joints, "t": now})
                    print(f"  [校准] 像素({px[0]},{px[1]}) 时间={now:.1f}s")
                last_cap = now
            dt = time.perf_counter() - loop_t
            time.sleep(max(1 / args.fps - dt, 0))
    except KeyboardInterrupt:
        print("\n录制中断")
    finally:
        data = {"fps": args.fps, "total_frames": len(frames), "duration_s": args.episode_time_s, "frames": frames}
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print("{} 帧, 已保存 {}".format(len(frames), out_path))

        if calib_data:
            calib_path = out_path.replace(".json", "_calib.json")
            with open(calib_path, "w") as f:
                json.dump({"points": calib_data}, f, indent=2)
            print("{} 个校准点, 已保存 {}".format(len(calib_data), calib_path))

        if cam:
            cam.disconnect()
        try:
            follower.disconnect()
        except Exception as e:
            print("断开从臂时出错: {}".format(e))
        try:
            leader.disconnect()
        except Exception as e:
            print("断开主臂时出错: {}".format(e))

if __name__ == "__main__":
    main()
