#!/usr/bin/env python3
"""遥操作标定 — 拖拽主臂到不同位置，自动采集机器人坐标+像素坐标"""

import sys, os, json, time, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def find_marker(img):
    """检测红色标记物中心像素坐标"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([179, 255, 255]))
    mask = m1 | m2
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def main():
    from lerobot.robots.so_follower.so_follower import SO100Follower
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SOLeader
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from camera import D435iCamera
    from config.settings import CAMERA_MATRIX

    import argparse
    p = argparse.ArgumentParser(description="遥操作标定")
    p.add_argument("--follower", default="/dev/ttyACM0")
    p.add_argument("--leader", default="/dev/ttyACM1")
    p.add_argument("--points", type=int, default=12, help="采集点数")
    args = p.parse_args()

    # 连接从臂和主臂
    cfg_f = SO101FollowerConfig(port=args.follower, use_degrees=True, id="calib_follower")
    follower = SO100Follower(cfg_f)
    follower.connect()
    cfg_l = SO101LeaderConfig(port=args.leader, id="calib_leader")
    leader = SOLeader(cfg_l)
    leader.connect()

    # 连接相机
    cam = D435iCamera()
    cam.connect()

    print("\n=== 遥操作标定 ===")
    print("请拖拽主臂到不同位置，每到一个位置按 Enter 采集")
    print("在从臂夹爪上贴红色标记物")
    print(f"共采集 {args.points} 个点\n")

    data = []
    collected = 0
    while collected < args.points:
        # 读主臂关节角度
        leader_action = leader.get_action()
        leader_joints = {k: float(v) for k, v in zip(follower.bus.motors.keys(), leader_action)}

        # 拍照
        rgb, depth = cam.read()
        px = find_marker(rgb)
        if px is None:
            print(f"  [{collected+1}] 未检测到红色标记，请确保在画面中")
            continue

        # 保存预览图
        cv2.imwrite(f"/tmp/calib_point_{collected}.jpg",
                     cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        print(f"  [{collected+1}/{args.points}] 像素({px[0]},{px[1]}) 关节: ", end="")
        print({k: f"{v:.1f}" for k, v in leader_joints.items()})

        data.append({
            "pixel": [px[0], px[1]],
            "joints": {k: round(v, 4) for k, v in leader_joints.items()}
        })
        collected += 1

        if collected < args.points:
            input("    按 Enter 采集下一个点...")

    leader.disconnect()
    follower.disconnect()
    cam.disconnect()

    # 保存原始数据
    out = "calib_data.json"
    with open(out, "w") as f:
        json.dump({"points": data, "camera_matrix": CAMERA_MATRIX.tolist()}, f, indent=2)
    print(f"\n已保存 {out}")

    # 计算标定
    print("\n=== 计算标定 ===")
    print("需要从关节角度算出末端笛卡尔坐标")
    print("请运行: python3 scripts/calib_from_data.py")


if __name__ == "__main__":
    main()
