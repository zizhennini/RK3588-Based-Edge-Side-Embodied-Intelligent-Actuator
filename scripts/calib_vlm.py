#!/usr/bin/env python3
"""VLM 标定 — 放方块 → VLM 识别像素 → 遥操作到方块 → 记录"""

import sys, os, json, time, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voice_assistant.voice_assistant.qwen_runner import QwenRunner
from voice_assistant.voice_assistant.config import load_config
from camera import D435iCamera
from vla.control import ArmController
from config.settings import CAMERA_MATRIX
from vla.kinematics import forward_kinematics as fk
import serial, struct

from lerobot.robots.so_follower.so_follower import SO100Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NUM_POINTS = 5


def detect_block(qwen, cam):
    """VLM 识别紫色方块，返回像素坐标"""
    rgb, _ = cam.read()
    snap = "/tmp/calib_block.jpg"
    cv2.imwrite(snap, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    prompt = "<image>找到紫色方块，输出中心像素坐标: {\"cx\": 中心x, \"cy\": 中心y}。只输出JSON。"
    result = qwen.ask(snap, prompt)
    import re
    m = re.search(r"\{[^}]+\}", result, re.DOTALL)
    if m:
        import json as _j
        d = _j.loads(m.group())
        return int(d.get("cx", 0)), int(d.get("cy", 0))
    return None


def main():
    cfg = load_config()
    qwen = QwenRunner(cfg)
    cam = D435iCamera()
    cam.connect()

    # 连接主从臂（LeRobot 遥操作）
    cfg_f = SO101FollowerConfig(port="/dev/ttyACM0", use_degrees=True, id="calib_follower")
    follower = SO100Follower(cfg_f)
    follower.connect()
    cfg_l = SO101LeaderConfig(port="/dev/ttyACM1", id="calib_leader")
    leader = SOLeader(cfg_l)
    leader.connect()

    # 用 ArmController 做归零和读数
    arm = ArmController("/dev/ttyACM0")
    arm.home(steps=30)

    print("=" * 50)
    print("  VLM 标定 — 放方块 → 识别 → 遥操作 → 记录")
    print("=" * 50)
    print(f"\n共 {NUM_POINTS} 个点")
    print()

    data = []

    for i in range(NUM_POINTS):
        input(f"\n位置 {i+1}/{NUM_POINTS}: 放好紫色方块后按 Enter...")

        # VLM 识别
        det = detect_block(qwen, cam)
        if det is None:
            print("  ❌ 未识别到紫色方块，请重试")
            continue
        cx, cy = det
        print(f"  VLM 像素坐标: ({cx}, {cy})")

        # 遥操作：拖拽主臂 → 从臂跟随到方块位置
        print("  请拖拽主臂，让从臂到达紫色方块正上方")
        print("  到位后按 Enter 记录数据")
        while True:
            action = leader.get_action()
            follower.send_action(action)
            # 每循环检查是否有 Enter 键
            import select
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                input()  # 消耗 Enter
                break

        # 读从臂关节角度
        obs = follower.get_observation()
        joints = [obs.get(f"{n}.pos", 0) for n in JOINT_NAMES]
        joints_rad = [np.radians(j) for j in joints]
        print(f"  从臂关节角度: {[round(float(j),1) for j in joints]}")

        # 用 FK 算末端位置
        T = fk(np.array(joints_rad))
        ee_pos = [round(T[0,3],3), round(T[1,3],3), round(T[2,3],3)]
        print(f"  末端位置: {ee_pos}")

        data.append({
            "pixel": [cx, cy],
            "joints_deg": [round(float(j),1) for j in joints],
            "ee_pos": ee_pos,
        })

    leader.disconnect()
    follower.disconnect()
    cam.disconnect()
    arm.close()

    out = "vlm_calib_data.json"
    with open(out, "w") as f:
        json.dump({"points": data, "camera_matrix": CAMERA_MATRIX.tolist()}, f, indent=2)
    print(f"\n已保存 {out}，共 {len(data)} 个点")


if __name__ == "__main__":
    main()
