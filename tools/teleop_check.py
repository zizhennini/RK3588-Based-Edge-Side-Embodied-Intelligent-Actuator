#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/teleop_check.py — 主从零点对齐检查（只读，不驱动舵机）

背景: 遥操作跟随的"对不上/恒定偏差"几乎都来自**两臂角度零点不在同一物理位置**。
本工具把两臂当前姿态按各自标定换算成角度，打印逐关节差值，把主观感受变成可测量数字。

用法::

    python tools/teleop_check.py \
        --leader  /dev/ttyACM1 --leader-calib  config/calibration_leader.json \
        --follower /dev/ttyACM0 --follower-calib config/calibration.json

判读（把两臂摆成**目测完全相同的姿态**后运行）:
  - 各关节差值都 < 5°          → 零点对齐 ✓ 可以开始跟随
  - 差值近似为**恒定偏移**      → 两臂标定时的"中位姿态"物理位置不同：
                                重标两只臂，且必须摆成同一个标准姿态
                                （底座朝前、大臂竖直向上、小臂水平向前、腕水平、夹爪半开）
  - 差值**符号相反/数值很大**   → 该关节编码器计数方向不一致（跑一次 configure
                                或检查 Phase bit6），此时跟随会反向
  - 夹爪(id6) 单独偏差大        → 夹爪零点沿用旧标定，属已知语义，可忽略
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from hardware.feetech_bus import FeetechBus, GRIPPER_MOTOR_ID, raw_to_rad  # noqa: E402

JOINT_LABELS = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
                4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}


def read_deg(port: str, calib_path: str, name: str) -> dict:
    """按该臂自己的标定把当前 raw 换算为角度（度），夹爪按既有语义走 homing 零点"""
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    bus = FeetechBus(port, name=name)
    bus.connect(handshake=False)
    try:
        raw = bus.sync_read("Present_Position", num_retry=2)
        return {m: float(np.rad2deg(raw_to_rad(v, calib[str(m)],
                                               m != GRIPPER_MOTOR_ID)))
                for m, v in raw.items()}
    finally:
        bus.disconnect(disable_torque=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="主从零点对齐检查（只读）")
    ap.add_argument("--leader", default="/dev/ttyACM1")
    ap.add_argument("--follower", default="/dev/ttyACM0")
    ap.add_argument("--leader-calib", default="config/calibration_leader.json")
    ap.add_argument("--follower-calib", default="config/calibration.json")
    args = ap.parse_args()

    lead = read_deg(args.leader, args.leader_calib, "leader")
    foll = read_deg(args.follower, args.follower_calib, "follower")
    if len(lead) < 6 or len(foll) < 6:
        print(f"读取不全: leader {sorted(lead)} / follower {sorted(foll)}")
        return 1

    print("\n关节            主臂(°)   从臂(°)   差值(从-主)")
    print("-" * 48)
    diffs = {}
    for m in range(1, 7):
        d = foll[m] - lead[m]
        diffs[m] = d
        print(f"{JOINT_LABELS[m]:<14} {lead[m]:8.1f} {foll[m]:8.1f} {d:9.1f}")

    body = {m: diffs[m] for m in range(1, 6)}
    worst = max(body, key=lambda m: abs(body[m]))
    print("-" * 48)
    print(f"体关节最大偏差: {JOINT_LABELS[worst]} {body[worst]:+.1f}°")

    if all(abs(v) < 5.0 for v in body.values()):
        print("结论: 零点对齐 ✓ 可以开始跟随（两臂当前姿态目测重合度越准，此判定越可靠）")
        rc = 0
    elif max(abs(v) for v in body.values()) > 90:
        rev = [JOINT_LABELS[m] for m in body if abs(body[m]) > 90]
        print(f"结论: {rev} 偏差过大/疑似方向相反 ✗")
        print("  → 先跑一次 configure 对齐 Phase 计数方向，再重标该臂；")
        print("    若重标后仍反向，两臂装配方向不同（需 drive_mode 软件翻转兜底）")
        rc = 2
    else:
        print("结论: 存在恒定偏差 ✗ —— 两臂标定时的中位姿态不在同一物理位置")
        print("  → 重标两只臂，且必须摆成同一个标准姿态（底座朝前、大臂竖直向上、")
        print("    小臂水平向前、腕水平、夹爪半开），再跑一次本检查")
        rc = 3
    return rc


if __name__ == "__main__":
    sys.exit(main())
