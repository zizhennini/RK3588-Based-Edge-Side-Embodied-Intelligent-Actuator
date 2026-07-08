#!/usr/bin/env python3
"""微调 CAMERA_POSITION — 逐步调整 x/y/z 偏移直到抓取到位"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from camera import D435iCamera
from vla.control import ArmController
from config.settings import (
    CAMERA_MATRIX, CAMERA_POSITION, SERIAL_PORT, SERIAL_BAUD
)
from vla.vision import ColorLocator, PCAGrasper

print("=" * 50)
print("  CAMERA_POSITION 微调")
print("=" * 50)
print()

# 当前值
pos = CAMERA_POSITION.copy()
print(f"当前 CAMERA_POSITION = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
print()

# 拍照 + PCA
cam = D435iCamera()
cam.connect()
rgb, depth = cam.read()
cam.disconnect()

loc = ColorLocator(CAMERA_MATRIX)
pg = PCAGrasper(CAMERA_MATRIX)

mask = loc.get_mask(rgb, "红色")
if np.sum(mask > 0) < 200:
    print("未检测到红色物体！")
    sys.exit(1)

result = pg.compute_from_mask(mask, depth)
if result is None:
    print("PCA 失败！")
    sys.exit(1)

cc = result["center_cam"]
print(f"物体相机坐标: x={cc[0]:.3f}  y={cc[1]:.3f}  z={cc[2]:.3f}")
print()

# 用当前 CAMERA_POSITION 算机器人坐标
robot_x = cc[0] + pos[0]
robot_y = cc[1] + pos[1]
robot_z = pos[2] - cc[2]
print("按当前 CAMERA_POSITION 算出的机械臂目标:")
print(f"  x={robot_x:.3f}  y={robot_y:.3f}  z={robot_z:.3f}")
print()

# 手动调整
print("请输入修正值（直接回车保持不变）：")
try:
    dx = float(input(f"  x 偏移 [{pos[0]:.3f}] → ") or pos[0])
except ValueError:
    dx = pos[0]
try:
    dy = float(input(f"  y 偏移 [{pos[1]:.3f}] → ") or pos[1])
except ValueError:
    dy = pos[1]
try:
    dz = float(input(f"  z 偏移 [{pos[2]:.3f}] → ") or pos[2])
except ValueError:
    dz = pos[2]

print()
print(f"新 CAMERA_POSITION = np.array([{dx:.3f}, {dy:.3f}, {dz:.3f}], dtype=float)")
print()
print("写入 settings.py 后运行:")
print(f"  python scripts/voice_grasp.py '抓住红色色块'")
