#!/usr/bin/env python3
"""快速一键标定：物体放相机下，机械臂移到物体上，自动计算 CAMERA_POSITION"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from camera import D435iCamera
from vla.control import ArmController
from vla.control.controller import NumericalIK
from config.settings import SERIAL_PORT, SERIAL_BAUD

print("=" * 50)
print("  快速手眼标定")
print("=" * 50)
print()
print("步骤 1: 把物体放在相机视野正中央")
input("准备好后按 Enter...")

# 读取深度
cam = D435iCamera()
cam.connect()
rgb, depth = cam.read()
h, w = depth.shape
cy, cx = h // 2, w // 2
dz = float(depth[cy, cx])
if dz <= 0 or np.isnan(dz):
    patch = depth[cy-4:cy+5, cx-4:cx+5]
    valid = patch[patch > 0]
    dz = float(np.median(valid)) if len(valid) > 0 else 0.35
cam.disconnect()
print(f"图像中心深度: {dz:.3f}m")

print()
print("步骤 2: 手动把机械臂夹爪移到物体正上方（可用遥控或手推）")
print("        脚本将读取当前关节角并自动计算末端位置")
input("移好后按 Enter...")

# 读取关节角 → FK 算末端位置
arm = ArmController(SERIAL_PORT, SERIAL_BAUD)
joints = arm._read_current_pos()
arm.close()

ik = NumericalIK()
T = ik.forward_kinematics(np.rad2deg(joints[:5]))
robot_pos = T[:3, 3]
print(f"关节角: {np.round(np.rad2deg(joints[:5]), 1)} deg")
print(f"末端位置 (机器人坐标): x={robot_pos[0]:.3f}  y={robot_pos[1]:.3f}  z={robot_pos[2]:.3f}")

# 计算 CAMERA_POSITION
rx, ry, rz = robot_pos
cam_pos_x = rx
cam_pos_y = ry
cam_pos_z = rz + dz

print()
print("=" * 50)
print("  标定结果")
print("=" * 50)
print(f"\nCAMERA_POSITION = np.array([{cam_pos_x:.3f}, {cam_pos_y:.3f}, {cam_pos_z:.3f}], dtype=float)")
print(f"\n请写入 config/settings.py:")
print(f'  CAMERA_POSITION = np.array([{cam_pos_x:.3f}, {cam_pos_y:.3f}, {cam_pos_z:.3f}], dtype=float)')
