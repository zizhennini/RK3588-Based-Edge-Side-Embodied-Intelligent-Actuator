#!/usr/bin/env python3
"""简单抓取 — 直接传相机坐标到 move_to（不做任何变换）"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from camera import D435iCamera
from config.settings import CAMERA_MATRIX, SERIAL_PORT, SERIAL_BAUD
from vla.control import ArmController
from vla.vision import ColorLocator, PCAGrasper

def main():
    arm = ArmController(SERIAL_PORT, SERIAL_BAUD)
    arm.home(steps=30, delay_s=0.03)
    print("归零完成")

    cam = D435iCamera()
    cam.connect()
    rgb, depth = cam.read()
    cam.disconnect()

    loc = ColorLocator(CAMERA_MATRIX)
    pg = PCAGrasper(CAMERA_MATRIX)

    mask = loc.get_mask(rgb, "红色")
    if np.sum(mask > 0) < 200:
        print("未检测到红色")
        arm.close()
        return

    pose = pg.compute_from_mask(mask, depth)
    if pose is None:
        print("PCA 失败")
        arm.close()
        return

    cx, cy, cz = pose["center_cam"]
    # 相机坐标 → 机器人坐标（加偏移）
    rx = cx - 0.10   # 相机在基座左侧 10cm
    ry = cy + 0.0    # Y 对齐
    rz = cz         # 深度直接作为高度
    print(f"PCA 相机坐标: x={cx:.3f}  y={cy:.3f}  z={cz:.3f}")
    print(f"机器人坐标: x={rx:.3f}  y={ry:.3f}  z={rz:.3f}")
    print(f"角度: {np.rad2deg(pose['angle']):.1f}deg  宽度: {pose['width']:.3f}m")

    arm.gripper(True)
    time.sleep(0.3)

    approach_z = rz - 0.05
    arm.move_to(rx, ry, max(approach_z, 0.05))
    print(f"接近: ({rx:.3f}, {ry:.3f}, {max(approach_z, 0.05):.3f})")
    time.sleep(0.5)

    arm.move_to(rx, ry, rz)
    print(f"抓取: ({rx:.3f}, {ry:.3f}, {rz:.3f})")
    time.sleep(0.3)

    arm.gripper(False)
    print("闭合夹爪")
    time.sleep(0.8)

    lift_z = rz + 0.10
    arm.move_to(rx, ry, min(lift_z, 0.50))
    print(f"抬升: ({rx:.3f}, {ry:.3f}, {min(lift_z, 0.50):.3f})")

    print("完成")
    arm.close()

if __name__ == "__main__":
    main()
