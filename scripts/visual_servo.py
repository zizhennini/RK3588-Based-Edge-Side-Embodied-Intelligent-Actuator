"""颜色视觉跟随 — 指定颜色，跟着色块走"""
import sys, time, cv2, numpy as np
sys.path.insert(0, ".")

from camera import USBCamera
from vla.vision import ColorLocator
from vla.control import ArmController
from config.settings import CAMERA_INDEX, SERIAL_PORT, SERIAL_BAUD, CAMERA_MATRIX


def main():
    target_color = sys.argv[1] if len(sys.argv) > 1 else "红色"
    print(f"跟踪颜色: {target_color}")

    cam = USBCamera(CAMERA_INDEX)
    cam.connect()
    arm = ArmController(SERIAL_PORT, SERIAL_BAUD)
    locator = ColorLocator(CAMERA_MATRIX)

    print("移到拍照位...")
    arm._write_angle(1, np.deg2rad(0))
    arm._write_angle(2, np.deg2rad(60))
    arm._write_angle(3, np.deg2rad(-60))
    arm._write_angle(4, np.deg2rad(0))
    time.sleep(3)

    cx, cy = 320, 240
    scale = 0.0005

    while True:
        rgb, _ = cam.read()
        pos = locator.locate(rgb, np.zeros((120, 160), dtype=np.float32), target_color)
        if pos is None:
            time.sleep(0.1)
            continue

        du = pos["u"] - cx
        dv = pos["v"] - cy
        if abs(du) < 10 and abs(dv) < 10:
            continue

        arm._write_angle(1, np.deg2rad(-du * scale))
        arm._write_angle(2, np.deg2rad(dv * scale))
        time.sleep(0.2)


if __name__ == "__main__":
    main()
