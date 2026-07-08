#!/usr/bin/env python3
"""D435i 倾斜标定 — 拍桌面深度图，自动算出倾角补偿参数"""

import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def main():
    from camera import D435iCamera
    from config.settings import CAMERA_POSITION, CAMERA_MATRIX

    print("=" * 50)
    print("  D435i 倾斜标定")
    print("=" * 50)
    print()
    print("请确保桌面上无物体，D435i 为上帝视角")
    input("准备好后按 Enter 拍照...")
    print()

    cam = D435iCamera()
    cam.connect()
    _, depth = cam.read()
    cam.disconnect()

    h, w = depth.shape[:2]
    print(f"深度图: {w}x{h}")
    print()

    # 取 4 个角落和中心的深度均值
    margin = 50
    regions = {
        "左上": depth[margin:margin+30, margin:margin+30],
        "右上": depth[margin:margin+30, w-margin-30:w-margin],
        "左下": depth[h-margin-30:h-margin, margin:margin+30],
        "右下": depth[h-margin-30:h-margin, w-margin-30:w-margin],
        "中心": depth[h//2-15:h//2+15, w//2-15:w//2+15],
    }

    print("桌面各区域深度(m):")
    depths = {}
    for name, roi in regions.items():
        valid = roi[(roi > 0.01) & (~np.isnan(roi))]
        d = float(np.median(valid)) if len(valid) > 0 else 0
        depths[name] = d
        print(f"  {name}: {d:.3f}")
    print()

    # 计算倾斜
    if depths["左上"] > 0 and depths["右上"] > 0:
        dx = depths["左上"] - depths["右上"]  # 左右深度差
        dy = depths["左上"] - depths["左下"]  # 上下深度差
        center_depth = depths["中心"]
        print(f"左右深度差: {dx*1000:.0f}mm ({'左低右高' if dx>0 else '右低左高'})")
        print(f"上下深度差: {dy*1000:.0f}mm ({'上低下高' if dy>0 else '下低上高'})")

        # 估算倾角
        fov_h = 69.4  # D435i RGB 水平视场角(度)
        fov_v = 42.5  # D435i RGB 垂直视场角
        deg_per_pixel_h = fov_h / w
        deg_per_pixel_v = fov_v / h

        pixels_lr = (center_depth - abs(dx)) / center_depth * (w / 2)  # 粗略估算
        pixels_ud = (center_depth - abs(dy)) / center_depth * (h / 2)

        tilt_lr = deg_per_pixel_h * abs(dx) / center_depth * 180 / np.pi
        tilt_ud = deg_per_pixel_v * abs(dy) / center_depth * 180 / np.pi

        print(f"\n估算倾角:")
        print(f"  水平方向: {tilt_lr:.1f}°")
        print(f"  垂直方向: {tilt_ud:.1f}°")

        if abs(dx) < 0.005 and abs(dy) < 0.005:
            print("\n✅ 倾斜很小，不需要补偿")
        else:
            print(f"\n⚠ 倾斜 {max(tilt_lr, tilt_ud):.1f}°")

    print(f"\n当前 CAMERA_POSITION_Z = {CAMERA_POSITION[2]:.2f}m")
    print(f"桌面中心深度 = {depths['中心']:.3f}m")
    print(f"建议 CAMERA_POSITION_Z >= {depths['中心']+0.02:.2f}m（桌面深度+2cm余量）")


if __name__ == "__main__":
    main()
