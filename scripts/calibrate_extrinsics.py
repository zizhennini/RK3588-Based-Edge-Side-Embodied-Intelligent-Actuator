#!/usr/bin/env python3
"""相机外参标定 — 测量相机相对机械臂基座的位置并写入配置"""

import sys, os, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def get_input(prompt: str, default: float = 0.0) -> float:
    while True:
        try:
            val = input(prompt)
            if not val.strip():
                return default
            return float(val)
        except ValueError:
            print("  请输入数字")


def main():
    print("=" * 50)
    print("  相机外参标定")
    print("=" * 50)
    print()
    print("请用尺子测量以下三个距离（单位：米）:")
    print()
    print("  坐标系说明:")
    print("    X: 机械臂正前方为正")
    print("    Y: 机械臂正左方为正")
    print("    Z: 垂直向上为正")
    print()
    print("  测量参考点:")
    print("    相机: D435i RGB 镜头光心")
    print("    基座: 机械臂底座中心点")
    print()

    x = get_input("  相机在基座前方(+) / 后方(-) (X) [m]: ", 0.0)
    y = get_input("  相机在基座左方(+) / 右方(-) (Y) [m]: ", 0.0)
    z = get_input("  相机高度(镜头到桌面的垂直距离) (Z) [m]: ", 0.0)

    print()
    print(f"  输入值: X={x:.3f}  Y={y:.3f}  Z={z:.3f}")
    try:
        r = input("  确认写入? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        r = "n"

    if r != "y":
        print("  已取消")
        return

    # 更新 settings.py
    settings_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.py")
    with open(settings_path) as f:
        content = f.read()

    new_line = f"CAMERA_POSITION = np.array([{x:.4f}, {y:.4f}, {z:.4f}], dtype=float)  # [x, y, z] 标定值"
    content = re.sub(
        r"CAMERA_POSITION = np\.array\(\[.*?\], dtype=float\).*",
        new_line, content
    )

    with open(settings_path, "w") as f:
        f.write(content)

    print(f"  已写入 CAMERA_POSITION = [{x:.4f}, {y:.4f}, {z:.4f}]")


if __name__ == "__main__":
    main()
