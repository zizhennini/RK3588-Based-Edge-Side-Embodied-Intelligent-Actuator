#!/usr/bin/env python3
"""VLM + SSD + 相机 三联测（不需要机械臂）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    CAMERA_MATRIX, VLM_MODEL_NAME, VLM_MODEL_PATH, VLM_DEMO_BIN,
)

SSD_PROTOTXT = "./models/MobileNetSSD/MobileNetSSD_deploy.prototxt"
SSD_CAFFEMODEL = "./models/MobileNetSSD/MobileNetSSD_deploy.caffemodel"
SSD_CONFIDENCE = 0.5
from camera import D435iCamera as AstraProCamera
from vla.vlm import create_vlm
from vla.vision import MobileNetSSD, ColorLocator


def main():
    print("=" * 50)
    print("VLM + SSD + 相机 三联测试")
    print("=" * 50)

    # 1. 加载 VLM
    print("\n[1/4] 加载 VLM ...")
    vlm = create_vlm(VLM_MODEL_NAME, demo_bin=VLM_DEMO_BIN)
    vlm.load(VLM_MODEL_PATH)
    print("  ✓ VLM 加载完成")

    # 2. 加载 SSD
    print("\n[2/4] 加载 MobileNet SSD ...")
    detector = MobileNetSSD(SSD_PROTOTXT, SSD_CAFFEMODEL, SSD_CONFIDENCE)
    print("  ✓ SSD 加载完成")

    # 3. 打开相机
    print("\n[3/4] 打开 Astra Pro 相机 ...")
    cam = AstraProCamera(rgb_index=21)
    cam.connect()
    print("  ✓ 相机已连接")

    # 4. 拍摄 + VLM + SSD
    print("\n[4/4] 拍摄并推理 ...\n")

    rgb, depth = cam.read()
    tmp = "/tmp/vla_test.jpg"
    import cv2
    cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  相机: RGB {rgb.shape}, Depth {depth.shape}")

    # VLM 推理
    print("\n  ▶ VLM 推理中（等待 5-15 秒）...")
    result = vlm.infer(tmp)
    print(f"  VLM: 目标={result.object}, 颜色={result.color}")

    # SSD 检测
    print("\n  ▶ SSD 目标检测中 ...")
    objs = detector.detect(rgb)
    print(f"  SSD: 检测到 {len(objs)} 个物体")
    for o in objs:
        print(f"    {o['label']}: {o['confidence']:.2f} @ ({o['cx']},{o['cy']})")

    # SSD 按 VLM 语义筛选
    target = result.object.lower()
    filtered = detector.detect(rgb, target_class=target)
    if filtered:
        best = filtered[0]
        z = float(depth[best["cy"], best["cx"]])
        print(f"\n  ▶ 匹配 VLM 语义 '{target}'")
        print(f"    位置: ({best['cx']}, {best['cy']}), 深度: {z:.3f}m")
    else:
        print(f"\n  ▶ SSD 未检测到 '{target}'，降级为颜色分割")
        locator = ColorLocator(CAMERA_MATRIX)
        pos = locator.locate(rgb, depth, result.color)
        if pos:
            print(f"    颜色定位: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
        else:
            print("    颜色定位失败（未找到目标）")

    cam.disconnect()
    vlm.unload()
    print("\n测试完成")


if __name__ == "__main__":
    main()
