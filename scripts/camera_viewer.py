#!/usr/bin/env python3
"""相机实时取景 + 自动 VLM 推理（按 Enter 触发）"""
import sys, os, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from camera import USBCamera
from vla.vlm import create_vlm
from config.settings import VLM_MODEL_NAME, VLM_MODEL_PATH, VLM_DEMO_BIN, CAMERA_INDEX

inferring = False
result_text = ""
WINDOW_SCALE = 1.2  # 显示缩放倍率（640x480 放大到 768x576）

def do_inference(vlm, rgb):
    global inferring, result_text
    try:
        cv2.imwrite("/tmp/snap.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        r = vlm.infer("/tmp/snap.jpg")
        result_text = f"{r.color} {r.object[:30]}"
        print(f"\n📸 VLM: {result_text}")
    except Exception as e:
        result_text = f"错误: {e}"
        print(f"  ✗ {e}")
    finally:
        inferring = False

def main():
    global inferring, result_text
    vlm = create_vlm(VLM_MODEL_NAME, demo_bin=VLM_DEMO_BIN)
    vlm.load(VLM_MODEL_PATH)

    cam = USBCamera(CAMERA_INDEX)
    cam.connect()
    print("Enter → VLM | q 退出")
    print("首次启动请按 Enter 开始推理")

    try:
        while True:
                rgb = cam.read_rgb()
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if result_text:
                    cv2.putText(display, result_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if inferring:
                    cv2.putText(display, "推理中...", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if WINDOW_SCALE != 1.0:
                    h, w = display.shape[:2]
                    display = cv2.resize(display, (int(w * WINDOW_SCALE), int(h * WINDOW_SCALE)))
                cv2.imshow("RK3588-EIA", display)
                key = cv2.waitKey(10) & 0xFF

                if key == ord("q"):
                    break
                if key in (13, 10) and not inferring:
                    inferring = True
                    result_text = "推理中..."
                    t = threading.Thread(target=do_inference, args=(vlm, rgb))
                    t.daemon = True
                    t.start()
    except KeyboardInterrupt:
        pass
    finally:
        cam.disconnect()
        cv2.destroyAllWindows()
        vlm.unload()

if __name__ == "__main__":
    main()
