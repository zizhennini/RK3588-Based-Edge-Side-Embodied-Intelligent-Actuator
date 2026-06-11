#!/usr/bin/env python3
"""相机实时取景 + 自动 VLM 推理（按 Enter 触发）"""
import sys, os, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from astra import AstraProCamera
from vla.vlm import create_vlm
from config.settings import VLM_MODEL_NAME, VLM_MODEL_PATH, VLM_DEMO_BIN

inferring = False
result_text = ""

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

    cam = AstraProCamera(21)
    cam.connect()
    print("Enter → VLM | q 退出")
    print("首次启动请按 Enter 开始推理")

    try:
        while True:
            ret, frame = cam.rgb_cap.read()
            if not ret:
                continue

            if result_text:
                cv2.putText(frame, result_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if inferring:
                cv2.putText(frame, "推理中...", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("RK3588-EIA", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key in (13, 10) and not inferring:
                inferring = True
                result_text = "推理中..."
                t = threading.Thread(target=do_inference, args=(vlm, frame))
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
