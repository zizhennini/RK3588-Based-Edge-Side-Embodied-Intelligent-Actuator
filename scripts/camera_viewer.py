#!/usr/bin/env python3
"""相机取景器（实时画面不卡顿）+ Enter 触发 VLM+Depth（后台线程）"""
import sys, os, threading, glob, subprocess, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from config.settings import VLM_MODEL_PATH, VLM_DEMO_BIN
from astra import AstraProCamera

inferring = False


def do_inference(cam, rknn, rkllm):
    global inferring
    try:
        rgb, depth = cam.read()
        print(f"\n📸 Depth: {depth.min():.2f}~{depth.max():.2f}m")

        cmd = [VLM_DEMO_BIN, "/tmp/snap.jpg", rknn, rkllm,
               "128", "2048", "3",
               "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
        try:
            result = subprocess.run(cmd,
                                    input="<image>有什么物体？什么颜色？用JSON回答\n",
                                    capture_output=True, text=True, timeout=20)
            output = result.stdout
        except subprocess.TimeoutExpired as e:
            output = (e.stdout or b"").decode()

        import re
        m = re.search(r'\{[^}]+\}', output)
        if m:
            print(f"  VLM: {m.group()}")
        else:
            lines = [l for l in output.split('\n') if 'robot:' in l]
            print(f"  VLM: {lines[-1][:100] if lines else '无输出'}")
    except Exception as e:
        print(f"  ✗ {e}")
    finally:
        inferring = False
        print("  按 Enter 继续")


def main():
    global inferring
    rknn = glob.glob(VLM_MODEL_PATH + "/*.rknn")[0]
    rkllm = glob.glob(VLM_MODEL_PATH + "/*.rkllm")[0]

    cam = AstraProCamera(21)
    cam.connect()
    print("  Enter → VLM+Depth | q 退出")

    try:
        while True:
            ret, frame = cam.rgb_cap.read()
            if not ret:
                continue

            if inferring:
                cv2.putText(frame, "VLM inference...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("RK3588-EIA", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key in (13, 10) and not inferring:
                inferring = True
                cv2.imwrite("/tmp/snap.jpg", frame)
                t = threading.Thread(target=do_inference, args=(cam, rknn, rkllm))
                t.daemon = True
                t.start()
    except KeyboardInterrupt:
        pass
    finally:
        cam.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
