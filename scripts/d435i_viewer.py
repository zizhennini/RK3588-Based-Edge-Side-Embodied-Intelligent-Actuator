#!/usr/bin/env python3
"""D435i 实时预览 — pyrealsense2 驱动，按 q 退出"""
import cv2, numpy as np, pyrealsense2 as rs

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)

# 开启自动曝光和自动白平衡
sensor = profile.get_device().first_color_sensor()
for opt in (rs.option.enable_auto_exposure, rs.option.enable_auto_white_balance):
    if sensor.supports(opt):
        sensor.set_option(opt, 1.0)
        print(f"  {opt}: 已开启")
    else:
        print(f"  {opt}: 不支持")

# 等待自动曝光稳定
import time
time.sleep(0.5)

print("D435i 实时预览 — 按 q 退出")
try:
    while True:
        frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())
        cv2.imshow("D435i Preview", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()
