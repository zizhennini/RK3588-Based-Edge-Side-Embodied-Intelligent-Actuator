#!/usr/bin/env python3
"""RK3588-EIA 主入口 — VLM 端侧具身智能系统"""

import sys
sys.path.insert(0, ".")

from config.settings import (
    CAMERA_MATRIX, SERIAL_PORT, SERIAL_BAUD,
    VLM_MODEL_NAME, RKLLM_BIN, VLM_MODEL_PATH, VLM_DEMO_BIN,
)
from astra import AstraProCamera
from vla.vlm import create_vlm
from vla.control import ArmController
from vla.pipe import VLApipeline
from vla.pipe.pipeline import State


def main():
    vlm = None
    cam = None
    arm = None
    try:
        print("[VLA] 加载 VLM ...")
        vlm = create_vlm(VLM_MODEL_NAME, RKLLM_BIN, VLM_DEMO_BIN)
        vlm.load(VLM_MODEL_PATH)

        print("[VLA] 打开 Astra Pro ...")
        cam = AstraProCamera()
        cam.connect()

        print("[VLA] 连接 SO-ARM101 ...")
        arm = ArmController(SERIAL_PORT, SERIAL_BAUD)

        pipe = VLApipeline(arm, vlm, CAMERA_MATRIX)

        print("\n=== VLA 端侧具身智能系统 ===")
        print("VLM 正在理解场景（约 5-8 秒）...")

        rgb, depth = cam.read()
        pipe.start()
        while True:
            status = pipe.step(rgb, depth)
            print(f"  [{status}]")
            if pipe.state == State.DONE:
                break
    except KeyboardInterrupt:
        print("\n[VLA] 用户中断")
    except Exception as e:
        print(f"\n[VLA] 错误: {e}")
    finally:
        if cam:
            cam.disconnect()
        if arm:
            arm.close()
        if vlm:
            vlm.unload()
        print("完成")


if __name__ == "__main__":
    main()
