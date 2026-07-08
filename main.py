#!/usr/bin/env python3
"""RK3588-EIA VLA 自主抓取流水线"""
import sys, time
sys.path.insert(0, ".")
from config.settings import CAMERA_MATRIX, CAMERA_INDEX, SERIAL_PORT, SERIAL_BAUD, VLM_MODEL_NAME, VLM_MODEL_PATH, VLM_DEMO_BIN, VLM_IDLE_UNLOAD_TIMEOUT
from config.cpu_affinity import bind_current_thread, BIG_CORES, affinity_summary
from config.memory import MemoryMonitor
from camera import D435iCamera
from vla.vlm.smart_vlm import SmartVLM, IdleUnloader
from vla.vision import ColorLocator
from vla.control import ArmController
from vla.pipe import VLApipeline
from vla.pipe.pipeline import State

def main():
    bind_current_thread(BIG_CORES)
    print(affinity_summary())
    mem = MemoryMonitor()
    smart_vlm = None; cam = None; arm = None; unloader = None
    try:
        print("[VLA] 创建 SmartVLM...")
        smart_vlm = SmartVLM(VLM_MODEL_NAME, VLM_MODEL_PATH, demo_bin=VLM_DEMO_BIN, idle_timeout=VLM_IDLE_UNLOAD_TIMEOUT)
        unloader = IdleUnloader(smart_vlm, interval=5.0); unloader.start()
        print("[VLA] 打开 D435i...")
        cam = D435iCamera(CAMERA_INDEX); cam.connect()
        locator = ColorLocator(CAMERA_MATRIX)
        print("[VLA] 连接 SO-ARM101...")
        arm = ArmController(SERIAL_PORT, SERIAL_BAUD)
        pipe = VLApipeline(arm, smart_vlm, CAMERA_MATRIX)
        print("\n=== VLA 自主抓取 ==="); print(f"内存: {mem.summary()}"); input()
        pipe.start()
        while True:
            rgb, depth = cam.read()
            status = pipe.step(rgb, depth)
            print(f"  [{status}]")
            if pipe.state == State.DONE: break
    except KeyboardInterrupt: print("\n[VLA] 中断")
    except Exception as e: print(f"\n[VLA] 错误: {e}")
    finally:
        if unloader: unloader.stop()
        if cam: cam.disconnect()
        if arm: arm.close()
        if smart_vlm: smart_vlm.unload()
        print("完成")

if __name__ == "__main__": main()
