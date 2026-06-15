#!/usr/bin/env python3
"""语音控制 VLA 抓取演示 — whisper.cpp + VLM + 机械臂"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    CAMERA_MATRIX, SERIAL_PORT, SERIAL_BAUD,
    VLM_MODEL_NAME, VLM_MODEL_PATH, VLM_DEMO_BIN,
)
from astra import AstraProCamera
from vla.vlm import create_vlm
from vla.vision import ColorLocator
from vla.control import ArmController
from vla.voice import VoiceControl


def main():
    print("=" * 50)
    print("  语音控制 VLA 系统")
    print("  说：\"抓红色杯子\" → VLM 确认 → 机械臂抓取")
    print("=" * 50)

    # 1. 加载组件
    vlm = create_vlm(VLM_MODEL_NAME, demo_bin=VLM_DEMO_BIN)
    vlm.load(VLM_MODEL_PATH)
    cam = AstraProCamera(21)
    cam.connect()
    arm = ArmController(SERIAL_PORT, SERIAL_BAUD)
    locator = ColorLocator(CAMERA_MATRIX)
    voice = VoiceControl(lang="zh")

    if not voice._ready:
        print("\n⚠ whisper.cpp 未安装，运行: bash scripts/install_whisper.sh")
        print("   先测试 VLM + 相机\n")

    try:
        while True:
            print("\n" + "-" * 40)
            rgb, depth = cam.read()

            if voice._ready:
                inp = input("按 Enter 录音指令, 或输入文字指令, q 退出: ")
            else:
                inp = input("输入指令 (如 \"抓红色杯子\"), q 退出: ")

            if inp.lower() == "q":
                break
            if not inp.strip():
                continue

            # 检查是否为语音指令（用户输入的是音频路径）
            if inp.startswith("/audio/") and os.path.exists(inp) and voice._ready:
                print("🎤 语音识别中...")
                text = voice.listen(inp)
                cmd = voice.parse_command(text)
            else:
                # 文字指令直接解析
                cmd = voice.parse_command(inp)

            if cmd is None:
                print("  未识别到有效指令，试试: 抓红色杯子")
                continue

            color = cmd.get("color", "红色")
            obj = cmd.get("object", "物体")
            print(f"  🎯 指令: {cmd['raw']} → 颜色={color} 物体={obj}")

            # VLM 确认
            import cv2
            tmp = "/tmp/vla_voice.jpg"
            cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            result = vlm.infer(tmp)
            print(f"  VLM: {result.color} {result.object[:20]}")

            # 定位
            pos = locator.locate(rgb, depth, color)
            if pos is None:
                print(f"  定位失败: 未找到{color}色物体")
                continue
            print(f"  定位: ({pos['x']:.3f},{pos['y']:.3f},{pos['z']:.3f})")

            # 抓取
            if cmd["action"] == "grasp":
                print("  🤖 开始抓取...")
                arm.move_to(pos["x"], pos["y"], pos["z"])
                arm.gripper(False)
                time.sleep(1)
                arm.move_to(0.30, 0.0, 0.10)
                arm.gripper(True)
                print("  ✅ 完成!")

    except KeyboardInterrupt:
        pass
    finally:
        cam.disconnect()
        arm.close()
        vlm.unload()
        print("\n结束")


if __name__ == "__main__":
    main()
