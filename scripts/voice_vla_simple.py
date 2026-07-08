#!/usr/bin/env python3
"""简易语音 VLA — 按回车录音 → ASR → 拍照 → VLM → TTS"""
import sys, os, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.cpu_affinity import bind_current_thread, BIG_CORES
from config.settings import VLM_MODEL_NAME, VLM_MODEL_PATH, VLM_DEMO_BIN
from vla.command_queue import CommandQueue, create_voice_motion_callback
from vla.vlm.smart_vlm import SmartVLM, IdleUnloader
from voice_assistant.voice_assistant.asr import SherpaAsr
from voice_assistant.voice_assistant.config import load_config


def main():
    bind_current_thread(BIG_CORES)

    print("[VLA] 加载 ASR...")
    config = load_config()
    asr = SherpaAsr(config)

    print("[VLA] 连接相机...")
    cam = None
    try:
        from camera import D435iCamera
        cam = D435iCamera()
        cam.connect()
    except Exception as e:
        print(f"  相机: {e}")

    print("[VLA] 创建 SmartVLM...")
    smart_vlm = SmartVLM(VLM_MODEL_NAME, VLM_MODEL_PATH, demo_bin=VLM_DEMO_BIN)
    unloader = IdleUnloader(smart_vlm)
    unloader.start()

    print("[VLA] 启动指令队列...")
    queue = CommandQueue(smart_vlm=smart_vlm, camera=cam)
    queue.start()
    on_text = create_voice_motion_callback(queue)

    print("\n" + "=" * 50)
    print("简易语音 VLA — 按回车录音 4 秒")
    print("=" * 50)
    print("对着麦克风说指令，例如:")
    print("  '这是什么'  → 拍照 + VLM 识别 + TTS 播报")
    print("  '抓取'      → 匹配动作库")
    print("  '停止'      → 中断")
    print("  q+回车      → 退出")
    print()

    try:
        while True:
            cmd = input("按回车录音> ").strip().lower()
            if cmd == "q":
                break

            print("  录音中...", end=" ", flush=True)
            subprocess.run(["arecord", "-D", "hw:rockchipnau8822,0",
                           "-d", "4", "-f", "cd", "-t", "wav",
                           "/tmp/vla_cmd.wav"], capture_output=True)
            print("识别中...", flush=True)

            text = asr.transcribe_wav("/tmp/vla_cmd.wav")
            os.unlink("/tmp/vla_cmd.wav")

            if not text or not text.strip():
                print("  [未听清，请重试]")
                continue

            print(f"  [{text}]")
            on_text(text.strip())

    except KeyboardInterrupt:
        pass
    finally:
        unloader.stop()
        queue.stop()
        smart_vlm.unload()
        if cam:
            cam.disconnect()


if __name__ == "__main__":
    main()
