#!/usr/bin/env python3
"""VLA 交互入口 — 语音/文字双输入 → 队列 → 按需拍照 → VLM → TTS"""
import sys, os, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.cpu_affinity import bind_current_thread, BIG_CORES
from config.settings import VLM_MODEL_NAME, VLM_MODEL_PATH, VLM_DEMO_BIN
from vla.command_queue import CommandQueue, create_voice_motion_callback
from vla.vlm.smart_vlm import SmartVLM, IdleUnloader
from voice_assistant.voice_assistant.config import load_config


def snapshot() -> str:
    import cv2
    try:
        from astra import D435iCamera
        cam = D435iCamera()
        cam.connect()
        rgb, _ = cam.read()
        cam.disconnect()
        path = "/tmp/vla_snap.jpg"
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"  照片: {os.path.getsize(path)} bytes", flush=True)
        return path
    except Exception as e:
        print(f"  拍照失败: {e}", flush=True)
        return ""


def text_input_loop(queue, on_text):
    """后台线程：读取终端文字输入，推入队列"""
    while True:
        try:
            cmd = input().strip()
            if cmd:
                print(f"[文字] \"{cmd}\"", flush=True)
                on_text(cmd)
        except (EOFError, KeyboardInterrupt):
            break
        except Exception:
            pass


def main():
    bind_current_thread(BIG_CORES)

    print("[VLA] 加载语音助手...", flush=True)
    config = load_config()
    from voice_assistant.voice_assistant.orchestrator import VoiceAssistant
    assistant = VoiceAssistant(config)

    print("[VLA] 创建 SmartVLM...", flush=True)
    smart_vlm = SmartVLM(VLM_MODEL_NAME, VLM_MODEL_PATH, demo_bin=VLM_DEMO_BIN)
    unloader = IdleUnloader(smart_vlm)
    unloader.start()

    print("[VLA] 预热 TTS（首次较慢）...", flush=True)
    from voice_assistant.voice_assistant.tts import SherpaTts
    from vla.command_queue import CommandQueue
    _warm = SherpaTts(load_config())
    _warm.synthesize_samples("预热")
    CommandQueue._tts = _warm

    print("[VLA] 启动指令队列...", flush=True)
    queue = CommandQueue(smart_vlm=smart_vlm, snapshot_cb=snapshot)
    queue.start()
    on_text = create_voice_motion_callback(queue)

    # 文字输入线程
    t = threading.Thread(target=text_input_loop, args=(queue, on_text), daemon=True)
    t.start()

    print("\n" + "=" * 50)
    print("VLA 交互 — 文字输入 / 语音唤醒")
    print("=" * 50)
    print("【文字】直接打字按回车")
    print("【语音】说唤醒词 '鲁班猫' 后说话")
    print("  示例: 这是什么 | 抓取 | 停止 | 看看有什么")
    print("  Ctrl+C 退出\n")

    try:
        while True:
            print("[等待唤醒词 '鲁班猫' 或输入文字...]", flush=True)
            keyword = assistant.wait_for_wake(mode="kws", timeout=None)
            print(f"[唤醒] {keyword}", flush=True)

            command_wav = assistant.record_command()
            text = assistant.transcribe_wav(command_wav)
            command_wav.unlink(missing_ok=True)

            if not text or not text.strip():
                print("[未听清]\n", flush=True)
                continue

            print(f"[语音] \"{text}\"", flush=True)
            on_text(text)

    except KeyboardInterrupt:
        print("\n[退出]", flush=True)
    finally:
        unloader.stop()
        queue.stop()
        smart_vlm.unload()


if __name__ == "__main__":
    main()
