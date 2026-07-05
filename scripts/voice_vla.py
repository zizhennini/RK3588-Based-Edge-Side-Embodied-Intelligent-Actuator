#!/usr/bin/env python3
"""VLA 交互 — 语音/文字 分离模式"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.cpu_affinity import bind_current_thread, BIG_CORES
from config.settings import VLM_MODEL_NAME, VLM_MODEL_PATH, VLM_DEMO_BIN
from vla.command_queue import CommandQueue, create_voice_motion_callback
from vla.vlm.smart_vlm import SmartVLM, IdleUnloader


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
        return path
    except Exception as e:
        print(f"  拍照失败: {e}", flush=True)
        return ""


def run_text(q, on_text):
    """文字模式：一直等待输入"""
    print("\n文字模式 — 直接打字按回车，输入 q 退出\n")
    while True:
        try:
            c = input("> ").strip()
            if c.lower() == "q":
                break
            if c:
                print(f"[文字] \"{c}\"", flush=True)
                on_text(c)
        except (EOFError, KeyboardInterrupt):
            break


def run_voice(q, on_text):
    """语音模式：KWS 唤醒 → 录音 → ASR → 执行"""
    from voice_assistant.voice_assistant.config import load_config
    from voice_assistant.voice_assistant.wake import SherpaKeywordWake
    from voice_assistant.voice_assistant.asr import SherpaAsr
    from voice_assistant.voice_assistant.audio_io import AudioRecorder, apply_mic_mixer_settings

    cfg = load_config()
    apply_mic_mixer_settings(cfg)
    wake = SherpaKeywordWake(cfg)
    asr = SherpaAsr(cfg)
    recorder = AudioRecorder(cfg)
    temp_dir = cfg["paths"]["temp_dir"]
    os.makedirs(temp_dir, exist_ok=True)

    print("\n语音模式 — 说\"鲁班猫\"唤醒\n")
    try:
        while True:
            print("[等待唤醒词 '鲁班猫' ...]", flush=True)
            keyword = wake.wait(timeout=None)
            print(f"[唤醒] {keyword}", flush=True)

            wav = recorder.record_wav(os.path.join(temp_dir, "cmd.wav"), 4)
            text = asr.transcribe_wav(wav)
            os.unlink(wav)

            if text and text.strip():
                print(f"[指令] \"{text}\"", flush=True)
                on_text(text.strip())
            else:
                print("[未听清]", flush=True)
    except KeyboardInterrupt:
        print("\n[退出]", flush=True)


def main():
    bind_current_thread(BIG_CORES)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="voice",
                        choices=["voice", "text", "ask"])
    parser.add_argument("text", nargs="?", default="",
                        help="文本问答内容")
    args = parser.parse_args()

    print("[VLA] 创建 SmartVLM...", flush=True)
    smart_vlm = SmartVLM(VLM_MODEL_NAME, VLM_MODEL_PATH, demo_bin=VLM_DEMO_BIN)
    unloader = IdleUnloader(smart_vlm)
    unloader.start()

    print("[VLA] 启动指令队列...", flush=True)
    q = CommandQueue(smart_vlm=smart_vlm, snapshot_cb=snapshot)
    q.start()
    on_text = create_voice_motion_callback(q)

    try:
        if args.mode == "ask" and args.text:
            print(f"[文字] \"{args.text}\"", flush=True)
            on_text(args.text)
            import time
            time.sleep(30)
        elif args.mode == "text":
            run_text(q, on_text)
        else:
            run_voice(q, on_text)
    except KeyboardInterrupt:
        print("\n[退出]", flush=True)
    finally:
        unloader.stop()
        q.stop()
        smart_vlm.unload()


if __name__ == "__main__":
    main()
