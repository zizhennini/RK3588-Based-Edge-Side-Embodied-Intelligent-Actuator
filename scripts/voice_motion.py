#!/usr/bin/env python3
"""语音指令入口 — ASR → CommandQueue → 串行执行"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.cpu_affinity import bind_current_thread, BIG_CORES
from vla.command_queue import CommandQueue, create_voice_motion_callback
from voice_assistant.voice_assistant.orchestrator import VoiceAssistant
from voice_assistant.voice_assistant.config import load_config

MOTION_DIR = os.path.join(os.path.dirname(__file__), "..", "motion_library")


def main():
    bind_current_thread(BIG_CORES)

    queue = CommandQueue(motion_dir=MOTION_DIR)
    queue.start()

    config = load_config()
    assistant = VoiceAssistant(config)

    on_text = create_voice_motion_callback(queue)

    print("=== 语音指令系统 ===")
    print("说关键词触发动作，或直接对话 Qwen")
    print("说 '停止' 中断当前任务")
    print("可用动作:", queue.motion_matcher.list_actions())
    print("按 Ctrl+C 退出\n")

    try:
        while True:
            keyword = assistant.wait_for_wake(mode="kws", timeout=None)
            print(f"[唤醒] {keyword}")

            command_wav = assistant.record_command()
            text = assistant.transcribe_wav(command_wav)
            command_wav.unlink(missing_ok=True)

            if not text or not text.strip():
                print("[空指令]")
                continue

            print(f"[指令] {text}")
            on_text(text)

            count = queue.pending_count
            if count > 0:
                print(f"[队列] 等待 {count} 个任务...")

    except KeyboardInterrupt:
        print("\n退出")
    finally:
        queue.stop()


if __name__ == "__main__":
    main()
