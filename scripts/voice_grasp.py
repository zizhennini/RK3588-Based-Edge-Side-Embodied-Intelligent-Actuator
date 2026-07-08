#!/usr/bin/env python3
"""语音抓取入口：KWS 唤醒 → ASR → VLM 识别 → 视觉定位 → IK → 机械臂抓取 → TTS 反馈"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.cpu_affinity import bind_current_thread, BIG_CORES, affinity_summary
from config.settings import (
    CAMERA_MATRIX, SERIAL_PORT, SERIAL_BAUD,
    VLM_MODEL_NAME, VLM_MODEL_PATH, VLM_DEMO_BIN,
    VLM_IDLE_UNLOAD_TIMEOUT,
)
from vla.command_queue import CommandQueue, create_voice_motion_callback
from vla.vlm.smart_vlm import SmartVLM, IdleUnloader
from config.memory import MemoryMonitor


def main():
    import argparse
    parser = argparse.ArgumentParser(description="语音/文字抓取系统")
    parser.add_argument("text", nargs="?", default=None,
                        help="直接传入文字指令（跳过语音唤醒）")
    args = parser.parse_args()

    bind_current_thread(BIG_CORES)
    print(affinity_summary())
    mem = MemoryMonitor()
    print(f"[内存] {mem.summary()}")

    # ── 初始化硬件 ──
    arm = None
    cam = None
    smart_vlm = None
    unloader = None

    try:
        # VLM
        print("[VLA] 创建 SmartVLM...")
        smart_vlm = SmartVLM(
            VLM_MODEL_NAME, VLM_MODEL_PATH,
            demo_bin=VLM_DEMO_BIN,
            idle_timeout=VLM_IDLE_UNLOAD_TIMEOUT,
        )
        unloader = IdleUnloader(smart_vlm, interval=5.0)
        unloader.start()

        # 相机
        print("[VLA] 打开 D435i 相机...")
        try:
            from camera import D435iCamera
            cam = D435iCamera()
            cam.connect()
        except Exception as e:
            print(f"[VLA] 相机打开失败（降级为无相机模式）: {e}")
            cam = None

        # 机械臂
        print("[VLA] 连接 SO-ARM101...")
        try:
            from vla.control import ArmController
            arm = ArmController(SERIAL_PORT, SERIAL_BAUD)
        except Exception as e:
            print(f"[VLA] 机械臂连接失败: {e}")
            arm = None

        # 指令队列
        print("[VLA] 启动指令队列...")
        q = CommandQueue(
            arm=arm,
            smart_vlm=smart_vlm,
            camera=cam,
            camera_matrix=CAMERA_MATRIX,
        )
        q.start()
        on_text = create_voice_motion_callback(q)

        # ── 语音助手 ──
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

        print("\n" + "=" * 50)
        print("  语音/文字抓取系统已就绪")
        print('  语音模式：说"你好同学"唤醒')
        print("  文字模式：python voice_grasp.py \"抓住红色色块\"")
        print("  指令示例：")
        print('    - "抓住那个红色杯子"')
        print('    - "拿起蓝色方块"')
        print('    - "抓取"(VLM 自动识别目标)')
        print('    - "归零" / "停止"')
        print("=" * 50 + "\n")

        if args.text:
            text = args.text.strip()
            print(f"[文字] \"{text}\"", flush=True)
            on_text(text)
            waited = 0
            while q.is_busy and waited < 30:
                import time as _t
                _t.sleep(0.5)
                waited += 0.5
            print("[完成]")
            return

        while True:
            print("[等待唤醒词 '你好同学' ...]", flush=True)
            keyword = wake.wait(timeout=None)
            print(f"[唤醒] {keyword}", flush=True)

            wav = recorder.record_wav(os.path.join(temp_dir, "cmd.wav"), 4)
            text = asr.transcribe_wav(wav)
            os.unlink(wav)

            if text and text.strip():
                print(f"[指令] \"{text}\"", flush=True)
                on_text(text.strip())
                waited = 0
                while q.is_busy and waited < 30:
                    import time as _t
                    _t.sleep(0.5)
                    waited += 0.5
            else:
                print("[未听清]", flush=True)

    except KeyboardInterrupt:
        print("\n[退出] 用户中断")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        if unloader:
            unloader.stop()
        if cam:
            try:
                cam.disconnect()
            except Exception:
                pass
        if arm and hasattr(arm, "close"):
            arm.close()
        if smart_vlm:
            smart_vlm.unload()
        print("完成")


if __name__ == "__main__":
    main()
