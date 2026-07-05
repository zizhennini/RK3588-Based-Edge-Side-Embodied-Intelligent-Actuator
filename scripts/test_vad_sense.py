#!/usr/bin/env python3
"""VAD + SenseVoice 端到端测试 — 说话即识别，无唤醒词"""
import sys, os, time, wave, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sherpa_onnx

VAD_MODEL = "/home/elf/work/RK3588-EIA/voice_assistant/models/silero_vad/silero_vad.onnx"
SV_MODEL = "/home/elf/work/RK3588-EIA/voice_assistant/models/sense-voice/model.int8.onnx"
SV_TOKENS = "/home/elf/work/RK3588-EIA/voice_assistant/models/sense-voice/tokens.txt"
MIC_DEVICE = "hw:rockchipnau8822,0"
SAMPLE_RATE = 16000
CHANNELS = 1


def main():
    print("=== 初始化 VAD ===")
    vad_cfg = sherpa_onnx.VadModelConfig()
    vad_cfg.silero_vad.model = VAD_MODEL
    vad_cfg.silero_vad.threshold = 0.5
    vad_cfg.silero_vad.min_silence_duration = 0.8
    vad_cfg.silero_vad.min_speech_duration = 0.3
    vad_cfg.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=30)
    print("  VAD 加载成功")

    print("=== 初始化 SenseVoice ===")
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=SV_MODEL, tokens=SV_TOKENS,
        num_threads=2, use_itn=True, language="zh",
    )
    print("  SenseVoice 加载成功")

    print("\n" + "=" * 40)
    print("VAD 侦听中 — 直接说话即可")
    print("  Ctrl+C 退出")
    print("=" * 40)

    import subprocess
    CHUNK = 3200  # 0.2s @ 16kHz

    proc = subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE",
         "-r", str(SAMPLE_RATE), "-c", str(CHANNELS), "-t", "raw"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None

    try:
        tick = 0
        while True:
            data = proc.stdout.read(CHUNK * 2)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767
            vad.accept_waveform(samples)

            tick += 1
            if tick % 25 == 0:
                print(".", end="", flush=True)

            if vad.has_result():
                buf = vad.pop_front_speech_buffer()
                print(f"\n  语音段: {len(buf)} samples", flush=True)
                stream = recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, buf)
                recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                if text:
                    print(f"  → {text}", flush=True)
                print("  [继续侦听]", flush=True)
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
