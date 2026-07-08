#!/usr/bin/env python3
"""官方 VAD + SenseVoice 示例（基于 sherpa-onnx 官方代码）"""
import numpy as np
import sounddevice as sd
import sherpa_onnx

VAD = "/home/elf/work/RK3588-EIA/voice_assistant/models/silero_vad/silero_vad.onnx"
MODEL = "/home/elf/work/RK3588-EIA/voice_assistant/models/sense-voice/model.int8.onnx"
TOKENS = "/home/elf/work/RK3588-EIA/voice_assistant/models/sense-voice/tokens.txt"


def main():
    print("检查麦克风...")
    devices = sd.query_devices()
    print(devices)
    default = sd.default.device[0]
    print(f"使用设备: {devices[default]['name']}")

    print("加载 SenseVoice...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=MODEL, tokens=TOKENS, num_threads=2, use_itn=True,
    )

    print("加载 VAD...")
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = VAD
    cfg.silero_vad.min_silence_duration = 0.3
    cfg.sample_rate = 16000
    window = cfg.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

    print("\n说话吧！按 Ctrl+C 退出\n")

    GAIN = 8.0
    tick = 0
    buf = np.array([], dtype=np.float32)
    with sd.InputStream(channels=1, dtype="float32", samplerate=16000) as s:
        while True:
            samples, _ = s.read(int(0.1 * 16000))
            samples = samples.reshape(-1) * GAIN
            level = np.abs(samples).mean()
            tick += 1
            if tick % 25 == 0:
                bar = "█" * int(level * 500)
                print(f"\r  音量: {level:.5f} {bar}", end="", flush=True)

            buf = np.concatenate([buf, samples])
            while len(buf) > window:
                vad.accept_waveform(buf[:window])
                buf = buf[window:]

            while not vad.empty():
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, vad.front.samples)
                vad.pop()
                recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                if text:
                    print(f"\n  → {text}\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出")
