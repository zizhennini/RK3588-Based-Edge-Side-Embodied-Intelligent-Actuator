"""TTS 播报 — 先杀Qwen释放内存，再播报"""
import sys, os, subprocess, time

# 1. 杀 Qwen 进程释放内存
subprocess.run(["pkill", "-f", "Qwen3.5-0.8B/demo"], capture_output=True)
time.sleep(0.5)

text = sys.argv[1][:100]

# 2. 尝试 sherpa-onnx TTS
try:
    import numpy as np, yaml
    PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJ)
    with open(os.path.join(PROJ, "voice_assistant", "config", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    from voice_assistant.voice_assistant.tts import SherpaTts
    tts = SherpaTts(cfg)
    sr, samples = tts.synthesize_samples(text)
    (samples * 32767).astype(np.int16).tofile("/tmp/tts_out.pcm")
    subprocess.run(["aplay", "-D", "hw:rockchipnau8822,0", "-r", str(sr), "-f", "S16_LE", "-c", "1", "/tmp/tts_out.pcm"])
except Exception:
    # 3. 降级 espeak
    subprocess.run(["espeak-ng", "-v", "zh", text], timeout=10)
