"""TTS 播报 — 给 test_grounding 调用"""
import sys, os, numpy as np, subprocess, yaml

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

with open(os.path.join(PROJ, "voice_assistant", "config", "default.yaml")) as f:
    config = yaml.safe_load(f)

from voice_assistant.voice_assistant.tts import SherpaTts
tts = SherpaTts(config)
sr, samples = tts.synthesize_samples(sys.argv[1])
(samples * 32767).astype(np.int16).tofile("/tmp/tts_out.pcm")
subprocess.run(["aplay", "-D", "hw:rockchipnau8822,0", "-r", str(sr), "-f", "S16_LE", "-c", "1", "/tmp/tts_out.pcm"])
