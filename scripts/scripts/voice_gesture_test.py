"""语音控制机械臂动作测试"""
import sys, os, json, yaml, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vla.control import ArmController
from config.settings import SERIAL_PORT, SERIAL_BAUD
from scripts.gestures import Gestures

# 加载语音助手配置
CFG = os.path.join(os.path.dirname(__file__), "..", "voice_assistant", "config", "default.yaml")
with open(CFG) as f:
    config = yaml.safe_load(f)

PROJ = os.path.join(os.path.dirname(__file__), "..")


def recognize(audio_path: str) -> str:
    """调用 voice_assistant 的 STT 识别"""
    result = subprocess.run(
        [sys.executable, "voice_assistant/voice_assistant.py", "stt", audio_path],
        capture_output=True, text=True, timeout=30,
        cwd=PROJ
    )
    return result.stdout.strip()


arm = ArmController(SERIAL_PORT, SERIAL_BAUD)
g = Gestures(arm)

print("语音控制测试:")
print("  说 '你好' → 打招呼")
print("  说 '挥手' → 挥手")
print("  说 '点头' → 点头")
print("  说 '归位' → 归位")
print("  q 退出")
print()

rec_cmd = PROJ + "/scripts/voice_rec_test.py"
if not os.path.exists(rec_cmd):
    # 直接用 arecord 录音
    rec_cmd = None

while True:
    cmd = input("按 Enter 录音（或输入文字指令）: ").strip().lower()
    if cmd == "q":
        break

    if not cmd:
        # 录音
        print("  🎤 录音3秒...")
        subprocess.run([
            "arecord", "-D", "hw:rockchipnau8822,0",
            "-d", "3", "-f", "cd", "-t", "wav", "/tmp/voice_cmd.wav"
        ], capture_output=True)
        print("  识别中...")
        cmd = recognize("/tmp/voice_cmd.wav")
        print(f"  识别: [{cmd}]")

    if "你好" in cmd or "打招呼" in cmd:
        print("  👋 打招呼")
        g.greeting()
    elif "挥手" in cmd:
        print("  👋 挥手")
        g.wave()
    elif "点头" in cmd:
        print("  👍 点头")
        g.nod()
    elif "归位" in cmd:
        print("  🏠 归位")
        g.home()
    else:
        print(f"  未识别的指令: {cmd}")

arm.close()
