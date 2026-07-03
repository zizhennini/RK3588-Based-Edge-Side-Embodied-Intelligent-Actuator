"""语音+视觉+深度 全链路测试（sherpa-onnx TTS）"""
import sys, os, json, yaml, cv2, numpy as np, re, subprocess, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from astra import D435iCamera
from voice_assistant.voice_assistant.qwen_runner import QwenRunner

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(PROJ, "voice_assistant", "config", "default.yaml")


def tts(text):
    say = os.path.join(PROJ, "scripts", "say.py")
    r = subprocess.run([sys.executable, say, text], capture_output=True, timeout=60)
    if r.returncode != 0:
        subprocess.run(["espeak-ng", "-v", "zh", text[:80]], timeout=10)


print("初始化 D435i...")
cam = D435iCamera()
cam.connect()

while True:
    cmd = input("\n按 Enter 录音，或输入文字指令，q 退出: ").strip()
    if cmd == "q":
        break

    if not cmd:
        print("  录音中...")
        subprocess.run([
            "arecord", "-D", "hw:rockchipnau8822,0",
            "-d", "3", "-f", "cd", "-t", "wav", "/tmp/voice_cmd.wav"
        ], capture_output=True)
        result = subprocess.run([
            sys.executable, os.path.join(PROJ, "voice_assistant", "voice_assistant.py"),
            "stt", "/tmp/voice_cmd.wav"
        ], capture_output=True, text=True)
        cmd = result.stdout.strip()
        t = "识别到: " + cmd
        print("  " + t)

    if not cmd:
        continue

    print("  拍照...")
    rgb, depth = cam.read()
    snap = "/tmp/snap_vla.jpg"
    cv2.imwrite(snap, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print("  Qwen3-VL 分析...")
    with open(CFG) as f:
        config = yaml.safe_load(f)
    runner = QwenRunner(config)
    prompt = "<image>用JSON格式输出画面中物体的名称和边界框坐标"
    result = runner.ask(snap, prompt) if "坐标" in cmd else runner.ask(snap, "<image>" + cmd)
    answer = result.strip()
    print("  回答: {}".format(answer[:300]))
    tts(answer)

    # 尝试解析坐标并补充深度
    try:
        data = json.loads(re.search(r'\[.*\]|{.*}', answer, re.DOTALL).group())
        items = data if isinstance(data, list) else [data]
        for item in items:
            name = item.get("label", item.get("object", "物体"))
            box = item.get("bbox_2d") or item.get("position") or []
            if len(box) >= 4:
                maxv = max(box)
                scale = 640 / maxv if maxv > 1 else 1
                cx = int((box[0] + box[2]) / 2 * scale)
                cy = int((box[1] + box[3]) / 2 * scale)
                if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    z = float(depth[cy, cx])
                    if z > 0:
                        pt = cam.deproject(cx, cy, z)
                        txt = "{}位置在({:.0f},{:.0f})毫米".format(name, pt[0]*1000, pt[1]*1000)
                        print("  " + txt)
                        tts(txt)
    except Exception as e:
        print("  坐标解析: {}".format(e))

cam.disconnect()
print("完成")
