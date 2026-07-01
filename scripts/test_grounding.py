import sys, os, cv2, json, yaml, re, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from astra import USBCamera
from config.settings import CAMERA_INDEX
from vla.vision.ipm import IPM
from voice_assistant.voice_assistant.qwen_runner import QwenRunner

OUT = "/home/elf/work/RK3588-EIA/scripts/test_grounding_output"
os.makedirs(OUT, exist_ok=True)

print("拍照...")
cam = USBCamera(CAMERA_INDEX)
cam.connect()
rgb, _ = cam.read()
cam.disconnect()

cv2.imwrite(f"{OUT}/01_raw.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

ipm = IPM()
ipm.set_workspace([[50, 100], [50, 380], [590, 380], [590, 100]], (448, 448))
bird = ipm.transform(rgb)
cv2.imwrite(f"{OUT}/03_bird_ipm.jpg", cv2.cvtColor(bird, cv2.COLOR_RGB2BGR))
print(f"  IPM: {OUT}/03_bird_ipm.jpg")

cfg_path = os.path.join(os.path.dirname(__file__), "..", "voice_assistant", "config", "default.yaml")
with open(cfg_path) as f:
    config = yaml.safe_load(f)

runner = QwenRunner(config)
prompt = "<image>输出所有物体的名称和边界框坐标，格式JSON: [{\"name\":\"物体名\",\"bbox_2d\":[x1,y1,x2,y2]}]"
result = runner.ask(f"{OUT}/03_bird_ipm.jpg", prompt)
print(f"\nQwen: {result}")

# 解析
obj_name = "物体"
cx, cy = None, None
try:
    data = json.loads(re.search(r'\[.*\]|\{.*\}', result, re.DOTALL).group())
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    obj_name = data.get("label", data.get("name", data.get("object", "物体")))
    box = data.get("bbox_2d") or data.get("position") or data.get("bbox") or []
    if len(box) >= 4:
        max_val = max(box)
        if max_val > 1:
            cx = int((box[0] + box[2]) / 2 * 448 / max_val)
            cy = int((box[1] + box[3]) / 2 * 448 / max_val)
        else:
            cx = int((box[0] + box[2]) / 2 * 448)
            cy = int((box[1] + box[3]) / 2 * 448)
except Exception:
    pass

if cx:
    wx, wy = cx * 300 / 448, (448 - cy) * 300 / 448
    msg = f"{obj_name}在({wx:.0f},{wy:.0f})毫米"
    print(f"\n{msg}")
else:
    msg = f"看到{obj_name}"
    print(f"\n{msg}")

if cx:
    wx, wy = cx * 300 / 448, (448 - cy) * 300 / 448
    print(f"\n{obj_name} @ ({wx:.0f},{wy:.0f})mm")
    subprocess.run(["espeak-ng", "-v", "zh", f"{obj_name}在{wx:.0f}乘{wy:.0f}毫米位置"], timeout=20)
else:
    print(f"\n{obj_name}（无坐标）")

