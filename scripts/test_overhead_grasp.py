"""测试上帝视角俯拍 → Qwen grounding 坐标输出"""
import sys, os, cv2, json, yaml, re, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import CAMERA_OVERHEAD
from voice_assistant.voice_assistant.qwen_runner import QwenRunner

OUT = "/home/elf/work/RK3588-EIA/scripts/test_grounding_output"
os.makedirs(OUT, exist_ok=True)

print("拍照（上帝视角）...")
cap = cv2.VideoCapture(CAMERA_OVERHEAD, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.5)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("拍照失败")
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

h, w = rgb.shape[:2]

# 截取工作区中心区域
h, w = rgb.shape[:2]
crop = rgb[h//4:3*h//4, w//4:3*w//4]
bird = cv2.resize(crop, (448, 448))
cv2.imwrite(f"{OUT}/01_overhead.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{OUT}/02_bird.jpg", cv2.cvtColor(bird, cv2.COLOR_RGB2BGR))
print(f"  原始: {OUT}/01_overhead.jpg")
print(f"  裁剪: {OUT}/02_bird.jpg")

# 2. Qwen grounding
cfg_path = os.path.join(os.path.dirname(__file__), "..", "voice_assistant", "config", "default.yaml")
with open(cfg_path) as f:
    config = yaml.safe_load(f)

runner = QwenRunner(config)
prompt = "<image>输出画面中所有物体的名称和边界框坐标，JSON格式"
result = runner.ask(f"{OUT}/02_bird.jpg", prompt)
print(f"\nQwen:\n{result}")

# 3. 解析
try:
    data = json.loads(re.search(r'\[.*\]|{.*}', result, re.DOTALL).group())
    items = data if isinstance(data, list) else [data]
    for item in items:
        name = item.get("name", item.get("label", "物体"))
        box = item.get("bbox_2d") or item.get("position") or []
        if len(box) >= 4:
            maxv = max(box)
            scale = 448 / maxv if maxv > 1 else 1
            cx = int((box[0] + box[2]) / 2 * scale)
            cy = int((box[1] + box[3]) / 2 * scale)
            print(f"  {name}: 像素中心 ({cx}, {cy})")
except Exception as e:
    print(f"  解析失败: {e}")
