"""集成测试：D435i + Qwen3-VL grounding + 3D坐标"""
import sys, os, json, yaml, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera import D435iCamera
from voice_assistant.voice_assistant.qwen_runner import QwenRunner

# 1. 初始化 D435i
print("初始化 D435i...")
cam = D435iCamera()
cam.connect()

# 2. 拍照
print("拍照...")
rgb, depth = cam.read()
snap_path = "/tmp/d435i_snap.jpg"
cv2.imwrite(snap_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print("  已保存 {}".format(snap_path))

# 3. Qwen3-VL grounding
print("Qwen grounding...")
cfg_path = os.path.join(os.path.dirname(__file__), "..", "voice_assistant", "config", "default.yaml")
with open(cfg_path) as f:
    config = yaml.safe_load(f)

runner = QwenRunner(config)
prompt = "<image>用JSON格式输出所有物体的名称和边界框坐标"
result = runner.ask(snap_path, prompt)
print("  Qwen输出:", result[:200])

# 4. 解析坐标
try:
    import re
    data = json.loads(re.search(r'\[.*\]|{.*}', result, re.DOTALL).group())
    items = data if isinstance(data, list) else [data]
    for item in items:
        name = item.get("label", item.get("name", item.get("object", "物体")))
        box = item.get("bbox_2d") or item.get("position") or []
        if len(box) >= 4:
            maxv = max(box)
            scale = 640 / maxv if maxv > 1 else 1
            cx = int((box[0] + box[2]) / 2 * scale)
            cy = int((box[1] + box[3]) / 2 * scale)
            # 获取深度
            z = float(depth[cy, cx]) if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1] else 0
            if z > 0:
                pt = cam.deproject(cx, cy, z)
                print("  {} @ ({},{}) → 深度{:.3f}m → 3D({:.3f},{:.3f},{:.3f})m".format(name, cx, cy, z, pt[0], pt[1], pt[2]))
            else:
                print("  {} @ ({},{}) 无效深度".format(name, cx, cy))
        else:
            print("  {}: 无坐标".format(name))
except Exception as e:
    print("  解析失败: {}".format(e))

cam.disconnect()
print("集成测试完成")
