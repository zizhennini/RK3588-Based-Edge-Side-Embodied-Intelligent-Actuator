"""检查录制数据"""
import json
d = json.load(open("greeting_raw.json"))
print("keys:", list(d["frames"][0].keys()))
print("第1帧:", d["frames"][0])
g = [f.get("gripper", 0) for f in d["frames"]]
print("gripper范围: {:.1f}~{:.1f}".format(min(g), max(g)))
