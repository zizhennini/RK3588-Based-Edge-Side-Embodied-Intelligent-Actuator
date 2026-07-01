"""回放快速测试"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vla.control import ArmController

d = json.load(open("teleop_record_20260701_110854.json"))
arm = ArmController(port="/dev/ttyACM0")
frames = d["frames"]

print("回放前5帧测试...")
for i in range(min(5, len(frames))):
    deg = frames[i].get("shoulder_pan", 0.0)
    arm._write_angle(1, float(np.deg2rad(deg)))
    print("帧{}: J1={}度".format(i, deg))
    time.sleep(0.5)

print("回放后5帧...")
for i in range(max(0, len(frames)-5), len(frames)):
    deg = frames[i].get("shoulder_pan", 0.0)
    arm._write_angle(1, float(np.deg2rad(deg)))
    print("帧{}: J1={}度".format(i, deg))
    time.sleep(0.5)

arm.close()
print("完成")
