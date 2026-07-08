"""自适应抓取：示教基准位 → Qwen 识别偏移 → 轨迹修正 → 执行"""
import sys, os, json, yaml, time, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera import D435iCamera
from vla.control import ArmController
from config.settings import SERIAL_PORT, SERIAL_BAUD

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(PROJ, "voice_assistant", "config", "default.yaml")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def tts(text):
    subprocess.run(["pkill", "-f", "Qwen3.5-0.8B/demo"], capture_output=True)
    time.sleep(0.5)
    subprocess.run([sys.executable, os.path.join(PROJ, "scripts", "say.py"), text[:100]], timeout=60)


def load_task(task_file):
    with open(task_file) as f:
        return json.load(f)


def detect_object():
    """Qwen3.5 检测物体 → 返回像素坐标 (cx, cy) 或 None"""
    cam = D435iCamera()
    cam.connect()
    rgb, depth = cam.read()
    snap = "/tmp/grasp_snap.jpg"
    cv2.imwrite(snap, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    with open(CFG) as f:
        config = yaml.safe_load(f)
    from voice_assistant.voice_assistant.qwen_runner import QwenRunner
    runner = QwenRunner(config)

    import re, json as j
    result = runner.ask(snap, "<image>用JSON格式输出物体的名称和边界框坐标")
    try:
        data = j.loads(re.search(r"\[.*\]|{.*}", result, re.DOTALL).group())
        item = data[0] if isinstance(data, list) else data
        box = item.get("bbox_2d") or item.get("position") or []
        if len(box) >= 4:
            maxv = max(box)
            s = 640 / maxv if maxv > 1 else 1
            cx = int((box[0] + box[2]) / 2 * s)
            cy = int((box[1] + box[3]) / 2 * s)
            z = float(depth[cy, cx]) if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1] else 0
            cam.disconnect()
            if z > 0:
                return cx, cy, z
    except:
        pass
    cam.disconnect()
    return None


def main():
    import argparse, cv2
    parser = argparse.ArgumentParser(description="自适应抓取")
    parser.add_argument("task_file", help="示教任务 JSON")
    parser.add_argument("--ref_cx", type=float, required=True, help="示教时物体像素x")
    parser.add_argument("--ref_cy", type=float, required=True, help="示教时物体像素y")
    parser.add_argument("--ref_z", type=float, required=True, help="示教时物体深度m")
    parser.add_argument("--port", default=SERIAL_PORT)
    args = parser.parse_args()

    task = load_task(args.task_file)
    print("加载任务: {} 段".format(task["total"]))

    # 检测物体当前位置
    tts("正在识别物体位置")
    det = detect_object()
    if det is None:
        tts("未检测到物体")
        return
    cx, cy, z = det
    print("  检测到物体 @ ({}, {}) 深度 {:.3f}m".format(cx, cy, z))

    # 计算偏移量
    cam = D435iCamera()
    cam.connect()
    _, _ = cam.read()
    intr = cam._intr
    import pyrealsense2 as rs
    x0, y0, z0 = rs.rs2_deproject_pixel_to_point(intr, [args.ref_cx, args.ref_cy], args.ref_z)
    x1, y1, z1 = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], z)
    cam.disconnect()
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    print("  偏移: ({:.3f}, {:.3f}, {:.3f})m".format(dx, dy, dz))

    if abs(dx) + abs(dy) + abs(dz) < 0.01:
        tts("物体位置无偏移")
        return

    # 执行轨迹（叠加偏移）
    arm = ArmController(args.port, SERIAL_BAUD)
    for seg in task["segments"]:
        name = seg["name"]
        joints = seg["joints"]
        print("  执行: {}".format(name))
        for i, jn in enumerate(JOINTS):
            deg = joints[jn]
            if jn in ("shoulder_pan", "shoulder_lift", "elbow_flex"):
                deg += np.rad2deg(np.arctan2(dx, dz)) * 0.3  # 简化偏移映射
            arm._write_angle(i + 1, np.deg2rad(deg))
        time.sleep(1.5)

    arm.close()
    tts("抓取完成")


if __name__ == "__main__":
    main()
