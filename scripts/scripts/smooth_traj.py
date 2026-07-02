"""轨迹平滑工具 — 中值滤波 + Savitzky–Golay + 重采样"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

JOINTS = ["J1", "J2", "J3", "J4", "J5", "J6"]
JOINT_ALIAS = {
    "J1": "shoulder_pan", "J2": "shoulder_lift", "J3": "elbow_flex",
    "J4": "wrist_flex", "J5": "wrist_roll", "J6": "gripper",
}


def get_val(frame, jname):
    """支持 J1 / shoulder_pan 两种命名"""
    v = frame.get(jname) or frame.get(JOINT_ALIAS.get(jname))
    return v if v is not None else 0.0


def median_filter(data, window=3):
    """中值滤波去毛刺"""
    half = window // 2
    result = data.copy()
    for i in range(half, len(data) - half):
        result[i] = np.median(data[i - half:i + half + 1])
    return result


def savgol_filter(data, window=5, order=2):
    """Savitzky–Golay 滤波，保留趋势的同时平滑"""
    if len(data) < window:
        return data
    result = data.copy()
    for i in range(len(data)):
        left = max(0, i - window // 2)
        right = min(len(data), i + window // 2 + 1)
        x = np.arange(right - left)
        y = data[left:right]
        if len(x) >= order + 1:
            coeffs = np.polyfit(x, y, order)
            result[i] = np.polyval(coeffs, i - left)
    return result


def resample(frames, target_fps=30):
    """重采样到固定帧率"""
    if len(frames) < 2:
        return frames
    t_start = frames[0]["t"]
    t_end = frames[-1]["t"]
    dt = 1.0 / target_fps
    new_times = np.arange(t_start, t_end, dt)

    result = []
    for t in new_times:
        frame = {"t": round(t, 3)}
        for joint in JOINTS:
            vals = [f[joint] for f in frames]
            old_t = [f["t"] for f in frames]
            frame[joint] = round(float(np.interp(t, old_t, vals)), 1)
        result.append(frame)
    return result


def smooth_file(input_path, output_path=None, median_win=3, sg_win=5, fps=30):
    with open(input_path) as f:
        data = json.load(f)

    frames = data["frames"]
    print("原始: {}帧, {}FPS".format(len(frames), data.get("fps", "?")))

    # 每列关节分别滤波
    arr = np.array([[get_val(f, j) for j in JOINTS] for f in frames])
    for col in range(6):
        arr[:, col] = median_filter(arr[:, col], median_win)
        arr[:, col] = savgol_filter(arr[:, col], sg_win)

    for i, f in enumerate(frames):
        for j, name in enumerate(JOINTS):
            f[name] = round(float(arr[i, j]), 1)

    # 也保存一份命名兼容格式
    for i, f in enumerate(frames):
        for j, name in enumerate(JOINTS):
            alias = JOINT_ALIAS[name]
            f[alias] = f[name]

    # 重采样
    frames = resample(frames, fps)

    data["frames"] = frames
    data["fps"] = fps

    out_path = output_path or input_path.replace(".json", "_smoothed.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("平滑后: {}帧, {}FPS, 已保存 {}".format(len(frames), fps, out_path))
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="轨迹平滑处理")
    parser.add_argument("input", help="原始轨迹 JSON")
    parser.add_argument("--output", default=None)
    parser.add_argument("--median", type=int, default=3, help="中值滤波窗口")
    parser.add_argument("--sg-window", type=int, default=5, help="SG滤波窗口")
    parser.add_argument("--fps", type=int, default=30, help="目标帧率")
    args = parser.parse_args()
    smooth_file(args.input, args.output, args.median, args.sg_window, args.fps)
