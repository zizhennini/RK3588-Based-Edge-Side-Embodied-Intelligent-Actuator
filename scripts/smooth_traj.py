"""轨迹平滑工具 — 中值滤波 + 速度限幅 + 重采样"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

JOINTS = ["J1", "J2", "J3", "J4", "J5", "J6"]
JOINT_ALIAS = {
    "J1": "shoulder_pan", "J2": "shoulder_lift", "J3": "elbow_flex",
    "J4": "wrist_flex", "J5": "wrist_roll", "J6": "gripper",
}
# 每帧最大变化（度/帧 @30fps），超过此值的帧视为异常
MAX_DELTA = {"J1": 5, "J2": 5, "J3": 5, "J4": 8, "J5": 8, "J6": 10}


def get_val(frame, jname):
    v = frame.get(jname) or frame.get(JOINT_ALIAS.get(jname))
    return v if v is not None else 0.0


def clamp_velocity(data, max_delta):
    """速度限幅：每帧变化不超过 max_delta"""
    result = data.copy()
    for i in range(1, len(data)):
        diff = result[i] - result[i - 1]
        if abs(diff) > max_delta:
            result[i] = result[i - 1] + (max_delta if diff > 0 else -max_delta)
    return result


def smooth_file(input_path, output_path=None, median_win=3, fps=30):
    with open(input_path) as f:
        data = json.load(f)

    frames = data["frames"]
    print("原始: {}帧, {}FPS".format(len(frames), data.get("fps", "?")))

    # 读出所有数据
    arr = np.array([[get_val(f, j) for j in JOINTS] for f in frames])

    for col, name in enumerate(JOINTS):
        vals = arr[:, col].copy()

        if name == "J6":
            # gripper 是 0-100，只去毛刺（中值）和钳位
            half = median_win // 2
            for i in range(half, len(vals) - half):
                vals[i] = np.median(vals[i - half:i + half + 1])
            vals = np.clip(np.round(vals), 0, 100)
        else:
            # 中值滤波（去毛刺）
            half = median_win // 2
            for i in range(half, len(vals) - half):
                vals[i] = np.median(vals[i - half:i + half + 1])
            # 速度限幅
            max_d = MAX_DELTA.get(name, 5)
            for i in range(1, len(vals)):
                diff = vals[i] - vals[i - 1]
                if abs(diff) > max_d:
                    vals[i] = vals[i - 1] + (max_d if diff > 0 else -max_d)

        arr[:, col] = vals

    # 写回
    for i, f in enumerate(frames):
        for j, name in enumerate(JOINTS):
            f[name] = round(float(arr[i, j]), 1)
        for j, name in enumerate(JOINTS):
            alias = JOINT_ALIAS[name]
            f[alias] = f[name]

    # 重采样到固定帧率
    if len(frames) >= 2:
        t_start, t_end = frames[0]["t"], frames[-1]["t"]
        dt = 1.0 / fps
        n_steps = max(2, int(round((t_end - t_start) / dt)) + 1)
        new_times = np.linspace(t_start, t_end, n_steps)
        # 预计算原数据
        old_t = [f["t"] for f in frames]
        old_by_joint = {name: [f[name] for f in frames] for name in JOINTS}
        new_frames = []
        for t in new_times:
            nf = {"t": round(t, 3)}
            for j, name in enumerate(JOINTS):
                val = np.interp(t, old_t, old_by_joint[name])
                nf[name] = round(float(val), 1)
                nf[JOINT_ALIAS[name]] = nf[name]
            new_frames.append(nf)
        frames = new_frames

    data["frames"] = frames
    data["fps"] = fps
    out_path = output_path or input_path.replace(".json", "_smoothed.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("平滑后: {}帧, {}FPS, 已保存 {}".format(len(frames), fps, out_path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="轨迹平滑处理")
    parser.add_argument("input")
    parser.add_argument("--output", default=None)
    parser.add_argument("--median", type=int, default=3, help="中值滤波窗口")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    smooth_file(args.input, args.output, args.median, args.fps)
