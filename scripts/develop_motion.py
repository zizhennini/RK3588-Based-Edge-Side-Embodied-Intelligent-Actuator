"""动作轨迹开发流程：录制 → 处理 → 验证 → 入库"""
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(PROJ, "scripts")
LIB_DIR = os.path.join(PROJ, "motion_library")


def step(msg):
    print("\n" + "=" * 50)
    print("  {}".format(msg))
    print("=" * 50)


def run(cmd, capture=False):
    print("  $ {}".format(" ".join(cmd)))
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        for line in result.stdout.strip().split("\n"):
            if line.strip(): print("    {}".format(line.strip()))
        if result.returncode != 0:
            for line in result.stderr.strip().split("\n")[-3:]:
                print("    {}".format(line.strip()))
    else:
        result = subprocess.run(cmd)
    return result


def record_one(name, category, record_seconds, fps):
    """录制单个动作：录制→平滑→回放→入库"""
    raw_file = os.path.join(PROJ, "{}_raw.json".format(name))
    sub_dir = category if category else ""
    out_dir = os.path.join(LIB_DIR, sub_dir) if sub_dir else LIB_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "{}_{:02d}.json".format(name.replace("_", ""), 1))

    step("录制: {}".format(name))
    print("  录制中...(按 Ctrl+C 停止)")
    try:
        run([
            sys.executable, os.path.join(SCRIPT_DIR, "teleop_record.py"),
            "--follower_port", "/dev/ttyACM0",
            "--leader_port", "/dev/ttyACM1",
            "--episode_time_s", str(record_seconds),
            "--out", raw_file,
        ], capture=False)
    except KeyboardInterrupt:
        print("录制结束，继续处理...")

    if not os.path.exists(raw_file):
        print("  录制已取消")
        return

    # 去重
    with open(raw_file) as f:
        raw_data = json.load(f)
    frames = raw_data["frames"]
    keys = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    cut = len(frames) - 1
    while cut > 0:
        f1, f2 = frames[cut - 1], frames[cut]
        diff = sum(abs(f1.get(k, 0) - f2.get(k, 0)) for k in keys)
        if diff > 0.1: break
        cut -= 1
    if cut < len(frames) - 1:
        raw_data["frames"] = frames[:cut + 1]
        with open(raw_file, "w") as f:
            json.dump(raw_data, f, indent=2)
        print("  去重: 移除{}帧尾部重复".format(len(frames) - cut - 1))

    run([sys.executable, os.path.join(SCRIPT_DIR, "smooth_traj.py"),
         raw_file, "--output", out_file, "--fps", str(fps)], capture=True)

    # 回放验证
    step("回放验证: {}".format(name))
    print("  即将回放，按 Ctrl+C 跳过")
    try:
        run([sys.executable, os.path.join(SCRIPT_DIR, "replay_traj.py"),
             out_file, "--fps", str(fps), "--port", "/dev/ttyACM0"], capture=True)
    except KeyboardInterrupt:
        print("  跳过回放")

    # 入库
    idx_path = os.path.join(LIB_DIR, "index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    rel_path = os.path.relpath(out_file, LIB_DIR)
    idx[name] = {"file": rel_path, "keywords": [], "duration_s": round(raw_data.get("duration_s", record_seconds), 1), "category": category or ""}
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    print("  已入库: {}".format(name))
    os.remove(raw_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="动作轨迹开发流程")
    parser.add_argument("name", nargs="?", help="动作名称，如 reach_center")
    parser.add_argument("--category", default="", help="分类目录, 如 reach/grasp/lift/place/retract")
    parser.add_argument("--record_seconds", type=int, default=120, help="录制最大时长")
    parser.add_argument("--fps", type=int, default=30, help="目标帧率")
    parser.add_argument("--no_record", action="store_true", help="跳过录制")
    parser.add_argument("--batch", nargs="+", metavar="NAME", help="连续录制多个动作: --batch reach_center reach_left reach_right --category reach")
    args = parser.parse_args()

    if args.batch:
        for name in args.batch:
            record_one(name, args.category, args.record_seconds, args.fps)
        return

    if args.name:
        record_one(args.name, args.category, args.record_seconds if not args.no_record else 0, args.fps)


if __name__ == "__main__":
    main()
