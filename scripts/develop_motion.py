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


def run(cmd):
    print("  $ {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print("    {}".format(line.strip()))
    if result.returncode != 0:
        for line in result.stderr.strip().split("\n")[-3:]:
            print("    {}".format(line.strip()))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="动作轨迹开发流程")
    parser.add_argument("name", help="动作名称，如 greeting / grasp")
    parser.add_argument("--record_seconds", type=int, default=15, help="录制时长")
    parser.add_argument("--fps", type=int, default=30, help="目标帧率")
    parser.add_argument("--no_record", action="store_true", help="跳过录制，直接用已有raw文件")
    args = parser.parse_args()

    raw_file = os.path.join(PROJ, "{}_raw.json".format(args.name))
    out_file = os.path.join(LIB_DIR, "{}_{:02d}.json".format(args.name, 1))

    # 第1步：数据录制
    if not args.no_record:
        step("第1步：数据录制")
        print("  请在遥操作终端运行 lerobot-teleoperate...")
        r = run([
            sys.executable, os.path.join(SCRIPT_DIR, "teleop_record.py"),
            "--follower_port", "/dev/ttyACM0",
            "--leader_port", "/dev/ttyACM1",
            "--episode_time_s", str(args.record_seconds),
            "--out", raw_file,
        ])
        if r.returncode != 0:
            print("  录制失败")
            return

    # 第2步：算法处理（去重+截取+平滑）
    step("第2步：算法处理")
    # 去重：去掉尾部重复帧
    with open(raw_file) as f:
        raw_data = json.load(f)
    frames = raw_data["frames"]
    keys = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    cut = len(frames) - 1
    while cut > 0:
        f1, f2 = frames[cut - 1], frames[cut]
        diff = sum(abs(f1.get(k, 0) - f2.get(k, 0)) for k in keys)
        if diff > 0.1:
            break
        cut -= 1
    if cut < len(frames) - 1:
        raw_data["frames"] = frames[:cut + 1]
        with open(raw_file, "w") as f:
            json.dump(raw_data, f, indent=2)
        print("  去重: 移除{}帧尾部重复".format(len(frames) - cut - 1))

    run([sys.executable, os.path.join(SCRIPT_DIR, "smooth_traj.py"),
         raw_file, "--output", out_file, "--fps", str(args.fps)])

    # 第3步：回放验证
    step("第3步：回放验证")
    print("  即将回放，按 Ctrl+C 跳过验证")
    try:
        run([sys.executable, os.path.join(SCRIPT_DIR, "replay_traj.py"),
             out_file, "--fps", str(args.fps), "--port", "/dev/ttyACM0"])
    except KeyboardInterrupt:
        print("  跳过回放")

    # 第4步：更新动作库索引
    step("第4步：更新动作库")
    idx_path = os.path.join(LIB_DIR, "index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    if args.name not in idx:
        print("  动作名 {} 不在 index.json 中，请手动添加".format(args.name))
    else:
        idx[args.name]["file"] = os.path.basename(out_file)
        idx[args.name]["duration_s"] = args.record_seconds
        with open(idx_path, "w") as f:
            json.dump(idx, f, indent=2, ensure_ascii=False)
        print("  已更新 {}".format(args.name))

    print("\n完成！轨迹文件：{}".format(out_file))


if __name__ == "__main__":
    main()
