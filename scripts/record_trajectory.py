#!/usr/bin/env python3
"""轨迹录制 — 遥操作时记录双臂关节角度到动作库"""

import sys, os, json, time, numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MOTION_DIR = Path(__file__).resolve().parent.parent / "motion_library"
INDEX_FILE = MOTION_DIR / "index.json"
SAMPLING_HZ = 20  # 采样频率
DT = 1.0 / SAMPLING_HZ


def load_index() -> dict:
    if not INDEX_FILE.exists():
        return {}
    with open(INDEX_FILE) as f:
        return json.load(f)


def save_index(index: dict):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def list_actions():
    index = load_index()
    if not index:
        print("动作库为空")
        return
    print(f"{'名称':<12} {'关键词':<24} {'时长':<6} {'文件':<20}")
    print("-" * 62)
    for name, info in index.items():
        if not isinstance(info, dict):
            continue
        kw = ", ".join(info.get("keywords", []))
        dur = info.get("duration_s", 0)
        fn = info.get("file", "")
        print(f"{name:<12} {kw:<24} {dur:<6} {fn:<20}")


def delete_action(name: str):
    index = load_index()
    if name not in index:
        print(f"动作 '{name}' 不存在")
        return
    info = index.pop(name)
    fpath = MOTION_DIR / info["file"]
    if fpath.exists():
        fpath.unlink()
    save_index(index)
    print(f"已删除动作 '{name}'")


def record(name: str, keywords: list[str], arm):
    """录制轨迹：传入已连接的 ArmController，手动操控完成后 Ctrl+C 停止"""
    index = load_index()
    if name in index:
        print(f"动作 '{name}' 已存在，删除旧记录")
        delete_action(name)

    print(f"\n开始录制轨迹 '{name}'")
    print("请通过遥操作控制机械臂完成动作")
    print("按 Ctrl+C 停止录制")
    print()

    traj = []
    start = time.time()
    try:
        while True:
            t = time.time() - start
            if arm and hasattr(arm, "_read_current_pos"):
                joints = arm._read_current_pos()
                if isinstance(joints, np.ndarray):
                    traj.append({"t": round(t, 3), "joints": [round(float(j), 4) for j in joints]})
                else:
                    traj.append({"t": round(t, 3)})
            else:
                traj.append({"t": round(t, 3)})
            time.sleep(DT)
    except KeyboardInterrupt:
        pass

    duration = time.time() - start
    print(f"\n录制完成，时长 {duration:.1f}s，共 {len(traj)} 帧")

    if not traj or all("joints" not in f for f in traj):
        print("  ⚠ 注意：未读取到关节数据（arm 未传入或 _read_current_pos 未实现），仅保存了时间戳")

    # 保存轨迹文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = MOTION_DIR / f"{name}_{timestamp}.json"
    traj_data = {
        "meta": {
            "name": name,
            "recorded_at": datetime.now().isoformat(),
            "duration_s": round(duration, 1),
            "frames": len(traj),
            "sample_rate_hz": SAMPLING_HZ,
        },
        "frames": traj,
    }
    with open(filepath, "w") as f:
        json.dump(traj_data, f, indent=2)

    # 更新索引
    index[name] = {
        "file": filepath.name,
        "keywords": keywords,
        "duration_s": round(duration, 1),
    }
    save_index(index)
    print(f"轨迹已保存: {filepath.name}")
    print(f"关键词: {keywords}")


def inspect(filepath: str):
    """查看轨迹信息"""
    fp = Path(filepath)
    if not fp.is_absolute():
        fp = MOTION_DIR / fp
    if not fp.exists():
        print(f"文件不存在: {fp}")
        return
    with open(fp) as f:
        data = json.load(f)
    meta = data.get("meta", {})
    print(f"名称: {meta.get('name', '?')}")
    print(f"时长: {meta.get('duration_s', '?')}s")
    print(f"帧数: {meta.get('frames', '?')}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="动作库管理")
    sub = p.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="列出所有动作")

    p_record = sub.add_parser("record", help="录制新轨迹")
    p_record.add_argument("name", help="动作名称")
    p_record.add_argument("--keywords", nargs="+", default=[], help="触发关键词")

    p_delete = sub.add_parser("delete", help="删除动作")
    p_delete.add_argument("name", help="动作名称")

    p_inspect = sub.add_parser("inspect", help="查看轨迹信息")
    p_inspect.add_argument("file", help="轨迹文件名")

    args = p.parse_args()

    if args.cmd == "list":
        list_actions()
    elif args.cmd == "delete":
        delete_action(args.name)
    elif args.cmd == "inspect":
        inspect(args.file)
    elif args.cmd == "record":
        print("录制需在遥操作界面内启动，请使用 teleop_dual_arm.py")
    else:
        p.print_help()
