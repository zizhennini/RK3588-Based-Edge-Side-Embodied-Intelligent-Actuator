#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/json_to_lerobot.py — 遥操作录制 JSON → 训练数据（PC 端运行）

输入: scripts/lerobot-record-lite 或 hardware.teleop.TeleopPair 输出的 JSON
      {"fps": 30, "frames": [{"J1": deg, ..., "J6": deg, "t": sec}, ...]}

两种输出格式:
  --format npz（默认，仅 numpy）:
      <out>/<name>.npz  state=(N,6) float32 弧度 + timestamps + meta JSON
  --format lerobot（需 PC 端 lerobot env, pip lerobot）:
      LeRobotDataset v2.x（API 写入，自动兼容 0.4.x/0.6.x import 路径）
      每个输入文件 = 一个 episode

用法:
    python scripts/json_to_lerobot.py record_*.json --format npz --out episodes/
    python scripts/json_to_lerobot.py record_a.json record_b.json \
        --format lerobot --repo-id local/so101_teleop --task "拿起杯子" \
        --root ~/datasets/so101_teleop

注: 板端零依赖此脚本（数据流: 板端录制 JSON → rsync → PC 端转换 → 训练）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def load_episode(path: Path) -> dict:
    """读取录制 JSON → {"fps": int, "state": (N,6) 弧度, "timestamps": (N,)}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"{path}: frames 为空")
    state_deg = []
    timestamps = []
    for fr in frames:
        try:
            row = [float(fr[f"J{i}"]) for i in range(1, 7)]
        except (KeyError, TypeError):
            continue  # 丢帧（个别关节读取失败）跳过
        state_deg.append(row)
        timestamps.append(float(fr.get("t", len(timestamps) / data.get("fps", 30))))
    state = np.deg2rad(np.asarray(state_deg, dtype=np.float32))
    return {
        "fps": int(data.get("fps", 30)),
        "state": state,
        "timestamps": np.asarray(timestamps, dtype=np.float32),
        "source": str(path),
    }


def save_npz(episodes: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        name = Path(ep["source"]).stem
        np.savez_compressed(
            out_dir / f"{name}.npz",
            state=ep["state"],
            timestamps=ep["timestamps"],
            meta=json.dumps({
                "fps": ep["fps"],
                "units": "rad",
                "joint_names": JOINT_NAMES,
                "source": ep["source"],
                "total_frames": len(ep["state"]),
            }, ensure_ascii=False),
        )
        print(f"✓ {out_dir / (name + '.npz')}  ({len(ep['state'])} 帧)")


def save_lerobot(episodes: list, repo_id: str, task: str, root: Path) -> None:
    """LeRobotDataset API 写入（兼容 0.4.x / 0.6.x import 路径）"""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 0.6.x
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # 0.4.x

    fps = episodes[0]["fps"]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
    }
    try:
        ds = LeRobotDataset.create(
            repo_id=repo_id, fps=fps, features=features,
            robot_type="so101", root=str(root), push_to_hub=False,
        )
    except TypeError:
        # lerobot 0.4.x 的 create() 无 push_to_hub 参数（0.6.x 引入）
        ds = LeRobotDataset.create(
            repo_id=repo_id, fps=fps, features=features,
            robot_type="so101", root=str(root),
        )
    for ep in episodes:
        for i in range(len(ep["state"])):
            frame = {"observation.state": ep["state"][i]}
            # 0.4.x/0.6.x add_frame 的 task 传递方式差异 → 双路尝试
            try:
                ds.add_frame({**frame, "task": task})
            except (TypeError, KeyError):
                ds.add_frame(frame)
        try:
            ds.save_episode(task=task)
        except TypeError:
            ds.save_episode()
        print(f"✓ episode 写入: {ep['source']} ({len(ep['state'])} 帧)")
    try:
        ds.consolidate()
    except AttributeError:
        pass  # 0.6.x 自动 consolidate
    print(f"✓ LeRobot dataset 完成: repo_id={repo_id} root={root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="遥操作录制 JSON → 训练数据")
    parser.add_argument("inputs", nargs="+", help="录制 JSON 文件（可多个=多 episode）")
    parser.add_argument("--format", choices=["npz", "lerobot"], default="npz")
    parser.add_argument("--out", default="episodes", help="npz 输出目录")
    parser.add_argument("--repo-id", default="local/so101_teleop",
                        help="lerobot repo_id")
    parser.add_argument("--task", default="teleop recording",
                        help="lerobot task 描述")
    parser.add_argument("--root", default=None, help="lerobot dataset root")
    args = parser.parse_args()

    episodes = []
    for p in args.inputs:
        try:
            episodes.append(load_episode(Path(p)))
        except Exception as e:
            print(f"✗ 跳过 {p}: {e}", file=sys.stderr)
    if not episodes:
        print("无有效输入", file=sys.stderr)
        return 1

    total = sum(len(ep["state"]) for ep in episodes)
    print(f"载入 {len(episodes)} 个 episode，共 {total} 帧")

    if args.format == "npz":
        save_npz(episodes, Path(args.out))
    else:
        root = Path(args.root or f"./datasets/{args.repo_id.split('/')[-1]}")
        save_lerobot(episodes, args.repo_id, args.task, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
