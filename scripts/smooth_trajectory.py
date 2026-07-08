#!/usr/bin/env python3
"""轨迹平滑与校验工具 — 滤波、去抖、安全阈值检查"""

import sys, os, json, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vla.control import ArmController

MOTION_DIR = Path(__file__).resolve().parent.parent / "motion_library"


def smooth_trajectory(filepath: Path, window: int = 5) -> dict:
    """滑动窗口平均滤波"""
    with open(filepath) as f:
        data = json.load(f)

    joints = []
    for frame in data.get("trajectory", []):
        joints.append(frame.get("joints", []))
    joints = np.array(joints)

    if joints.size == 0:
        return data

    # 滑动平均
    kernel = np.ones(window) / window
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=joints)

    # 限幅滤波：每帧变化不超过 max_delta 度/帧
    max_delta = 5.0
    for i in range(1, len(smoothed)):
        delta = smoothed[i] - smoothed[i - 1]
        over = np.abs(delta) > max_delta
        smoothed[i, over] = smoothed[i - 1, over] + np.sign(delta[over]) * max_delta

    # 写回
    for i, frame in enumerate(data.get("trajectory", [])):
        frame["joints"] = [round(float(v), 4) for v in smoothed[i]]

    data["meta"]["smoothed"] = True
    data["meta"]["smooth_window"] = window
    return data


def validate_trajectory(filepath: Path) -> list[str]:
    """校验轨迹安全性，返回警告列表"""
    warnings = []
    with open(filepath) as f:
        data = json.load(f)

    controller = ArmController  # 只取静态常量
    limits = controller.JOINT_LIMITS
    joint_names = controller.JOINT_NAMES

    for i, frame in enumerate(data.get("trajectory", [])):
        joints = frame.get("joints", [])
        if len(joints) < 6:
            warnings.append(f"帧 {i}: 关节数据不足 ({len(joints)})")
            continue
        for j, name in enumerate(joint_names):
            low, high = limits[name]
            if joints[j] < low or joints[j] > high:
                warnings.append(f"帧 {i} 关节 {j}({name}): {joints[j]:.3f} rad 超限 [{low:.3f}, {high:.3f}]")

    dur = data.get("meta", {}).get("duration_s", 0)
    if dur < 1:
        warnings.append(f"轨迹时长过短: {dur}s")

    return warnings


def process(filepath: str):
    """一步完成平滑+校验"""
    fp = Path(filepath)
    if not fp.is_absolute():
        fp = MOTION_DIR / fp
    if not fp.exists():
        print(f"文件不存在: {fp}")
        return

    print(f"处理: {fp.name}")

    # 校验
    warns = validate_trajectory(fp)
    if warns:
        print(f"  ⚠ 发现 {len(warns)} 个问题:")
        for w in warns:
            print(f"    - {w}")
        print("  请修正后再使用")
    else:
        print("  ✅ 安全校验通过")

    # 平滑
    smoothed = smooth_trajectory(fp, window=5)
    backup = fp.with_suffix(".json.bak")
    fp.rename(backup)
    with open(fp, "w") as f:
        json.dump(smoothed, f, indent=2)
    print(f"  ✅ 平滑完成 (window=5)")
    print(f"  备份: {backup.name}")


def batch_process():
    """批量处理所有未平滑的轨迹"""
    for f in MOTION_DIR.glob("*.json"):
        if f.name == "index.json":
            continue
        if f.name.endswith(".bak"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            if data.get("meta", {}).get("smoothed"):
                continue
            process(f.name)
        except json.JSONDecodeError:
            print(f"  ⚠ {f.name}: JSON 解析失败")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="轨迹平滑与校验")
    p.add_argument("file", nargs="?", help="轨迹文件名（为空则批量处理所有）")
    a = p.parse_args()

    if a.file:
        process(a.file)
    else:
        batch_process()
