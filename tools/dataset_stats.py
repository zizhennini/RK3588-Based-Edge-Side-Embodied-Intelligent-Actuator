#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/dataset_stats.py — M1 数据集统计与质量报告（PC 端运行，只读）

扫描数据目录（tools/collect_episodes.py 的产物），输出逐条与汇总统计：
帧率/时长/丢帧、逐关节角度分布（min/max/幅度/均值）、追踪误差、关节限位覆盖率，
并给出体检结论。

用法::

    # 单任务
    python tools/dataset_stats.py data/raw/pick_place
    # 全部任务（data/raw/*/）
    python tools/dataset_stats.py data/raw --all-tasks
    # 落盘报告
    python tools/dataset_stats.py data/raw/pick_place --json report.json --md report.md
    # 附带标定文件 → 检查关节实际使用幅度是否覆盖行程（数据多样性体检）
    python tools/dataset_stats.py data/raw/pick_place --calib config/calibration.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.episode_quality import (JOINT_NAMES, check_episode, check_images,  # noqa: E402
                                   dataset_stats, format_report)
from scripts.json_to_lerobot import episode_sort_key, find_images  # noqa: E402


def load_frames(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("frames", []), int(data.get("fps", 30)), data.get("duration_s")


def scan_task(task_dir: Path, fps: int, requested_s=None, thresholds=None) -> dict:
    """扫描单任务目录 → {"episodes": [...], "stats": {...}}"""
    files = [p for p in task_dir.glob("*.json")
             if p.name not in ("manifest.json", "conversion_summary.json")]
    files.sort(key=episode_sort_key)
    episodes = []
    for p in files:
        try:
            frames, file_fps, _ = load_frames(p)
            q = check_episode(frames, target_fps=file_fps or fps,
                              requested_s=requested_s, thresholds=thresholds)
            imgs = find_images(p)
            img_counts = {c: len(v) for c, v in imgs.items()}
            if img_counts:
                iq = check_images(img_counts, q["metrics"]["frames"])
                q["images"] = iq
                if not iq["ok"]:
                    q["ok"] = False
                    q["reasons"] += iq["reasons"]
            episodes.append({"file": p.name, "frames": frames, "quality": q,
                             "images": img_counts})
        except Exception as e:
            print(f"✗ 跳过 {p.name}: {e}", file=sys.stderr)
    return {"task": task_dir.name, "episodes": episodes,
            "stats": dataset_stats(episodes)}


def calib_coverage(stats: dict, calib_path: str) -> list:
    """对比数据关节幅度与标定行程幅度（数据多样性体检）→ 文本行列表"""
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    lines = ["", "标定行程覆盖（数据幅度 / 标定行程）:"]
    for i, name in enumerate(JOINT_NAMES, start=1):
        entry = calib.get(str(i))
        s = stats["per_joint_deg"].get(name)
        if not entry or not s:
            continue
        cal_span = abs(entry["range_max"] - entry["range_min"]) * 360.0 / 4095.0
        ratio = s["span"] / cal_span if cal_span else 0.0
        flag = "⚠ 幅度偏小（数据多样性不足）" if ratio < 0.3 else "✓"
        lines.append(f"  {name:<14} {s['span']:7.1f}° / {cal_span:7.1f}° "
                     f"= {ratio:5.1%}  {flag}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="M1 数据集统计与质量报告")
    ap.add_argument("path", help="任务目录（如 data/raw/pick_place）或数据根目录")
    ap.add_argument("--all-tasks", action="store_true",
                    help="把 path 当作根目录，统计其下所有子目录")
    ap.add_argument("--calib", default=None, help="标定 JSON（附覆盖率体检）")
    ap.add_argument("--json", default=None, help="统计 JSON 输出路径")
    ap.add_argument("--md", default=None, help="报告 Markdown 输出路径")
    ap.add_argument("--fps", type=int, default=30, help="默认目标帧率（文件缺 fps 时）")
    ap.add_argument("--requested-s", type=float, default=None,
                    help="期望单条时长（用于时长门控，默认关闭）")
    args = ap.parse_args()

    root = Path(args.path)
    if not root.is_dir():
        print(f"✗ 不是目录: {root}", file=sys.stderr)
        return 1
    if args.all_tasks:
        task_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    else:
        task_dirs = [root]
    if not task_dirs:
        print(f"✗ 未发现任务目录: {root}", file=sys.stderr)
        return 1

    results = []
    report = ["# M1 数据集统计报告", ""]
    for td in task_dirs:
        res = scan_task(td, args.fps, args.requested_s)
        results.append(res)
        st = res["stats"]
        cams = sorted({c for e in res["episodes"] for c in (e.get("images") or {})})
        cam_line = ""
        if cams:
            cam_line = "图像: " + ", ".join(
                f"{c}={sum((e.get('images') or {}).get(c, 0) for e in res['episodes'])} 帧"
                for c in cams) + f"（{len(res['episodes'])} 条 episode）"
        report += [f"## 任务: {res['task']}", "", "```", format_report(st),
                   cam_line, "```", ""]
        print(f"\n=== 任务 {res['task']} ===")
        print(format_report(st))
        if cam_line:
            print(cam_line)
        if args.calib:
            cov = calib_coverage(st, args.calib)
            print("\n".join(cov))
            report += ["```"] + cov + ["```", ""]
        if st["episodes_fail"]:
            report.append(f"> ⚠ 不合格 {st['episodes_fail']} 条，见下方逐条列表")
            report.append("")
            for e in res["episodes"]:
                if not e["quality"]["ok"]:
                    report.append(f"- `{e['file']}`: "
                                  + "；".join(e["quality"]["reasons"]))
            report.append("")

    total = dataset_stats([e for r in results for e in r["episodes"]])
    report += ["## 全部任务汇总", "", "```", format_report(total), "```", ""]

    if args.json:
        out = {"tasks": [{"task": r["task"], "stats": r["stats"]} for r in results],
               "total": total}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 统计 JSON: {Path(args.json).resolve()}")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(report), encoding="utf-8")
        print(f"✓ 报告: {Path(args.md).resolve()}")
    return 0 if total["episodes_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
