#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/episode_review.py — 每集人工审核卡片生成（M1 采集闭环，无 GUI 也能核对）

参考 so101-nexus 的"录完回放 + 人工 Approve 才入库"流程（本仓库 docs/open_source_reference.md §3 #7），
在无图形界面的板端用**静态审核卡片**替代视频回放：把一条 episode 的抽样帧与关节曲线画进一张 PNG，
供操作者在 PC 上（或板端桌面）快速判断该集是否合格。

    python tools/episode_review.py data/raw/pick_place                 # 全部 episode
    python tools/episode_review.py data/raw/pick_place --out review/   # 指定输出目录
    python tools/episode_review.py data/raw/pick_place --frames 8      # 每相机抽样帧数

卡片内容（自上而下）:
  1. 每相机一行抽样帧（按时间均匀抽取，标注帧号）
  2. 关节曲线图: J1..J6（主臂指令，实线）与 F1..F6（从臂实际，虚线）叠加 → 直接看出跟随好坏
  3. 底部文字: 帧数/时长/fps/丢帧/追踪误差/图像帧数/质检结论
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.episode_quality import (JOINT_NAMES, check_episode, check_images,  # noqa: E402
                                  tracking_summary)
from scripts.json_to_lerobot import episode_sort_key, find_images  # noqa: E402

THUMB_W, THUMB_H = 320, 240
PLOT_H = 220
LINE_COLORS = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
               (171, 71, 188), (0, 172, 193)]


def _workspace_poly():
    """读取 cam_align 工具标定的工作区多边形 → (camera_name, np.ndarray) 或 (None, None)

    由 tools/cam_align_d435i.py 生成 config/cam_align.json；审核卡片把它画在对应相机的帧上，
    便于一眼判断物体/夹爪是否落在工作区内（采集一致性）。
    """
    cfg = Path("config/cam_align.json")
    if not cfg.exists():
        return None, None
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("corners"):
            return d.get("camera"), np.array(d["corners"], dtype=np.float32)
    except Exception:
        pass
    return None, None


def _thumb(path: Path, poly=None, src_wh=None):
    """读图缩放为缩略图；给出多边形时按比例叠加工作区标记"""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return None
    th = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    if poly is not None and src_wh:
        sx, sy = THUMB_W / src_wh[0], THUMB_H / src_wh[1]
        p = (poly * np.array([sx, sy], dtype=np.float32)).astype(np.int32)
        cv2.polylines(th, [p], True, (0, 165, 255), 1, cv2.LINE_AA)
    return th


def _label(img, text: str, y: int = 20):
    import cv2
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _plot_tracks(frames, width: int) -> np.ndarray:
    """关节曲线: 上=主臂指令 J*，下=从臂实际 F*（若存在）"""
    import cv2
    canvas = np.full((PLOT_H, width, 3), 255, np.uint8)
    if not frames:
        return _label(canvas, "无帧数据", 30)

    t = np.array([float(f.get("t", i)) for i, f in enumerate(frames)])
    span = max(1e-6, float(t[-1] - t[0]))
    vals = {f"J{i}": [float(f[f"J{i}"]) for f in frames if f"J{i}" in f]
            for i in range(1, 7)}
    allv = [v for lst in vals.values() for v in lst]
    if not allv:
        return _label(canvas, "无关节数据", 30)
    lo, hi = min(allv), max(allv)
    pad = max(5.0, (hi - lo) * 0.08)
    lo, hi = lo - pad, hi + pad

    def xy(i, v):
        x = int(20 + (width - 40) * (t[i] - t[0]) / span)
        y = int(PLOT_H - 24 - (PLOT_H - 44) * (v - lo) / (hi - lo))
        return x, y

    for grid in range(5):                       # 横向参考线
        y = int(PLOT_H - 24 - (PLOT_H - 44) * grid / 4)
        cv2.line(canvas, (20, y), (width - 20, y), (232, 232, 232), 1)
    cv2.putText(canvas, f"关节曲线 J*(实) vs F*(虚)  {lo:.0f}~{hi:.0f} deg",
                (22, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    for i in range(1, 7):
        key = f"J{i}"
        if not vals[key]:
            continue
        pts = np.array([xy(k, v) for k, v in enumerate(vals[key])], np.int32)
        cv2.polylines(canvas, [pts], False, LINE_COLORS[i - 1], 2, cv2.LINE_AA)
        fkey = f"F{i}"
        if all(fkey in f for f in frames):
            fpts = np.array([xy(k, float(frames[k][fkey]))
                             for k in range(len(frames))], np.int32)
            cv2.polylines(canvas, [fpts], False,
                          tuple(int(c * 0.55 + 255 * 0.45) for c in LINE_COLORS[i - 1]),
                          1, cv2.LINE_AA)
        cv2.putText(canvas, f"J{i}", (width - 34, 18 + (i - 1) * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, LINE_COLORS[i - 1], 1, cv2.LINE_AA)
    return canvas


def build_card(episode_json: Path, out_path: Path, n_frames: int = 6,
               target_fps: int = 30, requested_s=None) -> dict:
    """生成单集审核卡片 → (卡片路径, 质检结果)"""
    import cv2
    with open(episode_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    fps = int(data.get("fps", target_fps))
    q = check_episode(frames, target_fps=fps, requested_s=requested_s)

    cams = find_images(episode_json)
    img_counts = {c: len(v) for c, v in cams.items()}
    if img_counts:
        iq = check_images(img_counts, q["metrics"]["frames"])
        q["images"] = iq
        if not iq["ok"]:
            q["ok"] = False
            q["reasons"] += iq["reasons"]

    rows = []
    poly_cam, poly = _workspace_poly()
    for cam, files in cams.items():
        if not files:
            continue
        use_poly, src_wh = (poly, None) if (poly is not None and cam == poly_cam) \
            else (None, None)
        if use_poly is not None:
            probe = cv2.imread(str(files[0]))
            if probe is not None:
                src_wh = (probe.shape[1], probe.shape[0])
            else:
                use_poly = None
        idx = np.linspace(0, len(files) - 1, min(n_frames, len(files))).astype(int)
        tiles = []
        for k in idx:
            th = _thumb(files[k], use_poly, src_wh)
            if th is None:
                th = np.full((THUMB_H, THUMB_W, 3), 40, np.uint8)
                th = _label(th, "读图失败")
            tiles.append(_label(th, f"{cam} #{k}"))
        row = np.hstack(tiles)
        rows.append(row)

    width = max([r.shape[1] for r in rows] + [THUMB_W * n_frames])
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for r in rows]
    rows.append(_plot_tracks(frames, width))

    tr = tracking_summary(frames)
    m = q["metrics"]
    info = (f"{episode_json.stem}  帧 {m['frames']}  时长 {m['duration_s']}s  fps {m['fps']}  "
            f"丢帧 {m['dropped']}  "
            + (f"追踪最差 J{tr['worst_joint']} {tr['worst_mean']:.2f}deg  " if tr["has_data"] else "")
            + (f"图像 " + ",".join(f"{c}:{v}" for c, v in img_counts.items()) + "  " if img_counts else "")
            + ("✓ 合格" if q["ok"] else "✗ " + "; ".join(q["reasons"])))
    bar = np.full((30, width, 3), (250, 250, 250), np.uint8)
    color = (30, 130, 60) if q["ok"] else (30, 30, 200)
    cv2.putText(bar, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    rows.append(bar)

    card = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), card, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return {"card": str(out_path), "quality": q, "images": img_counts}


def main() -> int:
    ap = argparse.ArgumentParser(description="每集审核卡片生成（M1 采集闭环）")
    ap.add_argument("task_dir", help="任务目录（如 data/raw/pick_place）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <task_dir>/review）")
    ap.add_argument("--frames", type=int, default=6, help="每相机抽样帧数")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--requested-s", type=float, default=None, help="期望时长（用于时长门控）")
    args = ap.parse_args()

    task = Path(args.task_dir)
    if not task.is_dir():
        print(f"✗ 不是目录: {task}", file=sys.stderr)
        return 1
    out_dir = Path(args.out) if args.out else task / "review"
    files = [p for p in task.glob("*.json")
             if p.name not in ("manifest.json", "conversion_summary.json")]
    files.sort(key=episode_sort_key)
    if not files:
        print(f"✗ 未发现 episode JSON: {task}", file=sys.stderr)
        return 1

    ok = bad = 0
    for p in files:
        r = build_card(p, out_dir / f"{p.stem}_review.jpg", args.frames,
                       args.fps, args.requested_s)
        m = r["quality"]["metrics"]
        status = "✓" if r["quality"]["ok"] else "✗"
        print(f"{status} {r['card']}  帧{m['frames']} fps{m['fps']} "
              + ("  ".join(r["quality"]["reasons"]) if not r["quality"]["ok"] else ""))
        ok += 1 if r["quality"]["ok"] else 0
        bad += 0 if r["quality"]["ok"] else 1
    print(f"\n审核卡片已生成: {out_dir.resolve()}  合格 {ok} / 不合格 {bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
