#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/check_camera_coverage.py — 相机覆盖度验收（8 位姿边界测试）

背景（见 docs/realsense_camera_survey.md）：第三方固定相机的首要职责是"腕部相机被抓握遮挡时的
兜底"，但**没有任何开源项目文档化"夹爪遍历工作空间边界做覆盖度验收"的流程**——本工具补这一环。

做法：引导操作者把夹爪依次摆到工作空间的 8 个边界位姿（四角×高低 / 最近 / 最远），
每个位姿拍一帧 → 自动检测夹爪（浅色目标 = SO101 白色打印件）位置 → 计算其到画面四边的余量，
保存标注图并给出结论；全部位姿通过后相机才可以锁死并开始采集。

    python3 tools/check_camera_coverage.py                 # 8 位姿交互
    python3 tools/check_camera_coverage.py --poses 4       # 只测 4 个角位姿
    python3 tools/check_camera_coverage.py --camera front --out coverage_check
    python3 tools/check_camera_coverage.py --no-detect      # 关闭自动检测（纯存图人工判读）

判读：夹爪到画幅四边余量 **≥10%** 为合格；某位姿检测不到夹爪 = 疑似出画面或被遮挡。
工作区多边形（tools/cam_align_d435i.py 标定）存在时会一并叠加并检查夹爪是否落在区内。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.cam_sink import CameraSet  # noqa: E402

POSES = [
    ("1 左前角·低位", "LF-low", "把夹爪移到工作区左前角、贴近桌面（低位）"),
    ("2 左前角·高位", "LF-high", "同左前角，抬高约 15cm（高位）"),
    ("3 右前角·低位", "RF-low", "移到右前角、贴近桌面"),
    ("4 右前角·高位", "RF-high", "同右前角，抬高约 15cm"),
    ("5 左后角·低位", "LB-low", "移到左后角（靠近基座一侧）、贴近桌面"),
    ("6 右后角·低位", "RB-low", "移到右后角、贴近桌面"),
    ("7 最远端", "farthest", "尽量伸到离相机最远处（工作区前缘外侧一点）"),
    ("8 最近端", "nearest", "尽量收回到离相机最近处（靠近基座/画面下缘）"),
]
MIN_MARGIN_PCT = 10.0        # 夹爪到画幅四边的最小余量（%）


def draw_pixel_grid(img, step: int = 40, label_every: int = 80):
    """像素网格 + 坐标标注（便于人工读夹爪尖端坐标；与 cam_align 工具同款）"""
    if not step:
        return img
    import cv2
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (70, 70, 70), 1)
        if x % label_every == 0:
            for col, th in (((0, 0, 0), 3), ((0, 255, 255), 1)):
                cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, col, th, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (70, 70, 70), 1)
        if y % label_every == 0:
            for col, th in (((0, 0, 0), 3), ((0, 255, 255), 1)):
                cv2.putText(img, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, col, th, cv2.LINE_AA)
    return img


def detect_gripper(bgr: np.ndarray, min_area_frac: float = 0.001,
                   max_area_frac: float = 0.15, tophat_thresh: int = 25,
                   tophat_ksize: int = 61, sat_max: int = 90,
                   bg_area_frac: float = 0.03, max_aspect: float = 3.0,
                   min_thickness: int = 35):
    """自动检测夹爪（**尽力而为，不可靠时请人工标点**）→ (bbox, 最大候选面积占比)

    实测结论（板端真实位姿图，见 CHANGELOG 2026-10-05）：本场景**颜色阈值不可靠**——
      ① 桌面有亮木纹带，局部提亮后呈细长条（~200×20），会被误判为夹爪；
      ② 夹爪白色在不同位置差异极大：受光处 V≈254/S≈40，暗部因白平衡偏蓝 V≈130/S≈133
         ⇒ 既不能只靠"亮"，也不能只靠"低饱和"。
    因此本函数只做"尽力检测"（top-hat 局部提亮 + 低饱和 + **剔除细长杂斑** + 面积上下限）；
    **人工标点优先**（--manual 或自动失败时输入夹爪尖端像素坐标）。
    """
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    sat = hsv[:, :, 1]

    # ① 局部提亮
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (int(tophat_ksize), int(tophat_ksize)))
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, kernel)

    # ② 显著提亮 且 低饱和（白色/浅色物体）
    mask = cv2.inRange(tophat, int(tophat_thresh), 255)
    mask = cv2.bitwise_and(mask, cv2.inRange(sat, 0, int(sat_max)))

    # ③ 清理（小核，避免把本就不大的夹爪腐蚀掉）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((7, 7), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(bgr.shape[0] * bgr.shape[1])
    best, best_frac, seen = None, 0.0, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        frac = area / frame_area
        seen = max(seen, frac)
        x, y, w, h = cv2.boundingRect(c)
        if frac < min_area_frac or frac > max_area_frac:
            continue
        if frac > bg_area_frac and y <= 2:      # 贴顶边的大面积 = 背景（白墙等）
            continue
        aspect = max(w, h) / max(1.0, float(min(w, h)))
        if aspect > max_aspect or min(w, h) < min_thickness:
            continue                             # 剔除细长杂斑（桌面木纹亮带 ~200×20）
        if best is None or frac > best_frac:
            best, best_frac = (x, y, w, h), frac
    return best, (best_frac if best else seen)


def point_margins(px: float, py: float, width: int, height: int) -> dict:
    """人工标点（夹爪尖端像素）→ 到画面四边的余量（%）

    比自动检测可靠：只需一个点即可判断"是否贴边/出画"，操作者从叠加了像素网格的
    画面里读数（网格每 40px、坐标每 80px 标注一次）。
    """
    px = min(max(float(px), 0.0), float(width))
    py = min(max(float(py), 0.0), float(height))
    return {
        "left_pct": 100.0 * px / width,
        "right_pct": 100.0 * (width - px) / width,
        "top_pct": 100.0 * py / height,
        "bottom_pct": 100.0 * (height - py) / height,
        "center_pct": (round(100.0 * px / width, 1), round(100.0 * py / height, 1)),
        "manual_point": [round(px, 1), round(py, 1)],
    }


def parse_point(raw: str):
    """解析人工标点输入 → (x, y) 或 None（支持 "300,380" / "300 380"；空/非法返回 None）"""
    if not raw:
        return None
    s = raw.replace("，", ",").replace("(", "").replace(")", "").strip()
    for sep in (",", " ", ";", "x"):
        if sep in s:
            parts = [p for p in s.split(sep) if p.strip()]
            break
    else:
        parts = [s]
    try:
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except (ValueError, TypeError):
        pass
    print(f"    ⚠ 无法解析坐标 {raw!r}（应形如 \"300,380\"），本次按自动结果处理")
    return None


def bbox_margins(bbox, width: int, height: int) -> dict:
    """bbox → 到画面四边的余量（%）"""
    x, y, w, h = bbox
    return {
        "left_pct": 100.0 * x / width,
        "right_pct": 100.0 * (width - (x + w)) / width,
        "top_pct": 100.0 * y / height,
        "bottom_pct": 100.0 * (height - (y + h)) / height,
        "center_pct": (round(100.0 * (x + w / 2) / width, 1),
                       round(100.0 * (y + h / 2) / height, 1)),
    }


def load_poly():
    cfg = Path("config/cam_align.json")
    if not cfg.exists():
        return None
    try:
        d = json.loads(cfg.read_text(encoding="utf-8"))
        return np.array(d["corners"], dtype=np.float32) if d.get("corners") else None
    except Exception:
        return None


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="相机覆盖度验收（8 位姿边界测试）")
    ap.add_argument("--cameras", default="./config/cameras.json")
    ap.add_argument("--camera", default="front")
    ap.add_argument("--out", default="coverage_check")
    ap.add_argument("--poses", type=int, default=8, help="测试位姿数（4=只测四角低位）")
    ap.add_argument("--min-margin", type=float, default=MIN_MARGIN_PCT)
    ap.add_argument("--no-detect", action="store_true", help="关闭自动检测（纯存图）")
    ap.add_argument("--tophat", type=int, default=25,
                    help="局部提亮阈值（top-hat 后，默认 25；检不到就调小，误检就调大）")
    ap.add_argument("--tophat-ksize", type=int, default=61,
                    help="top-hat 核尺寸（默认 61，应大于夹爪尺寸）")
    ap.add_argument("--sat-max", type=int, default=90,
                    help="低饱和上限 S（默认 90；白色物体通常 <60）")
    ap.add_argument("--debug-detect", action="store_true", help="打印检测中间统计")
    ap.add_argument("--no-manual", action="store_true",
                    help="不提示人工标点（纯自动检测，失败记未检测）")
    ap.add_argument("--pixel-grid", type=int, default=40,
                    help="人工标点时叠加的像素网格步长（默认 40，0=不叠）")
    args = ap.parse_args()

    import cv2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = POSES[:4] + POSES[4:8] if args.poses >= 8 else POSES[:args.poses]

    cs = CameraSet.from_config(args.cameras)
    cs.start()
    if not cs.wait_ready(timeout_s=25.0).get(args.camera):
        print(f"✗ 相机 {args.camera} 未就绪")
        cs.stop()
        return 1
    poly = load_poly()
    print(f"相机 {args.camera} | 位姿 {len(poses)} 个 | 余量合格线 {args.min_margin:.0f}%"
          + ("| 已载入工作区多边形" if poly is not None else "| 无工作区多边形（可先用 cam_align 标定）"))

    results = []
    for i, (name, ascii_name, hint) in enumerate(poses, start=1):
        print(f"\n--- [{i}/{len(poses)}] {name} ({ascii_name}) ---")
        print(f"    {hint}")
        try:
            input("    摆好后按 Enter 拍照...")
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            break
        time.sleep(0.4)                       # 等一帧新的（避免拍到运动中的模糊帧）
        got = cs.latest_age(args.camera)
        if got is None:
            print("    ✗ 未取到图像")
            continue
        bgr, ts, age = got
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        vis = bgr.copy()
        if poly is not None and len(poly) >= 3:
            cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 165, 255), 2)
        bbox, frac = (None, 0.0) if args.no_detect else detect_gripper(
            bgr, tophat_thresh=args.tophat, tophat_ksize=args.tophat_ksize,
            sat_max=args.sat_max)
        verdict, detail = "手工判读", {}
        if bbox is None and not args.no_detect:
            print(f"    ⚠ 未自动检测到夹爪（最大候选占比 {frac * 100:.2f}%）")
        elif bbox is not None:
            print(f"    自动检测: bbox={bbox} 占比 {frac * 100:.2f}%")

        # ── 人工标点优先（本场景颜色检测不可靠；叠加像素网格便于读数）──
        manual = None
        if not args.no_manual:
            vis = draw_pixel_grid(vis, step=args.pixel_grid)
            print("    图中夹爪尖端坐标（看叠加的像素网格读数）: ", end="")
            try:
                raw = input('形如 "300,380"，直接回车=采用自动检测结果: ').strip()
            except (KeyboardInterrupt, EOFError):
                raw = ""
            manual = parse_point(raw)

        if manual is not None:
            m = point_margins(manual[0], manual[1], w, h)
            detail = m
            mn = min(m["left_pct"], m["right_pct"], m["top_pct"], m["bottom_pct"])
            verdict = "合格" if mn >= args.min_margin else f"余量不足({mn:.0f}%)"
            cx, cy = int(manual[0]), int(manual[1])
            cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
            if poly is not None:
                inside = cv2.pointPolygonTest(poly.astype(np.int32),
                                              (float(cx), float(cy)), False) >= 0
                detail["inside_workspace"] = bool(inside)
                if not inside:
                    verdict += " 且不在工作区内"
        elif bbox is not None:
            x, y, bw, bh = bbox
            m = bbox_margins(bbox, w, h)
            detail = m
            mn = min(m["left_pct"], m["right_pct"], m["top_pct"], m["bottom_pct"])
            verdict = "合格" if mn >= args.min_margin else f"余量不足({mn:.0f}%)"
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            if poly is not None:
                inside = cv2.pointPolygonTest(
                    poly.astype(np.int32),
                    (float(x + bw / 2), float(y + bh / 2)), False) >= 0
                detail["inside_workspace"] = bool(inside)
                if not inside:
                    verdict += " 且不在工作区内"
        # 画面文字只用 ASCII（cv2.putText 不支持中文，中文位姿名会渲染成 ???）
        txt = f"[{i}] {ascii_name}  {verdict}  min_margin=" + (
            f"{min(detail.get('left_pct', 0), detail.get('right_pct', 0), detail.get('top_pct', 0), detail.get('bottom_pct', 0)):.0f}%"
            if detail else "n/a")
        for k, line in enumerate((txt, f"age={age * 1000:.0f}ms  blob={frac * 100:.2f}%")):
            cv2.putText(vis, line, (8, 22 + k * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, line, (8, 22 + k * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        p = out_dir / f"pose_{i}_{ascii_name}.jpg"
        cv2.imwrite(str(p), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f"    → {verdict}  图像延迟 {age * 1000:.0f}ms" +
              ("（人工标点）" if manual is not None else "（自动检测）"))
        if detail:
            print(f"    余量: 左{detail['left_pct']:.0f}% 右{detail['right_pct']:.0f}% "
                  f"上{detail['top_pct']:.0f}% 下{detail['bottom_pct']:.0f}%"
                  + (f"  在工作区内={detail.get('inside_workspace')}"
                     if "inside_workspace" in detail else ""))
        print(f"    标注图: {p}")
        results.append({"pose": name, "verdict": verdict, "age_ms": round(age * 1000, 1),
                        "blob_frac": round(frac, 5), **detail})

    cs.stop()
    bad = [r for r in results if r["verdict"].startswith("余量不足") or "不在工作区" in r["verdict"]]
    none_found = [r for r in results if r["verdict"] == "手工判读"]
    print("\n" + "=" * 62)
    print(f"覆盖度验收: {len(results)} 个位姿，合格 {len(results) - len(bad) - len(none_found)}，"
          f"不合格 {len(bad)}，未检测 {len(none_found)}（需人工看标注图）")
    for r in bad:
        print(f"  ✗ {r['pose']}: {r['verdict']}")
    if none_found:
        print(f"  ⚠ 未自动检测到夹爪的位姿: {[r['pose'] for r in none_found]}"
              f"（浅色阈值未命中：可能出画面/被遮挡，请人工核对标注图）")
    if not bad and not none_found:
        print("  ✓ 全部位姿合格 —— 可锁死相机并开始采集")
    else:
        print("  建议: 抬高/后移相机或扩大视野，使夹爪在所有边界位姿下距画幅四边 ≥"
              f"{args.min_margin:.0f}%")
    (out_dir / "coverage_report.json").write_text(
        json.dumps({"camera": args.camera, "min_margin_pct": args.min_margin,
                    "poses": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"报告: {(out_dir / 'coverage_report.json').resolve()}")
    return 0 if (not bad and not none_found) else 2


if __name__ == "__main__":
    sys.exit(main())
