#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-b：CV 精定位验收（方块中心 ≤3 px、朝向 ≤10°）

判据来自 docs/测试方案_VLM+外部IK+抓取.md P0-b。分两部分测，缺一不可：

A. **合成图（精确真值）** —— 只有一个已知真值才能谈"误差"。
   在低饱和木色背景上渲染**已知中心/朝向**的彩色方块（4× 超采样抗锯齿，
   保证光栅化本身不是误差来源），再叠噪声/模糊/JPEG 压缩，测：
     - 中心误差（质心 / minAreaRect 中心 / 最小外接圆心 三种估计量对比）
     - 朝向误差（minAreaRect vs PCA，mod 90）
   为什么必须合成：真实图里"方块中心"本身没有客观真值（旋转正方形的
   bbox 中心有偏、质心无偏），手工标注精度也只有 ±5 px，压不住 3 px 判据。

B. **真实图（鲁棒性）** —— 合成图不能代表真实光照。在板端真实采集图上测：
     - 检出率（6 种颜色是否都稳定找到）
     - 机械臂误检（白色塑料被 D435i 白平衡染成淡蓝，H 与蓝色方块重合）
     - 中心对分割扰动的敏感度（S 门限 ±20、腐蚀/膨胀 1 px）
     - 三种中心估计量的离散度（互为交叉校验）
     - 导出叠加图供目视复核

用法::
    python tools/cv_localization_bench.py --data data/raw/pick_place --images 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.cube_locator import (  # noqa: E402
    COLOR_HUE_WINDOWS, COLOR_ZH, CubeCandidate, draw_candidates, find_cubes,
)

#: 实测的各方块中位 HSV（真机图像统计得到）——合成图用它才贴近真实
MEASURED_HSV = {
    "red": (178, 145, 146), "orange": (11, 166, 168), "yellow": (42, 114, 195),
    "green": (86, 250, 141), "blue": (103, 253, 228), "purple": (120, 135, 194),
}
SS = 4          # 超采样倍数
W, H = 640, 480


# ---------------------------------------------------------------------------
# 合成图
# ---------------------------------------------------------------------------
def render_scene(color: str, cx: float, cy: float, side: float, angle_deg: float,
                 rng: np.random.Generator, noise: float = 6.0,
                 blur: int = 0, jpeg: int = 0) -> np.ndarray:
    """渲染一张含单个已知方块的真实感 RGB 图（返回 uint8 RGB）"""
    h, s, v = MEASURED_HSV[color]
    # 背景：低饱和木色（实测木桌 S<90 才会被彩色门限滤掉）
    bg_hsv = np.zeros((H * SS, W * SS, 3), np.uint8)
    bg_hsv[..., 0] = 20
    bg_hsv[..., 1] = 40 + rng.integers(-8, 8)
    bg_hsv[..., 2] = 170 + rng.integers(-12, 12)
    big = cv2.cvtColor(bg_hsv, cv2.COLOR_HSV2RGB)

    # 方块（HSV2RGB 后再画，避免色相插值）
    rect = ((cx * SS, cy * SS), (side * SS, side * SS), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    patch = np.zeros((H * SS, W * SS, 3), np.uint8)
    cv2.fillPoly(patch, [box], (h, s, v))
    m = patch.any(axis=2)
    cube_rgb = cv2.cvtColor(patch, cv2.COLOR_HSV2RGB)
    big[m] = cube_rgb[m]

    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    if blur > 0:
        img = cv2.GaussianBlur(img, (blur | 1, blur | 1), 0)
    if noise > 0:
        img = np.clip(img.astype(np.float32) +
                      rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    if jpeg > 0:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img


def _circle_center(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    (x, y), _ = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return (float(x), float(y))


def _rect_center(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    (x, y), _, _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return (float(x), float(y))


def _ang_err(a: float, b: float) -> float:
    """正方形朝向误差（mod 90，0~45）"""
    d = abs(a - b) % 90.0
    return float(min(d, 90.0 - d))


def part_a(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    angles = list(np.arange(0, 90, 5.0))
    sides = [20.0, 28.0, 39.0, 50.0]
    colors = list(MEASURED_HSV)
    rows = []
    for color in colors:
        for side in sides:
            for ang in angles:
                cx = float(rng.uniform(120, W - 120))
                cy = float(rng.uniform(100, H - 160))
                img = render_scene(color, cx, cy, side, ang, rng, noise=6.0,
                                   blur=3 if side < 25 else 0, jpeg=85)
                cands = find_cubes(img, colors=[color])
                if not cands:
                    rows.append(dict(color=color, side=side, angle=ang, found=False))
                    continue
                c = max(cands, key=lambda k: k.area)
                m8 = c.extra["centroid"]
                d_centroid = float(np.hypot(c.center[0] - cx, c.center[1] - cy))
                rc = _rect_center(m8)
                cc = _circle_center(m8)
                d_rect = float(np.hypot(rc[0] - cx, rc[1] - cy)) if rc else float("nan")
                d_circ = float(np.hypot(cc[0] - cx, cc[1] - cy)) if cc else float("nan")
                rows.append(dict(
                    color=color, side=side, angle=ang, found=True,
                    d_centroid=d_centroid, d_rect=d_rect, d_circle=d_circ,
                    ang_err_minarearect=_ang_err(c.angle_deg, ang),
                    ang_err_pca=_ang_err(c.extra["angle_pca"], ang),
                    area=c.area, extent=c.extent,
                    side_est=c.side_px, side_true=side))
    found = [r for r in rows if r["found"]]
    miss = [r for r in rows if not r["found"]]

    def stats(key):
        v = np.array([r[key] for r in found], dtype=float)
        v = v[np.isfinite(v)]
        return dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                    p99=float(np.percentile(v, 99)), max=float(v.max()))

    # PCA 在 45° 附近的退化
    by_ang = {}
    for r in found:
        by_ang.setdefault(round(r["angle"]), []).append(r["ang_err_pca"])
    pca_by_angle = {a: float(np.median(v)) for a, v in sorted(by_ang.items())}

    side_err = np.array([abs(r["side_est"] - r["side_true"]) for r in found])
    res = dict(
        n_total=len(rows), n_found=len(found), n_miss=len(miss),
        miss_detail=[dict(color=m["color"], side=m["side"], angle=m["angle"])
                     for m in miss[:20]],
        d_centroid=stats("d_centroid"), d_rect=stats("d_rect"), d_circle=stats("d_circle"),
        ang_err_minarearect=stats("ang_err_minarearect"),
        ang_err_pca=stats("ang_err_pca"),
        pca_median_by_angle=pca_by_angle,
        side_est_median_abs_err=float(np.median(side_err)),
        # 逐样本原始值（画分布图用）
        raw=dict(d_centroid=[r["d_centroid"] for r in found],
                 d_rect=[r["d_rect"] for r in found],
                 d_circle=[r["d_circle"] for r in found],
                 ang_err_minarearect=[r["ang_err_minarearect"] for r in found],
                 ang_err_pca=[r["ang_err_pca"] for r in found],
                 angles=[r["angle"] for r in found],
                 sides=[r["side"] for r in found],
                 side_err=[float(v) for v in side_err]),
    )

    print("\n=== A. 合成图（精确真值，%d 张）===" % len(rows))
    print("  检出 %d / %d（漏检 %d）" % (len(found), len(rows), len(miss)))
    if miss:
        print("  漏检明细(前 20):", res["miss_detail"])
    print("  中心误差 (px)         中位      p90      p99      max")
    for key, label in (("d_centroid", "质心"), ("d_rect", "minAreaRect 中心"),
                       ("d_circle", "最小外接圆心")):
        s = res[key]
        print("    %-18s %7.3f %8.3f %8.3f %8.3f"
              % (label, s["median"], s["p90"], s["p99"], s["max"]))
    s = res["ang_err_minarearect"]
    print("  朝向误差 minAreaRect  中位 %.2f°  p90 %.2f°  max %.2f°"
          % (s["median"], s["p90"], s["max"]))
    s = res["ang_err_pca"]
    print("  朝向误差 PCA          中位 %.2f°  p90 %.2f°  max %.2f°  ← 45° 附近退化"
          % (s["median"], s["p90"], s["max"]))
    worst = sorted(pca_by_angle.items(), key=lambda kv: -kv[1])[:5]
    print("  PCA 最差的 5 个角度: " + ", ".join("%d°→%.1f°" % (a, e) for a, e in worst))
    print("  等效边长误差 中位 %.2f px" % res["side_est_median_abs_err"])
    return res


# ---------------------------------------------------------------------------
# 真实图
# ---------------------------------------------------------------------------
def sample_images(data_dir: Path, n_images: int) -> list:
    eps = sorted(data_dir.glob("episode_*_images"))
    if not eps:
        return []
    out = []
    per = max(1, n_images // max(1, len(eps)))
    for ep in eps:
        fr = sorted((ep / "front").glob("*.jpg"))
        if not fr:
            continue
        idx = np.linspace(0, len(fr) - 1, per).astype(int)
        out += [fr[i] for i in idx]
        if len(out) >= n_images:
            break
    return out[:n_images]


def part_b(data_dir: Path, n_images: int, out_dir: Path, seed: int = 0) -> dict:
    files = sample_images(data_dir, n_images)
    if not files:
        print("\n（跳过 B：%s 下无 episode_*_images）" % data_dir)
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    per_img = []
    shifts, spreads, big_areas = [], [], []
    n_lost = 0
    color_hits = {c: 0 for c in COLOR_HUE_WINDOWS}
    n_saved = 0
    for k, f in enumerate(files):
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        base = find_cubes(rgb)
        for c in base:
            color_hits[c.color] += 1
            big_areas.append(c.area)
        per_img.append(dict(file=f.name, n=len(base),
                            colors=[c.color for c in base],
                            areas=[c.area for c in base]))

        # 三种中心估计量的离散度
        for c in base:
            m8 = c.extra["centroid"]
            rc, cc = _rect_center(m8), _circle_center(m8)
            if rc and cc:
                spreads.append(max(np.hypot(c.center[0] - rc[0], c.center[1] - rc[1]),
                                   np.hypot(c.center[0] - cc[0], c.center[1] - cc[1]),
                                   np.hypot(rc[0] - cc[0], rc[1] - cc[1])))

        # 扰动敏感度：换分割参数后同一方块中心漂移多少
        # 注意区分两种情况——「候选项消失/换了个目标」不等于「中心不稳」，
        # 若把前者算进漂移，最大值会被无关事件污染（实测 max 14 px 就是这种）。
        for sat in (70, 90, 110):
            for kw in (dict(), dict(erode=1), dict(dilate=1)):
                alt = find_cubes(rgb, sat_min=sat, **kw)
                for c in base:
                    best, bd = None, 1e9
                    for a in alt:
                        if a.color != c.color:
                            continue
                        d = np.hypot(a.center[0] - c.center[0], a.center[1] - c.center[1])
                        if d < bd:
                            best, bd = a, d
                    if best is None or bd > 15:
                        n_lost += 1
                        continue
                    # 面积变化过大 → 匹配到的其实是另一个分量（合并/拆裂）
                    if abs(best.area - c.area) > 0.5 * c.area:
                        n_lost += 1
                        continue
                    shifts.append(bd)

        if n_saved < 6:
            vis = draw_candidates(rgb, base)
            cv2.imwrite(str(out_dir / ("cv_overlay_%02d_%s" % (k, f.name))),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            n_saved += 1

    ns = np.array(per_img and [p["n"] for p in per_img] or [0])
    sh = np.array(shifts) if shifts else np.array([np.nan])
    sp = np.array(spreads) if spreads else np.array([np.nan])
    res = dict(
        n_images=len(per_img),
        candidates_per_image=dict(median=float(np.median(ns)), min=int(ns.min()),
                                  max=int(ns.max()),
                                  hist={str(int(v)): int((ns == v).sum())
                                        for v in sorted(set(ns.tolist()))}),
        color_hits=color_hits,
        missing_colors=[c for c, v in color_hits.items() if v == 0],
        n_candidates_total=int(ns.sum()),
        shift_median_px=float(np.nanmedian(sh)),
        shift_p90_px=float(np.nanpercentile(sh, 90)) if np.isfinite(sh).any() else float("nan"),
        shift_max_px=float(np.nanmax(sh)) if np.isfinite(sh).any() else float("nan"),
        n_perturb_matched=int(sh.size), n_perturb_lost=int(n_lost),
        center_spread_median_px=float(np.nanmedian(sp)),
        center_spread_max_px=float(np.nanmax(sp)),
        max_area=int(max(big_areas)) if big_areas else 0,
        per_image=per_img,
        overlay_dir=str(out_dir),
    )
    print("\n=== B. 真实图（%d 张，板端采集）===" % len(per_img))
    print("  每图候选数: 中位 %.0f  范围 %d~%d  直方图 %s"
          % (res["candidates_per_image"]["median"], res["candidates_per_image"]["min"],
             res["candidates_per_image"]["max"], res["candidates_per_image"]["hist"]))
    print("  各颜色累计命中: " + ", ".join(
        "%s %d" % (COLOR_ZH[c], v) for c, v in color_hits.items()))
    if res["missing_colors"]:
        print("  ⚠ 未检出的颜色: %s" % [COLOR_ZH[c] for c in res["missing_colors"]])
    print("  最大候选面积 %d px（机械臂大分量实测 17969 → 应被面积/形状过滤掉）"
          % res["max_area"])
    print("  扰动敏感度（S 门限 ±20 / 腐蚀 / 膨胀）中心漂移: 中位 %.2f px  p90 %.2f px  max %.2f px"
          % (res["shift_median_px"], res["shift_p90_px"], res["shift_max_px"]))
    print("    匹配成功 %d 次 / 丢失或换目标 %d 次（丢失单列，不计入漂移）"
          % (res["n_perturb_matched"], res["n_perturb_lost"]))
    print("  三种中心估计量互差: 中位 %.2f px  max %.2f px"
          % (res["center_spread_median_px"], res["center_spread_max_px"]))
    print("  叠加图已写 %s（cv_overlay_*.jpg）" % out_dir)
    return res


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="P0-b CV 精定位验收")
    ap.add_argument("--data", type=Path, default=Path("data/raw/pick_place"))
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/p0b_overlay"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    print("=" * 66)
    print("P0-b CV 精定位验收（判据：中心 ≤3 px、朝向 ≤10°）")
    print("=" * 66)

    a = {} if args.skip_synthetic else part_a(args.seed)
    b = part_b(args.data, args.images, args.out_dir, args.seed)

    print("\n" + "=" * 66)
    print("判据")
    print("=" * 66)
    checks = []
    if a:
        checks.append(("① 合成图中心误差中位 ≤3 px", a["d_centroid"]["median"] <= 3.0,
                       "%.3f px" % a["d_centroid"]["median"]))
        checks.append(("② 合成图中心误差 p99 ≤3 px", a["d_centroid"]["p99"] <= 3.0,
                       "%.3f px" % a["d_centroid"]["p99"]))
        checks.append(("③ 合成图朝向误差中位 ≤10°",
                       a["ang_err_minarearect"]["median"] <= 10.0,
                       "%.2f°" % a["ang_err_minarearect"]["median"]))
        checks.append(("④ 合成图检出率 ≥95%",
                       a["n_found"] / max(1, a["n_total"]) >= 0.95,
                       "%.1f%%" % (100.0 * a["n_found"] / max(1, a["n_total"]))))
    if b:
        checks.append(("⑤ 真实图中心对分割扰动漂移中位 ≤3 px",
                       b["shift_median_px"] <= 3.0, "%.2f px" % b["shift_median_px"]))
        checks.append(("⑥ 真实图无大面积误检（<5000 px）",
                       b["max_area"] < 5000, "max %d px" % b["max_area"]))
        checks.append(("⑦ 真实图 6 色均检出", not b["missing_colors"],
                       "缺 %s" % b["missing_colors"]))
    for label, ok, extra in checks:
        print("  %-34s %s   (%s)" % (label, "✓ 通过" if ok else "✗ 未通过", extra))
    all_ok = all(ok for _, ok, _ in checks)
    print("\n结论: P0-b %s" % ("通过 ✅" if all_ok else "未通过 ❌"))

    if args.json_out:
        args.json_out.write_text(json.dumps(
            dict(synthetic=a, real=b,
                 checks={l: bool(o) for l, o, _ in checks}, pass_all=bool(all_ok)),
            ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        print("结果已写 %s" % args.json_out)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
