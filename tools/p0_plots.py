#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0 三张分布图（VLM 偏差 / CV 误差 / IK 时间）

读 tools/ 里三个基准脚本导出的 JSON，输出 PNG。图内一律用 ASCII 标签 ——
matplotlib 默认字体没有中文字形，写中文会全变成方框。

用法::
    python tools/p0_plots.py --results-dir .research/p0_results --out-dir docs/p0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(d: Path, name: str):
    p = d / name
    if not p.exists():
        print("  (缺 %s，跳过相关子图)" % p.name)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="P0 分布图")
    ap.add_argument("--results-dir", type=Path, default=Path(".research/p0_results"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/p0"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coord = _load(args.results_dir, "p0a_coord.json")
    som = _load(args.results_dir, "p0a_both20.json")
    cv = _load(args.results_dir, "p0b.json")
    ik = _load(args.results_dir, "p0_ik.json")
    made = []

    # ── 图 1：VLM 像素偏差 + SoM 准确率 ──────────────────────────────
    if coord or som:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        if coord:
            e = np.array([r["err_min"] for r in coord["coord"]["rows"]
                          if r["err_min"] is not None], dtype=float)
            ax = axes[0]
            ax.hist(e, bins=np.arange(0, 700, 25), color="#c44", edgecolor="k")
            ax.axvline(10, color="g", ls="--", lw=1.5, label="target <=10 px")
            ax.axvline(50, color="orange", ls="--", lw=1.5, label="unusable >50 px")
            ax.axvline(np.median(e), color="b", lw=1.5,
                       label="median %.0f px" % np.median(e))
            ax.set_xlabel("VLM coordinate error (px)")
            ax.set_ylabel("count")
            ax.set_title("P0-a  VLM direct coordinate output (n=%d)\nimage is 640x480"
                         % e.size)
            ax.legend(fontsize=8)
        if som and "som" in som:
            rows = som["som"]["rows"]
            ok = sum(1 for r in rows if r["correct"])
            n = len(rows)
            ax = axes[1]
            ax.bar(["correct", "wrong"], [ok, n - ok],
                   color=["#4a4", "#c44"], edgecolor="k")
            ax.set_ylim(0, max(1, n) * 1.2)
            ax.set_ylabel("count")
            ax.set_title("P0-a  Set-of-Mark index pick (n=%d)\naccuracy %.0f%%"
                         % (n, 100.0 * ok / max(1, n)))
            for i, v in enumerate([ok, n - ok]):
                ax.text(i, v, str(v), ha="center", va="bottom")
        fig.tight_layout()
        p = args.out_dir / "p0a_vlm_deviation.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)

    # ── 图 2：CV 中心 / 朝向误差 ────────────────────────────────────
    if cv and cv.get("synthetic", {}).get("raw"):
        raw = cv["synthetic"]["raw"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        for key, lab, col in (("d_centroid", "centroid", "#36c"),
                              ("d_rect", "minAreaRect center", "#c93"),
                              ("d_circle", "enclosing-circle center", "#3a3")):
            v = np.array(raw[key], dtype=float)
            v = v[np.isfinite(v)]
            ax.hist(v, bins=np.arange(0, 2.0, 0.05), histtype="step", lw=1.6,
                    label="%s (median %.3f px)" % (lab, np.median(v)), color=col)
        ax.axvline(3.0, color="r", ls="--", lw=1.5, label="target <=3 px")
        ax.set_xlabel("center error (px)")
        ax.set_ylabel("count")
        ax.set_title("P0-b  CV center error, synthetic GT (n=%d)"
                     % len(raw["d_centroid"]))
        ax.legend(fontsize=8)

        ax = axes[1]
        a1 = np.array(raw["ang_err_minarearect"], dtype=float)
        a2 = np.array(raw["ang_err_pca"], dtype=float)
        ax.hist(a1[np.isfinite(a1)], bins=np.arange(0, 46, 1), color="#36c",
                alpha=0.85, label="minAreaRect (median %.2f deg)" % np.nanmedian(a1))
        ax.hist(a2[np.isfinite(a2)], bins=np.arange(0, 46, 1), color="#c44",
                alpha=0.55, label="PCA (median %.1f deg)" % np.nanmedian(a2))
        ax.axvline(10.0, color="r", ls="--", lw=1.5, label="target <=10 deg")
        ax.set_xlabel("orientation error (deg, mod 90)")
        ax.set_ylabel("count")
        ax.set_title("P0-b  orientation error\n(PCA degenerate for a square)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = args.out_dir / "p0b_cv_error.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)

    # ── 图 3：IK 求解时间 ──────────────────────────────────────────
    if ik:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        for r in ik["results"]:
            t = np.array(r["raw"]["times_ms"], dtype=float)
            ax.hist(t, bins=np.logspace(-2, 1.3, 40), histtype="step", lw=1.6,
                    label="%s (med %.2f ms)" % (r["name"].split()[0], np.median(t)))
        ax.axvline(10.0, color="r", ls="--", lw=1.5, label="budget 10 ms")
        ax.set_xscale("log")
        ax.set_xlabel("IK solve time (ms, log)")
        ax.set_ylabel("count")
        ax.set_title("P0-c  placo IK solve time (RK3588)")
        ax.legend(fontsize=8)

        ax = axes[1]
        names, meds = [], []
        for r in ik["results"]:
            names.append(r["name"].split()[0])
            meds.append(r["med_ms"])
        ax.bar(names, meds, color="#36c", edgecolor="k")
        ax.axhline(10.0, color="r", ls="--", lw=1.5, label="budget 10 ms")
        ax.set_ylabel("median solve time (ms)")
        ax.set_title("P0-c  median solve time by scenario")
        for i, v in enumerate(meds):
            ax.text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(meds) * 1.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = args.out_dir / "p0c_ik_time.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)

    for p in made:
        print("已写 %s (%d bytes)" % (p, p.stat().st_size))
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
