#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-a：VLM 定位能力验收（判据：中位像素偏差 ≤10 px；>50 px 则 VLM 只做语义决策）

对应 docs/测试方案_VLM+外部IK+抓取.md P0-a。测两件事，一负一正：

**mode=coord（负向对照）** —— 按方案原文的 prompt 直接问坐标：
    <image>图中<颜色>方块的中心像素坐标？只输出 JSON {"cube":[x,y]}
若这条走不通，就证明「VLM 不输出坐标」的设计前提成立。真值用 P0-b 的 CV 定位
（已独立验证中心误差中位 0.705 px，远优于判据 3 px，可当真值）。
模型可能输出归一化坐标，故同时按「像素」和「归一化×宽高」两种解释算偏差，
报出更贴近的那一种 —— 这本身也是有价值的观测。

**mode=som（正向对照）** —— Set-of-Mark：CV 出候选并编号画在图上，
只问「<颜色>方块是几号」，把回归问题变成分类问题，测选择准确率。

延迟按**整次子进程调用**计（demo 每次都要重新加载 1.2 GB rkllm），
这才是真实部署会付出的代价。

用法::
    python tools/vlm_bench.py --mode both --samples 20 --json-out /tmp/p0a.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.cube_locator import COLOR_ZH, draw_candidates, find_cubes  # noqa: E402

MODEL_DIR = Path("models/vlm/Qwen3.5-0.8B")

#: 方案原文的坐标 prompt（保持原样，便于说明"按方案测也不行"）
COORD_PROMPT = ('<image>\n图中{color}方块的中心像素坐标？\n'
                '只输出 JSON {{"cube":[x,y]}}\n仅输出 JSON，不要其他文字。')

#: Set-of-Mark prompt：只让模型选编号，不做回归
SOM_PROMPT = ('<image>\n图中已用红框标出若干物体并编号，框旁标有编号。\n'
              '请问{color}方块是几号？\n只输出数字，不要其他文字。')


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------
def parse_answer(raw: str) -> str:
    """从 demo 原始输出里取模型回答（'robot:' 之后、下一个 'user:' 之前）"""
    i = raw.find("robot:")
    if i < 0:
        return raw.strip()
    after = raw[i + 6:]
    j = after.find("user:")
    return (after[:j] if j >= 0 else after).strip()


def parse_xy_variants(ans: str):
    """把模型输出解析成**所有合理的坐标解释**，返回 [{'x','y','how'}]

    公平性考虑：模型经常不听格式要求，回的是 ``{"bbox_2d":[x1,y1,x2,y2]}``。
    若只把前两个数当中心，等于故意把误差算大。这里对每种合理解释都算一遍，
    最终取"最有利解释"下的误差 —— 这样"不可用"的结论才站得住脚。
    """
    out = []
    for m in re.finditer(r"[\[\(]([^\]\)]*)[\]\)]", ans):
        nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(1))
        if len(nums) >= 4:
            x1, y1, x2, y2 = (float(v) for v in nums[:4])
            out.append(dict(x=(x1 + x2) / 2, y=(y1 + y2) / 2, how="bbox4_center"))
            out.append(dict(x=min(x1, x2), y=min(y1, y2), how="bbox4_corner"))
        if len(nums) >= 2:
            out.append(dict(x=float(nums[0]), y=float(nums[1]), how="pair_first_two"))
    # 无括号形式（如 "592,748"）
    nums = re.findall(r"-?\d+(?:\.\d+)?", ans)
    if len(nums) >= 4:
        x1, y1, x2, y2 = (float(v) for v in nums[:4])
        out.append(dict(x=(x1 + x2) / 2, y=(y1 + y2) / 2, how="bare_bbox4_center"))
    if len(nums) >= 2:
        out.append(dict(x=float(nums[0]), y=float(nums[1]), how="bare_pair"))
    return out


def parse_index(ans: str):
    """解析编号（1 起）。返回 (idx 或 None, 方式)"""
    m = re.search(r"\d+", ans)
    if not m:
        return None, "none"
    return int(m.group(0)), "first_int"


# ---------------------------------------------------------------------------
# VLM 调用（复用生产路径 perception/vlm.py）
# ---------------------------------------------------------------------------
class VlmRunner:
    """薄封装：直接复用 VLMPerception 的子进程调用，保证测的就是生产代码路径"""

    def __init__(self, model_dir: Path, timeout: float = 180.0,
                 max_new_tokens: int = 128):
        from perception.vlm import VLMPerception
        # 必须绝对路径：_run_inference 用 cwd=model_dir 起子进程，
        # 相对路径的 demo 会被解析到 model_dir 之下而找不到（踩过）
        model_dir = Path(model_dir).resolve()
        self.v = VLMPerception()
        self.v.setup(dict(model_dir=str(model_dir), timeout=timeout,
                          max_context_len=4096, max_new_tokens=max_new_tokens))
        if not self.v.is_available:
            raise RuntimeError("VLM demo 不可用：%s" % self.v._get_demo_path())

    def ask(self, image_path: Path, prompt: str):
        t0 = time.perf_counter()
        raw = self.v._run_inference(str(image_path), prompt)
        ms = (time.perf_counter() - t0) * 1000.0
        return parse_answer(raw), raw, ms


# ---------------------------------------------------------------------------
# 样本
# ---------------------------------------------------------------------------
def build_samples(data_dir: Path, n: int, colors, seed: int = 0):
    """挑 n 个 (图像, 颜色) 样本：优先选 CV 能稳定找到的方块"""
    rng = np.random.default_rng(seed)
    eps = sorted(data_dir.glob("episode_*_images"))[:12]
    pool = []
    for ep in eps:
        fr = sorted((ep / "front").glob("*.jpg"))
        if not fr:
            continue
        for i in np.linspace(0, len(fr) - 1, 8).astype(int):
            pool.append(fr[i])
    rng.shuffle(pool)

    samples, tmp = [], Path("/tmp/p0a_imgs")
    tmp.mkdir(parents=True, exist_ok=True)
    for f in pool:
        if len(samples) >= n:
            break
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cands = find_cubes(rgb, colors=colors)
        if not cands:
            continue
        # 每张图取一个候选，尽量分散颜色
        c = cands[int(rng.integers(0, len(cands)))]
        # VLM demo 不接受中文路径，统一拷成 ASCII 名
        dst = tmp / ("s%03d_%s.jpg" % (len(samples), f.name))
        cv2.imwrite(str(dst), bgr)
        samples.append(dict(image=str(dst), src=str(f), color=c.color,
                            gt_center=list(c.center), gt_bbox=list(c.bbox),
                            gt_area=c.area))
    return samples


# ---------------------------------------------------------------------------
# 两个模式
# ---------------------------------------------------------------------------
def run_coord(runner: VlmRunner, samples, w: int, h: int, verbose: bool):
    rows = []
    for k, s in enumerate(samples):
        prompt = COORD_PROMPT.format(color=COLOR_ZH[s["color"]])
        ans, raw, ms = runner.ask(Path(s["image"]), prompt)
        variants = parse_xy_variants(ans)
        gx, gy = s["gt_center"]
        best = None
        for v in variants:
            e_px = float(np.hypot(v["x"] - gx, v["y"] - gy))
            e_nm = float(np.hypot(v["x"] * w - gx, v["y"] * h - gy))
            cand = dict(v, err_px=e_px, err_norm=e_nm, err_best=min(e_px, e_nm))
            if best is None or cand["err_best"] < best["err_best"]:
                best = cand
        rows.append(dict(idx=k, color=s["color"], color_zh=COLOR_ZH[s["color"]],
                         src=Path(s["src"]).name, answer=ans[:200],
                         n_variants=len(variants),
                         best=best, err_min=best["err_best"] if best else None,
                         ms=ms, illegal=best is None))
        if verbose:
            print("  [%2d] %-6s %-28r best=%-14s err=%s (%.0f ms)"
                  % (k, COLOR_ZH[s["color"]], ans[:28],
                     best["how"] if best else "-",
                     "%.1f" % best["err_best"] if best else "None", ms))
    return rows


def run_som(runner: VlmRunner, samples, verbose: bool):
    rows = []
    for k, s in enumerate(samples):
        bgr = cv2.imread(s["image"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cands = find_cubes(rgb)
        if not cands:
            continue
        vis = draw_candidates(rgb, cands, numbered=True)
        marked = Path("/tmp/p0a_imgs") / ("som%03d.jpg" % k)
        cv2.imwrite(str(marked), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        # 目标 = CV 认为是该颜色的候选序号（1 起）
        want = [i for i, c in enumerate(cands, 1) if c.color == s["color"]]
        if not want:
            continue
        prompt = SOM_PROMPT.format(color=COLOR_ZH[s["color"]])
        ans, raw, ms = runner.ask(marked, prompt)
        idx, how = parse_index(ans)
        rows.append(dict(idx=k, color=s["color"], color_zh=COLOR_ZH[s["color"]],
                         n_candidates=len(cands),
                         candidate_colors=[c.color for c in cands],
                         want=want, got=idx, answer=ans, parse=how, ms=ms,
                         correct=bool(idx in want),
                         illegal=idx is None or not (1 <= idx <= len(cands))))
        if verbose:
            print("  [%2d] %-6s 候选%d %s  期望%s 得到%s %s (%.0f ms)"
                  % (k, COLOR_ZH[s["color"]], len(cands),
                     [c.color[:2] for c in cands], want, idx,
                     "✓" if rows[-1]["correct"] else "✗", ms))
    return rows


# ---------------------------------------------------------------------------
def _stats(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(n=0, median=None, p90=None, max=None)
    return dict(n=int(v.size), median=float(np.median(v)),
                p90=float(np.percentile(v, 90)), max=float(v.max()))


def main() -> int:
    ap = argparse.ArgumentParser(description="P0-a VLM 定位验收")
    ap.add_argument("--mode", choices=["coord", "som", "both"], default="both")
    ap.add_argument("--data", type=Path, default=Path("data/raw/pick_place"))
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--colors", default="", help="逗号分隔，如 red,blue（默认全部）")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="生成上限。生产默认 512 会让长回答拖到 20 s+；"
                         "SoM 只需一个数字，实测 16 足够")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    colors = [c for c in args.colors.split(",") if c] or None

    print("=" * 66)
    print("P0-a VLM 定位验收（Qwen3.5-0.8B @ RKNN+RKLLM）")
    print("=" * 66)
    print("模型目录:", args.model_dir)
    runner = VlmRunner(args.model_dir, timeout=args.timeout,
                       max_new_tokens=args.max_new_tokens)
    print("demo 就绪:", runner.v._get_demo_path(),
          "| max_context_len", runner.v._max_context_len,
          "| max_new_tokens", runner.v._max_new_tokens)

    samples = build_samples(args.data, args.samples, colors, args.seed)
    print("样本: %d 个（图 + 目标颜色），颜色分布 %s"
          % (len(samples), {c: sum(1 for s in samples if s["color"] == c)
                            for c in sorted({s["color"] for s in samples})}))
    if not samples:
        print("✗ 没有可用样本")
        return 1

    out = dict(samples=len(samples), model_dir=str(args.model_dir))
    checks = []

    if args.mode in ("coord", "both"):
        print("\n=== mode=coord：直接问坐标（方案原文 prompt）===")
        rows = run_coord(runner, samples, 640, 480, args.verbose)
        illegal = sum(1 for r in rows if r["illegal"])
        e = _stats([r["err_min"] for r in rows])
        how = {}
        for r in rows:
            if r["best"]:
                how[r["best"]["how"]] = how.get(r["best"]["how"], 0) + 1
        lat = _stats([r["ms"] for r in rows])
        out["coord"] = dict(rows=rows, illegal=illegal, err_best=e,
                            interpretation_used=how, latency_ms=lat)
        print("  非法/无法解析输出: %d/%d = %.1f%%" % (illegal, len(rows),
                                                      100.0 * illegal / len(rows)))
        print("  像素偏差（取对模型**最有利**的解释）: n=%d 中位 %.1f px  p90 %.1f px  max %.1f px"
              % (e["n"], e["median"], e["p90"], e["max"]))
        print("  各样本最有利解释: %s" % how)
        print("  （图像尺寸 640×480，故偏差 >640 px 即等于完全没定位到）")
        print("  单次耗时（含子进程+模型加载）: 中位 %.0f ms  max %.0f ms"
              % (lat["median"], lat["max"]))
        checks.append(("① coord 模式中位偏差 ≤10 px（方案 P0-a 判据）",
                       e["median"] is not None and e["median"] <= 10.0,
                       "%.1f px" % e["median"] if e["median"] is not None else "n/a"))
        checks.append(("② coord 模式 >50 px（是 → 只能用语义）",
                       e["median"] is not None and e["median"] > 50.0,
                       "%.1f px" % e["median"] if e["median"] is not None else "n/a"))

    if args.mode in ("som", "both"):
        print("\n=== mode=som：Set-of-Mark 选编号（正向对照）===")
        rows = run_som(runner, samples, args.verbose)
        if rows:
            acc = sum(1 for r in rows if r["correct"]) / len(rows)
            out["som"] = dict(rows=rows, accuracy=acc,
                              latency_ms=_stats([r["ms"] for r in rows]))
            print("  选择正确: %d/%d = %.1f%%" % (sum(1 for r in rows if r["correct"]),
                                                  len(rows), 100.0 * acc))
            lat = out["som"]["latency_ms"]
            print("  单次耗时: 中位 %.0f ms" % lat["median"])
            checks.append(("③ SoM 选编号准确率 ≥70%", acc >= 0.70, "%.1f%%" % (100 * acc)))
        else:
            print("  （无有效样本）")

    print("\n" + "=" * 66)
    print("结论")
    print("=" * 66)
    for label, ok, extra in checks:
        print("  %-46s %s   (%s)" % (label, "✓" if ok else "✗", extra))
    if args.mode in ("coord", "both") and out.get("coord"):
        m = out["coord"]["err_best"]["median"]
        if m is not None and m > 50.0:
            print("\n  → VLM 坐标回归不可用（中位 %.0f px，远超 50 px 门槛）：" % m)
            print("    证实方案「VLM 只出语义、定位交给 CV」的前提 ✓")
        elif m is not None and m <= 10.0:
            print("\n  → VLM 坐标输出意外可用（中位 %.1f px），可考虑直接用它做粗定位" % m)
    if args.json_out:
        args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                            default=float), encoding="utf-8")
        print("结果已写 %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
