# -*- coding: utf-8 -*-
"""感知层单元测试：立方体精定位 + VLM prompt 处理

覆盖 perception/cube_locator.py 与 perception/vlm.py 的纯逻辑部分。
不需要模型/NPU/硬件，PC 端只要 opencv + numpy 就能跑。

每个用例都对应一个**实测踩过的坑**，见 docs/p0_report.md：
  - 45° 方块被形状判据误杀（漏检 65/432 全在 30°~60°）
  - 白色机械臂被白平衡染蓝，色相与蓝色方块重合 → 必须靠形状/饱和度剔除
  - 正方形用 PCA 求朝向会退化 → 必须用 minAreaRect
  - 多行 prompt 被 demo 拆成多次提问
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.cube_locator import (  # noqa: E402
    COLOR_HUE_WINDOWS, COLOR_ZH, MEASURED_HSV_REF, RECT_FILL_MIN, classify_hue,
    colorful_mask, draw_candidates, find_cube, find_cubes,
)
from perception.vlm import flatten_prompt  # noqa: E402

W, H = 640, 480
SS = 4      # 超采样，保证光栅化不是误差来源


def _render(color_hsv, cx, cy, side, angle_deg, bg_sat=40, rng=None):
    """渲染一张低饱和背景 + 单个已知方块（RGB uint8）"""
    h, s, v = color_hsv
    big = np.zeros((H * SS, W * SS, 3), np.uint8)
    big[..., 0] = 20
    big[..., 1] = bg_sat
    big[..., 2] = 170
    bg = cv2.cvtColor(big, cv2.COLOR_HSV2RGB)
    box = cv2.boxPoints(((cx * SS, cy * SS), (side * SS, side * SS), angle_deg)).astype(np.int32)
    patch = np.zeros((H * SS, W * SS, 3), np.uint8)
    cv2.fillPoly(patch, [box], (h, s, v))
    m = patch.any(axis=2)
    cube = cv2.cvtColor(patch, cv2.COLOR_HSV2RGB)
    bg[m] = cube[m]
    img = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
    if rng is not None:
        img = np.clip(img.astype(np.float32) + rng.normal(0, 4, img.shape),
                      0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# 形状判据
# ---------------------------------------------------------------------------
def test_rotated_square_never_missed():
    """0°~85° 每 5° 都必须检出 —— 回归 45° 被误杀那个坑

    根因：用「面积/轴对齐 bbox 面积」当形状判据时，正方形旋转 θ 的该比值
    = 1/(cosθ+sinθ)²，45° 时恰好 0.5，任何 >0.5 的门限都会在 30°~60° 整批漏检。
    """
    rng = np.random.default_rng(0)
    miss = []
    for ang in range(0, 90, 5):
        img = _render((103, 253, 228), 320, 240, 39, float(ang), rng=rng)
        if not find_cubes(img, colors=["blue"]):
            miss.append(ang)
    assert not miss, "以下角度漏检: %s" % miss
    print("  PASS: 18 个角度全部检出（含 45° 附近）")


def test_rect_fill_invariant_to_rotation():
    """拟合矩形填充率与旋转无关（≈1），这是采用它而非 bbox 比值的原因"""
    fills = []
    for ang in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0):
        img = _render((103, 253, 228), 320, 240, 39, ang)
        c = find_cube(img, "blue")
        assert c is not None, ang
        fills.append(c.extent)
    assert min(fills) > RECT_FILL_MIN, fills
    assert max(fills) - min(fills) < 0.15, "填充率随旋转变化过大: %s" % fills
    print("  PASS: 拟合矩形填充率 %.3f~%.3f（与旋转无关）" % (min(fills), max(fills)))


def test_arm_like_blob_rejected():
    """淡蓝白大块（模拟被白平衡染蓝的机械臂）不得被当成方块"""
    img = _render((101, 149, 228), 320, 470, 200, 0.0)   # 又大又低饱和、且触底
    cands = find_cubes(img)
    assert not cands, "机械臂样大块被误检为方块: %s" % [c.as_dict() for c in cands]
    print("  PASS: 淡蓝白大块被剔除（面积/填充率/触底）")


def test_small_noise_not_detected():
    """零散小噪点不应产生候选"""
    rng = np.random.default_rng(1)
    img = np.full((H, W, 3), 200, np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, W), rng.integers(0, H)
        img[y:y + 3, x:x + 3] = (0, 0, 255)
    assert not find_cubes(img), "噪点被检成方块"
    print("  PASS: 小噪点不产生候选")


# ---------------------------------------------------------------------------
# 精度与朝向
# ---------------------------------------------------------------------------
def test_centroid_accuracy_synthetic():
    """已知真值下质心误差 < 1 px"""
    worst = 0.0
    for cx, cy, side in ((150.0, 120.0, 39.0), (480.0, 300.0, 28.0), (320.0, 240.0, 50.0)):
        img = _render((178, 145, 146), cx, cy, side, 25.0)
        c = find_cube(img, "red")
        assert c is not None, (cx, cy, side)
        worst = max(worst, float(np.hypot(c.center[0] - cx, c.center[1] - cy)))
    assert worst < 1.0, "质心误差 %.3f px" % worst
    print("  PASS: 合成图质心误差 max %.3f px" % worst)


def test_orientation_minarearect_beats_pca():
    """朝向必须用 minAreaRect；PCA 对正方形退化（误差应明显更大）"""
    ma, pc = [], []
    for ang in (10.0, 25.0, 40.0, 55.0, 70.0):
        img = _render((86, 250, 141), 320, 240, 45, ang)
        c = find_cube(img, "green", diagnostics=True)
        assert c is not None, ang

        def err(a):
            d = abs(a - ang) % 90.0
            return min(d, 90.0 - d)
        ma.append(err(c.angle_deg))
        pc.append(err(c.extra["angle_pca"]))
    assert max(ma) < 5.0, "minAreaRect 朝向误差过大: %s" % ma
    assert np.median(pc) > np.median(ma), "PCA 竟然不差于 minAreaRect: %s vs %s" % (pc, ma)
    print("  PASS: minAreaRect 误差 max %.2f°，PCA 中位 %.1f°（退化）"
          % (max(ma), float(np.median(pc))))


# ---------------------------------------------------------------------------
# 颜色
# ---------------------------------------------------------------------------
def test_all_six_colors_classified():
    """6 种实测色相都能正确归类"""
    for name, hsv in MEASURED_HSV_REF.items():
        img = _render(hsv, 320, 240, 39, 0.0)
        c = find_cube(img, name)
        assert c is not None, "%s 未检出" % name
        assert c.color == name, "期望 %s 得到 %s (H=%d)" % (name, c.color, c.median_hsv[0])
    print("  PASS: 6 种颜色全部正确归类")


def test_hue_windows_do_not_overlap():
    """色相窗口两两不重叠（否则 classify_hue 的先后顺序会决定结果）"""
    spans = []
    for name, wins in COLOR_HUE_WINDOWS.items():
        for lo, hi in wins:
            spans.append((lo, hi, name))
    spans.sort()
    for (lo1, hi1, n1), (lo2, hi2, n2) in zip(spans, spans[1:]):
        # red 的 (0,8) 与 (165,180) 是同一颜色的两段，允许
        assert hi1 < lo2, "窗口重叠: %s[%d,%d] 与 %s[%d,%d]" % (n1, lo1, hi1, n2, lo2, hi2)
    # 未被任何窗口覆盖的色相应返回 None（149~164 实测无主色）
    assert classify_hue(155) is None, "155 不该落到任何颜色窗口"
    assert classify_hue(103) == "blue"
    print("  PASS: 色相窗口无重叠（%d 段），未覆盖色相返回 None" % len(spans))


def test_color_zh_mapping_complete():
    for c in COLOR_HUE_WINDOWS:
        assert c in COLOR_ZH, c
    print("  PASS: 颜色中文名映射完整")


# ---------------------------------------------------------------------------
# Set-of-Mark 绘制
# ---------------------------------------------------------------------------
def test_draw_candidates_marks_all():
    """编号绘制不应改变原图，且每个候选都有标记"""
    img = _render((103, 253, 228), 320, 240, 39, 0.0)
    cands = find_cubes(img)
    assert cands
    vis = draw_candidates(img, cands, numbered=True)
    assert vis.shape == img.shape
    assert not np.array_equal(vis, img), "没画出任何东西"
    # 每个候选中心附近应出现绿色标记
    for c in cands:
        cx, cy = int(c.center[0]), int(c.center[1])
        patch = vis[max(0, cy - 5):cy + 6, max(0, cx - 5):cx + 6]
        assert (patch[..., 1] > 200).any(), "候选中心无标记"
    print("  PASS: Set-of-Mark 绘制正常（%d 个候选）" % len(cands))


def test_colorful_mask_thresholds():
    """彩色掩码：高饱和进、低饱和出"""
    rng = np.random.default_rng(2)
    img = _render((103, 253, 228), 320, 240, 39, 0.0, bg_sat=40, rng=rng)
    m = colorful_mask(img)
    assert m[240, 320] == 255, "方块中心未被判为彩色"
    assert m[10, 10] == 0, "低饱和背景被判为彩色"
    print("  PASS: 彩色掩码高低饱和区分正确")


# ---------------------------------------------------------------------------
# VLM prompt
# ---------------------------------------------------------------------------
def test_flatten_prompt_single_line():
    """多行 prompt 必须被压成单行（否则 demo 会拆成多次提问）"""
    p = '<image>\n图中红色方块在哪？\n只输出 JSON\n不要解释'
    f = flatten_prompt(p)
    assert "\n" not in f and "\r" not in f
    assert f.startswith("<image>"), f
    assert "只输出 JSON" in f
    assert f.count("<image>") == 1
    print("  PASS: 多行 prompt 压成单行: %r" % f[:60])


def test_flatten_prompt_preserves_content():
    """折叠空白不得吞掉实质内容"""
    p = "  <image>\n\n  请检测图中\"红色杯子\"的位置。\n输出格式: {\"bbox_2d\":[x1,y1,x2,y2]}\n  "
    f = flatten_prompt(p)
    for token in ("<image>", "红色杯子", "bbox_2d", "[x1,y1,x2,y2]"):
        assert token in f, token
    assert "  " not in f, "仍有连续空格"
    print("  PASS: 折叠后内容完整")


def test_vlm_prompt_templates_are_flattenable():
    """生产 prompt 模板（多行书写）必须能被安全压平，且仍含 <image>"""
    from perception.vlm import VLMPerception
    for tmpl in (VLMPerception.DETECT_PROMPT_TEMPLATE, VLMPerception.GENERAL_DETECT_PROMPT):
        txt = tmpl.format(target="红色方块") if "{target}" in tmpl else tmpl
        f = flatten_prompt(txt)
        assert f.startswith("<image>"), f[:40]
        assert "\n" not in f
    print("  PASS: 生产 prompt 模板压平后仍含 <image> 且为单行")


if __name__ == "__main__":
    print("=" * 58)
    print("Perception Tests (cube locator + VLM prompt)")
    print("=" * 58)
    tests = [
        ("45° 不失检", test_rotated_square_never_missed),
        ("填充率与旋转无关", test_rect_fill_invariant_to_rotation),
        ("机械臂样大块剔除", test_arm_like_blob_rejected),
        ("噪点不误检", test_small_noise_not_detected),
        ("质心精度", test_centroid_accuracy_synthetic),
        ("朝向 minAreaRect vs PCA", test_orientation_minarearect_beats_pca),
        ("6 色归类", test_all_six_colors_classified),
        ("色相窗口不重叠", test_hue_windows_do_not_overlap),
        ("颜色中文名映射", test_color_zh_mapping_complete),
        ("SoM 绘制", test_draw_candidates_marks_all),
        ("彩色掩码门限", test_colorful_mask_thresholds),
        ("prompt 压成单行", test_flatten_prompt_single_line),
        ("prompt 内容完整", test_flatten_prompt_preserves_content),
        ("生产模板可压平", test_vlm_prompt_templates_are_flattenable),
    ]
    passed = 0
    for name, fn in tests:
        print("\n[%s]" % name)
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print("  FAIL: %s" % e)
        except Exception as e:  # noqa: BLE001
            print("  ERROR: %s: %s" % (type(e).__name__, e))
    print("\n" + "=" * 58)
    print("Results: %d/%d passed" % (passed, len(tests)))
