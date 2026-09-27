"""M1 采集质量层单元测试（纯逻辑，无硬件、无 numpy 依赖）

覆盖: 帧率/丢帧/追踪误差/时长四道门控、无效帧统计、数据集汇总、报告渲染。
运行: python3 tests/test_episode_quality.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.episode_quality import (DEFAULT_THRESHOLDS, check_episode,  # noqa: E402
                                   dataset_stats, format_report, interval_stats,
                                   tracking_summary)


def make_frames(n=150, fps=30, track_err=0.0, jitter=None, drop_at=()):
    """合成 frames: J 正弦运动，F = J + track_err；drop_at 中的索引处加大时间间隔"""
    frames = []
    t = 0.0
    for i in range(n):
        if i in drop_at:
            t += 3.0 / fps          # 丢 2 帧
        ang = 30.0 * (i % 60) / 60.0 - 15.0
        fr = {f"J{k}": round(ang + k, 1) for k in range(1, 7)}
        fr["t"] = round(t, 4)
        if track_err:
            for k in range(1, 7):
                fr[f"F{k}"] = round(fr[f"J{k}"] + track_err, 1)
        elif jitter is not None:
            for k in range(1, 7):
                fr[f"F{k}"] = round(fr[f"J{k}"] + jitter, 1)
        frames.append(fr)
        t += 1.0 / fps
    return frames


def test_pass_clean_episode():
    """满帧率 + 无丢帧 + 追踪误差小 + 时长足够 → 合格"""
    frames = make_frames(150, 30, track_err=1.5)
    q = check_episode(frames, target_fps=30, requested_s=5.0)
    assert q["ok"], q["reasons"]
    assert q["metrics"]["frames"] == 150, q["metrics"]
    assert abs(q["metrics"]["fps"] - 30.0) < 0.5, q["metrics"]
    assert q["metrics"]["dropped"] == 0
    assert q["metrics"]["track_worst_mean"] == 1.5
    print("  PASS: 干净 episode 合格 + 指标正确")


def test_fps_gate():
    """帧率不足 → 不合格（150 帧铺在 7.5s 上 = 20fps）"""
    frames = make_frames(150, 30)
    for i, fr in enumerate(frames):
        fr["t"] = round(i / 20.0, 4)          # 人为放慢到 20fps
    q = check_episode(frames, target_fps=30, requested_s=None)
    assert not q["ok"] and any("帧率不足" in r for r in q["reasons"]), q
    print("  PASS: 帧率门控生效")


def test_drop_gate():
    """丢帧比例超限 → 不合格；少量丢帧仍可通过"""
    heavy = make_frames(60, 30, drop_at=tuple(range(5, 50, 5)))   # 9/59 ≈ 15%
    q = check_episode(heavy, target_fps=30)
    assert not q["ok"] and any("丢帧过多" in r for r in q["reasons"]), q
    mild = make_frames(300, 30, drop_at=(100,))
    q2 = check_episode(mild, target_fps=30)
    assert q2["ok"], q2["reasons"]
    assert q2["metrics"]["dropped"] == 1, q2["metrics"]
    print("  PASS: 丢帧门控（15% 拒绝 / 1 帧放行）")


def test_track_gate():
    """追踪误差超限 → 不合格；夹爪(J6)单独偏大不影响体关节门控"""
    bad = make_frames(150, 30, track_err=8.0)
    q = check_episode(bad, target_fps=30)
    assert not q["ok"] and any("追踪误差超限" in r for r in q["reasons"]), q

    frames = make_frames(150, 30, track_err=1.0)
    for fr in frames:                       # 只让夹爪偏大
        fr["F6"] = fr["J6"] + 25.0
    q2 = check_episode(frames, target_fps=30)
    assert q2["ok"], q2["reasons"]
    tr = tracking_summary(frames)
    assert tr["worst_joint"] in (1, 2, 3, 4, 5), tr
    assert tr["per_joint_mean"][6] == 25.0, tr["per_joint_mean"]
    print("  PASS: 追踪门控只看体关节，夹爪单独报告")


def test_duration_gate():
    """时长不足 → 不合格（请求 20s 只录 5s）"""
    frames = make_frames(150, 30)
    q = check_episode(frames, target_fps=30, requested_s=20.0)
    assert not q["ok"] and any("时长不足" in r for r in q["reasons"]), q
    print("  PASS: 时长门控生效")


def test_invalid_frames():
    """缺关节数据的帧计入无效帧并拒绝"""
    frames = make_frames(150, 30)
    for fr in frames[:5]:
        fr.pop("J3")
    q = check_episode(frames, target_fps=30, requested_s=5.0)
    assert q["metrics"]["invalid_frames"] == 5, q["metrics"]
    assert not q["ok"] and any("无效帧" in r for r in q["reasons"]), q
    print("  PASS: 无效帧统计与拒绝")


def test_interval_stats():
    """帧间隔统计（中位/最大/丢帧计数）"""
    frames = make_frames(30, 30)
    iv = interval_stats(frames, 30)
    # 合成数据 t 保留 4 位小数（真实 JSON 为 3 位小数）→ 容差取 1e-3
    assert abs(iv["median_dt"] - 1 / 30) < 1e-3, iv
    assert iv["dropped"] == 0
    assert len(frames) == 30
    print("  PASS: 帧间隔统计")


def test_dataset_stats_and_report():
    """数据集汇总：合格/不合格计数、逐关节分布、报告渲染"""
    ok = make_frames(150, 30, track_err=1.0)
    bad = make_frames(60, 30, track_err=9.0)
    eps = [
        {"file": "episode_0001.json", "frames": ok,
         "quality": check_episode(ok, 30, 5.0)},
        {"file": "episode_0002.json", "frames": bad,
         "quality": check_episode(bad, 30, 5.0)},
    ]
    st = dataset_stats(eps)
    assert st["episodes"] == 2 and st["episodes_ok"] == 1 and st["episodes_fail"] == 1, st
    assert st["total_frames"] == 210, st
    assert len(st["per_joint_deg"]) == 6, st
    for name, s in st["per_joint_deg"].items():
        assert s["span"] >= 0 and s["min"] <= s["max"], (name, s)
    assert st["track_worst_mean"] is not None
    rep = format_report(st)
    assert "episode 数: 2" in rep and "shoulder_pan" in rep, rep
    print("  PASS: 数据集汇总 + 报告渲染")


def test_threshold_override():
    """阈值可覆盖；关闭追踪门控后误差大也合格"""
    frames = make_frames(150, 30, track_err=20.0)
    q = check_episode(frames, 30, 5.0)
    assert not q["ok"]
    q2 = check_episode(frames, 30, 5.0,
                       thresholds={**DEFAULT_THRESHOLDS, "max_track_err_deg": 30.0})
    assert q2["ok"], q2["reasons"]
    print("  PASS: 阈值覆盖生效")


def test_image_check():
    """相机帧完整性检查: 少量缺失放行、缺失超阈值拒绝、多相机独立判定"""
    from tools.episode_quality import check_images
    ok = check_images({"front": 300}, 300)
    assert ok["ok"] and ok["detail"]["front"]["missing"] == 0, ok
    # 1/300 缺失 = 0.33% < 0.5% 放行
    assert check_images({"front": 299}, 300)["ok"]
    # 3/300 缺失 = 1% > 0.5% 拒绝
    bad = check_images({"front": 297}, 300)
    assert not bad["ok"] and any("缺帧" in r for r in bad["reasons"]), bad
    # 多相机: 一个合格一个不合格 → 整体不合格
    mixed = check_images({"front": 300, "wrist": 200}, 300)
    assert not mixed["ok"] and mixed["detail"]["front"]["missing"] == 0, mixed
    # 无相机（空字典）→ 合格（未启用相机时不应误判）
    assert check_images({}, 300)["ok"]
    print("  PASS: 相机帧完整性检查")


if __name__ == "__main__":
    print("=" * 52)
    print("M1 Episode Quality Tests")
    print("=" * 52)
    tests = [
        ("clean episode pass", test_pass_clean_episode),
        ("fps gate", test_fps_gate),
        ("drop gate", test_drop_gate),
        ("track gate", test_track_gate),
        ("duration gate", test_duration_gate),
        ("invalid frames", test_invalid_frames),
        ("interval stats", test_interval_stats),
        ("dataset stats + report", test_dataset_stats_and_report),
        ("threshold override", test_threshold_override),
        ("image completeness check", test_image_check),
    ]
    passed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
    print(f"\n{'=' * 52}")
    print(f"Results: {passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
