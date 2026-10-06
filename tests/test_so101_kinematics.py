# -*- coding: utf-8 -*-
"""SO-101 运动学（placo 路线）单元测试

覆盖 hardware/so101_kinematics.py。设计目标：**PC 端无 placo 也能跑绝大部分**
（标定换算 + URDF 生成是纯文本/纯数值逻辑），placo 相关用例在缺库时打印 SKIP。

注意：**用例不写死具体限位数值**。``config/calibration.json`` 是每次实机标定的产物
（板端/PC 部署目录都不是 git 仓库，该文件由标定工具运行时写入），换一台臂或重新标定
就会变。因此这里验证的是**换算约定与结构不变式**，而非某一组标定数字：

  - 换算公式: deg = (raw - zero) * 360 / (4095)，zero=行程中点（夹爪用 homing_offset）
  - q 布局: q[0:7] 浮动基座, q[7:12] = 6 关节（板端实测）
  - TCP frame 必须叫 ``gripper_frame_link``（``gripper`` 是关节名，用错会拿到错位姿）
  - 生成的 URDF 恰好 6 个 ``<limit>``，不得污染 ``<transmission>``
"""
import json
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hardware.so101_kinematics import (  # noqa: E402
    DEFAULT_CALIB, DEFAULT_URDF_SRC, JOINT_NAMES, TCP_FRAME,
    build_calibrated_urdf, joint_limits_rad,
)

RESOLUTION = 4095.0


def _calib() -> dict:
    with open(DEFAULT_CALIB, encoding="utf-8") as f:
        return json.load(f)


def _independent_limits(entry: dict, motor_id: int):
    """独立实现一遍换算公式（不复用被测代码），作为交叉验证基准"""
    lo_raw, hi_raw = float(entry["range_min"]), float(entry["range_max"])
    if motor_id == 6:
        zero = float(entry["homing_offset"])
    else:
        zero = (lo_raw + hi_raw) / 2.0
    a = np.deg2rad((lo_raw - zero) * 360.0 / RESOLUTION)
    b = np.deg2rad((hi_raw - zero) * 360.0 / RESOLUTION)
    return (a, b) if a <= b else (b, a)


def _placo_available() -> bool:
    try:
        import placo  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 纯逻辑：标定换算
# ---------------------------------------------------------------------------
def test_joint_limits_conversion_formula():
    """限位换算与独立实现的公式逐项一致（防公式漂移，不依赖具体标定数值）"""
    cal = _calib()
    lo, hi, detail = joint_limits_rad(DEFAULT_CALIB)
    assert lo.shape == (6,) and hi.shape == (6,)
    assert len(detail) == 6
    assert [d["joint"] for d in detail] == list(JOINT_NAMES)

    worst = 0.0
    for i in range(6):
        e_lo, e_hi = _independent_limits(cal[str(i + 1)], i + 1)
        worst = max(worst, abs(lo[i] - e_lo), abs(hi[i] - e_hi))
    assert worst < 1e-12, "换算与独立公式不符，最大偏差 %.3e rad" % worst
    print("  PASS: 6 关节换算与独立公式一致（最大偏差 %.2e rad）" % worst)
    print("        当前标定: " + " / ".join(
        "J%d %+.2f°~%+.2f°" % (i + 1, np.degrees(lo[i]), np.degrees(hi[i]))
        for i in range(6)))


def test_joint_limits_sanity():
    """结构不变式：lo<hi、行程处于物理合理区间、raw 顺序与 rad 顺序一致"""
    cal = _calib()
    lo, hi, detail = joint_limits_rad(DEFAULT_CALIB)
    for i in range(6):
        e = cal[str(i + 1)]
        assert lo[i] < hi[i], "J%d lo>=hi" % (i + 1)
        span_deg = float(np.degrees(hi[i] - lo[i]))
        assert 20.0 < span_deg <= 360.0 + 1e-6, \
            "J%d 行程 %.1f° 不合理" % (i + 1, span_deg)
        # raw_min<raw_max 时 rad 也必须 lo<hi（同号同序）
        assert float(e["range_min"]) < float(e["range_max"]), "J%d raw 反序" % (i + 1)
        # 行程应与 raw 行程成比例
        expect = (float(e["range_max"]) - float(e["range_min"])) * 360.0 / RESOLUTION
        assert abs(span_deg - expect) < 1e-6, "J%d 行程与 raw 不成比例" % (i + 1)
    print("  PASS: 6 关节 lo<hi、行程 %s° 均在合理区间"
          % "/".join("%.0f" % np.degrees(hi[i] - lo[i]) for i in range(6)))


def test_joint_limits_gripper_uses_homing_offset():
    """夹爪零点是 homing_offset，不是行程中点——两种约定在数值上必须可区分

    若某次标定恰好让 homing_offset ≈ 行程中点，该用例前提失效，直接跳过（不算失败）。
    """
    cal = _calib()
    e = cal["6"]
    mid = (float(e["range_min"]) + float(e["range_max"])) / 2.0
    gap_deg = abs(float(e["homing_offset"]) - mid) * 360.0 / RESOLUTION
    lo, hi, _ = joint_limits_rad(DEFAULT_CALIB)

    if gap_deg <= 1.0:
        print("  SKIP: 本次标定 homing_offset 与行程中点仅差 %.2f°，两种约定不可区分" % gap_deg)
        return

    # 误用行程中点时会得到什么
    wrong = _independent_limits({"range_min": e["range_min"], "range_max": e["range_max"],
                                 "homing_offset": mid}, 6)
    assert abs(np.degrees(wrong[0] - lo[5])) > 0.5, "夹爪似乎用了行程中点零点"
    assert abs(np.degrees(wrong[1] - hi[5])) > 0.5, "夹爪似乎用了行程中点零点"
    print("  PASS: 夹爪零点用 homing_offset（%.2f° ~ %.2f°，与中点约定差 %.1f°）"
          % (np.degrees(lo[5]), np.degrees(hi[5]), gap_deg))


# ---------------------------------------------------------------------------
# 纯逻辑：URDF 生成
# ---------------------------------------------------------------------------
def _build_to_temp(**kw):
    tmp = tempfile.mkdtemp(prefix="so101_urdf_")
    dst = os.path.join(tmp, "so101_calibrated.urdf")
    rep = build_calibrated_urdf(dst=dst, **kw)
    with open(dst, encoding="utf-8") as f:
        return rep, f.read()


def test_urdf_injects_exactly_six_limits():
    """只替换 6 个顶层 revolute 关节的 <limit>，不得污染 <transmission> 内的 <joint>

    官方 URDF 里每个关节名出现 2 次（1 个顶层 <joint> + 1 个 <transmission> 子元素），
    早期正则因此注入 12 次并把 <limit> 塞进了 transmission。
    """
    rep, txt = _build_to_temp(strip_meshes=False)
    assert rep["replaced"] == 6, "注入次数 %d != 6" % rep["replaced"]
    assert len(re.findall(r"<limit\b", txt)) == 6, \
        "<limit> 总数 %d != 6（transmission 被污染）" % len(re.findall(r"<limit\b", txt))
    # transmission 块内不得有 <limit>
    for m in re.finditer(r"<transmission\b.*?</transmission>", txt, re.S):
        assert "<limit" not in m.group(0), "transmission 内出现 <limit>"
    print("  PASS: 恰好 6 个 <limit>，transmission 未被污染")


def test_urdf_limit_values_match_calibration():
    """生成的 URDF 里每个关节的 limit 数值 == 标定换算值"""
    _, txt = _build_to_temp(strip_meshes=False)
    lo, hi, _ = joint_limits_rad(DEFAULT_CALIB)
    for i, jname in enumerate(JOINT_NAMES):
        pat = re.compile(
            r'<joint\s+(?=[^>]*\bname="%s")(?=[^>]*\btype=")[^>]*>(.*?)</joint>'
            % re.escape(jname), re.S)
        ms = pat.findall(txt)
        assert len(ms) == 1, "%s 顶层 joint 匹配到 %d 个" % (jname, len(ms))
        lm = re.search(r'<limit[^>]*lower="([-\d.eE]+)"[^>]*upper="([-\d.eE]+)"', ms[0])
        assert lm, "%s 缺 <limit>" % jname
        assert abs(float(lm.group(1)) - lo[i]) < 1e-6, jname
        assert abs(float(lm.group(2)) - hi[i]) < 1e-6, jname
    print("  PASS: 6 个关节的 limit 数值与标定一致（1e-6）")


def test_urdf_strips_meshes_when_requested():
    """剥离 <visual>/<collision>/<mesh>：placo 缺 STL 会直接报错，运动学不需要网格"""
    _, txt = _build_to_temp(strip_meshes=True)
    for tag in ("<visual", "<collision", "<mesh"):
        assert tag not in txt, "仍残留 %s" % tag
    assert "<inertial>" in txt, "inertial 不应被剥离（pinocchio 需要质量）"
    print("  PASS: mesh/visual/collision 已剥离，inertial 保留")


def test_urdf_preserves_kinematic_chain():
    """剥离网格不得破坏运动链：7 个 joint + 8 个 link 全部保留"""
    _, txt = _build_to_temp(strip_meshes=True)
    joints = re.findall(r'<joint\s+name="([^"]+)"\s+type=', txt)
    links = re.findall(r'<link\s+name="([^"]+)"', txt)
    assert len(joints) == 7, joints
    assert len(links) == 8, links
    for j in JOINT_NAMES + ("gripper_frame_joint",):
        assert j in joints, "缺关节 %s" % j
    assert "gripper_frame_link" in links
    print("  PASS: 7 关节 / 8 连杆完整，TCP link 存在")


def test_tcp_frame_is_link_not_joint():
    """TCP 必须是 frame（gripper_frame_link），不能退回关节名 gripper"""
    assert TCP_FRAME == "gripper_frame_link"
    _, txt = _build_to_temp(strip_meshes=True)
    assert '<link name="%s"' % TCP_FRAME in txt, "TCP frame 不是 URDF 里的 link"
    assert '<joint name="%s"' % TCP_FRAME not in txt, "TCP frame 不应该是关节名"
    print("  PASS: TCP = gripper_frame_link（link，非关节）")


# ---------------------------------------------------------------------------
# placo 相关（缺库则 SKIP）
# ---------------------------------------------------------------------------
def test_placo_joint_layout_and_limits():
    """q 布局 q[7:12] 为 6 关节，且模型限位 == 标定限位"""
    if not _placo_available():
        print("  SKIP: 未安装 placo")
        return
    from hardware.so101_kinematics import So101Kinematics
    kin = So101Kinematics()
    assert kin.joint_idx.tolist() == [7, 8, 9, 10, 11, 12], kin.joint_idx.tolist()
    lo_m = np.asarray(kin.model.lowerPositionLimit).ravel()[kin.joint_idx]
    hi_m = np.asarray(kin.model.upperPositionLimit).ravel()[kin.joint_idx]
    assert np.abs(lo_m - kin.q_lo).max() < 1e-5
    assert np.abs(hi_m - kin.q_hi).max() < 1e-5
    print("  PASS: q[7:12] 为 6 关节，模型限位与标定一致")


def test_placo_fk_ik_roundtrip():
    """FK → IK 往返：位置 < 0.2 mm，且解在标定限位内"""
    if not _placo_available():
        print("  SKIP: 未安装 placo")
        return
    from hardware.so101_kinematics import So101Kinematics
    kin = So101Kinematics()
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(50):
        q = rng.uniform(kin.q_lo, kin.q_hi)
        T = kin.fk(q)
        r = kin.ik(T, q_seed=kin.clamp(q + np.deg2rad(rng.uniform(-20, 20, 6))))
        assert r.in_limits, "解越限 %.3e rad" % r.max_limit_violation
        worst = max(worst, r.pos_err)
    assert worst < 2e-4, "最大位置误差 %.3e m" % worst
    print("  PASS: 50 组 FK→IK 往返，最大位置误差 %.2e m" % worst)


def test_placo_fk_convention_matches_dataset():
    """关节零点约定：真实数据正解后桌面应在 z≈0、臂展不超过物理臂长

    该用例是「URDF 零点 == 数据集角度零点」的回归防线。约定一旦漂移，
    相机→IK→抓取整条链会系统性偏掉，且很难在实机上定位。
    """
    if not _placo_available():
        print("  SKIP: 未安装 placo")
        return
    import glob
    import json
    from hardware.so101_kinematics import So101Kinematics
    files = sorted(glob.glob(os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "pick_place", "episode_*.json")))[:5]
    if not files:
        print("  SKIP: 无采集数据 data/raw/pick_place/episode_*.json")
        return
    kin = So101Kinematics()
    zs, reach = [], []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            frames = json.load(fh).get("frames", [])
        for fr in frames:
            p = kin.fk_pos(np.deg2rad([fr["F%d" % i] for i in range(1, 7)]))
            zs.append(p[2]); reach.append(float(np.linalg.norm(p)))
    zs = np.array(zs); reach = np.array(reach)
    assert zs.min() > -0.03, "末端穿到桌面下方 %.4f m（零点约定漂移）" % zs.min()
    assert reach.max() < 0.45, "臂展 %.4f m 超出物理臂长" % reach.max()
    assert np.percentile(zs, 5) < 0.05, "5%% 分位高度 %.4f，桌面不在 z≈0" % np.percentile(zs, 5)
    print("  PASS: 约定一致（z_min %+.4f m, 臂展 max %.4f m, n=%d）"
          % (zs.min(), reach.max(), zs.size))


if __name__ == "__main__":
    print("=" * 58)
    print("SO-101 Kinematics (placo) Tests")
    print("=" * 58)
    tests = [
        ("标定限位换算公式", test_joint_limits_conversion_formula),
        ("标定限位结构不变式", test_joint_limits_sanity),
        ("夹爪零点=homing_offset", test_joint_limits_gripper_uses_homing_offset),
        ("URDF 恰好 6 个 limit", test_urdf_injects_exactly_six_limits),
        ("URDF limit 数值", test_urdf_limit_values_match_calibration),
        ("URDF 剥离 mesh", test_urdf_strips_meshes_when_requested),
        ("URDF 运动链完整", test_urdf_preserves_kinematic_chain),
        ("TCP = link 非关节", test_tcp_frame_is_link_not_joint),
        ("placo q 布局/限位", test_placo_joint_layout_and_limits),
        ("placo FK/IK 往返", test_placo_fk_ik_roundtrip),
        ("placo 约定 vs 数据集", test_placo_fk_convention_matches_dataset),
    ]
    passed = skipped = 0
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
