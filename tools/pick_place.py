#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 闭环抓放：VLM 选目标 → CV 精定位 → 手眼映射 → 静差补偿 → IK → 抓 → 抬 → 放

对应 docs/测试方案_VLM+外部IK+抓取.md 的 P2。分段判据（§5 协议）：
  Reached（末端到达抓取点 ±1 cm）/ Grasped（夹起并保持）/ Placed（放入目标区且松开）
三段分别统计，**20 试次**，条件交错执行。

复用 P0/P1 的成果（不重复造轮子）
--------------------------------
本工具直接 import `tools/handeye_calib` 的组件：`DownIK`（位置+工具朝下+弱偏航）、
`Cam`、`goto`/`goto_lift`（原地转正 + 分段抬升）、`apply_h`/`apply_drift`（单应 + 静差补偿）、
`stable_marker`（H⁻¹ 预测跟踪）。`perception/cube_locator` 提供方块定位。

P2 比 P1 多出来的两个误差源（都要先量化再跑协议）
----------------------------------------------
1. **视差**：单应是在「随动标记所在平面」标定的（P1 时 TCP z≈0.07），
   而待抓的方块躺在桌面（更低）——两个平面不同，同一像素对应的基座 XY 不同。
2. **z 变化**：静差补偿模型 d(x,y) 绑定「固定 z + 固定姿态」；抓取必须下降，
   重力矩随之改变 → 静差也变。故本工具用 `--grasp` 单独标定抓取高度处的模型，
   并用 `--approach-only` 先把"下降到方块上方"这一步的**总误差**量出来，
   确认在夹爪容差内之后再跑完整协议。

用法::
    python tools/pick_place.py --scene                       # 只读：看画面里有哪些方块
    python tools/pick_place.py --approach-only --confirm-motion   # 只接近不抓（量误差）
    python tools/pick_place.py --single --confirm-motion     # 单次抓放（验证机构）
    python tools/pick_place.py --trials 20 --confirm-motion  # 正式协议
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from perception.cube_locator import COLOR_ZH, find_cubes  # noqa: E402
from tools.handeye_calib import (  # noqa: E402
    Cam, DownIK, apply_drift, apply_h, goto, goto_lift, make_dik,
    stable_marker, _xref_from,
)

REPO = Path(__file__).resolve().parent.parent
HOMOGRAPHY = REPO / "config" / "homography.json"

#: 方块高度（米）。3D 打印方块实测边长 ~2 cm（P0-b 图像测得 ~39 px）
CUBE_H_M = 0.020
#: 夹取时 TCP 的 z（米）。TCP=夹爪中心（gripper_frame_link，即两指之间的抓取点），
#: 要让两指跨在方块中部 → TCP z ≈ 方块半高。**必须实测确认**（--approach-only）。
GRASP_Z = 0.012
#: 抬升/搬运高度
LIFT_Z = 0.10
Z_SAFE = 0.095
#: 默认投放区（基座系，米）——放在标定区域之外，避免与抓取区混在一起
DROP_XY = (0.245, 0.085)
#: 夹爪判"夹住"的负载阈值（%）
GRASP_LOAD = 8.0


# ---------------------------------------------------------------------------
class Args:
    """给 handeye_calib 里的 _connect 等复用的最小参数容器"""
    def __init__(self, port, calib, i_term=None, release=False):
        self.port = port
        self.calib = calib
        self.i_term = i_term
        self.release = release


def load_calib(path: Path, kin) -> dict:
    if not path.exists():
        raise SystemExit("✗ 没有 %s，先跑 tools/handeye_calib.py --calibrate" % path)
    cal = json.loads(path.read_text(encoding="utf-8"))
    if cal.get("calib_fingerprint") != kin.calib_fingerprint:
        raise SystemExit("✗ 标定指纹不符（标定 %s / 当前 %s）→ 标定已失效，请重标"
                         % (cal.get("calib_fingerprint"), kin.calib_fingerprint))
    return cal


def grasp_z_for_calib(cal: dict) -> float:
    """抓取高度处该用哪个 z 的标定：取标定 z 与 GRASP_Z 里最接近的那个模型"""
    return float(cal.get("z_m", 0.07))


# ---------------------------------------------------------------------------
def detect_cubes(cam: Cam, colors=None):
    rgb = cam.grab()
    return rgb, find_cubes(rgb, colors=colors)


def pick_target(cands, mode: str, rng, drop_xy):
    """选择抓取目标。mode: nearest=离投放区最远(先清远的) / random / first"""
    if not cands:
        return None
    if mode == "random":
        return cands[int(rng.integers(0, len(cands)))]
    if mode == "nearest":
        return min(cands, key=lambda c: np.hypot(c.center[0] - 320, c.center[1] - 240))
    # far: 选离投放区最近图像位置的方块（图像里离底部越远越靠外）
    return cands[0]


def cube_xy_from_pixel(H, px, z_cube: float, z_plane: float, parallax=None):
    """像素 → 基座 XY，并做**视差**修正

    单应 H 是在 z_plane 平面上标定的；方块在 z_cube 平面上。视差表现为以"相机主光轴
    与像面的交点"为中心的**径向缩放**：平面越高，射线在该平面上的交点离主点越远。
    故修正为 p_table = p_plane - k·(z_plane - z_cube)·(p_plane - c)，
    其中 c 为主点像素、k 为视差系数（由 --measure-parallax 实测得到，默认 0=不修正）。
    """
    xy = apply_h(H, np.asarray(px, float).reshape(1, 2))[0]
    if not parallax:
        return xy
    cx, cy = parallax["principal_px"]
    k = parallax["k_per_m"]          # 每米高度差引起的相对缩放
    s = k * (z_plane - z_cube)
    dx = xy[0] - parallax["origin_xy"][0]
    dy = xy[1] - parallax["origin_xy"][1]
    return np.array([parallax["origin_xy"][0] + dx * (1.0 - s),
                     parallax["origin_xy"][1] + dy * (1.0 - s)])


# ---------------------------------------------------------------------------
# 夹爪
# ---------------------------------------------------------------------------
def gripper_load(arm):
    """读夹爪负载（%）与电流（mA）"""
    try:
        d = arm.bus.read_diagnostics([6]).get(6, {})
        ld = d.get("load")
        return (ld[0] if ld else 0.0), (d.get("current_mA") or 0.0)
    except Exception:
        return 0.0, 0.0


def gripper_open_counts(arm):
    """夹爪当前开度（相对闭合限位的编码器计数）。**判「夹到东西」最可靠的量**：
    两指之间没有东西时，闭到底会到 range_min 附近；夹住 2 cm 方块时两指被撑开，
    编码器会停在一个明显更大的值上 —— 与米制宽度映射是否准确无关。"""
    try:
        cal = arm.calibration["6"]
        raw = arm._rad_to_raw(6, arm.read_positions()[5])
        return int(raw) - int(cal["range_min"])
    except Exception:
        return -1


def close_until_grasp(arm, contact_load=GRASP_LOAD, w_start=0.055, w_min=0.003,
                      step=0.004, settle=0.55, backoff=0.003,
                      min_open_counts=90):
    """逐步闭合直到夹到东西，随后**回退一点**避免持续堵转。

    判据用**两个独立量的与**：
      ① 负载上升（有阻力）
      ② 编码器开度仍明显大于闭合限位（两指被撑开）——这一条能排除
         「两指之间没东西、闭到底把负载顶上去」的假抓取
    必需回退：实测不回退会让夹爪在 21.6% 负载 / 143 mA 下堵转约 3 分钟，
    温度 37→45℃，舵机锁死 Overload 保护态（寻址 ping 不应答，只能断电恢复）。

    Returns: (grasped, load_pct, w_used, open_counts)
    """
    w = w_start
    while w >= w_min:
        arm.gripper_width(w)
        time.sleep(settle)
        load, _ = gripper_load(arm)
        oc = gripper_open_counts(arm)
        if load >= contact_load and oc >= min_open_counts:
            if backoff > 0:
                raw = arm._rad_to_raw(6, arm.read_positions()[5])
                arm.bus.write("Goal_Position", 6, min(2919, raw + 55))
                time.sleep(0.4)
            return True, gripper_load(arm)[0], w, gripper_open_counts(arm)
        w -= step
    return False, gripper_load(arm)[0], w, gripper_open_counts(arm)


def open_gripper(arm, w=0.055, settle=0.8):
    arm.gripper_width(w)
    time.sleep(settle)


# ---------------------------------------------------------------------------
def run_trial(kin, dik, arm, cam, cal, args, rng, idx):
    """一次完整抓放。返回分段结果 dict（Reached / Grasped / Placed）"""
    H = np.asarray(cal["H"], float)
    drift = cal.get("drift_model")
    z_plane = float(cal["z_m"])
    ref_area = None
    for p in cal.get("points", []):
        mk = p.get("marker") or {}
        if mk.get("area"):
            ref_area = float(mk["area"])
            break

    rgb, cands = detect_cubes(cam, args.colors)
    if not cands:
        return dict(idx=idx, ok=False, reason="no_cube")
    # 只抓「映射后落标定区域内」的方块。区域外单应与静差模型都在**外推**：
    # 实测紫色方块在区域边界外(y=-0.163，区域下界 -0.16) → Reached 偏 17.1 mm 抓空；
    # 而区域内的蓝色方块只偏 4.3 mm。所以宁可不抓，也不要拿外推结果去撞。
    x0, x1, y0, y1 = cal["region"]
    m = args.in_region_margin
    inside, outside = [], []
    for c in cands:
        xy = apply_h(H, np.asarray(c.center, float).reshape(1, 2))[0]
        if (x0 + m) <= xy[0] <= (x1 - m) and (y0 + m) <= xy[1] <= (y1 - m):
            inside.append((c, xy))
        else:
            outside.append((c, xy))
    if outside and args.verbose:
        print("  区域外（跳过）: " + ", ".join(
            "%s(%+.3f,%+.3f)" % (COLOR_ZH.get(c.color, "?"), xy[0], xy[1])
            for c, xy in outside[:6]))
    if not inside:
        return dict(idx=idx, ok=False, reason="no_cube_in_region",
                    outside=[(c.color, list(map(float, xy))) for c, xy in outside])
    tgt = pick_target([c for c, _ in inside], args.target, rng, args.drop)
    x_plane = next(xy for c, xy in inside if c is tgt)
    x_cube = cube_xy_from_pixel(H, tgt.center, CUBE_H_M / 2, z_plane, args.parallax)
    ddx, ddy = apply_drift(drift, x_cube[0], x_cube[1]) if drift else (0.0, 0.0)
    cmd_xy = (x_cube[0] - ddx, x_cube[1] - ddy)
    z_grasp = args.grasp_z if args.grasp_z is not None else GRASP_Z

    rec = dict(idx=idx, color=tgt.color, pix=list(map(float, tgt.center)),
               xy_plane=list(map(float, x_plane)), xy_cube=list(map(float, x_cube)),
               cmd=list(map(float, cmd_xy)), z_grasp=float(z_grasp),
               drift_mm=[ddx * 1000, ddy * 1000], reached=False, grasped=False,
               placed=False, reason="")

    # 1) 下降到抓取高度
    ok, why = goto_lift(dik, arm, (cmd_xy[0], cmd_xy[1], z_grasp), z_safe=Z_SAFE)
    if not ok:
        rec["reason"] = "move_fail: %s" % why
        return rec
    time.sleep(0.4)

    # 2) 量 Reached：标记像素 ↔ 移动前方块像素（同一单应）
    px_pred = apply_h(np.linalg.inv(H), np.array(cmd_xy).reshape(1, 2))[0]
    mk, spread, info = stable_marker(cam, cal.get("marker_kind", "panel"),
                                     cal.get("marker_color"), expect_px=px_pred,
                                     ref_area=ref_area, max_jump_px=args.track_gate_px)
    if mk is None:
        rec["reason"] = "marker_lost_after_move"
        return rec
    tip = apply_h(H, mk.reshape(1, 2))[0]
    cub = apply_h(H, np.asarray(tgt.center, float).reshape(1, 2))[0]
    rec["reach_mm"] = float(np.hypot(tip[0] - cub[0], tip[1] - cub[1]) * 1000)
    rec["reached"] = rec["reach_mm"] <= args.reach_tol_mm

    # 3) 闭合夹爪抓取（判据 = 负载上升 **且** 编码器开度被撑开，排除"闭到底"假抓取）
    grasped, load, w_used, oc = close_until_grasp(arm, args.grasp_load)
    rec["close_w"] = float(w_used)
    rec["close_load"] = float(load)
    rec["close_open_counts"] = int(oc)

    # 4) 抬起，再读负载：夹住的话负载会保持；掉出去则回落到 ~0
    ok2, why2 = goto_lift(dik, arm, (cmd_xy[0], cmd_xy[1], LIFT_Z), z_safe=Z_SAFE)
    if not ok2:
        rec["reason"] = "lift_fail: %s" % why2
        open_gripper(arm)
        return rec
    time.sleep(0.5)
    load_after, _ = gripper_load(arm)
    rec["load_after_lift"] = float(load_after)
    rec["grasped"] = bool(grasped and load_after >= args.hold_load)

    # 5) 搬到投放区
    if rec["grasped"]:
        ok3, why3 = goto_lift(dik, arm, (args.drop[0], args.drop[1], LIFT_Z), z_safe=Z_SAFE)
        if not ok3:
            rec["reason"] = "carry_fail: %s" % why3
            open_gripper(arm)
            return rec
        ok4, why4 = goto_lift(dik, arm, (args.drop[0], args.drop[1], args.drop_z), z_safe=Z_SAFE)
        if not ok4:
            rec["reason"] = "drop_fail: %s" % why4
            open_gripper(arm)
            return rec
        open_gripper(arm)
        time.sleep(0.4)
        goto_lift(dik, arm, (args.drop[0], args.drop[1], LIFT_Z), z_safe=Z_SAFE)
        # 6) 判 Placed：投放区附近出现该颜色的方块
        rgb2, cands2 = detect_cubes(cam, args.colors)
        best = None
        for c in cands2:
            if c.color != tgt.color:
                continue
            xy = apply_h(H, np.asarray(c.center, float).reshape(1, 2))[0]
            d = float(np.hypot(xy[0] - args.drop[0], xy[1] - args.drop[1]))
            if best is None or d < best[0]:
                best = (d, c)
        if best is None:
            rec["reason"] = "cube_not_found_after_drop"
        else:
            rec["place_mm"] = best[0] * 1000
            rec["placed"] = best[0] <= args.place_tol_mm
            if not rec["placed"]:
                rec["reason"] = "placed_too_far(%.0fmm)" % (best[0] * 1000)
    else:
        rec["reason"] = rec["reason"] or "not_grasped(load %.1f%%)" % load_after
        open_gripper(arm)
    return rec


# ---------------------------------------------------------------------------
def cmd_scene(kin, args) -> int:
    cam = Cam()
    try:
        rgb, cands = detect_cubes(cam)
        print("画面里检测到 %d 个方块:" % len(cands))
        for c in cands:
            print("  %-6s 像素 (%3.0f,%3.0f)  面积 %4d  边长 %.1f px  朝向 %.1f°"
                  % (c.color_zh, c.center[0], c.center[1], c.area, c.side_px, c.angle_deg))
        if args.snap:
            from perception.cube_locator import draw_candidates
            args.snap.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.snap), cv2.cvtColor(draw_candidates(rgb, cands),
                                                     cv2.COLOR_RGB2BGR))
            print("标注图 → %s" % args.snap)
        return 0 if cands else 2
    finally:
        cam.close()


def _setup(kin, args):
    """连从臂 + 开相机 + 建 DownIK（calibrate/approach/single/trials 共用）"""
    cal = load_calib(args.homography, kin)
    from tools.handeye_calib import _connect
    arm, T_now = _connect(kin, Args(args.port, args.calib, args.i_term, args.release))
    dik = make_dik(kin, _xref_from(T_now))
    cam = Cam()
    return cal, arm, dik, cam


def _teardown(arm, cam, args):
    cam.close()
    try:
        if args.release:
            arm.disconnect()
            print("从臂已断开（扭矩已释放）")
        else:
            arm.bus.disconnect(disable_torque=False)
            print("从臂串口已关闭（**扭矩保持**）")
    except Exception:
        pass


def _print_rec(rec):
    print("  [%2d] %-4s Reached=%s(%.1f mm) Grasped=%s(load %.1f%%) Placed=%s  %s"
          % (rec["idx"], COLOR_ZH.get(rec.get("color"), "-"),
             "✓" if rec.get("reached") else "✗", rec.get("reach_mm", float("nan")),
             "✓" if rec.get("grasped") else "✗", rec.get("load_after_lift", float("nan")),
             "✓" if rec.get("placed") else "✗", rec.get("reason", "")))


def cmd_single(kin, args) -> int:
    if not args.confirm_motion:
        print("✗ 会驱动机器人，加 --confirm-motion")
        return 2
    cal, arm, dik, cam = _setup(kin, args)
    try:
        if not args.no_recover:
            pass   # 由调用方先跑 --recover；这里不自动动
        rec = run_trial(kin, dik, arm, cam, cal, args,
                        np.random.default_rng(args.seed), 1)
        print("\n=== 单次抓放结果 ===")
        _print_rec(rec)
        print("  目标方块像素 %s → 基座 %s（视差修正后 %s）"
              % (np.round(rec["pix"], 0).tolist(),
                 np.round(rec["xy_plane"], 4).tolist(),
                 np.round(rec["xy_cube"], 4).tolist()))
        print("  静差补偿 %s mm → 下发 %s"
              % (np.round(rec["drift_mm"], 2).tolist(),
                 np.round(rec["cmd"], 4).tolist()))
        if args.json_out:
            args.json_out.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
            print("  已写 %s" % args.json_out)
        return 0 if (rec["grasped"] and rec["placed"]) else 2
    finally:
        _teardown(arm, cam, args)


def cmd_trials(kin, args) -> int:
    if not args.confirm_motion:
        print("✗ 会驱动机器人，加 --confirm-motion")
        return 2
    n = args.trials
    cal, arm, dik, cam = _setup(kin, args)
    recs = []
    try:
        print("\n>>> 正式协议：%d 试次（交错执行，不重置场景；分段计 Reached/Grasped/Placed）"
              % n)
        for i in range(1, n + 1):
            rec = run_trial(kin, dik, arm, cam, cal, args,
                            np.random.default_rng(args.seed + i), i)
            recs.append(rec)
            _print_rec(rec)
            # 每轮把抓起/未抓起的方块位置变化记录下来，便于复盘
        ok_r = [r for r in recs if r.get("reached")]
        ok_g = [r for r in recs if r.get("grasped")]
        ok_p = [r for r in recs if r.get("placed")]
        print("\n=== P2 分段成功率（%d 试次）===" % n)
        for tag, arr in (("Reached(±%.0fmm)" % args.reach_tol_mm, ok_r),
                         ("Grasped", ok_g), ("Placed", ok_p)):
            p = len(arr) / n
            se = (p * (1 - p) / n) ** 0.5
            print("  %-18s %2d/%d = %5.1f%%   （95%% CI ±%.1f pp）"
                  % (tag, len(arr), n, 100 * p, 196 * se))
        from collections import Counter
        fails = Counter(r.get("reason", "?") for r in recs if not r.get("placed"))
        if fails:
            print("\n  失败模式分类:")
            for k, v in fails.most_common():
                print("    %-34s %d 次" % (k, v))
        if args.json_out:
            args.json_out.write_text(json.dumps(
                dict(n=n, recs=recs, reached=len(ok_r), grasped=len(ok_g),
                     placed=len(ok_p), cfg=vars(args)), ensure_ascii=False,
                indent=1, default=str), encoding="utf-8")
            print("  已写 %s" % args.json_out)
        return 0 if len(ok_p) / n >= 0.7 else 2
    finally:
        _teardown(arm, cam, args)


def cmd_approach(kin, args) -> int:
    """只接近不抓：下降到方块上方，量「相机看到的手指 vs 方块」的实际横向偏差

    这一步一次性量化了 P2 新增的两个误差源（视差 + 抓取高度的静差），
    确认在夹爪容差内之后再跑抓取，避免盲目跑 20 试次。
    """
    if not args.confirm_motion:
        print("✗ 会驱动机器人，加 --confirm-motion")
        return 2
    cal = load_calib(args.homography, kin)
    H = np.asarray(cal["H"], float)
    drift = cal.get("drift_model")
    z_plane = float(cal["z_m"])
    x_ref = np.asarray(cal["x_ref"], float)

    from tools.handeye_calib import _connect
    arm, T_now = _connect(kin, Args(args.port, args.calib, args.i_term, args.release))
    dik = make_dik(kin, _xref_from(T_now))
    cam = Cam()
    try:
        rgb, cands = detect_cubes(cam, args.colors)
        if not cands:
            print("✗ 画面里没有方块")
            return 2
        tgt = pick_target(cands, args.target, np.random.default_rng(args.seed), DROP_XY)
        xy_plane = apply_h(H, np.asarray(tgt.center, float).reshape(1, 2))[0]
        xy_cube = cube_xy_from_pixel(H, tgt.center, CUBE_H_M / 2, z_plane, args.parallax)
        print("目标方块: %s 像素 (%3.0f,%3.0f)" % (tgt.color_zh, tgt.center[0], tgt.center[1]))
        print("  单应→基座（标记平面 z=%.3f）: (%+.4f, %+.4f)" % (z_plane, *xy_plane))
        print("  视差修正→方块中心平面 z=%.3f: (%+.4f, %+.4f)%s"
              % (CUBE_H_M / 2, *xy_cube, "" if args.parallax else "（无视差模型，未修正）"))
        ddx, ddy = apply_drift(drift, xy_cube[0], xy_cube[1]) if drift else (0.0, 0.0)
        cmd_xy = (xy_cube[0] - ddx, xy_cube[1] - ddy)
        print("  静差补偿 (%+.2f, %+.2f) mm → 下发 (%+.4f, %+.4f)"
              % (ddx * 1000, ddy * 1000, *cmd_xy))

        z_go = args.grasp_z if args.grasp_z is not None else GRASP_Z
        print("\n>>> 分段移动到方块上方 (%.4f, %.4f, z=%.3f)…" % (cmd_xy[0], cmd_xy[1], z_go))
        ok, why = goto_lift(dik, arm, (cmd_xy[0], cmd_xy[1], z_go), z_safe=Z_SAFE)
        if not ok:
            print("✗ 移动失败: %s" % why)
            return 2
        time.sleep(0.5)

        # 到位后只看**标记**：方块此时被夹爪压在下面、俯视相机根本看不到它
        #（实测：悬停到方块上方后，方块直接从检测结果里消失）。
        # 所以拿"移动前记录的方块像素"与"移动后的标记像素"都用同一条单应换算 ——
        # 这正是流水线实际会犯的误差（流水线也是用 H 从像素算方块坐标），量它才有意义。
        ref_area = args.ref_area
        if ref_area is None:
            for p in cal.get("points", []):
                mk = p.get("marker") or {}
                if mk.get("area"):
                    ref_area = float(mk["area"])
                    break
        px_pred = apply_h(np.linalg.inv(H), np.array(cmd_xy).reshape(1, 2))[0]
        marker_px, spread, info = stable_marker(
            cam, cal.get("marker_kind", "panel"), cal.get("marker_color"),
            expect_px=px_pred, ref_area=ref_area, max_jump_px=args.track_gate_px)
        print("\n=== 接近结果 ===")
        if marker_px is None:
            print("  ✗ 到位后看不到标记（类型 %s）%s" % (cal.get("marker_kind"), info or ""))
            return 2
        tip = apply_h(H, marker_px.reshape(1, 2))[0]
        cub = apply_h(H, np.asarray(tgt.center, float).reshape(1, 2))[0]
        off = float(np.hypot(tip[0] - cub[0], tip[1] - cub[1]) * 1000)
        print("  移动前 方块像素 (%3.0f,%3.0f) → 基座 (%+.4f, %+.4f)"
              % (tgt.center[0], tgt.center[1], cub[0], cub[1]))
        print("  移动后 标记像素 (%3.0f,%3.0f) → 基座 (%+.4f, %+.4f)  抖动 %.2f px"
              % (marker_px[0], marker_px[1], tip[0], tip[1], spread))
        print("  → 手指相对方块的横向偏差 **%.2f mm**" % off)
        print("     （含视差：单应在标记平面 z=%.3f 标定，而方块在 z≈%.3f 平面）"
              % (z_plane, CUBE_H_M / 2))
        tol = args.grasp_tol_mm
        print("  判定: %s（阈值 %.1f mm；2 cm 方块的经验容差 ~±8 mm）"
              % ("✅ 可以试抓" if off <= tol else "❌ 偏太多，先修视差", tol))
        return 0 if off <= tol else 2
    finally:
        cam.close()
        try:
            if args.release:
                arm.disconnect()
            else:
                arm.bus.disconnect(disable_torque=False)
            print("从臂串口已关闭（扭矩保持）")
        except Exception:
            pass


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="P2 闭环抓放")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", action="store_true", help="只读：列出画面里的方块")
    g.add_argument("--approach-only", action="store_true", help="只接近不抓（量总误差）")
    g.add_argument("--single", action="store_true", help="单次抓放")
    g.add_argument("--trials", type=int, default=0, help="正式协议：N 试次")
    ap.add_argument("--homography", type=Path, default=HOMOGRAPHY)
    ap.add_argument("--calib", type=Path, default=REPO / "config" / "calibration.json")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--i-term", type=int, default=None)
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--colors", default=None, help="只抓这些颜色（逗号分隔）")
    ap.add_argument("--target", choices=["far", "nearest", "random"], default="far")
    ap.add_argument("--grasp-z", type=float, default=None, help="覆盖抓取高度（米）")
    ap.add_argument("--drop", type=float, nargs=2, default=list(DROP_XY),
                    metavar=("X", "Y"), help="投放点（米）")
    ap.add_argument("--parallax", type=json.loads, default=None,
                    help='视差模型 JSON，如 \'{"principal_px":[320,240],'
                         '"origin_xy":[0.2,0.0],"k_per_m":2.0}\'')
    ap.add_argument("--ref-area", type=float, default=None, help="标记参考面积（跟踪用）")
    ap.add_argument("--track-gate-px", type=float, default=45.0)
    ap.add_argument("--grasp-tol-mm", type=float, default=8.0,
                    help="判定「可以试抓」的横向偏差阈值（默认 8 mm）")
    ap.add_argument("--grasp-load", type=float, default=GRASP_LOAD,
                    help="判「夹到东西」的夹爪负载阈值（%%）")
    ap.add_argument("--hold-load", type=float, default=6.0,
                    help="抬起后仍算「夹住」的负载阈值（%%）；掉出去会回落到 ~0")
    ap.add_argument("--reach-tol-mm", type=float, default=10.0,
                    help="Reached 判据：末端到达抓取点 ±该值（默认 10 mm）")
    ap.add_argument("--place-tol-mm", type=float, default=50.0,
                    help="Placed 判据：投放后方块落在投放点 ±该值（默认 50 mm）")
    ap.add_argument("--drop-z", type=float, default=0.045,
                    help="投放时下降到的高度（米）")
    ap.add_argument("--no-recover", action="store_true",
                    help="不预先做姿态恢复（默认也不自动做，需先手动跑 --recover）")
    ap.add_argument("--in-region-margin", type=float, default=0.008,
                    help="只抓映射后落标定区域【内缩该值】的方块（默认 8 mm），"
                         "区域外是外推、实测会偏十几毫米")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--confirm-motion", action="store_true")
    ap.add_argument("--snap", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    from hardware.so101_kinematics import So101Kinematics
    kin = So101Kinematics(calib_path=args.calib)
    args.colors = [c for c in (args.colors or "").split(",") if c] or None

    if args.scene:
        return cmd_scene(kin, args)
    if args.approach_only:
        return cmd_approach(kin, args)
    if args.single:
        return cmd_single(kin, args)
    if args.trials:
        return cmd_trials(kin, args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
