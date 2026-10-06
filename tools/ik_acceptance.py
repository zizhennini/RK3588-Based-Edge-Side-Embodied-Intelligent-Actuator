#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-2 离线 IK 数值验收（placo 后端，无需硬件）

对应 docs/测试方案_VLM+外部IK+抓取.md 的 P0-2。在板端实测（RK3588 aarch64）。

场景设计依据「真实工况」而非随机测试
------------------------------------
VLM+IK 的实际控制方式是：VLM 给出一个比当前位姿稍远的目标（厘米级），
IK 从**上一次的解**热启动迭代过去。因此真实工况 = 短距离 + 热启动。
故验收以场景 D（热启动轨迹跟踪）为准，A/B 为静态复核，C 只作诊断：

  A 静态近邻  起点 = 目标 ±5°，冷启动
  B 静态中距  起点 = 目标 ±30°，冷启动
  C 静态跨半空间 起点 = 目标 ±90°，冷启动（压力测试，只报告）
  D 热启动轨迹 关节空间插值轨迹 100 步，每步以上一步的解热启动（真实工况）

判据（P0-2）
  ① 场景 D 单步中位耗时 < 10 ms（板端 CPU 预算）
  ② 场景 D 位置误差中位 ≤ 2 mm
  ③ A/B/D 收敛率 ≥ 95%
  ④ 所有解 100% 落在标定限位内（越界即实机堵转，零容忍）

用法::
    python tools/ik_acceptance.py                 # 板端/PC 端均可（需 placo）
    python tools/ik_acceptance.py --targets 200 --seed 0 --json-out /tmp/p0_ik.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hardware.so101_kinematics import (  # noqa: E402
    JOINT_NAMES, MAX_ITER, POS_TOL_M, ROT_TOL_RAD, So101Kinematics,
    build_calibrated_urdf, joint_limits_rad,
)

DEFAULT_TARGETS = 200
TRAJ_STEPS = 100


def fmt_deg(rad_arr) -> str:
    return "[" + ", ".join("%+7.2f" % d for d in np.degrees(np.atleast_1d(rad_arr))) + "]"


# ---------------------------------------------------------------------------
def report_limits(kin: So101Kinematics) -> None:
    print("=== 标定限位（从 config/calibration.json → rad）===")
    for d in kin.limit_detail:
        print("  J%d %-14s raw[%4.0f,%4.0f] → %s  行程 %5.1f°"
              % (d["id"], d["joint"], d["raw_min"], d["raw_max"],
                 fmt_deg([d["lo_rad"], d["hi_rad"]]), d["span_deg"]))

    # 模型里的限位必须与标定一致，否则 IK 会解出实机到不了的角
    lo_m = np.asarray(kin.model.lowerPositionLimit).ravel()[kin.joint_idx]
    hi_m = np.asarray(kin.model.upperPositionLimit).ravel()[kin.joint_idx]
    print("\n=== 模型限位 vs 标定限位校验 ===")
    dlo = np.abs(lo_m - kin.q_lo).max()
    dhi = np.abs(hi_m - kin.q_hi).max()
    print("  最大偏差: lower %.3e rad  upper %.3e rad  → %s"
          % (dlo, dhi, "一致 ✓" if max(dlo, dhi) < 1e-6 else "不一致 ✗"))


def report_convention(kin: So101Kinematics, data_dir: Path, n_ep: int = 8) -> dict:
    """用真实采集数据正解，验证 URDF 关节零点 == 数据集角度零点

    判据（物理）：桌面应在 z≈0；臂展不应超过 SO-101 物理臂长；抓取时刻末端贴近桌面。
    """
    eps = sorted(data_dir.glob("episode_*.json"))[:n_ep]
    if not eps:
        print("\n（跳过约定验证：%s 下无 episode_*.json）" % data_dir)
        return {}
    zs, reaches, gz = [], [], []
    n_out = np.zeros(6, dtype=int)
    n_frames_lim = 0
    for f in eps:
        try:
            frames = json.loads(f.read_text(encoding="utf-8")).get("frames", [])
        except Exception:
            continue
        if not frames:
            continue
        F = np.array([[fr["F%d" % i] for i in range(1, 7)] for fr in frames])
        P = np.array([kin.fk_pos(np.deg2rad(row)) for row in F])
        zs.append(P[:, 2])
        reaches.append(np.linalg.norm(P, axis=1))
        g6 = F[:, 5]
        closed = g6 <= (g6.min() + 0.15 * (g6.max() - g6.min()))
        if closed.any():
            gz.append(P[closed, 2])
        # F* 是从臂实测角 → 必然落在「录制时那份标定」的行程内。
        # 若用别的标定去量却大量越界，说明这份标定不是录制时用的那份。
        Q = np.deg2rad(F)
        n_out += (np.maximum(kin.q_lo - Q, Q - kin.q_hi) > np.deg2rad(0.5)).sum(axis=0)
        n_frames_lim += len(F)
    if not zs:
        return {}
    z = np.concatenate(zs)
    r = np.concatenate(reaches)
    out = dict(n_episodes=len(zs), frames=int(z.size),
               z_min=float(z.min()), z_p1=float(np.percentile(z, 1)),
               z_median=float(np.median(z)), z_max=float(z.max()),
               reach_median=float(np.median(r)), reach_max=float(r.max()))
    print("\n=== 约定验证：真实数据正解（%d 个 episode / %d 帧）===" % (out["n_episodes"], out["frames"]))
    print("  末端高度 z: min %+.4f  p1 %+.4f  中位 %+.4f  max %+.4f  (m)"
          % (out["z_min"], out["z_p1"], out["z_median"], out["z_max"]))
    print("  臂展 |p|: 中位 %.4f  max %.4f  (m)" % (out["reach_median"], out["reach_max"]))
    if gz:
        g = np.concatenate(gz)
        out["grasp_z_median"] = float(np.median(g))
        print("  夹爪闭合(抓取)时刻末端高度: 中位 %+.4f  (m)" % out["grasp_z_median"])
    ok = (out["z_min"] > -0.03) and (out["reach_max"] < 0.45)
    print("  判据（桌面 z≈0 且臂展<0.45m）: %s" % ("通过 ✓" if ok else "未通过 ✗"))
    out["convention_ok"] = bool(ok)

    # 「这份标定是不是录制数据时用的那份」——F* 是实测角，必须落在录制标定的行程内
    out["limit_out_of_range"] = n_out.tolist()
    out["frames_checked"] = int(n_frames_lim)
    total_out = int(n_out.sum())
    print("  F* 越界（用当前标定衡量 %d 帧，容差 0.5°）: %s"
          % (n_frames_lim, "全部在行程内 ✓" if total_out == 0
             else "越界 %d 次 %s ← 该标定很可能不是录制数据时用的那份"
                  % (total_out, dict(zip(JOINT_NAMES[:6], n_out.tolist())))))
    return out


# ---------------------------------------------------------------------------
def scenario_static(kin: So101Kinematics, name: str, noise_deg: float,
                    n: int, rng: np.random.Generator) -> dict:
    """目标 = 限位内随机关节角的正解；起点 = 目标 ± noise（冷启动）"""
    conv = inlim = 0
    iters, times, perrs, rerrs = [], [], [], []
    worst = 0.0
    for _ in range(n):
        q_t = rng.uniform(kin.q_lo, kin.q_hi)
        T_goal = kin.fk(q_t)
        q_seed = kin.clamp(q_t + np.deg2rad(rng.uniform(-noise_deg, noise_deg, 6)))
        # 冷启动：显式给 seed，不沿用上次解
        r = kin.ik(T_goal, q_seed=q_seed, max_iter=MAX_ITER)
        iters.append(r.iters); times.append(r.ms)
        perrs.append(r.pos_err); rerrs.append(r.rot_err)
        worst = max(worst, r.max_limit_violation)
        conv += int(r.converged)
        inlim += int(r.in_limits)
    iters, times = np.array(iters), np.array(times)
    perrs, rerrs = np.array(perrs), np.array(rerrs)
    res = dict(name=name, n=n, mode="cold", noise_deg=noise_deg,
               converged=conv, in_limits=inlim,
               med_ms=float(np.median(times)), p90_ms=float(np.percentile(times, 90)),
               max_ms=float(times.max()), med_iter=float(np.median(iters)),
               med_pos_err=float(np.median(perrs)), max_pos_err=float(max(perrs)),
               med_rot_err=float(np.median(rerrs)), max_rot_err=float(max(rerrs)),
               max_violation_deg=float(np.degrees(worst)))
    print("\n=== 场景 %s ===" % name)
    print("  收敛 %d/%d = %.1f%%   限位内 %d/%d = %.1f%%"
          % (conv, n, 100.0 * conv / n, inlim, n, 100.0 * inlim / n))
    print("  迭代 中位 %.0f  max %d | 耗时 中位 %.2f ms  p90 %.2f ms  max %.2f ms"
          % (np.median(iters), max(iters), res["med_ms"], res["p90_ms"], res["max_ms"]))
    print("  位置误差 中位 %.2e m  max %.2e m | 姿态误差 中位 %.2e rad  max %.2e rad"
          % (res["med_pos_err"], res["max_pos_err"], res["med_rot_err"], res["max_rot_err"]))
    print("  最大限位越界 %.3e°" % res["max_violation_deg"])
    return res


def scenario_trajectory(kin: So101Kinematics, name: str, steps: int,
                        rng: np.random.Generator) -> dict:
    """热启动轨迹跟踪（真实工况）：关节空间插值，每步用上一步的解热启动"""
    q_a = rng.uniform(kin.q_lo, kin.q_hi)
    q_b = rng.uniform(kin.q_lo, kin.q_hi)
    ts = np.linspace(0.0, 1.0, steps)
    conv = inlim = 0
    iters, times, perrs, rerrs = [], [], [], []
    worst = 0.0
    # 先让 solver 进入轨迹起点（冷启动一次）
    kin.ik(kin.fk(q_a), q_seed=q_a, max_iter=MAX_ITER)
    for t in ts:
        q_ref = (1.0 - t) * q_a + t * q_b
        T_goal = kin.fk(q_ref)
        r = kin.ik(T_goal, warm_start=True, max_iter=MAX_ITER)  # 热启动
        iters.append(r.iters); times.append(r.ms)
        perrs.append(r.pos_err); rerrs.append(r.rot_err)
        worst = max(worst, r.max_limit_violation)
        conv += int(r.converged)
        inlim += int(r.in_limits)
    n = steps
    iters, times = np.array(iters), np.array(times)
    perrs, rerrs = np.array(perrs), np.array(rerrs)
    res = dict(name=name, n=n, mode="warm", converged=conv, in_limits=inlim,
               med_ms=float(np.median(times)), p90_ms=float(np.percentile(times, 90)),
               max_ms=float(times.max()), med_iter=float(np.median(iters)),
               med_pos_err=float(np.median(perrs)), max_pos_err=float(max(perrs)),
               med_rot_err=float(np.median(rerrs)), max_rot_err=float(max(rerrs)),
               max_violation_deg=float(np.degrees(worst)))
    print("\n=== 场景 %s ===" % name)
    print("  收敛 %d/%d = %.1f%%   限位内 %d/%d = %.1f%%"
          % (conv, n, 100.0 * conv / n, inlim, n, 100.0 * inlim / n))
    print("  迭代 中位 %.0f  max %d | 耗时 中位 %.2f ms  p90 %.2f ms  max %.2f ms"
          % (np.median(iters), max(iters), res["med_ms"], res["p90_ms"], res["max_ms"]))
    print("  位置误差 中位 %.2e m  max %.2e m | 姿态误差 中位 %.2e rad  max %.2e rad"
          % (res["med_pos_err"], res["max_pos_err"], res["med_rot_err"], res["max_rot_err"]))
    print("  最大限位越界 %.3e°" % res["max_violation_deg"])
    return res


def report_workspace(kin: So101Kinematics, n: int = 20000,
                     rng: np.random.Generator = None) -> dict:
    """采样可达工作空间（P0-3 像素→3D 映射要用）"""
    rng = rng or np.random.default_rng(1)
    Q = rng.uniform(kin.q_lo, kin.q_hi, size=(n, 6))
    P = np.array([kin.fk_pos(q) for q in Q])
    out = {}
    print("\n=== 可达工作空间（%d 随机取样）===" % n)
    for i, ax in enumerate("xyz"):
        out[ax] = [float(P[:, i].min()), float(P[:, i].max())]
        print("  %s: [%+.4f, %+.4f] m" % (ax, out[ax][0], out[ax][1]))
    r = np.linalg.norm(P, axis=1)
    out["reach"] = [float(r.min()), float(r.max())]
    print("  臂展 |p|: [%.4f, %.4f] m" % (out["reach"][0], out["reach"][1]))
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="P0-2 离线 IK 数值验收（placo）")
    ap.add_argument("--targets", type=int, default=DEFAULT_TARGETS, help="每场景目标数")
    ap.add_argument("--steps", type=int, default=TRAJ_STEPS, help="轨迹场景步数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--urdf", type=Path, default=None, help="已带限位的 URDF（默认自动生成）")
    ap.add_argument("--calib", type=Path, default=None,
                    help="标定文件（默认 config/calibration.json）；"
                         "用别的标定跑可诊断「标定文件与数据集是否配套」")
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/raw/pick_place"), help="用于约定验证的采集数据")
    ap.add_argument("--rebuild-urdf", action="store_true", help="强制重新生成标定 URDF")
    ap.add_argument("--json-out", type=Path, default=None, help="结果写入 JSON")
    ap.add_argument("--skip-workspace", action="store_true")
    ap.add_argument("--convention-only", action="store_true",
                    help="只跑「关节零点约定」验证（诊断标定文件是否与数据集配套）")
    args = ap.parse_args()

    if args.rebuild_urdf:
        rep = build_calibrated_urdf(dst=args.urdf) if args.urdf else build_calibrated_urdf()
        print("已生成标定 URDF: %s（替换 %d/%d 个关节限位，剥离 mesh=%s）"
              % (rep["dst"], rep["replaced"], rep["expected"], rep["strip_meshes"]))

    import placo
    t_load = time.perf_counter()
    if args.calib is not None:
        # 用指定标定现场生成一份临时 URDF（诊断用，不覆盖默认产物）
        import tempfile
        tmp_urdf = Path(tempfile.mkdtemp(prefix="ik_calib_")) / "so101_calibrated.urdf"
        build_calibrated_urdf(calib_path=args.calib, dst=tmp_urdf)
        kin = So101Kinematics(urdf=tmp_urdf, calib_path=args.calib)
    else:
        kin = So101Kinematics(urdf=args.urdf)
    load_ms = (time.perf_counter() - t_load) * 1000.0

    print("=" * 66)
    print("P0-2 离线 IK 数值验收  |  placo @ 本机")
    print("=" * 66)
    print("URDF: %s" % kin.urdf)
    print("TCP : %s | 关节索引 %s | q 维数 %d | 加载耗时 %.1f ms"
          % (kin.tcp_frame, kin.joint_idx.tolist(),
             int(np.asarray(kin.model.lowerPositionLimit).ravel().size), load_ms))
    print("标定: %s  (指纹 %s) — 必须与本机实机标定一致"
          % (kin.calib_path, kin.calib_fingerprint))
    print("容差: 位置 %.0e m  姿态 %.0e rad  最大迭代 %d" % (POS_TOL_M, ROT_TOL_RAD, MAX_ITER))

    report_limits(kin)
    conv_rep = report_convention(kin, args.data_dir)

    if args.convention_only:
        return 0 if conv_rep.get("convention_ok", False) else 2

    rng = np.random.default_rng(args.seed)
    results = [
        scenario_static(kin, "A 静态近邻 (±5°)", 5.0, args.targets, rng),
        scenario_static(kin, "B 静态中距 (±30°)", 30.0, args.targets, rng),
        scenario_trajectory(kin, "D 热启动轨迹跟踪（真实工况）", args.steps, rng),
        scenario_static(kin, "C 跨半空间 (±90°, 诊断用)", 90.0, args.targets, rng),
    ]

    ws = {} if args.skip_workspace else report_workspace(kin, rng=rng)

    # ── 判据 ────────────────────────────────────────────────────────
    by_name = {r["name"]: r for r in results}
    d = by_name["D 热启动轨迹跟踪（真实工况）"]
    c1 = d["med_ms"] < 10.0
    c2 = d["med_pos_err"] <= 0.002
    realistic = [by_name["A 静态近邻 (±5°)"], by_name["B 静态中距 (±30°)"], d]
    c3 = all(r["converged"] / r["n"] >= 0.95 for r in realistic)
    c4 = all(r["in_limits"] == r["n"] for r in results)
    c5 = conv_rep.get("convention_ok", True)

    print("\n" + "=" * 66)
    print("验收汇总")
    print("=" * 66)
    print("%-30s %8s %8s %10s %10s" % ("场景", "收敛率", "限位内", "中位耗时", "中位误差"))
    for r in results:
        print("%-30s %7.1f%% %7.1f%% %8.2fms %9.2e" % (
            r["name"], 100.0 * r["converged"] / r["n"],
            100.0 * r["in_limits"] / r["n"], r["med_ms"], r["med_pos_err"]))

    print("\n判据:")
    checks = [
        ("① 场景 D 单步中位耗时 < 10 ms", c1, "%.2f ms" % d["med_ms"]),
        ("② 场景 D 位置误差中位 ≤ 2 mm", c2, "%.2e m" % d["med_pos_err"]),
        ("③ A/B/D 收敛率 ≥ 95%", c3,
         " / ".join("%.1f%%" % (100.0 * r["converged"] / r["n"]) for r in realistic)),
        ("④ 所有解 100% 在标定限位内", c4,
         "越界 %.1e°" % max(r["max_violation_deg"] for r in results)),
        ("⑤ 关节零点约定与数据集一致", c5,
         "z_min %+.4f m" % conv_rep.get("z_min", float("nan"))),
    ]
    for label, ok, extra in checks:
        print("  %-34s %s   (%s)" % (label, "✓ 通过" if ok else "✗ 未通过", extra))
    all_ok = all(ok for _, ok, _ in checks)
    print("\n结论: P0-2 %s" % ("通过 ✅" if all_ok else "未通过 ❌"))

    if args.json_out:
        payload = dict(urdf=str(kin.urdf), tcp=kin.tcp_frame,
                       pos_tol=POS_TOL_M, rot_tol=ROT_TOL_RAD, max_iter=MAX_ITER,
                       limits=kin.limit_detail, results=results, workspace=ws,
                       convention=conv_rep,
                       checks={lbl: bool(ok) for lbl, ok, _ in checks},
                       pass_all=bool(all_ok))
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print("结果已写 %s" % args.json_out)

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
