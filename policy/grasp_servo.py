"""抓取下探闭环视觉伺服(融合 lerobot-mujoco-kinematics 机制,2026-10-08)

设计文档: .research/融合设计_闭环视觉伺服下探.md
参照: acharjee07/lerobot-mujoco-kinematics(像素误差 P 控制闭环)+ dougsm/ggcnn_kinova_grasping
     (SERVO 状态机 / 位姿滑动平均 / 深度盲切)

与开环执行的差异: 下探过程中每轮重新检测目标,像素→基座 XY 的映射误差
(标定残差 / 目标被碰移)在执行中持续修正,而不是"感知一次后闭眼摸完"。

复用资产(全部为 P1/P2 已验证组件,本文件不含新的运动学代码):
    - tools/handeye_calib.DownIK / ik_path   placo 伺服 IK(持久 solver,直线渐近)
    - tools/pick_place.detect_cubes          HSV 目标检测(实测 3.56ms)
    - tools/pick_place.cube_xy_from_pixel    像素→基座 XY(含视差修正)
    - tools/pick_place.close_until_grasp     负载判定闭合(防堵转)
    - config.settings SERIAL_PORT
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.pick_place import (  # noqa: E402
    close_until_grasp,
    cube_xy_from_pixel,
    detect_cubes,
    open_gripper,
)
from tools.handeye_calib import goto  # noqa: E402


def _pick_nearest(cands, last_px):
    """连续性跟踪: 选与上一帧目标像素最近的候选,避免多方块间跳变。"""
    if not cands:
        return None
    if last_px is None:
        return cands[0]
    return min(cands, key=lambda c: np.hypot(c.center[0] - last_px[0],
                                             c.center[1] - last_px[1]))


def servo_descent(
    dik, arm, cam, H, cal,
    xy_init, *,
    z_start: float | None = None,
    z_grasp: float | None = None,
    colors=None,
    parallax: dict | None = None,
    v_descent: float = 0.015,      # m/s 恒速下探
    k_p: float = 0.5,              # XY 修正比例增益(每帧修正误差的 50%)
    max_step: float = 0.002,       # 单帧 XY 修正钳制(m)
    deadband: float = 0.003,       # XY 死区(m)——防到位后抖动
    lost_tol: int = 15,            # 目标丢失容忍帧数(30Hz 下约 0.5s)
    timeout: float = 12.0,         # 伺服总超时(s)
    servo_period: float = 0.08,    # 伺服节拍(s)——每轮检测+修正+小步下探
    z_plane: float | None = None,  # 标定平面高度(默认取 cal 抓取平面)
    close_kwargs: dict | None = None,
    dry_run: bool = False,         # True: 只验证 XY 修正收敛,不下爪/不闭合
) -> dict:
    """闭环下探: 视觉实时修正 XY + 恒速下探 Z,到位后负载判定闭合。

    Returns:
        dict(success, reason, grasped, load_pct, iterations, xy_final, z_final)
    """
    from config.settings import SERIAL_PORT  # noqa: F401  (goto 内部走 dik/arm)

    z_grasp_val = z_grasp if z_grasp is not None else float(
        __import__("tools.pick_place", fromlist=["grasp_z_for_calib"]
                   ).grasp_z_for_calib(cal))
    z_plane_val = z_plane if z_plane is not None else float(
        cal.get("z_plane", 0.0)) if isinstance(cal, dict) else z_plane_val_default(cal)

    xy_cur = np.asarray(xy_init, float).copy()
    z_cur = float(z_start) if z_start is not None else float(
        dik.kin.fk(arm.read_positions())[2, 3])
    last_px = None
    lost = 0
    it = 0
    t0 = time.perf_counter()
    dt = servo_period

    # 张开夹爪准备抓取
    open_gripper(arm)

    while True:
        loop_t0 = time.perf_counter()
        it += 1
        if time.perf_counter() - t0 > timeout:
            return {"success": False, "reason": "伺服超时", "iterations": it,
                    "xy_final": xy_cur.tolist(), "z_final": z_cur}

        # ---- 感知 ----
        _, cands = detect_cubes(cam, colors=colors)
        target = _pick_nearest(cands, last_px)
        if target is None:
            lost += 1
            if lost > lost_tol:
                return {"success": False, "reason": "目标丢失", "iterations": it,
                        "xy_final": xy_cur.tolist(), "z_final": z_cur}
        else:
            lost = 0
            last_px = np.asarray(target.center, float)
            # ---- 像素 → 基座 XY(带视差修正)→ 误差 → P 修正 ----
            xy_meas = cube_xy_from_pixel(H, last_px, z_grasp_val,
                                         z_plane_val, parallax)
            err = xy_meas - xy_cur
            if np.hypot(*err) > deadband:
                step = np.clip(k_p * err, -max_step, max_step)
                xy_cur = xy_cur + step

        # ---- Z 恒速下探 ----
        z_cur = max(z_cur - v_descent * dt, z_grasp_val)

        # ---- IK + 逐步写出(复用 goto: 内部 ik_path 直线渐近) ----
        ok, why, _, _ = goto(dik, arm,
                             np.array([xy_cur[0], xy_cur[1], z_cur]))
        if not ok:
            return {"success": False, "reason": f"IK/路径失败: {why}",
                    "iterations": it, "xy_final": xy_cur.tolist(),
                    "z_final": z_cur}

        # ---- 到位判定: Z 达标 → 负载判定闭合 ----
        if z_cur <= z_grasp_val + 1e-6:
            if dry_run:
                return {"success": True, "reason": "dry_run 完成(XY 已收敛到抓取位)",
                        "grasped": None, "iterations": it,
                        "xy_final": xy_cur.tolist(), "z_final": z_cur}
            grasped, load, w, oc = close_until_grasp(
                arm, **(close_kwargs or {}))
            if grasped:
                # 抬升到安全高度(复用 P2 逻辑: 回等待位由调用方决定)
                return {"success": True, "reason": "已闭合", "grasped": True,
                        "load_pct": load, "w_used": w,
                        "iterations": it, "xy_final": xy_cur.tolist(),
                        "z_final": z_cur}
            return {"success": False, "reason": "到位但夹取失败", "grasped": False,
                    "load_pct": load, "iterations": it,
                    "xy_final": xy_cur.tolist(), "z_final": z_cur}

        # ---- 节拍 ----
        elapsed = time.perf_counter() - loop_t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


def z_plane_val_default(cal):
    """从标定 dict 取标定平面高度(兼容缺失)。"""
    try:
        return float(cal.get("z_plane", 0.0))
    except Exception:
        return 0.0
