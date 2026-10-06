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

        # 到了之后再看一眼：方块还在原像素附近吗？手指（标记）离方块多远？
        rgb2, cands2 = detect_cubes(cam, args.colors)
        same = None
        for c in cands2:
            if c.color == tgt.color and np.hypot(c.center[0] - tgt.center[0],
                                                c.center[1] - tgt.center[1]) < 60:
                same = c
                break
        if same is None:
            print("⚠ 下降后方块不在原位（可能被手指挡住或被碰走）→ 无法量偏差")
        # 标记的参考面积与预测像素：优先用标定时记录的实际值，避免跟踪误锁
        ref_area = args.ref_area
        if ref_area is None:
            for p in cal.get("points", []):
                mk = p.get("marker") or {}
                if mk.get("area"):
                    ref_area = float(mk["area"])
                    break
        px_pred = apply_h(np.linalg.inv(H), np.array(cmd_xy).reshape(1, 2))[0]
        expect_marker, _, info = stable_marker(cam, cal.get("marker_kind", "panel"),
                                               cal.get("marker_color"),
                                               expect_px=px_pred,
                                               ref_area=ref_area,
                                               max_jump_px=args.track_gate_px)
        print("\n=== 接近结果 ===")
        if same is not None and expect_marker is not None:
            # 把两者的像素都换算到基座系比较（用同一条单应，故视差被抵消一半，
            # 这里主要看"手指相对方块"的横向偏差）
            tip = apply_h(H, expect_marker.reshape(1, 2))[0]
            cub = apply_h(H, same.center.reshape(1, 2))[0]
            off = np.hypot(tip[0] - cub[0], tip[1] - cub[1]) * 1000
            print("  手指(标记) 像素 (%3.0f,%3.0f)" % tuple(expect_marker))
            print("  方块      像素 (%3.0f,%3.0f)" % tuple(same.center))
            print("  → 横向偏差 **%.2f mm**（应小于夹爪可容差；方块 2 cm，经验容差 ~±8 mm）"
                  % off)
            print("  判定: %s" % ("✅ 可以试抓" if off <= 8.0 else "❌ 偏太多，先查视差/静差"))
            return 0 if off <= 8.0 else 2
        print("  未能量化偏差（见上面告警）")
        return 2
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
    print("（--single / --trials 尚未实现，先用 --approach-only 量化误差）")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
