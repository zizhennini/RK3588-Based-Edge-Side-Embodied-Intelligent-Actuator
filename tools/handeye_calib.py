#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 静态定位：固定俯视相机手眼标定（平面单应）+ 定位精度验证

判据（docs/测试方案_VLM+外部IK+抓取.md P1）：
    指定目标点 → IK → 执行 → 相机实测末端偏差 **≤5 mm**（10 个目标点）

标记的选择（换过一次，原因是实测踩坑）
--------------------------------------
要测"末端实际到哪了"，就得让相机**看见**末端。最初的设计是让从臂**夹一个彩色方块**
当随动标记 —— 理由是白色机械臂在木桌背景上难稳定分割（P0 实测白平衡把塑料染成淡蓝、
H=101，与蓝方块 H=103 几乎重合），而方块有 0.7 px 的定位精度。

**但这个设计实测失败**：相机是前上方俯视，而夹爪朝下，**夹爪本体把夹在下面的方块
完全挡住了** —— 夹住红方块后画面里只剩桌上 5 个候选，被夹的红方块一个像素都看不到。

改用**夹爪上本来就有的黑色矩形面板**当标记。它在原始帧里非常干净：
V<60/80/100 三个阈值下稳定给出 bbox≈(251,315,42,19)、**填充率 0.78~0.84**（实心矩形），
而画面里其它暗分量（线材/底座/云台）填充率只有 0.18~0.45，故「高填充率 + 矩形」即可
唯一锁定。好处是不需要任何额外物品，也不用往夹爪里塞东西。

刚性说明：该面板在 wrist_flex 之后的连杆上。本工具**已经用弱约束把偏航钉死**（实测跨度
0.14°），而偏航对应的正是 wrist_roll 的转角 —— 所以 wrist_flex 之后的任何特征相对 TCP
都是刚性的，不要求它必须在 gripper_link 上。

⚠️ 教训：做颜色/亮度统计**必须在未标注的原始帧上做**。我曾在 draw_candidates 画过红框的
图上统计"红色像素"，把注释框（纯红 255,0,0）当成了"露出来的红方块"。

为什么不需要尺子、也不会有视差误差
----------------------------------
标定与验证都让标记停在**同一高度**（固定 TCP z），于是标记始终在一个平面上，
平面单应严格成立。标定时用「标记像素 ↔ 下发的 TCP 坐标」拟合 H_fit；真实关系是
``pixel → 标记基座坐标 = 下发坐标 + c``（c = 标记相对 TCP 的常量偏移），即
``H_fit = T_{-c} ∘ H_true``，而**平移本身也是单应**，故 c 被精确吸收；
验证时 ``H_fit(pixel) = 标记坐标 − c = 下发坐标 + 真实误差``
→ **c 精确抵消，量到的就是真实定位误差**。所以既不用量 c，也不用尺子。

约束怎么给（这是本文件最关键的设计，踩过两次坑）
------------------------------------------------
必须让 c 在整个扫掠中保持**恒定**，同时又不能把姿态锁死：

1. ❌ **锁死完整 6 自由度姿态**：区域内大部分点**略微不可达**，QP 稳定在折中解。
   实测残差是**地板**而非收敛慢 —— 50 次和 3000 次迭代都是 7.908e-04 m，个别点 1.1e-02 m。
2. ❌ **只给位置、完全放开姿态**：残差降到 1e-11 m（机器精度），但偏航跨度 **49.9°**；
   偏航一转，标记的横向偏移就跟着转 → 破坏"c 恒定"（3 mm 偏移配 50° ≈ 2.1 mm 虚假误差）。
3. ✅ **位置 + 工具 z 轴朝下 + 对工具 x 轴加"弱"约束**（本实现）：
   位置(3)+z 轴对齐(2) 用掉 5 个自由度，6 关节恰好剩 **1 个零空间自由度**就是偏航。
   对它加**弱**约束只作用在零空间 —— 位置精度不受影响，而偏航跨度实测压到 **0.14°**
   （由它引起的标记漂移 ≈0.01 mm，可忽略）。

用法::
    python tools/handeye_calib.py --plan-only                 # 离线自检（不动机器人）
    python tools/handeye_calib.py --check                     # 只读：相机能否看到标记
    python tools/handeye_calib.py --calibrate --confirm-motion
    python tools/handeye_calib.py --verify --points 10 --confirm-motion
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402  （模块级导入：detect_dark_panel / fit_homography / 存图都要用）

from hardware.so101_kinematics import So101Kinematics  # noqa: E402
from perception.cube_locator import COLOR_ZH, find_cubes  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "config" / "homography.json"

#: 默认扫掠区域（基座系，米）。依据：演示数据 TCP 到过 x[0.153,0.316] y[-0.168,0.085]
#: （相机必看得见），再取其中段并避开 x>0.26 的边缘（弱偏航约束下那里残差会变大）。
DEF_X = (0.16, 0.26)
DEF_Y = (-0.09, 0.05)
#: 扫掠高度（TCP=夹爪中心，米）。夹着方块时方块底面还要低 ~2~3 cm，留足余量
DEF_Z = 0.07

STEPS_PER_MOVE = 12          # 每次移动的笛卡尔插值步数
STEP_DELAY_S = 0.03          # 步间延时
SETTLE_S = 0.35              # 到位后稳定时间
FRAMES_PER_POINT = 5         # 每点取帧数（取中位抑制抖动）

#: IK 容差。判据是 5 mm，故 0.5 mm 的 IK 残差只占 10%。
#: （So101Kinematics 默认 1e-4 会让 QP 在 0.1 mm 处擦边，把数值擦边误判成不可达。）
IK_POS_TOL = 5e-4
IK_ROT_TOL = 5e-3
IK_MAX_ITER = 300
YAW_WEIGHT = 0.02            # 偏航弱约束权重（只作用于零空间）


# ---------------------------------------------------------------------------
# 位置 + 工具朝下 + 弱偏航 的 IK
# ---------------------------------------------------------------------------
class DownIK:
    """「TCP 到指定位置 + 工具 z 轴朝下 + 偏航弱约束」的 IK

    为什么不用 ``So101Kinematics.ik()``（它用 add_frame_task 锁死完整姿态）：
    锁死姿态会让区域内多数点略微不可达，残差是地板（见模块 docstring）。

    注：这是 P1 专用实现。P2（抓取）同样需要"夹爪朝下 + 位置"，
    届时宜把本类上提到 ``hardware/so101_kinematics.py`` 复用。
    """

    def __init__(self, kin: So101Kinematics, x_ref=None, yaw_weight: float = YAW_WEIGHT):
        import placo
        self.kin = kin
        self.solver = placo.KinematicsSolver(kin.robot)
        self.solver.mask_fbase(True)
        self.solver.dt = 0.01
        self.solver.enable_joint_limits(True)
        self.pos = self.solver.add_position_task("gripper_frame_link", np.zeros(3))
        # 工具 z 轴对齐世界 -z（夹爪朝下）
        self.az = self.solver.add_axisalign_task(
            "gripper_frame_link", np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))
        # 偏航弱约束：把工具 x 轴拉向水平参考方向。注意 API 是
        # configure(name, priority[, weight])；PositionTask 的接口是 target_world（没有 target）。
        self.ax = None
        if x_ref is not None and yaw_weight > 0:
            self.ax = self.solver.add_axisalign_task(
                "gripper_frame_link", np.array([1.0, 0.0, 0.0]), np.asarray(x_ref, float))
            try:
                self.ax.configure("yaw_reg", "soft", float(yaw_weight))
            except Exception as e:      # pragma: no cover
                print("⚠ 偏航弱约束权重设置失败(%s)，偏航可能漂移" % e)

    def ik(self, xyz, q_seed, max_iter: int = IK_MAX_ITER,
           pos_tol: float = IK_POS_TOL, rot_tol: float = IK_ROT_TOL):
        """返回 (q, pos_err, axis_err, yaw_deg, in_limits, iters)"""
        kin = self.kin
        xyz = np.asarray(xyz, dtype=float)
        kin._set_joints(q_seed)
        kin.robot.update_kinematics()
        self.pos.target_world = xyz
        ep = er = float("inf")
        it = 0
        for it in range(1, max_iter + 1):
            self.solver.solve(True)
            kin.robot.update_kinematics()
            T = kin.robot.get_T_world_frame("gripper_frame_link")
            ep = float(np.linalg.norm(T[:3, 3] - xyz))
            er = float(np.arccos(np.clip(np.dot(T[:3, 2], [0.0, 0.0, -1.0]), -1.0, 1.0)))
            if ep < pos_tol and er < rot_tol:
                break
        T = kin.robot.get_T_world_frame("gripper_frame_link")
        q = kin.joint_angles()
        yaw = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        inl, _ = kin.limit_violation(q)
        conv = (ep < pos_tol and er < rot_tol)
        return q, ep, er, yaw, inl, it, conv


def ik_path(dik: DownIK, q_start, xyz_goal, steps: int = STEPS_PER_MOVE):
    """沿笛卡尔直线渐近到 xyz_goal。

    为什么必须渐近不能一步到位：placo 的 QP 是**局部**求解器。实测从一个有效构型
    出发、目标只挪 1 cm 时 2 次迭代即收敛；但直接跳到 10 cm 外的网格角点会卡在局部
    极小，而且后续热启动会被这个坏种子污染，整片网格全废。

    Returns:
        (ok, reason, q_end, results) —— results 为 (t, 结果元组) 列表
    """
    kin = dik.kin
    q = np.asarray(q_start, dtype=float).copy()
    p0 = kin.fk(q)[:3, 3].copy()
    p1 = np.asarray(xyz_goal, dtype=float)
    results = []
    for i in range(1, steps + 1):
        t = i / steps
        p = p0 + (p1 - p0) * t
        q, ep, er, yaw, inl, it, conv = dik.ik(p, q)
        results.append((t, (q, ep, er, yaw, inl, it, conv)))
        if not conv:
            # 两个误差都要报：只印位置误差会把"姿态没收敛"伪装成"位置没收敛"（踩过）
            return False, ("IK 未收敛 @步%d/%d (位置 %.2e m/容差 %.0e，"
                           "轴 %.2e rad/容差 %.0e)"
                           % (i, steps, ep, IK_POS_TOL, er, IK_ROT_TOL)), q, results
        if not inl:
            return False, "IK 解越限 @步%d/%d" % (i, steps), q, results
    return True, "ok", q, results


def goto(dik: DownIK, arm, xyz_goal):
    """规划并实际写出舵机目标。任一步失败则**不写**舵机（避免半途卡住）"""
    ok, why, q_end, results = ik_path(dik, arm.read_positions(), xyz_goal)
    if not ok:
        return False, why
    for _, r in results:
        arm.write_positions(r[0])
        time.sleep(STEP_DELAY_S)
    return True, "ok"


# ---------------------------------------------------------------------------
# 相机 / 标记
# ---------------------------------------------------------------------------
class Cam:
    """D435i RGB 取帧（复用项目 CameraManager）

    ⚠ 冷启动需要时间：板端实测 **3.5 s** 才出第一帧。等待窗口必须留够，
    否则会误报「相机取不到帧」而其实只是还没热起来（踩过：3.0 s 超时刚好差 0.5 s）。
    """

    def __init__(self, width=640, height=480, fps=30, warmup_s=12.0):
        from hardware.camera_d435i import CameraManager
        self.cam = CameraManager(width=width, height=height, fps=fps)
        self.cam.start()
        t0 = time.time()
        while time.time() - t0 < warmup_s:
            if self.cam.get_rgb() is not None:
                print("相机就绪（冷启动 %.1f s）" % (time.time() - t0))
                return
            time.sleep(0.1)
        raise RuntimeError("相机 %.0f s 内取不到帧（检查 D435i 是否插好 / 是否被别的进程占用）"
                           % warmup_s)

    def grab(self) -> np.ndarray:
        f = self.cam.get_rgb()
        if f is None:
            raise RuntimeError("取帧失败")
        return f

    def close(self):
        try:
            self.cam.stop()
        except Exception:
            pass


def detect_dark_panel(rgb, v_max: int = 80, min_area: int = 180, max_area: int = 3000,
                      min_fill: float = 0.62, aspect=(1.15, 3.4), y_min: int = 250):
    """找夹爪上的**黑色矩形面板**（比"夹在夹爪里的方块"更好的标记）

    为什么换掉方块标记：相机是从前上方俯视，**夹爪本体把夹在下面的方块完全挡住了**
    （实测：夹住红方块后画面里只剩桌上 5 个候选，红方块一个像素都看不到）。
    而夹爪上本来就有一块明显的黑色矩形面板，实测在 V<60/80/100 三个阈值下都稳定给出
    bbox≈(251,315,42,19)、**填充率 0.78~0.84**（实心矩形）；画面里其它暗分量填充率只有
    0.18~0.45（线材、底座、云台），因此"高填充率 + 矩形"就能唯一锁定它。

    刚性说明：该面板位于 wrist_flex 之后的连杆上（手腕/夹爪段）。而本工具**已经用弱约束
    把偏航钉死**（实测跨度 0.14°），偏航对应的正是 wrist_roll 的转角，所以 wrist_flex 之后
    的任何特征相对 TCP 都是刚性的 —— 不需要它是 gripper_link 上的东西。

    Returns: [(area, (cx,cy), (x,y,w,h), fill, aspect)] 按面积降序
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    mask = ((v < v_max).astype(np.uint8)) * 255
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area or a > max_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if y < y_min:
            continue
        fill = a / float(w * h)
        if fill < min_fill:
            continue
        asp = max(w, h) / float(min(w, h))
        if not (aspect[0] <= asp <= aspect[1]):
            continue
        out.append((a, (float(cents[i][0]), float(cents[i][1])), (x, y, w, h), fill, asp))
    out.sort(key=lambda c: -c[0])
    return out


def detect_marker(rgb, kind: str, color=None, expect_px=None):
    """按标记类型检测。返回 (center_px, info_dict) 或 (None, None)"""
    if kind == "panel":
        cands = detect_dark_panel(rgb)
        if not cands:
            return None, None
        if expect_px is not None:
            cands.sort(key=lambda c: np.hypot(c[1][0] - expect_px[0],
                                             c[1][1] - expect_px[1]))
        a, c, bbox, fill, asp = cands[0]
        return np.asarray(c, float), dict(kind="panel", area=a, bbox=bbox,
                                          fill=fill, aspect=asp)
    cands = find_cubes(rgb)
    if not cands:
        return None, None
    if color:
        sel = [c for c in cands if c.color == color]
        cands = sel or cands
    if expect_px is not None:
        cands.sort(key=lambda c: np.hypot(c.center[0] - expect_px[0],
                                         c.center[1] - expect_px[1]))
    cub = cands[0]
    return np.asarray(cub.center, float), dict(kind="cube", color=cub.color, area=cub.area,
                                               bbox=list(cub.bbox), fill=cub.extent,
                                               aspect=0.0)


def stable_marker(cam: Cam, kind: str, color=None, expect_px=None,
                  n=FRAMES_PER_POINT):
    """连续取 n 帧取中位。返回 (center, spread_px, info)"""
    pts, last = [], None
    for _ in range(n):
        c, info = detect_marker(cam.grab(), kind, color, expect_px)
        if c is not None:
            pts.append(c)
            last = info
            expect_px = c
        time.sleep(0.03)
    if not pts:
        return None, float("nan"), None
    P = np.array(pts)
    return np.median(P, axis=0), float(np.max(np.linalg.norm(P - P.mean(0), axis=1))), last


# ---------------------------------------------------------------------------
# 单应
# ---------------------------------------------------------------------------
def fit_homography(px, xy, ransac_mm: float = 5.0):
    import cv2
    px = np.asarray(px, float).reshape(-1, 2)
    xy = np.asarray(xy, float).reshape(-1, 2)
    if px.shape[0] < 4:
        raise ValueError("至少需要 4 个点，当前 %d" % px.shape[0])
    H, mask = cv2.findHomography(px, xy, cv2.RANSAC, ransac_mm / 1000.0)
    inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(px), bool)
    if inl.sum() >= 4:
        H2, _ = cv2.findHomography(px[inl], xy[inl], 0)
        if H2 is not None:
            H = H2
    res = np.linalg.norm(apply_h(H, px) - xy, axis=1) * 1000.0
    return H, inl, res


def apply_h(H, px):
    px = np.asarray(px, float).reshape(-1, 2)
    v = np.hstack([px, np.ones((px.shape[0], 1))]) @ np.asarray(H, float).T
    return v[:, :2] / v[:, 2:3]


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def snake_points(x0, x1, y0, y1, n, z):
    """n×n 网格 + **蛇形**顺序：相邻点距离最小，利于热启动。

    按行正序遍历会在每行末尾横跨整行，那正是卡死的高发点。
    """
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    pts = []
    for k, y in enumerate(ys):
        for x in (xs if k % 2 == 0 else xs[::-1]):
            pts.append((float(x), float(y), float(z)))
    return pts


def demo_q_and_pose(kin, data_dir: Path):
    """取演示数据里一个真实构型与末端位姿（离线预检 / 参考姿态用）"""
    for f in sorted(data_dir.glob("episode_*.json"))[:4]:
        try:
            frames = json.loads(f.read_text(encoding="utf-8")).get("frames", [])
        except Exception:
            continue
        if frames:
            q = np.deg2rad([frames[len(frames) // 2]["F%d" % i] for i in range(1, 7)])
            return q, kin.fk(q)
    return None, None


def dataset_hint(kin, data_dir: Path, n_ep: int = 8) -> dict:
    """从采集数据统计 TCP 实际 XY 分布 —— 那些位置相机一定看得见（图里有）"""
    P = []
    for f in sorted(data_dir.glob("episode_*.json"))[:n_ep]:
        try:
            frames = json.loads(f.read_text(encoding="utf-8")).get("frames", [])
        except Exception:
            continue
        for fr in frames[::15]:
            P.append(kin.fk_pos(np.deg2rad([fr["F%d" % i] for i in range(1, 7)])))
    if not P:
        return {}
    P = np.array(P)
    out = {}
    print("\n=== 数据集提示：采集时 TCP 实际到过的区域（相机必定看得见）===")
    for i, ax in enumerate("xyz"):
        q = np.percentile(P[:, i], [1, 50, 99])
        out[ax] = [float(v) for v in q]
        print("  %s: p1 %+.3f  中位 %+.3f  p99 %+.3f  (m)" % (ax, *q))
    return out


def make_dik(kin, x_ref=None):
    return DownIK(kin, x_ref=x_ref)


def feasibility_sweep(dik: DownIK, q_seed, pts) -> dict:
    """纯运动学地把 pts 走一遍（渐近 + 蛇形），返回可达性与残差统计。

    ``cmd_plan``（离线）与 ``cmd_calibrate``（上机前）共用同一套判据 —— 避免
    "离线说能跑、上机才发现不行"。
    """
    q = np.asarray(q_seed, float).copy()
    n_ok, bad, worst, worst_ax = 0, [], 0.0, 0.0
    yaws = []
    for i, (x, y, z) in enumerate(pts):
        ok, why, q2, res = ik_path(dik, q, (x, y, z))
        if ok:
            n_ok += 1
            q = q2
            worst = max(worst, res[-1][1][1])
            worst_ax = max(worst_ax, res[-1][1][2])
            yaws.append(res[-1][1][3])
        else:
            bad.append((i, x, y, why))
    span = (max(yaws) - min(yaws)) if yaws else float("nan")
    return dict(n_ok=n_ok, n=len(pts), bad=bad, worst_pos=worst, worst_ax=worst_ax,
                yaw_span=span,
                c_drift_mm=3.0 * 2 * np.sin(np.deg2rad(span) / 2) if yaws else float("nan"))


def _print_feasibility(r: dict) -> None:
    print("  可达且在限位内: %d/%d" % (r["n_ok"], r["n"]))
    print("  残差 max: 位置 %.2e m（容差 %.0e）  轴 %.2e rad（容差 %.0e）"
          % (r["worst_pos"], IK_POS_TOL, r["worst_ax"], IK_ROT_TOL))
    if r["yaw_span"] == r["yaw_span"]:
        print("  偏航跨度 %.3f°  → 若 |c_横向|=3mm，标记漂移 ≈ %.3f mm"
              % (r["yaw_span"], r["c_drift_mm"]))
    for b in r["bad"][:12]:
        print("    ✗ #%d (%.3f, %+.3f) %s" % b)


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------
def cmd_plan(kin, args) -> int:
    print("=" * 68)
    print("P1 离线自检（不动机器人）")
    print("=" * 68)
    hint = dataset_hint(kin, args.data)
    q_demo, T_demo = demo_q_and_pose(kin, args.data)
    if q_demo is None:
        print("⚠ 无演示数据，用行程中点起种子（结论仅供参考）")
        q_demo = 0.5 * (kin.q_lo + kin.q_hi)
    else:
        x_ref = T_demo[:3, 0].copy()
        x_ref[2] = 0.0
        x_ref /= np.linalg.norm(x_ref)
        print("\n参考姿态（演示数据）位置 %s" % np.round(T_demo[:3, 3], 4).tolist())
        print("  工具 z 轴 %s（应≈(0,0,-1)）" % np.round(T_demo[:3, 2], 4).tolist())

    dik = make_dik(kin, x_ref if q_demo is not None else None)
    x0, x1, y0, y1 = args.region
    pts = snake_points(x0, x1, y0, y1, args.grid, args.z)
    print("\n=== 标定点可达性预检（渐近接近 + 蛇形遍历，模拟真实工具）===")
    print("  区域 x[%.3f, %.3f] y[%+.3f, %+.3f] z=%.3f  共 %d 点（%d×%d）"
          % (x0, x1, y0, y1, args.z, len(pts), args.grid, args.grid))
    q = np.asarray(q_demo, float).copy()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ok, why, q, _ = ik_path(dik, q, (cx, cy, args.z), steps=20)
    if not ok:
        print("  ✗ 连区域中心都到不了: %s" % why)
        return 2
    print("  到达区域中心 ✓")

    r = feasibility_sweep(dik, q, pts)
    _print_feasibility(r)
    if r["n_ok"] == r["n"]:
        print("  ✅ 全部可达 → 可以上实机")
    elif hint:
        print("\n  建议把区域收缩到演示数据覆盖范围: --region %.3f %.3f %.3f %.3f"
              % (hint["x"][0], hint["x"][1], hint["y"][0], hint["y"][1]))
    return 0 if r["n_ok"] == r["n"] else 2

def cmd_check(kin, args) -> int:
    cam = Cam()
    try:
        rgb = cam.grab()
        kind = args.marker
        cands = detect_dark_panel(rgb) if kind == "panel" else find_cubes(rgb)
        print("标记类型=%s，候选 %d 个" % (kind, len(cands)))
        for c in cands[:8]:
            if kind == "panel":
                print("   area %5d  bbox %s  填充率 %.2f  长宽比 %.2f  质心 (%.0f,%.0f)"
                      % (c[0], c[2], c[3], c[4], c[1][0], c[1][1]))
            else:
                print("   %-7s 质心 (%3.0f,%3.0f)  面积 %4d  边长 %.1f"
                      % (c.color, c.center[0], c.center[1], c.area, c.side_px))
        if args.snap:
            import cv2
            from perception.cube_locator import draw_candidates
            args.snap.parent.mkdir(parents=True, exist_ok=True)
            vis = draw_candidates(rgb, cands) if kind == "cube" else rgb.copy()
            if kind == "panel":
                for a, cc, bb, fill, asp in cands[:8]:
                    x, y, w, h = bb
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.drawMarker(vis, (int(cc[0]), int(cc[1])), (0, 255, 0),
                                   cv2.MARKER_CROSS, 11, 2)
            cv2.imwrite(str(args.snap), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print("  标注图已存 %s" % args.snap)
        c, info = detect_marker(rgb, kind, args.marker_color)
        if c is None:
            print("✗ 没找到标记（类型 %s）" % kind)
            return 2
        print("✓ 标记: %s  像素 (%d, %d)" % (info, c[0], c[1]))
        return 0
    finally:
        cam.close()


def _connect(kin, args):
    from hardware.arm import SO101Arm
    from hardware.feetech_bus import resolve_port
    port = resolve_port("follower", args.port)
    print("从臂串口: %s" % port)
    arm = SO101Arm(port=port, calibration_path=str(args.calib))
    arm.connect(handshake=True)
    q = arm.read_positions()
    T = kin.fk(q)
    print("当前 TCP: x=%+.4f y=%+.4f z=%+.4f m" % tuple(T[:3, 3]))
    print("当前关节(度): %s" % np.round(np.degrees(q), 1).tolist())
    return arm, T


def _xref_from(T):
    x = T[:3, 0].copy()
    x[2] = 0.0
    n = np.linalg.norm(x)
    return x / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def cmd_gripper(kin, args) -> int:
    """只动夹爪一个关节（用来把标记方块放进夹爪 / 夹住）。

    两种模式：
      --gripper W         直接写到宽度 W（米）
      --grip-until-contact 从当前开度**逐步闭合**，读到负载上升就停

    为什么需要后一种：``gripper_width`` 把米制宽度按 **0.08 m 满量程线性**折算到 raw 行程，
    但夹爪真实最大开口并未实测过。按不准确的映射一次闭到底，可能夹过头 → 舵机持续堵转
    （虽有 Max_Torque 50% / Protection_Current 50% 保护，仍不该长时间这样）。
    逐步闭合 + 负载判接触是仓库既有的抓取判据（``gripper_current``），也更适合复用。
    """
    from hardware.arm import SO101Arm
    from hardware.feetech_bus import resolve_port
    port = resolve_port("follower", args.port)
    arm = SO101Arm(port=port, calibration_path=str(args.calib))
    arm.connect(handshake=True)
    try:
        cal = arm.calibration["6"]
        rmin, rmax = int(cal["range_min"]), int(cal["range_max"])

        def show(tag):
            q = arm.read_positions()
            raw = arm._rad_to_raw(6, q[5])
            d = arm.bus.read_diagnostics([6]).get(6, {})
            load = d.get("load")
            print("  %-8s 角度 %+7.2f°  raw %4d  负载 %s  电流 %s mA  温度 %s"
                  % (tag, np.degrees(q[5]), raw,
                     ("%.1f%% %s" % load) if load else "n/a",
                     d.get("current_mA"), d.get("temperature")))
            return raw, (load[0] if load else 0.0)

        print("夹爪标定行程 %d~%d（越小越闭）" % (rmin, rmax))
        show("闭合前")

        if args.grip_until_contact:
            print(">>> 逐步闭合直到接触（负载 ≥ %.1f%% 即停）" % args.contact_load)
            hit = False
            for i in range(24):
                w = 0.060 - i * 0.004
                if w < 0.002:
                    break
                arm.gripper_width(w)
                time.sleep(0.7)
                raw, load = show("w=%.3f" % w)
                if load >= args.contact_load:
                    print("  ✅ 检测到接触（负载 %.1f%% ≥ %.1f%%）" % (load, args.contact_load))
                    hit = True
                    break
                if raw <= rmin + 5:
                    print("  ⚠ 已到闭合行程下限仍未检测到负载上升")
                    break
            if hit and args.backoff > 0:
                # 关键：接触后必须**回退一点**，否则舵机持续堵转。
                # 实测教训：不回退导致夹爪在 21.6% 负载 / 143 mA 下堵转约 3 分钟，
                # 温度 37→45℃，舵机锁死 Overload 保护态（寻址 ping 不再应答，
                # 只能读寄存器；须断电重上电才能恢复）。
                back = 0.002 + args.backoff
                print(">>> 回退 %.4f m 释放堵转（必须做，否则会再次锁过载）" % (args.backoff))
                q = arm.read_positions()
                raw_now = arm._rad_to_raw(6, q[5])
                target_raw = min(rmax, raw_now + 62)   # 62 count ≈ 5.5°，够松开夹持力
                arm.bus.write("Goal_Position", 6, target_raw)
                time.sleep(0.8)
                show("回退后")
                print("  提示：夹持已放松。**不要**让它长时间保持夹紧 —— 过载保护会锁死。")
        else:
            print(">>> 写入宽度 %.4f m" % args.gripper)
            arm.gripper_width(args.gripper)
            time.sleep(0.9)
            raw, load = show("闭合后")
            if load >= args.contact_load:
                print("  ✅ 负载 %.1f%% → 已夹住东西" % load)
            elif raw <= rmin + 5:
                print("  ⚠ 已到行程下限、负载仍为 0 → 夹爪是空的")
        print("（若没到位可再执行一次）")
        return 0
    finally:
        try:
            arm.bus.disconnect(disable_torque=False)   # 保持扭矩，别把夹爪松开
            print("串口已关闭（扭矩状态保持不变）")
        except Exception:
            pass


def cmd_status(kin, args) -> int:
    """只读：读从臂当前姿态 + 判断"从这里出发能不能走完整片区域"。

    **不改扭矩状态** —— 常规 disconnect() 会禁扭矩让机械臂松脱下垂，
    在"只是看一眼"的场景里那是危险的（臂可能举着东西）。这里显式保持扭矩不变。
    """
    from hardware.arm import SO101Arm
    from hardware.feetech_bus import resolve_port
    port = resolve_port("follower", args.port)
    arm = SO101Arm(port=port, calibration_path=str(args.calib))
    arm.connect(handshake=True)
    try:
        q = arm.read_positions()
        T = kin.fk(q)
        x_ref = _xref_from(T)
        print("从臂串口: %s" % port)
        print("当前关节(度): %s" % np.round(np.degrees(q), 1).tolist())
        print("当前 TCP: x=%+.4f y=%+.4f z=%+.4f m" % tuple(T[:3, 3]))
        print("工具 z 轴: %s   （夹爪朝下应≈(0,0,-1)）" % np.round(T[:3, 2], 4).tolist())
        print("工具 x 轴水平投影(偏航参考): %s" % np.round(x_ref, 4).tolist())
        ok_lim, viol = kin.limit_violation(q)
        print("关节是否全在标定限位内: %s"
              % ("是" if ok_lim else "否（越限 %.3e rad）" % viol))
        x0, x1, y0, y1 = args.region
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        d = float(np.hypot(T[0, 3] - cx, T[1, 3] - cy))
        print("\n区域中心 (%.3f, %+.3f)  z=%.3f ；当前离中心水平距离 %.1f cm"
              % (cx, cy, args.z, d * 100))
        print("当前 TCP z=%.4f → 目标 z=%.4f" % (T[2, 3], args.z))
        pts = snake_points(x0, x1, y0, y1, args.grid, args.z)
        print("\n=== 从当前位置出发的可行性（%d 点，不写舵机）===" % len(pts))
        if d > 0.10:
            print("  ⚠ 离区域中心 >10 cm，直接渐近过去可能不可达 —— "
                  "建议先把臂摆到区域中心附近（夹爪朝下）再跑 --calibrate")
        # 必须和 cmd_calibrate 的流程一致：**先渐近到区域中心**，再从那里扫掠。
        # 否则从远处当前姿态直接跳网格角点会失败，给出假警报（踩过）。
        dik = make_dik(kin, x_ref)
        ok_c, why_c, q_c, _ = ik_path(dik, q, (cx, cy, args.z), steps=20)
        if not ok_c:
            print("  ✗ 连区域中心都到不了: %s" % why_c)
            print("\n结论: 需要先摆臂或改区域 ❌")
            return 2
        print("  渐近到区域中心 ✓（此后从中心扫掠）")
        r = feasibility_sweep(dik, q_c, pts)
        _print_feasibility(r)
        print("\n结论: %s" % ("可以跑 --calibrate ✅" if r["n_ok"] == r["n"]
                              else "需先调整区域或摆姿（见上面 ✗）"))
        return 0 if r["n_ok"] == r["n"] else 2
    finally:
        # 关键：不禁扭矩（只看不动，不能把臂放瘫）
        try:
            arm.bus.disconnect(disable_torque=False)
            print("串口已关闭（扭矩状态保持不变）")
        except Exception:
            pass


def cmd_calibrate(kin, args) -> int:
    if not args.confirm_motion:
        print("✗ 该命令会驱动机器人。确认现场安全后加 --confirm-motion")
        return 2
    arm, T_now = _connect(kin, args)
    x_ref = _xref_from(T_now)
    dik = make_dik(kin, x_ref)
    print("偏航参考方向（工具 x 轴水平投影）: %s" % np.round(x_ref, 4).tolist())

    # 上机前先做一次**基于从臂当前姿态**的可行性预检，任何舵机动作之前。
    # 离线预检用的是演示姿态；实机姿态不同，偏航约束锁定的方向也不同，
    # 因此必须用真实起点复核一遍，否则可能出现"离线说能跑、上机卡住"。
    x0, x1, y0, y1 = args.region
    pts = snake_points(x0, x1, y0, y1, args.grid, args.z)
    print("\n=== 上机前可行性预检（从臂当前姿态，%d 点，不写舵机）===" % len(pts))
    q_now = arm.read_positions()
    r = feasibility_sweep(make_dik(kin, x_ref), q_now, pts)
    _print_feasibility(r)
    if r["n_ok"] != r["n"]:
        print("\n✗ 有 %d 个点不可达。未驱动任何舵机。" % (r["n"] - r["n_ok"]))
        print("  处理：① 按上面提示缩小 --region；② 或把从臂摆到更接近区域中心的姿态"
              "（夹爪朝下）后重试；③ 或加 --force 跳过预检（不推荐）")
        if not args.force:
            arm.disconnect()
            return 2
        print("  --force 已指定，继续。")
    else:
        print("  ✅ 预检通过")

    cam = Cam()
    try:
        x0, x1, y0, y1 = args.region
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        print("\n>>> 渐近移动到区域中心（x=%.3f y=%.3f z=%.3f）" % (cx, cy, args.z))
        ok, why = goto(dik, arm, (cx, cy, args.z))
        if not ok:
            print("✗ 移动失败: %s" % why)
            return 2
        time.sleep(SETTLE_S)
        c, spread, info = stable_marker(cam, args.marker, args.marker_color)
        if c is None:
            print("✗ 中心处看不到标记。确认方块在夹爪里且在相机视野内。")
            return 2
        print("  标记像素 (%d, %d)  抖动 %.2f px  %s" % (c[0], c[1], spread, info))
        expect = c

        pts = snake_points(x0, x1, y0, y1, args.grid, args.z)
        print("\n>>> 扫掠 %d 个标定点" % len(pts))
        px_list, xy_list, rows = [], [], []
        for i, (x, y, z) in enumerate(pts, 1):
            ok, why = goto(dik, arm, (x, y, z))
            if not ok:
                print("  [%2d/%d] (%.3f,%+.3f) 跳过: %s" % (i, len(pts), x, y, why))
                rows.append(dict(i=i, x=x, y=y, ok=False, reason=why))
                continue
            time.sleep(SETTLE_S)
            c, spread, info = stable_marker(cam, args.marker, args.marker_color,
                                            expect_px=expect)
            if c is None:
                print("  [%2d/%d] (%.3f,%+.3f) 跳过: 看不到标记" % (i, len(pts), x, y))
                rows.append(dict(i=i, x=x, y=y, ok=False, reason="marker_not_found"))
                continue
            expect = c
            q_act = arm.read_positions()
            T_act = kin.fk(q_act)
            yaw = float(np.degrees(np.arctan2(T_act[1, 0], T_act[0, 0])))
            px_list.append(c)
            xy_list.append([x, y])
            fk_mm = float(np.hypot(T_act[0, 3] - x, T_act[1, 3] - y) * 1000)
            rows.append(dict(i=i, x=x, y=y, ok=True, px=c.tolist(), spread_px=spread,
                             marker=info, yaw_deg=yaw,
                             fk_xy=[float(T_act[0, 3]), float(T_act[1, 3])],
                             fk_err_mm=fk_mm))
            print("  [%2d/%d] (%.3f,%+.3f) px=(%4d,%4d) 抖动%.2f 偏航%+7.2f°  舵机偏差 %5.2f mm"
                  % (i, len(pts), x, y, c[0], c[1], spread, yaw, fk_mm))

        good = [r for r in rows if r.get("ok")]
        if len(good) < 4:
            print("\n✗ 有效点只有 %d 个，不足以拟合单应（需 ≥4）" % len(good))
            return 2
        H, inl, res = fit_homography(np.array(px_list), np.array(xy_list),
                                     args.ransac_mm)
        yaws = np.array([r["yaw_deg"] for r in good])
        print("\n=== 标定结果 ===")
        print("  有效点 %d，RANSAC 内点 %d" % (len(good), int(np.sum(inl))))
        print("  重投影残差: 中位 %.2f mm  p90 %.2f mm  max %.2f mm"
              % (np.median(res), np.percentile(res, 90), res.max()))
        print("  偏航跨度 %.3f°（应很小 —— 它是「c 恒定」前提的量化指标）"
              % (yaws.max() - yaws.min()))
        print("  舵机到位偏差（不含相机）: 中位 %.2f mm  max %.2f mm"
              % (np.median([r["fk_err_mm"] for r in good]),
                 np.max([r["fk_err_mm"] for r in good])))
        print("  单应矩阵 H（像素 → 基座 XY, 米）:")
        for row in H:
            print("    [%+.6e %+.6e %+.6e]" % tuple(row))

        out = dict(
            kind="planar_homography_pixel_to_base_xy",
            H=H.tolist(), created=time.strftime("%Y-%m-%d %H:%M:%S"),
            z_m=float(args.z), x_ref=x_ref.tolist(), yaw_weight=YAW_WEIGHT,
            marker_kind=args.marker, marker_color=args.marker_color,
            region=list(map(float, args.region)),
            grid=int(args.grid), calib_fingerprint=kin.calib_fingerprint,
            urdf=str(kin.urdf), n_points=len(good), n_inliers=int(np.sum(inl)),
            yaw_span_deg=float(yaws.max() - yaws.min()),
            residual_mm=dict(median=float(np.median(res)),
                             p90=float(np.percentile(res, 90)), max=float(res.max())),
            fk_err_mm=dict(median=float(np.median([r["fk_err_mm"] for r in good])),
                           max=float(np.max([r["fk_err_mm"] for r in good]))),
            points=rows,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("  已写 %s" % args.out)
        print("\n  提示: 舵机偏差 = 舵机实际到位 vs 下发目标（不含相机）；"
              "重投影残差 还含标记检测噪声。")
        return 0 if np.median(res) <= 5.0 else 2
    finally:
        cam.close()
        try:
            arm.disconnect()
            print("从臂已断开（扭矩已释放，机械臂会松脱下垂，请扶住）")
        except Exception:
            pass


def cmd_verify(kin, args) -> int:
    if not args.out.exists():
        print("✗ 还没有标定结果 %s，先跑 --calibrate" % args.out)
        return 2
    cal = json.loads(args.out.read_text(encoding="utf-8"))
    if cal.get("calib_fingerprint") != kin.calib_fingerprint:
        print("⚠ 标定文件指纹变了（标定时 %s，当前 %s）→ 标定已失效，请重新标定"
              % (cal.get("calib_fingerprint"), kin.calib_fingerprint))
        return 2
    H = np.array(cal["H"], float)
    x_ref = np.array(cal.get("x_ref", [1.0, 0.0, 0.0]), float)
    z = float(cal["z_m"])

    if not args.confirm_motion:
        print("✗ 该命令会驱动机器人。确认现场安全后加 --confirm-motion")
        return 2
    arm, _ = _connect(kin, args)
    dik = make_dik(kin, x_ref)
    cam = Cam()
    try:
        x0, x1, y0, y1 = cal["region"]
        rng = np.random.default_rng(args.seed)
        pts = []
        for _ in range(args.points * 8):
            if len(pts) >= args.points:
                break
            x, y = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
            if all(np.hypot(x - a, y - b) > 0.02 for a, b, _ in pts):
                pts.append((x, y, z))
        print("\n>>> 验证 %d 个随机目标点（随机且相互间隔 >2 cm，不是标定点）" % len(pts))

        goto(dik, arm, ((x0 + x1) / 2, (y0 + y1) / 2, z))
        time.sleep(SETTLE_S)
        mkind = cal.get("marker_kind", "cube")
        mcolor = cal.get("marker_color")
        expect, _, _ = stable_marker(cam, mkind, mcolor)
        if expect is None:
            print("✗ 看不到标记")
            return 2

        rows = []
        for i, (x, y, zz) in enumerate(pts, 1):
            ok, why = goto(dik, arm, (x, y, zz))
            if not ok:
                print("  [%2d/%d] 跳过: %s" % (i, len(pts), why))
                continue
            time.sleep(SETTLE_S)
            c, spread, info = stable_marker(cam, mkind, mcolor, expect_px=expect)
            if c is None:
                print("  [%2d/%d] 跳过: 看不到标记" % (i, len(pts)))
                continue
            expect = c
            pred = apply_h(H, c.reshape(1, 2))[0]
            dx, dy = float(pred[0] - x), float(pred[1] - y)
            err = float(np.hypot(dx, dy) * 1000)
            T_act = kin.fk(arm.read_positions())
            fk_mm = float(np.hypot(T_act[0, 3] - x, T_act[1, 3] - y) * 1000)
            rows.append(dict(i=i, cmd=[x, y, zz], px=c.tolist(),
                             cam_xy=[float(pred[0]), float(pred[1])], err_mm=err,
                             dx_mm=dx * 1000, dy_mm=dy * 1000, fk_err_mm=fk_mm,
                             spread_px=spread))
            print("  [%2d/%d] 目标(%.3f,%+.3f) 相机(%.3f,%+.3f) 偏差 %6.2f mm"
                  "  (dx%+6.2f dy%+6.2f)  舵机 %5.2f mm"
                  % (i, len(pts), x, y, pred[0], pred[1], err,
                     dx * 1000, dy * 1000, fk_mm))

        if not rows:
            print("✗ 没有有效验证点")
            return 2
        e = np.array([r["err_mm"] for r in rows])
        f = np.array([r["fk_err_mm"] for r in rows])
        summary = dict(n=len(rows), median_mm=float(np.median(e)),
                       p90_mm=float(np.percentile(e, 90)), max_mm=float(e.max()),
                       mean_dx_mm=float(np.mean([r["dx_mm"] for r in rows])),
                       mean_dy_mm=float(np.mean([r["dy_mm"] for r in rows])),
                       fk_median_mm=float(np.median(f)), fk_max_mm=float(f.max()))
        print("\n=== P1 定位精度 ===")
        print("  相机实测偏差: 中位 %.2f mm  p90 %.2f mm  max %.2f mm（%d 点）"
              % (summary["median_mm"], summary["p90_mm"], summary["max_mm"], summary["n"]))
        print("  系统性偏移: dx %+.2f mm  dy %+.2f mm"
              % (summary["mean_dx_mm"], summary["mean_dy_mm"]))
        print("  舵机到位偏差（不含相机）: 中位 %.2f mm  max %.2f mm"
              % (summary["fk_median_mm"], summary["fk_max_mm"]))
        c1 = summary["median_mm"] <= 5.0
        c2 = summary["max_mm"] <= 10.0
        print("\n  判据: ① 中位 ≤5 mm %s   ② max ≤10 mm（工程余量） %s"
              % ("✓" if c1 else "✗", "✓" if c2 else "✗"))
        print("  结论: P1 %s" % ("通过 ✅" if (c1 and c2) else "未通过 ❌"))
        if args.json_out:
            args.json_out.write_text(json.dumps(
                dict(summary=summary, rows=rows,
                     calib={k: v for k, v in cal.items() if k != "points"}),
                ensure_ascii=False, indent=1), encoding="utf-8")
            print("  已写 %s" % args.json_out)
        return 0 if (c1 and c2) else 2
    finally:
        cam.close()
        try:
            arm.disconnect()
            print("从臂已断开（扭矩已释放，请扶住机械臂）")
        except Exception:
            pass


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="P1 手眼标定 + 定位精度验证")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan-only", action="store_true", help="离线自检（不动机器人）")
    g.add_argument("--check", action="store_true", help="只读检查相机能否看到标记")
    g.add_argument("--status", action="store_true",
                   help="只读：读从臂当前姿态 + 从该姿态出发的可行性（不改扭矩）")
    g.add_argument("--gripper", type=float, metavar="W_M",
                   help="只动夹爪：把夹爪开到宽度 W_M 米（注意米制映射是近似的）")
    g.add_argument("--grip-until-contact", action="store_true",
                   help="只动夹爪：逐步闭合直到负载上升（夹住标记方块，比按宽度闭更安全）")
    ap.add_argument("--contact-load", type=float, default=5.0,
                    help="判定「已接触」的负载阈值（%%，默认 5）")
    ap.add_argument("--backoff", type=float, default=0.004,
                    help="检测到接触后回退的宽度（米），用于释放堵转，默认 4 mm")
    g.add_argument("--calibrate", action="store_true", help="扫掠标定（动机器人）")
    g.add_argument("--verify", action="store_true", help="定位精度验证（动机器人）")
    ap.add_argument("--region", type=float, nargs=4,
                    default=[DEF_X[0], DEF_X[1], DEF_Y[0], DEF_Y[1]],
                    metavar=("X0", "X1", "Y0", "Y1"), help="扫掠区域（米）")
    ap.add_argument("--z", type=float, default=DEF_Z, help="扫掠高度 TCP z（米）")
    ap.add_argument("--grid", type=int, default=3, help="标定网格边长（3=9 点）")
    ap.add_argument("--points", type=int, default=10, help="验证点数")
    ap.add_argument("--marker", choices=["panel", "cube"], default="panel",
                    help="标记类型。panel=夹爪上的黑色矩形面板（默认，无需额外物品，"
                         "实测方块被夹爪挡住看不见）；cube=夹在夹爪里的彩色方块")
    ap.add_argument("--marker-color", default=None,
                    help="--marker cube 时指定方块颜色（red/orange/...）；默认自动")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--calib", type=Path, default=REPO / "config" / "calibration.json")
    ap.add_argument("--data", type=Path, default=REPO / "data" / "raw" / "pick_place")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--ransac-mm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snap", type=Path, default=None,
                    help="--check 时把标注图存到该路径（便于人眼复核相机看到的场景）")
    ap.add_argument("--confirm-motion", action="store_true",
                    help="确认现场安全、允许驱动机器人（--calibrate/--verify 必需）")
    ap.add_argument("--force", action="store_true",
                    help="跳过上机前可行性预检（不推荐，仅在明确知道风险时用）")
    args = ap.parse_args()

    t0 = time.time()
    kin = So101Kinematics(calib_path=args.calib)
    print("运动学就绪: %s | 标定指纹 %s | 加载 %.0f ms"
          % (kin.urdf.name, kin.calib_fingerprint, (time.time() - t0) * 1000))

    if args.plan_only:
        return cmd_plan(kin, args)
    if args.check:
        return cmd_check(kin, args)
    if args.status:
        return cmd_status(kin, args)
    if args.gripper is not None or args.grip_until_contact:
        return cmd_gripper(kin, args)
    if args.calibrate:
        return cmd_calibrate(kin, args)
    return cmd_verify(kin, args)


if __name__ == "__main__":
    raise SystemExit(main())
