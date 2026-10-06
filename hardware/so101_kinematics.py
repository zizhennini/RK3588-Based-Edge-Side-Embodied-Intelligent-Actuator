# hardware/so101_kinematics.py
"""SO-101 运动学（placo / pinocchio）—— 「VLM + 外部 IK」路线的运动学基座

设计依据 docs/测试方案_VLM+外部IK+抓取.md（P0-2）。三条板端实测结论驱动了本实现：

1. **URDF 必须带「本机标定限位」**
   官方 ``so101_new_calib.urdf`` 里的 ``<limit>`` 是 SO-101 的标称值
   （shoulder_pan ±110°、gripper -10°~100° 等），与本机实测行程不同
   （本项目实测 ±119.4° / ±104.9° / ±95.4° / ±103.2° / ±180° / -54.1°~+76.6°）。
   IK 若不带上真实限位，会解出实机到不了的角 → 舵机堵转发热。
   故本模块从 ``config/calibration.json`` 现场生成 ``so101_calibrated.urdf``。

2. **TCP 是 frame，不是关节名**
   ``gripper`` 是**关节**；``add_frame_task("gripper")`` 会拿到错误位姿。
   正确 TCP 是 ``gripper_frame_link``（固定在 gripper_link 上的 dummy link，
   ``gripper_frame_joint`` 把它接在 gripper_link 上，与 lerobot 约定一致）。

3. **placo 的 QP 是局部求解器，必须迭代**
   单次 ``solve()`` 只走一小步：从行程中点一次迭代剩 ~9 cm 误差（实测）。
   迭代到收敛后中位耗时 0.10 ms（±5° 起点）/ 0.65 ms（±90° 起点）。

坐标/角度约定（已用 53 条真实采集数据正解验证，见 docs）：
  - q[0:7] = 浮动基座 (x,y,z,qx,qy,qz,qw)，用 ``mask_fbase(True)`` 锁死
  - q[7:12] = shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
  - q[12]   = gripper
  - 关节角单位 rad，零点 = 标定行程中点（夹爪用 homing_offset），与数据集
    ``F*``/``J*``（JSON 里存的角度值）经 ``np.deg2rad`` 后完全一致：
    真实数据正解得到桌面 z≈0.000 m、最大臂展 0.327 m、抓取时刻末端高度 +0.014 m。

依赖: ``pip install placo``（aarch64 有预编译 wheel，无需编译）
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 复用总线层的角度换算，避免两处约定漂移（feetech_bus 模块级只依赖 stdlib+numpy）
from hardware.feetech_bus import GRIPPER_MOTOR_ID, raw_to_rad

#: 仓库根目录（hardware/ 的上一级）
REPO_ROOT = Path(__file__).resolve().parent.parent
#: 官方 URDF（限位为标称值，本模块不直接使用）
DEFAULT_URDF_SRC = REPO_ROOT / "models" / "so101_urdf" / "so101_new_calib.urdf"
#: 从臂标定（range_min/range_max/homing_offset）
DEFAULT_CALIB = REPO_ROOT / "config" / "calibration.json"
#: 生成物：带本机标定限位的 URDF
DEFAULT_DERIVED_URDF = REPO_ROOT / "models" / "so101_urdf" / "so101_calibrated.urdf"

#: 关节顺序（与数据集 action/state 的 6 维顺序一致，官方核实）
JOINT_NAMES: Tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
#: 工具坐标系（TCP）frame 名 —— 注意不是关节名 "gripper"
TCP_FRAME = "gripper_frame_link"

#: 默认收敛判据
POS_TOL_M = 1e-4      # 0.1 mm
ROT_TOL_RAD = 1e-3    # ≈0.057°
#: 默认最大迭代（50 次足以覆盖 ±90° 起点；实测中位 2~15 次）
MAX_ITER = 50


# ---------------------------------------------------------------------------
# 标定限位换算
# ---------------------------------------------------------------------------
def calibration_fingerprint(calib_path: Path = DEFAULT_CALIB) -> str:
    """标定文件短指纹（sha1 前 12 位）

    为什么需要：``config/calibration.json`` 是**每次实机标定的产物**，部署目录
    （板端 / PC）通常不是 git 仓库，该文件由标定工具运行时写入，因此不同机器上
    可能是不同版本（实测：板端 J1 行程 2717 counts，另一台 2341 counts）。
    日志里带上指纹，才能在出问题时区分「IK 算错」还是「标定文件不对」。
    """
    import hashlib
    data = Path(calib_path).read_bytes()
    return hashlib.sha1(data).hexdigest()[:12]


def joint_limits_rad(calib_path: Path = DEFAULT_CALIB,
                     joint_names: Sequence[str] = JOINT_NAMES
                     ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """从标定文件读各关节行程 → (lo6, hi6) 弧度限位 + 明细

    零点约定与数据集一致：普通关节用「行程中点」，夹爪用 ``homing_offset``
    （见 ``feetech_bus.angle_zero`` 的说明）。
    """
    cal = json.loads(Path(calib_path).read_text(encoding="utf-8"))
    lo = np.zeros(len(joint_names))
    hi = np.zeros(len(joint_names))
    detail: List[dict] = []
    for i, jname in enumerate(joint_names, start=1):
        entry = cal.get(str(i))
        if not isinstance(entry, dict):
            raise KeyError(f"标定文件缺少 motor {i}（{jname}）: {calib_path}")
        use_mid = (i != GRIPPER_MOTOR_ID)
        raw_lo = float(entry["range_min"])
        raw_hi = float(entry["range_max"])
        a = raw_to_rad(raw_lo, entry, use_range_midpoint=use_mid)
        b = raw_to_rad(raw_hi, entry, use_range_midpoint=use_mid)
        lo[i - 1], hi[i - 1] = (a, b) if a <= b else (b, a)
        detail.append(dict(id=i, joint=jname, raw_min=raw_lo, raw_max=raw_hi,
                           lo_rad=lo[i - 1], hi_rad=hi[i - 1],
                           span_deg=float(np.degrees(hi[i - 1] - lo[i - 1]))))
    return lo, hi, detail


# ---------------------------------------------------------------------------
# URDF 生成（注入本机标定限位）
# ---------------------------------------------------------------------------
#: 只匹配「顶层关节」——要求开标签里带 ``type=``，从而排除 <transmission> 内的
#: ``<joint name="X">`` 子元素（官方 URDF 里每个关节名因此出现 2 次）
_JOINT_RE = r'<joint\s+(?=[^>]*\bname="{name}")(?=[^>]*\btype=")[^>]*>(.*?)</joint>'
_LIMIT_RE = re.compile(r"<limit[^>]*/>")
_VISUAL_RE = re.compile(r"<visual>.*?</visual>", re.S)
_COLLISION_RE = re.compile(r"<collision>.*?</collision>", re.S)
_MESH_RE = re.compile(r"<mesh[^>]*/>")


def build_calibrated_urdf(src: Path = DEFAULT_URDF_SRC,
                          calib_path: Path = DEFAULT_CALIB,
                          dst: Path = DEFAULT_DERIVED_URDF,
                          strip_meshes: bool = True) -> dict:
    """把标定限位注入官方 URDF，写出运动学专用 URDF

    Args:
        strip_meshes: 去掉 ``<visual>/<collision>/<mesh>``。
            placo 加载时会去磁盘找 ``assets/*.stl``，缺失即报
            ``ValueError Mesh assets/... could not be found``（板端实测）。
            IK/FK 只需要运动链，故默认剥离。

    Returns:
        报告 dict（限位明细、替换计数、输出路径）
    """
    src, dst = Path(src), Path(dst)
    lo, hi, detail = joint_limits_rad(calib_path)
    txt = src.read_text(encoding="utf-8", errors="ignore")

    if strip_meshes:
        txt = _VISUAL_RE.sub("", txt)
        txt = _COLLISION_RE.sub("", txt)
        txt = _MESH_RE.sub("", txt)

    n_replaced = 0
    for i, jname in enumerate(JOINT_NAMES):
        pat = re.compile(_JOINT_RE.format(name=re.escape(jname)), re.S)

        def _sub(m, lo=lo[i], hi=hi[i]):
            nonlocal n_replaced
            body = _LIMIT_RE.sub("", m.group(1))
            n_replaced += 1
            limit = ('<limit lower="%.6f" upper="%.6f" effort="10" velocity="10"/>'
                     % (lo, hi))
            return m.group(0).replace(m.group(1), body + limit)

        txt = pat.sub(_sub, txt)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(txt, encoding="utf-8")
    return dict(src=str(src), dst=str(dst), replaced=n_replaced, expected=len(JOINT_NAMES),
                strip_meshes=strip_meshes, limits=detail,
                calib_fingerprint=calibration_fingerprint(calib_path))


# ---------------------------------------------------------------------------
# IK 结果
# ---------------------------------------------------------------------------
@dataclass
class IkResult:
    """一次 IK 求解的结果（角度单位 rad，位置单位 m）"""
    q: np.ndarray                    # (6,) 关节角解
    converged: bool                  # 是否达到 pos/rot 容差
    iters: int                       # 实际迭代次数
    ms: float                        # 求解总耗时（毫秒）
    pos_err: float                   # 末端位置误差
    rot_err: float                   # 末端姿态误差
    in_limits: bool = True           # 解是否在标定限位内
    max_limit_violation: float = 0.0  # 最大越限量（rad）
    detail: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 运动学封装
# ---------------------------------------------------------------------------
class So101Kinematics:
    """SO-101 正/逆运动学封装（placo 后端，单一 solver 复用 + 热启动）

    典型用法（实时回路）::

        kin = So101Kinematics()
        r = kin.ik(T_target)            # 每次调用自动热启动上一次的解
        if r.converged and r.in_limits:
            bus.sync_write("Goal_Position", {...})

    或限时单步（控制周期内只算一次 QP 迭代）::

        kin.set_target(T_target)
        r = kin.step()

    Args:
        urdf: 已带标定限位的 URDF；None 时用 ``DEFAULT_DERIVED_URDF``，
              不存在则自动由官方 URDF 生成
        calib_path: 标定文件
        tcp_frame: 工具坐标系 frame 名
        auto_build: URDF 缺失时是否自动生成
    """

    def __init__(self, urdf: Optional[Path] = None,
                 calib_path: Path = DEFAULT_CALIB,
                 tcp_frame: str = TCP_FRAME,
                 auto_build: bool = True):
        try:
            import placo  # 延迟导入：PC 端无 placo 也能 import 本模块
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SO-101 运动学需要 placo: pip install placo"
                "（aarch64 有预编译 wheel，无需编译）") from e
        self._placo = placo
        self.calib_path = Path(calib_path)

        urdf = Path(urdf) if urdf is not None else DEFAULT_DERIVED_URDF
        if not urdf.exists():
            if not auto_build:
                raise FileNotFoundError(f"缺少 URDF: {urdf}")
            build_calibrated_urdf(calib_path=self.calib_path, dst=urdf)

        self.urdf = urdf
        self.tcp_frame = tcp_frame
        self.robot = placo.RobotWrapper(str(urdf))
        self.model = self.robot.model

        # 关节占 q 的最后 6 维（前 7 维是浮动基座）
        nq = int(np.asarray(self.model.lowerPositionLimit).ravel().size)
        if nq < 7:
            raise ValueError(f"模型 q 维数 {nq} 异常，期望 ≥7（浮动基座 7 + 6 关节）")
        self.joint_idx = np.arange(nq - 6, nq)

        frames = [str(f.name) for f in self.model.frames]
        if tcp_frame not in frames:
            raise ValueError(f"URDF 里没有 frame {tcp_frame!r}；可用: {frames}")

        self.q_lo, self.q_hi, self.limit_detail = joint_limits_rad(self.calib_path)
        self.calib_fingerprint = calibration_fingerprint(self.calib_path)

        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)          # 锁死浮动基座：IK 只动 6 个关节
        self.solver.dt = 0.01
        self.solver.enable_joint_limits(True)  # 关键：不调用则 QP 完全忽略限位
        try:
            self.solver.enable_velocity_limits(True)
        except Exception:
            pass
        self.task = self.solver.add_frame_task(self.tcp_frame, np.eye(4))

    # -- 基础 ------------------------------------------------------------
    @property
    def n_joints(self) -> int:
        return 6

    def _set_joints(self, q6: Sequence[float]) -> None:
        q = np.asarray(self.robot.state.q).copy()
        q[self.joint_idx] = np.asarray(q6, dtype=np.float64)
        self.robot.state.q = q

    def joint_angles(self) -> np.ndarray:
        """当前关节角 (6,)"""
        return np.asarray(self.robot.state.q).ravel()[self.joint_idx].copy()

    def clamp(self, q6: Sequence[float], margin: float = 0.0) -> np.ndarray:
        """把关节角夹到标定限位内（margin 为额外内缩量，rad）"""
        return np.clip(np.asarray(q6, dtype=np.float64),
                       self.q_lo + margin, self.q_hi - margin)

    def limit_violation(self, q6: Sequence[float]) -> Tuple[bool, float]:
        """返回 (是否在限位内, 最大越限量 rad)"""
        q = np.asarray(q6, dtype=np.float64)
        v = float(np.maximum(self.q_lo - q, q - self.q_hi).max())
        return v <= 1e-6, v

    # -- 正解 ------------------------------------------------------------
    def fk(self, q6: Sequence[float]) -> np.ndarray:
        """正运动学 → TCP 的 4x4 齐次矩阵（世界系）"""
        self._set_joints(q6)
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.tcp_frame).copy()

    def fk_pos(self, q6: Sequence[float]) -> np.ndarray:
        """正运动学 → TCP 位置 (3,)"""
        return np.asarray(self.fk(q6)[:3, 3])

    # -- 逆解 ------------------------------------------------------------
    @staticmethod
    def _pose_err(T_now: np.ndarray, T_goal: np.ndarray) -> Tuple[float, float]:
        ep = float(np.linalg.norm(T_now[:3, 3] - T_goal[:3, 3]))
        R = T_now[:3, :3].T @ T_goal[:3, :3]
        c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        return ep, float(np.arccos(c))

    def set_target(self, T_target: np.ndarray) -> None:
        """设置当前任务目标（配合 step() 使用）"""
        self.task.T_world_frame = np.asarray(T_target, dtype=np.float64)

    def step(self) -> IkResult:
        """单次 QP 迭代（约 0.05 ms）——控制周期内预算有限时用"""
        import time
        t0 = time.perf_counter()
        self.solver.solve(True)
        self.robot.update_kinematics()
        ms = (time.perf_counter() - t0) * 1000.0
        T_goal = np.asarray(self.task.T_world_frame)
        ep, er = self._pose_err(self.robot.get_T_world_frame(self.tcp_frame), T_goal)
        q = self.joint_angles()
        ok_lim, viol = self.limit_violation(q)
        return IkResult(q=q, converged=(ep < POS_TOL_M and er < ROT_TOL_RAD),
                        iters=1, ms=ms, pos_err=ep, rot_err=er,
                        in_limits=ok_lim, max_limit_violation=viol)

    def ik(self, T_target: np.ndarray,
           q_seed: Optional[Sequence[float]] = None,
           max_iter: int = MAX_ITER,
           pos_tol: float = POS_TOL_M,
           rot_tol: float = ROT_TOL_RAD,
           warm_start: bool = True) -> IkResult:
        """迭代求解 IK 到收敛

        Args:
            T_target: 目标 4x4 齐次矩阵
            q_seed: 起点关节角；None 时用上一次的解（warm_start）
            max_iter: 最大迭代次数（超出即判未收敛，用于限制最坏延迟）
            warm_start: False 时忽略上次解，用关节角上下限中点起步

        Returns:
            IkResult（``q`` 始终给出当前最好解，即使未收敛）
        """
        import time
        T_target = np.asarray(T_target, dtype=np.float64)

        if q_seed is not None:
            self._set_joints(q_seed)
        elif warm_start:
            cur = self.joint_angles()
            # 全零/未初始化时退化为限位中点
            if not np.any(np.isfinite(cur)) or np.allclose(cur, 0.0):
                self._set_joints(0.5 * (self.q_lo + self.q_hi))
        else:
            self._set_joints(0.5 * (self.q_lo + self.q_hi))
        self.robot.update_kinematics()

        self.set_target(T_target)
        t0 = time.perf_counter()
        ep = er = float("inf")
        it = 0
        for it in range(1, max_iter + 1):
            self.solver.solve(True)
            self.robot.update_kinematics()
            ep, er = self._pose_err(self.robot.get_T_world_frame(self.tcp_frame), T_target)
            if ep < pos_tol and er < rot_tol:
                break
        ms = (time.perf_counter() - t0) * 1000.0

        q = self.joint_angles()
        ok_lim, viol = self.limit_violation(q)
        return IkResult(q=q, converged=(ep < pos_tol and er < rot_tol),
                        iters=it, ms=ms, pos_err=ep, rot_err=er,
                        in_limits=ok_lim, max_limit_violation=viol)

    def ik_pos(self, xyz: Sequence[float],
               q_seed: Optional[Sequence[float]] = None,
               keep_orientation: bool = True,
               **kw) -> IkResult:
        """只给位置目标的 IK（姿态沿用当前或指定的起点姿态）"""
        base = self.fk(q_seed if q_seed is not None else self.joint_angles())
        T = base.copy()
        T[:3, 3] = np.asarray(xyz, dtype=np.float64)
        if not keep_orientation:
            T[:3, :3] = np.eye(3)
        return self.ik(T, q_seed=q_seed, **kw)
