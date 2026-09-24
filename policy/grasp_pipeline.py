"""抓取管线 -- VLM 定位 + GGCNN/PCA 抓取 + IK 执行

整合完整感知-规划-执行三段管线，实现 PolicyModule 接口。

降级路径:
  1. GGCNN 可用 -> VLM 定位 ROI + GGCNN 抓取检测
  2. 仅 VLM 可用 -> bbox 中心 + 简单深度提取（无抓取角度）
  3. 都不可用 -> 返回失败

参考: scripts/vlm_grasp.py（已验证可工作的完整流程）
"""
import time
import math
import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np

from hardware.interfaces import (
    PolicyModule, Observation, Action, TaskRequest, TaskResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class GraspCandidate:
    """抓取候选"""
    center_px: tuple            # (u, v) 原图像素坐标
    position_3d: np.ndarray     # (x, y, z) 机器人坐标系 (米)
    angle_rad: float            # 抓取角度 (wrist_roll, rad)
    width_m: float              # 夹爪开度 (米)
    quality: float              # 质量分数 [0, 1]
    label: str = ""             # 物体标签
    source: str = ""            # 来源: "ggcnn" | "vlm" | "pca"


# ---------------------------------------------------------------------------
# 抓取管线
# ---------------------------------------------------------------------------
class GraspPipeline(PolicyModule):
    """完整抓取管线: VLM定位 -> 深度3D -> GGCNN/PCA抓取 -> IK -> 执行

    实现 PolicyModule 接口，同时提供更直观的 execute_grasp() 主入口。

    依赖模块（均通过构造函数注入，可为 None 以支持降级）：
        - arm: SO101Arm (硬件层，必需)
        - camera: CameraManager (硬件层，必需)
        - vlm: VLMPerception (感知层，可选)
        - ggcnn: GGCNNDetector (感知层，可选)
        - kinematics: Kinematics (策略层，可选)
    """

    # ---- 手眼标定参数（实测标定角，移植自 vlm_grasp.py） ----
    # 手眼标定旋转角: 像素X+69px → 机器人 +X0.050m -Y0.104m
    CAM_ANGLE = math.radians(101.9)

    # 相机外参: 相机光心在机械臂基座坐标系下的位置 (米)
    CAM_POSITION = np.array([0.182, -0.129, 0.47], dtype=float)

    # 相机内参（D435i 出厂标定，来自 config/settings.py）
    CAM_FX = 604.2294
    CAM_FY = 604.0748
    CAM_PPX = 315.1330
    CAM_PPY = 250.8858

    # ---- 抓取几何参数（移植自 vlm_grasp.py） ----
    PRE_GRASP_OFFSET = 0.08     # 接近距离: pre_grasp 在目标上方 8cm
    LIFT_OFFSET = 0.06          # 抬升距离: lift 在目标上方 6cm
    GRIPPER_FACTOR = 0.7        # 夹爪开度系数: 取物体短边的 70%
    DEFAULT_GRIPPER_WIDTH = 0.05  # 降级时的默认夹爪开度 (米)

    # ---- 工作空间限制 [x_min, x_max, y_min, y_max, z_min, z_max] ----
    WORKSPACE = [0.03, 0.45, -0.30, 0.45, 0.01, 0.40]

    # ---- 执行时序参数 (秒) ----
    T_PRE_GRASP = 1.0           # 移动到预抓取点后等待
    T_OPEN_GRIPPER = 0.3        # 张开夹爪后等待
    T_GRASP = 0.5               # 下探到抓取点后等待
    T_CLOSE_GRIPPER = 0.5       # 闭合夹爪后等待
    T_LIFT = 0.5                # 抬升后等待

    def __init__(self, arm, camera, vlm=None, ggcnn=None, kinematics=None):
        """
        Args:
            arm: SO101Arm 实例（必需，提供 move_to/gripper/home/emergency_stop）
            camera: CameraManager 实例（必需，提供 get_frame）
            vlm: VLMPerception 实例（可选，用于目标定位）
            ggcnn: GGCNNDetector 实例（可选，用于抓取位姿检测）
            kinematics: Kinematics 实例（可选，用于额外的工作空间钳制）
        """
        self.arm = arm
        self.camera = camera
        self.vlm = vlm
        self.ggcnn = ggcnn
        self.kinematics = kinematics
        self._config: dict = {}
        self._last_grasp: Optional[GraspCandidate] = None

    # ------------------------------------------------------------------
    # Module 生命周期接口
    # ------------------------------------------------------------------
    def setup(self, config: dict) -> None:
        """初始化管线配置

        Args:
            config: 支持以下 key（均可选）:
                cam_angle_deg: float    手眼标定角(度)，覆盖默认 101.9
                pre_grasp_offset: float 接近距离(米)
                lift_offset: float      抬升距离(米)
                gripper_factor: float   夹爪开度系数
                workspace: list         工作空间 [xmin,xmax,ymin,ymax,zmin,zmax]
        """
        self._config = dict(config or {})

        if "cam_angle_deg" in self._config:
            self.CAM_ANGLE = math.radians(float(self._config["cam_angle_deg"]))
        if "pre_grasp_offset" in self._config:
            self.PRE_GRASP_OFFSET = float(self._config["pre_grasp_offset"])
        if "lift_offset" in self._config:
            self.LIFT_OFFSET = float(self._config["lift_offset"])
        if "gripper_factor" in self._config:
            self.GRIPPER_FACTOR = float(self._config["gripper_factor"])
        if "workspace" in self._config:
            self.WORKSPACE = list(self._config["workspace"])

        logger.info(
            "GraspPipeline 配置完成: cam_angle=%.1f° pre=%.2fm lift=%.2fm",
            math.degrees(self.CAM_ANGLE), self.PRE_GRASP_OFFSET, self.LIFT_OFFSET,
        )

    def start(self) -> None:
        """启动管线（无独立后台线程，占位实现）"""
        logger.info("GraspPipeline 已启动")

    def stop(self) -> None:
        """停止管线（释放引用，占位实现）"""
        self._last_grasp = None
        logger.info("GraspPipeline 已停止")

    @property
    def is_available(self) -> bool:
        """管线可用的最低要求: 机械臂 + 相机"""
        return self.arm is not None and self.camera is not None

    def on_failure(self) -> str:
        """抓取失败时中止当前任务（避免危险动作）"""
        return "abort"

    # ------------------------------------------------------------------
    # PolicyModule 接口: predict
    # ------------------------------------------------------------------
    def predict(self, obs: Observation) -> Action:
        """PolicyModule 接口: 根据观测返回下一步动作

        说明: 抓取管线本质是"感知-规划-执行"的离散流程，而非连续控制策略。
        此处为兼容 PolicyModule 接口提供适配实现：基于当前观测计算 IK 目标，
        返回单步动作。实际完整抓取请使用 execute_grasp()。

        Args:
            obs: 当前观测（含 rgb/depth/state/timestamp）

        Returns:
            Action: positions=IK 解算的 6 维关节角, gripper=开合度, execution_time
        """
        if not self.is_available:
            return self._idle_action(obs)

        # 使用深度图中心点作为简易目标（无 VLM/GGCNN 时的兜底策略）
        depth = obs.depth
        h, w = depth.shape[:2]
        cu, cv = w // 2, h // 2
        pos_3d = self._pixel_to_robot((cu, cv), depth)
        if pos_3d is None:
            return self._idle_action(obs)

        pos_3d = self._clamp_workspace(pos_3d)
        angles = self._solve_ik(pos_3d, obs.state, wrist_roll_rad=None)
        return Action(
            positions=angles,
            gripper=1.0,
            execution_time=1.0,
        )

    def _idle_action(self, obs: Observation) -> Action:
        """保持当前姿态的空动作"""
        state = obs.state if obs.state is not None else np.zeros(6)
        return Action(positions=np.asarray(state, dtype=float), gripper=0.0,
                      execution_time=0.0)

    # ------------------------------------------------------------------
    # 主入口: 完整抓取流程
    # ------------------------------------------------------------------
    def execute_grasp(self, target_desc: str, verify: bool = False) -> TaskResult:
        """完整抓取流程（主入口）

        流程:
            1. perceive: 获取图像 + VLM 定位 + GGCNN 抓取检测
            2. plan:     像素->3D + 坐标变换 + 轨迹生成 + 安全校验
            3. execute:  pre_grasp -> grasp -> close gripper -> lift -> home

        Args:
            target_desc: 目标描述 (如 "红色杯子")
            verify: 是否需要人工确认 (默认 False)

        Returns:
            TaskResult(success, message, data)
        """
        if not self.is_available:
            return TaskResult(False, "管线不可用: 缺少机械臂或相机", {})

        # ---- 阶段 1: 感知 ----
        logger.info("[1/3] 感知阶段: 目标='%s'", target_desc)
        grasp = self._perceive(target_desc)
        if grasp is None:
            return TaskResult(False, "感知失败: 未定位到可抓取目标", {})
        self._last_grasp = grasp
        logger.info(
            "  抓取候选: source=%s quality=%.2f pos=(%.3f,%.3f,%.3f) "
            "angle=%.1f° width=%.3fm",
            grasp.source, grasp.quality, *grasp.position_3d,
            math.degrees(grasp.angle_rad), grasp.width_m,
        )

        # ---- 阶段 2: 规划 ----
        logger.info("[2/3] 规划阶段")
        trajectory = self._plan(grasp)
        if trajectory is None:
            return TaskResult(
                False,
                f"规划失败: 目标超出工作空间 {grasp.position_3d.tolist()}",
                {"position_3d": grasp.position_3d.tolist()},
            )

        # ---- 人工确认 ----
        if verify:
            ok = self._human_verify(grasp, trajectory)
            if not ok:
                return TaskResult(False, "用户取消执行", {})

        # ---- 阶段 3: 执行 ----
        logger.info("[3/3] 执行阶段")
        return self._execute(trajectory, grasp.angle_rad, grasp.width_m)

    # ------------------------------------------------------------------
    # 阶段 1: 感知
    # ------------------------------------------------------------------
    def _perceive(self, target_desc: str) -> Optional[GraspCandidate]:
        """感知阶段: 获取图像 + VLM 定位 + GGCNN 抓取检测

        降级顺序:
            1. GGCNN 可用 -> 在 VLM ROI 内做抓取检测
            2. 仅 VLM -> bbox 中心 + 深度提取
            3. 都不可用 -> None
        """
        # 1. 获取图像帧
        frame = self.camera.get_frame()
        if frame is None:
            logger.error("感知失败: 相机无可用帧")
            return None
        rgb, depth, ts = frame

        state = self._read_arm_state()
        obs = Observation(rgb=rgb, depth=depth, state=state, timestamp=ts)

        # 2. VLM 目标定位（如果可用）
        roi_bbox = None
        vlm_center = None
        vlm_label = ""
        if self.vlm is not None and self.vlm.is_available and target_desc:
            try:
                vlm_result = self.vlm.detect(obs, target_desc)
            except Exception as e:
                logger.warning("VLM 检测异常: %s", e)
                vlm_result = None
            if vlm_result and vlm_result.get("confidence", 0) > 0:
                bbox = vlm_result.get("bbox")  # 归一化 (x1,y1,x2,y2) [0,1]
                vlm_label = vlm_result.get("label", target_desc)
                if bbox is not None:
                    h, w = rgb.shape[:2]
                    roi_bbox = (
                        int(bbox[0] * w), int(bbox[1] * h),
                        int(bbox[2] * w), int(bbox[3] * h),
                    )
                vlm_center = vlm_result.get("center_px")
                logger.info("  VLM 定位: label='%s' roi=%s center=%s",
                            vlm_label, roi_bbox, vlm_center)

        # 3. GGCNN 抓取检测（如果可用）
        if self.ggcnn is not None and self.ggcnn.is_available:
            try:
                grasp_result = self.ggcnn.detect(obs, roi_bbox=roi_bbox)
            except Exception as e:
                logger.warning("GGCNN 检测异常: %s", e)
                grasp_result = None
            grasps = (grasp_result or {}).get("grasps", [])
            if grasps:
                best = grasps[0]  # 质量最高
                pos_3d = self._pixel_to_robot(best["center_px"], depth)
                if pos_3d is not None:
                    return GraspCandidate(
                        center_px=tuple(best["center_px"]),
                        position_3d=pos_3d,
                        angle_rad=float(best["angle_rad"]),
                        width_m=float(best["width_m"]),
                        quality=float(best["quality"]),
                        label=vlm_label,
                        source="ggcnn",
                    )
                logger.warning("  GGCNN 抓取点深度无效，尝试降级")

        # 4. 降级: VLM bbox 中心 + 简单深度提取（无 GGCNN 时）
        center = None
        if roi_bbox is not None:
            cx = (roi_bbox[0] + roi_bbox[2]) // 2
            cy = (roi_bbox[1] + roi_bbox[3]) // 2
            center = (cx, cy)
        elif vlm_center is not None:
            center = tuple(vlm_center)

        if center is not None:
            pos_3d = self._pixel_to_robot(center, depth)
            if pos_3d is not None:
                width = self._estimate_width_from_bbox(roi_bbox, depth, center)
                return GraspCandidate(
                    center_px=(int(center[0]), int(center[1])),
                    position_3d=pos_3d,
                    angle_rad=0.0,
                    width_m=width,
                    quality=0.5,
                    label=vlm_label,
                    source="vlm",
                )

        logger.error("感知失败: 无可用的抓取候选（VLM/GGCNN 均未定位到目标）")
        return None

    def _estimate_width_from_bbox(self, roi_bbox: Optional[tuple],
                                  depth: np.ndarray,
                                  center: tuple) -> float:
        """从 VLM bbox 估算夹爪开度（短边物理尺寸 * GRIPPER_FACTOR）

        移植自 vlm_grasp.py pca_grasp_pose 的宽度计算逻辑。
        """
        if roi_bbox is None:
            return self.DEFAULT_GRIPPER_WIDTH
        x1, y1, x2, y2 = roi_bbox
        bbox_w = max(0.0, x2 - x1)
        bbox_h = max(0.0, y2 - y1)
        z = self._get_depth_at(center[0], center[1], depth)
        if z is None or z <= 0.01:
            return self.DEFAULT_GRIPPER_WIDTH
        metric_w = bbox_w * z / self.CAM_FX
        metric_h = bbox_h * z / self.CAM_FY
        return float(min(metric_w, metric_h) * self.GRIPPER_FACTOR)

    def _read_arm_state(self) -> np.ndarray:
        """安全读取机械臂当前关节角度"""
        try:
            return self.arm.read_positions()
        except Exception as e:
            logger.warning("读取机械臂状态失败: %s", e)
            return np.zeros(6)

    # ------------------------------------------------------------------
    # 阶段 2: 规划
    # ------------------------------------------------------------------
    def _plan(self, grasp: GraspCandidate) -> Optional[list]:
        """规划阶段: 生成三段轨迹（接近->抓取->抬升）+ 安全校验

        接近方向: 从正上方垂直下移（上帝视角最安全，移植自 vlm_grasp.py）
        抓取角度通过 wrist_roll 关节实现。

        Returns:
            [pre_grasp, grasp_pos, lift_pos] 三个 np.ndarray(x,y,z)，
            校验失败返回 None
        """
        x, y, z = grasp.position_3d

        # 工作空间钳制（可选，若注入 kinematics）
        target = self._clamp_workspace(np.array([x, y, z], dtype=float))
        x, y, z = target

        pre_grasp = np.array([x, y, z + self.PRE_GRASP_OFFSET], dtype=float)
        grasp_pos = np.array([x, y, z], dtype=float)
        lift_pos = np.array([x, y, z + self.LIFT_OFFSET], dtype=float)

        # 安全校验: 三个轨迹点都必须在工作空间内
        for name, point in (("预抓取", pre_grasp), ("抓取", grasp_pos),
                            ("抬升", lift_pos)):
            if not self._in_workspace(point):
                logger.warning("目标超出工作空间 [%s]: %s", name,
                               [round(v, 3) for v in point])
                return None

        logger.info("  轨迹: pre=%s grasp=%s lift=%s",
                    [round(v, 3) for v in pre_grasp],
                    [round(v, 3) for v in grasp_pos],
                    [round(v, 3) for v in lift_pos])
        return [pre_grasp, grasp_pos, lift_pos]

    def _clamp_workspace(self, xyz: np.ndarray) -> np.ndarray:
        """工作空间钳制: 优先使用注入的 Kinematics，否则用 WORKSPACE 边界裁剪"""
        if self.kinematics is not None:
            try:
                return self.kinematics.clamp_workspace(xyz)
            except Exception as e:
                logger.warning("Kinematics.clamp_workspace 失败: %s", e)
        ws = self.WORKSPACE
        out = np.asarray(xyz, dtype=float).copy()
        out[0] = float(np.clip(out[0], ws[0], ws[1]))
        out[1] = float(np.clip(out[1], ws[2], ws[3]))
        out[2] = float(np.clip(out[2], ws[4], ws[5]))
        return out

    def _in_workspace(self, xyz: np.ndarray) -> bool:
        """检查点是否在工作空间内"""
        ws = self.WORKSPACE
        return bool(
            ws[0] <= xyz[0] <= ws[1] and
            ws[2] <= xyz[1] <= ws[3] and
            ws[4] <= xyz[2] <= ws[5]
        )

    def _solve_ik(self, xyz: np.ndarray, current: np.ndarray,
                  wrist_roll_rad: Optional[float]) -> np.ndarray:
        """IK 解算（供 predict 使用）；无 kinematics 时返回当前状态"""
        if self.kinematics is None:
            return np.asarray(current, dtype=float)
        try:
            return self.kinematics.inverse_kinematics(
                xyz, np.asarray(current, dtype=float), wrist_roll_rad)
        except Exception as e:
            logger.warning("IK 解算失败: %s", e)
            return np.asarray(current, dtype=float)

    def _human_verify(self, grasp: GraspCandidate, trajectory: list) -> bool:
        """人工复核（命令行确认）"""
        pre, gpos, lift = trajectory
        print("\n[人工复核]")
        print(f"  目标: {grasp.label or '未知'}  来源: {grasp.source}")
        print(f"  角度={math.degrees(grasp.angle_rad):.1f}° "
              f"宽度={grasp.width_m:.3f}m 质量={grasp.quality:.2f}")
        print(f"  坐标=({gpos[0]:.3f}, {gpos[1]:.3f}, {gpos[2]:.3f})")
        try:
            r = input("  确认执行? (y/n): ").strip().lower()
            return r == "y"
        except (EOFError, KeyboardInterrupt):
            print("  已取消")
            return False

    # ------------------------------------------------------------------
    # 阶段 3: 执行
    # ------------------------------------------------------------------
    def _execute(self, trajectory: list, angle_rad: float,
                 width_m: float) -> TaskResult:
        """执行阶段: 机械臂三段运动

        时序: 预抓取(带角度) -> 张开夹爪 -> 下探抓取 -> 闭合夹爪 -> 抬升 -> home
        """
        pre_grasp, grasp_pos, lift_pos = trajectory

        try:
            # 1. 移动到预抓取位置（带 PCA/GGCNN 抓取角度）
            logger.info("  接近: (%.3f,%.3f,%.3f) wrist_roll=%.2f",
                        *pre_grasp, angle_rad)
            self.arm.move_to(pre_grasp[0], pre_grasp[1], pre_grasp[2],
                             wrist_roll_rad=angle_rad)
            time.sleep(self.T_PRE_GRASP)

            # 2. 张开夹爪
            if width_m > 0:
                self.arm.gripper_width(width_m * self.GRIPPER_FACTOR)
            else:
                self.arm.gripper(True)
            time.sleep(self.T_OPEN_GRIPPER)

            # 3. 下探到抓取位置（保持角度）
            logger.info("  抓取: (%.3f,%.3f,%.3f)", *grasp_pos)
            self.arm.move_to(grasp_pos[0], grasp_pos[1], grasp_pos[2],
                             wrist_roll_rad=angle_rad)
            time.sleep(self.T_GRASP)

            # 4. 闭合夹爪
            logger.info("  夹爪闭合 (width=%.3fm)", width_m)
            self.arm.gripper(False)
            time.sleep(self.T_CLOSE_GRIPPER)

            # 5. 抬升
            logger.info("  抬升: (%.3f,%.3f,%.3f)", *lift_pos)
            self.arm.move_to(lift_pos[0], lift_pos[1], lift_pos[2],
                             wrist_roll_rad=angle_rad)
            time.sleep(self.T_LIFT)

            return TaskResult(
                success=True,
                message="抓取成功",
                data={
                    "position": grasp_pos.tolist(),
                    "pre_grasp": pre_grasp.tolist(),
                    "lift": lift_pos.tolist(),
                    "angle": angle_rad,
                    "width_m": width_m,
                },
            )

        except Exception as e:
            logger.error("抓取执行失败: %s", e, exc_info=True)
            try:
                self.arm.emergency_stop()
            except Exception as stop_err:
                logger.error("急停失败: %s", stop_err)
            return TaskResult(False, f"执行失败: {e}", {})

        finally:
            # 回归 home（异常也不影响主流程返回）
            try:
                self.arm.home(steps=30, delay_s=0.03)
            except Exception as home_err:
                logger.warning("回归 home 失败: %s", home_err)

    # ------------------------------------------------------------------
    # 坐标变换（移植自 vlm_grasp.py）
    # ------------------------------------------------------------------
    def _pixel_to_robot(self, center_px: tuple,
                        depth: np.ndarray) -> Optional[np.ndarray]:
        """像素坐标 -> 机器人基座坐标

        移植自 vlm_grasp.py 的 depth_to_3d + robot_coords(mode=4) 组合:
          1. 相机内参反投影: 像素 + 深度 -> 相机坐标系 (x_cam, y_cam, z_cam)
          2. 旋转矩阵手眼变换 (mode=4): 相机坐标 -> 机器人基座坐标

        Args:
            center_px: (u, v) 原图像素坐标
            depth: (H, W) float32 深度图，单位米

        Returns:
            np.ndarray([rx, ry, rz]) 机器人基座坐标 (米)，深度无效返回 None
        """
        u, v = float(center_px[0]), float(center_px[1])

        # 5x5 邻域中值容错获取深度
        z = self._get_depth_at(u, v, depth)
        if z is None or z <= 0.01:
            return None

        # 1. 相机内参反投影 (像素 -> 相机坐标系)
        x_cam = (u - self.CAM_PPX) * z / self.CAM_FX
        y_cam = (v - self.CAM_PPY) * z / self.CAM_FY

        # 2. 相机 -> 机器人基座坐标 (mode=4 旋转矩阵，精确标定)
        cos_a = math.cos(self.CAM_ANGLE)
        sin_a = math.sin(self.CAM_ANGLE)
        rx = x_cam * cos_a - y_cam * sin_a + self.CAM_POSITION[0]
        ry = x_cam * sin_a + y_cam * cos_a + self.CAM_POSITION[1]
        rz = self.CAM_POSITION[2] - z

        return np.array([rx, ry, rz], dtype=float)

    def _get_depth_at(self, u: float, v: float,
                      depth: np.ndarray) -> Optional[float]:
        """获取指定像素的深度（5x5 邻域中值容错，移植自 vlm_grasp.py）

        Args:
            u, v: 像素坐标（浮点，内部取整）
            depth: (H, W) 深度图，单位米

        Returns:
            中值深度 (米)，邻域内无有效深度返回 None
        """
        h, w = depth.shape[:2]
        ui, vi = int(round(u)), int(round(v))
        ui = max(2, min(w - 3, ui))
        vi = max(2, min(h - 3, vi))
        patch = depth[vi - 2:vi + 3, ui - 2:ui + 3]
        valid = patch[np.isfinite(patch) & (patch > 0.01)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    # ------------------------------------------------------------------
    # TaskRequest 适配（供上层任务调度器调用）
    # ------------------------------------------------------------------
    def handle_request(self, request: TaskRequest) -> TaskResult:
        """处理 TaskRequest（type='grasp' 时执行抓取）

        Args:
            request: TaskRequest(type, target, bbox, params)

        Returns:
            TaskResult
        """
        if request.type != "grasp":
            return TaskResult(False, f"GraspPipeline 不支持任务类型: {request.type}", {})
        verify = bool(request.params.get("verify", False))
        return self.execute_grasp(request.target, verify=verify)
