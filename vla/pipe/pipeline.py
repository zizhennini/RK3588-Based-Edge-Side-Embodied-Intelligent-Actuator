"""主流水线 — VLM 场景理解 + ColorLocator 定位 → Depth → IK"""
import cv2
import time
import numpy as np
from vla.vision import ColorLocator
from vla.control import ArmController


class State:
    IDLE = 0
    VLM_INFER = 1
    GRASP = 2
    PLACE = 3
    DONE = 4


class VLApipeline:
    """VLM 场景理解 + ColorLocator 定位 → Depth → IK"""

    MAX_DEPTH_RETRY = 3       # 深度读取失败最大重试次数
    DEFAULT_DEPTH = 0.35       # 默认深度回退值（米）
    DEPTH_VALID_MIN = 0.05     # 有效深度最小值（米）
    DEPTH_VALID_MAX = 2.0      # 有效深度最大值（米）

    def __init__(self, arm: ArmController, vlm, camera_matrix):
        self.arm = arm
        self.vlm = vlm
        self.K = camera_matrix
        self.locator = ColorLocator(camera_matrix)
        self.state = State.IDLE
        self.target_3d: tuple | None = None
        self.vlm_color = "红色"

    def start(self):
        self.state = State.VLM_INFER

    def _validate_depth(self, depth, u: int, v: int) -> float:
        """安全读取深度值，含多重容错"""
        if depth is None:
            return self.DEFAULT_DEPTH

        if not isinstance(depth, np.ndarray) or depth.size == 0:
            return self.DEFAULT_DEPTH

        if np.all(depth == 0):
            return self.DEFAULT_DEPTH

        h, w = depth.shape[:2]
        if not (0 <= v < h and 0 <= u < w):
            return self.DEFAULT_DEPTH

        dz = float(depth[v, u])

        if np.isnan(dz) or np.isinf(dz):
            return self.DEFAULT_DEPTH

        if self.DEPTH_VALID_MIN <= dz <= self.DEPTH_VALID_MAX:
            return dz

        # 尝试取 u,v 周围 3x3 区域的中值
        y0, y1 = max(0, v - 1), min(h, v + 2)
        x0, x1 = max(0, u - 1), min(w, u + 2)
        patch = depth[y0:y1, x0:x1]
        valid = patch[(patch > self.DEPTH_VALID_MIN) & (patch < self.DEPTH_VALID_MAX)]
        if len(valid) > 0:
            return float(np.median(valid))

        return self.DEFAULT_DEPTH

    def step(self, rgb, depth) -> str:
        if self.state == State.IDLE:
            return "idle"

        elif self.state == State.VLM_INFER:
            if rgb is None:
                self.state = State.DONE
                return "vlm_failed: rgb is None"

            tmp = "/tmp/vla_frame.jpg"
            cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            result = self.vlm.infer(tmp)
            self.vlm_color = result.color or "红色"

            if result.cx is not None and result.cy is not None:
                u, v = int(result.cx), int(result.cy)
                status = "vlm_coord"
            else:
                # VLM 未输出坐标，降级为颜色定位
                if not result.color and not result.object:
                    self.state = State.DONE
                    return f"vlm_failed: 无法识别目标物体"

                # 先用 VLM 识别的物体名尝试定位，失败后用颜色定位
                pos = self.locator.locate(rgb, depth, result.color)
                if pos is None:
                    # 遍历所有颜色尝试
                    for color_name in ["红色", "绿色", "蓝色", "黄色", "橙色", "紫色"]:
                        pos = self.locator.locate(rgb, depth, color_name)
                        if pos is not None:
                            self.vlm_color = color_name
                            break

                if pos is None:
                    self.state = State.DONE
                    return f"locate_failed: {result.color} {result.object}"

                u, v = pos["u"], pos["v"]
                status = "color_locate"

            # 带容错的深度读取
            z = self._validate_depth(depth, u, v)

            # 相机坐标系 → 3D 坐标（使用 `move_to_camera` 内部转换）
            x_cam = (u - self.K[0, 2]) * z / self.K[0, 0]
            y_cam = (v - self.K[1, 2]) * z / self.K[1, 1]

            self.target_3d = (x_cam, y_cam, z)
            self.state = State.GRASP
            return f"{status}: {result.color} {result.object[:20]} @ ({u},{v}) z={z:.2f}"

        elif self.state == State.GRASP:
            if self.target_3d is None:
                self.state = State.DONE
                return "grasp_failed: no target"

            # 使用 move_to_camera 自动处理坐标转换
            self.arm.move_to_camera(*self.target_3d)
            self.arm.gripper(False)
            time.sleep(1)
            self.state = State.PLACE
            return f"grasp: cam({self.target_3d[0]:.2f},{self.target_3d[1]:.2f},{self.target_3d[2]:.2f})"

        elif self.state == State.PLACE:
            self.arm.move_to(0.30, 0.0, 0.10)
            self.arm.gripper(True)
            self.state = State.DONE
            return "place_done"

        return "done"
