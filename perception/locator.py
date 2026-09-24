# perception/locator.py — 颜色定位降级模块
#
# 重构方案 v9: VLM 不可用时作为降级方案。
# 从 vla/vision/locator.py 移植，无需 OpenCV 以外依赖。
"""轻量颜色定位 — HSV 颜色分割 + 轮廓检测 + Depth 反投影

当 VLM (Qwen3.5) 不可用时，使用纯 CPU 颜色分割定位目标。
支持 6 种常见颜色，返回像素坐标和 3D 空间坐标。

模块接口:
    ColorLocator(camera_matrix) -> locate(rgb, depth, color) -> dict | None
"""
import cv2
import numpy as np


class ColorLocator:
    """基于 HSV 颜色分割的目标定位（VLM 降级方案）

    使用纯 OpenCV CPU 操作，无需任何 ML 模型。
    支持 6 种预定义颜色，每种颜色可配置多组 HSV 范围。

    Args:
        camera_matrix: (3, 3) 相机内参矩阵
    """

    # HSV 颜色范围 (OpenCV H: 0-179, S: 0-255, V: 0-255)
    COLOR_RANGES = {
        "红色": [
            [(0, 30, 30), (30, 255, 255)],       # 低角度红
            [(150, 30, 30), (180, 255, 255)],     # 高角度红（环绕）
        ],
        "绿色": [[(35, 50, 50), (85, 255, 255)]],
        "蓝色": [[(100, 100, 50), (140, 255, 255)]],
        "黄色": [[(20, 80, 80), (40, 255, 255)]],
        "橙色": [[(5, 80, 100), (30, 255, 255)]],
        "紫色": [[(125, 60, 50), (160, 255, 255)]],
    }

    # 最小轮廓面积（像素）
    MIN_CONTOUR_AREA = 500

    def __init__(self, camera_matrix: np.ndarray):
        """
        Args:
            camera_matrix: (3, 3) float 相机内参矩阵
        """
        self.K = camera_matrix

    # ── 颜色掩码 ──────────────────────────────────────────────────────────

    def get_mask(self, rgb: np.ndarray, color: str) -> np.ndarray:
        """获取指定颜色的二值掩码

        支持多区间合并（如红色需要低角度和高角度两个区间）。

        Args:
            rgb: (H, W, 3) uint8 RGB 图像
            color: 颜色名称，见 COLOR_RANGES 键

        Returns:
            二值掩码 (H, W) uint8
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        ranges = self.COLOR_RANGES.get(color, self.COLOR_RANGES["红色"])
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        if mask is None:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        # 形态学操作去噪
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    # ── 主色检测 ──────────────────────────────────────────────────────────

    def dominant_color(self, rgb: np.ndarray) -> tuple[str, np.ndarray]:
        """遍历所有颜色，返回像素最多的颜色及其掩码

        Args:
            rgb: (H, W, 3) uint8 RGB 图像

        Returns:
            (color_name, mask) — 颜色名和对应的二值掩码
        """
        best_color = "红色"
        best_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        best_count = 0
        for color in self.COLOR_RANGES:
            mask = self.get_mask(rgb, color)
            cnt = int(np.sum(mask > 0))
            if cnt > best_count:
                best_count = cnt
                best_color = color
                best_mask = mask
        return best_color, best_mask

    # ── 目标定位 ──────────────────────────────────────────────────────────

    def locate(self, rgb: np.ndarray, depth: np.ndarray,
               color: str) -> dict | None:
        """定位指定颜色目标的 3D 坐标

        流程:
          1. HSV 颜色分割 → 二值掩码
          2. 轮廓检测 → 最大轮廓质心
          3. 深度采样 → 相机内参反投影 → 3D 坐标 (相机坐标系)

        Args:
            rgb: (H, W, 3) uint8 RGB 图像
            depth: (H, W) float32 深度图 (米)
            color: 目标颜色名称

        Returns:
            dict with keys:
                x, y, z: 3D 坐标 (相机坐标系, 米)
                u, v:    像素坐标
            None: 未找到目标
        """
        mask = self.get_mask(rgb, color)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 取最大轮廓
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.MIN_CONTOUR_AREA:
            return None

        # 质心
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])

        # 深度采样（5x5 邻域中值）
        z = self._sample_depth(depth, v, u)
        if z <= 0 or np.isnan(z):
            z = 0.35  # 兜底深度

        # 相机内参反投影: (u, v) → (x, y, z)
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return {"x": x, "y": y, "z": z, "u": u, "v": v}

    # ── 内部工具 ──────────────────────────────────────────────────────────

    @staticmethod
    def _sample_depth(depth: np.ndarray, v: int, u: int,
                      radius: int = 2) -> float:
        """取 (v, u) 邻域有效深度中值

        Args:
            depth: 深度图 (米)
            v, u: 像素坐标
            radius: 邻域半径

        Returns:
            中值深度 (米)，无效时返回 0.35
        """
        h, w = depth.shape
        y1 = max(0, v - radius)
        y2 = min(h, v + radius + 1)
        x1 = max(0, u - radius)
        x2 = min(w, u + radius + 1)
        patch = depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return 0.35
        return float(np.median(valid))