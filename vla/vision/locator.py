"""轻量视觉定位：颜色分割 + 轮廓检测 + Depth 反投影"""
import cv2
import numpy as np


class ColorLocator:
    """基于颜色分割的目标定位，无 YOLO，纯 CPU"""

    # 每个颜色可有多组 HSV 范围（OpenCV H:0-179, S:0-255, V:0-255）
    COLOR_RANGES = {
        "红色": [
            [(0, 30, 30), (30, 255, 255)],      # 低角度红（大幅放宽）
            [(150, 30, 30), (180, 255, 255)],    # 高角度红环绕
        ],
        "绿色": [[(35, 50, 50), (85, 255, 255)]],
        "蓝色": [[(100, 100, 50), (140, 255, 255)]],
        "黄色": [[(20, 80, 80), (40, 255, 255)]],
        "橙色": [[(5, 80, 100), (30, 255, 255)]],
        "紫色": [[(125, 60, 50), (160, 255, 255)]],
    }

    def __init__(self, camera_matrix: np.ndarray):
        self.K = camera_matrix

    def get_mask(self, rgb: np.ndarray, color: str) -> np.ndarray:
        """获取指定颜色的二值掩码，支持多区间合并"""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        ranges = self.COLOR_RANGES.get(color, self.COLOR_RANGES["红色"])
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        if mask is None:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def dominant_color(self, rgb: np.ndarray, min_pixels: int = 500) -> tuple[str, np.ndarray]:
        """遍历所有颜色，返回像素最多的颜色及其掩码"""
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

    def locate(self, rgb: np.ndarray, depth: np.ndarray, color: str) -> dict | None:
        mask = self.get_mask(rgb, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:
            return None
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        u, v = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        z = float(depth[v, u]) if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1] else 0.35
        if z <= 0 or np.isnan(z):
            z = 0.35
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return {"x": x, "y": y, "z": z, "u": u, "v": v}
