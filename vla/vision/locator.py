"""轻量视觉定位：颜色分割 + 轮廓检测 + Depth 反投影"""
import cv2
import numpy as np


class ColorLocator:
    """基于颜色分割的目标定位，无 YOLO，纯 CPU"""

    COLOR_RANGES = {
        "红色": [(0, 120, 70), (10, 255, 255)],
        "绿色": [(40, 70, 70), (80, 255, 255)],
        "蓝色": [(100, 150, 70), (130, 255, 255)],
        "黄色": [(20, 100, 100), (35, 255, 255)],
        "橙色": [(5, 100, 100), (20, 255, 255)],
        "紫色": [(130, 80, 80), (160, 255, 255)],
    }

    def __init__(self, camera_matrix: np.ndarray):
        self.K = camera_matrix

    def locate(self, rgb: np.ndarray, depth: np.ndarray, color: str) -> dict | None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lower, upper = self.COLOR_RANGES.get(color, self.COLOR_RANGES["红色"])
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
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
