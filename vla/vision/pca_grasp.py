"""PCA 抓取位姿生成 — 基于物体掩码点云的主方向分析"""
import numpy as np


class PCAGrasper:
    """从物体分割掩码 + 深度图生成抓取位姿（中心、角度、夹爪宽度）"""

    def __init__(self, camera_matrix: np.ndarray):
        self.K = camera_matrix
        self.fx = float(camera_matrix[0, 0])
        self.fy = float(camera_matrix[1, 1])
        self.cx = float(camera_matrix[0, 2])
        self.cy = float(camera_matrix[1, 2])

    def compute_from_mask(self, mask: np.ndarray, depth: np.ndarray) -> dict | None:
        """从二值掩码和深度图计算抓取位姿

        参数:
            mask:  二值掩码 (H, W)，非零为物体区域
            depth: 深度图 (H, W)，单位米，float32

        返回:
            dict | None: {
                "center_cam": [x, y, z],  相机坐标系抓取中心
                "angle": float,            抓取角度（弧度，相机 XY 平面）
                "width": float,            自适应夹爪宽度（米）
                "point_count": int,        有效点云数量
            }
        """
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            return None

        zs = depth[ys, xs].astype(np.float32)

        valid = (zs > 0.05) & (zs < 2.0) & (~np.isnan(zs)) & (~np.isinf(zs))
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        if len(xs) < 10:
            return None

        # 反投影到三维点云
        px = (xs.astype(np.float32) - self.cx) * zs / self.fx
        py = (ys.astype(np.float32) - self.cy) * zs / self.fy
        points = np.column_stack([px, py, zs])

        # Z 轴百分位去离群（去除桌面/背景）
        z_low = np.percentile(zs, 5)
        z_high = np.percentile(zs, 95)
        inlier = (zs >= z_low) & (zs <= z_high)
        object_points = points[inlier]
        if len(object_points) < 10:
            object_points = points

        # PCA 主成分分析
        centroid = np.mean(object_points, axis=0)
        centered = object_points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 主轴方向（最大特征值对应）
        main_axis = eigenvectors[:, -1]

        # 抓取角度：主轴在相机 XY 平面上的投影角
        angle = float(np.arctan2(main_axis[1], main_axis[0]))

        # 夹爪宽度：沿次轴方向物体跨度
        second_axis = eigenvectors[:, -2]
        proj = np.dot(centered, second_axis)
        width = max(float(np.ptp(proj)), 0.02)

        return {
            "center_cam": centroid.tolist(),
            "angle": angle,
            "width": width,
            "point_count": len(object_points),
        }

    def compute_from_bbox(self, bbox, depth):
        """从矩形 bbox 裁剪深度后计算抓取位姿

        参数:
            bbox: (x1, y1, x2, y2)
            depth: 深度图 (H, W)
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi = depth[y1:y2, x1:x2]
        mask = np.zeros_like(depth, dtype=np.uint8)
        mask[y1:y2, x1:x2] = (roi > 0.05).astype(np.uint8) * 255
        return self.compute_from_mask(mask[y1:y2, x1:x2], roi)
