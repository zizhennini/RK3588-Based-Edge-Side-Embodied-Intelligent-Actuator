"""IPM 逆透视变换 + 畸变矫正 — 将相机斜视角转为俯视图"""
import cv2
import numpy as np


class IPM:
    def __init__(self):
        # 相机内参（需要标定后替换）
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((1, 5), dtype=np.float64)

        # 源点：原始图像中工作区的四个角（需实测）
        self.src_pts = np.float32([
            [50, 120],   # 左上
            [50, 360],   # 左下
            [590, 360],  # 右下
            [590, 120],  # 右上
        ])

        # 目标点：俯视图中的矩形
        self.dst_pts = np.float32([
            [0, 0],
            [0, 299],
            [299, 299],
            [299, 0],
        ])

        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.out_size = (300, 300)

    def calibrate(self, chessboard_size=(9, 6), square_size=25.0, image_paths=None):
        """棋盘格标定相机内参"""
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        obj_points = []
        img_points = []

        if image_paths:
            for path in image_paths:
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                if ret:
                    obj_points.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    img_points.append(corners2)

            if len(obj_points) > 0:
                ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
                    obj_points, img_points, gray.shape[::-1], None, None
                )
                return True
        return False

    def set_workspace(self, src_pts, out_size=(300, 300)):
        """设置工作区四个角（像素坐标）"""
        self.src_pts = np.float32(src_pts)
        self.out_size = out_size
        self.dst_pts = np.float32([
            [0, 0],
            [0, out_size[1] - 1],
            [out_size[0] - 1, out_size[1] - 1],
            [out_size[0] - 1, 0],
        ])
        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

    def undistort(self, img):
        """畸变矫正"""
        h, w = img.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None,
            self.camera_matrix, (w, h), cv2.CV_32FC1
        )
        return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    def transform(self, img):
        """透视变换 → 俯视图"""
        return cv2.warpPerspective(img, self.H, self.out_size, flags=cv2.INTER_LINEAR)

    def pixel_to_world(self, u, v, mm_per_pixel=1.0):
        """俯视图像素坐标 → 桌面毫米坐标"""
        return u * mm_per_pixel, v * mm_per_pixel

    def draw_debug(self, img):
        """在原始图上画出工作区"""
        out = img.copy()
        pts = self.src_pts.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        for pt in self.src_pts:
            cv2.circle(out, tuple(pt.astype(int)), 4, (0, 0, 255), -1)
        return out
