#!/usr/bin/env python3
"""手眼标定 — 计算相机坐标系到机械臂基坐标系的变换"""
import cv2
import numpy as np


CHESSBOARD_SIZE = (7, 10)
SQUARE_MM = 25.0


class HandEyeCalibrator:
    """使用 Tsai-Lenz 方法进行手眼标定（眼在手上）"""

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []

    def add_pose(self, rvec_gripper, t_gripper, rgb: np.ndarray):
        """添加一组机械臂位姿 + 对应棋盘格图像"""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if not ret:
            return False
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        objp *= SQUARE_MM
        _, rvec, tvec = cv2.solvePnP(objp, corners, self.K, self.dist)
        R, _ = cv2.Rodrigues(rvec_gripper)
        self.R_gripper2base.append(R)
        self.t_gripper2base.append(t_gripper)
        self.R_target2cam.append(cv2.Rodrigues(rvec)[0])
        self.t_target2cam.append(tvec)
        return True

    def calibrate(self) -> np.ndarray:
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            self.R_gripper2base, self.t_gripper2base,
            self.R_target2cam, self.t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )
        H = np.eye(4)
        H[:3, :3] = R_cam2gripper
        H[:3, 3] = t_cam2gripper.flatten()
        return H
