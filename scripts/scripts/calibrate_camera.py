#!/usr/bin/env python3
"""相机内参标定 — 使用棋盘格标定 Astra Pro RGB 相机"""
import cv2
import numpy as np
import glob


CHESSBOARD_SIZE = (7, 10)
SQUARE_MM = 25.0


def calibrate(image_dir: str = "./calib_images"):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_MM

    objpoints, imgpoints = [], []
    images = glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print(f"标定完成，使用 {len(images)} 张图像")
    print(f"相机内参矩阵:\n{K}")
    print(f"畸变系数:\n{dist}")
    print(f"重投影误差: {ret:.4f}")
    return K, dist


if __name__ == "__main__":
    calibrate()
