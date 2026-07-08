#!/usr/bin/env python3
"""相机内参标定 — D435i 出厂标定读取 / 棋盘格标定"""
import cv2
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def load_d435i_intrinsics() -> dict:
    """从 D435i 读取出厂标定参数"""
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pipe.stop()
    return {
        "fx": intr.fx, "fy": intr.fy,
        "cx": intr.ppx, "cy": intr.ppy,
        "coeffs": list(intr.coeffs),
        "model": str(intr.model),
    }

def update_settings(intr: dict):
    """将标定参数写入 config/settings.py"""
    import re
    path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.py")
    with open(path) as f:
        content = f.read()
    new_matrix = f"CAMERA_MATRIX = np.array([\n    [{intr['fx']:.4f}, 0.0, {intr['cx']:.4f}],\n    [0.0, {intr['fy']:.4f}, {intr['cy']:.4f}],\n    [0.0, 0.0, 1.0],\n], dtype=np.float64)"
    content = re.sub(r"CAMERA_MATRIX = np\.array\(\[.*?\],\s*dtype=np\.float64\)", new_matrix, content, flags=re.DOTALL)
    with open(path, "w") as f:
        f.write(content)
    print(f"已更新 CAMERA_MATRIX")


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
    import argparse
    p = argparse.ArgumentParser(description="相机标定")
    p.add_argument("--d435i", action="store_true", help="从 D435i 读取出厂标定")
    p.add_argument("--apply", action="store_true", help="将标定参数写入 settings.py")
    p.add_argument("--image-dir", default="./calib_images", help="棋盘格图片目录(仅棋盘格模式)")
    args = p.parse_args()

    if args.d435i:
        print("读取 D435i 出厂标定...")
        intr = load_d435i_intrinsics()
        print(f"  fx={intr['fx']:.4f}, fy={intr['fy']:.4f}")
        print(f"  cx={intr['cx']:.4f}, cy={intr['cy']:.4f}")
        print(f"  畸变: {intr['model']} {intr['coeffs']}")
        if args.apply:
            update_settings(intr)
    else:
        K, dist = calibrate(args.image_dir)
        if args.apply:
            intr = {"fx": K[0,0], "fy": K[1,1], "cx": K[0,2], "cy": K[1,2]}
            update_settings(intr)
