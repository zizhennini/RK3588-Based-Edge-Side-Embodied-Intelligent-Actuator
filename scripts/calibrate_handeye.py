#!/usr/bin/env python3
"""手眼标定 — 基于 OpenCV calibrateHandEye，自动采集棋盘格数据并计算相机→机械臂变换"""

import sys, os, time, glob, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PATTERN = (7, 5)  # 棋盘格内角点 (宽, 高)
SQUARE_MM = 25    # 棋盘格方格大小(mm)
CHESSBOARD_DIR = "/tmp/chessboard_data"


def main():
    from vla.control import ArmController
    from camera import D435iCamera
    from config.settings import CAMERA_MATRIX

    print("=" * 50)
    print("  手眼标定 — 基于 OpenCV calibrateHandEye")
    print("=" * 50)
    print()
    print("请在从臂夹爪上固定棋盘格（%d×%d 内角点，%dmm方格）" % (PATTERN[0], PATTERN[1], SQUARE_MM))
    print("从臂将自动移动采集 %d 组数据" % 15)
    input("准备好后按 Enter...")
    print()

    os.makedirs(CHESSBOARD_DIR, exist_ok=True)
    cam = D435iCamera()
    cam.connect()
    arm = ArmController("/dev/ttyACM0")
    arm.home(steps=30)

    # 采集 15 个不同位姿
    poses = []
    for i in range(15):
        rx = 0.15 + 0.12 * np.sin(i * 0.8)
        ry = 0.08 * np.sin(i * 1.2 + 0.5)
        rz = 0.10 + 0.04 * np.sin(i * 0.6)
        print(f"  采集 {i+1}/15: ({rx:.3f},{ry:.3f},{rz:.3f})...", end=" ", flush=True)
        arm.move_to(rx, ry, rz)
        time.sleep(1.5)

        # 拍照
        rgb, _ = cam.read()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, PATTERN)
        if not ret:
            print("未检测到棋盘格")
            continue

        # 亚像素角点
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        print(f"✅ 检测到 {len(corners)} 个角点")

        # 保存数据
        _, rvec, tvec = cv2.solvePnP(
            np.zeros((PATTERN[0]*PATTERN[1], 3), np.float32),
            corners, CAMERA_MATRIX, None)
        R_target2cam, _ = cv2.Rodrigues(rvec)

        # 读取当前关节角 → T_base2ee
        joints = arm._read_current_pos()
        T_base2ee = np.eye(4)
        # 简化: 用 IK 算末端位姿
        T_base2ee[:3, :3] = cv2.Rodrigues(np.array([joints[0], joints[1], joints[2]]))[0]
        T_base2ee[:3, 3] = [rx, ry, rz]

        poses.append((R_target2cam, tvec, T_base2ee))

    arm.home(steps=30)
    arm.close()
    cam.disconnect()

    if len(poses) < 5:
        print("\n标定点不足，请确保棋盘格清晰可见")
        return

    print(f"\n共采集 {len(poses)} 组有效数据")

    # 解算手眼标定
    R_target2cam_list = [p[0] for p in poses]
    t_target2cam_list = [p[1] for p in poses]
    T_base2ee_list = [p[2] for p in poses]

    # 解 AX = XB
    print("\n=== 标定结果 ===")
    for method in range(5):
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_target2cam_list, t_target2cam_list,
            [T[:3,:3] for T in T_base2ee_list],
            [T[:3,3] for T in T_base2ee_list],
            method=method)
        print(f"\n  Method {method}:")
        print(f"    旋转矩阵:\n{R_cam2base.round(4)}")
        print(f"    平移: [{t_cam2base[0,0]:.4f}, {t_cam2base[1,0]:.4f}, {t_cam2base[2,0]:.4f}]")

    # 推荐结果（平移向量应与 CAMERA_POSITION 一致）
    print(f"\n  当前 CAMERA_POSITION = [0.0, 0.21, 0.47]")
    print(f"  建议选择平移最接近此值的 method")


if __name__ == "__main__":
    main()
