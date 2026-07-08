#!/usr/bin/env python3
"""ArUco 手眼标定 — 打印二维码贴桌面，从臂触碰标记点，拍照自动计算相机位姿"""

import sys, os, time, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def generate_marker():
    """生成 ArUco 标记并保存为 PDF 可打印文件"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    img = cv2.aruco.generateImageMarker(aruco_dict, 1, 400)
    cv2.imwrite("/tmp/aruco_marker.png", img)
    print("ArUco 标记已生成: /tmp/aruco_marker.png")
    print(f"尺寸: {img.shape[1]}x{img.shape[0]}px")
    print("请将此图打印出来（建议边长 4cm），贴在桌面上")
    return "/tmp/aruco_marker.png"


def detect_marker(img):
    """检测 ArUco 标记，返回角点+ID+位姿"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return None, None, None
    return corners[0], ids[0][0], None


def main():
    from camera import D435iCamera
    from vla.control import ArmController
    from config.settings import CAMERA_MATRIX

    print("=" * 50)
    print("  ArUco 手眼标定")
    print("=" * 50)
    print()

    # 生成标记
    marker_path = generate_marker()
    input("请打印 ArUco 标记并贴在桌面上，然后按 Enter...")

    # 启动相机
    cam = D435iCamera()
    cam.connect()

    # 检测 ArUco
    print("\n检测 ArUco 标记...")
    for _ in range(20):
        rgb, _ = cam.read()
        corners, marker_id, _ = detect_marker(rgb)
        if corners is not None:
            break
        time.sleep(0.1)

    if corners is None:
        print("未检测到 ArUco 标记")
        cam.disconnect()
        return

    # 标记的 4 个角点
    marker_size_m = 0.04  # 4cm
    obj_points = np.array([
        [-marker_size_m/2,  marker_size_m/2, 0],
        [ marker_size_m/2,  marker_size_m/2, 0],
        [ marker_size_m/2, -marker_size_m/2, 0],
        [-marker_size_m/2, -marker_size_m/2, 0],
    ], dtype=np.float32)

    # 解算位姿
    _, rvec, tvec = cv2.solvePnP(obj_points, corners, CAMERA_MATRIX, None)
    R_marker2cam, _ = cv2.Rodrigues(rvec)
    T_marker2cam = tvec.flatten()

    # 标记中心在相机坐标系下的位置
    center_cam = T_marker2cam.copy()
    print(f"\n检测到 ArUco ID={marker_id}")
    print(f"标记中心在相机坐标: ({center_cam[0]:.3f}, {center_cam[1]:.3f}, {center_cam[2]:.3f})m")

    cam.disconnect()

    # 让用户移动从臂到标记位置
    print(f"\n请将机器人从臂的夹爪移动到 ArUco 标记中心正上方")
    input(f"然后按 Enter 读取当前位置...")

    arm = ArmController("/dev/ttyACM0")
    joints = arm._read_current_pos()
    arm.close()

    from vla.control.controller import ArmController as AC
    import math
    # 粗略估算末端位置（基于关节角度）
    j = [np.rad2deg(j) for j in joints]
    l1, l2 = 0.12, 0.15
    # SO-101 简化 FK
    theta2 = np.radians(j[2])
    theta1 = np.radians(j[1])
    pan = np.radians(j[0])
    x_approx = (l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)) * np.cos(pan)
    y_approx = (l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)) * np.sin(pan)
    z_approx = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + 0.06

    print(f"\n关节角度: {[round(j,1) for j in j]}")
    print(f"末端近似位置: ({x_approx:.3f}, {y_approx:.3f}, {z_approx:.3f})m")
    print(f"标记相机坐标: ({center_cam[0]:.3f}, {center_cam[1]:.3f})m")
    print(f"标记机器人坐标: ({x_approx:.3f}, {y_approx:.3f})m")

    # 计算相机→基座的旋转平移
    # robot_pos = R * cam_pos + T
    cam_v = np.array([center_cam[0], center_cam[1]])
    robot_v = np.array([x_approx, y_approx])

    if np.linalg.norm(cam_v) > 0.01:
        angle = np.degrees(np.arctan2(robot_v[1], robot_v[0]) - np.arctan2(cam_v[1], cam_v[0]))
        print(f"\n旋转角: {angle:.1f}°")
        R = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                       [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]])
        # T = robot_pos - R * cam_pos
        T = robot_v - R @ cam_v
        print(f"相机位姿: X={T[0]:.3f}m, Y={T[1]:.3f}m, Z=0.47m")
        print(f"旋转角: {angle:.1f}°")
        print(f"\n建议更新 CAMERA_POSITION = [{T[0]:.3f}, {T[1]:.3f}, 0.47]")
        print(f"_CAM_ANGLE = {angle:.1f}")


if __name__ == "__main__":
    main()
