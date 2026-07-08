#!/usr/bin/env python3
"""最优方案视觉抓取 — VLM+PCA+6D 位姿 → IK 轨迹，无需示教轨迹"""

import sys, os, json, time, re, yaml, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera import D435iCamera
from vla.control import ArmController
from vla.kinematics import Kinematics, WORKSPACE
from config.settings import CAMERA_MATRIX, CAMERA_POSITION, SERIAL_PORT, SERIAL_BAUD

PROJ = Path(__file__).resolve().parent.parent
CFG = PROJ / "voice_assistant" / "config" / "default.yaml"

# ── IK 参考参数 ──
ARM_LENGTHS = [0.12, 0.15, 0.18]  # 大臂/小臂/腕部长度(m)


def load_qwen():
    import yaml as _yaml
    with open(CFG) as f:
        config = _yaml.safe_load(f)
    from voice_assistant.voice_assistant.qwen_runner import QwenRunner
    return QwenRunner(config)


def detect_object(qwen, target_desc: str, rgb, depth):
    """VLM 识别目标 → 像素坐标 (cx, cy) + 深度 z"""
    import cv2
    snap = "/tmp/vlm_pca_snap.jpg"
    cv2.imwrite(snap, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    prompt = f"<image>图像中有\"{target_desc}\"吗? 输出目标边界框: {{\"bbox_2d\":[x1,y1,x2,y2],\"label\":\"物体名\"}}。只输出JSON。"
    result = qwen.ask(snap, prompt)
    try:
        data = json.loads(re.search(r"\[.*\]|\{.*\}", result, re.DOTALL).group())
        if isinstance(data, list):
            data = data[0]
        bbox = data["bbox_2d"]
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        h, w = depth.shape[:2]
        cx, cy = np.clip(cx, 0, w-1), np.clip(cy, 0, h-1)
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        label = data.get("label", target_desc)
        return cx, cy, bbox_w, bbox_h, label
    except Exception as e:
        print(f"  VLM 解析失败: {e}")
        return None


def depth_to_3d(cx, cy, depth, camera_matrix=CAMERA_MATRIX) -> tuple:
    """像素 → 3D 坐标（邻域中值去噪）"""
    h, w = depth.shape[:2]
    cy, cx = min(cy, h-1), min(cx, w-1)
    z = float(depth[cy, cx])
    if z <= 0 or np.isnan(z):
        roi = depth[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
        valid = roi[(roi > 0) & (~np.isnan(roi))]
        z = float(np.median(valid)) if len(valid) > 0 else 0
    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
    ppx, ppy = camera_matrix[0,2], camera_matrix[1,2]
    x = (cx - ppx) * z / fx
    y = (cy - ppy) * z / fy
    return x, y, z


def pca_grasp_pose(depth, cx, cy, bbox_w, bbox_h):
    """PCA 计算抓取角度和夹爪宽度"""
    # 1. 提取 bbox 区域深度 ROI
    x1 = max(0, cx - int(bbox_w//2) - 10)
    x2 = min(depth.shape[1], cx + int(bbox_w//2) + 10)
    y1 = max(0, cy - int(bbox_h//2) - 10)
    y2 = min(depth.shape[0], cy + int(bbox_h//2) + 10)
    roi = depth[y1:y2, x1:x2]

    # 2. 有效深度点 → 3D 点云
    fx, fy = CAMERA_MATRIX[0,0], CAMERA_MATRIX[1,1]
    ppx, ppy = CAMERA_MATRIX[0,2], CAMERA_MATRIX[1,2]
    ys, xs = np.where((roi > 0.01) & (~np.isnan(roi)))
    if len(ys) < 10:
        return None, None, None
    zs = roi[ys, xs]
    points = np.zeros((len(ys), 3), dtype=np.float32)
    points[:, 0] = ((xs + x1 - 0.5) - ppx) * zs / fx
    points[:, 1] = ((ys + y1 - 0.5) - ppy) * zs / fy
    points[:, 2] = zs

    # 3. PCA 主方向（物体在桌面上的朝向）
    cov = np.cov(points[:, :2].T)  # 仅 X-Y 平面
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    main_dir = eigenvectors[:, -1]  # 主方向向量
    angle = np.degrees(np.arctan2(main_dir[1], main_dir[0]))

    # 4. 夹爪宽度 = bbox 短边（投影到深度平面）
    # 取 bbox 实际物理尺寸的较小值
    z_center = zs[len(zs)//2]
    metric_w = bbox_w * z_center / fx  # bbox 物理宽度
    metric_h = bbox_h * z_center / fy  # bbox 物理高度
    gripper_width = min(metric_w, metric_h) * 0.7  # 夹爪开度取 70%

    return angle, gripper_width, points


# 手眼标定旋转角（实测: 像素X+69px → 机器人 +X0.050m -Y0.104m）
_CAM_ANGLE = np.radians(101.9)
_CAM_C = np.cos(_CAM_ANGLE)
_CAM_S = np.sin(_CAM_ANGLE)

def robot_coords(x_cam, y_cam, z_cam, mode=0):
    """相机 → 从臂坐标系
       mode=0~3: 轴交换（兼容旧版）
       mode=4: 旋转矩阵（精确标定）
    """
    if mode == 5:
        rx = -(x_cam * _CAM_C - y_cam * _CAM_S)
        ry = x_cam * _CAM_S + y_cam * _CAM_C
    elif mode == 4:
        rx = x_cam * _CAM_C - y_cam * _CAM_S
        ry = x_cam * _CAM_S + y_cam * _CAM_C
    elif mode == 0:
        rx, ry = x_cam, y_cam
    elif mode == 1:
        rx, ry = -x_cam, y_cam
    elif mode == 2:
        rx, ry = y_cam, -x_cam
    elif mode == 3:
        rx, ry = -y_cam, x_cam
    else:
        rx, ry = x_cam, y_cam
    rx += CAMERA_POSITION[0]
    ry += CAMERA_POSITION[1]
    rz = CAMERA_POSITION[2] - z_cam
    return rx, ry, rz


def generate_grasp_trajectory(pos_3d, angle_deg, width_m, approach_dist=0.08, lift_dist=0.06):
    """从 6D 位姿生成接近→抓取→抬升三段轨迹
       接近方向：从正上方垂直下移（上帝视角最安全）
       PCA 角度通过 wrist_roll 关节实现夹爪旋转"""
    pre = np.array([pos_3d[0], pos_3d[1], pos_3d[2] + approach_dist])
    grasp = np.array(pos_3d)
    lift = np.array([pos_3d[0], pos_3d[1], pos_3d[2] + lift_dist])

    return {
        "approach_dist_m": approach_dist,
        "lift_dist_m": lift_dist,
        "gripper_width_m": round(width_m, 3),
        "grasp_angle_deg": round(angle_deg, 1),
        "pre_grasp": [round(v, 4) for v in pre],
        "grasp": [round(v, 4) for v in grasp],
        "lift": [round(v, 4) for v in lift],
    }


def execute_6d(arm, traj: dict):
    """执行三段轨迹：预抓取(带PCA角度) → 抓取 → 抬升"""
    pre = traj["pre_grasp"]
    grasp = traj["grasp"]
    lift = traj["lift"]
    angle_rad = np.radians(traj["grasp_angle_deg"])

    print(f"  接近: ({pre[0]:.3f}, {pre[1]:.3f}, {pre[2]:.3f})  wrist_roll={angle_rad:.2f}")
    arm.move_to(*pre, wrist_roll_rad=angle_rad, use_current=False)
    time.sleep(1)

    print(f"  抓取: ({grasp[0]:.3f}, {grasp[1]:.3f}, {grasp[2]:.3f})")
    arm.move_to(*grasp, wrist_roll_rad=angle_rad, use_current=False)
    time.sleep(0.5)

    w = traj["gripper_width_m"]
    print(f"  夹爪闭合 (宽度={w:.3f}m)")
    arm.gripper(False)
    time.sleep(0.5)

    print(f"  抬升: ({lift[0]:.3f}, {lift[1]:.3f}, {lift[2]:.3f})")
    arm.move_to(*lift, use_current=False)
    time.sleep(0.5)

    print("  ✅ 抓取完成")


def main():
    import argparse, cv2
    parser = argparse.ArgumentParser(description="最优方案视觉抓取 — VLM+PCA+6D")
    parser.add_argument("target", help="目标描述，如\"红色方块\"")
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--mode", type=int, default=0, choices=range(6),
                        help="坐标映射模式 (0-3=轴交换 4=旋转矩阵 5=旋转+X取反)")
    args = parser.parse_args()

    print(f"[1/5] 加载 Qwen3.5...")
    qwen = load_qwen()

    print(f"[2/5] 启动 D435i...")
    cam = D435iCamera()
    cam.connect()
    rgb, depth = cam.read()
    print(f"  RGB: {rgb.shape}  深度: {depth.shape}")

    print(f"[3/5] VLM 识别: {args.target}...")
    det = detect_object(qwen, args.target, rgb, depth)
    if det is None:
        print("  ❌ 未识别到目标"); cam.disconnect(); return
    cx, cy, bw, bh, label = det
    print(f"  识别: {label} @ 像素({cx},{cy})  bbox({bw:.0f}x{bh:.0f})")

    # PCA 计算
    print(f"[4/5] PCA 抓取位姿计算...")
    angle, width, points = pca_grasp_pose(depth, cx, cy, bw, bh)
    if angle is None:
        print("  ❌ 深度点云太少，无法计算朝向"); cam.disconnect(); return

    # 3D 坐标
    x_cam, y_cam, z_cam = depth_to_3d(cx, cy, depth)
    rx, ry, rz = robot_coords(x_cam, y_cam, z_cam, args.mode)
    print(f"  抓取角度: {angle:.1f}°")
    print(f"  夹爪宽度: {width:.3f}m")
    print(f"  从臂坐标: ({rx:.3f}, {ry:.3f}, {rz:.3f})")

    # 生成轨迹
    traj = generate_grasp_trajectory((rx, ry, rz), angle, width)
    print(f"  预抓取点: {[round(v,3) for v in traj['pre_grasp']]}")
    print(f"  抓取点:   {[round(v,3) for v in traj['grasp']]}")
    print(f"  抬升点:   {[round(v,3) for v in traj['lift']]}")

    cam.disconnect()

    # 安全校验
    print(f"\n[安全校验]")
    safe = True
    ws = WORKSPACE
    for name, pt in [("预抓取", traj["pre_grasp"]), ("抓取", traj["grasp"]), ("抬升", traj["lift"])]:
        for j, (v, lo, hi) in enumerate(zip(pt, [ws[0],ws[2],ws[4]], [ws[1],ws[3],ws[5]])):
            if v < lo or v > hi:
                print(f"  ⚠ {name} {['X','Y','Z'][j]}: {v:.3f} 超限 [{lo:.3f},{hi:.3f}]")
                safe = False
    if not safe:
        print("  ❌ 安全校验未通过"); return
    print("  ✅ 安全校验通过")

    # 人工复核
    if not args.no_verify:
        print(f"\n[人工复核]")
        try:
            r = input(f"  角度={angle:.1f}° 宽度={width:.3f}m 坐标=({rx:.3f},{ry:.3f},{rz:.3f}) 确认执行? (y/n): ").strip().lower()
            if r != "y":
                print("  已取消"); return
        except (EOFError, KeyboardInterrupt):
            print("  已取消"); return

    # 执行
    if args.dry_run:
        print(f"\n[5/5] 模拟运行 (dry-run)")
        print(f"  三段轨迹: 接近 {traj['pre_grasp']} → 抓取 {traj['grasp']} → 抬升 {traj['lift']}")
        return

    print(f"\n[5/5] 执行抓取...")
    arm = ArmController(args.port, SERIAL_BAUD)
    try:
        execute_6d(arm, traj)
    finally:
        arm.home(steps=30, delay_s=0.03)
        arm.close()
    print("  ✅ 完成")


if __name__ == "__main__":
    main()
