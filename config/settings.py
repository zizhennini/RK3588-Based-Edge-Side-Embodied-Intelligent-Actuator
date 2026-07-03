"""VLA 系统配置 — 统一管理所有参数"""
import numpy as np


# ── 相机配置 ──
CAMERA_INDEX = 21  # 默认相机设备号
CAMERA_OVERHEAD = 21  # 俯拍上帝视角（XWF-1080p6）
CAMERA_ARM = 23       # 机械臂局部（icspring）
CAMERA_D435I = True   # 使用 D435i 深度相机
# 相机内参（需实标后替换）
CAMERA_MATRIX = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
# D435i 深度相机内参（自动获取）
D435_INTRINSICS = None  # 运行时自动填充

# ── 串口配置 ──
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 1000000

# ── VLM 配置 ──
VLM_MODEL_NAME = "qwen3-vl-2b"
VLM_MODEL_PATH = "./models/Qwen3-VL-2B"
VLM_DEMO_BIN = "/usr/bin/demo"

# ── MobileNet SSD 配置 ──
SSD_PROTOTXT = "./models/MobileNetSSD/MobileNetSSD_deploy.prototxt"
SSD_CAFFEMODEL = "./models/MobileNetSSD/MobileNetSSD_deploy.caffemodel"
SSD_CONFIDENCE = 0.5
