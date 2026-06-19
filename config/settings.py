"""VLA 系统配置 — 统一管理所有参数"""
import numpy as np


# ── 相机配置 ──
CAMERA_INDEX = 21  # USB 相机设备号
# 相机内参（需实标后替换）
CAMERA_MATRIX = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

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
