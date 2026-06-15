"""VLA 系统配置 — 统一管理所有参数"""
import numpy as np


# ── Astra Pro 相机内参（需实标后替换） ──
CAMERA_MATRIX = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# ── 串口配置 ──
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 1000000

# ── VLM 配置 ──
VLM_MODEL_NAME = "qwen3-vl-2b"
RKLLM_BIN = "/usr/bin/rkllm_demo"
VLM_MODEL_PATH = "./models/Qwen3-VL-2B"
# multimodal demo 可执行文件路径（Qwen3-VL / InternVL3 等需要）
VLM_DEMO_BIN = "/usr/bin/demo"

# ── MobileNet SSD 配置 ──
SSD_PROTOTXT = "./models/MobileNetSSD/MobileNetSSD_deploy.prototxt"
SSD_CAFFEMODEL = "./models/MobileNetSSD/MobileNetSSD_deploy.caffemodel"
SSD_CONFIDENCE = 0.5
