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
VLM_MODEL_NAME = "smolvlm2-256m"
RKLLM_BIN = "/usr/bin/rkllm_demo"
VLM_MODEL_PATH = "./models/SmolVLM-256M"
VLM_DEMO_BIN = "/usr/bin/demo"
