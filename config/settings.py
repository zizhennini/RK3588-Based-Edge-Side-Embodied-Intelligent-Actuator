"""VLA 系统配置 — 统一管理所有参数"""
import numpy as np


# ── 相机配置 ──
CAMERA_INDEX = 21  # D435i RGB 相机设备号
CAMERA_OVERHEAD = 27  # 上帝视角（USB 高清摄像头，用于实验录像）
CAMERA_ARM = 23       # 机械臂局部（icspring）
# 相机内参（D435i 出厂标定，可通过 scripts/calibrate_camera.py --d435i 更新）
CAMERA_MATRIX = np.array([
    [604.2294, 0.0, 315.1330],
    [0.0, 604.0748, 250.8858],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
# 相机→机械臂基座外参（实测填入，单位：米）
# 相机光心在机械臂基座坐标系下的位置
CAMERA_POSITION = np.array([0.182, -0.129, 0.47], dtype=float)  # [x, y, z] 标定值

# ── 串口配置 ──
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 1000000

# ── VLM 配置 ──
VLM_MODEL_NAME = "qwen3.5"
VLM_MODEL_PATH = "./models/Qwen3.5-0.8B"
VLM_DEMO_BIN = "./models/Qwen3.5-0.8B/demo"

# ── 内存管理配置 ──
VLM_IDLE_UNLOAD_TIMEOUT = 30  # VLM 闲置秒数后自动卸载
VLM_MEMORY_BUDGET_MB = 900     # VLM 推理预估内存占用
RECORDING_MEMORY_BUDGET_MB = 256  # 录像编码预估内存占用
MEMORY_RESERVE_MB = 200         # 系统预留内存余量

# ── MobileNet SSD 配置 ──
SSD_PROTOTXT = "./models/MobileNetSSD/MobileNetSSD_deploy.prototxt"
SSD_CAFFEMODEL = "./models/MobileNetSSD/MobileNetSSD_deploy.caffemodel"
SSD_CONFIDENCE = 0.5
