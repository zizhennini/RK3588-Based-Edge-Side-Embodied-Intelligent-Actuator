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
# 舵机标定文件路径 (refactor_plan_v9 §6.5)
SERVO_CALIB = "./config/calibration.json"

# ── VLM 配置 ──
VLM_MODEL_NAME = "qwen3.5"
VLM_MODEL_PATH = "./models/vlm/Qwen3.5-0.8B"
VLM_DEMO_BIN = "./models/vlm/Qwen3.5-0.8B/demo"

# ── 内存管理配置 ──
VLM_IDLE_UNLOAD_TIMEOUT = 30  # VLM 闲置秒数后自动卸载
VLM_MEMORY_BUDGET_MB = 900     # VLM 推理预估内存占用
RECORDING_MEMORY_BUDGET_MB = 256  # 录像编码预估内存占用
MEMORY_RESERVE_MB = 200         # 系统预留内存余量

# ── 子进程运行时配置 (refactor_plan_v9 §1.6 / §4.1) ──
# True 时相机/ACT/GGCNN 的 CPU 推理走独立子进程 + 共享内存帧传输 (runtime/)，
# 规避 Python GIL 抖动、适配 RK3588 NPU 单进程单核限制；
# False（默认）走已验证的进程内路径。
# 注意: 完整多进程编排需板端实测后启用 (T4.2 / 风险 R2)。
USE_SUBPROCESS_RUNTIME = False

# 各子进程 CPU 绑核（A55 小核 0-3: 相机/串口/语音/安全；A76 大核 4-7: 推理）
# 子进程不继承父进程亲和性，runtime.worker 在子进程内用以下集合重新绑核。
CORES_CAMERA = {0, 1}     # 相机采集 (A55, 30Hz)
CORES_ARM_IO = {1}        # 串口 IO (A55, 50Hz)
CORES_VOICE = {2, 3}      # 语音 KWS/ASR/TTS (A55)
CORES_SAFETY = {2, 3}     # 安全监控 (A55)
CORES_ACT = {4, 5}        # ACT 推理 (A76)
CORES_GGCNN = {5}         # GGCNN 抓取检测 (A76)
CORES_VLM = {6, 7}        # VLM 推理 (A76, 已是 RKLLM 子进程)

# 共享内存帧缓冲名称（相机子进程为生产者，推理子进程为消费者）
SHARED_FRAME_NAME = "eia_camera_frame"
