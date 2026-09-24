# RK3588 端侧部署完整流程

> 与 refactor_plan_v9 对齐: **板端 Python 3.10 仅推理，不安装 LeRobot**；
> LeRobot（数据集处理 / ACT 训练 / ONNX 导出）在 PC 端 Python 3.12 环境。

## 资源索引

| 资源 | 链接 |
|------|------|
| LeRobot v0.6.1（仅 PC 端） | https://github.com/huggingface/lerobot |
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx |
| Qwen3.5 | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| SO-ARM100/101 | https://github.com/TheRobotStudio/SO-ARM100 |
| RealSense D435i | https://github.com/IntelRealSense/librealsense |
| librga | https://github.com/airockchip/librga |

## 1. 系统准备

```bash
# Ubuntu 22.04 / Arm64
sudo apt update
sudo apt install -y python3-pip python3-opencv cmake build-essential wget git ffmpeg
```

## 2. 项目安装（板端）

```bash
conda create -n rk3588 python=3.10 -y
conda activate rk3588

# 安装板端依赖（推理/相机/串口/语音/编码，不含 LeRobot）
cd /home/elf/work/rk3588-eia
pip install -r requirements.txt

# RKNN Lite（NPU 推理库）
pip install /path/to/rknn_toolkit_lite2-*-cp310-*.whl

# 验证
python3 -c "import cv2, numpy, sherpa_onnx, pyrealsense2; print('OK')"
```

PC 端（数据集处理 / ACT 训练 / 模型导出）另建环境：

```bash
conda create -n rk3588 python=3.12 -y
conda activate rk3588
pip install -r requirements-dev.txt   # torch / onnx / LeRobot v0.6.1
```

## 3. 模型部署

### VLM 模型

`models/vlm/Qwen3.5-0.8B/` 包含（gitignore，需单独部署到板端）：

```
demo                          # Qwen3.5 C++ demo 程序（RKLLM 子进程调用）
lib/                          # RKLLM 运行时库
Qwen3.5-0.8B_vision_rk3588.rknn   # 视觉编码器
Qwen3.5-0.8B_w8a8_rk3588.rkllm    # LLM 模型
demo.jpg                      # 占位图
```

路径由 `config/settings.py` 的 `VLM_MODEL_PATH` / `VLM_DEMO_BIN` 指向。

### ACT / GGCNN 模型

- `models/act/`: ACT ONNX（PC 端 `tools/export_act_onnx.py` 导出后拷贝到板端）
- `models/ggcnn/`: GGCNN ONNX（PC 端 `tools/export_ggcnn_onnx.py` 导出，Cornell 预训练权重）

### 语音模型

`voice_assistant/voice_assistant/models/` 包含：

```bash
voice_assistant/voice_assistant/models/
├── sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/   # KWS 唤醒词
├── sherpa-onnx-conformer-zh-stateless2-2023-05-23/   # ASR 语音识别
├── matcha-icefall-zh-baker/                          # TTS 声学模型
└── vocos-22khz-univ.onnx                             # TTS 声码器
```

## 4. 配置

```bash
# 默认配置直接使用；相机设备号/串口按实际检测修改：
#   config/settings.py            → CAMERA_INDEX / SERIAL_PORT（单一事实来源）
#   voice/config/default.yaml     → 语音链路模型路径
# 子进程运行时（runtime/）默认关闭：settings.USE_SUBPROCESS_RUNTIME = False
#   （需板端实测 GIL/延迟/NPU 绑定后再启用，见 refactor_plan_v9 T4.2）
```

## 5. 运行

```bash
conda activate rk3588
cd /home/elf/work/rk3588-eia

# 统一入口 — 5 种模式（autonomous/voice/teleop/record/menu）
python3 main.py
python3 menu.py            # 交互式菜单

# 语音助手 — 完整链路
python3 va.py once

# 文字问答
python3 va.py ask "画面中有什么"

# 语音触发动作
python3 scripts/voice_motion.py
```

## 6. 标定

```bash
# 相机取景检查
python3 scripts/d435i_viewer.py       # D435i 深度流
python3 scripts/camera_viewer.py      # USB 相机

# 相机内参标定（回写 config/settings.py CAMERA_MATRIX）
python3 scripts/calibrate_camera.py --d435i

# 相机→机械臂基座外参标定（回写 config/settings.py CAMERA_POSITION）
python3 scripts/calibrate_extrinsics.py

# 手眼标定辅助
python3 scripts/calibrate_handeye.py

# 舵机标定参数: config/calibration.json（SO101Arm 启动时加载）
```

> 标定脚本只回写 `config/settings.py`（单一事实来源）；
> `hardware/arm.py`、`policy/grasp_pipeline.py` 均引用 settings，无需手工同步副本。

## 7. 单元测试

```bash
python3 tests/test_kinematics.py      # 运动学 FK/IK 一致性（6 用例，仅 numpy）
python3 runtime/shared_frame.py       # 共享内存帧缓冲自检
```
