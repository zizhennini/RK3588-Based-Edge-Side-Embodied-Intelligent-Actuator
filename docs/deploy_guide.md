# RK3588 端侧部署完整流程

## 资源索引

| 资源 | 链接 |
|------|------|
| LeRobot v0.4.4 | https://github.com/huggingface/lerobot/tree/v0.4.4 |
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
sudo apt install -y python3-pip python3-opencv cmake build-essential wget git
```

## 2. 项目安装

```bash
conda create -n rkvla python=3.10 -y
conda activate rkvla

# 安装项目依赖
cd /home/elf/work/RK3588-EIA
pip install -r requirements.txt

# LeRobot
cd lerobot && pip install -e .[feetech] && cd ..

# RKNN Lite（NPU 推理库）
pip install /path/to/rknn_toolkit_lite2-*-cp310-*.whl

# 验证
python3 -c "import cv2, numpy, sherpa_onnx, pyrealsense2; print('OK')"
```

## 3. 模型部署

### VLM 模型

`models/Qwen3.5-0.8B/` 包含：

```
demo                          # Qwen3.5 C++ demo 程序
lib/                          # RKLLM 运行时库
Qwen3.5-0.8B_vision_rk3588.rknn   # 视觉编码器
Qwen3.5-0.8B_w8a8_rk3588.rkllm    # LLM 模型
demo.jpg                      # 占位图
```

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
# 默认配置直接使用
# 如需修改相机索引，编辑：
voice_assistant/config/default.yaml
  → audio > camera_index: 21   # D435i RGB
```

## 5. 运行

```bash
conda activate rkvla
cd /home/elf/work/RK3588-EIA

# 语音助手 — 完整链路
python3 va.py once

# 文字问答
python3 va.py ask "画面中有什么"

# VLA 自主抓取
python3 main.py

# 语音触发动作
python3 scripts/voice_motion.py
```

## 6. 标定

```bash
# 机械臂标定
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_awesome_leader_arm

# 相机标定
python3 astra/viewer.py
```
