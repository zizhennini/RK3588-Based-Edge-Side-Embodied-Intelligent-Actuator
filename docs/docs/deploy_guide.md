# RK3588 端侧部署完整流程

## 资源索引

| 资源 | 链接 |
|------|------|
| LeRobot v0.4.4 | https://github.com/huggingface/lerobot/tree/v0.4.4 |
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx |
| Qwen3-VL | https://github.com/QwenLM/Qwen3-VL |
| SO-ARM100/101 | https://github.com/TheRobotStudio/SO-ARM100 |

## 1. 系统准备

```bash
# Ubuntu 22.04 / Arm64
sudo apt update
sudo apt install -y python3-pip python3-opencv cmake build-essential wget
```

## 2. 项目安装

```bash
# 创建 conda 环境
conda create -n rkvla python=3.10 -y
conda activate rkvla

# LeRobot
cd lerobot && pip install -e .[feetech] && cd ..

# 项目依赖
pip install -r requirements.txt

# RKNN Lite（NPU 推理库）
pip install /path/to/rknn_toolkit_lite2-*-cp310-*.whl

# 启动相机取景
python scripts/camera_viewer.py
```

## 3. VLM 模型部署

模型文件放到 `models/Qwen3-VL-2B/`：

```
qwen3-vl-2b_vision_rk3588.rknn   # 视觉编码器
qwen3-vl-2b-instruct_w8a8_rk3588.rkllm  # LLM
```

## 4. 语音模型部署

```bash
# sherpa-onnx 语音模型放到 voice_assistant/models/
voice_assistant/models/
├── sherpa-onnx-kws-zipformer-zh-en-3M-*/   # KWS 唤醒词
├── sherpa-onnx-conformer-zh-stateless2-*/  # ASR 语音识别
├── matcha-icefall-zh-baker/                # TTS 声学模型
└── vocos-22khz-univ.onnx                   # TTS 声码器
```

## 5. 标定

```bash
# 机械臂标定
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_awesome_leader_arm
```

## 6. 运行

```bash
# 语音助手
conda activate rkvla
cd voice_assistant && python voice_assistant.py once

# VLA 自主抓取
python main.py
```
