# RK3588-EIA 技术手册

> 端侧具身智能教育平台 — 命令参考 · 开发指南

## 资料链接

| 内容 | 地址 |
|------|------|
| LeRobot v0.4.4 | https://github.com/huggingface/lerobot/tree/v0.4.4 |
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx |
| Qwen3-VL | https://github.com/QwenLM/Qwen3-VL |
| SO-ARM101 | https://github.com/TheRobotStudio/SO-ARM100 |

## 环境管理

```bash
# rkvla（主力环境）
conda activate rkvla

# 查看已安装包
conda list
```

## 机械臂控制

### 关节限位（实测）

| 关节 | mid | min | max | 物理范围 |
|------|-----|-----|-----|---------|
| shoulder_pan | 2117 | 946 | 3287 | ±103° |
| shoulder_lift | 2014 | 821 | 3206 | -105°~+102° |
| elbow_flex | 1997 | 888 | 3105 | ±97° |
| wrist_flex | 2022 | 851 | 3192 | ±103° |
| wrist_roll | 2058 | 130 | 3985 | -169°~+170° |
| gripper | 2178 | 1495 | 2860 | — |

### 角度→脉冲公式

```python
mid = (range_min + range_max) / 2
pulse = int(deg * 4095 / 360 + mid)
```

### 串口协议

```python
packet = 0xFF 0xFF ID LEN INST ADDR DATA CKSUM
LEN = INST + ADDR + DATA_SIZE + CKSUM = 5 (写2字节)
```

## 语音助手

```bash
cd voice_assistant

# 语音唤醒监听
python voice_assistant.py listen

# 单次指令
python voice_assistant.py once

# 文字问Qwen（带图片）
python voice_assistant.py ask "描述图片" --image demo.jpg --no-speak

# 录音
python voice_assistant.py record --seconds 3

# 语音识别
python voice_assistant.py stt input.wav

# 语音合成
python voice_assistant.py tts "你好"
```

## LeRobot 命令

| 命令 | 用途 |
|------|------|
| `lerobot-teleoperate` | 主从遥操作 |
| `lerobot-calibrate` | 机器人标定 |
| `lerobot-find-port` | 查找串口 |
| `lerobot-setup-motors` | 舵机配置 |
