# RK3588-EIA 技术手册

> 端侧具身智能教育平台 — 命令参考 · 开发指南

## 环境管理

```bash
conda activate rkvla
python3 --version  # Python 3.10
```

## 语音助手 (va.py)

```bash
cd /home/elf/work/RK3588-EIA

# 录音 → ASR → VLM → TTS（完整链路）
python3 va.py once --seconds 4

# 文字问答（跳过语音）
python3 va.py ask "画面中有什么" --no-speak

# 唤醒一次
python3 va.py listen --wake-mode kws

# 循环唤醒
python3 va.py listen-forever

# 纯录音
python3 va.py record --seconds 3

# 语音识别
python3 va.py stt input.wav

# 纯 TTS 合成
python3 va.py tts-stream "你好"
```

### va.py 参数

| 参数 | 说明 |
|------|------|
| `--seconds N` | 录音秒数（默认 6） |
| `--no-speak` | 不播报（静默模式） |
| `--no-play` | 不播放 |
| `--wake-mode kws/stt` | 唤醒模式 |
| `--image PATH` | 指定图片 |

## 语音模块文件结构

```
voice_assistant/
├── config/default.yaml       # 语音模块配置（模型路径、音频参数）
├── scripts/capture-photo.sh  # 拍照脚本（备用）
└── voice_assistant/          # Python 包
    ├── cli.py                # 命令行入口
    ├── orchestrator.py       # 流程编排（含 MemoryLimiter）
    ├── asr.py                # sherpa-onnx 语音识别
    ├── tts.py                # sherpa-onnx 语音合成
    ├── streaming_tts.py      # 流式 TTS（缓冲防 underrun）
    ├── wake.py               # KWS 唤醒词检测
    ├── audio_io.py           # 录音/播放
    ├── camera.py             # 相机拍照（pyrealsense2）
    ├── intent.py             # 意图识别
    ├── qwen_runner.py        # Qwen3.5 demo 子进程通信
    └── models/               # 语音模型文件
        ├── sherpa-onnx-conformer-*/     # ASR
        ├── sherpa-onnx-kws-*/          # KWS
        ├── matcha-icefall-zh-baker/    # TTS
        └── vocos-22khz-univ.onnx       # 声码器
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
packet = [0xFF, 0xFF, ID, LEN, INST, ADDR, DATA..., CKSUM]
CKSUM = ~sum(packet[2:]) & 0xFF
```

## 指令队列 (CommandQueue)

```python
from vla.command_queue import CommandQueue, create_voice_motion_callback

q = CommandQueue(smart_vlm=vlm, arm=arm)
q.start()
on_text = create_voice_motion_callback(q)

# 语音指令自动入队
on_text("你好")   # → motion 任务
on_text("停止")   # → 紧急中断

q.stop()
```

### 任务类型

| 类型 | 触发 | 说明 |
|------|------|------|
| `motion` | 关键词匹配 | 回放示教轨迹 |
| `vlm_ask` | 默认 | VLM 问答 |
| `vlm_grasp` | "抓" | VLM 抓取 |
| `tts` | 播报 | TTS 语音 |
| `emergency_stop` | "停止" | 清空队列 + 物理停机 |

## 分级内存管控

```python
from config.memory import MemoryMonitor, MemoryLimiter

mem = MemoryMonitor()
mem.acquire("vlm")    # 申请 900MB VLM 预算
mem.release("vlm")    # 释放

with MemoryLimiter(mem, "recording"):  # 自动释放冲突组件
    do_recording()
```

## 大小核算力隔离

| 核心 | 类型 | 绑定组件 |
|------|------|---------|
| 0-3 | Cortex-A76（大核） | VLM 推理、主线程、语音 |
| 4-7 | Cortex-A55（小核） | 机械臂控制、相机采集 |

```python
from config.cpu_affinity import bind_current_thread, BIG_CORES, LITTLE_CORES

bind_current_thread(BIG_CORES)      # 大核：推理
bind_current_thread(LITTLE_CORES)   # 小核：运动
```

## VLA 流水线

```bash
# 全自主抓取
python3 main.py

# 语音 VLA
python3 scripts/voice_vla.py voice
python3 scripts/voice_vla.py text
```

## LeRobot 命令

| 命令 | 用途 |
|------|------|
| `lerobot-teleoperate` | 主从遥操作 |
| `lerobot-calibrate` | 机器人标定 |
| `lerobot-find-port` | 查找串口 |
| `lerobot-setup-motors` | 舵机配置 |
