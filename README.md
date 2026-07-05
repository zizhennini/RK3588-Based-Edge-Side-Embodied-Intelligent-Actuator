# RK3588-EIA

**Embedded Intelligent Actuator** — 端侧具身智能教育平台

RK3588 NPU 端侧部署 VLA（Vision-Language-Action），支持语音交互 + VLM 理解 + 机械臂抓取，全链路在端侧运行，无需联网。

---

## 架构

```
🎤 语音输入 (sherpa-onnx ASR + KWS)
       ↓
📷 D435i 深度相机 ──► Qwen3.5-0.8B (NPU) ──► 文字描述 + 2D 坐标
       ↓
深度图 ──► 3D 反投影 (x, y, z)
       ↓
几何 IK ──► Feetech 串口 ──► SO-ARM101 抓取
       ↓
🔊 TTS 播报 (sherpa-onnx Matcha-TTS)
```

## 资源索引

| 资源 | 链接 |
|------|------|
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx |
| Intel RealSense | https://github.com/IntelRealSense/librealsense |
| Qwen3.5 | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| SO-ARM101 | https://github.com/TheRobotStudio/SO-ARM100 |
| librga | https://github.com/airockchip/librga |

## 项目结构

```
RK3588-EIA/
├── main.py                  # VLA 主入口
├── va.py                    # 语音助手入口
├── voice_assistant/         # 语音助手（sherpa-onnx）
│   ├── voice_assistant/     #   Python 包（ASR/TTS/KWS）
│   ├── config/              #   语音模块配置
│   └── scripts/             #   拍照脚本
├── vla/                     # VLA 核心
│   ├── vlm/                 #   VLM 引擎（Qwen3.5）
│   ├── control/             #   IK + Feetech 串口
│   ├── vision/              #   视觉定位
│   ├── pipe/                #   流水线状态机
│   └── command_queue.py     #   指令队列 + 中断
├── astra/                   # 相机模块（D435iCamera）
├── config/                  # 统一配置
│   ├── settings.py
│   ├── cpu_affinity.py      # 大小核算力隔离
│   └── memory.py            # 分级内存管控
├── scripts/                 # 测试/工具脚本
├── motion_library/          # 示教轨迹库
├── models/
│   └── Qwen3.5-0.8B/       # Qwen3.5 模型 (rknn + rkllm)
└── docs/                    # 文档
```

## 快速开始

```bash
conda activate rkvla
cd /home/elf/work/RK3588-EIA

# 语音助手 — 录音 → ASR → Qwen → TTS
python va.py once

# 语音唤醒循环
python va.py listen-forever

# 文字问答
python va.py ask "画面中有什么" --no-speak

# VLA 自主抓取
python main.py

# 语音触发动作回放
python scripts/voice_motion.py

# VLA 语音交互
python scripts/voice_vla.py voice
```

## 语音指令

| 命令 | 功能 |
|------|------|
| `va.py once` | 录音 → ASR → Qwen3.5 → TTS |
| `va.py ask <text>` | 文字 → Qwen3.5 |
| `va.py listen` | 唤醒词 → 录音 → 识别 → 回答 |
| `va.py listen-forever` | 循环唤醒 |
| `scripts/voice_motion.py` | 唤醒 → 指令 → 动作/VLM |
| `scripts/voice_vla.py voice` | 唤醒 → 指令 → SmartVLM → 机械臂 |

## 示教轨迹回放

| 命令 | 动作 |
|------|------|
| "你好" | 打招呼 |
| "抓" | 抓取 |
| "挥手" | 挥手 |
| "鞠躬" | 鞠躬 |
| "停止" | 紧急停止 |

## 核心特性

| 特性 | 说明 |
|------|------|
| 大小核算力隔离 | motion→小核(A55), inference→大核(A76) |
| 分级内存管控 | VLM 按需加载/闲置30s卸载, 录音与推理互斥 |
| 指令队列+中断 | FIFO 串行执行, 紧急中断清空队列+物理停机 |
| 全离线语音 | sherpa-onnx KWS + ASR + TTS, 无需联网 |

## 测试完成状态

| 模块 | 状态 |
|------|------|
| 机械臂6关节控制 | ✅ |
| D435i 深度相机 | ✅ |
| Qwen3.5-0.8B VLM | ✅ |
| KWS 唤醒词 | ✅ |
| ASR 语音识别 | ✅ |
| TTS 语音合成 | ✅ |
| 流式 TTS 播报 | ✅ |
| 示教录制→回放 | ✅ |
| 指令队列+中断 | ✅ |
| 分级内存管控 | ✅ |
| 大小核算力隔离 | ✅ |
