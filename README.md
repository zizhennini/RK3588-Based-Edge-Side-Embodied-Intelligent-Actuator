# RK3588-EIA

**Embedded Intelligent Actuator** — 端侧具身智能教育平台

RK3588 NPU 端侧部署 VLA（Vision-Language-Action），支持语音交互 + VLM 理解 + 机械臂抓取，全链路在端侧运行，无需联网。

---

## 架构

```
🎤 语音输入 (sherpa-onnx ASR + KWS)
       ↓
📷 D435i 深度相机 ──► Qwen3.5-0.8B (NPU) ──► 文字描述 + 2D grounding 坐标
       ↓                          (视觉 ~0.7s + LLM ~2.1s)
深度图 ──► 3D 反投影 (x, y, z)
       ↓
IK ──► 串口 ──► SO-ARM101 抓取
       ↓
🔊 语音播报 (espeak-ng)
```

## 资源索引

| 资源 | 链接 |
|------|------|
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx |
| Intel RealSense | https://github.com/IntelRealSense/librealsense |
| Qwen3.5 | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| SO-ARM101 | https://github.com/TheRobotStudio/SO-ARM100 |

## 项目结构

```
RK3588-EIA/
├── astra/                # 相机模块（USBCamera + D435iCamera）
├── vla/                  # VLA 核心
│   ├── vlm/              # VLM 引擎（Qwen3.5）
│   ├── control/          # IK + Feetech 串口
│   └── pipe/             # 流水线
├── voice_assistant/      # 语音助手（sherpa-onnx）
├── motion_library/       # 示教轨迹库
├── scripts/              # 测试/工具脚本
├── models/
│   └── Qwen3.5-0.8B/    # Qwen3.5 模型 (rknn + rkllm)
├── config/settings.py    # 统一配置
└── va.py                 # 语音助手入口
```

## 快速开始

```bash
conda activate rkvla

# 语音助手
python va.py once

# 动作演示
python scripts/gestures.py

# 示教录制 → 平滑 → 入库
python scripts/develop_motion.py greeting --record_seconds 15

# 语音触发动作回放
python scripts/voice_motion.py

# VLA 全链路测试（D435i + Qwen3.5）
python scripts/test_vla_voice.py
```

## 示教轨迹回放

| 命令 | 动作 |
|------|------|
| "你好" | 打招呼 |
| "抓" | 抓取 |
| "挥手" | 挥手 |
| "点头" | 点头 |
| "画面中有什么" | VLM 看图回答 |

## 测试完成状态

| 模块 | 状态 |
|------|------|
| 机械臂6关节控制 | ✅ |
| D435i 深度相机 | ✅ |
| Qwen3.5-0.8B VLM | ✅ |
| 语音识别/合成 | ✅ |
| 示教录制→回放 | ✅ |
| 双摄像头 | ✅ |
| IPM 俯视图 | ✅ |
