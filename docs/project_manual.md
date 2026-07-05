# RK3588-EIA 项目完整说明

> **Embedded Intelligent Actuator** — 端侧具身智能教育平台

## 一、项目架构

```
🎤 语音输入 (sherpa-onnx ASR + KWS)
       ↓
📷 D435i ──► Qwen3.5-0.8B (NPU) ──► 文字描述 + 坐标
       ↓
深度图 ──► 3D 反投影 ──► IK ──► 串口 ──► SO-ARM101
       ↓
🔊 TTS 播报
```

## 二、目录结构

```
RK3588-EIA/
├── main.py                   # VLA 主入口
├── va.py                     # 语音助手入口
├── voice_assistant/          # 语音助手
│   ├── voice_assistant/      #   Python 包（ASR/TTS/KWS）
│   │   └── models/           #   语音模型文件
│   └── config/               #   语音模块配置
├── vla/                      # VLA 核心系统
│   ├── vlm/                  #   VLM 引擎（工厂模式）
│   ├── vision/               #   视觉定位
│   ├── control/controller.py #   IK + Feetech 串口
│   ├── pipe/pipeline.py      #   有限状态机
│   └── command_queue.py      #   指令队列 + 中断
├── astra/                    # 相机封装（D435i）
├── config/                   # 系统配置
│   ├── settings.py           #   统一参数
│   ├── cpu_affinity.py       #   大小核算力隔离
│   └── memory.py             #   分级内存管控
├── scripts/                  # 测试/工具
├── motion_library/           # 示教轨迹库
├── models/Qwen3.5-0.8B/     # VLM 模型
└── docs/                     # 文档
```

## 三、核心特性

### 3.1 大小核算力隔离

| 核心 | 架构 | 用途 |
|------|------|------|
| 0-3 | Cortex-A76（大核） | VLM 推理、语音主线程 |
| 4-7 | Cortex-A55（小核） | 机械臂控制、相机采集 |

文件：`config/cpu_affinity.py`

### 3.2 分级内存管控

- SmartVLM 按需加载，闲置 30s 自动卸载（释放 ~900MB）
- CommandQueue 每类任务 acquire/release 内存预算
- 录音/ASR 用 MemoryLimiter 保护，自动释放冲突组件

文件：`config/memory.py`、`vla/vlm/smart_vlm.py`

### 3.3 指令队列 + 中断

- FIFO 串行执行，6 种任务类型
- 紧急中断：清空队列 + arm.emergency_stop() 物理停机
- 中断感知的轨迹回放（每帧检查标志）

文件：`vla/command_queue.py`

### 3.4 全离线语音

| 模块 | 模型 | 功能 |
|------|------|------|
| KWS | sherpa-onnx-kws-zipformer | 唤醒词"鲁班猫" |
| ASR | sherpa-onnx-conformer | 中文语音识别 |
| TTS | Matcha-TTS + vocos | 中文语音合成 |

### 3.5 VLM 引擎

- 工厂模式：`factory.create_vlm()` 一行切换模型
- 当前支持：Qwen3.5-0.8B（RKNN + RKLLM 格式）
- NPU 推理：视觉 ~0.7s + LLM ~2.1s

## 四、测试完成状态

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
