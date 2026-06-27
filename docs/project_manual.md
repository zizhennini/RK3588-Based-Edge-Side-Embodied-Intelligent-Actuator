# RK3588-EIA 项目完整说明

> **Embedded Intelligent Actuator** — 端侧具身智能教育平台

## 一、项目架构

```
USB 相机 ──► Qwen3-VL (NPU) ──► 文字描述 + 2D grounding 坐标
                      │
              ┌───────┴───────┐
              ▼               ▼
         语义理解          坐标定位
     "红色方块"          center_point
              │               │
              └───────┬───────┘
                      ▼
                  3D 反投影 + IK → 串口 → SO-ARM101
```

## 二、目录结构

```
RK3588-EIA/
├── main.py                   # VLA 主入口
├── va.py                     # 语音助手入口
├── requirements.txt
├── README.md
├── vla/                      # VLA 核心系统
│   ├── vlm/                  #   VLM 引擎（工厂模式）
│   ├── vision/               #   视觉定位
│   ├── control/controller.py #   IK + Feetech 串口
│   └── pipe/pipeline.py      #   有限状态机
├── voice_assistant/          # 语音助手
│   ├── voice_assistant/      #   语音包（ASR/TTS/KWS）
│   └── models/               #   语音模型
├── astra/                    # USB 相机封装
├── lerobot/                  # LeRobot v0.4.4
├── config/settings.py
└── models/                   # VLM 模型
```

## 三、创新点

1. **纯端侧 VLA** — 全链路在 RK3588 本地，无需云端
2. **Qwen3-VL grounding** — 同时输出语义 + 坐标，一步到位
3. **sherpa-onnx 离线语音** — KWS + ASR + TTS 全离线中文交互
4. **通用 VLM 框架** — 工厂模式，一行配置切换模型
5. **LeRobot 遥操作** — 主从示教 + 数据录制
