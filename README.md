# RK3588-EIA

**Embedded Intelligent Actuator** — 端侧具身智能教育平台

RK3588 NPU 端侧部署 VLA（Vision-Language-Action），支持语音交互 + VLM 理解 + 机械臂抓取，全链路在端侧运行，无需联网。

---

## 资源索引

| 资源 | 链接 |
|------|------|
| LeRobot v0.4.4 | https://github.com/huggingface/lerobot/tree/v0.4.4 |
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx |
| SO-ARM100 机械臂 | https://github.com/TheRobotStudio/SO-ARM100 |
| Qwen3-VL | https://github.com/QwenLM/Qwen3-VL |

### VLM 模型资源

| 模型 | 说明 | 链接 |
|------|------|------|
| **Qwen3-VL-2B-Instruct** ⭐ | **主力模型，支持 2D grounding 坐标输出** | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct |
| SmolVLM-256M-Instruct | 轻量 VLM | https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct |

---

## 架构

```
               🎤 语音输入 (sherpa-onnx ASR + KWS)
                      ↓
   USB 相机 ──► Qwen3-VL (RKLLM NPU) ──► 文字描述 + 2D grounding 坐标
                         (支持 center_point / bbox_2d 输出)
                      ↓
                3D 反投影 (x, y, z)
                      ↓
                IK → 串口 → SO-ARM101 抓取
                      ↓
               🔊 语音播报 (sherpa-onnx TTS)
```

---

## 项目结构

```
RK3588-EIA/
├── lerobot/              # LeRobot v0.4.4（遥操作模块）
├── astra/                # USB 相机封装
├── vla/                  # VLA 核心系统
│   ├── vlm/              # VLM 引擎（工厂模式）
│   ├── vision/           # ColorLocator 颜色定位
│   ├── control/          # IK + Feetech 串口
│   └── pipe/             # 流水线 FSM
├── voice_assistant/      # 语音助手模块（sherpa-onnx）
│   ├── voice_assistant/  # 语音包（ASR / TTS / KWS / Qwen 调度）
│   ├── config/           # 语音配置
│   ├── scripts/          # 拍照脚本
│   └── models/           # 语音模型（KWS + ASR + TTS）
├── config/settings.py    # 统一配置
├── models/
│   ├── Qwen3-VL-2B/      # Qwen3-VL-2B 模型 (rknn + rkllm)
│   ├── SmolVLM-256M/     # SmolVLM-256M 模型
│   └── so101_urdf/       # SO-ARM101 URDF
├── scripts/
│   ├── demo/             # RKLLM 推理二进制
│   ├── camera_viewer.py  # 实时取景+VLM 推理
│   └── test_*.py         # 各模块测试脚本
├── va.py                 # 语音助手入口
└── main.py               # VLA 主入口
```

---

## 快速开始

```bash
# 安装 LeRobot
cd lerobot && pip install -e .[feetech]

# 安装 RKLLM Runtime
sudo cp scripts/demo/demo /usr/bin/
sudo cp scripts/demo/lib/*.so /usr/lib/
sudo ldconfig

# 启动相机实时取景（按 Enter 触发 VLM）
python scripts/camera_viewer.py

# 语音助手
cd voice_assistant && python voice_assistant.py once

# 启动 VLA 自主抓取（接好机械臂）
python main.py
```

---

## VLM 模型对比

| 模型 | 参数量 | 推理速度 | 内存 | 输出方式 | 状态 |
|------|--------|---------|------|---------|------|
| **Qwen3-VL-2B** ⭐ | 2B | ~5s | 2.5GB | 文字描述 + **2D grounding 坐标** | ✅ 已跑通 |
| SmolVLM-256M | 256M | ~1.3s | <1GB | 文字描述 | ✅ 已跑通 |

---

## 机械臂校准值

6 个关节经实测标定，限位已写入 `controller.py`：

| 关节 | 零位(mid) | 最小 | 最大 | 物理范围 |
|------|----------|------|------|---------|
| 1 shoulder_pan | 2117 | 946 | 3287 | -103° ~ +103° |
| 2 shoulder_lift | 2014 | 821 | 3206 | -105° ~ +102° |
| 3 elbow_flex | 1997 | 888 | 3105 | -97° ~ +97° |
| 4 wrist_flex | 2022 | 851 | 3192 | -103° ~ +103° |
| 5 wrist_roll | 2058 | 130 | 3985 | -169° ~ +170° |
| 6 gripper | 2178 | 1495 | 2860 | 夹紧~张开 |

角度→脉冲公式（官方 v0.4.4）：
```python
mid = (range_min + range_max) / 2
pulse = deg × 4095 / 360 + mid
```

---

## 标定

```bash
# 机械臂标定
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_awesome_leader_arm
```

## LeRobot 命令

`lerobot-teleoperate` `lerobot-calibrate` `lerobot-find-port`
`lerobot-find-cameras` `lerobot-setup-motors` `lerobot-info`
