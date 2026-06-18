# RK3588-EIA

**Embedded Intelligent Actuator** — 端侧具身智能教育平台

RK3588 NPU 端侧部署 VLA（Vision-Language-Action），支持语音交互 + VLM 理解 + 机械臂抓取，全链路在端侧运行，无需联网。

---

## 资源索引

| 资源 | 链接 |
|------|------|
| LeRobot v0.4.4 | https://github.com/huggingface/lerobot/tree/v0.4.4 |
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| whisper.cpp | https://github.com/ggml-org/whisper.cpp |
| SO-ARM100 机械臂 | https://github.com/TheRobotStudio/SO-ARM100 |
| 嵌赛官网 | https://www.socchina.net/ |

### VLM 模型资源

| 模型 | 说明 | 链接 |
|------|------|------|
| **SmolVLM2-500M-vqa-position** ⭐ | **机器人抓取微调版，直出 JSON 坐标** | https://huggingface.co/robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged |
| Qwen3-VL-2B-Instruct | 通用 VLM | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct |
| SmolVLM-256M-Instruct | 通用 VLM | https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct |
| InternVL2-1B | 轻量 VLM | https://huggingface.co/OpenGVLab/InternVL2-1B |

---

## 架构

```
               🎤 语音输入 (whisper.cpp)
                      ↓
   USB 相机 ──► VLM (RKLLM NPU) ──► JSON: color, object, cx, cy
                      ↓
               3D 反投影 (x, y, z)
                      ↓
               IK → 串口 → SO-ARM101 抓取
                      ↓
               🔊 语音播报 (espeak-ng)
```

**VLM 定位方案**（二选一）：
- **SmolVLM2-500M 微调版**：直出 JSON 坐标 `{color, objectName, centerPosition}`
- **ColorLocator**：HSV 颜色分割定位（降级方案）

---

## 项目结构

```
RK3588-EIA/
├── lerobot/              # LeRobot v0.4.4（遥操作模块，不动）
├── astra/                # USB 相机封装（后台线程无卡顿）
├── vla/                  # VLA 核心系统
│   ├── vlm/              # VLM 引擎（5个引擎+工厂模式）
│   ├── vision/           # ColorLocator 颜色定位
│   ├── control/          # IK + Feetech 串口
│   ├── voice/            # 语音识别 (whisper) + 播报 (espeak)
│   └── pipe/             # 流水线 FSM
├── config/settings.py    # 统一配置
├── models/
│   ├── Qwen3-VL-2B/      # Qwen3-VL-2B 模型
│   ├── SmolVLM-256M/     # SmolVLM-256M 模型
│   ├── SmolVLM-500M/     # SmolVLM-500M 模型（含微调版）
│   ├── whisper/          # whisper 模型
│   └── so101_urdf/       # SO-ARM101 URDF（placo IK）
├── scripts/
│   ├── demo/             # RKLLM 推理二进制
│   ├── camera_viewer.py  # 实时取景+VLM 推理
│   ├── voice_demo.py     # 语音控制演示
│   └── test_*.py         # 各模块测试脚本
├── main.py               # 主入口
└── config/settings.py    # 统一配置
```

---

## 快速开始

```bash
# 安装 LeRobot
cd lerobot && pip install -e .[feetech] && cd ..

# 安装项目依赖
pip install -r requirements.txt

# 安装 RKLLM Runtime
sudo cp scripts/demo/demo /usr/bin/
sudo cp scripts/demo/lib/*.so /usr/lib/
sudo ldconfig

# 安装语音 TTS
sudo apt install espeak-ng

# 启动相机实时取景（按 Enter 触发 VLM）
python scripts/camera_viewer.py

# 语音控制演示
python scripts/voice_demo.py

# 启动 VLA 自主抓取（接好机械臂）
python main.py
```

---

## VLM 模型对比

| 模型 | 参数量 | 推理速度 | 内存 | 输出方式 | 状态 |
|------|--------|---------|------|---------|------|
| **SmolVLM2-500M-robot** ⭐ | 500M | ~2s | <1GB | **JSON 坐标** | ✅ 已转换 |
| Qwen3-VL-2B | 2B | ~5s | 2.5GB | 文本描述 | ✅ 已跑通 |
| SmolVLM-256M | 256M | ~1.3s | <1GB | 文本描述 | ✅ 已跑通 |
| InternVL2-1B | 1B | ~2s | 1GB | 文本描述 | ✅ 已跑通 |
| SmolVLM2-500M-color | 500M | ~2s | <1GB | 颜色识别 | 🔍 待测试 |

---

## 机械臂校准值

6个关节经实测标定，限位已写入 `controller.py`：

| 关节 | 零位 | 最小 | 最大 | 物理范围 |
|------|------|------|------|---------|
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

# 相机标定（如需）
python scripts/calibrate_camera.py
```

## LeRobot 命令

`lerobot-teleoperate` `lerobot-calibrate` `lerobot-find-port`
`lerobot-find-cameras` `lerobot-setup-motors` `lerobot-info`
