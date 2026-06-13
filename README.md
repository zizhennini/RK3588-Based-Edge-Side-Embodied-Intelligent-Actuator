# RK3588-EIA

**Embedded Intelligent Actuator** — 端侧具身智能执行器

RK3588 NPU 端侧部署 VLA（Vision-Language-Action），使用 LeRobot + VLM + Astra Pro + SO-ARM101 实现自主抓取搬运。

---

## 资源索引

| 资源 | 链接 |
|------|------|
| LeRobot v0.4.4 | https://github.com/huggingface/lerobot/tree/v0.4.4 |
| Orbbec SDK v1 (arm64) | https://github.com/orbbec/OrbbecSDK/releases/tag/v1.10.27 |
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| RKLLM 模型仓库 | https://console.box.lenovo.com/l/l0tXb8 (提取码: rkllm) |
| 嵌赛官网 | https://www.socchina.net/ |
| 瑞芯微赛题解读 | https://mp.weixin.qq.com/s/W_yiAElw6HTVnroYEp3pmg |
| SO-ARM100 机械臂 | https://github.com/TheRobotStudio/SO-ARM100 |
| whisper.cpp (端侧语音) | 🎤 CPU 语音识别，50K ⭐ | https://github.com/ggml-org/whisper.cpp |

### VLM 模型资源

| 模型 | 说明 | 链接 |
|------|------|------|
| **SmolVLM2-500M-vqa-position** ⭐ | **机器人抓取微调版，输出坐标+颜色** | https://huggingface.co/robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged |
| SmolVLM2-500M-color-aware | 机器人物体识别（颜色感知） | https://huggingface.co/robot-learning-group47/smolvlm2-500m-color-aware |
| SmolVLM-256M-Instruct | 通用 VLM（RKLLM 已验证） | https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct |
| Qwen3-VL-2B-Instruct | 通用 VLM（已在 RK3588 跑通） | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct |
| InternVL2-1B | 轻量 VLM | https://huggingface.co/OpenGVLab/InternVL2-1B |

---

## 架构

```
Astra Pro ─┬── RGB ──► VLM (RKLLM NPU) ──► "红色 杯子"
           │              ~1-5s/次
           │                   ▲
           │                   │ 确认
           │              ┌────┴────┐
           │         🎤 语音 → whisper.cpp ──► "抓红色杯子"
           │              (CPU, ~200ms)
           │                   │
           │                   ▼
           │              ColorLocator (CPU) ──► (u,v)
           │                                     ↓
           └── Depth ───────────────────────► z 值查表
                                                    ↓
                                               3D 坐标 (x,y,z)
                                                    ↓
                                              IK → SO-ARM101 执行
```

**定位方案**（二选一）：
- **ColorLocator**：颜色分割定位（CPU < 5%，无需额外模型）
- **VLM 直出坐标**：VLM 输出 cx,cy（需微调版 VLM）

---

## 项目结构

```
RK3588-EIA/
├── lerobot/              # LeRobot v0.4.4（遥操作模块）
├── astra/                # Astra Pro 相机封装
├── vla/                  # VLA 核心系统
│   ├── vlm/              # 通用 VLM 框架（5个引擎）
│   ├── vision/           # ColorLocator 颜色定位
│   ├── control/          # IK + Feetech 串口
│   └── pipe/             # 流水线 FSM
├── config/settings.py    # 统一配置
├── models/
│   ├── Qwen3-VL-2B/      # Qwen3-VL-2B 模型
│   ├── SmolVLM-256M/     # SmolVLM-256M 模型
│   └── so101_urdf/       # SO-ARM101 URDF（placo IK）
├── scripts/
│   ├── demo/             # RKLLM 推理二进制
│   ├── camera_viewer.py  # 实时取景+VLM 推理
│   └── test_pipeline.py  # VLM+相机 联调
├── main.py               # 主入口
└── docs/
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

# 启动相机实时取景（按 Enter 触发 VLM）
python scripts/camera_viewer.py

# 启动 VLA 自主抓取（接好机械臂）
python main.py
```

---

## VLM 模型对比

| 模型 | 参数量 | 推理速度 | 内存 | 定位方式 | 状态 |
|------|--------|---------|------|---------|------|
| **SmolVLM2-500M-vqa-position** ⭐ | 500M | ~2s | <1GB | **直接输出坐标** | 🔍 待测试 |
| Qwen3-VL-2B | 2B | ~5s | 2.5GB | 文本描述 | ✅ 已跑通 |
| SmolVLM-256M | 256M | ~1.3s | <1GB | 文本描述 | ✅ 已跑通 |
| InternVL2-1B | 1B | ~2s | 1GB | 文本描述 | ⚪ 待测试 |
| SmolVLM2-500M-color | 500M | ~2s | <1GB | 颜色识别 | 🔍 待测试 |

---

## 标定

```bash
python scripts/calibrate_camera.py
python scripts/calibrate_handeye.py
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyUSB0
```

## LeRobot 命令

`lerobot-teleoperate` `lerobot-calibrate` `lerobot-find-port`
`lerobot-find-cameras` `lerobot-setup-motors` `lerobot-info`
