# RK3588-EIA

**Embedded Intelligent Actuator** — 端侧具身智能执行器

---

## 项目资源索引

| 资源 | 链接 |
|------|------|
| LeRobot v0.4.4 | https://github.com/huggingface/lerobot/tree/v0.4.4 |
| Orbbec SDK v1 (arm64) | https://github.com/orbbec/OrbbecSDK/releases/tag/v1.10.27 |
| MobileNet-SSD (Caffe) | https://github.com/chuanqi305/MobileNet-SSD |
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| RKLLM 模型仓库 | https://console.box.lenovo.com/l/l0tXb8 (提取码: rkllm) |
| RK3588 NPU 文档 | https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1676.html |
| SO-ARM101 机械臂 | https://github.com/TheRobotStudio/SO-ARM100 |
| 奥比中光 Astra Pro | https://shop.orbbec.com/products/astra-pro |
| 嵌赛官网 | https://www.socchina.net/ |

---

## 架构

```
Astra Pro ─┬── RGB (OpenCV) ──► VLM (RKLLM NPU) ──► "拿红色杯子"
           │                    Qwen3-VL-2B / InternVL2-1B
           │                               ↓
           │                        Prompt 解析器
           │                        {object:"bottle", color:"red"}
           │                               ↓
           └── RGB (OpenCV) ──► MobileNet SSD (CPU) ──► 检测 "bottle"
                                              │
                                              ▼
                                         找对应检测框 → HSV 验证颜色
                                              │
                                         ┌────┴────┐
                                         ▼         ▼
                                      匹配     不匹配 → ColorLocator 降级
                                         │
                                         ▼
Astra Pro ── Depth (C++ SDK) ──► (cx, cy) + Depth z
                                         │
                                         ▼
                                    IK 解算 → SO-ARM101 执行
```

## 项目结构

```
RK3588-EIA/
├── lerobot/              # LeRobot v0.4.4（遥操作必需模块）
├── astra/                # Astra Pro 相机封装
├── vla/                  # VLA 核心系统
│   ├── vlm/              # 通用 VLM 框架（3个引擎：Qwen3/InternVL2/Qwen2.5）
│   ├── vision/           # MobileNet SSD + ColorLocator
│   ├── control/          # IK + Feetech 串口
│   └── pipe/             # 流水线 FSM
├── config/settings.py    # 统一配置
├── models/
│   ├── Qwen3-VL-2B/      # VLM 模型（rknn + rkllm）
│   └── MobileNetSSD/     # SSD 模型（prototxt + caffemodel）
├── scripts/demo/         # RKLLM 多模态推理可执行文件
├── main.py
└── docs/
```

---

## RK3588 完整安装

```bash
# ── 1. 系统依赖 ──
sudo apt install -y python3-pip python3-opencv python3-serial cmake build-essential wget unzip

# ── 2. Orbbec SDK（Depth 采集） ──
wget https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C%2B%2B_v1.10.27_20250925_0549823_linux_arm64_release.zip
unzip OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_arm64_release.zip
cd OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_arm64_release
sudo cp -r lib/* /usr/local/lib/ && sudo cp -r include/* /usr/local/include/ && sudo ldconfig
cd Script && sudo ./install_udev_rules.sh && sudo udevadm control --reload && sudo udevadm trigger
cd ../..

# ── 3. LeRobot ──
cd lerobot && pip install -e .[feetech] && cd ..

# ── 4. 项目依赖 ──
pip install -r requirements.txt

# ── 5. MobileNet SSD ──
python scripts/download_mobilenet.py

# ── 6. 编译 astra_capture ──
cd astra && g++ -std=c++17 capture.cpp -I/usr/local/include -L/usr/local/lib -lOrbbecSDK -lpthread -o build/astra_capture && cd ..

# ── 7. RKLLM Runtime ──
sudo cp scripts/demo/demo /usr/bin/
sudo cp scripts/demo/lib/librkllmrt.so /usr/local/lib/
sudo ldconfig

# ── 8. 验证 ──
python -c "from astra import AstraProCamera; c=AstraProCamera(21); c.connect(); r,d=c.read(); print(f'RGB:{r.shape} Depth:{d.shape}'); c.disconnect()"
```

---

## VLM 模型

| 模型 | 配置名 | 内存 | 推理速度 | 硬件 |
|------|--------|------|---------|------|
| **Qwen3-VL-2B** | `qwen3-vl-2b` | ~2.5GB | 5-8s/次 | 16GB / 8GB |
| InternVL2-1B | `internvl2-1b` | ~1GB | 2-3s/次 | 8GB/16GB |
| Qwen2.5-VL-3B | `qwen2.5-vl-3b` | ~3GB | 6-10s/次 | 16GB |
| SmolVLM-256M |  | 256M | 1s/每次 | 8G |

切换：改 `config/settings.py` 中 `VLM_MODEL_NAME`。

### 模型部署

Qwen3-VL-2B 需要两个文件（从 model zoo 下载）：
```
models/Qwen3-VL-2B/
├── qwen3-vl-2b_vision_rk3588.rknn         # 视觉编码器
└── qwen3-vl-2b-instruct_w8a8_rk3588.rkllm # 语言模型
```

以及 RKLLM Runtime（从 SDK 下载）：
```bash
sudo cp scripts/demo/demo /usr/bin/
sudo cp scripts/demo/lib/librkllmrt.so /usr/local/lib/
sudo ldconfig
```

## 标定

```bash
python scripts/calibrate_camera.py
python scripts/calibrate_handeye.py
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyUSB0
```

## 快速开始

```bash
bash scripts/teleop.sh          # 遥操作
bash scripts/run_vla.sh         # VLA 抓取
python scripts/monitor.py       # 资源监控
```

## LeRobot

可用命令：`lerobot-teleoperate` `lerobot-calibrate` `lerobot-find-port`
`lerobot-find-cameras` `lerobot-setup-motors` `lerobot-info`

