# RK3588-EIA 技术手册

> 端侧具身智能执行器 — 环境搭建 · 命令参考 · 开发指南

---

## 资料链接

| 内容 | 地址 |
|------|------|
| Orbbec SDK v1.10.27 arm64 | https://github.com/orbbec/OrbbecSDK/releases/tag/v1.10.27 |
| LeRobot v0.4.4 | https://github.com/huggingface/lerobot/tree/v0.4.4 |
| MobileNet-SSD | https://github.com/chuanqi305/MobileNet-SSD |
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| RKLLM 模型仓库 | https://console.box.lenovo.com/l/l0tXb8 (提取码: rkllm) |
| SO-ARM101 | https://github.com/TheRobotStudio/SO-ARM100 |
| 嵌赛官网 | https://www.socchina.net/ |
| Python 3.10 | https://www.python.org/downloads/ |
| Miniconda | https://docs.anaconda.com/miniconda/ |

---

## 一、硬件清单

| 组件 | 型号 | 数量 | 备注 |
|------|------|------|------|
| 边缘计算板 | ELF RK3588 | 1 | 8GB/16GB，运行 VLA 系统 |
| 深度相机 | 奥比中光 Astra Pro | 1 | RGB 640×480, Depth 160×120 |
| 机械臂从臂 | SO-ARM101 Follower | 1 | 6-DOF，Feetech STS3215 |
| 机械臂主臂 | SO-ARM101 Leader | 1 | 6-DOF，遥操作示教用 |
| USB 线 | USB-A to USB-C | 2 | 连接机械臂到 RK3588 |
| USB 线 | USB-A to Micro-USB | 1 | 连接 Astra Pro |

---

## 二、开发环境搭建

### 2.1 RK3588 端

**系统要求**：Ubuntu 22.04 / ARM64, NPU 驱动已安装, Python 3.10+

```bash
# ── 系统依赖 ──
sudo apt update
sudo apt install -y python3-pip python3-opencv python3-serial \
    cmake build-essential wget unzip

# ── Orbbec SDK 安装 ──
# Depth 数据依赖 Orbbec SDK arm64
wget https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C%2B%2B_v1.10.27_20250925_0549823_linux_arm64_release.zip
unzip OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_arm64_release.zip
cd OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_arm64_release
sudo cp -r lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/
sudo ldconfig
cd Script
sudo chmod +x install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo udevadm control --reload && sudo udevadm trigger
cd ../..

# ── 安装 LeRobot ──
cd lerobot && pip install -e .[feetech] && cd ..

# ── 安装项目依赖 ──
pip install -r requirements.txt

# ── 下载 MobileNet SSD ──
python scripts/download_mobilenet.py

# ── 编译 astra_capture ──
cd astra
g++ -std=c++17 capture.cpp -I/usr/local/include -L/usr/local/lib -lOrbbecSDK -lpthread -o build/astra_capture
cd ..
```

### 2.2 VLM 模型部署

Qwen3-VL-2B（当前使用的模型）需要两个文件 + RKLLM Runtime：

```bash
# 1. 从 model zoo 下载预转换模型
#    https://console.box.lenovo.com/l/l0tXb8 (提取码: rkllm)
#    放到:
mkdir -p models/Qwen3-VL-2B
#    需要的文件:
#      qwen3-vl-2b_vision_rk3588.rknn         ← 视觉编码器
#      qwen3-vl-2b-instruct_w8a8_rk3588.rkllm ← 语言模型

# 2. 安装 RKLLM Runtime（多模态 demo 可执行文件）
sudo cp scripts/demo/demo /usr/bin/
sudo cp scripts/demo/lib/librkllmrt.so /usr/local/lib/
sudo ldconfig

# 3. 验证推理
/usr/bin/demo scripts/demo/demo.jpg \
    models/Qwen3-VL-2B/qwen3-vl-2b_vision_rk3588.rknn \
    models/Qwen3-VL-2B/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm \
    1024 2048 3 \
    "<|vision_start|>" "<|vision_end|>" "<|image_pad|>" \
    "描述图像"
```

其他 VLM 模型（InternVL2-1B / Qwen2.5-VL-3B）通过 `rkllm_demo` 部署：

```bash
# 下载或转换 model.rkllm
mkdir -p models/InternVL2-1B-rkllm
# 放入 model.rkllm
rkllm_demo --model models/InternVL2-1B-rkllm/model.rkllm --prompt "描述" --image test.jpg
```

### 2.3 MobileNet SSD 部署

```bash
python scripts/download_mobilenet.py
ls -la models/MobileNetSSD/
# 应看到: prototxt + caffemodel (约 23 MB)
```

---

## 三、标定流程

### 3.1 相机内参标定

```bash
python scripts/calibrate_camera.py
# 将输出的 K 矩阵填入 config/settings.py
```

### 3.2 手眼标定

```bash
python scripts/calibrate_handeye.py
```

### 3.3 机械臂零点校准

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyUSB0
```

---

## 四、命令参考

### 遥操作

```bash
bash scripts/teleop.sh
```

### VLA 自主抓取

```bash
bash scripts/run_vla.sh
```

### 舵机工具

```bash
lerobot-setup-motors --port /dev/ttyUSB0
lerobot-find-port
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyUSB0
lerobot-info
```

### 资源监控

```bash
python scripts/monitor.py
```

---

## 五、系统架构

### 5.1 VLM 框架 (`vla/vlm/`)

工厂模式，支持多模型一键切换：

```python
from vla.vlm import create_vlm

vlm = create_vlm("internvl2-1b")        # InternVL2-1B
# vlm = create_vlm("qwen2.5-vl-3b")     # Qwen2.5-VL-3B

vlm.load("./models/InternVL2-1B-rkllm/model.rkllm")
result = vlm.infer("/tmp/frame.jpg")
print(result.color, result.object)      # "红色", "方块"
```

添加新模型：继承 `VLMBase`，在 `factory.py` 注册。

### 5.2 视觉定位 (`vla/vision/`)

双引擎检测流程：

```
VLM 输出 {object:"bottle", color:"red"}
         ↓
MobileNet SSD (CPU 15-20%) ──► COCO 20 类检测
         ↓
         在检测框内取 HSV 平均值验证颜色
         ↓
         匹配 → (cx, cy) → Depth 查值
         ↓  不匹配
ColorLocator (CPU < 5%) ──► 颜色分割 → 轮廓 → 质心 → Depth
```

COCO 20 类：aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

### 5.3 机械臂控制 (`vla/control/`)

```
3D 坐标 → 串联几何 IK → 6 个关节角 → Feetech STS3215 串口协议 → 舵机
```

舵机 ID：1(肩部旋转) ~ 6(夹爪)，波特率 1000000。

### 5.4 相机 (`astra/`)

| 数据 | 方式 | 分辨率 | 帧率 |
|------|------|--------|------|
| RGB | OpenCV VideoCapture (/dev/video21) | 640×480 | 30fps |
| Depth | C++ Orbbec SDK (astra_capture) | 160×120 | ~30fps |

### 5.5 流水线 (`vla/pipe/`)

```
IDLE → VLM_INFER → LOCATE → GRASP → PLACE → DONE
```

| 状态 | 动作 |
|------|------|
| VLM_INFER | VLM 推理 → 输出目标颜色/物体 |
| LOCATE | SSD 检测 + HSV 验证 (降级 ColorLocator) → 3D 坐标 |
| GRASP | IK 解算 → 移动到目标 → 闭合夹爪 |
| PLACE | 移动到放置点 → 打开夹爪 |

---

## 六、配置说明

`config/settings.py`：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `CAMERA_MATRIX` | 相机内参矩阵（需标定） | 占位值 |
| `SERIAL_PORT` | 机械臂串口路径 | `/dev/ttyUSB0` |
| `SERIAL_BAUD` | 串口波特率 | 1000000 |
| `VLM_MODEL_NAME` | VLM 模型选择 | `internvl2-1b` |
| `RKLLM_BIN` | RKLLM 推理可执行文件 | `/usr/bin/rkllm_demo` |
| `VLM_MODEL_PATH` | RKLLM 模型文件路径 | `./models/InternVL2-1B-rkllm/model.rkllm` |
| `SSD_PROTOTXT` | SSD 模型结构文件 | `./models/MobileNetSSD_deploy.prototxt` |
| `SSD_CAFFEMODEL` | SSD 模型权重 | `./models/MobileNetSSD_deploy.caffemodel` |
| `SSD_CONFIDENCE` | SSD 检测置信度阈值 | 0.5 |

---

## 七、性能指标

| 指标 | 预期值 | 说明 |
|------|--------|------|
| VLM 推理时间 | 2-4s | InternVL2-1B INT4 量化 |
| SSD 目标检测 | 30-50ms | OpenCV DNN CPU |
| 颜色定位时间 | < 10ms | ColorLocator CPU |
| Depth 分辨率 | 160×120 | Astra Pro 硬件限制 |
| 单次抓取总时间 | 4-7s | VLM + 定位 + IK + 舵机 |
| 静态抓取成功率 | 70-85% | 颜色鲜明、背景可控 |
| 系统空闲内存 | ~3GB (8GB) | 含 VLM 模型约 1GB |
| CPU 占用 | ~20% | 含系统 + SSD + IK |
| NPU 占用 | 独占 | VLM 推理时满载 |

---

## 八、调试

```bash
# 串口通信
ls /dev/ttyUSB* /dev/ttyACM*
sudo chmod 666 /dev/ttyACM*

# 相机 RGB
python -c "import cv2; cap=cv2.VideoCapture(21); ret,f=cap.read(); print(f.shape if ret else '失败')"

# 相机 Depth
cd astra && ./build/astra_capture && python -c "
import numpy as np; d=np.fromfile('_depth.f32',dtype=np.float32).reshape(120,160)
print(f'Depth: {d.min():.2f}-{d.max():.2f}m, 非零: {(d>0.01).sum()}/{d.size}')"

# VLM 推理
rkllm_demo --model models/InternVL2-1B-rkllm/model.rkllm --prompt "描述图像" --image test.jpg

# NPU 负载
cat /sys/kernel/debug/rknpu/load

# 模块导入
python -c "
from vla.vlm import create_vlm; print('VLM OK')
from vla.vision import MobileNetSSD; print('SSD OK')
from vla.control import ArmController; print('Arm OK')
from vla.pipe import VLApipeline; print('Pipe OK')
from astra import AstraProCamera; print('Astra OK')
"
```

---

## 九、常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError` | 缺少依赖 | `pip install -r requirements.txt` |
| `rkllm_demo: not found` | RKLLM 未装 | 检查 NPU 驱动 |
| `astra_capture: No such file` | 未编译 | cd astra && g++ ... |
| `cannot reshape depth` | Depth 尺寸不符 | 检查 _depth.info 中的宽高 |
| `SerialException` | 串口未连接 | 检查 USB + 权限 |
| `Camera index out of range` | 相机未接 | 重新插拔 USB |
| `NPU load: N/A` | NPU 驱动问题 | `sudo dmesg \| grep rknpu` |
| `libobsensor.so: not found` | SDK 未安装 | `sudo ldconfig` |
