# RK3588 端侧部署完整流程

## 资源索引

| 资源 | 链接 |
|------|------|
| Orbbec SDK v1.10.27 arm64 zip | https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C%2B%2B_v1.10.27_20250925_0549823_linux_arm64_release.zip |
| OrbbecSDK GitHub | https://github.com/orbbec/OrbbecSDK |
| LeRobot v0.4.4 | https://github.com/huggingface/lerobot/tree/v0.4.4 |
| MobileNet-SSD Caffe 模型 | https://github.com/chuanqi305/MobileNet-SSD |
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| SO-ARM100/101 | https://github.com/TheRobotStudio/SO-ARM100 |
| 奥比中光官网 | https://www.orbbec.com/ |

---

## 1. 系统准备

```bash
# Ubuntu 22.04 / Arm64
# 确认 NPU 驱动
ls /dev/dri/
# 应显示: card0 card1 renderD128 ...

cat /sys/kernel/debug/rknpu/load

# 安装系统依赖
sudo apt update
sudo apt install -y python3-pip python3-opencv python3-serial \
    cmake build-essential wget unzip
```

## 2. Orbbec SDK 安装

Astra Pro 使用 OpenNI 协议，需通过 Orbbec SDK v1 读取 Depth。官方提供 arm64 预编译包。

```bash
# 下载 arm64 SDK（约 18MB）
wget https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C%2B%2B_v1.10.27_20250925_0549823_linux_arm64_release.zip

# 解压
unzip OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_arm64_release.zip
cd OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_arm64_release

# 安装 C 库
sudo cp -r lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/
sudo ldconfig

# 安装 udev 规则（相机权限）
cd Script
sudo chmod +x install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo udevadm control --reload && sudo udevadm trigger
cd ../..
```

## 3. 项目安装

```bash
# LeRobot
cd lerobot && pip install -e .[feetech] && cd ..

# 项目依赖
pip install -r requirements.txt

# MobileNet SSD 模型（约 23MB）
python scripts/download_mobilenet.py

# 编译 Depth 采集程序
cd astra
g++ -std=c++17 capture.cpp -I/usr/local/include -L/usr/local/lib -lOrbbecSDK -lpthread -o build/astra_capture
cd ..

# 验证相机
python -c "
from astra import AstraProCamera
cam = AstraProCamera(rgb_index=21)
cam.connect()
rgb, depth = cam.read()
print(f'RGB: {rgb.shape}, Depth: {depth.shape}')
cam.disconnect()
"
```

## 4. VLM 模型部署

```bash
# 将转换好的 RKLLM 模型放入
mkdir -p models/InternVL2-1B-rkllm
# model.rkllm 放进去

# 验证
rkllm_demo \
    --model models/InternVL2-1B-rkllm/model.rkllm \
    --prompt "描述图像" \
    --image test.jpg
```

**模型获取方式：**
1. 瑞芯微官方下载预转换模型
2. 自行在 x86 PC 上使用 RKLLM-Toolkit 转换：`pip install rkllm-toolkit`

## 5. 标定

```bash
# 相机内参标定
python scripts/calibrate_camera.py

# 手眼标定
python scripts/calibrate_handeye.py

# 机械臂零点校准
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0
```

## 6. 运行

```bash
# 遥操作示教
bash scripts/teleop.sh

# VLA 自主抓取
bash scripts/run_vla.sh

# 资源监控
python scripts/monitor.py
```

## 系统结构图

```
┌────────────────────────────────────────────────────────────┐
│                       RK3588 (ELF)                        │
│                                                            │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Astra Pro │  │ VLM (RKLLM)  │  │ SO-ARM101        │   │
│  │ RGB OpenCV│  │ NPU 独占     │  │ Feetech 串口     │   │
│  │ Depth SDK │  │ 2-4s/次      │  │ /dev/ttyUSB0     │   │
│  └─────┬─────┘  └──────┬───────┘  └──────────────────┘   │
│        │               │                                  │
│        ▼               ▼                                  │
│  ┌──────────────┐ ┌──────────┐                            │
│  │ MobileNet    │◀│ Prompt   │                            │
│  │ SSD (CPU)    │ │ 解析器   │                            │
│  │ 15-20%       │ └──────────┘                            │
│  └──────┬───────┘                                         │
│         │ 不匹配降级                                       │
│         ▼                                                 │
│  ┌──────────────┐                                         │
│  │ ColorLocator │                                         │
│  │ CPU < 5%     │                                         │
│  └──────┬───────┘                                         │
│         │ Depth 反投影 + IK                                │
│         ▼                                                 │
│  3D 坐标 → 关节角 → 串口 → 舵机                              │
└────────────────────────────────────────────────────────────┘
```
