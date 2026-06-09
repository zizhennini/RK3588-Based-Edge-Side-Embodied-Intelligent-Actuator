# Astra Pro RK3588 部署指南

## 资料链接

| 内容 | 地址 |
|------|------|
| Orbbec SDK GitHub | https://github.com/orbbec/OrbbecSDK |
| v1.10.27 arm64 zip | https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C%2B%2B_v1.10.27_20250925_0549823_linux_arm64_release.zip |
| 奥比中光官网 | https://www.orbbec.com/ |
| Astra Pro 产品页 | https://shop.orbbec.com/products/astra-pro |

---

## 安装 Orbbec SDK

Astra Pro 使用 OpenNI 协议，通过 Orbbec SDK v1 读取 Depth。官方提供 arm64 预编译包。

```bash
# 下载（约 18MB）
wget https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.27/OrbbecSDK_C_C%2B%2B_v1.10.27_20250925_0549823_linux_arm64_release.zip

unzip OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_arm64_release.zip
cd OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_arm64_release

# 安装 C 库
sudo cp -r lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/
sudo ldconfig

# 安装 udev 规则
cd Script
sudo chmod +x install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo udevadm control --reload && sudo udevadm trigger
```

## 相机数据

| 数据 | 接口 | 分辨率 | 帧率 |
|------|------|--------|------|
| RGB | OpenCV VideoCapture (/dev/video21) | 640×480 | 30fps |
| Depth | Orbbec SDK (C++ helper) | 160×120 | ~30fps |

## 验证

```bash
# 确认设备被识别
v4l2-ctl --list-devices

# RGB 测试
python -c "import cv2; cap=cv2.VideoCapture(21); ret,f=cap.read(); print(f'RGB: {f.shape}')"

# Depth 测试
cd astra
g++ -std=c++17 capture.cpp -I/usr/local/include -L/usr/local/lib -lOrbbecSDK -lpthread -o build/astra_capture
./build/astra_capture
python -c "
import numpy as np
d = np.fromfile('_depth.f32', dtype=np.float32).reshape(120,160)
print(f'Depth: {d.min():.2f}-{d.max():.2f}m')
rm -f _depth.f32 _depth.info
"
```

## 常见问题

| 问题 | 解决 |
|------|------|
| `libobsensor.so: cannot open` | `sudo ldconfig` |
| `USB camera not found` | 重新插拔 USB 线 |
| `Permission denied` | 运行 udev 规则脚本 |
| `Match openni video mode failed` | RGB 用 OpenCV，SDK 只用于 Depth |
