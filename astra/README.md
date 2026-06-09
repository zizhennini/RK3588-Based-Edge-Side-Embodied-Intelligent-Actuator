# astra — 奥比中光 Astra Pro 相机封装

基于 Orbbec SDK (pyorbbecsdk)，同时获取 RGB + Depth 流。

## 依赖

```bash
# 安装 Orbbec SDK arm64
wget https://github.com/orbbec/OrbbecSDK/releases/download/v1.10.35/OrbbecSDK_v1.10.35_arm64.deb
sudo dpkg -i OrbbecSDK_v1.10.35_arm64.deb
sudo apt install -f
sudo apt install -y python3-pip cmake build-essential

# 安装 Python 绑定
pip install pyorbbecsdk

# 安装 udev 规则
cd /usr/local/lib/orbbec_sdk/misc/scripts
sudo chmod +x install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo udevadm control --reload && sudo udevadm trigger
```

## 文件

| 文件 | 说明 |
|------|------|
| `camera.py` | `AstraProCamera` — connect() / read() / disconnect() |

## 使用

```python
from astra import AstraProCamera

cam = AstraProCamera()
cam.connect()
rgb, depth = cam.read()  # rgb: H×W×3 uint8, depth: H×W float32（米）
cam.disconnect()
```

## 输出格式

| 数据 | 格式 | 说明 |
|------|------|------|
| rgb | (H, W, 3) uint8 | RGB 彩色图像 |
| depth | (H, W) float32 | 深度图，单位：米 |
