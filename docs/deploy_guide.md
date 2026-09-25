# RK3588 端侧部署完整流程

> 与 refactor_plan_v9 对齐: **板端 Python 3.10 仅推理，不安装 LeRobot**；
> LeRobot（数据集处理 / ACT 训练 / ONNX 导出）在 PC 端 Python 3.12 环境。

## 资源索引

| 资源 | 链接 |
|------|------|
| LeRobot v0.6.1（仅 PC 端） | https://github.com/huggingface/lerobot |
| RKLLM 工具链 | https://github.com/airockchip/rknn-llm |
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx |
| Qwen3.5 | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| SO-ARM100/101 | https://github.com/TheRobotStudio/SO-ARM100 |
| RealSense D435i | https://github.com/IntelRealSense/librealsense |
| librga | https://github.com/airockchip/librga |

## 1. 系统准备

```bash
# Ubuntu 22.04 / Arm64
sudo apt update
sudo apt install -y python3-pip python3-opencv cmake build-essential wget git ffmpeg
```

## 2. 项目安装（板端）

```bash
conda create -n rk3588 python=3.10 -y
conda activate rk3588

# 安装板端依赖（推理/相机/串口/语音/编码，不含 LeRobot）
cd /home/elf/work/rk3588-eia
pip install -r requirements.txt

# RKNN Lite（NPU 推理库）
pip install /path/to/rknn_toolkit_lite2-*-cp310-*.whl

# 验证
python3 -c "import cv2, numpy, sherpa_onnx, pyrealsense2; print('OK')"
```

PC 端（数据集处理 / ACT 训练 / 模型导出）另建环境：

```bash
conda create -n rk3588 python=3.12 -y
conda activate rk3588
pip install -r requirements-dev.txt   # torch / onnx / safetensors 等
pip install lerobot==0.6.1            # 仅 PC 端需要（数据集 / ACT 训练）
```

## 2.5 环境与版本矩阵（避免冲突）

### 已验证环境基线（CHANGELOG v0.3.0 实测记录）

| 端 | Python | 关键包实测版本 |
|----|--------|----------------|
| 板端 RK3588 | 3.10（conda `rk3588`） | onnxruntime 1.23.2、rknn-toolkit-lite2 2.3.2（本地 whl cp310）、torch 2.7.0+cpu（系统预装） |
| PC 端 WSL2 | 3.12（conda `rk3588`） | LeRobot v0.6.1、torch 2.11.0+cpu |

### 逐包版本对照

| 包 | 板端 (requirements.txt) | PC 端 (requirements-dev.txt) | 约束理由 / 冲突点 |
|----|------------------------|------------------------------|-------------------|
| Python | 3.10 | 3.12 | **分离根因**: rknn-toolkit-lite2 2.3.2 仅提供 cp310 whl；LeRobot v0.6.1 强制 >=3.12。两端不可混用 |
| numpy | >=1.24,<2.0（实测 1.26.4） | >=1.24（实测 2.2.6） | **红线仅板端**: rknn-toolkit-lite2 不兼容 numpy 2.x；PC 端随 LeRobot/torch 生态放宽（2026-09 核查决策），导出产物为 ONNX 文件、与 numpy 版本无关 |
| torch | 系统预装 2.7.0+cpu（`/usr/local` dist-packages，厂商镜像与 rknn_toolkit_lite2 并存；conda env 内零 torch） | >=2.0（实测 2.11.0+cpu） | **红线**: 板端勿 pip 安装/升级 torch，避免覆盖系统 RK 构建；主链路零 torch 用途 |
| onnxruntime | >=1.16（实测 1.23.2） | >=1.16 | 两端对齐，保证导出 ONNX 的算子支持一致 |
| onnx | —（板端不装） | >=1.14 | ACT 导出需 opset 14+（scaled_dot_product_attention）；GGCNN 用 opset 12；onnx>=1.14 均覆盖 |
| safetensors | — | >=0.4 | `export_act_onnx.py` 加载 LeRobot checkpoint 所需；此前靠 LeRobot 传递安装，已显式化防环境漂移 |
| rknn-toolkit-lite2 | 2.3.2（本地 whl，非 PyPI） | — | NPU 推理仅板端 |
| pyrealsense2 | >=2.54 | — | aarch64/py310 wheel 板端已验证 |
| feetech-servo-sdk | >=1.10 | — | 提供 `scservo_sdk` 模块（hardware/arm.py） |
| sherpa-onnx | >=1.9 | — | 语音 KWS/ASR/TTS |
| pexpect | >=4.8 | — | VLM RKLLM demo 子进程管理 |
| opencv-python | >=4.8,<5.0 | >=4.8,<5.0 | 两端对齐 |
| LeRobot | 不安装（v9 决策）；vendored 子集已于 2026-09 审计删除（src 布局须 `pip install -e` 拉入 torch 依赖链，违反板端红线；主链路为自研 scservo_sdk 封装） | v0.6.1（site-packages） | 板端推理链路零 LeRobot 依赖；板端遥操作用自研工具链 `hardware/teleop.py`（TeleopPair 30Hz 跟随）+ `scripts/lerobot-record-lite`（`--follow` 跟随或纯串口只录），标定 `tools/calibrate_arm.py`，录制数据 PC 端 `scripts/json_to_lerobot.py` 转换 |
| ffmpeg | 系统包（需含 h264_rkmpp / scale_rkrga） | — | apt 或板卡厂商 RK 构建 |

### 环境自检命令

板端：

```bash
python3 - <<'EOF'
import sys, numpy, cv2, onnxruntime, pyrealsense2, sherpa_onnx, serial, pexpect
print("python ", sys.version.split()[0], "(want 3.10.x)")
print("numpy  ", numpy.__version__, "(want <2.0)")
print("ort    ", onnxruntime.__version__)
print("cv2    ", cv2.__version__)
EOF
python3 -c "from rknnlite.api import RKNNLite; print('rknn-toolkit-lite2 OK')"
ffmpeg -encoders 2>/dev/null | grep h264_rkmpp
```

PC 端：

```bash
python3 - <<'EOF'
import sys, numpy, torch, onnx, onnxruntime, safetensors
print("python      ", sys.version.split()[0], "(want 3.12.x)")
print("numpy       ", numpy.__version__, "(want <2.0)")
print("torch       ", torch.__version__)
print("onnx        ", onnx.__version__)
print("onnxruntime ", onnxruntime.__version__)
print("safetensors ", safetensors.__version__)
EOF
python3 -c "import lerobot; print('LeRobot', lerobot.__version__, '(want 0.6.1)')"
```

### 冲突红线汇总

1. **numpy 2.x 禁入板端**（rknn-toolkit-lite2 2.3.2 不兼容）；PC 端放宽 >=1.24（实测 2.2.6 随 LeRobot 生态，ONNX 产物与 numpy 版本无关）
2. **板端不 pip 安装/升级 torch**（保留系统 RK 优化构建）
3. **Python 3.10（板）/ 3.12（PC）严格分离**，不跨端复用 site-packages 或 conda env
4. **ONNX opset**: GGCNN=12、ACT>=14；两端 onnxruntime>=1.16 才能加载全部导出产物
5. **LeRobot 仅 PC 端**（`pip install lerobot==0.6.1`，site-packages 安装）；repo 内 vendored 子集已于 2026-09 删除，板端遥操作/标定走自研工具链（`hardware/teleop.py` + `scripts/lerobot-record-lite --follow` + `tools/calibrate_arm.py` + `tools/feetech_scan.py`，scservo_sdk/纯 pyserial 实现，零 lerobot 依赖），录制数据 PC 端 `scripts/json_to_lerobot.py` 转 npz/LeRobotDataset

## 3. 模型部署

### VLM 模型

`models/vlm/Qwen3.5-0.8B/` 包含（gitignore，需单独部署到板端）：

```
demo                          # Qwen3.5 C++ demo 程序（RKLLM 子进程调用）
lib/                          # RKLLM 运行时库
Qwen3.5-0.8B_vision_rk3588.rknn   # 视觉编码器
Qwen3.5-0.8B_w8a8_rk3588.rkllm    # LLM 模型
demo.jpg                      # 占位图
```

路径由 `config/settings.py` 的 `VLM_MODEL_PATH` / `VLM_DEMO_BIN` 指向。

### ACT / GGCNN 模型

- `models/act/`: ACT ONNX（PC 端 `tools/export_act_onnx.py` 导出后拷贝到板端）
- `models/ggcnn/`: GGCNN ONNX（PC 端 `tools/export_ggcnn_onnx.py` 导出，Cornell 预训练权重）

### 语音模型

`voice_assistant/voice_assistant/models/` 包含：

```bash
voice_assistant/voice_assistant/models/
├── sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/   # KWS 唤醒词
├── sherpa-onnx-conformer-zh-stateless2-2023-05-23/   # ASR 语音识别
├── matcha-icefall-zh-baker/                          # TTS 声学模型
└── vocos-22khz-univ.onnx                             # TTS 声码器
```

## 4. 配置

```bash
# 默认配置直接使用；相机设备号/串口按实际检测修改：
#   config/settings.py            → CAMERA_INDEX / SERIAL_PORT（单一事实来源）
#   voice/config/default.yaml     → 语音链路模型路径
# 子进程运行时（runtime/）默认关闭：settings.USE_SUBPROCESS_RUNTIME = False
#   （需板端实测 GIL/延迟/NPU 绑定后再启用，见 refactor_plan_v9 T4.2）
```

## 5. 运行

```bash
conda activate rk3588
cd /home/elf/work/rk3588-eia

# 统一入口 — 5 种模式（autonomous/voice/teleop/record/menu）
python3 main.py
python3 menu.py            # 交互式菜单

# 语音助手 — 完整链路
python3 va.py once

# 文字问答
python3 va.py ask "画面中有什么"

# 语音触发动作
python3 scripts/voice_motion.py
```

## 6. 标定

```bash
# 相机取景检查
python3 scripts/d435i_viewer.py       # D435i 深度流
python3 scripts/camera_viewer.py      # USB 相机

# 相机内参标定（回写 config/settings.py CAMERA_MATRIX）
python3 scripts/calibrate_camera.py --d435i

# 相机→机械臂基座外参标定（回写 config/settings.py CAMERA_POSITION）
python3 scripts/calibrate_extrinsics.py

# 手眼标定辅助
python3 scripts/calibrate_handeye.py

# 舵机标定参数: config/calibration.json（SO101Arm 启动时加载）
```

> 标定脚本只回写 `config/settings.py`（单一事实来源）；
> `hardware/arm.py`、`policy/grasp_pipeline.py` 均引用 settings，无需手工同步副本。

## 7. 单元测试

```bash
python3 tests/test_kinematics.py      # 运动学 FK/IK 一致性（6 用例，仅 numpy）
python3 runtime/shared_frame.py       # 共享内存帧缓冲自检
```
