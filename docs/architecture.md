# ELF2 RK3588 自主抓取系统架构文档

## 系统概览

基于 RK3588 (8GB RAM, 6 TOPS NPU) 的边缘侧具身智能执行器，实现语音指令驱动的自主抓取。

## 四层架构

```
应用层 (main.py System)
  ├── 模式管理: autonomous / voice / teleop / record / menu
  ├── 降级逻辑: 根据模块可用性自动调整
  └── 生命周期: init_hardware → run → shutdown

策略层 (policy/)
  ├── GraspPipeline: VLM + GGCNN + IK 完整抓取管线
  ├── Kinematics: 6DOF IK (XLeRobot 偏移补偿)
  └── ACTPolicy: ONNX 推理 (第三阶段)

感知层 (perception/)
  ├── VLMPerception: Qwen3.5-0.8B 目标检测 (NPU 子进程)
  ├── GGCNNDetector: 实时抓取位姿检测 (ONNX Runtime)
  └── CameraManager: D435i 深拷贝帧缓冲

硬件层 (hardware/)
  ├── SO101Arm: scservo_sdk + SYNC_READ/WRITE + 串口恢复
  ├── SafetyMonitor: 深度避障 + 急停
  └── interfaces.py: 核心数据结构 + 模块接口
```

## 数据流

```
CameraManager (30Hz) → FrameBuffer (深拷贝)
     ↓
VLMPerception (1-3Hz) → bbox [0,1] 归一化
     ↓
GGCNNDetector (10-30Hz) → grasp_pose (angle, width, quality)
     ↓
GraspPipeline → 三段轨迹 (pre_grasp → grasp → lift)
     ↓
SO101Arm → SYNC_WRITE → 舵机执行
```

## 资源分配

| 组件 | CPU 核 | NPU | 内存 |
|------|--------|-----|------|
| CameraManager | A55 核 0-1 | - | ~200MB |
| SO101Arm IO | A55 核 1 | - | ~10MB |
| VLM | A76 核 6-7 | 核 0 | ~900MB (按需) |
| GGCNN | A76 核 5 | - | ~50MB |
| ACT (P3) | A76 核 4-5 | 核 1 (可选) | ~350-450MB |
| Voice + Safety | A55 核 2-3 | - | ~300MB |

## 环境配置

- 板端: Python 3.10, conda env `rk3588`, onnxruntime + rknn-toolkit-lite2
- PC 端: Python 3.12, conda env `rk3588`, LeRobot v0.6.1 + torch
- 模型路径: `/home/elf/work/rk3588-eia/models/`

## 关键参数

- 手眼标定角: 101.9°
- 相机外参: [0.182, -0.129, 0.47]
- 相机内参: fx=604.23, fy=604.07, ppx=315.13, ppy=250.89
- 工作空间: x[0.03,0.45] y[-0.30,0.45] z[0.01,0.40]
- IK 参数: L1=0.1159, L2=0.1350, 偏移补偿 θ1≈14.0° θ2≈16.2°
