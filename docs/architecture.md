# ELF2 RK3588 自主抓取系统架构文档

> 本文档为技术手册。标注 **[已落地]** 的为当前代码已实现并（静态）验证；
> 标注 **[规划中]** 的为设计目标，尚未默认启用或需板端实测。
> 最近一次架构清理见文末「架构债清理记录」。

## 系统概览

基于 RK3588 (8GB RAM, 6 TOPS NPU) 的边缘侧具身智能执行器，实现语音指令驱动的自主抓取。

## 四层架构

```
应用层 (main.py System)                                    [已落地]
  ├── 模式管理: autonomous / voice / teleop / record / menu
  ├── 降级逻辑: 根据模块可用性自动调整 (ACT→GGCNN→VLM)
  └── 生命周期: init_hardware → init_policy → run → shutdown

策略层 (policy/)
  ├── GraspPipeline: VLM + GGCNN + IK 完整抓取管线 (PolicyModule)   [已落地]
  ├── Kinematics: 6DOF 解析 IK + FK (XLeRobot 偏移补偿)            [已落地, FK/IK 互逆 <1mm]
  └── ACTPolicy: ONNX 推理 (PolicyModule)                          [已落地, 模型待导出]

感知层 (perception/)
  ├── VLMPerception: Qwen3.5-0.8B 目标检测 (RKLLM 子进程)          [已落地]
  ├── GGCNNDetector: 实时抓取位姿检测 (ONNX Runtime)               [已落地, 模型待导出]
  ├── ColorLocator: HSV 颜色定位降级                               [已落地]
  └── (CameraManager 见硬件层)

硬件层 (hardware/) — 均实现 HardwareModule 接口
  ├── SO101Arm: scservo_sdk + SYNC_READ/WRITE + 串口恢复           [已落地, 接口已实现]
  ├── CameraManager: D435i 深拷贝帧缓冲                            [已落地, 接口已实现]
  ├── SafetyMonitor: 深度避障 + 急停                               [已落地]
  ├── H264Encoder: ffmpeg + h264_rkmpp 硬件编码 (Module)           [已落地]
  └── interfaces.py: 核心数据结构 + 模块接口 (Module/Perception/Policy/Hardware)

运行时层 (runtime/) — 子进程 + 共享内存                            [规划中, opt-in]
  ├── SharedFrameBuffer: 跨进程零拷贝帧传输 (seqlock)              [已落地, 往返已验证]
  └── SubprocessWorker/InferenceWorker: 推理子进程 + 绑核          [脚手架, 需板端实测]
```

## 数据流

```
CameraManager (采集线程, 30Hz) → FrameBuffer (深拷贝)
     ↓
VLMPerception (1-3Hz, RKLLM 子进程) → bbox [0,1] 归一化
     ↓
GGCNNDetector (10-30Hz, 进程内 ONNX) → grasp_pose (angle, width, quality)
     ↓
GraspPipeline → 三段轨迹 (pre_grasp → grasp → lift) → Kinematics IK
     ↓
SO101Arm → SYNC_WRITE → 舵机执行
```

## 并发模型

| 组件 | 当前实现 | 目标 (refactor_plan_v9 §1.6) | 状态 |
|------|----------|------------------------------|------|
| CameraManager | 进程内采集**线程** | 独立子进程 (A55 0-1) | [规划中] |
| SO101Arm IO | 进程内 | 串口 IO 子进程 (A55 1) | [规划中] |
| VLM | RKLLM **子进程** | 子进程 (A76 6-7) | [已落地] |
| ACT / GGCNN | 进程内 ONNX Runtime | 独立子进程 (A76 4-5 / 5) | [规划中] |
| 帧 IPC | FrameBuffer 深拷贝 (线程) | multiprocessing.Queue + shared_memory | [规划中] |

- **默认路径**: 进程内（CameraManager 采集线程 + 推理同进程），已在板端验证可工作。
- **子进程路径 (opt-in)**: `config.settings.USE_SUBPROCESS_RUNTIME=True` 启用 `runtime/`
  的子进程 + 共享内存帧传输，规避 Python GIL 抖动、适配 NPU 单进程单核限制。
  `SharedFrameBuffer` 已通过往返 + 跨句柄 attach 自检；完整多进程编排需板端实测 (T4.2)。
- **绑核**: 子进程不继承父进程 CPU 亲和性，`runtime.worker.rebind_affinity()` 在子进程内
  用 `config.settings.CORES_*` 重新绑核。

## 资源分配（refactor_plan_v9 §4.1）

> 下表为**目标分配**（子进程路径）；进程内默认路径下由 OS 调度，语音/安全已绑 A55 核 2-3。

| 组件 | CPU 核 | NPU | 内存 | settings 键 |
|------|--------|-----|------|-------------|
| CameraManager | A55 核 0-1 | - | ~200MB | `CORES_CAMERA` |
| SO101Arm IO | A55 核 1 | - | ~10MB | `CORES_ARM_IO` |
| Voice + Safety | A55 核 2-3 | - | ~300MB | `CORES_VOICE` / `CORES_SAFETY` |
| GGCNN | A76 核 5 | - | ~50MB | `CORES_GGCNN` |
| ACT | A76 核 4-5 | 核 1 (可选) | ~350-450MB | `CORES_ACT` |
| VLM | A76 核 6-7 | 核 0 | ~900MB (按需) | `CORES_VLM` |

## 配置单一事实来源

- **相机内参** `CAMERA_MATRIX`、**外参** `CAMERA_POSITION` 统一定义于 `config/settings.py`。
- `hardware/arm.py`、`policy/grasp_pipeline.py` 均**引用** settings，不再复制字面量
  （修复前存在硬编码副本，标定脚本回写 settings 后会静默漂移）。
- 标定脚本 `scripts/calibrate_extrinsics.py` / `calibrate_camera.py` 只回写 `settings.py`。

## 环境配置

- 板端: Python 3.10, conda env `rk3588`, onnxruntime + rknn-toolkit-lite2
- PC 端: Python 3.12, conda env `rk3588`, LeRobot v0.6.1 + torch (见 requirements-dev.txt)
- 模型路径: `./models/`（相对项目根；VLM 子进程经 RKLLM demo 加载）

## 关键参数

- 手眼标定角: 101.9°（`GraspPipeline.CAM_ANGLE`，抓取管线专有，单处定义）
- 相机外参: `[0.182, -0.129, 0.47]`（settings.CAMERA_POSITION）
- 相机内参: fx=604.23, fy=604.07, ppx=315.13, ppy=250.89（settings.CAMERA_MATRIX）
- 工作空间: x[0.03,0.45] y[-0.30,0.45] z[0.01,0.40]
- IK 参数: L1=0.1159, L2=0.1350, 偏移补偿 θ1≈14.0° θ2≈16.2°

## 架构债清理记录

### 2026-09-24 — 4 项架构债清理（进入第三阶段前）

| # | 债 | 处置 | 验证 |
|---|-----|------|------|
| 1 | 依赖倒置: `hardware/arm.py` 导入 `policy.kinematics` | 改为**依赖注入**: `SO101Arm.set_kinematics()`，由组合根 `main.py` 注入；硬件层不再 import 策略层 | py_compile；arm.py 无残留 policy 导入 |
| 2 | 配置硬编码: 相机内外参在 arm.py/grasp_pipeline.py 重复 | 统一**引用** `config.settings.CAMERA_POSITION/CAMERA_MATRIX`，消除字面量副本 | py_compile；settings 单一源 |
| 3 | 子进程 + 共享内存未落地（`import mp` 死导入） | 新建 `runtime/`: `SharedFrameBuffer`(seqlock 零拷贝) + `SubprocessWorker/InferenceWorker`(绑核)；opt-in 开关；移除死导入 | 往返 + 跨句柄 attach 自检通过；多进程编排待板端实测 |
| 4 | `SO101Arm`/`CameraManager` 未实现 `HardwareModule` | 两类继承 `HardwareModule`，实现 setup/start/stop/is_available/on_failure/execute/get_observation | CameraManager 接口一致性导入测试通过；SO101Arm 同模式（板端运行时确认） |

**说明**: 债 3 的完整多进程编排（相机/ACT/GGCNN 各独立子进程）为 opt-in 脚手架，
默认仍走已验证的进程内路径；启用前需板端实测 GIL/延迟/NPU 绑定（refactor_plan_v9 T4.2 / 风险 R2）。
业务功能推进快于架构清理的问题已在本次集中偿还。
