# 开发日志 (CHANGELOG)

## v0.2.0 - 2026-09-23 (架构重构第一阶段)

### 新增
- `hardware/interfaces.py`: 核心数据结构（Observation/Action/TaskRequest/TaskResult）+ 模块生命周期接口（Module + 降级契约）
- `hardware/arm.py`: SO101Arm 类（scservo_sdk 封装、SYNC_READ/WRITE、单例+崩溃恢复、串口异常恢复）
- `hardware/camera_d435i.py`: CameraManager（FrameBuffer 深拷贝保护、预热机制、采集线程绑核）
- `hardware/safety.py`: SafetyMonitor（深度避障、急停触发/清除、独立监控线程）
- `policy/kinematics.py`: 完整 6DOF IK（XLeRobot 偏移补偿、URDF 精确参数、FK/IK 一致性 < 1mm）
- `tests/test_kinematics.py`: IK 单元测试（FK/IK 一致性、工作空间钳制、关节限位）
- `main.py`: 统一入口 System 类（5 种模式、降级逻辑、逆序资源释放、信号处理）
- `config/calibration.json`: 6 个舵机默认标定参数
- 新目录结构: `hardware/`, `policy/`, `perception/`, `voice/`, `tools/`, `models/`

### 删除（废弃文件）
- `vla/control/controller.py` — 被 `hardware/arm.py` 替代
- `vla/control/kinematics.py` — 被 `policy/kinematics.py` 替代
- `vla/kinematics.py` — 被 `policy/kinematics.py` 替代（含 XLeRobot 偏移补偿）
- `vla/command_queue.py` — 被 `main.py` System 类替代
- `tests/test_ik.py` — 被 `tests/test_kinematics.py` 替代
- `tests/test_locator.py` — 依赖旧 ColorLocator，无独立价值

### 环境配置
- PC 端 (WSL2): conda env `rk3588`, Python 3.12, LeRobot v0.6.1, torch CPU-only
- 板端 (RK3588): conda env `rk3588`, Python 3.10, onnxruntime 1.23.2, rknn-toolkit-lite2 2.3.2, torch 2.7.0+cpu
- WSL2 代理: mirrored 网络模式 + autoProxy

### 技术决策
- Python 版本分离: 板端 3.10（复用系统 rknn/torch）/ PC 端 3.12（LeRobot v0.6.1 要求）
- 子进程架构: ACT/GGCNN/VLM 各走独立子进程，规避 GIL
- NPU 单进程单核: RK3588 NPU 每核独立子进程
- IK 统一为 XLeRobot 偏移补偿方案（L1=0.1159, L2=0.1350）
- ACT 延迟预估修正: 70-120ms（RK3588 A76，非 x86 的 35-60ms）
- FrameBuffer 深拷贝保护: 读写均 .copy()

## v0.1.0 - 2026-09-20 (初始版本)
- 原始项目代码（VLA pipeline + 语音助手 + D435i 相机）
