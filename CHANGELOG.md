# 开发日志 (CHANGELOG)

## v0.3.0 - 2026-09-24 (第二阶段: GGCNN 通用抓取 + VLM 感知)

### 新增
- `perception/grasp_detect.py`: GGCNNDetector — GGCNN 实时抓取检测（ONNX Runtime 推理，62K 参数，预处理/后处理/NMS/坐标反映射/物理宽度换算）
- `perception/vlm.py`: VLMPerception — Qwen3.5-0.8B VLM 封装（子进程调 RKLLM demo，bbox 归一化兼容，闲置 30s 自动卸载释放 ~900MB）
- `policy/grasp_pipeline.py`: GraspPipeline — 完整抓取管线（感知→规划→执行三段式，VLM定位+GGCNN抓取+IK执行，三条降级路径）
- `tools/export_ggcnn_onnx.py`: GGCNN ONNX 导出脚本（含模型定义，opset=12，自动验证）
- `perception/__init__.py`: 模块导出
- `policy/__init__.py`: 更新导出（GraspPipeline, GraspCandidate）

### 模型文件
- `models/vlm/Qwen3.5-0.8B/`: 正确的 VLM 模型（.rknn 209MB + .rkllm 1.2GB + demo + lib/）
- 清理误放的 Qwen3-VL-2B、SmolVLM-500M/256M、yolo_world、MobileNetSSD（释放 ~4.3GB）

### 环境配置
- 板端 (RK3588): conda env `rk3588`, Python 3.10, onnxruntime 1.23.2, rknn-toolkit-lite2 2.3.2, torch 2.7.0+cpu
- PC 端 (WSL2): conda env `rk3588`, Python 3.12, LeRobot v0.6.1, torch 2.11.0+cpu
- WSL2 代理: 已清除（恢复默认 NAT 模式）
- Git: 分支 `refactor/architecture-stage1` 已推送

### 修复
- `perception/vlm.py` 模型路径更新为 `/home/elf/work/rk3588-eia/models/vlm/Qwen3.5-0.8B`
- `config/settings.py` VLM_MODEL_PATH 同步更新
- `voice_assistant/config/default.yaml` 路径同步更新
- `.gitignore` 更新覆盖 models/vlm/ 目录

### 技术决策
- GGCNN 仓库更正: dougsm/ggcnn（非 angusdk/ggcnn，后者 404）
- GGCNN 推理策略: P0 用 ONNX Runtime CPU（~30-50ms），P1 可选 RKNN NPU（<10ms）
- VLM 调用方式: 保持子进程调 RKLLM demo（输入为文件路径，需 cv2.imwrite 落盘）
- 抓取管线降级: GGCNN可用→GGCNN; 仅VLM→bbox中心+深度; 都不可用→返回失败
- 坐标变换: 手眼标定角 101.9°，相机外参 [0.182,-0.129,0.47]，精确移植自 vlm_grasp.py

### 待办
- [ ] 下载 GGCNN Cornell 预训练权重并导出 ONNX
- [ ] 板端部署验证（chmod +x demo imgenc）
- [ ] 清理遗留代码引用（vla/vision/detector.py 引用已删除的 MobileNetSSD）

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
