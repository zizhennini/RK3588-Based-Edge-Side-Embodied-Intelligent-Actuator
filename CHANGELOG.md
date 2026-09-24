# 开发日志 (CHANGELOG)

## v0.5.0 - 2026-09-24 (架构债清理 + 文档同步)

对应提交 `0e7ed92`。详见 `docs/refactor_plan_v9.md` v9.1 节与 `docs/architecture.md` 架构债清理记录。

### 修复（4 项架构债）
- **依赖倒置**: `hardware/arm.py` `move_to()` 不再 `from policy.kinematics import Kinematics`，改为依赖注入（`SO101Arm(kinematics=)` / `set_kinematics()`），由组合根 `main.py:init_arm` 注入；硬件层零 policy 导入
- **配置硬编码**: `arm.py` / `grasp_pipeline.py` 相机内外参不再复制字面量，统一引用 `config.settings.CAMERA_POSITION/CAMERA_MATRIX`（单一事实来源，消除标定回写后的静默漂移）
- **子进程未落地**: 移除 `main.py` 死导入 `import multiprocessing as mp`；新增 `USE_SUBPROCESS_RUNTIME` opt-in 开关（默认 False 走已验证的进程内路径）
- **接口未落地**: `SO101Arm` / `CameraManager` 继承 `HardwareModule` 并实现全部接口方法（arm `on_failure="abort"` 硬依赖；camera `on_failure="skip"` 可降级；camera `execute()` 为文档化 no-op）

### 新增
- `runtime/` 包（refactor_plan_v9 §1.6/§4.1 落地组件）:
  - `shared_frame.py`: `SharedFrameBuffer` — 基于 `multiprocessing.shared_memory` 的 seqlock 零拷贝跨进程帧传输
  - `worker.py`: `SubprocessWorker`（spawn + 命令/响应 Queue + 子进程内重新绑核）+ `InferenceWorker`（ACT/GGCNN 推理子进程，模型子进程内构建）
- `config/settings.py`: `USE_SUBPROCESS_RUNTIME`、`CORES_*` 各子进程绑核表（§4.1 资源分配集中定义）、`SHARED_FRAME_NAME`

### 清理（无用文件 / 死代码链）
- 删除 `vla/vision/detector.py`（MobileNetSSD）— 模型文件已于 v0.3.0 删除，运行时必崩；`vla/vision/__init__.py` 同步移除导出，README 重写（完成 v0.3.0 待办"清理遗留代码引用"）
- 删除 `scripts/test_pipeline.py` — 以 MobileNetSSD 为核心的遗留三联测试，随模型删除已不可运行
- 删除 `scripts/download_mobilenet.py` / `scripts/download_smolvlm2.py` — 下载 v0.3.0 已清理模型的失效脚本
- 删除 `config/settings.py` 中 `SSD_PROTOTXT/SSD_CAFFEMODEL/SSD_CONFIDENCE` 死配置（无任何消费方）
- 本地清理: 空遗留文件 `test_output.txt`、各目录 `__pycache__/`（均未跟踪）
- 保留决策: `camera/`、`vla/`（除 detector）、`voice_assistant/` 仍被约 20 个脚本及 `voice/orchestrator.py` 引用，暂不删除；`lerobot/`（85 文件 vendored 子集）为遥操作/录制脚本依赖（`pip install -e` 后被 `calib_teleop.py`/`teleop_record.py` 引用），第三阶段数据录制可能复用，暂不删除

### 文档
- `docs/architecture.md`: 全面修订 — [已落地]/[规划中] 标注、并发模型对照表、配置单一源说明、架构债清理记录
- `docs/refactor_plan_v9.md`: 开发日志追加 v9.1 + 债处置表 + 遗留项
- `README.md` §5 代码结构树、`config/README.md`、`tests/README.md`、`docs/deploy_guide.md`、`scripts/README.md` 同步当前架构

### 验证
- 语法 23/23、本地导入一致性 172/0、运动学 FK/IK 6/6、SharedFrameBuffer 往返 + 跨句柄 attach 自检通过
- 注: 完整多进程编排（默认启用子进程）需板端实测后开启（T4.2 / 风险 R2）

## v0.4.0 - 2026-09-24 (ACT 策略集成 + 第一阶段架构对齐)

对应提交 `6dc4e5f`（stage3 ACT）+ `eabd747`（stage1 架构对齐）。

### 新增
- `policy/act_policy.py`: ACTPolicy — ACT 策略 ONNX 推理封装（PolicyModule 接口，action chunk 逐步执行）
- `tools/export_act_onnx.py`: ACT ONNX 导出脚本（PC 端 LeRobot checkpoint → ONNX，opset 14+，自动验证）
- `policy/grasp_pipeline.py`: 双模式切换 — ACT（优先，50Hz 连续控制）/ VLM+GGCNN（兜底，三段式），运行时经 `use_act` 切换、ACT 不可用自动降级
- `hardware/encoder.py`: H264Encoder — ffmpeg + h264_rkmpp 硬件编码封装（Module 接口），`main.py`/`menu.py` 录像链路接入
- `voice/`: 独立语音包（KWS 唤醒/ASR/TTS/意图/编排器/CLI，自 voice_assistant 迁移重构）
- `perception/locator.py`: ColorLocator — HSV 颜色定位降级路径
- `vla/`: 遗留兼容层（controller/command_queue/kinematics，供旧脚本引用，待第三阶段评估移除）
- `requirements-dev.txt`: PC 端开发依赖（Python 3.12: torch/onnx/LeRobot v0.6.1），与板端 `requirements.txt`（Python 3.10）分离

### 修复
- `policy/kinematics.py`: FK/IK 严格互逆（修复 FK 双重偏移 Bug，误差 394mm → <1mm）；统一偏移减法约定（θ1 仰角/θ2 相对伸直弯折）
- `tests/test_kinematics.py`: 扩展至 6 用例（FK/IK 往返、工作空间钳制、不可达点钳制、关节限位）

### 技术决策
- ACT 延迟预估 70-120ms（RK3588 A76），ONNX 运行时内存 ~280-350MB
- Python 版本分离: 板端 3.10（仅推理，不装 LeRobot）/ PC 端 3.12（数据集处理与训练）

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
- [x] 清理遗留代码引用（vla/vision/detector.py 引用已删除的 MobileNetSSD）— 已于 v0.5.0 完成

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
