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
- `docs/test_plan.md`: 架构对齐重写 — M1→SO101Arm（含 M1.5 接口符合性）、M2→voice/ 新栈、M5 增 H264Encoder、M6→perception/+GraspPipeline 双模式、M7→config/safety+SafetyMonitor，新增 M8 相机 / M9 运动学 / M10 runtime、P0 集成验证前置项、I5/I6 端到端链路、S4 性能基准扩充（ACT/GGCNN 延迟预算、8GB 内存阈值修正）
- `docs/deploy_guide.md`: 新增 §2.5 环境与版本矩阵 — 两端实测基线（板 3.10 + ort 1.23.2 + rknn-lite 2.3.2 / PC 3.12 + LeRobot 0.6.1）、逐包版本对照、5 条冲突红线、两端环境自检命令
- `requirements-dev.txt`: 显式补 `safetensors>=0.4`（export_act_onnx.py 加载 checkpoint 所需，此前靠 LeRobot 传递安装）+ 版本约束注释
- numpy PC 端约束放宽为 `>=1.24`（实测 2.2.6 随 LeRobot/torch 生态）；**<2.0 红线仅约束板端**（rknn-toolkit-lite2），requirements-dev.txt 与 deploy_guide §2.5 同步修订

### 部署与环境核查（2026-09-24）
- 两端环境实测（矩阵见 deploy_guide §2.5）：板端 conda `rk3588` = py3.10.21 / numpy 1.26.4 / ort 1.23.2 / rknnlite 2.3.2 / **新装 pyrealsense2 2.58.4 + sherpa-onnx 1.13.8**；PC 端 WSL2 conda `rk3588` = py3.12.14 / torch 2.11.0+cpu / lerobot 0.6.1 / safetensors 0.8.0 / **新装 onnx + onnxruntime + scipy**
- 板端 `/home/elf/work/rk3588-eia/` **首次部署本项目**（此前板上无本项目——v0.3.0 所记旧部署与实板不符，`~/work/rkrobot` 为另一项目 RK3588-SO-ARM101）：代码 2.7MB + VLM 模型 1476MB（rsync 校验一致，demo/imgenc 具可执行位）；`voice/config/default.yaml` 写死的板端路径随之生效
- PC 端（WSL2 Ubuntu 22.04）代码部署至 `/home/shimuzi/work/rk3588-eia/`（导出工具链）
- 板端外网仅 ~40KB/s → 大资产一律 WSL2 中转（外网口下载 → 内网 rsync 板端）
- ACT 链路暂缓：checkpoint 两端均不存在（待 LeRobot 训练产物）；GGCNN 优先（Cornell 权重 → ONNX opset 12，PC 端导出 → 传板）
- 语音模型四件套部署板端（ASR conformer-zh 474M / matcha-icefall-zh-baker 88M / KWS zipformer-3M 39M / vocos 52M），对照 `voice/config/default.yaml` 12/12 关键文件校验通过；下载通道：板端外网仅 ~40KB/s、WSL2 GitHub 直连 ~11KB/s、gh-proxy ~277KB/s，最终 **Windows 系统代理（Clash）直下最稳**，经 WSL2 rsync 中转上板
- GGCNN 权重实际发布于 dougsm/ggcnn **release v0.1** `ggcnn_weights_cornell.zip`（repo raw 路径返回 404 HTML——v0.3.0 待办"下载 Cornell 权重"所记 URL 有误）；`ggcnn_epoch_23_cornell_statedict.pt` 为 2018 老格式 pickle
- `tools/export_ggcnn_onnx.py` 修复：torch>=2.6 下 `weights_only=True` 解析老权重抛 `UnpicklingError`，增加回退 `weights_only=False`（仅限可信来源）
- GGCNN 导出实测：torch 2.11 新导出器拒绝 opset 12、**实际保持 opset 18**（板端 ort 1.23.2 可加载）；新导出器默认拆分权重为 `.onnx.data` → 已内联合并为**单文件 271,701B** 传板；板端单帧推理实测 **50.7ms**（S4.10 预算 30-50ms 上限，未绑核，worker 绑 A76 后预期改善）
- `requirements-dev.txt` 补 `onnxscript>=0.10`（torch>=2.9 `torch.onnx.export` 新导出器依赖，部署实测发现）
- 板端 L1 冒烟全绿：kinematics 单测 6/6、shared_frame 自检通过、**21/21 模块导入链 OK**（含 scservo_sdk/pyrealsense2/sherpa_onnx/rknnlite 全部硬件依赖）；PC 端 WSL2 自检 9/9（pyrealsense2 缺失时 CameraManager 优雅降级提示生效）
- 环境审计后续（2026-09-25）：板端 requests 依赖链补装（urllib3/idna/certifi——系统 dist-packages 的 requests 缺依赖所致）；**删除 vendored `lerobot/` 子集（85 文件，repo+两端部署副本）**——src 布局必须 `pip install -e`（拉入 torch 重依赖）违反板端红线，且主链路为自研 scservo_sdk 封装、对其零依赖，遥操作录制改走 `scripts/lerobot-record-lite` 轻量替代或 PC 端 pip lerobot（推翻 9251cd0 的暂留决策）；deploy_guide/test_plan 相关指引同步更新
- 板端 env pip list 可见 torch 2.7.0+cpu/draccus/deepdiff 的归属澄清（2026-09-25 审计）：**均非本项目残留、全部保留**——torch=厂商镜像系统预装（`/usr/local/lib/python3.10/dist-packages`，396MB，与 rknn_toolkit_lite2 并存，conda env sys.path 含系统目录故 pip 可见）；draccus/deepdiff=user site（`~/.local`）历史积累，rkvla env 的 lerobot 0.4.4 共用其中 draccus，卸载会破坏 rkvla；conda rk3588 env 自身零 torch/零 lerobot 确认无误
- 新增 `docs/pc_board_feetech_plan.md`：**PC 端/板端功能矩阵**（板端=实时推理执行端 11 功能域已全部部署、PC 端=开发/导出/训练端，数据流单向闭环：板端录制→PC 训练→导出 ONNX→传板推理）+ **自研 Feetech 协议层构建方案**——通读 lerobot 0.6.1 motors_bus/feetech/so_follower/so_leader 四文件后逐点对照，列出缺口 G1-G12（P0 四项：连接握手校验、STS3215 Phase bit4 防角度溢出、目标突变限幅、夹爪防烧三参数），规划 feetech_bus.py 协议层 + 标定向导 + 自研遥操作录制三阶段路线（零新增 pip 依赖，明确不做多品牌抽象避免过度设计）

### 自研 Feetech 协议层：阶段 A-D 全量落地（2026-09-25）

按 `docs/pc_board_feetech_plan.md` 四阶段全部实现并两端验证（方案文档 §6 有验收明细）：

- **阶段 A 协议层**: 新增 `hardware/feetech_bus.py`（控制表驱动 STS3215 封装，lerobot Apache-2.0 设计借鉴+独立实现）——握手校验 G1（逐 ID ping+型号码 777）、Phase bit4 清除 G2、目标突变限幅纯函数 G3、夹爪防烧 G4、控制表 G7、Operating_Mode G8、sign-magnitude 编解码、串口异常恢复、broadcast_ping/多波特率扫描/setup_motor G10、标定 EEPROM 读写 G6；`hardware/arm.py` 重构为委托 FeetechBus——**公开 API 零破坏**（main.py/grasp_pipeline/safety 调用点零改动），`execute()` 新增可选 `max_relative_step_deg` 帧间限幅（G3 策略流第二道防线）；PID G11/固件校验 G12 接口预留
- **阶段 B 标定**: 新增 `tools/calibrate_arm.py` 交互向导（握手→中位归零→30Hz 全行程录制→arm.py 兼容 JSON；可选 `--write-eeprom` 写入舵机 + `--verify` 读回校验，全程禁扭矩+退出自动恢复）
- **阶段 C 遥操作**: 新增 `hardware/teleop.py`（LeaderArm 只读主臂 + TeleopPair 30Hz 跟随环：起步插值平滑对齐防突跳、G3 限幅、record-lite 兼容 JSON 录制）；`scripts/lerobot-record-lite` 补 `--follow` 跟随模式；新增 `scripts/json_to_lerobot.py`（PC 端录制 JSON→npz 或 LeRobotDataset，兼容 lerobot 0.4.x/0.6.x API 签名）
- **阶段 D 工具**: 新增 `tools/feetech_scan.py`（当前/全波特率广播扫描 + 单电机出厂初始化改 ID/波特率）
- **测试**: 新增 `tests/test_feetech_bus.py` 8 项（sign-magnitude 往返、控制表 24 地址回归、EPROM 可写集合、G3 限幅四路径、raw↔rad 换算、表一致性、无 SDK 优雅失败、arm 导入链）
- **修复两个存量 Bug（构建中发现）**:
  1. `arm.py` 寄存器地址对调——原把 0x29(41)=Acceleration 当 "Return_Delay_Time" 写、0x1A(26)=CW_Dead_Zone 当 "Acceleration" 写，真正的 Return_Delay_Time(addr 7) 从未被设置；控制表驱动修复，真机读回证实
  2. `scripts/lerobot-record-lite` argparse dest Bug——`--robot.port` 定义后用 `args.robot.port` 点号访问（属性名含点，运行即崩）；显式 `dest=` 修复
- **验证**: PC 端（WSL2）单测 8/8 + kinematics 回归 6/6 + py_compile/import 链/4 工具 --help 全过 + npz 端到端 + **lerobot 0.4.4 env 数据转换验收 rc=0**（v2.x parquet+meta 10 产物）；板端单测 8/8 + **导入链 23/23**（含 feetech_bus/teleop 新模块）+ **真实硬件握手**（follower/leader 各 6×STS3215 model=777，broadcast_ping 6 ID error=0，read_positions 实测换算正常，id1 电压 4.9V/温度 28°C）+ **configure 真机验收**（用户在场确认）：写前快照 Phase bit4=1 共 5 台（G2 隐患实机证实）、写后读回全部一致（bit4 清除/Return_Delay=0/Acceleration=16/POSITION/扭矩恢复/夹爪防烧 500-250-25）、耗时 0.2s、位置读数全 [0,4096)
- 已知平台差异（非缺陷）：PC 端 `voice.wake/asr/tts/orchestrator` 4 模块导入失败 = sherpa_onnx 未装（语音运行时仅板端职责，deploy_guide 红线 5）；真实标定（calibrate_arm）与双臂跟随（teleop）需手搬交互，按 test_plan P0.4/P1.1 执行

### Follower 实机标定 + 标定链路两修复（2026-09-25，P0.4）
- **follower(ttyACM0) 实机标定完成**：`calibrate_arm.py --write-eeprom --verify` 向导（中位归零 + 3634 帧全行程录制），Homing_Offset（带符号 +30/+5/−31/−19/−38/−90）与 Min/Max_Position_Limit 写入 6 舵机 EEPROM，只读复核 **6/6 EEPROM↔JSON 一致**；`config/calibration.json` 已更新为实机标定值
- **修复 verify 负偏移误报**：Homing_Offset 为有符号寄存器（bit11 sign-magnitude），`read()` 读回已解码带符号，旧 verify 却把期望编码成无符号再比——中位 raw<2047 的 4 个舵机全被误报 MISMATCH；统一带符号域比对
- **修复 enable_torque 突跳隐患（真机复现）**：禁扭矩手搬臂后 Goal_Position 停留旧值，恢复扭矩瞬间舵机冲向旧目标（标定退出时 4 关节冲出 range 最多 ~64°）；`FeetechBus.enable_torque` 现先 sync_read Present → sync_write Goal 对齐再上扭矩
- 负偏移说明：中位 raw 读数 <2047 的舵机（id3-6）需负 Homing_Offset 把读数上移回半圈中点，属 STS3215 正常语义

### 协议完整性复审第二轮（2026-09-25）
- 对照 lerobot 0.6.1 全 API 面 + 社区 commanderfun/STS3215 Servo 类 + 官方新 SDK（ftservo-python-sdk 2.0.0）逐项核查（明细与跳过理由见 plan 文档 §6）
- **补齐**：G12 落地（`firmware_versions` + 握手固件一致性检查）、`read_diagnostics`（Status 错误标志解码/温度/电压/电流/负载+方向/Moving 一次只读全总线）、`decode_status_flags`/`decode_load` 纯函数（STS3215 位定义：bit0 Voltage/bit1 Sensor/bit2 Temperature/bit3 Current/bit5 Overload；Load bit10 方向；电流 6.5mA/step）、`wait_until_stopped`（move_sync 语义）+ `move_to(wait=True)`、`gripper_current()` 抓取闭环判据、teleop 实测 fps 统计
- **明确跳过**（防过度设计）：reset_calibration（危险）、REG_WRITE+ACTION（SYNC_WRITE 已覆盖）、NORMALIZE_MODES（rad 语义已有）、多品牌 Protocol 1（单型号决策）、wheel mode（无场景）、官方新 SDK 引入（scservo_sdk 已部署实测+补丁就位）
- **验证**：单测扩至 **11/11**（PC+板端双过）；板端真机只读实测——固件 6 台全 3.10 一致 ✓、诊断全健康（33-36°C / 11.9-12.0V / 错误标志空 / 静态 0mA）✓

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
