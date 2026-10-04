# 开源项目参考评估（快照分析）

> 快照位置：`D:\Project\RK3588\开源项目（部分）\`（**仓库外**，用户收集的部分项目，未必是最新成果）
> 分析日期：2026-09-27 ｜ 分析方式：只读代码/文档走查，逐条给出**路径证据**
> 用途：判断哪些值得参考、哪些明确不采纳，避免重复劳动与踩已踩过的坑

## 0. 怎么用这份文档

- 每条结论都带**证据路径 + 行号**，需要复核时直接跳转源码。
- **采纳清单**（§3）按优先级排列，标注对应的我方阶段：M1 采集 / M2 训练 / M3 部署。
- **不采纳清单**（§4）写明原因，避免下次再评估一遍。
- 快照**不是最新版本**，凡与官方上游冲突时以上游为准（例如 lerobot 我们自己就同步到 192a0b92）。

## 1. 项目速览与定位

| 项目 | 是什么 | 与我们的关系 | 结论 |
|---|---|---|---|
| `so101-nexus-0.6.0` | SO-100/101 全栈学习库（遥操作→LeRobot 数据集→MuJoCo→训练循环）；**follower 是仿真的** | 标定语义 + 采集工程质量 | **借语义与流程**，代码不可用 |
| `rdk_LeRobot_tools-stable` | 地瓜 RDK（BPU）板的 LeRobot/ACT 导出与部署工具（5 个文件） | 部署骨架（与 NPU 无关的部分） | **借架构**，工具链换 RKNN |
| `IB_Robot` | openEuler 具身智能框架（SO-101 + STS3215 + LeRobot v0.6.0 patch 栈 + ROS2） | 数据契约 + 质量门控 + 训练/导出配置 | **借点状资产** |
| `IB_Robot_ACT_banana_pick_distill` | HF 上的 ACT 多后端推理 bundle（torch + Ascend `.om` + **RK3588 `.rknn`**，340MiB） | **我方目标的直接对标物** | **重点参考** |
| `act-starryos-rk3588-main` / `-rust-master` | 同一起源（智能车判向任务）的 C++ / Rust 版：ACT → RKNN 上板 | RKNN 量化与板上推理工程实证 | 借**推理工程与量化教训** |
| `ffmpeg-rockchip-master` | FFmpeg 8.1.2 非官方 fork，提供 MPP 硬件编解码 + RGA 滤镜 | 落盘/预处理加速的可选项 | **暂缓**（见 §4） |
| `librga-rockchip-master` | RGA 2D 加速用户态库（im2d API 1.10.5） | 图像预处理卸载 | **暂缓**（无 Python 绑定） |
| `lerobot-rk3588-main` | lerobot 的 RK3588 变体快照 | 数据集格式/相机/训练配置 | 见 §2.8 |

## 2. 逐项目结论

### 2.1 so101-nexus-0.6.0 —— 借"语义"与"流程"，不借代码

- **标定语义与我们完全等价**（这是最重要的一条）：`mid = (range_min+range_max)/2`，
  `qpos = sign*(ticks-mid)/TICKS_PER_RADIAN` —— 角度零点 = **行程中点**，
  `homing_offset` 只负责把物理中位拉回中点。
  证据：`so101-nexus-0.6.0/.../src/so101_nexus/lerobot_adapter/normalization.py:99-101,121-126`；
  设计依据见 `docs/content/docs/concepts/observations.mdx:196-199`、`concepts/lerobot.mdx:93-120`。
  ⇒ **我方无需为 LeRobot 训练改标定语义**（我们已实现同一语义，见 `docs/pc_board_feetech_plan.md` §6 第三轮）。
- 守卫：拒绝 `range_min == range_max`（`normalization.py:77-81`）；夹爪单独 `RANGE_0_100`（`:103-109`）。
- **采集工程**（实体 leader → 仿真 follower，但同步逻辑通用）：`src/so101_nexus/teleop/recorder.py`
  - 帧率用**绝对时间轴补偿** `sleep(frame_duration - elapsed)`，超时不补睡（`:300-330`）——等价于我方 `period - elapsed`。
  - **先 `get_observation()` 再 `send_action()`**，保证 `(s_t, a_t)` 同帧配对（`:307-317`）——与我方"先读 leader 再写 follower"一致。
  - **每集人工审核闭环**（录完给视频回放 + 关节曲线，Approve/Discard 才入库）：`teleop/app.py:936-943`；
    流程见 `docs/content/docs/workflow/teleoperation.mdx:37-46` —— **我方目前缺这一环**（见 §3 采纳 #7）。
  - schema 自存档：finalize 写 `meta/so101_nexus_env.json`（`teleop/dataset.py:122-168`）——我方 `manifest.json` 已覆盖。
- **陷阱（务必遵守）**：本体关节存"度"、夹爪存 `RANGE_0_100`，`meta/info.json` **不标注单位**；
  对整条 6 维向量糊 `np.deg2rad` 会**静默只毁夹爪**（0.113 vs 0.262 rad），足以训出"永远夹不上"的策略。
  证据：`docs/content/docs/concepts/lerobot.mdx:137-145`。
  ⇒ 我方约定：**全 6 维统一弧度**（`scripts/json_to_lerobot.py` 对整条向量 `deg2rad`），
  夹爪不用 `RANGE_0_100`；该约定必须在 **M2 训练与 M3 板端推理两侧严格一致**（记入红线）。
- 不采纳：整库上板（需 Python≥3.12 + mujoco + torch；Warp 后端 **NVIDIA CUDA only**）；
  其"合成标定"自认不能替代真机标定（`synthetic_calibration.py:31-34`）。

### 2.2 rdk_LeRobot_tools-stable —— 借部署骨架，工具链换 RKNN

- 定位：把 LeRobot 训练的 ACT 部署到 **RDK S100（BPU/Nash）**；仅 5 个文件：
  `export_bpu_actpolicy.py`、`bpu_export_config.yaml`、`bpu_control_robot.py`、`damo/replace.py`、`doc/WORKFLOW_GUIDE_CN.md`。
- 与 NPU 无关、**可直接迁移到 RKNN 的骨架**（见 §3 采纳 #10-15）：
  - ACT 拆 **两张全静态图**：`VisionEncoder`（backbone + `encoder_img_feat_input_proj`）与
    `TransformerLayers`（encoder+decoder+`action_head`）；latent 置零、decoder 固定 chunk_size、
    `dynamic_axes=None`；多相机复用同一 vision encoder。证据：`export_bpu_actpolicy.py:578-640,354,388,595,626-630`。
  - **归一化参数外置 `.npy`**（每相机 mean/std + action mean/std + 反归一化参数），板端 numpy 前后处理（`:313-331`；`bpu_control_robot.py:230-246`）。
  - **chunk 队列**：`deque(maxlen=n_action_steps)`，**队列空了才推理**，逐周期 popleft（`bpu_control_robot.py:185-228,86`）——这是 30Hz 还能跑得动大模型的根本原因。
  - **链式量化校准**：Transformer 的标定输入 = VisionEncoder 的**实时输出特征**（不是原图），视觉分支每 4 个样本存 1 份（`export_bpu_actpolicy.py:516-551`）——做错精度会崩。
  - 精度验收：导出时存 PyTorch 侧 `new_actions.npy`，板端逐元素比对（`:374-377`）。
- 不采纳：`.hbm`/`hb_mapper`/OpenExplorer 工具链（地瓜专属）；`bpu_control_robot.py` 原样上板
  （**板端 `import torch`** + 依赖旧路径 `lerobot.common.*`）。

### 2.3 IB_Robot —— 借点状资产（契约/门控/配置），不引 ROS 包结构

- 定位：openEuler 具身智能框架，融合 LeRobot 与 ROS2；硬件与我方同构（SO-101 + Feetech STS3215，
  `so101_hardware/README.md:17` 记电流 `1 LSB = 6.5 mA`、`/dev/ttyACM0`，`:97-103` 同步读 15 字节反馈）。
  基线 **LeRobot v0.6.0**（`model_training_guide.md:7`）+ 12 个 patch（KD / mt_act / 样本加权 / QAT）。
- **数据契约**（可当 SSOT 参考）：录制 ROS2 bag → 转 **LeRobot v3.0**
  （`videos/<obs key>/chunk-000/file-000.mp4` + `data/...parquet` + `meta/`；
  `dataset_tools/README.md:350-364`）；`fps = int(contract.rate_hz)` 显式写入（`bag_to_lerobot.py:844`）；
  **rate_hz = 20**（单臂 `so101_single_arm.yaml:249-250`，单条上限 90s）；
  图像 `observation.images.{front,top,wrist}` 各 `resize [480,640]`（`dataset_tools/README.md:671-720`）。
- **帧质量门控**（利用 STS3215 的 `Present_Current`，我方协议层已支持读取）：
  关键帧 = 电流大速度小（`current>0.5 & vel<0.01`，权重 ×2.0，前后各扩 30 帧）；
  冻结帧 = `vel<0.1 & current<0.2`（权重 0）。证据：`model_training_guide.md:57-111`。
- 训练/导出配置（可直接抄的经验值）：小模型蒸馏 `dim_model=1024, enc_layers=4, dec_layers=2, ffn=1200`、
  AdamW lr 1e-5/wd 1e-4/grad_clip 10、`batch_size=60, steps=500000`（`model_training_guide.md:186-210`）；
  ONNX 导出只包 `act_policy.model`，`opset_version=13`，输出名 `action`，
  **归一化/反归一化全在图外**（`export_onnx_rknn.py:99-132`）；RKNN 编译**不设 mean/std**
  （`convert_to_rknn.py:137-142`）。
- 不采纳：ROS2/colcon 包结构；`third_party/`（含 NVIDIA 非商业许可的 grasp_gen wheel，
  `src/manipulation_service/README.md:67`）；其板端环境（**装了 torch ~200MB**，CPU 后路 ~80s 不可用）。
  另注：仓库声明 Apache-2.0 但根 LICENSE 缺失，`libs/lerobot` 在快照里是空目录（只有 patch 栈）。

### 2.4 IB_Robot_ACT_banana_pick_distill —— 我方目标的直接对标物（重点）

- 是什么：HF 上的推理 bundle（非数据集/非训练目录），同一策略带 **torch + Ascend `.om` + RK3588 `.rknn`** 三套产物。
  清单：`config.json`、`train_config.json`、`policy_preprocessor.json`、`policy_postprocessor.json`、
  两个 processor stats（各 7,536B）、`model.safetensors` 340MiB、`inference_manifest.json`、
  `artifacts/rknn/rk3588/policy-*.rknn` 152MB、`artifacts/ascend/.../policy-*.om` 128MB。
  来源：`https://huggingface.co/openEuler/IB_Robot_ACT_banana_pick_distill`。
- **张量契约（我方 M3 的目标形态）**：`observation.state [1,6]` f32 +
  `observation.images.top [1,3,480,640]` + `observation.images.wrist [1,3,480,640]` → `action [1,100,6]`。
  注意 480×640 = H×W，与我方 640×480 一致（**2 相机**）。
- **策略配置**：`chunk_size=100`、`n_action_steps=100`（整块播放，无时序集成）、resnet18 IMAGENET1K_V1、
  `dim_model=1024 / heads 8 / ffn 1200 / enc 4 / dec 2`、**use_vae + latent_dim 32 + kl_weight 10**、`MEAN_STD` 归一化。
- **RK3588 实测延迟（最有价值的数字）**：RKNN 推理 **~470 ms**、端到端 **~570 ms**（chunk 100 → 5 s 动作缓冲，余量充足）；
  CPU 后路 ~80 s（不可用）。证据：`README.OpenHarmony.md:333-341`。
- RKNN 编译参数：`target_platform="rk3588", float_dtype="float16", optimization_level=3,
  single_core_mode=False, do_quantization=False`（float16 为 ACT/Transformer 默认推荐）。
- **"板端禁 torch"有解**：ONNX 图内不含归一化，两个 stats 文件合计仅 15KB ⇒
  **归一化/反归一化可用纯 numpy 复现**，板端只需 rknn runtime + numpy。
- ABI 驱动：用编译器 ABI JSON + `inference_manifest.json` 驱动板端，**不要硬编码输入名/顺序**
  （编译器会重排；只有 `layout: NHWC` 时才需转置）；RKNN 输出名 `action`、Ascend 名
  `/model/action_head/Add:0:action` → 必须靠 semantic 映射。
- **负面结论（重要）**：该项目**全仓没有任何"需要多少条 episode"的经验值**（全仓正则扫描零命中）。
  ⇒ 我方 M1 的"50 条"只能当**待验证假设**，用帧质量指标（关键帧/冻结帧占比）做采集端自检，而不是只数条数。
  另：其配置 50 万步却只发布 16 万步 student ⇒ **M2 要按 eval 曲线挑 checkpoint，不要取最后一个**。

#### 2.4.1 上游版本核对（2026-09-27 实查 HF / GitCode）

| 对象 | 本地快照 | 上游最新 | 结论 |
|---|---|---|---|
| HF `openEuler/IB_Robot_ACT_banana_pick_distill` | `5ccab2fe`（2026-08-26） | **`5ccab2fe`（2026-08-26）** | **未更新，本地即最新**（4 次提交：06-30 初始 → 07-28 加 Ascend OM+RKNN → 08-26 补 manifest） |
| HF `openEuler/IB_Robot_ACT_banana_pick` | — | `37549a4d`（**2026-09-08**） | **新版，结构更完整（见下）** |
| HF `openEuler/IB_Robot_ACT_dual_arm_banana_pick` | — | `21662c68`（**2026-09-08**） | **双臂版**（12 维 + 3 相机） |
| GitCode `openeuler/IB_Robot`（源码） | `b2d87d7c`（2026-09-17） | **`f208ee90`（2026-09-30）** | **落后 19 个提交**（见下） |

**新版单臂 bundle（09-08）比我方参考的 distill 版更有参考价值**——它同时含**教师与蒸馏学生**：

| | distill（08-26） | banana_pick（09-08） |
|---|---|---|
| 目录布局 | 平铺 + `inference_manifest.json` | **`pytorch_model/` + `rknn_model/`**（各自带 `config.json`） |
| PyTorch 侧结构 | dim_model 1024 / dec 2 / ffn 1200（学生） | **dim_model 2048 / dec 7 / ffn 3200（教师）** |
| RKNN 侧结构 | 单独 `artifacts/rknn/rk3588/policy-*.rknn` | **`rknn_model/act_ros2_rknn.rknn` + config**：dim_model 1024 / dec 2（学生）、`kd: true`、`pretrained_path: ./models/502000/pretrained_model` |
| 输入契约 | state 6 + top/wrist `[3,480,640]` | PyTorch: state 6 + top/wrist `[3,480,640]`；**RKNN: state 6 + `observation.current` 6 + `hand_view`/`top_view` `[3,240,320]`** |
| 训练配置 | steps 500000、KD/ada_weight、batch 60 | `steps=500000, batch_size=16, seed=1000, num_workers=10, save_freq=10000`；数据集 `1arm_2cam_banana_pick_v1_20260514`（本地路径，非 HF）；**`image_transforms.enable=false`**（无图像增强）；`use_imagenet_stats=true`；`video_backend=torchcodec` |

值得注意的三点（**采纳前需自行验证**）：

1. **RKNN 契约 ≠ PyTorch 契约**（相机名 `hand_view/top_view`、分辨率 240×320、多一路 `observation.current`）
   ⇒ 说明其 RKNN 产物是从**另一次训练/微调**导出的，不是同一份权重的直接转换。
   **对我方的教训**：导出 RKNN 时必须**用编译期 ABI JSON 校验输入名/顺序/形状**，不能假设"和训练配置一样"。
2. **`observation.current` 作为额外状态输入**（6 维，电机电流）——与我们 §3 #10 的帧质量门控同源信号；
   我方协议层已能读 `Present_Current`，M2 可评估是否加入观测。
3. 部署侧用 **240×320**（3 相机时）/480×640（2 相机时）：分辨率越低 NPU 越省，但**必须与模型训练分辨率一致**
   （ACT 不做 resize，见 §2.8）。

**源码仓库 IB_Robot 的 19 个新提交**（2026-09-17 → 09-30，摘要）：imitation retargeting 包 +
"play retargeted imitation plans as one trajectory"、HRI 执行器由 YOLOX/PEAR 驱动、
`ibrobot_msgs` 增加 YOLOX/PEAR/trace-id 服务契约、embodied agent 计划控制面 + tracing、
GraspGen 延迟兼容、LeKiwi 运行时。
⇒ 近期重心在**感知/执行/Agent 编排**而非 ACT 训练本身；与我们 M1/M2 直接相关的只有
"imitation retargeting（把演示轨迹重定向到本体）"一条，M3 再评估。

### 2.5 act-starryos-rk3588（C++ / Rust 两版）—— 借推理工程与量化教训

- 前提：两者是同一"智能车判向"竞赛项目的两个提交版本，**任务不是机械臂**
  （action = `1×8×3` 轮速），所以**动作头结构不可抄**，可迁移的是工程链路。
- **实测数字（关键）**：
  - 预处理 **96.5 ms** vs 纯 NPU **33.7 ms** —— 证据：`experiments/final_10_images/raw_outputs/original_00_frame_000000.jpg.txt:72-105`
    ⇒ **图像预处理才是瓶颈**，不是 NPU。
  - fp16 96.95 MiB / 33.758 ms；hybrid（骨干 INT8）50.23 MiB / 33.607 ms —— **只快 0.45%**；ONNX 193.26 MiB。
  - 普通 Linux 真 NPU fp16：**24.7 ms / 峰值 206 MB**（`results/board_rk3588_real.md`）。
- **量化教训（必读）**：RKNN 模拟器判定"无损"的 hybrid 在**真 NPU 上反而最差**
  （右召回 3/19 vs fp16 18/19），根因是信号量纲与硬件量化噪声同量级（`results/board_rk3588_real.md:15-26`）
  ⇒ **量化必须真板验收，且先默认 fp16**。
- **零拷贝 IO 模式**：`rknn_create_mem` + `rknn_set_io_mem` 一次绑定输入输出，之后每帧只 `rknn_run`，
  不再 `outputs_get/release`（消除每帧输出 DMA 泄漏导致的"第 96 帧卡死"）——
  `act-starryos-rk3588-rust-master/.../board/act-rknn/src/main.rs:226-261,509-545`；
  C++ 参考实现：`act-starryos-rk3588-main/.../inference/rknn_runtime/main.cpp:336-448`（含 `core_mask` 三核全开与分段计时）。
- 不采纳：**StarryOS 路线整体不碰**——其工程主体（自烧镜像、dtb 裁剪、只读挂载绕 SD 写死锁）
  全是在解决 StarryOS 自身缺陷（根因 dwmmc 中断竞态，见 `starryos_npu/上板调试历程.md`）；
  我方 aarch64 Ubuntu 22.04 + glibc 生态不能自毁。

### 2.6 ffmpeg-rockchip —— 暂缓

- 能力：MPP 硬编解码（`h264_rkmpp/hevc_rkmpp/mjpeg_rkmpp`，异步 frame-parallel）+
  RGA 滤镜（`scale_rkrga/vpp_rkrga/overlay_rkrga`）+ DRM_PRIME 零拷贝；编码器**可直接吃 RGB24/BGR24**
  （`libavcodec/rkmppenc.h:231-240`），`rc_mode` 支持 CQP（**数据集更适合固定 QP**）。
- 暂缓理由：① 需 vendor 内核（5.10/6.1）+ `rockchip_mpp ≥1.3.9` + 强制 `--enable-libdrm`，
  要重新编译整个 FFmpeg（1 万+ 文件）；② **无 Python 绑定**，集成需 subprocess/C 胶水；
  ③ 与 LeRobotDataset v3 视频规范（编码器/时间戳/色域）契合度**未验证**，风险是"采得下、训不了"；
  ④ 我方瓶颈不在编码（见 §2.5：在预处理）。
  ⇒ 先跑通 JPEG + ACT 闭环，再单独做旁路实验验证 v3 可读后再切。

### 2.7 librga-rockchip —— 暂缓（先确认现有 OpenCV 是否已是 rk 版）

- 能力：`imresize/imcvtcolor/imcrop/imrotate/imblend/improcess`（一次组合 crop+scale+csc）；
  零拷贝 `importbuffer_fd`（性能 physical > fd > virtual，`docs/...RGA_CN.md:1476-1480`）；
  RGB888 width stride 须 4 对齐（`:450-578`）。
- 暂缓理由：**全仓库零 Python 绑定**（无 .py/.pyx/pyi，文档无 ctypes/pybind 说明）；
  ctypes 全量绑定结构体多、无编译期保护，出错表现为花屏而非报错。
  ⇒ 先确认板上 OpenCV 是否已带 RGA 后端；若是则不需要此项；若不是且实测预处理是瓶颈，
  **优先写小型 C++ 预处理扩展**而非纯 ctypes 全量绑定。

### 2.8 lerobot-rk3588-main —— 名字误导；可取的是相机校验/录制调优/ACT 前后处理清单

- **"rk3588" 与内容无关**：全仓 `.py` 搜 `rknn|RKNN|rk3588|rockchip|npu|rga|mpp` **0 命中**。
  真正的 fork 改动是**跨机遥操作**（WebSocket + ROS2 + TCP）：新增根目录
  `ros_web.py`/`ros2_to_ws.py`/`ros2_control.py` + `src/lerobot/scripts/leader_ws_stream.py`、
  `follower_ws_record.py`、`check_parquet.py`。
- **版本比我们的 0.4.4 还旧**：`pyproject.toml:28` 写 0.3.4，实际是**上游 main 在 0.4.0 发布前**
  的窗口（依赖对比：本快照 `rerun-sdk>=0.21,<0.23`、`gymnasium<1.0`；上游 v0.4.0 已是
  `rerun>=0.24,<0.27`、`gymnasium>=1.1.1`；`src/lerobot/__init__.py:54-61` 本快照仍含 `xarm`，v0.4.0 已删）。
- **快照不完整**：`src/lerobot/datasets/` 目录物理缺失（`SOURCES.txt` 列有 18 条但文件不存在），
  数据集写入细节无法从此快照取证（我们已有官方 192a0b92 的完整实现作依据，不受影响）。
- **数据集 v3 契约**（`docs/source/lerobot-dataset-v3.mdx`）：`:16-20` 多 episode 合并进同一
  parquet/mp4（v2 是一集一文件），靠元数据切 episode；`:53-59` 三支柱（Parquet 表格 / 每相机一路 MP4 /
  JSON+Parquet 元数据含 episode offset）；`:107-117` 帧级 `timestamp` + 加载期 `delta_timestamps` 时间窗。
  编码依赖 `av>=15`（PyAV 硬依赖）且 **`torchcodec` 的 marker 显式排除 aarch64/arm64**
  （`pyproject.toml:78`）——即**即便想在 ARM 上用 lerobot 的读取路径也装不上 torchcodec**，
  反向印证我方"JPEG 落盘 + 自研转换"的路线是对的。
- **相机实现细节（官方 0.4.x 线的真实行为）**：`cameras/opencv/camera_opencv.py:160` `cv2.setNumThreads(1)`；
  `:173-177` warmup 读帧 + 默认 `warmup_s=1`；**`:219-244` set→get 回读校验，失败即 RuntimeError**；
  `:375-463` 每相机 daemon 读线程 + latest_frame + `new_frame_event`（200ms 超时）。
  注意：其配置**没有 fourcc、没有 CAP_PROP_BUFFERSIZE**（我们自己踩出来的 MJPG/buffersize 经验是对官方的补充）；
  `docs/source/cameras.mdx` **没有** codec/带宽/曝光最佳实践章节。
- **录制循环的真实水位**：`scripts/lerobot_record.py:420-430` 用 `busy_wait(1/fps - dt)` 控帧率，
  **无丢帧检测、无帧级时间戳对齐**——"帧数不保证 fps×时长"；
  多相机取帧逐相机 `async_read()`（`robots/so101_follower/so101_follower.py:187`），
  **同一帧内多相机无共同时间戳、无硬件同步**。
  ⇒ **我方采集质量门控（fps/丢帧/追踪误差/图像完整性）+ 每帧 `ct_<cam>` 相机时间戳，比上游更严格**（这是我们的优势，应保留）。
  图像写出调优建议：`num_image_writer_processes=0` + `num_image_writer_threads_per_camera=4`（`lerobot_record.py:154-162`）。
- **ACT 超参基线（官方实现取值，M2 起点）**：`policies/act/configuration_act.py:94-138`
  —— `chunk_size=100` / `n_action_steps=100` / `n_obs_steps=1` / `temporal_ensemble_coeff=None` /
  `vision_backbone=resnet18 (IMAGENET1K_V1)` / `dim_model=512` / `n_heads=8` / `dim_feedforward=3200` /
  `n_encoder_layers=4` / **`n_decoder_layers=1`** / `use_vae=True` `latent_dim=32` `n_vae_encoder_layers=4` /
  `dropout=0.1` `kl_weight=10` / AdamW `lr=1e-5 wd=1e-4 lr_backbone=1e-5`（无 scheduler）/
  归一化 VISUAL+STATE+ACTION 全 `MEAN_STD`。
- **两条对部署极重要的实现事实**：
  1. **ACT 不 resize、不内置 ImageNet Normalize**（`modeling_act.py` 搜不到 resize/interpolate/normalize）
     ⇒ 图像以**原分辨率**进 ResNet-18 ⇒ **ONNX 输入 H/W 必须等于训练分辨率**；归一化由
     `NormalizerProcessorStep` 用 `dataset_stats` 做（`configs/default.py:35 use_imagenet_stats=True`）
     ⇒ 导出常量必须从 `meta/stats.json` **实读**，不能凭约定。**这直接要求我方 M1 固定分辨率（640×480）**，
     与对标 bundle 的 `[1,3,480,640]` 一致 ✓。
  2. 动作执行 = `deque(maxlen=n_action_steps)`，空则整 chunk 入队再逐步 popleft
     （`modeling_act.py:99-121`）——与 §2.2 的 chunk 队列同构，chunk=100 @30Hz = 3.3s 开环。
- **`docs/source/act.mdx:28-29`：ACT 约 80M 参数，"50 条演示常能出效果"**（`:70` 100k steps 几小时、
  `:71` batch 从 8 起）⇒ 这是我方 M1"50 条"计划**唯一找到的官方侧支持性数据点**
  （注意：是"常能出效果"的经验说法，不是保证值；IB_Robot 侧则完全未公布条数建议）。
- **导出必须自研**：该快照全仓 `onnx|rknn|tensorrt|torchscript` **0 命中**——ACT→ONNX→RKNN 的脚本
  只能自己写（可参考 §2.2/§2.4 的骨架与参数）。
- 不采纳：① 其跨机 TCP 动作通道（**无过期判定、act 与 obs 无时间戳对齐 → 跨机延迟直接变标签噪声**，
  `follower_ws_record.py:100-147,252-256`）；② `pyproject.toml:122` 的本地路径 pin
  （`pi = ["transformers @ file:///root/Desktop/..."]`，照抄会让 pip 直接失败）；
  ③ RealSense depth 相关（其实现**无 `rs.align` 对齐**、depth 复用 color 分辨率、disconnect 只 `pipeline.stop()`
  无 hardware reset；ACT 只用 RGB，与我方当前一致）。

## 3. 采纳清单（按优先级）

| # | 项 | 具体做法 | 证据 | 阶段 |
|---|---|---|---|---|
| 1 | **预处理分段计时** | 在采集/推理链路里插 decode / preprocess / NPU / postprocess 分段计时 | starryos 实测 96.5ms vs 33.7ms | M3（现在就可埋点） |
| 2 | **夹爪单位红线** | 全 6 维统一弧度；禁用"整向量 deg2rad 却把夹爪当百分比"的混用 | nexus `concepts/lerobot.mdx:137-145` | M1/M2/M3 |
| 3 | **标定语义已对齐** | 无需为 LeRobot 改标定；`(range_min+range_max)/2` 为零点 | nexus `normalization.py:121-126` | M1（已满足） |
| 4 | **chunk 队列 + 空队列才推理** | `deque(maxlen=chunk)`，逐周期 popleft，队列空才跑模型 | rdk `bpu_control_robot.py:185-228` | M3 |
| 5 | **归一化外置 + 板端 numpy** | mean/std 存 `.npy`/json，板端纯 numpy 前后处理（天然满足禁 torch） | rdk `:313-331`；banana bundle stats 15KB | M3 |
| 6 | **ACT 双静态图拆分** | VisionEncoder / TransformerLayers 两张全静态 ONNX（latent 置零、decoder 固定 chunk） | rdk `export_bpu_actpolicy.py:578-640` | M3 |
| 7 | **每集人工审核闭环** | 录完生成可视化（抽样帧拼图/短视频 + 关节曲线）供人工 Approve | nexus `teleop/app.py:936-943` | **M1（当前缺口）** |
| 8 | **RKNN 用 fp16，真板验收** | 默认 `float_dtype=float16`；量化后必须真板比对，不信模拟器 | starryos `board_rk3588_real.md:15-26` | M3 |
| 9 | **zero-copy IO** | `rknn_create_mem`+`rknn_set_io_mem` 绑定输入输出，避免每帧 get/release | starryos `main.rs:226-261` | M3 |
| 10 | **帧质量门控（采集端自检）** | 用 `Present_Current` 判关键帧/冻结帧占比，替代"只数条数" | IB_Robot `model_training_guide.md:57-111` | M1/M2 |
| 11 | **按 eval 曲线挑 checkpoint** | 不要取最后一个（对标物 50 万步配置只发布 16 万步） | banana `config.json:66` | M2 |
| 12 | **ABI/manifest 驱动推理** | 用编译器 ABI JSON 驱动，不硬编码输入名/顺序 | banana `inference_manifest.json:143-196` | M3 |
| 13 | **M2 起手配置参考** | chunk 100、2 相机 480×640、dim_model 1024/enc4/dec2/ffn1200、resnet18、use_vae latent32、MEAN_STD | banana `config.json` | M2 |
| 14 | **数据集 fps 显式写入契约** | `fps = contract.rate_hz`，并加低采样率告警 | IB_Robot `bag_to_lerobot.py:844,597-600` | M1（我方已写 fps） |
| 15 | **片段后保持窗口** | 成功后按 `success_hold*fps` 续录（默认 0.5s） | nexus `recorder.py:222-237` | M3（策略回放） |
| 16 | **ACT 超参基线** | chunk 100 / n_action_steps 100 / dim_model 512 / heads 8 / ffn 3200 / enc 4 / **dec 1** / resnet18 / use_vae latent32 / kl 10 / lr 1e-5 + AdamW 无 scheduler / 全 MEAN_STD | lerobot `policies/act/configuration_act.py:94-138` | M2 |
| 17 | **固定分辨率（不可变）** | ACT **不做 resize**，图像按训练分辨率进 ResNet-18 ⇒ ONNX 输入 H/W = 训练分辨率；M1 必须锁定 640×480 不再变 | lerobot `modeling_act.py`（无 resize）+ 对标 bundle `[1,3,480,640]` | **M1 红线** |
| 18 | **归一化常量从 meta/stats.json 实读** | `use_imagenet_stats=True` ⇒ stats 由数据集统计得出，导出时必须读实际文件，不能凭约定写死 | lerobot `configs/default.py:35`；banana `policy_preprocessor.json` | M3 |
| 19 | **录制 fps 一致性校验** | 追加录制前校验 dataset fps，防 10/30fps 混录 | lerobot `scripts/lerobot_record.py:259-261,476` | M1（我方 manifest 已记 fps） |
| 20 | **图像写出线程调优** | 落盘不稳时调 `num_image_writer_threads_per_camera`，优先线程而非子进程 | lerobot `scripts/lerobot_record.py:154-162` | M1（我方已是单写线程+有界队列） |
| 21 | **我方已优于上游的两点（保持）** | ① 采集侧有 fps/丢帧/追踪误差/图像完整性四道门控（上游**无丢帧检测**）；② 每帧记 `ct_<cam>` 相机时间戳（上游多相机**无共同时间戳**） | lerobot `lerobot_record.py:420-430`、`so101_follower.py:187` | M1 |
| 22 | **每集审核卡片** | `tools/episode_review.py`：抽样帧拼图 + 关节/追踪曲线 + 质检结论 → 单张 PNG，供人工 Approve | nexus `teleop/app.py:936-943`（我方 §3 #7 落地） | **M1（已实现）** |

## 4. 明确不采纳

| 项 | 原因 |
|---|---|
| StarryOS 相关全部内容 | 工程主体是在解决 StarryOS 自身缺陷（dwmmc 竞态）；我方 Ubuntu 22.04 链路不能自毁 |
| 地瓜 BPU 工具链（`.hbm`/`hb_mapper`/OpenExplorer） | 与 RKNN 是两套后端；只借架构不借工具链 |
| `bpu_control_robot.py` 原样上板 | 板端 `import torch`，违反我方"板端禁 torch"红线；须重写为 numpy + onnxruntime/rknn |
| IB_Robot 的 ROS2/colcon 结构与 `third_party/` | 引入重依赖；`third_party` 含 NVIDIA 非商业许可 wheel |
| so101-nexus 整库 / 其 MuJoCo 仿真标定 | 需 Python≥3.12+mujoco+torch；仿真标定非真机替代品 |
| 现在就把 JPEG 序列换成硬件编码 mp4 | 与 LeRobot v3 视频规范契合度未验证 + 瓶颈不在编码（先做旁路实验） |
| 纯 ctypes 全量绑定 librga | 无编译期保护；先确认现有 OpenCV 是否已带 RGA 后端 |
| 把"50 条 episode"当作已验证经验值 | IB_Robot 未公布任何条数建议；**唯一支持性数据点是 lerobot 官方 `docs/source/act.mdx:29`"50 条演示常能出效果"**（经验说法，非保证）→ 仍应按帧质量指标自检 |
| 照抄 lerobot-rk3588 快照的依赖与跨机通道 | 该快照版本比我们旧（0.4.0 发布前）、`datasets/` 缺失；其跨机 TCP 动作通道无过期判定与时间戳对齐（会引入标签噪声）；`pyproject.toml:122` 本地路径 pin 会让 pip 失败 |

## 5. 待办（板端只读核查，决定 §2.6/§2.7 能否立即走通）

板端目前网络不可达（连续 ssh 超时），以下命令待板端恢复后执行：

```bash
cat /proc/version                                   # 是否 Rockchip BSP 内核（5.10/6.1）
ls -l /dev/rga /dev/mpp_service /dev/dma_heap        # RGA/MPP 设备节点是否存在
ffmpeg -hide_banner -encoders | grep -i rkmpp        # 是否已有 rkmpp 硬件编码器
ldconfig -p | grep -E 'librga|rockchip_mpp'          # librga / mpp 库是否就绪
python3 -c "import cv2;print(cv2.getBuildInformation())" | grep -iE 'rga|mpp|ffmpeg'
```

判读：若 `/dev/rga` 与 `librga` 就绪且 OpenCV 已带 RGA 后端 → 预处理卸载可低成本落地（采纳 #1 的加速手段）；
若 ffmpeg 已有 `rkmpp` 编码器 → 视频落盘旁路实验可行。

## 6. 待验证问题

1. 我方 30Hz / 单条 20s / 双相机 640×480 与对标物（20Hz / ≤90s / 2 相机 480×640）差异是否影响 ACT 效果？
   → M2 用同一批数据做 30Hz vs 20Hz 下采样对比。
2. 我方夹爪用弧度（非官方 `RANGE_0_100`）是否影响 ACT 收敛？→ M2 观察夹爪维度的 loss 与动作范围。
3. RKNN fp16 下 ACT（我方 chunk 大小待定）端到端延迟能否满足 30Hz 采集回路之外的控制需求？→ M3 实测。
