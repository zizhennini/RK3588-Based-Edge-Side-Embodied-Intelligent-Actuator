# RK3588-EIA 系统测试方案

> 覆盖模块测试 → 阶段测试 → 全阶段测试 → 总测试四级。
> **架构基线**: `refactor/architecture-stage1` v0.5.0（架构债清理后）；模块命名与
> `docs/architecture.md` 一致（SO101Arm / CameraManager / GraspPipeline / runtime 等）。
> **环境与版本矩阵**: 见 `docs/deploy_guide.md`（两端 Python/numpy/torch 版本红线）。
> 旧版方案中以 `ArmController`（vla 遗留层）为对象的条目已迁移到 `hardware/arm.py SO101Arm`；
> 遗留脚本路径（scripts/replay_traj.py 等走 vla/ 兼容层）单独标注。

---

## 测试环境

### 板端（硬件在环，L1-L4 层测试）

| 项目 | 配置 |
|------|------|
| 硬件 | RK3588 (8GB) + D435i + SO-ARM101 主臂(ttyACM1) + SO-ARM101 从臂(ttyACM0) + 北通蝙蝠4 |
| 软件 | conda env `rk3588` (Python 3.10)，依赖见 `requirements.txt`；rknn-toolkit-lite2 2.3.2 (本地 whl)；ffmpeg-rkmpp (系统包) |
| 工作目录 | `/home/elf/work/rk3588-eia` |
| 前置激活 | `source /home/elf/work/miniconda/etc/profile.d/conda.sh && conda activate rk3588` |

### PC 端 WSL2（无硬件单测 / 导出工具）

| 项目 | 配置 |
|------|------|
| 软件 | conda env `rk3588` (Python 3.12)，依赖见 `requirements-dev.txt`；LeRobot v0.6.1（仅数据集/训练） |
| 可测范围 | `tests/test_kinematics.py`、`runtime/shared_frame.py` 自检、ONNX 导出工具、GraspPipeline 离线逻辑（mock 观测） |
| 不可测 | 串口/相机/NPU/RKLLM/硬件编码（无对应硬件与 aarch64 运行库） |

### 测试状态标记

| 标记 | 含义 |
|------|------|
| 🔴 未通过 | 测试失败，需修复 |
| 🟡 部分通过 | 功能可用但有缺陷 |
| 🟢 通过 | 测试通过（标注验证环境） |
| ⬜ 未测试 | 尚未执行 |
| ⏸ 阻塞 | 依赖项未就绪（当前主要为 `models/act/`、`models/ggcnn/` ONNX 资产未部署，见 P1-P3 前置项） |

---

## 第一部分：模块测试

### M1. 机械臂控制 (hardware/arm.py — SO101Arm, HardwareModule)

#### M1.1 串口通信

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M1.1.1 | 串口连接 | `python3 -c "from hardware.arm import SO101Arm; a=SO101Arm.get_instance(); a.connect(); print(a.is_available); a.close()"` | `True`，无串口异常 | ⬜ |
| M1.1.2 | 关节位置读取 (SYNC_READ) | `python3 -c "from hardware.arm import SO101Arm; a=SO101Arm.get_instance(); a.connect(); print(a.read_positions()); a.close()"` | 输出 6 个浮点数（弧度） | ⬜ |
| M1.1.3 | 急停/力矩释放 | `python3 -c "from hardware.arm import SO101Arm; a=SO101Arm.get_instance(); a.connect(); a.emergency_stop(); a.close()"` | 舵机力矩关闭可自由转动 | ⬜ |
| M1.1.4 | 串口崩溃恢复 | 通信中拔掉 USB → 重新插回，观察 `_reset_serial()` 重试 | 3 次重试内恢复或抛出明确异常；单例经 `close()` 清除后可重建 | ⬜ |

#### M1.2 关节控制

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M1.2.1 | 批量写入 (SYNC_WRITE) | `python3 -c "from hardware.arm import SO101Arm; import numpy as np, time; a=SO101Arm.get_instance(); a.connect(); a.write_positions(np.array([0,0,-1.2,0,0,1.5])); time.sleep(2); a.close()"` | 机械臂移动到目标位 | ⬜ |
| M1.2.2 | 平滑归零 | `python3 -c "from hardware.arm import SO101Arm; a=SO101Arm.get_instance(); a.connect(); a.home(); a.close()"` | 默认 50 步平滑过渡到 HOME 位 | ⬜ |
| M1.2.3 | 关节限位钳制 | 写入超限角度（结合 `Kinematics.clamp_joint`） | 脉冲被钳制到标定 range | ⬜ |
| M1.2.4 | 标定加载/保存 | `python3 -c "from hardware.arm import SO101Arm; a=SO101Arm.get_instance(); print(a._load_calibration('./config/calibration.json') is not None)"` | True；`save_calibration()` 可写回 | ⬜ |

#### M1.3 夹爪控制

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M1.3.1 | 夹爪开合（布尔） | `python3 -c "from hardware.arm import SO101Arm; import time; a=SO101Arm.get_instance(); a.connect(); a.gripper(True); time.sleep(1); a.gripper(False); a.close()"` | 张开→闭合 | ⬜ |
| M1.3.2 | 夹爪宽度（米） | `a.gripper_width(0.04)` | 开度约 4cm | ⬜ |

#### M1.4 IK 运动（依赖注入 Kinematics — v0.5.0 架构）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M1.4.1 | 未注入时报错 | `SO101Arm.get_instance()` 后直接 `move_to(...)`（不 set_kinematics） | 抛 RuntimeError，提示注入 Kinematics（硬件层不反向依赖 policy） | ⬜ |
| M1.4.2 | 可达位置运动 | `python3 -c "from hardware.arm import SO101Arm; from policy.kinematics import Kinematics; import time; a=SO101Arm.get_instance(); a.set_kinematics(Kinematics()); a.connect(); a.move_to(0.25,0.0,0.15); time.sleep(2); a.close()"` | 末端到达 (0.25, 0, 0.15) | ⬜ |
| M1.4.3 | 不可达位置钳制 | 同上，目标 (0.5, 0.5, 0.5) | 钳制到工作空间边界后运动，无异常 | ⬜ |
| M1.4.4 | 真机坐标系对齐 | 低速执行 M1.4.2，实测末端与目标偏差 | 偏差 < 10mm（验证 URDF 系限位 vs XLeRobot 舵机系偏移 θ1≈14.0°/θ2≈16.2° 的一致性 — v9.1 遗留风险） | ⬜ |

#### M1.5 HardwareModule 接口符合性（v0.5.0 新增）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M1.5.1 | 接口方法齐备 | `python3 -c "from hardware.arm import SO101Arm; from hardware.interfaces import HardwareModule; print(issubclass(SO101Arm, HardwareModule))"` | True（抽象方法未全实现会在此处实例化失败） | 🟢 静态验证（Windows py_compile+AST；板端 ⬜） |
| M1.5.2 | 生命周期 | `setup({})` → `start()` → `is_available` → `stop()` | start=connect、stop=disconnect、is_available 反映连接态 | ⬜ |
| M1.5.3 | 失败策略 | `a.on_failure()` | 返回 `"abort"`（机械臂为硬依赖） | ⬜ |
| M1.5.4 | execute/get_observation | `a.execute(Action(positions=..., gripper=0.5, execution_time=1.0))`；`a.get_observation()` | execute 返回 True；observation.state 为 6 关节角，rgb/depth=None（由 System 与相机观测合并） | ⬜ |

### M2. 语音模块 (voice/ — va.py 入口)

> v0.4.0 起 va.py 走 `voice/cli.py` 新栈；`voice_assistant/` 为遗留实现（模型文件仍在
> `voice_assistant/voice_assistant/models/`，由 `voice/config/default.yaml` 引用）。

#### M2.1 KWS 唤醒词

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M2.1.1 | KWS 模型加载 | `python3 -c "from voice.config import load_config; from voice.wake import SherpaKeywordWake; w=SherpaKeywordWake(load_config())"` | 无异常 | 🟢 已验证（旧栈同模型；新栈 ⬜ 复跑） |
| M2.1.2 | 唤醒词检测 | `python3 va.py listen --wake-mode kws --seconds 4 --no-speak` | 说"小咪"后唤醒并录音 | 🟢 已验证 |

#### M2.2 ASR 语音识别

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M2.2.1 | ASR 模型加载 | `python3 -c "from voice.config import load_config; from voice.asr import SherpaAsr; SherpaAsr(load_config())"` | 无异常 | 🟢 已验证（旧栈；新栈 ⬜ 复跑） |
| M2.2.2 | WAV 文件识别 | `python3 va.py stt /tmp/test.wav` | 输出识别文字 | ⬜ |
| M2.2.3 | 实时录音识别 | `python3 va.py once --seconds 4 --no-speak` | 录音→ASR→打印文字 | 🟢 已验证 |

#### M2.3 TTS 语音合成

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M2.3.1 | TTS 模型加载 | `python3 -c "from voice.config import load_config; from voice.tts import SherpaTts; SherpaTts(load_config())"` | 无异常 | 🟢 已验证（旧栈；新栈 ⬜ 复跑） |
| M2.3.2 | 流式 TTS 播报 | `python3 va.py tts-stream "你好，我是RK3588智能助手"` | 喇叭播放语音 | 🟢 已验证 |
| M2.3.3 | 长文本播报(>220字) | `python3 va.py tts-stream "..."`（长文字，经 `text_clean.sanitize_tts_text`） | 按句分段流式播放 | ⬜ |

#### M2.4 全链路测试

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M2.4.1 | 语音→VLM→TTS | `python3 va.py once --seconds 4` | 录音→ASR→Qwen→TTS 完整流程 | 🟢 已验证 |
| M2.4.2 | 文字→VLM→TTS | `python3 va.py ask "介绍一下自己"` | 文字→Qwen→TTS 播报 | 🟢 已验证 |
| M2.4.3 | 唤醒→录音→回答 | `python3 va.py listen-forever` | 说"小咪"→唤醒→录音→回答 | 🟢 已验证 |
| M2.4.4 | 拍照→VLM 分析 | `python3 va.py ask "画面中有什么" --no-speak` | 拍照→Qwen 分析→打印描述 | 🟢 已验证 |
| M2.4.5 | 意图路由 | `python3 -c "from voice.intent import IntentRouter; ..."`（按语音指令样本） | 动作/问答/控制意图正确分类 | ⬜ |

### M3. 轨迹模块（motion_library，遗留脚本路径）

> 本组脚本走 `vla/` 兼容层（ArmController），第三阶段评估迁移到 SO101Arm 后再改对象。

#### M3.1 动作库管理

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M3.1.1 | 动作列表 | `python3 scripts/record_trajectory.py list` | 列出已注册动作 | ⬜ |
| M3.1.2 | 轨迹信息查看 | `python3 scripts/record_trajectory.py inspect greeting_01.json` | 显示元数据 | ⬜ |
| M3.1.3 | 动作删除 | `python3 scripts/record_trajectory.py delete test` | 删除指定动作 | ⬜ |

#### M3.2 轨迹平滑

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M3.2.1 | 单文件平滑 | `python3 scripts/smooth_traj.py motion_library/grasp/grasp_01.json --output /tmp/test_smoothed.json` | 生成 smoothed 文件 | ⬜ |
| M3.2.2 | 批量平滑 | `python3 scripts/smooth_trajectory.py` | 处理所有未平滑轨迹 | ⬜ |
| M3.2.3 | 速度限幅验证 | 检查平滑后逐帧角度差 | 均不超过 MAX_DELTA | ⬜ |

#### M3.3 轨迹回放

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M3.3.1 | 基本回放 | `python3 scripts/replay_traj.py motion_library/greeting_01.json --port /dev/ttyACM0` | 从臂执行 greeting 动作 | ⬜ |
| M3.3.2 | 轨迹合法性校验 | `python3 scripts/smooth_trajectory.py motion_library/grasp/grasp_01.json` | 输出校验结果 | ⬜ |

### M4. 任务控制器（scripts/task_controller.py，遗留脚本路径）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M4.1 | task_library 自动创建 | `python3 -c "from scripts.task_controller import TaskController; tc=TaskController(None); print(tc.list_tasks())"` | 输出空列表，task_library 目录已创建 | ⬜ |
| M4.2 | 创建任务 | 编程调用 `tc.create_task()` | 返回合法 JSON | ⬜ |
| M4.3 | 保存+加载 | `tc.save_task(task); tc.load_task(name)` | 保存后加载内容一致 | ⬜ |
| M4.4 | 安全校验(假数据) | `tc.verify_safety(invalid_task)` | 返回警告列表 | ⬜ |
| M4.5 | 模拟运行 | `python3 scripts/task_controller.py run <name> --dry-run` | 打印各步骤但不驱动臂 | ⬜ |

### M5. 录像模块 (hardware/encoder.py H264Encoder + scripts/recorder.py)

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M5.1 | 硬件编码可用 | `ffmpeg -f lavfi -i color=c=black:s=640x480:d=2 -c:v h264_rkmpp -b:v 5M -y /tmp/test.mp4` | 输出 2 秒 MP4，无报错 | ✅ 已验证 |
| M5.2 | RGA 缩放可用 | `ffmpeg -f lavfi -i color=c=black:s=1280x720:d=2 -vf scale_rkrga=640:480 -c:v h264_rkmpp -y /tmp/test_rga.mp4` | 输出 MP4，无报错 | ⬜ |
| M5.3 | H264Encoder 接口 | `python3 -c "from hardware.encoder import H264Encoder; e=H264Encoder(); e.setup({}); print(e.is_available)"` | ffmpeg+h264_rkmpp 探测成功则 True；`on_failure()=="skip"` | ⬜ |
| M5.4 | H264Encoder 写帧 | `e.open(out_path)` → 循环 `e.write_frame(rgb)` → `e.close()`；检查 `frames_written()`/`is_recording()` | 生成可播放 MP4，帧数一致 | ⬜ |
| M5.5 | menu/main 录像模式 | `python3 menu.py` → record 模式（或 `main.py --mode record`） | run_record 走 H264Encoder 链路，正常启停 | ⬜ |
| M5.6 | D435i 实时录像（遗留） | `python3 scripts/recorder.py --duration 5` | 录制 5 秒 MP4 到 recordings/ | ⬜ |
| M5.7 | OSD 叠加录像 | `python3 scripts/recorder.py --duration 5 --osd "实验记录"` | 视频画面叠加文字 | ⬜ |
| M5.8 | 录像启停 | 运行后按 Ctrl+C | 正常停止并保存文件 | ⬜ |

### M6. 感知与抓取管线 (perception/ + policy/grasp_pipeline.py)

#### M6.1 VLMPerception（RKLLM 子进程）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M6.1.1 | 模块可用性 | `python3 -c "from perception.vlm import VLMPerception; v=VLMPerception(); v.setup({}); print(v.is_available)"` | True（模型已部署 `models/vlm/Qwen3.5-0.8B`；demo 需 `chmod +x`） | ⬜ |
| M6.1.2 | 目标检测 | 相机取 obs 后 `v.detect(obs, "红色杯子")` | 返回 dict 含 bbox（[0,1] 归一化）/label | ⬜ |
| M6.1.3 | 闲置自动卸载 | 推理后等待 >30s（`VLM_IDLE_UNLOAD_TIMEOUT`），观察 `v.is_loaded()`/`v.idle_seconds()` | 自动卸载释放 ~900MB | ⬜ |
| M6.1.4 | 统计信息 | `v.stats()` | 输出推理次数/延迟摘要 | ⬜ |

#### M6.2 GGCNNDetector（⏸ 阻塞：models/ggcnn/*.onnx 未部署）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M6.2.1 | 模型缺失降级 | `d=GGCNNDetector(); d.setup({...}); print(d.is_available)`（不放模型文件） | False，无异常抛出；`on_failure()=="skip"` | ⬜ |
| M6.2.2 | 抓取检测 | 部署 ONNX 后 `d.detect(obs, roi_bbox)` | 返回 angle/width/quality/center_px | ⏸ 模型资产 |
| M6.2.3 | 推理延迟 | 计时 `detect()` | 30-50ms（ONNX Runtime CPU 预估） | ⏸ 模型资产 |

#### M6.3 ACTPolicy（⏸ 阻塞：models/act/*.onnx 未部署）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M6.3.1 | 模型缺失降级 | `p=ACTPolicy(); p.setup({...}); print(p.is_available, p.setup_error)` | False + 明确错误信息，无异常 | ⬜ |
| M6.3.2 | chunk 推理 | 部署后 `p.predict(obs)`；`p.buffer_remaining()`/`p.chunk_size()`/`p.n_action_steps()` | 返回 Action；chunk 缓冲逐步消耗 | ⏸ 模型资产 |
| M6.3.3 | 推理延迟 | `p.last_infer_ms()` | 70-120ms（RK3588 A76 预算，v9 决策） | ⏸ 模型资产 |
| M6.3.4 | 缓冲重置 | `p.reset_buffer()` | 缓冲清空，下次 predict 重新推理 | ⏸ 模型资产 |

#### M6.4 GraspPipeline（双模式 + 降级链）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M6.4.1 | 构造与注入 | `GraspPipeline(arm, camera, vlm=..., ggcnn=..., act_policy=..., kinematics=...)`（均可为 None 支持降级） | 构造成功；`is_available` 反映依赖状态 | ⬜ |
| M6.4.2 | 模式切换 | `pipeline.use_act` 属性读写 | ACT 可用时 True；不可用自动回落 False | ⏸ 模型资产 |
| M6.4.3 | 降级链 | 依次移除 ggcnn/vlm 依赖构造管线，调 `execute_grasp("目标")` | GGCNN→仅 VLM(bbox 中心+深度)→失败，三级降级路径正确 | ⬜（仅 VLM 级可先测） |
| M6.4.4 | 三段式执行 | `execute_grasp(target_desc, verify=True)` | pre_grasp→grasp→lift 轨迹；人工确认门生效 | ⏸ 模型资产 |
| M6.4.5 | 相机参数单一源 | 检查 `GraspPipeline.CAM_POSITION/CAM_FX` 与 `settings.CAMERA_POSITION/CAMERA_MATRIX` 一致 | 完全一致（v0.5.0 债 2 修复验证） | 🟢 静态验证（Windows；板端 ⬜） |

#### M6.5 ColorLocator（降级定位）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M6.5.1 | 颜色定位 | `loc=ColorLocator(settings.CAMERA_MATRIX); loc.locate(rgb, depth, "红色")` | 返回 3D 坐标或 None | ⬜ |
| M6.5.2 | 主色识别 | `loc.dominant_color(rgb)` | 输出颜色名 + mask | ⬜ |

#### M6.6 遗留 VLM 抓取脚本（vla/ 兼容层）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M6.6.1 | 像素坐标解析 | `python3 scripts/vlm_grasp.py "红色杯子" --teach-trajectory grasp_01.json --ref-cx 320 --ref-cy 240 --ref-z 0.3 --dry-run` | Qwen 输出坐标+打印偏移量 | ⬜ |
| M6.6.2 | 深度 3D 解算 | 同上（检查输出坐标） | 输出合理的 3D 坐标 | ⬜ |
| M6.6.3 | 安全校验 | 带超限偏移量的假数据 | 校验未通过，提示超限 | ⬜ |

### M7. 安全防护 (config/safety.py + hardware/safety.py + config/teaching.py)

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M7.1 | 深度避障状态机 | `mon=SafetyMonitor(arm, camera); mon.start()` 后手动遮挡 D435i | 状态 safe→warn→stop；`mon.is_emergency()` 变 True | ⬜ |
| M7.2 | 电流读取 | 运行中 `CurrentMonitor.read_current(1)`（config/safety.py） | 输出合理电流值（几十~几百 mA） | ⬜ |
| M7.3 | 急停触发/清除 | `mon.trigger_emergency_stop()` → `mon.clear_emergency()` | 臂停止；清除后可恢复 | ⬜ |
| M7.4 | 教学模式加载 | `python3 -c "from config.teaching import TeachingMode; m=TeachingMode('student','manual')"` | 无异常 | ⬜ |
| M7.5 | 模式切换(教师) | `python3 -c "from config.teaching import TeachingMode; m=TeachingMode('teacher','ai'); print(m.get_label())"` | 输出"AI自主抓取模式" | ⬜ |
| M7.6 | 功能可见性 | `python3 -c "from config.teaching import TeachingMode; m=TeachingMode('student','manual'); print(m.is_enabled('vlm'))"` | 输出 False | ⬜ |

### M8. 相机 (hardware/camera_d435i.py — CameraManager, HardwareModule)（v0.5.0 新增）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M8.1 | 采集启动+预热 | `python3 -c "from hardware.camera_d435i import CameraManager; import time; c=CameraManager(); c.start(); time.sleep(4); print(c.is_running, c.is_warmed_up, c.has_frame); c.stop()"` | 三项均 True（预热默认 3s） | ⬜ |
| M8.2 | 帧读取 | `c.get_frame()` / `c.get_rgb()` / `c.get_depth()` | (640,480,3) uint8 + (640,480) float32 + 时间戳；深拷贝无脏数据 | ⬜ |
| M8.3 | 接口符合性 | `issubclass(CameraManager, HardwareModule)`；`c.on_failure()`；`c.get_observation()` | True；`"skip"`（可降级）；Observation 含 rgb/depth、state=None | 🟢 接口一致性已验证（Windows 无 pyrealsense2 环境，实例化+方法齐备；帧链路板端 ⬜） |
| M8.4 | setup 配置 | `c.setup({'width':320,'height':240,'fps':15})` | 参数生效 | 🟢 已验证（Windows） |
| M8.5 | 采集线程绑核 | 运行中检查采集线程亲和性 | 绑定 A55 核 {0,1}（`os.sched_setaffinity`） | ⬜ |

### M9. 运动学 (policy/kinematics.py)（v0.5.0 新增）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M9.1 | 单元测试全量 | `python3 tests/test_kinematics.py` | 6/6 通过（FK/IK 互逆 <1mm、工作空间钳制、不可达钳制、关节限位） | 🟢 6/6（Windows；WSL2/板端 ⬜ 复跑） |
| M9.2 | FK/IK 往返 | `k.inverse_kinematics(xyz)` → `k.forward_kinematics(joints)` | 往返误差 <1mm | 🟢 同上 |
| M9.3 | 工作空间钳制 | `k.clamp_workspace(np.array([0.5,0,0]))` | `[0.45, 0, 0]`（x 钳到上限） | 🟢 同上 |
| M9.4 | 板端 numpy 兼容 | 板端 conda 环境重跑 M9.1 | 6/6（验证 numpy<2.0 约束下数值一致） | ⬜ |

### M10. 运行时 (runtime/ — opt-in 子进程+共享内存)（v0.5.0 新增）

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M10.1 | SharedFrameBuffer 自检 | `python3 runtime/shared_frame.py` | 往返测试通过（seqlock/numpy 视图/完整性） | 🟢 通过（Windows 含跨句柄 attach；WSL2/板端 ⬜ 复跑） |
| M10.2 | 子进程 spawn+绑核 | `SubprocessWorker` 子类在板端启动，检查子进程亲和性 | 子进程内 `rebind_affinity` 生效（不继承父进程） | ⏸ 板端 |
| M10.3 | InferenceWorker 推理 | GGCNN 模型子进程内构建 + predict 往返 | Queue 返回 Action；帧走共享内存零拷贝 | ⏸ 模型资产+板端 |
| M10.4 | 全编排 (T4.2) | `USE_SUBPROCESS_RUNTIME=True` 跑 main.py autonomous | 相机/推理分进程，延迟与 GIL 抖动对比进程内路径 | ⏸ 风险 R2，需板端实测后决定默认值 |

---

## 第二部分：阶段测试

### P0. 集成验证前置项（v0.5.0 评估新增）

| 编号 | 前置项 | 环境 | 状态 |
|------|--------|------|------|
| P0.1 | GGCNN Cornell 预训练权重下载 → `tools/export_ggcnn_onnx.py` 导出（opset 12） | PC WSL2 | ⬜（v0.3.0 待办遗留） |
| P0.2 | ACT checkpoint（safetensors）就位 → `tools/export_act_onnx.py` 导出（opset 14+）；依赖含 safetensors（requirements-dev.txt 已补） | PC WSL2 | ⬜ |
| P0.3 | 导出产物传板 `models/ggcnn/`、`models/act/`（gitignored，不随仓库分发） | 板端 | ⬜ |
| P0.4 | `chmod +x models/vlm/Qwen3.5-0.8B/demo` 与 imgenc（v0.3.0 待办遗留） | 板端 | ⬜ |
| P0.5 | 两端版本矩阵核对（deploy_guide.md，numpy<2.0 / Python 3.10 vs 3.12 红线） | 两端 | ⬜ |

### P1. 第一期（基础功能）

| 编号 | 测试项 | 测试步骤 | 预期 | 涉及模块 | 状态 |
|------|--------|---------|------|---------|------|
| P1.1 | 遥操作录制 | 板端自研工具链：标定 `tools/calibrate_arm.py --port /dev/ttyACM0`；跟随录制 `scripts/lerobot-record-lite --follow`（hardware.teleop.TeleopPair：30Hz leader→follower + G3 限幅 + 起步平滑对齐）或只录不跟随（默认，纯 pyserial）；录制 JSON 传 PC 后 `scripts/json_to_lerobot.py --format npz|lerobot` 转换（lerobot 0.4.4 env 已验收）；总线排查 `tools/feetech_scan.py --port ... [--scan-all]`；PC 端带硬件时亦可 `scripts/teleop_record.py`（需 pip lerobot>=0.6） | 拖拽主臂 15 秒，从臂跟随，生成 JSON | M1 | ⬜ |
| P1.2 | 平滑处理 | `python3 scripts/develop_motion.py greeting --no_record` | 处理 raw 文件→smoothed | M3.2 | ⬜ |
| P1.3 | 回放验证 | `python3 scripts/replay_traj.py motion_library/greeting_01.json --port /dev/ttyACM0` | 从臂执行轨迹 | M3.3, M1 | ⬜ |
| P1.4 | 语音触发回放 | 运行 `va.py listen-forever`，说"你好" | 关键词匹配→回放 greeting | M2 | ⬜ |
| P1.5 | 语音循环回放 | 说"循环回放你好" | 从臂回放 3 次 | M2, M3.3 | ⬜ |
| P1.6 | 语音暂停/继续 | 回放中说"暂停"→"继续" | 暂停→恢复 | M2, M3.3 | ⬜ |
| P1.7 | 语音归零 | 说"归零" | 从臂回到 HOME 位 | M2, M1.2 | ⬜ |
| P1.8 | 语音动作列表 | 说"有什么动作" | TTS 播报动作列表 | M2, M3.1 | ⬜ |

### P2. 第二期（核心功能）

| 编号 | 测试项 | 测试步骤 | 预期 | 涉及模块 | 状态 |
|------|--------|---------|------|---------|------|
| P2.1 | 任务控制器模拟 | `python3 scripts/task_controller.py list` | 列出任务(或空) | M4 | ⬜ |
| P2.2 | 创建组合任务 | 创建 2 段轨迹的组合任务→保存 | 存在 task JSON 文件 | M4 | ⬜ |
| P2.3 | 组合任务模拟运行 | `python3 scripts/task_controller.py run <name> --dry-run` | 依次打印各步骤 | M4 | ⬜ |
| P2.4 | 硬件录像 | `python3 scripts/recorder.py --duration 10` 或 menu record 模式 | 录制 10 秒 MP4 | M5 | ⬜ |
| P2.5 | OSD 录像 | `python3 scripts/recorder.py --duration 5 --osd "测试"` | 带文字叠加的 MP4 | M5 | ⬜ |
| P2.6 | VLM 抓取模拟（遗留） | `vlm_grasp.py --dry-run` | 打印偏移量 | M6.6 | ⬜ |
| P2.7 | GGCNN 单模块推理 | 部署 ONNX 后跑 M6.2.2/M6.2.3 | 检测输出 + 延迟达标 | M6.2 | ⏸ P0.1/P0.3 |
| P2.8 | ACT 单模块推理 | 部署 ONNX 后跑 M6.3.2/M6.3.3 | chunk 推理 + 延迟达标 | M6.3 | ⏸ P0.2/P0.3 |
| P2.9 | GraspPipeline 三段式端到端 | `execute_grasp("红色杯子", verify=True)`（GGCNN 模式） | 感知→规划→执行完整链路 | M6.4, M1, M8 | ⏸ P0.1/P0.3 |
| P2.10 | ACT/GGCNN 双模式切换 | 切换 `use_act`，对比两种模式抓取 | 模式切换与自动降级正确 | M6.4 | ⏸ P0.2/P0.3 |

### P3. 第三期（安全+教学）

| 编号 | 测试项 | 测试步骤 | 预期 | 涉及模块 | 状态 |
|------|--------|---------|------|---------|------|
| P3.1 | 深度避障联调 | 运行 main.py 时手动遮挡 D435i | 从臂减速→停止 | M7.1 | ⬜ |
| P3.2 | 电流防护联调 | 运行中手动阻挡机械臂 | 电流上升→柔顺回退→停机 | M7.2, M7.3 | ⬜ |
| P3.3 | 教学模式切换 | 教师角色切换三种模式 | 功能可见性正确变更 | M7.4-M7.6 | ⬜ |
| P3.4 | 一键还原 | 教师执行还原 | 动作库/任务库/录像清空 | M3.1, M4 | ⬜ |

---

## 第三部分：全阶段测试

跨功能模块的集成测试，验证端到端流程。

### I1. 语音→轨迹回放全链路

| 步骤 | 操作 | 预期 | 涉及模块 |
|------|------|------|---------|
| 1 | 启动 `python3 va.py listen-forever` | 进入唤醒监听状态 | M2.1 |
| 2 | 说"小咪" | 唤醒成功 | M2.1 |
| 3 | 说"你好" | ASR 识别→意图路由→motion 任务入队 | M2.2, M2.4.5 |
| 4 | 从臂执行 greeting 轨迹 | 机械臂平滑运动 | M3.3, M1 |
| 5 | TTS 播报"动作 greeting 执行完成" | 喇叭播报 | M2.3 |
| 6 | 说"暂停" | 回放暂停 | M3.3 |
| 7 | 说"继续" | 回放恢复 | M3.3 |
| 8 | 说"归零" | 机械臂平滑回 HOME | M1.2 |

### I2. 录制→平滑→入库→回放全链路

| 步骤 | 操作 | 预期 | 涉及模块 |
|------|------|------|---------|
| 1 | `python3 scripts/teleop_record.py --episode_time_s 15`（或 lerobot-record-lite） | 拖拽主臂追帧录制 | P1.1, M1 |
| 2 | 检查生成的 JSON 文件 | 包含 frames 和 joints 数据 | P1.1 |
| 3 | `python3 scripts/develop_motion.py <name> --no_record` | 平滑处理+入库 | M3.2 |
| 4 | `python3 scripts/record_trajectory.py list` | 动作库中可见新动作 | M3.1 |
| 5 | `python3 scripts/replay_traj.py ...` 或说动作名 | 从臂执行 | M3.3 |

### I3. 视觉→VLM→抓取全链路（遗留 vlm_grasp 路径）

| 步骤 | 操作 | 预期 | 涉及模块 |
|------|------|------|---------|
| 1 | 放置目标物体于示教基准位 | 机械臂可到达 | — |
| 2 | 运行 `python3 scripts/vlm_grasp.py "目标" --teach-trajectory ...` | Qwen 识别+深度定位 | M6.6, M6.1 |
| 3 | 安全校验通过 | 打印校验结果 | M7 |
| 4 | 人工确认(y) | 机械臂执行修正后轨迹 | M1 |
| 5 | 抓取完成，TTS 播报 | 喇叭播报 | M2.3 |

### I4. 录像+VLM 资源互斥

| 步骤 | 操作 | 预期 | 涉及模块 |
|------|------|------|---------|
| 1 | 开始录像 | ffmpeg/H264Encoder 启动 | M5 |
| 2 | 尝试启动 VLM 问答 | VLM 推理正常（内存预算内共存；MemoryMonitor 仲裁） | M6.1, M7 |
| 3 | 停止录像 | 资源释放 | M5 |

### I5. GraspPipeline GGCNN 三段式端到端（v0.5.0 新增）

| 步骤 | 操作 | 预期 | 涉及模块 |
|------|------|------|---------|
| 1 | System 初始化（main.py autonomous 模式） | 依赖注入完成：SO101Arm(含 Kinematics)/CameraManager/VLM/GGCNN | M1.4, M8, M6.1, M6.2 |
| 2 | 相机观测 → VLM ROI → GGCNN 抓取位姿 | detect 链路输出 GraspCandidate | M6.1, M6.2 |
| 3 | IK 解算三段轨迹（pre_grasp→grasp→lift） | 工作空间钳制生效 | M9, M6.4 |
| 4 | SO101Arm 执行 + 夹爪闭合 | 抓起了目标物 | M1.2, M1.3 |
| 5 | SafetyMonitor 全程监控 | 无急停触发（或正确触发） | M7.1 |

**状态**: ⏸ 阻塞于 P0.1/P0.3（GGCNN ONNX 部署）

### I6. ACT 双模式切换与降级（v0.5.0 新增）

| 步骤 | 操作 | 预期 | 涉及模块 |
|------|------|------|---------|
| 1 | ACT 模型部署，`pipeline.use_act=True` | predict 走 ACT chunk，50Hz 逐步执行 | M6.3, M6.4 |
| 2 | 移除 ACT 模型重启 | 自动降级 GGCNN 三段式，`use_act` 回落 False | M6.4.2 |
| 3 | 再移除 GGCNN | 降级仅 VLM（bbox 中心+深度） | M6.4.3 |
| 4 | 全部移除 | execute_grasp 返回失败 TaskResult，系统不崩溃 | M6.4.3 |

**状态**: ⏸ 阻塞于 P0.2/P0.3（ACT ONNX 部署）

---

## 第四部分：总测试

全系统端到端验收测试，所有模块联合运行。

### S1. 正常场景

| 编号 | 场景 | 步骤 | 预期 | 状态 |
|------|------|------|------|------|
| S1.1 | 语音唤醒→视觉分析→播报 | 说"小咪"唤醒→"画面中有什么" | 拍照→VLM 分析→TTS 播报→30s 后 VLM 自动卸载 | ⬜ |
| S1.2 | 语音触发动作回放 | 说"小咪"→"你好" | 从臂执行 greeting→TTS 播报完成 | ⬜ |
| S1.3 | 多指令排队 | "小咪"→"你好"→"再见" | 两条指令串行执行 | ⬜ |
| S1.4 | 指令中断 | 回放中说"停止" | 机械臂急停，队列清空 | ⬜ |
| S1.5 | 自主抓取（GraspPipeline） | main.py autonomous 模式下达抓取指令 | I5 或 I6 全链路成功 | ⏸ 模型资产 |
| S1.6 | 实验录像 | 录像 5 分钟 | 正常录制，OSD 信息可见 | ⬜ |

### S2. 异常场景

| 编号 | 场景 | 步骤 | 预期 | 状态 |
|------|------|------|------|------|
| S2.1 | 深度遮挡 | 机械臂运行时遮挡 D435i | 减速→停止 | ⬜ |
| S2.2 | 机械臂碰撞 | 手动阻挡运动中的机械臂 | 电流上升→急停→回退 | ⬜ |
| S2.3 | VLM 识别失败 | 画面中无指定目标 | detect 返回失败 dict，提示"未检测到目标" | ⬜ |
| S2.4 | ASR 识别失败 | 环境噪音过大 | 不触发指令或提示无法识别 | ⬜ |
| S2.5 | 串口断开 | 运行中拔掉机械臂 USB | SO101Arm `_safe_write` 重试→`_reset_serial` 恢复；不可恢复时 on_failure="abort" 安全停机 | ⬜ |
| S2.6 | 内存不足 | 连续 VLM 推理不卸载 | MemoryMonitor 触发阈值→自动回收（VLM 闲置卸载兜底） | ⬜ |
| S2.7 | ffmpeg 编码失败 | 磁盘空间不足 | H264Encoder on_failure="skip"，录像停止并提示，主流程不崩 | ⬜ |
| S2.8 | 相机断连 | 运行中拔掉 D435i USB | CameraManager on_failure="skip"，系统降级无视觉模式 | ⬜ |

### S3. 学生模式权限验证

| 编号 | 场景 | 步骤 | 预期 | 状态 |
|------|------|------|------|------|
| S3.1 | 学生模式功能限制 | `--role student --mode manual` | VLM/录像不可用 | ⬜ |
| S3.2 | 教师模式全功能 | `--role teacher` | 所有功能可用，可切换模式 | ⬜ |
| S3.3 | 教师一键还原 | 教师执行还原命令 | 动作库/任务库/录像清空，index 重置 | ⬜ |

### S4. 性能基准

| 编号 | 指标 | 测试方法 | 可接受阈值 | 状态 |
|------|------|---------|----------|------|
| S4.1 | VLM 推理延迟 | 计时 `VLMPerception.detect()` | < 5s（含模型加载） | ⬜ |
| S4.2 | ASR 识别延迟 | 计时 4s 录音+识别 | < 5s（含录音时间） | 🟢 已验证 <5s |
| S4.3 | TTS 合成延迟(每句) | 计时 StreamingTtsPlayer | < 2s | ⬜ |
| S4.4 | 轨迹回放帧率 | 录制时长/执行时长 | ≥ 15fps | ⬜ |
| S4.5 | 录像帧率 | 检查 MP4 metadata | ≥ 15fps | ⬜ |
| S4.6 | 唤醒词响应延迟 | 从说话到唤醒 | < 1s | 🟢 已验证 <0.5s |
| S4.7 | 闲置 VLM 卸载时间 | 最后一次推理后计时 `idle_seconds()` | ≈ 30s（VLM_IDLE_UNLOAD_TIMEOUT） | ⬜ |
| S4.8 | 内存峰值 | 同时运行 VLM+ACT+机械臂+录像 | < 6GB（8GB 板；预算: VLM 900MB + ACT 350-450MB + 录像 256MB + 系统余量 200MB，见 settings 内存配置） | ⬜ |
| S4.9 | ACT chunk 推理延迟 | `ACTPolicy.last_infer_ms()` | 70-120ms（A76 预算） | ⏸ 模型资产 |
| S4.10 | GGCNN 单帧延迟 | 计时 `GGCNNDetector.detect()` | 30-50ms（ONNX CPU） | ⏸ 模型资产 |
| S4.11 | 相机采集帧率 | CameraManager 30s 帧计数 | ≥ 30Hz | ⬜ |
| S4.12 | 控制循环频率 | ACT 模式外层执行循环计时 | ≥ 50Hz（chunk 内逐步执行不重推理） | ⏸ 模型资产 |

---

## 测试报告模板

每次测试后记录：

```
## [编号] [测试项名称]

日期: YYYY-MM-DD
测试人: 
测试环境: 板端 rk3588 / PC WSL2
前置条件: 
实际结果: 
预期结果: 
结论: 🟢/🟡/🔴/⏸
备注: 
```
