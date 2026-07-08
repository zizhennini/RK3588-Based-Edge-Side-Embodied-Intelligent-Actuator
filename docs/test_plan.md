# RK3588-EIA 系统测试方案

> 覆盖模块测试 → 阶段测试 → 全阶段测试 → 总测试四级

---

## 测试环境

| 项目 | 配置 |
|------|------|
| 硬件 | RK3588 (16GB) + D435i + SO-ARM101 主臂(ttyACM1) + SO-ARM101 从臂(ttyACM0) + 北通蝙蝠4 |
| 软件 | Conda rkvla (Python 3.10), sherpa-onnx, pyrealsense2, ffmpeg-rkmpp |
| 工作目录 | `/home/elf/work/RK3588-EIA` |
| 前置激活 | `source /home/elf/work/miniconda/etc/profile.d/conda.sh && conda activate rkvla` |

### 测试状态标记

| 标记 | 含义 |
|------|------|
| 🔴 未通过 | 测试失败，需修复 |
| 🟡 部分通过 | 功能可用但有缺陷 |
| 🟢 通过 | 测试通过 |
| ⬜ 未测试 | 尚未执行 |
| ⏸ 阻塞 | 依赖项未就绪 |

---

## 第一部分：模块测试

### M1. 机械臂控制 (ArmController)

#### M1.1 串口通信

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M1.1.1 | 串口连接 | `python3 -c "from vla.control import ArmController; a=ArmController('/dev/ttyACM0'); a.close()"` | 无串口异常 | ⬜ |
| M1.1.2 | 舵机位置读取 | `python3 -c "from vla.control import ArmController; a=ArmController('/dev/ttyACM0'); print(a._read_current_pos()); a.close()"` | 输出6个浮点数(弧度) | ⬜ |
| M1.1.3 | 舵机力矩开关 | `python3 -c "from vla.control import ArmController; a=ArmController('/dev/ttyACM0'); a.emergency_stop(); a.close()"` | 舵机力矩关闭可自由转动 | ⬜ |

#### M1.2 关节控制

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M1.2.1 | 单关节写入 | `python3 -c "from vla.control import ArmController; import numpy as np; a=ArmController('/dev/ttyACM0'); a._write_angle(1, 0.5); time.sleep(1); a.close()"` | 关节1转动到约28° | ⬜ |
| M1.2.2 | 批量写入 | `python3 -c "from vla.control import ArmController; import numpy as np; a=ArmController('/dev/ttyACM0'); a.write_angles(np.array([0,0,-1.2,0,0,1.5])); time.sleep(2); a.close()"` | 机械臂移动到归零位 | ⬜ |
| M1.2.3 | 工作空间钳制 | `python3 -c "from vla.control import ArmController; a=ArmController(); print(a.clamp_workspace(np.array([0.5,0,0]))); a.close()"` | 输出 `[0.45, 0, 0]`（x被钳制到上限） | ⬜ |
| M1.2.4 | 平滑归零 | `python3 scripts/smooth_traj.py 或 a.home()` | 50步平滑过渡到HOME_POSE | ⬜ |
| M1.2.5 | 关节限位超界 | 尝试 `_write_angle(1, 3.0)`（超出 ±1.92） | 脉冲被钳制到 `range_max` | ⬜ |

#### M1.3 夹爪控制

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M1.3.1 | 夹爪张开 | `python3 -c "from vla.control import ArmController; a=ArmController('/dev/ttyACM0'); a.gripper(True); time.sleep(1); a.close()"` | 夹爪张开 | ⬜ |
| M1.3.2 | 夹爪闭合 | `python3 -c "from vla.control import ArmController; a=ArmController('/dev/ttyACM0'); a.gripper(False); time.sleep(1); a.close()"` | 夹爪闭合 | ⬜ |

#### M1.4 IK 解算

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M1.4.1 | IK 可达位置 | `python3 -c "from vla.control import ArmController; import numpy as np; a=ArmController(); a.move_to(0.25, 0.0, 0.15); time.sleep(2); a.close()"` | 机械臂移动到(0.25,0,0.15) | ⬜ |
| M1.4.2 | IK 不可达位置 | `python3 -c "from vla.control import ArmController; a=ArmController(); a.move_to(0.5, 0.5, 0.5); time.sleep(2); a.close()"` | 被钳制到工作空间边界 | ⬜ |

### M2. 语音模块

#### M2.1 KWS 唤醒词

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M2.1.1 | KWS 模型加载 | `python3 -c "from voice_assistant.voice_assistant.config import load_config; from voice_assistant.voice_assistant.wake import SherpaKeywordWake; w=SherpaKeywordWake(load_config())"` | 无异常 | ✅ 已验证 |
| M2.1.2 | 唤醒词检测 | `python3 va.py listen --wake-mode kws --seconds 4 --no-speak` | 说"小咪"后唤醒并录音 | 🟢 已验证 |

#### M2.2 ASR 语音识别

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M2.2.1 | ASR 模型加载 | `python3 -c "from voice_assistant.voice_assistant.config import load_config; from voice_assistant.voice_assistant.asr import SherpaAsr; SherpaAsr(load_config())"` | 无异常 | ✅ 已验证 |
| M2.2.2 | WAV 文件识别 | `python3 va.py stt /tmp/test.wav` | 输出识别文字 | ⬜ |
| M2.2.3 | 实时录音识别 | `python3 va.py once --seconds 4 --no-speak` | 录音→ASR→打印文字 | 🟢 已验证 |

#### M2.3 TTS 语音合成

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M2.3.1 | TTS 模型加载 | `python3 -c "from voice_assistant.voice_assistant.config import load_config; from voice_assistant.voice_assistant.tts import SherpaTts; SherpaTts(load_config())"` | 无异常 | ✅ 已验证 |
| M2.3.2 | 流式TTS播报 | `python3 va.py tts-stream "你好，我是RK3588智能助手"` | 喇叭播放语音 | 🟢 已验证 |
| M2.3.3 | 长文本播报(>220字) | `python3 va.py tts-stream "..."`（长文字） | 按句分段流式播放 | ⬜ |

#### M2.4 全链路测试

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M2.4.1 | 语音→VLM→TTS | `python3 va.py once --seconds 4` | 录音→ASR→Qwen→TTS完整流程 | 🟢 已验证 |
| M2.4.2 | 文字→VLM→TTS | `python3 va.py ask "介绍一下自己"` | 文字→Qwen→TTS播报 | 🟢 已验证 |
| M2.4.3 | 唤醒→录音→回答 | `python3 va.py listen-forever` | 说"小咪"→唤醒→录音→回答 | 🟢 已验证 |
| M2.4.4 | 拍照→VLM 分析 | `python3 va.py ask "画面中有什么" --no-speak` | 拍照→Qwen 分析→打印描述 | 🟢 已验证 |

### M3. 轨迹模块

#### M3.1 动作库管理

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M3.1.1 | 动作列表 | `python3 scripts/record_trajectory.py list` | 列出已注册动作 | ⬜ |
| M3.1.2 | 轨迹信息查看 | `python3 scripts/record_trajectory.py inspect greeting_01.json` | 显示元数据 | ⬜ |
| M3.1.3 | 动作删除 | `python3 scripts/record_trajectory.py delete test` | 删除指定动作 | ⬜ |

#### M3.2 轨迹平滑

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M3.2.1 | 单文件平滑 | `python3 scripts/smooth_traj.py motion_library/grasp_01.json --output /tmp/test_smoothed.json` | 生成 smoothed 文件 | ⬜ |
| M3.2.2 | 批量平滑 | `python3 scripts/smooth_trajectory.py` | 处理所有未平滑轨迹 | ⬜ |
| M3.2.3 | 速度限幅验证 | 检查平滑后逐帧角度差 | 均不超过 MAX_DELTA | ⬜ |

#### M3.3 轨迹回放

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M3.3.1 | 基本回放 | `python3 scripts/replay_traj.py motion_library/greeting_01.json --port /dev/ttyACM0` | 从臂执行greeting动作 | ⬜ |
| M3.3.2 | 轨迹合法性校验 | `python3 scripts/smooth_trajectory.py motion_library/grasp_01.json` | 输出校验结果 | ⬜ |

### M4. 任务控制器

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M4.1 | task_library 自动创建 | `python3 -c "from scripts.task_controller import TaskController; tc=TaskController(None); print(tc.list_tasks())"` | 输出空列表，task_library 目录已创建 | ⬜ |
| M4.2 | 创建任务 | 编程调用 `tc.create_task()` | 返回合法 JSON | ⬜ |
| M4.3 | 保存+加载 | `tc.save_task(task); tc.load_task(name)` | 保存后加载内容一致 | ⬜ |
| M4.4 | 安全校验(假数据) | `tc.verify_safety(invalid_task)` | 返回警告列表 | ⬜ |
| M4.5 | 模拟运行 | `python3 scripts/task_controller.py run <name> --dry-run` | 打印各步骤但不驱动臂 | ⬜ |

### M5. 录像模块

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M5.1 | 硬件编码可用 | `ffmpeg -f lavfi -i color=c=black:s=640x480:d=2 -c:v h264_rkmpp -b:v 5M -y /tmp/test.mp4` | 输出2秒MP4，无报错 | ✅ 已验证 |
| M5.2 | RGA 缩放可用 | `ffmpeg -f lavfi -i color=c=black:s=1280x720:d=2 -vf scale_rkrga=640:480 -c:v h264_rkmpp -y /tmp/test_rga.mp4` | 输出MP4，无报错 | ⬜ |
| M5.3 | D435i 实时录像 | `python3 scripts/recorder.py --duration 5` | 录制5秒MP4到 recordings/ | ⬜ |
| M5.4 | OSD 叠加录像 | `python3 scripts/recorder.py --duration 5 --osd "实验记录"` | 视频画面叠加文字 | ⬜ |
| M5.5 | 录像启停 | 运行后按 Ctrl+C | 正常停止并保存文件 | ⬜ |

### M6. VLM 抓取

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M6.1 | 像素坐标解析 | `python3 scripts/vlm_grasp.py "红色杯子" --teach-trajectory grasp_01.json --ref-cx 320 --ref-cy 240 --ref-z 0.3 --dry-run` | Qwen 输出坐标+打印偏移量 | ⬜ |
| M6.2 | 深度3D解算 | 同上（检查输出坐标） | 输出合理的3D坐标 | ⬜ |
| M6.3 | 安全校验 | 带超限偏移量的假数据 | 校验未通过，提示超限 | ⬜ |

### M7. 安全防护

| 编号 | 测试项 | 命令 | 预期 | 状态 |
|------|--------|------|------|------|
| M7.1 | 深度避障状态机 | 手动遮挡D435i前方 | depth状态从safe→warn→stop | ⬜ |
| M7.2 | 电流读取 | 运行中 `CurrentMonitor.read_current(1)` | 输出合理电流值(几十~几百mA) | ⬜ |
| M7.3 | 教学模式加载 | `python3 -c "from config.teaching import TeachingMode; m=TeachingMode('student','manual')"` | 无异常 | ⬜ |
| M7.4 | 模式切换(教师) | `python3 -c "from config.teaching import TeachingMode; m=TeachingMode('teacher','ai'); print(m.get_label())"` | 输出"AI自主抓取模式" | ⬜ |
| M7.5 | 功能可见性 | `python3 -c "from config.teaching import TeachingMode; m=TeachingMode('student','manual'); print(m.is_enabled('vlm'))"` | 输出 False | ⬜ |

---

## 第二部分：阶段测试

### P1. 第一期（基础功能）

| 编号 | 测试项 | 测试步骤 | 预期 | 涉及模块 | 状态 |
|------|--------|---------|------|---------|------|
| P1.1 | 遥操作录制 | `python3 scripts/teleop_record.py --leader /dev/ttyACM1 --follower /dev/ttyACM0 --episode_time_s 15` | 拖拽主臂15秒，从臂跟随，生成JSON | B1, M1 | ⬜ |
| P1.2 | 平滑处理 | `python3 scripts/develop_motion.py greeting --no_record` | 处理raw文件→smoothed | B2, M3.2 | ⬜ |
| P1.3 | 回放验证 | `python3 scripts/replay_traj.py motion_library/greeting_01.json --port /dev/ttyACM0` | 从臂执行轨迹 | B6, M3.3, M1 | ⬜ |
| P1.4 | 语音触发回放 | 运行va.py listen-forever，说"你好" | 关键词匹配→回放greeting | B7, C1, M2 | ⬜ |
| P1.5 | 语音循环回放 | 说"循环回放你好" | 从臂回放3次 | B7, M2 | ⬜ |
| P1.6 | 语音暂停/继续 | 回放中说"暂停"→"继续" | 暂停→恢复 | B6, M2 | ⬜ |
| P1.7 | 语音归零 | 说"归零" | 从臂回到HOME_POSE | C1, M2 | ⬜ |
| P1.8 | 语音动作列表 | 说"有什么动作" | TTS播报动作列表 | C1, M2 | ⬜ |

### P2. 第二期（核心功能）

| 编号 | 测试项 | 测试步骤 | 预期 | 涉及模块 | 状态 |
|------|--------|---------|------|---------|------|
| P2.1 | 任务控制器模拟 | `python3 scripts/task_controller.py list` | 列出任务(或空) | D6 | ⬜ |
| P2.2 | 创建组合任务 | 创建2段轨迹的组合任务→保存 | 存在task JSON文件 | D1, D6 | ⬜ |
| P2.3 | 组合任务模拟运行 | `python3 scripts/task_controller.py run <name> --dry-run` | 依次打印各步骤 | D2, D7 | ⬜ |
| P2.4 | 硬件录像 | `python3 scripts/recorder.py --duration 10` | 录制10秒MP4 | E1-E5 | ⬜ |
| P2.5 | OSD 录像 | `python3 scripts/recorder.py --duration 5 --osd "测试"` | 带文字叠加的MP4 | E3 | ⬜ |
| P2.6 | VLM 抓取模拟 | `python3 scripts/vlm_grasp.py "红色杯子" --teach-trajectory grasp_01.json --ref-cx 320 --ref-cy 240 --ref-z 0.3 --dry-run` | 打印偏移量 | F1-F4 | ⬜ |

### P3. 第三期（安全+教学）

| 编号 | 测试项 | 测试步骤 | 预期 | 涉及模块 | 状态 |
|------|--------|---------|------|---------|------|
| P3.1 | 深度避障联调 | 运行main.py时手动遮挡D435i | 从臂减速→停止 | G1 | ⬜ |
| P3.2 | 电流防护联调 | 运行中手动阻挡机械臂 | 电流上升→柔顺回退→停机 | G2, G3 | ⬜ |
| P3.3 | 教学模式切换 | 教师角色切换三种模式 | 功能可见性正确变更 | H1, H2 | ⬜ |
| P3.4 | 一键还原 | 教师执行还原 | 动作库/任务库/录像清空 | H3 | ⬜ |

---

## 第三部分：全阶段测试

跨功能模块的集成测试，验证端到端流程。

### I1. 语音→轨迹回放全链路

| 步骤 | 操作 | 预期 | 涉及模块 |
|------|------|------|---------|
| 1 | 启动 `python3 va.py listen-forever` | 进入唤醒监听状态 | M2.1 |
| 2 | 说"小咪" | 唤醒成功，指示灯响应(若有) | M2.1 |
| 3 | 说"你好" | ASR识别→关键词匹配→motion任务入队 | M2.2, B7 |
| 4 | 从臂执行greeting轨迹 | 机械臂平滑运动 | M3.3, M1 |
| 5 | TTS播报"动作greeting执行完成" | 喇叭播报 | M2.3 |
| 6 | 说"暂停" | 回放暂停 | B6 |
| 7 | 说"继续" | 回放恢复 | B6 |
| 8 | 说"归零" | 机械臂平滑回到HOME_POSE | C1, M1.2 |

### I2. 录制→平滑→入库→回放全链路

| 步骤 | 操作 | 预期 | 涉及模块 |
|------|------|------|---------|
| 1 | `python3 scripts/teleop_record.py --episode_time_s 15` | 拖拽主臂追帧录制 | B1, M1 |
| 2 | 检查生成的 JSON 文件 | 包含 frames 和 joints 数据 | B1 |
| 3 | `python3 scripts/develop_motion.py <name> --no_record` | 平滑处理+入库 | B2, B3, B5 |
| 4 | `python3 scripts/record_trajectory.py list` | 动作库中可见新动作 | B5 |
| 5 | `python3 scripts/replay_traj.py ...` 或说动作名 | 从臂执行 | B6, B7 |

### I3. 视觉→VLM→抓取全链路

| 步骤 | 操作 | 预期 | 涉及模块 |
|------|------|------|---------|
| 1 | 放置目标物体于示教基准位 | 机械臂可到达 | — |
| 2 | 运行 `python3 scripts/vlm_grasp.py "目标" --teach-trajectory ...` | Qwen识别+深度定位 | F1, F2, M2.4 |
| 3 | 安全校验通过 | 打印校验结果 | F3, D4 |
| 4 | 人工确认(y) | 机械臂执行修正后轨迹 | D5, M1 |
| 5 | 抓取完成，TTS播报 | 喇叭播报 | F4, M2.3 |

### I4. 录像+VLM 互斥

| 步骤 | 操作 | 预期 | 涉及模块 |
|------|------|------|---------|
| 1 | 开始录像 | ffmpeg 进程启动 | E2, E4 |
| 2 | 尝试启动 VLM 问答 | VLM 推理正常（录像非互斥项，仅资源互斥） | M2.4, H4 |
| 3 | 停止录像 | 资源释放 | E5 |

---

## 第四部分：总测试

全系统端到端验收测试，所有模块联合运行。

### S1. 正常场景

| 编号 | 场景 | 步骤 | 预期 | 状态 |
|------|------|------|------|------|
| S1.1 | 语音唤醒→视觉分析→播报 | 说"小咪"唤醒→"画面中有什么" | 拍照→VLM分析→TTS播报→30s后VLM自动卸载 | ⬜ |
| S1.2 | 语音触发动作回放 | 说"小咪"→"你好" | 从臂执行greeting→TTS播报完成 | ⬜ |
| S1.3 | 多指令排队 | "小咪"→"你好"→"再见" | 两条指令串行执行 | ⬜ |
| S1.4 | 指令中断 | 回放中说"停止" | 机械臂急停，队列清空 | ⬜ |
| S1.5 | VLM 零样本抓取(模拟) | 运行 `vlm_grasp.py --dry-run` | 目标识别→偏移计算→打印轨迹修正方案 | ⬜ |
| S1.6 | 实验录像 | 录像5分钟 | 正常录制，OSD信息可见 | ⬜ |

### S2. 异常场景

| 编号 | 场景 | 步骤 | 预期 | 状态 |
|------|------|------|------|------|
| S2.1 | 深度遮挡 | 机械臂运行时遮挡D435i | 减速→停止 | ⬜ |
| S2.2 | 机械臂碰撞 | 手动阻挡运动中的机械臂 | 电流上升→急停→回退 | ⬜ |
| S2.3 | VLM 识别失败 | 画面中无指定目标 | 提示"未检测到目标" | ⬜ |
| S2.4 | ASR 识别失败 | 环境噪音过大 | 不触发指令或提示无法识别 | ⬜ |
| S2.5 | 串口断开 | 运行中拔掉机械臂USB | ArmController 抛出异常→紧急停止 | ⬜ |
| S2.6 | 内存不足 | 连续VLM推理不卸载 | MemoryMonitor 触发阈值→自动回收 | ⬜ |
| S2.7 | ffmpeg 编码失败 | 磁盘空间不足 | Recorder 停止并提示错误 | ⬜ |

### S3. 学生模式权限验证

| 编号 | 场景 | 步骤 | 预期 | 状态 |
|------|------|------|------|------|
| S3.1 | 学生模式功能限制 | `--role student --mode manual` | VLM/录像不可用 | ⬜ |
| S3.2 | 教师模式全功能 | `--role teacher` | 所有功能可用，可切换模式 | ⬜ |
| S3.3 | 教师一键还原 | 教师执行还原命令 | 动作库/任务库/录像清空，index重置 | ⬜ |

### S4. 性能基准

| 编号 | 指标 | 测试方法 | 可接受阈值 | 状态 |
|------|------|---------|----------|------|
| S4.1 | VLM推理延迟 | 计时 `ask_qwen()` | < 5s（含模型加载） | ⬜ |
| S4.2 | ASR识别延迟 | 计时4s录音+识别 | < 5s（含录音时间） | 🟢 已验证 <5s |
| S4.3 | TTS合成延迟(每句) | 计时 `synthesize_samples()` | < 2s | ⬜ |
| S4.4 | 轨迹回放帧率 | 录制时长/执行时长 | ≥ 15fps | ⬜ |
| S4.5 | 录像帧率 | 检查MP4 metadata | ≥ 15fps | ⬜ |
| S4.6 | 唤醒词响应延迟 | 从说话到唤醒 | < 1s | 🟢 已验证 <0.5s |
| S4.7 | 闲置VLM卸载时间 | 最后一次推理后计时 | ≈ 30s | ⬜ |
| S4.8 | 内存峰值 | 同时运行VLM+机械臂 | < 12GB | ⬜ |

---

## 测试报告模板

每次测试后记录：

```
## [编号] [测试项名称]

日期: YYYY-MM-DD
测试人: 
前置条件: 
实际结果: 
预期结果: 
结论: 🟢/🟡/🔴
备注: 
```
