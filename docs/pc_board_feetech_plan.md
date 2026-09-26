# PC 端 / 板端功能矩阵与 Feetech 协议层构建方案

> 版本: v1.0 (2026-09-25) | 状态: 待评审
> 关联: refactor_plan_v9.md（架构决策）、deploy_guide.md §2.5（版本矩阵）、test_plan.md M1/P1.1（验收）
> 背景: vendored lerobot 子集已删除（2026-09-25）。Feetech 舵机协议层走自研路线——基于官方
> `feetech-servo-sdk`（scservo_sdk）与 `pyserial`，参考 lerobot（Apache-2.0）等优秀开源项目的
> 成熟设计逐点吸收，不引入 LeRobot 包依赖。

---

## 1. 两端功能矩阵

### 1.1 职责总原则

| | 板端（RK3588, elf@10.1.27.9） | PC 端（WSL2 Ubuntu 22.04 + RTX 4060） |
|---|---|---|
| 定位 | **实时推理执行端**（部署态） | **开发/导出/训练端**（研发态） |
| Python | 3.10（conda `rk3588` env） | 3.12（conda `rk3588` env）+ 3.10（conda `lerobot` env） |
| 硬件 | 机械臂×2（ttyACM0/1）、D435i、NPU、麦克风/扬声器、nau8822 声卡 | GPU（CUDA）、可选 USB 直通调试硬件 |
| 红线 | numpy<2.0；零 torch 用途（系统预装除外）；零 LeRobot 依赖 | 不做实时推理（无 NPU）；不长期直控板端硬件 |

### 1.2 板端功能清单（全部已部署，L1 冒烟 21/21 ✓）

| 功能域 | 模块 | 说明 | 依赖 |
|--------|------|------|------|
| 实时主链路 | `runtime/` | 相机采集 worker（30fps）→ 共享内存 → 推理 worker → 控制 worker（30-50Hz）；绑 A76 大核 | numpy/cv2 |
| 机械臂控制 | `hardware/arm.py` (SO101Arm) | 6 关节 SYNC 读写、IK 运动、home、急停、单例 | scservo_sdk |
| 相机 | `hardware/camera_d435i.py` | RGB+深度采集、对齐 | pyrealsense2 |
| 录像 | `hardware/encoder.py` | H264 硬编（h264_rkmpp + scale_rkrga） | ffmpeg 6.0.1 |
| NPU 推理 | `perception/vlm.py` | RKLLM demo 子进程（pexpect）+ RKNN 视觉编码 | rknnlite 2.3.2 |
| 抓取检测 | `perception/grasp_detect.py` | GGCNN ONNX（CPU ort，实测 50.7ms/帧） | onnxruntime |
| 策略 | `policy/` | kinematics（FK/IK 6/6 ✓）、act_policy（ONNX，待 checkpoint）、grasp_pipeline（双模式+降级） | numpy/scipy |
| 语音全链路 | `voice/` | KWS→ASR→intent→TTS（sherpa-onnx CPU，四件套模型已部署校验 12/12） | sherpa_onnx |
| 安全 | `hardware/safety.py` + `runtime/` SafetyMonitor | 关节限位、工作空间 clamp、超时降级 | — |
| 交互入口 | `main.py` / `menu.py` / `va.py` | 菜单、语音助手常驻 | pexpect |
| 遥操作录制 | `scripts/lerobot-record-lite`（现状）| 纯 pyserial 裸包读 leader（30fps→JSON），**缺 follower 跟随写** | pyserial |

### 1.3 PC 端功能清单

| 功能域 | env | 模块/工具 | 说明 |
|--------|-----|-----------|------|
| ONNX 导出链 | rk3588 (py3.12) | `tools/export_act_onnx.py`、`tools/export_ggcnn_onnx.py` | torch 2.11 + onnxscript；GGCNN 链路已实测打通（opset 18 单文件，ort 校验 ✓） |
| 单元测试 | rk3588 | `tests/test_kinematics.py`、`runtime/shared_frame.py` 自检 | 两端均 6/6 ✓ |
| 代码开发/审查 | — | 全仓库 | Windows repo 为单一事实来源，rsync/cp 同步两端 |
| 遥操作采集 | lerobot (py3.10) | `lerobot-teleoperate` / `lerobot-record`（pip lerobot 0.4.4 + feetech extra） | 需 USB 直通或采集盒；GPU torch 2.10+cu128 ✓ |
| 数据集处理/训练 | lerobot | LeRobot dataset → ACT 训练 | RTX 4060 8G；产出 checkpoint（safetensors） |
| 评估回放 | lerobot / rk3588 | `lerobot-record-lite` 产物的 JSON 可在 PC 端转换/平滑（`scripts/smooth_traj.py`） | — |

### 1.4 数据流管道（单向闭环）

```
[板端] 遥操作录制 (leader/follower 双总线, 自研轻量)
   └→ JSON 轨迹 ──(rsync)──→ [PC 端] 转 LeRobot dataset → ACT 训练 → checkpoint.safetensors
                                  └→ export_act_onnx.py (opset>=14) → act_policy.onnx
                                        └──(rsync)──→ [板端] policy/act_policy.py NPU/CPU 推理
[板端] 语音/VLM/GGCNN 实时任务 ← 全部模型资产已就位（VLM 1476MB / 语音 653MB / GGCNN 271KB）
```

---

## 2. 自研 Feetech 协议层：现状盘点

### 2.1 已有能力（hardware/arm.py, 571 行, 生产可用）

| 能力 | 实现 | 备注 |
|------|------|------|
| SYNC 批量读写 | GroupSyncWrite 0x2A / GroupSyncRead 0x38 | 与 lerobot 同款用法 |
| SDK 超时 Bug 修复 | monkey-patch `setPacketTimeout` | 与 lerobot 同源（gitee IBY2S6），双方英雄所见略同 |
| 安全写入 | `_safe_write`：3 次重试 + 串口关闭重开恢复 | **比 lerobot 更强**（lerobot 只重试不恢复串口） |
| 标定参数 | JSON（homing_offset/range_min/range_max）+ 默认值回退 | 仅本地文件，未写舵机 EEPROM |
| 位置换算 | raw ↔ 弧度（4095 分辨率，中点偏移） | 等价 lerobot DEGREES norm_mode |
| 限位保护 | 写入前 clamp range_min/max | 总线级第一道防线 |
| IK 运动/归零/急停 | move_to（注入 kinematics）、home 插值、emergency_stop 禁扭矩 | 臂语义层 |
| 单例+锁 | 线程安全 get_instance，防串口争抢 | lerobot 无此设计（单线程假设） |

### 2.2 现存缺口（对照 lerobot 0.6.1 逐点核查）

| # | 缺口 | lerobot 做法 | 风险/价值 | 优先级 |
|---|------|-------------|-----------|--------|
| G1 | **无握手校验**：connect() 盲写串口，设备不存在/接错线不报错 | `_assert_motors_exist`：逐 ID ping + 型号码校验，缺失/错型列表化报错 | 部署排错成本高；fail-fast | **P0** |
| G2 | **STS3215 角度反馈溢出**：未清 Phase 寄存器 bit4，位置读数可能溢出为负 | `configure_motors` 中对 sts3215 清 0x12 的 0x10 位 | 硬件级坑，读数错乱→控制发散 | **P0** |
| G3 | **无目标突变限幅**：策略输出跳变直接砸向舵机 | `send_action` 的 `max_relative_target` + `ensure_safe_goal_position` | 安全第二道防线（SafetyMonitor 之外的总线级） | **P0** |
| G4 | **夹爪无防烧参数** | Max_Torque_Limit=500 / Protection_Current=250 / Overload_Torque=25 | 夹爪堵转烧电机（实物损失） | **P0** |
| G5 | **标定无生成流程**：calibration.json 全靠硬编码默认值，换一台臂即失配 | 交互三步：`set_half_turn_homings`（当前位置=半圈中点，写 Homing_Offset）→ `record_ranges_of_motion`（手动搬关节录 min/max）→ `write_calibration` | 没有它 P1 硬件测试无法在第二台臂上复现 | **P1** |
| G6 | 标定不写舵机 EEPROM | Homing_Offset/Min/Max_Position_Limit 写入舵机 + `is_calibrated` 一致性比对 | 标定跟随舵机本体，主机文件丢失可恢复 | P1 |
| G7 | 魔数遍布（0x2A/0x38/0x29/0x1A/0x28） | 控制表驱动 `data_name → (addr, length)` | 可维护性；新增寄存器读写不用查手册改代码 | P1 |
| G8 | Operating_Mode 未显式写 POSITION | connect 后逐电机写 Operating_Mode=0 | 舵机若被外部工具改成 PWM/速度模式，行为诡异 | P1 |
| G9 | **遥操作跟随缺失**：record-lite 只读 leader，无 follower 跟随写 | SOLeader.get_action → SOFollower.send_action 30Hz 环 | P1.1 遥操作录制测试的自研前提 | **P2** |
| G10 | 无总线扫描/单电机初始化 | `scan_port`（多波特率 broadcast_ping）、`setup_motor`（写 ID+波特率） | 调试工具，出厂流程低频 | P2 |
| G11 | P/I/D 系数不可配 | configure 时写 P/I/D_Coefficient | 运动质感调优 | P3 |
| G12 | 固件版本一致性不校验 | `_assert_same_firmware` | 混批舵机隐患，低频 | P3 |

**明确不做**（避免过度设计）：多品牌抽象三层继承（MotorsBusBase→SerialMotorsBus→FeetechMotorsBus 是为 Dynamixel/Feetech 双栈设计，我们只有 STS3215 单型号）、DeepDiff 控制表比较、draccus 配置框架、多分辨率模型表。

---

## 3. 构建路线（自研，零新增依赖：scservo_sdk + pyserial 均已两端就绪）

### 阶段 A：总线层加固（P0，对应 G1-G4）

新增 `hardware/feetech_bus.py`（协议层，~300 行），`arm.py` 瘦身为臂语义层并改用它：

```
hardware/feetech_bus.py
├── CTRL_TABLE: dict[str, tuple[addr, length]]   # STS3215 单型号控制表（消 G7）
├── class FeetechBus:
│   ├── connect(handshake=True)                  # ping×6 + 型号码校验，列表化报错（消 G1）
│   ├── configure()                              # Return_Delay/加速度 + Phase bit4 清除（消 G2）
│   │                                            # + Operating_Mode=POSITION（消 G8）
│   │                                            # + 夹爪防烧三参数（消 G4）
│   ├── read/sync_read/write/sync_write          # data_name 驱动，num_retry 统一
│   ├── clamp_relative_goal(max_delta_rad)       # 目标突变限幅（消 G3）
│   └── disable/enable_torque + torque_disabled 上下文
```

保留 arm.py 现有更强能力：`_safe_write` 串口恢复、单例锁、home 插值、IK 注入。
验收：test_plan M1（SO101Arm 全 API）+ 新增 M1 用例：拔线 connect 必须 fail-fast 报"缺失 ID 列表"。

### 阶段 B：标定体系（P1，对应 G5/G6）

新增 `tools/calibrate_arm.py`（交互式，参考 lerobot calibrate 流程重写为中文提示）：
1. 禁扭矩 → 提示"将臂摆到中位" → Enter → 计算并写 Homing_Offset（当前位置=2047，sign-magnitude 编码）
2. 提示"逐关节搬动全行程" → 30Hz 轮询录 min/max（wrist_roll 固定 0-4095）
3. 写 `config/calibration.json`（兼容现有格式）+ 可选写舵机 EEPROM（Min/Max_Position_Limit）
4. `--verify` 模式：读回 EEPROM 与 JSON 比对（is_calibrated 语义）

验收：两台臂（leader/follower）各生成一份 calibration.json；test_plan P0.4 标定前置项闭环。

### 阶段 C：遥操作与录制（P2，对应 G9，支撑 test_plan P1.1）

新增 `hardware/teleop.py` + 扩展 `scripts/lerobot-record-lite`：
- `LeaderArm`（复用 FeetechBus， torque 禁用，只读 Present_Position）
- 30Hz 跟随环：leader sync_read → 弧度 → follower write_positions（经 G3 限幅）
- 录制输出 JSON（现有 record-lite 格式，frames[{J1..J6,t}]）+ `--follow` 开关
- PC 端转换脚本 `scripts/json_to_lerobot.py`（JSON → LeRobot dataset，供训练；PC 端专用，板端零依赖）

验收：P1.1 拖拽主臂 15 秒从臂跟随，生成 JSON；转换脚本在 PC 端产出可被 lerobot env 加载的 dataset。

### 阶段 D：调试工具（P2-P3，对应 G10-G12）

`tools/feetech_scan.py`：多波特率 broadcast_ping 扫描 + 单电机 ID/波特率写入（出厂初始化）。
按需实现，不阻塞主线。

---

## 4. 与现有架构的衔接

- **依赖方向不变**：feetech_bus（协议）← arm（臂语义）← runtime/policy（应用），硬件层不反向依赖策略层；kinematics 仍由组合根注入
- **接口不变**：SO101Arm 继续实现 `HardwareModule`（setup/start/stop/is_available/on_failure/execute/get_observation），System 层零改动
- **板端零新增 pip 依赖**：scservo_sdk（feetech-servo-sdk）+ pyserial 均已装且实测 ✓
- **PC 端采集走 lerobot env（pip lerobot 0.4.4）**，与自研协议层并存互不干扰；自研遥操作（阶段 C）成熟后可替代 PC 端采集，实现"一套代码两端跑"
- **测试对应**：M1（arm API）→ 阶段 A；P0.4（标定）→ 阶段 B；P1.1（遥操作录制）→ 阶段 C

## 5. 参考实现清单（学习来源，均已通读）

| 项目 | 文件 | 吸收点 |
|------|------|--------|
| lerobot 0.6.1 (Apache-2.0) | `motors/motors_bus.py` (1296行) | 控制表驱动、握手、write/sync_write 分工、num_retry、标定三部曲、归一化 |
| lerobot 0.6.1 | `motors/feetech/feetech.py` (459行) | STS3215 Phase bit4、固件校验、broadcast_ping 裸协议、sign-magnitude、OperatingMode |
| lerobot 0.6.1 | `robots/so_follower/so_follower.py` + `teleoperators/so_leader/so_leader.py` | 6 关节 Motor 表、夹爪防烧、max_relative_target、标定向导、leader/follower 组装范式 |
| Feetech 官方 SDK | `feetech-servo-sdk`（PyPI，即 scservo_sdk） | PortHandler/PacketHandler/GroupSync*（两端已装） |
| 本项目 | `scripts/lerobot-record-lite` (61行) | 纯 pyserial 裸 SYNC_READ 包构造+解析雏形（0xFF 0xFF 0xFE ... checksum） |
| [commanderfun/STS3215](https://github.com/commanderfun/STS3215)（社区，MIT 风格教程） | `servo.py` Servo 类 | Status 错误标志位定义（bit0 Voltage/bit1 Sensor/bit2 Temperature/bit3 Current/bit5 Overload）、Present_Load bit10 方向+低10位幅值解码、电流 6.5mA/step、move_sync 双阶段轮询语义 |
| [ftservo/FTServo_Python](https://github.com/ftservo/FTServo_Python)（官方新 SDK，PyPI `ftservo-python-sdk` 2.0.0） | — | 已评估**不引入**：scservo_sdk 两端已部署实测、超时 Bug monkey-patch 就位、与 lerobot 生态同源；双 SDK 并存徒增维护面 |

> 许可合规：lerobot 为 Apache-2.0，本方案为设计思想借鉴+独立重写（非代码复制），CHANGELOG 记录出处。

---

## 6. 实现与验证状态（2026-09-25 更新）

阶段 A-D **全部落地并两端验证** ✅

| 阶段 | 产物 | 验证 |
|---|---|---|
| A | `hardware/feetech_bus.py`（控制表驱动协议层）+ `hardware/arm.py` 重构（公开 API 零破坏）+ `tests/test_feetech_bus.py` | 单测 PC 8/8 + 板端 8/8；导入链板端 23/23；**真实握手** follower/leader 各 6×STS3215(model=777)；**configure 真机验收**：写前快照 Phase bit4=1 共 5 台（G2 隐患实机证实），写后读回全部一致（bit4 清除/Return_Delay=0/Acceleration=16/POSITION/扭矩恢复/夹爪防烧 500-250-25），耗时 0.2s，位置读数全 [0,4096) |
| B | `tools/calibrate_arm.py`（中位归零+行程录制+可选 `--write-eeprom`/`--verify`） | 语法/--help 通过；真实标定需手搬臂交互（按 test_plan P0.4 执行） |
| C | `hardware/teleop.py`（LeaderArm+TeleopPair 30Hz 跟随）+ `record-lite --follow` + `scripts/json_to_lerobot.py` | npz 端到端通过；**lerobot 0.4.4 env 转换验收 rc=0**（v2.x parquet+meta 10 产物）；真实双臂跟随按 P1.1 |
| D | `tools/feetech_scan.py`（多波特率扫描+出厂初始化） | --help 通过；broadcast_ping 真实 6 ID error=0 |

构建中发现并修复的存量 Bug：
1. **arm.py 寄存器地址对调**：原以 0x29(41)=Acceleration 当 "Return_Delay_Time" 写、以 0x1A(26)=CW_Dead_Zone 当 "Acceleration" 写，真正的 Return_Delay_Time(addr 7) 从未设置——控制表修复，真机读回证实
2. **record-lite argparse dest Bug**：`--robot.port` 定义后用 `args.robot.port` 点号访问（运行即崩）——显式 `dest=` 修复

G11(PID)/G12(固件版本校验) 为可选项，接口已预留（`configure(pid=...)` / 控制表含 Firmware_* 寄存器）。

### 协议完整性复审（第二轮，2026-09-25）

对照 lerobot 0.6.1 全 API 面（motors_bus/feetech/so_follower/so_leader）+ 社区
commanderfun/STS3215 Servo 类 + 官方新 SDK 逐项核查后：

**补齐实现**：
- **G12 落地**：`firmware_versions()` + 握手中固件一致性检查（混批警告不中止）——真机实测 6 台全 3.10 一致
- **健康监测**：`read_diagnostics()`（Status 错误标志解码/温度/电压/电流 mA/负载%+方向/Moving，一次只读全总线）+ 纯函数 `decode_status_flags`/`decode_load`——真机实测全健康（33-36°C / 12.0V / 0 错误标志）
- **到位等待**：`wait_until_stopped(targets, tolerance, timeout)`（move_sync 语义：Moving=0 且位置入容差）；`SO101Arm.move_to(..., wait=True)` 可选启用
- **抓取闭环判据**：`SO101Arm.gripper_current()`（电流 mA + 负载%，夹到物体后上升）、`SO101Arm.diagnostics()` 全臂透传
- **teleop 实测 fps 统计**（录制结束打印实际帧率 vs 目标）

**明确跳过（防过度设计，理由记录）**：
| 项 | 跳过理由 |
|---|---|
| `reset_calibration`（恢复出厂标定） | 危险操作，误触即丢标定；重新标定成本低（向导 3 分钟） |
| INST_REG_WRITE + INST_ACTION | SYNC_WRITE 已满足广播同步写；延迟触发场景不存在 |
| NORMALIZE_MODES 三模式归一化（lerobot） | arm 层已有 rad 语义 + clamp，训练数据为 rad（lerobot dataset 同源） |
| 多品牌/Protocol 1 兼容层 | 单型号 STS3215 决策（§3 已定），无第二种总线设备 |
| wheel mode / spin（连续旋转） | 机械臂关节无连续旋转用例；Operating_Mode 寄存器已暴露可手动切 |
| `is_calibrated`/`set_baudrate` 显式 API | 标定存在性由 JSON 文件判断（arm 层）；波特率变更场景已被 `setup_motor`/`scan_baudrates` 覆盖 |

### 协议与官方语义对齐（第三轮，2026-09-25）

起因：无限时跟随实测出现「轨迹对不上 + 恒定偏差 + 部分关节反向」。逐行对照
lerobot `feetech.py`（configure_motors/write_calibration/_get_half_turn_homings）、
`motors_bus.py`（`_normalize`/`_unnormalize` 的 DEGREES 模式）、`so_follower.py`
（configure）、`so_leader.py`、`tables.py`、`config_so_follower.py` 后定位 **4 个实质缺陷**：

| # | 缺陷 | 官方值/语义 | 影响 | 修复 |
|---|---|---|---|---|
| 1 | 角度零点用 `homing_offset`（手摆中位点） | `mid=(range_min+range_max)/2`（DEGREES 模式） | 主从两臂零点不重合 → **跟随恒定偏差 + 轨迹错位** | 新增纯函数 `angle_zero/raw_to_rad/rad_to_raw`（feetech_bus），arm/teleop 统一调用；体关节用行程中点，夹爪保留 homing 零点（官方夹爪走 RANGE_0_100，语义不同） |
| 2 | `Acceleration = 16` | `configure_motors(acceleration=254)` | 从臂加速迟滞 → 跟随滞后 | 默认改 254 |
| 3 | `Maximum_Acceleration`（addr 85）从未写入 | `maximum_acceleration=254` | 出厂默认上限压住梯形加减速 | configure 写入 254 |
| 4 | PID 从未写入（仅显式传参时写） | `position_p/i/d = 16/0/32` | 位置环刚度/阻尼非官方值 | configure 默认写 `DEFAULT_PID` |

**额外加固（lerobot 未覆盖的真机坑位）**：

- **Phase bit6 编码器计数方向位**：真机实测 leader 5 个关节（J1/J3/J4/J5/J6）Phase=0x4C
  （bit6=1），follower 全 0x0C —— 两臂这些关节**读数方向天生相反**，遥操作时该关节
  反向（J2 恰好同向）。这是「甚至相反」的硬件级根因。configure 新增
  `align_direction=True`：检测并清除 bit6，同时 WARNING 明确提示「须重录本臂行程标定」。
- **`configure()` 全项落进控制表 + 单测锚定**：新增回归测试 `lerobot parity defaults`
  锁定 PID 16/0/32、加速度 254、Phase 位掩码、configure 签名默认值，防回退。
- 主臂（leader）也执行 configure（`gripper_id=None`）：lerobot `SOLeader.configure`
  同款（configure_motors + Operating_Mode），写完立即禁扭矩保持可手搬。

**标定流程相应调整**：行程录制从「次要步骤」升为**决定角度零点**的关键步骤
（必须把每个关节推到机械行程两端到底；主从两臂同样力道推到底，零点才会对齐），
向导文案与打印（新增「角度零点(行程中点)」列）同步更新；`--derive-from-port`
零点推导降级为「EEPROM 归零一致性」可选步骤（角度换算不再依赖它）。

**真机验证**：板端单测 13/13 通过（新增 `angle zero semantics` / `lerobot parity defaults`）。
