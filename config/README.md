# config — 配置文件与系统级工具

所有可配置参数和系统工具集中管理。

> **单一事实来源**: `settings.py` 是相机内参/外参等标定参数的唯一定义处。
> `hardware/arm.py`、`policy/grasp_pipeline.py` 均直接**引用**（不复制字面量）；
> 标定脚本 `scripts/calibrate_camera.py` / `calibrate_extrinsics.py` 只回写本文件。

## 文件

| 文件 | 说明 |
|------|------|
| `settings.py` | 全系统参数单一来源（相机/串口/VLM/内存/SSD/子进程运行时） |
| `cpu_affinity.py` | 大小核算力隔离工具（A55 小核 0-3 / A76 大核 4-7） |
| `memory.py` | 分级内存管控框架（MemoryMonitor / MemoryLimiter） |
| `safety.py` | 安全阈值配置：关节限位、最大运动速度、急停参数 |
| `teaching.py` | 示教实验配置：默认抓取轨迹、手眼标定参数 |
| `calibration.json` | 6 舵机标定参数（SO101Arm 启动时加载） |
| `README.md` | 本文件 |

## 参数一览 (settings.py)

### 相机

| 参数 | 类型 | 说明 |
|------|------|------|
| `CAMERA_INDEX` | int | D435i RGB 相机设备号（21） |
| `CAMERA_OVERHEAD` | int | 上帝视角 USB 相机设备号（27，实验录像用） |
| `CAMERA_ARM` | int | 机械臂局部相机设备号（23） |
| `CAMERA_MATRIX` | np.ndarray (3×3) | 相机内参 fx/fy/ppx/ppy（`calibrate_camera.py --d435i` 回写更新） |
| `CAMERA_POSITION` | np.ndarray (3,) | 相机光心在机械臂基座系下的外参 [x,y,z] 米（`calibrate_extrinsics.py` 回写更新） |

### 串口 / 标定

| 参数 | 类型 | 说明 |
|------|------|------|
| `SERIAL_PORT` | str | 机械臂串口路径（/dev/ttyACM0） |
| `SERIAL_BAUD` | int | 串口波特率（1000000） |
| `SERVO_CALIB` | str | 舵机标定文件路径（./config/calibration.json） |

### VLM

| 参数 | 类型 | 说明 |
|------|------|------|
| `VLM_MODEL_NAME` | str | 模型名 |
| `VLM_MODEL_PATH` | str | .rkllm 模型目录路径 |
| `VLM_DEMO_BIN` | str | RKLLM demo 程序路径 |
| `VLM_IDLE_UNLOAD_TIMEOUT` | float | 闲置卸载超时（秒，默认 30） |

### 内存管理

| 参数 | 类型 | 说明 |
|------|------|------|
| `VLM_MEMORY_BUDGET_MB` | int | VLM 推理预估内存占用（900） |
| `RECORDING_MEMORY_BUDGET_MB` | int | 录像编码预估内存占用（256） |
| `MEMORY_RESERVE_MB` | int | 系统预留内存余量（200） |

### 子进程运行时 (refactor_plan_v9 §1.6/§4.1, v0.5.0 新增)

| 参数 | 类型 | 说明 |
|------|------|------|
| `USE_SUBPROCESS_RUNTIME` | bool | True 时相机/ACT/GGCNN 走 `runtime/` 子进程 + 共享内存；默认 False（进程内路径，已验证）。完整多进程编排需板端实测后启用 |
| `CORES_CAMERA` | set | 相机采集绑核（A55 {0,1}，30Hz） |
| `CORES_ARM_IO` | set | 串口 IO 绑核（A55 {1}，50Hz） |
| `CORES_VOICE` / `CORES_SAFETY` | set | 语音 / 安全监控绑核（A55 {2,3}） |
| `CORES_ACT` | set | ACT 推理绑核（A76 {4,5}） |
| `CORES_GGCNN` | set | GGCNN 检测绑核（A76 {5}） |
| `CORES_VLM` | set | VLM 推理绑核（A76 {6,7}，已是 RKLLM 子进程） |
| `SHARED_FRAME_NAME` | str | 共享内存帧缓冲名（"eia_camera_frame"，相机子进程为生产者） |

## CPU 亲和性 (cpu_affinity.py)

| 函数 | 说明 |
|------|------|
| `bind_current_thread(cores)` | 当前线程绑核 |
| `bind_process(pid, cores)` | 指定 PID 绑核 |
| `bind_subprocess_args(cores)` | 返回 taskset 参数 |
| `make_preexec_bind(cores)` | preexec_fn 回调 |

### 核心定义

```python
LITTLE_CORES = {0, 1, 2, 3}   # Cortex-A55（小核）— 相机/串口/语音/安全监控
BIG_CORES = {4, 5, 6, 7}      # Cortex-A76（大核）— ACT/GGCNN/VLM 推理
```

> 注意: **子进程不继承父进程绑核亲和性**。`runtime/worker.py` 在子进程内用
> `rebind_affinity()` + 上表 `CORES_*` 重新绑核（refactor_plan_v9 §1.6）。

## 内存管控 (memory.py)

| 类 | 说明 |
|------|------|
| `MemoryMonitor` | 内存监控器，管理组件间内存争用 |
| `MemoryLimiter` | 上下文管理器，保护大内存操作 |

### 内存预算

| 组件 | 预算 |
|------|------|
| VLM | 900MB |
| Recording | 256MB |
| TTS | 64MB |
| ASR | 50MB |
