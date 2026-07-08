# config — 配置文件与系统级工具

所有可配置参数和系统工具集中管理。

## 文件

| 文件 | 说明 |
|------|------|
| `settings.py` | 相机内参/串口/VLM 模型/设备索引 |
| `cpu_affinity.py` | 大小核算力隔离工具 |
| `memory.py` | 分级内存管控框架 |
| `README.md` | 本文件 |

## 参数一览 (settings.py)

| 参数 | 类型 | 说明 |
|------|------|------|
| `CAMERA_MATRIX` | np.ndarray (3×3) | 相机内参 |
| `CAMERA_INDEX` | int | 默认相机设备号 |
| `SERIAL_PORT` | str | 机械臂串口路径 |
| `SERIAL_BAUD` | int | 串口波特率 |
| `VLM_MODEL_NAME` | str | 模型名 |
| `VLM_MODEL_PATH` | str | .rkllm 模型文件路径 |
| `VLM_DEMO_BIN` | str | demo 程序路径 |
| `VLM_IDLE_UNLOAD_TIMEOUT` | float | 闲置卸载超时(秒) |

## CPU 亲和性 (cpu_affinity.py)

| 函数 | 说明 |
|------|------|
| `bind_current_thread(cores)` | 当前线程绑核 |
| `bind_process(pid, cores)` | 指定 PID 绑核 |
| `bind_subprocess_args(cores)` | 返回 taskset 参数 |
| `make_preexec_bind(cores)` | preexec_fn 回调 |

### 核心定义

```python
BIG_CORES = {0, 1, 2, 3}     # Cortex-A76（大核）
LITTLE_CORES = {4, 5, 6, 7}   # Cortex-A55（小核）
```

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
