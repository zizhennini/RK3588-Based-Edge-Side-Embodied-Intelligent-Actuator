
# ELF2 RK3588 自主抓取项目重构方案（v9）

## 一、核心决策与技术修正

### 1.1 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| **策略模型** | ACT (~50M 参数) | LeRobot 生态，SO-ARM101 适配 |
| **推理引擎** | ONNX Runtime CPU（首选）-> RKNN NPU（中期） | A76 实测预估 70-120ms，非 x86 的 35-60ms |
| **舵机通信** | `feetech-servo-sdk` (PyPI)，导入名 `scservo_sdk` | 飞特官方 SDK，aarch64 已验证，8.4KB 极轻量 |
| **视觉理解** | Qwen3.5-0.8B VLM (已部署) | 开放词汇目标识别 |
| **抓取检测** | GGCNN (69K 参数) | 实时抓取位姿，处理未训练物体 |
| **语音交互** | sherpa-onnx (保持现有) | 已是最优方案 |
| **框架策略** | **轻量级自研，不依赖完整 LeRobot** | 仅板端推理 + 舵机 SDK |
| **Python 环境** | **Miniconda 统一管理，板端 Python 3.10 / PC 端 Python 3.12** | LeRobot v0.6.1 强制要求 Python >=3.12；板端不装 LeRobot 可用 3.10 |
| **并发模型** | **子进程（非多线程）用于 CPU 推理** | 规避 GIL 抖动，ACT/GGCNN/VLM 各走独立进程 |

### 1.2 v9 关键技术修正（相对 v8）

| 编号 | v8 表述 | v9 修正 | 依据 |
|------|---------|---------|------|
| C1 | Python 板端 3.11 / PC 端 3.12 | **板端 Python 3.10 / PC 端 Python 3.12**（Miniconda） | LeRobot v0.6.1 强制要求 Python >=3.12；板端不安装 LeRobot，保持 3.10 |
| C2 | ACT ONNX CPU 延迟 35-60ms | **70-120ms**（RK3588 A76） | 35-60ms 是 x86 PC 数据；A76 IPC 低于桌面 Zen，需实测 |
| C3 | ONNX 模型 ~200MB | **~280-350MB**（含优化器状态/缓冲区） | 50M 参数 * 4B = 200MB 纯权重，实际导出含额外张量和元数据 |
| C4 | NPU 3 核进程内分配 | **每核独立子进程** | RK3588 NPU 单进程只能绑定 1 个分片，不可进程内跨核 |
| C5 | ACT/GGCNN 多线程推理 | **子进程推理** | Python GIL 导致 CPU 推理多线程抖动，子进程隔离更稳定 |
| C6 | AV1 编码为 v3 默认 | **MP4 编码可配置**（H.264 默认，AV1 可选） | 板端 AV1 解码环境复杂，P0 阶段用 H.264 |
| C7 | 板端录制为 P0 | **板端录制降为 P2** | pyarrow 在 aarch64 编译困难，P0 阶段数据由 PC 端处理 |
| C8 | FrameBuffer 直接返回引用 | **返回 .copy() 深拷贝** | 防止多线程/多进程读取时出现脏数据 |
| C9 | ONNX opset 12 + scaled_dot_product_attention | **opset 14+** | scaled_dot_product_attention 需 opset 14+，opset 12 不支持 |
| C10 | rdk_LeRobot_tools 可直接复用 | **仅架构参考** | 该项目面向地平线 BPU，未适配 RKNN；IB-Robot 无开箱 ACT-RKNN 脚本 |

### 1.3 Feetech SDK 导入名澄清

**经项目源码验证**: `lerobot/src/lerobot/motors/feetech/feetech.py` 第 69/125/198/335 行均使用 `import scservo_sdk as scs`。多个独立来源（SO-ARM100 剥离方案、piwheels、FTServo_Python 官方仓库）一致确认：

- **PyPI 包名**: `feetech-servo-sdk`（安装用 `pip install feetech-servo-sdk`）
- **Python 导入名**: `scservo_sdk`（代码中 `import scservo_sdk`）
- **注意**: 另有 `ftservo-python-sdk` 包，导入名不同，本项目不使用

### 1.4 舵机通信方案

**决策**: 使用 `feetech-servo-sdk`（PyPI）作为底层通信，导入名 `scservo_sdk`。

**现有 ArmController 的问题**（`vla/control/controller.py`）:
1. 手工 pyserial 打包协议包，脆弱且难维护
2. 无 SYNC_READ，6 个舵机需 6 次串口交互
3. 缺少 `camera_to_robot()` / `move_to_camera_with_angle()` / `gripper_width()` 方法
4. 外参/端口与 `settings.py` 不一致
5. `command_queue.py` 调用不存在方法（致命 Bug）

**新方案**: 自研 `hardware/arm.py` 基于 `scservo_sdk` 封装，参考 LeRobot `FeetechMotorsBus` API 设计。

**注意**: `feetech-servo-sdk` v1.0.0 有 `setPacketTimeout` Bug，需 monkey-patch 修复（参考 `feetech.py:85-95`）。

### 1.5 ACT 部署路径

```
ACT Policy (PyTorch)
  |-- VisionEncoder (ResNet18) -> vision_encoder.onnx -> [可选] RKNN
  +-- TransformerLayers (4+1层) -> transformer.onnx -> ONNX Runtime CPU
```

| 优先级 | 方案 | 延迟（RK3588 A76） | 内存 | 工作量 |
|--------|------|------|------|--------|
| **P0** | ONNX Runtime CPU FP32 | 70-120ms | ~350-450MB | 1-2 天 |
| P1 | VisionEncoder RKNN + Transformer CPU | 30-60ms | ~100-150MB | 3-5 天 |

**ACT 推理代码参考**: 从 Shaka-Labs/ACT 或 tonyzhaozh/act 提取模型定义（~400 行），不依赖 LeRobot。

**ACT 默认配置**: chunk_size=100, n_action_steps=100, dim_model=512, n_heads=8, n_encoder_layers=4, n_decoder_layers=1, use_vae=True, latent_dim=32

### 1.6 统一入口 + 子进程架构

**问题**: 原项目 `main.py`、`va.py`、`menu.py` 三个独立入口争抢资源；Python GIL 导致多线程 CPU 推理抖动。

**方案**: 单一 `main.py` 入口 + **子进程隔离 CPU 密集型推理**。

```
main.py (主进程, 协调器)
  |-- CameraManager 子进程 (A55 核 0-1, 30Hz)
  |-- SO101Arm 串口 IO 子进程 (A55 核 1, 50Hz)
  |-- ACTPolicy 子进程 (A76 核 4-5, 推理后通过队列输出动作)
  |-- GGCNN 子进程 (A76 核 5, 10-30Hz)
  |-- VLM 子进程 (A76 核 6-7, 1-3Hz, 按需启动)
  |-- VoiceAssistant (主进程内线程, A55 核 2-3)
  +-- SafetyMonitor (主进程内线程, A55 核 3, 最高优先级)
```

**进程间通信**: `multiprocessing.Queue` + `shared_memory`（帧数据走共享内存避免序列化开销）。

**注意**: 子进程不继承父进程 CPU 绑核亲和性，**每个子进程内部必须重新调用 `os.sched_setaffinity()`**。

---

## 二、板端/PC 端分工方案

### 2.1 分工架构

```
+-- 板端 RK3588 (8+64) ---------------------------+
| 职责: 推理 + 实时控制                              |
|  Python: 3.10 (Miniconda, conda env: rkeia)       |
|  [遥操作] -- 录制演示轨迹 --> [TF 卡]              |
|  [ACT ONNX] <-- 模型权重 <-- [TF 卡]              |
|  [实时推理] -- 动作序列 --> [SO-ARM101]            |
+---------------------------------------------------+
         |  TF 卡离线搬运（主要）/ USB 局域网（备选）
         v
+-- PC 端 WSL2 Ubuntu 22.04 -----------------------+
| 职责: 训练 + 模型导出 + 数据处理                    |
|  Python: 3.10 (Miniconda, conda env: rkeia-train)  |
|  [训练数据] <-- [TF 卡]                            |
|  [LeRobot] -- lerobot-train --> [checkpoint]       |
|  [checkpoint] -- ONNX 导出 --> [TF 卡]             |
+---------------------------------------------------+
```

### 2.2 数据格式

| 数据类型 | 格式 | 说明 |
|---------|------|------|
| 训练数据 | LeRobot Dataset v3 (Parquet + MP4) | PC 端录制/训练；视频编码 H.264（默认）或 AV1（可选） |
| 模型权重 | ONNX (.onnx) ~280-350MB FP32 | PC 端导出，板端加载推理 |
| 标定参数 | JSON | 两端共享 |

**板端录制降为 P2**: P0 阶段遥操作数据直接由 PC 端 LeRobot 录制，避免板端编译 pyarrow 的困难。P2 阶段若需板端独立录制，使用自定义 HDF5 格式，PC 端转换。

### 2.3 TF 卡注意事项

- TF 卡格式化为 **ext4**（Linux 原生），Windows 原生不可读
- PC 端访问需: (1) WSL2 直接挂载物理磁盘，或 (2) 安装 Ext2Ft/Paragon ExtFS 等工具
- 单次传输 ~350MB 模型 + ~1GB 数据，USB 3.0 约 2-3 分钟

---

## 三、模块化架构设计

### 3.1 四层架构

```
+--------------------------------------------------+
|  应用层 (application/)                            |
|  - System: 统一入口，进程/硬件生命周期管理          |
|  - VoiceAssistant: sherpa-onnx KWS/ASR/TTS       |
|  - IntentRouter: 语音文本 -> 任务类型              |
+------------------+-------------------------------+
                   |  TaskRequest / TaskResult
+------------------+-------------------------------+
|  策略层 (policy/)                                 |
|  - ACTPolicy: ONNX 推理 -> action chunk           |
|  - GGCNNDetector: 抓取质量图 -> grasp pose        |
|  - GraspPipeline: VLM + ACT/GGCNN 模式切换       |
+------------------+-------------------------------+
                   |  Observation / Action
+------------------+-------------------------------+
|  感知层 (perception/)                             |
|  - CameraManager: 单一实例，深拷贝帧缓冲          |
|  - VLM: 开放词汇目标检测 (bbox)                   |
|  - DepthProcessor: 深度滤波 + 点云投影            |
+------------------+-------------------------------+
                   |  Frame (RGB + Depth + Timestamp)
+------------------+-------------------------------+
|  硬件层 (hardware/)                               |
|  - SO101Arm: scservo_sdk + IK + 标定              |
|  - D435iCamera: pyrealsense2 封装                 |
|  - SafetyMonitor: 深度避障 + 电流监测 + 急停      |
+--------------------------------------------------+
```

### 3.2 模块接口（含降级契约）

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Observation:
    rgb: np.ndarray          # (480, 640, 3) uint8
    depth: np.ndarray        # (480, 640) float32, 单位: 米
    state: np.ndarray        # (6,) 关节角度 (rad)
    timestamp: float

@dataclass
class Action:
    positions: np.ndarray    # (6,) 目标关节角度 (rad)
    gripper: float           # [0=全闭, 1=全开]
    execution_time: float

@dataclass
class TaskRequest:
    type: str                # "grasp" | "move" | "ask" | "replay"
    target: str
    bbox: Optional[tuple]    # VLM 检测框 (x1, y1, x2, y2) 归一化
    params: dict

@dataclass
class TaskResult:
    success: bool
    message: str
    data: dict

class Module(ABC):
    """所有模块的统一生命周期接口（含降级契约）"""
    @abstractmethod
    def setup(self, config: dict) -> None: ...
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @property
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def on_failure(self) -> str:
        """返回降级策略: "skip" | "retry" | "fallback" | "abort"
        System 根据此策略决定模块启动失败后的行为"""
        ...

class PerceptionModule(Module):
    @abstractmethod
    def detect(self, obs: Observation) -> dict: ...

class PolicyModule(Module):
    @abstractmethod
    def predict(self, obs: Observation) -> Action: ...

class HardwareModule(Module):
    @abstractmethod
    def execute(self, action: Action) -> bool: ...
    @abstractmethod
    def get_observation(self) -> Observation: ...
```

**System 降级逻辑**:
- VLM 不可用 -> `on_failure() = "fallback"` -> 仅支持 GGCNN 抓取
- ACT 不可用 -> `on_failure() = "fallback"` -> 使用 GGCNN 兜底
- 相机不可用 -> `on_failure() = "abort"` -> 仅支持遥操作和动作回放
- 舵机不可用 -> `on_failure() = "abort"` -> 系统拒绝启动

### 3.3 FrameBuffer 深拷贝保护

```python
class FrameBuffer:
    """线程/进程安全的帧缓冲区，写入时深拷贝防止脏数据"""
    def __init__(self):
        self._latest: tuple[np.ndarray, np.ndarray, float] | None = None
        self._lock = threading.Lock()

    def update(self, rgb: np.ndarray, depth: np.ndarray, ts: float):
        with self._lock:
            # 深拷贝：防止 pyrealsense2 FrameBuffer 回收后 numpy 引用失效
            self._latest = (rgb.copy(), depth.copy(), ts)

    def get_frame(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        with self._lock:
            if self._latest is None:
                return None
            rgb, depth, ts = self._latest
            return rgb.copy(), depth.copy(), ts  # 读取也拷贝
```

---

## 四、资源管理

### 4.1 子进程资源分配

| 子进程 | CPU 核 | 频率 | NPU | 内存预估 |
|--------|--------|------|-----|---------|
| CameraManager | A55 核 0-1 | 30 Hz | - | ~200MB |
| SO101Arm IO | A55 核 1 | 50 Hz | - | ~10MB |
| ACTPolicy | A76 核 4-5 | 1-5 Hz | -（P0）/ 核 1（P1） | ~350-450MB |
| GGCNNDetector | A76 核 5 | 10-30 Hz | - / 核 2（可选） | ~50MB |
| VLM | A76 核 6-7 | 1-3 Hz | 核 0（按需） | ~900MB |
| 主进程 (Voice+Safety) | A55 核 2-3 | - | - | ~300MB |

**NPU 核分配（每核一个独立进程）**:
- NPU 核 0: VLM 进程（按需加载，闲置 30s 卸载释放 ~900MB）
- NPU 核 1: ACT VisionEncoder（P1 阶段，独立子进程）
- NPU 核 2: GGCNN（可选，独立子进程）

**关键约束**: RK3588 NPU 单进程只能绑定 1 个分片，不可在同一 Python 进程内跨核。多 NPU 模型必须拆子进程。

### 4.2 内存预算

| 组件 | 内存 |
|------|------|
| 系统 + OS | ~1.5 GB |
| Qwen3.5 VLM | ~900 MB（闲置卸载） |
| ACT (ONNX Runtime) | ~350-450 MB（含运行时缓冲区） |
| GGCNN | ~50 MB |
| D435i 相机 | ~200 MB |
| 语音系统 | ~64 MB |
| Python + 依赖（无 LeRobot） | ~250 MB |
| **总计（VLM 加载时）** | **~3.3-3.5 GB，剩余 ~4.5 GB** |
| **总计（VLM 卸载后）** | **~2.4 GB，剩余 ~5.6 GB** |

### 4.3 存储规划（64GB eMMC + 32GB TF）

| 内容 | 大小 | 位置 |
|------|------|------|
| 系统 + 依赖 | ~8 GB | eMMC |
| 项目代码 | ~500 MB | eMMC |
| 模型文件 | ~1.5 GB（ACT 350MB + VLM 900MB + GGCNN 1MB + 语音 200MB） | eMMC |
| 训练数据集 | ~5-20 GB | TF 卡 |

---

## 五、机械臂控制层

### 5.1 SO101Arm 类设计

基于 `scservo_sdk` 封装，参考 LeRobot `FeetechMotorsBus` API。

**串口异常恢复**:
```python
class SO101Arm:
    def _safe_write(self, func, *args, max_retries=3):
        """带串口恢复的安全写入"""
        for attempt in range(max_retries):
            try:
                return func(*args)
            except serial.SerialException as e:
                logger.error(f"串口异常 (尝试 {attempt+1}/{max_retries}): {e}")
                self._reset_serial()
        raise RuntimeError("串口通信连续失败，无法恢复")

    def _reset_serial(self):
        """串口重置：关闭 -> 等待 -> 重新打开"""
        try:
            self.port_handler.closePort()
        except Exception:
            pass
        time.sleep(0.5)  # 等待 USB 串口重置
        self.port_handler.openPort()
        self.port_handler.setBaudRate(self.baud)
        self._configure_motors()
        logger.info("串口已重置")
```

**单例 + 崩溃恢复**:
```python
class SO101Arm:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, **kwargs) -> "SO101Arm":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    def connect(self):
        try:
            self.port_handler.openPort()
            self.port_handler.setBaudRate(self.baud)
            self._configure_motors()
            self._connected = True
        except Exception as e:
            # 进程崩溃后串口句柄可能残留，强制清理
            logger.error(f"连接失败: {e}, 尝试强制重置")
            self._force_reset()
            raise

    def _force_reset(self):
        """强制重置串口状态"""
        try:
            self.port_handler.closePort()
        except Exception:
            pass
        self._connected = False

    def close(self):
        """释放资源，允许重新创建"""
        self.emergency_stop()
        self._force_reset()
        SO101Arm._instance = None  # 清除单例，允许重建
```

### 5.2 完整 6DOF IK 输出

现有 IK 仅实现 2DOF 平面解算（joint2/joint3），**必须封装为 6 维关节向量输出**。

```python
class Kinematics:
    """SO-ARM101 完整 6DOF 运动学（XLeRobot 偏移补偿）"""

    # URDF 精确参数
    L1 = 0.1159   # 上臂
    L2 = 0.1350   # 前臂
    THETA1_OFFSET = math.atan2(0.028, 0.11257)    # ~14.0 deg
    THETA2_OFFSET = math.atan2(0.0052, 0.1349) + THETA1_OFFSET  # ~16.2 deg

    def inverse_kinematics(self, target_xyz: np.ndarray,
                           current_angles: np.ndarray,
                           wrist_roll_rad: float | None = None
                           ) -> np.ndarray:
        """
        完整 6DOF IK，返回 6 维关节角度 (rad)。

        输入:
            target_xyz: (3,) 目标笛卡尔坐标 [x, y, z] (米)
            current_angles: (6,) 当前关节角度 (rad)
            wrist_roll_rad: 腕部旋转角度 (rad)，None 则保持当前

        输出:
            angles: (6,) 目标关节角度 [shoulder_pan, shoulder_lift,
                      elbow_flex, wrist_flex, wrist_roll, gripper]
        """
        x, y, z = target_xyz
        # Joint 1: shoulder_pan (水平旋转)
        joint1 = math.atan2(y, x)

        # Joint 2-3: 2DOF 平面 IK（含偏移补偿）
        r = math.sqrt(x**2 + y**2)
        h = z - 0.0624  # 基座高度
        j2_deg, j3_deg = self._ik_2dof(r, h)

        # Joint 4: wrist_flex（保持与桌面平行或跟随）
        joint4 = current_angles[3] if wrist_roll_rad is None else wrist_roll_rad

        # Joint 5: wrist_roll
        joint5 = current_angles[4]

        # Joint 6: gripper（保持不变）
        joint6 = current_angles[5]

        return np.array([
            math.atan2(y, x),           # joint 1
            math.radians(j2_deg),        # joint 2
            math.radians(j3_deg),        # joint 3
            joint4,                       # joint 4
            joint5,                       # joint 5
            joint6,                       # joint 6
        ])

    def _ik_2dof(self, r: float, h: float) -> tuple[float, float]:
        """2DOF 平面 IK + 偏移补偿，返回角度 (deg)"""
        # ... 余弦定理 + 偏移补偿（同 v8 13.2 节）
        ...

    def forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """FK 验证: FK(IK(target)) == target (偏差 < 1mm)"""
        ...
```

### 5.3 标定参数统一

统一使用 `settings.py` 实测标定值 `[0.182, -0.129, 0.47]`，串口统一 `/dev/ttyACM0`。

---

## 六、项目目录结构

```
RK3588-EIA/
|-- main.py                      # [唯一入口] System 类 + 子进程管理
|-- config/                      # 配置管理（保留现有）
|   |-- settings.py              # 硬件参数（统一外参标定值）
|   |-- safety.py / cpu_affinity.py / memory.py
|   +-- calibration.json         # [新] 舵机标定参数
|-- hardware/                    # 硬件抽象层
|   |-- arm.py                   # SO101Arm (scservo_sdk + 6DOF IK + 标定)
|   |-- camera_d435i.py          # CameraManager (深拷贝帧缓冲)
|   +-- safety.py                # SafetyMonitor
|-- perception/                  # 感知模块
|   |-- vlm.py                   # Qwen3.5 VLM（独立子进程）
|   |-- grasp_detect.py          # GGCNN 实时抓取检测
|   +-- locator.py               # 颜色定位（降级方案）
|-- policy/                      # 策略模块
|   |-- act_model.py             # ACT 模型定义（从 Shaka-Labs/ACT 提取）
|   |-- act_policy.py            # ACT ONNX 推理（独立子进程）
|   |-- grasp_pipeline.py        # 抓取管线（VLM + ACT/GGCNN）
|   +-- kinematics.py            # 6DOF IK（XLeRobot 偏移补偿）
|-- voice/                       # 语音模块（从 voice_assistant/ 迁移）
|-- tools/                       # 开发工具（PC 端使用）
|   |-- export_act_onnx.py
|   +-- convert_rknn.py
|-- models/                      # 模型文件
|   |-- act/                     # ACT ONNX (~350MB FP32)
|   |-- ggcnn/                   # GGCNN ONNX (~1MB)
|   |-- vlm/                     # Qwen3.5 RKNN/RKLLM (~900MB)
|   +-- speech/                  # sherpa-onnx 语音模型 (~200MB)
|-- data/                        # 训练数据（TF 卡）
|-- scripts/                     # 辅助脚本
+-- tests/
```

---

## 七、分阶段实施计划

### 第一阶段：基础框架 + 统一入口 + 舵机 SDK + IK 修复（1-2 周）

| 任务 | 内容 | 参考 |
|------|------|------|
| T1.0 | 板端/PC 端 Miniconda 环境创建（板端 Python 3.10 / PC 端 Python 3.12） | 见第八章 |
| T1.1 | 创建 `main.py` 单一入口 + `System` 类 + 子进程管理 | -- |
| T1.2 | 创建 `hardware/arm.py`：基于 `scservo_sdk` 封装 SO101Arm，含串口恢复、单例崩溃重建 | `controller.py` + `feetech.py` |
| T1.3 | **修复 IK**：重写 `policy/kinematics.py`，完整 6DOF 输出，移植 XLeRobot 偏移补偿，FK/IK 偏差 < 1mm | XLeRobot `SO101Robot.py` |
| T1.4 | 创建 `hardware/camera_d435i.py`：深拷贝帧缓冲 | 现有 `camera/camera.py` |
| T1.5 | 迁移安全控制、配置管理（统一外参） | 现有 `config/` |
| T1.6 | 迁移语音模块到 `voice/` | 现有 `voice_assistant/` |
| T1.7 | 验证：模式切换、SYNC_WRITE/READ、IK 偏差 < 1mm、串口断连恢复 | -- |

### 第二阶段：GGCNN 通用抓取（1-2 周）

| 任务 | 内容 | 参考 |
|------|------|------|
| T2.1 | 移植 GGCNN，导出 ONNX，独立子进程运行 | github.com/angusdk/ggcnn |
| T2.2 | 创建 `perception/grasp_detect.py` | GGCNN |
| T2.3 | 创建 `perception/vlm.py`（含 bbox 归一化兼容） | 现有 `vla/vlm/qwen3_vl.py` |
| T2.4 | 创建 `policy/grasp_pipeline.py`（VLM + GGCNN + IK） | 参考 `scripts/vlm_grasp.py` |
| T2.5 | 验证：抓取 5 种未训练日常物体 | -- |

### 第三阶段：ACT 训练与部署（2-3 周）

| 任务 | 内容 | 参考 |
|------|------|------|
| T3.1 | PC 端 LeRobot 采集 50+ episode | `lerobot-record` |
| T3.2 | PC+GPU 训练 ACT | `lerobot-train` |
| T3.3 | 导出 ONNX（分模块），实际大小预期 ~280-350MB | D-Robotics 导出流程参考 |
| T3.4 | 板端 ONNX Runtime CPU 推理（独立子进程，实测延迟） | -- |
| T3.5 | （可选）VisionEncoder RKNN（独立 NPU 子进程） | -- |
| T3.6 | 集成 ACT + GGCNN 模式切换 | -- |

### 第四阶段：系统联调（1 周）

| 任务 | 内容 |
|------|------|
| T4.1 | 语音 -> 模式切换 -> ACT/GGCNN 完整链路 |
| T4.2 | 多速率异步架构（子进程间队列通信） |
| T4.3 | 性能优化（绑核/内存/延迟实测） |
| T4.4 | systemd 服务 + 部署文档 |

---

## 八、Python 环境管理（Miniconda 统一方案）

### 8.1 核心原则

**板端和 PC 端使用相同 Python 版本（3.10）**，通过 Miniconda 管理，避免版本不一致引发的依赖兼容问题。

### 8.2 板端环境（RK3588 aarch64）

```bash
conda create -n rkeia python=3.10 -y  # 板端不装 LeRobot，3.10 即可
conda activate rkeia

pip install onnxruntime           # ACT 推理 (aarch64 wheel)
pip install feetech-servo-sdk     # 舵机通信 (import scservo_sdk)
pip install numpy opencv-python-headless pyserial
pip install pyrealsense2          # D435i 相机
pip install sherpa-onnx           # 语音
# rknn-toolkit-lite2 通过 .whl 安装（已有）
```

### 8.3 PC 端环境（WSL2 Ubuntu 22.04 x86_64）

```bash
conda create -n rkeia-train python=3.12 -y  # LeRobot v0.6.1 要求 >=3.12
conda activate rkeia-train

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install lerobot[feetech]      # v0.6.1 + 舵机支持
pip install datasets huggingface-hub
pip install onnx onnxruntime      # 导出验证
```

### 8.4 两端依赖差异

| 依赖包 | 板端 (rkeia) | PC 端 (rkeia-train) |
|--------|:---:|:---:|
| Python | 3.10 | 3.10 |
| onnxruntime | Y | Y |
| feetech-servo-sdk | Y | Y |
| pyrealsense2 | Y | - |
| torch | - | Y (CUDA) |
| lerobot | - | Y |
| datasets | - | Y |
| sherpa-onnx | Y | - |
| rknn-toolkit-lite2 | Y | - |

---

## 九、风险与缓解

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| ACT CPU 推理 70-120ms | 中 | Action chunking 一次输出 100 步，推理频率 1-5Hz 即可；舵机响应 ~30ms 才是瓶颈 |
| ACT RKNN 转换失败 | 中 | ONNX Runtime CPU 保底；VisionEncoder 可单独上 NPU |
| ONNX 模型 ~280-350MB 超预期 | 低 | 64GB eMMC 充足；INT8 量化可降至 ~80MB |
| NPU 单进程单核限制 | 中 | VLM/ACT/GGCNN 各走独立子进程，每进程绑 1 个 NPU 核 |
| 串口崩溃后句柄残留 | 高 | `_reset_serial()` 强制关闭+重开；`close()` 清除单例允许重建 |
| GIL 导致推理抖动 | 中 | ACT/GGCNN/VLM 全部走子进程，非多线程 |
| FrameBuffer 脏数据 | 中 | 读写均 `.copy()` 深拷贝 |
| IK 精度不足 | 中 | 移植 XLeRobot 偏移补偿，修正连杆参数，FK/IK 偏差 < 1mm |
| 外参不一致 | 中 | 统一使用 `settings.py` 实测值 `[0.182, -0.129, 0.47]` |
| 子进程不继承 CPU 亲和性 | 低 | 每个子进程内部调用 `os.sched_setaffinity()` |
| TF 卡 ext4 Windows 不可读 | 低 | WSL2 直接挂载物理磁盘，或安装 Ext2Ft |
| 训练数据质量 | 中 | 严格筛选，先录 30 集验证 |
| 第三方项目复用度高估 | 低 | rdk_LeRobot_tools 仅架构参考（面向 BPU）；IB-Robot 无开箱 ACT-RKNN 脚本 |

---

## 十、废弃文件处理策略

**原则**: 已知 Bug 的文件**不修复**，直接废弃，参考优秀开源项目重新实现。

| 废弃文件 | 已知 Bug | 替代实现 |
|---------|---------|----------|
| `vla/command_queue.py` | 调用不存在方法，运行时必崩 | `policy/grasp_pipeline.py`（参考 `scripts/vlm_grasp.py`） |
| `vla/control/controller.py` | 手工 pyserial、无 SYNC_READ、缺方法、外参不一致 | `hardware/arm.py`（scservo_sdk + 完整 API） |
| `vla/kinematics.py` | 无偏移补偿、连杆参数偏差 37mm、NumericalIK 不存在 | `policy/kinematics.py`（XLeRobot IK + 6DOF） |
| `vla/pipe/pipeline.py` | 状态机耦合过重，不支持异步 | `policy/grasp_pipeline.py`（子进程架构） |
| `vla/vision/pca_grasp.py` | 2D PCA 精度不足 | `perception/grasp_detect.py`（GGCNN） |
| `lerobot/` (v0.4.4) | 板端不需要完整 LeRobot | 板端仅用 `scservo_sdk` + `onnxruntime` |
| `main.py` / `va.py` / `menu.py` | 三入口争抢资源 | 单一 `main.py` + `System` 类 |

**可迁移保留**: `camera/camera.py` -> `hardware/camera_d435i.py`；`config/` 原位保留；`voice_assistant/` -> `voice/`；`vla/vlm/qwen3_vl.py` -> `perception/vlm.py`

---

## 十一、关键参考项目（修正评估）

| 优先级 | 项目 | 关键价值 | 实际可用性说明 |
|--------|------|---------|---------------|
| **P0** | XLeRobot | IK 偏移补偿（已移植） | 直接可用的 IK 公式和参数 |
| **P0** | Shaka-Labs/ACT | 最轻量独立 ACT 实现 | 提取 ~400 行推理代码 |
| P1 | D-Robotics/rdk_LeRobot_tools | ACT 分模块 ONNX 导出流程 | **仅架构参考**：面向地平线 BPU，未适配 RKNN |
| P1 | DORA-RS | 数据流架构参考 | 设计理念参考，不直接集成 |
| P1 | IB-Robot | RK3588 ACT 参考 | **无开箱 ACT-RKNN 脚本**，仅有抽象接入层 |
| P2 | Embodied.cpp | C++ 多速率运行时 | 架构参考（多速率执行理念） |
| P2 | GGCNN | 69K 参数实时抓取检测 | 可直接导出 ONNX 使用 |
| 参考 | feetech-servo-sdk | 舵机通信 SDK | PyPI 包，导入名 `scservo_sdk` |

---

## 十二、开发日志

| 日期 | 版本 | 变更摘要 |
|------|------|---------|
| 2026-09-20 | v1-v2 | 初始方案 + ACT 部署案例调研 |
| 2026-09-21 | v3 | 移除完整 LeRobot 依赖，统一入口 |
| 2026-09-22 | v4-v5 | DORA-RS/Embodied.cpp 参考，板端/PC 分工，模块化架构 |
| 2026-09-23 | v6-v7 | Miniconda 环境、IK Bug 分析、废弃文件策略、数据格式确认 |
| 2026-09-23 | v8 | 技术校验：ACT 50M、数据格式 v3、Python 分离 |
| 2026-09-23 | **v9** | **全面修正**: Python 统一 3.10；ACT 延迟 70-120ms；ONNX ~280-350MB；NPU 单进程单核；子进程替代多线程；FrameBuffer 深拷贝；串口崩溃恢复；IK 完整 6DOF；pyarrow 降 P2；第三方项目评估修正；方案精简至 ~800 行 |
| 2026-09-24 | v9.1 | **架构债清理**（进入第三阶段前）: 修复依赖倒置(arm→policy 改依赖注入)；相机内外参统一引用 settings 单一源；新建 runtime/ 子进程+共享内存运行时(opt-in)；SO101Arm/CameraManager 落地 HardwareModule 接口；同步更新 architecture.md |

### v9 关键决策记录

1. **Python 版本分离**: 板端 Python 3.10（仅推理，不装 LeRobot）；PC 端 Python 3.12（LeRobot v0.6.1 强制要求 >=3.12）
2. **子进程架构**: ACT/GGCNN/VLM 各走独立子进程，规避 GIL，适配 NPU 单进程单核限制
3. **ACT 延迟修正**: 70-120ms（RK3588 A76 预估），非 x86 的 35-60ms
4. **ONNX 大小修正**: ~280-350MB（含缓冲区），非纯权重的 ~200MB
5. **串口崩溃恢复**: `_reset_serial()` + `close()` 清除单例，防止进程崩溃后串口残留
6. **IK 完整 6DOF**: 封装为 6 维关节向量输出，而非仅 2DOF 平面解算
7. **FrameBuffer 深拷贝**: 读写均 `.copy()`，防止多线程/多进程脏数据
8. **板端录制降 P2**: pyarrow aarch64 编译困难，P0 阶段数据由 PC 端处理
9. **第三方项目评估修正**: rdk_LeRobot_tools/IB-Robot 仅架构参考，无开箱可用脚本
10. **ONNX opset**: scaled_dot_product_attention 需 opset 14+（非 12）

### v9.1 架构债清理（2026-09-24）

进入第三阶段（数据录制 / 遥操作）前，集中偿还 4 项架构债（业务功能推进快于架构清理）：

| # | 债 | 处置 | 落地文件 | 验证 |
|---|-----|------|---------|------|
| 1 | **依赖倒置**: `hardware/arm.py:288` 在 `move_to()` 内 `from policy.kinematics import Kinematics`，硬件层反向依赖策略层 | 改**依赖注入**: `SO101Arm.__init__(kinematics=)` + `set_kinematics()`，由组合根 `main.py:init_arm` 注入 `Kinematics()`；`move_to` 用 `self._kinematics`，缺失则抛清晰错误。硬件层零 policy 导入 | `hardware/arm.py`, `main.py` | py_compile 通过；arm.py 无残留 `from policy` |
| 2 | **配置硬编码**: 相机外参 `[0.182,-0.129,0.47]`、内参 `604.2294...` 在 `arm.py:113`、`grasp_pipeline.py:72-78` 复制字面量，标定脚本回写 settings 后静默漂移 | 统一**引用** `config.settings.CAMERA_POSITION/CAMERA_MATRIX`（grasp_pipeline 类属性从 settings 求值；arm.py 构造时读取，带兜底）。settings.py 成为单一事实来源 | `hardware/arm.py`, `policy/grasp_pipeline.py` | py_compile；settings 单一源 |
| 3 | **子进程未落地**: 方案 §1.6/§4.1 承诺子进程 + shared_memory，实际 `main.py` 仅有 `import multiprocessing as mp` 死导入；CameraManager 用线程非子进程；ACT/GGCNN 进程内推理有 GIL 抖动风险 | 新建 `runtime/` 包: `SharedFrameBuffer`(seqlock 零拷贝跨进程帧传输) + `SubprocessWorker`(spawn+Queue+子进程内绑核) + `InferenceWorker`(ACT/GGCNN 子进程)；`settings.USE_SUBPROCESS_RUNTIME` opt-in 开关 + `CORES_*` 集中绑核配置；移除 main.py 死导入 | `runtime/shared_frame.py`, `runtime/worker.py`, `runtime/__init__.py`, `config/settings.py`, `main.py` | SharedFrameBuffer 往返 + 跨句柄 attach 自检通过；worker 导入/构造通过；**多进程编排需板端实测** |
| 4 | **接口未落地**: `SO101Arm`/`CameraManager` 为裸类，未实现 `HardwareModule` | 两类继承 `HardwareModule`，实现 `setup/start/stop/is_available/on_failure/execute/get_observation`。arm on_failure="abort"(硬依赖)，camera on_failure="skip"(可降级)；get_observation 各填本设备可得字段(arm→state, camera→rgb/depth)，由 System 合并 | `hardware/arm.py`, `hardware/camera_d435i.py` | CameraManager 接口一致性导入测试通过；SO101Arm 同模式(py_compile 通过，板端运行时确认) |

**遗留 / 后续（需板端）**:
- 债 3 完整多进程编排（相机/ACT/GGCNN 各独立子进程 + `USE_SUBPROCESS_RUNTIME=True` 默认启用）需板端实测 GIL/延迟/NPU 单核绑定后方可开启（对应 T4.2、风险 R2）。当前默认仍走已验证的进程内路径。
- `WORKSPACE` 在 arm.py / grasp_pipeline.py / kinematics.py 仍各有一份（非相机参数，本次未纳入）；关节限位表 arm.py(URDF 系) 与 kinematics.py(xlerobot 舵机系) 的坐标系对齐仍需板端标定确认。
- `hardware/arm.py` 的 `SO101Arm.get_observation()` 返回 rgb/depth=None，System 层观测合并逻辑待在第三阶段遥操作/录制中实现。
