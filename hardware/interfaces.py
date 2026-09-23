"""核心数据结构和模块接口定义"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Observation:
    """感知层 -> 策略层 的标准输入"""
    rgb: np.ndarray          # (480, 640, 3) uint8
    depth: np.ndarray        # (480, 640) float32, 单位: 米
    state: np.ndarray        # (6,) 关节角度 (rad)
    timestamp: float         # 采集时间戳 (time.time())


@dataclass
class Action:
    """策略层 -> 硬件层 的标准输出"""
    positions: np.ndarray    # (6,) 目标关节角度 (rad)
    gripper: float           # 夹爪开合度 [0=全闭, 1=全开]
    execution_time: float    # 预计执行时间 (s)


@dataclass
class TaskRequest:
    """应用层 -> 策略层 的任务请求"""
    type: str                # "grasp" | "move" | "ask" | "replay"
    target: str              # 目标描述 ("红色杯子")
    bbox: Optional[tuple] = None  # VLM 检测框 (x1, y1, x2, y2) 归一化 [0,1]
    params: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    """策略层 -> 应用层 的任务结果"""
    success: bool
    message: str
    data: dict = field(default_factory=dict)


class Module(ABC):
    """所有模块的统一生命周期接口（含降级契约）"""

    @abstractmethod
    def setup(self, config: dict) -> None:
        """初始化模块（加载模型/配置）"""
        ...

    @abstractmethod
    def start(self) -> None:
        """启动模块（开始线程/子进程）"""
        ...

    @abstractmethod
    def stop(self) -> None:
        """停止模块（释放资源）"""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """返回模块是否可用"""
        ...

    @abstractmethod
    def on_failure(self) -> str:
        """返回降级策略: "skip" | "retry" | "fallback" | "abort"
        System 根据此策略决定模块启动失败后的行为"""
        ...


class PerceptionModule(Module):
    """感知层接口"""
    @abstractmethod
    def detect(self, obs: Observation) -> dict:
        """返回检测结果 {"bbox": ..., "label": ..., "confidence": ...}"""
        ...


class PolicyModule(Module):
    """策略层接口"""
    @abstractmethod
    def predict(self, obs: Observation) -> Action:
        """根据观测返回动作"""
        ...


class HardwareModule(Module):
    """硬件层接口"""
    @abstractmethod
    def execute(self, action: Action) -> bool:
        """执行动作，返回是否成功"""
        ...

    @abstractmethod
    def get_observation(self) -> Observation:
        """获取当前观测"""
        ...
