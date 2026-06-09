"""VLM 抽象基类 — 支持多模型接入"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VLMResult:
    color: str
    object: str
    cx: int | None = None
    cy: int | None = None
    raw: str = ""


class VLMBase(ABC):
    """所有 VLM 引擎的统一接口"""

    @abstractmethod
    def load(self, model_path: str):
        ...

    @abstractmethod
    def infer(self, image_path: str, prompt: str | None = None) -> VLMResult:
        ...

    @abstractmethod
    def unload(self):
        ...
