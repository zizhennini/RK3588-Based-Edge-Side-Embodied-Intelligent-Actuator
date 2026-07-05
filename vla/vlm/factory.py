"""VLM 工厂 — 通过配置字符串创建具体引擎"""
from .base import VLMBase
from .qwen3_vl import Qwen3VLEngine


VLM_REGISTRY = {
    "qwen3-vl": Qwen3VLEngine,
    "qwen3-vl-2b": Qwen3VLEngine,
    "qwen3.5": Qwen3VLEngine,
    "qwen3.5-0.8b": Qwen3VLEngine,
}


def create_vlm(
    model_name: str,
    demo_bin: str = "demo",
) -> VLMBase:
    if model_name not in VLM_REGISTRY:
        raise ValueError(
            f"不支持的 VLM 模型: {model_name}，可选: {list(VLM_REGISTRY.keys())}"
        )
    cls = VLM_REGISTRY[model_name]
    engine = cls()
    if hasattr(engine, "set_demo_bin"):
        engine.set_demo_bin(demo_bin)
    return engine
