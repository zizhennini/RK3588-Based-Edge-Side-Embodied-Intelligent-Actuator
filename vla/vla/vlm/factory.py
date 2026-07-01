"""VLM 工厂 — 通过配置字符串创建具体引擎"""
from .base import VLMBase
from .internvl2 import InternVL2Engine
from .qwen_vl import QwenVLEngine
from .qwen3_vl import Qwen3VLEngine
from .smolvlm2 import SmolVLM2Engine
from .smolvlm2_500m import SmolVLM2_500MEngine


VLM_REGISTRY = {
    "internvl2": InternVL2Engine,
    "internvl2-1b": InternVL2Engine,
    "qwen2.5-vl": QwenVLEngine,
    "qwen2.5-vl-3b": QwenVLEngine,
    "qwen-vl": QwenVLEngine,
    "qwen3-vl": Qwen3VLEngine,
    "qwen3-vl-2b": Qwen3VLEngine,
    "smolvlm2": SmolVLM2Engine,
    "smolvlm2-256m": SmolVLM2Engine,
    "smolvlm-500m": SmolVLM2_500MEngine,
    "smolvlm2-500m": SmolVLM2_500MEngine,
}


def create_vlm(
    model_name: str,
    rkllm_bin: str = "rkllm_demo",
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
    if hasattr(engine, "set_rkllm_bin"):
        engine.set_rkllm_bin(rkllm_bin)
    return engine
