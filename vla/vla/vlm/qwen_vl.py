"""Qwen2.5-VL-3B RKLLM 部署实现"""
import json
import subprocess
from .base import VLMBase, VLMResult


class QwenVLEngine(VLMBase):
    def __init__(self, rkllm_bin: str = "rkllm_demo"):
        self.rkllm_bin = rkllm_bin
        self.model_path = ""
        self.default_prompt = (
            "你是一个机械臂抓取系统。根据图像回答问题。\n"
            "输出JSON格式：{\"color\": \"红色\", \"object\": \"方块\"}\n"
        )

    def load(self, model_path: str):
        self.model_path = model_path

    def infer(self, image_path: str, prompt: str | None = None) -> VLMResult:
        cmd = [
            self.rkllm_bin,
            "--model", self.model_path,
            "--prompt", prompt or self.default_prompt,
            "--image", image_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        raw = result.stdout.strip()
        return self._parse(raw)

    def _parse(self, raw: str) -> VLMResult:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            return VLMResult(
                color=data.get("color", "红色"),
                object=data.get("object", "方块"),
                raw=raw,
            )
        except (ValueError, json.JSONDecodeError):
            return VLMResult(color="红色", object="方块", raw=raw)

    def unload(self):
        self.model_path = ""
