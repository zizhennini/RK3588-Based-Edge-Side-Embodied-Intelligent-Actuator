"""Qwen3-VL-2B RKLLM 多模态部署实现 — VLM 直出坐标"""
import json
import subprocess
from .base import VLMBase, VLMResult


class Qwen3VLEngine(VLMBase):
    def __init__(self):
        self.demo_bin = "demo"
        self.encoder_path = ""
        self.llm_path = ""
        self.max_new_tokens = 48
        self.max_context_len = 2048
        self.rknn_core_num = 3
        self.img_start = "<|vision_start|>"
        self.img_end = "<|vision_end|>"
        self.img_pad = "<|image_pad|>"
        self.default_prompt = (
            '<image>输出JSON格式: {"color":"红色","object":"杯子","cx":320,"cy":240}'
            " cx,cy是物体在图像中的像素坐标(0-640,0-480)"
        )

    def set_demo_bin(self, path: str):
        self.demo_bin = path

    def load(self, model_path: str):
        import glob, os
        rknn_files = glob.glob(os.path.join(model_path, "*.rknn"))
        rkllm_files = glob.glob(os.path.join(model_path, "*.rkllm"))
        if not rknn_files or not rkllm_files:
            raise FileNotFoundError(f"在 {model_path} 中未找到 .rknn 或 .rkllm 文件")
        self.encoder_path = rknn_files[0]
        self.llm_path = rkllm_files[0]

    def infer(self, image_path: str, prompt: str | None = None) -> VLMResult:
        cmd = [
            self.demo_bin, image_path,
            self.encoder_path, self.llm_path,
            str(self.max_new_tokens), str(self.max_context_len),
            str(self.rknn_core_num),
            self.img_start, self.img_end, self.img_pad,
        ]
        stdin_input = (prompt or self.default_prompt) + "\n"
        try:
            result = subprocess.run(cmd, input=stdin_input,
                                    capture_output=True, text=True, timeout=20)
            raw = result.stdout.strip()
        except subprocess.TimeoutExpired as e:
            raw = (e.stdout or b"").decode().strip() if isinstance(e.stdout, bytes) \
                  else (e.stdout or "").strip()
        return self._parse(raw)

    def _parse(self, raw: str) -> VLMResult:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            return VLMResult(
                color=data.get("color", "红色"),
                object=data.get("object", "方块"),
                cx=data.get("cx"),
                cy=data.get("cy"),
                raw=raw,
            )
        except (ValueError, json.JSONDecodeError):
            return VLMResult(color="红色", object="方块", raw=raw)

    def unload(self):
        self.encoder_path = ""
        self.llm_path = ""
