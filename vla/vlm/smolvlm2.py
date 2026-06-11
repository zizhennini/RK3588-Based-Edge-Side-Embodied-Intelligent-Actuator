"""SmolVLM-256M RKLLM 部署实现 — 文本描述 + 颜色定位"""
import re
import subprocess
import numpy as np
from .base import VLMBase, VLMResult


COLORS = {
    "红色": ["红", "red"], "绿色": ["绿", "green"], "蓝色": ["蓝", "blue"],
    "黄色": ["黄", "yellow"], "白色": ["白", "white", "白色"], "黑色": ["黑", "black"],
    "橙色": ["橙", "orange"], "紫色": ["紫", "purple"], "灰色": ["灰", "gray"],
}


class SmolVLM2Engine(VLMBase):
    def __init__(self):
        self.demo_bin = "demo"
        self.encoder_path = ""
        self.llm_path = ""
        self.max_new_tokens = 64
        self.max_context_len = 2048
        self.rknn_core_num = 3
        self.img_start = "<image>"
        self.img_end = ""
        self.img_pad = "<image>"

    def set_demo_bin(self, path: str):
        self.demo_bin = path

    def load(self, model_path: str):
        import glob, os
        rknn_files = glob.glob(os.path.join(model_path, "*.rknn"))
        rkllm_files = glob.glob(os.path.join(model_path, "*.rkllm"))
        if not rknn_files or not rkllm_files:
            raise FileNotFoundError(f"在 {model_path} 中未找到模型文件")
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
        stdin_input = (prompt or "1") + "\n"
        try:
            result = subprocess.run(cmd, input=stdin_input,
                                    capture_output=True, text=True, timeout=30)
            raw = result.stdout
        except subprocess.TimeoutExpired as e:
            raw = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return self._parse(raw)

    def _parse(self, raw: str) -> VLMResult:
        m = re.search(r'robot:\s*(.*?)<end_of_utterance>', raw, re.DOTALL)
        text = m.group(1).strip().lower() if m else ""

        color = "红色"
        object_name = text[:20] if text else "物体"

        for cn_name, en_names in COLORS.items():
            for name in en_names:
                if name in text:
                    color = cn_name
                    break

        return VLMResult(color=color, object=object_name, cx=None, cy=None, raw=raw)

    def unload(self):
        self.encoder_path = ""
        self.llm_path = ""
