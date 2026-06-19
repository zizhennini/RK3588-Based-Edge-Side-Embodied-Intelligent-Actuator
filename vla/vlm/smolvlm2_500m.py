"""SmolVLM2-500M VQA 微调版 — 直出 JSON 坐标"""
import json
import subprocess
from .base import VLMBase, VLMResult


COLOR_MAP = {
    "red": "红色", "green": "绿色", "blue": "蓝色",
    "yellow": "黄色", "white": "白色", "black": "黑色",
    "orange": "橙色", "purple": "紫色", "gray": "灰色",
    "红": "红色", "绿": "绿色", "蓝": "蓝色",
    "黄": "黄色", "白": "白色", "黑": "黑色",
}


def _parse_color(raw: str) -> str:
    """解析颜色，支持英文名和 #ff0000 格式"""
    raw = raw.lower().strip().replace(" ", "")
    # 十六进制 #ff0000 → red
    hex_map = {
        "#ff": "红色", "#00ff00": "绿色", "#0000ff": "蓝色",
        "#ffff00": "黄色", "#ffffff": "白色", "#000000": "黑色",
        "#ffa500": "橙色", "#800080": "紫色", "#808080": "灰色",
    }
    for hex_code, cn in hex_map.items():
        if raw.startswith(hex_code):
            return cn
    # 英文名匹配
    for en, cn in COLOR_MAP.items():
        if en in raw:
            return cn
    return "红色"


class SmolVLM2_500MEngine(VLMBase):
    def __init__(self):
        self.demo_bin = "demo"
        self.encoder_path = ""
        self.llm_path = ""
        self.max_new_tokens = 128
        self.max_context_len = 2048
        self.rknn_core_num = 3
        self.img_start = "<image>"
        self.img_end = ""
        self.img_pad = "<image>"
        self.default_prompt = (
            "What color is the object? Output JSON with color, objectName, and centerPosition [cx,cy]"
        )

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
        stdin_input = (prompt or self.default_prompt) + "\n"
        try:
            result = subprocess.run(cmd, input=stdin_input,
                                    capture_output=True, text=True, timeout=30)
            raw = result.stdout
        except subprocess.TimeoutExpired as e:
            raw = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return self._parse(raw)

    def _parse(self, raw: str) -> VLMResult:
        import re
        m = re.search(r'robot:\s*(.*?)<end_of_utterance>', raw, re.DOTALL)
        text = m.group(1).strip() if m else raw.strip()

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return VLMResult(color="红色", object="物体", cx=None, cy=None, raw=raw)

        color = _parse_color(data.get("color", ""))

        object_name = str(data.get("objectName", data.get("object", "物体")))

        cx, cy = None, None
        pos = data.get("centerPosition", data.get("position", data.get("cxcy")))
        if isinstance(pos, list) and len(pos) >= 2:
            cx, cy = int(pos[0]), int(pos[1])

        return VLMResult(color=color, object=object_name, cx=cx, cy=cy, raw=raw)

    def unload(self):
        self.encoder_path = ""
        self.llm_path = ""
