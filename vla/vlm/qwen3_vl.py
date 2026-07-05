"""Qwen3.5 VLM — 直接调 demo（已验证可靠）"""
import re, os, subprocess
from .base import VLMBase, VLMResult

MD = "/home/elf/work/RK3588-EIA/models/Qwen3.5-0.8B"


class Qwen3VLEngine(VLMBase):
    def load(self, model_path: str, demo_bin: str | None = None):
        pass

    def infer(self, image_path: str, prompt: str | None = None) -> VLMResult:
        text = (prompt or "<image>画面中有什么？") + "\nexit\n"
        env = {**os.environ, "LD_LIBRARY_PATH": f"{MD}/lib", "RKLLM_LOG_LEVEL": "0"}
        cmd = [f"{MD}/demo", os.path.abspath(image_path),
               f"{MD}/Qwen3.5-0.8B_vision_rk3588.rknn",
               f"{MD}/Qwen3.5-0.8B_w8a8_rk3588.rkllm",
               "128", "2048", "3", "rk3588",
               "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
        try:
            r = subprocess.run(cmd, input=text, capture_output=True, text=True,
                              timeout=120, env=env, cwd=MD)
            raw = r.stdout
        except subprocess.TimeoutExpired as e:
            raw = (e.stdout or "").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        except Exception as e:
            raw = str(e)
        return self._parse(raw)

    def _parse(self, raw: str) -> VLMResult:
        ans = ""
        i = raw.find("robot:")
        if i >= 0:
            after = raw[i + 6:]
            j = after.find("user:")
            ans = after[:j].strip() if j >= 0 else after.strip()
        lines = [l for l in ans.split("\n") if l.strip()
                 and not l.strip().startswith(("I rkllm:", "rkllm", "main:"))
                 and l.strip() not in ("robot:", "user:")]
        ans = "\n".join(lines)
        return VLMResult(color="", object="", raw=ans)

    def unload(self):
        pass
