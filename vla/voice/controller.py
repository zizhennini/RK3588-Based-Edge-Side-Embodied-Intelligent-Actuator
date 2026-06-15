"""语音控制模块 — 基于 whisper.cpp 的端侧语音识别"""
import subprocess
import json
import tempfile
import os
from pathlib import Path

WHISPER_BIN = "/usr/bin/whisper"
WHISPER_MODEL = "/usr/local/share/whisper-tiny.bin"


class VoiceControl:
    """语音识别 + 指令解析"""

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self._ready = os.path.exists(WHISPER_BIN) and os.path.exists(WHISPER_MODEL)

    def listen(self, audio_path: str | None = None) -> str:
        """识别语音，返回文字"""
        if not self._ready:
            raise RuntimeError("whisper.cpp 未安装，请运行 scripts/install_whisper.sh")

        if audio_path and not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        cmd = [
            WHISPER_BIN,
            "-m", WHISPER_MODEL,
            "-f", audio_path,
            "-l", self.lang,
            "--no-timestamps",
            "--print-progress", "false",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        lines = [ln.strip() for ln in result.stdout.strip().split("\n") if ln.strip()]
        text = lines[-1] if lines else ""
        return text

    def parse_command(self, text: str) -> dict | None:
        """解析语音指令，提取目标物体和动作"""
        text = text.lower().strip()

        # 动作识别
        action = "grasp"
        if any(w in text for w in ["放", "放回", "放置", "place", "put"]):
            action = "place"

        # 颜色识别
        colors = ["红色", "绿色", "蓝色", "黄色", "白色", "黑色", "橙色", "紫色"]
        color = None
        for c in colors:
            if c in text:
                color = c
                break
        # 英文颜色
        en_colors = {"red": "红色", "green": "绿色", "blue": "蓝色",
                     "yellow": "黄色", "white": "白色", "black": "黑色"}
        for en, cn in en_colors.items():
            if en in text:
                color = cn
                break

        # 常见物体关键字
        objects = {
            "杯子": ["杯子", "杯", "cup", "mug"],
            "瓶子": ["瓶子", "瓶", "bottle", "水瓶"],
            "手机": ["手机", "phone", "cellphone"],
            "方块": ["方块", "块", "积木", "block", "cube"],
            "球": ["球", "ball"],
            "笔": ["笔", "pen"],
            "书": ["书", "book"],
        }
        obj = None
        for name, keywords in objects.items():
            if any(k in text for k in keywords):
                obj = name
                break

        if not color and not obj:
            return None

        return {"action": action, "color": color, "object": obj, "raw": text}

    def listen_and_parse(self, audio_path: str) -> dict | None:
        """一步完成：录音 → 识别 → 解析"""
        text = self.listen(audio_path)
        if not text:
            return None
        return self.parse_command(text)
