"""语音控制模块 — 基于 whisper.cpp 的端侧语音识别"""
import subprocess
import json
import tempfile
import os
import time
from pathlib import Path

WHISPER_BIN = "/usr/bin/whisper"
WHISPER_MODEL = "/usr/local/share/whisper-base.bin"
RECORD_DEVICE = "hw:rockchipnau8822,0"

# 常用繁体→简体映射
TRAD2SIMP = str.maketrans({
    "紅": "红", "綠": "绿", "藍": "蓝", "黃": "黄", "黑": "黑", "白": "白",
    "塊": "块", "筆": "笔", "書": "书", "體": "体", "機": "机", "械": "械",
    "個": "个", "問": "问", "題": "题", "說": "说", "話": "话", "講": "讲",
    "時": "时", "間": "间", "確": "确", "認": "认", "會": "会", "聲": "声",
    "音": "音", "顏": "颜", "色": "色", "標": "标", "準": "准", "簡": "简",
    "單": "单", "繁": "繁", "轉": "转", "換": "换", "對": "对", "應": "应",
    "動": "动", "作": "作", "指": "指", "令": "令", "識": "识", "別": "别",
    "開": "开", "始": "始", "結": "结", "束": "束", "關": "关", "閉": "闭",
    "連": "连", "接": "接", "斷": "断", "輸": "输", "出": "出", "入": "入",
})


def _trad2simp(text: str) -> str:
    return text.translate(TRAD2SIMP)


class VoiceControl:
    """语音识别 + 指令解析"""

    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self._ready = os.path.exists(WHISPER_BIN) and os.path.exists(WHISPER_MODEL)

    def listen(self, audio_path: str | None = None) -> str:
        """识别语音，返回简体文字"""
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
            "--no-prints",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        text = result.stdout.strip()
        return _trad2simp(text)

    def parse_command(self, text: str) -> dict | None:
        """解析语音指令，提取目标物体和动作"""
        text = text.lower().strip()

        # 动作识别
        action = "grasp"
        if any(w in text for w in ["放", "放回", "放置", "place", "put"]):
            action = "place"

        # 颜色识别
        color_map = {
            "红色": ["红色", "红", "red"],
            "绿色": ["绿色", "绿", "green"],
            "蓝色": ["蓝色", "蓝", "blue"],
            "黄色": ["黄色", "黄", "yellow"],
            "白色": ["白色", "白", "white"],
            "黑色": ["黑色", "黑", "black"],
            "橙色": ["橙色", "橙", "orange"],
            "紫色": ["紫色", "紫", "purple"],
        }
        color = None
        for cn, keywords in color_map.items():
            if any(k in text for k in keywords):
                color = cn
                break

        # 常见物体关键字
        objects = {
            "杯子": ["杯子", "杯", "cup", "mug"],
            "瓶子": ["瓶子", "瓶", "bottle", "水瓶"],
            "手机": ["手机", "phone"],
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

    def record(self, duration: int = 3, output_path: str = "/tmp/vla_voice.wav") -> str:
        """实时录音，返回音频文件路径"""
        if not self._ready:
            raise RuntimeError("whisper.cpp 未安装")
        cmd = [
            "arecord", "-D", RECORD_DEVICE,
            "-d", str(duration),
            "-f", "cd", "-t", "wav",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=duration + 5)
        return output_path if os.path.exists(output_path) else ""

    def record_and_parse(self, duration: int = 3) -> dict | None:
        """实时录音 → 识别 → 解析，一步完成"""
        path = self.record(duration)
        if not path:
            return None
        return self.listen_and_parse(path)

    def listen_and_parse(self, audio_path: str) -> dict | None:
        """一步完成：识别 → 解析"""
        text = self.listen(audio_path)
        if not text:
            return None
        return self.parse_command(text)
