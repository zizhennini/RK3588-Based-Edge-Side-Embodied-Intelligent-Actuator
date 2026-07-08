"""Qwen3.5 VLM — 直接调 demo（已验证可靠）+ 结构化输出解析"""
import re
import os
import subprocess
from .base import VLMBase, VLMResult

MD = "/home/elf/work/RK3588-EIA/models/Qwen3.5-0.8B"


# 颜色关键词表（用于从中文文本中提取颜色）
COLOR_KEYWORDS = [
    "红色", "粉色", "橙色", "黄色", "绿色", "蓝色", "紫色",
    "黑色", "白色", "灰色", "棕色", "金色", "银色",
    "深红", "深蓝", "深绿", "浅红", "浅蓝", "浅绿",
    "透明", "彩色",
]


def _extract_color(text: str) -> str:
    """从文本中提取第一个出现的颜色关键词"""
    for c in COLOR_KEYWORDS:
        if c in text:
            return c
    # 尝试匹配英文颜色
    en_map = {
        "red": "红色", "pink": "粉色", "orange": "橙色",
        "yellow": "黄色", "green": "绿色", "blue": "蓝色",
        "purple": "紫色", "black": "黑色", "white": "白色",
        "gray": "灰色", "grey": "灰色", "brown": "棕色",
        "gold": "金色", "silver": "银色",
    }
    for en, zh in en_map.items():
        if en in text.lower():
            return zh
    return ""


FILLER_PATTERNS = [
    r'的(?:名称|名字|颜色)?是', r'位于', r'在画面', r'画面[中里]',
    r'它是', r'这是一个', r'那里有', r'看起来是',
]


def _extract_object(text: str) -> str:
    """从文本中提取物体名称（去掉颜色前缀和填充词后的主体）"""
    for c in COLOR_KEYWORDS:
        text = text.replace(c, "")
    for pat in FILLER_PATTERNS:
        text = re.sub(pat, "", text)
    # 取第一个有意义的词作为物体名
    parts = re.split(r'[，。,.\s]+', text.strip())
    for p in parts:
        p = p.strip()
        if p and len(p) <= 6 and not p.isdigit():
            return p
    return parts[0] if parts else text[:8]


def _extract_coords(text: str) -> tuple:
    """尝试从文本中提取图像坐标 (cx, cy)

    支持的格式：
      - bbox: [x1,y1,x2,y2] → 计算中心
      - coords: (cx, cy)
    """
    # 匹配 bbox 格式: [x1,y1,x2,y2] 或 (x1,y1,x2,y2)
    m = re.search(r'[\[\(](\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)[\]\)]', text)
    if m:
        x1, y1, x2, y2 = map(int, m.groups())
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    # 匹配坐标格式: (cx, cy)
    m = re.search(r'[\[\(](\d+)\s*[,，]\s*(\d+)[\]\)]', text)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    return (None, None)


# 结构化提示词模板
SYSTEM_PROMPT = """请仔细观察画面，用以下格式回答（不要有多余的解释）：

颜色: <物体主要颜色>
物体: <物体名称>
位置: <在画面中描述物体的位置，如"画面中央偏左">

如果画面中没有明显的目标物体，请回答：无法识别"""

STRUCTURED_PROMPT = """<image>请仔细观察画面中的物体，按以下格式回答：

颜色: 红色/蓝色/绿色/黄色/橙色/紫色/粉色/白色/黑色/棕色/灰色/金色/银色/透明
物体: 物体的具体名称（如"杯子"、"方块"、"玩偶"等）
位置: 物体在画面中的位置描述

如果画面中没有明显物体，回答：无法识别"""


class Qwen3VLEngine(VLMBase):
    def load(self, model_path: str, demo_bin: str | None = None):
        pass

    def infer(self, image_path: str, prompt: str | None = None) -> VLMResult:
        text = (prompt or STRUCTURED_PROMPT) + "\nexit\n"
        env = {**os.environ,
               "LD_LIBRARY_PATH": f"{MD}/lib",
               "RKLLM_LOG_LEVEL": "0"}
        cmd = [f"{MD}/demo", os.path.abspath(image_path),
               f"{MD}/Qwen3.5-0.8B_vision_rk3588.rknn",
               f"{MD}/Qwen3.5-0.8B_w8a8_rk3588.rkllm",
               "512", "2048", "3", "rk3588",
               "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
        try:
            r = subprocess.run(cmd, input=text, capture_output=True, text=True,
                              timeout=120, env=env, cwd=MD)
            raw = r.stdout
        except subprocess.TimeoutExpired as e:
            raw = ""
            if e.stdout:
                raw = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
        except Exception as e:
            raw = str(e)
        return self._parse(raw)

    def _parse(self, raw: str) -> VLMResult:
        # 提取 robot: 之后的回答
        ans = ""
        i = raw.find("robot:")
        if i >= 0:
            after = raw[i + 6:]
            j = after.find("user:")
            ans = after[:j].strip() if j >= 0 else after.strip()

        # 过滤日志行
        lines = [l for l in ans.split("\n") if l.strip()
                 and not l.strip().startswith(("I rkllm:", "rkllm", "main:"))
                 and l.strip() not in ("robot:", "user:")]
        clean_ans = "\n".join(lines)

        # === 尝试解析 JSON 格式 bbox 输出 ===
        # 格式: {"bbox_2d": [x1,y1,x2,y2], "label": "红色色块"}
        import json as _json
        json_match = re.search(r'\{.*"bbox_2d".*\}', clean_ans, re.DOTALL)
        if json_match:
            try:
                obj = _json.loads(json_match.group(0))
                label = obj.get("label", "")
                bbox = obj.get("bbox_2d", [])
                cx, cy = None, None
                if len(bbox) == 4:
                    cx = (bbox[0] + bbox[2]) // 2
                    cy = (bbox[1] + bbox[3]) // 2
                color = _extract_color(label)
                obj_name = _extract_object(label)
                if not obj_name and label:
                    obj_name = label[:8]
                return VLMResult(color=color, object=obj_name,
                                 cx=cx, cy=cy, raw=clean_ans)
            except Exception:
                pass

        # === 结构化字段提取 ===

        # 1. 提取颜色
        color = ""
        m = re.search(r'颜色[：:]\s*(.+)', clean_ans)
        if m:
            color = m.group(1).strip().rstrip("，。,.")
        if not color:
            color = _extract_color(clean_ans)

        # 2. 提取物体名称
        obj_name = ""
        m = re.search(r'物体[：:]\s*(.+)', clean_ans)
        if m:
            obj_name = m.group(1).strip().rstrip("，。,.;;")
        if not obj_name:
            obj_name = _extract_object(clean_ans)

        # 3. 提取坐标（从完整文本中搜索）
        cx, cy = _extract_coords(clean_ans)

        # 4. 检查是否无法识别
        if "无法识别" in clean_ans or "没有" in clean_ans and "物体" in clean_ans:
            return VLMResult(color="", object="", raw=clean_ans)

        return VLMResult(color=color, object=obj_name,
                         cx=cx, cy=cy, raw=clean_ans)

    def unload(self):
        pass
