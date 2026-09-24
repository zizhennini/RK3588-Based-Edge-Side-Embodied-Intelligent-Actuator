"""Qwen3.5 VLM 感知模块 -- 开放词汇目标检测

封装 RKLLM demo 子进程调用，实现 PerceptionModule 接口。
支持：
  - 文本描述 → bbox 检测（开放词汇）
  - bbox 归一化兼容（绝对像素 / [0,1]）
  - 闲置自动卸载（释放 ~900MB NPU 内存）
  - 按需冷启动
"""
import os
import gc
import re
import json
import time
import logging
import subprocess
import threading
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from hardware.interfaces import PerceptionModule, Observation

logger = logging.getLogger(__name__)

# ─── 颜色关键词表 ────────────────────────────────────────────────────────────
COLOR_KEYWORDS = [
    "红色", "粉色", "橙色", "黄色", "绿色", "蓝色", "紫色",
    "黑色", "白色", "灰色", "棕色", "金色", "银色",
    "深红", "深蓝", "深绿", "浅红", "浅蓝", "浅绿",
    "透明", "彩色",
]

# ─── 默认路径（板端） ─────────────────────────────────────────────────────────
_DEFAULT_MODEL_DIR = "/home/elf/work/rk3588-eia/models/vlm/Qwen3.5-0.8B"
_DEFAULT_DEMO_BIN = "demo"
_DEFAULT_RKNN_MODEL = "Qwen3.5-0.8B_vision_rk3588.rknn"
_DEFAULT_RKLLM_MODEL = "Qwen3.5-0.8B_w8a8_rk3588.rkllm"

# 临时图片路径（板端 /tmp）
_TMP_IMAGE_PATH = "/tmp/vlm_detect_frame.jpg"


class VLMPerception(PerceptionModule):
    """Qwen3.5 VLM 目标检测 -- 子进程调用 RKLLM demo

    生命周期：
        setup(config) → start() → detect(obs) ... → stop()

    检测流程：
        1. RGB numpy → BGR → imwrite 临时文件
        2. 构造 prompt（含目标描述）
        3. subprocess 调用 RKLLM demo（NPU 推理）
        4. 解析输出 → bbox JSON / 正则提取
        5. bbox 归一化到 [0,1]
        6. 返回结构化检测结果
    """

    # 结构化检测提示词模板
    DETECT_PROMPT_TEMPLATE = (
        '<image>\n'
        '请检测图中"{target}"的位置。\n'
        '输出格式: {{"bbox_2d": [x1, y1, x2, y2], "label": "物体名称"}}\n'
        '仅输出 JSON，不要其他文字。'
    )

    # 无特定目标时的通用检测提示词
    GENERAL_DETECT_PROMPT = (
        '<image>\n'
        '请仔细观察画面中的主要物体，按以下格式输出：\n'
        '{{"bbox_2d": [x1, y1, x2, y2], "label": "物体名称"}}\n'
        '仅输出 JSON，不要其他文字。\n'
        '如果画面中没有明显物体，输出: {{"bbox_2d": [], "label": "无"}}'
    )

    def __init__(self):
        # 模型路径配置
        self._model_dir: str = _DEFAULT_MODEL_DIR
        self._demo_bin: str = _DEFAULT_DEMO_BIN
        self._rknn_model: str = _DEFAULT_RKNN_MODEL
        self._rkllm_model: str = _DEFAULT_RKLLM_MODEL

        # 推理参数
        self._max_new_tokens: int = 512
        self._max_context_len: int = 2048
        self._n_threads: int = 3
        self._platform: str = "rk3588"
        self._timeout: float = 120.0

        # 图像尺寸
        self._image_width: int = 640
        self._image_height: int = 480

        # 状态跟踪
        self._loaded: bool = False
        self._last_infer_time: float = 0.0
        self._idle_timeout: float = 30.0
        self._infer_count: int = 0
        self._total_infer_ms: float = 0.0

        # 闲置卸载
        self._unloader_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock: threading.Lock = threading.Lock()

        # 可用性
        self._available: bool = False

    # ─── Module 接口实现 ──────────────────────────────────────────────────────

    def setup(self, config: dict) -> None:
        """初始化 VLM 模块配置

        Args:
            config: 配置字典，支持以下 key:
                model_dir: str       模型目录路径
                demo_bin: str        demo 可执行文件名或绝对路径
                rknn_model: str      RKNN 模型文件名
                rkllm_model: str     RKLLM 模型文件名
                idle_timeout: float  闲置卸载超时(s)，默认 30
                image_width: int     图像宽度，默认 640
                image_height: int    图像高度，默认 480
                timeout: float       子进程超时(s)，默认 120
                max_new_tokens: int  最大生成 token 数，默认 512
        """
        self._model_dir = config.get("model_dir", self._model_dir)
        self._demo_bin = config.get("demo_bin", self._demo_bin)
        self._rknn_model = config.get("rknn_model", self._rknn_model)
        self._rkllm_model = config.get("rkllm_model", self._rkllm_model)
        self._idle_timeout = config.get("idle_timeout", self._idle_timeout)
        self._image_width = config.get("image_width", self._image_width)
        self._image_height = config.get("image_height", self._image_height)
        self._timeout = config.get("timeout", self._timeout)
        self._max_new_tokens = config.get("max_new_tokens", self._max_new_tokens)

        # 检查 demo 二进制是否存在
        demo_path = self._get_demo_path()
        self._available = os.path.isfile(demo_path) and os.access(demo_path, os.X_OK)

        if self._available:
            logger.info(f"VLM 模块配置完成: model_dir={self._model_dir}")
        else:
            logger.warning(
                f"VLM demo 不可用: {demo_path} "
                f"(exists={os.path.isfile(demo_path)}, "
                f"executable={os.access(demo_path, os.X_OK) if os.path.isfile(demo_path) else 'N/A'})"
            )

    def start(self) -> None:
        """启动闲置卸载后台线程"""
        if self._running:
            return
        self._running = True
        self._unloader_thread = threading.Thread(
            target=self._idle_unload_loop,
            name="VLM-IdleUnloader",
            daemon=True,
        )
        self._unloader_thread.start()
        logger.info(f"VLM 闲置卸载线程已启动 (timeout={self._idle_timeout}s)")

    def stop(self) -> None:
        """停止后台线程并释放资源"""
        self._running = False
        if self._unloader_thread is not None:
            self._unloader_thread.join(timeout=10.0)
            self._unloader_thread = None
        self._unload()
        logger.info(f"VLM 模块已停止 (总推理 {self._infer_count} 次)")

    @property
    def is_available(self) -> bool:
        """VLM 是否可用（demo_bin 存在且可执行）"""
        return self._available

    def on_failure(self) -> str:
        """VLM 推理失败可重试（冷启动可能偶尔超时）"""
        return "retry"

    # ─── 核心检测方法 ─────────────────────────────────────────────────────────

    def detect(self, obs: Observation, target_desc: str = "") -> dict:
        """检测目标物体位置

        Args:
            obs: Observation 数据（含 rgb 480x640x3 uint8）
            target_desc: 目标描述文本（如 "红色杯子"），为空时检测最显著物体

        Returns:
            dict: {
                "bbox": (x1, y1, x2, y2),  # 归一化 [0,1]
                "label": str,               # 物体标签
                "confidence": float,        # 置信度（VLM 无显式分数，成功=0.8）
                "center_px": (cx, cy),      # 绝对像素中心坐标
                "raw": str,                 # VLM 原始输出文本
            }
            检测失败时返回 {"bbox": None, "label": "", "confidence": 0.0,
                          "center_px": None, "raw": ..., "error": ...}
        """
        if not self._available:
            return self._fail_result("VLM demo 不可用")

        try:
            # 1. RGB → BGR → 写入临时文件
            image_path = self._save_frame(obs.rgb)

            # 2. 构造 prompt
            prompt = self._build_prompt(target_desc)

            # 3. 调用 RKLLM demo 推理
            self._ensure_loaded()
            t0 = time.perf_counter()
            raw_output = self._run_inference(image_path, prompt)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # 4. 更新统计
            self._last_infer_time = time.time()
            self._infer_count += 1
            self._total_infer_ms += elapsed_ms

            # 5. 解析输出
            result = self._parse_output(raw_output)
            result["infer_ms"] = elapsed_ms

            logger.debug(
                f"VLM 检测完成: label={result.get('label')}, "
                f"bbox={result.get('bbox')}, 耗时={elapsed_ms:.0f}ms"
            )
            return result

        except subprocess.TimeoutExpired:
            logger.error(f"VLM 推理超时 (>{self._timeout}s)")
            return self._fail_result("推理超时")
        except Exception as e:
            logger.error(f"VLM 检测异常: {e}", exc_info=True)
            return self._fail_result(str(e))

    # ─── 内部方法 ─────────────────────────────────────────────────────────────

    def _get_demo_path(self) -> str:
        """获取 demo 二进制的完整路径"""
        if os.path.isabs(self._demo_bin):
            return self._demo_bin
        return str(Path(self._model_dir) / self._demo_bin)

    def _save_frame(self, rgb: np.ndarray) -> str:
        """将 RGB numpy 数组保存为临时 BGR JPEG 文件

        Args:
            rgb: (H, W, 3) uint8 RGB 图像

        Returns:
            临时文件路径
        """
        # RGB → BGR（OpenCV 使用 BGR）
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # 更新实际图像尺寸（用于 bbox 归一化）
        h, w = bgr.shape[:2]
        self._image_height = h
        self._image_width = w
        # 写入临时文件
        cv2.imwrite(_TMP_IMAGE_PATH, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return _TMP_IMAGE_PATH

    def _build_prompt(self, target_desc: str) -> str:
        """构造检测提示词"""
        if target_desc.strip():
            return self.DETECT_PROMPT_TEMPLATE.format(target=target_desc)
        return self.GENERAL_DETECT_PROMPT

    def _ensure_loaded(self) -> None:
        """确保 VLM 处于可用状态（标记为已加载）

        RKLLM demo 是单次子进程调用模式，无需显式 load/unload 进程。
        但保留 loaded 状态以支持闲置检测逻辑。
        """
        with self._lock:
            if not self._loaded:
                self._loaded = True
                logger.debug("VLM 标记为已加载（首次推理冷启动）")

    def _unload(self) -> None:
        """卸载 VLM（标记为未加载，触发 GC）"""
        with self._lock:
            if self._loaded:
                self._loaded = False
                gc.collect()
                logger.info("VLM 已卸载（释放内存标记）")

    def _run_inference(self, image_path: str, prompt: str) -> str:
        """调用 RKLLM demo 子进程执行推理

        Args:
            image_path: 输入图片绝对路径
            prompt: 提示词文本

        Returns:
            demo 进程的 stdout 原始文本

        Raises:
            subprocess.TimeoutExpired: 推理超时
            FileNotFoundError: demo 二进制不存在
        """
        model_dir = Path(self._model_dir)
        demo_path = self._get_demo_path()
        rknn_path = str(model_dir / self._rknn_model)
        rkllm_path = str(model_dir / self._rkllm_model)

        # 环境变量
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(model_dir / "lib")
        env["RKLLM_LOG_LEVEL"] = "0"

        # 构造命令（与现有 Qwen3VLEngine 一致）
        cmd = [
            demo_path,
            os.path.abspath(image_path),
            rknn_path,
            rkllm_path,
            str(self._max_new_tokens),
            str(self._max_context_len),
            str(self._n_threads),
            self._platform,
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
        ]

        # prompt 通过 stdin 传入，以 "exit" 结束会话
        stdin_text = prompt + "\nexit\n"

        logger.debug(f"VLM 推理命令: {' '.join(cmd[:4])}...")

        result = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=self._timeout,
            env=env,
            cwd=str(model_dir),
        )

        if result.returncode != 0:
            stderr_snippet = result.stderr[:200] if result.stderr else ""
            logger.warning(f"VLM demo 返回码 {result.returncode}: {stderr_snippet}")

        return result.stdout

    def _parse_output(self, raw: str) -> dict:
        """解析 RKLLM demo 的原始输出

        支持三种 bbox 格式：
          1. JSON: {"bbox_2d": [x1,y1,x2,y2], "label": "..."}
          2. 方括号: [x1,y1,x2,y2]（无 label）
          3. 坐标对: (cx, cy)

        所有 bbox 最终归一化到 [0,1]。
        """
        # 提取 "robot:" 之后的回答内容
        answer = self._extract_answer(raw)

        # 尝试 JSON 格式解析
        result = self._try_parse_json_bbox(answer)
        if result is not None:
            return result

        # 尝试正则提取 bbox
        result = self._try_parse_regex_bbox(answer)
        if result is not None:
            return result

        # 无法解析坐标 → 返回无检测结果
        logger.warning(f"VLM 输出无法解析 bbox: {answer[:100]}")
        return {
            "bbox": None,
            "label": "",
            "confidence": 0.0,
            "center_px": None,
            "raw": answer,
        }

    def _extract_answer(self, raw: str) -> str:
        """从 demo 原始输出中提取模型回答文本"""
        answer = ""
        # 找 "robot:" 标记后的内容
        idx = raw.find("robot:")
        if idx >= 0:
            after = raw[idx + 6:]
            # 截断到下一个 "user:" 标记
            end_idx = after.find("user:")
            answer = after[:end_idx].strip() if end_idx >= 0 else after.strip()
        else:
            answer = raw.strip()

        # 过滤 RKLLM 日志行
        lines = []
        for line in answer.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("I rkllm:", "rkllm", "main:", "W rkllm:")):
                continue
            if stripped in ("robot:", "user:"):
                continue
            lines.append(line)

        return "\n".join(lines)

    def _try_parse_json_bbox(self, text: str) -> Optional[dict]:
        """尝试从文本中解析 JSON 格式的 bbox"""
        # 匹配包含 bbox_2d 的 JSON 对象
        json_match = re.search(r'\{[^{}]*"bbox_2d"[^{}]*\}', text, re.DOTALL)
        if not json_match:
            return None

        try:
            obj = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            # 尝试修复常见问题（单引号、尾逗号）
            raw_json = json_match.group(0)
            raw_json = raw_json.replace("'", '"')
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)
            try:
                obj = json.loads(raw_json)
            except json.JSONDecodeError:
                return None

        bbox = obj.get("bbox_2d", [])
        label = obj.get("label", "")

        if not bbox or len(bbox) != 4:
            # 空 bbox → 未检测到目标
            return {
                "bbox": None,
                "label": label,
                "confidence": 0.0,
                "center_px": None,
                "raw": text,
            }

        # 归一化 bbox
        norm_bbox = self._normalize_bbox(bbox)
        center_px = self._bbox_to_center_px(norm_bbox)

        return {
            "bbox": norm_bbox,
            "label": label,
            "confidence": 0.8,  # VLM 无显式置信度，成功解析给 0.8
            "center_px": center_px,
            "raw": text,
        }

    def _try_parse_regex_bbox(self, text: str) -> Optional[dict]:
        """尝试用正则从文本中提取 bbox 坐标"""
        # 模式1: 4 个数值 [x1,y1,x2,y2] 或 (x1,y1,x2,y2)
        # 支持整数和浮点数
        m = re.search(
            r'[\[\(]\s*(\d+\.?\d*)\s*[,，]\s*(\d+\.?\d*)\s*[,，]'
            r'\s*(\d+\.?\d*)\s*[,，]\s*(\d+\.?\d*)\s*[\]\)]',
            text
        )
        if m:
            bbox = [float(m.group(i)) for i in range(1, 5)]
            norm_bbox = self._normalize_bbox(bbox)
            center_px = self._bbox_to_center_px(norm_bbox)
            # 尝试提取 label
            label = self._extract_label_from_text(text)
            return {
                "bbox": norm_bbox,
                "label": label,
                "confidence": 0.6,  # 正则提取置信度较低
                "center_px": center_px,
                "raw": text,
            }

        # 模式2: 坐标对 (cx, cy)
        m = re.search(r'[\[\(]\s*(\d+\.?\d*)\s*[,，]\s*(\d+\.?\d*)\s*[\]\)]', text)
        if m:
            cx, cy = float(m.group(1)), float(m.group(2))
            # 判断是否归一化坐标
            if cx <= 1.0 and cy <= 1.0:
                cx_px = int(cx * self._image_width)
                cy_px = int(cy * self._image_height)
            else:
                cx_px, cy_px = int(cx), int(cy)
            # 从中心点构造一个小 bbox（±20 像素范围）
            half_w = 20 / self._image_width
            half_h = 20 / self._image_height
            norm_cx = cx_px / self._image_width
            norm_cy = cy_px / self._image_height
            norm_bbox = (
                max(0.0, norm_cx - half_w),
                max(0.0, norm_cy - half_h),
                min(1.0, norm_cx + half_w),
                min(1.0, norm_cy + half_h),
            )
            label = self._extract_label_from_text(text)
            return {
                "bbox": norm_bbox,
                "label": label,
                "confidence": 0.4,  # 仅坐标对，置信度更低
                "center_px": (cx_px, cy_px),
                "raw": text,
            }

        return None

    def _normalize_bbox(self, bbox) -> tuple:
        """确保 bbox 为归一化 [0,1] 格式

        处理 Qwen3.5-VL 输出格式不统一问题：
          - 绝对像素坐标 [120, 80, 340, 290] → 除以 (W, H)
          - 已归一化坐标 [0.19, 0.17, 0.53, 0.60] → 直接使用

        Args:
            bbox: [x1, y1, x2, y2] 原始坐标

        Returns:
            (x1, y1, x2, y2) 归一化到 [0,1]
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]

        # 判断是否为绝对像素坐标（任一值 > 1.0）
        if max(x1, y1, x2, y2) > 1.0:
            x1 = x1 / self._image_width
            y1 = y1 / self._image_height
            x2 = x2 / self._image_width
            y2 = y2 / self._image_height

        # clamp 到 [0, 1] 范围
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        # 确保 x1 < x2, y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        return (x1, y1, x2, y2)

    def _bbox_to_center_px(self, norm_bbox: tuple) -> tuple:
        """归一化 bbox → 绝对像素中心坐标

        Args:
            norm_bbox: (x1, y1, x2, y2) 归一化 [0,1]

        Returns:
            (cx, cy) 绝对像素坐标
        """
        x1, y1, x2, y2 = norm_bbox
        cx = int((x1 + x2) / 2 * self._image_width)
        cy = int((y1 + y2) / 2 * self._image_height)
        return (cx, cy)

    def _extract_label_from_text(self, text: str) -> str:
        """从非 JSON 文本中尝试提取物体标签"""
        # 尝试 "label": "xxx" 模式
        m = re.search(r'["\']label["\']\s*[:：]\s*["\']([^"\']+)["\']', text)
        if m:
            return m.group(1)
        # 尝试 "物体: xxx" 模式
        m = re.search(r'物体[：:]\s*(.+)', text)
        if m:
            return m.group(1).strip().rstrip("，。,.")
        return ""

    def _fail_result(self, error: str) -> dict:
        """构造失败结果"""
        return {
            "bbox": None,
            "label": "",
            "confidence": 0.0,
            "center_px": None,
            "raw": "",
            "error": error,
        }

    # ─── 闲置卸载 ─────────────────────────────────────────────────────────────

    def _idle_unload_loop(self) -> None:
        """后台线程：定期检查闲置状态并卸载"""
        while self._running:
            try:
                self._unload_if_idle()
            except Exception as e:
                logger.debug(f"闲置卸载检查异常: {e}")
            time.sleep(5.0)

    def _unload_if_idle(self) -> bool:
        """如果闲置超时则卸载，返回是否执行了卸载"""
        if not self._loaded:
            return False
        if self._last_infer_time == 0:
            return False
        idle_seconds = time.time() - self._last_infer_time
        if idle_seconds > self._idle_timeout:
            logger.info(
                f"VLM 闲置 {idle_seconds:.0f}s > {self._idle_timeout:.0f}s，执行卸载"
            )
            self._unload()
            return True
        return False

    # ─── 状态查询 ─────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """VLM 是否处于加载状态"""
        return self._loaded

    @property
    def idle_seconds(self) -> float:
        """距上次推理的闲置秒数"""
        if self._last_infer_time == 0:
            return float("inf")
        return time.time() - self._last_infer_time

    def stats(self) -> str:
        """返回模块运行统计信息"""
        avg_ms = self._total_infer_ms / max(self._infer_count, 1)
        return (
            f"VLMPerception("
            f"{'已加载' if self._loaded else '未加载'}, "
            f"闲置{self.idle_seconds:.0f}s/{self._idle_timeout:.0f}s, "
            f"推理{self._infer_count}次, "
            f"均耗时{avg_ms:.0f}ms)"
        )
