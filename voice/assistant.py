"""语音助手封装 -- 实现 Module 接口，集成到 System。

设计说明：
    - 复用 voice/orchestrator.py 中的 sherpa-onnx KWS/ASR/TTS 能力；
    - 唤醒 -> 录音 -> ASR 文本 -> 关键词意图路由 -> on_intent 回调；
    - sherpa-onnx / yaml / 模型缺失时优雅降级（is_available=False），
      不影响核心抓取功能（on_failure 返回 "skip"）。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from hardware.interfaces import Module

logger = logging.getLogger(__name__)

# 语音线程绑定到 A55 小核（0-3），把大核留给视觉/推理
_LITTLE_CORES = {2, 3}

# 意图关键词（优先级：stop > home > grasp > ask）
_STOP_KEYWORDS = ("停止", "急停", "停下", "别动", "暂停", "停一停")
_HOME_KEYWORDS = ("归零", "回零", "复位", "归位", "回家", "初始位置", "回初始", "回到原位")
_GRASP_KEYWORDS = ("抓取", "抓起", "拿起", "夹起", "捡", "抓", "拿")
# 目标提取时需要剔除的语气/指代词
_TARGET_STOPWORDS = (
    "帮我", "帮忙", "麻烦", "请", "帮我", "一下", "把", "给", "它", "他",
    "那个", "这个", "那", "这", "的", "东西", "物体",
)


class VoiceAssistant(Module):
    """语音助手 -- 唤醒词 + ASR + 意图路由 + TTS"""

    def __init__(self):
        self._orchestrator = None
        self._config: Optional[dict] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_intent: Optional[Callable] = None  # (intent_type, params) -> None
        self._wake_timeout = 5  # 唤醒等待超时（秒），用于响应 stop()
        self._command_seconds: Optional[int] = None

    # ------------------------------------------------------------------
    # Module 生命周期
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        """初始化语音模块。

        config keys:
            config_path: str      # voice/config/default.yaml 路径
            on_intent: Callable   # 意图回调 (intent_type: str, params: dict) -> None
            wake_timeout: int     # 唤醒等待超时秒数（可选，默认 5）
        """
        self._on_intent = config.get("on_intent")
        self._wake_timeout = int(config.get("wake_timeout", 5))
        config_path = config.get("config_path", "voice/config/default.yaml")

        # 延迟导入，避免 sherpa-onnx / yaml 未安装时整个 System 崩溃
        try:
            from voice.config import load_config
            from voice.orchestrator import VoiceAssistant as _Orchestrator

            self._config = load_config(config_path)
            self._command_seconds = self._config.get("audio", {}).get("command_seconds")
            self._orchestrator = _Orchestrator(self._config)
            logger.info("VoiceAssistant 已配置")
        except ImportError as e:
            logger.warning(f"语音模块依赖缺失: {e}")
            self._orchestrator = None
        except Exception as e:
            logger.warning(f"语音模块配置失败: {e}")
            self._orchestrator = None

    def start(self) -> None:
        """启动语音监听线程"""
        if self._orchestrator is None:
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="VoiceAssistant")
        self._thread.start()
        logger.info("VoiceAssistant 已启动")

    def stop(self) -> None:
        """停止语音监听并释放资源"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._wake_timeout + 3.0)
            self._thread = None
        if self._orchestrator is not None:
            try:
                self._orchestrator.cleanup_temp()
            except Exception:
                pass
        logger.info("VoiceAssistant 已停止")

    @property
    def is_available(self) -> bool:
        return self._orchestrator is not None

    def on_failure(self) -> str:
        # 语音为非关键模块，不可用时跳过，不影响核心抓取
        return "skip"

    # ------------------------------------------------------------------
    # 对外能力
    # ------------------------------------------------------------------

    def say(self, text: str) -> None:
        """TTS 播报（流式合成 -> PCM 喇叭）"""
        if not text or self._config is None:
            return
        try:
            from voice.streaming_tts import StreamingTtsPlayer

            player = StreamingTtsPlayer(self._config)
            try:
                player.enqueue(text)
            finally:
                player.close()
        except Exception as e:
            logger.error(f"TTS 失败: {e}")

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        """语音监听主循环：唤醒 -> 录音 -> ASR -> 意图路由 -> 回调"""
        # 绑核到 A55 小核，避免抢占视觉/推理的大核资源
        try:
            import os
            os.sched_setaffinity(0, _LITTLE_CORES)
        except (AttributeError, OSError):
            pass

        while self._running:
            try:
                keyword = self._orchestrator.wait_for_wake(
                    mode="kws", timeout=self._wake_timeout)
            except TimeoutError:
                # 超时属正常轮询，回到循环顶部检查 self._running
                continue
            except Exception as e:
                if not self._running:
                    break
                logger.error(f"唤醒检测异常: {e}")
                time.sleep(0.5)
                continue

            if not self._running:
                break

            logger.info(f"检测到唤醒词: {keyword}")
            try:
                text = self._record_and_transcribe()
            except Exception as e:
                logger.error(f"录音/识别异常: {e}")
                time.sleep(0.2)
                continue

            if not text:
                logger.info("未识别到有效语音内容")
                continue

            logger.info(f"识别文本: {text}")
            intent, params = self._route(text)
            if self._on_intent is not None:
                try:
                    self._on_intent(intent, params)
                except Exception as e:
                    logger.error(f"意图回调处理异常: {e}")

    def _record_and_transcribe(self) -> str:
        """录一段指令音频并转写为文本"""
        wav = self._orchestrator.record_command(self._command_seconds)
        try:
            return self._orchestrator.transcribe_wav(wav)
        finally:
            try:
                wav.unlink(missing_ok=True)
            except Exception:
                pass

    def _route(self, text: str) -> tuple[str, dict]:
        """将识别文本映射为 (intent, params)"""
        t = text.strip()
        if not t:
            return "ask", {"question": ""}

        # 1. 急停（最高优先级）
        if any(k in t for k in _STOP_KEYWORDS) or t in ("停", "停一下"):
            return "stop", {}

        # 2. 归零 / 复位
        if any(k in t for k in _HOME_KEYWORDS):
            return "home", {}

        # 3. 抓取
        for kw in _GRASP_KEYWORDS:
            if kw in t:
                target = self._extract_target(t)
                return "grasp", {"target": target}

        # 4. 兜底：VLM 问答
        return "ask", {"question": t}

    @staticmethod
    def _extract_target(text: str) -> str:
        """从抓取指令中剔除动词/语气词，提取目标物体描述"""
        target = text
        for w in _GRASP_KEYWORDS:
            target = target.replace(w, "")
        for w in _TARGET_STOPWORDS:
            target = target.replace(w, "")
        target = target.strip(" ，,。、！!？?")
        return target or text.strip()
