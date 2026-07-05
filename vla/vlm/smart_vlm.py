"""SmartVLM — 按需加载/自动卸载包装器"""
import time
import gc
from .base import VLMBase, VLMResult
from .factory import create_vlm


class SmartVLM:
    """VLM 按需加载/闲置卸载包装

    核心策略：
    - infer() 前自动加载（~4s 冷启动），加载后计时
    - 闲置超过 idle_timeout 秒后自动卸载，释放 ~900MB 内存
    - 短间隔多次推理只加载一次
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        demo_bin: str = "demo",
        idle_timeout: float = 30.0,
    ):
        self._model_name = model_name
        self._model_path = model_path
        self._demo_bin = demo_bin
        self._idle_timeout = idle_timeout
        self._engine: VLMBase | None = None
        self._loaded = False
        self._last_infer = 0.0
        self._load_count = 0
        self._total_load_ms = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def idle_seconds(self) -> float:
        if self._last_infer == 0:
            return float("inf")
        return time.time() - self._last_infer

    def ensure_loaded(self):
        """确保模型已加载，若未加载则冷启动"""
        if self._loaded and self._engine is not None:
            return
        t0 = time.perf_counter()
        if self._engine is None:
            self._engine = create_vlm(self._model_name, demo_bin=self._demo_bin)
        self._engine.load(self._model_path)
        self._loaded = True
        elapsed = (time.perf_counter() - t0) * 1000
        self._load_count += 1
        self._total_load_ms += elapsed

    def unload(self):
        """卸载模型，释放内存"""
        if self._engine is not None and self._loaded:
            self._engine.unload()
            self._loaded = False
            gc.collect()

    def infer(self, image_path: str, prompt: str | None = None) -> VLMResult:
        self.ensure_loaded()
        self._last_infer = time.time()
        return self._engine.infer(image_path, prompt)

    def unload_if_idle(self) -> bool:
        """闲置超时自动卸载，返回 True 表示已卸载"""
        if self._loaded and self.idle_seconds > self._idle_timeout:
            self.unload()
            return True
        return False

    def stats(self) -> str:
        return (
            f"SmartVLM({'已加载' if self._loaded else '未加载'}, "
            f"闲置{self.idle_seconds:.0f}s/{self._idle_timeout:.0f}s超时, "
            f"加载{self._load_count}次, 均耗时"
            f"{self._total_load_ms / max(self._load_count, 1):.0f}ms)"
        )


class IdleUnloader:
    """后台空闲卸载线程 — 定期检查 SmartVLM 并卸载"""

    def __init__(self, smart_vlm: SmartVLM, interval: float = 5.0):
        self._vlm = smart_vlm
        self._interval = interval
        self._thread = None
        self._running = False

    def start(self):
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                self._vlm.unload_if_idle()
            except Exception:
                pass
            time.sleep(self._interval)
