"""D435i 深度相机管理 -- 线程安全帧缓冲 + 深拷贝保护"""
import time
import threading
import logging
import numpy as np
from typing import Optional, Callable

from hardware.interfaces import HardwareModule, Observation

logger = logging.getLogger(__name__)

# 尝试导入 pyrealsense2（板端环境）
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
    logger.warning("pyrealsense2 未安装，CameraManager 将不可用")


class FrameBuffer:
    """线程/进程安全的帧缓冲区，写入和读取均深拷贝防止脏数据"""

    def __init__(self):
        self._latest: Optional[tuple[np.ndarray, np.ndarray, float]] = None
        self._lock = threading.Lock()

    def update(self, rgb: np.ndarray, depth: np.ndarray, ts: float):
        """写入新帧（深拷贝）"""
        with self._lock:
            self._latest = (rgb.copy(), depth.copy(), ts)

    def get_frame(self) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
        """获取最新帧（深拷贝，非阻塞）"""
        with self._lock:
            if self._latest is None:
                return None
            rgb, depth, ts = self._latest
            return rgb.copy(), depth.copy(), ts

    @property
    def has_frame(self) -> bool:
        with self._lock:
            return self._latest is not None


class CameraManager(HardwareModule):
    """D435i 相机管理器 -- 单一实例，线程安全帧缓冲

    实现 HardwareModule 接口 (refactor_plan_v9 §4.2):
      - setup/start/stop/is_available/on_failure: 生命周期
      - get_observation(): 返回含 rgb/depth 的 Observation（state 由 System 与机械臂合并）
      - execute(): 相机为只读传感器，无执行动作，no-op 返回 True

    用法::

        cam = CameraManager(width=640, height=480, fps=30)
        cam.start()
        ...
        frame = cam.get_frame()      # (rgb, depth, timestamp) or None
        rgb   = cam.get_rgb()         # np.ndarray or None
        depth = cam.get_depth()       # np.ndarray or None
        ...
        cam.stop()
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 warmup_seconds: float = 3.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_seconds = warmup_seconds
        self._pipeline: Optional["rs.pipeline"] = None
        self._config: Optional["rs.config"] = None
        self._frame_buffer = FrameBuffer()
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._warmed_up = False
        self._subscribers: list[Callable[[np.ndarray, np.ndarray, float], None]] = []
        self._subscribers_lock = threading.Lock()

    # ── 生命周期 ──────────────────────────────────────────

    def start(self) -> None:
        """启动相机采集"""
        if rs is None:
            raise RuntimeError("pyrealsense2 未安装")

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        self._config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )

        self._pipeline.start(self._config)

        self._running = True
        self._warmed_up = False
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraCapture"
        )
        self._capture_thread.start()
        logger.info(
            "CameraManager 已启动 (%dx%d@%dfps), 预热 %.1fs",
            self.width, self.height, self.fps, self.warmup_seconds,
        )

    def stop(self) -> None:
        """停止采集"""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        logger.info("CameraManager 已停止")

    # ── HardwareModule 接口 (refactor_plan_v9 §4.2) ──────────────

    def setup(self, config: dict) -> None:
        """配置相机。支持 config 键: width, height, fps, warmup_seconds"""
        if not config:
            return
        self.width = int(config.get("width", self.width))
        self.height = int(config.get("height", self.height))
        self.fps = int(config.get("fps", self.fps))
        self.warmup_seconds = float(config.get("warmup_seconds", self.warmup_seconds))

    @property
    def is_available(self) -> bool:
        """pyrealsense2 可用且采集线程运行中"""
        return rs is not None and self.is_running

    def on_failure(self) -> str:
        """相机为可选依赖（menu/voice 模式可降级），失败跳过（与 main.py init_camera 语义一致）"""
        return "skip"

    def execute(self, action) -> bool:
        """相机为只读传感器，无执行动作；no-op 返回 True"""
        return True

    def get_observation(self) -> Observation:
        """返回含 rgb/depth 的 Observation

        state（关节角）相机未知，置 None，由 System 层与机械臂观测合并。
        尚无帧时 rgb/depth 返回 None。
        """
        frame = self.get_frame()
        if frame is None:
            return Observation(rgb=None, depth=None, state=None, timestamp=time.time())
        rgb, depth, ts = frame
        return Observation(rgb=rgb, depth=depth, state=None, timestamp=ts)

    # ── 采集线程 ──────────────────────────────────────────

    def _capture_loop(self) -> None:
        """采集线程主循环"""
        # 绑核到 A55 小核（板端环境）
        try:
            import os
            os.sched_setaffinity(0, {0, 1})  # A55 核 0-1
        except (AttributeError, OSError):
            pass  # Windows 不支持

        # ── 预热阶段：丢弃帧，等待自动曝光稳定 ──
        warmup_deadline = time.monotonic() + self.warmup_seconds
        logger.info("相机预热中 (%.1fs)，丢弃帧数据...", self.warmup_seconds)
        while self._running and time.monotonic() < warmup_deadline:
            try:
                self._pipeline.wait_for_frames(timeout_ms=500)
            except Exception:
                pass
        if not self._running:
            return
        self._warmed_up = True
        logger.info("相机预热完成，开始写入帧缓冲")

        # ── 正常采集阶段 ──
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # 转为 numpy 数组
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # BGR -> RGB
                rgb = color_image[:, :, ::-1].copy()

                # 深度转米（D435i 深度单位是 mm）
                depth_m = depth_image.astype(np.float32) / 1000.0

                ts = time.time()
                self._frame_buffer.update(rgb, depth_m, ts)

                # 通知订阅者（传入的数组已是独立副本，可安全持有）
                self._notify_subscribers(rgb, depth_m, ts)

            except Exception as e:
                logger.error("采集帧异常: %s", e)
                time.sleep(0.01)

    # ── 订阅者模式 ────────────────────────────────────────

    def subscribe(self, callback: Callable[[np.ndarray, np.ndarray, float], None]) -> None:
        """订阅每帧回调（采集线程同步调用，务必轻量）

        Parameters
        ----------
        callback : callable
            签名 ``callback(rgb, depth, timestamp)``，在采集线程中同步执行。
            回调中收到的 rgb / depth 是深拷贝后的副本，可安全持有。
            **注意**：回调应快速返回（<5ms），耗时操作请自行异步分发。
        """
        with self._subscribers_lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
                logger.debug("新增帧订阅者: %s", getattr(callback, '__qualname__', callback))

    def unsubscribe(self, callback: Callable[[np.ndarray, np.ndarray, float], None]) -> None:
        """取消订阅"""
        with self._subscribers_lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def _notify_subscribers(self, rgb: np.ndarray, depth: np.ndarray, ts: float) -> None:
        """通知所有订阅者（采集线程内调用）"""
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(rgb, depth, ts)
            except Exception as e:
                logger.error("帧订阅回调异常: %s", e)

    # ── 帧读取接口 ────────────────────────────────────────

    def get_frame(self) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
        """获取最新帧（深拷贝，非阻塞）

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float] | None
            (rgb, depth_m, timestamp) 或 None（尚无帧时）
        """
        return self._frame_buffer.get_frame()

    def get_rgb(self) -> Optional[np.ndarray]:
        """仅获取 RGB 帧"""
        frame = self._frame_buffer.get_frame()
        return frame[0] if frame else None

    def get_depth(self) -> Optional[np.ndarray]:
        """仅获取深度帧（单位：米）"""
        frame = self._frame_buffer.get_frame()
        return frame[1] if frame else None

    # ── 状态属性 ──────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return (
            self._running
            and self._capture_thread is not None
            and self._capture_thread.is_alive()
        )

    @property
    def has_frame(self) -> bool:
        return self._frame_buffer.has_frame

    @property
    def is_warmed_up(self) -> bool:
        """相机是否已完成预热（自动曝光稳定）"""
        return self._warmed_up
