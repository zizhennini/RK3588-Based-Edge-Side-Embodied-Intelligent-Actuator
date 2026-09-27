#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/cam_sink.py — 采集侧相机流与 JPEG 落盘（板端）

设计要点:
- 复用 hardware/camera_d435i.CameraManager（D435i RGB+Depth，非阻塞 get_frame）；
  USB 相机走 cv2.VideoCapture + 抓帧线程。两者统一暴露 latest(name)。
- JpegSink 异步写盘（有界队列 + 写线程），保证 30Hz 遥操作环不被磁盘 IO 阻塞；
  队列溢出/写失败计数上报，不静默丢数据。
- 命名与关节帧一一对应: ``<episode_dir>/<cam>/000000.jpg`` 序号 = 关节帧序号，
  json_to_lerobot 据此对齐（同一 index 即同一帧）。

配置文件 config/cameras.json::

    {
      "cameras": [
        {"name": "front", "type": "realsense", "width": 640, "height": 480, "fps": 30},
        {"name": "wrist", "type": "usb", "device": 0, "width": 640, "height": 480}
      ],
      "jpeg_quality": 90
    }
"""
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "cameras": [{"name": "front", "type": "realsense",
                 "width": 640, "height": 480, "fps": 30}],
    "jpeg_quality": 90,
}


class _UsbCamera:
    """USB 相机抓帧线程（cv2.VideoCapture）"""

    def __init__(self, name: str, device: int = 0, width: int = 640,
                 height: int = 480, fps: int = 30):
        self.name = name
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self._cap = None
        self._latest: Optional[np.ndarray] = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"USB 相机 {self.name} 打开失败 (device={self.device})")
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.name}",
                                        daemon=True)
        self._thread.start()
        time.sleep(0.5)   # 等首帧

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._latest = frame
                    self._ts = time.perf_counter()

    def latest(self):
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy(), self._ts

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()


class CameraSet:
    """多相机集合: 统一 latest(name) 接口（D435i + 可选 USB）"""

    def __init__(self, specs: List[dict], warmup_s: float = 3.0):
        self.specs = specs
        self.warmup_s = warmup_s
        self._cams: Dict[str, object] = {}
        self._kind: Dict[str, str] = {}

    @classmethod
    def from_config(cls, path: Optional[str]) -> "CameraSet":
        cfg = dict(DEFAULT_CONFIG)
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        warmup = float(cfg.get("warmup_seconds", 3.0))
        return cls(cfg.get("cameras", []), warmup_s=warmup)

    @property
    def names(self) -> List[str]:
        return list(self._cams.keys())

    def start(self) -> None:
        for spec in self.specs:
            name, kind = spec["name"], spec.get("type", "realsense")
            if kind == "realsense":
                from hardware.camera_d435i import CameraManager
                cam = CameraManager(width=spec.get("width", 640),
                                    height=spec.get("height", 480),
                                    fps=spec.get("fps", 30),
                                    warmup_seconds=self.warmup_s)
                cam.start()
                self._cams[name] = cam
                self._kind[name] = "realsense"
                logger.info("相机 %s (RealSense) 已启动", name)
            elif kind == "usb":
                cam = _UsbCamera(name, device=spec.get("device", 0),
                                 width=spec.get("width", 640),
                                 height=spec.get("height", 480),
                                 fps=spec.get("fps", 30))
                cam.start()
                self._cams[name] = cam
                self._kind[name] = "usb"
                logger.info("相机 %s (USB device=%s) 已启动", name, spec.get("device", 0))
            else:
                raise ValueError(f"未知相机类型: {kind}（支持 realsense/usb）")

    def latest(self, name: str):
        """→ (bgr ndarray, ts) 或 None（非阻塞）

        统一返回 **BGR**（cv2.imwrite 约定）：CameraManager 内部把 bgr8 流转成 RGB 存储，
        此处再转回 BGR，保证多相机落盘格式一致。
        """
        cam = self._cams.get(name)
        if cam is None:
            return None
        if self._kind[name] == "realsense":
            rgb = cam.get_rgb()
            if rgb is None:
                return None
            return rgb[:, :, ::-1].copy(), time.perf_counter()
        return cam.latest()

    def wait_ready(self, timeout_s: float = 15.0) -> Dict[str, bool]:
        """等待各相机出首帧（CameraManager 有预热期，期间无帧）

        Returns: {cam_name: 是否就绪}
        """
        ready = {}
        deadline = time.perf_counter() + timeout_s
        for name in self.names:
            ok = False
            while time.perf_counter() < deadline:
                if self.latest(name) is not None:
                    ok = True
                    break
                time.sleep(0.1)
            ready[name] = ok
            if ok:
                logger.info("相机 %s 就绪", name)
            else:
                logger.warning("相机 %s 等待首帧超时（%.0fs）", name, timeout_s)
        return ready

    def stop(self) -> None:
        for name, cam in self._cams.items():
            try:
                cam.stop()
            except Exception as e:
                logger.warning("相机 %s 停止失败: %s", name, e)
        self._cams.clear()


class JpegSink:
    """异步 JPEG 写盘（有界队列 + 写线程），按相机分目录、序号对齐关节帧"""

    def __init__(self, root: Path, cameras: List[str], quality: int = 90,
                 queue_size: int = 512):
        self.root = Path(root)
        self.cameras = list(cameras)
        self.quality = quality
        self.seq = {c: 0 for c in cameras}
        self.written = {c: 0 for c in cameras}
        self.dropped = {c: 0 for c in cameras}
        self.failed = {c: 0 for c in cameras}
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        for c in self.cameras:
            (self.root / c).mkdir(parents=True, exist_ok=True)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="jpeg-sink", daemon=True)
        self._thread.start()

    def push(self, cam: str, bgr: np.ndarray) -> None:
        """入队一帧（复制以免被相机线程复用）；队列满则计数丢弃"""
        if cam not in self.seq:
            return
        idx = self.seq[cam]
        self.seq[cam] += 1
        try:
            self._q.put_nowait((cam, idx, bgr.copy()))
        except queue.Full:
            self.dropped[cam] += 1

    def _loop(self) -> None:
        import cv2
        while self._running or not self._q.empty():
            try:
                cam, idx, img = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            path = self.root / cam / f"{idx:06d}.jpg"
            try:
                ok = cv2.imwrite(str(path), img,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)])
                if ok:
                    self.written[cam] += 1
                else:
                    self.failed[cam] += 1
            except Exception:
                self.failed[cam] += 1
            finally:
                self._q.task_done()

    def finish(self) -> dict:
        """排空队列并返回统计 {cam: {written, dropped, failed}}"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30.0)
        return {c: {"written": self.written[c], "dropped": self.dropped[c],
                    "failed": self.failed[c]} for c in self.cameras}
