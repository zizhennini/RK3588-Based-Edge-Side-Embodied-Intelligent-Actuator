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
    """USB/UVC 相机抓帧线程（cv2.VideoCapture）

    与 lerobot OpenCVCameraConfig 对齐（Apache-2.0，独立实现）:
      - ``index_or_path``: 支持整数索引 **或设备路径**（推荐 ``/dev/v4l/by-id/...``，
        断电/重插后节点号会变；按索引还会被 cv2 的深度相机探测逻辑干扰）
      - ``fourcc``: 默认 MJPG（UVC 相机常见必须设 MJPG 才能跑到 30fps@640x480）
      - 显式 V4L2 后端（避免 cv2 误走 obsensor/深度探测路径）
      - ``backend``/``warmup_s``/``rotation``: 预热丢弃帧、可选旋转（0/90/180/270）
    """

    def __init__(self, name: str, device=0, width: int = 640, height: int = 480,
                 fps: int = 30, fourcc: str = "MJPG", warmup_s: float = 1.0,
                 rotation: int = 0, buffersize: int = 4):
        self.name = name
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self.fourcc = fourcc
        self.warmup_s = warmup_s
        self.rotation = int(rotation or 0)
        # 缓冲深度: 实测本机 UVC 相机 bufsize=1 只有 15fps、=4 才 29.8fps
        # （1 帧缓冲时驱动在任何读取抖动下直接丢帧）。lerobot 官方不设该项（用默认）。
        # 代价: 最多滞后 buffersize/fps ≈ 130ms；采集数据集可接受。
        self.buffersize = int(buffersize)
        self._cap = None
        self._latest: Optional[np.ndarray] = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        if self.rotation == 90:
            import cv2
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 180:
            import cv2
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation == 270:
            import cv2
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def start(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        # 顺序很重要: 先分辨率/帧率、后 FOURCC。反过来的话驱动会在设分辨率时
        # 把像素格式重置回 YUYV，640x480 YUYV 在 USB2 上只能跑 ~15fps（实测踩过）。
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*self.fourcc[:4].ljust(4)))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffersize)
        if not cap.isOpened():
            raise RuntimeError(f"USB 相机 {self.name} 打开失败 (device={self.device})")

        # 回读实际协商参数（不匹配会静默降帧率/改格式，必须显式告警）
        def _fourcc_str(v: float) -> str:
            v = int(v)
            return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
        real_cc = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info("USB 相机 %s 协商: %s %dx%d @%.0ffps (请求 %s %dx%d @%d)",
                    self.name, real_cc, real_w, real_h, real_fps,
                    self.fourcc or "auto", self.width, self.height, self.fps)
        if self.fourcc and real_cc.strip("\x00") != self.fourcc[:4]:
            logger.warning("USB 相机 %s 像素格式未生效: 请求 %s 实际 %s"
                           "（会导致帧率下降，如 YUYV@USB2 只有 ~15fps）",
                           self.name, self.fourcc, real_cc)
        if self.fps and real_fps and real_fps < self.fps * 0.8:
            logger.warning("USB 相机 %s 实际帧率 %.0f 低于请求 %d",
                           self.name, real_fps, self.fps)

        self._cap = cap
        # 预热: 丢弃帧等自动曝光/白平衡稳定（lerobot warmup_s 同款语义）
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < self.warmup_s:
            cap.read()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.name}",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                if self.rotation:
                    frame = self._rotate(frame)
                with self._lock:
                    self._latest = frame
                    self._ts = time.perf_counter()

    def latest(self, copy: bool = False):
        with self._lock:
            if self._latest is None:
                return None
            # 不拷贝: cv2.read() 每帧分配新数组，旧数组在引用期间依然有效；
            # 编码在写线程里做（contiguous + imwrite），不占 30Hz 主环
            return (self._latest.copy() if copy else self._latest), self._ts

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
        # lerobot 官方做法: OpenCV 内部线程数设为 1，避免多线程采集时 cv2 内部
        # 线程池与我们的采集/写盘线程争抢（板端 4 小核，争抢会直接掉帧）
        try:
            import cv2
            cv2.setNumThreads(1)
        except Exception:
            pass
        for spec in self.specs:
            name, kind = spec["name"], spec.get("type", "realsense")
            if kind == "realsense":
                from hardware.camera_d435i import CameraManager
                cam = CameraManager(width=spec.get("width", 640),
                                    height=spec.get("height", 480),
                                    fps=spec.get("fps", 30),
                                    warmup_seconds=self.warmup_s,
                                    use_depth=spec.get("use_depth", True))
                cam.start()
                self._cams[name] = cam
                self._kind[name] = "realsense"
                logger.info("相机 %s (RealSense) 已启动", name)
            elif kind == "usb":
                cam = _UsbCamera(name, device=spec.get("device", 0),
                                 width=spec.get("width", 640),
                                 height=spec.get("height", 480),
                                 fps=spec.get("fps", 30),
                                 fourcc=spec.get("fourcc", "MJPG"),
                                 warmup_s=spec.get("warmup_s", 1.0),
                                 rotation=spec.get("rotation", 0),
                                 buffersize=spec.get("buffersize", 4))
                cam.start()
                self._cams[name] = cam
                self._kind[name] = "usb"
                logger.info("相机 %s (USB device=%s, fourcc=%s) 已启动",
                            name, spec.get("device", 0), spec.get("fourcc", "MJPG"))
            else:
                raise ValueError(f"未知相机类型: {kind}（支持 realsense/usb）")

    def latest(self, name: str, copy: bool = False):
        """→ (bgr ndarray, ts) 或 None（非阻塞，**不拷贝**）

        性能取舍（板端实测）: 抓帧路径每多一次整帧 memcpy（640×480×3 ≈ 0.9MB），
        30Hz 双相机就会把 GIL 拖满、USB 相机唯一帧率从 29.8 掉到 14.4fps。
        因此默认返回**内部数组引用**：调用方只读、且不得跨帧持有（下一帧会换新数组，
        旧数组由 cv2/CameraManager 各自分配，引用期间数据有效——见 JpegSink 编码时机）。
        需要长期持有请传 copy=True。

        统一返回 **BGR**（cv2.imwrite 约定）：CameraManager 内部把 bgr8 流转成 RGB 存储，
        此处给出 RGB→BGR 视图（负步长视图，编码线程再做 contiguous）。
        """
        cam = self._cams.get(name)
        if cam is None:
            return None
        if self._kind[name] == "realsense":
            # 用**帧自身时间戳**（而非调用时刻）：采集端据此判断是否新帧、记录 ct_<cam>
            got = cam.get_rgb_ts()
            if not got:
                return None
            rgb, ts = got
            view = rgb[:, :, ::-1]
            return (view.copy() if copy else view), ts
        got = cam.latest(copy=copy)
        return got

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
        """入队一帧（**不拷贝**，编码在线程里做）；队列满则计数丢弃

        前提: bgr 由 CameraSet.latest 返回且未被就地修改（cv2/CameraManager 每帧新数组）。
        """
        if cam not in self.seq:
            return
        idx = self.seq[cam]
        self.seq[cam] += 1
        try:
            self._q.put_nowait((cam, idx, bgr))
        except queue.Full:
            self.dropped[cam] += 1

    def _loop(self) -> None:
        import cv2
        # 绑到 A55 小核（0-1 与相机采集线程同簇），把 A76 大核留给 30Hz 遥操作主环
        try:
            os.sched_setaffinity(0, {0, 1})
        except (AttributeError, OSError):
            pass
        while self._running or not self._q.empty():
            try:
                cam, idx, img = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            path = self.root / cam / f"{idx:06d}.jpg"
            try:
                # 负步长视图/非连续数组在 imwrite 前需转连续（在写线程做，不占主环）
                arr = img if img.flags["C_CONTIGUOUS"] else np.ascontiguousarray(img)
                ok = cv2.imwrite(str(path), arr,
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
