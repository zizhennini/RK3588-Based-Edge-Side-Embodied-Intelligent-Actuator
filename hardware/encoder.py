#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hardware/encoder.py — RK3588 硬件 H.264 编码模块

refactor_plan_v9 §6.3: 硬件编码封装（录像）。
技术选型: ffmpeg 子进程 + h264_rkmpp（RK3588 MPP 硬件编码器，CPU 占用 ~0），
该方案已在 scripts/recorder.py 验证可用；比 GStreamer mpph264enc 管线
在 RK3588 常见系统镜像上更普遍可用。

帧流协议: RGB24 rawvideo 写入 ffmpeg stdin。

降级链:
    h264_rkmpp (硬件) → libx264 (软件, PC 调试用) → 不可用 (is_available=False)

用法:
    enc = H264Encoder()
    enc.setup({"fps": 15, "bitrate": "5M", "out_dir": "./recordings"})
    enc.open("clip.mp4", width=640, height=480)   # 或省略路径自动生成时间戳名
    enc.write_frame(rgb)                           # (H, W, 3) uint8 RGB ndarray
    enc.close()
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from hardware.interfaces import Module

logger = logging.getLogger(__name__)

# 候选编码器（按优先级）: (ffmpeg 编码器名, 说明)
_ENCODER_CANDIDATES = [
    ("h264_rkmpp", "RK3588 MPP 硬件编码"),
    ("libx264", "x264 软件编码（降级）"),
]


def _ffmpeg_available() -> Optional[str]:
    """返回 ffmpeg 可执行文件路径，未找到返回 None"""
    return shutil.which("ffmpeg")


def _probe_encoders(ffmpeg: str) -> set:
    """探测 ffmpeg 支持的编码器集合（一次性，失败返回空集）"""
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return {line.split()[1] for line in out.splitlines()
                if line.startswith(" ") and len(line.split()) >= 2}
    except Exception as e:
        logger.debug("ffmpeg 编码器探测失败: %s", e)
        return set()


class H264Encoder(Module):
    """H.264 视频编码器（硬件加速优先，自动降级）

    生命周期遵循 Module 接口；单次录像用 open()/write_frame()/close()。
    start()/stop() 为模块级启停（预热探测/释放），与单次录像无关。
    """

    def __init__(self):
        self._config: dict = {}
        self._ffmpeg: Optional[str] = None
        self._encoder: Optional[str] = None      # 选定的 ffmpeg 编码器名
        self._proc: Optional[subprocess.Popen] = None
        self._out_path: Optional[str] = None
        self._width = 0
        self._height = 0
        self._frames_written = 0
        self._setup_done = False

    # ------------------------------------------------------------------
    # Module 生命周期
    # ------------------------------------------------------------------

    def setup(self, config: dict) -> None:
        """初始化：探测 ffmpeg 与可用编码器

        config 可选键:
            fps (int): 帧率，默认 15
            bitrate (str): 目标码率，默认 "5M"
            out_dir (str): 输出目录，默认 "./recordings"
            encoder (str): 强制指定编码器（跳过自动探测）
        """
        self._config = dict(config or {})
        self._ffmpeg = _ffmpeg_available()
        if self._ffmpeg is None:
            logger.warning("H264Encoder: 未找到 ffmpeg，录像功能不可用")
            self._encoder = None
            self._setup_done = True
            return

        forced = self._config.get("encoder")
        if forced:
            self._encoder = forced
        else:
            supported = _probe_encoders(self._ffmpeg)
            self._encoder = None
            for name, desc in _ENCODER_CANDIDATES:
                if name in supported:
                    self._encoder = name
                    logger.info("H264Encoder: 使用 %s（%s）", name, desc)
                    break
            if self._encoder is None:
                logger.warning("H264Encoder: ffmpeg 无可用 H.264 编码器")
        self._setup_done = True

    def start(self) -> None:
        """模块级启动（无后台线程，探测在 setup 完成）"""
        if not self._setup_done:
            self.setup({})

    def stop(self) -> None:
        """模块级停止：确保进行中的录像已关闭"""
        self.close()

    @property
    def is_available(self) -> bool:
        return self._ffmpeg is not None and self._encoder is not None

    def on_failure(self) -> str:
        # 录像属非关键路径：失败跳过，不影响主系统
        return "skip"

    # ------------------------------------------------------------------
    # 单次录像 API
    # ------------------------------------------------------------------

    def open(self, out_path: Optional[str] = None,
             width: int = 640, height: int = 480) -> Optional[str]:
        """开始一段录像，返回实际输出文件路径；不可用时返回 None"""
        if not self.is_available:
            logger.error("H264Encoder 不可用（ffmpeg/编码器缺失）")
            return None
        if self._proc is not None:
            logger.warning("H264Encoder: 已有录像进行中，先关闭旧录像")
            self.close()

        if out_path is None:
            out_dir = Path(self._config.get("out_dir", "./recordings"))
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = str(out_dir / f"record_{ts}.mp4")
        else:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        fps = str(self._config.get("fps", 15))
        bitrate = str(self._config.get("bitrate", "5M"))
        cmd = [
            self._ffmpeg, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", fps,
            "-i", "-",
            "-c:v", self._encoder, "-b:v", bitrate,
            "-r", fps,
            out_path,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            logger.error("H264Encoder: ffmpeg 启动失败: %s", e)
            self._proc = None
            return None

        self._out_path = out_path
        self._width, self._height = width, height
        self._frames_written = 0
        logger.info("H264Encoder: 开始录像 %s (%dx%d @%sfps, %s)",
                    out_path, width, height, fps, self._encoder)
        return out_path

    def write_frame(self, rgb: np.ndarray) -> bool:
        """写入一帧 RGB24 图像 (H, W, 3) uint8；成功返回 True"""
        if self._proc is None or self._proc.stdin is None:
            return False
        if rgb is None:
            return False
        try:
            # 尺寸防御: 与 open() 不一致时拒绝写入（避免 rawvideo 流错位）
            if rgb.shape[1] != self._width or rgb.shape[0] != self._height:
                logger.warning(
                    "H264Encoder: 帧尺寸 %s 与录像尺寸 %dx%d 不符，丢弃该帧",
                    rgb.shape[:2], self._width, self._height)
                return False
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            self._proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
            self._frames_written += 1
            return True
        except (BrokenPipeError, OSError) as e:
            logger.error("H264Encoder: 写帧失败（ffmpeg 已退出？）: %s", e)
            self._proc = None
            return False

    def close(self) -> Optional[str]:
        """结束录像并等待 ffmpeg 落盘，返回输出路径（无录像时返回 None）"""
        out = self._out_path
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("H264Encoder: ffmpeg 未在 10s 内退出，强制终止")
                self._proc.kill()
            logger.info("H264Encoder: 录像结束 %s（%d 帧）", out, self._frames_written)
        self._proc = None
        self._out_path = None
        return out if self._frames_written > 0 or out else None

    @property
    def is_recording(self) -> bool:
        return self._proc is not None

    @property
    def frames_written(self) -> int:
        return self._frames_written
