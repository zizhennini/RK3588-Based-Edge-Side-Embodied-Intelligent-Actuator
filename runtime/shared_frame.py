# runtime/shared_frame.py
"""跨进程共享内存帧缓冲 — refactor_plan_v9 §1.6 / §4.1 落地组件

方案原文: "进程间通信: multiprocessing.Queue + shared_memory（帧数据走共享内存
避免序列化开销）"。本模块提供其中的 **共享内存帧传输** 原语：

  - 生产者（相机子进程）create=True 分配共享内存，write_frame() 写入 RGB+深度
  - 消费者（推理子进程/主进程）create=False 按 name attach，read_frame() 零拷贝读出

同步采用 **seqlock**（顺序锁）：单生产者/单消费者、"最新帧"语义、无锁、
按 name 跨进程 attach 无需共享 Lock 对象。读者遇到写入中途（seq 为奇数）或
读取前后 seq 变化（帧被覆盖）时自旋重试，保证不会读到撕裂帧。

内存布局（单块 SharedMemory）::

    [0:64]      Header: seq(u64) ts(f64) h(i32) w(i32) ch(i32) has_depth(i32) 保留
    [64:...]    RGB   : h*w*ch  uint8
    [...:...]    Depth : h*w     float32   （has_depth=1 时存在）

用法::

    # 生产者
    buf = SharedFrameBuffer.create(width=640, height=480, name="eia_cam")
    buf.write_frame(rgb, depth, ts)

    # 消费者（另一进程）
    buf = SharedFrameBuffer.attach("eia_cam")
    rgb, depth, ts = buf.read_frame()
"""
from __future__ import annotations

import logging
import struct
import time
from typing import Optional, Tuple

import numpy as np

try:
    from multiprocessing import shared_memory
except ImportError:  # pragma: no cover - Python < 3.8
    shared_memory = None

logger = logging.getLogger(__name__)

# Header: seq(u64) ts(f64) h(i32) w(i32) ch(i32) has_depth(i32)
_HEADER_FMT = "<Qdiiii"
_HEADER_SIZE = 64  # 固定 64 字节头，预留对齐/扩展
assert struct.calcsize(_HEADER_FMT) <= _HEADER_SIZE


class SharedFrameBuffer:
    """共享内存帧缓冲（seqlock 同步，单生产者/单消费者）

    形状在创建时固定（width/height/channels/has_depth）；write_frame 要求
    输入形状匹配，避免运行期重分配。
    """

    def __init__(self, shm: "shared_memory.SharedMemory", width: int, height: int,
                 channels: int = 3, has_depth: bool = True, owner: bool = False):
        if shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory 不可用（需 Python 3.8+）")
        self._shm = shm
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.has_depth = bool(has_depth)
        self._owner = owner  # owner 负责 unlink 释放系统资源
        self._closed = False

        # 各区域偏移
        self._rgb_off = _HEADER_SIZE
        self._rgb_n = self.width * self.height * self.channels
        self._depth_off = self._rgb_off + self._rgb_n
        self._depth_n = self.width * self.height * 4  # float32

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, width: int, height: int, channels: int = 3,
               has_depth: bool = True, name: Optional[str] = None) -> "SharedFrameBuffer":
        """分配新的共享内存（生产者调用）"""
        rgb_n = width * height * channels
        depth_n = width * height * 4 if has_depth else 0
        size = _HEADER_SIZE + rgb_n + depth_n
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        buf = cls(shm, width, height, channels, has_depth, owner=True)
        buf._write_header(0, 0.0, height, width, channels, 1 if has_depth else 0)
        logger.info("SharedFrameBuffer 创建: name=%s %dx%dx%d size=%d",
                    shm.name, width, height, channels, size)
        return buf

    @classmethod
    def attach(cls, name: str) -> "SharedFrameBuffer":
        """按名字附着到已存在的共享内存（消费者调用）"""
        shm = shared_memory.SharedMemory(name=name, create=False)
        seq, ts, h, w, ch, has_depth = cls._read_header_from(shm)
        return cls(shm, w, h, ch, bool(has_depth), owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    # ------------------------------------------------------------------
    # Header 读写
    # ------------------------------------------------------------------
    def _write_header(self, seq, ts, h, w, ch, has_depth) -> None:
        struct.pack_into(_HEADER_FMT, self._shm.buf, 0, seq, ts, h, w, ch, has_depth)

    @staticmethod
    def _read_header_from(shm) -> Tuple[int, float, int, int, int, int]:
        return struct.unpack_from(_HEADER_FMT, shm.buf, 0)

    def _read_seq(self) -> int:
        return struct.unpack_from("<Q", self._shm.buf, 0)[0]

    def _write_seq(self, seq: int) -> None:
        struct.pack_into("<Q", self._shm.buf, 0, seq)

    def _write_ts(self, ts: float) -> None:
        struct.pack_into("<d", self._shm.buf, 8, ts)

    # ------------------------------------------------------------------
    # 帧读写
    # ------------------------------------------------------------------
    def write_frame(self, rgb: np.ndarray, depth: Optional[np.ndarray],
                    ts: Optional[float] = None) -> bool:
        """写入一帧（生产者）。seqlock: seq→奇数(写入中)→偶数(稳定)。"""
        if self._closed:
            return False
        if ts is None:
            ts = time.time()
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            logger.warning("write_frame: RGB 形状 %s 与缓冲 %dx%d 不符，丢弃",
                           rgb.shape[:2], self.width, self.height)
            return False

        seq = self._read_seq()
        self._write_seq(seq + 1)  # 奇数 = 写入中

        # RGB
        rgb_view = np.ndarray((self.height, self.width, self.channels),
                              dtype=np.uint8, buffer=self._shm.buf,
                              offset=self._rgb_off)
        np.copyto(rgb_view, np.ascontiguousarray(rgb, dtype=np.uint8))

        # Depth
        if self.has_depth:
            if depth is None:
                depth = np.zeros((self.height, self.width), dtype=np.float32)
            depth_view = np.ndarray((self.height, self.width),
                                    dtype=np.float32, buffer=self._shm.buf,
                                    offset=self._depth_off)
            np.copyto(depth_view, np.ascontiguousarray(depth, dtype=np.float32))

        self._write_ts(float(ts))
        self._write_seq(seq + 2)  # 偶数 = 稳定
        return True

    def read_frame(self, max_retry: int = 50
                   ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], float]]:
        """读取最新帧（消费者，零拷贝后返回独立副本）。

        seqlock 读: 读 seq(偶) → 拷数据 → 再读 seq，若变化说明写入中被覆盖，重试。
        返回 (rgb_copy, depth_copy|None, ts) 或 None（尚无帧/重试耗尽）。
        """
        if self._closed:
            return None
        for _ in range(max_retry):
            seq1 = self._read_seq()
            if seq1 == 0 or (seq1 & 1):
                # 尚无帧 或 写入中 → 短暂让步重试
                if seq1 == 0:
                    return None
                time.sleep(0.0005)
                continue
            _, ts, h, w, ch, has_depth = self._read_header_from(self._shm)
            if h != self.height or w != self.width:
                return None
            rgb_view = np.ndarray((h, w, ch), dtype=np.uint8,
                                  buffer=self._shm.buf, offset=self._rgb_off)
            rgb = rgb_view.copy()
            depth = None
            if has_depth:
                depth_view = np.ndarray((h, w), dtype=np.float32,
                                        buffer=self._shm.buf, offset=self._depth_off)
                depth = depth_view.copy()
            seq2 = self._read_seq()
            if seq1 == seq2:
                return rgb, depth, ts
            # seq 变化 → 读取期间被覆盖，重试
        logger.debug("read_frame: 重试 %d 次仍撕裂，放弃本帧", max_retry)
        return None

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------
    def close(self, unlink: Optional[bool] = None) -> None:
        """关闭句柄。owner 默认 unlink 释放系统级共享内存。"""
        if self._closed:
            return
        self._closed = True
        do_unlink = self._owner if unlink is None else unlink
        try:
            self._shm.close()
        except Exception:
            pass
        if do_unlink:
            try:
                self._shm.unlink()
            except Exception:
                pass

    def __enter__(self) -> "SharedFrameBuffer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


# ----------------------------------------------------------------------
# 自检: 单进程 write→read 往返（验证 header 打包 / numpy 视图 / 数据完整性）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    W, H = 64, 48
    buf = SharedFrameBuffer.create(width=W, height=H, name="eia_selftest")
    try:
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        depth = rng.random((H, W), dtype=np.float32)
        ts = 1234.5678
        assert buf.write_frame(rgb, depth, ts), "write_frame 失败"
        got = buf.read_frame()
        assert got is not None, "read_frame 返回 None"
        gr, gd, gts = got
        assert np.array_equal(gr, rgb), "RGB 往返不一致"
        assert gd is not None and np.allclose(gd, depth), "Depth 往返不一致"
        assert abs(gts - ts) < 1e-6, f"ts 不一致 {gts} != {ts}"
        print(f"[OK] SharedFrameBuffer 往返测试通过 ({W}x{H}, name={buf.name})")
    finally:
        buf.close()
