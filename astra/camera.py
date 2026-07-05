"""相机模块 — USBCamera + AstraProCamera + D435iCamera"""
import cv2
import numpy as np
import subprocess
import threading
from pathlib import Path
from collections import deque
from config.cpu_affinity import bind_current_thread, LITTLE_CORES


ASTRA_BIN = Path(__file__).parent / "build" / "astra_capture"


class USBCamera:
    def __init__(self, device_index: int = 0, resolution: tuple = (640, 480)):
        self.device_index = device_index
        self.resolution = resolution
        self.cap = None
        self._latest_rgb = None
        self._grab_thread = None
        self._running = False

    def connect(self):
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开相机 /dev/video{self.device_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._running = True
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()
        import time
        for _ in range(50):
            if self._latest_rgb is not None:
                break
            time.sleep(0.01)

    def _grab_loop(self):
        bind_current_thread(LITTLE_CORES)
        while self._running:
            try:
                ret, bgr = self.cap.read()
                if ret:
                    self._latest_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        rgb = self.read_rgb()
        depth = np.zeros((rgb.shape[0] // 4, rgb.shape[1] // 4), dtype=np.float32)
        return rgb, depth

    def read_rgb(self) -> np.ndarray:
        for _ in range(30):
            if self._latest_rgb is not None:
                return self._latest_rgb
        raise RuntimeError("相机尚未就绪")

    def disconnect(self):
        self._running = False
        if self._grab_thread:
            self._grab_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None


class AstraProCamera:
    def __init__(self, rgb_index: int = 21):
        self.rgb_index = rgb_index
        self.cap = None
        self._cwd = Path(__file__).parent
        self._depth_proc = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.rgb_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开 RGB 相机 /dev/video{self.rgb_index}")
        if not ASTRA_BIN.exists():
            raise FileNotFoundError("astra_capture 未编译，请先 make")
        self._start_depth()

    def _start_depth(self):
        self._depth_proc = subprocess.Popen(
            [str(ASTRA_BIN)], cwd=str(self._cwd),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(3):
            ret, bgr = self.cap.read()
            if ret:
                break
        if not ret:
            raise RuntimeError("读取 RGB 帧失败")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = self._read_depth()
        return rgb, depth

    def read_rgb(self) -> np.ndarray:
        return self.read()[0]

    def _read_depth(self) -> np.ndarray:
        if self._depth_proc is None or self._depth_proc.poll() is not None:
            return np.zeros((120, 160), dtype=np.float32)
        depth_path = self._cwd / "_depth.f32"
        info_path = self._cwd / "_depth.info"
        if not depth_path.exists() or not info_path.exists():
            return np.zeros((120, 160), dtype=np.float32)
        try:
            with open(info_path) as f:
                w_str, h_str = f.read().strip().split()
                w, h = int(w_str), int(h_str)
            depth = np.fromfile(depth_path, dtype=np.float32).reshape(h, w)
            return depth
        except Exception:
            return np.zeros((120, 160), dtype=np.float32)

    def disconnect(self):
        if self._depth_proc:
            self._depth_proc.terminate()
            try:
                self._depth_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._depth_proc.kill()
        if self.cap:
            self.cap.release()
        for f in ["_depth.f32", "_depth.info"]:
            (self._cwd / f).unlink(missing_ok=True)


class D435iCamera:
    """Intel RealSense D435i — RGB + 深度"""
    def __init__(self, depth_res=(848, 480), rgb_res=(640, 480), fps=30):
        self._pipeline = None
        self._align = None
        self._intr = None
        self._depth_res = depth_res
        self._rgb_res = rgb_res
        self._fps = fps
        self._depth_buf = deque(maxlen=5)
        self._spa = None
        self._tmp = None

    def connect(self):
        import pyrealsense2 as rs
        import time
        for attempt in range(3):
            pipe = rs.pipeline()
            cfg = rs.config()
            dw, dh = self._depth_res
            rw, rh = self._rgb_res
            cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, self._fps)
            cfg.enable_stream(rs.stream.color, rw, rh, rs.format.bgr8, self._fps)
            try:
                profile = pipe.start(cfg)
                pipe.wait_for_frames(3000)
                self._pipeline = pipe
                self._intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
                self._align = rs.align(rs.stream.color)
                self._spa = rs.spatial_filter()
                self._tmp = rs.temporal_filter()
                return
            except Exception as e:
                pipe.stop()
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise

    def read(self):
        import pyrealsense2 as rs
        frames = None
        for _ in range(3):
            try:
                frames = self._pipeline.wait_for_frames(5000)
                break
            except RuntimeError:
                continue
        if frames is None:
            h, w = self._rgb_res
            return np.zeros((h, w, 3), dtype=np.uint8), np.zeros(self._depth_res[::-1], dtype=np.float32)
        aligned = self._align.process(frames)
        color = aligned.get_color_frame()
        depth_raw = aligned.get_depth_frame()
        if not color or not depth_raw:
            h, w = self._rgb_res
            return np.zeros((h, w, 3), dtype=np.uint8), np.zeros(self._depth_res[::-1], dtype=np.float32)
        f = self._spa.process(depth_raw)
        f = self._tmp.process(f)
        d = np.asanyarray(f.get_data()).astype(np.float32) / 1000.0
        d[d > 2.0] = 0
        self._depth_buf.append(d)
        if len(self._depth_buf) == 5:
            d = np.median(np.array(self._depth_buf), axis=0)
        rgb = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB)
        return rgb, d

    def read_rgb(self):
        return self.read()[0]

    def get_depth_at(self, u, v):
        import pyrealsense2 as rs
        frames = self._pipeline.wait_for_frames()
        d = frames.get_depth_frame()
        if d:
            z = d.get_distance(u, v)
            return z if z > 0 else 0.35
        return 0.35

    def deproject(self, u, v, z):
        import pyrealsense2 as rs
        return rs.rs2_deproject_pixel_to_point(self._intr, [u, v], z)

    def disconnect(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
