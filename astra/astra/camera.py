"""相机模块 — USBCamera + AstraProCamera"""
import cv2
import numpy as np
import subprocess
import threading
from pathlib import Path


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
