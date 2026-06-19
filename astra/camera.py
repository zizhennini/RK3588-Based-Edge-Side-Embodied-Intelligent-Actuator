"""通用 USB 相机封装 — 1080P 30fps + 后台线程无卡顿"""
import cv2
import numpy as np
import threading


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
