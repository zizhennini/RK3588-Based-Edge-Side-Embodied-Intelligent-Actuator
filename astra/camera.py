"""奥比中光 Astra Pro 相机封装 — RGB OpenCV + Depth C++ helper"""
import subprocess
import cv2
import numpy as np
from pathlib import Path

ASTRA_BIN = Path(__file__).parent / "build" / "astra_capture"


class AstraProCamera:
    def __init__(self, rgb_index: int = 21):
        self.rgb_index = rgb_index
        self.rgb_cap = None
        self._cwd = Path(__file__).parent
        self._depth_h = 0
        self._depth_w = 0

    def connect(self):
        self.rgb_cap = cv2.VideoCapture(self.rgb_index, cv2.CAP_V4L2)
        if not self.rgb_cap.isOpened():
            raise RuntimeError(f"无法打开 RGB 相机 /dev/video{self.rgb_index}")
        if not ASTRA_BIN.exists():
            raise FileNotFoundError("astra_capture 未编译")

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(3):
            ret, bgr = self.rgb_cap.read()
            if ret:
                break
        if not ret:
            raise RuntimeError("读取 RGB 帧失败")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 清理上一次残留
        for f in ["_depth.f32", "_depth.info"]:
            (self._cwd / f).unlink(missing_ok=True)

        result = subprocess.run(
            [str(ASTRA_BIN)], cwd=str(self._cwd),
            capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"depth 失败: {result.stderr.decode().strip()}")

        depth_path = self._cwd / "_depth.f32"
        info_path = self._cwd / "_depth.info"
        if not depth_path.exists() or not info_path.exists():
            raise RuntimeError("depth 文件未生成")

        with open(info_path) as f:
            w_str, h_str = f.read().strip().split()
            self._depth_w, self._depth_h = int(w_str), int(h_str)
        info_path.unlink()

        depth = np.fromfile(depth_path, dtype=np.float32).reshape(self._depth_h, self._depth_w)
        depth_path.unlink()
        return rgb, depth

    def disconnect(self):
        if self.rgb_cap:
            self.rgb_cap.release()
            self.rgb_cap = None
        # 清理 Depth 临时文件
        for f in ["_depth.f32", "_depth.info"]:
            (self._cwd / f).unlink(missing_ok=True)
