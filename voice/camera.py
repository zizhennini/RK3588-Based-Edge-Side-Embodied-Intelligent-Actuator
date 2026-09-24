from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class CameraAdapter:
    def __init__(self, config: dict):
        self.photo_dir = Path(config["paths"]["photo_dir"])
        self.camera_index = int(config["audio"].get("camera_index", 21))

    def capture(self) -> Path:
        self.photo_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.photo_dir / f"voice_{timestamp}.jpg"
        bgr = self._capture_bgr(timestamp)
        cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
        return out

    def _capture_bgr(self, timestamp: str) -> np.ndarray:
        try:
            import pyrealsense2 as rs
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipe.start(cfg)
            sensor = profile.get_device().first_color_sensor()
            for opt in (rs.option.enable_auto_exposure,
                        rs.option.enable_auto_white_balance):
                if sensor.supports(opt):
                    sensor.set_option(opt, 1)
            for _ in range(15):
                pipe.wait_for_frames()
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            pipe.stop()
            if color:
                return np.asanyarray(color.get_data())
        except Exception:
            pass
        import subprocess
        dev = f"/dev/video{self.camera_index}"
        tmp = f"/tmp/voice_cap_{timestamp}.jpg"
        subprocess.run(["ffmpeg", "-f", "v4l2", "-video_size", "640x480",
            "-i", dev, "-vframes", "1", "-y", tmp], capture_output=True, timeout=10)
        img = cv2.imread(tmp)
        if img is not None:
            Path(tmp).unlink(missing_ok=True)
            return img
        raise RuntimeError(f"无法从 {dev} 获取图像")
