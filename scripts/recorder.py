#!/usr/bin/env python3
"""硬件加速录像 — D435i/USB 摄像头 → h264_rkmpp 编码"""

import sys, os, time, subprocess, signal, cv2
from pathlib import Path
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera import D435iCamera


class Recorder:
    def __init__(self, out_dir="./recordings", fps=15,
                 width=640, height=480, bitrate="5M",
                 camera="d435i", camera_index=27):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps; self.width = width; self.height = height
        self.bitrate = bitrate; self.camera = camera
        self.camera_index = camera_index
        self._process = None; self._cam = None
        self._cap = None; self._running = False

    def _init_camera(self):
        src_h, src_w = 480, 640
        if self.camera == "usb":
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                raise RuntimeError(f"无法打开 USB 摄像头 /dev/video{self.camera_index}")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, frame = self._cap.read()
            if ret: src_h, src_w = frame.shape[:2]
            print(f"[Recorder] USB: {src_w}x{src_h}")
        else:
            self._cam = D435iCamera()
            self._cam.connect()
            rgb, _ = self._cam.read()
            src_h, src_w = rgb.shape[:2]
            print(f"[Recorder] D435i: {src_w}x{src_h}")
        return src_w, src_h

    def start(self, name=None):
        if self._running: raise RuntimeError("已在录像中")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{name or 'record'}_{ts}.mp4"
        out = str(self.out_dir / fname)
        src_w, src_h = self._init_camera()
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{src_w}x{src_h}", "-r", str(self.fps), "-i", "-",
               "-c:v", "h264_rkmpp", "-b:v", self.bitrate,
               "-r", str(self.fps), out]
        self._process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._running = True
        print(f"[Recorder] 开始: {out}")
        return out

    def _read_frame(self):
        if self.camera == "usb":
            ret, frame = self._cap.read()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes() if ret else None
        rgb, _ = self._cam.read()
        return rgb.tobytes()

    def record(self, duration=5):
        """同步录制（duration=0 表示手动 Ctrl+C 停止）"""
        ok = (self._process and self._process.stdin and
              ((self.camera == "usb" and self._cap and self._cap.isOpened()) or self._cam))
        if not ok: return
        t0 = time.time()
        try:
            while self._running and (duration <= 0 or time.time() - t0 < duration):
                data = self._read_frame()
                if data: self._process.stdin.write(data)
                time.sleep(1.0 / self.fps)
        except (BrokenPipeError, OSError):
            pass
        except Exception as e:
            print(f"[Recorder] 异常: {e}")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self._process:
            try: self._process.stdin.close()
            except: pass
            self._process.wait(timeout=5)
            self._process = None
        if self.camera == "usb" and self._cap:
            self._cap.release(); self._cap = None
        if self._cam:
            try: self._cam.disconnect()
            except: pass
            self._cam = None
        print("[Recorder] 已停止")


_recorder = None

def handler(sig, frame):
    if _recorder: _recorder.stop()
    sys.exit(0)

def main():
    global _recorder
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="./recordings")
    p.add_argument("--name"); p.add_argument("--fps", type=int, default=15)
    p.add_argument("--bitrate", default="5M")
    p.add_argument("--duration", type=float, default=5, help="秒数, 0=手动Ctrl+C")
    p.add_argument("--camera", default="d435i", choices=["d435i","usb"])
    p.add_argument("--camera-index", type=int, default=27)
    p.add_argument("--osd", default="")
    a = p.parse_args()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    kw = dict(camera=a.camera, camera_index=a.camera_index)

    if a.osd:
        _recorder = OSDRecorder(out_dir=a.out_dir, fps=a.fps, bitrate=a.bitrate, **kw)
        _recorder.start(name=a.name, osd_text=a.osd)
    else:
        _recorder = Recorder(out_dir=a.out_dir, fps=a.fps, bitrate=a.bitrate, **kw)
        _recorder.start(name=a.name)

    if a.duration > 0:
        print(f"[Recorder] 自动停止 ({a.duration}s)")
        _recorder.record(a.duration)
    else:
        print("[Recorder] Ctrl+C 停止")
        _recorder.record(0)


class OSDRecorder(Recorder):
    def start(self, name=None, osd_text=""):
        if self._running: raise RuntimeError("已在录像中")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{name or 'record'}_{ts}.mp4"
        out = str(self.out_dir / fname)
        src_w, src_h = self._init_camera()
        escaped = osd_text.replace("'", "'\\\\''").replace(":", "\\\\:")
        drawtext = f"drawtext=text='{escaped}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5"
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{src_w}x{src_h}", "-r", str(self.fps), "-i", "-",
               "-vf", drawtext,
               "-c:v", "h264_rkmpp", "-b:v", self.bitrate,
               "-r", str(self.fps), out]
        self._process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._running = True
        print(f"[Recorder] OSD 开始: {out}")
        return out


if __name__ == "__main__":
    main()
