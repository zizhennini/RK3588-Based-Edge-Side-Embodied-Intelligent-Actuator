#!/usr/bin/env python3
"""RK3588-EIA Web 控制界面"""

import sys, os, json, io, base64, threading, time, subprocess, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOST = "0.0.0.0"
PORT = 8080
PROJ = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "命令超时"
    except Exception as e:
        return str(e)


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        elif self.path == "/snapshot":
            self._snapshot()
        elif self.path == "/dashboard":
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        elif self.path.startswith("/run/"):
            cmd = urllib.parse.unquote(self.path[5:])
            out = run_cmd(cmd)
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(out.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def _snapshot(self):
        try:
            import cv2, pyrealsense2 as rs
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipe.start(cfg)
            for _ in range(10): pipe.wait_for_frames()
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            img = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2JPEG)
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            pipe.stop()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(buf.tobytes())
        except:
            import numpy as np
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Camera N/A", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            _, buf = cv2.imencode(".jpg", img)
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(buf.tobytes())

    def log_message(self, *a):
        pass


HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RK3588-EIA 控制面板</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
h1 { text-align: center; margin-bottom: 20px; color: #e94560; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-bottom: 20px; }
.btn { padding: 15px; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; color: white; }
.btn-fn { background: #16213e; }
.btn-fn:hover { background: #0f3460; }
.btn-stop { background: #e94560; }
.btn-stop:hover { background: #c23152; }
#output { background: #0f0f23; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 13px; height: 300px; overflow-y: auto; white-space: pre-wrap; margin-bottom: 20px; }
#snapshot { width: 100%; max-width: 640px; border-radius: 8px; display: block; margin: 0 auto 20px; }
.controls { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
.controls input { flex: 1; min-width: 200px; padding: 10px; border-radius: 6px; border: none; background: #16213e; color: white; font-size: 14px; }
.controls button { padding: 10px 20px; border-radius: 6px; border: none; cursor: pointer; font-size: 14px; }
.status { text-align: center; color: #888; margin-bottom: 20px; }
</style>
</head>
<body>
<h1>RK3588-EIA</h1>
<div class="status" id="status">就绪</div>

<div class="grid">
<button class="btn btn-fn" onclick="run('python3 va.py once --seconds 4 --no-speak --no-play')">语音识别</button>
<button class="btn btn-fn" onclick="run('python3 va.py ask 画面中有什么 --no-speak --no-play')">VLM 拍照</button>
<button class="btn btn-fn" onclick="run('python3 va.py ask 1+1等于几 --no-speak --no-play')">VLM 文字</button>
<button class="btn btn-fn" onclick="run('python3 scripts/recorder.py --duration 5')">录像 5s</button>
<button class="btn btn-fn" onclick="run('python3 scripts/record_trajectory.py list')">动作库</button>
<button class="btn btn-fn" onclick="refreshSnapshot()">刷新画面</button>
<button class="btn btn-stop" onclick="run('pkill -f va.py; pkill -f demo')">停止所有</button>
</div>

<div class="controls">
<input id="cmdInput" placeholder="输入自定义命令..." onkeydown="if(event.key==='Enter') run(this.value)">
<button class="btn btn-fn" onclick="run(document.getElementById('cmdInput').value)">执行</button>
</div>

<div id="output">欢迎使用 RK3588-EIA</div>
<img id="snapshot" src="/snapshot" alt="相机画面">

<script>
function run(cmd) {
    if (!cmd) return;
    document.getElementById('status').textContent = '执行中...';
    var out = document.getElementById('output');
    out.textContent += '\\n$ ' + cmd + '\\n';
    out.scrollTop = out.scrollHeight;
    fetch('/run/' + encodeURIComponent(cmd)).then(r => r.text()).then(t => {
        out.textContent += t + '\\n';
        out.scrollTop = out.scrollHeight;
        document.getElementById('status').textContent = '就绪';
    });
}
function refreshSnapshot() {
    document.getElementById('snapshot').src = '/snapshot?' + new Date().getTime();
}
setInterval(refreshSnapshot, 5000);
</script>
</body>
</html>
"""


def main():
    import argparse
    p = argparse.ArgumentParser(description="RK3588-EIA Web 控制面板")
    p.add_argument("--port", type=int, default=PORT)
    a = p.parse_args()

    server = HTTPServer((HOST, a.port), Handler)
    print(f"打开浏览器访问: http://{HOST}:{a.port}")
    print(f"本机访问: http://localhost:{a.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止")
        server.server_close()


if __name__ == "__main__":
    main()
