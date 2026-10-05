#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/cam_align_d435i.py — D435i 位置调整专用取景工具（实时预览 + 工作区覆盖度）

用途：**物理调整相机位置**时实时看视角，并量化"当前视角是否覆盖机械臂工作空间"。
标定出的工作区多边形会存到 `config/cam_align.json`，供后续采集与审核复用。

两种模式（自动切换）：
  * **实时预览**（有 DISPLAY 时，板端桌面直接跑）:
        python3 tools/cam_align_d435i.py
    快捷键:  q 退出 | s 存图 | m 鼠标点 4 角标定工作区并保存 | r 重置标定
             g 三分线 | c 中心十字 | t 目标区框 | v 覆盖度数值 | h 帮助
  * **无显示/远程（保存标注图，反复看同一张）**:
        python3 tools/cam_align_d435i.py --headless --out cam_align --interval 3
    每次覆盖写 `<out>/latest.jpg`（另有按序号归档），打印指标与建议。

判读要点（工具会实时给建议）:
  - 工作区占画面 **40%~70%** 为宜（太小=看不清细节，太大=末端容易出画面）
  - 工作区四角到画面边缘留 **≥5% 余量**（夹爪到最远端/最近端都不能出画面）
  - 画面亮度 60~190、清晰度（Laplacian 方差）≥80（糊了=没对焦/太近）
  - 机械臂本体（白色结构）不宜占据过多画面（会遮挡物体）

参数：
  --camera NAME     用 config/cameras.json 里的哪个相机（默认 front）
  --corners "x,y x,y x,y x,y"   直接给定工作区多边形（像素坐标，顺时针）
  --target-zone P   目标区边长占比（默认 0.6，画在画面中央作参考）
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.cam_sink import CameraSet  # noqa: E402

CFG_PATH = Path("config/cam_align.json")


# ---------------------------------------------------------------------------
# 指标与建议（纯函数，便于复用与测试）
# ---------------------------------------------------------------------------
def frame_metrics(bgr: np.ndarray) -> dict:
    """亮度/清晰度指标"""
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "shape": f"{bgr.shape[1]}x{bgr.shape[0]}",
        "brightness": float(gray.mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def coverage_metrics(poly: np.ndarray, width: int, height: int) -> dict:
    """工作区多边形相对画面的覆盖度指标"""
    import cv2
    area = abs(cv2.contourArea(poly.astype(np.float32)))
    frame_area = float(width * height)
    xs, ys = poly[:, 0], poly[:, 1]
    return {
        "coverage_pct": 100.0 * area / frame_area if frame_area else 0.0,
        "margin_left_pct": 100.0 * xs.min() / width,
        "margin_right_pct": 100.0 * (width - xs.max()) / width,
        "margin_top_pct": 100.0 * ys.min() / height,
        "margin_bottom_pct": 100.0 * (height - ys.max()) / height,
        "center_dx_pct": 100.0 * (xs.mean() - width / 2) / width,
        "center_dy_pct": 100.0 * (ys.mean() - height / 2) / height,
        "out_of_frame": bool((xs < 0).any() or (ys < 0).any()
                             or (xs >= width).any() or (ys >= height).any()),
    }


def advice(m: dict, cov: dict = None, web: bool = False) -> list:
    """把指标翻译成可执行建议 → [{"level","zh","en"}]

    zh 用于终端输出，en 用于画面叠加（cv2.putText 只支持 ASCII，中文会变 `?`）。
    """
    out = []

    def add(level, zh, en):
        out.append({"level": level, "zh": zh, "en": en})

    b, s = m["brightness"], m["sharpness"]
    if b < 60:
        add("warn", f"画面偏暗(亮度{b:.0f})：调亮环境光或避免逆光",
            f"! dark ({b:.0f}): add light / avoid backlight")
    elif b > 190:
        add("warn", f"画面偏亮(亮度{b:.0f})：避开强反光/直射光源",
            f"! overexposed ({b:.0f}): avoid glare")
    # 清晰度阈值取低值：桌面等平滑场景 Laplacian 方差本来就低（实测 ~35），
    # 只有明显异常才提示；调节位置时更应看该值的相对变化（越大越锐）
    if s < 20:
        add("warn", f"画面细节偏少(清晰度{s:.0f})：检查对焦/距离/光线",
            f"! low detail ({s:.0f}): check focus/distance/light")

    if not cov:
        if web:
            add("info",
                "工作区未标定：从网页图上读四角像素坐标（已叠 40px 网格），"
                "再用 --corners \"x,y x,y x,y x,y\" 重启本工具（会自动存 config/cam_align.json）",
                "workspace not marked: read 4 corners from grid, "
                "restart with --corners \"x,y x,y x,y x,y\"")
        else:
            add("info", "尚未标定工作区：按 m 用鼠标点工作区四角（或 --corners 传入）",
                "workspace not marked: press m or pass --corners")
        return out

    c = cov["coverage_pct"]
    if cov["out_of_frame"]:
        add("warn", "⚠ 工作区有角点超出画面：把相机后移/抬高，或调整俯仰",
            "! workspace corner OUT OF FRAME: pull back / raise camera")
    if c < 40:
        add("info", f"工作区只占画面 {c:.0f}%（偏小）：相机靠近或缩窄视野",
            f"coverage {c:.0f}% too small: move closer")
    elif c > 70:
        add("info", f"工作区占画面 {c:.0f}%（偏大）：相机后移，留出末端余量",
            f"coverage {c:.0f}% too large: move back")
    else:
        add("ok", f"工作区占画面 {c:.0f}% ✓ 合理", f"coverage {c:.0f}% OK")

    min_margin = min(cov["margin_left_pct"], cov["margin_right_pct"],
                     cov["margin_top_pct"], cov["margin_bottom_pct"])
    if min_margin < 5:
        label, _ = min((("左", "L"), cov["margin_left_pct"]),
                       (("右", "R"), cov["margin_right_pct"]),
                       (("上", "T"), cov["margin_top_pct"]),
                       (("下", "B"), cov["margin_bottom_pct"]),
                       key=lambda kv: kv[1])
        side_zh, side_en = label
        add("warn", f"⚠ {side_zh}侧余量仅 {min_margin:.1f}%（<5%）：该方向末端易出画面",
            f"! {side_en} margin {min_margin:.1f}% (<5%): arm may leave frame")
    if abs(cov["center_dx_pct"]) > 8:
        d = "右" if cov["center_dx_pct"] > 0 else "左"
        add("info", f"工作区中心偏{d} {abs(cov['center_dx_pct']):.0f}%：相机左右平移对齐",
            f"center off {d} {abs(cov['center_dx_pct']):.0f}%: pan camera")
    if abs(cov["center_dy_pct"]) > 8:
        d = "下" if cov["center_dy_pct"] > 0 else "上"
        add("info", f"工作区中心偏{d} {abs(cov['center_dy_pct']):.0f}%：相机上下平移对齐",
            f"center off {d} {abs(cov['center_dy_pct']):.0f}%: tilt camera")
    return out


def advice_lines(adv: list, key: str = "zh") -> list:
    """从建议列表取出指定语言的文本行"""
    return [a[key] for a in adv]


class MjpegServer:
    """极小 MJPEG over HTTP 服务（纯 stdlib）——不依赖 OpenCV GUI

    板端 OpenCV 为 `GUI: NONE` 构建（无 GTK/Qt），`cv2.imshow` 不可用；且 SSH 会话无 DISPLAY。
    因此实时预览改走浏览器：`http://<板端IP>:<port>/` 看流，`/snap.jpg` 取单帧。
    只依赖 http.server + cv2.imencode，无新增依赖。
    """

    PAGE = ("<html><head><title>cam_align</title></head>"
            "<body style='margin:0;background:#111;color:#ddd;font-family:monospace'>"
            "<div style='padding:6px'>实时取景（MJPEG）｜单帧: <a style='color:#6cf' "
            "href='/snap.jpg'>/snap.jpg</a>｜读数: <a style='color:#6cf' href='/state'>"
            "/state</a></div>"
            "<img src='/stream' style='width:100%;max-width:960px;display:block'>"
            "</body></html>")

    def __init__(self, port: int = 8080, quality: int = 80):
        self.port = int(port)
        self.quality = int(quality)
        self._jpeg = None
        self._state = {}
        self._count = 0
        self._lock = threading.Lock()
        self._srv = None

    def update(self, vis, state: dict = None):
        import cv2
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._count += 1
            if state:
                self._state = state

    def _current(self):
        with self._lock:
            return self._jpeg, self._count, dict(self._state)

    def start(self) -> str:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        me = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):        # 静音访问日志
                pass

            def do_GET(self):                 # noqa: N802
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    body = me.PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/state":
                    jpeg, cnt, state = me._current()
                    body = json.dumps({"frames": cnt, **state},
                                      ensure_ascii=False, indent=2).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path in ("/snap.jpg", "/snapshot"):
                    jpeg, _, _ = me._current()
                    if not jpeg:
                        self.send_error(503, "no frame yet")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                elif path == "/stream":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = -1
                    try:
                        while True:
                            jpeg, cnt, _ = me._current()
                            if jpeg is not None and cnt != last:
                                last = cnt
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                                 b"Content-Length: "
                                                 + str(len(jpeg)).encode()
                                                 + b"\r\n\r\n" + jpeg + b"\r\n")
                            time.sleep(0.02)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

        self._srv = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True,
                         name="mjpeg-server").start()
        return f"http://<板端IP>:{self.port}/"

    def stop(self):
        if self._srv:
            try:
                self._srv.shutdown()
            except Exception:
                pass
            self._srv = None


def local_ips() -> list:
    """本机所有 IPv4 地址（用于提示浏览器访问地址；排除 127.*）"""
    import socket
    import subprocess
    ips: list = []

    def _add(ip: str):
        if ip and ip not in ips and not ip.startswith("127."):
            ips.append(ip)

    try:                                    # iproute2 最可靠（板端已装）
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"],
                             capture_output=True, text=True, timeout=3).stdout
        for line in out.splitlines():
            parts = line.split()
            for i, tok in enumerate(parts):
                if tok == "inet" and i + 1 < len(parts):
                    _add(parts[i + 1].split("/")[0])
    except Exception:
        pass
    if not ips:                             # 回退：默认路由出口 + 主机名解析
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            _add(s.getsockname()[0])
            s.close()
        except Exception:
            pass
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None,
                                           socket.AF_INET):
                _add(info[4][0])
        except Exception:
            pass
    return ips


def cv2_gui_ok() -> tuple:
    """OpenCV 是否带 GUI 支持 → (ok, 原因)"""
    import cv2
    try:
        info = cv2.getBuildInformation()
        for line in info.splitlines():
            if line.strip().startswith("GUI:"):
                val = line.split(":", 1)[1].strip()
                if val.upper() in ("NONE", ""):
                    return False, "OpenCV 为 GUI: NONE 构建（无 GTK/Qt），cv2.imshow 不可用"
                return True, f"OpenCV GUI={val}"
    except Exception as e:      # noqa: BLE001
        return False, f"无法判定 OpenCV GUI 支持: {e}"
    return True, "未知（按可用处理）"


def draw_pixel_grid(img, step: int = 40, label_every: int = 80):
    """像素网格 + 坐标标注 —— 便于在浏览器里读工作区四角坐标（供 --corners 用）"""
    import cv2
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (70, 70, 70), 1)
        if x % label_every == 0:
            cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (70, 70, 70), 1)
        if y % label_every == 0:
            cv2.putText(img, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# 绘制
# ---------------------------------------------------------------------------
def draw_overlay(bgr: np.ndarray, show: dict, poly=None, target_zone=0.6,
                 metrics=None, fps=None, adv=None, help_text=False,
                 pixel_grid: int = 0) -> np.ndarray:
    """画面叠加（**只用 ASCII**：cv2.putText 不支持中文，否则渲染成 `?`）"""
    import cv2
    img = bgr.copy()
    h, w = img.shape[:2]
    if pixel_grid:
        img = draw_pixel_grid(img, step=pixel_grid, label_every=pixel_grid * 2)

    if show.get("thirds", True):
        for i in (1, 2):
            cv2.line(img, (w * i // 3, 0), (w * i // 3, h), (90, 90, 90), 1)
            cv2.line(img, (0, h * i // 3), (w, h * i // 3), (90, 90, 90), 1)
    if show.get("cross", True):
        cv2.line(img, (w // 2 - 14, h // 2), (w // 2 + 14, h // 2), (0, 220, 255), 2)
        cv2.line(img, (w // 2, h // 2 - 14), (w // 2, h // 2 + 14), (0, 220, 255), 2)
    if show.get("zone", True):
        bw, bh = int(w * target_zone), int(h * target_zone)
        x0, y0 = (w - bw) // 2, (h - bh) // 2
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (0, 180, 0), 1)
        cv2.putText(img, "target zone", (x0 + 4, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1, cv2.LINE_AA)
    if poly is not None and len(poly) >= 3:
        cv2.polylines(img, [poly.astype(np.int32)], True, (0, 165, 255), 2)
        for i, (x, y) in enumerate(poly.astype(int)):
            cv2.circle(img, (int(x), int(y)), 4, (0, 165, 255), -1)
            cv2.putText(img, str(i + 1), (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

    lines = []
    if metrics:
        lines.append(f"{metrics['shape']}  fps {fps:.1f}" if fps else metrics["shape"])
        lines.append(f"bright {metrics['brightness']:.0f}  sharp {metrics['sharpness']:.0f}")
    if show.get("cov", True) and poly is not None and len(poly) >= 3:
        cov = coverage_metrics(poly, w, h)
        lines.append(f"workspace {cov['coverage_pct']:.0f}%  margins "
                     f"L{cov['margin_left_pct']:.0f} R{cov['margin_right_pct']:.0f} "
                     f"T{cov['margin_top_pct']:.0f} B{cov['margin_bottom_pct']:.0f}%")
    lines += advice_lines(adv or [], key="en")
    y = 22
    for t in lines[:12]:
        color = (0, 0, 255) if t.startswith("!") else (255, 255, 255)
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 20
    if help_text:
        txt = ("q quit | s snap | m mark workspace(4 clicks) | r reset | "
               "g grid | c cross | t zone | v coverage | h help")
        cv2.putText(img, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if poly is None or len(poly) < 3:
        cv2.putText(img, "corner order when marking: 1 (x1,y1) 2 3 4",
                    (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
    return img


def load_poly(corners_arg, cfg_path=CFG_PATH):
    if corners_arg:
        pts = [tuple(float(v) for v in p.split(",")) for p in corners_arg.split()]
        return np.array(pts, dtype=np.float32)
    if cfg_path.exists():
        try:
            d = json.loads(cfg_path.read_text(encoding="utf-8"))
            if d.get("corners"):
                return np.array(d["corners"], dtype=np.float32)
        except Exception:
            pass
    return None


def save_poly(poly, camera: str, cfg_path=CFG_PATH):
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({
        "camera": camera,
        "corners": [[float(x), float(y)] for x, y in poly],
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "工作区多边形（像素，顺时针）；由 tools/cam_align_d435i.py 标定，"
                "供采集/审核工具绘制参考",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ 工作区多边形已保存: {cfg_path.resolve()}")


def main() -> int:
    # 重定向到文件/管道时默认全缓冲会让 headless 模式下看不到实时指标 → 改行缓冲
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="D435i 位置调整取景工具")
    ap.add_argument("--cameras", default="./config/cameras.json")
    ap.add_argument("--camera", default="front", help="用哪个相机（默认 front=D435i）")
    ap.add_argument("--headless", action="store_true", help="强制无界面（保存标注图）")
    ap.add_argument("--out", default="cam_align", help="headless 输出目录")
    ap.add_argument("--interval", type=float, default=3.0, help="headless 存图间隔（秒）")
    ap.add_argument("--corners", default=None,
                    help='工作区四角像素坐标，如 "120,90 520,90 540,420 100,430"')
    ap.add_argument("--target-zone", type=float, default=0.6, help="目标区边长占比（画面参考框）")
    ap.add_argument("--workspace-w", type=float, default=30.0,
                    help="工作区宽度（cm，默认 30；用于架设距离建议）")
    ap.add_argument("--workspace-h", type=float, default=40.0,
                    help="工作区进深受限尺寸（cm，默认 40；用于架设距离建议）")
    ap.add_argument("--target-coverage", type=float, default=0.55,
                    help="期望工作区占画面比例（默认 0.55；用于架设距离建议）")
    ap.add_argument("--serve", type=int, default=0, nargs="?",
                    help="启动 MJPEG 网页实时预览（端口，默认 8080）；"
                         "OpenCV 无 GUI 或无 DISPLAY 时自动启用")
    ap.add_argument("--pixel-grid", type=int, default=0,
                    help="叠加像素网格步长（如 40）；便于从网页图读工作区四角坐标")
    args = ap.parse_args()

    cov_target = args.target_coverage

    import cv2
    # 是否能开本地窗口：需要 (a) 有 DISPLAY (b) OpenCV 带 GUI 支持
    gui_ok, gui_why = cv2_gui_ok()
    have_display = bool(os.environ.get("DISPLAY"))
    gui = (not args.headless) and have_display and gui_ok
    if not gui and not args.headless:
        reasons = []
        if not have_display:
            reasons.append("本会话无 DISPLAY（SSH 会话常见；板端桌面在 :1）")
        if not gui_ok:
            reasons.append(gui_why)
        print("无法开本地窗口 —— " + "；".join(reasons))
        print("→ 改用 **网页实时预览**（MJPEG）")
    server = None
    if not gui or args.serve:
        port = args.serve or 8080
        server = MjpegServer(port=port)
        url = server.start()
        ips = local_ips() or ["<板端IP>"]
        print(f"✓ 实时预览已启动：请用浏览器打开 "
              + " 或 ".join(f"http://{ip}:{port}/" for ip in ips))
        print(f"  单帧快照 http://{ips[0]}:{port}/snap.jpg ｜ 指标 JSON "
              f"http://{ips[0]}:{port}/state")
        print("  （网页是实时的，直接看着画面调相机位置；按 Ctrl-C 结束）")
        if not args.pixel_grid:
            args.pixel_grid = 40      # 网页模式默认叠像素网格，便于读工作区四角坐标
            print("  已默认叠加像素网格（每 40px，标注每 80px）——"
                  "读四角坐标后用 --corners \"x,y x,y x,y x,y\" 标定工作区")

    cs = CameraSet.from_config(args.cameras)
    cs.start()
    ready = cs.wait_ready(timeout_s=25.0)
    if not ready.get(args.camera):
        print(f"✗ 相机 {args.camera} 未就绪（可用: {ready}）")
        cs.stop()
        return 1

    # 视场角/架设距离建议（基于相机实测内参；D435i RGB 为针孔模型）
    try:
        from hardware.camera_d435i import plan_distance_m, visible_size_m, video_fov_deg
        intr = None
        cam_obj = cs._cams.get(args.camera)
        if hasattr(cam_obj, "intrinsics_summary"):
            intr = cam_obj.intrinsics_summary()
        if intr:
            print(f"\n相机内参实测: {intr['width']}x{intr['height']} "
                  f"fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
                  f"→ HFOV {intr['hfov_deg']:.1f}°  VFOV {intr['vfov_deg']:.1f}°")
            hf, vf = intr["hfov_deg"], intr["vfov_deg"]
            print("可见范围估算（相机到桌面作业面的垂直距离 d）:")
            for d in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
                w, h = visible_size_m(d, hf, vf)
                print(f"   d={d:.1f}m → 可见 {w * 100:.0f}×{h * 100:.0f} cm"
                      f"（工作区 30×40cm 占画面 {30 / (w * 100) * 100:.0f}%）")
            dw, dh, d_use = plan_distance_m(args.workspace_w / 100.0,
                                            args.workspace_h / 100.0, hf, vf,
                                            cov_target)
            print(f"→ 建议架设距离 ≈ {d_use:.2f} m"
                  f"（按工作区 {args.workspace_w:.0f}×{args.workspace_h:.0f}cm "
                  f"占画面 {cov_target:.0%}；宽/高约束分别在 "
                  f"{dw:.2f}m / {dh:.2f}m，取大者再留 10% 余量）")
            print("   注：以上为光轴垂直正对时的估算；相机倾斜时按沿光轴距离计，"
                  "并用 m 标定工作区后的覆盖率实测值校正\n")
    except Exception as e:
        print(f"（内参/距离建议不可用: {e}）\n")

    poly = load_poly(args.corners)
    if poly is not None:
        print(f"已载入工作区多边形（{len(poly)} 点）: {poly.astype(int).tolist()}")
        if args.corners:   # 命令行给定的多边形也持久化，供采集/审核复用
            save_poly(poly, args.camera)
    show = {"thirds": True, "cross": True, "zone": True, "cov": True}
    help_text = True
    clicks = []
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq, fps, last_t, uniq, last_ts = 0, 0.0, time.perf_counter(), 0, None
    last_save = 0.0

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            if len(clicks) == 4:
                p = np.array(clicks, dtype=np.float32)
                save_poly(p, args.camera)
                print(f"  覆盖度: {coverage_metrics(p, w, h)['coverage_pct']:.0f}%")

    if gui:
        cv2.namedWindow("D435i align", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("D435i align", on_mouse)

    print(__doc__.split("用途：")[0].strip())
    print(f"相机: {args.camera} | 模式: {'实时预览' if gui else '存图'} | 目标区 {args.target_zone:.0%}")
    if gui:
        print("快捷键: q 退出 | s 存图 | m 标定工作区 | r 重置 | g/c/t/v 切换叠加 | h 帮助")

    try:
        while True:
            got = cs.latest(args.camera)
            if got is None:
                time.sleep(0.02)
                continue
            bgr, ts = got
            bgr = np.ascontiguousarray(bgr)
            h, w = bgr.shape[:2]
            now = time.perf_counter()
            # fps 只统计**相机新帧**（按时间戳变化计），否则会把"拉取率"误报成 2 倍帧率
            if ts != last_ts:
                uniq += 1
                last_ts = ts
            if now - last_t >= 1.0:
                fps, uniq, last_t = uniq / (now - last_t), 0, now
            metrics = frame_metrics(bgr)
            cur_poly = np.array(clicks, dtype=np.float32) if len(clicks) == 4 else poly
            adv = advice(metrics, coverage_metrics(cur_poly, w, h)
                         if cur_poly is not None and len(cur_poly) >= 3 else None,
                         web=server is not None)
            vis = draw_overlay(bgr, show, cur_poly, args.target_zone, metrics, fps,
                               adv, help_text, pixel_grid=args.pixel_grid)

            if server is not None:
                server.update(vis, {
                    "mode": "web", "fps": round(fps, 1),
                    "brightness": round(metrics["brightness"], 1),
                    "sharpness": round(metrics["sharpness"], 1),
                    "advice_zh": advice_lines(adv, "zh"),
                    "coverage_pct": (round(coverage_metrics(cur_poly, w, h)["coverage_pct"], 1)
                                     if cur_poly is not None and len(cur_poly) >= 3 else None),
                })

            if gui:
                cv2.imshow("D435i align", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    p = out_dir / f"snap_{time.strftime('%H%M%S')}.jpg"
                    cv2.imwrite(str(p), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    print(f"  存图: {p.resolve()}")
                if key == ord("m"):
                    clicks.clear()
                    print("  请用鼠标依次点击工作区四角（顺时针：左下→左上→右上→右下）")
                if key == ord("r"):
                    clicks.clear()
                    print("  已重置标定（继续用已保存的多边形）")
                if key == ord("g"):
                    show["thirds"] = not show["thirds"]
                if key == ord("c"):
                    show["cross"] = not show["cross"]
                if key == ord("t"):
                    show["zone"] = not show["zone"]
                if key == ord("v"):
                    show["cov"] = not show["cov"]
                if key == ord("h"):
                    help_text = not help_text
            elif server is None and now - last_save >= args.interval:
                last_save = now
                cv2.imwrite(str(out_dir / "latest.jpg"), vis,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                seq += 1
                cv2.imwrite(str(out_dir / f"snap_{seq:03d}.jpg"), vis,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                print(f"[{seq:03d}] {metrics['shape']} fps {fps:.1f} "
                      f"亮度 {metrics['brightness']:.0f} 清晰度 {metrics['sharpness']:.0f}")
                for t in advice_lines(adv, "zh"):
                    print(f"      {t}")
                print("      → 看 <out>/latest.jpg（远程: scp 回来或 rsync）")
            elif server is not None and now - last_save >= args.interval:
                # 网页模式下降低打印频率（画面在浏览器里看），但仍周期性输出指标
                last_save = now
                seq += 1
                print(f"[{seq:03d}] fps {fps:.1f} 亮度 {metrics['brightness']:.0f} "
                      f"清晰度 {metrics['sharpness']:.0f}")
                for t in advice_lines(adv, "zh"):
                    print(f"      {t}")
    except KeyboardInterrupt:
        print("\n结束")
    finally:
        cs.stop()
        if server is not None:
            server.stop()
        if gui:
            cv2.destroyAllWindows()

    if len(clicks) == 4:
        save_poly(np.array(clicks, dtype=np.float32), args.camera)
    print(f"输出目录: {out_dir.resolve()}")
    if poly is None and len(clicks) < 4:
        print("提示: 标定工作区后（m 或 --corners），工具才能给出覆盖度/余量建议")
    return 0


if __name__ == "__main__":
    sys.exit(main())
