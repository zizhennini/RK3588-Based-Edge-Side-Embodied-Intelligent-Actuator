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


def advice(m: dict, cov: dict = None) -> list:
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


# ---------------------------------------------------------------------------
# 绘制
# ---------------------------------------------------------------------------
def draw_overlay(bgr: np.ndarray, show: dict, poly=None, target_zone=0.6,
                 metrics=None, fps=None, adv=None, help_text=False) -> np.ndarray:
    """画面叠加（**只用 ASCII**：cv2.putText 不支持中文，否则渲染成 `?`）"""
    import cv2
    img = bgr.copy()
    h, w = img.shape[:2]

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
    ap.add_argument("--target-zone", type=float, default=0.6, help="目标区边长占比")
    args = ap.parse_args()

    import cv2
    gui = (not args.headless) and bool(os.environ.get("DISPLAY"))
    if not gui and not args.headless:
        print("未检测到 DISPLAY（远程/无桌面）→ 自动转为保存标注图模式；"
              "每 %.0fs 覆盖写 <out>/latest.jpg" % args.interval)

    cs = CameraSet.from_config(args.cameras)
    cs.start()
    ready = cs.wait_ready(timeout_s=25.0)
    if not ready.get(args.camera):
        print(f"✗ 相机 {args.camera} 未就绪（可用: {ready}）")
        cs.stop()
        return 1

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
                         if cur_poly is not None and len(cur_poly) >= 3 else None)
            vis = draw_overlay(bgr, show, cur_poly, args.target_zone, metrics, fps,
                               adv, help_text)

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
            elif now - last_save >= args.interval:
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
    except KeyboardInterrupt:
        print("\n结束")
    finally:
        cs.stop()
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
