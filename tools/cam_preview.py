#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/cam_preview.py — 相机取景检查（固定安装/调位用，只读，不动机械臂）

按 config/cameras.json 逐个相机抓帧存图 + 打印实际协商参数与实测帧率，
用于确认【安装位置/视野/朝向】：

    python tools/cam_preview.py                       # 存到 ./cam_preview/
    python tools/cam_preview.py --out /tmp/preview --seconds 5 --frames 3

输出: <out>/<cam>_1.jpg ... 每相机若干张 + 实测 fps（<目标 0.8 倍会告警）。
检查要点:
  - front（D435i 第三人称）: 整个工作区（夹爪、物体、目标区）都在画面内，
    且与 config/settings.py CAMERA_POSITION 的手眼标定视角一致
  - wrist（末端 USB）: 夹爪与夹持目标清晰、不被自身结构遮挡
  - 图像上下/左右是否正确（不对就改 config/cameras.json 的 rotation）
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.cam_sink import CameraSet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="相机取景检查（只读）")
    ap.add_argument("--cameras", default="./config/cameras.json")
    ap.add_argument("--out", default="cam_preview", help="输出目录")
    ap.add_argument("--seconds", type=float, default=3.0, help="每相机测帧率时长")
    ap.add_argument("--frames", type=int, default=2, help="每相机存图张数")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cs = CameraSet.from_config(args.cameras)
    cs.start()
    ready = cs.wait_ready(timeout_s=25.0)
    print("就绪:", ready)
    names = [c for c, ok in ready.items() if ok]
    if not names:
        print("✗ 无可用相机")
        cs.stop()
        return 1

    import cv2
    for c in names:
        # 先取首帧存图
        saved = 0
        t0 = time.perf_counter()
        seen = set()
        n = 0
        while time.perf_counter() - t0 < args.seconds:
            got = cs.latest(c)
            if got is None:
                time.sleep(0.01)
                continue
            bgr, ts = got
            n += 1
            seen.add(ts)
            if saved < args.frames:   # 存最先拿到的若干帧（确定性，避免摩数取模漏存）
                saved += 1
                p = out / f"{c}_{saved}.jpg"
                cv2.imwrite(str(p), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                print(f"  存图 {p}  {bgr.shape}")
            time.sleep(1 / 30)
        dt = time.perf_counter() - t0
        fps = len(seen) / dt
        print(f"{c}: 唯一帧 {len(seen)} / 拉取 {n} → {fps:.1f} fps "
              f"({bgr.shape[1]}x{bgr.shape[0]})"
              + ("  ⚠ 帧率偏低" if fps < 24 else "  ✓"))
    cs.stop()
    print(f"\n取景图已存到 {out.resolve()} —— 打开确认视野/朝向，必要时调 rotation 或相机位置")
    return 0


if __name__ == "__main__":
    sys.exit(main())
