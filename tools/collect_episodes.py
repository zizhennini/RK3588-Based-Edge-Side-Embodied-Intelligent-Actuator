#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/collect_episodes.py — M1 演示数据采集（板端运行，一次连接多集连采）

数据规范::

    data/raw/<task>/episode_0001.json     单条 episode（与 lerobot-record-lite 同格式）
    data/raw/<task>/manifest.json         会话清单（质量校验结果 + 元数据 + 汇总统计）

用法::

    python tools/collect_episodes.py --task pick_place --episodes 50 --episode-time 20 \
        --leader-calib config/calibration_leader.json \
        --follower-calib config/calibration.json --max-step 15

流程（每条 episode 一致）:
    1. 把两臂摆回起始姿态 → Enter 开始
    2. 30Hz 跟随录制 episode-time 秒（每帧读回从臂角度，自动算追踪误差）
    3. 质量门控（帧率/丢帧/追踪误差/时长，阈值可用 CLI 覆盖）
       不合格 → [R] 重录该条 / [A] 仍然保留（标记 accepted_by_user）/ [Q] 退出
    4. 合格自动落盘 episode_XXXX.json 并写入 manifest.json

Ctrl-C: 录制中 = 结束该条并进入质量判定；等待输入中 = 结束会话并打印汇总。
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.episode_quality import (DEFAULT_THRESHOLDS, check_episode,  # noqa: E402
                                   check_images, dataset_stats, format_report)
from tools.cam_sink import CameraSet, JpegSink  # noqa: E402


def load_calib(path: str):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def next_index(task_dir: Path, start: int = 1) -> int:
    """扫描已有 episode_*.json，返回下一个可用序号（支持断点续采）"""
    idx = start
    while (task_dir / f"episode_{idx:04d}.json").exists():
        idx += 1
    return idx


def write_manifest(task_dir: Path, manifest: dict) -> None:
    manifest["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    manifest["summary"] = dataset_stats([
        {"file": e["file"], "frames": None, "quality": e["quality"]}
        for e in manifest["episodes"]
    ])
    tmp = task_dir / "manifest.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, task_dir / "manifest.json")


def ask_retry(ep_idx: int, quality: dict) -> str:
    """不合格时询问处置 → "retry" / "accept" / "quit" """
    print(f"\n✗ episode {ep_idx:04d} 质量不合格:")
    for r in quality["reasons"]:
        print(f"    - {r}")
    while True:
        try:
            ans = input("  [R] 重录  [A] 仍然保留(标记)  [Q] 退出会话: ").strip().lower()
        except KeyboardInterrupt:
            print()
            return "quit"
        if ans in ("r", "", "retry"):
            return "retry"
        if ans in ("a", "accept"):
            return "accept"
        if ans in ("q", "quit"):
            return "quit"


def main() -> int:
    ap = argparse.ArgumentParser(description="M1 演示数据采集（一次连接多集连采 + 质量门控）")
    ap.add_argument("--task", required=True, help="任务名（决定数据子目录，如 pick_place）")
    ap.add_argument("--episodes", type=int, default=50, help="本次采集条数")
    ap.add_argument("--episode-time", type=float, default=20.0, help="单条时长（秒）")
    ap.add_argument("--out-root", default="data/raw", help="数据根目录")
    ap.add_argument("--start-index", type=int, default=0,
                    help="起始序号（0=自动接续已有最大序号+1）")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--leader-port", default="/dev/ttyACM1")
    ap.add_argument("--follower-port", default="/dev/ttyACM0")
    ap.add_argument("--leader-calib", default="./config/calibration_leader.json")
    ap.add_argument("--follower-calib", default="./config/calibration.json")
    ap.add_argument("--max-step", type=float, default=15.0, help="G3 帧间限幅（度/帧）")
    # 质量门控阈值（默认见 episode_quality.DEFAULT_THRESHOLDS）
    ap.add_argument("--min-fps-ratio", type=float,
                    default=DEFAULT_THRESHOLDS["min_fps_ratio"])
    ap.add_argument("--max-drop-ratio", type=float,
                    default=DEFAULT_THRESHOLDS["max_drop_ratio"])
    ap.add_argument("--max-track-err", type=float,
                    default=DEFAULT_THRESHOLDS["max_track_err_deg"],
                    help="体关节平均追踪误差上限（度）")
    ap.add_argument("--min-duration-ratio", type=float,
                    default=DEFAULT_THRESHOLDS["min_duration_ratio"])
    ap.add_argument("--no-track-check", action="store_true",
                    help="关闭追踪误差门控（不读回从臂时使用）")
    # 相机（默认读 config/cameras.json，存在即启用；--no-camera 强制关闭）
    ap.add_argument("--cameras", default="./config/cameras.json",
                    help="相机配置 JSON（含 cameras 列表: realsense/usb）")
    ap.add_argument("--no-camera", action="store_true", help="本次不录图像（仅关节）")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--no-review", action="store_true",
                    help="不生成每集审核卡片（默认 <task>/review/episode_XXXX_review.jpg）")
    args = ap.parse_args()

    thresholds = {
        "min_fps_ratio": args.min_fps_ratio,
        "max_drop_ratio": args.max_drop_ratio,
        "max_track_err_deg": args.max_track_err,
        "min_duration_ratio": args.min_duration_ratio,
    }

    task_dir = Path(args.out_root) / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    start = args.start_index or next_index(task_dir)
    manifest_path = task_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest.setdefault("episodes", [])
    else:
        manifest = {
            "task": args.task,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "episode_time_s": args.episode_time,
            "target_fps": args.fps,
            "max_step_deg": args.max_step,
            "leader_calib": args.leader_calib,
            "follower_calib": args.follower_calib,
            "thresholds": thresholds,
            "episodes": [],
        }

    print("=" * 62)
    print(f"M1 采集: 任务={args.task}  计划 {args.episodes} 条 × {args.episode_time:.0f}s "
          f"@ {args.fps}fps")
    print(f"输出目录: {task_dir.resolve()}")
    print(f"起始序号: {start}")
    print("=" * 62)

    from hardware.teleop import TeleopPair
    pair = TeleopPair(
        leader_port=args.leader_port, follower_port=args.follower_port,
        fps=args.fps, leader_calibration=load_calib(args.leader_calib),
        follower_calibration_path=args.follower_calib,
        max_relative_step_deg=args.max_step,
    )

    cam_set = None
    cam_names = []
    cam_cfg_path = None if args.no_camera else args.cameras
    if cam_cfg_path and Path(cam_cfg_path).exists():
        try:
            cam_set = CameraSet.from_config(cam_cfg_path)
            cam_set.start()
            ready = cam_set.wait_ready(timeout_s=20.0)
            cam_names = [c for c, ok in ready.items() if ok]
            print(f"相机: {cam_names}（配置 {cam_cfg_path}）"
                  + ("" if len(cam_names) == len(ready) else
                     f"  ⚠ 未就绪: {[c for c, ok in ready.items() if not ok]}"))
        except Exception as e:
            print(f"⚠ 相机启动失败，本次仅录关节: {e}")
            cam_set = None
            cam_names = []
    else:
        print("相机: 未启用（无配置或无 --no-camera 之外的配置）")

    done = fails = 0
    try:
        with pair:
            idx = start
            while done < args.episodes:
                print(f"\n--- episode {idx:04d}（第 {done + 1}/{args.episodes} 条）---")
                print("把两臂摆到起始姿态，按 Enter 开始（Ctrl-C 结束会话）")
                try:
                    input()
                except (KeyboardInterrupt, EOFError):
                    print("\n会话结束（未开始本条）")
                    break

                img_dir = task_dir / f"episode_{idx:04d}_images"
                sink = None
                frame_cb = None
                if cam_set and cam_names:
                    sink = JpegSink(img_dir, cam_names, quality=args.jpeg_quality)
                    sink.start()

                    def frame_cb(i, elapsed, frame, _sink=sink,
                                 _cams=list(cam_names)):
                        """每关节帧抓一次各相机最新帧（序号与关节帧一一对应）"""
                        for c in _cams:
                            got = cam_set.latest(c)
                            if got is None:
                                continue
                            bgr, ts = got
                            _sink.push(c, bgr)
                            frame[f"ct_{c}"] = round(ts, 4)

                frames = pair.run(args.episode_time, out_path=None, follow=True,
                                  log_follower=not args.no_track_check,
                                  frame_cb=frame_cb)
                img_counts = sink.finish() if sink else {}
                quality = check_episode(frames, target_fps=args.fps,
                                        requested_s=args.episode_time,
                                        thresholds=thresholds)
                if img_counts:
                    img_q = check_images({c: v["written"] for c, v in img_counts.items()},
                                         quality["metrics"]["frames"])
                    quality["images"] = img_q
                    if not img_q["ok"]:
                        quality["ok"] = False
                        quality["reasons"] += img_q["reasons"]
                m = quality["metrics"]
                print(f"  帧数 {m['frames']}  时长 {m['duration_s']}s  "
                      f"fps {m['fps']}  丢帧 {m['dropped']}  "
                      + (f"追踪最差 J{m['track_worst_joint']} "
                         f"{m['track_worst_mean']}°" if m["track_has_data"] else "追踪未测")
                      + ("  图像 " + ", ".join(f"{c}:{v['written']}"
                                               for c, v in img_counts.items())
                         if img_counts else ""))

                accepted_by_user = False
                if quality["ok"]:
                    print("  ✓ 质量合格")
                else:
                    fails += 1
                    action = ask_retry(idx, quality)
                    if action == "retry":
                        continue
                    if action == "quit":
                        print("会话结束（当前条未保存）")
                        break
                    accepted_by_user = True

                out = task_dir / f"episode_{idx:04d}.json"
                pair.save(frames, str(out), duration_s=m["duration_s"])

                # 每集审核卡片（抽样帧拼图 + 关节/追踪曲线 + 质检结论）→ 人工 Approve 依据
                entry = {
                    "index": idx,
                    "file": out.name,
                    "frames": m["frames"],
                    "duration_s": m["duration_s"],
                    "fps": m["fps"],
                    "image_dir": img_dir.name if img_counts else None,
                    "images": img_counts or None,
                    "quality": quality,
                    "accepted_by_user": accepted_by_user,
                    "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
                }
                if not args.no_review:
                    try:
                        from tools.episode_review import build_card
                        card = build_card(out, task_dir / "review" /
                                          f"{out.stem}_review.jpg",
                                          target_fps=args.fps,
                                          requested_s=args.episode_time)
                        entry["review_card"] = str(Path(card["card"]).name)
                        print(f"  审核卡片: {card['card']}")
                    except Exception as e:
                        print(f"  ⚠ 审核卡片生成失败（不影响数据）: {e}")

                manifest["episodes"] = [e for e in manifest["episodes"]
                                        if e["index"] != idx] + [entry]
                manifest["episodes"].sort(key=lambda e: e["index"])
                write_manifest(task_dir, manifest)
                done += 1
                idx += 1
    except KeyboardInterrupt:
        print("\n会话中断")
    finally:
        if cam_set is not None:
            try:
                cam_set.stop()
            except Exception as e:
                print(f"⚠ 相机关闭异常: {e}")
        if manifest["episodes"]:
            write_manifest(task_dir, manifest)
            s = manifest["summary"]
            print("\n" + "=" * 62)
            kept = sum(1 for e in manifest["episodes"] if e["accepted_by_user"])
            print(f"会话汇总: 本次保存 {done} 条（其中用户强制保留 {kept} 条）")
            print(format_report(s))
            print(f"\nmanifest: {manifest_path.resolve()}")
            print("=" * 62)
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
