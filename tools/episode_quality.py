#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/episode_quality.py — M1 采集质量校验与数据集统计（纯函数，无硬件依赖）

被 `tools/collect_episodes.py`（采集时门控）与 `tools/dataset_stats.py`（事后统计）共用，
也用于单元测试。输入为录制 JSON 的 frames 列表::

    [{"J1": deg, ..., "J6": deg, "t": sec, "F1": deg, ..., "F6": deg}, ...]
    J* = 主臂（指令）角度；F* = 从臂实际角度（--log-follower 时存在，用于追踪误差）

质量门控项（默认阈值见 DEFAULT_THRESHOLDS）:
  - 帧率: 实测 fps ≥ 目标 × min_fps_ratio
  - 丢帧: 帧间隔 > 1.5×周期 的帧占比 ≤ max_drop_ratio
  - 追踪: 体关节(J1-J5)逐关节平均 |从臂实际−指令| ≤ max_track_err_deg
  - 时长: 实测时长 ≥ 请求时长 × min_duration_ratio
"""
from typing import Dict, List, Optional, Sequence

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
BODY_JOINTS = (1, 2, 3, 4, 5)      # 追踪门控只看体关节（夹爪响应特性不同，单独报告）

DEFAULT_THRESHOLDS = {
    "min_fps_ratio": 0.9,
    "max_drop_ratio": 0.01,
    "max_track_err_deg": 5.0,
    "min_duration_ratio": 0.95,
    "max_stale_ratio": 0.01,      # 陈旧图像帧占比上限（>0.2s 的帧，见 stale_frame_stats）
}


def frame_keys_ok(fr: dict) -> bool:
    """该帧六个关节角度是否齐全（缺任一 = 该帧无效）"""
    return all(f"J{i}" in fr for i in range(1, 7))


def interval_stats(frames: Sequence[dict], target_fps: float = 30.0) -> Dict[str, float]:
    """帧间隔统计 → {median_dt, max_dt, drop_ratio, dropped}"""
    ts = [float(fr.get("t", 0.0)) for fr in frames]
    if len(ts) < 2:
        return {"median_dt": 0.0, "max_dt": 0.0, "drop_ratio": 0.0, "dropped": 0.0}
    dts = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    srt = sorted(dts)
    median_dt = srt[len(srt) // 2]
    limit = 1.5 / float(target_fps)
    dropped = sum(1 for d in dts if d > limit)
    return {
        "median_dt": median_dt,
        "max_dt": max(dts),
        "drop_ratio": dropped / len(dts),
        "dropped": float(dropped),
    }


def tracking_errors(frames: Sequence[dict]) -> Dict[int, List[float]]:
    """逐关节追踪误差列表 {joint_id: [|F−J|, ...]}（无 F 字段则返回空）"""
    errs: Dict[int, List[float]] = {i: [] for i in range(1, 7)}
    for fr in frames:
        for i in range(1, 7):
            j, fkey = fr.get(f"J{i}"), fr.get(f"F{i}")
            if j is None or fkey is None:
                continue
            errs[i].append(abs(float(fkey) - float(j)))
    return errs


def tracking_summary(frames: Sequence[dict]) -> Dict[str, object]:
    """追踪误差汇总 → {per_joint_mean, per_joint_max, worst_joint, worst_mean, has_data}"""
    errs = tracking_errors(frames)
    mean = {i: (sum(v) / len(v) if v else None) for i, v in errs.items()}
    mx = {i: (max(v) if v else None) for i, v in errs.items()}
    body = {i: mean[i] for i in BODY_JOINTS if mean[i] is not None}
    worst_joint = max(body, key=lambda i: body[i]) if body else None
    return {
        "per_joint_mean": mean,
        "per_joint_max": mx,
        "worst_joint": worst_joint,
        "worst_mean": body[worst_joint] if worst_joint else None,
        "has_data": bool(body),
    }


def check_episode(frames: Sequence[dict], target_fps: float = 30.0,
                  requested_s: Optional[float] = None,
                  thresholds: Optional[dict] = None) -> Dict[str, object]:
    """单条 episode 质量校验

    Returns:
        {"ok": bool, "reasons": [str, ...], "metrics": {...}}
    """
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    reasons: List[str] = []

    valid = [fr for fr in frames if frame_keys_ok(fr)]
    invalid = len(frames) - len(valid)
    ts = [float(fr.get("t", 0.0)) for fr in valid]
    duration = (ts[-1] + 1.0 / target_fps) if ts else 0.0
    measured_fps = len(valid) / duration if duration > 0 else 0.0
    iv = interval_stats(valid, target_fps)
    tr = tracking_summary(valid)
    sf = stale_frame_stats(valid)

    metrics = {
        "frames": len(valid),
        "invalid_frames": invalid,
        "duration_s": round(duration, 3),
        "fps": round(measured_fps, 2),
        "median_dt": round(iv["median_dt"], 4),
        "max_dt": round(iv["max_dt"], 4),
        "drop_ratio": round(iv["drop_ratio"], 4),
        "dropped": int(iv["dropped"]),
        "stale_frames": int(sf["total"]),
        "stale_ratio": round(sf["ratio"], 4),
        "stale_max_age": round(sf["max_age"], 3),
        "track_worst_joint": tr["worst_joint"],
        "track_worst_mean": (round(tr["worst_mean"], 2)
                             if tr["worst_mean"] is not None else None),
        "track_has_data": tr["has_data"],
    }

    if not valid:
        reasons.append("无有效帧（全部帧缺关节数据）")
    else:
        if measured_fps < target_fps * th["min_fps_ratio"]:
            reasons.append(f"帧率不足: 实测 {measured_fps:.1f} < 目标 {target_fps}×"
                           f"{th['min_fps_ratio']:.2f}={target_fps * th['min_fps_ratio']:.1f}")
        if iv["drop_ratio"] > th["max_drop_ratio"]:
            reasons.append(f"丢帧过多: {int(iv['dropped'])} 帧间隔 >{1.5 / target_fps:.3f}s"
                           f"（占比 {iv['drop_ratio']:.1%} > {th['max_drop_ratio']:.0%}）")
        if invalid:
            reasons.append(f"无效帧 {invalid} 条（缺关节数据）")
        if tr["has_data"] and tr["worst_mean"] > th["max_track_err_deg"]:
            reasons.append(f"追踪误差超限: 最差 J{tr['worst_joint']} 平均 "
                           f"{tr['worst_mean']:.1f}° > {th['max_track_err_deg']}°")
        if requested_s and duration < requested_s * th["min_duration_ratio"]:
            reasons.append(f"时长不足: {duration:.1f}s < 请求 {requested_s}s×"
                           f"{th['min_duration_ratio']:.2f}")
        if sf["ratio"] > th["max_stale_ratio"]:
            reasons.append(f"陈旧图像帧过多: {int(sf['total'])} 帧（占比 {sf['ratio']:.1%} > "
                           f"{th['max_stale_ratio']:.0%}，最大 {sf['max_age']:.2f}s）"
                           f"—— 图像与关节状态配对不可靠")

    return {"ok": not reasons, "reasons": reasons, "metrics": metrics}


def stale_frame_stats(frames: Sequence[dict]) -> Dict[str, float]:
    """陈旧图像帧统计（采集端写入 stale_<cam> 时）→ {total, ratio, max_age}

    陈旧帧 = 相机最新帧距该关节帧超过新鲜度上限（默认 0.2s），此时图像与关节状态
    配对不可靠，采集端不落盘并标记；本函数用于事后统计与门控。
    """
    total = sum(1 for fr in frames if any(k.startswith("stale_") for k in fr))
    ages = [float(v) for fr in frames for k, v in fr.items()
            if k.startswith("stale_")]
    return {"total": float(total),
            "ratio": (total / len(frames)) if frames else 0.0,
            "max_age": max(ages) if ages else 0.0}


def check_images(image_counts: Dict[str, int], frames: int,
                 max_missing_ratio: float = 0.005) -> Dict[str, object]:
    """相机帧完整性检查: 各相机已写盘帧数 vs 关节帧数

    Args:
        image_counts: {cam_name: written_jpeg_count}
        frames: 关节帧数
    """
    reasons = []
    detail = {}
    for cam, n in (image_counts or {}).items():
        missing = max(0, frames - int(n))
        ratio = missing / frames if frames else 0.0
        detail[cam] = {"written": int(n), "missing": missing,
                       "missing_ratio": round(ratio, 4)}
        if ratio > max_missing_ratio:
            reasons.append(f"相机 {cam} 缺帧 {missing}/{frames}（{ratio:.1%} > "
                           f"{max_missing_ratio:.1%}）")
    return {"ok": not reasons, "reasons": reasons, "detail": detail}


def dataset_stats(episodes: Sequence[dict]) -> Dict[str, object]:
    """数据集汇总统计

    Args:
        episodes: [{"file": str, "frames": [...] or None, "quality": check_episode 结果}]
    """
    ok_eps = [e for e in episodes if e.get("quality", {}).get("ok")]
    all_frames: List[dict] = []
    for e in episodes:
        fr = e.get("frames")
        if isinstance(fr, (list, tuple)):      # 兼容：frames 可能是帧列表，也可能是帧数(int)
            all_frames.extend(fr)
    # 帧列表不可用时，退回累加各集的 quality.metrics.frames
    counted = sum(int((e.get("quality") or {}).get("metrics", {}).get("frames", 0))
                  for e in episodes)

    per_joint = {}
    valid = [fr for fr in all_frames if frame_keys_ok(fr)]
    for i in range(1, 7):
        vals = [float(fr[f"J{i}"]) for fr in valid]
        if not vals:
            continue
        per_joint[JOINT_NAMES[i - 1]] = {
            "min": round(min(vals), 1),
            "max": round(max(vals), 1),
            "span": round(max(vals) - min(vals), 1),
            "mean": round(sum(vals) / len(vals), 1),
        }

    fps_list = [e["quality"]["metrics"]["fps"] for e in episodes
                if e.get("quality", {}).get("metrics")]
    dur_list = [e["quality"]["metrics"]["duration_s"] for e in episodes
                if e.get("quality", {}).get("metrics")]
    tr = tracking_summary(valid)

    return {
        "episodes": len(episodes),
        "episodes_ok": len(ok_eps),
        "episodes_fail": len(episodes) - len(ok_eps),
        "total_frames": len(valid) if valid else counted,
        "total_duration_s": round(sum(dur_list), 2),
        "fps_min": round(min(fps_list), 2) if fps_list else None,
        "fps_mean": round(sum(fps_list) / len(fps_list), 2) if fps_list else None,
        "duration_min_s": round(min(dur_list), 2) if dur_list else None,
        "duration_max_s": round(max(dur_list), 2) if dur_list else None,
        "per_joint_deg": per_joint,
        "track_per_joint_mean": {JOINT_NAMES[i - 1]: (round(v, 2) if v is not None else None)
                                 for i, v in tr["per_joint_mean"].items()},
        "track_worst_joint": tr["worst_joint"],
        "track_worst_mean": (round(tr["worst_mean"], 2)
                             if tr["worst_mean"] is not None else None),
    }


def format_report(stats: Dict[str, object]) -> str:
    """人类可读的统计报告（Markdown 片段）"""
    lines = [
        f"episode 数: {stats['episodes']}（合格 {stats['episodes_ok']} / "
        f"不合格 {stats['episodes_fail']}）",
        f"总帧数: {stats['total_frames']}    总时长: {stats['total_duration_s']}s",
        f"帧率: min {stats['fps_min']} / mean {stats['fps_mean']}",
        f"单条时长: {stats['duration_min_s']}s ~ {stats['duration_max_s']}s",
        "",
        "| 关节 | min(°) | max(°) | 幅度(°) | mean(°) | 追踪平均误差(°) |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in stats["per_joint_deg"].items():
        terr = stats["track_per_joint_mean"].get(name)
        lines.append(f"| {name} | {s['min']} | {s['max']} | {s['span']} | "
                     f"{s['mean']} | {terr if terr is not None else '-'} |")
    if stats["track_worst_mean"] is not None:
        lines.append("")
        lines.append(f"追踪最差关节: J{stats['track_worst_joint']} "
                     f"平均 {stats['track_worst_mean']}°")
    return "\n".join(lines)
