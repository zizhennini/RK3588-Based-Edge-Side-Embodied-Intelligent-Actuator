#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/calibrate_arm.py — SO-ARM101 交互式关节标定向导（阶段 B / G5+G6）

流程（参考 lerobot 标定三部曲，独立实现，中文交互）:
  1. 握手校验连接（fail-fast，G1）
  2. 全程禁扭矩，可自由手搬关节
  3. 中位归零: 将臂摆到各关节行程中点 → 记录 homing（半圈中点）
  4. 行程录制: 逐关节搬动全行程 → 30Hz 轮询记录 min/max raw
     （wrist_roll 固定 0-4095 全圈；夹爪开合数次录行程）
  5. 生成 config/calibration.json —— 与 SO101Arm._load_calibration 完全兼容
     （homing_offset = 中位 raw 值，换算公式 (raw-mid)*360/4095 不变）
  6. 可选 --write-eeprom: 把 Homing_Offset/Min/Max_Position_Limit 写入舵机
     EEPROM（标定跟随舵机本体，sign-magnitude bit11 编码），写入后 JSON
     自动切换到舵机偏移后的新坐标系
  7. 可选 --verify: 读回 EEPROM 与期望值比对（容差 ±2）

用法:
    # 板端（follower 臂）
    python tools/calibrate_arm.py --port /dev/ttyACM0
    # leader 臂（遥操作主臂）
    python tools/calibrate_arm.py --port /dev/ttyACM1 --output config/calibration_leader.json
    # 标定写入舵机 EEPROM + 校验
    python tools/calibrate_arm.py --port /dev/ttyACM0 --write-eeprom --verify

安全: 向导全程扭矩禁用（手搬无阻力）；退出（含 Ctrl-C）自动恢复。
依赖: 仅 scservo_sdk + pyserial（板端已装）；PC 端带 USB 直通亦可运行。
"""
import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

# 允许从仓库根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hardware.feetech_bus import (  # noqa: E402
    FeetechBus, RESOLUTION, encode_sign_magnitude,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("calibrate_arm")

HALF_TURN = RESOLUTION // 2 - 1  # 2047: lerobot 同款半圈中点
POLL_HZ = 30
JOINT_LABELS = {
    1: "shoulder_pan (底座旋转)",
    2: "shoulder_lift (大臂俯仰)",
    3: "elbow_flex (肘部)",
    4: "wrist_flex (腕俯仰)",
    5: "wrist_roll (腕旋转, 全圈)",
    6: "gripper (夹爪)",
}


def wait_enter(prompt: str) -> None:
    print(f"\n>>> {prompt}")
    input("    完成后按 Enter 继续...")


def record_ranges(bus: FeetechBus, motor_ids) -> dict:
    """手搬关节全行程，30Hz 轮询记录 min/max raw（Enter 结束）"""
    mins = {mid: RESOLUTION - 1 for mid in motor_ids}
    maxs = {mid: 0 for mid in motor_ids}
    print("\n>>> 请用手缓慢搬动每个关节走完全部行程（夹爪开合数次）。")
    print("    录制中... 完成后按 Enter 结束（Ctrl-C 同样安全退出）")
    stop = {"flag": False}

    def _on_sigint(signum, frame):
        stop["flag"] = True
    old_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        import threading
        t_end = threading.Event()

        def _reader():
            input()
            t_end.set()
        th = threading.Thread(target=_reader, daemon=True)
        th.start()
        n_frames = 0
        while not t_end.is_set() and not stop["flag"]:
            raw_map = bus.sync_read("Present_Position", motor_ids=motor_ids,
                                    num_retry=0)
            for mid, raw in raw_map.items():
                mins[mid] = min(mins[mid], raw)
                maxs[mid] = max(maxs[mid], raw)
            n_frames += 1
            time.sleep(1.0 / POLL_HZ)
        print(f"    录制结束: {n_frames} 帧")
    finally:
        signal.signal(signal.SIGINT, old_handler)
    return {"min": mins, "max": maxs}


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-ARM101 关节标定向导")
    parser.add_argument("--port", default="/dev/ttyACM0", help="串口（follower 默认 ttyACM0）")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--output", default="config/calibration.json",
                        help="标定 JSON 输出路径（与 SO101Arm 兼容格式）")
    parser.add_argument("--write-eeprom", action="store_true",
                        help="把标定写入舵机 EEPROM（Homing_Offset/Min/Max_Position_Limit）")
    parser.add_argument("--verify", action="store_true",
                        help="写入后读回 EEPROM 校验（容差 ±2）")
    parser.add_argument("--no-handshake", action="store_true",
                        help="跳过握手校验（调试用）")
    args = parser.parse_args()

    print("=" * 60)
    print("SO-ARM101 关节标定向导 (Feetech STS3215 ×6)")
    print("=" * 60)
    bus = FeetechBus(args.port, baud=args.baud, name="calib")
    bus.connect(handshake=not args.no_handshake)

    exit_code = 0
    try:
        with bus.torque_disabled():
            # ---- 步骤 1: 中位归零 ----
            wait_enter("请用手将机械臂摆到【中位姿态】：每个关节位于其行程的中点附近"
                       "（参考: 大臂竖直、小臂水平、夹爪半开）")
            mid_raw = bus.sync_read("Present_Position", num_retry=2)
            if len(mid_raw) != len(bus.motor_ids):
                missing = set(bus.motor_ids) - set(mid_raw)
                raise ConnectionError(f"中位读取失败，无响应 ID: {sorted(missing)}")
            print("    中位 raw:", {mid: v for mid, v in sorted(mid_raw.items())})

            # ---- 步骤 2: 行程录制 ----
            ranges = record_ranges(bus, bus.motor_ids)

            # ---- 步骤 3: 组装标定（JSON 用 arm.py 语义: homing_offset=中位 raw）----
            calibration = {}
            for mid in bus.motor_ids:
                lo, hi = ranges["min"][mid], ranges["max"][mid]
                if mid == 5:
                    # wrist_roll 全圈: 固定 0-4095（lerobot 同款处理）
                    lo, hi = 0, RESOLUTION - 1
                if hi - lo < 100:
                    logger.warning("id=%d 行程过窄 (%d-%d)，疑似未搬动该关节", mid, lo, hi)
                calibration[str(mid)] = {
                    "homing_offset": int(mid_raw[mid]),
                    "range_min": int(lo),
                    "range_max": int(hi),
                }
                print(f"    {JOINT_LABELS[mid]}: mid={mid_raw[mid]} "
                      f"range=[{lo}, {hi}] span={hi - lo}")

            # ---- 步骤 4: 可选写 EEPROM ----
            if args.write_eeprom:
                print("\n>>> 写入舵机 EEPROM（Homing_Offset/Min/Max_Position_Limit）...")
                eeprom_calib = {}
                for mid in bus.motor_ids:
                    c = calibration[str(mid)]
                    # 半圈归零: 让当前位置(中位)读数为 2047
                    offset = c["homing_offset"] - HALF_TURN
                    shift = offset  # 写入后读数 raw' = raw - offset
                    eeprom_calib[mid] = {
                        "homing_offset": offset,  # write_calibration 内做 sign 编码
                        "range_min": max(0, c["range_min"] - shift),
                        "range_max": min(RESOLUTION - 1, c["range_max"] - shift),
                    }
                bus.write_calibration(eeprom_calib)
                # JSON 切换到偏移后的新坐标系（读数中位=2047）
                for mid in bus.motor_ids:
                    calibration[str(mid)] = {
                        "homing_offset": HALF_TURN + 1,  # 2048, 与 arm.py 默认语义一致
                        "range_min": eeprom_calib[mid]["range_min"],
                        "range_max": eeprom_calib[mid]["range_max"],
                    }
                print("    EEPROM 写入完成，JSON 已切换到舵机偏移后坐标系")

                if args.verify:
                    print(">>> 读回 EEPROM 校验（容差 ±2）...")
                    time.sleep(0.3)
                    readback = bus.read_calibration()
                    ok = True
                    for mid in bus.motor_ids:
                        exp, got = eeprom_calib[mid], readback[mid]
                        # Homing_Offset 期望值按 sign-magnitude 编码后比对
                        exp_off = encode_sign_magnitude(exp["homing_offset"], 11) \
                            if exp["homing_offset"] < 0 else exp["homing_offset"]
                        for key, e_v in (("homing_offset", exp_off),
                                         ("range_min", exp["range_min"]),
                                         ("range_max", exp["range_max"])):
                            if abs(got[key] - e_v) > 2:
                                print(f"    MISMATCH id={mid} {key}: "
                                      f"期望 {e_v} 读回 {got[key]}")
                                ok = False
                    print("    校验通过 ✓" if ok else "    校验存在不一致 ✗")
                    if not ok:
                        exit_code = 2

            # ---- 步骤 5: 保存 JSON ----
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(calibration, f, indent=2, ensure_ascii=False)
            print(f"\n✓ 标定已保存: {out.resolve()}")
            print("  SO101Arm 下次连接将自动加载（calibration_path 指向该文件）")
    except KeyboardInterrupt:
        print("\n已取消（扭矩自动恢复）")
        exit_code = 130
    finally:
        bus.disconnect(disable_torque=False)  # torque_disabled 上下文已恢复扭矩
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
