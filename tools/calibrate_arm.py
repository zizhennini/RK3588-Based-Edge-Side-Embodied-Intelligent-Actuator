#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/calibrate_arm.py — SO-ARM101 交互式关节标定向导（阶段 B / G5+G6）

流程（参考 lerobot 标定三部曲，独立实现，中文交互）:
  1. 握手校验连接（fail-fast，G1）
  2. 全程禁扭矩，可自由手搬关节
  3. 中位归零: 将臂摆到各关节行程中点 → 记录 homing（半圈中点）
  4. 行程录制: 逐关节搬动全行程 → 30Hz 轮询记录 min/max raw
     （wrist_roll 固定 0-4095 全圈；夹爪开合数次录行程）
     **行程决定角度零点**（官方 DEGREES 语义: mid=(range_min+range_max)/2），
     所以必须推到机械行程两端；homing 仅用于 EEPROM 归零，不参与角度换算
  5. 生成 config/calibration.json —— 与 SO101Arm._load_calibration 完全兼容
     （换算公式 (raw-行程中点)*360/4095，夹爪仍用 homing_offset 为零点）
  6. 可选 --write-eeprom: 把 Homing_Offset/Min/Max_Position_Limit 写入舵机
     EEPROM（标定跟随舵机本体，sign-magnitude bit11 编码），写入后 JSON
     自动切换到舵机偏移后的新坐标系
  7. 可选 --verify: 读回 EEPROM 与期望值比对（容差 ±2）

用法:
    # 板端（follower 臂，第一只）
    python tools/calibrate_arm.py --port /dev/ttyACM0
    # leader 臂（第二只，零点从 follower 推导——两臂摆相同姿态即可对齐零点，
    # 官方语义"相同物理姿态 = 相同角度值"，消除主从恒定偏差）
    python tools/calibrate_arm.py --port /dev/ttyACM1 --output config/calibration_leader.json \
        --derive-from-port /dev/ttyACM0 --derive-from-calib config/calibration.json
    # 标定写入舵机 EEPROM + 校验（官方对 leader/follower 都写入）
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

from hardware.feetech_bus import FeetechBus, RESOLUTION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("calibrate_arm")

HALF_TURN = RESOLUTION // 2 - 1  # 2047: lerobot 同款半圈中点
POLL_HZ = 30


def eeprom_offset_update(old_offset: int, mid_raw: int,
                         half_turn: int = HALF_TURN) -> int:
    """计算要写入舵机的新 Homing_Offset（**累加**语义）

    舵机语义 ``Present_Position = 实际位置 - Homing_Offset``。要让当前中位姿态读数为
    ``half_turn``，绝对偏移必须是【舵机现有偏移 + 本次坐标平移量】::

        new = old + (mid_raw - half_turn)

    直接用 ``mid_raw - half_turn`` 覆盖只在 ``old == 0``（从未写过偏移）时正确；
    对已经写过偏移的臂会把中位读成 ``old + half_turn``，整个坐标系错位。
    """
    return int(old_offset) + (int(mid_raw) - int(half_turn))


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
    print("    ⚠ 本步骤决定角度零点（官方 DEGREES 语义: 零点 = 行程中点），")
    print("      请把每个关节都推到【机械行程两端到底】（左右/上下都推到位），")
    print("      主从两臂用同样的力度推到底，零点才会对齐（避免跟随恒定偏差）。")
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
    parser.add_argument("--derive-from-port", default=None,
                        help="参考臂串口（零点推导模式）：标第二只臂时用。两臂摆到完全相同的"
                             "物理姿态，homing 由参考臂标定推导（mid_本臂 = raw_本臂 − raw_参考 + "
                             "homing_参考），保证官方语义『相同物理姿态 = 相同角度值』，"
                             "消除自由摆放造成的主从零点错位（恒定跟随偏差）")
    parser.add_argument("--derive-from-calib", default="config/calibration.json",
                        help="参考臂标定 JSON（--derive-from-port 时读取其 homing_offset）")
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
            if args.derive_from_port:
                print("\n>>> 【零点推导模式】（官方语义: 相同物理姿态 = 相同角度值）")
                print("    1. 参考臂（%s）摆到其中位姿态并固定，之后不要再碰它" % args.derive_from_port)
                print("    2. 把【被标定臂】搬到与参考臂【完全相同的姿态】（目测重合、同形状）")
                print("    3. 两臂都不要再碰")
                wait_enter("两臂姿态已重合")
                ref_bus = FeetechBus(args.derive_from_port, baud=args.baud, name="ref")
                ref_bus.connect(handshake=False)
                try:
                    ref_raw = ref_bus.sync_read("Present_Position", num_retry=2)
                finally:
                    ref_bus.disconnect(disable_torque=False)
                if len(ref_raw) != len(bus.motor_ids):
                    raise ConnectionError(
                        f"参考臂读取不全: {sorted(ref_raw)}（检查 {args.derive_from_port}）")
                self_raw = bus.sync_read("Present_Position", num_retry=2)
                if len(self_raw) != len(bus.motor_ids):
                    raise ConnectionError(f"被标定臂读取不全: {sorted(self_raw)}")
                with open(args.derive_from_calib, "r", encoding="utf-8") as f:
                    ref_calib = json.load(f)
                mid_raw = {}
                print("    零点推导（mid = raw本臂 − raw参考 + homing参考）:")
                for m in bus.motor_ids:
                    ref_mid = int(ref_calib[str(m)]["homing_offset"])
                    derived = int(round(self_raw[m] - ref_raw[m] + ref_mid))
                    mid_raw[m] = derived
                    print("      id%d: raw本=%d raw参=%d homing参=%d → 推导 homing=%d"
                          % (m, self_raw[m], ref_raw[m], ref_mid, derived))
            else:
                wait_enter("请用手将机械臂摆到【中位姿态】：所有关节位于各自行程的中点"
                           "（官方语义 middle of range of motion）。\n"
                           "    推荐参考姿态: 底座朝前、大臂竖直向上、小臂水平向前、腕水平、夹爪半开朝前。\n"
                           "    ⚠ 标定第二只臂时必须与第一只臂使用【完全相同】的物理参考姿态，"
                           "否则主从零点错位会造成跟随恒定偏差（推荐用 --derive-from-port 自动对齐）")
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
                      f"range=[{lo}, {hi}] span={hi - lo} "
                      f"角度零点(行程中点)={(lo + hi) / 2:.1f}")

            # 夹爪(6) 零点保持既有约定（0 rad = 夹爪闭合位）：重标定时沿用旧 JSON 的
            # homing_offset，避免因中位姿态摆放差异导致夹爪角度语义漂移
            out_path_obj = Path(args.output)
            if out_path_obj.exists():
                try:
                    with open(out_path_obj, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    if "6" in old and "homing_offset" in old["6"]:
                        keep = int(old["6"]["homing_offset"])
                        if calibration["6"]["homing_offset"] != keep:
                            print(f"    夹爪零点沿用旧标定 homing_offset={keep}"
                                  f"（本次中位读数 {calibration['6']['homing_offset']} 仅存档）")
                        calibration["6"]["homing_offset"] = keep
                except Exception as e:
                    logger.warning("读取旧标定失败（夹爪零点沿用规则跳过）: %s", e)

            # ---- 步骤 4: 可选写 EEPROM ----
            if args.write_eeprom:
                print("\n>>> 写入舵机 EEPROM（Homing_Offset/Min/Max_Position_Limit）...")
                eeprom_calib = {}
                for mid in bus.motor_ids:
                    c = calibration[str(mid)]
                    # 舵机语义: Present_Position = 实际位置 - Homing_Offset。
                    # delta = 本次坐标平移量（写入后读数 = 写入前读数 - delta），
                    # 新偏移必须【累加】到舵机现有偏移上——follower 此前已写过偏移，
                    # 早先实现直接用 delta 覆盖，会把中位读成 O_old+2047（坐标系写坏）。
                    old_offset = int(bus.read("Homing_Offset", mid, num_retry=1))
                    delta = c["homing_offset"] - HALF_TURN
                    new_offset = eeprom_offset_update(old_offset, c["homing_offset"])
                    eeprom_calib[mid] = {
                        "homing_offset": new_offset,  # write_calibration 内做 sign 编码
                        "range_min": max(0, c["range_min"] - delta),
                        "range_max": min(RESOLUTION - 1, c["range_max"] - delta),
                    }
                    if old_offset:
                        print(f"    id{mid}: 现有偏移 {old_offset} → 累加后 {new_offset}")
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
                        # 注意: read() 已把 Homing_Offset 按 bit11 解码为带符号值，
                        # 期望值同样是带符号 offset —— 两边统一在带符号域直接比对。
                        # （旧版误将期望编码成无符号再比，负偏移会误报 MISMATCH）
                        for key in ("homing_offset", "range_min", "range_max"):
                            if abs(got[key] - exp[key]) > 2:
                                print(f"    MISMATCH id={mid} {key}: "
                                      f"期望 {exp[key]} 读回 {got[key]}")
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
