#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/calibrate_arm.py — SO-ARM101 交互式关节标定向导（阶段 B / G5+G6）

流程（与 lerobot 标定三部曲逐项对齐，独立实现，中文交互）:
  1. 握手校验连接（fail-fast，G1）
  2. 全程禁扭矩，可自由手搬关节
  3. 中位归零: 各关节摆到行程【正中间】→ 读 mid_raw → **写入 Homing_Offset
     使中位读数=2047**（lerobot set_half_turn_homings 同款，强制前置）
     —— 这一步是行程录制不跨越 0/4095 边界的前提
  4. 行程录制: 逐关节搬动全行程 → 30Hz 轮询记录 min/max raw + 跨边界检测
     （检测到绕回则中止，不保存废数据；wrist_roll 固定 0-4095 全圈）
     **行程决定角度零点**（官方 DEGREES 语义: mid=(range_min+range_max)/2）
  5. 写 Min/Max_Position_Limit（与行程同坐标系）+ 可选 --verify 读回校验（±2）
  6. 保存 config/calibration.json（homing_offset=2048，range 为归零后坐标系）
  7. 可选 --reset-eeprom: 单独复位模式（清错误偏移、限位恢复 0/4095）后退出

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
    """手搬关节全行程，30Hz 轮询记录 min/max raw（Enter 结束）

    返回 {"min": {...}, "max": {...}, "wrap": {mid: bool}}。
    wrap 为**跨 0/4095 边界检测**：相邻采样从 >3300 跳到 <800（或反向）即判定读数
    绕回——此时 min/max 会录成接近满量程的假行程（真机实测 span 4083/4095 = 废数据），
    必须让调用方拒绝保存。前提是已写入中位归零偏移（读数围绕 2047），见 main()。
    """
    mins = {mid: RESOLUTION - 1 for mid in motor_ids}
    maxs = {mid: 0 for mid in motor_ids}
    wrap = {mid: False for mid in motor_ids}
    prev = {}
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
                p = prev.get(mid)
                if p is not None and ((p > 3300 and raw < 800)
                                      or (p < 800 and raw > 3300)):
                    wrap[mid] = True
                prev[mid] = raw
                mins[mid] = min(mins[mid], raw)
                maxs[mid] = max(maxs[mid], raw)
            n_frames += 1
            time.sleep(1.0 / POLL_HZ)
        print(f"    录制结束: {n_frames} 帧")
    finally:
        signal.signal(signal.SIGINT, old_handler)
    bad = [m for m, w in wrap.items() if w]
    if bad:
        print(f"    ⚠ 检测到读数跨越 0/4095 边界: {bad}")
    return {"min": mins, "max": maxs, "wrap": wrap}


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-ARM101 关节标定向导")
    parser.add_argument("--port", default="/dev/ttyACM0", help="串口（follower 默认 ttyACM0）")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--output", default="config/calibration.json",
                        help="标定 JSON 输出路径（与 SO101Arm 兼容格式）")
    parser.add_argument("--write-eeprom", action="store_true",
                        help="（已内置，保留仅为命令兼容）EEPROM 写入现在始终执行："
                             "中位归零偏移在行程录制【之前】写，否则行程会跨越 0/4095 边界")
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
    parser.add_argument("--reset-eeprom", action="store_true",
                        help="单独模式: 把 Homing_Offset 归零、Min/Max_Position_Limit 恢复 "
                             "0/4095（读数恢复为物理原始值）后退出。用于清除错误偏移——"
                             "偏移写歪时读数会被压到 0/4095 边界，行程录制必然失真")
    args = parser.parse_args()

    print("=" * 60)
    print("SO-ARM101 关节标定向导 (Feetech STS3215 ×6)")
    print("=" * 60)
    bus = FeetechBus(args.port, baud=args.baud, name="calib")
    bus.connect(handshake=not args.no_handshake)

    exit_code = 0
    try:
        # ---- 复位模式: 清错误 EEPROM 偏移后退出（不进入向导）----
        if args.reset_eeprom:
            print("\n>>> 复位 EEPROM: Homing_Offset=0, Min=0, Max=%d" % (RESOLUTION - 1))
            bus.disable_torque()   # 保持禁扭矩，便于随后手搬标定
            for mid in bus.motor_ids:
                old = bus.read("Homing_Offset", mid, num_retry=1)
                bus.write("Homing_Offset", mid, 0, num_retry=1)
                bus.write("Min_Position_Limit", mid, 0, num_retry=1)
                bus.write("Max_Position_Limit", mid, RESOLUTION - 1, num_retry=1)
                print(f"    id{mid}: Homing_Offset {old} → 0")
            time.sleep(0.3)
            rb = bus.read_calibration()
            print("    读回:", {k: v for k, v in sorted(rb.items())})
            ok = all(v["homing_offset"] == 0 and v["range_min"] == 0
                     and v["range_max"] == RESOLUTION - 1 for v in rb.values())
            print("    复位成功 ✓（接着跑一次常规标定）" if ok
                  else "    复位存在不一致 ✗（可重复执行一次）")
            return 0 if ok else 2

        with bus.torque_disabled(enable_on_exit=False):
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
                wait_enter("请用手将机械臂摆到【中位姿态】：每个关节都摆到其机械行程的"
                           "【正中间】（官方语义 middle of range of motion）。\n"
                           "    ⚠ 这一步很关键: 随后会写入中位归零偏移，读数围绕 2047，"
                           "关节行程才不会跨越 0/4095 边界。\n"
                           "      若摆到极限位置附近，行程会绕回、标定必然失真"
                           "（向导检测到会中止并要求重来）。\n"
                           "    推荐参考姿态: 底座朝前、大臂竖直向上、小臂水平向前、"
                           "腕水平、夹爪半开朝前。")
                mid_raw = bus.sync_read("Present_Position", num_retry=2)
                if len(mid_raw) != len(bus.motor_ids):
                    missing = set(bus.motor_ids) - set(mid_raw)
                    raise ConnectionError(f"中位读取失败，无响应 ID: {sorted(missing)}")
            print("    中位 raw:", {mid: v for mid, v in sorted(mid_raw.items())})

            # ---- 步骤 2: 中位归零写入 EEPROM（lerobot set_half_turn_homings 同款，
            #      强制前置）------------------------------------------------------
            # 为什么必须写: 读数随后围绕 2047，关节行程才不会跨越 0/4095 边界。
            # 不写时行程会绕回，min/max 录成接近满量程的假行程（真机实测 span 4083，
            # 真值仅 ~3100），标定必然失真——这是本向导此前最大的缺陷。
            print("\n>>> 写入中位归零偏移（Homing_Offset，使中位读数 = 2047）...")
            eeprom_calib = {}
            for mid in bus.motor_ids:
                old_offset = int(bus.read("Homing_Offset", mid, num_retry=1))
                new_offset = eeprom_offset_update(old_offset, mid_raw[mid])
                bus.write("Homing_Offset", mid, new_offset, num_retry=1)
                eeprom_calib[mid] = {"homing_offset": new_offset}
                if old_offset:
                    print(f"    id{mid}: 现有偏移 {old_offset} → 累加后 {new_offset}")
            time.sleep(0.2)
            mid_now = bus.sync_read("Present_Position", num_retry=2)
            print("    写入后中位读数:", dict(sorted(mid_now.items())))
            off = {m: v for m, v in mid_now.items() if abs(v - HALF_TURN) > 5}
            if off:
                print(f"    ⚠ 与 {HALF_TURN} 偏差 >5 步: {off}（写入可能未生效）")

            # ---- 步骤 3: 行程录制（含跨边界检测）----
            ranges = record_ranges(bus, bus.motor_ids)
            wrap_bad = [m for m in (1, 2, 3, 4, 6) if ranges["wrap"].get(m)]
            if wrap_bad:
                print("\n✗ 标定中止（未保存、未写限位）——行程跨越 0/4095 边界: "
                      f"{wrap_bad}")
                print("  原因: 步骤 1 的中位姿态不在关节行程【中间】，"
                      "导致搬动行程时读数绕回。")
                print("  处理: 把每个关节都摆到行程中间再重跑本向导"
                      "（不要摆到极限位置）。")
                return 3

            # ---- 步骤 4: 组装标定（读数已在中位归零坐标系，围绕 2047）----
            calibration = {}
            for mid in bus.motor_ids:
                lo, hi = ranges["min"][mid], ranges["max"][mid]
                if mid == 5:
                    # wrist_roll 全圈: 固定 0-4095（lerobot 同款处理）
                    lo, hi = 0, RESOLUTION - 1
                if hi - lo < 100:
                    logger.warning("id=%d 行程过窄 (%d-%d)，疑似未搬动该关节", mid, lo, hi)
                calibration[str(mid)] = {
                    "homing_offset": HALF_TURN + 1,  # 2048（写入后中位=2047）
                    "range_min": int(lo),
                    "range_max": int(hi),
                }
                print(f"    {JOINT_LABELS[mid]}: 中位读数={mid_now.get(mid)} "
                      f"range=[{lo}, {hi}] span={hi - lo} "
                      f"角度零点(行程中点)={(lo + hi) / 2:.1f}")

            # 中位姿态质量自检: 归零点(2047)越接近行程中点，说明步骤 1 越接近
            # 机械行程正中。偏离过大时，两臂零点会随各自摆放姿态错开（恒定偏差根因），
            # 且 lerobot 官方也要求"middle of range of motion"。
            pose_bad = {}
            for mid in (1, 2, 3, 4, 6):
                lo, hi = calibration[str(mid)]["range_min"], calibration[str(mid)]["range_max"]
                off = (lo + hi) / 2 - HALF_TURN
                if abs(off) > 150:      # >150 步 ≈ 13°
                    pose_bad[mid] = round(off, 1)
            if pose_bad:
                print("\n⚠ 中位姿态偏离行程中点较大（{id: 偏差步数}）: "
                      f"{pose_bad}")
                print("  含义: 步骤 1 摆的位置不是关节行程正中间。"
                      "同一只臂自身可用，但**两臂零点会错开**，")
                print("        跟随会出现恒定偏差。建议重跑并把每个关节摆到行程正中"
                      "（两臂用同一个标准姿态）。")

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

            # ---- 步骤 5: 写限位（与行程同坐标系）+ 读回校验 ----
            print("\n>>> 写入 Min/Max_Position_Limit...")
            for mid in bus.motor_ids:
                eeprom_calib[mid]["range_min"] = calibration[str(mid)]["range_min"]
                eeprom_calib[mid]["range_max"] = calibration[str(mid)]["range_max"]
            bus.write_calibration(eeprom_calib)
            if args.verify:
                print(">>> 读回 EEPROM 校验（容差 ±2）...")
                time.sleep(0.3)
                readback = bus.read_calibration()
                ok = True
                for mid in bus.motor_ids:
                    exp, got = eeprom_calib[mid], readback[mid]
                    # read() 已把 Homing_Offset 按 bit11 解码为带符号值，期望值同样是
                    # 带符号 offset —— 两边统一在带符号域直接比对。
                    for key in ("homing_offset", "range_min", "range_max"):
                        if abs(got[key] - exp[key]) > 2:
                            print(f"    MISMATCH id={mid} {key}: "
                                  f"期望 {exp[key]} 读回 {got[key]}")
                            ok = False
                print("    校验通过 ✓" if ok else "    校验存在不一致 ✗")
                if not ok:
                    exit_code = 2

            # ---- 步骤 6: 保存 JSON ----
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
