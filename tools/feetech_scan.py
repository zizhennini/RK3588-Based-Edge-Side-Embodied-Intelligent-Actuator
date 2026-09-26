#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/feetech_scan.py — Feetech 总线扫描与出厂初始化（阶段 D / G10）

用法:
    # 当前波特率扫描在线设备（broadcast ping）
    python tools/feetech_scan.py --port /dev/ttyACM0

    # 全波特率扫描（9600..1M，找被改过波特率的舵机）
    python tools/feetech_scan.py --port /dev/ttyACM0 --scan-all

    # 单电机出厂初始化（改 ID + 波特率；须只连接该电机）
    python tools/feetech_scan.py --port /dev/ttyACM0 --set-id \
        --initial-id 1 --target-id 7 --target-baud 1000000

依赖: scservo_sdk + pyserial（板端已装）。
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hardware.feetech_bus import (  # noqa: E402
    FeetechBus, MODEL_NUMBER_STS3215, SCAN_BAUDRATES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("feetech_scan")

MODEL_NAMES = {777: "STS3215", 2825: "STS3250", 11272: "SM8512BL", 1284: "SCS0009"}


def do_scan(bus: FeetechBus, scan_all: bool) -> int:
    if not bus.port_handler.openPort():
        print(f"✗ 无法打开串口 {bus.port}")
        return 1
    try:
        if scan_all:
            print(f"全波特率扫描: {SCAN_BAUDRATES}")
            result = bus.scan_baudrates()
            if not result:
                print("✗ 未发现任何设备（检查接线/供电/USB 转串口）")
                return 1
            for baud, models in sorted(result.items()):
                for mid, model in sorted(models.items()):
                    name = MODEL_NAMES.get(model, f"unknown({model})")
                    print(f"  @ {baud:>9}  ID={mid:<3} model={model} ({name})")
            return 0
        else:
            found = bus.broadcast_ping()
            if not found:
                print(f"✗ @{bus.baud} 未发现设备（可试 --scan-all）")
                return 1
            print(f"发现 {len(found)} 个设备 @ {bus.baud}:")
            for mid, err in sorted(found.items()):
                model = bus.ping(mid, num_retry=1)
                name = MODEL_NAMES.get(model, f"unknown({model})") \
                    if model is not None else "?"
                mark = "✓" if model == MODEL_NUMBER_STS3215 else "⚠ 型号不符"
                print(f"  ID={mid:<3} model={model} ({name}) {mark} "
                      f"error_status={err}")
            return 0
    finally:
        bus.port_handler.closePort()


def do_set_id(bus: FeetechBus, initial_id: int, target_id: int,
              target_baud: int) -> int:
    print(f"出厂初始化: ID {initial_id} → {target_id}, 波特率 → {target_baud}")
    print("⚠ 请确认总线上只连接了这一个电机！")
    input("确认后按 Enter（Ctrl-C 取消）...")
    bus.motor_ids = (initial_id,)
    bus.connect(handshake=False)
    try:
        ok = bus.setup_motor(initial_id, target_id, target_baud)
        if not ok:
            print("✗ 初始化写入失败")
            return 1
        print("✓ 写入完成，复扫验证...")
        bus.disconnect(disable_torque=False)
        verify = FeetechBus(bus.port, motor_ids=(target_id,),
                            baud=target_baud, name="verify")
        model = None
        try:
            verify.connect(handshake=False)
            model = verify.ping(target_id, num_retry=2)
        finally:
            verify.disconnect(disable_torque=False)
        if model is not None:
            print(f"✓ 验证通过: ID={target_id} model={model} "
                  f"({MODEL_NAMES.get(model, '?')}) @ {target_baud}")
            return 0
        print(f"✗ 验证失败: ID={target_id} 无响应")
        return 1
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


def do_identify(ports: list, baud: int) -> int:
    """识别每个串口连的是哪只臂（防 USB 重插后主从端口互换）

    判据: Homing_Offset —— follower 标定时写过 EEPROM 偏移（非 0），leader 通常为 0。
    同时打印当前读数与 Phase 方向位，便于确认两侧配置一致。
    """
    found = 0
    for port in ports:
        bus = FeetechBus(port, baud=baud, name="ident")
        try:
            bus.connect(handshake=False)
        except Exception as e:
            print(f"{port}: 打开失败 ({e})")
            continue
        try:
            offs, phases = {}, {}
            for mid in bus.motor_ids:
                offs[mid] = int(bus.read("Homing_Offset", mid, num_retry=1))
                phases[mid] = bus.read("Phase", mid, num_retry=1)
            pos = bus.sync_read("Present_Position", num_retry=2)
            guess = ("follower（已写 EEPROM 偏移）" if any(offs.values())
                     else "leader（偏移全 0，未写过 EEPROM）")
            print(f"\n{port}: {len(pos)}/6 舵机响应 → 疑似 {guess}")
            print(f"  Homing_Offset: {offs}")
            print("  Phase: " + " ".join(f"id{k}=0x{v:02X}"
                                       for k, v in sorted(phases.items())))
            print(f"  Present_Position: {dict(sorted(pos.items()))}")
            high = [k for k, v in phases.items() if v & 0x40]
            if high:
                print(f"  ⚠ Phase bit6 置位（计数方向可能反向）: {high} —— 建议重跑 configure")
            found += 1
        finally:
            bus.disconnect(disable_torque=False)
    if found == 0:
        print("未识别到任何臂（检查 USB 连接/供电）")
        return 1
    return 0


def do_release(ports: list, baud: int) -> int:
    """释放扭矩（松臂巡视模式）: 对每个端口禁扭矩+解锁，进程退出后舵机不再锁死

    用途: 程序异常退出后舵机仍保持 Torque_Enable=1（串口关闭不会自动断扭矩），
    此模式把两侧（或指定端口）全部舵机置为自由状态。
    """
    done = 0
    for port in ports:
        bus = FeetechBus(port, baud=baud, name="release")
        try:
            bus.connect(handshake=False)
        except Exception as e:
            print(f"{port}: 打开失败 ({e})")
            continue
        try:
            bus.disable_torque()
            tq = {mid: bus.read("Torque_Enable", mid, num_retry=1)
                  for mid in bus.motor_ids}
            free = [m for m, v in tq.items() if v == 0]
            print(f"{port}: 扭矩已释放 {len(free)}/6 {tq}")
            done += 1
        finally:
            bus.disconnect(disable_torque=False)
    if done == 0:
        print("未找到可用端口")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Feetech 总线扫描/出厂初始化/端口识别/松臂")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--scan-all", action="store_true", help="全波特率扫描")
    parser.add_argument("--set-id", action="store_true", help="出厂初始化模式")
    parser.add_argument("--initial-id", type=int, default=1)
    parser.add_argument("--target-id", type=int)
    parser.add_argument("--target-baud", type=int, default=1_000_000)
    parser.add_argument("--identify", action="store_true",
                        help="识别 /dev/ttyACM*/ttyUSB* 各端口对应哪只臂（主从判别）")
    parser.add_argument("--release-all", action="store_true",
                        help="对所有 /dev/ttyACM*/ttyUSB* 端口释放扭矩（松臂，防锁死）")
    args = parser.parse_args()

    if args.identify or args.release_all:
        import glob
        ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        if not ports:
            print("未发现任何 /dev/ttyACM* 或 /dev/ttyUSB* 设备")
            return 1
        if args.release_all:
            return do_release(ports, args.baud)
        return do_identify(ports, args.baud)

    bus = FeetechBus(args.port, baud=args.baud, name="scan")
    if args.set_id:
        if args.target_id is None:
            parser.error("--set-id 需要 --target-id")
        return do_set_id(bus, args.initial_id, args.target_id, args.target_baud)
    return do_scan(bus, args.scan_all)


if __name__ == "__main__":
    sys.exit(main())
