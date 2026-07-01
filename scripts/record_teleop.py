"""遥操作时录制主臂关节角度（不干扰从臂）"""
import sys, os, json, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scservo_sdk import *
from config.settings import SERIAL_BAUD

LEADER_PORT = "/dev/ttyACM1"
calib = [(946,3287),(821,3206),(888,3105),(851,3192),(0,4095),(1495,2860)]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out_path = args.out or f"teleop_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print("  启动遥操作后，此脚本将录制主臂关节角度")
    print(f"  请运行：lerobot-teleoperate ...")
    print(f"  录制 {args.seconds} 秒...")

    port_handler = PortHandler(LEADER_PORT)
    port_handler.openPort()
    port_handler.setBaudRate(SERIAL_BAUD)
    packet_handler = PacketHandler(port_handler)
    group = GroupSyncRead(port_handler, packet_handler, 0x38, 2)
    for sid in range(1, 7):
        group.addParam(sid)

    frames = []
    t0 = time.perf_counter()

    while time.perf_counter() - t0 < args.seconds:
        loop_t = time.perf_counter()
        group.txRxPacket()
        frame = {"t": round(time.perf_counter() - t0, 3)}
        for sid in range(1, 7):
            if group.isAvailable(sid, 0x38, 2):
                pos = group.getData(sid, 0x38, 2)
                lo, hi = calib[sid - 1]
                mid = (lo + hi) / 2
                frame[f"J{sid}"] = round((pos - mid) * 360 / 4095, 1)
        frames.append(frame)

        dt = time.perf_counter() - loop_t
        time.sleep(max(1 / args.fps - dt, 0))

    port_handler.closePort()

    data = {"fps": args.fps, "total_frames": len(frames), "duration_s": args.seconds, "frames": frames}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  ✅ {len(frames)}帧, 已保存 {out_path}")


if __name__ == "__main__":
    main()
