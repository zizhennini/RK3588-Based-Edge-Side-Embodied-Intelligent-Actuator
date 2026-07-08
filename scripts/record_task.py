"""多段工序录制 — 按 Enter 标记工序节点"""
import sys, os, json, time, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import SERIAL_PORT, SERIAL_BAUD
from scservo_sdk import *

calib = [(946,3287),(821,3206),(888,3105),(851,3192),(0,4095),(1495,2860)]
JOINTS = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]


def read_joints(ph, pkt) -> list:
    grp = GroupSyncRead(ph, pkt, 0x38, 2)
    for sid in range(1, 7):
        grp.addParam(sid)
    result = grp.txRxPacket()
    if result != COMM_SUCCESS:
        print("  !! SyncRead 失败: {}".format(result))
        return [0.0] * 6
    angles = []
    for sid in range(1, 7):
        if grp.isAvailable(sid, 0x38, 2):
            pos = grp.getData(sid, 0x38, 2) & 0x0FFF
            lo, hi = calib[sid - 1]
            mid = (lo + hi) / 2
            angles.append(round((pos - mid) * 360 / 4095, 1))
        else:
            print("  !! 关节 {} 读取超时".format(sid))
            angles.append(0.0)
    return angles


def main():
    import argparse
    parser = argparse.ArgumentParser(description="多段工序录制")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or "task_{}.json".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    ph = PortHandler(args.port)
    if not ph.openPort():
        print("!! 无法打开串口: {}".format(args.port))
        sys.exit(1)
    ph.setBaudRate(SERIAL_BAUD)
    pkt = PacketHandler(ph)

    segments = []
    while True:
        name = input("  节点名: ").strip()
        if name == "q":
            break
        if name == "s":
            data = {"segments": segments, "total": len(segments), "created": str(datetime.datetime.now())}
            with open(out, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("  已保存: {}".format(out))
            break
        if not name:
            continue

        angles = read_joints(ph, pkt)
        joint_map = dict(zip(JOINTS, angles))
        segments.append({"name": name, "joints": joint_map})
        print("  记录: {}  {}".format(name, joint_map))


if __name__ == "__main__":
    main()
