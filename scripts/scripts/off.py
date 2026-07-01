"""关闭所有关节使能（手动掰动前运行）"""
import sys, os, struct, serial, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import SERIAL_PORT, SERIAL_BAUD

ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.3)
for sid in range(1, 7):
    cmd = struct.pack("<BBBBBBB", 0xFF, 0xFF, sid, 4, 0x03, 0x28, 0)
    cks = (~sum(cmd[2:]) & 0xFF)
    ser.write(cmd + struct.pack("<B", cks))
    time.sleep(0.05)
    print(f"关节{sid} 使能关闭")
ser.close()
print("所有关节已关闭，可以手动掰动")
