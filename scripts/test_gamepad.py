#!/usr/bin/env python3
"""北通蝙蝠4 (BD4A) 手柄测试"""
import sys, time
from evdev import InputDevice, list_devices, ecodes

BTN = {304:"A",305:"B",306:"C",307:"X",308:"Y",309:"Z",
       310:"LB",311:"RB",312:"LT",313:"RT",
       314:"Sel",315:"Start",316:"Home",398:"Func",
       317:"L3",318:"R3"}

AXIS_Y = {1:"上/下"}
AXIS_X = {0:"左/右",2:"右X",5:"右Y",9:"RT",10:"LT"}

dev = None
for p in list_devices():
    d = InputDevice(p)
    if "betop" in d.name.lower():
        dev = d
        break
if not dev:
    print("未检测到手柄"); sys.exit(1)

print(f"设备: {dev.path}  {dev.name}")
print("按任意键/Ctrl+C 退出\n")

keys = {c:0 for c in (304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,398)}
axes = {c:128 for c in (0,1,2,5,9,10)}

dev.grab()
try:
    for ev in dev.read_loop():
        if ev.type == 1 and ev.code in keys:
            keys[ev.code] = ev.value
        elif ev.type == 3 and ev.code in axes:
            axes[ev.code] = ev.value
        if ev.type != 0:
            continue

        lx, ly, rx, ry, rt, lt = (axes.get(i,128) for i in (0,1,2,5,9,10))
        print("\033[H\033[J", end="")
        print("=== 北通蝙蝠4 ===")
        print()
        print("[按键]")
        print(f"  {BTN[304]}:{keys[304]}  {BTN[305]}:{keys[305]}  {BTN[307]}:{keys[307]}  {BTN[308]}:{keys[308]}  {BTN[306]}:{keys[306]}  {BTN[309]}:{keys[309]}")
        print(f"  {BTN[310]}:{keys[310]}  {BTN[311]}:{keys[311]}  {BTN[312]}:{keys[312]}  {BTN[313]}:{keys[313]}")
        print(f"  {BTN[314]}:{keys[314]}  {BTN[315]}:{keys[315]}  {BTN[316]}:{keys[316]}  {BTN[398]}:{keys[398]}")
        print(f"  {BTN[317]}:{keys[317]}  {BTN[318]}:{keys[318]}")
        print()
        print("[十字键/左摇杆]")
        def d(v): return f"{v:3d}"
        print(f"  X:{d(lx)}  Y:{d(ly)}  上:{'1' if ly==0 else '0'} 下:{'1' if ly==255 else '0'} 左:{'1' if lx==0 else '0'} 右:{'1' if lx==255 else '0'}")
        print()
        print("[右摇杆]")
        print(f"  X:{d(rx)}  Y:{d(ry)}")
        print()
        print("[扳机]")
        print(f"  LT:{d(lt)}  RT:{d(rt)}")
except KeyboardInterrupt:
    pass
finally:
    dev.ungrab()
    print("\n退出")
