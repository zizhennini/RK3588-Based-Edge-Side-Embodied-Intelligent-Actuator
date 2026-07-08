#!/usr/bin/env python3
"""RK3588-EIA 主菜单"""
import sys, os, subprocess

def menu():
    print()
    print('==============================================')
    print('  RK3588-EIA 端侧具身智能系统')
    print('==============================================')
    print()
    print('  1. 语音唤醒（说"你好同学"唤醒）')
    print('  2. 语音触发动作回放')
    print('  3. 文字问答（拍照+分析+播报）')
    print('  4. 文字问答（纯文字+播报）')
    print('  5. 遥操作（主从同步）')
    print('  6. 示教录制-平滑-入库')
    print('  7. 动作库管理')
    print('  8. 实验录像（USB摄像头）')
    print('  9. 系统资源监控')
    print('  0. 退出')
    print()

def main():
    while True:
        menu()
        c = input('请选择: ').strip()

        if c == '0':
            break
        elif c == '1':
            os.system('python3 va.py listen-forever --wake-mode kws')
        elif c == '2':
            os.system('python3 va.py listen-forever --wake-mode kws')
        elif c == '3':
            print('连续问答模式（拍照+分析+播报），输入 q 退出')
            while True:
                q = input('\n问题: ').strip()
                if q.lower() in ('q', 'quit', 'exit', '退出'):
                    break
                if q:
                    os.system('python3 va.py ask "' + q + '"')
            print('退出问答模式')
        elif c == '4':
            print('连续问答模式（纯文字+播报），输入 q 退出')
            while True:
                q = input('\n问题: ').strip()
                if q.lower() in ('q', 'quit', 'exit', '退出'):
                    break
                if q:
                    os.system('python3 va.py ask "' + q + '"')
            print('退出问答模式')
        elif c == '5':
            print('主从遥操作 — 拖拽主臂，从臂跟随')
            os.system('python3 scripts/teleop_record.py --leader /dev/ttyACM1 --follower /dev/ttyACM0 --episode_time_s 300')
        elif c == '6':
            name = input('动作名称: ').strip() or 'new_motion'
            cat = input('分类(reach/grasp/lift/place/retract): ').strip() or 'reach'
            secs = input('录制秒数(默认30, Ctrl+C随时停): ').strip() or '30'
            if secs == '0' or int(secs) <= 0: secs = '30'
            os.system('python3 scripts/develop_motion.py ' + name + ' --category ' + cat + ' --record_seconds ' + secs)
        elif c == '7':
            os.system('python3 scripts/record_trajectory.py list')
        elif c == '8':
            dur = input('录像秒数 (0=手动): ').strip() or '5'
            os.system('python3 scripts/recorder.py --camera usb --duration ' + dur)
        elif c == '9':
            print('监控显示，Ctrl+C 退出')
            os.system('python3 scripts/monitor.py --interval 1')
        else:
            print('无效选择')
            continue

        input('\n按 Enter 返回菜单...')

if __name__ == '__main__':
    main()
