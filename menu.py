#!/usr/bin/env python3
"""RK3588-EIA 主菜单 — 教学/演示便捷入口

refactor_plan_v9 第 1.6 节: 统一入口为 main.py（System 类 + 模式管理）。
此菜单保留旧版交互习惯，并接入 v9 新架构模式:
  - 自主抓取  → main.py --mode autonomous (ACT/GGCNN 视觉引导)
  - 系统诊断  → main.py --mode menu       (硬件自检 + 降级状态报告)

安全改进（相对旧版）:
  - subprocess 参数列表启动，消除 os.system 字符串拼接的 shell 注入隐患
  - sys.executable 替代硬编码 python3，兼容 conda 环境
"""
import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run(args):
    """安全启动子进程（参数列表传递，不经 shell）"""
    try:
        subprocess.run(args, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print("\n已中断")
    except FileNotFoundError as e:
        print(f"启动失败: {e}")


def menu():
    print()
    print('==============================================')
    print('  RK3588-EIA 端侧具身智能系统 (v9 架构)')
    print('==============================================')
    print()
    print('  1. 语音唤醒（说"你好同学"唤醒）')
    print('  2. 语音触发动作回放')
    print('  3. 文字问答（拍照+分析+播报）')
    print('  4. 文字问答（纯文字+播报）')
    print('  5. 自主抓取（ACT/GGCNN 视觉引导）')
    print('  6. 遥操作（主从同步）')
    print('  7. 示教录制-平滑-入库')
    print('  8. 动作库管理')
    print('  9. 实验录像（USB摄像头）')
    print('  10. 系统资源监控')
    print('  11. 系统诊断（硬件自检+降级状态）')
    print('  0. 退出')
    print()


def ask_loop(no_photo: bool):
    """连续文字问答循环

    Args:
        no_photo: True=纯文字问答（禁用拍照意图），False=允许意图触发拍照
    """
    mode = "纯文字" if no_photo else "拍照+分析"
    print(f'连续问答模式（{mode}+播报），输入 q 退出')
    while True:
        q = input('\n问题: ').strip()
        if q.lower() in ('q', 'quit', 'exit', '退出'):
            break
        if q:
            cmd = [PYTHON, 'va.py', 'ask', q]
            if no_photo:
                cmd.append('--no-photo')
            run(cmd)
    print('退出问答模式')


def main():
    while True:
        menu()
        c = input('请选择: ').strip()

        if c == '0':
            break
        elif c == '1':
            run([PYTHON, 'va.py', 'listen-forever', '--wake-mode', 'kws'])
        elif c == '2':
            # 动作回放由语音意图路由自动触发（motion_library/index.json 关键词匹配）
            run([PYTHON, 'va.py', 'listen-forever', '--wake-mode', 'kws'])
        elif c == '3':
            ask_loop(no_photo=False)
        elif c == '4':
            ask_loop(no_photo=True)
        elif c == '5':
            print('自主抓取 — 输入目标描述（如"红色杯子"）回车执行，q 退出')
            run([PYTHON, 'main.py', '--mode', 'autonomous'])
        elif c == '6':
            print('主从遥操作 — 拖拽主臂，从臂跟随')
            run([PYTHON, 'scripts/teleop_record.py', '--leader', '/dev/ttyACM1',
                 '--follower', '/dev/ttyACM0', '--episode_time_s', '300'])
        elif c == '7':
            name = input('动作名称: ').strip() or 'new_motion'
            cat = input('分类(reach/grasp/lift/place/retract): ').strip() or 'reach'
            secs = input('录制秒数(默认30, Ctrl+C随时停): ').strip() or '30'
            if not secs.isdigit() or int(secs) <= 0:
                secs = '30'
            run([PYTHON, 'scripts/develop_motion.py', name, '--category', cat,
                 '--record_seconds', secs])
        elif c == '8':
            run([PYTHON, 'scripts/record_trajectory.py', 'list'])
        elif c == '9':
            dur = input('录像秒数 (0=手动): ').strip() or '5'
            run([PYTHON, 'scripts/recorder.py', '--camera', 'usb', '--duration', dur])
        elif c == '10':
            print('监控显示，Ctrl+C 退出')
            run([PYTHON, 'scripts/monitor.py', '--interval', '1'])
        elif c == '11':
            print('系统诊断 — 初始化全部硬件并报告模块可用性/降级状态')
            run([PYTHON, 'main.py', '--mode', 'menu'])
        else:
            print('无效选择')
            continue

        input('\n按 Enter 返回菜单...')


if __name__ == '__main__':
    main()