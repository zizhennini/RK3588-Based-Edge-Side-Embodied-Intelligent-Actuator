#!/usr/bin/env python3
"""旧版语音助手入口 (Deprecated) — 重定向到新 voice/ 包

refactor_plan_v9 T1.6: 语音模块已迁移至 voice/ 包，统一入口为:
    python3 va.py <子命令>            (项目根目录)
    python3 main.py --mode voice     (集成系统语音交互模式)

此文件保留兼容旧的启动习惯（systemd/别名/文档引用），
实际实现委托 voice.cli，不再依赖嵌套的 voice_assistant/voice_assistant/ 旧包。
"""
import os
import sys
import warnings

warnings.warn(
    "voice_assistant/voice_assistant.py 已废弃，请使用项目根目录的 va.py "
    "或 python3 main.py --mode voice",
    DeprecationWarning, stacklevel=2,
)

# 项目根目录 = 本文件所在目录(voice_assistant/)的上一级
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from voice.cli import main

if __name__ == "__main__":
    main()