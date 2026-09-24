#!/usr/bin/env python3
"""语音助手独立入口 — 兼容旧文档用法 (Deprecated wrapper)

refactor_plan_v9 第 1.6 节: 统一入口为 main.py（--mode voice 进入语音交互模式）。
此文件保留兼容 README/旧习惯的独立语音 CLI 用法，实际实现已迁移至 voice/ 包。

Usage:
    python3 va.py listen-forever --wake-mode kws
    python3 va.py ask "画面里有什么物体？"
    python3 va.py once --seconds 4
"""
import sys
import os

# 确保项目根目录在 sys.path（voice/ 包按项目根解析）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice.cli import main

if __name__ == "__main__":
    main()