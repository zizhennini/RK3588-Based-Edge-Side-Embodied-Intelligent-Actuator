#!/usr/bin/env bash
# 启动 VLA 自主抓取模式
set -e
cd "$(dirname "$0")/.."
python main.py
