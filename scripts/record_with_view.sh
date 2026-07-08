#!/usr/bin/env bash
# 一键录制：D435i 画面 + 轨迹录制同步启动
set -e
cd "$(dirname "$0")/.."
source /home/elf/work/miniconda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate rkvla 2>/dev/null || true

# 启动 D435i 画面（后台）
python3 scripts/d435i_viewer.py &
VIEWER_PID=$!
sleep 2

echo ""
echo "=== D435i 画面已启动，按 Ctrl+C 停止录制后画面自动关闭 ==="
echo ""

# 启动录制
python3 scripts/develop_motion.py "$@"

# 录制结束，关闭画面
kill $VIEWER_PID 2>/dev/null || true
wait $VIEWER_PID 2>/dev/null || true
echo "=== 录制完成 ==="
