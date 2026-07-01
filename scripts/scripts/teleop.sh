#!/usr/bin/env bash
# 启动 LeRobot 主从遥操作示教 + 数据录制
set -e
cd "$(dirname "$0")/.."

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyUSB1
