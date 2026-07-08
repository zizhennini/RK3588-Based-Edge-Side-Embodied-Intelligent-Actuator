#!/usr/bin/env bash
# 启动 LeRobot 主从遥操作（已完善的官方实现）
set -e
cd "$(dirname "$0")/.."

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1
