#!/usr/bin/env bash
# 启动 LeRobot 示教数据采集
set -e
cd "$(dirname "$0")/.."

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyUSB1 \
    --fps=30 \
    --root=data \
    --repo-id=my-vla-dataset
