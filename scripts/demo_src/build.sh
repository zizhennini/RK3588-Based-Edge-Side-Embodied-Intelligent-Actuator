#!/bin/bash
# 编译 RKLLM 官方多模态 demo（v1.2.3）并安装到 /usr/bin/
set -e
cd "$(dirname "$0")"

echo "编译 RKLLM 多模态 demo ..."

OpenCV_LIBS=$(pkg-config --cflags --libs opencv4 2>/dev/null || echo "-lopencv_core -lopencv_imgproc -lopencv_imgcodecs")

g++ -std=c++11 src/main.cpp src/image_enc.cc \
    -I src \
    -I 3rdparty \
    -I 3rdparty/opencv/opencv-linux-aarch64/include \
    -I 3rdparty/librknnrt/Linux/librknn_api/include \
    /usr/local/lib/librkllmrt.so /usr/local/lib/librknnrt.so \
    /usr/lib/aarch64-linux-gnu/libopencv_core.so \
    /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so \
    /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so \
    -o demo -lpthread

echo "编译成功"

sudo cp demo /usr/bin/demo_v1.2.3
echo "已安装到 /usr/bin/demo_v1.2.3"
