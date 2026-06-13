#!/bin/bash
# 编译 whisper.cpp（RK3588 ARM64 优化）
set -e
cd "$(dirname "$0")/.."

WHISPER_DIR="scripts/whisper"

if [ ! -d "$WHISPER_DIR" ]; then
    echo "下载 whisper.cpp ..."
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$WHISPER_DIR"
fi

cd "$WHISPER_DIR"

echo "编译 whisper.cpp（ARM64 优化）..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NO_FLASH_ATTN=1
cmake --build build -j4 --config Release

echo "下载 tiny 模型（~75MB，最快）..."
bash models/download-ggml-model.sh tiny

echo "安装 whisper-cli 到系统..."
sudo cp build/bin/whisper-cli /usr/bin/whisper
sudo cp models/ggml-tiny.bin /usr/local/share/whisper-tiny.bin

echo ""
echo "✓ whisper.cpp 编译完成"
echo ""
echo "测试："
echo "  whisper -f samples/jfk.wav -m /usr/local/share/whisper-tiny.bin -l zh"
