#!/usr/bin/env python3
"""下载 SmolVLM2-256M-Video-Instruct 模型（HuggingFace）"""
import os, sys
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
TARGET_DIR = Path("./models/SmolVLM2-256M")


def download():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("安装 huggingface-hub: pip install huggingface-hub")
        sys.exit(1)

    print(f"下载 {MODEL_ID} 到 {TARGET_DIR}/")
    print("文件: model.safetensors + config.json + tokenizer + onnx 等")
    print(f"约 5.7GB，请确保网络通畅\n")

    import subprocess
    cmd = [
        sys.executable, "-m", "huggingface_hub", "download",
        MODEL_ID,
        "--local-dir", str(TARGET_DIR),
        "--resume-download",
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"\n✓ 下载完成: {TARGET_DIR}/")
        print(f"  模型文件: {TARGET_DIR / 'model.safetensors'}")
        print(f"  ONNX: {TARGET_DIR / 'onnx/'}")
    else:
        print("\n✗ 下载失败，请手动下载:")
        print(f"  https://huggingface.co/{MODEL_ID}")


if __name__ == "__main__":
    download()
