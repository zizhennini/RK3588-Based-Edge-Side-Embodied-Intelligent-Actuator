"""GGCNN ONNX 导出脚本（PC 端运行）

从 dougsm/ggcnn 官方权重 (state_dict) 导出固定输入 300x300 的 ONNX 模型,
供 RK3588 端 perception/grasp_detect.py (ONNX Runtime CPU) 推理使用。

使用:
    python tools/export_ggcnn_onnx.py \
        --weights path/to/statedict.pt \
        --output models/ggcnn/ggcnn_cornell_300.onnx

权重来源: https://github.com/dougsm/ggcnn (models/ggcnn_cornell.pth, 纯 state_dict)

输入:  depth (1, 1, 300, 300) float32
输出:  pos / cos / sin / width 各 (1, 1, 300, 300) float32
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("export_ggcnn")

INPUT_SIZE = 300


class GGCNN(nn.Module):
    """GGCNN: 62K 参数实时抓取检测（dougsm/ggcnn）"""

    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=3, padding=3)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.convt1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(8, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(16, 32, kernel_size=9, stride=3, padding=3, output_padding=1)
        self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
        self.width_output = nn.Conv2d(32, 1, kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.convt1(x))
        x = self.relu(self.convt2(x))
        x = self.relu(self.convt3(x))
        pos = self.pos_output(x)
        cos = self.cos_output(x)
        sin = self.sin_output(x)
        width = self.width_output(x)
        return pos, cos, sin, width


def load_state_dict(weights_path: Path) -> dict:
    """加载权重, 兼容 {'state_dict': ...} 包装和 'module.' 前缀 (DataParallel)

    安全: 使用 weights_only=True 避免反序列化任意 Python 对象
    """
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        # torch < 1.13 无 weights_only 参数
        logger.warning("当前 torch 版本不支持 weights_only, 请确保权重文件来源可信")
        ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"无法从 {weights_path} 解析 state_dict (类型: {type(ckpt)})")
    return {k.removeprefix("module."): v for k, v in ckpt.items()}


def verify_onnx(onnx_path: Path, model: GGCNN, dummy: torch.Tensor) -> None:
    """用 onnxruntime 推理并与 PyTorch 输出对比, 校验导出正确性"""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_outs = sess.run(None, {input_name: dummy.numpy()})

    model.eval()
    with torch.no_grad():
        pt_outs = model(dummy)

    names = ["pos", "cos", "sin", "width"]
    for name, pt, ort_out in zip(names, pt_outs, ort_outs):
        max_diff = float(np.abs(pt.numpy() - ort_out).max())
        logger.info(f"输出 {name}: shape={ort_out.shape}, 最大误差={max_diff:.2e}")
        assert ort_out.shape == (1, 1, INPUT_SIZE, INPUT_SIZE), f"{name} 输出形状错误"
        assert max_diff < 1e-4, f"{name} 输出与 PyTorch 不一致 (diff={max_diff})"
    logger.info("✓ ONNX 与 PyTorch 输出一致")


def main():
    parser = argparse.ArgumentParser(description="GGCNN ONNX 导出")
    parser.add_argument("--weights", type=Path, required=True,
                        help="官方 state_dict 路径 (如 ggcnn_cornell.pth)")
    parser.add_argument("--output", type=Path,
                        default=Path("models/ggcnn/ggcnn_cornell_300.onnx"),
                        help="导出 ONNX 路径")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 onnxruntime 一致性校验")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"权重文件不存在: {args.weights}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 1. 构建模型并加载权重
    model = GGCNN(input_channels=1)
    state_dict = load_state_dict(args.weights)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"missing keys: {missing}")
        logger.warning(f"unexpected keys: {unexpected}")
        if missing:
            raise RuntimeError("权重与 GGCNN 结构不匹配, 存在缺失参数, 终止导出")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型加载完成: {n_params / 1e3:.1f}K 参数")

    # 2. 导出 ONNX (固定输入 300x300)
    model.eval()
    dummy = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(args.output),
        opset_version=args.opset,
        input_names=["depth"],
        output_names=["pos", "cos", "sin", "width"],
        dynamic_axes=None,  # 固定尺寸, 利于 ORT 图优化
    )
    size_kb = args.output.stat().st_size / 1024
    logger.info(f"✓ 导出完成: {args.output} ({size_kb:.0f} KB, opset={args.opset})")

    # 3. 一致性校验
    if not args.no_verify:
        verify_onnx(args.output, model, dummy)


if __name__ == "__main__":
    main()
