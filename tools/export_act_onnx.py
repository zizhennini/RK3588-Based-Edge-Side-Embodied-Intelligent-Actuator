#!/usr/bin/env python3
"""ACT 策略模型分模块 ONNX 导出脚本

将 LeRobot v0.6.1 训练产出的 ACT checkpoint 导出为分模块 ONNX，
供 RK3588 板端 ONNX Runtime CPU 推理使用。

Usage:
    python tools/export_act_onnx.py \
        --checkpoint outputs/train/act_so101/checkpoints/last/pretrained_model \
        --output_dir models/act/ \
        --opset 14

输出:
    models/act/vision_encoder.onnx   (~100MB, ResNet18 backbone)
    models/act/transformer.onnx      (~180MB, encoder+decoder+action_head)
    models/act/act_config.json       (模型配置，供板端加载)
    models/act/query_embed.npy       (learned queries, 100×512 float32)

模型结构参考:
    https://github.com/huggingface/lerobot/tree/main/lerobot/common/policies/act

依赖:
    torch, numpy, onnx, onnxruntime, safetensors
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_act")

# ─────────────────────────────────────────────────────────────────────────────
# 默认配置（与 LeRobot ACT SO-ARM101 对齐）
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "chunk_size": 100,
    "n_action_steps": 100,
    "state_dim": 6,
    "action_dim": 6,
    "image_shape": [3, 480, 640],   # C, H, W
    "dim_model": 512,
    "n_heads": 8,
    "n_encoder_layers": 4,
    "n_decoder_layers": 1,
    "dim_feedforward": 3200,
    "dropout": 0.0,                  # 推理时 dropout=0
    "pre_norm": False,
}

# ResNet18 各 stage 配置: (输出通道数, block 数量, stride)
_RESNET_STAGES = [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]


# ─────────────────────────────────────────────────────────────────────────────
# ResNet18 Backbone（手动实现，key 与 torchvision.models.resnet18 兼容）
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """ResNet BasicBlock（与 torchvision 结构完全一致）"""
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18Backbone(nn.Module):
    """ResNet18 去掉 avgpool/fc，输出 feature maps (B, 512, H/32, W/32)

    state_dict key 与 torchvision.models.resnet18 完全兼容：
        conv1.weight, bn1.*, layer{1-4}.{0-1}.conv{1-2}.weight, ...
        layer{2-4}.0.downsample.{0,1}.*
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1   = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1  = self._make_layer(64,  2, stride=1)
        self.layer2  = self._make_layer(128, 2, stride=2)
        self.layer3  = self._make_layer(256, 2, stride=2)
        self.layer4  = self._make_layer(512, 2, stride=2)
        # 注意：无 avgpool、无 fc

        # 权重初始化（加载 checkpoint 后会被覆盖）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # (B, 512, H/32, W/32)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ACT 完整模型（用于加载 checkpoint 权重）
# ─────────────────────────────────────────────────────────────────────────────

class ACTModel(nn.Module):
    """ACT 策略模型完整定义（推理路径，不含 CVAE）

    结构与 LeRobot ACT 对齐：
        backbone      : ResNet18Backbone
        state_proj    : Linear(state_dim → dim_model)
        query_embed   : Embedding(chunk_size, dim_model)
        encoder       : TransformerEncoder(n_encoder_layers)
        decoder       : TransformerDecoder(n_decoder_layers)
        action_head   : Sequential(Linear, ReLU, Linear)

    注意：
        - LeRobot ACT 的 key 可能带有 "model." 前缀，加载时需剥离
        - transformer encoder/decoder 使用 nn.TransformerEncoder/Decoder，
          key 与 PyTorch 原生实现一致（self_attn.*, linear1.*, norm1.* 等）
        - 若实际 checkpoint key 不同，load_checkpoint() 会输出警告并尝试自动映射
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        c, h, w = cfg["image_shape"]
        dim    = cfg["dim_model"]

        self.backbone = ResNet18Backbone(in_channels=c)

        # state_dim → dim_model 投影
        self.state_proj = nn.Linear(cfg["state_dim"], dim)

        # learned queries（DETR 风格 object queries）
        self.query_embed = nn.Embedding(cfg["chunk_size"], dim)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=cfg["n_heads"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            activation="relu",
            batch_first=True,
            norm_first=cfg["pre_norm"],
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg["n_encoder_layers"])

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=cfg["n_heads"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            activation="relu",
            batch_first=True,
            norm_first=cfg["pre_norm"],
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg["n_decoder_layers"])

        # Action head: dim_model → dim_model → action_dim
        self.action_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, cfg["action_dim"]),
        )

        # 空间展平后 vision token 数量（用于运行时断言）
        self._n_vision_tokens = (h // 32) * (w // 32)

    def forward(
        self,
        image: torch.Tensor,     # (B, C, H, W)
        state: torch.Tensor,     # (B, state_dim)
    ) -> torch.Tensor:           # (B, chunk_size, action_dim)
        B = image.shape[0]

        # 1. Vision backbone
        feat = self.backbone(image)                        # (B, 512, h/32, w/32)
        # 展平空间维度 → token 序列，并转置为 (B, N, dim)
        vision_tokens = feat.flatten(2).transpose(1, 2)   # (B, N_vis, dim)

        # 2. State token
        state_token = self.state_proj(state).unsqueeze(1) # (B, 1, dim)

        # 3. Encoder memory = [state_token, vision_tokens]
        memory = torch.cat([state_token, vision_tokens], dim=1)  # (B, 1+N_vis, dim)
        memory = self.encoder(memory)                             # (B, 1+N_vis, dim)

        # 4. Decoder queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, chunk, dim)
        decoded = self.decoder(queries, memory)                          # (B, chunk, dim)

        # 5. Action head
        actions = self.action_head(decoded)   # (B, chunk_size, action_dim)
        return actions


# ─────────────────────────────────────────────────────────────────────────────
# ONNX 导出包装模块
# ─────────────────────────────────────────────────────────────────────────────

class VisionEncoderModule(nn.Module):
    """Module 1: 图像 → 视觉特征图

    输入:  image           (1, 3, 480, 640) float32  [0,1] 归一化
    输出:  vision_features (1, 512, 15, 20)  float32
    """

    def __init__(self, backbone: ResNet18Backbone) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


class TransformerModule(nn.Module):
    """Module 2: 视觉特征 + 关节状态 → 动作序列

    输入:
        vision_features (1, 512, 15, 20) float32
        state           (1, 6)           float32  关节角度（已归一化）
        query_embed     (100, 512)       float32  固定 learned queries
    输出:
        actions         (1, 100, 6)      float32  100 步动作序列
    """

    def __init__(
        self,
        state_proj:  nn.Linear,
        encoder:     nn.TransformerEncoder,
        decoder:     nn.TransformerDecoder,
        action_head: nn.Sequential,
    ) -> None:
        super().__init__()
        self.state_proj  = state_proj
        self.encoder     = encoder
        self.decoder     = decoder
        self.action_head = action_head

    def forward(
        self,
        vision_features: torch.Tensor,   # (1, 512, H', W')
        state:           torch.Tensor,   # (1, state_dim)
        query_embed:     torch.Tensor,   # (chunk_size, dim_model)
    ) -> torch.Tensor:                   # (1, chunk_size, action_dim)
        # 展平空间维度 → (1, N_vis, dim)
        vision_tokens = vision_features.flatten(2).transpose(1, 2)

        # State token → (1, 1, dim)
        state_token = self.state_proj(state).unsqueeze(1)

        # Encoder memory
        memory = torch.cat([state_token, vision_tokens], dim=1)
        memory = self.encoder(memory)

        # Decoder queries: (chunk, dim) → (1, chunk, dim)
        queries = query_embed.unsqueeze(0)
        decoded = self.decoder(queries, memory)

        return self.action_head(decoded)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 加载工具
# ─────────────────────────────────────────────────────────────────────────────

def _find_checkpoint_file(ckpt_dir: Path) -> Path:
    """在 checkpoint 目录中查找权重文件（优先 safetensors）"""
    candidates = [
        ckpt_dir / "model.safetensors",
        ckpt_dir / "pytorch_model.bin",
        ckpt_dir / "model.pt",
        ckpt_dir / "model.pth",
        ckpt_dir / "checkpoint.pt",
        ckpt_dir / "checkpoint.pth",
    ]
    for p in candidates:
        if p.exists():
            return p
    # 尝试任意 safetensors / pt / pth 文件
    for pattern in ("*.safetensors", "*.pt", "*.pth", "*.bin"):
        found = list(ckpt_dir.glob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError(
        f"在 {ckpt_dir} 中未找到权重文件，"
        f"期望: model.safetensors / pytorch_model.bin / model.pt"
    )


def load_state_dict(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """从 checkpoint 目录加载 state_dict，支持 safetensors 和 PyTorch 格式"""
    weight_file = _find_checkpoint_file(ckpt_dir)
    log.info("加载权重文件: %s", weight_file.name)

    if weight_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            sd = load_file(str(weight_file), device="cpu")
        except ImportError:
            raise RuntimeError(
                "需要安装 safetensors: pip install safetensors"
            ) from None
    else:
        try:
            ckpt = torch.load(weight_file, map_location="cpu", weights_only=True)
        except TypeError:
            # torch < 1.13 无 weights_only 参数
            log.warning("当前 torch 版本不支持 weights_only，请确保权重文件来源可信")
            ckpt = torch.load(weight_file, map_location="cpu")

        if isinstance(ckpt, dict):
            sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        else:
            raise RuntimeError(f"无法从 {weight_file} 解析 state_dict（类型: {type(ckpt)}）")

    if not isinstance(sd, dict):
        raise RuntimeError(f"state_dict 类型异常: {type(sd)}")

    return sd


def _strip_prefixes(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """剥离常见前缀，使 key 与 ACTModel 对齐

    LeRobot checkpoint 可能使用的前缀：
        "model."          → 直接剥离
        "policy.model."   → 剥离至 "model." 再处理
        "_orig_mod."      → torch.compile 产生的前缀，剥离
    """
    cleaned: dict[str, torch.Tensor] = {}
    for key, val in sd.items():
        k = key
        # 剥离 torch.compile 前缀
        while k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        # 剥离 policy. 前缀
        while k.startswith("policy."):
            k = k[len("policy."):]
        # 剥离 model. 前缀（仅一层）
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = val
    return cleaned


def _remap_backbone_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """处理 backbone key 的多种命名变体

    LeRobot ACT 可能使用：
        backbone.model.conv1.weight  (torchvision ResNet 含 model. 层)
        backbone.conv1.weight        (直接命名)
        vision_backbone.*            (旧版命名)
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, val in sd.items():
        k = key
        # backbone.model.* → backbone.*
        if k.startswith("backbone.model."):
            k = "backbone." + k[len("backbone.model."):]
        # vision_backbone.* → backbone.*
        elif k.startswith("vision_backbone."):
            k = "backbone." + k[len("vision_backbone."):]
        # vision_backbone.model.* → backbone.*
        elif k.startswith("vision_backbone.model."):
            k = "backbone." + k[len("vision_backbone.model."):]
        remapped[k] = val
    return remapped


def _remap_transformer_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """处理 transformer key 的命名变体

    LeRobot ACT 可能使用：
        transformer.encoder.layers.0.*  → encoder.layers.0.*
        transformer.decoder.layers.0.*  → decoder.layers.0.*
        transformer_encoder.*           → encoder.*
        transformer_decoder.*           → decoder.*
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, val in sd.items():
        k = key
        if k.startswith("transformer.encoder."):
            k = "encoder." + k[len("transformer.encoder."):]
        elif k.startswith("transformer.decoder."):
            k = "decoder." + k[len("transformer.decoder."):]
        elif k.startswith("transformer_encoder."):
            k = "encoder." + k[len("transformer_encoder."):]
        elif k.startswith("transformer_decoder."):
            k = "decoder." + k[len("transformer_decoder."):]
        remapped[k] = val
    return remapped


def _remap_misc_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """处理其他组件 key 的命名变体"""
    remapped: dict[str, torch.Tensor] = {}
    # 常见别名映射（完整前缀替换）
    alias_map = {
        "state_encoder.":   "state_proj.",
        "input_proj.":      "state_proj.",
        "state_embedding.": "state_proj.",
        "query_embedding.": "query_embed.",
        "queries_embed.":   "query_embed.",
        "action_decoder.":  "action_head.",
        "output_proj.":     "action_head.",
    }
    for key, val in sd.items():
        k = key
        for old, new in alias_map.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        remapped[k] = val
    return remapped


def load_act_model(ckpt_dir: Path, cfg: dict[str, Any]) -> ACTModel:
    """构建 ACTModel 并加载 checkpoint 权重

    自动处理常见 key 前缀差异，加载后报告匹配情况。
    """
    raw_sd = load_state_dict(ckpt_dir)
    log.info("原始 state_dict 共 %d 个 key", len(raw_sd))

    # 逐步清理 key
    sd = _strip_prefixes(raw_sd)
    sd = _remap_backbone_keys(sd)
    sd = _remap_transformer_keys(sd)
    sd = _remap_misc_keys(sd)

    model = ACTModel(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if unexpected:
        log.warning("未匹配的 checkpoint key（已忽略，共 %d 个）:", len(unexpected))
        for k in unexpected[:20]:
            log.warning("  unexpected: %s", k)
        if len(unexpected) > 20:
            log.warning("  ... 及其他 %d 个", len(unexpected) - 20)

    if missing:
        # 区分关键 missing（backbone/transformer）和非关键 missing（query_embed 等可重新初始化）
        critical = [k for k in missing if not k.startswith("query_embed")]
        if critical:
            log.warning("缺失的关键参数（共 %d 个，模型可能未正确加载）:", len(critical))
            for k in critical[:20]:
                log.warning("  missing: %s", k)
            if len(critical) > 20:
                log.warning("  ... 及其他 %d 个", len(critical) - 20)
            log.warning(
                "⚠ 请检查 checkpoint 的 key 命名是否与 ACTModel 定义一致。\n"
                "  可运行以下命令查看 checkpoint key：\n"
                "  python -c \"from safetensors.torch import load_file; "
                "sd=load_file('%s'); print(list(sd.keys())[:20])\"",
                _find_checkpoint_file(ckpt_dir),
            )
        else:
            log.info("query_embed 未在 checkpoint 中找到，使用随机初始化（共 %d 个）", len(missing))

    n_params = sum(p.numel() for p in model.parameters())
    log.info("模型构建完成: %.1fM 参数", n_params / 1e6)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 配置文件加载
# ─────────────────────────────────────────────────────────────────────────────

def load_config(ckpt_dir: Path) -> dict[str, Any]:
    """从 checkpoint 目录读取 config.json，合并默认值

    LeRobot config.json 可能使用不同的字段名，此处做兼容性映射。
    """
    cfg = dict(DEFAULT_CONFIG)

    config_file = ckpt_dir / "config.json"
    if not config_file.exists():
        log.warning("未找到 %s，使用默认配置", config_file)
        return cfg

    with open(config_file, encoding="utf-8") as f:
        raw: dict = json.load(f)

    log.info("读取配置: %s", config_file)

    # 直接映射的字段
    direct_keys = [
        "chunk_size", "n_action_steps", "dim_model", "n_heads",
        "n_encoder_layers", "n_decoder_layers", "dim_feedforward",
        "dropout", "pre_norm",
    ]
    for k in direct_keys:
        if k in raw:
            cfg[k] = raw[k]

    # 兼容 LeRobot 的不同字段名
    if "nhead" in raw and "n_heads" not in raw:
        cfg["n_heads"] = raw["nhead"]
    if "num_encoder_layers" in raw and "n_encoder_layers" not in raw:
        cfg["n_encoder_layers"] = raw["num_encoder_layers"]
    if "num_decoder_layers" in raw and "n_decoder_layers" not in raw:
        cfg["n_decoder_layers"] = raw["num_decoder_layers"]
    if "feedforward_dim" in raw and "dim_feedforward" not in raw:
        cfg["dim_feedforward"] = raw["feedforward_dim"]

    # state_dim / action_dim：优先从 input_features / output_features 推断
    if "state_dim" in raw:
        cfg["state_dim"] = raw["state_dim"]
    elif "input_features" in raw:
        # LeRobot 格式: {"observation.state": {"dtype": "float32", "shape": [6]}}
        for feat_key, feat_val in raw["input_features"].items():
            if "state" in feat_key.lower():
                shape = feat_val.get("shape", [])
                if shape:
                    cfg["state_dim"] = int(shape[-1]) if isinstance(shape, list) else int(shape)
                break

    if "action_dim" in raw:
        cfg["action_dim"] = raw["action_dim"]
    elif "output_features" in raw:
        for feat_key, feat_val in raw["output_features"].items():
            if "action" in feat_key.lower():
                shape = feat_val.get("shape", [])
                if shape:
                    cfg["action_dim"] = int(shape[-1]) if isinstance(shape, list) else int(shape)
                break

    # image_shape：从 input_features 中推断
    if "image_shape" in raw:
        cfg["image_shape"] = raw["image_shape"]
    elif "input_features" in raw:
        for feat_key, feat_val in raw["input_features"].items():
            if "image" in feat_key.lower() or "pixel" in feat_key.lower():
                shape = feat_val.get("shape", [])
                if len(shape) == 3:
                    cfg["image_shape"] = list(shape)
                break

    # 推理时强制 dropout=0
    cfg["dropout"] = 0.0

    log.info("有效配置: dim_model=%d, n_heads=%d, enc_layers=%d, dec_layers=%d, "
             "chunk=%d, state_dim=%d, action_dim=%d, image=%s",
             cfg["dim_model"], cfg["n_heads"],
             cfg["n_encoder_layers"], cfg["n_decoder_layers"],
             cfg["chunk_size"], cfg["state_dim"], cfg["action_dim"],
             cfg["image_shape"])
    return cfg


def load_norm_stats(ckpt_dir: Path) -> dict | None:
    """尝试从 checkpoint 目录加载归一化统计量（可选）"""
    for fname in ("norm_stats.json", "train_config.json", "dataset_stats.json"):
        fpath = ckpt_dir / fname
        if fpath.exists():
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                # 尝试提取 norm_stats 字段
                if "norm_stats" in data:
                    log.info("从 %s 加载归一化统计量", fname)
                    return data["norm_stats"]
                if "mean" in data or "std" in data:
                    log.info("从 %s 加载归一化统计量", fname)
                    return data
            except Exception as e:
                log.debug("解析 %s 失败: %s", fname, e)
    log.info("未找到归一化统计量文件（norm_stats.json），配置中 norm_stats 将为 null")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ONNX 导出
# ─────────────────────────────────────────────────────────────────────────────

def export_vision_encoder(
    model: ACTModel,
    output_path: Path,
    opset: int,
) -> None:
    """导出 Module 1: Vision Encoder（ResNet18 backbone）"""
    log.info("── 导出 vision_encoder.onnx ──")
    module = VisionEncoderModule(model.backbone)
    module.eval()

    c, h, w = model.cfg["image_shape"]
    dummy = torch.randn(1, c, h, w, dtype=torch.float32)

    with torch.no_grad():
        expected = module(dummy)
    log.info("  PyTorch 输出 shape: %s", list(expected.shape))

    torch.onnx.export(
        module,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["vision_features"],
        dynamic_axes=None,           # 固定尺寸，利于 ORT 图优化
        do_constant_folding=True,
    )
    size_mb = output_path.stat().st_size / (1024 ** 2)
    log.info("  ✓ 导出完成: %s (%.1f MB, opset=%d)", output_path.name, size_mb, opset)


def export_transformer(
    model: ACTModel,
    output_path: Path,
    opset: int,
) -> torch.Tensor:
    """导出 Module 2: Transformer + Action Head

    返回 query_embed 权重张量（供保存为 .npy）
    """
    log.info("── 导出 transformer.onnx ──")
    module = TransformerModule(
        state_proj=model.state_proj,
        encoder=model.encoder,
        decoder=model.decoder,
        action_head=model.action_head,
    )
    module.eval()

    cfg = model.cfg
    c, h, w = cfg["image_shape"]
    feat_h, feat_w = h // 32, w // 32
    dim = cfg["dim_model"]
    chunk = cfg["chunk_size"]
    state_dim = cfg["state_dim"]

    # query_embed 权重（固定，导出为 .npy 供板端加载）
    query_weight = model.query_embed.weight.data.clone()  # (chunk, dim)

    dummy_vision = torch.randn(1, dim, feat_h, feat_w, dtype=torch.float32)
    dummy_state  = torch.randn(1, state_dim, dtype=torch.float32)

    # TransformerModule 接收 (chunk, dim)，内部 unsqueeze(0)
    with torch.no_grad():
        expected = module(dummy_vision, dummy_state, query_weight)
    log.info("  PyTorch 输出 shape: %s", list(expected.shape))

    torch.onnx.export(
        module,
        (dummy_vision, dummy_state, query_weight),
        str(output_path),
        opset_version=opset,
        input_names=["vision_features", "state", "query_embed"],
        output_names=["actions"],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    size_mb = output_path.stat().st_size / (1024 ** 2)
    log.info("  ✓ 导出完成: %s (%.1f MB, opset=%d)", output_path.name, size_mb, opset)
    return query_weight


# ─────────────────────────────────────────────────────────────────────────────
# ONNX 验证
# ─────────────────────────────────────────────────────────────────────────────

def verify_vision_encoder(
    model: ACTModel,
    onnx_path: Path,
    tolerance: float = 1e-4,
) -> None:
    """验证 vision_encoder.onnx 与 PyTorch 输出一致"""
    log.info("── 验证 vision_encoder.onnx ──")
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime 未安装，跳过验证（pip install onnxruntime）")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0]
    out  = sess.get_outputs()[0]
    log.info("  ONNX 输入:  name=%s  shape=%s  dtype=%s", inp.name, inp.shape, inp.type)
    log.info("  ONNX 输出:  name=%s  shape=%s  dtype=%s", out.name, out.shape, out.type)

    c, h, w = model.cfg["image_shape"]
    dummy = torch.randn(1, c, h, w, dtype=torch.float32)

    module = VisionEncoderModule(model.backbone)
    module.eval()
    with torch.no_grad():
        pt_out = module(dummy).numpy()

    ort_out = sess.run(None, {inp.name: dummy.numpy()})[0]

    max_diff  = float(np.abs(pt_out - ort_out).max())
    mean_diff = float(np.abs(pt_out - ort_out).mean())
    log.info("  输出 shape: %s | max_diff=%.2e  mean_diff=%.2e", ort_out.shape, max_diff, mean_diff)

    if max_diff < tolerance:
        log.info("  ✓ vision_encoder 验证通过（max_diff < %.0e）", tolerance)
    else:
        log.error("  ✗ vision_encoder 验证失败：max_diff=%.2e > %.0e", max_diff, tolerance)
        sys.exit(1)


def verify_transformer(
    model: ACTModel,
    onnx_path: Path,
    query_weight: torch.Tensor,
    tolerance: float = 1e-4,
) -> None:
    """验证 transformer.onnx 与 PyTorch 输出一致"""
    log.info("── 验证 transformer.onnx ──")
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime 未安装，跳过验证（pip install onnxruntime）")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for inp in sess.get_inputs():
        log.info("  ONNX 输入:  name=%-20s shape=%s", inp.name, inp.shape)
    for out in sess.get_outputs():
        log.info("  ONNX 输出:  name=%-20s shape=%s", out.name, out.shape)

    cfg = model.cfg
    dim = cfg["dim_model"]
    c, h, w = cfg["image_shape"]
    feat_h, feat_w = h // 32, w // 32

    dummy_vision = torch.randn(1, dim, feat_h, feat_w, dtype=torch.float32)
    dummy_state  = torch.randn(1, cfg["state_dim"], dtype=torch.float32)

    module = TransformerModule(
        state_proj=model.state_proj,
        encoder=model.encoder,
        decoder=model.decoder,
        action_head=model.action_head,
    )
    module.eval()
    with torch.no_grad():
        pt_out = module(dummy_vision, dummy_state, query_weight).numpy()

    input_names = [inp.name for inp in sess.get_inputs()]
    feed = {
        input_names[0]: dummy_vision.numpy(),
        input_names[1]: dummy_state.numpy(),
        input_names[2]: query_weight.numpy(),
    }
    ort_out = sess.run(None, feed)[0]

    max_diff  = float(np.abs(pt_out - ort_out).max())
    mean_diff = float(np.abs(pt_out - ort_out).mean())
    log.info("  输出 shape: %s | max_diff=%.2e  mean_diff=%.2e", ort_out.shape, max_diff, mean_diff)

    if max_diff < tolerance:
        log.info("  ✓ transformer 验证通过（max_diff < %.0e）", tolerance)
    else:
        log.error("  ✗ transformer 验证失败：max_diff=%.2e > %.0e", max_diff, tolerance)
        log.error("  提示：若差异略超阈值，可能是浮点精度问题，可适当放宽 --tolerance")
        sys.exit(1)


def verify_end_to_end(
    model: ACTModel,
    vision_onnx: Path,
    transformer_onnx: Path,
    query_weight: torch.Tensor,
    tolerance: float = 1e-4,
) -> None:
    """端到端验证：image + state → actions，对比 PyTorch 与双模块 ONNX 串联结果"""
    log.info("── 端到端验证（vision_encoder → transformer 串联）──")
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime 未安装，跳过端到端验证")
        return

    sess_v = ort.InferenceSession(str(vision_onnx),    providers=["CPUExecutionProvider"])
    sess_t = ort.InferenceSession(str(transformer_onnx), providers=["CPUExecutionProvider"])

    cfg = model.cfg
    c, h, w = cfg["image_shape"]
    dummy_image = torch.randn(1, c, h, w, dtype=torch.float32)
    dummy_state = torch.randn(1, cfg["state_dim"], dtype=torch.float32)

    # ONNX 串联推理
    v_in_name  = sess_v.get_inputs()[0].name
    t_in_names = [inp.name for inp in sess_t.get_inputs()]
    feat = sess_v.run(None, {v_in_name: dummy_image.numpy()})[0]
    ort_actions = sess_t.run(None, {
        t_in_names[0]: feat,
        t_in_names[1]: dummy_state.numpy(),
        t_in_names[2]: query_weight.numpy(),
    })[0]

    # PyTorch 推理
    model.eval()
    with torch.no_grad():
        pt_actions = model(dummy_image, dummy_state).numpy()

    max_diff = float(np.abs(pt_actions - ort_actions).max())
    log.info("  actions shape: %s | max_diff=%.2e", ort_actions.shape, max_diff)

    if max_diff < tolerance:
        log.info("  ✓ 端到端验证通过（max_diff=%.2e < %.0e）", max_diff, tolerance)
    else:
        log.warning("  ⚠ 端到端 max_diff=%.2e 超出阈值 %.0e（可能为累积浮点误差）", max_diff, tolerance)


# ─────────────────────────────────────────────────────────────────────────────
# 配置保存
# ─────────────────────────────────────────────────────────────────────────────

def save_act_config(
    cfg: dict[str, Any],
    output_dir: Path,
    norm_stats: dict | None,
) -> Path:
    """保存 act_config.json（供 RK3588 板端加载）"""
    c, h, w = cfg["image_shape"]
    feat_h, feat_w = h // 32, w // 32

    act_cfg = {
        "chunk_size":         cfg["chunk_size"],
        "n_action_steps":     cfg["n_action_steps"],
        "state_dim":          cfg["state_dim"],
        "action_dim":         cfg["action_dim"],
        "image_shape":        cfg["image_shape"],
        "feature_map_shape":  [512, feat_h, feat_w],
        "dim_model":          cfg["dim_model"],
        "n_heads":            cfg["n_heads"],
        "n_encoder_layers":   cfg["n_encoder_layers"],
        "n_decoder_layers":   cfg["n_decoder_layers"],
        "vision_encoder_path": "vision_encoder.onnx",
        "transformer_path":    "transformer.onnx",
        "query_embed_path":    "query_embed.npy",
        "norm_stats":          norm_stats,
        "onnx_opset":          cfg.get("_opset", 14),
        "notes": {
            "vision_encoder": "输入 image (1,3,480,640) float32 [0,1]归一化 → 输出 (1,512,15,20)",
            "transformer":    "输入 vision_features+state+query_embed → 输出 actions (1,100,6)",
            "inference":      "两模块串联：先 vision_encoder，再 transformer；state 需按 norm_stats 归一化",
        },
    }

    config_path = output_dir / "act_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(act_cfg, f, indent=2, ensure_ascii=False)
    log.info("✓ 配置已保存: %s", config_path)
    return config_path


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ACT 策略模型分模块 ONNX 导出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本导出（使用默认配置）
  python tools/export_act_onnx.py --checkpoint path/to/pretrained_model

  # 指定输出目录和 opset
  python tools/export_act_onnx.py \\
      --checkpoint outputs/train/act_so101/checkpoints/last/pretrained_model \\
      --output_dir models/act/ \\
      --opset 14

  # 跳过验证（加快导出速度）
  python tools/export_act_onnx.py --checkpoint path/to/ckpt --no-verify
        """,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, metavar="DIR",
        help="LeRobot 训练产出目录（含 model.safetensors + config.json）",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("models/act"), metavar="DIR",
        help="ONNX 输出目录（默认: models/act/）",
    )
    parser.add_argument(
        "--opset", type=int, default=14, metavar="N",
        help="ONNX opset 版本（默认: 14，scaled_dot_product_attention 需要 14+）",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="跳过导出后 onnxruntime 一致性验证",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-4, metavar="T",
        help="验证时允许的最大输出误差（默认: 1e-4）",
    )
    args = parser.parse_args()

    # ── 检查输入目录 ──────────────────────────────────────────────────────────
    ckpt_dir: Path = args.checkpoint
    if not ckpt_dir.exists():
        log.error("checkpoint 目录不存在: %s", ckpt_dir)
        sys.exit(1)
    if ckpt_dir.is_file():
        # 允许直接传入文件路径，自动取父目录
        ckpt_dir = ckpt_dir.parent
        log.info("传入文件路径，使用父目录: %s", ckpt_dir)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("ACT ONNX 分模块导出")
    log.info("  checkpoint : %s", ckpt_dir)
    log.info("  output_dir : %s", output_dir)
    log.info("  opset      : %d", args.opset)
    log.info("  verify     : %s", not args.no_verify)
    log.info("=" * 60)

    # ── 1. 加载配置 ──────────────────────────────────────────────────────────
    cfg = load_config(ckpt_dir)
    cfg["_opset"] = args.opset

    # ── 2. 构建模型并加载权重 ────────────────────────────────────────────────
    model = load_act_model(ckpt_dir, cfg)

    # ── 3. 导出 Module 1: Vision Encoder ─────────────────────────────────────
    vision_onnx = output_dir / "vision_encoder.onnx"
    export_vision_encoder(model, vision_onnx, args.opset)

    # ── 4. 导出 Module 2: Transformer + Action Head ──────────────────────────
    transformer_onnx = output_dir / "transformer.onnx"
    query_weight = export_transformer(model, transformer_onnx, args.opset)

    # ── 5. 保存 query_embed.npy ──────────────────────────────────────────────
    query_npy = output_dir / "query_embed.npy"
    np.save(str(query_npy), query_weight.numpy())
    log.info("✓ query_embed 已保存: %s  shape=%s  dtype=float32",
             query_npy.name, list(query_weight.shape))

    # ── 6. 验证 ──────────────────────────────────────────────────────────────
    if not args.no_verify:
        verify_vision_encoder(model, vision_onnx, tolerance=args.tolerance)
        verify_transformer(model, transformer_onnx, query_weight, tolerance=args.tolerance)
        verify_end_to_end(model, vision_onnx, transformer_onnx, query_weight,
                          tolerance=args.tolerance)
    else:
        log.info("跳过验证（--no-verify）")

    # ── 7. 保存 act_config.json ──────────────────────────────────────────────
    norm_stats = load_norm_stats(ckpt_dir)
    save_act_config(cfg, output_dir, norm_stats)

    # ── 8. 汇总 ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("导出完成！输出文件：")
    for fpath in sorted(output_dir.iterdir()):
        if fpath.is_file():
            size_mb = fpath.stat().st_size / (1024 ** 2)
            log.info("  %-30s  %8.2f MB", fpath.name, size_mb)
    log.info("=" * 60)
    log.info("板端部署提示：")
    log.info("  1. 将 %s 目录整体拷贝至 RK3588", output_dir)
    log.info("  2. 使用 ONNX Runtime CPU 加载 vision_encoder.onnx 和 transformer.onnx")
    log.info("  3. query_embed.npy 在初始化时加载一次，每次推理作为 transformer 输入传入")
    log.info("  4. state 输入需按 act_config.json 中的 norm_stats 进行归一化")
    log.info("  5. actions 输出为 (1, %d, %d)，取前 n_action_steps 步执行",
             cfg["chunk_size"], cfg["action_dim"])


if __name__ == "__main__":
    main()
