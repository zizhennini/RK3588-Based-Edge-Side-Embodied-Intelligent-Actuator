#!/usr/bin/env bash
# =============================================================================
# scripts/train_act.sh — PC 端 ACT 训练一键封装（WSL2 / conda env `rk3588`）
#
# 用法:
#   bash scripts/train_act.sh <task> [steps] [batch_size] [device]
#   # 例: bash scripts/train_act.sh pick_place 50000 16 cuda
#
# 环境变量（均可覆盖）:
#   RAW_ROOT   采集原始数据根目录（默认 ~/data/raw）   —— 内含 <task>/episode_*.json
#   DS_ROOT    转换后数据集根目录（默认 ~/datasets）
#   OUT_ROOT   训练输出根目录（默认 ~/train）
#   FORCE      设为 1 时强制重新转换数据集（默认 0，已有则跳过）
#
# 做了什么:
#   ① 环境自检（lerobot/torch/CUDA）
#   ② 原始 JSON(+图像) → LeRobotDataset v3（scripts/json_to_lerobot.py）
#   ③ lerobot-train 训练 ACT（含 push_to_hub=false 等必需开关）
#   ④ 打印 checkpoint 路径与下一步导出命令
#
# 已知坑（详见 docs/m2_m3_training_export_guide.md）:
#   - lerobot 0.6.1 的 dataset/training 栈是可选依赖：pip install "lerobot[dataset]" "lerobot[training]"
#   - 必须显式 --policy.push_to_hub=false，否则报 repo_id missing
#   - 训练数据必须有 action 特征（本仓库转换器已写入：action←J*、observation.state←F*）
# =============================================================================
set -euo pipefail

TASK=${1:?用法: bash scripts/train_act.sh <task> [steps] [batch_size] [device]}
STEPS=${2:-50000}
BATCH=${3:-16}
DEVICE=${4:-cuda}
RAW_ROOT=${RAW_ROOT:-$HOME/data/raw}
DS_ROOT=${DS_ROOT:-$HOME/datasets}
OUT_ROOT=${OUT_ROOT:-$HOME/train}
FORCE=${FORCE:-0}

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RAW_DIR="$RAW_ROOT/$TASK"
DS_DIR="$DS_ROOT/$TASK"
OUT_DIR="$OUT_ROOT/$TASK"

echo "=============================================================="
echo " ACT 训练: task=$TASK  设备=$DEVICE  steps=$STEPS  batch=$BATCH"
echo " 原始数据: $RAW_DIR"
echo " 数据集  : $DS_DIR"
echo " 输出    : $OUT_DIR"
echo "=============================================================="

# ── ① 环境自检 ────────────────────────────────────────────────
python - <<'PYEOF' || { echo "✗ 环境自检失败：请先 conda activate rk3588 并安装 lerobot[dataset,training]"; exit 1; }
import sys
try:
    import torch, lerobot
except ImportError as e:
    print('✗ 缺依赖:', e); sys.exit(1)
print(f"  python {sys.version.split()[0]} | lerobot {lerobot.__version__} | torch {torch.__version__}")
if torch.cuda.is_available():
    print(f"  CUDA 可用: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
elif 'cuda' in sys.argv:
    print('  ⚠ CUDA 不可用，将回退 CPU（很慢）')
PYEOF

# ── ② 转换数据集 ──────────────────────────────────────────────
if [ -f "$DS_DIR/meta/info.json" ] && [ "$FORCE" != "1" ]; then
  echo "② 数据集已存在，跳过转换（FORCE=1 可强制重建）: $DS_DIR"
else
  echo "② 转换: 原始 JSON → LeRobotDataset"
  n_json=$(find "$RAW_DIR" -maxdepth 1 -name 'episode_*.json' 2>/dev/null | wc -l)
  if [ "$n_json" -eq 0 ]; then
    echo "✗ $RAW_DIR 下没有 episode_*.json（先用 tools/collect_episodes.py 采集）"; exit 1
  fi
  echo "   找到 $n_json 条 episode"
  python "$REPO_DIR/scripts/json_to_lerobot.py" \
      --input-dir "$RAW_DIR" --format lerobot \
      --repo-id "local/$TASK" --task "$TASK" --root "$DS_DIR" \
      --summary "$DS_DIR/summary.json"
fi

# ── ③ 训练 ────────────────────────────────────────────────────
echo "③ 训练 ACT（device=$DEVICE）"
if [ "$DEVICE" = "cuda" ]; then DEV_ARG=cuda; else DEV_ARG=cpu; fi
mkdir -p "$OUT_DIR"
lerobot-train \
    --dataset.repo_id="local/$TASK" --dataset.root="$DS_DIR" \
    --policy.type=act --policy.device="$DEV_ARG" --policy.push_to_hub=false \
    --batch_size="$BATCH" --num_workers=4 \
    --steps="$STEPS" --save_freq=5000 --log_freq=100 \
    --output_dir="$OUT_DIR" --job_name="$TASK" --wandb.enable=false

CKPT="$OUT_DIR/checkpoints/last/pretrained_model"
echo
echo "=============================================================="
echo "✓ 训练完成。checkpoint: $CKPT"
echo "  下一步导出 ONNX（板端推理用，opset 用默认 18）:"
echo "    python tools/export_act_onnx.py --checkpoint $CKPT \\"
echo "        --output_dir models/act/"
echo "=============================================================="
