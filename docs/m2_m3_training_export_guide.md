# M2/M3 训练与导出链路（已验证）

> 状态：**端到端已跑通并实测**（2026-10-05，合成数据冒烟）。真实数据采集完成后按本文档执行即可。
> PC 侧环境：**WSL2 `rk3588` conda env**（Python 3.12.14 + lerobot 0.6.1 + torch 2.11.0）。
> 板端：onnxruntime 1.23.2（CPU）。

## 0. 一句话流程

```
板端遥操作采集(data/raw/<task>/episode_XXXX.json + _images/)        [M1]
  → PC 转换 scripts/json_to_lerobot.py --format lerobot             [M2 入口]
  → PC 训练 lerobot-train --policy.type=act                         [M2]
  → PC 导出 tools/export_act_onnx.py（拆 vision_encoder + transformer）[M3]
  → 板端 policy/act_policy.py 加载推理（ORT CPU）                     [M3]
```

## 1. 环境准备（一次性）

```bash
wsl -u shimuzi      # 进入 WSL2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rk3588
```

**关键：lerobot 0.6.1 的数据集/训练栈是可选依赖，缺了会在运行时才报错**（`ImportError: 'datasets' is required…` /
`'accelerate' is required…`）：

```bash
pip install "lerobot[dataset]"     # datasets / pyarrow / av / torchcodec
pip install "lerobot[training]"    # accelerate / wandb / einops / torchvision
```

已实测版本：`pyarrow 25.0.1`、`av 15.1.0`、`datasets 4.8.5`、`torchcodec 0.11.1`、
`accelerate 1.15.0`、`torchvision 0.26.0`（numpy 保持 2.2.6 未被改动）。

> ⚠️ 该 env 的 torch 是 **CPU 版**（2.11.0+cpu）。冒烟训练 CPU 约 1.6~1.8 s/step 可用；
> 真正训练（数万步）需换 CUDA 版 torch —— 见 §5 待办。

## 2. 数据转换（M2 入口）

```bash
cd ~/work/rk3588-eia
python scripts/json_to_lerobot.py --input-dir ~/data/raw/<task> --format lerobot \
    --repo-id local/<task> --task "拿起方块" --root ~/datasets/<task> \
    --summary ~/datasets/<task>/summary.json
```

**特征语义（imitation learning 约定，勿弄反）**：

| 输出特征 | 来源 | 含义 |
|---|---|---|
| `action` | `J1..J6` | 主臂指令 / 下发给从臂的目标（策略输出） |
| `observation.state` | `F1..F6` | 从臂实际角度（机器人本体状态，策略输入） |
| `observation.images.<cam>` | `<episode>_images/<cam>/*.jpg` | `dtype=video`，SVT-AV1 编码 |

- ⚠️ **缺 `action` 会训练崩溃**：`modeling_act.py` 里 `config.action_feature.shape` → `NoneType`。
  官方由数据集特征键推断 `input/output_features`（`action` → `FeatureType.ACTION`）。
- 缺 `F*` 的历史数据会退回用 `J*` 当 state 并打印告警（可训练但状态含目标值，精度受影响）。
- 产出为标准 **LeRobotDataset v3.0**：`meta/info.json`、`data/chunk-*/file-*.parquet`、
  `videos/observation.images.*/chunk-*/*.mp4`、`meta/{stats,tasks}.*`。

## 3. 训练（M2）

```bash
lerobot-train \
  --dataset.repo_id=local/<task> --dataset.root=~/datasets/<task> \
  --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --batch_size=8 --num_workers=4 --steps=50000 --save_freq=5000 \
  --output_dir=~/train/<task> --job_name=<task> --wandb.enable=false
```

要点：
- **`--policy.push_to_hub=false` 必须显式给**：0.6.1 默认 `push_to_hub=true`，缺 `repo_id` 会在
  `configs/train.py::validate()` 报 `'repo_id' argument missing`（顶层 `--repo_id` 不是合法参数）。
- 冒烟实测（CPU，batch 2，3 集 120 帧）：`num_learnable_params=51,597,190 (52M)`，
  `loss 85.98 → 27.54`（含 `l1_loss` + `kld_loss`，VAE 生效），约 1.6~1.8 s/step。
- 输出 `checkpoints/last/pretrained_model/`：`config.json`、`model.safetensors`、
  `policy_preprocessor*.json/safetensors`（**归一化统计量在这里**）、`policy_postprocessor*`、`train_config.json`。

## 4. 导出 ONNX（M3）

```bash
python tools/export_act_onnx.py \
    --checkpoint ~/train/<task>/checkpoints/last/pretrained_model \
    --output_dir models/act/
```

> **opset 用默认的 18**：torch>=2.9 的导出器对 ACT 图（含 `LayerNormalization`）**无法降到 14**，
> 请求 14 会打印 `Please consider setting opset_version >=18` 并保留 18。
> 板端 onnxruntime 1.23.2 支持 ≥21，故 18 可直接用。`act_config.json` 同时记录
> `onnx_opset`(实际) 与 `onnx_opset_requested`(请求)。

产出：`vision_encoder.onnx(+.data)`、`transformer.onnx(+.data)`、`query_embed.npy`、`act_config.json`。

**已验证的契约与数值等价**（冒烟模型，PC 侧 torch vs 两段 ONNX）：

| 项 | 值 |
|---|---|
| 分模块数值校验 | vision `max_diff < 1e-4`；transformer `max_diff < 1e-4`；端到端 `max_diff=5.2e-07` |
| 板端（RK3588/ORT 1.23.2）与 PC 参考 | vision `max|Δ|=5.7e-06`；actions `max|Δ|=6.6e-07` |
| RK3588 延迟（10 次中位） | vision **364 ms** + transformer **152 ms** = 端到端 **619 ms（1.62 Hz）** |
| 可行性 | chunk=100 步按 30 Hz 执行需 3.33 s ≫ 0.619 s，约 **5 倍余量** |

**两个容易踩的坑（已在本仓库修复）**：

1. **`norm_stats` 必须从 0.6.1 的 processor safetensors 提取**。
   0.6.1 把归一化搬到 pre/postprocessor，统计量在
   `policy_preprocessor_step_N_normalizer_processor.safetensors`（键形如 `observation.state.mean`）。
   旧版 json 路径取不到 → `norm_stats: null` → 板端无法归一化。
2. **ONNX 实际 opset 不是请求值**：torch 2.x 因 `LayerNormalization` 等算子无法降到 14，
   `--opset 14` 会"降级失败但保持原版本"，实测产物为 **opset 18**。
   `act_config.json` 现在同时记录 `onnx_opset`（实际）与 `onnx_opset_requested`（请求）。
   ORT 1.23.2 支持 ≥21，故 18 可用。

## 5. 板端推理（M3）

```bash
# 把 onnx 目录整体拷到板端后
python3 -c "
from policy.act_policy import ACTPolicy"   # 或经 runtime 装配
```

`policy/act_policy.py` 从 `act_config.json` 读取：`vision_encoder/transformer/query_embed` 路径、
`chunk_size`/`n_action_steps`、`norm_stats`、`image_mean/image_std`；`predict(obs)` 在 action 缓冲
耗尽时才触发一次完整推理（官方 ACT 的 action queue 语义），可选 `temporal_ensemble`。

**归一化必须与训练侧一致**（官方 `normalizer_processor` 语义）：

```
state :  (s - mean) / (std + 1e-8)
image :  (x/255 - image_mean) / image_std      # use_imagenet_stats → ImageNet 常量
action:  a * (std + 1e-8) + mean               # 反归一化
```

> ⚠️ 导出的 `vision_encoder.onnx` 内部**不含**图像归一化（backbone 从 `conv1` 开始），
> 因此板端**必须**先做 `(x/255 - image_mean)/image_std`。漏掉会静默喂错尺度、动作全错。
> 板端已做健壮化：缺顶层 `image_mean/std` 时从 `norm_stats.images` 推导，两者都没有则**显式告警**。

## 6. 待办

- [ ] **真数据**：等相机支架到位 → 采集 ≥50 集 → 按 §2~§4 全流程跑一遍
- [ ] **CUDA torch**：`rk3588` env 目前是 CPU 版；真实训练前装 CUDA 版
      （或新建 `act` env：py3.12 + torch(cu) + `lerobot[dataset,training]`）；GPU 为 RTX 4060 Laptop 8GB
- [ ] **板端 RKNN/NPU 加速**（可选）：ORT CPU 端到端 619 ms 已够用（5 倍余量）；
      若要提速再评估 fp16 RKNN（官方 bundle 实测 ~470ms 推理，同为 CPU 路径）
- [ ] 训练超参：官方 ACT 参考 `steps 500000 / batch 16 / lr 1e-5 / save_freq 10000`（见调研文档）
