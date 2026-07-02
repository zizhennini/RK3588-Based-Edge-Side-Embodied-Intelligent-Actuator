# 预训练视觉抓取模型资源

LeRobot 官方在 HuggingFace 上提供了大量可直接下载的预训练模型，无需训练即可用于机械臂视觉控制。

## 模型列表

### 轻量级（适合 RK3588 评估）

| 模型 | 用途 | 参数量 | 下载量 | 链接 |
|------|------|--------|-------|------|
| `lerobot/smolvla_base` | 视觉-语言-动作基础模型 | 轻量 | 50k+ | https://huggingface.co/lerobot/smolvla_base |
| `lerobot/diffusion_pusht` | Diffusion Policy 推/抓 | CNN级别 | 2.2k | https://huggingface.co/lerobot/diffusion_pusht |
| `lerobot/act_aloha_sim_transfer_cube_human` | ACT 抓取/放置 | ~80M | 1.7k | https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human |
| `lerobot/resnet10` | 极轻量视觉特征提取器 | ~5M | 8.2k | https://huggingface.co/lerobot/resnet10 |
| `lerobot/smolvla_libero` | LIBERO 场景抓取微调 | 轻量 | 8.7k | https://huggingface.co/lerobot/smolvla_libero |

### 中等/大型 VLA 模型（需 GPU，仅供参考）

| 模型 | 用途 | 参数量 | 链接 |
|------|------|--------|------|
| `lerobot/pi0fast-base` | Pi0Fast 基础 VLA | 3B | https://huggingface.co/lerobot/pi0fast-base |
| `lerobot/pi05_base` | Pi0.5 基础 VLA | 3B+ | https://huggingface.co/lerobot/pi05_base |
| `lerobot/pi0_base` | Pi0 基础 VLA | 3B+ | https://huggingface.co/lerobot/pi0_base |

### 社区模型

| 模型 | 用途 | 参数量 | 链接 |
|------|------|--------|------|
| `qualia-robotics/act-makermods-pick-book` | ACT 抓取书本 | 51.7M | https://huggingface.co/qualia-robotics/act-makermods-pick-book-parallel-grasp-2a8bc186 |

## 使用方式

```bash
pip install lerobot
```

```python
# 以 diffusion_pusht 为例
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")

# 推理
obs = {"top_camera": rgb_image, "state": joint_positions}
action = policy.select_action(obs)
```

## 完整模型列表

所有 LeRobot 预训练模型：https://huggingface.co/lerobot
