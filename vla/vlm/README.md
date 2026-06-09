# vla/vlm — 通用 VLM 框架

工厂模式，支持多模型一键切换。

## 文件

| 文件 | 说明 |
|------|------|
| `base.py` | `VLMBase` 抽象基类 + `VLMResult` 数据类 |
| `internvl2.py` | InternVL2-1B (rkllm_demo) |
| `qwen_vl.py` | Qwen2.5-VL-3B (rkllm_demo) |
| `qwen3_vl.py` | Qwen3-VL-2B (多模态 demo，需要 .rknn+.rkllm) |
| `factory.py` | 工厂函数 `create_vlm()` |

## 使用

```python
from vla.vlm import create_vlm

# Qwen3-VL-2B（推荐）
vlm = create_vlm("qwen3-vl-2b", demo_bin="/usr/bin/demo")
vlm.load("./models/Qwen3-VL-2B")
result = vlm.infer("/tmp/frame.jpg")
print(result.color, result.object)

# InternVL2-1B
vlm = create_vlm("internvl2-1b", rkllm_bin="rkllm_demo")
vlm.load("./models/InternVL2-1B-rkllm/model.rkllm")
```

## 支持模型

| 模型 | 配置名 | 引擎 | 模型文件 |
|------|--------|------|---------|
| Qwen3-VL-2B | `qwen3-vl-2b` | `Qwen3VLEngine` | .rknn + .rkllm（目录） |
| InternVL2-1B | `internvl2-1b` | `InternVL2Engine` | .rkllm（文件路径） |
| Qwen2.5-VL-3B | `qwen2.5-vl-3b` | `QwenVLEngine` | .rkllm（文件路径） |

## 添加新模型

1. 继承 `VLMBase`，实现 `load()` `infer()` `unload()`
2. 在 `factory.py` 的 `VLM_REGISTRY` 中注册
