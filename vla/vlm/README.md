# vla/vlm — 通用 VLM 框架

工厂模式，支持多模型一键切换。含按需加载/闲置卸载机制。

## 文件

| 文件 | 说明 |
|------|------|
| `base.py` | `VLMBase` 抽象基类 + `VLMResult` 数据类 |
| `qwen3_vl.py` | Qwen3.5-0.8B 引擎（子进程 demo） |
| `smart_vlm.py` | 按需加载 + 闲置自动卸载封装 |
| `factory.py` | 工厂函数 `create_vlm()` |

## 使用

```python
from vla.vlm import create_vlm
from vla.vlm.smart_vlm import SmartVLM, IdleUnloader

# 带闲置卸载的 SmartVLM
vlm = SmartVLM("qwen3-vl-2b", model_path, demo_bin=demo_bin, idle_timeout=30)
vlm.ensure_loaded()          # 加载模型
result = vlm.infer(path)     # 推理（自动 ensure_loaded）
vlm.unload()                 # 卸载（释放 ~900MB）

# 后台自动卸载
unloader = IdleUnloader(vlm, interval=5.0)
unloader.start()
```

## 支持模型

| 模型 | 配置名 | 引擎 | 模型文件 |
|------|--------|------|---------|
| Qwen3.5-0.8B | `qwen3.5-0.8b` | `Qwen3VLEngine` | .rknn + .rkllm |
| Qwen3-VL-2B | `qwen3-vl-2b` | `Qwen3VLEngine` | .rknn + .rkllm |
| SmolVLM-256M | — | 实验性 | ONNX |
| SmolVLM-500M | — | 实验性 | ONNX |
