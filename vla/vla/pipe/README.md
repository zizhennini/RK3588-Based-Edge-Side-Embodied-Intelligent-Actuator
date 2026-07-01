# vla/pipe — 主流水线编排

有限状态机 (FSM) 编排完整 VLA 抓取流程。

## 文件

| 文件 | 说明 |
|------|------|
| `pipeline.py` | `VLApipeline` — FSM: VLM_INFER → LOCATE → GRASP → PLACE → DONE |

## 状态机

```
IDLE ──start()──▶ VLM_INFER ──▶ LOCATE ──▶ GRASP ──▶ PLACE ──▶ DONE
                                              │
                                              └── locate_failed ──▶ DONE
```

## 使用

```python
from vla.pipe import VLApipeline

pipe = VLApipeline(arm, vlm, locator)
pipe.start()
status = pipe.step(rgb, depth)  # 每次调用推进一帧
```
