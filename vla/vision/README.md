# vla/vision — 视觉定位（遗留模块）

颜色定位 + PCA 抓取估计 + 逆透视变换。

> 注: 原 `detector.py`（MobileNetSSD）已于 v0.5.0 移除 — 其模型文件在 v0.3.0
> 清理时删除，检测能力由 `perception/grasp_detect.py`（GGCNN）与
> `perception/vlm.py`（VLM）承接。

## 文件

| 文件 | 说明 |
|------|------|
| `locator.py` | `ColorLocator` — HSV 颜色分割 + Depth 反投影 |
| `pca_grasp.py` | `PCAGrasper` — PCA 主轴抓取角度估计 |
| `ipm.py` | `IPM` — 逆透视变换（上帝视角映射） |

## 定位流程

```
VLM 输出 "红色杯子"
         ↓
ColorLocator (CPU) ──► HSV 颜色分割
         ↓
匹配 → (cx, cy) → Depth 反投影 → 3D 坐标
```

## 使用

```python
from vla.vision import ColorLocator

locator = ColorLocator(camera_matrix)
pos = locator.locate(rgb, depth, "红色")
```
