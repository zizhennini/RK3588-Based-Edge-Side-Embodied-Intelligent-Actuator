# vla/vision — 视觉定位

双检测引擎：**MobileNet SSD (CPU) + ColorLocator 降级**。

## 文件

| 文件 | 说明 |
|------|------|
| `detector.py` | `MobileNetSSD` — CPU 目标检测，支持 COCO 20 类 |
| `locator.py` | `ColorLocator` — HSV 颜色分割 + Depth 反投影（降级用） |

## 检测流程

```
VLM 输出 "红色杯子"
         ↓
MobileNet SSD (CPU) ──► 检测 "bottle" 类别
         ↓
         检测框内 HSV 验证是否为红色
         ↓
         匹配 → (cx, cy) → Depth 反投影 → 3D 坐标
         ↓  不匹配
ColorLocator (CPU) ──► 降级为纯颜色分割
```

## 使用

```python
from vla.vision import MobileNetSSD, ColorLocator

detector = MobileNetSSD("./models/MobileNetSSD_deploy.prototxt",
                        "./models/MobileNetSSD_deploy.caffemodel")
results = detector.detect(rgb, target_class="bottle")

locator = ColorLocator(camera_matrix)
pos = locator.locate(rgb, depth, "红色")
```

## 模型下载

```bash
python scripts/download_mobilenet.py
```

## COCO 20 类

background, aeroplane, bicycle, bird, boat, **bottle**, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
