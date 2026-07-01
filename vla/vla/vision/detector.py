"""MobileNet SSD 目标检测器 — CPU 推理，无 NPU 占用"""
import cv2
import numpy as np

COCO_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]


class MobileNetSSD:
    """基于 OpenCV DNN 的 MobileNet SSD 目标检测"""

    def __init__(
        self,
        prototxt: str = "./models/MobileNetSSD_deploy.prototxt",
        caffemodel: str = "./models/MobileNetSSD_deploy.caffemodel",
        confidence_threshold: float = 0.5,
    ):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.confidence_threshold = confidence_threshold

    def detect(self, rgb: np.ndarray, target_class: str | None = None) -> list[dict]:
        """检测目标，返回 [{"label": "bottle", "confidence": 0.85, "box": (x1,y1,x2,y2), "cx": int, "cy": int}, ...]"""
        h, w = rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(rgb, 0.007843, (300, 300), (127.5, 127.5, 127.5))
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        target_id = -1
        if target_class and target_class in COCO_CLASSES:
            target_id = COCO_CLASSES.index(target_class)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue
            class_id = int(detections[0, 0, i, 1])
            if target_id != -1 and class_id != target_id:
                continue
            label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            results.append({
                "label": label,
                "confidence": float(confidence),
                "box": (x1, y1, x2, y2),
                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,
            })
        return results

    def detect_and_match_color(
        self, rgb: np.ndarray, depth: np.ndarray,
        target_class: str, target_color: str | None,
        color_locator,
    ) -> dict | None:
        """检测目标类别，并按颜色筛选"""
        dets = self.detect(rgb, target_class=target_class if target_color is None else None)
        if not dets:
            return None

        if target_color is None:
            best = max(dets, key=lambda d: d["confidence"])
            return self._to_3d(best, depth)

        # 按颜色筛选：在检测框内取 HSV 中位数验证
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        for det in sorted(dets, key=lambda d: d["confidence"], reverse=True):
            x1, y1, x2, y2 = det["box"]
            roi = hsv[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            median_hue = np.median(roi[:, :, 0])
            lower, upper = color_locator.COLOR_RANGES.get(target_color, [(0, 0, 0), (180, 255, 255)])
            if lower[0] <= median_hue <= upper[0]:
                return self._to_3d(det, depth)

        # 颜色没匹配上，返回置信度最高的
        best = max(dets, key=lambda d: d["confidence"])
        return self._to_3d(best, depth)

    def _to_3d(self, det: dict, depth: np.ndarray):
        z = float(depth[det["cy"], det["cx"]])
        return {**det, "z": z} if z > 0 and not np.isnan(z) else None
