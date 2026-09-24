"""GGCNN 实时抓取检测 -- ONNX Runtime 推理

模型: dougsm/ggcnn (Cornell, 300x300 输入, ~62K 参数)
输入: (1, 1, 300, 300) float32 单通道深度图
输出: pos(质量) / cos / sin / width 四张 300x300 map

依赖: numpy, opencv, onnxruntime, scipy (skimage 可选)
"""
import logging
import numpy as np
import cv2
from typing import Optional

from hardware.interfaces import PerceptionModule, Observation

logger = logging.getLogger(__name__)

# ---- 可选依赖: scipy (高斯平滑) / skimage (峰值检测) ----
try:
    from scipy.ndimage import gaussian_filter, maximum_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from skimage.feature import peak_local_max as _sk_peak_local_max
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def _gaussian_smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """高斯平滑, scipy 缺失时退化为 cv2.GaussianBlur"""
    if _HAS_SCIPY:
        return gaussian_filter(img, sigma)
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _peak_local_max_np(img: np.ndarray, min_distance: int,
                       threshold_abs: float, num_peaks: int) -> np.ndarray:
    """纯 numpy/scipy 版 peak_local_max (贪心 NMS), skimage 不可用时的替代

    Returns: (N, 2) int 数组, 每行为 (row, col), 按响应值降序
    """
    flat = img.copy()
    # 局部最大值: 与 (2*min_distance+1) 邻域最大值比较
    if _HAS_SCIPY:
        size = 2 * min_distance + 1
        local_max = maximum_filter(flat, size=size, mode="constant")
        candidates = (flat == local_max) & (flat >= threshold_abs)
    else:
        # 无 scipy: cv2.dilate 等效局部最大值
        k = np.ones((2 * min_distance + 1, 2 * min_distance + 1), np.uint8)
        local_max = cv2.dilate(flat, k)
        candidates = (flat == local_max) & (flat >= threshold_abs)

    rows, cols = np.nonzero(candidates)
    if len(rows) == 0:
        return np.zeros((0, 2), dtype=int)

    values = flat[rows, cols]
    order = np.argsort(-values)  # 响应降序
    rows, cols, values = rows[order], cols[order], values[order]

    # 贪心非极大值抑制: 抑制已选峰值 min_distance 半径内的候选
    kept: list = []
    for r, c, v in zip(rows, cols, values):
        if all((r - kr) ** 2 + (c - kc) ** 2 > min_distance ** 2
               for kr, kc in kept):
            kept.append((r, c))
            if len(kept) >= num_peaks:
                break
    return np.array(kept, dtype=int) if kept else np.zeros((0, 2), dtype=int)


def _find_peaks(img: np.ndarray, min_distance: int,
                threshold_abs: float, num_peaks: int) -> np.ndarray:
    """峰值检测统一入口, 优先 skimage"""
    if _HAS_SKIMAGE:
        return _sk_peak_local_max(
            img, min_distance=min_distance,
            threshold_abs=threshold_abs, num_peaks=num_peaks)
    return _peak_local_max_np(img, min_distance, threshold_abs, num_peaks)


class GGCNNDetector(PerceptionModule):
    """GGCNN 抓取检测器 -- ONNX Runtime CPU 推理"""

    INPUT_SIZE = 300          # 网络输入边长
    WIDTH_SCALE = 150.0       # width 输出 -> 300x300 像素空间宽度的换算系数

    def __init__(self):
        self._session = None
        self._input_name = "depth"
        self._config: dict = {}

    # ---------------- Module 生命周期 ----------------
    def setup(self, config: dict) -> None:
        """
        config keys:
            model_path: str       # ONNX 模型路径 (默认 models/ggcnn/ggcnn_cornell_300.onnx)
            q_threshold: float    # 抓取质量阈值 (默认 0.3)
            max_grasps: int       # 最大返回抓取数 (默认 3)
            camera_matrix: np.ndarray  # 相机内参 3x3, 用于像素->物理换算
            inpaint: bool         # 是否 inpaint 填补缺失深度 (默认 True)
        """
        import onnxruntime as ort

        model_path = config.get("model_path", "models/ggcnn/ggcnn_cornell_300.onnx")
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._config = config
        logger.info(f"GGCNN 已加载: {model_path} (input={self._input_name})")

    def start(self) -> None:
        pass  # ONNX session 无后台线程

    def stop(self) -> None:
        self._session = None

    @property
    def is_available(self) -> bool:
        return self._session is not None

    def on_failure(self) -> str:
        return "fallback"  # 降级到 PCA 抓取

    # ---------------- 检测主流程 ----------------
    def detect(self, obs: Observation, roi_bbox: Optional[tuple] = None) -> dict:
        """检测最佳抓取位姿

        Args:
            obs: Observation (depth 480x640 float32, 单位米)
            roi_bbox: 可选 (x1, y1, x2, y2) 原图绝对像素坐标, 限制检测区域

        Returns:
            {"grasps": [{"center_px": (u, v),   # 原图像素坐标
                         "angle_rad": float,     # 抓取角度 [-π/2, π/2]
                         "width_m": float,       # 夹爪开度 (米)
                         "quality": float}, ...]}  # 按质量降序
        """
        empty = {"grasps": []}
        if not self.is_available:
            logger.warning("GGCNN 未初始化, 跳过检测")
            return empty

        depth = np.asarray(obs.depth, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            logger.warning(f"深度图形状异常: {depth.shape}")
            return empty
        orig_h, orig_w = depth.shape

        # 单位启发式: 中位有效深度 > 100 认为单位是毫米, 转米
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size == 0:
            logger.warning("深度图无有效像素")
            return empty
        if np.median(valid) > 100.0:
            depth = depth / 1000.0

        # 1. ROI 裁剪
        roi, rx1, ry1 = self._crop_roi(depth, roi_bbox)
        roi_h, roi_w = roi.shape

        # 2. 预处理: inpaint -> 归一化 -> resize(300,300)
        net_in = self._preprocess(roi)

        # 3. 推理
        pos, cos, sin, width = self._session.run(
            None, {self._input_name: net_in})
        pos = np.squeeze(pos)
        cos = np.squeeze(cos)
        sin = np.squeeze(sin)
        width = np.squeeze(width)

        # 4. 后处理: 角度/宽度还原 + 高斯平滑 + 峰值提取
        ang = np.arctan2(sin, cos) / 2.0          # [-π/2, π/2]
        width_px = width * self.WIDTH_SCALE       # 300x300 像素空间宽度
        q_img = _gaussian_smooth(pos, sigma=2)
        ang_img = _gaussian_smooth(ang, sigma=2)
        width_img = _gaussian_smooth(width_px, sigma=1)

        q_threshold = float(self._config.get("q_threshold", 0.3))
        max_grasps = int(self._config.get("max_grasps", 3))
        peaks = _find_peaks(q_img, min_distance=20,
                            threshold_abs=q_threshold,
                            num_peaks=max_grasps)
        if len(peaks) == 0:
            logger.debug("GGCNN 未找到满足阈值的抓取候选")
            return empty

        # 5. 峰值坐标反映射到原图 + 像素->物理宽度换算
        fx = self._get_fx()
        scale_x = roi_w / self.INPUT_SIZE   # 300 空间 -> ROI 空间
        scale_y = roi_h / self.INPUT_SIZE
        orig_scale_x = orig_w / self.INPUT_SIZE
        grasps = []
        for row, col in peaks:
            # 原图像素坐标 (u, v)
            u = float(col * scale_x + rx1)
            v = float(row * scale_y + ry1)
            u = float(np.clip(u, 0, orig_w - 1))
            v = float(np.clip(v, 0, orig_h - 1))
            # 抓取点深度 (5x5 邻域有效中值, 抗噪)
            z = self._sample_depth(depth, int(round(v)), int(round(u)))
            # 宽度: 300 空间像素 -> 原图像素 -> 米
            w_orig_px = float(width_img[row, col]) * orig_scale_x
            width_m = w_orig_px * z / fx if fx > 0 else 0.0
            grasps.append({
                "center_px": (u, v),
                "angle_rad": float(ang_img[row, col]),
                "width_m": float(width_m),
                "quality": float(q_img[row, col]),
            })

        grasps.sort(key=lambda g: -g["quality"])
        logger.debug(f"GGCNN 检出 {len(grasps)} 个抓取候选, "
                     f"最优 q={grasps[0]['quality']:.2f} @ {grasps[0]['center_px']}")
        return {"grasps": grasps[:max_grasps]}

    # ---------------- 内部工具 ----------------
    def _crop_roi(self, depth: np.ndarray,
                  roi_bbox: Optional[tuple]) -> tuple[np.ndarray, int, int]:
        """裁剪 ROI; bbox 缺失/无效时退化为图像中心方形区域

        Returns: (roi, rx1, ry1) -- rx1/ry1 为 ROI 左上角在原图的坐标
        """
        h, w = depth.shape
        if roi_bbox is not None:
            x1, y1, x2, y2 = (int(round(c)) for c in roi_bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 >= 20 and y2 - y1 >= 20:
                return depth[y1:y2, x1:x2].copy(), x1, y1
            logger.warning(f"roi_bbox 无效或过小 {roi_bbox}, 退化为中心裁剪")
        # 中心正方形裁剪 (边长取短边, 上限 300)
        side = min(h, w, self.INPUT_SIZE)
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        return depth[y1:y1 + side, x1:x1 + side].copy(), x1, y1

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """inpaint -> 归一化 clip(x - mean, -1, 1) -> resize(300,300) -> NCHW"""
        img = roi.copy()
        # 无效深度 (0 / NaN / inf) -> inpaint 填补
        invalid = ~np.isfinite(img) | (img <= 0)
        if self._config.get("inpaint", True) and 0 < invalid.sum() < invalid.size:
            mask = (invalid * 255).astype(np.uint8)
            fill = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img = cv2.inpaint(fill, mask, 5, cv2.INPAINT_TELEA)
        img[invalid] = 0.0
        # 归一化: 减均值后裁剪到 ±1 (GGCNN 官方预处理, 非除 255)
        img = np.clip(img - img.mean(), -1.0, 1.0).astype(np.float32)
        # resize 到网络输入
        img = cv2.resize(img, (self.INPUT_SIZE, self.INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        return img.reshape(1, 1, self.INPUT_SIZE, self.INPUT_SIZE)

    def _get_fx(self) -> float:
        """从 config 的 camera_matrix 取 fx, 缺失时返回 D435i 出厂标定值"""
        cam = self._config.get("camera_matrix")
        if cam is not None:
            try:
                return float(np.asarray(cam)[0, 0])
            except (IndexError, ValueError, TypeError):
                logger.warning("camera_matrix 格式异常, 使用默认 fx=604.23")
        return 604.2294  # config/settings.py CAMERA_MATRIX 默认值

    @staticmethod
    def _sample_depth(depth: np.ndarray, v: int, u: int) -> float:
        """取 (v,u) 附近 5x5 邻域内有效深度的中值; 全无效时返回 0.3m 兜底"""
        h, w = depth.shape
        y1, y2 = max(0, v - 2), min(h, v + 3)
        x1, x2 = max(0, u - 2), min(w, u + 3)
        patch = depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        return float(np.median(valid)) if valid.size > 0 else 0.3
