# perception/cube_locator.py — 桌面立方体精定位（HSV 分割 + 连通域 + 最小外接矩形）
"""P0-b：传统 CV 精定位，输出「抓取点像素 + 抓取朝向」+ 多候选（供 Set-of-Mark）

为什么不用现成的 ``perception/locator.py::ColorLocator``
-------------------------------------------------------
本模块的阈值全部来自**板端真实图像实测**（`data/raw/pick_place` 前 12 个 episode
× 6 帧，72 张），而不是沿用旧模块的经验值。实测发现两处旧实现会踩的坑：

1. **白色机械臂会被判成"彩色"**
   D435i 自动白平衡把白色塑料染成淡蓝：实测机械臂区域中位 RGB=(96,181,228)
   → S=149, V=228, **H=101**。而蓝色方块是 H=103 —— **色相几乎重合，无法靠 H 区分**。
   旧实现只按 H 找颜色，会把机械臂像素归到蓝色里。
   本模块用「饱和度 + 形状 + 底部连通」三重条件剔除，见 ``_reject_arm_like``。

2. **方块饱和度的跨度远超直觉**
   实测各方块中位饱和度：蓝 253 / 绿 250 / 橙 166 / 红 145 / 紫 135 / **黄 114**。
   黄色方块比机械臂(S=149)还低 → 单靠 S 阈值也不行，必须联合形状判据。

实测尺寸（72 张图）：方块面积中位 **1096 px**、w≈39 h≈38（不是想象里的 20 px）。
故 ``MIN_AREA`` 取 200、``MAX_AREA`` 取 5000，可容纳远近与遮挡变化。

⚠️ **正方形的朝向本质上是模糊的**（旋转 90° 后图形不变），且 **PCA 主轴在 45° 附近
退化**（两个特征值相等 → 主轴方向无定义）。实测对 45° 附近的方块 PCA 角度抖动可达
±20°，故本模块朝向**以 minAreaRect 为准**，PCA 只作交叉校验（见 P0-b 报告）。

坐标与颜色约定
--------------
- 输入 **RGB**（与 ``hardware/interfaces.Observation.rgb`` 一致），不是 BGR
- OpenCV H∈[0,180)，乘 2 得角度制
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 实测标定的颜色色相窗口（OpenCV H ∈ [0,180)）
# 依据：真实图像里各方块分量的中位 H —— 红 178 / 橙 11 / 黄 42 / 绿 86 / 蓝 103 / 紫 120
# ---------------------------------------------------------------------------
COLOR_HUE_WINDOWS: Dict[str, List[Tuple[int, int]]] = {
    "red":    [(165, 180), (0, 8)],
    "orange": [(9, 20)],
    "yellow": [(21, 55)],
    "green":  [(56, 95)],
    "blue":   [(96, 115)],
    "purple": [(116, 148)],
}

#: 中文名（与 VLM prompt / 日志统一）
COLOR_ZH: Dict[str, str] = {
    "red": "红色", "orange": "橙色", "yellow": "黄色",
    "green": "绿色", "blue": "蓝色", "purple": "紫色",
}
ZH_TO_COLOR: Dict[str, str] = {v: k for k, v in COLOR_ZH.items()}

#: 板端真实图像实测的各方块中位 HSV（72 张图统计，供合成图/测试复现真实分布）
#: 注意饱和度跨度极大：蓝 253 → 黄 114，比被白平衡染蓝的机械臂（149）还低，
#: 所以单靠 S 阈值也分不开方块与机械臂。
MEASURED_HSV_REF: Dict[str, Tuple[int, int, int]] = {
    "red": (178, 145, 146), "orange": (11, 166, 168), "yellow": (42, 114, 195),
    "green": (86, 250, 141), "blue": (103, 253, 228), "purple": (120, 135, 194),
}

#: 彩色像素门限（实测：方块 S 114~253、V 141~228；木桌 S 低）
SAT_MIN = 90
VAL_MIN = 70

#: 连通域筛选（实测方块面积中位 1096 px）
MIN_AREA = 200
MAX_AREA = 5000
#: 面积 / **拟合矩形**面积。注意不能用「面积/轴对齐bbox面积」当形状判据：
#: 正方形旋转 θ 后轴对齐 bbox 边长 = s(cosθ+sinθ)，该比值 = 1/(cosθ+sinθ)²，
#: **在 45° 恰好等于 0.5** → 任何 >0.5 的门限都会把 45° 的方块整批误杀
#: （实测漏检 65/432 全部集中在 30°~60°）。用拟合矩形则恒 ≈1，与旋转无关。
RECT_FILL_MIN = 0.75
#: 拟合矩形的长短边比（正方形应接近 1）；用来排除细长形状
RECT_ASPECT_MAX = 1.60

#: 机械臂误检的第二道防线：淡蓝白（H∈[88,118] 且 S 偏低 且 很亮）
ARM_HUE = (88, 118)
ARM_SAT_MAX = 200
ARM_VAL_MIN = 195

#: 机械臂从画面底边进入 → 触底的分量一律丢弃（本场景方块不会压到画面底边）
REJECT_BOTTOM_MARGIN = 2


@dataclass
class CubeCandidate:
    """一个候选方块（抓取点 + 朝向）"""
    color: str                     # 英文色名，见 COLOR_HUE_WINDOWS
    center: Tuple[float, float]    # (cx, cy) 像素，**质心**（正方形对称中心的无偏估计）
    angle_deg: float               # 朝向，归一化到 [0,90)（正方形 90° 周期）
    area: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) 轴对齐外接框
    extent: float                  # 面积 / 拟合矩形面积（≈1 表示实心矩形，与旋转无关）
    side_px: float                 # 等效边长 ≈ sqrt(area)
    median_hsv: Tuple[int, int, int] = (0, 0, 0)
    extra: Dict = field(default_factory=dict)

    @property
    def color_zh(self) -> str:
        return COLOR_ZH.get(self.color, self.color)

    def as_dict(self) -> dict:
        return dict(color=self.color, color_zh=self.color_zh,
                    center=[round(self.center[0], 2), round(self.center[1], 2)],
                    angle_deg=round(self.angle_deg, 2), area=self.area,
                    bbox=list(self.bbox), extent=round(self.extent, 3),
                    side_px=round(self.side_px, 2))


# ---------------------------------------------------------------------------
# 掩码
# ---------------------------------------------------------------------------
def colorful_mask(rgb: np.ndarray, sat_min: int = SAT_MIN,
                  val_min: int = VAL_MIN) -> np.ndarray:
    """高饱和像素掩码 (H,W) uint8 0/255

    sat_min/val_min 可调 —— P0-b 的鲁棒性测试会扰这两个门限看中心漂移多少。
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    return (((s > sat_min) & (v > val_min)).astype(np.uint8)) * 255


def hue_mask(rgb: np.ndarray, color: str) -> np.ndarray:
    """按实测色相窗口取某颜色的掩码（仅用 H，不含 S/V 门限）"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    m = np.zeros(h.shape, dtype=np.uint8)
    for lo, hi in COLOR_HUE_WINDOWS[color]:
        m |= (((h >= lo) & (h <= hi)).astype(np.uint8)) * 255
    return m


def classify_hue(median_h: float) -> Optional[str]:
    """中位色相 → 颜色名；落在所有窗口之外返回 None"""
    for name, windows in COLOR_HUE_WINDOWS.items():
        for lo, hi in windows:
            if lo <= median_h <= hi:
                return name
    return None


# ---------------------------------------------------------------------------
# 几何量
# ---------------------------------------------------------------------------
def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    """连通域质心（正方形对称中心的无偏估计）"""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0:
        return (float("nan"), float("nan"))
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def _fitted_rect(mask: np.ndarray):
    """最小外接矩形 → (center, (w,h), angle_raw)；无轮廓返回 None"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return rect


def _angle_from_rect(rect) -> float:
    """拟合矩形朝向 → [0,90)（正方形 90° 周期）"""
    (_, _), (w, h), ang = rect
    if w < h:
        ang += 90.0
    return float(ang % 90.0)


def _angle_minarearect(mask: np.ndarray) -> float:
    """最小外接矩形朝向 → [0,90)"""
    rect = _fitted_rect(mask)
    return float("nan") if rect is None else _angle_from_rect(rect)


def _angle_pca(mask: np.ndarray) -> float:
    """PCA 主轴朝向 → [0,90)（仅作交叉校验，45° 附近会退化）"""
    ys, xs = np.nonzero(mask)
    if xs.size < 4:
        return float("nan")
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    w, vec = np.linalg.eigh(cov)              # 特征值升序
    v = vec[:, -1]                            # 最大特征值对应主轴
    ang = np.degrees(np.arctan2(v[1], v[0]))
    return float(ang % 90.0)


def _angle_spread(mask: np.ndarray) -> float:
    """PCA 与 minAreaRect 的朝向差（mod 90，0~45）—— 用来暴露 PCA 退化程度"""
    a, b = _angle_minarearect(mask), _angle_pca(mask)
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    d = abs(a - b) % 90.0
    return float(min(d, 90.0 - d))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def _reject_arm_like(comp_mask: np.ndarray, bbox, hsv: np.ndarray,
                     img_h: int) -> Optional[str]:
    """白臂误检剔除。返回拒绝原因，None 表示保留"""
    x, y, w, h = bbox
    # 1) 触底（机械臂总是从画面底边进入）
    if y + h >= img_h - REJECT_BOTTOM_MARGIN:
        return "touches_bottom"
    # 2) 淡蓝白（臂的塑料被白平衡染蓝）：H 与蓝色方块重合，靠 S 区分
    h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mh = float(np.median(h_[comp_mask]))
    ms = float(np.median(s_[comp_mask]))
    mv = float(np.median(v_[comp_mask]))
    if ARM_HUE[0] <= mh <= ARM_HUE[1] and ms <= ARM_SAT_MAX and mv >= ARM_VAL_MIN:
        return "arm_like_color"
    return None


def find_cubes(rgb: np.ndarray,
               colors: Optional[Sequence[str]] = None,
               sat_min: int = SAT_MIN,
               val_min: int = VAL_MIN,
               morph: int = 0,
               erode: int = 0,
               dilate: int = 0) -> List[CubeCandidate]:
    """在 RGB 图中找所有立方体候选

    Args:
        rgb: (H,W,3) uint8 **RGB**
        colors: 只保留这些颜色（英文名）；None = 全部
        sat_min/val_min: 彩色门限（鲁棒性测试用）
        morph: 形态学开运算迭代次数（默认 0 —— 实测开运算会啃掉小方块边缘）
        erode/dilate: 额外腐蚀/膨胀，鲁棒性测试用

    Returns:
        按面积降序的候选列表
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = colorful_mask(rgb, sat_min=sat_min, val_min=val_min)
    if erode > 0:
        mask = cv2.erode(mask, None, iterations=erode)
    if dilate > 0:
        mask = cv2.dilate(mask, None, iterations=dilate)
    if morph > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=morph)

    n, lab = cv2.connectedComponents(mask, connectivity=8)
    out: List[CubeCandidate] = []
    H = rgb.shape[0]
    for i in range(1, n):
        comp = (lab == i)
        area = int(comp.sum())
        if area < MIN_AREA or area > MAX_AREA:
            continue
        ys, xs = np.nonzero(comp)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1

        m8 = (comp.astype(np.uint8)) * 255
        rect = _fitted_rect(m8)
        if rect is None:
            continue
        (_, _), (rw, rh), _ = rect
        if rw < 1 or rh < 1:
            continue
        rect_fill = area / float(rw * rh)
        rect_aspect = max(rw, rh) / min(rw, rh)
        if rect_fill < RECT_FILL_MIN or rect_aspect > RECT_ASPECT_MAX:
            continue

        reason = _reject_arm_like(comp, (x0, y0, w, h), hsv, H)
        if reason is not None:
            continue

        mh = float(np.median(hsv[..., 0][comp]))
        ms = int(np.median(hsv[..., 1][comp]))
        mv = int(np.median(hsv[..., 2][comp]))
        color = classify_hue(mh)
        if color is None:
            continue
        if colors is not None and color not in colors:
            continue

        cx, cy = _centroid(m8)
        out.append(CubeCandidate(
            color=color, center=(cx, cy), angle_deg=_angle_from_rect(rect),
            area=area, bbox=(x0, y0, w, h), extent=rect_fill,
            side_px=float(np.sqrt(area)), median_hsv=(int(mh), ms, mv),
            extra=dict(angle_pca=_angle_pca(m8), angle_spread=_angle_spread(m8),
                       rect_fill=rect_fill, rect_aspect=rect_aspect,
                       rect_size=(float(rw), float(rh)),
                       centroid=m8, algo="minarearect"),
        ))
    out.sort(key=lambda c: -c.area)
    return out


def find_cube(rgb: np.ndarray, color: str, **kw) -> Optional[CubeCandidate]:
    """找指定颜色的最大候选（抓取用）"""
    cs = find_cubes(rgb, colors=[color], **kw)
    return cs[0] if cs else None


# ---------------------------------------------------------------------------
# Set-of-Mark 辅助
# ---------------------------------------------------------------------------
def draw_candidates(rgb: np.ndarray, cands: Sequence[CubeCandidate],
                    numbered: bool = True, thickness: int = 2) -> np.ndarray:
    """画出候选框与编号（Set-of-Mark 供 VLM 选编号）

    编号从 1 开始，画在框左上角外侧；同时写色名（ASCII，避免中文字体缺失）。
    """
    vis = rgb.copy()
    for idx, c in enumerate(cands, start=1):
        x, y, w, h = c.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), thickness)
        if numbered:
            tag = "%d" % idx if not c.color else "%d:%s" % (idx, c.color)
            ty = max(12, y - 4)
            cv2.rectangle(vis, (x, ty - 11), (x + 8 * len(tag) + 2, ty + 2), (255, 0, 0), -1)
            cv2.putText(vis, tag, (x + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cx, cy = int(round(c.center[0])), int(round(c.center[1]))
        cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 9, 1)
    return vis
