"""单元测试：视觉定位（ColorLocator + MobileNetSSD 导入）"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vla.vision import ColorLocator

K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)


def test_color_locate_red():
    locator = ColorLocator(K)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[200:280, 300:380] = [255, 0, 0]
    depth = np.ones((480, 640), dtype=np.float32) * 0.5
    result = locator.locate(rgb, depth, "红色")
    assert result is not None
    assert abs(result["z"] - 0.5) < 0.01


def test_color_locate_none():
    locator = ColorLocator(K)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32) * 0.5
    result = locator.locate(rgb, depth, "红色")
    assert result is None


def test_color_locate_too_small():
    locator = ColorLocator(K)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[240:245, 320:325] = [255, 0, 0]
    depth = np.ones((480, 640), dtype=np.float32) * 0.5
    result = locator.locate(rgb, depth, "红色")
    assert result is None


def test_ssd_import():
    from vla.vision import MobileNetSSD
    import os
    assert os.path.exists(".")


if __name__ == "__main__":
    test_color_locate_red()
    test_color_locate_none()
    test_color_locate_too_small()
    test_ssd_import()
    print("所有测试通过")
