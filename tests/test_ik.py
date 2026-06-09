import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vla.control import ArmController


def test_ik_basic():
    arm = ArmController.__new__(ArmController)
    angles = arm._ik(0.3, 0.0, 0.15)
    assert len(angles) == 6, "IK 应返回 6 个关节角"
    for a in angles:
        assert -np.pi <= a <= np.pi, f"关节角越界: {a}"


if __name__ == "__main__":
    test_ik_basic()
    print("IK 测试通过")
