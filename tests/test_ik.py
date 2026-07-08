import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vla.control.controller import NumericalIK


def test_numerical_ik_fk_consistency():
    """测试 IK 与 FK 的一致性：IK(FK(q)) ≈ q"""
    ik = NumericalIK()
    np.random.seed(42)
    for _ in range(5):
        q_orig_deg = np.random.uniform(
            low=[ik.joint_limits_rad[i][0] for i in range(5)],
            high=[ik.joint_limits_rad[i][1] for i in range(5)],
        )
        q_orig_deg = np.rad2deg(q_orig_deg)
        T = ik.forward_kinematics(q_orig_deg)
        q_result_deg = ik.inverse_kinematics(q_orig_deg, T)
        err = np.max(np.abs(q_result_deg[:5] - q_orig_deg))
        assert err < 1.0, f"IK-FK 一致性误差过大: {err:.4f}°"
    print("数值 IK — FK/IK 一致性测试通过")


def test_numerical_ik_reach_target():
    """测试 IK 能正确到达目标位置"""
    ik = NumericalIK()
    targets = [
        np.array([0.20, 0.00, 0.10]),
        np.array([0.15, 0.10, 0.08]),
        np.array([0.10, -0.05, 0.15]),
    ]
    q_init = np.zeros(6)
    for target in targets:
        T = np.eye(4)
        T[:3, 3] = target
        q_deg = ik.inverse_kinematics(q_init, T)
        pos = ik.fk(np.deg2rad(q_deg[:5]))
        err = np.linalg.norm(pos - target)
        assert err < 0.02, f"位置误差过大: {err:.4f}m, target={target}, got={pos}"
    print("数值 IK — 目标到达测试通过")


def test_numerical_ik_joint_limits():
    """测试 IK 输出在关节限位内"""
    ik = NumericalIK()
    targets = [
        np.array([0.30, 0.00, 0.20]),
        np.array([-0.20, 0.00, 0.05]),
        np.array([0.00, 0.20, 0.15]),
    ]
    q_init = np.zeros(6)
    for target in targets:
        T = np.eye(4)
        T[:3, 3] = target
        q_deg = ik.inverse_kinematics(q_init, T)
        q_rad = np.deg2rad(q_deg[:5])
        for i in range(5):
            low, high = ik.joint_limits_rad[i]
            assert low - 0.01 <= q_rad[i] <= high + 0.01, \
                f"关节 {i} 越界: {q_rad[i]:.4f} rad, limit=[{low:.4f}, {high:.4f}]"
    print("数值 IK — 关节限位测试通过")


if __name__ == "__main__":
    test_numerical_ik_fk_consistency()
    test_numerical_ik_reach_target()
    test_numerical_ik_joint_limits()
    print("\n所有 IK 测试通过")
