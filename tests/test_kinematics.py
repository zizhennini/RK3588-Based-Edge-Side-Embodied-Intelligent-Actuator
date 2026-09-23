"""SO-ARM101 运动学单元测试"""
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from policy.kinematics import Kinematics


def test_fk_ik_consistency():
    """FK(IK(target)) == target（偏差 < 1mm）"""
    kin = Kinematics()
    test_points = [
        np.array([0.20, 0.05, 0.15]),
        np.array([0.15, 0.10, 0.12]),
        np.array([0.10, 0.15, 0.18]),
        np.array([0.22, 0.00, 0.10]),
        np.array([0.05, 0.20, 0.14]),
        np.array([0.18, 0.12, 0.20]),
        np.array([0.12, -0.08, 0.16]),
        np.array([0.20, -0.10, 0.11]),
    ]
    current = np.array([0.0, 1.0, 1.5, 0.0, 0.0, 0.5])

    for target in test_points:
        ik_result = kin.inverse_kinematics(target, current)
        fk_result = kin.forward_kinematics(ik_result)
        error_mm = np.linalg.norm(fk_result - target) * 1000
        assert error_mm < 1.0, (
            f"target={target}, fk={fk_result}, error={error_mm:.3f}mm"
        )
    print(f"  PASS: {len(test_points)} points, all < 1mm")


def test_ik_fk_consistency():
    """IK(FK(angles)) == angles（偏差 < 1deg）"""
    kin = Kinematics()
    test_angles = [
        np.array([0.0, 1.0, 1.5, 0.0, 0.0, 0.5]),
        np.array([0.5, 0.8, 1.2, -0.3, 0.1, 0.5]),
        np.array([-0.3, 1.5, 0.8, 0.2, -0.1, 0.3]),
        np.array([1.0, 1.2, 1.0, 0.5, 0.0, 0.7]),
        np.array([-0.8, 2.0, 0.5, -0.2, 0.3, 0.4]),
    ]

    for angles in test_angles:
        xyz = kin.forward_kinematics(angles)
        ik_result = kin.inverse_kinematics(xyz, angles)
        # 只比较前 3 个关节（j4/j5/j6 保持当前值）
        error_deg = np.max(np.abs(np.degrees(ik_result[:3] - angles[:3])))
        assert error_deg < 1.0, (
            f"angles={angles}, ik={ik_result}, error={error_deg:.2f}deg"
        )
    print(f"  PASS: {len(test_angles)} configs, all < 1deg")


def test_workspace_clamp():
    """超范围目标被正确钳制"""
    kin = Kinematics()
    r_max = kin.L1 + kin.L2  # 0.2509

    # 超出最大 reach
    far_point = np.array([0.30, 0.0, 0.10])
    clamped = kin.clamp_workspace(far_point)
    r_clamped = math.sqrt(clamped[0]**2 + clamped[1]**2)
    assert r_clamped <= r_max + 1e-6, f"r_clamped={r_clamped} > r_max={r_max}"

    # 高度过低
    low_point = np.array([0.15, 0.0, 0.01])
    clamped = kin.clamp_workspace(low_point)
    assert clamped[2] >= kin.BASE_HEIGHT, f"z={clamped[2]} < BASE_HEIGHT"

    # 高度过高
    high_point = np.array([0.15, 0.0, 0.50])
    clamped = kin.clamp_workspace(high_point)
    assert clamped[2] <= kin.BASE_HEIGHT + r_max + 1e-6, (
        f"z={clamped[2]} > max"
    )

    # 正常范围内的点不应被修改
    normal = np.array([0.15, 0.05, 0.15])
    clamped = kin.clamp_workspace(normal)
    np.testing.assert_array_almost_equal(clamped, normal)
    print("  PASS: all workspace clamp tests")


def test_joint_limits():
    """输出角度在限位内"""
    kin = Kinematics()
    # 极端目标点
    extreme_targets = [
        np.array([0.25, 0.0, 0.06]),
        np.array([0.01, 0.25, 0.20]),
        np.array([0.20, 0.20, 0.25]),
        np.array([-0.20, 0.10, 0.08]),
    ]
    current = np.array([0.0, 1.0, 1.5, 0.0, 0.0, 0.5])

    for target in extreme_targets:
        ik_result = kin.inverse_kinematics(target, current)
        for i in range(6):
            low, high = kin.JOINT_LIMITS[i + 1]
            assert low - 1e-9 <= ik_result[i] <= high + 1e-9, (
                f"joint {i+1}: {ik_result[i]:.4f} not in [{low:.4f}, {high:.4f}]"
            )

    # clamp_joint 直接测试
    over_limit = np.array([5.0, -5.0, 5.0, 5.0, 5.0, 5.0])
    clamped = kin.clamp_joint(over_limit)
    for i in range(6):
        low, high = kin.JOINT_LIMITS[i + 1]
        assert low - 1e-9 <= clamped[i] <= high + 1e-9
    print("  PASS: all joint limits respected")


def test_known_poses():
    """已知位姿验证"""
    kin = Kinematics()

    # 全伸展：j1=0, theta1=0, theta2=pi → 末端在正前方最大距离
    # FK: fx = L1*cos(0) + L2*cos(0 + pi - pi) = L1 + L2
    #     fy = L1*sin(0) + L2*sin(pi - pi) = 0
    #     x = fx, y = 0, z = BASE_HEIGHT + fy = BASE_HEIGHT
    full_extend = np.array([0.0, 0.0, math.pi, 0.0, 0.0, 0.0])
    fk = kin.forward_kinematics(full_extend)
    expected = np.array([kin.L1 + kin.L2, 0.0, kin.BASE_HEIGHT])
    np.testing.assert_array_almost_equal(fk, expected, decimal=4)

    # 直接上方：theta1=pi/2, theta2=pi → 末端在正上方
    # FK: fx = L1*cos(pi/2) + L2*cos(pi/2 + pi - pi) = 0 + L2*cos(pi/2) = 0
    #     fy = L1*sin(pi/2) + L2*sin(pi/2) = L1 + L2
    straight_up = np.array([0.0, math.pi / 2, math.pi, 0.0, 0.0, 0.0])
    fk = kin.forward_kinematics(straight_up)
    expected_z = kin.BASE_HEIGHT + kin.L1 + kin.L2
    assert abs(fk[2] - expected_z) < 0.001, f"z={fk[2]}, expected={expected_z}"
    assert abs(fk[0]) < 0.001, f"x={fk[0]}, expected ~0"

    # IK → 全伸展位姿
    target_extend = np.array([kin.L1 + kin.L2, 0.0, kin.BASE_HEIGHT])
    current = np.array([0.0, 0.0, math.pi, 0.0, 0.0, 0.0])
    ik = kin.inverse_kinematics(target_extend, current)
    fk_back = kin.forward_kinematics(ik)
    error_mm = np.linalg.norm(fk_back - target_extend) * 1000
    assert error_mm < 1.0, f"full extend error: {error_mm:.3f}mm"

    print("  PASS: all known pose checks")


if __name__ == "__main__":
    print("=" * 50)
    print("SO-ARM101 Kinematics Tests")
    print("=" * 50)
    tests = [
        ("FK(IK(target)) < 1mm", test_fk_ik_consistency),
        ("IK(FK(angles)) < 1deg", test_ik_fk_consistency),
        ("workspace clamp", test_workspace_clamp),
        ("joint limits", test_joint_limits),
        ("known poses", test_known_poses),
    ]
    passed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{len(tests)} passed")
