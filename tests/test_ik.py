import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from vla.control.kinematics import RobotKinematics


def test_geo_ik_basic():
    from vla.control.controller import ArmController
    arm = ArmController.__new__(ArmController)
    arm.JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    ik = arm._init_ik("./models/so101_urdf/so101_new_calib.urdf")
    current = np.zeros(7)
    t_des = np.eye(4)
    t_des[:3, 3] = [0.3, 0.0, 0.15]
    angles_deg = ik.inverse_kinematics(current, t_des)
    assert len(angles_deg) >= 6, "IK 应返回至少 6 个关节角"
    angles_rad = np.deg2rad(angles_deg[:6])
    for a in angles_rad:
        assert -np.pi <= a <= np.pi, f"关节角越界: {a}"


if __name__ == "__main__":
    test_geo_ik_basic()
    print("IK 测试通过")
