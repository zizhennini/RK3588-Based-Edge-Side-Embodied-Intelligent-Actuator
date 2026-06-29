"""测试 XLeRobot 的 IK 公式 — 带机械偏移补偿"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def xlerobot_ik(x, y, l1=0.1159, l2=0.1350):
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    if r > r_max:
        scale = r_max / r; x *= scale; y *= scale; r = r_max
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale = r_min / r; x *= scale; y *= scale; r = r_min
    cos_t2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    theta2 = math.pi - math.acos(cos_t2)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))
    return 90 - math.degrees(joint2), math.degrees(joint3) - 90


def xlerobot_fk(j2_deg, j3_deg, l1=0.1159, l2=0.1350):
    j2_rad = math.radians(90 - j2_deg)
    j3_rad = math.radians(j3_deg + 90)
    to = math.atan2(0.028, 0.11257)
    t2o = math.atan2(0.0052, 0.1349) + to
    theta1 = j2_rad - to
    theta2 = j3_rad - t2o
    fx = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2 - math.pi)
    fy = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2 - math.pi)
    return fx, fy


def old_geo_ik(x, y, z, L1=0.120, L2=0.150, L3=0.180):
    r = math.sqrt(x**2 + y**2)
    d = math.sqrt((r - 0.025)**2 + (z - L1)**2)
    ct = (d**2 - L2**2 - L3**2) / (2 * L2 * L3)
    ct = max(-1.0, min(1.0, ct))
    t3 = -abs(math.acos(ct))
    t2 = math.atan2(z - L1, r - 0.025) - math.atan2(L3 * math.sin(t3), L2 + L3 * math.cos(t3))
    return math.degrees(t2), math.degrees(t3)


print("=" * 60)
print("比较 IK 公式: XLeRobot vs 原几何IK")
print("=" * 60)

test_points = [
    (0.20, 0.15, "前上方"),
    (0.25, 0.10, "前方"),
    (0.15, 0.20, "更上方"),
    (0.20, 0.05, "桌面近处"),
]

print(f"\n{'目标':<22} {'XLeRobot':<18} {'原几何':<18}")
print("-" * 58)

for i, (px, py, label) in enumerate(test_points):
    xj2, xj3 = xlerobot_ik(px, py)
    oj2, oj3 = old_geo_ik(px, py, 0.20)
    fkx, fky = xlerobot_fk(xj2, xj3)
    dev = math.hypot(fkx - px, fky - py) * 1000
    flag = " ✅" if dev < 1 else ""
    print(f"{label} ({px:.2f},{py:.2f}):  {xj2:>6.1f}°/{xj3:<6.1f}°  {oj2:>6.1f}°/{oj3:<6.1f}°  FK偏差{dev:.2f}mm{flag}")

print()
print("P 控制收敛测试:")
tx, ty = 0.20, 0.15
cx, cy = 0.15, 0.10
kp = 0.5
for step in range(8):
    jt2, jt3 = xlerobot_ik(tx, ty)
    jc2, jc3 = xlerobot_ik(cx, cy)
    cx += (jt2 - jc2) * kp * 0.001
    cy += (jt3 - jc3) * kp * 0.001
    err = math.hypot(tx - cx, ty - cy)
    print(f"  步{step+1}: ({cx:.4f},{cy:.4f}) 误差{err*1000:.1f}mm")
