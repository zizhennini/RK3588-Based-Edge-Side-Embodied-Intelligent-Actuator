"""FeetechBus 协议层单元测试（纯逻辑，PC 端无硬件可跑）

覆盖: sign-magnitude 编解码 / STS3215 控制表回归（防地址对调类 Bug）/
EPROM 可写集合 / 目标突变限幅 G3 / raw-rad 换算 / 无 SDK 时优雅失败
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hardware.feetech_bus import (
    STS3215_TABLE, EPROM_WRITABLE, SIGN_BITS, BAUDRATE_TABLE, SCAN_BAUDRATES,
    GRIPPER_PROTECTION, MODEL_NUMBER_STS3215, RESOLUTION,
    STATUS_ERROR_FLAGS, CURRENT_MA_PER_STEP, VOLTAGE_V_PER_STEP,
    FeetechBus, encode_sign_magnitude, decode_sign_magnitude,
    decode_status_flags, decode_load,
)


def test_sign_magnitude_roundtrip():
    """sign-magnitude 编解码往返一致（bit10/bit11/bit15 + 边界 + 溢出）"""
    for sign_bit in (10, 11, 15):
        max_mag = (1 << sign_bit) - 1
        for v in (0, 1, -1, max_mag, -max_mag, max_mag // 2, -(max_mag // 2)):
            enc = encode_sign_magnitude(v, sign_bit)
            assert decode_sign_magnitude(enc, sign_bit) == v, f"v={v} bit={sign_bit}"
        # 符号位正确落位
        assert encode_sign_magnitude(-5, 11) == (1 << 11) | 5
        assert encode_sign_magnitude(5, 11) == 5
        # 溢出必须报错
        try:
            encode_sign_magnitude(max_mag + 1, sign_bit)
            assert False, "溢出未报错"
        except ValueError:
            pass
    print("  PASS: bit10/11/15 往返 + 边界 + 溢出报错")


def test_control_table_regression():
    """关键寄存器地址回归（Feetech STS/SMS 手册值，防 arm.py 曾有的地址对调 Bug）"""
    expected = {
        "Firmware_Major_Version": (0, 1),
        "Model_Number": (3, 2),
        "ID": (5, 1),
        "Baud_Rate": (6, 1),
        "Return_Delay_Time": (7, 1),     # 原 arm.py 误写 0x29
        "Min_Position_Limit": (9, 2),
        "Max_Position_Limit": (11, 2),
        "Max_Torque_Limit": (16, 2),
        "Phase": (18, 1),
        "P_Coefficient": (21, 1),
        "D_Coefficient": (22, 1),
        "I_Coefficient": (23, 1),
        "Protection_Current": (28, 2),
        "Homing_Offset": (31, 2),
        "Operating_Mode": (33, 1),
        "Overload_Torque": (36, 1),
        "Torque_Enable": (40, 1),        # 0x28 ✓（原 arm.py 此值正确）
        "Acceleration": (41, 1),         # 0x29 —— 原 arm.py 误写 0x1A
        "Goal_Position": (42, 2),        # 0x2A ✓
        "Lock": (55, 1),
        "Present_Position": (56, 2),     # 0x38 ✓
        "Present_Voltage": (62, 1),
        "Present_Temperature": (63, 1),
        "Maximum_Acceleration": (85, 1),
    }
    for name, (addr, length) in expected.items():
        got = STS3215_TABLE[name]
        assert got == (addr, length), f"{name}: 期望 {(addr, length)} 实际 {got}"
    print(f"  PASS: {len(expected)} 个关键地址与手册一致")


def test_eprom_writable_set():
    """EPROM 可写集合: 含标定/坑位寄存器, 不含只读与 SRAM"""
    must_in = {"Homing_Offset", "Min_Position_Limit", "Max_Position_Limit",
               "Phase", "Return_Delay_Time", "Operating_Mode",
               "Max_Torque_Limit", "Protection_Current", "Overload_Torque",
               "P_Coefficient", "I_Coefficient", "D_Coefficient",
               "ID", "Baud_Rate"}
    must_out = {"Model_Number", "Firmware_Major_Version", "Firmware_Minor_Version",
                "Torque_Enable", "Goal_Position", "Present_Position", "Lock",
                "Acceleration"}
    assert must_in <= EPROM_WRITABLE, f"缺失: {must_in - EPROM_WRITABLE}"
    assert not (must_out & EPROM_WRITABLE), f"误入: {must_out & EPROM_WRITABLE}"
    # 防烧三参数键名都在控制表（防笔误）
    for key in GRIPPER_PROTECTION:
        assert key in STS3215_TABLE, f"GRIPPER_PROTECTION 键不在控制表: {key}"
    print("  PASS: EPROM 集合边界正确 + 防烧键名有效")


def test_clamp_relative_goal():
    """G3 目标突变限幅: 限内透传 / 超限截断 / present 缺失放行 / None 关闭"""
    goal = {1: 10.0, 2: -10.0, 3: 0.5}
    present = {1: 8.0, 2: -8.0}          # 3 无 present
    out = FeetechBus.clamp_relative_goal(goal, present, max_delta=1.0)
    assert out[1] == 9.0, out              # 10 → 8+1 截断
    assert out[2] == -9.0, out             # -10 → -8-1 截断
    assert out[3] == 0.5, out              # 无 present 放行（首帧语义）
    # 限内不动
    out2 = FeetechBus.clamp_relative_goal({1: 8.5}, {1: 8.0}, 1.0)
    assert out2[1] == 8.5
    # None/0 = 关闭
    out3 = FeetechBus.clamp_relative_goal(goal, present, None)
    assert out3 == goal
    print("  PASS: 截断/透传/首帧/关闭 四路径")


def test_raw_rad_roundtrip():
    """raw↔rad 换算往返 + 与 SO101Arm 历史公式一致 + clamp"""
    mid = 2048
    for raw in (946, 2048, 3287, 130, 3985):
        rad = FeetechBus.raw_to_rad(raw, mid)
        back = FeetechBus.rad_to_raw(rad, mid, 0, 4095)
        assert abs(back - raw) <= 1, f"raw={raw} back={back}"
    # 与 arm.py 公式一致性: deg = (raw-mid)*360/4095
    rad = FeetechBus.raw_to_rad(2048 + 4095 / 2, 2048)
    assert abs(np.rad2deg(rad) - 180.0) < 0.1
    # clamp
    assert FeetechBus.rad_to_raw(99.0, mid, 946, 3287) == 3287
    assert FeetechBus.rad_to_raw(-99.0, mid, 946, 3287) == 946
    # 分辨率
    step = FeetechBus.raw_to_rad(2049, mid) - FeetechBus.raw_to_rad(2048, mid)
    assert abs(np.rad2deg(step) - 360 / 4095) < 1e-6
    print("  PASS: 往返/公式一致/clamp/分辨率")


def test_tables_consistency():
    """波特率表与扫描序列一致 + 型号常量"""
    for baud in BAUDRATE_TABLE:
        assert baud in SCAN_BAUDRATES, f"BAUDRATE_TABLE 波特率不在扫描序列: {baud}"
    assert MODEL_NUMBER_STS3215 == 777
    assert RESOLUTION == 4096
    assert SIGN_BITS["Homing_Offset"] == 11
    assert SIGN_BITS["Present_Position"] == 15
    print("  PASS: 波特率表/型号/分辨率/符号位常量")


def test_status_flags_decode():
    """Status 寄存器错误标志解码（STS3215 位定义，社区实现同源）"""
    assert decode_status_flags(0) == []
    assert decode_status_flags(0x01) == ["Voltage"]
    assert decode_status_flags(0x04) == ["Temperature"]
    got = decode_status_flags(0x04 | 0x20)  # Temperature + Overload
    assert sorted(got) == sorted(["Temperature", "Overload"]), got
    # bit4（未定义位）不产生条目
    assert decode_status_flags(0x10) == []
    # 全标志
    all_names = set(STATUS_ERROR_FLAGS.values())
    assert set(decode_status_flags(0x2F)) == all_names
    print("  PASS: 0/单位/组合/未定义位/全标志 五路径")


def test_load_decode():
    """Present_Load 解码: bit10 方向 + 低 10 位幅值（0-1000 → 0-100%）"""
    assert decode_load(0) == (0.0, "CCW")
    assert decode_load(500) == (50.0, "CCW")
    assert decode_load(0x400 | 500) == (50.0, "CW")
    assert decode_load(0x400 | 1000) == (100.0, "CW")
    assert decode_load(1000) == (100.0, "CCW")
    # 幅值截断在低 10 位
    pct, d = decode_load(0x400 | 0x3FF)
    assert abs(pct - 102.3) < 0.1 and d == "CW"
    print("  PASS: 零/幅值/方向/满量程/截断")


def test_diagnostic_constants():
    """诊断换算常量（STS3215: 电流 6.5mA/step, 电压 0.1V/step）"""
    assert CURRENT_MA_PER_STEP == 6.5
    assert VOLTAGE_V_PER_STEP == 0.1
    # 真机实测锚点: 电压寄存器读 49 → 4.9V（2026-09-25 板端 verify 输出）
    assert round(49 * VOLTAGE_V_PER_STEP, 1) == 4.9
    print("  PASS: 常量与真机实测锚点一致")


def test_no_sdk_graceful():
    """无 scservo_sdk 环境: 模块可导入, 实例化给出清晰指引"""
    try:
        import scservo_sdk  # noqa: F401
        has_sdk = True
    except ImportError:
        has_sdk = False
    if has_sdk:
        print("  SKIP: 当前环境有 scservo_sdk（板端跑此分支属预期）")
        return
    try:
        FeetechBus("/dev/null")
        assert False, "无 SDK 却实例化成功?"
    except ImportError as e:
        assert "feetech-servo-sdk" in str(e), f"错误指引不清晰: {e}"
    print("  PASS: 无 SDK 时 ImportError + pip 指引")


def test_arm_module_importable():
    """hardware.arm 模块级导入不依赖硬件 SDK（PC 端导入链保障）"""
    import hardware.arm as arm_mod
    assert arm_mod.GRIPPER_OPEN_PULSE == 2600
    assert arm_mod.GRIPPER_CLOSE_PULSE == 1781
    assert len(arm_mod.DEFAULT_CALIBRATION) == 6
    assert arm_mod.MOTOR_IDS["gripper"] == 6
    # move_to 无 kinematics 时的报错语义存在
    assert "kinematics" in arm_mod.SO101Arm.move_to.__doc__
    print("  PASS: arm 模块可导入 + 常量回归")


if __name__ == "__main__":
    print("=" * 50)
    print("FeetechBus Protocol Layer Tests")
    print("=" * 50)
    tests = [
        ("sign-magnitude roundtrip", test_sign_magnitude_roundtrip),
        ("control table regression", test_control_table_regression),
        ("eprom writable set", test_eprom_writable_set),
        ("clamp relative goal (G3)", test_clamp_relative_goal),
        ("raw<->rad roundtrip", test_raw_rad_roundtrip),
        ("tables consistency", test_tables_consistency),
        ("status flags decode", test_status_flags_decode),
        ("load decode", test_load_decode),
        ("diagnostic constants", test_diagnostic_constants),
        ("no-sdk graceful", test_no_sdk_graceful),
        ("arm module importable", test_arm_module_importable),
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
    sys.exit(0 if passed == len(tests) else 1)
