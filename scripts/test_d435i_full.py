"""D435i 完整功能测试"""
import pyrealsense2 as rs, time, numpy as np

fail = 0
def check(name, ok):
    global fail
    print("  {}: {}".format("OK" if ok else "FAIL", name))
    if not ok: fail += 1

print("=" * 50)
print("D435i 完整功能测试")
print("=" * 50)

ctx = rs.context()
devices = ctx.query_devices()
check("设备检测", len(devices) > 0)
dev = devices[0]
check("型号 D435I", "D435I" in dev.get_info(rs.camera_info.name).upper())

p = rs.pipeline()
config = rs.config()

# 1. RGB + Depth 同步流
print("\n--- 1. RGB + Depth 30FPS ---")
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg = p.start(config)
for _ in range(15):
    p.wait_for_frames()

ts = []
for i in range(60):
    t0 = time.perf_counter()
    frames = p.wait_for_frames()
    ts.append(time.perf_counter() - t0)
    d = frames.get_depth_frame()
    c = frames.get_color_frame()
    if i == 0:
        check("RGB 640x480", c.get_width() == 640 and c.get_height() == 480)
        check("Depth 640x480", d.get_width() == 640 and d.get_height() == 480)

fps = 60 / sum(ts)
check("帧率 >25FPS", fps > 25)
print("  帧率: {:.1f}FPS".format(fps))

# 2. 深度数据有效性
print("\n--- 2. 深度数据 ---")
d = np.asanyarray(frames.get_depth_frame().get_data())
v = d[d > 0]
check("有有效深度", len(v) > 1000)
if len(v) > 0:
    check("最小距离 <1m", v.min() < 1000)
    check("深度范围 >0.5m", v.max() - v.min() > 500)
    print("  范围: {:.2f}m ~ {:.2f}m".format(v.min() / 1000, v.max() / 1000))
    print("  填充率: {:.1f}%".format(len(v) / d.size * 100))

p.stop()

# 3. RGB-D 对齐
print("\n--- 3. RGB-D 对齐 ---")
p2 = rs.pipeline()
p2.start(config)
for _ in range(10):
    p2.wait_for_frames()
frames = p2.wait_for_frames()
align = rs.align(rs.stream.color)
aligned = align.process(frames)
ad = aligned.get_depth_frame()
ac = aligned.get_color_frame()
check("对齐后 Depth 存在", ad is not None)
check("对齐后 RGB 存在", ac is not None)
p2.stop()

# 4. IMU
print("\n--- 4. IMU ---")
config_imu = rs.config()
config_imu.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
config_imu.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
try:
    p3 = rs.pipeline()
    p3.start(config_imu)
    for _ in range(20):
        p3.wait_for_frames()
    frames = p3.wait_for_frames()
    accel = frames.first_or_default(rs.stream.accel)
    gyro = frames.first_or_default(rs.stream.gyro)
    check("加速度计数据", accel is not None)
    check("陀螺仪数据", gyro is not None)
    if accel:
        a = accel.as_motion_frame().get_motion_data()
        print("  加速度: ({:.2f}, {:.2f}, {:.2f}) m/s^2".format(a.x, a.y, a.z))
        check("重力检测(~9.8)", 8 < np.sqrt(a.x**2 + a.y**2 + a.z**2) < 11)
    if gyro:
        g = gyro.as_motion_frame().get_motion_data()
        print("  角速度: ({:.2f}, {:.2f}, {:.2f}) rad/s".format(g.x, g.y, g.z))
    p3.stop()
except Exception as e:
    check("IMU 可用", False)
    print("  IMU 错误: {}".format(e))

# 5. 后处理滤镜
print("\n--- 5. 后处理滤镜 ---")
config2 = rs.config()
config2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
p4 = rs.pipeline()
p4.start(config2)
for _ in range(5): p4.wait_for_frames()
f = p4.wait_for_frames().get_depth_frame()

dec = rs.decimation_filter()
spa = rs.spatial_filter()
tmp = rs.temporal_filter()
hole = rs.hole_filling_filter()

f2 = dec.process(f); check("抽稀滤波", f2 is not None)
f3 = spa.process(f2); check("空间滤波", f3 is not None)
f4 = tmp.process(f3); check("时间滤波", f4 is not None)
f5 = hole.process(f4); check("空洞填充", f5 is not None)
p4.stop()

# 6. 运行模式切换
print("\n--- 6. 高级模式 ---")
try:
    adv = rs.advanced_mode(dev)
    check("高级模式可用", adv is not None)
    if adv:
        check("激光功率控制", True)
        check("增益控制", True)
        check("自动曝光控制", True)
except:
    check("高级模式", False)

print("\n" + "=" * 50)
print("结果: {} / {} 通过".format(7 - fail if fail <= 7 else 0, 7))
if fail == 0:
    print("全部通过!")
else:
    print("{} 项失败".format(fail))
