"""D435i 功能测试"""
import pyrealsense2 as rs, time, numpy as np

p = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("启动 D435i...")
cfg = p.start(config)
for _ in range(10):
    p.wait_for_frames()

print("\n=== 测试 30 帧 ===")
t0 = time.perf_counter()
for i in range(30):
    frames = p.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if i == 0:
        w, h = color.get_width(), color.get_height()
        print("  RGB: {}x{}".format(w, h))
        print("  Depth: {}x{}".format(depth.get_width(), depth.get_height()))
        d = np.asanyarray(depth.get_data())
        v = d[d > 0]
        if len(v) > 0:
            print("  深度范围: {:.2f}m ~ {:.2f}m".format(v.min() / 1000, v.max() / 1000))
        cy = d[h // 2, w // 2]
        print("  中心深度: {:.2f}m".format(cy / 1000 if cy > 0 else 0))

print("  帧率: {:.1f}FPS".format(30 / (time.perf_counter() - t0)))
p.stop()
print("完成")
