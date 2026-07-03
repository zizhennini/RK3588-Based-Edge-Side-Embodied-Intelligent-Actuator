"""D435i 深度精度测试 — 鼠标实时显示坐标"""
import pyrealsense2 as rs, cv2, numpy as np

p = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = p.start(cfg)
intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# 高精度模式
sensor = profile.get_device().first_depth_sensor()
sensor.set_option(rs.option.visual_preset, 3)
sensor.set_option(rs.option.laser_power, 150)
sensor.set_option(rs.option.enable_auto_exposure, 1)

# 滤镜
dec = rs.decimation_filter()
spa = rs.spatial_filter()
tmp = rs.temporal_filter()
hole = rs.hole_filling_filter()

for _ in range(15): p.wait_for_frames()

coord_text, status_text = "", ""

def mouse_cb(event, x, y, flags, param):
    global coord_text
    u, v = int(x * 640 / 960), int(y * 480 / 720)
    if v >= 480 or u >= 640: return
    z = latest_depth.get_distance(u, v) if latest_depth is not None else 0
    if z > 0:
        pt = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
        coord_text = "({},{}) → ({:.3f},{:.3f},{:.3f})m".format(u, v, pt[0], pt[1], pt[2])

latest_depth = None
filtered = True
cv2.namedWindow("D435i")
cv2.setMouseCallback("D435i", mouse_cb)

while True:
    frames = p.wait_for_frames()
    latest_depth = frames.get_depth_frame()

    if filtered:
        f = dec.process(latest_depth)
        f = spa.process(f)
        f = tmp.process(f)
        f = hole.process(f)
    else:
        f = latest_depth

    d = np.asanyarray(f.get_data())
    d_cm = cv2.applyColorMap(np.clip((d / 3000 * 255).astype(np.uint8), 0, 255), cv2.COLORMAP_JET)

    if coord_text:
        cv2.putText(d_cm, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(d_cm, "Filtered" if filtered else "Raw", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("D435i", cv2.resize(d_cm, (960, 720)))
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"): break
    if key == ord("f"): filtered = not filtered

p.stop()
cv2.destroyAllWindows()
