"""D435i 最佳精度模式 — 简化稳定版"""
import pyrealsense2 as rs, cv2, numpy as np
from collections import deque

p = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
profile = p.start(cfg)
intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

sensor = profile.get_device().first_depth_sensor()
sensor.set_option(rs.option.visual_preset, 3)
sensor.set_option(rs.option.laser_power, 150)
sensor.set_option(rs.option.enable_auto_exposure, 1)

spa = rs.spatial_filter()
tmp = rs.temporal_filter()

buf = deque(maxlen=5)
for _ in range(20): p.wait_for_frames()

coord_text, depth_frame = "", None

def mouse_cb(event, x, y, flags, param):
    global coord_text
    if depth_frame is None: return
    u, v = int(x * 848 / 1272), int(y * 480 / 720)
    if u >= 848 or v >= 480: return
    z = depth_frame.get_distance(u, v)
    if z > 0:
        pt = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
        coord_text = "px({:>3d},{:>3d}) → ({:.3f},{:.3f},{:.3f})m".format(u, v, pt[0], pt[1], pt[2])

cv2.namedWindow("D435i Best")
cv2.setMouseCallback("D435i Best", mouse_cb)

while True:
    frames = p.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    f = spa.process(depth_frame)
    f = tmp.process(f)

    d = np.asanyarray(f.get_data()).astype(np.float32)
    d[d > 2000] = 0
    buf.append(d)
    d = np.median(np.array(buf), axis=0).astype(np.uint16) if len(buf) == 5 else d.astype(np.uint16)

    d_cm = cv2.applyColorMap(np.clip((d / 2000 * 255).astype(np.uint8), 0, 255), cv2.COLORMAP_JET)

    if coord_text:
        cv2.putText(d_cm, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(d_cm, "848x480 HighAccuracy +5帧中值", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("D435i Best", cv2.resize(d_cm, (1272, 720)))
    if cv2.waitKey(10) & 0xFF == ord("q"): break

p.stop()
cv2.destroyAllWindows()
