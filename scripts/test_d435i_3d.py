"""D435i 3D坐标可视化 — 鼠标悬停显示3D坐标"""
import pyrealsense2 as rs, cv2, numpy as np

p = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = p.start(cfg)
intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

for _ in range(15): p.wait_for_frames()

last_text = ""

def mouse_cb(event, x, y, flags, param):
    global last_text
    u, v = int(x / 960 * 640), int(y / 720 * 480)
    z = depth_frame.get_distance(u, v)
    if z > 0:
        pt = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
        last_text = "({:>3d},{:>3d}) → ({:.3f},{:.3f},{:.3f})m".format(u, v, pt[0], pt[1], pt[2])

cv2.namedWindow("D435i")
cv2.setMouseCallback("D435i", mouse_cb)

while True:
    frames = p.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame: continue

    color = np.asanyarray(color_frame.get_data())
    if last_text:
        cv2.putText(color, last_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("D435i", cv2.resize(color, (960, 720)))
    if cv2.waitKey(1) & 0xFF == ord("q"): break

p.stop()
cv2.destroyAllWindows()
