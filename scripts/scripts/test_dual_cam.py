"""双摄像头实时取景 — 上帝视角 + 机械臂局部"""
import sys, os, cv2, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import CAMERA_OVERHEAD, CAMERA_ARM

cap1 = cv2.VideoCapture(CAMERA_OVERHEAD, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(CAMERA_ARM, cv2.CAP_V4L2)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(0.3)
frame_count = 0
t0 = time.perf_counter()

while True:
    r1, f1 = cap1.read()
    r2, f2 = cap2.read()
    if not r1 or not r2:
        continue
    
    frame_count += 1
    if frame_count % 30 == 0:
        fps = frame_count / (time.perf_counter() - t0)
        print(f"  {frame_count}帧  平均{fps:.1f} FPS")

    cv2.imshow("Overhead", f1)
    cv2.imshow("Arm", f2)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
print(f"完成，共{frame_count}帧")
