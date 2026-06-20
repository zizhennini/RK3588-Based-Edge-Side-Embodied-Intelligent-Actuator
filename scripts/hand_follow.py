"""YOLO-World 手部检测 + 视觉跟随"""
import sys, time, cv2, numpy as np
sys.path.insert(0, ".")

from astra import USBCamera
from vla.control import ArmController
from config.settings import CAMERA_INDEX, SERIAL_PORT, SERIAL_BAUD
from rknnlite.api import RKNNLite


def encode_text(rknn_clip, text):
    vocab = {chr(i): i for i in range(256)}
    ids = np.array([[vocab.get(c, 0) for c in text.lower()[:20]] + [0] * (20 - len(text))], dtype=np.int64)
    return rknn_clip.inference(inputs=[ids])[0]


def decode_yolo(outputs, conf_thresh=0.3):
    """简易 YOLO 解码，输出检测框 (cx, cy)"""
    scales = [(80, 80), (40, 40), (20, 20)]
    boxes = []
    for i, (h, w) in enumerate(scales):
        scores = outputs[i * 2][0]  # (80, h, w)
        deltas = outputs[i * 2 + 1][0]  # (4, h, w)
        score_map = scores.max(axis=0)  # (h, w)
        yy, xx = np.where(score_map > conf_thresh)
        for y, x in zip(yy, xx):
            conf = float(score_map[y, x])
            dx = float(deltas[0, y, x])
            dy = float(deltas[1, y, x])
            dw = float(deltas[2, y, x])
            dh = float(deltas[3, y, x])
            cx = (x + dx) * 640 / w
            cy = (y + dy) * 640 / h
            bw = np.exp(dw) * 640 / w
            bh = np.exp(dh) * 640 / h
            boxes.append((cx, cy, bw, bh, conf))
    if not boxes:
        return None
    boxes.sort(key=lambda b: b[4], reverse=True)
    cx, cy = int(boxes[0][0]), int(boxes[0][1])
    return cx, cy


def main():
    print("加载 YOLO-World ...")
    rknn_clip = RKNNLite()
    rknn_clip.load_rknn("./models/yolo_world/clip_text.rknn")
    rknn_clip.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    text_feat = encode_text(rknn_clip, "hand")
    text_feat = np.tile(text_feat, (80, 1)).reshape(1, 80, -1).astype(np.float32)

    rknn = RKNNLite()
    rknn.load_rknn("./models/yolo_world/yolo_world_v2s.rknn")
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)

    cam = USBCamera(CAMERA_INDEX)
    cam.connect()
    arm = ArmController(SERIAL_PORT, SERIAL_BAUD)

    print("移到拍照位...")
    arm._write_angle(1, np.deg2rad(0))
    arm._write_angle(2, np.deg2rad(60))
    arm._write_angle(3, np.deg2rad(-60))
    arm._write_angle(4, np.deg2rad(0))
    time.sleep(3)

    while True:
        rgb = cam.read_rgb()
        display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        inp = cv2.resize(rgb, (640, 640))
        inp = np.transpose(inp, (2, 0, 1)).reshape(1, 3, 640, 640).astype(np.float32) / 255.0
        outputs = rknn.inference(inputs=[inp, text_feat])
        det = decode_yolo(outputs)

        if det:
            cx, cy = det
            cv2.circle(display, (cx, cy), 10, (0, 255, 0), -1)

        cv2.imshow("Hand Follow", cv2.resize(display, (960, 720)))
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        # 打印分数看模型有没有输出
        scores = outputs[0][0]
        print(f" score_range=[{scores.min():.2f},{scores.max():.2f}] top5={np.sort(scores.flatten())[-5:].tolist()}")


if __name__ == "__main__":
    main()
