from rknn.api import RKNN
import numpy as np
import os

RKNN_MODEL_PATH = "./onnx/qwen3.5_vision.onnx"
TARGET = "rk3588"

mean_val = [[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]]
std_val = [[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]]

grid_const = np.array([[1, 28, 28]], dtype=np.int64)

rknn = RKNN(verbose=False)
rknn.config(target_platform=TARGET, mean_values=mean_val, std_values=std_val)
rknn.load_onnx(RKNN_MODEL_PATH,
    inputs=["pixel", "grid_thw"],
    input_size_list=[[1, 3, 448, 448], [1, 3]],
    input_initial_val=[None, grid_const])
rknn.build(do_quantization=False, dataset=None)
os.makedirs("rknn", exist_ok=True)
rknn.export_rknn("./rknn/qwen3.5_vision_rk3588.rknn")
print("导出完成: ./rknn/qwen3.5_vision_rk3588.rknn")
rknn.release()
