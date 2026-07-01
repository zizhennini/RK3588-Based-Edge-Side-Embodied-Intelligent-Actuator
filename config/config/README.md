# config — 配置文件

所有可配置参数集中管理。

## 文件

| 文件 | 说明 |
|------|------|
| `settings.py` | 相机内参/串口/VLM 模型/设备索引 |

## 参数一览

| 参数 | 类型 | 说明 |
|------|------|------|
| `CAMERA_MATRIX` | np.ndarray (3×3) | 相机内参（需标定） |
| `SERIAL_PORT` | str | 机械臂串口路径 |
| `SERIAL_BAUD` | int | 串口波特率 |
| `VLM_MODEL_NAME` | str | 模型名：`internvl2-1b` / `qwen2.5-vl-3b` |
| `RKLLM_BIN` | str | rkllm_demo 路径 |
| `VLM_MODEL_PATH` | str | .rkllm 模型文件路径 |
| `ASTRA_RGB_INDEX` | int | RGB 相机索引 |
| `ASTRA_DEPTH_INDEX` | int | Depth 相机索引 |
