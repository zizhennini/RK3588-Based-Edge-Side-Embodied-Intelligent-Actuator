# scripts — 工具脚本

> 完整脚本清单以目录为准，此处收录常用脚本。标定脚本只回写
> `config/settings.py`（单一事实来源），代码层（arm.py / grasp_pipeline.py）引用不复制。

## 语音交互

| 脚本 | 说明 |
|------|------|
| `voice_motion.py` | 唤醒 → 录音 → ASR → CommandQueue → 动作/VLM（绑大核） |
| `voice_vla.py` | 语音/文字 VLA 交互（带 SmartVLM + IdleUnloader，绑大核） |
| `test_vla_voice.py` | VLA 全链路测试 |

## 标定

| 脚本 | 说明 |
|------|------|
| `calibrate_camera.py` | 相机内参标定（`--d435i` 回写 `settings.CAMERA_MATRIX`） |
| `calibrate_extrinsics.py` | 相机→机械臂基座外参标定（回写 `settings.CAMERA_POSITION`） |
| `calibrate_handeye.py` | 手眼标定辅助（像素→机器人坐标映射验证） |
| `calibrate_aruco.py` / `calibrate_quick.py` / `calibrate_tilt.py` / `calibrate_tune.py` | 其他标定辅助工具 |

## 机械臂 / 示教

| 脚本 | 说明 |
|------|------|
| `gestures.py` | 动作演示 |
| `develop_motion.py` | 示教录制 → 平滑 → 入库 |
| `teleop_dual_arm.py` | 主从双臂遥操作 |
| `record_trajectory.py` / `replay_traj.py` / `smooth_trajectory.py` | 运动库管理：录制 / 回放 / 平滑 |
| `check_gripper.py` / `off.py` | 夹爪检查 / 舵机卸载 |

## 抓取实验

| 脚本 | 说明 |
|------|------|
| `vlm_grasp.py` | VLM 视觉引导抓取完整流程（`policy/grasp_pipeline.py` 的参考实现） |
| `grasp_test.py` | 抓取测试 |
| `adaptive_grasp.py` / `visual_servo.py` / `hand_follow.py` | 自适应抓取 / 视觉伺服 / 手部跟随实验 |

## 相机

| 脚本 | 说明 |
|------|------|
| `camera_viewer.py` | USB 相机取景 |
| `d435i_viewer.py` | D435i 深度流取景 |
| `test_d435i*.py` / `test_dual_cam.py` | 相机测试系列 |

## 录像

| 脚本 | 说明 |
|------|------|
| `recorder.py` | 硬件加速录像（h264_rkmpp，支持时长与 OSD 文字） |
| `record.sh` / `record_with_view.sh` | 录像 shell 封装 |
| `record_teleop.py` / `record_task.py` | 遥操作 / 任务过程录像 |

## 系统

| 脚本 | 说明 |
|------|------|
| `monitor.py` | 系统资源监控 |
| `rkeia.service` | systemd 服务单元模板 |
| `web_demo.py` | Web 演示 |
