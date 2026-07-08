# RK3588-EIA

**Embedded Intelligent Actuator** — 端侧具身智能教育平台

RK3588 NPU 端侧部署 VLA（Vision-Language-Action），支持语音交互 + VLM 理解 + 机械臂抓取，全链路在端侧运行，无需联网。

---

## 快速开始

```bash
conda activate rkvla
cd /home/elf/work/RK3588-EIA

# 统一菜单
python3 menu.py

# 语音唤醒
python3 va.py listen-forever --wake-mode kws

# 录音问答
python3 va.py once

# 文字问答
python3 va.py ask "1+1等于几"

# 文字问答（拍照）
python3 va.py ask "画面中有什么"

# Web 控制面板
python3 webui.py
# 浏览器打开 http://192.168.137.100:8080
```

## 功能列表

| 功能 | 命令 | 说明 |
|------|------|------|
| 语音唤醒 | `va.py listen-forever` | KWS 检测"你好同学"唤醒 |
| 语音识别 | `va.py once` | 录音 -> ASR -> VLM -> TTS |
| 文字问答 | `va.py ask <text>` | Qwen3.5 VLM 回答 + TTS 播报 |
| 遥操作 | `scripts/teleop_record.py` | 主臂拖拽 -> 从臂跟随 |
| 动作录制 | `scripts/develop_motion.py` | 录制 -> 平滑 -> 入库 |
| 动作回放 | `scripts/replay_traj.py` | 30fps 帧间插值回放 |
| 动作库管理 | `scripts/record_trajectory.py` | 查看/删除动作 |
| 硬件录像 | `scripts/recorder.py` | h264_rkmpp 编码 |
| 系统监控 | `scripts/monitor.py` | CPU/内存/NPU 实时监控 |
| 任务控制器 | `scripts/task_controller.py` | 多工序任务编排 |
| VLM 抓取 | `scripts/vlm_grasp.py` | VLM+PCA+IK 视觉抓取 |

## 语音指令（动作库）

| 说 | 动作 |
|----|------|
| "你好" | 打招呼 |
| "抬起" | 抬升动作 |
| "抓" | 抓取动作 |
| "停止" | 紧急停止 |
| "有什么动作" | 列出动作库 |
| "归零" | 归零位 |

## 核心特性

| 特性 | 说明 |
|------|------|
| 大小核算力隔离 | A76 大核(4-7)跑推理, A55 小核(0-3)跑控制 |
| 分级内存管控 | VLM 按需加载/闲置30s卸载(~900MB) |
| 指令队列+中断 | FIFO 串行执行, 紧急中断清空队列+物理停机 |
| 全离线语音 | sherpa-onnx KWS + ASR + TTS, 无需联网 |
| 流式 TTS 缓冲 | 独立线程预填缓冲, 消除音频中断 |

## 硬件

| 组件 | 型号 |
|------|------|
| SoC | Rockchip RK3588 (4xA76 + 4xA55, 6TOPS NPU, 8GB) |
| 机械臂 | SO-ARM101 6-DOF + 夹爪, Feetech STS3215 |
| 深度相机 | Intel RealSense D435i |
| 手柄 | 北通蝙蝠4 (BD4A) |

## 测试完成状态

| 模块 | 状态 |
|------|------|
| KWS 唤醒词 | 完成 |
| ASR 语音识别 | 完成 (0.17s) |
| TTS 语音合成 | 完成 (0.43s) |
| VLM 视觉问答 | 完成 (2.8s) |
| 机械臂 IK 控制 | 完成 |
| 示教录制 -> 回放 | 完成 |
| 语音触发回放 | 完成 |
| 硬件加速录像 | 完成 |
| 指令队列+中断 | 完成 |
| 分级内存管控 | 完成 |
| 大小核算力隔离 | 完成 |
| 系统资源监控 | 完成 |
| VLM 自主抓取 | 开发中 |
