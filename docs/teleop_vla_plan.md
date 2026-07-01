# 示教回放 + VLM 自主抓取方案

## 概述

通过遥操作采集示教轨迹 → 平滑预处理 → VLM 识别目标 → 偏差修正 → 自主抓取

## 第一阶段：示教数据采集

### 硬件准备
- 机械臂（follower）连接 `/dev/ttyACM0`
- 主臂（leader）连接 `/dev/ttyACM1`
- 上帝视角相机（overhead）固定俯拍工作区
- 局部相机（arm）装在机械臂上（可选）

### 采集流程
```
1. 人操作主臂做一次完整抓取
2. 记录每帧：
   - 6个关节角度 (J1-J6)
   - 夹爪开合状态
   - 时间戳
3. 保存为 JSON
```

### 调用方式
```bash
python scripts/record_teleop.py --follower /dev/ttyACM0 --leader /dev/ttyACM1 --out demo_grasp.json
```

## 第二阶段：轨迹平滑处理

### 算法
| 步骤 | 方法 | 参考 |
|------|------|------|
| 异常值剔除 | 中值滤波、3σ 去噪 | icar_autopilot_2025th |
| 平滑 | 滑动平均 / Savitzky–Golay 滤波 | 通用信号处理 |
| 重采样 | 线性/样条插值为固定频率 | LeRobot LERP 插值 |
| 速度约束 | 每帧最大角度变化限幅 | 项目已有 MAX_J2 |

### 参考
- `D:\桌面\icar_autopilot_2025th`：C++ 信号处理实现
- `D:\桌面\lerobot-v0.4.4`：LeRobot 数据采集流程

## 第三阶段：VLM 目标识别 + 偏差修正

### 自动执行流程
```
语音指令 "抓红色杯子"
    ↓ ASR
文字指令
    ↓ 查找匹配的示教轨迹
🔍 示教路径: demo_grasp.json
    ↓
📷 上帝视角拍照
    ↓
🧠 Qwen3-VL grounding → 输出目标物体的像素坐标 (u, v)
    ↓
📐 对比示教时物体坐标 → 计算偏移量 (Δu, Δv)
    ↓
🔄 偏移量映射到笛卡尔空间 → 修正抓取点 (x, y, z)
    ↓
🤖 按修正后的轨迹回放 → 自适应抓取
```

### 关键技术
- **VLM grounding**：Qwen3-VL 输出 bbox_2d + label（已实现）
- **坐标偏差计算**：示教时记录物体位置 → 执行时重新识别 → 计算偏移
- **轨迹修正**：将偏移量叠加到示教轨迹的笛卡尔坐标上

## 实现步骤

1. **录** — `record_teleop.py`：遥操作数据采集（参考 LeRobot）
2. **平滑** — `smooth_traj.py`：轨迹滤波+插值（参考 icar_autopilot）
3. **执行** — `replay_adaptive.py`：VLM 识别 + 偏差修正 + 轨迹回放
