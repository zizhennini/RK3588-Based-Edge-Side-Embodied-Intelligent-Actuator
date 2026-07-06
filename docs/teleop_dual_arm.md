# 双臂遥操作 — 北通蝙蝠4 (BD4A) 手柄控制

## 硬件连接

| 组件 | 接口 |
|------|------|
| 北通蝙蝠4 无线接收器 | USB |
| SO101 主臂 | `/dev/ttyACM1` |
| SO101 从臂 | `/dev/ttyACM1`（可配置） |

## 启动

```bash
conda activate rkvla
cd /home/elf/work/RK3588-EIA

# 单臂模式（默认）
python3 scripts/teleop_dual_arm.py

# 双臂模式
python3 scripts/teleop_dual_arm.py --arm2-port /dev/ttyACM2
```

## 控制映射

### 臂1（主臂）

| 手柄操作 | 控制量 | 说明 |
|---------|--------|------|
| 左摇杆 ←→ | X 轴平移 | 左摇杆左右 |
| 左摇杆 ↑↓ | Y 轴平移 | 左摇杆上下 |
| 十字键 ↑ | Z 轴上升 | 上推 |
| 十字键 ↓ | Z 轴下降 | 下推 |
| LB | 夹爪开关 | 按一次切换开/关 |

### 臂2（从臂）

| 手柄操作 | 控制量 | 说明 |
|---------|--------|------|
| 右摇杆 ←→ | X 轴平移 | 右摇杆左右 |
| 右摇杆 ↑↓ | Y 轴平移 | 右摇杆上下 |
| LT | Z 轴上升 | 左扳机 |
| RT | Z 轴下降 | 右扳机 |
| RB | 夹爪开关 | 按一次切换开/关 |

### 功能键

| 按键 | 功能 |
|------|------|
| Start | 双臂归零位 |
| Select | 紧急停止（关机力矩） |
| Home | 退出程序 |
| C (Turbo) | （预留） |
| Z (Shift) | （预留） |

## 速度参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 控制频率 | 50Hz | 每 20ms 一次控制循环 |
| 最大移动速度 | 0.08 m/s | 摇杆推满时的线速度 |
| 死区 | 15% | 摇杆中心 ±15% 忽略 |

## 软件架构

```
GamepadReader (evdev) ──→ DualArmTeleop ──→ ArmController × 2
                           │
                           ├── _map_controls()  手柄→笛卡尔增量
                           ├── _execute()       发送 IK 指今
                           └── run()            主循环 50Hz
```

### 核心类

| 类 | 文件 | 说明 |
|------|------|------|
| `GamepadReader` | `teleop_dual_arm.py` | 非阻塞轮询手柄状态 |
| `DualArmTeleop` | `teleop_dual_arm.py` | 双臂控制编排 |
| `ArmController` | `vla/control/controller.py` | IK + Feetech 串口 |

## 串口协议

Feetech STS3215 舵机，波特率 1000000bps。

```python
packet = [0xFF, 0xFF, ID, LEN, INST, ADDR, DATA..., CKSUM]
CKSUM = ~sum(packet[2:]) & 0xFF
```

## 安全机制

- `emergency_stop()` — 清空串口缓冲区 + 关闭全部舵机力矩
- Ctrl+C — 自动急停 + 关闭串口
- Select 键 — 物理急停
- 关节限位钳制 — 超限自动截断
