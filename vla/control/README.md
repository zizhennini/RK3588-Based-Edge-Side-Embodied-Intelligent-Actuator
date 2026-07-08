# vla/control — 机械臂控制

SO-ARM101 6-DOF 机械臂控制，含串联几何逆运动学 + Feetech STS3215 串口协议。

## 文件

| 文件 | 说明 |
|------|------|
| `controller.py` | `ArmController` — IK + 串口控制 + emergency_stop |

## 使用

```python
from vla.control import ArmController
from config.cpu_affinity import LITTLE_CORES

arm = ArmController("/dev/ttyUSB0", 1000000, bind_little=True)  # 绑小核
arm.move_to(0.25, 0.0, 0.15)   # 移动到 3D 坐标（自动 IK）
arm.gripper(False)              # 闭合夹爪
arm.emergency_stop()            # 紧急停止：清空串口+关闭力矩
arm.close()
```

## 功能

- 闭式解析 IK（6-DOF 专用）
- Feetech 串口协议（1000000bps）
- `emergency_stop()` — 清空串口缓冲区 + 关闭全部舵机力矩
- 绑小核 (Cortex-A55, ID 4-7)
- 6 关节 + 夹爪控制

## 注意事项

- 串口波特率：1000000（STS3215 默认）
- 舵机 ID：1（肩部旋转）~ 6（夹爪）
