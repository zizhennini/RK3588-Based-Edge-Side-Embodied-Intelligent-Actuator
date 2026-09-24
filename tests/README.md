# tests — 单元测试

| 文件 | 测试内容 | 状态 |
|------|---------|------|
| `test_kinematics.py` | `policy/kinematics.py`: FK/IK 严格互逆往返（<1mm）、工作空间钳制、不可达点钳制、关节限位 | 6/6 通过（仅依赖 numpy，PC/板端均可跑） |
| `test_vlm.py` | VLM 框架接口: 模拟引擎 + 结果解析 | 需 VLM 环境（板端） |

## 相关自检（不在 tests/ 下）

| 位置 | 内容 |
|------|------|
| `runtime/shared_frame.py` | SharedFrameBuffer 共享内存帧缓冲往返自检（seqlock / numpy 视图 / 数据完整性），仅依赖 numpy |

## 运行

```bash
python tests/test_kinematics.py    # 运动学 FK/IK 一致性
python tests/test_vlm.py           # VLM 接口（板端）
python runtime/shared_frame.py     # 共享内存帧缓冲自检
```

## 已移除

- `test_ik.py` / `test_locator.py` — v0.2.0 架构重构时被 `test_kinematics.py` 替代（旧 `ArmController._ik` 与旧 ColorLocator 已废弃）
