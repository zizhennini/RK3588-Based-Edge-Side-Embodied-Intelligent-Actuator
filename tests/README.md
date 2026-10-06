# tests — 单元测试

| 文件 | 测试内容 | 状态 |
|------|---------|------|
| `test_feetech_bus.py` | `hardware/feetech_bus.py`: sign-magnitude 编解码、STS3215 控制表回归、EPROM 可写集合、目标突变限幅、raw↔rad 换算、无 SDK 时优雅失败 | 14 用例（仅依赖 numpy，PC/板端均可跑） |
| `test_episode_quality.py` | `tools/episode_quality.py`: 采集质量判据（帧率/丢帧/追踪误差/时长四道门槛、无效帧统计、数据集汇总、报告渲染） | 10 用例（纯逻辑，无硬件无 numpy） |
| `test_so101_kinematics.py` | `hardware/so101_kinematics.py`: 标定限位换算与结构不变式、URDF 限位注入（恰好 6 个、不污染 transmission）、mesh 剥离、运动链完整、TCP frame 正确；placo 用例含 q 布局、FK/IK 往返、**关节零点约定 vs 真实数据集** | 11 用例（8 个纯逻辑 PC 可跑；3 个需 placo，缺库自动 SKIP） |
| `test_perception.py` | `perception/cube_locator.py` 与 `perception/vlm.py`: 45° 方块不失检、拟合矩形填充率与旋转无关、机械臂样淡蓝白大块剔除、质心精度、**朝向必须用 minAreaRect（PCA 对正方形退化）**、6 色归类、色相窗口不重叠、SoM 绘制、**多行 prompt 必须压成单行** | 14 用例（依赖 opencv+numpy，PC/板端均可跑） |
| `test_kinematics.py` | `policy/kinematics.py`（**旧解析 IK，待废弃**）: FK/IK 互逆往返、工作空间钳制、关节限位 | 6 用例（仅依赖 numpy） |
| `test_vlm.py` | VLM 框架接口: 模拟引擎 + 结果解析 | 需 VLM 环境（板端） |

## 两条 IK 路线的关系（重要）

项目里同时存在两套运动学，**新代码一律用 `hardware/so101_kinematics.py`**：

| | `policy/kinematics.py`（旧，待废弃） | `hardware/so101_kinematics.py`（新） |
|---|---|---|
| 方法 | 平面 2 连杆解析解，手写 L1/L2 | placo/pinocchio 全 6 自由度数值解 |
| 限位 | 代码里硬编码常量 | 由 `config/calibration.json` 现场生成 URDF 限位 |
| 联动 | 只有关节 1-3 真正参与 IK | 全 6 关节 |
| 精度 | 板端验收未过 | 板端验收通过（P0-2，误差中位 <0.1mm，单步 0.06ms） |
| 依赖 | numpy | placo（aarch64 有预编译 wheel） |

旧模块仍被 `main.py` / `vla/control/controller.py` 引用，替换需同步改动这两处调用方，
属独立任务（见 `docs/测试方案_VLM+外部IK+抓取.md`）。

## 相关自检（不在 tests/ 下）

| 位置 | 内容 |
|------|------|
| `runtime/shared_frame.py` | SharedFrameBuffer 共享内存帧缓冲往返自检（seqlock / numpy 视图 / 数据完整性），仅依赖 numpy |
| `tools/ik_acceptance.py` | P0-c 离线 IK 数值验收（多场景 + 工作空间 + 关节零点约定），需 placo |
| `tools/cv_localization_bench.py` | P0-b CV 精定位验收（合成图精确真值 + 真实图鲁棒性），需 opencv |
| `tools/vlm_bench.py` | P0-a VLM 定位验收（直接问坐标 vs Set-of-Mark 选编号），需板端 NPU |
| `tools/p0_plots.py` | 由上三个基准的 JSON 生成 P0 三张分布图（需 matplotlib） |

## 运行

```bash
python tests/test_feetech_bus.py        # 总线协议层（无需硬件）
python tests/test_episode_quality.py    # 采集质量判据
python tests/test_so101_kinematics.py   # 运动学（placo 用例缺库自动 SKIP）
python tests/test_perception.py         # 立方体定位 + VLM prompt 处理
python tests/test_kinematics.py         # 旧解析 IK（待废弃）
python tests/test_vlm.py                # VLM 接口（板端）
python tools/ik_acceptance.py           # IK 数值验收（板端，需 placo）
python tools/cv_localization_bench.py   # CV 精定位验收（PC/板端）
python tools/vlm_bench.py               # VLM 定位验收（板端，需 NPU）
python runtime/shared_frame.py          # 共享内存帧缓冲自检
```

## 已移除

- `test_ik.py` / `test_locator.py` — v0.2.0 架构重构时被 `test_kinematics.py` 替代（旧 `ArmController._ik` 与旧 ColorLocator 已废弃）
