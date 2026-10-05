# 与官方 / 开源实现的对齐审计

> 目的：确认本仓库中**自研的部分不是"乱来"**——载荷最重的设计逐条对照官方
> lerobot 与主流开源项目的实现，给出代码行级证据；同时明确哪些是**有意偏离**。
> 审计环境：WSL2 `rk3588` env，**lerobot 0.6.1**（安装于 site-packages，路径见下）。
> 日期：2026-10-05。

## 1. 数据集语义（最关键，错了会崩或学出反向策略）

| 官方代码 | 官方行为 | 本仓库实现 | 结论 |
|---|---|---|---|
| `robots/so_follower/so_follower.py:180 get_observation()` → `sync_read("Present_Position")` | `observation.state` = **从臂实际位置** | `observation.state ← F*`（从臂实际角度） | ✅ 一致 |
| `teleoperators/so_leader/so_leader.py:146 get_action()` → `sync_read("Present_Position")` | `action` = **主臂位置（遥操作动作）** | `action ← J*`（主臂指令） | ✅ 一致 |
| `robots/so_follower/so_follower.py:229 sync_write("Goal_Position", goal_pos)` | 动作最终写为从臂目标 | 同 | ✅ |

**背景**：本仓库最初把 `J*` 写成了 `observation.state` 且**完全没有 `action`** —— 训练在
`policies/act/modeling_act.py` 以 `action_feature=None → NoneType.shape` 崩溃；成因是
LeRobot 由数据集特征键推断 `input_features`/`output_features`（`policy/factory.py:292,304`，
`action` → `FeatureType.ACTION`）。已修正，并在此得到官方实现证实。

## 2. 关节顺序（顺序错会静默学乱）

官方 `robots/so_follower/so_follower.py:53-59`：

```
shoulder_pan(1) shoulder_lift(2) elbow_flex(3) wrist_flex(4) wrist_roll(5) gripper(6)
```

本仓库 `scripts/json_to_lerobot.py:47 JOINT_NAMES`：**完全相同的顺序**；
`J1..J6` 即总线 motor ID 1..6。 ✅ 一致

官方特征命名为 `{motor}.pos`（`so_follower.py:67 _motors_ft()`），由录制器组装为数据集里的
6 维向量；我们的 `observation.state`/`action` 也是 6 维向量 + `names=JOINT_NAMES` 元数据 ✅ 一致。

## 3. "相机覆盖/可见性"这件事：官方与开源**都做，但方式不同**

| 来源 | 具体做法 |
|---|---|
| **官方 lerobot 录制循环** `scripts/lerobot_record.py:353` | **运行时速率检查**：录制频率低于目标 FPS 即警告 `"Record loop is running slower ... Dataset frames might be dropped … Common causes are: 1) Camera FPS not keeping up …"` |
| ALOHA 2 | `"Sessions are automatically shut down for missing data"`（数据缺失直接停机） |
| Mobile ALOHA | `if freq_mean < 30: re-collecting`（帧率不足整集重采） |
| DROID | 采集前核对屏幕上有恰好 6 路图像；把 `"camera cannot see robot"` 列为常见错误 |
| NVIDIA SO-101 工作坊 | `"try picking up vials … only using the camera views, not your eyes"`（人工覆盖度测试） |

**结论**：业界做法 = **① 运行时数据检查 + ② 一次快速人工目视测试**，**没有**用几何位姿扫描。
因此本仓库据此调整定位：

- **不作为门槛**：`tools/check_camera_coverage.py`（8 位姿边界验收）降级为**可选预检**。
  它测的是"机械臂机械极限包络"，通常远大于任务实际范围，很容易给出与任务无关的"不合格"。
  若要预检，应改成**任务相关位姿**（取物点/放置点/最高抬升，3~4 个即可）。
- **正式判据**：**试采 3 条 + 审核卡片判读**（等价于业界做法 ①，且本仓库比官方更严：
  帧率/丢帧/追踪误差/时长/**陈旧图像** 五道门控 + 逐帧原图 + 审核卡片 + manifest）。

## 4. 本仓库真正"自研"的部分及其依据

| 自研点 | 依据（官方或实测） |
|---|---|
| 板端 **ONNX Runtime CPU**（而非 RKNN） | 实测端到端 **619 ms**（vision 364 + transformer 152），chunk=100 步按 30Hz 需 3.33 s → **约 5 倍余量**；官方 RK3588 bundle 亦走 CPU（实测 ~470/570 ms） |
| 归一化统计**外置**到 `act_config.json` | 官方帧差异 bundle 同样外置为 2 个 stats 文件、用 numpy 复现（板端零 torch） |
| 逐帧记录**相机真实时间戳** + **陈旧帧护栏** | 比 lerobot 更严（其实现用主机 `perf_counter`、无新鲜度门控）；实测 front 中位 14.7 ms / wrist 28.8 ms |
| **vision_encoder + transformer 拆分导出** | 与 RDK / IB_Robot 的拆分结构一致；**官方 lerobot 没有任何 ONNX 导出实现**（包内无 `*onnx*`、无 `torch.onnx.export`），故必须自研 |
| 相机官方采集选项（AE Priority=0 / PowerLine=50Hz / laser=0） | RealSense 官方数据手册与 librealsense 源码（见 `realsense_d435i_official_survey.md`） |

## 5. 有意偏离（需知悉，勿混用）

| 项 | 官方 | 本仓库 | 影响 |
|---|---|---|---|
| **关节单位** | 归一化电机单位：本体 `DEGREES`，**夹爪单独 `RANGE_0_100`**（`so_follower.py:50,59`） | **统一弧度**（含夹爪） | 对训练无影响（ACT 对 state/action 一律 MEAN_STD，只要训练与推理约定一致）；**但不能与官方 SO-101 数据集直接混用**，需一次单位换算 |
| 录制频率检查 | 仅警告 | 门控（不达标判失败） | 更严，可能拒掉官方会接受的数据 |
| 相机数量 | SO-101 官方数据集 2 路（固定+腕部） | 相同 2 路 | 一致 ✅ |

## 6. 复现方式

```bash
# 审计所依据的官方代码位置
python -c "import lerobot,os;print(os.path.dirname(lerobot.__file__))"
# 关注文件：
#   robots/so_follower/so_follower.py        （motor 顺序 / get_observation / send_action）
#   teleoperators/so_leader/so_leader.py     （get_action）
#   scripts/lerobot_record.py                （运行时速率检查）
#   policies/factory.py                      （由数据集特征推断 input/output features）
#   policies/act/modeling_act.py             （action_feature 用法）
```
