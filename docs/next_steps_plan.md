# 下一步规划（M1 → M4）

> 更新：2026-10-05。前置文档：`official_alignment_audit.md`（对齐审计）、
> `m2_m3_training_export_guide.md`（训练/导出/板端推理，已实测）、
> `deploy_guide.md` §8（采集/统计/审核/转换）、`realsense_*.md`（相机调研）。

## 0. 当前状态（已完成且验证）

| 环节 | 状态 | 证据 |
|---|---|---|
| 双臂协议/标定 | ✅ | 主从跟随 3593 帧 / 120 s / 29.9 fps，零错误；udev 稳定端口经重启验证 |
| M1 采集工具链 | ✅ | `collect_episodes.py`（五道质量门控 + 逐帧原图 + manifest）+ 审核卡片 + 统计 + 转换 |
| 相机接入 | ✅ | D435i(固定) + USB(腕部) 双路 640×480@30；官方采集选项已固化；逐帧真实时间戳 + 陈旧帧护栏 |
| 相机摆放 | ⏳ | 已抬到 ≈0.53~0.55 m（可见 ≈56×42 cm，0.87 mm/px）；**尚未最终锁死** |
| M2 训练环境 | ✅ | WSL2 `rk3588` env：lerobot 0.6.1 + **torch 2.11.0+cu128（GPU 可用）**；`scripts/train_act.sh` 一键 |
| M2 数据契约 | ✅ | `action←J*`/`observation.state←F*`（与官方逐行一致）；LeRobotDataset v3 转换实测通过 |
| M2 训练冒烟 | ✅ | ACT 52M，GPU **0.334 s/step**、显存 3.73 GB；loss 85.98→27.54 |
| M3 导出 | ✅ | 拆分 ONNX；数值校验端到端 **5.2e-07**；`norm_stats`/`image_mean` 已修正；opset 18 |
| M3 板端推理 | ✅ | RK3588 上 ORT 1.23.2：**619 ms 端到端（1.62 Hz）**，与 PC 参考 `max|Δ|=6.6e-07` |
| M4 交接文档 | ⏳ | 各环节文档齐；M1-M3 运行手册待补真实数据结果 |

## 1. 需要你先定的三件事

| # | 决策 | 选项 | 影响 |
|---|---|---|---|
| D1 | **任务定义** | 抓放方块（立方体簇一带，约 20×25 cm）／其他 | 决定工作区范围与是否需要抬高相机 |
| D2 | **单集时长** | 12 s（推荐，贴近官方 8.0/13.7 s）／20 s／按任务分档 | 影响每集有效信息量与采 50 条的耗时 |
| D3 | **相机取景** | A：维持现状（任务在中央区域，够用）／B：抬到 ≈0.70 m 换取全包络（分辨率降到 1.2 mm/px） | 决定是否再动相机（**动过就作废已有数据**） |

> 建议：先按 **A + 12 s + 抓放方块** 试采 3 条，用审核卡片判读"夹爪是否全程在画面内"。
> 若通过 → 直接采 50 条；若不通过 → 再按 B 调整并重采（试采数据量小，代价可接受）。

## 2. 执行计划

### 阶段 M1-α：试采 3 条（约 5 分钟）
```bash
# 板端（相机与机械臂均已就绪）
python3 tools/collect_episodes.py --task pilot --episodes 3 --episode-time 12
```
**判据（自动，工具已内建）**：manifest 中 3 条 `accepted_by_user` 之前的质量门控全绿
（帧率 ≥90% 目标、丢帧 ≤1%、追踪误差 ≤5°、时长 ≥95%、陈旧帧 ≤1%）+ 图像完整。
**人工判据**：看 `data/raw/pilot/review/episode_000X_review.jpg` → 夹爪/物体是否全程在画面内。
→ 我远程核对 manifest 与审核卡片，给"通过/需调整相机"的结论。

### 阶段 M1-β：正式采集 50 条（约 30~40 分钟，含人工审核）
```bash
python3 tools/collect_episodes.py --task pick_place --episodes 50 --episode-time 12
python3 tools/dataset_stats.py --all-tasks            # 数据集统计与健康度
```
**判据**：50 条全绿 + 标定行程覆盖率 ≥30%（多样性）+ 图像数 = 关节帧数。
→ **相机在整批采集中不得移动**（移动则前面数据作废）。

### 阶段 M2：训练（约 5 小时，GPU）
```bash
# WSL2
bash scripts/train_act.sh pick_place 50000 16 cuda
```
**判据**：loss 收敛、`checkpoints/last/pretrained_model` 生成、`meta/stats.json` 合理。
→ 我先用 5k 步短训验证数据无问题，再放长训。

### 阶段 M3：导出 + 板端推理验证（约 20 分钟）
```bash
python tools/export_act_onnx.py --checkpoint ~/train/pick_place/checkpoints/last/pretrained_model --output_dir models/act/
# 传板 → 板端 policy/act_policy.py 加载 → 实测延迟与动作合理性
```
**判据**：导出数值校验通过；板端延迟 <1 s；给相同 obs 时板端与 PC 输出一致（已在冒烟模型验证过该方法）。

### 阶段 M4：真机闭环与打磨
1. 板端策略驱动从臂跑一次（先空载/慢速，`n_action_steps` 调小）
2. 失败复盘：位置偏差 → 补数据；遮挡/出画 → 调相机或补位姿
3. 交付：运行手册（采集→训练→导出→部署）+ 数据集卡片 + 模型卡片

## 3. 风险与对策

| 风险 | 触发信号 | 对策 |
|---|---|---|
| 夹爪出画 | 审核卡片上夹爪贴边/不见 | 抬高相机（D3-B）并**重采**；或把任务收进中央区域 |
| 光照/曝光漂移 | 亮度序列漂移 | 已在固定光照下锁定选项；必要时开启 `lock_exposure/lock_white_balance` |
| 单集过长含静止尾帧 | 审核卡片后段关节曲线平直 | 缩短 `--episode-time` 或重采该条 |
| 数据多样性不足 | `dataset_stats.py` 标定行程覆盖率 <30% | 设计更分散的物体初始位置 |
| 训练不收敛/抓不住 | 训练 loss 平/真机成功率低 | 先补数据到 50~100 条；检查单位与关节顺序（已审计一致） |
| 板端延迟超标 | 端到端 >2 s | 减小 chunk、降分辨率（需重训）、或评估 RKNN fp16（见调研文档） |

## 4. 里程碑验收清单（每阶段一页纸）

- **M1**：`manifest.json` 全绿 + 审核卡片人工确认 + `dataset_stats` 健康
- **M2**：训练 loss 下降 + checkpoint 完整（`model.safetensors` + pre/postprocessor）
- **M3**：导出校验 `max_diff<1e-4` + 板端 vs PC `max|Δ|<1e-4` + 延迟 <1 s
- **M4**：真机连续成功 ≥3/5 次 + 运行手册可被他人照做复现
