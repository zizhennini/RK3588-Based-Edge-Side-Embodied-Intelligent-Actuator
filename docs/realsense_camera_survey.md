# 优秀开源机器人项目中的 RealSense / 多相机用法调研
### —— 面向「模仿学习数据采集的相机配置与摆放」

> 调研时间：本次会话。所有结论均附一手证据（仓库文件路径或文档链接 + 原文摘录）。
> 标注规则：**（已核实）** = 本次直接读到一手文件原文；**（未核实）** = 未能取到一手材料，仅作提示。
>
> ⚠️ 本次会话 `curl.exe` 因 schannel 凭证问题不可用；`github.com` HTML 页面间歇性超时，因此部分证据取自 `raw.githubusercontent.com` 原文、`hf-mirror.com` 镜像与官方文档站。

---

## 0. 先纠正三个前提（重要）

你方背景里写的三条"已参考/已知"，有两条与一手材料不符：

| 你的前提 | 一手材料实际 | 影响 |
|---|---|---|
| "DROID —— Franka + **3×RealSense**" | ❌ DROID **不用 RealSense**。用的是 **2× Stereolabs ZED 2 + 1× ZED Mini**（3 个双目相机 = 屏幕上 6 路图像） | DROID 的分辨率/时间同步经验不能直接套到 D435i，但它的"3 路布局 + 标定 + 采集协议"仍然高度可复用 |
| "Mobile ALOHA 腕部 GoPro/RealSense"（你问的） | 部分对：**开源代码里 3 路全是 Logitech C922 USB 摄像头**（`idProduct=085c`），论文/后来的 ALOHA 2 才换成 RealSense。ALOHA 2 用的是 **4× D405**，分辨率 **848×480** | 论文与代码存在"型号漂移"，引用时必须区分版本 |
| "lerobot 官方 cameras/realsense 无 `rs.align`" | ✅ 正确，且比你说的更彻底：连 `rs.align` 的调用点都不存在，`use_depth=False` 是默认值 | 这条可以放心沿用 |

---

## 1. 项目对比表

### 1.1 主表

| 项目 | 相机数量与位置 | 分辨率 / 帧率 | 型号 | 曝光 / 白平衡 | 时间戳与同步 | 数据格式 | 证据 |
|---|---|---|---|---|---|---|---|
| **ALOHA（原始，Stanford）** | **4 路**：`cam_high`（顶部俯视）、`cam_low`（前方低视角）、`cam_left_wrist`、`cam_right_wrist` | 640×480 @ **60**（相机采集）；**50 Hz**（记录/控制循环） | Logitech C922 USB 摄像头（代码用 `usb_cam`，`idProduct 085c`） | `autoexposure=true`（全程自动曝光）；`autofocus=false` + **手动锁焦**：high=5，low=35，双腕=40 | 每相机一个 ROS subscriber，只缓存"最新一帧"+`header.stamp`，无双相机对齐；相机 60 fps > 控制 50 Hz 保证帧新鲜度 | HDF5，JPEG quality=50（试过 20 也可） | [`launch/4arms_teleop.launch`](https://raw.githubusercontent.com/tonyzhaozh/aloha/main/launch/4arms_teleop.launch)（已核实）；[`aloha_scripts/constants.py`](https://raw.githubusercontent.com/MarkFzp/mobile-aloha/main/aloha_scripts/constants.py)（`FPS = 50`、`DT = 0.02`） |
| **Mobile ALOHA** | **3 路**（`cam_low` 被注释掉）：`cam_high`、`cam_left_wrist`、`cam_right_wrist` | 同上 640×480 @60 采集 / 50 Hz 记录 | 同上（README 明确 `roslaunch aloha 4arms_teleop.launch` = "4 robots and **3 cameras**"） | 同上 | 同上；另加 **帧率健康检查**：`if freq_mean < 30: re-collecting...` | 同上 | [`README.md`](https://raw.githubusercontent.com/MarkFzp/mobile-aloha/main/README.md)（"Step 3: Setup 3 cameras"；"maximum 2 cameras per hub for reasonable latency"）；[`launch/4arms_teleop.launch`](https://raw.githubusercontent.com/MarkFzp/mobile-aloha/main/launch/4arms_teleop.launch)（`usb_cam_low` 整段被 `<!-- -->` 注释）；[`record_episodes.py`](https://raw.githubusercontent.com/MarkFzp/mobile-aloha/main/aloha_scripts/record_episodes.py) |
| **ALOHA 2（Google DeepMind）** | **4 路**：**overhead**（顶部俯视）、**worms-eye**（虫眼，仰视）、**left wrist**、**right wrist** | **848 × 480** RGB（四路统一） | **4× Intel RealSense D405** | 文档未展开（D405 走 Stereo Module，无独立 RGB 传感器） | ROS 2；"sensor availability and **latency** 对操作员可见"；"**Sessions are automatically shut down for missing data**"（缺帧直接废会话） | 记录 leader/follower 关节数据 @ **50 Hz** + 图像流 | [arXiv:2405.02292 §2.5 Cameras / Fig.6](https://arxiv.org/html/2405.02292v1)（已核实） |
| **DROID** | **3 路双目**：`exterior_image_1` + `exterior_image_2`（两侧**可调三脚架/桌面夹持**，位置**刻意随机化**）+ `wrist_image`（ZED Mini 装在 Franka link8 上） | 公开 RLDS 为 **180×320×3 uint8**；原始数据是 **full-HD 双目 MP4 + `.svo`**；动作控制频率 **15 Hz**；相机 fps 未取到一手值**（未核实，推测 30）** | **2× ZED 2 + 1× ZED Mini**（StereoLabs，非 RealSense） | 未提及曝光锁定；反而把"改变房间光照"当作**主动增广**手段 | 论文称 "three **synchronized** stereo camera streams"；RLDS schema **无逐相机时间戳字段**，只有统一 step；原始 `.svo` 保留 ZED 硬件时间戳供事后使用 | RLDS（tf.Image 180×320×3）；原始 = `trajectory.h5`（低维）+ MP4（HD 双目）+ SVO | 购物单 [shopping-list](https://droid-dataset.github.io/droid/hardware-setup/shopping-list.html)（"Zed 2 ×2 / Zed Mini ×1 / Charuco ×1"）；[assembly.html](https://droid-dataset.github.io/droid/hardware-setup/assembly.html)；[the-droid-dataset](https://droid-dataset.github.io/droid/the-droid-dataset.html)；[data-collection](https://droid-dataset.github.io/droid/example-workflows/data-collection.html)；[arXiv:2403.12945 §III](https://arxiv.org/html/2403.12945v2) |
| **lerobot `RealSenseCamera`** | 单相机类，数量由上层配置决定 | 默认**不指定**宽高/fps（`0,0,0` → 用设备默认 profile）；文档示例给出 `1280×720@30`、`640×480@60` | 任意 D4xx（含 D435i）；D405 有专门兼容性说明 | **支持锁定**：`exposure` / `gain` / `white_balance` 任一非 None → **先关自动曝光/自动白平衡**再写死值；不设则完全不动 | 独立后台读线程 + `latest_color_frame`；时间戳用**主机 `time.perf_counter()`**，**不是设备时间戳**；无跨相机对齐；`read_latest(max_age_ms=500)` 做新鲜度护栏 | numpy RGB (H,W,3) | [`camera_realsense.py`](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/cameras/realsense/camera_realsense.py)；[`configuration_realsense.py`](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/cameras/realsense/configuration_realsense.py)（均**已核实**） |
| **`realsense-ros`（Intel 官方 wrapper）** | 单节点单相机；多相机靠 `camera_namespace` + `camera_name` 区分 | 由 `*_profile` 字符串控制，默认 `0,0,0`（=设备自动选） | D4xx / D5xx | `rgb_camera.enable_auto_exposure` 默认 **true**；`depth_module.enable_auto_exposure` 默认 **true**，`depth_module.exposure=8500`、`gain=16` | `enable_sync` 默认 **false**（且语义是**单相机内部** color↔depth 同 timetag）；`depth_module.inter_cam_sync_mode` 默认 **0**（0=Default / 1=Master / 2=Slave，这是**跨相机硬件同步**开关）；`unite_imu_method` 默认 **0**（0=None/1=copy/2=linear_interpolation） | ROS 2 topic（`sensor_msgs/Image`）+ 可 JSON 的 metadata topic | [`rs_launch.py`](https://raw.githubusercontent.com/IntelRealSense/realsense-ros/ros2-development/realsense2_camera/launch/rs_launch.py)（全文已核实）；[`README.md`](https://raw.githubusercontent.com/IntelRealSense/realsense-ros/ros2-development/README.md)（已核实） |

### 1.2 lerobot 官方数据集的图像路数 / 分辨率 / fps / 时长（你问的第 3 点）

| 数据集 | `robot_type` | 相机 keys | 单帧分辨率 | fps | episode 数 / 总帧数 | 平均单集时长 | 证据 |
|---|---|---|---|---|---|---|---|
| `lerobot/svla_so101_pickplace` | `so100_follower` | **2 路**：`observation.images.`**`up`** + **`.side`** | 480×640×3 | **30** | 50 ep / 11939 frames | **≈239 帧 ≈ 8.0 s** | [hf-mirror README](https://hf-mirror.com/datasets/lerobot/svla_so101_pickplace/raw/main/README.md)（`meta/info.json`，已核实） |
| `lerobot/svla_so100_stacking` | `so100` | **2 路**：`observation.images.`**`top`** + **`.wrist`** | 480×640×3 | **30** | 56 ep / 22956 frames | **≈410 帧 ≈ 13.7 s** | [hf-mirror README](https://hf-mirror.com/datasets/lerobot/svla_so100_stacking/raw/main/README.md)（已核实） |
| `lerobot/aloha_static_cups_open` | `aloha` | **4 路**：`cam_high` / `cam_low` / `cam_left_wrist` / `cam_right_wrist` | 480×640×3 | **50** | 50 ep / 20000 frames | **400 帧 = 8.0 s** | [hf-mirror README](https://hf-mirror.com/datasets/lerobot/aloha_static_cups_open/raw/main/README.md)（已核实） |

**要点**：lerobot 生态里 SO-100/SO-101 的官方数据集清一色是 **2 路（一路固定 + 一路腕部）**，**480×640 @ 30 fps**，视频编码 av1/yuv420p。你方"640×480@30 + 2 路"的设定与官方 SO-101 数据集**完全同构**——这是一个很强的对齐信号。

### 1.3 第 5 个项目：NVIDIA SO-101 Sim-to-Real 工作坊（最贴近你方栈）

官方文档明确写了"为什么两路"和"怎么验证摆放"，正好补上你的 Q2/Q5：

> "Each robot workspace today is equipped with **two cameras**: **Gripper camera**: Mounted on the robot's wrist/gripper; **External camera**: Stationary camera viewing the workspace from above or the side."
>
> "**Why Two Cameras?** The gripper camera becomes occluded after the robot grasps an object like a vial. The external camera provides continuous visibility of the workspace throughout the entire manipulation sequence, ensuring the policy always has usable visual input even when the gripper camera is blocked."
>
> "camera assignment is critical to policy performance. If they are swapped (gripper cam thinks it's the external cam, or vice versa), the policy will fail."

配置原文（两路都是 `640×480 @ 30`，key 命名 `wrist` / `front`）：

```
--robot.cameras='{
  "wrist": { "type":"opencv", "index_or_path":$CAMERA_GRIPPER, "width":640, "height":480, "fps":30 },
  "front": { "type":"opencv", "index_or_path":$CAMERA_EXTERNAL, "width":640, "height":480, "fps":30 }
}'
```

证据：[docs.nvidia.com — Operating the SO-101](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/08-operating-so101.html)（已核实）

---

## 2. 五个决策问题的结论与推荐取值

### Q1. 第三人称相机装多高、多远、什么角度？

**残酷的事实：没有任何一个项目给出"高度 X cm、距离 Y cm、俯角 Z°"的硬数字。** 所有项目都给的是"经验法则 + 事后标定"。

各项目的实际做法（已核实原文）：

| 项目 | 原文 | 解读 |
|---|---|---|
| DROID | "the data collector chooses views for the 3rd person cameras that can **capture a wide range of interesting behaviors** in the scene" | 不看"高度角度"，看"覆盖行为空间" |
| DROID | "The clamp and camera position of the stands should be **randomized as much as possible** during data collection. **Don't fix the stand to a single position each time.**" | 位置本身当作增广维度 → 换来 viewpoint 泛化 |
| DROID | "Whenever setting up the camera stand, **tighten the clamp and all joints as much as possible** to prevent the camera from shaking"（理由：机器人运动会晃动桌子） | 固定性 > 精确性 |
| ALOHA 2 | overhead（顶部俯视）+ **worms-eye（虫眼，从桌面附近仰视）** + 双腕 | "俯视"不是唯一选择，**低角度仰视**能补俯视的死角（尤其夹爪下方） |
| ALOHA 2 | "mount points for the overhead and worms-eye cameras" 在 20×20mm 铝型材框架上；桌面 48″×30″（**1.22 m × 0.76 m**） | 相机装在**工作台自带的框架**上，而不是独立三脚架 → 桌面晃不晃无所谓 |
| NVIDIA SO-101 | "External camera: Stationary camera viewing the workspace from **above or the side**" | "上方或侧面"二选一 |

**可用的工程推导（本人推导，非项目原值）**：

设工作空间宽度 `W`、需要的边距系数 `m`（建议 m ≥ 0.3，即画面里工作空间只占 ~70% 宽），相机水平视场角 `θh`：

```
d ≈ W·(1 + 2m) / (2·tan(θh/2))
```

- 你方 SO-ARM101 工作空间约 `W ≈ 0.4~0.5 m`。若 θh 取 60°（保守），则 `d ≈ 0.4×1.6/(2×0.577) ≈ 0.55 m`；`W=0.5` 时 `d ≈ 0.69 m`。
- **推荐取值（推导值，建议实测后微调）**：距工作空间中心 **0.6 ~ 0.9 m**，高于桌面 **0.7 ~ 1.1 m**，俯角 **30° ~ 45°**，方位**略偏操作者一侧 15°~30°**（避免正对时机械臂自身遮挡夹爪）。
- **D435i 的视场角请务必用命令实测，不要抄博客**（本次未取到 Intel 一手 datasheet，DNS 解析失败；社区常引用值 RGB ≈ 69.4°×42.5° @16:9、depth ≈ 87°×58°，**属未核实**）：

```bash
rs-enumerate-devices -c        # 打印每个 stream 的 intrinsics，含 fx/fy → 反推 FOV
```

> ⚠️ **注意 D435i RGB 的原生宽高比是 16:9**。你请求 640×480（4:3）时 RGB 流通常会被**裁剪或缩放到 4:3**，水平 FOV 会比 16:9 时**明显变窄**。这直接决定"相机要放多远"，必须先实测再用上面的公式。

---

### Q2. 几路相机是主流？各自适用场景与代价

统计（全部已核实）：

| 路数 | 项目 | 典型场景 |
|---|---|---|
| **1 路** | — 主流项目里**没有**纯 1 路的模仿学习配置（仅在工具型采集如 DobbE 中出现，DROID 论文明确批评："limits the data to wrist camera viewpoints"） | 不推荐作为唯一视角 |
| **2 路（固定 + 腕部）** | `svla_so100_stacking`（top+wrist）、`svla_so101_pickplace`（up+side）、NVIDIA SO-101 官方工作坊（external+gripper） | **SO-100/SO-101 单臂的事实标准**；ACT 等轻量策略的主流 |
| **3 路（固定 + 双腕）** | Mobile ALOHA（high + 双腕）、DROID（2 exterior + 1 wrist） | **双臂**标准；Mobile ALOHA 就是 2 腕 + 1 顶 |
| **4 路（顶部 + 低角 + 双腕）** | 原始 ALOHA（high/low/双腕）、ALOHA 2（overhead/worms-eye/双腕）、`aloha_static_cups_open` | 泛化性最好、成本最高 |

**结论与代价**：

- **你方是双臂 SO-ARM101**，按"每臂一个腕部视角 + 至少一个固定第三人称"的原则，**3 路是本配置的正统答案**（= Mobile ALOHA 同构）。但你现在是 **2 路**（D435i 固定 + 1 个腕部 USB），这在本届 lerobot SO-101 生态里也完全站得住（官方 SO-101 数据集就是 2 路）。
- 代价对比：
  - 加第 2 个腕部相机 → 数据量 +50%，RK3588 上的编码/写盘压力 +50%；但**单臂任务收益有限**，双臂任务收益明显（另一只手的遮挡无法从单腕视角恢复）。
  - 固定视角**单路**的风险：夹爪抓取后腕部相机被遮住时（NVIDIA 文档明确指出这一点），唯一可用视觉就只剩固定视角——**所以固定视角必须是"全局可用"的那个，不能省**。
- **推荐**：先做 **2 路**跑通全链路（与官方 SO-101 数据集对齐），**预留第 3 路**（第二腕部）的接口和存储带宽。

---

### Q3. 曝光/白平衡：自动 + 预热，还是锁定？

**主流是"自动曝光 + 预热"；但所有成熟栈都提供了锁定接口，且锁定是"可选增强"而非默认。**

证据（全部已核实）：

| 做法 | 证据 |
|---|---|
| **自动曝光是默认** | ALOHA/Mobile ALOHA：`usb_cam` 的 `autoexposure` 参数**显式写 `true`**（四个相机节点全部如此）。`realsense-ros`：`rgb_camera.enable_auto_exposure` 默认 **`true`** |
| **预热是默认** | lerobot：`warmup_s: int = 1`，且 `_run_warmup()` 里强制 `self.warmup_s = max(self.warmup_s, 1)`，注释原文：*"Enforcing at least one second of warmup as RS cameras need a bit of time before the first read. If we don't wait, the first read from the warmup will raise."* Mobile ALOHA：`ImageRecorder.__init__` 末尾 `time.sleep(0.5)` |
| **锁定方法（lerobot 给得最清楚）** | `RealSenseCameraConfig(..., exposure=N, gain=N, white_balance=N)`。源码逻辑：只要 exposure 或 gain 非 None → 先 `enable_auto_exposure = 0` 并打日志 `"auto-exposure disabled."`，再写值；white_balance 非 None → 先关自动白平衡。文档原文：*"Manual exposure value for the color sensor. When set, auto-exposure is disabled and this fixed value is used."* |
| **锁定的理由** | 源码注释：*"which also freezes exposure at its current value when no exposure is configured"*（只设 gain 会把曝光冻结在当前值）。更根本的理由是**模仿学习要求跨 episode 的像素统计一致**——自动曝光在"物体进入画面/手挡住"时会改变亮度，等价于给策略注入了与动作无关的噪声。这一点**没有任何项目用文字明确写出来**（未核实），属推论 |
| ⚠️ **D405 的坑（对你方不适用但有参考价值）** | lerobot 源码：手动色彩控制**只对拥有独立 "RGB Camera" sensor 的机型生效**。D405 的彩色流来自共享的 "Stereo Module"，所以会抛错：*"manual color controls require a dedicated 'RGB Camera' module, which this camera does not have."* → **D435i 有独立 RGB 模块，可以锁** |
| ⚠️ **`realsense-ros` 的暴露不完整** | `rs_launch.py` 只暴露了 `rgb_camera.enable_auto_exposure`，**没有** `rgb_camera.exposure` / `rgb_camera.gain`（只有 `depth_module.exposure/gain`）。README 说"sensor inner parameters 可运行时修改"，所以 `ros2 param set /camera/camera rgb_camera.exposure <N>` 理论可用，但**未文档化，属未核实** |
| **锁定的操作顺序（关键）** | 必须先关自动、再写值；且**锁定值要在"采集时真实光照 + 目标物在画面中"的状态下读取**，否则会全黑/全白。lerobot 的做法是"在 `connect()` 里预热 → 期间自动曝光收敛 → 之后你再决定是否锁" |

**推荐（针对你方 D435i + 末端 USB 相机）**：

1. **第一阶段（跑通）**：保持自动曝光 + 强制预热（D435i 采 `warmup_s >= 1~2 s`，USB 相机建议 `time.sleep(0.5~2)` 并丢掉前 N 帧）。这是 ALOHA/Mobile ALOHA/lerobot 的共同默认，风险最低。
2. **第二阶段（提升一致性）**：在**固定光照**（关窗、恒定补光灯）下，(a) 让自动曝光收敛 2~3 s，(b) 读出当前 `exposure`/`gain`/`white_balance` 值，(c) 用这些值**锁死**，并**在本次采集全程保持不变**。
3. **必须记录**到数据集 metadata 里：是否锁定、锁定值、光源类型。DROID 之所以能把"改变房间光照"当作增广，是因为它把这些都记进了 metadata。
4. ⚠️ **末端 USB 相机与 D435i 的自动曝光会各自独立振荡**，两路画面的亮度会不同步漂移——这是**锁定曝光最有价值的场景**。

---

### Q4. 时间同步：主流怎么处理？

**主流 = "同一循环里逐相机取最新帧"，辅以新鲜度护栏。硬件同步几乎没人用，时间戳最近邻对齐只有离线数据集才做。**

| 层次 | 做法 | 证据（已核实） |
|---|---|---|
| **① 最新帧 + 单循环（绝对主流）** | lerobot：每相机一个后台线程写 `latest_color_frame`，主循环 `async_read(timeout_ms=200)` 拿最新帧；`read_latest(max_age_ms=500)` 做"太旧就报错"的护栏。Mobile ALOHA：ROS subscriber 回调存最新帧，`get_images()` 一次取全部相机的最新帧 | `camera_realsense.py`；`robot_utils.py` |
| **② 相机帧率 > 控制频率（保证新鲜度）** | ALOHA：相机 **60 fps** vs 控制 **50 Hz** → 最坏 16.7 ms 陈旧。lerobot：`read_latest` 默认 `max_age_ms=500` 兜底 | `4arms_teleop.launch`（`framerate 60`）；`constants.py`（`FPS = 50`） |
| **③ 帧率健康检查 / 缺帧直接废数据** | Mobile ALOHA：`if freq_mean < 30: print('freq_mean is X, lower than 30, re-collecting...')` → **重采**。ALOHA 2：*"Sessions are **automatically shut down for missing data** to ensure downstream learning pipelines always receive complete data."* 且 *"sensor availability and **latency** are visible to the operator during collection"* | `record_episodes.py`；ALOHA 2 论文 §3 |
| **④ 单相机内部 color↔depth 同步** | `realsense-ros` 的 `enable_sync`：*"gathers closest frames of different sensors, infra red, color and depth, to be sent with the **same timetag**"*，且 *"let librealsense sync between frames, and get the frameset with color and depth images combined"*。**默认 `false`** | `README.md`；`rs_launch.py` |
| **⑤ 跨相机硬件同步（极少用）** | `realsense-ros` 有 `depth_module.inter_cam_sync_mode`，默认 `0`，取值 `[0-Default, 1-Master, 2-Slave]`。注意：D4xx 系列需要**物理连线**（多机同步线）才能生效，单靠软件设参不够**（未核实具体接线要求）** | `rs_launch.py` |
| **⑥ 时间戳来源** | **没有项目使用设备时间戳做对齐。** lerobot 用的是主机 `time.perf_counter()`；Mobile ALOHA 记录了 `header.stamp` 但**只用于 debug**，且代码有 bug：`data.header.stamp.secs + data.header.stamp.secs * 1e-9`（本应是 `nsecs`）→ 说明他们**没有认真做时间戳对齐** | `camera_realsense.py`；`robot_utils.py` |
| **⑦ "多相机无共同时间戳"问题的明确记录** | **没有找到任何一个项目用文字承认并解决这个问题。** 最接近的：DROID 原始数据保留 `.svo`（含 ZED 硬件时间戳）供"想用的人"事后处理；而公开 RLDS **没有逐相机时间戳字段**，只有一个全局 step。`realsense-ros` 里有一个相关但不同的问题记录：*"there is a time gap between the moment the image arrives at the wrapper and the moment the image is published... a situation is created where an image with earlier timestamp is published after Imu message with later timestamp... **Note that in either case, the timestamp in each message's header reflects the time of it's origin.**"* | `the-droid-dataset.html`；`README.md` |
| ⚠️ **`global_time_enabled`** | 在**当前** `rs_launch.py` 的完整参数列表（约 80 项）和**当前** README 中**都没有出现**。老版本 `ros2-legacy` 是否有此参数**本次未能核实**。不要基于它设计架构 | `rs_launch.py`（全文已读） |

**推荐（针对你方 ACT @ 640×480@30、D435i + USB 相机、RK3588）**：

1. **架构选 ① + ② + ③**：单采集循环里逐相机取"最新帧"，相机帧率 ≥ 30 且**略高于**控制频率（例如相机 30 fps、采集循环 25~28 Hz，或相机 60 fps、循环 30 Hz）。**不要试图让相机帧率等于控制频率**。
2. **在每帧数据里同时记录一个主机时间戳 + 每个相机的"帧到达时间"**。即使不做对齐，也要能**事后诊断抖动**。ALK 式做法：记录 `t_loop_start`、每个相机的 `t_arrival`、`frame_age_ms`。
3. **加硬性护栏**：`frame_age_ms > 100 ms` → 报警；`> 200 ms` → **丢弃该 episode**（对齐 ALOHA 2 的 "sessions automatically shut down for missing data" 与 Mobile ALOHA 的 `freq_mean < 30 → 重采`）。
4. **RK3588 上不要开跨相机硬件同步**：需要额外同步线，收益对 ACT 这种 30 Hz 离线策略极低；而且 D435i + 第三方 USB 相机本来就**无法硬件同步**。
5. **不要用 `rs.align`**：你只用 RGB，align 只影响 depth 输出，白白吃 CPU。与 lerobot 一致。
6. **不要开 `enable_sync`**（对 RGB-only 无意义，只增加延迟）；**不要开 `unite_imu_method`**（ACT 不用 IMU，开了只是白花带宽）。
7. 若同步需求上升，退路是**离线软对齐**：给每帧记录主机时间戳，训练前把各相机重采样到统一时间轴（最近邻或线性插值）。目前没有开源项目这么做，但这是标准做法。

---

### Q5. 如何避免自遮挡 / 末端出画面？

**有一个项目给出了明确的机制性答案（NVIDIA SO-101），其余项目给的是"检查清单"式做法。没有任何项目给出"用夹爪走到工作空间边界做覆盖度检查"的自动化脚本。**

| 类别 | 内容 | 证据（已核实） |
|---|---|---|
| **最明确的机制** | NVIDIA SO-101：**"The gripper camera becomes occluded after the robot grasps an object like a vial. The external camera provides continuous visibility of the workspace throughout the entire manipulation sequence, ensuring the policy always has usable visual input even when the gripper camera is blocked."** → **固定第三人称相机的首要职责就是"腕部被遮挡时的兜底"** | [NVIDIA SO-101](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/08-operating-so101.html) |
| **降遮挡的硬件设计** | ALOHA 2 把腕部相机换成**小体积 D405 + 3D 打印支架**，理由：*"The lower profile of the cameras on the wrists **reduces the number of collision states** and improves teleoperation for certain fine grained manipulation tasks, especially those that require close contact between the arms or navigating through tight spaces."* → **相机本体越大，越会自己遮挡自己 + 增加碰撞** | [ALOHA 2 §2.5](https://arxiv.org/html/2405.02292v1) |
| **采集前的检查流程** | DROID：*"**Press shift to see the camera feed.** Anytime you see a camera feed, confirm that there are **exactly 6 images on screen**. If there is anything different, **halt data collection**, unplug and replug the 3 camera wires, and restart the GUI."* | [data-collection.html](https://droid-dataset.github.io/droid/example-workflows/data-collection.html) |
| **协议级目标** | DROID 把防错列为采集协议的第一目标：*"(1) preventing common data collection mistakes like **'camera cannot see robot'** or **'teleoperator in camera view'**"* | [arXiv:2403.12945 §III-B](https://arxiv.org/html/2403.12945v2) |
| **畸光/反光干扰** | DROID：*"Place a small piece of thick Velcro (soft side) over the light on the gripper. Otherwise, it will **shine into the camera**."* | [assembly.html](https://droid-dataset.github.io/droid/hardware-setup/assembly.html) |
| **把"换视角"当增广** | DROID：GUI 会周期性要求 *"randomly sampled 'scene augmentations' like ... **moving and re-calibrating the 3rd person cameras**, changing the room lighting"* → 与其追求一个完美固定机位，不如**在多个位置都采一份** | [arXiv:2403.12945 §III-B](https://arxiv.org/html/2403.12945v2) |
| **人工覆盖度验证（最接近你问的"检查方法"）** | NVIDIA SO-101 的操作建议：打开 Rerun 双路视图，*"Make sure your camera views and props roughly match this setup"*，然后 **"try picking up vials and placing them in the rack only using the camera views, not your eyes. See if you can perform the task a few times."** → **"闭眼遥操作"就是覆盖度验收测试** | [NVIDIA SO-101](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/08-operating-so101.html) |
| **相机身份错配的后果** | NVIDIA SO-101：*"camera assignment is critical to policy performance. If they are swapped... the policy will fail."*；*"Camera indices may change any time cameras are unplugged or replugged... Always verify camera assignments before collecting data"* | 同上 |
| ❌ **未找到** | 没有任何项目文档化"驱动夹爪遍历工作空间边界/角落并统计覆盖度"的自动化脚本或流程。DROID 只有人工 `Shift` 看画面；ALOHA 2 只有"缺帧自动废会话" | — |

**推荐（结合项目证据，自建一套）**：

1. **把"覆盖度验收"做成一次性的、可复现的脚本化动作**（这是我在项目里没找到、但值得你补上的部分）：
   - 让从臂依次移动到工作空间的 **8 个关键位姿**：四角 ×（高位 / 贴桌低位），以及 **正对相机最近点**、**最远点**、**左右极限**。
   - 每一步停 0.5 s，同时录 D435i + 腕部相机的帧。
   - **验收标准**：(a) 8 个位姿里**夹爪指尖**都落在固定视角画面内，且**距画面边缘 ≥ 10% 画幅**；(b) 腕部相机在"抓握姿态"下画面中心仍是物体而非自身指节。
   - 这一步只需做一次（每次改动相机位置后重做），成本极低，收益极高。
2. **固定视角的硬性要求**：必须能同时看到**双臂的工作空间 + 机器人底座**（后者对 ACT 判断"物体在哪"很关键）。宁可画面里机械臂只占 1/4，也不要为了"看清夹爪"而拉近到夹爪经常出画。
3. **优先用"上方或侧上方 30°~45°"而非纯俯视**：纯俯视会丢失高度信息，且机械臂上行时会大面积遮挡工作空间。ALOHA 2 同时装 overhead + worms-eye 就是这个原因；你只有一路固定视角时，**30°~45° 的斜俯视是信息量最大的折中**。
4. **腕部相机支架要尽量小、尽量贴合夹爪**（ALOHA 2 降遮挡经验的直接引用）；并且**不要让它遮住 D435i 看夹爪的视线**——装上后立刻用第 1 步的脚本验证。
5. **两次采集之间不要动相机**；DROID 的"随机化机位"策略是为**跨场景泛化**服务的，与你**单场景 ACT** 的目标相反。你要的是**绝对固定 + 每次采前核对画面**。
6. **每次开始采集前跑一次"3 秒人工核对"**（照抄 DROID）：切换查看两路画面 → 确认机器人可见、操作者不在画面内、无强反光 → 才开始。
7. **夹爪 LED / 反光贴纸**：如果 D435i 视野里有红外/可见光干扰（D435i 有红外投射器），参考 DROID 用厚魔术贴遮光的做法；同时确认 **IR emitter 不会在你只用 RGB 时干扰**（D435i 的 IR 投射器对 RGB 基本无影响，但会在金属表面产生散斑被 RGB 视作噪点——低风险，可忽略）。

---

## 3. 对我方（D435i 固定第三人称 + 末端 USB 相机、ACT、640×480@30）的具体建议清单

### A. 立即可做（不改硬件）

- [ ] **A1. 放弃 `rs.align`，与 lerobot 保持一致。** 你只用 RGB；`align_depth.enable` 默认就是 `false`，保持不动即可。证据：[`rs_launch.py`](https://raw.githubusercontent.com/IntelRealSense/realsense-ros/ros2-development/realsense2_camera/launch/rs_launch.py)（`align_depth.enable` 默认 `false`）、[`camera_realsense.py`](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/cameras/realsense/camera_realsense.py)（全文件无 align）。
- [ ] **A2. 显式启用 D435i 的 RGB 流并显式指定 profile**，不要依赖 `0,0,0` 自动选：
  ```bash
  ros2 launch realsense2_camera rs_launch.py \
    camera_name:=d435i_front \
    enable_color:=true rgb_camera.color_profile:=640x480x30 rgb_camera.color_format:=RGB8 \
    enable_depth:=false enable_infra:=false enable_infra1:=false enable_infra2:=false \
    enable_sync:=false align_depth.enable:=false \
    enable_gyro:=false enable_accel:=false unite_imu_method:=0 \
    initial_reset:=false
  ```
  **理由**：`enable_depth` 默认是 `true`！你只用 RGB，**必须显式关掉 depth**，否则白白占用 USB 带宽和 CPU——这是最容易踩的坑之一。
- [ ] **A3. 用 `rs-enumerate-devices -c` 实测 640×480 下 RGB 的真实 fx/fy → 反推 FOV**，再据此定相机距离（见 Q1 公式）。**不要抄网上的 69.4°×42.5°**（本次未核实，且 640×480 是 4:3 会裁掉 16:9 的水平视场）。
- [ ] **A4. 加预热**：D435i `warmup ≥ 1~2 s`（lerobot 源码强制 `max(warmup_s, 1)`）；末端 USB 相机 `sleep 0.5~2 s` 并丢弃前 10~30 帧。证据：[`configuration_realsense.py`](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/cameras/realsense/configuration_realsense.py)（`warmup_s: int = 1`）。
- [ ] **A5. 加帧新鲜度护栏**：每帧记录 `frame_age_ms`；`> 200 ms` 丢弃该 episode；`> 100 ms` 告警。参考 lerobot `read_latest(max_age_ms=500)` 与 Mobile ALOHA 的 `freq_mean < 30 → 重采`。
- [ ] **A6. 采集循环频率与相机频率解耦**：让相机跑 30 fps（或 D435i 跑 60 fps），采集循环 25~30 Hz，每次取"最新帧"。不要试图做成"等齐所有相机"的阻塞同步——**没有项目这么做**。

### B. 需要一次实验（1~2 小时）

- [ ] **B1. 写一个 `check_camera_coverage.py`**（见 Q5 第 1 条）：驱动从臂遍历 8 个边界位姿，保存两路画面并打上网格与边缘 10% 禁区框。**这是我方当前最缺、且成本最低的一环。**
- [ ] **B2. 曝光策略 A/B**：在固定光照下采两条同任务数据，(a) 自动曝光 + 预热，(b) 锁定曝光/增益/白平衡；对比 ACT 训练 loss 曲线与 rollout 成功率。**这是唯一能回答"你方要不要锁曝光"的方法**——项目里没人给出结论。
- [ ] **B3. `enable_depth:=false` vs `true` 的 USB 带宽与 CPU 占用对比**（在 RK3588 上用 `dmesg | grep -i "usb\|bandwidth"` + `top`/`tegrastats`）。DROID 的经验是"**直接插主机 USB 口，绝不用 Many2One Hub 或延长线**"（否则 Segmentation Fault），Mobile ALOHA 的经验是"**每个 hub 最多 2 个相机**"。你方 D435i + 1 个 USB 相机应**各自独占一个 USB 控制器**。

### C. 架构与数据格式

- [ ] **C1. 数据集 key 命名对齐官方 SO-101**：用 `observation.images.front`（或 `.top`）+ `observation.images.wrist`，与 `lerobot/svla_so100_stacking`（`top`+`wrist`）和 NVIDIA SO-101（`front`+`wrist`）保持一致 → 便于日后直接复用 lerobot/openpi 的转换脚本与预训练权重。
- [ ] **C2. 分辨率/帧率直接用 640×480 @ 30 fps**。这与 `lerobot/svla_so101_pickplace`、`lerobot/svla_so100_stacking`（均 480×640 @30）**完全一致**，是本次调研中最强的"你就是主流配置"的证据。
- [ ] **C3. 单集时长目标 8~15 s**：官方 SO-101 数据集平均 **8.0 s**（pickplace）/ **13.7 s**（stacking）；ALOHA 400 帧 @50 fps = 8 s。Mobile ALOHA 的 `episode_len` 配置跨度 1000~8500 步（20~170 s @50 Hz），属于长程任务，**不要照抄**。
- [ ] **C4. 视频编码**：官方数据集用 **av1 / yuv420p**；Mobile ALOHA 用 JPEG q=50（"tried as low as 20, seems fine"）。RK3588 有硬件 AV1 编码器，**优先用 AV1 硬编**；若不可用，JPEG q=50 是可接受的回退（Mobile ALOHA 实证）。
- [ ] **C5. 在 metadata 里记录**：`exposure_locked: bool`、`exposure/gain/white_balance` 值、`camera_serial`、`camera_position_label`、`warmup_s`、`target_fps`、`actual_fps`。DROID 之所以能事后筛数据，就是因为它记了采集者、时间、机器人 ID。
- [ ] **C6. 预留第 3 路（第二个腕部相机）接口**，但**先不上**。理由：双臂任务的收益主要体现在第二腕视角，但 RK3588 的 USB 带宽与编码吞吐是瓶颈；先用 2 路跑通并与官方数据集对齐。

### D. 明确"不要做"的事

- [ ] **D1. 不要用 DROID 的"随机化相机位置"策略。** 那是为跨场景泛化设计的；单场景 ACT 需要**绝对固定**的机位。
- [ ] **D2. 不要开跨相机硬件同步**（`inter_cam_sync_mode`）。D435i + 第三方 USB 相机本来就无法硬件同步；ACT 在 30 Hz 离线训练下收益极低。
- [ ] **D3. 不要照抄 ALOHA 的 60 fps 相机 + 50 Hz 循环。** 那是为 50 Hz 遥操作设计的；ACT @30 fps 用 30~60 fps 相机即可。
- [ ] **D4. 不要为 D435i 的 IMU 做任何设计。** `enable_gyro`/`enable_accel` 默认 `false`，`unite_imu_method` 默认 `0`，保持关闭。
- [ ] **D5. 不要用 `global_time_enabled` 做设计假设。** 该参数在当前 `rs_launch.py`（约 80 项全量参数）与 README 中均**不存在**（老版本未核实）。

---

## 4. 未核实事项（明确标注，供后续补）

| 项 | 状态 | 建议核实方式 |
|---|---|---|
| D435i 在 640×480 RGB 下的实际 FOV | **未核实**（intelrealsense.com DNS 失败、ark.intel.com 跨域重定向、librealsense#7170 GitHub 超时）；社区常引用 RGB ≈ 69.4°×42.5°(16:9) / depth ≈ 87°×58° | `rs-enumerate-devices -c`；官方 [D400 Series Datasheet (Sept 2023)](https://www.intelrealsense.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf) |
| DROID 相机的实际采集 fps | **未核实**（`github.com/droid-dataset/droid` 页面超时）；推测 ZED 2 HD @30 | 克隆 `github.com/droid-dataset/droid` 查相机配置 |
| `rs_launch.py` 是否有 `global_time_enabled` | 当前版本**确认没有**；老版本（`ros1-legacy`/`ros2-legacy`）**未核实** | 查 `ros2-legacy` 分支的 `rs_launch.py` |
| `rgb_camera.exposure` / `rgb_camera.gain` 是否可通过 `ros2 param set` 设置 | **未核实**（`rs_launch.py` 未暴露，README 未文档化，但 README 声称 sensor 内参可运行时修改） | `ros2 param list \| grep -i rgb_camera` 实测 |
| D4xx `inter_cam_sync_mode` 的物理接线要求 | **未核实** | Intel [D457 Hardware Synchronization](https://www.intelrealsense.com/wp-content/uploads/2024/05/Intel-RealSense-D457-Hardware-Synchronization.pdf)（PDF，本次解析失败） |
| realsense-ros 官方对"多相机 USB 控制器/带宽"的建议 | **未核实**：README 中**没有**此类建议。有效的经验值来自 DROID（直插、不用 Hub/延长线）与 Mobile ALOHA（每 hub ≤ 2 相机） | 查 realsense-ros wiki |

---

## 5. 一句话总结

**你方"1 路 D435i 固定第三人称 + 1 路腕部 USB、640×480@30、ACT"的配置，与 lerobot 官方 SO-101 数据集（2 路 480×640@30）和 NVIDIA SO-101 官方工作坊（`front`+`wrist` 640×480@30）完全同构，是当前生态的主流答案，不需要为了"向大项目看齐"而加相机。** 真正需要补的是三件事：(1) **实测 640×480 下的 FOV 再定机位距离**；(2) **写一个 8 位姿覆盖度验收脚本**（这是所有项目都缺、但你最容易补上的一环）；(3) **加帧新鲜度护栏 + 在固定光照下做一次曝光锁定 A/B**。DROID 的"随机化机位"和 ALOHA 的"60 fps 相机 + 50 Hz 循环"是为不同目标设计的，**不要照抄**。
