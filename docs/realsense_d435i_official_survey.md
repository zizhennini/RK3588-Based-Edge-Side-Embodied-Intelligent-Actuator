# Intel RealSense D435i 用于「机器人模仿学习 RGB 数据采集」调研报告

> 适用场景：RK3588（Ubuntu 22.04 / aarch64）+ SO-ARM101 主从臂，D435i 作第三人称固定相机，另一 USB 相机装末端（腕部视角），ACT 策略训练数据采集，**只用 RGB 640×480@30**，深度暂不用（未来 GGCNN 可能用）。
> 调研日期：2026-07（以官方站点当时可访问版本为准）

---

## 0. 官方资料入口（重要：域名已迁移）

Intel 已于 2025 年前后把 RealSense 业务剥离为 **RealSense, Inc.**，官方资料域名迁移：

| 旧地址（Intel，多数已失效） | 新地址（官方，可用） |
|---|---|
| `www.intelrealsense.com/depth-camera-d435i/` | [`www.realsenseai.com/products/depth-camera-d435i/`](https://www.realsenseai.com/products/depth-camera-d435i/) |
| `www.intelrealsense.com/wp-content/uploads/.../Intel-RealSense-D400-Series-Datasheet-*.pdf` | [`www.realsenseai.com/product-datasheets/`](https://www.realsenseai.com/product-datasheets/) |
| `dev.intelrealsense.com/docs/...` | [`dev.realsenseai.com/docs/...`](https://dev.realsenseai.com/docs/) |
| `github.com/IntelRealSense/librealsense` | [`github.com/realsenseai/librealsense`](https://github.com/realsenseai/librealsense) |

**本次实际验证可访问的官方源**：
- D400 系列数据手册（2026-03 版，含 D435i）：<https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf>
  （2023-09 版镜像：<https://realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf>）
- D435i 官方产品页（Tech Specs）：<https://www.realsenseai.com/products/depth-camera-d435i/>
- 官方开发者文档：<https://dev.realsenseai.com/docs/>
- librealsense 源码/文档：<https://github.com/realsenseai/librealsense>
- 官方 wiki（Projection / Visual Presets / Release Notes）：<https://github.com/realsenseai/librealsense/wiki>

> 注：本报告所有数值均来自上列官方数据手册 / 官方文档页 / librealsense 官方仓库源码与文档。凡官方未明确的，均显式标注 **「官方未明确」**。

---

## 1. 光学与视场（FOV / 分辨率 / 基线 / 最小深度 / 精度）

### 1.1 深度模组（D435i = D430 模组 + D4 处理板）

出处：D400 数据手册 *Table「Depth Camera SKU Properties」* 与 *Table 3-13「Wide Left and Right Imager Properties ± D430」*。

| 项目 | 数值（D435i） | 官方出处 |
|---|---|---|
| 深度 FOV（HD，16:9，848×480 / 1280×720） | **H 87° / V 58° / D 95°** | 数据手册 SKU 属性表 |
| 深度 FOV（VGA，4:3，640×480） | **H 75° / V 62° / D 89°** | 数据手册 SKU 属性表 |
| 官方产品页 FOV 标称 | **87° × 58°** | [D435i 产品页](https://www.realsenseai.com/products/depth-camera-d435i/) |
| 左右图像传感器 | **OmniVision OV9282**，1280×800，8:5，10-bit RAW，f/2.0，焦距 **1.93 mm**，**全局快门**（Global Shutter），**Filter Type = None（无 IR-cut）** | 数据手册 Table 3-13 |
| 基线 baseline | **50 mm** | 数据手册 SKU 属性表（D415 为 55 mm，D455 为 95 mm，D405 为 18 mm） |
| IR 投影器 FOV | **H 90° / V 63° / D 99°** | 数据手册 SKU 属性表 |
| 深度输出分辨率 / 帧率 | 最高 1280×720 / 最高 90 fps | 产品页 Tech Specs |
| **最小深度 Min-Z** | 1280×720 → **280 mm**；**848×480 → 195 mm**；**640×480 → 175 mm**；640×360 → 150 mm；480×270 → 120 mm；424×240 → 105 mm | 数据手册 **Table 4-12「Minimum-Z Depth」**（2026-03 版；2023-09 版为 Table 4-11） |
| Min-Z（产品页标称） | **Max resolution 时 ~28 cm** | 产品页 Tech Specs |
| **深度精度** | **< 2% @ 2 m**（Z-accuracy / Absolute Error，条件：≤2 m、80% ROI、HD 分辨率） | 产品页 Tech Specs + 数据手册 *Table「Depth Quality Specification」* 中 `Z-accuracy (or Absolute Error) = ±2%` |
| **理想工作距离** | 产品页：**Ideal Range 0.3 m – 3 m**；数据手册特性列表：**「Range 0.2 m to over 3 m (varies with lighting conditions)」** | 产品页 Tech Specs；数据手册 §1 Features（D435/D435i 条目） |

补充（官方 tuning 白皮书，同样重要）：
- 「RealSense D435 建议使用 **848×480 @30fps** 深度，配 auto-exposure，后处理 downsample 2」——<https://dev.realsenseai.com/docs/d400-series-visual-presets/>
- 「D435 在 848×480 时 MinZ ≈ 16.8 cm」与数据手册 Table 4-12 的 195 mm 略有出入；**以数据手册 Table 4-12 为准（195 mm）**，白皮书那句属早期版本的估算值。——<https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/>
- 「depth error 随距离平方增长，所以尽量靠近物体，但不要进 MinZ 以内」——同上
- 关于 **disparity shift 降低 MinZ**：「把 disparity shift 从 0 增到 128，newMaxZ ≈ 原 MinZ，newMinZ ≈ 原 MinZ 的一半……我们通常只在已知没有比 MaxZ 更远物体时才建议这么做，**例如把深度相机吊在桌面上方俯拍桌面**。」——**这正是你的场景**，官方明确背书。——<https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/>

### 1.2 RGB（彩色）模组

出处：数据手册 *Table 3-18/3-19/3-20「Color Sensor Properties」*、SKU 属性表的 `Color Sensor` / `Color Camera FOV` 行、*Table 4-2「Image Formats (USB 3.1 Gen 1)」*；产品页 Tech Specs。

| 项目 | 数值（D435i） | 官方出处 |
|---|---|---|
| 传感器型号 | **OmniVision OV2740**（D415 与 D435/D435i 同型号；D455/D405 为 OV9782） | 数据手册 SKU 属性表 `Color Sensor` 行 |
| 有效像素 / 长宽比 | **1920 × 1080**，**16:9**，10-bit RAW RGB，f/2.0，焦距 **1.88 mm** | 数据手册 Table 3-18（D415 Color Sensor，同型号同 FOV） |
| 快门 | **Rolling Shutter（卷帘）** | 产品页 Tech Specs `RGB Sensor Technology` |
| **RGB FOV（1920×1080 原生）** | **H 69° / V 42° / D 77°**（数据手册 SKU 表；D415 Color Sensor 表更精确：**H 69.4° / V 42.5° / D 77°**）；产品页写 **H 69° × V 42°** | 数据手册 SKU 属性表 + Table 3-18；产品页 |
| FOV 容差 | **「Due to mechanical tolerances of ± 5%, Max and Min FOV values can vary from lens to lens」→ 逐台 ±5%** | 数据手册 FOV 章节注 |
| RGB 分辨率 / 帧率（YUY2 16-bit，USB3.1 Gen1） | 1920×1080 @ 6/15/30；1280×720 @ 6/15/30；**960×540 @ 6/15/30/60**；**848×480 @ 6/15/30/60**；**640×480 @ 6/15/30/60**；640×360 @ 6/15/30/60；424×240 @ 6/15/30/60；320×240 @ 6/30/60；320×180 @ 6/30/60 | 数据手册 Table 4-2（2026-03 版；2023-09 版为 Table 4-3） |
| 产品页汇总 | RGB Frame Rate **30 fps**；Frame Resolution **1920×1080**；Sensor Resolution **2 MP** | 产品页 Tech Specs |

**关于 640×480 的 RGB FOV（关键，官方未给出）**

- **官方未明确给出 640×480 下的 RGB FOV。** 数据手册只给 1920×1080（16:9）的 69°/42°/77°，并另列 640×480、320×240 这类 4:3 模式，但**不逐模式给 FOV**。
- 可推算：OV2740 原生只有 1920×1080（16:9）。要输出 4:3 的 640×480，只能**水平中心裁剪**（1920×1080 → 1440×1080 → 缩放到 640×480；垂直裁剪到 1440 行在物理上不可能）。因此：
  - **HFOV(640×480) ≈ 2·atan(0.75 · tan(69.4°/2)) ≈ 54.9°**（取 ±5% 容差 → **52° ~ 58°**）
  - **VFOV(640×480) ≈ 42.5°**（不变）
  - 这是**推算值，非官方数值**，请以实测为准。
- **官方给出的正确做法**：运行时读设备内参再换算。librealsense 官方文档明确 `rs2_intrinsics` 的 `fx/fy/cx/cy/width/height` 语义，且 `HFOV = 2·atan(width / (2·fx))`、`VFOV = 2·atan(height / (2·fy))`。——<https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0>
  ```python
  prof = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
  i = prof.get_intrinsics()          # 640x480 的真实内参
  import math
  hfov = 2*math.degrees(math.atan(i.width /(2*i.fx)))
  vfov = 2*math.degrees(math.atan(i.height/(2*i.fy)))
  ```
  **建议你先跑这段把 640×480 的真实 HFOV/VFOV 打印出来，再套第 6 节的摆放公式。**

---

## 2. RGB 采集官方最佳实践

### 2.1 官方对彩色流的推荐设置

- 官方没有任何「RGB 数据采集专用」的推荐配置页；官方给出的彩色流相关信息集中在数据手册 **Table 4-24「RGB Exposed Controls」** 与 librealsense 的 option 语义里。
- 数据手册 **Table 4-24「RGB Exposed Controls」**（D435i 有效范围）：

| 控件 | 官方描述 | Min | Max |
|---|---|---|---|
| Auto-Exposure Mode | 自动设置曝光时间与增益 | 0x1 | 0x8 |
| Auto Exposure ROI | 在选定 ROI 上做自动曝光（T/L/B/R） | T=0, L=0, B=1, R=1 | T=1079, L=1919, B=1080, R=1920 |
| Manual Exposure Time | 关闭自动曝光后的绝对曝光时间 | 1 | 10000 |
| Brightness | | −64 | 64 |
| Contrast | | 0 | 100 |
| Gain | 关闭 AE 时生效 | 0 | 128 |
| Hue / Saturation / Sharpness | | −180 / 0 / 0 | 180 / 100 / 100 |
| Gamma | | 100 | 500 |
| **White Balance Temperature Control** | **AWB 关闭时设置白平衡** | **2800** | **6500** |
| White Balance Temperature Auto (AWB) | 开/关 AWB | 0 | 1 |
| **Power Line Frequency** | 按当地市电频率设置以**避免闪烁** | **0** | **3**（Off/50Hz/60Hz/Auto） |
| Backlight Compensation / Low Light Comp | | 0 | 1 |

- 深度侧推荐（数据手册 Table 4-23「Depth Camera Controls」，D430 系列）：Manual Exposure **1–166 ms**；Manual Gain（Gain 1.0 = 16）**16–248**；Laser Power On/Off 0–1；**Manual Laser Power 0–360 mW，步进 30 mW**；Auto Exposure Mode 0–1；Auto Exposure ROI。
- 官方 tuning 白皮书：**「先调曝光，增益尽量保持最低（16）。提高增益会引入电子噪声」**，曝光单位 µs（33000 = 33 ms），过曝与欠曝一样糟。——<https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/>

### 2.2 Auto-Exposure Priority：机器人数据采集该怎么选

**官方定义（librealsense 源码原文）**：
- `RS2_OPTION_AUTO_EXPOSURE_PRIORITY` — *"Allows sensor to dynamically adjust the frame rate depending on lighting conditions"*（`include/librealsense2/h/rs_option.h`）
- 控件描述 — **"Restrict Auto-Exposure to enforce constant FPS rate. Turn ON to remove the restrictions (may result in FPS drop)"**（`src/platform/uvc-option.cpp`）
- 即映射到 V4L2 `V4L2_CID_EXPOSURE_AUTO_PRIORITY`（`src/platform/backend-v4l2.cpp`）。

**结论（用于 ACT 数据采集）：设 `RS2_OPTION_AUTO_EXPOSURE_PRIORITY = 0`（OFF）。**
理由（直接来自上面官方原话）：该控件本质是「是否限制 AE 以强制恒定帧率」。**OFF = 限制 AE = 恒定 30 fps**；ON = 放开限制 = 暗光下自动延长曝光时间、**帧率掉到 30 以下**。ACT 训练数据要求时间轴均匀（每个 demo 的帧间隔一致），掉帧会让动作-观测对齐错位，所以必须选恒定帧率。

> ⚠️ 注意：RealSense Viewer 里这个开关默认是打开的（ON = 允许掉帧），而且开关名字「Auto Exposure Priority」很容易让人误解成相反含义。**采集脚本里显式写成 0。**

> **「视觉预设 vs 高精度预设」**：Visual Presets（Default / High Density / High Accuracy / Hand）**只作用于深度 ASIC 参数，不影响 RGB 图像**。官方预设表把 High Accuracy 标为「Object Scanning, Collision Avoidance, **Robots**」，而 RGB 只受 Table 4-24 那些 UVC 控件影响。所以「RGB 采集用哪个预设」这个问题本身不成立——**预设不影响 RGB**；如果将来开深度，机器人抓取建议 `High Accuracy`（官方原话：*"very good for autonomous robots where false depth, aka. Hallucinations, are much worse than no depth"*）。——<https://dev.realsenseai.com/docs/d400-series-visual-presets/>、<https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/>

### 2.3 白平衡 / 曝光是否需要手动锁定

- **官方原文（librealsense）**：
  - `RS2_OPTION_WHITE_BALANCE` — *"Controls white balance of color image. **Setting any value will disable auto white balance**"*
  - `RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE` — *"Enable / disable color image auto-white-balance"*
  - `RS2_OPTION_ENABLE_AUTO_EXPOSURE` — *"Enable / disable auto-exposure"*
  - `RS2_OPTION_AUTO_EXPOSURE_LIMIT` — *"Set and get auto exposure limit in microseconds. If the requested exposure limit is greater than frame time, it will be set to frame time at runtime. Setting will not take effect until next streaming session."*
- **官方是否要求锁定？→ 官方未明确要求。** 官方只在深度侧建议「AE 或手动曝光都要保证正确曝光」，并说明 RGB 的 AE ROI 与 Setpoint 可单独设置（「The RGB ROI needs to be set separately... it does impact the color image quality」）。
- **工程结论（非官方，基于上面官方语义）**：
  1. 曝光：**建议锁定**。AE 在采集过程中随物体/手臂进入画面而重新收敛，会造成帧间亮度跳变，对 ACT 是纯噪声。做法：先让 AE 跑 10–30 s 收敛，读回 `RS2_OPTION_EXPOSURE`，再 `ENABLE_AUTO_EXPOSURE=0` + 写入该曝光值，同时把 `GAIN` 固定在最低可用值（官方建议尽量低）。**先把 AE Priority 设为 0** 再锁定，保证锁定值一定 ≤ 帧周期（33.3 ms）。
  2. 白平衡：**建议锁定**（官方只说「写 WB 会自动关 AWB」，并给出 2800–6500 K 可写范围）。若自动白平衡在手臂/阴影进入画面时反复漂移，颜色通道的帧间跳变同样会污染策略输入。做法：先开 AWB 采一段稳定画面，读回色温，再写入同一值（写入即自动关 AWB）。
  3. 若你的桌面光源固定（LED 灯条/环形灯），锁定的收益最明显；**同时建议关掉房间的自动调光**。

### 2.4 帧间一致性：flicker / anti-flicker（50Hz vs 60Hz）

- **官方定义**：`RS2_OPTION_POWER_LINE_FREQUENCY` — *"Power Line Frequency control for anti-flickering **Off/50Hz/60Hz/Auto**"*（`rs_option.h`）；数据手册 Table 4-24 也列出该控件，范围为 **0–3**，描述 *"Specified based on the local power line frequency for flicker avoidance"*。
- **该选项只注册在彩色传感器上**（`src/ds/d400/d400-color.cpp` 里对 color endpoint 注册 `uvc_pu_option(RS2_OPTION_POWER_LINE_FREQUENCY)`），**深度流不暴露它**。
- **中国（50 Hz 市电）→ 设为 50 Hz**（对应枚举值 1）。这能消除工频灯光下 100 Hz 亮度脉动与卷帘快门行曝光的拍频，避免逐帧亮度/条纹跳变。
- 顺带说明：`RS2_OPTION_AUTO_EXPOSURE_MODE`（*"Static, Anti-Flicker and Hybrid"*）在官方源码里**只注册在 IMU/运动模组上**（`src/ds/ds-motion-common.cpp`），**D435i 的深度与彩色流都不暴露这个选项**——所以对 RGB 抗闪烁，唯一可用的官方手段就是 `POWER_LINE_FREQUENCY`。

---

## 3. IR 发射器与多机干扰

### 3.1 IR projector 会影响 RGB 图像吗？

- **对 D435i 的专用 RGB 模组：官方未明确说明其是否内置 IR-cut 滤镜。** 数据手册分别列出：
  - 左右**立体图像传感器**（OV9282）：`Filter Type = None`，即**不遮挡 IR**，所以深度/IR 图上会看到投影点阵；
  - 专用 **Color Sensor** 表（D415 / D450 / D401）：`Filter Type = IR Cut Filter`。D435i 的 `Color Sensor` 行与之同为 **OV2740 且 FOV 相同（69/42/77）**，但数据手册**没有为 D435i 的 RGB 单独列 Filter 行**。
  - 官方光学滤镜白皮书只解释「D400 系列的**立体相机**故意不内置滤镜，以便看到 IR 投影」，并指出「几乎所有彩色相机都用 IR-cut 滤镜改善可见光色彩」。——<https://dev.realsenseai.com/docs/optical-filters-for-intel-realsense-depth-cameras-d400/>
- **可落地的做法**：**先实测确认**（关灯只留投影器，看 RGB 原图有无点阵/散斑）。若追求零风险与帧间绝对一致，**RGB-only 采集阶段把发射器关掉**：`RS2_OPTION_LASER_POWER = 0`（官方文档：*"Power of the laser emitter (mW), with **0 meaning projector turned off**"*）或 `RS2_OPTION_EMITTER_ENABLED = 0`（*"0 - disable all emitters"*）。代价：深度不可用——需要两套 profile（采集 RGB 时关 emitter，将来跑 GGCNN 时开）。

### 3.2 多台 RealSense 同场会不会互相干扰？

- **官方结论：不会，而且通常是正向的。** 原话：*"For Active Stereo Depth systems, no a priori knowledge of the projection pattern is needed... **it does not matter if other cameras point at the same scene with their projectors. To a first order, all additional projectors actually improve the overall performance by adding more light and more texture.**"*——<https://dev.realsenseai.com/docs/projectors/>
- tuning 白皮书同结论：*"**RealSense D4xx cameras do not interfere with each other.**"*——<https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/>
- 你只有 1 台 D435i + 1 个普通 USB 相机，**普通 USB 相机若带 IR 补光 LED**（很多工业 USB 相机有 850nm 补光），反而会给 D435i 的**立体图**增加纹理（有益），但要注意：
  - 若该 USB 相机补光很强且正对 D435i，可能造成局部过曝 / 眩光——建议**错开补光时间**或降低腕部相机补光功率。
  - 若腕部相机有 850nm 长通/带通滤镜而 D435i 开着投影器，两者不冲突（不同波段无关）。

### 3.3 `Emitter Enabled` / `Laser Power` 官方建议

- **默认值：150 mW**（数据手册深度测试条件原文：*"measured using a texture-less (white) target with **default laser power (150mW)** and auto exposure enabled"*）。
- **可调范围：0–360 mW，步进 30 mW**（数据手册 Table 4-23 `Manual Laser Power (mW)`，描述 *"Laser Power setting (30 mW steps)"*）。projectors 白皮书也确认「可从默认 150 mW 提升到 360 mW（>2×）」，但「2× 功率只带来 √2 ≈ 1.41× 的距离提升」。
- **官方调参建议**：*"Test on White wall: Turn on projector and adjust laser power. **We recommend using the nominal 150mW**, but adjust it up or down for better results. For example, **if you see localized laser point saturation, reduce laser power; if depth is sparse, try increasing laser power.**"*——<https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/>
- `RS2_OPTION_EMITTER_ENABLED` 取值（官方 `rs_option.h`）：**0 = disable all emitters；1 = enable laser；2 = enable auto laser；3 = enable LED**。
- **热保护（重要）**：数据手册 Table 3-26 —— *"When laser power and depth streaming is enabled and if stereo depth module temperature is **> 60 °C, laser power is halved**. If temperature is not lowered below temperature limit..."*（随后进入 `hot_laser_power_reduce` / `hot_laser_disable` 报错，librealsense 亦有 `Laser hot - power reduce` / `Laser hot - disabled` 错误码）。
- **USB2 特别说明**：当设备以 USB2 PID（`0x0AD6`）连接时，librealsense **不注册** `EMITTER_ENABLED` / `LASER_POWER` / `PROJECTOR_TEMPERATURE` 三个选项（`src/ds/d400/d400-device.cpp`）。所以**如果发现 `enable_emitter` 报 "option not supported"，先查是不是插到了 USB2 口**。

---

## 4. 深度相关（简述，面向未来 GGCNN 抓取）

### 4.1 `rs.align(rs.stream.color)` 的作用

- 官方定义与示例：`examples/align/rs-align.cpp`。深度和彩色来自**不同视口**（D435i 深度模组与 RGB 模组物理分离、基线不同），align 把某个流的像素**重投影**到另一个视口：
  ```cpp
  rs2::align align_to_color(RS2_STREAM_COLOR);   // 构造很贵，不要放在主循环里
  ...
  frameset = align_to_color.process(frameset);   // 之后 get_depth_frame() 就是彩色视口下的深度
  ```
- **官方明确列出的两个副作用（源码注释原文）**：
  1. **Sampling**：重采样会改变分辨率并对齐到目标视口，插值必须是最近邻（Nearest Neighbor）以避免引入不存在的值；
  2. **Occlusion**：结果图里有些像素对应的 3D 点原始传感器根本没看到（被遮挡），这些像素可能是无效纹理值。
- **对你的关键含义**：**align 到 color 之后，深度 FOV 被裁到 RGB FOV。** D435i 深度本来有 75°×62°（VGA）/87°×58°（HD），RGB 只有 ~69°×42.5°（1080p）或 ~55°×42.5°（640×480 推算）。所以「深度视场比 RGB 宽」的优势在 align-to-color 后**完全丧失**。若 GGCNN 需要更大深度视野，应改为 `rs2::align(RS2_STREAM_DEPTH)` 把彩色投到深度视口，或干脆不做 align、分别用 `rs2_project_point_to_pixel()` 自行映射。

### 4.2 官方推荐的后处理滤波链与参数

**官方推荐顺序**（`doc/post-processing-filters.md` 原文）：

```
Depth Frame >> Decimation Filter >> Depth2Disparity Transform >>
Spatial Filter >> Temporal Filter >> Disparity2Depth Transform >>
Hole Filling Filter >> Filtered Depth
```
- **注意：官方强调「没有任何软件强制的顺序约束」，但这是 librealsense 工具与 demo 采用的推荐链。**
- 官方示例代码 `examples/post-processing/rs-post-processing.cpp` 的实际链（比文档多一个 Threshold 与 Rotation）：
  `Decimate → Rotate → Threshold → Depth→Disparity → Spatial → Temporal → (回)Disparity→Depth`

**官方参数范围与默认值**（`doc/post-processing-filters.md`）：

| 滤波器 | 参数 | 范围 | 官方默认 |
|---|---|---|---|
| Decimation | Filter Magnitude（线性缩放因子） | 离散 [2–8] | **2** |
| Spatial（边缘保持） | Filter Magnitude（迭代次数） | [1–5] | **2** |
| Spatial | Smooth Alpha（α=1 不滤波，α=0 无限滤波） | [0.25–1] | **0.5** |
| Spatial | Smooth Delta（边缘保持阈值） | 离散 [1–50] | **20** |
| Spatial | Hole Filling（就地对称补洞） | [0–5] → [none,2,4,8,16,unlimited] px | **0（none）** |
| Temporal | Smooth Alpha | [0–1] | **0.4** |
| Temporal | Smooth Delta | 离散 [1–100] | **20** |
| Temporal | Persistency index | [0–8] 枚举 | **3（Valid in 2/last 4）** |
| Hole Filling | Hole Filling | [0–2]：`fill_from_left` / `farest_from_around` / `nearest_from_around` | **1（Farest from around）** |

**官方白皮书给出的"推荐默认值"（与上表略有差异，值得注意）**——<https://dev.realsenseai.com/docs/depth-post-processing-for-intel-realsense-depth-camera-d400-series/>：
- 降采样：「小因子（2、3）用 **non-zero median**，大因子（4、5…）用 **non-zero mean**」，且**忽略 depth=0 的空洞**。
- **Spatial：`Spatial Alpha = 0.6`，`Spatial Delta = 8`（delta 单位是 1/32 disparity，即 8/32 disparity）。**
- **Temporal：`alpha = 0.5`，`delta = 20`。**
- 必须在 **disparity 域**做空间/时间滤波（因为深度噪声随距离平方增长，α/δ 依赖距离会过平滑近处、欠平滑远处），这也是链里先 Depth→Disparity 的原因。
- 降采样 2× → 后续计算量降 4×；降采样 4× → 降 16×。
- **关于 threshold filter**：官方 `post-processing-filters.md` 的推荐链里**没有** threshold；它只出现在示例代码里，官方描述为 *"Threshold - removes values outside recommended range"*（对应 `RS2_OPTION_MIN_DISTANCE` / `RS2_OPTION_MAX_DISTANCE`）。**「官方未明确 threshold 在推荐链中的位置与参数」**，示例代码把它放在 Decimation 之后、disparity 变换之前。

### 4.3 对「未来用深度做抓取」的意义

- **空洞该不该补**：官方明确 *"For some applications it is best to leave the holes and do nothing. This is generally true for **robotic navigation and obstacle avoidance** as well as 3D scanning applications."* → **GGCNN 建议保守：Spatial/Temporal 打开，Hole Filling 关闭或只做轻量 spatial hole filling**，避免"猜"出来的深度污染抓取高度估计。
- **预设选 `High Accuracy`**（官方：机器人场景宁可没有深度，也不要错误深度）。
- **近距优化**：桌面俯拍场景官方明确支持用 **disparity shift** 把 MinZ 压到约一半（见 §1.1）。
- **深度单位**：默认 1000 µm（1 mm），最大量程 ~65 m；近距作业可改 100 µm 提高量化分辨（最大量程降到 ~6.5 m）。——官方 tuning 白皮书 & projectors 白皮书。
- **分辨率建议**：官方推荐 D435 用 **848×480 @30fps** 跑深度（精度最好），再用 decimation 降到 640×480 级别给网络。若你已经同时开 RGB 640×480，**深度用 848×480 + decimation 2 ≈ 428×240**，或直接用 640×480（MinZ 175 mm，比 848×480 的 195 mm 更近）。

---

## 5. 硬件与工程注意

### 5.1 USB3 vs USB2：带宽与分辨率限制

- **接口**：D435i 为 **USB-C 3.1 Gen 1**（产品页 Tech Specs）。
- **USB 3.1 Gen1 下的彩色流（YUY2）**：见 §1.2 表（640×480 可到 **60 fps**，1920×1080 只到 **30 fps**）。
- **USB 2.0 下的限幅**（官方数据手册 **Table 4-7「Image Formats (USB 2.0) ± D410/D415/D430/D435/D435i…」**）：

| 流 | 分辨率 | USB2.0 可用帧率 |
|---|---|---|
| Depth Z16 | 1280×720 | 6 |
| Depth Z16 | 848×480 | 6, 10 |
| Depth Z16 | **640×480** | **6, 15, 30** |
| Depth Z16 | 640×360 | 30 |
| Depth Z16 | 480×270 | 6, 15, 30, 60 |
| Y8（左目亮度） | 1280×720 / 848×480 / 640×480 / 480×270 | 6 / 6,10 / 6,15,30 / 6,15,30,60 |
| **Color YUY2（RGB 相机）** | 1280×720 | **6, 15** |
| **Color YUY2（RGB 相机）** | **640×480** | **6, 15, 30** |
| Color YUY2（RGB 相机） | 424×240 | 6, 15, 30, 60 |

  → **好消息：你的 640×480@30 RGB 在 USB2.0 下也能跑。** 但深度会大幅受限（848×480 只能 10 fps；1280×720 只能 6 fps），且 emitter/laser 选项不注册。**结论：务必确认跑在 USB3 上**（`rs-enumerate-devices` 会打印 USB 类型；`RS2_OPTION_*` 缺失 emitter 就是 USB2 的信号）。
- **带宽经验值（官方多相机白皮书）**：
  - USB3.0 SuperSpeed 理论 5 Gbps；*"accounting for the encoding overhead, the raw data throughput is 4 Gbit/s, the specification considers it reasonable to achieve 3.2 Gbit/s"*；
  - *"in general care should be taken to stay well below **0.3 × 4 Gbps = 1200 Mbps** to ensure robust continuous streaming"*；
  - 参考算例：*"Assuming 30fps and 16bit depth, this translates to **442 Mbps**… To transmit color at RGB at the same time, adds about 24×1280×720×30 = **663 Mbps**. However, by default the ASIC is in YUYV mode which reduces the bandwidth by encoding the color in 16bits as opposed to 24bits, so the color channel is then **442 Mbps**."*；*"reducing the resolution to 640×360 shows no problem streaming 4 channels of both color and depth as the total bandwidth is **882 Mbps**."*
  - → 你的配置（RGB 640×480@30 YUY2 ≈ 640×480×2×30 = 18.4 Mbps，即使加上 BGR8 转换后的主机侧内存带宽）**远低于限值**，单相机无带宽压力。
  - **延迟/队列**：官方建议 frame queue `capacity`：**只开深度设 1，深度+彩色设 2**（多相机白皮书 §H）。
  ——<https://dev.realsenseai.com/docs/multiple-depth-cameras-configuration/>

### 5.2 USB-C 线材要求

- 官方（多相机白皮书 §D Cabling and enumeration）原文要点：
  - *"the quality of USB3.0 cables can vary quite a bit. For best performance, we recommend using **high quality cables and using as short cables as possible – preferably less than 1m**."*
  - *"If it is necessary to use cables longer than 2m, we recommend using **USB3 repeaters, and not just extension cables**."*（并指出 repeater 质量差异很大，官方实测把 3 个串联可用。）
  - 数据手册另有 *Table「Recommended USB Type C cable Assemblies」*（Table 3-37/3-38，内容为图形表格）与枚举要求：*"To ensure proper USB 3.1 device enumeration, **connect cable to D400 camera first, then** [to host]"*。
- **RK3588 实战提示**：很多 RK3588 板子的 USB3 Type-A/Type-C 口由内部 HUB 分出，或供电能力不足。**优先用独立 USB3 控制器口 + 带外部供电的 USB3 HUB**。

### 5.3 发热与长时间运行

- **官方环境规格（数据手册 Table 3-55「Depth Camera D400 Series Storage and Powered Conditions」）**：

| 条件 | 项目 | Min | Max | 单位 |
|---|---|---|---|---|
| 存储（环境、未供电） | 温度（持续、受控） | 0 | **50** | °C |
| 存储（环境、未供电） | 温度（短时暴露，运输） | **−40** | **70** | °C |
| 存储 | 湿度 | — | 40 °C / 90% RH | |
| **供电（环境）** | **温度（性能）** | **0** | **35** | **°C** |
| 供电 | **背面外壳温度** | **0** | **50** | °C |

  官方注(3)原文：*"The camera ambient temperature when powered, **0 °C to 35 °C is the validated range in which [the vendor] qualified the camera**... The camera's internal thermal solution was designed to keep the internal components at or below their max powered temperatures."*
  官方注(5)：*"**KPIs may be negatively impacted by extended exposure to excessive temperatures and humidity**."*
- **热对激光/深度的硬约束**：立体模组温度 **> 60 °C → 激光功率减半**，继续升温会 `hot_laser_disable`（§3.3）。
- **AE 漂移 / 是否需要预热 → 官方未明确给出预热时长或 AE 漂移量。** 官方只给出上述温度对 KPI 的影响与 0–35 °C 的验证范围。可落地的工程做法：
  1. **预热判据用数据说话**：连续采 5–10 分钟，每秒记录 `RS2_OPTION_EXPOSURE`、`RS2_OPTION_GAIN`、`RS2_OPTION_ASIC_TEMPERATURE`、`RS2_OPTION_PROJECTOR_TEMPERATURE`，等这些量进入平台期再开始录数据。你现在脚本里的「丢 3 秒」**明显不够**（3 s 只够建立流，不够热平衡）。
  2. **锁定曝光/白平衡**（§2.3）能在很大程度上消除热漂移带来的帧间跳变——这比单纯延长预热更有效。
  3. **物理散热**：D435i 是 90×25×25 mm 铝合金外壳、全被动散热。**不要把它包在 3D 打印的封闭壳里**；留出自然对流通道。官方数据手册明确 *"It is best to **thermally isolate** Vision Processor D4 from the stereo depth module"*（针对集成设计），相机整机也同理。
  4. **长时间录制不要靠 emitters 发热**：RGB-only 阶段关掉投影器可显著降低整机发热（投影器是主要热源之一）。

### 5.4 固定安装（防震动、避免遮挡）

- **官方机械安装点（产品页 Tech Specs `Mounting Mechanism`）**：
  - **1 个 1/4″-20 UNC 螺纹孔**
  - **2 个 M3 螺纹孔**
- **官方对固定的相关表述**：
  - 立体模组章节有 *"Stereo Depth Module Mounting Guidance / Screw Mounting / Bracket Mounting"*，强调 *"Secure placement and mounting to system/chassis"*、*"Mounting holes … secure mounting"*（数据手册）。
  - 官方标定文档 & Projection wiki 强调：设备**外参（extrinsics）在程序生命周期内视为常量**，即任何相机与机械臂之间的相对位移都会让标定失效。
  - 光学滤镜白皮书：*"care should be taken when mounting any transparent material in front of the camera if it also covers the IR projector, because **back-reflection of light from the projector into the stereo cameras can significantly degrade their performance**."*
- **官方未针对"模仿学习/固定第三人称视角"给出专门指导** → 以下为工程结论：
  1. **相机与机械臂必须刚性连接在同一个基座/桌面上**。最佳做法：相机用铝型材 + 1/4″-20 螺丝刚性锁在同一块底板上，机械臂底座也锁在同一块底板上。任何"相机夹在另一张桌子/独立三脚架"的方案都会因微振动导致固定视角抖动，直接注入 ACT 观测噪声。
  2. **防震**：螺丝上加弹垫；型材连接处避免悬臂过长；相机本体不要和机械臂共用一个会共振的薄板。
  3. **避免遮挡与反光**：安装支架、线缆、理线夹不得进入 FOV（D435i 深度 FOV 高达 87°，比 RGB 宽得多，支架很容易"蹭进"深度画面）；相机前方不要贴保护玻璃（尤其覆盖投影器开口时）。
  4. **不要让相机光轴被机械臂自身长期挡住**（见 §6.3）。

### 5.5 多相机硬同步（hardware sync）能力与是否需要

- **能力**：D435i **有外部同步连接器**。数据手册有 *Figure「External Sensor Sync Connector Location on Depth Camera D435/D435i/D455」* 与 *Table 3-42「External Sensor Sync Connector Pin List」*（例：pin 5 = `Z_VSYNC`（Depth VSYNC），pin 9 = GND）。多相机白皮书给出接线：**pin 5 → pin 5，pin 9 → pin 9**，可菊花链或星型；相机间距 < ~3 m 可用无源互连（推荐屏蔽双绞线），更长用 RS-485/RS-422 有源电路。
- **API**：`RS2_OPTION_INTER_CAM_SYNC_MODE`（官方 `rs_option.h`：*"Impose Inter-camera HW synchronization mode. Applicable for D400/L500/Rolling Shutter SKUs"*）。取值（`src/ds/ds-private.h`）：**0 = Default，1 = Master，2 = Slave**（3 = Full Slave，4–258 = genlock burst count，259/260 = 激光交替帧）。官方建议：*"For normal operation we recommend using **Default**. For HW sync, one camera should be told to be Master, and all others Slave."*
- **触发要求**：外部触发脉冲需 **100 µs 正脉冲、标称帧率（30 Hz → 33.33 ms）**，输入高阻、**1.8 V CMOS** 电平；信号源频率必须与传感器实际帧率完全一致（例如设 30 fps 实际可能是 30.015 fps，需要示波器校准）。
- **一个反直觉的官方验证技巧**：*"If NO HW sync is enabled, the time stamps will surprisingly appear to be perfectly aligned… By contrast, if HW Sync is enabled, the time stamps will actually drift over time… **If you see NO DRIFT, then there is NO HW sync. If you see DRIFT, then the units are actually HW synced.**"*
- **你需要吗？→ 不需要（当前配置）。** 理由：
  - 你只有 1 台 D435i；另一路是 USB 相机，二者本来也不同步，且 ACT 用的固定视角与腕部视角是**软对齐**（按时间戳最近邻配对）即可。
  - 官方说明：不启用硬同步时，把 frame queue capacity 设最小（彩色+深度 = 2），"frames from different cameras can never misalign by more than 1 frame"。
  - **建议**：改为在录制端做**统一时间戳打标**（`frame.get_timestamp()` + `RS2_FRAME_METADATA_FRAME_TIMESTAMP`），并按时间戳做 30 fps 重采样对齐。若将来上第二台 D435i 且要求亚帧级同步，再上硬同步线。
- ⚠️ 官方警告：ESD/EMI 事件（静电）会导致 frame counter 复位；深度同步场景可忽略，**但 RGB 相机曾被观察到在 ESD 事件中冻结（D415 比 D435 更敏感）**。长时间无人值守采集建议加 ESD 防护。

---

## 6. 摆放建议（最重要）

### 6.1 计算方式

**约定**：
- `H` = 相机镜头到底面（桌面）的**垂直高度**
- `δ` = 光轴**低于水平面的俯角**（δ = 90° 即完全垂直向下 / nadir；δ = 70° 即离垂直方向偏 20°）
- 相机离底面的**斜距** `R = H / sin δ`（对准点）
- 像面横轴（图像左右）对应桌面**横向**，像面纵轴（图像上下）对应桌面**沿倾斜方向**

**(a) 垂直俯拍（δ = 90°，最简单、最可控）**

```
可见宽度 W = 2 · H · tan(HFOV/2)
可见高度 L = 2 · H · tan(VFOV/2)
```

这正是你给的公式；官方 D400 数据手册 **§4.4「Depth Field of View at Distance (Z)」** 就是同一关系的定义章节（其变量定义明确列出 `HFOV`（左目水平 FOV）、`B`（baseline）、`Z`（场景到模组距离））。
更精确的做法是用设备内参（官方 Projection 文档）：`W = Z · width / fx`，`L = Z · height / fy`，其中 `fx, fy, width, height` 来自 `get_intrinsics()`。

**(b) 倾斜俯拍（δ < 90°，远近平移更"立体"，但要算梯形）**

像面**上边缘**对应俯角 `δ + VFOV/2`，**下边缘**对应 `δ − VFOV/2`（要求 `δ > VFOV/2`）。台面上的近/远边界：

```
x_近 = H / tan(δ + VFOV/2)      （δ + VFOV/2 > 90° 时为负值，即看到相机后方桌面）
x_远 = H / tan(δ − VFOV/2)
沿倾斜方向可用长度 L = x_远 − x_近
```

桌面某点 `x` 处的横向宽度（梯形，近窄远宽）：

```
W(x) = 2 · tan(HFOV/2) · H / sin(β(x))，  β(x) = atan(H / x) 为该行的俯角
      = 2 · R(x) · tan(HFOV/2)，          R(x) = √(x² + H²) 为该行的斜距
```

> 工程上只需记住：**梯形最窄处在近端**，所以「能否装下 40 cm 宽」要用近端的宽度判定。

### 6.2 具体数字（用你的实际配置算）

**场景**：桌面作业区 **30 × 40 cm**，要求覆盖 + 约 20% 余量（即需要 36 × 48 cm）。

**(a) 垂直俯拍所需的 H**（`W = 2·H·tan(HFOV/2)`，取 20% 余量）

| 流 / FOV | 宽度限制 H ≥ | 高度限制 H ≥ | 结论 H ≥ |
|---|---|---|---|
| **RGB @640×480（推算 55° × 42.5°）** | 0.40 m | **0.46 m** | **0.46 m**（高度是瓶颈） |
| RGB @1920×1080（69° × 42.5°） | 0.35 m | **0.46 m** | 0.46 m |
| Depth @640×480（75° × 62°） | 0.31 m | 0.30 m | 0.31 m |
| Depth @848×480（87° × 58°） | 0.25 m | 0.33 m | 0.33 m |

**关键洞察：垂直俯拍时，16:9 / 4:3 的 VFOV（42.5°）永远是瓶颈，而 RGB 的 VFOV 比深度 VFOV 窄得多。所以"RGB 能不能覆盖作业区"才是决定相机高度的那个约束。**

**(b) 推荐安装高度处能看到多大范围**（δ=90°，推算的 55°×42.5° RGB）

| H | 可见宽度 | 可见高度 |
|---|---|---|
| 0.40 m | 41.6 cm | 31.1 cm |
| **0.50 m** | **52.1 cm** | **38.9 cm** |
| **0.55 m** | **57.3 cm** | **42.8 cm** |
| 0.60 m | 62.5 cm | 46.7 cm |
| 0.70 m | 72.9 cm | 54.4 cm |
| 0.80 m | 83.3 cm | 62.2 cm |

**(c) 倾斜安装时的台面足迹 + 遮挡阴影**（RGB 55°×42.5°，δ = 俯角，`x` 以相机正下方为 0、朝远端为正）

| H | δ | 台面 x 范围 | 足迹跨度 | 横向宽(近/远) | 横宽≥40cm 时的可用沿倾长度 | 斜距(近/远) |
|---|---|---|---|---|---|---|
| 0.50 | 90° | −19 ~ +19 cm | 39 cm | 52 / 52 cm | 39 cm | 54 / 54 cm |
| 0.50 | 80° | −10 ~ +30 cm | 40 cm | 49 / 57 cm | **40 cm** | 51 / 58 cm |
| 0.55 | 90° | −21 ~ +21 cm | 43 cm | 57 / 57 cm | **43 cm** | 59 / 59 cm |
| **0.55** | **80°** | −11 ~ +33 cm | **44 cm** | **54 / 62 cm** | **44 cm** | **56 / 64 cm** |
| 0.55 | 75° | −6 ~ +40 cm | 46 cm | 54 / 66 cm | 46 cm | 55 / 68 cm |
| 0.60 | 80° | −12 ~ +36 cm | 48 cm | 59 / 68 cm | 48 cm | 61 / 70 cm |
| 0.70 | 80° | −14 ~ +42 cm | 56 cm | 69 / 79 cm | 56 cm | 71 / 82 cm |
| 0.80 | 80° | −16 ~ +48 cm | 64 cm | 79 / 91 cm | 64 cm | 82 / 94 cm |

**遮挡阴影公式（本文档给出的定量判据）**：高度 `h` 的竖直结构（机械臂连杆/立柱）在桌面上产生的遮挡位移

```
Δx = h / tan(δ)      （朝远离相机的方向）
```

| δ（俯角） | 5 cm | 10 cm | 15 cm | 20 cm | 25 cm |
|---|---|---|---|---|---|
| 90°（垂直） | 0 | 0 | 0 | 0 | 0 |
| **80°** | 0.9 cm | 1.8 cm | 2.6 cm | **3.5 cm** | 4.4 cm |
| **75°** | 1.3 cm | 2.7 cm | 4.0 cm | **5.4 cm** | 6.7 cm |
| 70° | 1.8 cm | 3.6 cm | 5.5 cm | 7.3 cm | 9.1 cm |
| 60° | 2.9 cm | 5.8 cm | 8.7 cm | 11.5 cm | 14.4 cm |
| 45° | 5.0 cm | 10.0 cm | 15.0 cm | 20.0 cm | 25.0 cm |

> **判据：想让 20 cm 高的连杆遮挡阴影 ≤ 5 cm，必须 δ ≥ 75°（即离垂直方向不超过 15°）。** 低角度（δ ≤ 60°）在 20 cm 高物体上会投出 > 11 cm 的盲区，对 30×40 cm 的桌面作业区是致命的。

**分辨率检查**（RGB 640×480，55° HFOV）：桌面像素尺寸 ≈ `2·R·tan(27.5°) / 640`

| 斜距 R | mm/pixel | 5 cm 物体占多少像素 |
|---|---|---|
| 0.55 m | 0.89 mm | 56 px |
| 0.65 m | 1.06 mm | 47 px |
| 0.80 m | 1.30 mm | 38 px |
| 0.95 m | 1.55 mm | 32 px |

→ H = 0.5 ~ 0.8 m 全部落在「5 cm 物体 ≥ 30 px」的舒适区，**ACT 完全够用**。H 越高视野越大但物体像素越少；H ≥ 1.0 m 时 5 cm 物体不足 25 px，不建议。

### 6.3 推荐摆放区间（结论）

| 参数 | 推荐值 | 依据 |
|---|---|---|
| **垂直高度 H** | **0.55 m（可接受 0.50 – 0.65 m）** | δ=90° 时 0.55 m 可见 57×43 cm，覆盖 30×40 cm + 余量；且 5 cm 物体 ≈ 56 px |
| **俯角 δ** | **80°（可接受 75° – 90°）** | δ=80° 时 20 cm 连杆遮挡阴影仅 3.5 cm；δ=75° 仍只有 5.4 cm；δ≤70° 时遮挡迅速恶化 |
| **离垂直方向倾角** | **0° – 15°（δ = 75° – 90°）** | 同上 |
| **斜距 R（到作业区中心）** | **0.50 – 0.68 m** | 远大于 MinZ（175/195 mm），且远在理想区间 0.3–3 m 内，深度精度最优 |
| **光轴指向** | 对准**作业区中心略偏远端**（取足迹中心） | δ=80°、H=0.55 时足迹中心在相机正下方前方 **+11 cm** |
| **横向位置** | 相机水平位置**偏置到机械臂工作平面的一侧**，光轴不要穿过机械臂基座 | 让基座落在画面边缘而非中央，减少对作业区的遮挡 |
| **视野大小** | 光轴方向上至少容纳 **基座 + 30 cm 臂展 + 40 cm 作业区** | 见下 |

**若必须同时把机械臂基座和整个 30×40 cm 作业区都收进画面**（推荐做法）：
- 作业区沿倾斜方向 40 cm + 基座到作业区近边 ≥ 10 cm + 余量 → 需要沿倾长度 **≥ 55 cm**。
- 由 6.2(c) 表：**H = 0.70 m、δ = 80°** 给出 56 cm 沿倾跨度、69–79 cm 横向宽度，**刚好满足**；**H = 0.75 – 0.80 m 更从容**（64 cm 跨度）。
- 此时 5 cm 物体约 38 px，仍可用。**这是"看得全"和"看得清"的折中点。**

### 6.4 避免自遮挡（三种失效模式与对策）

1. **机械臂本体挡住作业区**
   - 对策 A（最重要）：**把基座放在"远端"（远离相机一侧），作业区放在"近端"**。因为遮挡阴影恒朝**远离相机**方向偏移（Δx = h/tanδ），基座在远端时，连杆的阴影落在更远处、落在作业区之外。
   - 对策 B：**提高 δ 到 80°–90°**（阴影趋近 0）。
   - 对策 C：相机不要吊在基座正上方——把基座偏到画面边缘或画面外。
   - 反例：把相机放在机械臂"肩后"、让手臂朝远离相机方向伸入作业区 → 连杆阴影正好压在作业区上，**必须避免**。

2. **末端夹爪在抓取瞬间挡住物体**
   - 这是物理必然（夹爪从上方接近）。对策：**保证腕部相机（你已有的末端 USB 相机）在该时刻提供互补视角**；ACT 的固定第三人称视角只需保证**物体与夹爪的相对位置可辨**，δ ≥ 75° 时夹爪投影偏移 ≤ 5 cm，物体不会被完全吃掉。若发现遮挡严重，把 δ 提高到 85°–90°。

3. **相机支架/线缆/理线进入画面**
   - 深度 FOV 最宽 87°，RGB 也有 69°，支架极易"蹭边"。
   - 对策：安装后**用 `realsense-viewer` 或保存若干帧原图目视检查四条边**；支架尽量从相机**后方**进入而非侧前方。

### 6.5 保证末端在最远端/最近端都在画面内（检查流程）

1. **测出可达域**：把 SO-ARM101 手动拖到工作空间边界，记录末端在桌面坐标系下的**最小/最大 X（沿倾斜方向）与 Y（横向）**，加上夹爪自身尺寸，得到包络盒 `(X_min, X_max, Y_min, Y_max)`（含基座）。
2. **把它投影到相机足迹里**：用 §6.1 的公式（或直接在脚本里对四个角做 pinhole 投影）算出该包络盒在图像中的像素范围，要求**四边各留 ≥ 5% 图像边距**（即像素坐标都在 `[0.05·W, 0.95·W] × [0.05·H, 0.95·H]` 内）。
3. **两个边界都要查**：
   - **最近端（末端朝相机方向伸到极限）**：像面上边缘。δ + VFOV/2 若 > 90°，上边缘会看到相机后方，说明垂直方向视野浪费了；此时要么把 δ 调小（≈ 90° − VFOV/2 = 68.75°，让上边缘恰好垂直），要么把相机后移/抬高。
   - **最远端（末端伸到最远）**：像面下边缘，对应 `x_远 = H/tan(δ − VFOV/2)`。**这是最容易"出画"的一侧**，优先按它选 H。
4. **深度（未来 GGCNN）附加检查**：
   - 最远端的**斜距**要 ≤ 有效深度量程（桌面 0.5–1.0 m 完全没问题，理想区间 0.3–3 m）；
   - 最近端的**斜距**要 > MinZ（深度 640×480 时 **175 mm**，848×480 时 **195 mm**）——你 0.5 m 级的高度天然满足；
   - 若将来要贴得很近，用官方背书的 **disparity shift**（桌面俯拍正是官方举的例子）把 MinZ 再压一半。
5. **落地校验脚本思路**：用 `get_intrinsics()` 拿到 640×480 的 `fx, fy, cx, cy` → 对末端包络盒 8 个顶点做 `rs2_project_point_to_pixel()`（需要相机外参，可用 `rs2_get_extrinsics()` 或手写 `R, t`）→ 检查像素是否越界。**比用 FOV 估算更准，也顺带把 §1.2 里"640×480 FOV 官方未明确"这个不确定性彻底消掉。**

### 6.6 一句话配方

> **H = 0.55 m（要连基座一起收进画面则 H = 0.70–0.80 m），光轴俯角 δ = 80°（离垂直方向偏 10°），光轴对准作业区足迹中心（δ=80°/H=0.55 时即相机正下方前方约 +11 cm），相机横向偏置使机械臂基座落在画面一侧边缘，作业区放在画面的"近端"、基座在"远端"。**

---

## 7. D435i 采集配置推荐表（RGB-only，ACT 数据采集）

| 项目 | 推荐取值 | 理由 / 官方依据 |
|---|---|---|
| **彩色分辨率** | **640×480** | 与 ACT 输入匹配；带宽极低；USB2 下也能 30 fps（数据手册 Table 4-7）。⚠️ 该模式 FOV 官方未给出，需运行时读内参确认 |
| **彩色帧率** | **30 fps** | 与腕部相机/every-frame 录制对齐；数据手册支持 640×480 @6/15/30/60 |
| **彩色格式** | **`rs.format.bgr8`（pyrealsense2）** | 直接喂 ACT/OpenCV，零额外转换。注意：**设备原生输出是 YUY2 16-bit**（数据手册 Table 4-2），`bgr8`/`rgb8` 是 librealsense 在主机侧转换的结果，会多一次拷贝；RK3588 上 640×480 开销可忽略。若极致省 CPU 可用 `rs.format.yuyv` 自己在 NPU/GPU 侧转换 |
| **深度流** | **暂不开启**（或开 640×480@30 Z16 备用录制） | 只用 RGB 时不开启可省 USB 带宽与主机 CPU；若想顺手留一份深度给 GGCNN，640×480 的 MinZ 175 mm 比 848×480 的 195 mm 更近 |
| **Auto-Exposure Priority** | **0（OFF）** | 官方原文：*"Restrict Auto-Exposure to enforce constant FPS rate. Turn ON to remove the restrictions (may result in FPS drop)"* → **OFF 才保证恒定 30 fps**，ACT 训练绝不允许掉帧 |
| **自动曝光（ENABLE_AUTO_EXPOSURE）** | **先 ON 收敛 → 读回 EXPOSURE → 置 0 并写入锁定值** | 官方未强制锁定；但 AE 在手臂/阴影入画时会重收敛，造成帧间亮度跳变（噪声）。锁定前务必先设 AE Priority=0，保证锁定值 ≤ 33.3 ms |
| **GAIN** | **固定为最低可用值（官方建议 16）** | 官方 tuning 白皮书：*"keeping the GAIN=16 (lowest setting). Increasing the gain tends to introduce electronic noise"*；数据手册 Manual Gain 范围 16–248 |
| **Auto White Balance** | **先 ON 收敛 → 读回色温 → 写入锁定（写入即自动关 AWB）** | 官方：*"Setting any value will disable auto white balance"*；可写范围 **2800–6500 K**（数据手册 Table 4-24）。锁定可消除色温漂移 |
| **Power Line Frequency** | **50 Hz**（中国市电） | 官方 `RS2_OPTION_POWER_LINE_FREQUENCY`：*"anti-flickering Off/50Hz/60Hz/Auto"*；数据手册 Table 4-24 范围 0–3。仅注册在彩色传感器上 |
| **AE ROI（彩色）** | 可选：把 ROI 收到作业区（避开画面边缘的强反光/灯光） | 官方 tuning 白皮书：*"The RGB ROI needs to be set separately... it does impact the color image quality"*；数据手册 AE ROI 范围 T0/L0/B1080/R1920 |
| **IR Emitter / Laser Power** | **采集 RGB 时 `LASER_POWER = 0`（或 `EMITTER_ENABLED = 0`）；将来用深度时改为 `150 mW`（默认）** | 官方：*"0 meaning projector turned off"*；默认 150 mW、可调 0–360 mW（30 mW 步进）。关闭可省热、避免任何 IR 渗色；代价是深度不可用。⚠️ USB2 下这两个选项**不会注册** |
| **帧同步（inter-cam sync）** | **不启用（Default）**，改在录制端用时间戳对齐 | 只有一台 RealSense；官方：不启用硬同步时把 frame queue capacity 设为 2，跨相机错位不超过 1 帧。若将来上第二台 D435i 再启用 Master/Slave + pin5/pin9 |
| **Frame queue capacity** | **2**（彩色+深度都开时）；只开彩色可设 1 | 官方多相机白皮书 §H：*"set the queue size to 1 if enabling depth-only, and 2 if enabling both depth and color streams"* |
| **预热** | **不靠固定秒数；监控 `EXPOSURE` / `ASIC_TEMPERATURE` / `PROJECTOR_TEMPERATURE` 进入平台期再录** | **官方未明确预热时长**；官方仅给出 0–35 °C 环境温度验证范围、>60 °C 激光功率减半、以及「长时间高温高湿会降低 KPI」。现有「丢 3 秒」不足，建议至少连续监控 5–10 分钟 |
| **`rs.align`** | **不用**（只用 RGB）；将来深度+RGB 联合用时，明确目标视口 | 官方 `rs-align` 示例说明 align 会重采样并产生**遮挡无效像素**；且 align-to-color 会把深度视野裁到 RGB 视野 |
| **后处理滤波** | **当前全部关闭**（只用 RGB） | 官方推荐链：`Decimation → Depth2Disparity → Spatial → Temporal → Disparity2Depth → Hole Filling`；参数默认值见 §4.2。将来 GGCNN 接入时：Decimation mag=2，Spatial(α=0.5–0.6, δ=8–20)，Temporal(α=0.4–0.5, δ=20, persistency=3)，**Hole Filling 建议关闭**（官方：机器人/避障场景"leave the holes"），Threshold 位置**官方未明确** |
| **深度预设（将来）** | **High Accuracy** | 官方 tuning 白皮书：*"very good for autonomous robots where false depth, aka. Hallucinations, are much worse than no depth"* |
| **安装** | **H = 0.55 m（含基座则 0.70–0.80 m），δ = 80°，1/4″-20 刚性固定在机械臂同一底板上** | 见 §6；官方产品页给出 1 个 1/4″-20 UNC + 2 个 M3 安装点；相机与臂必须共基座（官方未明确此点，属工程结论） |
| **USB** | **必须 USB 3.x**；线长 < 1 m（> 2 m 用有源 repeater）；优先独立控制器口/带供电 HUB | 官方多相机白皮书 §A/§B/§D；数据手册 USB2 表显示 USB2 下深度最高仅 10 fps |

---

## 8. 明确的"官方未明确"清单（不要当成官方结论）

1. **RGB 在 640×480 下的 HFOV/VFOV** —— 官方未给出；本文的 55°×42.5° 是基于「OV2740 只有 16:9 原生模式，4:3 必然水平裁剪」的**推算**（±5%，因官方数据手册声明 FOV 逐台有 ±5% 的机械公差）。**请用 §1.2 的 `get_intrinsics()` 代码实测覆盖。**
2. **D435i 专用 RGB 模组是否内置 IR-cut 滤镜** —— 数据手册只为 D415/D450/D401 的 Color Sensor 标注 `IR Cut Filter`，为 D435i 未单列该行。
3. **模仿学习数据采集场景下必须锁定白平衡/曝光** —— 官方没有这条规定；本文的「建议锁定」是由官方 option 语义 + 帧间一致性需求推出的工程结论。
4. **预热时长 / AE 热漂移量** —— 官方完全未给出数字。
5. **`threshold_filter` 在官方推荐后处理链中的位置与参数** —— 官方 `post-processing-filters.md` 的推荐链里没有它，仅在示例代码中出现。
6. **相机与机械臂的刚性安装要求** —— 官方对模仿学习无专门说明；「必须共基座」属工程结论。
7. **「Auto Exposure Priority」的默认值** —— 官方未在文档中写默认值（Viewer 里默认 ON）。
8. **Depth Quality Specification 中的 Fill rate / RMS Error / Temporal Noise 具体数值** —— 官方表格以图形呈现，本次文本提取无法可靠还原（只能确认 `Z-accuracy (±2%)`）。**引用这三个指标时请直接看 PDF 原表。**

---

## 9. 与本项目现有实现的差距（Action Items）

| 现状 | 问题 | 建议动作 |
|---|---|---|
| 只有 `rs.format.bgr8` 彩色流 | FOV/内参未确认；`bgr8` 是主机侧转换 | 打印 `get_intrinsics()` 确认 640×480 真实 HFOV/VFOV，用它替换 §6 里的 55° 推算值 |
| 「预热丢 3 秒」 | 远不足以热平衡；官方未给时长 | 改为监控 `EXPOSURE`/`ASIC_TEMPERATURE` 到平台期；同时**锁定曝光与白平衡**（比预热更有效） |
| 没有设 AE Priority | 暗光下帧率会掉，ACT 时间轴错位 | 显式 `enable_option(rs.option.auto_exposure_priority, 0)` |
| 没有 emitter 控制 | 投影器常开：发热 + 可能 IR 渗色；且 USB2 下该选项不存在 | RGB-only 阶段 `laser_power = 0`；顺便用 `rs-enumerate-devices` 确认 USB 类型 |
| 没有 `power_line_frequency` | 50 Hz 工频灯光下逐帧亮度/条纹跳变 | 设 `power_line_frequency = 1`（50 Hz） |
| 没有 `rs.align` | 当前只用 RGB，**不需要 align**（别乱加） | 保持不加；将来深度+RGB 联合时再明确 align 目标视口并接受 FOV 裁剪 |
| 没有后处理滤波 | 当前只用 RGB，**不需要** | 保持关闭；GGCNN 接入时按 §4.2 的官方链与参数开启 |
| 相机安装位置未定量 | 无法保证末端全程在画面内、无法避免自遮挡 | 按 §6.5 的 5 步流程做一次几何校验并固化支架 |
