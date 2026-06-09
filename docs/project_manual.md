# RK3588-EIA 项目完整说明

> **Embedded Intelligent Actuator** — 端侧具身智能执行器
>
> 方案：LeRobot v0.4.4 + VLM (RKLLM NPU) + MobileNet SSD + Astra Pro (RGB OpenCV / Depth SDK) + SO-ARM101

---

## 一、项目架构

```
Astra Pro ─┬── RGB (OpenCV) ──► VLM (RKLLM NPU) ──► "拿红色杯子"
           │                           ↓
           │                    Prompt 解析器
           │                    {object:"bottle", color:"red"}
           │                           ↓
           └── RGB (OpenCV) ──► MobileNet SSD (CPU) ──► 检测 "bottle"
                                          │
                                          ▼
                                     找对应检测框 → HSV 验证颜色
                                          │
                                     ┌────┴────┐
                                     ▼         ▼
                                  匹配     不匹配 → ColorLocator 降级
                                          │
                                          ▼
Astra Pro ── Depth (C++ SDK) ──► (cx, cy) + Depth z
                                          │
                                          ▼
                                     IK 解算 → SO-ARM101 执行
```

## 二、目录结构

```
RK3588-EIA/
├── main.py                      # VLA 自主抓取主入口
├── requirements.txt
├── README.md
│
├── vla/                         # VLA 核心系统
│   ├── vlm/                     #   通用 VLM 框架
│   ├── vision/                  #   双引擎视觉定位
│   │   ├── detector.py          #     MobileNet SSD (CPU)
│   │   └── locator.py           #     ColorLocator (降级)
│   ├── control/controller.py    #   IK + Feetech 串口
│   └── pipe/pipeline.py         #   有限状态机
│
├── astra/                       # Astra Pro 相机
│   ├── camera.py                #   Python 封装
│   ├── capture.cpp              #   C++ Depth 采集
│   └── build/astra_capture      #   编译产物
│
├── lerobot/                     # LeRobot v0.4.4
├── config/settings.py           # 统一配置
├── models/                      # VLM + SSD 模型
├── scripts/                     # 工具脚本
├── tests/                       # 单元测试
└── docs/                        # 文档
```

## 三、LeRobot 说明

已删除训练/RL 等由 VLM 替代的模块，仅保留遥操作必需。

可用命令：

| 命令 | 用途 |
|------|------|
| `lerobot-teleoperate` | 主从遥操作 |
| `lerobot-calibrate` | 机器人校准 |
| `lerobot-find-port` | 查找串口 |
| `lerobot-find-cameras` | 查找相机 |
| `lerobot-setup-motors` | 舵机配置 |
| `lerobot-info` | 系统信息 |

## 四、VLM 多模型支持

| 模型 | 配置名 | 内存 | 推理速度 | 推荐硬件 |
|------|--------|------|---------|---------|
| InternVL2-1B | `internvl2-1b` | ~1 GB | 2-3s | 8GB/16GB |
| Qwen2.5-VL-3B | `qwen2.5-vl-3b` | ~2.5 GB | 4-6s | 16GB |

切换模型：修改 `config/settings.py` 中 `VLM_MODEL_NAME`。

## 五、算力分配

| 硬件 | 任务 | 占用 |
|------|------|------|
| **NPU** | VLM 推理 | 独占 |
| **CPU** | MobileNet SSD 目标检测 | 15-20% |
| **CPU** | ColorLocator 颜色分割 | < 5% |
| **CPU** | IK 解算 + 串口通信 | < 10% |
| **CPU** | LeRobot + 系统任务 | ~ 20% |

## 六、部署流程

```
1. 系统准备
   ├── Ubuntu 22.04 / ARM64
   ├── NPU 驱动 /dev/dri/
   └── sudo apt install python3-pip python3-opencv cmake build-essential

2. Orbbec SDK 安装
   ├── wget 下载 arm64 release zip
   └── sudo cp -r lib/* /usr/local/lib/ && sudo ldconfig

3. 项目安装
   ├── cd lerobot && pip install -e .[feetech] && cd ..
   ├── pip install -r requirements.txt
   ├── python scripts/download_mobilenet.py
   └── cd astra && g++ ... -o build/astra_capture && cd ..

4. VLM 模型部署
   └── RKLLM 模型放到 ./models/InternVL2-1B-rkllm/model.rkllm

5. 标定
   ├── 相机内参 → scripts/calibrate_camera.py
   ├── 手眼标定 → scripts/calibrate_handeye.py
   └── 机器人校准 → lerobot-calibrate

6. 运行
   ├── bash scripts/teleop.sh
   └── bash scripts/run_vla.sh
```

## 七、创新点

1. **纯端侧 VLA 闭环** — 不依赖云端，全链路在 RK3588 本地
2. **VLM-as-Policy 替代 ACT** — 解决 ACT 不兼容 RKNPU 的问题
3. **VLM + MobileNet SSD 双引擎** — VLM 语义决策，SSD 空间定位，无 NPU 竞争
4. **RGB OpenCV + Depth C++ SDK 混搭** — 兼容 Astra Pro 老旧 OpenNI 协议
5. **通用 VLM 框架** — 工厂模式，一行配置切换模型
6. **LeRobot 生态** — 复用开源示教管线
