#!/usr/bin/env python3
"""ELF2 RK3588 自主抓取项目 -- 统一入口

Usage:
    python main.py --mode autonomous    # 自主抓取
    python main.py --mode voice         # 语音交互
    python main.py --mode teleop        # 遥操作（需连接主臂）
    python main.py --mode menu          # 交互菜单（默认）
    python main.py --mode record        # 数据录制
"""
import argparse
import logging
import signal
import sys
import time
from typing import Optional

# 注: 子进程 + 共享内存运行时（规避 GIL 抖动）见 runtime/ 包，
# 由 config.settings.USE_SUBPROCESS_RUNTIME 控制启用（默认进程内路径）。

logger = logging.getLogger(__name__)


class System:
    """统一管理所有硬件资源的生命周期"""

    def __init__(self, config: dict = None):
        self.config = config or self._load_default_config()

        # 硬件模块（按需初始化）
        self.arm = None           # SO101Arm
        self.camera = None        # CameraManager
        self.safety = None        # SafetyMonitor

        # 策略模块（由 init_policy() 按需加载）
        self.vlm = None           # VLMPerception
        self.act_policy = None    # ACTPolicy（ACT 连续控制策略）
        self.ggcnn = None         # GGCNNDetector
        self.grasp_pipeline = None  # GraspPipeline（统一抓取入口）

        # 兼容旧字段名（与 _check_degradation 对齐）
        self.act = None

        # 语音模块（后续迁移）
        self.voice = None         # VoiceAssistant

        # 状态
        self._running = False
        self._mode = "menu"
        self._arm_initialized = False
        self._camera_initialized = False
        self._safety_initialized = False
        self._policy_initialized = False

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_default_config(self) -> dict:
        """加载默认配置（从 config/settings.py）"""
        try:
            from config import settings
            return {
                "serial_port": settings.SERIAL_PORT,
                "serial_baud": settings.SERIAL_BAUD,
                "camera_position": settings.CAMERA_POSITION.tolist()
                    if hasattr(settings.CAMERA_POSITION, "tolist")
                    else list(settings.CAMERA_POSITION),
                "camera_width": 640,
                "camera_height": 480,
                "camera_fps": 30,
            }
        except ImportError:
            logger.warning("无法加载 config.settings，使用默认值")
            return {
                "serial_port": "/dev/ttyACM0",
                "serial_baud": 1000000,
                "camera_position": [0.182, -0.129, 0.47],
                "camera_width": 640,
                "camera_height": 480,
                "camera_fps": 30,
            }

    # ------------------------------------------------------------------
    # 硬件生命周期（细粒度按需初始化）
    # ------------------------------------------------------------------

    def init_arm(self) -> bool:
        """初始化机械臂（串口）

        Returns:
            True 表示成功，False 表示失败
        """
        if self._arm_initialized:
            return self.arm is not None
        self._arm_initialized = True

        try:
            from hardware.arm import SO101Arm
            from policy.kinematics import Kinematics
            self.arm = SO101Arm.get_instance(
                port=self.config["serial_port"],
                baud=self.config["serial_baud"],
            )
            # 组合根注入 kinematics：硬件层不再反向依赖策略层（修复依赖倒置债）
            # 用 set_kinematics 而非构造参数，确保单例已存在时也能注入
            self.arm.set_kinematics(Kinematics())
            self.arm.connect()
            logger.info("SO101Arm 已连接")
            return True
        except Exception as e:
            logger.error(f"SO101Arm 初始化失败: {e}")
            self.arm = None
            return False

    def init_camera(self) -> bool:
        """初始化相机

        Returns:
            True 表示成功，False 表示失败
        """
        if self._camera_initialized:
            return self.camera is not None
        self._camera_initialized = True

        try:
            from hardware.camera_d435i import CameraManager
            self.camera = CameraManager(
                width=self.config["camera_width"],
                height=self.config["camera_height"],
                fps=self.config["camera_fps"],
            )
            self.camera.start()
            logger.info("CameraManager 已启动")
            return True
        except Exception as e:
            logger.error(f"CameraManager 初始化失败: {e}")
            self.camera = None
            return False

    def init_safety(self) -> bool:
        """初始化安全监控（依赖 arm，可选依赖 camera）

        Returns:
            True 表示成功，False 表示失败
        """
        if self._safety_initialized:
            return self.safety is not None
        self._safety_initialized = True

        try:
            from hardware.safety import SafetyMonitor
            self.safety = SafetyMonitor(
                arm=self.arm,
                camera=self.camera,
            )
            self.safety.start()
            logger.info("SafetyMonitor 已启动")
            return True
        except Exception as e:
            logger.error(f"SafetyMonitor 初始化失败: {e}")
            self.safety = None
            return False

    def init_hardware(self):
        """初始化全部硬件（仅 menu 模式调用，用于诊断）"""
        logger.info("初始化全部硬件...")
        self.init_arm()
        self.init_camera()
        self.init_safety()

        # 机械臂是硬性依赖
        if self.arm is None:
            logger.critical("机械臂不可用，系统拒绝启动")
            raise RuntimeError("SO101Arm 初始化失败")

    # ------------------------------------------------------------------
    # 策略生命周期（ACT + GGCNN + VLM 降级加载）
    # ------------------------------------------------------------------

    def init_policy(self) -> bool:
        """初始化策略模块（需在 init_arm + init_camera 之后调用）。

        加载顺序（任一失败不影响系统启动）：
            1. ACTPolicy   -> 连续控制（models/act/*.onnx）
            2. GGCNNDetector -> 三段式抓取兼底（models/ggcnn/*.onnx）
            3. VLMPerception -> 目标定位（可选）
            4. GraspPipeline -> 统一抓取入口，根据 ACT 可用性设定默认模式

        Returns:
            True 表示 GraspPipeline 创建成功（不保证 ACT/GGCNN 均可用）
        """
        if self._policy_initialized:
            return self.grasp_pipeline is not None
        self._policy_initialized = True

        if self.arm is None or self.camera is None:
            logger.warning("init_policy: arm/camera 未就绪，GraspPipeline 不可用")

        # ---- 1. ACT Policy ----
        try:
            from policy.act_policy import ACTPolicy
            act = ACTPolicy()
            act.setup({
                "model_dir": self.config.get("act_model_dir", "models/act"),
            })
            if act.is_available:
                self.act_policy = act
                self.act = act  # 兼容旧字段
                logger.info("ACT Policy 已加载")
            else:
                logger.info(
                    "ACT Policy 不可用（模型文件缺失），将使用 GGCNN 兜底: %s",
                    act.setup_error or "unknown",
                )
                self.act_policy = None
        except Exception as e:
            logger.warning(f"ACT Policy 加载失败: {e}")
            self.act_policy = None

        # ---- 2. GGCNN ----
        try:
            from perception.grasp_detect import GGCNNDetector
            ggcnn = GGCNNDetector()
            ggcnn.setup({
                "model_path": self.config.get(
                    "ggcnn_model_path",
                    "models/ggcnn/ggcnn_cornell_300.onnx"),
            })
            if ggcnn.is_available:
                self.ggcnn = ggcnn
                logger.info("GGCNN 已加载")
            else:
                logger.info("GGCNN 不可用（模型文件缺失）")
                self.ggcnn = None
        except Exception as e:
            logger.warning(f"GGCNN 加载失败: {e}")
            self.ggcnn = None

        # ---- 3. VLM ----
        try:
            from perception.vlm import VLMPerception
            vlm = VLMPerception()
            vlm.setup(self.config.get("vlm_config", {}))
            if vlm.is_available:
                self.vlm = vlm
                logger.info("VLM 已加载")
            else:
                logger.info("VLM 不可用")
                self.vlm = None
        except Exception as e:
            logger.warning(f"VLM 加载失败: {e}")
            self.vlm = None

        # ---- 4. GraspPipeline ----
        try:
            from policy.grasp_pipeline import GraspPipeline
            self.grasp_pipeline = GraspPipeline(
                arm=self.arm,
                camera=self.camera,
                vlm=self.vlm,
                ggcnn=self.ggcnn,
                act_policy=self.act_policy,
            )
            self.grasp_pipeline.setup(self.config.get("grasp_pipeline_config", {}))
        except Exception as e:
            logger.error(f"GraspPipeline 创建失败: {e}")
            self.grasp_pipeline = None
            return False

        # ---- 5. 默认模式: ACT 优先，GGCNN 兜底 ----
        if self.act_policy is not None:
            self.grasp_pipeline.use_act = True
            logger.info("抓取模式: ACT 策略（连续控制）")
        else:
            self.grasp_pipeline.use_act = False
            if self.ggcnn is not None:
                logger.info("抓取模式: GGCNN 兜底（三段式）")
            else:
                logger.warning("抓取模式: 无可用策略（ACT/GGCNN 均缺失）")

        return True

    # ------------------------------------------------------------------
    # 语音模块生命周期（可选，缺失时降级）
    # ------------------------------------------------------------------

    def init_voice(self) -> bool:
        """初始化语音模块（sherpa-onnx KWS/ASR/TTS + 意图路由）。

        语音为非关键模块：依赖缺失时优雅降级，不影响核心抓取。

        Returns:
            True 表示 VoiceAssistant 可用并已启动
        """
        try:
            from voice.assistant import VoiceAssistant
            assistant = VoiceAssistant()
            assistant.setup({
                "config_path": self.config.get(
                    "voice_config_path", "voice/config/default.yaml"),
                "on_intent": self._handle_voice_intent,
            })
            if assistant.is_available:
                self.voice = assistant
                self.voice.start()
                logger.info("VoiceAssistant 已启动")
                return True
            logger.info("VoiceAssistant 不可用（依赖缺失）")
            self.voice = None
            return False
        except Exception as e:
            logger.warning(f"VoiceAssistant 初始化失败: {e}")
            self.voice = None
            return False

    def _handle_voice_intent(self, intent: str, params: dict):
        """语音意图处理回调 -- 连接到 GraspPipeline / arm / vlm"""
        logger.info(f"语音意图: {intent}, 参数: {params}")
        params = params or {}

        if intent == "grasp":
            if self.grasp_pipeline is None:
                if self.voice:
                    self.voice.say("抓取功能不可用")
                return
            target = params.get("target", "")
            if self.voice:
                self.voice.say(f"正在抓取{target}")
            try:
                result = self.grasp_pipeline.execute_grasp(target)
            except Exception as e:
                logger.error(f"语音触发抓取异常: {e}")
                if self.voice:
                    self.voice.say("抓取出现异常")
                return
            if self.voice:
                if result.success:
                    self.voice.say("抓取成功")
                else:
                    self.voice.say(f"抓取失败，{result.message}")

        elif intent == "stop":
            if self.arm:
                try:
                    self.arm.emergency_stop()
                except Exception as e:
                    logger.error(f"语音急停失败: {e}")
            if self.voice:
                self.voice.say("已急停")

        elif intent == "home":
            if self.arm:
                try:
                    self.arm.home()
                except Exception as e:
                    logger.error(f"语音归零失败: {e}")
            if self.voice:
                self.voice.say("已归零")

        elif intent == "ask":
            question = params.get("question", "")
            logger.info(f"VLM 问答（待实现）: {question}")
            if self.voice:
                self.voice.say("我还在思考中")

        else:
            logger.warning(f"未知语音意图: {intent}")

    def shutdown(self):
        """逆序释放所有资源"""
        logger.info("正在关闭系统...")
        self._running = False

        # 语音模块先于安全监控释放
        if self.voice:
            try:
                self.voice.stop()
            except Exception as e:
                logger.error(f"VoiceAssistant 关闭失败: {e}")
            self.voice = None

        # 逆序释放: grasp_pipeline -> act/ggcnn/vlm -> safety -> camera -> arm
        if self.grasp_pipeline is not None:
            try:
                self.grasp_pipeline.stop()
            except Exception as e:
                logger.error(f"GraspPipeline 关闭失败: {e}")
            self.grasp_pipeline = None

        if self.act_policy is not None:
            try:
                self.act_policy.stop()
            except Exception as e:
                logger.error(f"ACTPolicy 关闭失败: {e}")
            self.act_policy = None
            self.act = None

        if self.ggcnn is not None:
            try:
                self.ggcnn.stop()
            except Exception as e:
                logger.error(f"GGCNNDetector 关闭失败: {e}")
            self.ggcnn = None

        if self.vlm is not None:
            try:
                self.vlm.stop()
            except Exception as e:
                logger.error(f"VLMPerception 关闭失败: {e}")
            self.vlm = None

        if self.safety:
            try:
                self.safety.stop()
            except Exception as e:
                logger.error(f"SafetyMonitor 关闭失败: {e}")
            self.safety = None

        if self.camera:
            try:
                self.camera.stop()
            except Exception as e:
                logger.error(f"CameraManager 关闭失败: {e}")
            self.camera = None

        if self.arm:
            try:
                self.arm.close()
            except Exception as e:
                logger.error(f"SO101Arm 关闭失败: {e}")
            self.arm = None

        logger.info("系统已关闭")

    def _signal_handler(self, signum, frame):
        """信号处理：优雅关闭"""
        logger.info(f"收到信号 {signum}，准备关闭...")
        self.shutdown()
        sys.exit(0)

    # ------------------------------------------------------------------
    # 降级检查
    # ------------------------------------------------------------------

    def _check_degradation(self) -> dict:
        """检查模块可用性，返回降级策略"""
        act_ok = (self.act_policy is not None
                  and getattr(self.act_policy, "is_available", False))
        ggcnn_ok = (self.ggcnn is not None
                    and getattr(self.ggcnn, "is_available", False))
        vlm_ok = (self.vlm is not None
                  and getattr(self.vlm, "is_available", False))

        status = {
            "arm_available": self.arm is not None,
            "camera_available": self.camera is not None,
            "vlm_available": vlm_ok,
            "act_available": act_ok,
            "ggcnn_available": ggcnn_ok,
            "voice_available": self.voice is not None,
            "grasp_pipeline_available": self.grasp_pipeline is not None,
            "grasp_mode": (
                "act" if (self.grasp_pipeline is not None
                          and self.grasp_pipeline.use_act)
                else "ggcnn" if ggcnn_ok
                else "none"
            ),
        }

        # 确定可用模式
        if not status["arm_available"]:
            status["available_modes"] = []
            status["degradation"] = "abort"
        elif not status["camera_available"]:
            status["available_modes"] = ["teleop", "replay"]
            status["degradation"] = "limited"
        else:
            modes = ["teleop", "replay"]
            if status["grasp_pipeline_available"] and (act_ok or ggcnn_ok):
                modes.append("autonomous")
            if status["voice_available"]:
                modes.append("voice")
            status["available_modes"] = sorted(set(modes))
            status["degradation"] = (
                "full" if len(status["available_modes"]) >= 4 else "partial"
            )

        return status

    # ------------------------------------------------------------------
    # 模式入口
    # ------------------------------------------------------------------

    def run_menu(self):
        """交互菜单模式 -- 初始化全部硬件用于诊断"""
        self.init_hardware()
        self.init_policy()

        status = self._check_degradation()
        print("\n=== ELF2 RK3588 自主抓取系统 ===")
        print(f"降级状态: {status['degradation']}")
        print(f"可用模式: {', '.join(status['available_modes']) or '无'}")
        print(f"  机械臂: {'OK' if status['arm_available'] else 'FAIL'}")
        print(f"  相机:   {'OK' if status['camera_available'] else 'FAIL'}")
        print(f"  VLM:    {'OK' if status['vlm_available'] else 'N/A'}")
        print(f"  ACT:    {'OK' if status['act_available'] else 'N/A'}")
        print(f"  GGCNN:  {'OK' if status['ggcnn_available'] else 'N/A'}")
        print(f"  语音:   {'OK' if status['voice_available'] else 'N/A'}")
        print(f"  抓取模式: {status['grasp_mode']}")
        print("\n模式切换: python main.py --mode <mode>")
        print("按 Ctrl+C 退出\n")

        self._running = True
        while self._running:
            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                break
        self.shutdown()

    def run_autonomous(self):
        """自主抓取模式 -- 需要 arm + camera + GraspPipeline。

        交互循环：
            读取 stdin 目标描述 -> grasp_pipeline.execute_grasp(target)
            输入空行 / 'q' / EOF 退出
        """
        logger.info("自主抓取模式启动")

        # 1. 硬件初始化
        if not self.init_arm():
            logger.critical("自主抓取模式: 机械臂不可用")
            return
        if not self.init_camera():
            logger.critical("自主抓取模式: 相机不可用")
            return
        self.init_safety()

        # 2. 策略初始化（ACT 优先，GGCNN 兜底）
        if not self.init_policy():
            logger.critical("自主抓取模式: GraspPipeline 创建失败")
            return

        status = self._check_degradation()
        if not status["act_available"] and not status["ggcnn_available"]:
            logger.critical("自主抓取模式: ACT/GGCNN 均不可用，无法抓取")
            self.shutdown()
            return

        logger.info("自主抓取就绪：模式=%s【输入目标描述后回车执行，'q' 退出】",
                    status["grasp_mode"])
        print("\n[自主抓取模式]")
        print(f"  当前策略: {status['grasp_mode'].upper()}")
        print("  输入目标描述（如：红色杯子）后回车开始抓取；输入 'q' 退出")
        print("  输入 'mode act' / 'mode ggcnn' 运行时切换模式\n")

        self._running = True
        try:
            while self._running:
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not line:
                    continue
                if line.lower() in ("q", "quit", "exit"):
                    break

                # 运行时模式切换
                if line.lower().startswith("mode"):
                    parts = line.split()
                    if len(parts) >= 2:
                        target_mode = parts[1].lower()
                        if target_mode == "act":
                            if status["act_available"]:
                                self.grasp_pipeline.use_act = True
                                print("  已切换到 ACT 模式")
                            else:
                                print("  ACT 不可用，无法切换")
                        elif target_mode == "ggcnn":
                            self.grasp_pipeline.use_act = False
                            print("  已切换到 GGCNN 模式")
                        else:
                            print("  未知模式，可选: act / ggcnn")
                    continue

                # 执行抓取
                logger.info("执行抓取: target='%s'", line)
                result = self.grasp_pipeline.execute_grasp(line)
                if result.success:
                    print(f"  [成功] {result.message}")
                else:
                    print(f"  [失败] {result.message}")
                if result.data:
                    logger.debug("  抓取详情: %s", result.data)
        finally:
            self.shutdown()

    def run_voice(self):
        """语音交互模式 -- 需要 arm + camera + voice

        唤醒词 -> ASR -> 意图路由 -> GraspPipeline / arm，
        结果通过 TTS 播报。
        """
        logger.info("语音交互模式启动")

        # 1. 硬件初始化
        if not self.init_arm():
            logger.critical("语音交互模式: 机械臂不可用")
            return
        self.init_camera()
        self.init_safety()

        # 2. 策略初始化（供“抓取”意图使用，ACT 优先 GGCNN 兑底）
        self.init_policy()

        # 3. 语音初始化
        self.init_voice()
        if not self.voice:
            logger.error("语音模块不可用，无法进入语音模式")
            self.shutdown()
            return

        print("\n[语音交互模式]")
        print("  唤醒词：“你好同学” / “拍照助手”")
        print("  示例指令：“抓取红色杯子” / “归零” / “停止”")
        print("  按 Ctrl+C 退出\n")

        self._running = True
        try:
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def run_teleop(self):
        """遥操作模式 -- 仅需 arm"""
        logger.info("遥操作模式启动")

        # 按需初始化: 仅 arm
        if not self.init_arm():
            logger.critical("遥操作模式: 机械臂不可用")
            return

        self._running = True
        # TODO: 遥操作实现
        logger.warning("遥操作模式尚未完整实现")
        while self._running:
            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                break
        self.shutdown()

    def run_record(self):
        """数据录制模式 -- 相机必需，机械臂可选

        当前实现 (refactor_plan_v9 §6.3): H264Encoder 硬件录像
        (ffmpeg + h264_rkmpp, CPU 占用近零)，Ctrl+C 停止并落盘。
        P2 阶段扩展: 关节状态 + 图像同步的完整数据录制 (HDF5)。
        """
        logger.info("数据录制模式启动")

        # 按需初始化: 相机必需；机械臂失败仅降级（只录视频）
        if not self.init_arm():
            logger.warning("数据录制模式: 机械臂不可用，仅录制视频")
        if not self.init_camera():
            logger.critical("数据录制模式: 相机不可用")
            return

        # 硬件编码器（不可用时按 on_failure="skip" 跳过，不拖垮系统）
        try:
            from hardware.encoder import H264Encoder
            encoder = H264Encoder()
            encoder.setup({
                "fps": self.config.get("record_fps", 15),
                "bitrate": self.config.get("record_bitrate", "5M"),
                "out_dir": self.config.get("record_dir", "./recordings"),
            })
        except Exception as e:
            logger.critical("H264Encoder 加载失败: %s", e)
            return
        if not encoder.is_available:
            logger.critical("H264Encoder 不可用（缺少 ffmpeg 或 H.264 编码器）")
            return

        out = encoder.open(
            width=self.config["camera_width"],
            height=self.config["camera_height"],
        )
        if out is None:
            logger.critical("录像启动失败")
            return
        logger.info("录像中: %s （Ctrl+C 停止）", out)

        self._running = True
        fps = self.config.get("record_fps", 15)
        interval = 1.0 / fps
        try:
            while self._running:
                t0 = time.time()
                rgb = self.camera.get_rgb()
                if rgb is not None:
                    encoder.write_frame(rgb)
                dt = time.time() - t0
                if dt < interval:
                    time.sleep(interval - dt)
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C，停止录像")
        finally:
            encoder.close()
        self.shutdown()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, mode: str = "menu"):
        """主入口 -- 不直接初始化硬件，由各模式按需触发"""
        self._mode = mode

        mode_handlers = {
            "menu": self.run_menu,
            "autonomous": self.run_autonomous,
            "voice": self.run_voice,
            "teleop": self.run_teleop,
            "record": self.run_record,
        }

        handler = mode_handlers.get(mode)
        if handler is None:
            logger.error(f"未知模式: {mode}")
            print(f"可用模式: {', '.join(mode_handlers.keys())}")
            self.shutdown()
            sys.exit(1)

        try:
            handler()
        except Exception as e:
            logger.error(f"模式 {mode} 运行异常: {e}")
            self.shutdown()
            raise


def main():
    parser = argparse.ArgumentParser(description="ELF2 RK3588 自主抓取系统")
    parser.add_argument(
        "--mode", type=str, default="menu",
        choices=["autonomous", "voice", "teleop", "menu", "record"],
        help="运行模式 (default: menu)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (default: INFO)",
    )
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 创建系统并运行
    system = System()
    system.run(mode=args.mode)


if __name__ == "__main__":
    main()
