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
import multiprocessing as mp
from typing import Optional

logger = logging.getLogger(__name__)


class System:
    """统一管理所有硬件资源的生命周期"""

    def __init__(self, config: dict = None):
        self.config = config or self._load_default_config()

        # 硬件模块（按需初始化）
        self.arm = None           # SO101Arm
        self.camera = None        # CameraManager
        self.safety = None        # SafetyMonitor

        # 策略模块（后续阶段实现）
        self.vlm = None           # VLM（子进程）
        self.act = None           # ACTPolicy（子进程）
        self.ggcnn = None         # GGCNNDetector（子进程）

        # 语音模块（后续迁移）
        self.voice = None         # VoiceAssistant

        # 状态
        self._running = False
        self._mode = "menu"
        self._arm_initialized = False
        self._camera_initialized = False
        self._safety_initialized = False

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
            self.arm = SO101Arm.get_instance(
                port=self.config["serial_port"],
                baud=self.config["serial_baud"],
            )
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

    def shutdown(self):
        """逆序释放所有资源"""
        logger.info("正在关闭系统...")
        self._running = False

        # 逆序释放: safety -> camera -> arm
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
        status = {
            "arm_available": self.arm is not None,
            "camera_available": self.camera is not None,
            "vlm_available": self.vlm is not None,
            "act_available": self.act is not None,
            "ggcnn_available": self.ggcnn is not None,
            "voice_available": self.voice is not None,
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
            if status["ggcnn_available"]:
                modes.append("autonomous")
            if status["act_available"]:
                modes.append("autonomous")
            if status["voice_available"]:
                modes.append("voice")
            status["available_modes"] = sorted(set(modes))
            status["degradation"] = "full" if len(status["available_modes"]) >= 4 else "partial"

        return status

    # ------------------------------------------------------------------
    # 模式入口
    # ------------------------------------------------------------------

    def run_menu(self):
        """交互菜单模式 -- 初始化全部硬件用于诊断"""
        self.init_hardware()

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
        """自主抓取模式 -- 需要 arm + camera"""
        logger.info("自主抓取模式启动")

        # 按需初始化: arm + camera（后续阶段增加 vlm/ggcnn）
        if not self.init_arm():
            logger.critical("自主抓取模式: 机械臂不可用")
            return
        self.init_camera()
        self.init_safety()

        self._running = True
        # TODO: 第二阶段实现 GGCNN/VLM 抓取管线
        logger.warning("自主抓取模式尚未完整实现（待第二阶段）")
        while self._running:
            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                break
        self.shutdown()

    def run_voice(self):
        """语音交互模式 -- 需要 arm + camera + voice"""
        logger.info("语音交互模式启动")

        # 按需初始化: arm + camera（后续增加 voice）
        if not self.init_arm():
            logger.critical("语音交互模式: 机械臂不可用")
            return
        self.init_camera()
        self.init_safety()

        self._running = True
        # TODO: 语音模块迁移后实现
        logger.warning("语音交互模式尚未完整实现（待语音模块迁移）")
        while self._running:
            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                break
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
        """数据录制模式 -- 需要 arm + camera"""
        logger.info("数据录制模式启动")

        # 按需初始化: arm + camera
        if not self.init_arm():
            logger.critical("数据录制模式: 机械臂不可用")
            return
        self.init_camera()

        self._running = True
        # TODO: 数据录制实现（P2 阶段）
        logger.warning("数据录制模式尚未完整实现（P2 阶段）")
        while self._running:
            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                break
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
