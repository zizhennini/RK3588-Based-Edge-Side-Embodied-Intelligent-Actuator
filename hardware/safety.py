"""安全监控模块 -- 深度避障 + 电流监测 + 急停"""
import threading
import logging
import time
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """安全监控器 -- 独立线程运行，最高优先级"""

    def __init__(self, arm, camera=None,
                 min_obstacle_distance: float = 0.05,
                 max_current_threshold: float = 2.0,
                 check_frequency: int = 10):
        """
        Args:
            arm: SO101Arm 实例
            camera: CameraManager 实例（可选）
            min_obstacle_distance: 最小避障距离 (m)
            max_current_threshold: 最大电流阈值 (A)
            check_frequency: 检查频率 (Hz)
        """
        self.arm = arm
        self.camera = camera
        self.min_obstacle_distance = min_obstacle_distance
        self.max_current_threshold = max_current_threshold
        self.check_frequency = check_frequency

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._emergency_active = False

    def start(self):
        """启动安全监控线程"""
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SafetyMonitor")
        self._thread.start()
        logger.info("SafetyMonitor 已启动")

    def _monitor_loop(self):
        """监控主循环"""
        # 绑核到 A55 核 3
        try:
            import os
            os.sched_setaffinity(0, {3})
        except (AttributeError, OSError):
            pass

        period = 1.0 / self.check_frequency
        while self._running:
            try:
                self._check_obstacles()
                self._check_current()
            except Exception as e:
                logger.error(f"安全监控异常: {e}")
                self.trigger_emergency_stop()
            time.sleep(period)

    def _check_obstacles(self):
        """深度相机避障检查"""
        if self.camera is None:
            return
        frame = self.camera.get_frame()
        if frame is None:
            return
        _, depth, _ = frame
        # 检查中心区域最小深度
        h, w = depth.shape
        center_region = depth[h // 3:2 * h // 3, w // 3:2 * w // 3]
        valid = center_region[center_region > 0]
        if len(valid) > 0:
            min_dist = np.min(valid)
            if min_dist < self.min_obstacle_distance:
                logger.warning(f"障碍物过近: {min_dist:.3f}m")

    def _check_current(self):
        """电流监测（如果舵机支持）"""
        pass  # 后续实现

    def trigger_emergency_stop(self):
        """触发急停"""
        if not self._emergency_active:
            self._emergency_active = True
            logger.critical("急停触发!")
            try:
                self.arm.emergency_stop()
            except Exception as e:
                logger.error(f"急停失败: {e}")

    def clear_emergency(self):
        """清除急停状态"""
        self._emergency_active = False
        logger.info("急停已清除")

    def stop(self):
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("SafetyMonitor 已停止")

    @property
    def is_emergency(self) -> bool:
        return self._emergency_active
