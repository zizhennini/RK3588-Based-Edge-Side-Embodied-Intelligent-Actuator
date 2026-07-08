"""系统内存监控 — 分级内存管控"""
import time
import gc
import os


# 组件内存预算（MB）
MEMORY_BUDGET = {
    "vlm": 900,
    "recording": 256,
    "tts": 64,
    "asr": 50,
}


class MemoryMonitor:
    """内存监控器，管理组件间的内存争用"""

    def __init__(self, reserve_mb: int = 200):
        self._locks: dict[str, int] = {}
        self._reserve_mb = reserve_mb

    def get_available_mb(self) -> int:
        """读取系统可用内存（MB）"""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
                    if line.startswith("MemFree:"):
                        free = int(line.split()[1]) // 1024
                    if line.startswith("Cached:"):
                        cached = int(line.split()[1]) // 1024
            return free + cached
        except (FileNotFoundError, OSError):
            return 0

    def get_used_mb(self) -> int:
        """读取已用内存（MB）"""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) // 1024
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) // 1024
            return total - avail
        except (FileNotFoundError, OSError):
            return 0

    def acquire(self, component: str) -> bool:
        """申请组件内存预算，返回是否成功"""
        needed = MEMORY_BUDGET.get(component, 256)
        avail = self.get_available_mb()

        # 计算其他已锁定的组件占用
        other_locked = sum(v for k, v in self._locks.items() if k != component)

        if avail - other_locked > needed + self._reserve_mb:
            self._locks[component] = needed
            return True
        return False

    def release(self, component: str):
        """释放组件内存预算"""
        self._locks.pop(component, None)

    def get_holder(self, component: str) -> bool:
        """检查组件是否已持有预算"""
        return component in self._locks

    def force_release_for(self, priority_component: str) -> list[str]:
        """为高优先级组件强制释放冲突组件，返回被释放的组件列表"""
        released = []
        for comp in list(self._locks.keys()):
            if comp != priority_component:
                self._locks.pop(comp, None)
                released.append(comp)
        gc.collect()
        return released

    def summary(self) -> str:
        """返回内存状态摘要"""
        avail = self.get_available_mb()
        used = self.get_used_mb()
        locked = sum(self._locks.values())
        comps = ",".join(self._locks.keys()) or "无"
        return f"内存 {used}MB/已用 {avail}MB/可用 锁定{locked}MB({comps})"


class MemoryLimiter:
    """内存限制上下文管理器 — 用于录制等大内存操作"""

    def __init__(self, monitor: MemoryMonitor, component: str):
        self.monitor = monitor
        self.component = component
        self._acquired = False

    def __enter__(self):
        # 如果需要，先释放冲突组件
        if self.component == "recording":
            self.monitor.force_release_for("recording")
        self._acquired = self.monitor.acquire(self.component)
        if not self._acquired:
            raise MemoryError(f"无法分配 {self.component} 所需内存")
        return self

    def __exit__(self, *args):
        self.monitor.release(self.component)
