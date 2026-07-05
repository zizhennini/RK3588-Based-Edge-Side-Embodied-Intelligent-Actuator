"""CPU 大小核亲和性绑定工具 — RK3588 专用"""
import os
import subprocess

# RK3588 核心布局
# Cortex-A76 (大核): 0, 1, 2, 3
# Cortex-A55 (小核): 4, 5, 6, 7
BIG_CORES = {0, 1, 2, 3}
LITTLE_CORES = {4, 5, 6, 7}
ALL_CORES = BIG_CORES | LITTLE_CORES


def bind_current_thread(cores: set[int]):
    """将当前线程/进程绑定到指定核心"""
    try:
        os.sched_setaffinity(0, cores)
        return True
    except AttributeError:
        return False
    except OSError:
        return False


def bind_process(pid: int, cores: set[int]):
    """将指定 PID 绑定到指定核心"""
    try:
        os.sched_setaffinity(pid, cores)
        return True
    except (AttributeError, OSError, ProcessLookupError):
        return False


def bind_subprocess_args(cores: set[int]):
    """返回 taskset 包装参数，用于 subprocess 启动时绑定核心"""
    core_list = ",".join(str(c) for c in sorted(cores))
    return ["taskset", "-c", core_list]


def make_preexec_bind(cores: set[int]):
    """返回 preexec_fn，用于 subprocess.Popen 绑核"""
    def _bind():
        try:
            os.sched_setaffinity(0, cores)
        except OSError:
            pass
    return _bind


def get_current_affinity() -> set[int] | None:
    """获取当前线程的 CPU 亲和性掩码"""
    try:
        return os.sched_getaffinity(0)
    except AttributeError:
        return None


def affinity_summary() -> str:
    """返回当前绑核状态摘要"""
    aff = get_current_affinity()
    if aff is None:
        return "绑核不可用（非 Linux）"
    on_big = aff & BIG_CORES
    on_little = aff & LITTLE_CORES
    parts = []
    if on_big:
        parts.append(f"大核({sorted(on_big)})")
    if on_little:
        parts.append(f"小核({sorted(on_little)})")
    return f"当前绑定 {'+'.join(parts)}" if parts else "未绑定"


def verify_rk3588() -> bool:
    """验证是否在 RK3588 上运行，返回 False 则跳过绑核"""
    try:
        with open("/proc/cpuinfo") as f:
            data = f.read()
        return "RK3588" in data
    except (FileNotFoundError, OSError):
        return False


def current_core_id() -> int | None:
    """返回当前线程运行的物理 CPU 核心编号"""
    try:
        return os.sched_getcpu()
    except (AttributeError, OSError):
        return None
