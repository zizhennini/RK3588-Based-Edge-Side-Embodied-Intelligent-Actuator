# runtime/__init__.py
"""子进程 + 共享内存运行时（refactor_plan_v9 §1.6 / §4.1）

本包提供"子进程隔离 CPU 推理 + 共享内存帧传输"架构的可复用组件，
用于规避 Python GIL 抖动、适配 RK3588 NPU 单进程单核限制。

组件:
    - SharedFrameBuffer (shared_frame): 跨进程零拷贝帧传输（seqlock 同步）
    - SubprocessWorker  (worker):       子进程工作器基类（Queue 控制 + 绑核）

状态: opt-in（默认关闭，见 config.settings.USE_SUBPROCESS_RUNTIME）。
      SharedFrameBuffer 已通过单进程往返自检；完整多进程编排
      （相机/ACT/GGCNN 各独立子进程）需板端实测后启用（T4.2 / 风险 R2）。
"""
from runtime.shared_frame import SharedFrameBuffer

__all__ = ["SharedFrameBuffer"]

try:
    from runtime.worker import SubprocessWorker, InferenceWorker
    __all__ += ["SubprocessWorker", "InferenceWorker"]
except ImportError:
    # worker 依赖 multiprocessing，缺失时不阻断 shared_frame 使用
    pass
