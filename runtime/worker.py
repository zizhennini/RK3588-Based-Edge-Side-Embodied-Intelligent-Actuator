# runtime/worker.py
"""子进程工作器 — refactor_plan_v9 §1.6 / §4.1 落地组件

方案要点:
  - "子进程（非多线程）用于 CPU 推理"，规避 GIL 抖动（C5）
  - "ACT/GGCNN/VLM 各走独立进程"，适配 RK3588 NPU 单进程单核限制（C4）
  - "进程间通信: multiprocessing.Queue + shared_memory（帧数据走共享内存）"
  - "子进程不继承父进程 CPU 绑核亲和性，每个子进程内部必须重新调用
     os.sched_setaffinity()"（§1.6 注意 / 风险表）

本模块提供:
  - SubprocessWorker: 子进程工作器基类（spawn 上下文 + 命令/响应 Queue + 子进程内绑核）
  - InferenceWorker:  ACT/GGCNN 推理子进程（模型在子进程内构建，帧走 SharedFrameBuffer）

状态: 脚手架（opt-in）。SubprocessWorker 基于标准 multiprocessing，逻辑稳定；
      InferenceWorker 的模型集成需板端实测（Windows 侧无法验证 onnxruntime/NPU/spawn）。
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Optional

logger = logging.getLogger(__name__)


def rebind_affinity(cores) -> None:
    """子进程内重新绑核（子进程不继承父进程亲和性 — 方案关键点）

    Windows / 不支持 sched_setaffinity 的平台静默跳过。
    """
    if not cores:
        return
    try:
        os.sched_setaffinity(0, set(cores))
        logger.debug("子进程绑核: %s", sorted(cores))
    except (AttributeError, OSError):
        pass


class SubprocessWorker:
    """子进程工作器基类

    控制/小数据走 multiprocessing.Queue；大帧数据走 SharedFrameBuffer（见 shared_frame）。
    使用 spawn 上下文（不继承父进程状态/锁，跨平台一致，避免 fork 死锁）。

    子类实现 run_body(cmd_q, resp_q, stop_evt)，该方法在 **子进程** 内运行。
    """

    def __init__(self, name: str, cores=None):
        self.name = name
        self.cores = set(cores) if cores else None
        self._proc = None
        self._cmd_q = None
        self._resp_q = None
        self._stop_evt = None

    # ------------------------------------------------------------------
    # 生命周期（父进程侧）
    # ------------------------------------------------------------------
    def start(self) -> None:
        ctx = mp.get_context("spawn")
        self._cmd_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._stop_evt = ctx.Event()
        self._proc = ctx.Process(
            target=self._child_main, name=self.name, daemon=True)
        self._proc.start()
        logger.info("SubprocessWorker 启动: %s (pid=%s, cores=%s)",
                    self.name, self._proc.pid, sorted(self.cores) if self.cores else None)

    def _child_main(self) -> None:
        """子进程入口: 先绑核，再运行 body"""
        rebind_affinity(self.cores)
        try:
            self.run_body(self._cmd_q, self._resp_q, self._stop_evt)
        except Exception as e:  # noqa: BLE001
            logger.error("SubprocessWorker %s 异常退出: %s", self.name, e)

    def run_body(self, cmd_q, resp_q, stop_evt) -> None:
        """子类实现: 在子进程内运行的主循环"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 消息收发（父进程侧）
    # ------------------------------------------------------------------
    def send(self, cmd) -> None:
        if self._cmd_q is not None:
            self._cmd_q.put(cmd)

    def recv(self, timeout: Optional[float] = None):
        if self._resp_q is None:
            return None
        try:
            if timeout is None:
                return self._resp_q.get_nowait()
            return self._resp_q.get(timeout=timeout)
        except Exception:
            return None

    def stop(self, timeout: float = 3.0) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()
        self.send({"op": "stop"})
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
        logger.info("SubprocessWorker 停止: %s", self.name)

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()


class InferenceWorker(SubprocessWorker):
    """ACT / GGCNN 推理子进程

    模型在 **子进程内** 构建（spawn 不继承父进程对象），规避 GIL 抖动并满足
    NPU 单进程单核约束。帧数据经 SharedFrameBuffer 零拷贝传入，关节状态与
    控制命令经 Queue 传入，推理结果 (Action) 经 Queue 返回。

    命令协议 (cmd_q.put):
        {"op": "predict", "state": <(6,) ndarray>, "timestamp": float,
         "request": <可选 TaskRequest>}   → 触发一次推理
        {"op": "stop"}                     → 退出

    响应协议 (resp_q.get):
        {"event": "ready",  "kind": ..., "available": bool}
        {"event": "action", "action": Action}
        {"event": "error",  "message": str}

    Args:
        kind: "act" | "ggcnn"
        model_config: 传给模型 setup() 的 dict
        frame_name: SharedFrameBuffer 名字（相机子进程为生产者）；None 则命令需自带帧
        cores: 绑定的 CPU 核集合（如 ACT={4,5}, GGCNN={5}）
    """

    def __init__(self, kind: str, model_config: Optional[dict] = None,
                 frame_name: Optional[str] = None, cores=None,
                 name: Optional[str] = None):
        super().__init__(name or f"inference-{kind}", cores)
        self.kind = kind
        self.model_config = model_config or {}
        self.frame_name = frame_name

    # ---- 子进程内 ----
    def run_body(self, cmd_q, resp_q, stop_evt) -> None:
        from runtime.shared_frame import SharedFrameBuffer

        model = self._build_model()
        available = bool(getattr(model, "is_available", False))
        resp_q.put({"event": "ready", "kind": self.kind, "available": available})

        frame_buf = None
        if self.frame_name and available:
            try:
                frame_buf = SharedFrameBuffer.attach(self.frame_name)
            except Exception as e:  # noqa: BLE001
                resp_q.put({"event": "error",
                            "message": f"attach SharedFrameBuffer 失败: {e}"})

        from hardware.interfaces import Observation

        while not stop_evt.is_set():
            try:
                cmd = cmd_q.get(timeout=0.1)
            except Exception:
                continue
            if not cmd:
                continue
            op = cmd.get("op")
            if op == "stop":
                break
            if op == "predict":
                if not available:
                    resp_q.put({"event": "error", "message": "模型不可用"})
                    continue
                try:
                    obs = self._compose_obs(cmd, frame_buf, Observation)
                    action = model.predict(obs)
                    resp_q.put({"event": "action", "action": action})
                except Exception as e:  # noqa: BLE001
                    resp_q.put({"event": "error", "message": str(e)})

        if frame_buf is not None:
            frame_buf.close(unlink=False)

    def _build_model(self):
        """在子进程内构建模型（延迟导入，避免父进程加载 onnxruntime）"""
        if self.kind == "act":
            from policy.act_policy import ACTPolicy
            m = ACTPolicy()
            m.setup(self.model_config)
            return m
        if self.kind == "ggcnn":
            from perception.grasp_detect import GGCNNDetector
            m = GGCNNDetector()
            m.setup(self.model_config)
            return m
        raise ValueError(f"未知推理类型: {self.kind}（应为 'act' | 'ggcnn'）")

    @staticmethod
    def _compose_obs(cmd, frame_buf, Observation):
        """组合 Observation: rgb/depth 来自共享内存，state/ts 来自命令"""
        import time as _time
        rgb = depth = None
        ts = cmd.get("timestamp", _time.time())
        if frame_buf is not None:
            got = frame_buf.read_frame()
            if got is not None:
                rgb, depth, ts_shm = got
                ts = cmd.get("timestamp", ts_shm)
        if rgb is None:
            rgb = cmd.get("rgb")
            depth = cmd.get("depth", depth)
        import numpy as _np
        state = _np.asarray(cmd.get("state", _np.zeros(6)), dtype=float)
        return Observation(rgb=rgb, depth=depth, state=state, timestamp=ts)
