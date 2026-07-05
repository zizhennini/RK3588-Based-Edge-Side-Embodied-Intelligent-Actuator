"""指令队列 — 串行有序执行 + 中断机制 + 分级内存管控"""
import json
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from config.memory import MemoryMonitor


@dataclass
class Task:
    type: str  # motion | vlm_grasp | vlm_ask | tts | emergency_stop | idle
    data: dict = field(default_factory=dict)
    source: str = "voice"
    id: int = 0


class MotionMatcher:
    """语音文本 → 动作库匹配"""

    def __init__(self, index_path: str = "./motion_library/index.json"):
        self._index_path = Path(index_path)
        self._index = self._reload()

    def _reload(self) -> dict:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {}

    def reload(self):
        self._index = self._reload()

    def match(self, text: str) -> tuple[str | None, dict | None]:
        text_lower = text.lower()
        for action_name, info in self._index.items():
            for kw in info.get("keywords", []):
                if kw.lower() in text_lower:
                    return action_name, info
        return None, None

    def list_actions(self) -> list[str]:
        return list(self._index.keys())


class CommandQueue:
    """串行指令队列 — FIFO 执行 + 紧急中断"""

    def __init__(self, arm=None, smart_vlm=None, camera=None, snapshot_cb=None,
                 memory_monitor=None, motion_dir="./motion_library"):
        self._queue: deque[Task] = deque()
        self._lock = threading.Lock()
        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._current_task: Task | None = None
        self._task_counter = 0
        self._interrupted = False

        self.arm = arm
        self.smart_vlm = smart_vlm
        self.camera = camera
        self.snapshot_cb = snapshot_cb
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.motion_matcher = MotionMatcher(Path(motion_dir) / "index.json")
        self.motion_dir = Path(motion_dir)

        self._on_task_start = None
        self._on_task_done = None
        self._on_interrupt = None

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def is_busy(self) -> bool:
        return self._current_task is not None or self.pending_count > 0

    @property
    def current_task_type(self) -> str | None:
        return self._current_task.type if self._current_task else None

    # ── 回调注册 ──

    def on_task_start(self, cb):
        self._on_task_start = cb

    def on_task_done(self, cb):
        self._on_task_done = cb

    def on_interrupt(self, cb):
        self._on_interrupt = cb

    # ── 队列操作 ──

    def push(self, task: Task):
        with self._lock:
            self._task_counter += 1
            task.id = self._task_counter
            self._queue.append(task)

    def push_text(self, text: str, source: str = "voice"):
        """解析文本并推入对应任务"""
        text_stripped = text.strip().lower()

        # 中断指令
        if text_stripped in ("停止", "停", "stop", "紧急停止", "刹车"):
            self.interrupt()
            self.push(Task("tts", {"text": "已停止"}, source))
            return

        # 匹配动作库
        action_name, info = self.motion_matcher.match(text)
        if action_name and info.get("file"):
            self.push(Task("motion", {
                "action": action_name,
                "file": str(self.motion_dir / info["file"]),
            }, source))
            return

        # 默认：VLM 问答
        self.push(Task("vlm_ask", {"text": text}, source))

    def interrupt(self):
        """中断当前任务 + 清空队列 + 紧急停机"""
        self._interrupted = True
        with self._lock:
            self._queue.clear()
        if self.arm and hasattr(self.arm, "emergency_stop"):
            self.arm.emergency_stop()
        if self._on_interrupt:
            self._on_interrupt()

    # ── 启动/停止 ──

    def start(self):
        if self._running:
            return
        self._running = True
        self._interrupted = False
        self._thread = threading.Thread(target=self._execute_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.interrupt()

    # ── TTS 播报 ──

    _tts_player = None

    def _speak(self, text: str):
        if not text or not text.strip():
            return
        cleaned = text.strip().replace("**", "").replace("\n", "，")
        try:
            if self.__class__._tts_player is None:
                from voice_assistant.voice_assistant.streaming_tts import StreamingTtsPlayer
                from voice_assistant.voice_assistant.config import load_config
                self.__class__._tts_player = StreamingTtsPlayer(load_config())
            self.__class__._tts_player.enqueue(cleaned)
        except Exception as e:
            print(f"[TTS] 播报失败: {e}")

    # ── 执行循环 ──

    def _execute_loop(self):
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            task = None
            with self._lock:
                if self._queue:
                    task = self._queue.popleft()

            if task is None:
                time.sleep(0.05)
                continue

            self._current_task = task
            self._interrupted = False

            if self._on_task_start:
                self._on_task_start(task)

            try:
                self._execute_task(task)
            except Exception as e:
                print(f"[CommandQueue] 任务 {task.id} 失败: {e}")

            self._current_task = None

            if self._on_task_done:
                self._on_task_done(task)

    def _mem_acquire(self, comp: str) -> bool:
        if self.memory_monitor and hasattr(self.memory_monitor, "acquire"):
            ok = self.memory_monitor.acquire(comp)
            if not ok:
                print(f"[Memory] {comp} 内存预算不足")
            return ok
        return True

    def _mem_release(self, comp: str):
        if self.memory_monitor and hasattr(self.memory_monitor, "release"):
            self.memory_monitor.release(comp)

    def _execute_task(self, task: Task):
        if task.type == "emergency_stop":
            self.interrupt()
            return

        if task.type == "tts":
            self._mem_acquire("tts")
            try:
                text = task.data.get("text", "")
                self._speak(text)
            finally:
                self._mem_release("tts")
            return

        if task.type == "motion":
            self._mem_acquire("recording")
            try:
                file_path = task.data.get("file", "")
                action = task.data.get("action", "unknown")
                if not file_path or not Path(file_path).exists():
                    print(f"[Motion] 轨迹文件不存在: {file_path}")
                    return
                print(f"[Motion] 回放: {action}")
                if self.arm:
                    self._replay_trajectory(file_path)
            finally:
                self._mem_release("recording")
            return

        if task.type == "vlm_ask":
            self._mem_acquire("vlm")
            try:
                text = task.data.get("text", "")
                print(f"[VLM] 问答: {text}")
                if self.smart_vlm:
                    self._vlm_ask(text)
            finally:
                self._mem_release("vlm")
            return

        if task.type == "vlm_grasp":
            self._mem_acquire("vlm")
            try:
                print(f"[VLM] 抓取: {task.data}")
                if self.smart_vlm and self.arm:
                    self._vlm_grasp(task.data)
            finally:
                self._mem_release("vlm")
            return

    # ── 具体执行器 ──

    def _replay_trajectory(self, file_path: str):
        import json
        import numpy as np
        try:
            with open(file_path) as f:
                traj = json.load(f)
        except Exception as e:
            print(f"加载轨迹失败: {e}")
            return

        frames = traj if isinstance(traj, list) else traj.get("frames", [])
        if not frames:
            print("空轨迹")
            return

        fps = traj.get("fps", 30) if not isinstance(traj, list) else 30
        dt = 1.0 / fps

        for i, frame in enumerate(frames):
            if self._interrupted:
                print("[Motion] 被中断")
                break
            joints = frame if isinstance(frame, list) else frame.get("joints", [])
            if joints and self.arm and hasattr(self.arm, "write_angles"):
                self.arm.write_angles(np.array(joints))
            time.sleep(dt)

    def _vlm_ask(self, text: str):
        if not self.smart_vlm:
            return
        import cv2
        if hasattr(self.smart_vlm, "ensure_loaded"):
            self.smart_vlm.ensure_loaded()
        tmp = "/tmp/vla_cmd.jpg"
        got_image = False
        if self.snapshot_cb:
            path = self.snapshot_cb()
            if path:
                tmp = path
                got_image = True
        elif self.camera and hasattr(self.camera, "read"):
            rgb, _ = self.camera.read()
            cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            got_image = True
        if got_image:
            print("[VLM] 分析画面中...", flush=True)
        if got_image and "<image>" not in text:
            text = f"<image>{text}"
        result = self.smart_vlm.infer(tmp, prompt=text)
        print(f"[VLM] {result.raw[:120]}", flush=True)
        if got_image and result.raw:
            self._speak(result.raw)

    def _vlm_grasp(self, data: dict):
        target = data.get("target", "物体")
        color = data.get("color", None)
        print(f"[Grasp] 目标: {target} 颜色: {color}")


def create_voice_motion_callback(command_queue: CommandQueue):
    """创建语音识别回调，将识别结果自动推入队列"""
    def on_voice_text(text: str):
        if text and text.strip():
            command_queue.push_text(text.strip(), source="voice")
    return on_voice_text
