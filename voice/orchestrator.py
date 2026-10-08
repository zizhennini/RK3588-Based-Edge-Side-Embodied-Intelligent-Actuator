from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .asr import SherpaAsr
from .audio_io import AudioRecorder
from .camera import CameraAdapter
from .intent import IntentRouter
from .qwen_runner import QwenRunner
from .streaming_tts import StreamingTtsPlayer
from .wake import SherpaKeywordWake, SttKeywordWake
from config.cpu_affinity import bind_current_thread
from config.memory import MemoryMonitor, MemoryLimiter

# 语音线程绑定 A55 小核 2-3（refactor_plan_v9 第 4.1 节资源分配表：
# 主进程 Voice+Safety → A55 核 2-3；大核 4-7 保留给 ACT/GGCNN/VLM 推理子进程）
_VOICE_CORES = {2, 3}


class VoiceAssistant:
    def __init__(self, config: dict, motion_cb=None):
        bind_current_thread(_VOICE_CORES)
        self.config = config
        self.temp_dir = Path(config["paths"]["temp_dir"])
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.intent = IntentRouter.from_config(config)
        self.memory = MemoryMonitor()
        self.motion_cb = motion_cb

    def record_command(self, seconds: int | None = None) -> Path:
        with MemoryLimiter(self.memory, "recording"):
            seconds = seconds or int(self.config["audio"]["command_seconds"])
            out = self.temp_dir / "command.wav"
            out.unlink(missing_ok=True)
            return AudioRecorder(self.config).record_wav(out, seconds)

    def transcribe_wav(self, wav_path: str | Path) -> str:
        with MemoryLimiter(self.memory, "asr"):
            return SherpaAsr(self.config).transcribe_wav(wav_path)

    def wait_for_wake(self, mode: str = "kws", timeout: int | None = None) -> str:
        if mode == "stt":
            return SttKeywordWake(self.config, self.temp_dir).wait(timeout=timeout)
        return SherpaKeywordWake(self.config).wait(timeout=timeout)

    def detect_wake_wav(self, wav_path: str | Path) -> str:
        return SherpaKeywordWake(self.config).detect_wav(wav_path)

    def capture_photo(self) -> Path:
        return CameraAdapter(self.config).capture()

    def ask_qwen(
        self,
        text: str,
        image_path: str | Path | None = None,
        force_photo: bool = False,
        no_photo: bool = False,
        on_sentence=None,
    ) -> str:
        intent = self.intent.analyze(text)
        uses_photo = force_photo or (intent.need_photo and not no_photo)
        if uses_photo:
            print("检测到拍照意图，正在拍照...", flush=True)
            image = self.capture_photo()
            print(f"照片已保存：{image}", flush=True)
        else:
            image = Path(image_path or self.config["paths"]["placeholder_image"])
        qwen_text = self._prepare_qwen_text(intent.qwen_text, uses_photo)
        print("正在调用 Qwen demo，请等待模型回答...", flush=True)
        runner = QwenRunner(self.config)
        if on_sentence is not None:
            return runner.ask_stream(image, qwen_text, on_sentence=on_sentence)
        return runner.ask(image, qwen_text)

    def _prepare_qwen_text(self, qwen_text: str, uses_photo: bool) -> str:
        marker = str(self.config["qwen"].get("demo_image_marker", "<image>"))
        image_prefix = str(self.config["intent"]["image_prefix"])
        should_attach_image = uses_photo or qwen_text.startswith(image_prefix)
        if should_attach_image and marker not in qwen_text:
            return f"{marker}{qwen_text}"
        return qwen_text

    def run_once_from_text(
        self,
        text: str,
        *,
        image_path: str | Path | None = None,
        force_photo: bool = False,
        no_photo: bool = False,
        speak: bool = True,
        play: bool = True,
    ) -> str:
        # 匹配动作库指令
        if self.motion_cb:
            from vla.command_queue import MotionMatcher
            matcher = MotionMatcher()
            action_name, info = matcher.match(text)
            # 序列连接词检测:含"然后/再/接着/之后/最后"的指令是多步任务,
            # 不能被单步关键词匹配截胡(否则"先打招呼,然后抬起"只执行"抬起")
            _SEQ_WORDS = ("然后", "再", "接着", "之后", "最后")
            multi_step = any(w in text for w in _SEQ_WORDS)
            if action_name and info.get("file") and not multi_step:
                print(f"[动作] 匹配到: {action_name}")
                if speak and play:
                    tts = self.config["models"]["tts"]
                    from .streaming_tts import StreamingTtsPlayer as STP
                    p = STP(self.config)
                    p.enqueue(f"执行动作{action_name}")
                    p.close()
                self.motion_cb(action_name, info)
                return f"执行动作: {action_name}"
            # 多步编排(vlm_arm 函数清单模式):单步未命中时,把指令拆解为
            # 动作库动作序列;解析失败/无动作则落回下方问答路径。
            # 执行用同步 subprocess(replay_traj 完成才走下一步)——
            # 不能用 motion_cb 的后台 Popen:多个回放并发会争抢同一条舵机总线
            from .agent_plan import plan_from_text
            actions, response = plan_from_text(self.config, text)
            if actions:
                import subprocess
                from pathlib import Path as _P
                _root = _P(__file__).resolve().parent.parent
                actions_index = matcher._index
                if speak and play:
                    from .streaming_tts import StreamingTtsPlayer as STP
                    p = STP(self.config)
                    if response:
                        p.enqueue(response)
                    for i, act in enumerate(actions, 1):
                        p.enqueue(f"第{i}步,{act}")
                    p.close()
                for i, act in enumerate(actions, 1):
                    traj_rel = actions_index.get(act, {}).get("file", "")
                    print(f"[编排] 执行第 {i}/{len(actions)} 步: {act}", flush=True)
                    if not traj_rel:
                        print(f"[编排] 动作 {act} 暂无轨迹文件,跳过", flush=True)
                        continue
                    traj = _root / "motion_library" / traj_rel
                    if not traj.is_file():
                        print(f"[编排] 轨迹不存在: {traj}", flush=True)
                        continue
                    try:
                        subprocess.run(
                            [sys.executable, str(_root / "scripts" / "replay_traj.py"),
                             str(traj), "--port", "/dev/ttyACM0", "--fps", "30",
                             "--initial"],
                            cwd=str(_root), check=True,
                        )
                    except subprocess.CalledProcessError as exc:
                        print(f"[编排] 动作 {act} 回放失败(ret={exc.returncode}),终止序列",
                              flush=True)
                        break
                return f"执行动作序列: {' → '.join(actions)}"
        from .streaming_tts import StreamingTtsPlayer
        if speak and play:
            print("将使用流式 TTS：Qwen 每生成一句就直接写入喇叭 PCM 播放。", flush=True)
            player = StreamingTtsPlayer(self.config)
            try:
                ack_text = str(self.config["models"]["tts"].get("ack_text", "")).strip()
                if ack_text:
                    player.enqueue(ack_text)
                return self.ask_qwen(
                    text,
                    image_path=image_path,
                    force_photo=force_photo,
                    no_photo=no_photo,
                    on_sentence=player.enqueue,
                )
            finally:
                player.close()
        return self.ask_qwen(text, image_path=image_path, force_photo=force_photo,
                             no_photo=no_photo)

    def run_once_from_microphone(self, seconds: int | None = None, *,
                                 no_photo: bool = False,
                                 speak: bool = True, play: bool = True) -> str:
        command_wav = self.record_command(seconds)
        try:
            text = self.transcribe_wav(command_wav)
        finally:
            command_wav.unlink(missing_ok=True)
        if not text:
            raise RuntimeError("STT produced empty text")
        print(f"识别文本：{text}", flush=True)
        return self.run_once_from_text(text, no_photo=no_photo, speak=speak, play=play)

    def listen_once(
        self,
        *,
        wake_mode: str = "kws",
        wake_timeout: int | None = None,
        seconds: int | None = None,
        no_photo: bool = False,
        speak: bool = True,
        play: bool = True,
    ) -> str:
        keyword = self.wait_for_wake(mode=wake_mode, timeout=wake_timeout)
        print(f"wake={keyword}", flush=True)
        return self.run_once_from_microphone(seconds=seconds, no_photo=no_photo,
                                             speak=speak, play=play)

    def listen_forever(
        self,
        *,
        wake_mode: str = "kws",
        wake_timeout: int | None = None,
        seconds: int | None = None,
        no_photo: bool = False,
        speak: bool = True,
        play: bool = True,
    ) -> None:
        round_idx = 0
        while True:
            round_idx += 1
            try:
                print(f"等待唤醒词，第 {round_idx} 轮。", flush=True)
                answer = self.listen_once(
                    wake_mode=wake_mode,
                    wake_timeout=wake_timeout,
                    seconds=seconds,
                    no_photo=no_photo,
                    speak=speak,
                    play=play,
                )
                print(answer, flush=True)
                print("本轮结束，继续等待唤醒词。", flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"本轮失败，继续等待唤醒词：{exc}", file=sys.stderr, flush=True)
                self.cleanup_temp()

    def cleanup_temp(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
