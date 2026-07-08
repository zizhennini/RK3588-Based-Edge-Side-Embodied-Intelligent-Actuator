"""指令队列 — 串行有序执行 + 中断机制 + 分级内存管控 + VLM 抓取"""
import json
import re
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
            if not isinstance(info, dict):
                continue
            for kw in info.get("keywords", []):
                if kw.lower() in text_lower:
                    return action_name, info
        return None, None

    def list_actions(self) -> list[str]:
        return list(self._index.keys())


# ── 抓取意图解析 ──

# 颜色关键词表（与 vla/vlm/qwen3_vl.py 保持一致）
GRASP_COLORS = [
    "红色", "粉色", "橙色", "黄色", "绿色", "蓝色", "紫色",
    "黑色", "白色", "灰色", "棕色", "金色", "银色", "透明",
]

# 抓取动词
GRASP_VERBS = ["抓住", "拿起", "抓取", "抓", "拿", "捡起", "拾取", "夹取"]

# 抓取指令正则：动词 + (填充词) + (颜色) + (的) + (目标物体)
_GRASP_RE = re.compile(
    r'(?:' + '|'.join(GRASP_VERBS) + r')'
    r'\s*(?:那个|这个|一下|一个|那边|那个的|过来|一下下)?'
    r'\s*(?P<color>' + '|'.join(GRASP_COLORS) + r')?'
    r'\s*(?:的|的个)?'
    r'\s*(?P<target>\S+)?'
)


def _extract_color_from_text(text: str) -> str:
    """从文本中提取颜色关键词"""
    for c in GRASP_COLORS:
        if c in text:
            return c
    en_map = {"red": "红色", "pink": "粉色", "orange": "橙色",
              "yellow": "黄色", "green": "绿色", "blue": "蓝色",
              "purple": "紫色", "black": "黑色", "white": "白色",
              "gray": "灰色", "brown": "棕色", "gold": "金色", "silver": "银色"}
    for en, zh in en_map.items():
        if en in text.lower():
            return zh
    return ""


def _parse_grasp_intent(text: str) -> dict | None:
    """解析抓取意图，返回 {"color": str, "target": str} 或 None"""
    m = _GRASP_RE.search(text)
    if m:
        color = m.group("color") or ""
        target = (m.group("target") or "").strip().rstrip("，。,. ")
        return {"color": color, "target": target}
    return None


class CommandQueue:
    """串行指令队列 — FIFO 执行 + 紧急中断 + VLM 抓取"""

    def __init__(self, arm=None, smart_vlm=None, camera=None, snapshot_cb=None,
                 memory_monitor=None, motion_dir="./motion_library",
                 camera_matrix=None):
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
        self.motion_matcher = MotionMatcher(str(Path(motion_dir) / "index.json"))
        self.motion_dir = Path(motion_dir)
        self.camera_matrix = camera_matrix

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

        # 暂停/继续
        if text_stripped in ("暂停", "停一下"):
            self._paused = True
            self.push(Task("tts", {"text": "已暂停"}, source))
            return
        if text_stripped in ("继续", "继续回放", "resume"):
            self._paused = False
            self.push(Task("tts", {"text": "继续执行"}, source))
            return

        # 指令：列出动作库
        if text_stripped in ("有什么动作", "动作列表", "动作库", "list", "清单"):
            index_path = self.motion_dir / "index.json"
            if index_path.exists():
                import json as _json
                with open(index_path) as _f:
                    idx = _json.load(_f)
                names = ", ".join(idx.keys())
                self.push(Task("tts", {"text": f"动作库中有: {names}"}, source))
            else:
                self.push(Task("tts", {"text": "动作库为空"}, source))
            return

        # 指令：归零
        if text_stripped in ("归零", "复位", "回零", "home"):
            self.push(Task("motion", {"action": "home"}, source))
            self.push(Task("tts", {"text": "已归零"}, source))
            return

        # 抓取意图解析（放在动作库匹配之前，抓取优先级更高）
        grasp = _parse_grasp_intent(text)
        if grasp is not None:
            self.push(Task("vlm_grasp", grasp, source))
            return

        # 循环回放
        loop_match = re.search(r"循环(?:回放|播放)\s*(.+)|(.+)\s*(?:循环|重复)", text_stripped)
        if loop_match:
            name = loop_match.group(1) or loop_match.group(2)
            action_name, info = self.motion_matcher.match(name or text)
            if action_name and info.get("file"):
                self.push(Task("motion", {
                    "action": action_name,
                    "file": str(self.motion_dir / info["file"]),
                    "loop": 3, "source": source,
                }, source))
                self.push(Task("tts", {"text": f"循环回放{action_name}, 3次"}, source))
                return

        # 匹配动作库
        action_name, info = self.motion_matcher.match(text)
        if action_name and info.get("file"):
            self.push(Task("motion", {
                "action": action_name,
                "file": str(self.motion_dir / info["file"]),
                "loop": 1, "source": source,
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
        print(f"[TTS] {cleaned}", flush=True)
        try:
            if self.__class__._tts_player is None:
                from voice_assistant.voice_assistant.streaming_tts import StreamingTtsPlayer
                from voice_assistant.voice_assistant.config import load_config
                self.__class__._tts_player = StreamingTtsPlayer(load_config())
            self.__class__._tts_player.enqueue(cleaned)
        except Exception as e:
            print(f"[TTS] 播报失败: {e}")

    # ── 拍照 ──

    def _take_snapshot(self) -> str:
        """拍照保存到临时文件，返回路径或空字符串"""
        if self.snapshot_cb:
            return self.snapshot_cb()
        if self.camera and hasattr(self.camera, "read"):
            try:
                import cv2
                rgb, _ = self.camera.read()
                path = "/tmp/vla_grasp.jpg"
                cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                return path
            except Exception as e:
                print(f"[相机] 拍照失败: {e}")
        return ""
    
    def _take_rgb_depth(self):
        """获取 RGB 和深度图，返回 (rgb, depth)"""
        if self.camera and hasattr(self.camera, "read"):
            try:
                return self.camera.read()
            except Exception as e:
                print(f"[相机] 读取失败: {e}")
        return None, None

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
            except Exception:
                import traceback
                traceback.print_exc()
                print(f"[CommandQueue] 任务 {task.id} 失败")

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
                action = task.data.get("action", "unknown")
                file_path = task.data.get("file", "")
                loop = task.data.get("loop", 1)
                if action == "home":
                    print("[Motion] 归零")
                    if self.arm:
                        self.arm.home()
                    return
                if not file_path or not Path(file_path).exists():
                    print(f"[Motion] 轨迹文件不存在: {file_path}")
                    return
                print(f"[Motion] 回放: {action} ({loop}次)")
                if self.arm:
                    self._replay_trajectory(file_path, loop)
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

    def _replay_trajectory(self, file_path: str, loop: int = 1):
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

        for lap in range(loop):
            if self._interrupted:
                break
            for i, frame in enumerate(frames):
                while self._paused and not self._interrupted:
                    time.sleep(0.1)
                if self._interrupted:
                    break
                joints = frame if isinstance(frame, list) else frame.get("joints", [])
                if joints and self.arm and hasattr(self.arm, "write_angles"):
                    self.arm.write_angles(np.array(joints))
                if (i + 1) % max(1, len(frames) // 5) == 0:
                    pct = (i + 1) / len(frames) * 100
                    info = f" 第{lap+1}/{loop}轮 {pct:.0f}%"
                    print(f"[Motion] 回放中{info}", flush=True)
                time.sleep(dt)
            if loop > 1 and lap < loop - 1 and not self._interrupted:
                print(f"[Motion] 第{lap+1}轮完成，继续下一轮")

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
        """完整 VLM 抓取链路：拍照 → VLM 识别 → 视觉定位 → 深度 → IK → 机械臂 → TTS"""
        target = data.get("target", "")
        color = data.get("color", "")

        # ── 1. 自动归零（确保起始位置确定） ──
        print(f"[Grasp] 归零中...", flush=True)
        if self.arm and hasattr(self.arm, "home"):
            try:
                self.arm.home(steps=30, delay_s=0.03)
            except Exception as e:
                print(f"[Grasp] 归零失败: {e}")
        time.sleep(0.5)

        # ── 2. TTS: 告知开始 ──
        desc = f"{color}{target}" if color else (target or "目标物体")
        self._speak(f"{desc}，让我看看")

        # ── 3. 拍照 ──
        image_path = self._take_snapshot()
        if not image_path:
            self._speak("拍照失败")
            return

        # ── 4. 构建 VLM 提示词 ──
        prompt_parts = ["<image>"]
        if color and target:
            prompt_parts.append(f"请找到画面中的{color}{target}，输出它的颜色和名称")
        elif color:
            prompt_parts.append(f"请找到画面中{color}的物体，输出它的颜色和名称")
        elif target:
            prompt_parts.append(f"请找到画面中的{target}，输出它的颜色和名称")
        else:
            prompt_parts.append("请找到画面中最明显的可抓取物体，输出它的颜色和名称")
        prompt = "".join(prompt_parts)

        # ── 5. VLM 推理 ──
        print(f"[Grasp] VLM 推理中...", flush=True)
        if hasattr(self.smart_vlm, "ensure_loaded"):
            self.smart_vlm.ensure_loaded()
        result = self.smart_vlm.infer(image_path, prompt=prompt)
        print(f"[Grasp] VLM: color={repr(result.color)} object={repr(result.object)} raw={repr(result.raw[:80])}", flush=True)

        # ── 6. 确定抓取颜色和目标名称 ──
        # VLM 输出校验：如果输出过于简短或格式异常，完全回退到语音指令
        vlm_valid = (result.color or result.object) and len(result.raw) > 5
        if vlm_valid and result.object and result.object not in ("[", "(", "]", "）", "）"):
            grasp_color = result.color or color
            grasp_object = result.object or target
            print(f"[Grasp] 使用 VLM 结果: {grasp_color}{grasp_object}", flush=True)
        else:
            # VLM 无效，回退到语音指令中的目标
            grasp_color = color
            grasp_object = target
            print(f"[Grasp] VLM 输出无效，回退到语音指令: color={repr(color)} target={repr(target)}", flush=True)

            if not grasp_color and not grasp_object:
                # 连语音指令也没有明确目标，用默认
                grasp_color = "红色"
                grasp_object = "物体"
                print(f"[Grasp] 语音指令也无明确目标，使用默认: 红色物体", flush=True)

        self._speak(f"找到{grasp_color}{grasp_object}")

        # ── 5. 读取 RGB + 深度图 ──
        rgb, depth = self._take_rgb_depth()
        if rgb is None:
            import cv2
            rgb = cv2.imread(image_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = None

        # ── 6. 视觉定位 + PCA 抓取位姿 ──
        from vla.vision import ColorLocator, PCAGrasper
        K = self.camera_matrix
        if K is None:
            from config.settings import CAMERA_MATRIX
            K = CAMERA_MATRIX
        locator = ColorLocator(K)
        pca_grasper = PCAGrasper(K)

        # PCA 计算抓取位姿
        # 策略 A：颜色掩码
        mask = locator.get_mask(rgb, grasp_color)
        if int(np.sum(mask > 0)) > 200:
            grasp_pose = pca_grasper.compute_from_mask(mask, depth)
            if grasp_pose:
                print(f"[Grasp] 颜色掩码 PCA: {grasp_color} {int(np.sum(mask>0))}px")
        else:
            grasp_pose = None

        # 策略 B：深度聚类兜底（找桌面上的凸起物体）
        if grasp_pose is None:
            print(f"[Grasp] 颜色分割失败，尝试深度聚类...")
            # 以图像中心为原点，取中心区域做深度PCA
            h, w = depth.shape
            cy, cx = h // 2, w // 2
            crop = depth[cy-h//4:cy+h//4, cx-w//4:cx+w//4]
            crop_mask = (crop > 0.1) & (crop < 0.8)
            if np.any(crop_mask):
                full_mask = np.zeros_like(depth, dtype=np.uint8)
                full_mask[cy-h//4:cy+h//4, cx-w//4:cx+w//4] = crop_mask.astype(np.uint8) * 255
                grasp_pose = pca_grasper.compute_from_mask(full_mask, depth)
                if grasp_pose:
                    print(f"[Grasp] 深度聚类 PCA: {grasp_pose['point_count']}pts")

        if grasp_pose is None:
            # 质心回退
            pos = locator.locate(rgb, depth, grasp_color)
            if pos is None:
                self._speak("无法定位目标位置")
                return
            cam_x, cam_y, cam_z = pos["x"], pos["y"], pos["z"]
            grasp_angle = 0.0
            grasp_width = 0.08
            print(f"[Grasp] 回退到质心: x={cam_x:.3f} y={cam_y:.3f} z={cam_z:.3f}")
        else:
            cc = grasp_pose["center_cam"]
            cam_x, cam_y, cam_z = cc[0], cc[1], cc[2]
            grasp_angle = grasp_pose["angle"]
            grasp_width = grasp_pose["width"]
            print(f"[Grasp] PCA 位姿: center=({cam_x:.3f},{cam_y:.3f},{cam_z:.3f}) "
                  f"angle={np.rad2deg(grasp_angle):.1f}deg width={grasp_width:.3f}m "
                  f"points={grasp_pose['point_count']}")

        # ── 8. PCA 结果校验 ──
        # 如果 PCA 中心非常接近图像中心（cx,cy），可能没有找到真实物体，仅靠默认颜色分割
        cam_cx, cam_cy = K[0, 2], K[1, 2]
        pixel_u = int((cam_x * K[0, 0] / max(cam_z, 0.01)) + cam_cx)
        pixel_v = int((cam_y * K[1, 1] / max(cam_z, 0.01)) + cam_cy)
        dist_from_center = np.hypot(pixel_u - cam_cx, pixel_v - cam_cy)
        if dist_from_center < 20 and grasp_color == "红色":
            print(f"[Grasp] ⚠ PCA 中心距图像中心仅 {dist_from_center:.0f}px，可能未找到真实物体", flush=True)

        # ── 9. 打印机器人坐标系下的目标位姿 ──
        robot_xyz = self.arm.camera_to_robot(cam_x, cam_y, cam_z)
        clamped = self.arm.clamp_workspace(robot_xyz.copy())
        if np.any(robot_xyz != clamped):
            diff = np.abs(robot_xyz - clamped)
            print(f"[Grasp] ⚠ 坐标被工作空间钳制: {diff}", flush=True)
        robot_xyz = clamped
        print(f"[Grasp] 机器人坐标: x={robot_xyz[0]:.3f} y={robot_xyz[1]:.3f} z={robot_xyz[2]:.3f}")
        print(f"[Grasp] 夹爪角度: {np.rad2deg(grasp_angle):.1f}deg  宽度: {grasp_width:.3f}m")

        # 提示工作空间是否严重钳制（x 被钳制说明目标偏左，y 被钳制说明目标偏远）
        if abs(robot_xyz[0] - 0.05) < 0.001:
            print(f"[Grasp] ⚠ 目标被钳制到 x_min，可能偏左", flush=True)

        self._speak("准备抓取")

        # ── 10. IK + 机械臂执行（带角度 + 自适应夹爪宽度） ──
        try:
            # 先张开夹爪
            print(f"[Grasp] 张开夹爪", flush=True)
            self.arm.gripper(True)
            time.sleep(0.3)

            # 接近目标（在目标上方 5cm）
            approach_cam_z = max(cam_z - 0.05, 0.05)
            approach_robot = self.arm.camera_to_robot(cam_x, cam_y, approach_cam_z)
            approach_robot = self.arm.clamp_workspace(approach_robot)
            print(f"[Grasp] ① 接近: ({approach_robot[0]:.3f}, {approach_robot[1]:.3f}, {approach_robot[2]:.3f}) "
                  f"wrist={np.rad2deg(grasp_angle):.1f}deg", flush=True)
            self.arm.move_to_camera_with_angle(cam_x, cam_y, approach_cam_z, grasp_angle)
            time.sleep(0.5)

            if self._interrupted:
                return

            # 下降到目标
            print(f"[Grasp] ② 抓取: ({robot_xyz[0]:.3f}, {robot_xyz[1]:.3f}, {robot_xyz[2]:.3f})", flush=True)
            self.arm.move_to_camera_with_angle(cam_x, cam_y, cam_z, grasp_angle)
            time.sleep(0.5)

            if self._interrupted:
                return

            # 自适应夹爪闭合
            close_pulse = int(1781 + (2600 - 1781) * min(grasp_width / 0.10, 1.0))
            print(f"[Grasp] ③ 夹爪闭合: width={grasp_width:.3f}m pulse={close_pulse}", flush=True)
            self.arm.gripper_width(grasp_width)
            time.sleep(0.8)

            # 抬升（深度减小 = 向上抬）
            lift_cam_z = max(cam_z - 0.12, 0.05)
            lift_robot = self.arm.camera_to_robot(cam_x, cam_y, lift_cam_z)
            lift_robot = self.arm.clamp_workspace(lift_robot)
            print(f"[Grasp] ④ 抬升: ({lift_robot[0]:.3f}, {lift_robot[1]:.3f}, {lift_robot[2]:.3f})", flush=True)
            self.arm.move_to_camera_with_angle(cam_x, cam_y, lift_cam_z, grasp_angle)
            time.sleep(0.5)

            # ── 9. TTS: 抓取成功 ──
            print(f"[Grasp] ✅ 抓取完成")
            self._speak("抓取完成")

        except Exception as e:
            print(f"[Grasp] 抓取执行失败: {e}")
            self._speak("抓取失败")
            return

    # ── 动作库快照（供外部调用） ──

    def update_motion_index(self):
        self.motion_matcher.reload()


def create_voice_motion_callback(command_queue: CommandQueue):
    """创建语音识别回调，将识别结果自动推入队列"""
    def on_voice_text(text: str):
        if text and text.strip():
            command_queue.push_text(text.strip(), source="voice")
    return on_voice_text
