# vla/command_queue.py — 向后兼容桩 (Deprecated)
#
# 警告：此文件仅为旧版脚本（voice_vla.py、orchestrator.py）提供向后兼容。
# 新代码请使用 policy.grasp_pipeline、perception.vlm 等模块。
#
# 重构方案 v9 第 1.4 节/第 10 节决策：
#   - 原 command_queue.py 调用不存在方法（致命 Bug），不修复
#   - 替代实现：policy/grasp_pipeline.py + policy/act_policy.py
#   - 此桩文件为旧脚本提供接口兼容，委托到新架构
"""指令队列 — 向后兼容包装（Deprecated）

为旧版语音/文本入口提供 MotionMatcher 和 CommandQueue 接口。

v9 重构说明:
  - MotionMatcher: 匹配文本到动作库条目，委托索引文件 (motion_library/index.json)
  - CommandQueue: 任务队列包装，将 VLM/动作指令委托到新架构
  - create_voice_motion_callback: 语音/文本输入的标准化回调入口
"""
import json
import os
import time
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── 动作库路径 ────────────────────────────────────────────────────────────────
_PROJECT_DIR = Path(__file__).resolve().parent.parent
_MOTION_DIR = _PROJECT_DIR / "motion_library"
_INDEX_FILE = _MOTION_DIR / "index.json"


# ═══════════════════════════════════════════════════════════════════════════════
# MotionMatcher — 文本匹配动作库
# ═══════════════════════════════════════════════════════════════════════════════
class MotionMatcher:
    """文本 → 动作库条目匹配

    加载 motion_library/index.json，将输入文本与各动作的 keywords 列表匹配。
    返回 (action_name, info) 或 (None, None)。

    Usage::
        matcher = MotionMatcher()
        name, info = matcher.match("抓取红色杯子")
        if name:
            print(f"匹配动作: {name}, 文件: {info['file']}")
    """

    def __init__(self, index_path: str = ""):
        self._index_path = Path(index_path or _INDEX_FILE)
        self._index: dict = {}
        self._reload()

    def _reload(self) -> None:
        """重新加载动作索引"""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 过滤无效条目
                self._index = {
                    k: v for k, v in data.items()
                    if isinstance(v, dict) and isinstance(v.get("keywords"), list)
                }
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("动作索引加载失败: %s", e)
                self._index = {}
        else:
            logger.info("动作索引文件不存在: %s", self._index_path)
            self._index = {}

    def match(self, text: str) -> tuple[Optional[str], Optional[dict]]:
        """匹配文本到动作

        匹配策略:
          1. 检查 text 是否包含任何动作的 keywords
          2. 如果多个动作匹配，选关键词匹配最多的
          3. 检查匹配的动作是否有关联文件 (info.get("file"))

        Args:
            text: 输入文本（如语音识别结果）

        Returns:
            (action_name, info) 或 (None, None)
        """
        text_lower = text.lower().strip()
        best_name = None
        best_info = None
        best_score = 0

        for name, info in self._index.items():
            keywords = [kw.lower() for kw in info.get("keywords", [])]
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_name = name
                best_info = info

        if best_name and best_score > 0:
            logger.debug("MotionMatcher: '%s' -> '%s' (score=%d)",
                         text, best_name, best_score)
            return best_name, best_info

        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# CommandQueue — 指令队列（向后兼容包装）
# ═══════════════════════════════════════════════════════════════════════════════
class CommandQueue:
    """指令队列 — 兼容旧版 voice_vla.py 接口

    在 v9 架构中，指令由 System 类和 GraspPipeline 管理。
    此包装器提供旧版接口，内部委托到新架构模块。

    Args:
        smart_vlm: SmartVLM 实例（兼容旧版）
        snapshot_cb: 快照回调函数，返回图片路径
    """

    def __init__(self, smart_vlm=None, snapshot_cb=None):
        self._smart_vlm = smart_vlm
        self._snapshot_cb = snapshot_cb
        self._running = False
        self._matcher = MotionMatcher()

    def start(self):
        """启动队列"""
        self._running = True
        logger.info("CommandQueue 已启动（向后兼容模式）")

    def stop(self):
        """停止队列"""
        self._running = False
        logger.info("CommandQueue 已停止")

    def process_text(self, text: str) -> str:
        """处理文本指令

        Args:
            text: 输入文本

        Returns:
            执行结果描述
        """
        text = text.strip()
        if not text:
            return ""

        # 1. 尝试匹配动作库
        name, info = self._matcher.match(text)
        if name and info and info.get("file"):
            traj_file = str(_MOTION_DIR / info["file"])
            if os.path.isfile(traj_file):
                logger.info("CommandQueue: 执行动作 '%s' -> %s", name, traj_file)
                self._run_replay(traj_file)
                return f"执行动作: {name}"

        # 2. 否则交给 VLM（如果可用）
        if self._smart_vlm is not None and self._snapshot_cb is not None:
            try:
                image_path = self._snapshot_cb()
                if image_path:
                    logger.info("CommandQueue: VLM 推理: %s", text)
                    result = self._smart_vlm.infer(image_path)
                    return f"VLM: {result.raw[:100]}"
            except Exception as e:
                logger.error("VLM 推理失败: %s", e)

        return f"文本: {text}"

    @staticmethod
    def _run_replay(traj_file: str) -> None:
        """后台启动轨迹回放"""
        cmd = [
            sys.executable,
            str(_PROJECT_DIR / "scripts" / "replay_traj.py"),
            traj_file,
            "--port", "/dev/ttyACM0",
            "--fps", "30",
            "--initial",
        ]
        try:
            subprocess.Popen(
                cmd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error("动作回放启动失败: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# create_voice_motion_callback — 语音/文本回调工厂
# ═══════════════════════════════════════════════════════════════════════════════
def create_voice_motion_callback(queue: CommandQueue):
    """创建语音/文本处理回调

    供 voice_vla.py 等旧版脚本使用。返回的函数接收文本并委托给 CommandQueue。

    Args:
        queue: CommandQueue 实例

    Returns:
        Callable[[str], None] — 接受文本的处理函数
    """
    def on_text(text: str):
        """处理文本指令"""
        try:
            result = queue.process_text(text)
            if result:
                print(f"[结果] {result}", flush=True)
        except Exception as e:
            print(f"[错误] {e}", flush=True)

    return on_text