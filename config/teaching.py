"""教学模式与权限管理 — 三种模式 + 角色权限 + 一键还原"""

from __future__ import annotations
import json
import shutil
from pathlib import Path

MOTION_DIR = Path(__file__).resolve().parent.parent / "motion_library"
TASK_DIR = Path(__file__).resolve().parent.parent / "task_library"
# 配置默认值（学生不可修改的安全参数在 teacher 配置中）
TEACHER_CONFIG = {
    "depth_warn_dist": 0.25,
    "depth_stop_dist": 0.12,
    "current_warn": 600,
    "current_stop": 900,
}

MODES = {
    "manual": {
        "label": "手动示教模式",
        "desc": "遥操作+轨迹录制+回放，禁用AI和录像",
        "enabled": {
            "teleop": True, "record": True, "replay": True,
            "vlm": False, "video_recording": False, "task_control": False,
        },
    },
    "ai": {
        "label": "AI自主抓取模式",
        "desc": "VLM视觉识别+自主抓取+录像，禁用轨迹编辑",
        "enabled": {
            "teleop": False, "record": False, "replay": True,
            "vlm": True, "video_recording": True, "task_control": True,
        },
    },
    "benchmark": {
        "label": "性能对比实验模式",
        "desc": "展示各功能处理时间和识别率",
        "enabled": {
            "teleop": False, "record": False, "replay": True,
            "vlm": True, "video_recording": True, "task_control": False,
        },
    },
}


class TeachingMode:
    """教学模式与权限管理"""

    def __init__(self, role: str = "student", mode: str = "manual"):
        if role not in ("student", "teacher"):
            role = "student"
        if mode not in MODES:
            mode = "manual"
        self.role = role
        self.mode = mode
        self._mode_config = dict(MODES[mode])

    def is_teacher(self) -> bool:
        return self.role == "teacher"

    def is_enabled(self, feature: str) -> bool:
        """检查某功能在当前模式下是否可用"""
        return self._mode_config.get("enabled", {}).get(feature, False)

    def switch_mode(self, mode: str) -> bool:
        """切换模式（仅教师）"""
        if not self.is_teacher():
            return False
        if mode not in MODES:
            return False
        self.mode = mode
        self._mode_config = dict(MODES[mode])
        return True

    def get_label(self) -> str:
        return MODES[self.mode]["label"]

    def get_desc(self) -> str:
        return MODES[self.mode]["desc"]

    def get_enabled_summary(self) -> str:
        enabled = [k for k, v in self._mode_config.get("enabled", {}).items() if v]
        return ", ".join(enabled) if enabled else "无"


class SystemRestore:
    """系统一键还原"""

    @staticmethod
    def reset_motion_library() -> int:
        """清空动作库（保留 index.json 骨架）"""
        count = 0
        if MOTION_DIR.exists():
            for f in MOTION_DIR.glob("*.json"):
                if f.name == "index.json":
                    # 重置 index 为默认骨架
                    default = {
                        "greeting": {"file": "", "keywords": ["你好", "打招呼"], "duration_s": 0, "category": "social"},
                        "grasp": {"file": "", "keywords": ["抓", "拿", "取"], "duration_s": 0, "category": "manipulation"},
                    }
                    with open(f, "w", encoding="utf-8") as fh:
                        json.dump(default, fh, indent=2, ensure_ascii=False)
                else:
                    f.unlink()
                    count += 1
        return count

    @staticmethod
    def reset_task_library() -> int:
        """清空任务库"""
        count = 0
        if TASK_DIR.exists():
            for f in TASK_DIR.glob("*.json"):
                f.unlink()
                count += 1
        return count

    @staticmethod
    def reset_recordings() -> int:
        """清空录像文件"""
        rec_dir = Path(__file__).resolve().parent.parent / "recordings"
        count = 0
        if rec_dir.exists():
            for f in rec_dir.glob("*.mp4"):
                f.unlink()
                count += 1
        return count

    @staticmethod
    def reset_all() -> dict:
        """一键还原所有"""
        return {
            "motions": SystemRestore.reset_motion_library(),
            "tasks": SystemRestore.reset_task_library(),
            "recordings": SystemRestore.reset_recordings(),
        }
