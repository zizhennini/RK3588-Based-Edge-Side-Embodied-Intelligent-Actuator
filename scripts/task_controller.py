#!/usr/bin/env python3
"""多工序任务控制器 — 组合轨迹 + 坐标修正 + 安全校验 + 人工复核"""

import sys, os, json, time, numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MOTION_DIR = Path(__file__).resolve().parent.parent / "motion_library"
TASK_DIR = Path(__file__).resolve().parent.parent / "task_library"


# ── 任务格式 ──
#
# task_library/task_name.json:
# {
#   "meta": {
#     "name": "grasp_and_place",
#     "category": "manipulation",
#     "created_at": "2026-07-06T12:00:00",
#     "verified": false
#   },
#   "steps": [
#     {
#       "id": 0,
#       "name": "reach",
#       "trajectory": "reach_01.json",
#       "offset_xyz": [0.0, 0.0, 0.0],
#       "gripper": null
#     },
#     {
#       "id": 1,
#       "name": "grasp",
#       "trajectory": "grasp_01.json",
#       "offset_xyz": [0.0, 0.0, 0.0],
#       "gripper": false
#     }
#   ]
# }


class TaskFormatError(Exception):
    pass


class TaskSafetyError(Exception):
    pass


def validate_task(task: dict) -> list[str]:
    """校验任务合法性，返回警告列表"""
    warnings = []
    if "meta" not in task:
        warnings.append("缺少 meta 字段")
        return warnings
    if "steps" not in task or not task["steps"]:
        warnings.append("缺少 steps 或为空")
        return warnings
    for i, step in enumerate(task["steps"]):
        if "trajectory" not in step:
            warnings.append(f"步骤 {i}: 缺少 trajectory")
            continue
        traj_path = MOTION_DIR / step["trajectory"]
        if not traj_path.exists():
            warnings.append(f"步骤 {i}: 轨迹文件不存在 {step['trajectory']}")
        offset = step.get("offset_xyz", [0, 0, 0])
        if len(offset) != 3:
            warnings.append(f"步骤 {i}: offset_xyz 需为 3 个值")
    return warnings


class TaskController:
    """多工序任务执行器"""

    def __init__(self, arm, motion_dir: str | Path = None):
        self.arm = arm
        self.motion_dir = Path(motion_dir) if motion_dir else MOTION_DIR
        self.task_dir = self.motion_dir.parent / "task_library"
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._interrupted = False
        self._on_progress = None

    def on_progress(self, cb):
        self._on_progress = cb

    def interrupt(self):
        self._interrupted = True

    # ── 任务录入 ──

    def create_task(self, name: str, category: str, steps: list[dict]) -> dict:
        """创建新任务"""
        for s in steps:
            if "trajectory" not in s:
                raise TaskFormatError(f"步骤缺少 trajectory: {s}")
            if "name" not in s:
                s["name"] = f"step_{steps.index(s)}"
        task = {
            "meta": {
                "name": name,
                "category": category,
                "created_at": datetime.now().isoformat(),
                "verified": False,
            },
            "steps": steps,
        }
        return task

    def save_task(self, task: dict) -> Path:
        """保存任务到 task_library/"""
        name = task["meta"]["name"]
        path = self.task_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2, ensure_ascii=False)
        print(f"[Task] 已保存: {path.name}")
        return path

    def load_task(self, name: str) -> dict:
        """加载任务"""
        path = self.task_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"任务不存在: {name}")
        with open(path) as f:
            return json.load(f)

    def list_tasks(self) -> list[str]:
        """列出所有任务"""
        if not self.task_dir.exists():
            return []
        return sorted([f.stem for f in self.task_dir.glob("*.json")])

    def delete_task(self, name: str):
        path = self.task_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            print(f"[Task] 已删除: {name}")

    # ── 安全校验 ──

    def verify_safety(self, task: dict) -> list[str]:
        """安全校验：轨迹存在性 + 关节范围"""
        warnings = validate_task(task)
        if warnings:
            return warnings
        from vla.control import ArmController
        limits = ArmController.JOINT_LIMITS
        for i, step in enumerate(task["steps"]):
            traj_path = self.motion_dir / step["trajectory"]
            try:
                with open(traj_path) as f:
                    traj = json.load(f)
            except Exception as e:
                warnings.append(f"步骤 {i}: 无法加载轨迹 - {e}")
                continue
            frames = traj if isinstance(traj, list) else traj.get("frames", [])
            for j, frame in enumerate(frames):
                joints = frame if isinstance(frame, list) else frame.get("joints", [])
                if len(joints) < 6:
                    continue
                for k, name in enumerate(ArmController.JOINT_NAMES):
                    low, high = limits[name]
                    if joints[k] < low or joints[k] > high:
                        warnings.append(f"步骤 {i} 帧 {j} 关节 {k}({name}): {joints[k]:.3f} 超限")
        return warnings

    # ── VLM 视觉偏移注入 ──

    def apply_vlm_offset(self, task_name: str, step_index: int, offset_xyz: list[float]) -> dict:
        """接收 VLM 视觉识别的三维偏移，注入指定步骤并保存"""
        task = self.load_task(task_name)
        if step_index < 0 or step_index >= len(task["steps"]):
            raise ValueError(f"步骤索引 {step_index} 超出范围 (0-{len(task['steps'])-1})")
        task["steps"][step_index]["offset_xyz"] = [round(v, 4) for v in offset_xyz]
        self.save_task(task)
        print(f"[Task] VLM 偏移已注入: {task_name} 步骤{step_index} -> {offset_xyz}")
        return task

    # ── 执行 ──

    def execute(self, task: dict, loop: int = 1, dry_run: bool = False):
        """执行多工序任务"""
        self._interrupted = False
        name = task["meta"]["name"]
        steps = task["steps"]
        print(f"[Task] 开始: {name} ({len(steps)} 步, {loop} 次)")
        if dry_run:
            print("[Task] 模拟运行模式")
        for lap in range(loop):
            if self._interrupted:
                break
            print(f"[Task] 第 {lap+1}/{loop} 轮")
            for i, step in enumerate(steps):
                if self._interrupted:
                    break
                step_name = step.get("name", f"step_{i}")
                traj_file = step["trajectory"]
                offset = step.get("offset_xyz", [0, 0, 0])
                self._report(f"步骤 {i+1}/{len(steps)}: {step_name}")
                if dry_run:
                    print(f"  [模拟] 加载 {traj_file}, 偏移 {offset}")
                    time.sleep(0.5)
                    continue
                self._execute_step(traj_file, offset, step.get("gripper"))
            if not self._interrupted and lap < loop - 1:
                print(f"[Task] 第 {lap+1} 轮完成，继续下一轮")
        if not self._interrupted:
            print(f"[Task] 完成: {name}")

    def _execute_step(self, traj_file: str, offset_xyz: list[float], gripper: bool | None):
        """执行单步轨迹（关节空间轨迹，offset 仅用于 VLM 抓取的笛卡尔坐标预修正）"""
        import numpy as np
        traj_path = self.motion_dir / traj_file
        try:
            with open(traj_path) as f:
                traj = json.load(f)
        except Exception as e:
            print(f"  [错误] 加载轨迹失败: {e}")
            return
        frames = traj if isinstance(traj, list) else traj.get("frames", [])
        if not frames:
            return
        fps = traj.get("fps", 30) if not isinstance(traj, list) else 30
        dt = 1.0 / fps
        offset = np.array(offset_xyz, dtype=float)
        has_offset = not np.allclose(offset, 0)
        if has_offset:
            print(f"  [信息] offset_xyz={offset_xyz} 仅在前3关节生效（近似偏移，需 IK 才能精确）")
        for i, frame in enumerate(frames):
            if self._interrupted:
                print("  [中断]")
                return
            joints = frame if isinstance(frame, list) else frame.get("joints", [])
            if len(joints) < 6:
                print(f"  [警告] 帧{i}关节数不足({len(joints)})，跳过")
                continue
            arr = np.array(joints)
            if has_offset:
                arr[:3] += offset[:3]
            if self.arm and hasattr(self.arm, "write_angles"):
                self.arm.write_angles(arr)
            time.sleep(dt)
        if gripper is not None and self.arm and hasattr(self.arm, "gripper"):
            self.arm.gripper(gripper)

    def _report(self, msg: str):
        print(f"[Task] {msg}", flush=True)
        if self._on_progress:
            self._on_progress(msg)

    # ── 人工复核 ──

    def review(self, task: dict) -> bool:
        """人工复核：打印任务详情等待确认"""
        print("\n" + "=" * 50)
        print("  人工复核")
        print("=" * 50)
        meta = task["meta"]
        print(f"  任务: {meta['name']} ({meta['category']})")
        print(f"  步骤数: {len(task['steps'])}")
        for i, s in enumerate(task["steps"]):
            offset = s.get("offset_xyz", [0, 0, 0])
            grip = s.get("gripper", "不变")
            print(f"    {i+1}. {s['name']:12s}  轨迹:{s['trajectory']:20s}  偏移:{offset}  夹爪:{grip}")
        print()
        try:
            r = input("  确认执行? (y/n): ").strip().lower()
            return r == "y"
        except (EOFError, KeyboardInterrupt):
            return False


def main():
    import argparse
    p = argparse.ArgumentParser(description="多工序任务控制器")
    sub = p.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="列出所有任务")

    p_show = sub.add_parser("show", help="查看任务详情")
    p_show.add_argument("name")

    p_delete = sub.add_parser("delete", help="删除任务")
    p_delete.add_argument("name")

    p_run = sub.add_parser("run", help="执行任务")
    p_run.add_argument("name")
    p_run.add_argument("--loop", type=int, default=1)
    p_run.add_argument("--dry-run", action="store_true")

    p_verify = sub.add_parser("verify", help="安全校验")
    p_verify.add_argument("name")

    p_record = sub.add_parser("record", help="交互式录制任务")
    p_record.add_argument("name", help="任务名称")
    p_record.add_argument("--steps", type=int, default=2, help="步骤数")

    args = p.parse_args()
    tc = TaskController(arm=None)

    if args.cmd == "list":
        for t in tc.list_tasks():
            print(t)
    elif args.cmd == "show":
        task = tc.load_task(args.name)
        print(json.dumps(task, indent=2, ensure_ascii=False))
    elif args.cmd == "delete":
        tc.delete_task(args.name)
    elif args.cmd == "verify":
        task = tc.load_task(args.name)
        warns = tc.verify_safety(task)
        if warns:
            for w in warns:
                print(f"  ⚠ {w}")
        else:
            print("  ✅ 安全校验通过")
    elif args.cmd == "run":
        task = tc.load_task(args.name)
        tc.execute(task, loop=args.loop, dry_run=args.dry_run)
    elif args.cmd == "record":
        steps = []
        print(f"录制任务: {args.name} ({args.steps} 步)")
        for i in range(args.steps):
            print(f"\n--- 步骤 {i+1}/{args.steps} ---")
            traj = input(f"  轨迹文件名 (motion_library/ 下): ").strip()
            ox = float(input(f"  offset_x [0]: ") or "0")
            oy = float(input(f"  offset_y [0]: ") or "0")
            oz = float(input(f"  offset_z [0]: ") or "0")
            gripper = input(f"  夹爪状态 (open/close/无): ").strip()
            g = None
            if gripper == "open": g = True
            elif gripper == "close": g = False
            step_name = input(f"  步骤名称 [step_{i+1}]: ").strip() or f"step_{i+1}"
            steps.append({"name": step_name, "trajectory": traj,
                          "offset_xyz": [ox, oy, oz], "gripper": g})
        task = tc.create_task(args.name, "custom", steps)
        warns = tc.verify_safety(task)
        if warns:
            print("\n⚠ 安全校验警告:")
            for w in warns: print(f"  - {w}")
        else:
            print("\n✅ 安全校验通过")
        tc.save_task(task)
        print(f"任务已保存: {args.name}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
