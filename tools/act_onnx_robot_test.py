#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/act_onnx_robot_test.py — 真机最小闭环测试（act.onnx 单图三输入）

安全设计：
  --dry-run（默认）只推理不驱动电机，打印策略"想下发"的动作与幅度 → 先确认安全
  实际驱动时必须加 --live；带单步限幅（--max-step-deg）、总时长上限（--duration）、
  Ctrl-C 立即停机并松开扭矩。

输入约定（与训练一致）：
  observation.state       = 从臂当前弧度（原始，不做归一化）
  observation.images.front/wrist = 相机 BGR uint8 HWC → RGB [0,1] float32 CHW
输出：action (1,100,6) 已是物理单位（弧度）
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CAM_KEYS = {"front": "observation.images.front", "wrist": "observation.images.wrist"}
GRIPPER_ID = 6

# 采集时三条演示的起始姿态（由"首帧复现"校验从真实数据读出，三条几乎一致）
#   episode_0001/0002/0003 首帧记录动作(度) ≈ J1 0 / J2 -104 / J3 98 / J4 58 / J5 5 / J6 -55
DEMO_START_DEG = [0.0, -104.0, 98.0, 58.0, 5.0, -55.0]
WARN_DEG = 12.0        # 与演示起始姿态偏差超过该值就提示（策略对起始状态敏感）


def pose_report(state_rad: np.ndarray) -> tuple:
    """与演示起始姿态比较 → (每关节偏差(度), 最大偏差(度))"""
    cur = np.rad2deg(np.asarray(state_rad, dtype=np.float64).reshape(-1)[:6])
    dev = cur - np.asarray(DEMO_START_DEG, dtype=np.float64)
    return dev, float(np.abs(dev).max())


def to_chw_rgb01(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H,W,3) → RGB float32 (1,3,H,W) [0,1]"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))[None]


def main() -> int:
    ap = argparse.ArgumentParser(description="ACT(act.onnx) 真机最小闭环测试")
    ap.add_argument("--onnx", default="/home/elf/work/act_onnx2/act.onnx")
    ap.add_argument("--cameras", default="./config/cameras.json")
    ap.add_argument("--port", default="/dev/so101_follower")
    ap.add_argument("--n-action-steps", type=int, default=10, help="每推一次执行几步")
    ap.add_argument("--duration", type=float, default=10.0, help="总时长上限(s)")
    ap.add_argument("--max-step-deg", type=float, default=3.0, help="单步限幅(度)")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--live", action="store_true", help="真正驱动电机（默认只推理）")
    ap.add_argument("--check-pose", action="store_true",
                    help="只检查/打印当前姿态与演示起始姿态的偏差后退出（不推理、不动电机）")
    args = ap.parse_args()

    sys_path = str(Path(__file__).resolve().parent.parent)
    import sys
    sys.path.insert(0, sys_path)
    from tools.cam_sink import CameraSet

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    print("ONNX 输入:", in_names)

    cs = CameraSet.from_config(args.cameras)
    cs.start()
    ready = cs.wait_ready(timeout_s=25.0)
    print("相机就绪:", ready)

    arm = None
    if args.live or args.check_pose:
        from hardware.arm import SO101Arm
        arm = SO101Arm(port=args.port)
        arm.connect()
        arm.bus.disable_torque()
        print("从臂已连接（扭矩已关）")
        st = np.asarray(arm.read_positions(), dtype=np.float64)
        dev, worst = pose_report(st)
        print("当前姿态(度): " + "  ".join("J%d %6.1f" % (i + 1, v)
                                      for i, v in enumerate(np.rad2deg(st[:6]))))
        print("演示起始(度): " + "  ".join("J%d %6.1f" % (i + 1, v)
                                      for i, v in enumerate(DEMO_START_DEG)))
        print("偏差(度)    : " + "  ".join("J%d %+6.1f" % (i + 1, v)
                                      for i, v in enumerate(dev)))
        if worst > WARN_DEG:
            print("  ⚠ 与演示起始姿态最大偏差 %.1f° > %.0f°：请先手动把从臂摆回起始姿态，"
                  "否则策略在分布外、动作会不对（扭矩已松开，可直接用手摆）" % (worst, WARN_DEG))
        else:
            print("  ✓ 起始姿态 OK（最大偏差 %.1f°）" % worst)
        if args.check_pose:
            arm.close()
            cs.stop()
            return 0 if worst <= WARN_DEG else 2
    if args.live:
        print("=== live：将驱动从臂（Ctrl-C 可随时停并松扭矩）===")
    else:
        print("=== dry-run：只推理，不驱动电机 ===")

    limit = np.deg2rad(args.max_step_deg)
    t0 = time.perf_counter()
    n_infer = 0
    queue = []
    last_cmd = None
    try:
        while time.perf_counter() - t0 < args.duration:
            # 取图
            feed = {}
            for cam, key in CAM_KEYS.items():
                got = cs.latest(cam)
                if got is None:
                    print("  ⚠ 相机 %s 无帧" % cam)
                    continue
                feed[key] = to_chw_rgb01(np.ascontiguousarray(got[0]))
            if len(feed) != len(CAM_KEYS):
                time.sleep(0.05)
                continue
            # 读状态（dry-run 时用零位替代，仅验证链路）
            if arm is not None:
                state = np.asarray(arm.read_positions(), dtype=np.float32)
            else:
                state = np.zeros(6, dtype=np.float32)
            feed["observation.state"] = state.reshape(1, 6)

            if not queue:
                t_inf = time.perf_counter()
                out = sess.run(None, feed)[0][0]          # (100,6)
                dt = (time.perf_counter() - t_inf) * 1000
                n_infer += 1
                queue = [out[i] for i in range(args.n_action_steps)]
                print("[推理 %d] %.0f ms | chunk 前 3 步:\n%s" % (
                    n_infer, dt, np.round(out[:3], 4)))

            target = queue.pop(0)
            if last_cmd is None:
                last_cmd = state if arm is not None else target
            delta = np.clip(target - last_cmd, -limit, limit)
            cmd = last_cmd + delta
            if np.abs(target - last_cmd).max() > limit:
                print("  （限幅生效：最大需求 %.2f° → 实际 %.2f°）" % (
                    np.rad2deg(np.abs(target - last_cmd).max()),
                    np.rad2deg(np.abs(delta).max())))
            if arm is not None:
                if last_cmd is None:
                    arm.bus.enable_torque()
                arm.write_positions(cmd)
            last_cmd = cmd
            print("  cmd(rad): %s" % np.round(cmd, 3))
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        if arm is not None:
            try:
                arm.bus.disable_torque()
                print("已松开从臂扭矩")
            except Exception as e:
                print("松扭矩失败:", e)
            arm.close()
        cs.stop()
    print("结束（推理 %d 次）" % n_infer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
