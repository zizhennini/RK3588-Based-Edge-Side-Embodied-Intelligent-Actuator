"""主流水线 — VLM 直出坐标 → Depth 反投影 → 机械臂执行"""
import cv2
import time
from vla.vlm import VLMBase
from vla.control import ArmController


class State:
    IDLE = 0
    VLM_INFER = 1
    GRASP = 2
    PLACE = 3
    DONE = 4


class VLApipeline:
    """VLM 一次性场景理解 + 目标定位 → Depth → IK → 机械臂"""

    def __init__(self, arm: ArmController, vlm: VLMBase, camera_matrix):
        self.arm = arm
        self.vlm = vlm
        self.K = camera_matrix
        self.state = State.IDLE
        self.target_3d: tuple | None = None

    def start(self):
        self.state = State.VLM_INFER

    def step(self, rgb, depth) -> str:
        if self.state == State.IDLE:
            return "idle"

        elif self.state == State.VLM_INFER:
            tmp = "/tmp/vla_frame.jpg"
            cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            result = self.vlm.infer(tmp)
            if result.cx is None or result.cy is None:
                self.state = State.DONE
                return f"vlm_failed: no coordinates ({result.object})"

            # VLM 输出了像素坐标 → Depth 查深度
            u, v = int(result.cx), int(result.cy)
            z = float(depth[v, u])
            if z <= 0 or z > 5:
                z = 0.35
            x = (u - self.K[0, 2]) * z / self.K[0, 0]
            y = (v - self.K[1, 2]) * z / self.K[1, 1]
            self.target_3d = (x, y, z)
            self.state = State.GRASP
            return f"vlm: {result.color} {result.object} @ ({u},{v}) z={z:.2f}"

        elif self.state == State.GRASP:
            if self.target_3d is None:
                self.state = State.DONE
                return "grasp_failed"
            self.arm.move_to(*self.target_3d)
            self.arm.gripper(False)
            time.sleep(1)
            self.state = State.PLACE
            return f"grasp: ({self.target_3d[0]:.2f},{self.target_3d[1]:.2f},{self.target_3d[2]:.2f})"

        elif self.state == State.PLACE:
            self.arm.move_to(0.30, 0.0, 0.10)
            self.arm.gripper(True)
            self.state = State.DONE
            return "place_done"

        return "done"
