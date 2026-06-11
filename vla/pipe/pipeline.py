"""主流水线 — VLM 场景理解 + ColorLocator 定位 → Depth → IK"""
import cv2
import time
import numpy as np
from vla.vlm import VLMBase
from vla.vision import ColorLocator
from vla.control import ArmController


class State:
    IDLE = 0
    VLM_INFER = 1
    GRASP = 2
    PLACE = 3
    DONE = 4


class VLApipeline:
    """VLM 场景理解 + ColorLocator 定位 → Depth → IK"""

    def __init__(self, arm: ArmController, vlm: VLMBase, camera_matrix):
        self.arm = arm
        self.vlm = vlm
        self.K = camera_matrix
        self.locator = ColorLocator(camera_matrix)
        self.state = State.IDLE
        self.target_3d: tuple | None = None
        self.vlm_color = "红色"

    def start(self):
        self.state = State.VLM_INFER

    def step(self, rgb, depth) -> str:
        if self.state == State.IDLE:
            return "idle"

        elif self.state == State.VLM_INFER:
            tmp = "/tmp/vla_frame.jpg"
            cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            result = self.vlm.infer(tmp)
            self.vlm_color = result.color

            if result.cx is not None and result.cy is not None:
                u, v = int(result.cx), int(result.cy)
            else:
                # VLM 未输出坐标，降级为颜色定位
                pos = self.locator.locate(rgb, depth, result.color)
                if pos is None:
                    self.state = State.DONE
                    return f"locate_failed: {result.color} {result.object}"
                u, v = pos["u"], pos["v"]

            z = float(depth[v, u])
            if z <= 0 or z > 5:
                z = 0.35
            x = (u - self.K[0, 2]) * z / self.K[0, 0]
            y = (v - self.K[1, 2]) * z / self.K[1, 1]
            self.target_3d = (x, y, z)
            self.state = State.GRASP
            return f"vlm: {result.color} {result.object[:20]} @ ({u},{v}) z={z:.2f}"

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
