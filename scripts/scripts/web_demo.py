#!/usr/bin/env python3
"""Gradio Web 控制界面（可选启动，不影响主线流程）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import gradio as gr
from config.settings import (
    CAMERA_MATRIX, SERIAL_PORT, SERIAL_BAUD,
    VLM_MODEL_NAME, VLM_MODEL_PATH, VLM_DEMO_BIN,
)
from astra import AstraProCamera
from vla.vlm import create_vlm
from vla.vision import ColorLocator
from vla.control import ArmController


def capture_frame():
    cam = AstraProCamera(21)
    cam.connect()
    ret, frame = cam.rgb_cap.read()
    cam.disconnect()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def vlm_infer():
    rgb = capture_frame()
    if rgb is None:
        return "相机失败", None, "无画面"
    cv2.imwrite("/tmp/web_snap.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    result = vlm.infer("/tmp/web_snap.jpg")
    return result.color, result.object, result.raw[:200]


def grasp(color: str):
    rgb = capture_frame()
    if rgb is None:
        return "相机失败"
    cam2 = AstraProCamera(21)
    cam2.connect()
    rgb2, depth = cam2.read()
    cam2.disconnect()
    locator = ColorLocator(CAMERA_MATRIX)
    pos = locator.locate(rgb2, depth, color)
    if pos is None:
        return f"未找到{color}色物体"
    arm = ArmController(SERIAL_PORT, SERIAL_BAUD)
    arm.move_to(pos["x"], pos["y"], pos["z"])
    arm.gripper(False)
    import time
    time.sleep(1)
    arm.move_to(0.30, 0.0, 0.10)
    arm.gripper(True)
    arm.close()
    return f"✅ 抓取完成 @ ({pos['x']:.2f},{pos['y']:.2f},{pos['z']:.2f})"


def create_ui():
    with gr.Blocks(title="RK3588-EIA 具身智能系统") as ui:
        gr.Markdown("""
        # 🤖 RK3588-EIA 端侧具身智能系统
        LeRobot + VLM (RKLLM) + Astra Pro + SO-ARM101
        """)

        with gr.Tab("🎮 控制面板"):
            with gr.Row():
                with gr.Column():
                    cam_img = gr.Image(label="相机画面")
                    refresh_btn = gr.Button("📷 刷新画面")
                with gr.Column():
                    vlm_color = gr.Textbox(label="VLM 颜色", interactive=False)
                    vlm_obj = gr.Textbox(label="VLM 物体", interactive=False)
                    vlm_raw = gr.Textbox(label="VLM 原始输出", lines=4, interactive=False)
                    vlm_btn = gr.Button("🧠 VLM 推理")

            with gr.Row():
                color_input = gr.Dropdown(
                    choices=["红色", "绿色", "蓝色", "黄色", "白色", "黑色"],
                    label="抓取颜色", value="红色"
                )
                grasp_btn = gr.Button("🤖 开始抓取")
                grasp_result = gr.Textbox(label="抓取结果", interactive=False)

        with gr.Tab("📊 系统状态"):
            gr.Markdown("""
            **组件状态**
            - RK3588 NPU: ✅
            - VLM 模型: qwen3-vl-2b / smolvlm2-256m
            - Astra Pro: ✅
            - SO-ARM101: 串口连接中
            - whisper.cpp: 🎤 语音就绪
            """)

        refresh_btn.click(fn=capture_frame, outputs=cam_img)
        vlm_btn.click(fn=vlm_infer, outputs=[vlm_color, vlm_obj, vlm_raw])
        grasp_btn.click(fn=grasp, inputs=color_input, outputs=grasp_result)

    return ui


if __name__ == "__main__":
    print("启动 Web 控制界面...")
    print("浏览器访问: http://0.0.0.0:7860")
    vlm = create_vlm(VLM_MODEL_NAME, demo_bin=VLM_DEMO_BIN)
    vlm.load(VLM_MODEL_PATH)
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
