# Summary of RK3588\-Based Edge\-Side Embodied Intelligent Actuator

## 1\. Project Overview

This open\-source project developed by `zizhennini` is a lightweight offline edge embodied intelligent platform built on the Rockchip RK3588 development board\. Paired with the SO\-ARM101 6\-DOF robotic arm, it realizes a fully local closed\-loop pipeline: **Voice/Text Command → Vision\-Language Reasoning → Task Scheduling → Robotic Arm Execution**\. The whole system works offline without cloud network access, targeting three core scenarios: university robotics teaching, embodied intelligence research experiments, and desktop automated demonstrations\. The repository is primarily written in Python \(99\.4%\) with a small proportion of Shell scripts\.

## 2\. Hardware \& Software Stack

### 2\.1 Recommended Hardware

- Edge computing unit: RK3588 development board \(8GB / 16GB RAM recommended\)

- Actuator: SO\-ARM101 6\-DOF robotic arm, Feetech STS3215 serial bus servos

- Perception sensors: Intel RealSense D435i RGB\-D depth camera, USB microphone, speaker/USB audio device

- Communication: USB / TTL serial adapter

### 2\.2 Layered Software Architecture

1. System layer: Linux OS for RK3588, Python 3\.10

2. Speech module: sherpa\-onnx for offline KWS wake\-up, ASR speech recognition and TTS speech synthesis

3. Vision reasoning layer: OpenCV \+ pyrealsense2; lightweight Qwen VLM accelerated by RKNN/RKLLM NPU

4. Robotic control layer: LeRobot\-style control interface compatible with Feetech servo serial protocol

5. Video recording module: FFmpeg with Rockchip hardware encoder `h264_rkmpp`, RGA for image scaling and OSD text overlay

6. Task scheduling: FIFO command queue with interrupt handling mechanism

## 3\. Five Core Functional Modules

### 3\.1 Offline Edge Multimodal Interaction

Supports dual input modes \(voice \& text\)\. Local VLM delivers visual question answering and target object description\. All speech functions run completely offline without internet APIs\.

### 3\.2 Precise Control of 6\-DOF Robotic Arm

Provides Cartesian space motion commands and direct joint angle writing, gripper control and hardware emergency stop interface\. Real\-time reading of servo pulse values, joint angles and runtime status\.

### 3\.3 Leader\-Follower Demonstration \& Motion Library Management \(Core for Imitation Learning\)

- Collect joint data from the leader arm and map motion to the follower robotic arm automatically

- Standard trajectory processing pipeline: duplicate frame removal → median filtering smoothing → velocity limitation → safety verification → manual review

- Persist validated trajectories into motion library; support creation, inspection, deletion, batch smoothing, loop/pause replay, and one\-click trigger via voice/text commands

### 3\.4 Depth Vision\-Based Grasping Experiment

Combines 3D coordinate data from depth camera and VLM semantic recognition to locate target objects\. A `dry-run` simulation mode is provided to verify planning before physical execution to reduce collision risks\.

### 3\.5 RK Hardware\-Accelerated Experiment Recording

Utilizes the chip’s VPU and RGA hardware units for video encoding to drastically lower CPU consumption\. Supports custom recording duration and on\-screen text watermarks for experiment archiving, competition demos and debugging review\.

## 4\. Overall System Workflow

1. Receive voice or text input and conduct intent recognition

2. Distribute tasks into three branches:

    - VQA branch: Camera frame capture → local VLM inference → output scene/object descriptions

    - Motion replay branch: Load trajectory files from motion library → trajectory safety check → push into command queue

    - Manual robot control branch: Generate motion commands directly and enqueue

3. Unified scheduling from command queue sends instructions to arm controller to drive servos and SO\-ARM101 manipulator

4. Synchronously capture camera frames and encode experiment footage via hardware acceleration

## 5\. Repository Structure \& Core Scripts

The project adopts modular layered design, with newly added folders/files including `camera`, `lerobot`, `tests` and `menu.py`\. Key directories:

- Entry files: `main.py` \(main VLA program\), `va.py` \(voice assistant\), `menu.py` \(interactive menu\)

- Configuration layer: `config/` for device parameters, safety thresholds and teaching settings

- Core business modules: `vla/` \(scheduling, robot control, VLM inference\), `voice_assistant/` \(full offline speech toolkit\)

- Utility scripts under `scripts/`: demonstration recording, trajectory smoothing \& replay, VLM grasping, hardware recording, multi\-step task controller

- Resource directories: `motion_library` \(trajectory files\), `task_library` \(composite task templates\), `recordings` \(experiment videos\), `models` \(robot URDF and AI model files\)

- Auxiliary folders: `docs` \(project documentation\), `tests` \(unit test cases\)

## 6\. Deployment \& Quick Start Guide

1. Clone the repository, build a Python 3\.10 Conda environment and install dependencies

2. Detect serial ports and camera devices, update serial port ID, camera index and VLM model path in configuration files

3. Verify RK hardware encoding environment; compile FFmpeg with Rockchip MPP support if encoders/filters are missing

4. Main function launch methods:

    - Voice assistant: continuous listening, single\-round recognition, text dialogue, visual QA

    - Motion library: list trajectories, inspect files, batch smooth, replay motion records

    - Demonstration workflow: leader\-follower collection → automatic processing → library storage

    - Vision grasping: run dry\-run simulation first, remove simulation flag for physical grasping after safety confirmation

    - Video recording: customize duration and add OSD text to save footage

## 7\. Multi\-Layer Safety Mechanisms

Multiple protection rules to avoid mechanical collision and hardware damage:

1. All new trajectories must be tested in dry\-run simulation mode first

2. Motion execution with speed limit, joint range check and servo ID mapping validation

3. Hardware emergency stop interface reserved

4. Autonomous grasping tasks require full camera\-hand calibration and on\-site manual supervision

## 8\. Development Status \& Roadmap

### Completed Functions

Offline full speech stack, text interaction, VLM visual reasoning, leader\-follower demonstration, trajectory smoothing \& replay, motion library management, hardware video recording\.

- Vision grasping: experimental feature

- RGA scaling \& OSD overlay: subject to hardware environment compatibility

- General full autonomous manipulation: under development

### Long\-Term Iteration Plan

1. Optimize stability and parsing logic of VLM output; expand grasping task templates

2. Complete standardized camera\-robot hand\-eye calibration workflow

3. Add dual safety monitoring based on servo current and depth image data

4. Develop web\-based visualization dashboard for classroom demonstration

5. Support more types of robotic arms and end\-effectors

6. Enable imitation learning dataset export for algorithm research

## 9\. Application Value \& Open Source Statement

### Educational Value

Covers core knowledge points including edge AI deployment, multimodal human\-robot interaction, robot kinematics, imitation learning and Rockchip hardware acceleration, with standardized reproducible experimental pipelines\.

### Research Value

Provides a lightweight local embodied intelligence experimental base for rapid validation of VLM robot manipulation, offline multimodal interaction and demonstration imitation algorithms\.

### Open Source License

No LICENSE file included in the original repository\. An appropriate open\-source license must be added before public distribution or reuse\. The project draws inspiration and references from open\-source communities including Rockchip RK3588 ecosystem, LeRobot, sherpa\-onnx and Intel RealSense\.

