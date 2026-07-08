# 基于RK3588的边缘侧具身智能执行器项目总结文档

# Summary Document of RK3588\-Based Edge\-Side Embodied Intelligent Actuator

## Document Information

- Project Repository: [https://github\.com/zizhennini/RK3588\-Based\-Edge\-Side\-Embodied\-Intelligent\-Actuator](https://github.com/zizhennini/RK3588-Based-Edge-Side-Embodied-Intelligent-Actuator)

- Developer: zizhennini

- Main Programming Language: Python \(99\.4%\), Shell Script \(0\.6%\)

- Document Type: Complete Project Summary Document

- Applicable Scenarios: Academic Report, Project Introduction, Opening Report, Technical Document

# 1\. Detailed Project Introduction

## 1\.1 Project Background

In recent years, embodied intelligence robotics has become a core research direction in artificial intelligence and robotics\. Most mainstream embodied intelligent solutions rely on cloud large model computing power, which brings obvious drawbacks: high network latency, risk of user data privacy leakage, inability to work in offline environments, and high long\-term usage costs\.

On the hardware side, existing small robotic arm control tools only support simple point\-to\-point motion control\. They lack integrated capabilities such as multimodal vision\-language fusion, offline voice human\-computer interaction, and teaching imitation learning\. These tools cannot meet the lightweight teaching and experimental needs of university robotics laboratories\.

Rockchip RK3588 is a domestic high\-performance edge computing chip equipped with a 6 TOPS NPU hardware acceleration unit\. It natively supports RKNN/RKLLM local deployment of large multimodal models, hardware video encoding and decoding, and RGA image hardware acceleration\. With advantages of low power consumption, low cost, full domestic production and strong local computing capability, it is the optimal embedded hardware carrier for desktop lightweight embodied intelligent robots\.

Based on the above industry pain points and hardware advantages, this open\-source project develops a fully offline edge embodied intelligent actuator platform based on RK3588, filling the gap of integrated teaching and research equipment integrating offline voice, visual reasoning, robotic arm imitation teaching and automatic recording\.

## 1\.2 Core Project Positioning

This is a lightweight, fully offline local closed\-loop embodied intelligence platform based on RK3588 edge computing board and SO\-ARM101 six\-degree\-of\-freedom robotic arm\. The system integrates offline voice interaction, visual\-language multimodal reasoning, master\-slave teaching demonstration, motion trajectory library playback, full\-process safety detection and robotic arm physical execution modules to form an end\-to\-end working pipeline of **Voice/Text Instruction → Environmental Visual Understanding → Multi\-Task Scheduling → Robotic Arm Physical Execution**\.

The whole system operates completely offline without any dependence on cloud servers or network APIs\. It is specially designed for three core application scenarios:

1. University teaching: Courses of Robotics Engineering, Artificial Intelligence, Embedded Development and Embodied Intelligence;

2. Scientific research experiments: Small\-scale imitation learning, visual target grasping, offline multimodal human\-robot interaction algorithm verification;

3. Desktop demonstration: Offline AI exhibition, discipline competition demonstration, laboratory automatic sorting and display experiments\.

## 1\.3 Core Project Construction Goals

1. Full offline localization: Offline voice wake\-up/recognition/synthesis, multimodal VLM reasoning all run on local NPU without internet access;

2. Natural multimodal human\-computer interaction: Support voice and text dual instruction input, realize voice question and answer \+ visual scene description;

3. Complete imitation learning pipeline: Realize master arm teaching, automatic trajectory optimization filtering, safety detection and reusable motion library storage;

4. Vision\-guided autonomous operation: Combine RGB\-D depth camera spatial information and multimodal model semantic recognition to realize target positioning and automatic grasping;

5. Low\-load experimental recording: Use RK3588 native hardware encoding to record the whole experimental process, reduce CPU resource occupation and facilitate experimental data archiving;

6. Multi\-dimensional safety protection: Equipped with trajectory speed limit, joint angle limit, dry\-run simulation pre\-verification, hardware emergency stop and other multi\-layer safety mechanisms to avoid mechanical collision and equipment damage\.

# 2\. Complete Hardware \& Software Technical Stack

## 2\.1 Complete Hardware Configuration List

|Module Category|Recommended Hardware Model|Function Description|
|---|---|---|
|Edge Computing Mainboard|RK3588 Development Board \(8GB/16GB RAM recommended\)|NPU acceleration for VLM, voice model, video encoding; system operation and task scheduling|
|Execution Manipulator|SO\-ARM101 6\-DOF Robotic Arm|Complete six\-joint rotation movement, equipped with clamping gripper for grasping and placing|
|Drive Servo|Feetech STS3215 Serial Bus Servo|Control each joint rotation, support real\-time feedback of pulse value, joint angle and operating current|
|Perception Camera|Intel RealSense D435i RGB\-D Depth Camera|Synchronously output RGB color image \+ depth spatial data, realize target 3D coordinate positioning|
|Audio Input Device|USB Microphone|Collect voice commands for offline ASR recognition and keyword wake\-up|
|Audio Output Device|Speaker / USB Audio Adapter|Play voice feedback synthesized by TTS module|
|Communication Accessories|USB / TTL Serial Adapter|Realize serial communication between mainboard and servo bus of robotic arm|

## 2\.2 Layered Software Architecture \& Technical Details

The software adopts layered decoupling design, and each layer relies on independent open\-source technical frameworks to realize modular expansion and convenient secondary development:

1. Bottom System Layer

    - Operating System: Custom Linux system adapted for RK3588 edge mainboard

    - Main Development Language: Python 3\.10, compatible with all core algorithm modules

2. Offline Voice Interaction Layer

    - Core Framework: sherpa\-onnx

    - Included Functions: KWS offline keyword wake\-up, ASR offline speech recognition, TTS offline speech synthesis; all models run locally without network

3. Visual Perception \& Multimodal Reasoning Layer

    - Image Processing: OpenCV for image preprocessing, pyrealsense2 for depth camera data reading

    - Multimodal Model: Lightweight Qwen series VLM visual\-language model

    - Hardware Acceleration: RKNN / RKLLM toolchain, deploy VLM on RK3588 NPU to accelerate reasoning speed

4. Robotic Arm Motion Control Layer

    - Control Interface: LeRobot\-style standardized motion control interface

    - Communication Protocol: Feetech serial bus servo dedicated communication protocol

    - Motion Mode: Cartesian space coordinate control, direct joint angle writing, gripper switch control

5. Hardware Video Recording Layer

    - Encoding Tool: FFmpeg with Rockchip MPP hardware acceleration

    - Hardware Unit: h264\_rkmpp hardware encoder, RGA image scaling unit, OSD text overlay function

    - Advantage: All encoding and image processing tasks are offloaded to chip hardware, CPU occupancy rate remains below 10% during recording

6. Task Scheduling Layer

    - Core Mechanism: FIFO first\-in\-first\-out command queue \+ hardware interrupt response

    - Function: Unified scheduling of visual reasoning tasks, motion playback tasks, manual control tasks to avoid command conflict and motion disorder

# 3\. Detailed Introduction of Five Core Functional Modules

## 3\.1 Offline Edge Multimodal Human\-Computer Interaction Module

This module is the human\-computer interaction entrance of the whole system, realizing completely offline natural interaction without cloud support:

1. Dual input mode support: Voice command input collected by microphone, text command input manually entered by users;

2. Full offline voice workflow: Support fixed keyword wake\-up to activate the system, single\-round/long\-term continuous voice recognition, voice feedback of execution results;

3. Local VLM visual reasoning: After receiving visual query instructions, the system calls the depth camera to capture real\-time images, and the multimodal model completes local reasoning to realize visual question answering, target object classification, scene content description;

4. No network restriction: All voice and vision model reasoning calculation is completed on the local RK3588 NPU, which can work normally in offline laboratory environments without WiFi or Ethernet\.

## 3\.2 6\-DOF Robotic Arm Precise Motion Control Module

Realize full\-dimensional underlying control of SO\-ARM101 robotic arm and Feetech serial servos:

1. Dual motion control modes:

    - Cartesian space control: Input X/Y/Z spatial coordinates and rotation angles to realize automatic trajectory planning of end effector;

    - Direct joint control: Manually assign angle values to each of the six joints to realize fixed\-angle positioning;

2. Gripper control function: Independent opening and closing command interface to realize clamping and releasing of target objects;

3. Hardware emergency stop interface: Reserve external emergency stop signal access, cut off servo motion output in real time after receiving the signal;

4. Real\-time status feedback: Continuously read servo pulse values, convert pulse data into actual joint angles, and feed back real\-time operating current of each joint for abnormal detection\.

## 3\.3 Master\-Slave Teaching \& Motion Library Management Module \(Core Imitation Learning Module\)

This module is the core imitation learning function of the project, realizing the whole process from manual teaching to reusable motion storage:

1. Master\-slave motion mapping: The user manually operates the master demonstration arm, the system collects real\-time joint state data, and synchronously maps the motion track to the slave execution arm;

2. Standard trajectory automatic processing pipeline \(automatic one\-click execution\):
Step 1: Duplicate frame removal – eliminate redundant repeated sampling frames during teaching to reduce data volume;
Step 2: Median filtering smoothing – eliminate jitter and mutation points caused by manual hand shaking;
Step 3: Velocity limiting processing – limit the maximum movement speed of each joint to avoid sudden acceleration and collision;
Step 4: Multi\-dimensional safety verification – judge whether the joint angle exceeds the limit of the mechanical arm workspace;
Step 5: Manual review – users preview the optimized track in simulation mode to confirm no abnormal movement;

3. Motion library full life cycle management: All verified stable motion tracks are stored as JSON files in the motion library folder; support track list viewing, single track parameter inspection, redundant track deletion, batch smoothing optimization;

4. Flexible playback trigger: Stored motion tracks can be called and played through voice instructions or text input; support loop playback, pause/resume playback, adjustable playback frame rate\. Optimized motion templates can be directly called by the VLM visual grasping task module\.

## 3\.4 RGB\-D Depth Vision Autonomous Grasping Experimental Module

It is an experimental intelligent operation module combining multimodal vision and robotic arm motion, specially used for object grasping and placement teaching experiments:

1. Fusion of depth and semantic information: RealSense D435i outputs RGB images and depth values, the VLM model identifies the target object specified by the user, and combines depth data to calculate the three\-dimensional spatial coordinates of the target relative to the base of the robotic arm;

2. Dry\-run simulation safety mode: Before physical execution, users can enable the dry\-run parameter; the system only outputs planned motion trajectory data without driving the actual robotic arm, which is convenient for verifying whether the grasping coordinate logic is correct and avoiding collision risks;

3. Standard grasping template: Pre\-set fixed teaching motion tracks of grasping and placing; after the vision module outputs the target offset coordinate, the system automatically corrects the track and completes the grasping and placing action;

4. Suitable experimental scenarios: Desktop small objects such as cups, blocks, boxes classification and transfer experiments\.

## 3\.5 RK Hardware\-Accelerated Full\-Process Experiment Recording Module

Specially developed for teaching experiment archiving and competition video recording, making full use of RK3588 chip hardware acceleration resources:

1. Hardware encoding offloading: Adopt FFmpeg \+ h264\_rkmpp Rockchip dedicated hardware encoder, video encoding calculation does not occupy CPU resources;

2. RGA hardware image processing: Support hardware scaling of camera image resolution, add custom OSD text watermark on the picture \(such as experiment name, time, task number\);

3. Custom recording parameters: Users can set recording duration at will, and all recorded videos are automatically saved to the unified recordings folder;

4. Multi\-scene application: Used for experimental process review, algorithm debugging analysis, classroom teaching video production, science and technology competition display video output\.

# 4\. Complete System End\-to\-End Operation Workflow

1. Instruction input and intent recognition
Users input instructions through voice or text; the system analyzes the instruction content to identify the task type, and divides it into three independent task branches for processing\.

2. Branch 1: Visual question and reasoning task
Camera captures real\-time RGB\-D images → local VLM multimodal model NPU reasoning → output scene description, target position, object attribute information → voice module feeds back the result to the user\.

3. Branch 2: Motion track playback task
Read the specified motion file from the motion library → automatic multi\-dimensional safety check of track data → push the verified motion command into the global FIFO command queue\.

4. Branch 3: Manual robotic arm control task
Directly generate joint angle or Cartesian coordinate motion commands according to user input, and send them to the global command queue\.

5. Unified motion execution scheduling
The task scheduler reads commands from the queue in order, sends motion parameters to the robotic arm controller, and drives Feetech serial servos to control the six joints and gripper of SO\-ARM101 to complete physical movement\.

6. Synchronous experimental video recording
During the whole task execution process, the system synchronously collects camera image streams, uses RK hardware encoder for real\-time encoding, and saves the complete experimental video to the local folder after the task ends\.

# 5\. Detailed Repository Code Structure \& Core File Function Description

The project adopts a standardized modular layered code structure, and the warehouse adds engineering test folders and interactive menu files on the basis of the basic functional modules\. The complete directory structure and core file functions are as follows:

```Plain Text
RK3588-Based-Edge-Side-Embodied-Intelligent-Actuator/
├── main.py                         # Main entry of multimodal vision-motion integration (VLA) program
├── va.py                           # Voice assistant independent entry, realize voice listening, dialogue and visual QA
├── menu.py                         # Interactive text menu, convenient for beginners to quickly select functions without memorizing command lines
├── requirements.txt                # Python dependency list, including all vision, voice, control and encoding libraries
├── .gitignore                      # Git ignore file, exclude large model files, video cache, device logs
├── config/                         # Global system configuration folder
│   ├── settings.py                 # Basic hardware parameters: serial port, camera index, model path, memory allocation
│   ├── safety.py                   # Safety threshold configuration: joint limit, maximum motion speed, emergency stop parameters
│   └── teaching.py                # Teaching experiment configuration: default grasping track, camera hand-eye calibration parameters
├── vla/                            # Core vision-motion integration business module
│   ├── command_queue.py            # Global FIFO command queue + interrupt processing logic, resolve task conflict
│   ├── control/                    # Robotic arm underlying control code, servo communication and motion planning
│   └── vlm/                        # Multimodal model loading, preprocessing and local reasoning encapsulation interface
├── voice_assistant/                # Full offline voice module: KWS wake-up, ASR recognition, TTS synthesis
├── camera/                         # RealSense depth camera data reading, image correction, depth coordinate calculation code
├── lerobot/                        # LeRobot standardized motion control adaptation layer, compatible with mainstream robotic arm control logic
├── scripts/                        # Independent executable tool scripts for all experimental functions
│   ├── develop_motion.py           # One-click master-slave teaching track development full workflow script
│   ├── record_trajectory.py        # Motion library management: record, list, view details, delete track files
│   ├── smooth_trajectory.py        # Batch automatic filtering, smoothing and safety detection of all tracks in the library
│   ├── replay_traj.py              # Independent track playback script, support serial port parameter customization
│   ├── vlm_grasp.py                # VLM vision-guided automatic grasping experimental script, with dry-run simulation parameter
│   ├── recorder.py                 # Hardware accelerated video recording script, support custom duration and OSD text
│   └── task_controller.py          # Multi-step composite task controller, realize continuous action combination
├── motion_library/                 # Persistent storage directory of optimized and verified motion track JSON files
├── task_library/                   # Composite multi-step task template storage, such as "grasp + move + place" combined actions
├── recordings/                     # Automatically store all hardware-encoded experiment videos
├── models/so101_urdf/              # SO-ARM101 robotic arm URDF simulation model, used for track preview and simulation
├── tests/                          # Unit test script folder: serial communication test, camera test, model reasoning test, servo test
└── docs/                           # Project official documentation: deployment tutorial, parameter calibration guide, experimental teaching cases
```

# 6\. Step\-by\-Step Complete Deployment \& Quick Operation Guide

## 6\.1 Warehouse Code Acquisition

Execute the following command in the RK3588 Linux terminal to clone the source code repository:

```bash
git clone https://github.com/zizhennini/RK3588-Based-Edge-Side-Embodied-Intelligent-Actuator.git
cd RK3588-Based-Edge-Side-Embodied-Intelligent-Actuator
```

## 6\.2 Python Independent Environment Construction

Use Conda to build an isolated Python 3\.10 operating environment to avoid dependency conflicts:

```bash
conda create -n rk3588-eia python=3.10 -y
conda activate rk3588-eia
pip install -r requirements.txt
```

## 6\.3 Hardware Device Detection \& Configuration Modification

1. Detect serial port of robotic arm servo and camera device number:

```bash
ls /dev/ttyACM*
ls /dev/video*
```

2. Modify hardware parameters in `config/settings.py` according to actual detection results:

```python
SERIAL_PORT = "/dev/ttyACM0"       # Robotic arm serial port
CAMERA_INDEX = 21                   # RGB-D camera device index
VLM_MODEL_PATH = "./models/Qwen3.5-0.8B"  # Local multimodal model storage path
```

## 6\.4 RK Hardware Encoding Environment Inspection

Verify whether the RK dedicated hardware encoder and RGA image acceleration filter are available:

```bash
ffmpeg -encoders | grep h264_rkmpp
ffmpeg -filters | grep rkrga
```

If the command cannot query the corresponding hardware acceleration module, it is necessary to recompile FFmpeg adapted to Rockchip MPP and RGA hardware on the RK3588 mainboard\.

## 6\.5 Detailed Operation Guide of All Core Functions

### 6\.5\.1 Voice Assistant Multimodal Interaction

1. Long\-term continuous voice listening mode \(support keyword wake\-up\):

```bash
python3 va.py listen-forever
```

2. Single round voice recognition \(automatically stop after 4 seconds of recording\):

```bash
python3 va.py once --seconds 4
```

3. Direct text instruction interaction without voice input:

```bash
python3 va.py ask "Introduce your functional modules"
```

4. Visual question and answer \(only output text results without voice playback\):

```bash
python3 va.py ask "What objects are in the camera frame?" --no-speak
```

### 6\.5\.2 Motion Library Track Management

1. View all stored motion track files in the library:

```bash
python3 scripts/record_trajectory.py list
```

2. View detailed parameter information of a single track \(joint angle, sampling frame rate, duration\):

```bash
python3 scripts/record_trajectory.py inspect motion_library/greeting_01.json
```

3. Batch smooth and safety verify all tracks in the motion library:

```bash
python3 scripts/smooth_trajectory.py
```

4. Play the specified motion track, support custom serial port parameters:

```bash
python3 scripts/replay_traj.py motion_library/greeting_01.json --port /dev/ttyACM0
```

### 6\.5\.3 Master\-Slave Teaching One\-Click Full Process

Run the integrated teaching script to complete the whole process of teaching collection, automatic optimization, safety detection and storage to the motion library with one click:

```bash
python3 scripts/develop_motion.py
```

Internal automatic execution flow: Master\-slave teaching collection → trajectory recording → duplicate frame removal → median filtering → speed limit → safety check → manual preview → save to motion library → support voice trigger playback\.

### 6\.5\.4 VLM Vision Autonomous Grasping Experiment

1. Dry\-run simulation test \(only calculate coordinates, no robotic arm movement, safety pre\-verification\):

```bash
python3 scripts/vlm_grasp.py "red cup" \
  --teach-trajectory grasp_01.json \
  --ref-cx 320 \
  --ref-cy 240 \
  --ref-z 0.3 \
  --dry-run
```

2. After confirming that the coordinate output is normal and there is no collision risk, remove the `--dry-run` parameter to execute the real grasping action of the robotic arm\.

### 6\.5\.5 Hardware Accelerated Experiment Video Recording

1. Basic recording function \(record 5 seconds of experimental video, automatically save to recordings folder\):

```bash
python3 scripts/recorder.py --duration 5
```

2. Recording with custom OSD text watermark displayed on the screen:

```bash
python3 scripts/recorder.py --duration 5 --osd "RK3588 Embodied Intelligence Experiment Record"
```

# 7\. Multi\-Layer System Safety Protection Mechanism

The robotic arm belongs to mechanical moving equipment, which is easy to cause collision, equipment damage and safety accidents if the parameters are abnormal\. The project designs multi\-dimensional safety protection rules, and all operation specifications must be strictly followed:

1. Simulation pre\-verification rule: All newly recorded tracks and visual grasping tasks must be tested in dry\-run simulation mode first, and physical movement can only be executed after confirming no abnormality;

2. Joint parameter pre\-inspection: Before sending servo motion commands, the system automatically checks joint rotation direction, joint limit range, servo ID mapping relationship to avoid over\-limit collision;

3. Motion speed limit: The track playback module limits the maximum movement speed of each joint, and the first verification of new tracks must use the lowest speed parameter;

4. Hardware emergency stop support: Reserve external emergency stop signal interface, which can cut off all motion output in real time in case of abnormal conditions;

5. Supervision specification for autonomous grasping: The vision autonomous grasping function cannot run without completing camera hand\-eye calibration, and manual supervision must be kept on site during operation to prevent unexpected collision\.

# 8\. Project Development Completion Status \& Long\-Term Iteration Roadmap

## 8\.1 Current Function Completion Status

|Function Module|Development Status|Supplementary Description|
|---|---|---|
|Offline KWS wake\-up, ASR, TTS voice full stack|Fully Supported|All voice functions run offline stably|
|Voice \& text dual\-mode human\-computer interaction|Fully Supported|Free switching of two input modes|
|Local VLM visual reasoning \& scene QA|Fully Supported|Need to configure the correct local model path|
|Master\-slave arm teaching demonstration function|Fully Supported|Complete track automatic processing pipeline|
|Track smoothing filtering and playback function|Fully Supported|Support loop, pause, batch processing|
|Motion library full life cycle management|Fully Supported|Add, delete, view, optimize tracks|
|VLM vision\-guided automatic grasping|Experimental Version|Stable basic functions, to be optimized for complex scenes|
|RK hardware accelerated video recording|Fully Supported|Normal operation under standard RK3588 system|
|RGA hardware scaling and OSD text overlay|Environment\-Dependent|Subject to FFmpeg compilation environment of the mainboard|
|General full\-autonomous open\-ended manipulation|Under Development|Follow\-up iteration optimization|

## 8\.2 Long\-Term Project Iteration Roadmap

1. Optimize VLM model output parsing logic, improve the stability of target coordinate extraction under complex scenes, and expand multi\-type grasping task templates;

2. Develop standardized one\-click camera\-robotic arm hand\-eye calibration workflow to reduce manual calibration difficulty for students;

3. Add dual safety monitoring logic based on depth image obstacle detection and servo operating current abnormality judgment;

4. Develop web visualization dashboard, realize remote real\-time preview of camera screen, track playback control and task management for classroom demonstration;

5. Expand hardware compatibility, support multiple types of six\-degree\-of\-freedom robotic arms and different end\-effector grippers;

6. Add imitation learning dataset export function, which can export teaching track data into standard training format for follow\-up algorithm research of students and researchers\.

# 9\. Project Educational \& Scientific Research Value \& Open Source License Statement

## 9\.1 Educational Application Value

The project covers all core knowledge points of embedded edge AI and robotics courses, and provides standardized repeatable experimental cases for university teaching:

1. Edge computing large model deployment: RKNN/RKLLM multimodal model NPU offline deployment practice;

2. Multimodal human\-computer interaction: Offline voice \+ vision fusion interaction development;

3. Robotic arm kinematics: Cartesian coordinate and joint angle dual motion control practice;

4. Imitation learning engineering realization: Master\-slave teaching, track filtering optimization and reusable motion library design;

5. Embodied intelligence complete pipeline: End\-to\-end closed\-loop system construction from perception to execution;

6. RK3588 hardware acceleration development: Hardware video encoding, RGA image processing practical operation\.

## 9\.2 Scientific Research Value

Provide a lightweight, fully offline local experimental platform for embodied intelligence related algorithm research:

1. Rapid verification of VLM\-driven robotic arm manipulation algorithms without cloud computing resources;

2. Research of offline multimodal human\-robot interaction under network\-free environment;

3. Comparative experiment of different trajectory smoothing and filtering algorithms based on teaching imitation data;

4. Research on depth vision target positioning and autonomous grasping strategy of desktop small objects\.

## 9\.3 Open Source License \& Acknowledgement

1. Open Source License Description: There is no built\-in LICENSE file in the original warehouse\. If users need to carry out secondary development, modification and public distribution, they must add a standard open\-source license file \(MIT/Apache 2\.0 etc\.\) by themselves\.

2. Open Source Community Acknowledgement: The project draws technical inspiration and relies on multiple open\-source ecological projects, including Rockchip RK3588 official software ecosystem, LeRobot robotic arm control framework, sherpa\-onnx offline speech toolkit, FFmpeg Rockchip hardware acceleration module, Intel RealSense depth camera open\-source SDK\.

