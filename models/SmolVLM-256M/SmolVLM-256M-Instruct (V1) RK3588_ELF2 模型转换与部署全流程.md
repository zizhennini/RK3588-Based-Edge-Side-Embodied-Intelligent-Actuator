# SmolVLM\-256M\-Instruct \(V1\) RK3588/ELF2 模型转换与部署全流程

# SmolVLM\-256M\-Instruct \(V1\) RK3588/ELF2 模型转换 \& 部署全流程

## 一、项目基础信息

### 1\. 适配对象

- **目标模型**：`SmolVLM-256M-Instruct`（V1 纯图像版，基于 SigLIP 视觉编码器 \+ Idefics3 语言底座）

- **硬件平台**：RK3588 / ELF2 开发板

- **工具链**：瑞芯微 `rknn-llm` 官方仓库、`rknn-toolkit2`、`rkllm-toolkit`

- **核心难点**：SigLIP 原生位置编码 `ScatterND + Gather` 浮点索引与 RKNN 算子约束冲突、Protobuf 版本不兼容、模型属性层级错误。

### 2\. 全局目录结构

```Plain Text
/home/elf/work/rknn-llm/examples/SmolVLM-256M/
├── SmolVLM-256M/          # 模型权重目录
├── data/
│   └── datasets.json      # RKLLM量化校准数据集
├── export/                 # 模型转换脚本目录（核心）
│   ├── export_vision.py    # 视觉分支 → ONNX
│   ├── export_vision_rknn.py # ONNX → RKNN
│   └── export_rkllm.py     # 语言分支 → RKLLM
└── deploy/                 # C++推理程序编译目录
```

### 3\. 前置环境准备

#### 3\.1 激活虚拟环境

```bash
conda activate rknn
```

#### 3\.2 关键依赖版本（必守）

|依赖项|推荐版本|说明|
|---|---|---|
|Python|3\.10|全流程适配版本|
|protobuf|3\.20\.3|规避新版解析冲突|
|transformers|4\.38 \~ 4\.41|正常加载 SmolVLM 模型类|
|rknn\-toolkit2|2\.3\.2|当前使用版本（多模态建议降级 1\.5\.0 兜底）|

#### 3\.3 临时环境变量（解决 Protobuf 解析警告）

终端全局生效，每次执行转换前可直接使用：

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

---

## 二、视觉模型转换（SigLIP → ONNX → RKNN）

### 1\. 历史报错复盘（已全部修复）

1. `NameError: name 'image' is not defined`：`forward` 返回变量名拼写错误

2. `AttributeError: 'Idefics3Model' object has no attribute 'model'`：模型嵌套层级错误，多余 `.model`

3. `Gather 算子输入为 float 非法类型`：SigLIP 原生 `ScatterND` 输出浮点索引，RKNN 要求 Gather 索引必须为 `int64`

4. TracerWarning / CUDA 库警告：可直接忽略，不影响功能

5. 常量折叠篡改张量类型：**强制关闭 ****`do_constant_folding=False`**

### 2\. 脚本 1：`export_vision.py`（视觉分支导出 ONNX，最终稳定版）

> 核心改造：手动拆解视觉前向、预生成静态 `int64` 位置 ID，**彻底移除 ScatterND 算子**，从根源解决类型报错。
> 
> 

```python
import torch
import numpy as np
import os
import argparse
# 导入SmolVLM专属模型类
from transformers import SmolVLMForConditionalGeneration

class smolvlm_vision(torch.nn.Module):
    def __init__(self, vlm):
        super().__init__()
        # 修复：直接访问顶层属性，删除多余 .model 嵌套
        self.vpm = vlm.vision_model
        self.connector = vlm.connector
        
        # 固定 512*512 分辨率 + patch_size=16，预生成静态int64位置ID
        num_patches = (512 // 16) ** 2
        pos_ids = torch.arange(num_patches, dtype=torch.int64)
        self.register_buffer("pos_ids", pos_ids)

    def forward(self, pixel_values):
        # 手动拆解视觉前向逻辑，剥离动态位置编码算子
        hidden = self.vpm.embeddings.patch_embedding(pixel_values)
        hidden = hidden.flatten(2).transpose(1, 2)
        
        # 静态位置编码（全程int64，规避浮点索引）
        pos = self.vpm.embeddings.position_embedding(self.pos_ids)
        hidden = hidden + pos.unsqueeze(0)
        
        # 编码器 + 后处理
        hidden = self.vpm.encoder(inputs_embeds=hidden).last_hidden_state
        hidden = self.vpm.post_layernorm(hidden)
        
        # 模态维度投影 768 -> 512
        hidden = self.connector(hidden)
        return hidden

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, 
        default="/home/elf/work/rknn-llm/examples/SmolVLM-256M/SmolVLM-256M",
        help="SmolVLM 权重绝对路径")
    parser.add_argument('--model_name', type=str, default='smolvlm')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    args = parser.parse_args()

    # 加载模型：关闭FlashAttention，避免额外算子
    model = SmolVLMForConditionalGeneration.from_pretrained(
        args.path,
        torch_dtype=torch.float32,
        _attn_implementation="eager",
        trust_remote_code=True
    ).eval()

    # 封装自定义视觉分支
    model = smolvlm_vision(model).eval()

    # 固定静态输入 [1, 3, 512, 512]（RKNN禁止动态分辨率）
    dummy = torch.randn(1, 3, args.height, args.width)

    os.makedirs("onnx", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        f"./onnx/{args.model_name}_vision.onnx",
        input_names=["pixel"],
        opset_version=15,          # 固定兼容版本，高opset易触发类型异常
        do_constant_folding=False  # 核心：关闭常量折叠，防止篡改张量类型
    )
    print("✅ ONNX export success")
```

### 3\. 脚本 2：`export_vision_rknn.py`（ONNX 转 RKNN，最终稳定版）

> 核心配置：RGB 三通道归一化、视觉分支**禁止量化（FP16）**、静态输入尺寸。
> 
> 

```python
from rknn.api import RKNN
import numpy as np
import os
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, default='./onnx/smolvlm_vision.onnx', help='ONNX path')
argparse.add_argument('--model_name', type=str, default='smolvlm', help='Model name')
argparse.add_argument('--target-platform', type=str, default='rk3588', help='Target chip')
argparse.add_argument('--batch_size', type=int, default=1, help='Batch size')
argparse.add_argument('--height', type=int, default=512, help='Image height')
argparse.add_argument('--width', type=int, default=512, help='Image width')
args = argparse.parse_args()

model_path = args.path
modelname = args.model_name
target_platform = args.target_platform

# 标准归一化：RGB三通道（修复早期单通道bug）
mean_value = [[0.5 * 255, 0.5 * 255, 0.5 * 255]]
std_value  = [[0.5 * 255, 0.5 * 255, 0.5 * 255]]

# 静态输入配置
inputs = ['pixel']
input_size_list = [[args.batch_size, 3, args.height, args.width]]
input_initial_val = None
op_target = None
disable_rules = []

# 初始化RKNN
rknn = RKNN(verbose=False)
rknn.config(
    target_platform=target_platform,
    mean_values=mean_value,
    std_values=std_value,
    op_target=op_target,
    disable_rules=disable_rules
)
rknn.load_onnx(model_path, inputs=inputs, input_size_list=input_size_list)
# 视觉分支：官方要求 FP16 不量化
rknn.build(do_quantization=False, dataset=None)

os.makedirs("rknn", exist_ok=True)
out_rknn = f"./rknn/{modelname}_vision_{target_platform}.rknn"
rknn.export_rknn(out_rknn)
print(f"✅ RKNN export success: {out_rknn}")
rknn.release()
```

### 4\. 执行视觉模型转换命令

```bash
# 进入转换目录
cd /home/elf/work/rknn-llm/examples/SmolVLM-256M/export

# 清理旧缓存文件（必做）
rm -rf ./onnx ./rknn

# 1. 导出 ONNX
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python export_vision.py

# 2. ONNX 转 RKNN
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python export_vision_rknn.py
```

**成功产物**：`./rknn/smolvlm_vision_rk3588.rknn`

---

## 三、语言模型转换（LLM → RKLLM）

### 1\. 量化校准数据集 `data/datasets.json`

RKLLM 量化必需校准文件，格式完全符合官方规范：

```json
[
    {"image_path": "data/datasets", "image": "1.jpg", "input": "Question: What is correct Python code to generate the content of the image?\nOptions:\nA. for x in range(6):\n  print(x)\nelse:\n  print(\"Finally finished!\")\n\nB. thisdict = {\n  \"brand\": \"Ford\",\n  \"model\": \"Mustang\",\n  \"year\": 1964\n}\n\nprint(len(thisdict))\nC. x = 1\ny = 2.8\nz = 1j\n\nprint(type(x))\nprint(type(y))\nprint(type(z))\n\nD. fruits = [\"apple\", \"banana\", \"cherry\"]\nfor x in fruits:\n  print(x)\nPlease select the correct answer from the options above. \n", "target":"D"},
    {"image_path": "data/datasets", "image": "2.jpg", "input": "Question: What is correct Python code to generate the content of the image?\nOptions:\nA. class Person:\n  def __init__(self, name, age):\n    self.name = name\n    self.age = age\n\np1 = Person(\"John\", 36)\n\nprint(p1.name)\nprint(p1.age)\nB. fruits = [\"apple\", \"banana\", \"cherry\"]\nfor x in fruits:\n  print(x)\nC. x = min(5, 10, 25)\ny = max(5, 10, 25)\n\nprint(x)\nprint(y)\nD. a = 33\nb = 200\nif b > a:\n  print(\"b is greater than a\")\nPlease select the correct answer from the options above. \n", "target":"D"},
    {"image_path": "data/datasets", "image": "21.jpg", "input": "Question: Which one is the correct caption of this image?\nOptions:\nA. A man rides a surfboard on a large wave.\nB. a young boy barefoot holding an umbrella touching the horn of a cow\nC. A giraffe standing by a stall in a field.\nD. A stop sign that has been vandalized with graffiti.\nPlease select the correct answer from the options above. \n", "target":"B"},
    {"image_path": "data/datasets", "image": "22.jpg", "input": "Question: Which one is the correct caption of this image?\nOptions:\nA. A narrow kitchen filled with appliances and cooking utensils.\nB. A person with glasses and a tie in a room.\nC. Tray of vegetables with cucumber, carrots, broccoli and celery.\nD. A pretty young woman riding a surfboard on a wave in the ocean.\nPlease select the correct answer from the options above. \n", "target":"A"},
    {"image_path": "data/datasets", "image": "241.jpg", "input": "Hint: The passage below describes an experiment. Read the passage and then follow the instructions below.\n\nMadelyn applied a thin layer of wax to the underside of her snowboard and rode the board straight down a hill. Then, she removed the wax and rode the snowboard straight down the hill again. She repeated the rides four more times, alternating whether she rode with a thin layer of wax on the board or not. Her friend Tucker timed each ride. Madelyn and Tucker calculated the average time it took to slide straight down the hill on the snowboard with wax compared to the average time on the snowboard without wax.\nFigure: snowboarding down a hill.\nQuestion: Identify the question that Madelyn and Tucker's experiment can best answer.\nOptions:\nA. Does Madelyn's snowboard slide down a hill in less time when it has a thin layer of wax or a thick layer of wax?\nB. Does Madelyn's snowboard slide down a hill in less time when it has a layer of wax or when it does not have a layer of wax?\nPlease select the correct answer from the options above. \n", "target":"B"},
    {"image_path": "data/datasets", "image": "252.jpg", "input": "Hint: People can use the engineering-design process to develop solutions to problems. One step in the process is testing if a potential solution meets the requirements of the design.\nThe passage below describes how the engineering-design process was used to test a solution to a problem. Read the passage. Then answer the question below.\n\nLaura and Isabella were making batches of concrete for a construction project. To make the concrete, they mixed together dry cement powder, gravel, and water. Then, they checked if each batch was firm enough using a test called a slump test.\nThey poured some of the fresh concrete into an upside-down metal cone. They left the concrete in the metal cone for 30 seconds. Then, they lifted the cone to see if the concrete stayed in a cone shape or if it collapsed. If the concrete in a batch collapsed, they would know the batch should not be used.\nFigure: preparing a concrete slump test.\nQuestion: Which of the following could Laura and Isabella's test show?\nOptions:\nA. if the concrete from each batch took the same amount of time to dry\nB. if a new batch of concrete was firm enough to use\nPlease select the correct answer from the options above. \n", "target":"B"},
    {"image_path": "data/datasets", "image": "362.jpg", "input": "Hint: Native copper has the following properties:\nsolid\nnot made by living things\nfound in nature\nfixed crystal structure\nmade of the metal copper\nQuestion: Is native copper a mineral?\nOptions:\nA. no\nB. yes\nPlease select the correct answer from the options above. \n", "target":"B"},
    {"image_path": "data/datasets", "image": "364.jpg", "input": "Hint: Plastic has the following properties:\nsolid\nno fixed crystal structure\nnot a pure substance\nmade in a factory\nQuestion: Is plastic a mineral?\nOptions:\nA. yes\nB. no\nPlease select the correct answer from the options above. \n", "target":"B"},
    {"image_path": "data/datasets", "image": "448.jpg", "input": "Hint: Read the text.\nButterflies and moths are easily mistaken for each other, but one distinction between them often appears during their pupal stage. When most butterfly caterpillars reach full size, they attach themselves to a leaf or other object and shed their skin a final time, forming a chrysalis, a hard, shell-like skin, which protects the pupa inside. The chrysalis may be dull and rough or shiny and smooth, usually blending into its surroundings. Most moth caterpillars, by contrast, create a cocoon to protect the pupa, rather than forming a chrysalis. The cocoons usually resemble hard silk pouches, but some moths also incorporate materials like hairs and twigs.\nQuestion: Which term matches the picture?\nOptions:\nA. cocoon\nB. chrysalis\nPlease select the correct answer from the options above. \n", "target":"B"},
    {"image_path": "data/datasets", "image": "477.jpg", "input": "Hint: Read the text.\nHeat transfer can occur in different ways. Two common ways are through conduction and convection. Conduction occurs when molecules from one object collide with molecules from another object. Burning your hand by touching a hot car door on a sunny summer day is an example of conduction.\nConvection is another form of heat transfer. When a liquid or gas is heated, the heated matter rises upward, away from the heat source. Hot bubbles rising in a pot of water boiling on a stove is an example of convection.\nQuestion: Which term matches the picture?\nOptions:\nA. conduction\nB. convection\nPlease select the correct answer from the options above. \n", "target":"B"},
    {"image_path": "data/datasets", "image": "1231.jpg", "input": "Question: Which image is more brightful?\nOptions:\nA. The first image\nB. The second image\nPlease select the correct answer from the options above. \n", "target":"A"},
    {"image_path": "data/datasets", "image": "1232.jpg", "input": "Question: Which image is more brightful?\nOptions:\nA. The first image\nB. The second image\nPlease select the correct answer from the options above. \n", "target":"A"},
    {"image_path": "data/datasets", "image": "1085.jpg", "input": "Question: is this place crowded?\nOptions:\nA. yes\nB. no\nPlease select the correct answer from the options above. \n", "target":"A"},
    {"image_path": "data/datasets", "image": "1086.jpg", "input": "Question: is this place crowded?\nOptions:\nA. yes\nB. no\nPlease select the correct answer from the options above. \n", "target":"A"},
    {"image_path": "data/datasets", "image": "1128.jpg", "input": "Question: In this picture, are the two dolphins the same size?\nOptions:\nA. same\nB. Not the same\nC. Can't judge\nPlease select the correct answer from the options above. \n", "target":"B"},
    {"image_path": "data/datasets", "image": "1129.jpg", "input": "Question: In this picture, are the two butterfly wings the same shape?\nOptions:\nA. same\nB. Not the same\nC. Can't judge\nPlease select the correct answer from the options above. \n", "target":"B"},
    {"image_path": "data/datasets", "image": "1200.jpg", "input": "Question: What will happen next?\nOptions:\nA. the motorcyle is gonna go forward\nB. the motorcyle is gonna crash\nC. the motorcyle is gonna go backward\nD. both A,B, and C\nPlease select the correct answer from the options above. \n", "target":"B"},
    {"image_path": "data/datasets", "image": "1201.jpg", "input": "Question: What will happen next?\nOptions:\nA. this person is gonna stay still\nB. this person is gonna keep walking\nC. this person is gonna fall into the water\nD. both A,B, and C\nPlease select the correct answer from the options above. \n", "target":"C"},
    {"image_path": "data/datasets", "image": "1554.jpg", "input": "Question: The object shown in this figure:\nOptions:\nA. Is a colorless, flammable liquid that is commonly used as a solvent and fuel\nB. Has a boiling point of 64.7°C\nC. Can be toxic if ingested or absorbed through the skin\nD. None of these options are correct.\nPlease select the correct answer from the options above. \n", "target":"C"},
    {"image_path": "data/datasets", "image": "1555.jpg", "input": "Question: The object shown in this figure:\nOptions:\nA. Is a lustrous, white metal that is highly reflective and ductile\nB. Has the highest electrical and thermal conductivity of all metals\nC. Has a boiling point of 2,162°C\nD. All of these options are correct.\nPlease select the correct answer from the options above. \n", "target":"D"}
]
```

### 2\. 脚本：`export_rkllm.py`（语言模型转 RKLLM，最终稳定版）

```python
import os
from rkllm.api import RKLLM
import argparse

argparse = argparse.ArgumentParser()
# 模型权重路径（精准对齐本地目录）
argparse.add_argument('--path', type=str, 
    default='../SmolVLM-256M', 
    help='SmolVLM 模型权重路径')
argparse.add_argument('--target-platform', type=str, default='rk3588')
argparse.add_argument('--num_npu_core', type=int, default=3)  # RK3588 推荐3核NPU
argparse.add_argument('--quantized_dtype', type=str, default='w8a8') # 默认W8A8量化
argparse.add_argument('--device', type=str, default='cpu')
args = argparse.parse_args()

modelpath = args.path
target_platform = args.target_platform
num_npu_core = args.num_npu_core
quantized_dtype = args.quantized_dtype
savepath = os.path.join("./rkllm", f"smolvlm_256m_{quantized_dtype}_{target_platform}.rkllm")

os.makedirs(os.path.dirname(savepath), exist_ok=True)
llm = RKLLM()

# 加载HuggingFace格式语言模型
ret = llm.load_huggingface(model=modelpath, device=args.device)
if ret != 0:
    print("❌ Load SmolVLM LLM failed!")
    exit(ret)

# 量化校准集路径（精准对齐本地目录）
dataset = '../data/datasets.json'
qparams = None
ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype=quantized_dtype,
    quantized_algorithm='normal',
    target_platform=target_platform,
    num_npu_core=num_npu_core,
    extra_qparams=qparams,
    dataset=dataset
)
if ret != 0:
    print("❌ Build RKLLM failed!")
    exit(ret)

# 导出RKLLM模型
ret = llm.export_rkllm(savepath)
if ret != 0:
    print("❌ Export RKLLM failed!")
    exit(ret)

print(f"✅ RKLLM export success: {savepath}")
```

### 3\. 执行语言模型转换命令

```bash
# 8GB内存板卡（推荐：W8A8 精度+速度最优）
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python export_rkllm.py

# 4GB低内存板卡（W4A16 降低显存占用）
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python export_rkllm.py --quantized_dtype w4a16
```

**成功产物**：`./rkllm/smolvlm_256m_w8a8_rk3588.rkllm`

---

## 四、交叉编译 C\+\+ 推理程序

```bash
# 进入编译目录
cd ../deploy

# 执行官方编译脚本
bash build-linux.sh
```

编译产物：`deploy/install/demo_Linux_aarch64`（含可执行文件 \+ 依赖库）

---

## 五、RK3588/ELF2 板端部署 \& 运行推理

### 1\. 板端前置配置（SSH / 串口登录开发板）

```bash
# 安装RKNN/RKLLM运行时库
sudo apt update && sudo apt install -y librknnrt librknnrt-dev

# NPU定频（必做，防止降频、提升推理速度）
wget https://raw.githubusercontent.com/airockchip/rknn-llm/main/scripts/fix_freq_rk3588.sh
chmod +x fix_freq_rk3588.sh
sudo ./fix_freq_rk3588.sh

# 创建工作目录
mkdir -p /data/smolvlm /data/smolvlm/models
```

### 2\. PC 端推送文件（ADB 方式）

```bash
# 连接板卡（替换为你的板卡IP）
adb connect 你的板卡IP:5555

# 推送视觉RKNN模型
adb push ./rknn/smolvlm_vision_rk3588.rknn /data/smolvlm/models/

# 推送语言RKLLM模型
adb push ./rkllm/smolvlm_256m_w8a8_rk3588.rkllm /data/smolvlm/models/

# 推送推理程序+依赖库
adb push ../deploy/install/demo_Linux_aarch64/* /data/smolvlm/

# 推送测试图片（自行准备 test.jpg，分辨率建议 512*512）
adb push /本地图片路径/test.jpg /data/smolvlm/
```

### 3\. 板端启动推理

```bash
adb shell
cd /data/smolvlm

# 加载动态链接库
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

# 推理命令格式：./VLM_NPU 图片 RKNN路径 RKLLM路径 生成长度 上下文长度
./VLM_NPU test.jpg \
./models/smolvlm_vision_rk3588.rknn \
./models/smolvlm_256m_w8a8_rk3588.rkllm \
512 2048
```

---

## 六、全局避坑总结（核心约束，必须遵守）

1. **视觉 ONNX 导出强制规则**

    - `do_constant_folding=False`：永久关闭常量折叠，防止篡改整型索引为浮点；

    - `opset_version=15`：禁用高版本 opset，规避张量类型推导异常；

    - 输入固定 `512×512`：**禁止动态分辨率**；

    - 位置编码：必须使用静态 `int64` 索引，规避 `ScatterND+Gather` 组合。

2. **RKNN 预处理规则**

    - 图像归一化：RGB 三通道 `mean/std = [127.5, 127.5, 127.5]`，禁止单通道；

    - 视觉分支：`do_quantization=False`（FP16 不量化），量化会严重损失精度。

3. **工具版本搭配（RK3588 多模态最优组合）**

    - 稳定兜底：`rknn-toolkit2==1.5.0` \+ `protobuf==3.20.3`；

    - 当前使用：`rknn-toolkit2==2.3.2`（需配合模型层改造规避算子问题）。

4. **路径规范**
所有脚本使用**绝对路径 / 同级相对路径**，避免层级嵌套导致模型加载失败。

---

## 七、常见报错速查表

|报错信息|根因|解决方案|
|---|---|---|
|`Gather: Type 'tensor(float)' invalid`|ScatterND 输出浮点索引，RKNN Gather 仅支持 int|重写视觉前向，预生成静态 int64 位置 ID|
|`AttributeError: 'Idefics3Model' object has no attribute 'model'`|多余 `.model` 嵌套层级|改为 `vlm.vision_model` 直接访问顶层属性|
|`NameError: name 'image' is not defined`|函数返回变量名拼写错误|统一返回 `image_hidden_states`|
|Protobuf Descriptors cannot be created directly|Protobuf 版本过高|临时环境变量 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 或降级至 3\.20\.3|
|进程被 Killed / OOM|内存不足|关闭多余程序、改用 `W4A16` 低内存量化|
|`dataset xxx.json not found`|校准集路径错误|核对 `../data/datasets.json` 路径|
|板端 `error while loading shared libraries`|动态库未加载|执行 `export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH`|

> （注：文档部分内容可能由 AI 生成）
