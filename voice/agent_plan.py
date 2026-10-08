"""多步动作编排(vlm_arm 函数清单模式 + 白名单执行)

将一条自然语言指令拆解为动作库(motion_library/index.json)中动作的先后序列。
与 vlm_arm 的差异:不 eval LLM 输出字符串,仅按动作名白名单查表执行;解析失败
自动返回空序列,由调用方回落到现有问答路径。

用法(orchestrator 内):
    from .agent_plan import plan_from_text
    actions, response = plan_from_text(self.config, text)
    if actions:
        for act in actions:
            self.motion_cb(act, actions_index.get(act, {}))
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from .qwen_runner import QwenRunner

_MOTION_INDEX = Path("motion_library/index.json")


def load_actions(index_path: str | Path = "") -> dict:
    """加载动作索引,结构与 MotionMatcher._reload 保持一致。"""
    path = Path(index_path) if index_path else _MOTION_INDEX
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            k: v for k, v in data.items()
            if isinstance(v, dict) and isinstance(v.get("keywords"), list)
        }
    except (json.JSONDecodeError, OSError):
        return {}


def build_plan_prompt(actions: dict) -> str:
    """动态生成编排系统提示词(动作清单来自 motion_library,新增动作自动纳入)。"""
    lines = []
    for name, info in actions.items():
        kws = "、".join(info.get("keywords", []) or [])
        lines.append(f"- {name}" + (f"(触发词:{kws})" if kws else ""))
    return (
        "你是机械臂动作编排器。用户的一条指令可能包含多个先后动作,"
        "请你把它拆解为下述动作的先后序列。\n"
        "【可用动作】\n" + "\n".join(lines) + "\n"
        "【输出要求】直接输出一个 JSON 对象,不要输出 markdown 代码块或其他文字:\n"
        '{"actions": ["动作1", "动作2"], "response": "给用户的简短回复"}\n'
        "actions 只能使用上面列出的动作名,按执行先后排序;"
        "与动作无关的指令,actions 输出空列表,response 正常对话回复。\n"
        "【示例】\n"
        '用户指令:先打招呼,然后抓取方块,放到右边。'
        '你输出:{"actions": ["greeting", "grasp", "reach_right", "place"], "response": "看我的"}\n'
        '用户指令:抬起手臂。'
        '你输出:{"actions": ["抬起"], "response": "抬起来了"}\n'
        '用户指令:先打招呼,然后抬起手臂。'
        '你输出:{"actions": ["greeting", "抬起"], "response": "你好,看我的动作"}'
        '(注意:打招呼和抬起手臂是两个不同的动作,必须都输出)\n'
        '用户指令:今天天气怎么样。'
        '你输出:{"actions": [], "response": "我没法看天气,但桌面上有什么我可以帮你看"}'
    )


def _literal(blob: str):
    return ast.literal_eval(blob)


def parse_plan(raw: str, valid: set[str]) -> tuple[list[str], str]:
    """解析 RKLLM 输出 -> (动作序列, 回复话术)。

    容错链:标准 JSON -> ast.literal_eval(单引号) -> 正则抽取白名单动作名。
    动作名必须存在于 valid(白名单),保序、不去重(允许"抬起-放下-抬起")。
    """
    if not raw:
        return [], ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        # JSON 彻底失败:按白名单正则兜底抽取
        hits = [a for a in valid if a in raw]
        return hits, ""
    blob = m.group(0)
    data = None
    for loader in (json.loads, _literal):
        try:
            data = loader(blob)
            break
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
    if not isinstance(data, dict):
        return [], ""

    raw_actions = data.get("actions", [])
    if isinstance(raw_actions, str):
        raw_actions = [raw_actions]
    actions: list[str] = []
    if isinstance(raw_actions, list):
        for a in raw_actions:
            if isinstance(a, str) and a.strip() in valid:
                actions.append(a.strip())

    response = data.get("response", "")
    return actions, (response if isinstance(response, str) else "")


# ═══════════════════════════════════════════════════════════════════════
# 意图级编排(原项目主线融合):LLM 语义理解 → 意图序列 → on_intent 分发
#
# 解决关键词路由的语言多样性 bug:同一动作有无数种说法,枚举不完,
# 由 LLM 语义理解映射到结构化意图(带参数),经 on_intent 回调走原项目主线
# (grasp → GraspPipeline.execute_grasp 视觉定位抓取)。
# ═══════════════════════════════════════════════════════════════════════

_VALID_INTENTS = ("grasp", "place", "home", "stop", "ask", "play_motion")


def build_route_prompt(motion_names: list[str]) -> str:
    """生成意图编排提示词(动作库回放作为演示类意图纳入)。"""
    motion_list = "、".join(motion_names) if motion_names else "(无)"
    return (
        "你是机械臂助手。把用户的一条自然语言指令理解并拆解为下述意图的先后序列。"
        "同一个动作用户可能用各种说法表达,请理解语义而不是匹配字面。\n"
        "【可用意图】\n"
        '- grasp: 抓取物体。参数 target=物体描述(如"红色方块"、"糖"、'
        '"桌上那个杯子")。用户想让你拿起/抓取/捡起任何东西都属于此意图\n'
        "- place: 放下/放置手中物体。参数 target=放置位置描述(可选)\n"
        "- home: 机械臂归零/复位/回到初始位置\n"
        "- stop: 立即停止/急停\n"
        "- ask: 用户在提问或聊天(不涉及动作)。参数 question=用户的问题\n"
        f"- play_motion: 回放演示动作。参数 name 从这些动作中选:{motion_list}\n"
        "【输出要求】直接输出一个 JSON 对象,不要 markdown 代码块或其他文字:\n"
        '{"steps": [{"intent": "grasp", "target": "红色方块"}],'
        ' "response": "给用户的简短回复"}\n'
        "steps 按执行先后排序;每个元素的 intent 只能用上面的意图名;"
        "grasp/ask/play_motion 必须带参数;纯聊天时 steps 为空数组。\n"
        "【示例】\n"
        '用户指令:帮我把红色那个拿过来。'
        '你输出:{"steps": [{"intent": "grasp", "target": "红色方块"}],'
        ' "response": "好的,我去拿"}\n'
        '用户指令:先归零,然后把黄色的收了。'
        '你输出:{"steps": [{"intent": "home"},'
        ' {"intent": "grasp", "target": "黄色方块"}], "response": "马上安排"}\n'
        '用户指令:方块捡起来放盘子里。'
        '你输出:{"steps": [{"intent": "grasp", "target": "方块"},'
        ' {"intent": "place", "target": "盘子"}], "response": "看我的"}\n'
        '用户指令:打个招呼。'
        '你输出:{"steps": [{"intent": "play_motion", "name": "greeting"}],'
        ' "response": "你好呀"}\n'
        '用户指令:今天天气怎么样。'
        '你输出:{"steps": [], "response": "我没法看天气,但我可以帮你看桌面"}'
    )


def parse_route(raw: str, motion_names: set[str]) -> tuple[list[dict], str]:
    """解析 RKLLM 输出 -> (意图步骤序列, 回复话术)。

    白名单:intent 必须在 _VALID_INTENTS 内;play_motion 的 name 必须在
    motion_names 内。异常输入全部丢弃,由调用方回落关键词路由。
    """
    steps: list[dict] = []
    if not raw:
        return steps, ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return steps, ""
    blob = m.group(0)
    data = None
    for loader in (json.loads, _literal):
        try:
            data = loader(blob)
            break
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
    if not isinstance(data, dict):
        return steps, ""

    raw_steps = data.get("steps", [])
    if isinstance(raw_steps, dict):
        raw_steps = [raw_steps]
    if not isinstance(raw_steps, list):
        return steps, ""

    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        intent = str(s.get("intent", "")).strip()
        if intent not in _VALID_INTENTS:
            continue
        step: dict = {"intent": intent}
        if intent == "grasp" or intent == "ask" or intent == "place":
            step["target"] = str(s.get("target", "")).strip()
        elif intent == "play_motion":
            name = str(s.get("name", "")).strip()
            if motion_names and name not in motion_names:
                continue                     # 动作名不在库里,丢弃
            step["name"] = name
        steps.append(step)

    response = data.get("response", "")
    return steps, (response if isinstance(response, str) else "")


def route_from_text(
    config: dict,
    text: str,
    motion_names: list[str] | None = None,
) -> tuple[list[dict], str]:
    """意图编排入口:调一次 RKLLM,返回 (意图步骤, 回复话术)。

    任何异常吞掉并返回空序列,调用方回落到原关键词路由(_route)。
    """
    try:
        prompt = build_route_prompt(motion_names or [])
        placeholder = str(config["paths"].get(
            "placeholder_image",
            "models/vlm/Qwen3.5-0.8B/demo.jpg"))
        runner = QwenRunner(config)
        # Qwen demo 为 pexpect 逐行交互,多行提示词必须压成单行发送
        one_line = " ".join(prompt.split())
        raw = runner.ask(placeholder, one_line + " 用户指令:" + text.strip())
        return parse_route(raw, set(motion_names or []))
    except Exception as exc:  # noqa: BLE001
        print(f"[agent_plan] 意图编排失败,回落关键词路由: {exc}", flush=True)
        return [], ""


def plan_from_text(
    config: dict,
    text: str,
    index_path: str | Path = "",
) -> tuple[list[str], str]:
    """编排入口:调一次 RKLLM,返回 (动作序列, 回复话术)。

    任何异常都被吞掉并返回空序列,调用方据此回落到问答路径。
    """
    try:
        actions = load_actions(index_path)
        if not actions:
            return [], ""
        prompt = (
            build_plan_prompt(actions)
            + "\n【用户指令】"
            + text.strip()
        )
        placeholder = str(config["paths"].get("placeholder_image", "asset/placeholder.jpg"))
        runner = QwenRunner(config)
        # Qwen demo 通过 pexpect sendline 逐行交互:含换行的提示词会被 demo
        # 当成多条输入截断,必须压成单行再发送
        one_line = " ".join(prompt.split())
        raw = runner.ask(placeholder, one_line + " 用户指令:" + text.strip())
        return parse_plan(raw, set(actions.keys()))
    except Exception as exc:  # noqa: BLE001 —— 编排失败绝不阻塞语音主链路
        print(f"[agent_plan] 编排失败,回落问答: {exc}", flush=True)
        return [], ""
