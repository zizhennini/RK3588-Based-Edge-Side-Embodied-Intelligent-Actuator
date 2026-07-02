#!/usr/bin/env python3
"""语音触发动作回放 — 说关键词 → 回放对应轨迹"""
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MOTION_DIR = os.path.join(os.path.dirname(__file__), "..", "motion_library")
INDEX_PATH = os.path.join(MOTION_DIR, "index.json")
VA_DIR = os.path.join(os.path.dirname(__file__), "..", "voice_assistant")
STT_SCRIPT = os.path.join(VA_DIR, "voice_assistant.py")
REPLAY_SCRIPT = os.path.join(os.path.dirname(__file__), "replay_traj.py")


def load_index():
    with open(INDEX_PATH) as f:
        return json.load(f)


def find_action(index, text):
    text = text.lower()
    for action_name, info in index.items():
        for kw in info["keywords"]:
            if kw.lower() in text:
                return action_name, info
    return None, None


def replay(action_name, info):
    file_path = os.path.join(MOTION_DIR, info["file"])
    if not os.path.exists(file_path):
        print("轨迹文件不存在: {}".format(file_path))
        return
    print("回放: {} ({})".format(action_name, file_path))
    subprocess.run([
        sys.executable, REPLAY_SCRIPT, file_path, "--fps", "30", "--port", "/dev/ttyACM0"
    ])


def main():
    index = load_index()

    print("语音触发动作回放")
    print("  说 '你好' → 打招呼")
    print("  说 '抓' → 抓取")
    print("  说 '再见' → 挥手")
    print("  说 '谢谢' → 鞠躬")
    print("  q 退出\n")

    while True:
        cmd = input("按 Enter 录音，或输入文字指令: ").strip().lower()
        if cmd == "q":
            break

        if not cmd:
            subprocess.run([
                "arecord", "-D", "hw:rockchipnau8822,0",
                "-d", "3", "-f", "cd", "-t", "wav", "/tmp/voice_cmd.wav"
            ], capture_output=True)
            result = subprocess.run([
                sys.executable, STT_SCRIPT, "stt", "/tmp/voice_cmd.wav"
            ], capture_output=True, text=True)
            cmd = result.stdout.strip()

        print("  识别: [{}]".format(cmd))
        action_name, info = find_action(index, cmd)
        if action_name:
            replay(action_name, info)
        else:
            print("  未匹配到动作")


if __name__ == "__main__":
    main()
