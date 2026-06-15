"""纯语音测试 — 不涉及相机/VLM/机械臂"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vla.voice import VoiceControl

def main():
    voice = VoiceControl(lang="zh")

    if not voice._ready:
        print("whisper.cpp 未安装")
        return

    print("=" * 50)
    print("  语音识别测试（仅语音，无机械臂）")
    print("=" * 50)
    print("  按 Enter 录音 3 秒 → whisper 识别 → 显示结果")
    print("  输入 q 退出")
    print()

    while True:
        inp = input("按 Enter 开始录音 (q 退出): ")
        if inp.lower() == "q":
            break

        print("🎤 录音 3 秒，请说话...")
        audio_path = voice.record(duration=3)
        if not audio_path:
            print("  录音失败")
            continue
        text = voice.listen(audio_path)
        print(f"  whisper 识别结果: [{text}]")
        if not text:
            print("  未检测到语音")
            continue
        cmd = voice.parse_command(text)

        if cmd is None:
            print("  未识别到有效指令（试试说: 抓红色杯子）")
        else:
            print(f"  ✅ 识别成功")
            print(f"     原始文字: {cmd['raw']}")
            print(f"     动作: {cmd['action']}")
            print(f"     颜色: {cmd['color']}")
            print(f"     物体: {cmd['object']}")

    print("\n测试结束")


if __name__ == "__main__":
    main()
