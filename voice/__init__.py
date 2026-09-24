"""语音交互模块 -- sherpa-onnx KWS/ASR/TTS + 意图路由

对外仅暴露 VoiceAssistant（实现 hardware.interfaces.Module 接口），
其余为内部实现细节。sherpa-onnx 等依赖缺失时优雅降级，不影响核心抓取。
"""
from voice.assistant import VoiceAssistant

__all__ = ["VoiceAssistant"]
