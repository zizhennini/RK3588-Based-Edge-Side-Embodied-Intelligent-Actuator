#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "voice_assistant"))
from voice_assistant.cli import main


if __name__ == "__main__":
    main()
