from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# voice/ 目录为扁平结构：voice/config.py + voice/config/default.yaml
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path:
        config_path = Path(path)
        # 相对路径若在当前工作目录下不存在，则回退到 voice 包内解析
        if not config_path.is_absolute() and not config_path.exists():
            pkg_path = Path(__file__).resolve().parent / config_path
            if pkg_path.exists():
                config_path = pkg_path
    else:
        config_path = DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
