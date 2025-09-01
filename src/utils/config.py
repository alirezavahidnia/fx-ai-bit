from __future__ import annotations
import os, yaml
from typing import Any, Dict

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError("config.yaml not found")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
