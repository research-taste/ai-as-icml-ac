from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    """Load simple KEY=VALUE lines into os.environ if they are not already set."""

    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_env_value(value.strip())
        if not key:
            continue
        if key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
