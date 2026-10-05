from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from icml_ai_ac.models import PaperRecord


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
        temp_name = handle.name
    os.replace(temp_name, path)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
                yield row


def read_paper_records(path: Path) -> Iterator[PaperRecord]:
    for row in read_jsonl(path):
        yield PaperRecord.from_dict(row)


def read_jsonl_if_exists(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())
    return read_jsonl(path)
