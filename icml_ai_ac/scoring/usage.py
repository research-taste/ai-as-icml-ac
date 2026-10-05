from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def sum_usage(usages: Iterable[Any]) -> dict[str, int | float]:
    """Sum scalar provider usage fields while ignoring booleans and nested detail."""
    totals: dict[str, int | float] = {}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0) + value
    return totals
