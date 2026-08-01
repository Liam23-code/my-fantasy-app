"""Small, dependency-free helpers shared across the fantasy engine.

Kept separate from ``modules.data_quality`` in the sibling NFL analytics repo
on purpose: this package must run standalone (see the migration guide in
``README.md``), so it does not import anything from that project.
"""
from __future__ import annotations

import math
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to a finite float, never raising.

    Handles ``None``, numeric strings (``"245.0"``), bools, and already-numeric
    values. Anything non-numeric, ``NaN``, or infinite falls back to ``default``.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else default
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return default
        try:
            parsed = float(text)
        except ValueError:
            return default
        return parsed if math.isfinite(parsed) else default
    return default


def safe_int(value: Any, default: int = 0) -> int:
    return int(round(safe_float(value, float(default))))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
