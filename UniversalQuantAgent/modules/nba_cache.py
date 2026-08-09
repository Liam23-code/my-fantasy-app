"""Shared retry, cache, and fallback policy for public NBA API endpoints."""
from __future__ import annotations

from io import StringIO
from math import isfinite
import json
from pathlib import Path
import re
import time
from typing import Any, Callable

import pandas as pd

NBA_API_TIMEOUT = 10
NBA_API_RETRIES = 3
NBA_API_BACKOFF_SECONDS = 0.5
FALLBACK_WARNING = "NBA API unavailable. Using fallback projections."
ENHANCED_FALLBACK_WARNING = (
    "NBA API unavailable \u2014 using enhanced fallback projections."
)

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "nba_cache"
_MEMORY_CACHE: dict[str, list[pd.DataFrame]] = {}
_PROVIDER_UNAVAILABLE_UNTIL = 0.0
_PROVIDER_COOLDOWN_SECONDS = 60.0


def _key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_").lower()


def _frames(value: Any) -> list[pd.DataFrame]:
    if isinstance(value, pd.DataFrame):
        return [value.copy()]
    if isinstance(value, (list, tuple)):
        return [frame.copy() for frame in value if isinstance(frame, pd.DataFrame)]
    return []


def _copy(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    return [frame.copy() for frame in frames]


def _mark(
    frames: list[pd.DataFrame],
    source: str,
    warning: str = "",
) -> list[pd.DataFrame]:
    result = _copy(frames)
    for frame in result:
        frame.attrs["data_source"] = source
        existing = frame.attrs.get("warnings", [])
        warnings = list(existing) if isinstance(existing, list) else []
        if warning and warning not in warnings:
            warnings.append(warning)
        frame.attrs["warnings"] = warnings
    return result


def _save(cache_key: str, frames: list[pd.DataFrame]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "frames": [
                json.loads(frame.to_json(orient="split", date_format="iso"))
                for frame in frames
            ]
        }
        (CACHE_DIR / f"{_key(cache_key)}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    except (OSError, TypeError, ValueError):
        # Cache persistence is helpful, never required for an analysis.
        pass


def _load(cache_key: str) -> list[pd.DataFrame]:
    path = CACHE_DIR / f"{_key(cache_key)}.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = [
            pd.read_json(StringIO(json.dumps(frame)), orient="split")
            for frame in payload.get("frames", [])
            if isinstance(frame, dict)
        ]
        return [frame for frame in frames if not frame.empty]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []


def _positive_number(value: Any) -> float | None:
    """Return a finite positive measurement without inventing a default."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) and number > 0 else None


def select_fallback_minutes(
    season_average: Any,
    last_10: Any,
    last_5: Any,
) -> tuple[float, str]:
    """Select minutes in the required season -> last-10 -> last-5 order.

    Returning an error when all three are absent is intentional: a fabricated
    zero silently destroys every downstream rate projection. Provider adapters
    must supply at least one real or cached minutes window.
    """
    for source, candidate in (
        ("season_average", season_average),
        ("last_10", last_10),
        ("last_5", last_5),
    ):
        number = _positive_number(candidate)
        if number is not None:
            return number, source
    raise ValueError(
        "Fallback minutes require a season, last-10, or last-5 average."
    )


def weighted_available_average(
    candidates: list[tuple[Any, float]],
) -> tuple[float, float]:
    """Blend positive observations and automatically reweight missing windows.

    Returns the blended value and the fraction of requested weight supported by
    available data. The latter is used by the reliability model.
    """
    available = [
        (number, float(weight))
        for value, weight in candidates
        if (number := _positive_number(value)) is not None and float(weight) > 0
    ]
    if not available:
        raise ValueError("A weighted fallback requires at least one valid value.")
    available_weight = sum(weight for _, weight in available)
    requested_weight = sum(float(weight) for _, weight in candidates if float(weight) > 0)
    blend = sum(number * weight for number, weight in available) / available_weight
    coverage = available_weight / requested_weight if requested_weight else 0.0
    return blend, coverage

def fetch_nba_frames(
    cache_key: str,
    loader: Callable[[], Any],
    fallback_factory: Callable[[], Any] | None = None,
    retries: int = NBA_API_RETRIES,
    backoff_seconds: float = NBA_API_BACKOFF_SECONDS,
) -> list[pd.DataFrame]:
    """Load NBA frames with retries, then memory, disk, and explicit fallback.

    The endpoint loader must use NBA_API_TIMEOUT. Keeping timeout construction
    next to each endpoint makes that policy visible during code review.
    """
    global _PROVIDER_UNAVAILABLE_UNTIL
    attempts = max(1, int(retries))
    provider_is_cooling_down = time.monotonic() < _PROVIDER_UNAVAILABLE_UNTIL
    if not provider_is_cooling_down:
        for attempt in range(attempts):
            try:
                frames = [frame for frame in _frames(loader()) if not frame.empty]
                if frames:
                    _PROVIDER_UNAVAILABLE_UNTIL = 0.0
                    _MEMORY_CACHE[cache_key] = _copy(frames)
                    _save(cache_key, frames)
                    return _mark(frames, "live")
            except Exception:
                pass
            if attempt < attempts - 1:
                time.sleep(max(0.0, backoff_seconds) * (2 ** attempt))
        # Avoid making every component in one fused projection wait through the
        # same provider outage. A later request automatically retries after the
        # short cooldown.
        _PROVIDER_UNAVAILABLE_UNTIL = time.monotonic() + _PROVIDER_COOLDOWN_SECONDS

    memory = _MEMORY_CACHE.get(cache_key, [])
    if memory:
        return _mark(memory, "last_known_memory", FALLBACK_WARNING)

    disk = _load(cache_key)
    if disk:
        _MEMORY_CACHE[cache_key] = _copy(disk)
        return _mark(disk, "historical_cache", FALLBACK_WARNING)

    fallback = _frames(fallback_factory()) if fallback_factory else []
    if fallback:
        return _mark(fallback, "season_average_fallback", FALLBACK_WARNING)
    return _mark([pd.DataFrame()], "empty_fallback", FALLBACK_WARNING)


def collect_warnings(*frames: pd.DataFrame) -> list[str]:
    """Collect unique ingestion warnings attached to returned data frames."""
    warnings: list[str] = []
    for frame in frames:
        if not isinstance(frame, pd.DataFrame):
            continue
        values = frame.attrs.get("warnings", [])
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            warnings.extend(str(value) for value in values if value)
    return list(dict.fromkeys(warnings))


def used_fallback(*frames: pd.DataFrame) -> bool:
    return any(
        isinstance(frame, pd.DataFrame)
        and frame.attrs.get("data_source") not in (None, "live")
        for frame in frames
    )