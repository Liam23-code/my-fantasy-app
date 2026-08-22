"""Rolling trend, efficiency, usage, and momentum analytics.

The public functions accept either a numeric time series or weekly/game
records.  Records may be dictionaries, Pydantic models, or simple objects.
Missing football metrics are derived when possible and otherwise use a
neutral zero; malformed public numeric series still fail loudly.  Results are
plain JSON-friendly dictionaries and lists suitable for Graph Lab, Player
Detail, and the unified quant facade.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

_POINT_FIELDS = ("fantasy_points", "points", "actual_points", "actual", "score")
_PROJECTION_FIELDS = (
    "projection",
    "projected_points",
    "expected_fantasy_points",
    "baseline_projection",
)
_EFFICIENCY_FIELDS = (
    "efficiency_score",
    "efficiency",
    "fantasy_points_per_opportunity",
    "points_per_touch",
    "points_per_opportunity",
)
_USAGE_FIELDS = (
    "usage_rate",
    "usage",
    "opportunity_share",
    "touch_share",
    "target_share",
    "snap_share",
    "route_participation",
)
_NESTED_KEYS = ("stats", "metrics", "projection_data")
_CONTAINER_KEYS = (
    "history",
    "weekly_history",
    "weekly",
    "records",
    "game_log",
    "weekly_projection",
    "weekly_projections",
)


@dataclass(frozen=True, slots=True)
class _Observation:
    label: Any
    points: float
    actual_present: bool
    projection: float | None
    efficiency: float
    usage: float
    numeric_input: bool


def _validate_window(window: int) -> int:
    if not isinstance(window, int) or isinstance(window, bool):
        raise TypeError("window must be an integer")
    if window < 1:
        raise ValueError("window must be at least 1")
    return window


def _optional_number(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric or None") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _safe_number(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _is_series(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_record(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return None


def _columnar_records(history: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    sequence_keys = [key for key, value in history.items() if _is_series(value)]
    metric_keys = set(_POINT_FIELDS + _PROJECTION_FIELDS + _EFFICIENCY_FIELDS + _USAGE_FIELDS)
    if not sequence_keys or not any(key in metric_keys for key in sequence_keys):
        return None
    length = max(len(history[key]) for key in sequence_keys)
    rows: list[dict[str, Any]] = []
    for index in range(length):
        row: dict[str, Any] = {}
        for key in sequence_keys:
            values = history[key]
            if index < len(values):
                destination = {"weeks": "week", "games": "game", "dates": "date", "labels": "label"}.get(key, key)
                row[destination] = values[index]
        rows.append(row)
    return rows


def _ordered_mapping_records(container: Mapping[Any, Any], *, projected: bool) -> list[Any]:
    def sort_key(key: Any) -> tuple[int, float | str]:
        try:
            return (0, float(key))
        except (TypeError, ValueError):
            return (1, str(key))

    records: list[Any] = []
    for key in sorted(container, key=sort_key):
        value = container[key]
        record = _as_record(value)
        if record is not None:
            record.setdefault("week", key)
            records.append(record)
        else:
            records.append({"week": key, "projection" if projected else "points": value})
    return records


def _history_items(history: Any) -> list[Any]:
    if isinstance(history, Mapping):
        for key in _CONTAINER_KEYS:
            candidate = history.get(key)
            if _is_series(candidate):
                return list(candidate)
            if isinstance(candidate, Mapping):
                return _ordered_mapping_records(candidate, projected=key in {"weekly_projection", "weekly_projections"})
        columnar = _columnar_records(history)
        if columnar is not None:
            return columnar
        if history and all(str(key).strip().isdigit() for key in history):
            return _ordered_mapping_records(history, projected=False)
        return [history]
    if isinstance(history, (str, bytes, bytearray)):
        raise TypeError("history must be an iterable of numbers or records")
    try:
        return list(history)
    except TypeError as exc:
        raise TypeError("history must be an iterable of numbers or records") from exc


def _sources(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    nested = tuple(record[key] for key in _NESTED_KEYS if isinstance(record.get(key), Mapping))
    return (record, *nested)


def _find(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[str | None, Any]:
    for source in _sources(record):
        for field in fields:
            if field in source and source[field] is not None:
                return field, source[field]
    return None, None


def _opportunities(record: Mapping[str, Any]) -> float:
    _, explicit = _find(record, ("opportunities", "touches"))
    value = _safe_number(explicit)
    if value > 0.0:
        return value
    _, carries = _find(record, ("carries", "rushing_attempts", "rush_attempts"))
    _, targets = _find(record, ("targets", "target_volume"))
    _, attempts = _find(record, ("passing_attempts", "pass_attempts"))
    return max(0.0, _safe_number(carries)) + max(0.0, _safe_number(targets)) + max(0.0, _safe_number(attempts))


def _usage(record: Mapping[str, Any]) -> float:
    field, value = _find(record, _USAGE_FIELDS)
    if field is not None:
        number = _safe_number(value)
        if any(token in field for token in ("rate", "share", "participation")) and 1.0 < number <= 100.0:
            number /= 100.0
        return number

    opportunities = _opportunities(record)
    _, team_value = _find(record, ("team_opportunities", "team_touches", "team_plays"))
    team_opportunities = _safe_number(team_value)
    return opportunities / team_opportunities if team_opportunities > 0.0 else opportunities


def _efficiency(record: Mapping[str, Any], points: float) -> float:
    field, value = _find(record, _EFFICIENCY_FIELDS)
    if field is not None:
        return _safe_number(value)
    opportunities = _opportunities(record)
    return points / opportunities if opportunities > 0.0 else 0.0


def _label(record: Mapping[str, Any], index: int) -> Any:
    _, value = _find(record, ("week", "game", "game_number", "date", "label"))
    if value is None:
        return index + 1
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    return str(value)


def _observations(history: Any) -> list[_Observation]:
    observations: list[_Observation] = []
    for index, item in enumerate(_history_items(history)):
        record = _as_record(item)
        if record is None:
            number = _optional_number(item, label=f"history[{index}]")
            if number is None:
                number = 0.0
            observations.append(
                _Observation(
                    label=index + 1,
                    points=number,
                    actual_present=True,
                    projection=None,
                    efficiency=0.0,
                    usage=0.0,
                    numeric_input=True,
                )
            )
            continue

        actual_field, actual_value = _find(record, _POINT_FIELDS)
        projection_field, projection_value = _find(record, _PROJECTION_FIELDS)
        projection = _safe_number(projection_value) if projection_field is not None else None
        actual_present = actual_field is not None
        points = _safe_number(actual_value) if actual_present else (projection or 0.0)
        observations.append(
            _Observation(
                label=_label(record, index),
                points=points,
                actual_present=actual_present,
                projection=projection,
                efficiency=_efficiency(record, points),
                usage=_usage(record),
                numeric_input=False,
            )
        )
    return observations


def rolling_average(
    values: Iterable[float | int | None],
    window: int = 3,
    min_periods: int = 1,
) -> list[float | None]:
    """Return a trailing rolling mean with explicit missing-value handling.

    ``None`` values are ignored within a window.  An output remains ``None``
    until at least ``min_periods`` numeric observations are present.
    """

    size = _validate_window(window)
    if not isinstance(min_periods, int) or isinstance(min_periods, bool):
        raise TypeError("min_periods must be an integer")
    if not 1 <= min_periods <= size:
        raise ValueError("min_periods must be between 1 and window")
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError("values must be an iterable of numbers or None")
    try:
        supplied = list(values)
    except TypeError as exc:
        raise TypeError("values must be an iterable of numbers or None") from exc

    queue: deque[float | None] = deque()
    total = 0.0
    observed = 0
    result: list[float | None] = []
    for index, raw in enumerate(supplied):
        value = _optional_number(raw, label=f"values[{index}]")
        queue.append(value)
        if value is not None:
            total += value
            observed += 1
        if len(queue) > size:
            expired = queue.popleft()
            if expired is not None:
                total -= expired
                observed -= 1
        result.append(round(total / observed, 6) if observed >= min_periods else None)
    return result


def rolling_efficiency(history: Any, window: int = 3) -> list[float | None]:
    """Return the rolling mean of explicit or derived efficiency."""

    observations = _observations(history)
    values = [observation.points if observation.numeric_input else observation.efficiency for observation in observations]
    return rolling_average(values, window)


def rolling_usage(history: Any, window: int = 3) -> list[float | None]:
    """Return the rolling mean of usage rate/share or opportunity volume."""

    observations = _observations(history)
    values = [observation.points if observation.numeric_input else observation.usage for observation in observations]
    return rolling_average(values, window)


def _record_projection_deltas(observations: Sequence[_Observation]) -> list[float]:
    deltas: list[float] = []
    prior_projection: float | None = None
    for observation in observations:
        if observation.projection is None:
            deltas.append(0.0)
        elif observation.actual_present:
            deltas.append(observation.points - observation.projection)
        elif prior_projection is None:
            deltas.append(0.0)
        else:
            deltas.append(observation.projection - prior_projection)
        if observation.projection is not None:
            prior_projection = observation.projection
    return deltas


def rolling_projection_deltas(
    history_or_actuals: Any,
    projections: Iterable[float | int | None] | None = None,
    window: int = 3,
) -> list[float | None]:
    """Return rolling projection changes or actual-minus-projection deltas.

    With separate ``actuals`` and ``projections``, each raw delta is
    ``actual - projection``.  Records containing both fields use the same
    definition.  A projection-only numeric series uses week-over-week change.
    """

    size = _validate_window(window)
    if projections is not None:
        if isinstance(history_or_actuals, (str, bytes, bytearray, Mapping)):
            raise TypeError("actuals must be an iterable of numbers")
        if isinstance(projections, (str, bytes, bytearray, Mapping)):
            raise TypeError("projections must be an iterable of numbers")
        actual_values = list(history_or_actuals)
        projection_values = list(projections)
        if len(actual_values) != len(projection_values):
            raise ValueError("actuals and projections must have the same length")
        deltas: list[float | None] = []
        for index, (actual, projection) in enumerate(zip(actual_values, projection_values, strict=True)):
            actual_number = _optional_number(actual, label=f"actuals[{index}]")
            projection_number = _optional_number(projection, label=f"projections[{index}]")
            deltas.append(None if actual_number is None or projection_number is None else actual_number - projection_number)
        return rolling_average(deltas, size)

    observations = _observations(history_or_actuals)
    if observations and all(observation.numeric_input for observation in observations):
        values = [observation.points for observation in observations]
        deltas = [0.0, *(current - previous for previous, current in pairwise(values))]
    else:
        deltas = _record_projection_deltas(observations)
    return rolling_average(deltas, size)


def _points_series(values_or_history: Any) -> list[float]:
    return [observation.points for observation in _observations(values_or_history)]


def momentum_score(values_or_history: Any, window: int = 3) -> float:
    """Return recency-weighted momentum on a bounded ``-100`` to ``100`` scale."""

    size = _validate_window(window)
    values = _points_series(values_or_history)
    if len(values) < 2:
        return 0.0
    segment = values[-min(len(values), size * 2) :]
    if max(segment) == min(segment):
        return 0.0

    count = len(segment)
    x_mean = (count - 1) / 2.0
    y_mean = sum(segment) / count
    denominator = sum((index - x_mean) ** 2 for index in range(count))
    slope = (
        sum((index - x_mean) * (value - y_mean) for index, value in enumerate(segment)) / denominator
        if denominator > 0.0
        else 0.0
    )
    scale = max(sum(abs(value) for value in segment) / count, 1.0)
    span_change = slope * (count - 1) / scale

    recent_count = min(size, count)
    recent = segment[-recent_count:]
    previous = segment[max(0, count - 2 * recent_count) : count - recent_count]
    if not previous:
        previous = [segment[0]]
    level_change = ((sum(recent) / len(recent)) - (sum(previous) / len(previous))) / scale

    raw_momentum = 0.65 * span_change + 0.35 * level_change
    return round(max(-100.0, min(100.0, 100.0 * math.tanh(raw_momentum))), 2)


def trend_direction(
    values_or_history: Any,
    *,
    threshold: float = 0.05,
    window: int = 3,
) -> str:
    """Classify recent momentum as ``"up"``, ``"down"``, or ``"flat"``.

    ``threshold`` is the minimum material relative move (``0.05`` means five
    percent) represented on the momentum scale.
    """

    if isinstance(threshold, bool):
        raise TypeError("threshold must be numeric")
    try:
        cutoff = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError("threshold must be numeric") from exc
    if not math.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    score = momentum_score(values_or_history, window)
    if score > cutoff * 100.0:
        return "up"
    if score < -cutoff * 100.0:
        return "down"
    return "flat"


def compute_trend_lines(history: Any, window: int = 3) -> dict[str, Any]:
    """Build the complete trend payload used by fantasy analytics surfaces."""

    size = _validate_window(window)
    observations = _observations(history)
    points = [observation.points for observation in observations]
    efficiency = [observation.efficiency for observation in observations]
    usage = [observation.usage for observation in observations]
    projection_deltas = _record_projection_deltas(observations)
    momentum = momentum_score(points, size)
    return {
        "window": size,
        "games": [observation.label for observation in observations],
        "points": points,
        "rolling_points": rolling_average(points, size),
        "efficiency": efficiency,
        "rolling_efficiency": rolling_average(efficiency, size),
        "usage": usage,
        "rolling_usage": rolling_average(usage, size),
        "projection_deltas": projection_deltas,
        "rolling_projection_deltas": rolling_average(projection_deltas, size),
        "momentum": momentum,
        "direction": trend_direction(points, threshold=0.05, window=size),
    }


# Product-language aliases retained as public API conveniences.
calculate_rolling_average = rolling_average
compute_rolling_average = rolling_average
compute_momentum = momentum_score
determine_trend_direction = trend_direction


__all__ = [
    "calculate_rolling_average",
    "compute_momentum",
    "compute_rolling_average",
    "compute_trend_lines",
    "determine_trend_direction",
    "momentum_score",
    "rolling_average",
    "rolling_efficiency",
    "rolling_projection_deltas",
    "rolling_usage",
    "trend_direction",
]
