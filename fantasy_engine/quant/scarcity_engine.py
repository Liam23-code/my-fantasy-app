"""Deterministic positional-scarcity and replacement-value analytics.

The functions in this module intentionally accept ordinary mappings, Pydantic
models, dataclasses, or keyed mappings of players.  They do not mutate caller
data and do not depend on the UI or on the unified :mod:`quant.quant_engine`
facade, which keeps the analytics safe to reuse from drafts, waivers, trades,
and batch jobs.

All scores use a 0--100 scale and all multipliers are unitless.  Projection
values retain the caller's scoring scale (normally season fantasy points).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

DEFAULT_ROSTER_SLOTS: dict[str, int] = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,
    "DST": 1,
    "K": 1,
}
DEFAULT_FLEX_POSITIONS: tuple[str, ...] = ("RB", "WR", "TE")
PROJECTION_KEYS: tuple[str, ...] = (
    "final_projection",
    "projection",
    "projected_points",
    "expected_fantasy_points",
    "season_projection",
    "fantasy_points",
    "points",
    "median",
)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _finite_number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_mapping(value: Any, *, label: str = "player") -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"{label} must be a mapping or mapping-like object, not {type(value).__name__}")


def _looks_like_player(row: Mapping[str, Any]) -> bool:
    keys = set(row)
    return bool(keys.intersection({"player_id", "id", "name", "player_name", "position", "pos", *PROJECTION_KEYS}))


def _coerce_players(players: Any) -> list[dict[str, Any]]:
    """Normalize a player collection while preserving deterministic order."""

    if players is None:
        return []
    if isinstance(players, Mapping):
        if "players" in players and not _looks_like_player(players):
            return _coerce_players(players["players"])
        if "results" in players and not _looks_like_player(players):
            return _coerce_players(players["results"])
        if _looks_like_player(players):
            sources: list[Any] = [players]
        else:
            sources = []
            for key in sorted(players, key=lambda item: str(item)):
                source = _as_mapping(players[key])
                source.setdefault("player_id", str(key))
                sources.append(source)
    elif isinstance(players, Iterable) and not isinstance(players, (str, bytes)):
        sources = list(players)
    else:
        sources = [players]

    normalized: list[dict[str, Any]] = []
    identities: set[str] = set()
    for index, source in enumerate(sources):
        row = _as_mapping(source)
        position = _position(row)
        if not position:
            raise ValueError(f"player at index {index} is missing a position")
        projection = _projection(row)
        player_id = _player_id(row, index=index)
        # Duplicate feeds are common.  First occurrence wins so the result is
        # stable and source precedence remains under the caller's control.
        identity = f"{position}:{player_id.casefold()}"
        if identity in identities:
            continue
        identities.add(identity)
        normalized.append(
            {
                **row,
                "player_id": player_id,
                "name": _player_name(row, fallback=player_id),
                "position": position,
                "projection": projection,
            }
        )
    return normalized


def _position(row: Mapping[str, Any]) -> str:
    raw = row.get("position", row.get("pos", row.get("fantasy_position", "")))
    position = str(raw or "").strip().upper()
    return "DST" if position in {"D/ST", "DEF", "DEFENSE"} else position


def _player_name(row: Mapping[str, Any], *, fallback: str) -> str:
    name = row.get("name", row.get("player_name", row.get("full_name", "")))
    return str(name or fallback).strip()


def _player_id(row: Mapping[str, Any], *, index: int = 0) -> str:
    value = row.get("player_id", row.get("id", row.get("gsis_id", "")))
    if value is not None and str(value).strip():
        return str(value).strip()
    name = _player_name(row, fallback=f"player-{index + 1}")
    return "-".join(name.casefold().split()) or f"player-{index + 1}"


def _projection(row: Mapping[str, Any], key: str | None = None) -> float:
    keys = (key,) if key else PROJECTION_KEYS
    for candidate in keys:
        if not candidate:
            continue
        value = row.get(candidate)
        if isinstance(value, Mapping):
            value = value.get("points", value.get("value"))
        number = _finite_number(value)
        if number is not None:
            return max(0.0, number)
    return 0.0


def _adp(row: Mapping[str, Any]) -> float | None:
    for key in ("fused_adp", "adp", "average_draft_position", "ecr", "rank"):
        number = _finite_number(row.get(key))
        if number is not None and number > 0:
            return number
    return None


def _validate_teams(teams: int) -> int:
    if isinstance(teams, bool):
        raise ValueError("teams must be a positive integer")
    try:
        parsed = int(teams)
    except (TypeError, ValueError) as exc:
        raise ValueError("teams must be a positive integer") from exc
    if parsed < 1 or float(parsed) != float(teams):
        raise ValueError("teams must be a positive integer")
    return parsed


def _coerce_roster_slots(roster_slots: Any) -> tuple[dict[str, int], tuple[str, ...]]:
    if roster_slots is None:
        return dict(DEFAULT_ROSTER_SLOTS), DEFAULT_FLEX_POSITIONS
    row = _as_mapping(roster_slots, label="roster_slots")
    raw_flex = row.get("flex_eligible", row.get("flex_positions", DEFAULT_FLEX_POSITIONS))
    if "roster_requirements" in row:
        row = _as_mapping(row["roster_requirements"], label="roster_requirements")
    else:
        row.pop("flex_eligible", None)
        row.pop("flex_positions", None)
    if isinstance(raw_flex, str):
        raw_flex = [raw_flex]
    flex_positions = tuple(dict.fromkeys(_position({"position": value}) for value in raw_flex if str(value).strip()))
    slots: dict[str, int] = {}
    for raw_position, raw_count in row.items():
        position = _position({"position": raw_position})
        if position in {"BENCH", "IR", "TAXI"} or not position:
            continue
        number = _finite_number(raw_count)
        if number is None or number < 0 or not number.is_integer():
            raise ValueError(f"roster slot count for {raw_position!r} must be a non-negative integer")
        slots[position] = int(number)
    if not slots:
        raise ValueError("roster_slots must include at least one starting position")
    return slots, flex_positions or DEFAULT_FLEX_POSITIONS


def _positive_rank(value: Any, *, label: str = "replacement_rank") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} values must be positive integers")
    number = _finite_number(value)
    if number is None or number < 1 or not number.is_integer():
        raise ValueError(f"{label} values must be positive integers")
    return int(number)


def positional_depth_curves(players: Any, *, projection_key: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return independently ranked projection curves for every position."""

    normalized = _coerce_players(players)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        if projection_key:
            row = {**row, "projection": _projection(row, projection_key)}
        grouped.setdefault(row["position"], []).append(row)

    curves: dict[str, list[dict[str, Any]]] = {}
    for position in sorted(grouped):
        ordered = sorted(
            grouped[position],
            key=lambda player: (-player["projection"], _adp(player) or math.inf, player["player_id"].casefold()),
        )
        leader = ordered[0]["projection"] if ordered else 0.0
        count = len(ordered)
        curve: list[dict[str, Any]] = []
        for index, player in enumerate(ordered):
            next_projection = ordered[index + 1]["projection"] if index + 1 < count else player["projection"]
            percentile = 100.0 if count == 1 else 100.0 * (count - index - 1) / (count - 1)
            curve.append(
                {
                    "player_id": player["player_id"],
                    "name": player["name"],
                    "position": position,
                    "rank": index + 1,
                    "projection": round(player["projection"], 3),
                    "points_drop_to_next": round(max(0.0, player["projection"] - next_projection), 3),
                    "cumulative_drop_from_leader": round(max(0.0, leader - player["projection"]), 3),
                    "leader_share": round(player["projection"] / leader, 4) if leader > 0 else 0.0,
                    "depth_percentile": round(percentile, 2),
                }
            )
        curves[position] = curve
    return curves


def positional_depth_curve(players: Any, position: str | None = None, *, projection_key: str | None = None) -> list[dict[str, Any]]:
    """Return one position's depth curve.

    ``position`` may be omitted only when the input contains exactly one
    position; this makes the singular helper convenient without ambiguity.
    """

    curves = positional_depth_curves(players, projection_key=projection_key)
    if position is None:
        if len(curves) > 1:
            raise ValueError("position is required when players contain more than one position")
        return next(iter(curves.values()), [])
    return curves.get(_position({"position": position}), [])


def _replacement_context(
    players: Any,
    *,
    roster_slots: Any = None,
    teams: int = 12,
    replacement_rank: int | Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    normalized = _coerce_players(players)
    curves = positional_depth_curves(normalized)
    team_count = _validate_teams(teams)
    slots, flex_positions = _coerce_roster_slots(roster_slots)

    demand: dict[str, int] = {}
    for position in curves:
        base_slots = slots.get(position, 1 if position not in {"K", "DST"} else slots.get(position, 1))
        demand[position] = team_count * base_slots

    # Allocate flex slots to the best remaining eligible players, preserving
    # real cross-position value rather than arbitrarily splitting FLEX 1/3.
    for flex_slot in ("FLEX", "SUPERFLEX"):
        flex_count = slots.get(flex_slot, 0) * team_count
        if not flex_count:
            continue
        eligible = set(flex_positions)
        if flex_slot == "SUPERFLEX":
            eligible.add("QB")
        remaining: list[tuple[float, str, str]] = []
        for position in sorted(eligible):
            curve = curves.get(position, [])
            base = demand.get(position, 0)
            for row in curve[base:]:
                remaining.append((-row["projection"], position, row["player_id"]))
        remaining.sort()
        for _, position, _ in remaining[:flex_count]:
            demand[position] = demand.get(position, 0) + 1

    context: dict[str, dict[str, Any]] = {}
    for position, curve in curves.items():
        if not curve:
            continue
        if isinstance(replacement_rank, Mapping):
            raw_rank = replacement_rank.get(position, replacement_rank.get(position.lower()))
            rank = _positive_rank(raw_rank) if raw_rank is not None else demand.get(position, team_count) + 1
        elif replacement_rank is not None:
            rank = _positive_rank(replacement_rank)
        else:
            rank = demand.get(position, team_count) + 1
        effective_rank = min(rank, len(curve))
        replacement = curve[effective_rank - 1]["projection"]
        context[position] = {
            "replacement_level": float(replacement),
            "replacement_rank": effective_rank,
            "modeled_replacement_rank": rank,
            "starter_demand": max(0, rank - 1),
            "available_depth": len(curve),
            "shortage": max(0, rank - len(curve)),
        }
    return normalized, context


def replacement_level_model(
    players: Any,
    roster_slots: Any = None,
    *,
    teams: int = 12,
    flex_positions: Sequence[str] | None = None,
    replacement_rank: int | Mapping[str, int] | None = None,
) -> dict[str, float]:
    """Model replacement points per position from league starter demand."""

    if flex_positions is not None:
        base_slots, _ = _coerce_roster_slots(roster_slots)
        roster_slots = {**base_slots, "flex_eligible": list(flex_positions)}
    _, context = _replacement_context(
        players,
        roster_slots=roster_slots,
        teams=teams,
        replacement_rank=replacement_rank,
    )
    return {position: round(details["replacement_level"], 3) for position, details in context.items()}


def replacement_level(
    players: Any,
    position: str,
    roster_slots: Any = None,
    *,
    teams: int = 12,
    replacement_rank: int | Mapping[str, int] | None = None,
) -> float:
    """Return a single position's modeled replacement projection."""

    normalized_position = _position({"position": position})
    if not normalized_position:
        raise ValueError("position must not be empty")
    return replacement_level_model(
        players,
        roster_slots,
        teams=teams,
        replacement_rank=replacement_rank,
    ).get(normalized_position, 0.0)


def scarcity_report(
    players: Any,
    *,
    roster_slots: Any = None,
    teams: int = 12,
    replacement_rank: int | Mapping[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return interpretable scarcity components and multipliers by position."""

    normalized, context = _replacement_context(
        players,
        roster_slots=roster_slots,
        teams=teams,
        replacement_rank=replacement_rank,
    )
    curves = positional_depth_curves(normalized)
    report: dict[str, dict[str, Any]] = {}
    for position in sorted(curves):
        curve = curves[position]
        details = context[position]
        replacement = details["replacement_level"]
        starter_count = max(1, min(details["starter_demand"], len(curve)))
        starter_values = [row["projection"] for row in curve[:starter_count]]
        leader = starter_values[0]
        median_starter = statistics.median(starter_values)
        last_starter = starter_values[-1]
        elite_gap = (leader - replacement) / max(leader, 1.0)
        starter_gap = (median_starter - replacement) / max(median_starter, 1.0)
        cliff = (last_starter - replacement) / max(last_starter, 1.0)
        shortage_ratio = details["shortage"] / max(details["modeled_replacement_rank"], 1)
        scarcity_index = _clamp(0.25 * elite_gap + 0.45 * starter_gap + 0.2 * cliff + 0.1 * shortage_ratio, 0.0, 1.0)
        multiplier = _clamp(0.9 + 0.6 * scarcity_index, 0.9, 1.5)
        report[position] = {
            **details,
            "position": position,
            "starter_median": round(median_starter, 3),
            "elite_to_replacement_gap": round(max(0.0, leader - replacement), 3),
            "near_replacement_cliff": round(max(0.0, last_starter - replacement), 3),
            "scarcity_score": round(scarcity_index * 100.0, 2),
            "scarcity_multiplier": round(multiplier, 4),
        }
    return report


def scarcity_multipliers(
    players: Any,
    *,
    roster_slots: Any = None,
    teams: int = 12,
    replacement_rank: int | Mapping[str, int] | None = None,
) -> dict[str, float]:
    """Return only the position-to-multiplier mapping."""

    return {
        position: details["scarcity_multiplier"]
        for position, details in scarcity_report(
            players,
            roster_slots=roster_slots,
            teams=teams,
            replacement_rank=replacement_rank,
        ).items()
    }


def scarcity_multiplier(
    position: str,
    players: Any,
    *,
    roster_slots: Any = None,
    teams: int = 12,
    replacement_rank: int | Mapping[str, int] | None = None,
) -> float:
    """Return one position's multiplier, defaulting to neutral when absent."""

    normalized_position = _position({"position": position})
    if not normalized_position:
        raise ValueError("position must not be empty")
    return scarcity_multipliers(
        players,
        roster_slots=roster_slots,
        teams=teams,
        replacement_rank=replacement_rank,
    ).get(normalized_position, 1.0)


def compute_positional_scarcity(
    players: Any,
    *,
    replacement_rank: int | Mapping[str, int] | None = None,
    roster_slots: Any = None,
    teams: int = 12,
) -> dict[str, Any]:
    """Return a facade-friendly player and position scarcity envelope."""

    normalized, context = _replacement_context(
        players,
        roster_slots=roster_slots,
        teams=teams,
        replacement_rank=replacement_rank,
    )
    by_position = scarcity_report(
        normalized,
        roster_slots=roster_slots,
        teams=teams,
        replacement_rank=replacement_rank,
    )
    results: list[dict[str, Any]] = []
    for player in normalized:
        position = player["position"]
        replacement = context[position]["replacement_level"]
        scarcity = by_position[position]
        vor = player["projection"] - replacement
        results.append(
            {
                "player_id": player["player_id"],
                "name": player["name"],
                "position": position,
                "projection": round(player["projection"], 3),
                "replacement_level": round(replacement, 3),
                "value_over_replacement": round(vor, 3),
                "scarcity_multiplier": scarcity["scarcity_multiplier"],
                "scarcity_score": scarcity["scarcity_score"],
                "scarcity_adjusted_value": round(vor * scarcity["scarcity_multiplier"], 3),
            }
        )
    results.sort(key=lambda row: (-row["scarcity_adjusted_value"], row["player_id"].casefold()))
    return {
        "metric": "positional_scarcity",
        "results": results,
        "by_player": {row["player_id"]: row for row in results},
        "by_position": by_position,
        "metadata": {
            "teams": _validate_teams(teams),
            "player_count": len(results),
            "method": "demand-weighted replacement curve",
        },
    }


def compute_draft_value(
    players: Any,
    *,
    current_pick: float | None = None,
    roster_slots: Any = None,
    teams: int = 12,
    replacement_rank: int | Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return scarcity-adjusted draft values for a player pool."""

    if current_pick is not None:
        parsed_pick = _finite_number(current_pick)
        if parsed_pick is None or parsed_pick < 1:
            raise ValueError("current_pick must be a positive number")
        current_pick = parsed_pick
    scarcity = compute_positional_scarcity(
        players,
        replacement_rank=replacement_rank,
        roster_slots=roster_slots,
        teams=teams,
    )
    normalized = {row["player_id"]: row for row in _coerce_players(players)}
    raw_rows: list[dict[str, Any]] = []
    for row in scarcity["results"]:
        source = normalized[row["player_id"]]
        adp = _adp(source)
        market_delta = (current_pick - adp) if current_pick is not None and adp is not None else 0.0
        market_adjustment = _clamp(market_delta * 0.5, -20.0, 20.0)
        draft_value = row["projection"] + row["scarcity_adjusted_value"] + market_adjustment
        raw_rows.append(
            {
                **row,
                "adp": round(adp, 3) if adp is not None else None,
                "market_delta": round(market_delta, 3),
                "market_adjustment": round(market_adjustment, 3),
                "draft_value": round(draft_value, 3),
            }
        )
    values = [row["draft_value"] for row in raw_rows]
    low, high = (min(values), max(values)) if values else (0.0, 0.0)
    for row in raw_rows:
        row["draft_value_score"] = round(50.0 if high == low else 100.0 * (row["draft_value"] - low) / (high - low), 2)
    raw_rows.sort(key=lambda row: (-row["draft_value"], row["adp"] or math.inf, row["player_id"].casefold()))
    return {
        "metric": "draft_value",
        "results": raw_rows,
        "by_player": {row["player_id"]: row for row in raw_rows},
        "by_position": scarcity["by_position"],
        "metadata": {
            **scarcity["metadata"],
            "current_pick": current_pick,
            "method": "projection plus scarcity-adjusted VOR and market timing",
        },
    }


def draft_value_adjustment(
    player: Any,
    player_pool: Any,
    *,
    current_pick: float | None = None,
    roster_slots: Any = None,
    teams: int = 12,
) -> dict[str, Any]:
    """Return the complete draft adjustment record for one player."""

    target = _coerce_players(player)
    if len(target) != 1:
        raise ValueError("player must resolve to exactly one player")
    pool = _coerce_players(player_pool)
    if all(row["player_id"] != target[0]["player_id"] for row in pool):
        pool.append(target[0])
    result = compute_draft_value(pool, current_pick=current_pick, roster_slots=roster_slots, teams=teams)
    return result["by_player"][target[0]["player_id"]]


def draft_value_adjustments(
    players: Any,
    *,
    current_pick: float | None = None,
    roster_slots: Any = None,
    teams: int = 12,
) -> list[dict[str, Any]]:
    """Return the ranked batch of draft-value adjustment records."""

    return compute_draft_value(players, current_pick=current_pick, roster_slots=roster_slots, teams=teams)["results"]


# Descriptive aliases used by integration code and external consumers.
compute_positional_depth_curves = positional_depth_curves
compute_positional_depth_curve = positional_depth_curve
compute_replacement_levels = replacement_level_model
compute_replacement_level = replacement_level
compute_scarcity_multipliers = scarcity_multipliers
compute_scarcity_multiplier = scarcity_multiplier
compute_draft_value_adjustments = draft_value_adjustments
compute_draft_value_adjustment = draft_value_adjustment


__all__ = [
    "DEFAULT_FLEX_POSITIONS",
    "DEFAULT_ROSTER_SLOTS",
    "compute_draft_value",
    "compute_draft_value_adjustment",
    "compute_draft_value_adjustments",
    "compute_positional_depth_curve",
    "compute_positional_depth_curves",
    "compute_positional_scarcity",
    "compute_replacement_level",
    "compute_replacement_levels",
    "compute_scarcity_multiplier",
    "compute_scarcity_multipliers",
    "draft_value_adjustment",
    "draft_value_adjustments",
    "positional_depth_curve",
    "positional_depth_curves",
    "replacement_level_model",
    "replacement_level",
    "scarcity_multiplier",
    "scarcity_multipliers",
    "scarcity_report",
]
