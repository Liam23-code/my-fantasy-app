"""Unified deterministic analytics facade for every fantasy workflow.

The Quant Engine has two deliberate properties:

* **one input contract** -- mappings, objects, iterables and local data paths
  are normalized through :mod:`quant.data_loader` before analysis; and
* **one output contract** -- batch metrics return an envelope containing
  ``metric``, ``results``, ``by_player`` and ``metadata``.  A single-player
  call additionally mirrors the primary result fields at the top level for
  ergonomic UI and engine integrations.

No function performs network I/O, samples randomness, or depends on wall-clock
time.  Identical data and settings therefore produce identical output.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import data_loader

ENGINE_VERSION = "1.0.0"
REGULAR_SEASON_GAMES = 17

_PRIMARY_METRICS = {
    "base_projection": "base_projection",
    "confidence": "confidence_score",
    "volatility": "volatility",
    "positional_scarcity": "scarcity_score",
    "value_over_replacement": "value_over_replacement",
    "draft_value": "draft_value",
    "trade_value": "trade_value",
    "weekly_matchup": "weekly_matchup_score",
    "trend_lines": "trend_direction",
    "momentum": "momentum_score",
    "rarity_tier": "rarity_tier",
    "health_adjustment": "health_multiplier",
    "usage_rate": "usage_rate",
    "efficiency": "efficiency_score",
    "breakout_probability": "breakout_probability",
    "bust_probability": "bust_probability",
}

_POSITION_REPLACEMENT_RANKS: dict[str, int] = {
    "QB": 12,
    "RB": 30,
    "WR": 36,
    "TE": 12,
    "K": 12,
    "DST": 12,
}

_POSITION_VOLATILITY: dict[str, float] = {
    "QB": 0.18,
    "RB": 0.28,
    "WR": 0.32,
    "TE": 0.30,
    "K": 0.34,
    "DST": 0.38,
}

_SCORING_RULES = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "interceptions": -2.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "fumbles_lost": -2.0,
}


def _import_subengine(name: str) -> Any:
    """Import a sibling engine under either supported package layout."""
    return importlib.import_module(f"{__package__}.{name}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, _safe_float(value, lower)))


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return dict(value.model_dump())
    if hasattr(value, "_asdict") and callable(value._asdict):
        return dict(value._asdict())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return None


def _looks_like_player(row: Mapping[str, Any]) -> bool:
    return bool(
        set(row)
        & {
            "player_id",
            "id",
            "name",
            "player_name",
            "position",
            "projection",
            "expected_fantasy_points",
            "stats",
            "history",
        }
    )


def _raw_records(value: Any) -> tuple[list[Any], bool]:
    """Return raw records and whether the caller supplied one logical player."""
    if value is None:
        return [], False
    if isinstance(value, (str, Path)):
        return data_loader.load_player_stats(value), False
    mapping = _as_mapping(value)
    if mapping is not None:
        if "result" in mapping and isinstance(mapping["result"], Mapping):
            return [mapping["result"]], True
        for key in ("players", "results", "data", "records"):
            if isinstance(mapping.get(key), (list, tuple)):
                return list(mapping[key]), False
        if _looks_like_player(mapping):
            return [mapping], True
        if mapping and all(isinstance(item, Mapping) for item in mapping.values()):
            records = []
            for key, item in mapping.items():
                record = dict(item)
                record.setdefault("player_id", str(key))
                records.append(record)
            return records, False
        return [mapping], True
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return list(value), False
    return [value], True


def _coerce_players(value: Any) -> tuple[list[dict[str, Any]], bool]:
    raw_records, single = _raw_records(value)
    players: list[dict[str, Any]] = []
    for raw in raw_records:
        mapping = _as_mapping(raw)
        if mapping is None:
            continue
        if set(data_loader.CANONICAL_PLAYER_SCHEMA).issubset(mapping):
            players.append(copy.deepcopy(mapping))
            continue
        normalized = data_loader.normalize_player_record(mapping, source_name="player_stats")
        if normalized is None:
            continue
        # Canonical keys keep normalized types; unknown analytical features
        # remain available without weakening the loader's stable base schema.
        for key, item in mapping.items():
            if key not in data_loader.CANONICAL_PLAYER_SCHEMA:
                normalized[key] = copy.deepcopy(item)
        players.append(normalized)
    return players, single


def _lookup(player: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    containers: list[Mapping[str, Any]] = [player]
    for nested in (
        "stats",
        "injury",
        "depth_chart",
        "adp_data",
        "team_strength_data",
        "weather",
    ):
        value = player.get(nested)
        if isinstance(value, Mapping):
            containers.append(value)
    metadata = player.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("raw"), Mapping):
        containers.append(metadata["raw"])
    for container in containers:
        for key in keys:
            if key in container and container[key] not in (None, ""):
                return container[key]
    return default


def _player_id(player: Mapping[str, Any], index: int = 0) -> str:
    return str(player.get("player_id") or player.get("id") or f"quant:player:{index}")


def _identity(player: Mapping[str, Any], index: int = 0) -> dict[str, str]:
    return {
        "player_id": _player_id(player, index),
        "name": str(player.get("name") or player.get("player_name") or ""),
        "position": str(player.get("position") or "").strip().upper(),
        "team": str(player.get("team") or player.get("nfl_team") or "").strip().upper(),
    }


def _envelope(
    metric: str,
    rows: list[dict[str, Any]],
    *,
    single: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    by_player = {str(row.get("player_id") or index): row for index, row in enumerate(rows)}
    payload: dict[str, Any] = {
        "metric": metric,
        "results": rows,
        "by_player": by_player,
        "metadata": {
            "engine_version": ENGINE_VERSION,
            "deterministic": True,
            "count": len(rows),
            **dict(metadata or {}),
        },
    }
    if single and rows:
        payload["result"] = rows[0]
        for key, value in rows[0].items():
            payload.setdefault(key, value)
        primary = _PRIMARY_METRICS.get(metric)
        if primary and primary in rows[0]:
            payload["score"] = rows[0][primary]
    return payload


def _normalize_external_envelope(metric: str, payload: Any, *, single: bool = False) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    if not isinstance(payload.get("results"), list):
        return None
    normalized = dict(payload)
    normalized.setdefault("metric", metric)
    normalized.setdefault("by_player", {str(row.get("player_id") or index): row for index, row in enumerate(payload["results"])})
    normalized.setdefault("metadata", {})
    normalized["metadata"] = {
        "engine_version": ENGINE_VERSION,
        "deterministic": True,
        "count": len(payload["results"]),
        **dict(normalized["metadata"]),
    }
    if single and payload["results"]:
        row = payload["results"][0]
        normalized.setdefault("result", row)
        for key, value in row.items():
            normalized.setdefault(key, value)
    return normalized


def _history(player: Mapping[str, Any]) -> list[Any]:
    history = player.get("history")
    if isinstance(history, (list, tuple)) and history:
        return list(history)
    for key in ("weekly_points", "weekly_scores", "game_log", "recent_points"):
        value = _lookup(player, key)
        if isinstance(value, Mapping):
            return [value[key] for key in sorted(value, key=lambda item: _safe_int(item))]
        if isinstance(value, (list, tuple)):
            return list(value)
    weekly = _lookup(player, "weekly_projection", "weekly_projections")
    if isinstance(weekly, Mapping):
        return [weekly[key] for key in sorted(weekly, key=lambda item: _safe_int(item))]
    return []


def _history_value(item: Any, *keys: str) -> float:
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        return _safe_float(item)
    row = _as_mapping(item)
    if row is None:
        return 0.0
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return _safe_float(row[key])
    return 0.0


def _point_history(player: Mapping[str, Any]) -> list[float]:
    return [
        _history_value(item, "points", "fantasy_points", "actual", "actual_points", "score", "projection")
        for item in _history(player)
    ]


def _weighted_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    weights = range(1, len(values) + 1)
    denominator = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / denominator


def _reception_value(scoring_mode: str) -> float:
    mode = str(scoring_mode or "ppr").strip().lower().replace("-", "_")
    if mode in {"standard", "std", "non_ppr"}:
        return 0.0
    if mode in {"half", "half_ppr", "0.5_ppr"}:
        return 0.5
    return 1.0


def _stat_projection(player: Mapping[str, Any], scoring_mode: str) -> float:
    total = sum(_safe_float(_lookup(player, stat)) * weight for stat, weight in _SCORING_RULES.items())
    total += _safe_float(_lookup(player, "receptions")) * _reception_value(scoring_mode)
    return max(0.0, total)


def _base_projection_row(player: Mapping[str, Any], index: int, scoring_mode: str, games: int) -> dict[str, Any]:
    provided = _safe_float(
        _lookup(player, "base_projection", "projection", "expected_fantasy_points", "projected_points")
    )
    history = [value for value in _point_history(player) if math.isfinite(value)]
    historical = _weighted_mean(history) * games if history else 0.0
    stat_model = _stat_projection(player, scoring_mode)
    points_per_game = _safe_float(_lookup(player, "points_per_game", "ppg"))
    if points_per_game > 0 and historical <= 0:
        historical = points_per_game * games

    available = [("provided_projection", provided), ("weighted_history", historical), ("stat_model", stat_model)]
    positive = [(name, value) for name, value in available if value > 0]
    if provided > 0:
        # A caller-provided forward projection is the baseline contract.  The
        # ensemble projection engine consumes the two diagnostic components
        # below and decides how to blend them; changing the baseline here
        # would apply the same historical/stat correction twice.
        estimate = provided
    elif positive:
        estimate = sum(value for _name, value in positive) / len(positive)
    else:
        estimate = 0.0

    row = {
        **_identity(player, index),
        "base_projection": round(max(0.0, estimate), 3),
        "projected_points": round(max(0.0, estimate), 3),
        "points_per_game": round(max(0.0, estimate) / max(1, games), 3),
        "scoring_mode": scoring_mode,
        "components": {
            "provided_projection": round(provided, 3),
            "weighted_historical_projection": round(historical, 3),
            "stat_projection": round(stat_model, 3),
            "history_games": len(history),
        },
    }
    return row


def load_all_player_data(
    sources: Mapping[str, Any] | Any | None = None,
    *,
    players: Any = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Public Quant facade for :func:`quant.data_loader.load_all_player_data`."""
    return data_loader.load_all_player_data(sources, players=players, strict=strict)


def compute_base_projections(
    players: Any,
    *,
    scoring_mode: str = "ppr",
    games: int = REGULAR_SEASON_GAMES,
) -> dict[str, Any]:
    """Blend a supplied forecast, recency-weighted history and scored stats."""
    pool, single = _coerce_players(players)
    games = max(1, _safe_int(games, REGULAR_SEASON_GAMES))
    rows = [_base_projection_row(player, index, scoring_mode, games) for index, player in enumerate(pool)]
    return _envelope("base_projection", rows, single=single, metadata={"scoring_mode": scoring_mode, "games": games})


def compute_volatility(players: Any) -> dict[str, Any]:
    """Estimate weekly dispersion, with uncertainty rising as confidence falls."""
    pool, single = _coerce_players(players)
    rows: list[dict[str, Any]] = []
    for index, player in enumerate(pool):
        values = _point_history(player)
        mean = statistics.fmean(values) if values else 0.0
        observed_sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        observed_cv = observed_sd / abs(mean) if mean else 0.0
        projection = _base_projection_row(player, index, "ppr", REGULAR_SEASON_GAMES)["base_projection"]
        weekly_mean = projection / REGULAR_SEASON_GAMES if projection > 0 else mean
        floor = _safe_float(_lookup(player, "floor"))
        ceiling = _safe_float(_lookup(player, "ceiling"))
        if weekly_mean > 0 and ceiling > weekly_mean * 4.0:
            floor /= REGULAR_SEASON_GAMES
            ceiling /= REGULAR_SEASON_GAMES
        band_cv = (ceiling - floor) / max(2.0 * abs(weekly_mean), 1.0) if ceiling > floor else 0.0
        position = str(player.get("position") or "").upper()
        prior_cv = _POSITION_VOLATILITY.get(position, 0.30)
        supplied = _safe_float(_lookup(player, "volatility"))
        confidence = _clamp(_lookup(player, "projection_confidence", "confidence", default=0.5))
        candidates = [value for value in (observed_cv, band_cv, supplied) if value > 0]
        evidence_cv = statistics.fmean(candidates) if candidates else prior_cv
        sample_weight = min(0.8, len(values) / 8.0)
        blended_cv = evidence_cv * sample_weight + prior_cv * (1.0 - sample_weight)
        adjusted_cv = blended_cv * (1.35 - 0.7 * confidence)
        normalized = _clamp(adjusted_cv / 0.75)
        confidence_curve = 1.35 - 0.7 * confidence
        standard_deviation = max(observed_sd * confidence_curve, weekly_mean * adjusted_cv)
        risk = "low" if normalized < 0.33 else "medium" if normalized < 0.67 else "high"
        rows.append(
            {
                **_identity(player, index),
                "volatility": round(normalized, 4),
                "coefficient_of_variation": round(adjusted_cv, 4),
                "standard_deviation": round(standard_deviation, 3),
                "risk_level": risk,
                "sample_size": len(values),
                "components": {
                    "observed_cv": round(observed_cv, 4),
                    "position_prior_cv": prior_cv,
                    "confidence": round(confidence, 4),
                },
            }
        )
    return _envelope("volatility", rows, single=single)


def compute_confidence_scores(players: Any) -> dict[str, Any]:
    """Score projection reliability from evidence, completeness and stability."""
    pool, single = _coerce_players(players)
    volatility = compute_volatility(pool).get("by_player", {})
    rows: list[dict[str, Any]] = []
    for index, player in enumerate(pool):
        player_id = _player_id(player, index)
        history_size = len(_history(player))
        games = _safe_int(_lookup(player, "games_played", "games"), history_size)
        sample = _clamp(max(games, history_size) / REGULAR_SEASON_GAMES)
        fields = (
            bool(player.get("name")),
            bool(player.get("position")),
            bool(player.get("team")),
            _safe_float(_lookup(player, "projection", "expected_fantasy_points")) > 0,
            _lookup(player, "adp") is not None,
            bool(_history(player)),
        )
        completeness = sum(fields) / len(fields)
        vol = _safe_float(volatility.get(player_id, {}).get("volatility"), 0.5)
        stability = 1.0 - _clamp(vol)
        health = _clamp(_lookup(player, "health_score", default=1.0))
        supplied = _clamp(_lookup(player, "projection_confidence", "confidence", default=0.5))
        score = 0.25 * supplied + 0.25 * sample + 0.25 * completeness + 0.15 * stability + 0.10 * health
        rows.append(
            {
                **_identity(player, index),
                "confidence_score": round(_clamp(score), 4),
                "confidence": round(_clamp(score), 4),
                "grade": "high" if score >= 0.75 else "medium" if score >= 0.5 else "low",
                "components": {
                    "supplied_confidence": round(supplied, 4),
                    "sample_strength": round(sample, 4),
                    "data_completeness": round(completeness, 4),
                    "stability": round(stability, 4),
                    "health": round(health, 4),
                },
            }
        )
    return _envelope("confidence", rows, single=single)


def _replacement_rank(position: str, replacement_rank: Mapping[str, int] | int | None, teams: int) -> int:
    if isinstance(replacement_rank, Mapping):
        return max(1, _safe_int(replacement_rank.get(position), _POSITION_REPLACEMENT_RANKS.get(position, teams)))
    if replacement_rank is not None:
        return max(1, _safe_int(replacement_rank, teams))
    baseline = _POSITION_REPLACEMENT_RANKS.get(position, teams)
    return max(1, round(baseline * teams / 12))


def _local_scarcity(
    pool: list[dict[str, Any]],
    *,
    replacement_rank: Mapping[str, int] | int | None,
    teams: int,
) -> dict[str, Any]:
    projections = compute_base_projections(pool)["by_player"]
    by_position: dict[str, list[tuple[int, dict[str, Any], float]]] = {}
    for index, player in enumerate(pool):
        position = str(player.get("position") or "").upper()
        projection = _safe_float(projections.get(_player_id(player, index), {}).get("base_projection"))
        by_position.setdefault(position, []).append((index, player, projection))

    rows: list[dict[str, Any]] = []
    position_summary: dict[str, dict[str, Any]] = {}
    for position, group in by_position.items():
        ordered = sorted(group, key=lambda item: (-item[2], _player_id(item[1], item[0])))
        rank = _replacement_rank(position, replacement_rank, teams)
        replacement = ordered[min(len(ordered), rank) - 1][2] if ordered else 0.0
        elite = ordered[0][2] if ordered else replacement
        spread = max(elite - replacement, 1.0)
        for positional_rank, (index, player, projection) in enumerate(ordered, start=1):
            vor = projection - replacement
            scarcity_score = _clamp(max(0.0, vor) / spread)
            rows.append(
                {
                    **_identity(player, index),
                    "position_rank": positional_rank,
                    "projection": round(projection, 3),
                    "replacement_level": round(replacement, 3),
                    "value_over_replacement": round(vor, 3),
                    "scarcity_multiplier": round(1.0 + 0.35 * scarcity_score, 4),
                    "scarcity_score": round(scarcity_score * 100.0, 3),
                }
            )
        position_summary[position] = {
            "players": len(ordered),
            "replacement_rank": min(rank, len(ordered)),
            "replacement_level": round(replacement, 3),
            "elite_projection": round(elite, 3),
        }
    rows.sort(key=lambda row: (-row["scarcity_score"], -row["projection"], row["player_id"]))
    payload = _envelope("positional_scarcity", rows, metadata={"teams": teams})
    payload["by_position"] = position_summary
    return payload


def compute_positional_scarcity(
    players: Any,
    *,
    replacement_rank: Mapping[str, int] | int | None = None,
    roster_slots: Mapping[str, int] | None = None,
    teams: int = 12,
) -> dict[str, Any]:
    """Model depth curves, replacement levels and scarcity multipliers."""
    pool, single = _coerce_players(players)
    teams = max(1, _safe_int(teams, 12))
    try:
        module = _import_subengine("scarcity_engine")
        implementation = module.compute_positional_scarcity
        external = implementation(
            pool,
            replacement_rank=replacement_rank,
            roster_slots=roster_slots,
            teams=teams,
        )
        normalized = _normalize_external_envelope("positional_scarcity", external, single=single)
        if normalized is not None:
            return normalized
    except (ImportError, AttributeError):
        pass
    result = _local_scarcity(pool, replacement_rank=replacement_rank, teams=teams)
    if single and result["results"]:
        result = _normalize_external_envelope("positional_scarcity", result, single=True) or result
    return result


def compute_value_over_replacement(
    players: Any,
    *,
    replacement_rank: Mapping[str, int] | int | None = None,
    roster_slots: Mapping[str, int] | None = None,
    teams: int = 12,
) -> dict[str, Any]:
    """Return projected points above a league-size-adjusted replacement player."""
    pool, single = _coerce_players(players)
    scarcity = compute_positional_scarcity(
        pool,
        replacement_rank=replacement_rank,
        roster_slots=roster_slots,
        teams=teams,
    )
    rows = [
        {
            **{key: row.get(key) for key in ("player_id", "name", "position", "team")},
            "projection": _safe_float(row.get("projection")),
            "replacement_level": _safe_float(row.get("replacement_level")),
            "value_over_replacement": _safe_float(row.get("value_over_replacement")),
            "replacement_rank": _replacement_rank(str(row.get("position") or ""), replacement_rank, teams),
        }
        for row in scarcity.get("results", [])
    ]
    return _envelope("value_over_replacement", rows, single=single, metadata={"teams": teams})


def compute_draft_value(
    players: Any,
    *,
    current_pick: int | None = None,
    replacement_rank: Mapping[str, int] | int | None = None,
    roster_slots: Mapping[str, int] | None = None,
    teams: int = 12,
) -> dict[str, Any]:
    """Combine projections, scarcity and market price into a draft value score."""
    pool, single = _coerce_players(players)
    try:
        module = _import_subengine("scarcity_engine")
        implementation = module.compute_draft_value
        external = implementation(
            pool,
            current_pick=current_pick,
            replacement_rank=replacement_rank,
            roster_slots=roster_slots,
            teams=teams,
        )
        normalized = _normalize_external_envelope("draft_value", external, single=single)
        if normalized is not None:
            return normalized
    except (ImportError, AttributeError):
        pass

    scarcity = compute_positional_scarcity(
        pool,
        replacement_rank=replacement_rank,
        roster_slots=roster_slots,
        teams=teams,
    )["by_player"]
    projection_rows = compute_base_projections(pool)["by_player"]
    ordered_projection = sorted((_safe_float(row.get("base_projection")) for row in projection_rows.values()), reverse=True)
    maximum = ordered_projection[0] if ordered_projection else 1.0
    rows: list[dict[str, Any]] = []
    for index, player in enumerate(pool):
        player_id = _player_id(player, index)
        projection = _safe_float(projection_rows.get(player_id, {}).get("base_projection"))
        scarcity_row = scarcity.get(player_id, {})
        scarcity_score = _safe_float(scarcity_row.get("scarcity_score")) / 100.0
        adp = _safe_float(_lookup(player, "adp", "average_draft_position"), 0.0)
        market_delta = (adp - current_pick) if adp > 0 and current_pick is not None else 0.0
        projection_score = projection / max(maximum, 1.0)
        market_bonus = max(-0.2, min(0.2, market_delta / max(teams * 5.0, 1.0)))
        value_score = _clamp(0.65 * projection_score + 0.25 * scarcity_score + 0.10 + market_bonus)
        rows.append(
            {
                **_identity(player, index),
                "projection": round(projection, 3),
                "adp": round(adp, 2) if adp > 0 else None,
                "market_delta": round(market_delta, 2),
                "value_over_replacement": _safe_float(scarcity_row.get("value_over_replacement")),
                "scarcity_multiplier": _safe_float(scarcity_row.get("scarcity_multiplier"), 1.0),
                "draft_value": round(value_score * 100.0, 3),
                "draft_value_score": round(value_score * 100.0, 3),
            }
        )
    rows.sort(key=lambda row: (-row["draft_value"], row["adp"] or math.inf, row["player_id"]))
    return _envelope("draft_value", rows, single=single, metadata={"current_pick": current_pick, "teams": teams})


def _default_similarity_features(position: str) -> list[str]:
    if position == "QB":
        return ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds"]
    if position in {"RB", "WR", "TE"}:
        return ["rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds", "targets"]
    return ["projection", "points_per_game", "volatility", "usage_rate", "efficiency"]


def _archetype(player: Mapping[str, Any]) -> str:
    position = str(player.get("position") or "").upper()
    rushing = _safe_float(_lookup(player, "rushing_yards"))
    receptions = _safe_float(_lookup(player, "receptions"))
    receiving = _safe_float(_lookup(player, "receiving_yards"))
    targets = _safe_float(_lookup(player, "targets"))
    if position == "QB":
        return "dual-threat quarterback" if rushing >= 300 else "pocket quarterback"
    if position == "RB":
        return "receiving back" if receptions >= 35 or receiving >= 300 else "early-down back"
    if position == "WR":
        yards_per_target = receiving / targets if targets else 0.0
        return "field stretcher" if yards_per_target >= 10 else "volume receiver"
    if position == "TE":
        return "move tight end" if receptions >= 50 else "inline tight end"
    return f"{position.lower()} specialist" if position else "skill player"


def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
    numerator = sum(left * right for left, right in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 and second_norm == 0.0:
        return 1.0
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return _clamp(numerator / (first_norm * second_norm))


def _resolve_player(player: Any, pool: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    mapping = _as_mapping(player)
    if mapping is not None:
        normalized, _single = _coerce_players(mapping)
        return normalized[0] if normalized else None
    key = str(player)
    folded = key.strip().lower()
    for candidate in pool:
        if _player_id(candidate) == key or str(candidate.get("name") or "").strip().lower() == folded:
            return dict(candidate)
    return None


def compute_player_similarity(
    player: Any,
    candidates: Any = None,
    *,
    limit: int = 5,
    features: Sequence[str] | None = None,
    same_position: bool = True,
) -> dict[str, Any]:
    """Return cosine-similar player comps and an interpretable archetype."""
    pool, _single = _coerce_players(candidates)
    target = _resolve_player(player, pool)
    if target is None:
        return {
            **_envelope("player_similarity", [], metadata={"limit": max(0, limit)}),
            "player_id": str(player),
            "comparisons": [],
            "features": list(features or []),
            "archetype": "unknown",
        }
    if not pool:
        pool = [target]
    limit = max(0, _safe_int(limit, 5))

    try:
        module = _import_subengine("similarity_engine")
        implementation = module.compute_player_similarity
        structured = implementation(
            target,
            pool,
            limit=limit,
            features=features,
            same_position=same_position,
        )
        if isinstance(structured, Mapping):
            comparisons = list(structured.get("comparisons") or [])
            payload = _envelope("player_similarity", comparisons, metadata={"limit": limit})
            payload.update(dict(structured))
            payload["result"] = dict(structured)
            return payload
    except (ImportError, AttributeError):
        pass

    selected_features = list(features or _default_similarity_features(str(target.get("position") or "").upper()))
    eligible = [
        candidate
        for candidate in pool
        if _player_id(candidate) != _player_id(target)
        and (not same_position or not target.get("position") or candidate.get("position") == target.get("position"))
    ]
    maxima = {
        feature: max([abs(_safe_float(_lookup(target, feature))), *[abs(_safe_float(_lookup(item, feature))) for item in eligible], 1.0])
        for feature in selected_features
    }

    def vector(item: Mapping[str, Any]) -> list[float]:
        return [_safe_float(_lookup(item, feature)) / maxima[feature] for feature in selected_features]

    target_vector = vector(target)
    comparisons = []
    for index, candidate in enumerate(eligible):
        similarity = _cosine(target_vector, vector(candidate))
        comparisons.append(
            {
                **_identity(candidate, index),
                "similarity": round(similarity, 6),
                "similarity_percent": round(similarity * 100.0, 2),
                "archetype": _archetype(candidate),
            }
        )
    comparisons.sort(key=lambda row: (-row["similarity"], row["player_id"]))
    comparisons = comparisons[:limit]
    structured = {
        **_identity(target),
        "features": selected_features,
        "vector": [round(value, 6) for value in target_vector],
        "archetype": _archetype(target),
        "comparisons": comparisons,
    }
    payload = _envelope("player_similarity", comparisons, metadata={"limit": limit})
    payload.update(structured)
    payload["result"] = structured
    return payload


def compute_trade_value(players: Any, *, team_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Estimate dynasty/redraft trade value with confidence and health risk."""
    pool, single = _coerce_players(players)
    draft = compute_draft_value(pool)["by_player"]
    confidence = compute_confidence_scores(pool)["by_player"]
    volatility = compute_volatility(pool)["by_player"]
    rows: list[dict[str, Any]] = []
    module = None
    try:
        module = _import_subengine("trade_engine")
    except ImportError:
        pass
    for index, player in enumerate(pool):
        if module is not None and hasattr(module, "player_value_score"):
            context = team_context if team_context is not None else {"player_pool": pool}
            external = module.player_value_score(player, context=context)
            if isinstance(external, Mapping):
                row = {**_identity(player, index), **dict(external)}
                value = _safe_float(row.get("trade_value", row.get("value_score")))
                row["trade_value"] = round(value, 3)
                row.setdefault("trade_value_score", row["trade_value"])
                rows.append(row)
                continue
        player_id = _player_id(player, index)
        draft_value = _safe_float(draft.get(player_id, {}).get("draft_value"))
        confidence_score = _safe_float(confidence.get(player_id, {}).get("confidence_score"), 0.5)
        vol = _safe_float(volatility.get(player_id, {}).get("volatility"), 0.5)
        health = _clamp(_lookup(player, "health_score", default=1.0))
        age = _safe_float(_lookup(player, "age"), 27.0)
        position = str(player.get("position") or "").upper()
        age_peak = 27.0 if position in {"RB", "WR"} else 29.0
        age_factor = _clamp(1.0 - max(0.0, age - age_peak) * 0.055, 0.45, 1.0)
        context_need = 0.5
        if team_context:
            needs = team_context.get("position_needs", team_context.get("needs", {}))
            if isinstance(needs, Mapping):
                context_need = _clamp(needs.get(position, 0.5))
        score = draft_value * (0.55 + 0.20 * confidence_score + 0.15 * health + 0.10 * age_factor)
        score *= 0.95 + 0.10 * context_need
        score *= 1.0 - 0.08 * vol
        rows.append(
            {
                **_identity(player, index),
                "trade_value": round(max(0.0, score), 3),
                "trade_value_score": round(max(0.0, score), 3),
                "components": {
                    "draft_value": round(draft_value, 3),
                    "confidence": round(confidence_score, 4),
                    "health": round(health, 4),
                    "age_curve": round(age_factor, 4),
                    "team_need": round(context_need, 4),
                    "volatility": round(vol, 4),
                },
            }
        )
    rows.sort(key=lambda row: (-row["trade_value"], row["player_id"]))
    return _envelope("trade_value", rows, single=single)


def _weekly_projection(player: Mapping[str, Any], week: int) -> float | None:
    weekly = _lookup(player, "weekly_projection", "weekly_projections")
    if isinstance(weekly, Mapping):
        value = weekly.get(week, weekly.get(str(week)))
        if isinstance(value, Mapping):
            value = value.get("points", value.get("projection"))
        if value is not None:
            return max(0.0, _safe_float(value))
    if isinstance(weekly, Sequence) and not isinstance(weekly, (str, bytes)) and 1 <= week <= len(weekly):
        value = weekly[week - 1]
        if isinstance(value, Mapping):
            value = value.get("points", value.get("projection"))
        return max(0.0, _safe_float(value))
    return None


def _schedule_for_week(player: Mapping[str, Any], week: int) -> dict[str, Any]:
    schedule = player.get("schedule")
    if isinstance(schedule, Mapping):
        value = schedule.get(week, schedule.get(str(week), {}))
        return dict(value) if isinstance(value, Mapping) else {"opponent": value}
    if isinstance(schedule, (list, tuple)):
        for item in schedule:
            row = _as_mapping(item)
            if row is not None and _safe_int(row.get("week")) == week:
                return row
    return {}


def compute_weekly_matchup_score(
    players: Any,
    week: int | None = None,
    *,
    matchup: Mapping[str, Any] | None = None,
    scoring_mode: str = "ppr",
) -> dict[str, Any]:
    """Adjust a weekly baseline for opponent, health, weather and bye week."""
    pool, single = _coerce_players(players)
    target_week = max(1, _safe_int(week, 1))
    confidence = compute_confidence_scores(pool)["by_player"]
    rows: list[dict[str, Any]] = []
    for index, player in enumerate(pool):
        player_id = _player_id(player, index)
        schedule = _schedule_for_week(player, target_week)
        context = dict(schedule) | dict(matchup or {})
        bye_week = _safe_int(_lookup(player, "bye_week", "bye"), 0)
        is_bye = bye_week == target_week or bool(context.get("bye") or context.get("is_bye"))
        weekly = _weekly_projection(player, target_week)
        if weekly is None:
            weekly = _base_projection_row(player, index, scoring_mode, REGULAR_SEASON_GAMES)["base_projection"] / REGULAR_SEASON_GAMES
        defense = _clamp(
            context.get(
                "opponent_strength",
                context.get("defensive_strength", _lookup(player, "opponent_strength", "defensive_strength", default=0.5)),
            )
        )
        weather_risk = _clamp(context.get("weather_risk", _lookup(player, "weather_risk", default=0.0)))
        health = _clamp(_lookup(player, "health_score", default=1.0))
        matchup_factor = 1.0 + (0.5 - defense) * 0.30
        weather_factor = 1.0 - 0.12 * weather_risk
        adjusted = 0.0 if is_bye else weekly * matchup_factor * weather_factor * health
        composite_factor = matchup_factor * weather_factor * health
        matchup_score = 0.0 if is_bye else _clamp(0.5 + (composite_factor - 1.0), 0.0, 1.0) * 100.0
        base_confidence = _safe_float(confidence.get(player_id, {}).get("confidence_score"), 0.5)
        weekly_confidence = 0.0 if is_bye else _clamp(base_confidence * (1.0 - 0.2 * weather_risk))
        rows.append(
            {
                **_identity(player, index),
                "week": target_week,
                "opponent": str(context.get("opponent") or player.get("opponent") or "").upper(),
                "is_bye": is_bye,
                "baseline_projection": round(weekly, 3),
                "adjusted_projection": round(max(0.0, adjusted), 3),
                "weekly_projected_points": round(max(0.0, adjusted), 3),
                "weekly_matchup_score": round(matchup_score, 3),
                "confidence": round(weekly_confidence, 4),
                "components": {
                    "defensive_strength": round(defense, 4),
                    "matchup_factor": round(matchup_factor, 4),
                    "weather_factor": round(weather_factor, 4),
                    "health_factor": round(health, 4),
                },
            }
        )
    return _envelope("weekly_matchup", rows, single=single, metadata={"week": target_week, "scoring_mode": scoring_mode})


def _is_history_series(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return True
    mappings = [_as_mapping(item) for item in value]
    return all(
        row is not None
        and not any(key in row for key in ("player_id", "name", "position"))
        and any(key in row for key in ("week", "points", "fantasy_points", "actual", "projection"))
        for row in mappings
    )


def _trend_pool(players: Any) -> tuple[list[dict[str, Any]], bool]:
    if _is_history_series(players):
        record = data_loader.canonical_player_schema()
        record["player_id"] = "quant:series"
        record["name"] = "Series"
        record["history"] = list(players)
        return [record], True
    return _coerce_players(players)


def _fallback_trend(history: Sequence[Any], window: int) -> dict[str, Any]:
    points = [_history_value(item, "points", "fantasy_points", "actual", "actual_points", "projection") for item in history]
    usage = [_history_value(item, "usage", "usage_rate", "opportunity_share") for item in history]
    efficiency = [_history_value(item, "efficiency", "efficiency_score", "fantasy_points_per_opportunity") for item in history]
    deltas = []
    for item in history:
        actual = _history_value(item, "points", "fantasy_points", "actual", "actual_points")
        projection = _history_value(item, "projection", "projected_points")
        deltas.append(actual - projection if projection else 0.0)

    def rolling(values: Sequence[float]) -> list[float | None]:
        result: list[float | None] = []
        for end in range(1, len(values) + 1):
            if end < window:
                result.append(None)
            else:
                result.append(round(statistics.fmean(values[end - window : end]), 4))
        return result

    recent = statistics.fmean(points[-window:]) if points else 0.0
    prior_values = points[-2 * window : -window]
    prior = statistics.fmean(prior_values) if prior_values else (statistics.fmean(points[:-1]) if len(points) > 1 else recent)
    scale = max(abs(prior), statistics.fmean([abs(value) for value in points]) if points else 1.0, 1.0)
    momentum = max(-1.0, min(1.0, (recent - prior) / scale))
    direction = "up" if momentum > 0.05 else "down" if momentum < -0.05 else "flat"
    return {
        "window": window,
        "games": len(points),
        "points": points,
        "rolling_points": rolling(points),
        "efficiency": efficiency,
        "rolling_efficiency": rolling(efficiency),
        "usage": usage,
        "rolling_usage": rolling(usage),
        "projection_deltas": deltas,
        "rolling_projection_deltas": rolling(deltas),
        "momentum": round(momentum, 6),
        "direction": direction,
    }


def compute_trend_lines(players: Any, *, window: int = 3) -> dict[str, Any]:
    """Compute rolling points, usage, efficiency and projection deltas."""
    pool, single = _trend_pool(players)
    window = max(1, _safe_int(window, 3))
    implementation = None
    try:
        implementation = _import_subengine("trend_engine").compute_trend_lines
    except (ImportError, AttributeError):
        pass
    rows = []
    for index, player in enumerate(pool):
        history = _history(player)
        trend = implementation(history, window=window) if implementation is not None else _fallback_trend(history, window)
        rows.append(
            {
                **_identity(player, index),
                **dict(trend),
                "trend_line": list(trend.get("rolling_points") or []),
                "trend_direction": str(trend.get("direction") or "flat"),
            }
        )
    return _envelope("trend_lines", rows, single=single, metadata={"window": window})


def compute_momentum(players: Any, *, window: int = 3) -> dict[str, Any]:
    """Return normalized recent-vs-prior momentum and trend direction."""
    pool, single = _trend_pool(players)
    window = max(1, _safe_int(window, 3))
    score_function = direction_function = None
    try:
        module = _import_subengine("trend_engine")
        score_function = module.momentum_score
        direction_function = module.trend_direction
    except (ImportError, AttributeError):
        pass
    rows = []
    for index, player in enumerate(pool):
        history = _history(player)
        if score_function is not None and direction_function is not None:
            momentum = _safe_float(score_function(history, window=window))
            direction = str(direction_function(history, window=window))
        else:
            trend = _fallback_trend(history, window)
            momentum = _safe_float(trend["momentum"])
            direction = str(trend["direction"])
        normalized = max(-1.0, min(1.0, momentum / 100.0)) if score_function is not None else max(-1.0, min(1.0, momentum))
        rows.append(
            {
                **_identity(player, index),
                "momentum": round(normalized, 6),
                "momentum_score": round((normalized + 1.0) * 50.0, 3),
                "direction": direction if direction in {"up", "down", "flat"} else "flat",
                "sample_size": len(history),
                "window": window,
            }
        )
    return _envelope("momentum", rows, single=single, metadata={"window": window})


def compute_rarity_tier(players: Any) -> dict[str, Any]:
    """Assign HoopGrids-compatible rarity tiers from overall analytical rank."""
    if isinstance(players, (int, float)) and not isinstance(players, bool):
        record = data_loader.canonical_player_schema()
        record.update({"player_id": "quant:value", "name": "Value", "projection": _safe_float(players)})
        pool, single = [record], True
    else:
        pool, single = _coerce_players(players)
    draft = compute_draft_value(pool)
    ranked = sorted(draft.get("results", []), key=lambda row: (-_safe_float(row.get("draft_value")), row.get("player_id", "")))
    rows = []
    count = max(1, len(ranked))
    for rank, row in enumerate(ranked, start=1):
        percentile = 1.0 - (rank - 1) / count
        if rank == 1 and count >= 25:
            tier, symbol = "Mythic", "✦"
        elif percentile >= 0.95:
            tier, symbol = "Legendary", "◆"
        elif percentile >= 0.80:
            tier, symbol = "Elite", "⬟"
        elif percentile >= 0.50:
            tier, symbol = "Pro", "★"
        elif percentile >= 0.20:
            tier, symbol = "Starter", "●"
        else:
            tier, symbol = "Depth", "○"
        rows.append(
            {
                **{key: row.get(key) for key in ("player_id", "name", "position", "team")},
                "rank": rank,
                "percentile": round(percentile, 4),
                "rarity_tier": tier,
                "tier": tier,
                "symbol": symbol,
                "draft_value": _safe_float(row.get("draft_value")),
            }
        )
    return _envelope("rarity_tier", rows, single=single)


def compute_health_adjustments(players: Any) -> dict[str, Any]:
    """Apply status, injury risk and expected availability to projections."""
    pool, single = _coerce_players(players)
    projection = compute_base_projections(pool)["by_player"]
    rows = []
    status_factors = {
        "IR": 0.0,
        "PUP": 0.15,
        "OUT": 0.0,
        "DOUBTFUL": 0.25,
        "QUESTIONABLE": 0.75,
        "LIMITED": 0.88,
        "PROBABLE": 0.96,
        "ACTIVE": 1.0,
        "HEALTHY": 1.0,
    }
    for index, player in enumerate(pool):
        player_id = _player_id(player, index)
        status = str(_lookup(player, "injury_status", "status", default="ACTIVE") or "ACTIVE").strip().upper()
        status_factor = status_factors.get(status, 0.90)
        risk = _clamp(_lookup(player, "injury_risk", default=1.0 - _safe_float(_lookup(player, "health_score"), 1.0)))
        expected_games = _safe_float(_lookup(player, "expected_games"), REGULAR_SEASON_GAMES)
        availability = _clamp(expected_games / REGULAR_SEASON_GAMES) if expected_games > 0 else 0.0
        multiplier = _clamp(status_factor * (1.0 - 0.35 * risk) * availability)
        base = _safe_float(projection.get(player_id, {}).get("base_projection"))
        rows.append(
            {
                **_identity(player, index),
                "injury_status": status,
                "injury_risk": round(risk, 4),
                "health_multiplier": round(multiplier, 4),
                "base_projection": round(base, 3),
                "health_adjusted_projection": round(base * multiplier, 3),
                "expected_games": round(expected_games, 2),
            }
        )
    return _envelope("health_adjustment", rows, single=single)


def compute_usage_rates(players: Any) -> dict[str, Any]:
    """Blend opportunity, snaps and role shares into a 0-1 usage rate."""
    pool, single = _coerce_players(players)
    rows = []
    for index, player in enumerate(pool):
        supplied = _safe_float(_lookup(player, "usage_rate", "usage", "opportunity_share"))
        shares = [
            _safe_float(_lookup(player, "target_share")),
            _safe_float(_lookup(player, "rush_share", "carry_share")),
            _safe_float(_lookup(player, "snap_share")),
            _safe_float(_lookup(player, "route_participation", "route_share")),
        ]
        shares = [value / 100.0 if value > 1.0 else value for value in shares if value > 0]
        opportunities = _safe_float(_lookup(player, "opportunities", "touches"))
        if opportunities <= 0:
            opportunities = _safe_float(_lookup(player, "carries", "rushing_attempts")) + _safe_float(_lookup(player, "targets"))
        team_plays = _safe_float(_lookup(player, "team_plays", "team_opportunities"))
        opportunity_share = opportunities / team_plays if team_plays > 0 else 0.0
        candidates = [value / 100.0 if value > 1.0 else value for value in (supplied, opportunity_share) if value > 0] + shares
        usage = _clamp(statistics.fmean(candidates) if candidates else 0.0)
        rows.append(
            {
                **_identity(player, index),
                "usage_rate": round(usage, 4),
                "usage_percent": round(usage * 100.0, 2),
                "opportunities": round(opportunities, 2),
                "components": {
                    "opportunity_share": round(opportunity_share, 4),
                    "role_shares": [round(value, 4) for value in shares],
                    "supplied_usage": round(supplied, 4),
                },
            }
        )
    return _envelope("usage_rate", rows, single=single)


def compute_efficiency_scores(players: Any) -> dict[str, Any]:
    """Score fantasy output, yards and touchdowns per opportunity."""
    pool, single = _coerce_players(players)
    projection = compute_base_projections(pool)["by_player"]
    raw_rows = []
    for index, player in enumerate(pool):
        player_id = _player_id(player, index)
        games = max(1, _safe_int(_lookup(player, "games_played", "games"), REGULAR_SEASON_GAMES))
        fantasy_points = _safe_float(projection.get(player_id, {}).get("base_projection"))
        opportunities = _safe_float(_lookup(player, "opportunities", "touches"))
        if opportunities <= 0:
            opportunities = _safe_float(_lookup(player, "carries", "rushing_attempts")) + _safe_float(_lookup(player, "targets"))
        if opportunities <= 0:
            opportunities = games
        yards = (
            _safe_float(_lookup(player, "passing_yards"))
            + _safe_float(_lookup(player, "rushing_yards"))
            + _safe_float(_lookup(player, "receiving_yards"))
        )
        touchdowns = (
            _safe_float(_lookup(player, "passing_tds"))
            + _safe_float(_lookup(player, "rushing_tds"))
            + _safe_float(_lookup(player, "receiving_tds"))
        )
        supplied = _safe_float(_lookup(player, "efficiency", "efficiency_score"))
        raw = supplied if supplied > 0 else fantasy_points / opportunities
        raw_rows.append((index, player, raw, fantasy_points / opportunities, yards / opportunities, touchdowns / opportunities))

    by_position: dict[str, list[float]] = {}
    for _index, player, raw, *_rest in raw_rows:
        by_position.setdefault(str(player.get("position") or "").upper(), []).append(raw)
    rows = []
    for index, player, raw, points_per_opportunity, yards_per_opportunity, td_rate in raw_rows:
        position = str(player.get("position") or "").upper()
        peers = sorted(by_position.get(position, [raw]))
        below = sum(1 for value in peers if value < raw)
        equal = sum(1 for value in peers if value == raw)
        percentile = (below + 0.5 * equal) / max(1, len(peers))
        rows.append(
            {
                **_identity(player, index),
                "efficiency_score": round(percentile * 100.0, 3),
                "efficiency": round(raw, 5),
                "fantasy_points_per_opportunity": round(points_per_opportunity, 5),
                "yards_per_opportunity": round(yards_per_opportunity, 5),
                "touchdown_rate": round(td_rate, 5),
                "position_percentile": round(percentile, 4),
            }
        )
    return _envelope("efficiency", rows, single=single)


def _sigmoid(value: float) -> float:
    if value >= 0:
        term = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + term)
    term = math.exp(min(-value, 60.0))
    return 1.0 / (1.0 + term)


def compute_breakout_probability(players: Any) -> dict[str, Any]:
    """Estimate upside odds from age, role, efficiency, momentum and confidence."""
    pool, single = _coerce_players(players)
    usage = compute_usage_rates(pool)["by_player"]
    efficiency = compute_efficiency_scores(pool)["by_player"]
    momentum = compute_momentum(pool)["by_player"]
    confidence = compute_confidence_scores(pool)["by_player"]
    rows = []
    for index, player in enumerate(pool):
        player_id = _player_id(player, index)
        age = _safe_float(_lookup(player, "age"), 27.0)
        position = str(player.get("position") or "").upper()
        young_threshold = 25.0 if position == "RB" else 26.0 if position in {"WR", "TE"} else 27.0
        age_edge = max(-1.0, min(1.0, (young_threshold - age) / 4.0))
        usage_score = _safe_float(usage.get(player_id, {}).get("usage_rate"))
        efficiency_score = _safe_float(efficiency.get(player_id, {}).get("efficiency_score")) / 100.0
        momentum_score = _safe_float(momentum.get(player_id, {}).get("momentum"))
        confidence_score = _safe_float(confidence.get(player_id, {}).get("confidence_score"), 0.5)
        depth_order = _safe_int(_lookup(player, "depth_order"), 2)
        role_bonus = 0.3 if depth_order == 1 else 0.0 if depth_order == 2 else -0.2
        logit = -1.25 + 1.8 * usage_score + 1.2 * efficiency_score + 0.8 * momentum_score + 0.45 * age_edge + role_bonus + 0.3 * confidence_score
        probability = _clamp(_sigmoid(logit))
        rows.append(
            {
                **_identity(player, index),
                "breakout_probability": round(probability, 4),
                "probability": round(probability, 4),
                "label": "high" if probability >= 0.65 else "moderate" if probability >= 0.4 else "low",
                "components": {
                    "usage": round(usage_score, 4),
                    "efficiency": round(efficiency_score, 4),
                    "momentum": round(momentum_score, 4),
                    "age_edge": round(age_edge, 4),
                    "role_bonus": role_bonus,
                    "confidence": round(confidence_score, 4),
                },
            }
        )
    return _envelope("breakout_probability", rows, single=single)


def compute_bust_probability(players: Any) -> dict[str, Any]:
    """Estimate downside odds from health, variance, decline and market premium."""
    pool, single = _coerce_players(players)
    volatility = compute_volatility(pool)["by_player"]
    momentum = compute_momentum(pool)["by_player"]
    efficiency = compute_efficiency_scores(pool)["by_player"]
    rows = []
    for index, player in enumerate(pool):
        player_id = _player_id(player, index)
        vol = _safe_float(volatility.get(player_id, {}).get("volatility"), 0.5)
        momentum_value = _safe_float(momentum.get(player_id, {}).get("momentum"))
        decline = max(0.0, -momentum_value)
        efficiency_risk = 1.0 - _safe_float(efficiency.get(player_id, {}).get("efficiency_score"), 50.0) / 100.0
        injury_risk = _clamp(_lookup(player, "injury_risk", default=0.0))
        adp = _safe_float(_lookup(player, "adp"), 0.0)
        projection = _safe_float(_lookup(player, "projection", "expected_fantasy_points"))
        market_premium = _clamp((50.0 - adp) / 50.0) * _clamp((250.0 - projection) / 250.0) if adp > 0 else 0.0
        logit = -1.8 + 1.6 * injury_risk + 1.2 * vol + 1.0 * decline + 0.9 * efficiency_risk + 0.8 * market_premium
        probability = _clamp(_sigmoid(logit))
        rows.append(
            {
                **_identity(player, index),
                "bust_probability": round(probability, 4),
                "probability": round(probability, 4),
                "label": "high" if probability >= 0.65 else "moderate" if probability >= 0.4 else "low",
                "components": {
                    "injury_risk": round(injury_risk, 4),
                    "volatility": round(vol, 4),
                    "decline": round(decline, 4),
                    "efficiency_risk": round(efficiency_risk, 4),
                    "market_premium": round(market_premium, 4),
                },
            }
        )
    return _envelope("bust_probability", rows, single=single)


def compute_final_projection(
    player: Any,
    *,
    players: Any = None,
    scoring_mode: str = "ppr",
    **kwargs: Any,
) -> dict[str, Any]:
    """Call the advanced projection module lazily, with a safe base fallback."""
    pool, _single = _coerce_players(players)
    target = _resolve_player(player, pool)
    target_argument = target if isinstance(player, Mapping) else player
    for module_name in ("projections.projection_engine", "fantasy_engine.projections.projection_engine"):
        try:
            implementation = importlib.import_module(module_name).compute_final_projection
        except (ImportError, AttributeError):
            continue
        signature = inspect.signature(implementation)
        available = {
            "players": pool,
            "player_pool": pool,
            "player_data": pool,
            "scoring_mode": scoring_mode,
            **kwargs,
        }
        accepted = {key: value for key, value in available.items() if key in signature.parameters}
        result = implementation(target_argument, **accepted)
        if isinstance(result, Mapping):
            return dict(result)
        return {
            "player_id": _player_id(target or {"player_id": str(player)}),
            "final_projection": _safe_float(result),
            "projection": _safe_float(result),
            "method": "advanced_projection_engine",
        }
    if target is None:
        normalized, _ = _coerce_players(player)
        target = normalized[0] if normalized else None
    if target is None:
        raise KeyError(f"player {player!r} was not found in the Quant Engine player pool")
    base = compute_base_projections(target, scoring_mode=scoring_mode)
    projection = _safe_float(base.get("base_projection"))
    return {
        **_identity(target),
        "final_projection": projection,
        "projection": projection,
        "confidence": compute_confidence_scores(target).get("confidence_score", 0.0),
        "method": "quant_base_projection_fallback",
        "components": base.get("components", {}),
    }


class QuantEngine:
    """Stateful facade that applies every metric to one configured player pool."""

    def __init__(
        self,
        players: Any = None,
        *,
        sources: Mapping[str, Any] | Any | None = None,
        scoring_mode: str = "ppr",
    ) -> None:
        self.scoring_mode = str(scoring_mode or "ppr")
        if sources is not None:
            self.players = load_all_player_data(sources, players=players)
        else:
            self.players, _single = _coerce_players(players)

    def set_player_pool(self, players: Any) -> list[dict[str, Any]]:
        """Replace and return the normalized in-memory player pool."""
        self.players, _single = _coerce_players(players)
        return copy.deepcopy(self.players)

    def load_all_player_data(
        self,
        sources: Mapping[str, Any] | Any | None = None,
        *,
        players: Any = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        self.players = load_all_player_data(sources, players=players, strict=strict)
        return copy.deepcopy(self.players)

    def _pool(self, players: Any) -> Any:
        return self.players if players is None else players

    def compute_base_projections(self, players: Any = None, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("scoring_mode", self.scoring_mode)
        return compute_base_projections(self._pool(players), **kwargs)

    def compute_confidence_scores(self, players: Any = None) -> dict[str, Any]:
        return compute_confidence_scores(self._pool(players))

    def compute_volatility(self, players: Any = None) -> dict[str, Any]:
        return compute_volatility(self._pool(players))

    def compute_positional_scarcity(self, players: Any = None, **kwargs: Any) -> dict[str, Any]:
        return compute_positional_scarcity(self._pool(players), **kwargs)

    def compute_player_similarity(self, player: Any, candidates: Any = None, **kwargs: Any) -> dict[str, Any]:
        return compute_player_similarity(player, self.players if candidates is None else candidates, **kwargs)

    def compute_value_over_replacement(self, players: Any = None, **kwargs: Any) -> dict[str, Any]:
        return compute_value_over_replacement(self._pool(players), **kwargs)

    def compute_draft_value(self, players: Any = None, **kwargs: Any) -> dict[str, Any]:
        return compute_draft_value(self._pool(players), **kwargs)

    def compute_trade_value(self, players: Any = None, **kwargs: Any) -> dict[str, Any]:
        return compute_trade_value(self._pool(players), **kwargs)

    def compute_weekly_matchup_score(self, players: Any = None, week: int | None = None, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("scoring_mode", self.scoring_mode)
        return compute_weekly_matchup_score(self._pool(players), week, **kwargs)

    def compute_trend_lines(self, players: Any = None, **kwargs: Any) -> dict[str, Any]:
        return compute_trend_lines(self._pool(players), **kwargs)

    def compute_momentum(self, players: Any = None, **kwargs: Any) -> dict[str, Any]:
        return compute_momentum(self._pool(players), **kwargs)

    def compute_rarity_tier(self, players: Any = None) -> dict[str, Any]:
        return compute_rarity_tier(self._pool(players))

    def compute_health_adjustments(self, players: Any = None) -> dict[str, Any]:
        return compute_health_adjustments(self._pool(players))

    def compute_usage_rates(self, players: Any = None) -> dict[str, Any]:
        return compute_usage_rates(self._pool(players))

    def compute_efficiency_scores(self, players: Any = None) -> dict[str, Any]:
        return compute_efficiency_scores(self._pool(players))

    def compute_breakout_probability(self, players: Any = None) -> dict[str, Any]:
        return compute_breakout_probability(self._pool(players))

    def compute_bust_probability(self, players: Any = None) -> dict[str, Any]:
        return compute_bust_probability(self._pool(players))

    def compute_final_projection(self, player: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("scoring_mode", self.scoring_mode)
        player_pool = kwargs.pop("players", self.players)
        return compute_final_projection(player, players=player_pool, **kwargs)

    def analyze_player(self, player: Any, *, week: int = 1) -> dict[str, Any]:
        """Return the cross-module analytical bundle used by detail surfaces."""
        target = _resolve_player(player, self.players)
        if target is None:
            raise KeyError(f"player {player!r} was not found in the Quant Engine player pool")
        return {
            "player": copy.deepcopy(target),
            "projection": self.compute_final_projection(target),
            "confidence": self.compute_confidence_scores(target),
            "volatility": self.compute_volatility(target),
            "weekly_matchup": self.compute_weekly_matchup_score(target, week),
            "trends": self.compute_trend_lines(target),
            "momentum": self.compute_momentum(target),
            "rarity": self.compute_rarity_tier(target),
            "similarity": self.compute_player_similarity(target),
            "breakout": self.compute_breakout_probability(target),
            "bust": self.compute_bust_probability(target),
        }


default_engine = QuantEngine()
quant = default_engine


__all__ = [
    "ENGINE_VERSION",
    "QuantEngine",
    "compute_base_projections",
    "compute_breakout_probability",
    "compute_bust_probability",
    "compute_confidence_scores",
    "compute_draft_value",
    "compute_efficiency_scores",
    "compute_final_projection",
    "compute_health_adjustments",
    "compute_momentum",
    "compute_player_similarity",
    "compute_positional_scarcity",
    "compute_rarity_tier",
    "compute_trade_value",
    "compute_trend_lines",
    "compute_usage_rates",
    "compute_value_over_replacement",
    "compute_volatility",
    "compute_weekly_matchup_score",
    "default_engine",
    "load_all_player_data",
    "quant",
]
