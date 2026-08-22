"""Ensemble projection system for the unified fantasy Quant Engine.

The projection engine deliberately accepts ordinary mappings instead of
requiring a database model. A caller can pass a player record directly, look
one up in an explicit pool, or configure an in-process registry. Every model
returns an auditable object with its estimate and components; the final model
combines them without mutating the source data.

No network access occurs here. Data acquisition and normalization belong to
``quant.data_loader``. This separation makes projections deterministic in
tests, batch jobs, Streamlit reruns, and API workers alike.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fantasy.scoring import calculate_fantasy_points
from fantasy.utils import clamp, safe_float

REGULAR_SEASON_GAMES = 17

MODEL_WEIGHTS: dict[str, float] = {
    "historical": 0.31,
    "usage": 0.22,
    "matchup": 0.14,
    "regression": 0.18,
    "market": 0.15,
}

#: Per-position ensemble weight profiles. A single global blend treats a QB
#: (volume is fixed by the offense's play count, output is passing-efficient
#: and stable) the same as a kicker (almost no usage signal, dominated by
#: matchup/game script). Falls back to ``MODEL_WEIGHTS`` for any position not
#: listed here. Each profile still sums to 1.0.
POSITION_MODEL_WEIGHTS: dict[str, dict[str, float]] = {
    "QB": {"historical": 0.36, "usage": 0.14, "matchup": 0.14, "regression": 0.16, "market": 0.20},
    "RB": {"historical": 0.28, "usage": 0.26, "matchup": 0.16, "regression": 0.15, "market": 0.15},
    "WR": {"historical": 0.27, "usage": 0.25, "matchup": 0.15, "regression": 0.18, "market": 0.15},
    "TE": {"historical": 0.24, "usage": 0.22, "matchup": 0.14, "regression": 0.25, "market": 0.15},
    "K": {"historical": 0.20, "usage": 0.05, "matchup": 0.30, "regression": 0.30, "market": 0.15},
    "DST": {"historical": 0.20, "usage": 0.05, "matchup": 0.35, "regression": 0.25, "market": 0.15},
}

#: Regression-to-mean prior strength (in "games worth" of the positional
#: anchor) per position. Lower means the player's own sample overrides the
#: positional prior faster. QB production stabilizes within a season; TE
#: role/target competition is the noisiest, so it shrinks hardest; K/DST
#: are almost pure matchup-driven noise and shrink hardest of all.
POSITION_PRIOR_STRENGTH: dict[str, float] = {
    "QB": 4.0,
    "RB": 5.0,
    "WR": 6.0,
    "TE": 8.0,
    "K": 10.0,
    "DST": 10.0,
}

#: Age (years) at which the breakout model's "still developing" youth bonus
#: reaches zero, per position. RBs are drafted after less college workload
#: erosion and hit their role ceiling earliest; QBs keep developing (or at
#: least keep starting jobs on mechanics rather than legs) far longer.
POSITION_YOUTH_CEILING: dict[str, float] = {
    "QB": 32.0,
    "RB": 26.0,
    "WR": 28.0,
    "TE": 29.0,
    "K": 30.0,
    "DST": 30.0,
}

#: Age at which the bust model's aging-decline risk starts ramping, and the
#: number of years the ramp takes to reach its full penalty. RBs fall off a
#: workload cliff quickly after ~27; QBs decline gradually into their late
#: 30s; TEs (late bloomers who often peak in their late 20s) decline later
#: than WRs but faster than QBs once they start.
POSITION_DECLINE_ONSET: dict[str, float] = {
    "QB": 33.0,
    "RB": 27.0,
    "WR": 30.0,
    "TE": 31.0,
    "K": 33.0,
    "DST": 33.0,
}

POSITION_DECLINE_SPAN: dict[str, float] = {
    "QB": 9.0,
    "RB": 4.0,
    "WR": 6.0,
    "TE": 6.0,
    "K": 8.0,
    "DST": 8.0,
}

POSITION_USAGE_BASELINES: dict[str, float] = {
    "QB": 36.0,
    "RB": 16.0,
    "WR": 7.5,
    "TE": 5.5,
    "K": 5.0,
    "DST": 1.0,
}

POSITION_REPLACEMENT_BASELINES: dict[str, float] = {
    "QB": 255.0,
    "RB": 145.0,
    "WR": 140.0,
    "TE": 105.0,
    "K": 105.0,
    "DST": 100.0,
}

INJURY_MULTIPLIERS: dict[str, float] = {
    "IR": 0.08,
    "OUT": 0.15,
    "SUSPENDED": 0.20,
    "PUP": 0.35,
    "NFI": 0.35,
    "DOUBTFUL": 0.48,
    "QUESTIONABLE": 0.88,
    "PROBABLE": 0.98,
    "ACTIVE": 1.0,
    "HEALTHY": 1.0,
    "": 1.0,
}

_PLAYER_REGISTRY: dict[str, dict[str, Any]] = {}


def _mapping(value: Any, *, argument: str = "player") -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"{argument} must be a mapping or object with fields")


def _identity(player: Mapping[str, Any]) -> str:
    value = player.get("player_id") or player.get("id") or player.get("name") or player.get("player_name")
    return str(value or "").strip()


def _position(player: Mapping[str, Any]) -> str:
    position = str(player.get("position") or player.get("player_position") or "").strip().upper()
    return "DST" if position in {"DEF", "D/ST"} else position


def _finite(value: Any, default: float = 0.0) -> float:
    number = safe_float(value, default)
    return number if math.isfinite(number) else default


def _probability(value: Any, default: float = 0.0) -> float:
    number = _finite(value, default)
    if number > 1.0:
        number /= 100.0
    return clamp(number, 0.0, 1.0)


def _numeric_sequence(value: Any) -> list[float]:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return []
    if not isinstance(value, Iterable):
        return []
    numbers: list[float] = []
    for item in value:
        if isinstance(item, Mapping):
            raw = next(
                (
                    item.get(key)
                    for key in (
                        "fantasy_points",
                        "points",
                        "projection",
                        "value",
                        "score",
                    )
                    if item.get(key) is not None
                ),
                None,
            )
        else:
            raw = item
        if raw is None or isinstance(raw, bool):
            continue
        number = _finite(raw, float("nan"))
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def _history(player: Mapping[str, Any]) -> list[float]:
    for key in (
        "historical_points",
        "season_history",
        "weekly_points",
        "game_log",
        "history",
        "recent_points",
    ):
        values = _numeric_sequence(player.get(key))
        if values:
            return values
    return []


def _raw_projection(player: Mapping[str, Any], scoring_mode: str = "ppr") -> float:
    for key in (
        "projection",
        "expected_fantasy_points",
        "projected_points",
        "season_projection",
        "points",
    ):
        raw = player.get(key)
        if raw is not None and not isinstance(raw, bool):
            value = _finite(raw)
            if value > 0:
                return value
    scored = calculate_fantasy_points(dict(player), mode=scoring_mode, bonuses=False)["total_points"]
    if scored > 0:
        games = max(1.0, _finite(player.get("expected_games"), _finite(player.get("games_played"), 1.0)))
        return float(scored) * (REGULAR_SEASON_GAMES / games if games <= 3 else 1.0)
    history = _history(player)
    return history[-1] if history else 0.0


def _season_scale(player: Mapping[str, Any], value: float) -> float:
    """Scale per-game history to the source record's season projection units."""
    baseline = _raw_projection(player)
    history = _history(player)
    if not history or baseline <= 0:
        return value
    typical = statistics.fmean(history)
    if typical <= 0:
        return value
    # Weekly histories tend to be below 40 while season projections are much
    # larger. Season histories are already on the baseline's scale.
    return value * REGULAR_SEASON_GAMES if baseline / typical >= 5.0 else value


def _result(model: str, player: Mapping[str, Any], estimate: float, **components: Any) -> dict[str, Any]:
    return {
        "model": model,
        "player_id": _identity(player),
        "estimate": round(max(0.0, _finite(estimate)), 3),
        "components": components,
    }


def configure_projection_data(players: Iterable[Any], *, replace: bool = True) -> int:
    """Register normalized player records for id-only projection calls."""
    if replace:
        _PLAYER_REGISTRY.clear()
    count = 0
    for source in players or []:
        try:
            player = _mapping(source)
        except TypeError:
            continue
        player_id = _identity(player)
        if not player_id:
            continue
        _PLAYER_REGISTRY[player_id] = player
        _PLAYER_REGISTRY.setdefault(player_id.casefold(), player)
        name = str(player.get("name") or player.get("player_name") or "").strip()
        if name:
            _PLAYER_REGISTRY.setdefault(name.casefold(), player)
        count += 1
    return count


def _resolve_player(player_id: Any, player_data: Any = None) -> dict[str, Any]:
    if isinstance(player_id, Mapping) or hasattr(player_id, "model_dump") or hasattr(player_id, "__dict__"):
        return _mapping(player_id)

    requested = str(player_id or "").strip()
    if not requested:
        raise ValueError("player_id must not be empty")

    if player_data is not None:
        if isinstance(player_data, Mapping):
            if requested in player_data and isinstance(player_data[requested], Mapping):
                return dict(player_data[requested])
            candidates: Sequence[Any] = list(player_data.values())
        elif isinstance(player_data, Sequence) and not isinstance(player_data, (str, bytes, bytearray)):
            candidates = player_data
        else:
            raise TypeError("player_data must be a mapping or player sequence")
        for candidate in candidates:
            try:
                row = _mapping(candidate)
            except TypeError:
                continue
            if requested.casefold() in {
                _identity(row).casefold(),
                str(row.get("name") or row.get("player_name") or "").strip().casefold(),
            }:
                return row

    registered = _PLAYER_REGISTRY.get(requested) or _PLAYER_REGISTRY.get(requested.casefold())
    if registered is not None:
        return dict(registered)

    try:
        from quant.data_loader import load_all_player_data

        loaded = load_all_player_data()
    except (ImportError, OSError, TypeError, ValueError):
        loaded = []
    if loaded:
        configure_projection_data(loaded, replace=False)
        registered = _PLAYER_REGISTRY.get(requested) or _PLAYER_REGISTRY.get(requested.casefold())
        if registered is not None:
            return dict(registered)
    raise KeyError(f"No normalized data is available for player {requested!r}")


def weighted_historical_model(
    player: Any,
    *,
    decay: float = 0.72,
    scoring_mode: str = "ppr",
) -> dict[str, Any]:
    """Return a recency-weighted historical estimate.

    History is ordered oldest-to-newest. Missing history falls back to the
    source projection so sparse feeds remain usable without fabricated games.
    """
    row = _mapping(player)
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in the interval (0, 1]")
    values = _history(row)
    if not values:
        estimate = _raw_projection(row, scoring_mode)
        return _result("weighted_historical", row, estimate, samples=0, decay=decay, weights=[])
    weights = [decay ** (len(values) - index - 1) for index in range(len(values))]
    weighted = sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights)
    estimate = _season_scale(row, weighted)
    return _result(
        "weighted_historical",
        row,
        estimate,
        samples=len(values),
        decay=decay,
        weights=[round(weight, 5) for weight in weights],
        raw_weighted_average=round(weighted, 3),
    )


def compute_usage_rate(player: Any) -> float:
    """Return a normalized 0-1 opportunity rate from common usage signals."""
    row = _mapping(player)
    # A canonical record always carries a ``usage_rate`` key (default 0.0),
    # so ``is not None`` can't distinguish "no signal" from "explicit zero".
    # Treat a positive value as the explicit signal; a non-positive one
    # falls through to the opportunity-based estimate below.
    explicit = next(
        (
            row.get(key)
            for key in ("usage_rate", "opportunity_share", "snap_share")
            if _finite(row.get(key)) > 0
        ),
        None,
    )
    if explicit is not None:
        return round(_probability(explicit), 4)

    position = _position(row)
    if position == "QB":
        opportunities = _finite(row.get("pass_attempts")) + _finite(row.get("carries"))
    elif position == "RB":
        opportunities = _finite(row.get("carries")) + _finite(row.get("targets"))
    elif position in {"WR", "TE"}:
        opportunities = _finite(row.get("targets")) or _finite(row.get("receptions")) / 0.67
    else:
        opportunities = _finite(row.get("opportunities"), _finite(row.get("touches")))
    # POSITION_USAGE_BASELINES are per-game figures (an RB baseline of 16.0 is
    # ~16 carries in a game), but carries/targets/pass_attempts are commonly
    # supplied as season totals. Bring a season total back to a per-game rate
    # before comparing against the baseline -- otherwise every real starter
    # saturates to the same 1.0 regardless of actual volume. A true single-
    # game stat line (games_played absent or 1) is already on that scale.
    games = _finite(row.get("games_played"), 1.0)
    if games > 1.0:
        opportunities = opportunities / games
    baseline = POSITION_USAGE_BASELINES.get(position, 12.0)
    if opportunities <= 0:
        return 0.5
    return round(clamp(opportunities / max(baseline * 1.6, 1.0), 0.0, 1.0), 4)


def usage_based_projection_model(
    player: Any,
    *,
    base_projection: float | None = None,
    scoring_mode: str = "ppr",
) -> dict[str, Any]:
    """Adjust a projection around neutral usage, bounded to avoid overfit."""
    row = _mapping(player)
    base = max(0.0, _finite(base_projection, _raw_projection(row, scoring_mode)))
    usage_rate = compute_usage_rate(row)
    route_rate = _probability(row.get("route_participation"), usage_rate)
    red_zone_rate = _probability(row.get("red_zone_share"), usage_rate)
    composite = 0.62 * usage_rate + 0.23 * route_rate + 0.15 * red_zone_rate
    multiplier = clamp(0.72 + composite * 0.58, 0.72, 1.30)
    return _result(
        "usage_based",
        row,
        base * multiplier,
        base_projection=round(base, 3),
        usage_rate=round(usage_rate, 4),
        route_participation=round(route_rate, 4),
        red_zone_share=round(red_zone_rate, 4),
        multiplier=round(multiplier, 4),
    )


def _matchup_multiplier(player: Mapping[str, Any]) -> float:
    explicit = next(
        (
            player.get(key)
            for key in (
                "matchup_multiplier",
                "defensive_adjustment",
                "matchup_adjustment",
            )
            if player.get(key) is not None
        ),
        None,
    )
    if explicit is not None:
        return clamp(_finite(explicit, 1.0), 0.75, 1.25)
    score = player.get("matchup_score")
    if score is not None:
        normalized = _finite(score, 50.0)
        normalized = normalized * 100.0 if 0.0 <= normalized <= 1.0 else normalized
        return clamp(0.80 + normalized / 250.0, 0.80, 1.20)
    rank = player.get("opponent_defense_rank", player.get("defense_rank"))
    if rank is not None:
        normalized_rank = clamp(_finite(rank, 16.5), 1.0, 32.0)
        return clamp(0.82 + (normalized_rank - 1.0) / 31.0 * 0.36, 0.82, 1.18)
    return 1.0


def matchup_based_projection_model(
    player: Any,
    *,
    base_projection: float | None = None,
    scoring_mode: str = "ppr",
) -> dict[str, Any]:
    """Apply bounded opponent and weather context to a baseline."""
    row = _mapping(player)
    base = max(0.0, _finite(base_projection, _raw_projection(row, scoring_mode)))
    matchup = _matchup_multiplier(row)
    weather = clamp(_finite(row.get("weather_multiplier"), 1.0), 0.85, 1.08)
    pace = clamp(_finite(row.get("pace_multiplier"), 1.0), 0.90, 1.10)
    combined = clamp(matchup * weather * pace, 0.75, 1.25)
    return _result(
        "matchup_based",
        row,
        base * combined,
        base_projection=round(base, 3),
        matchup_multiplier=round(matchup, 4),
        weather_multiplier=round(weather, 4),
        pace_multiplier=round(pace, 4),
        multiplier=round(combined, 4),
    )


def _volatility(player: Mapping[str, Any], base: float) -> float:
    explicit = player.get("volatility", player.get("volatility_score"))
    if explicit is not None:
        return _probability(explicit)
    floor = player.get("floor")
    ceiling = player.get("ceiling")
    if floor is not None and ceiling is not None and base > 0:
        return clamp((_finite(ceiling) - _finite(floor)) / max(base, 1.0), 0.0, 1.0)
    history = _history(player)
    if len(history) >= 2:
        mean = statistics.fmean(history)
        return clamp(statistics.pstdev(history) / max(abs(mean), 1.0), 0.0, 1.0)
    confidence = _probability(player.get("projection_confidence"), 0.65)
    return clamp(1.0 - confidence, 0.08, 0.65)


def volatility_adjusted_projection_model(
    player: Any,
    *,
    base_projection: float | None = None,
    risk_tolerance: str = "balanced",
    scoring_mode: str = "ppr",
) -> dict[str, Any]:
    """Return floor/median/ceiling and a risk-adjusted estimate."""
    row = _mapping(player)
    base = max(0.0, _finite(base_projection, _raw_projection(row, scoring_mode)))
    volatility = _volatility(row, base)
    risk_weights = {"safe": -0.30, "balanced": -0.08, "aggressive": 0.16, "boom_bust": 0.16}
    if risk_tolerance not in risk_weights:
        raise ValueError(f"risk_tolerance must be one of {sorted(risk_weights)}")
    adjustment = 1.0 + volatility * risk_weights[risk_tolerance]
    floor = max(0.0, _finite(row.get("floor"), base * (1.0 - 0.70 * volatility)))
    ceiling = max(base, _finite(row.get("ceiling"), base * (1.0 + 0.90 * volatility)))
    result = _result(
        "volatility_adjusted",
        row,
        base * adjustment,
        base_projection=round(base, 3),
        volatility=round(volatility, 4),
        risk_tolerance=risk_tolerance,
        multiplier=round(adjustment, 4),
        floor=round(floor, 3),
        median=round(base, 3),
        ceiling=round(ceiling, 3),
    )
    result.update({"volatility": round(volatility, 4), "floor": round(floor, 3), "median": round(base, 3), "ceiling": round(ceiling, 3)})
    return result


def injury_adjusted_projection_model(
    player: Any,
    *,
    base_projection: float | None = None,
    scoring_mode: str = "ppr",
) -> dict[str, Any]:
    """Discount projection for current status and expected availability."""
    row = _mapping(player)
    base = max(0.0, _finite(base_projection, _raw_projection(row, scoring_mode)))
    status = str(row.get("injury_status") or row.get("status") or "").strip().upper()
    status_multiplier = INJURY_MULTIPLIERS.get(status, 0.92 if status else 1.0)
    expected_games = clamp(
        _finite(row.get("expected_games"), REGULAR_SEASON_GAMES),
        0.0,
        float(REGULAR_SEASON_GAMES),
    )
    availability = expected_games / REGULAR_SEASON_GAMES
    # A season projection that already embeds expected games should not take
    # the full availability discount twice. Half-weight keeps injury context
    # useful while preserving the supplied baseline.
    availability_multiplier = 0.5 + 0.5 * availability
    combined = clamp(status_multiplier * availability_multiplier, 0.0, 1.0)
    return _result(
        "injury_adjusted",
        row,
        base * combined,
        base_projection=round(base, 3),
        injury_status=status or "ACTIVE",
        expected_games=round(expected_games, 2),
        status_multiplier=round(status_multiplier, 4),
        availability_multiplier=round(availability_multiplier, 4),
        multiplier=round(combined, 4),
    )


def regression_to_mean_model(
    player: Any,
    *,
    base_projection: float | None = None,
    position_mean: float | None = None,
    prior_strength: float = 6.0,
    scoring_mode: str = "ppr",
) -> dict[str, Any]:
    """Shrink small-sample production toward a positional prior."""
    row = _mapping(player)
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")
    base = max(0.0, _finite(base_projection, _raw_projection(row, scoring_mode)))
    anchor = max(
        0.0,
        _finite(
            position_mean,
            _finite(row.get("position_mean"), POSITION_REPLACEMENT_BASELINES.get(_position(row), base)),
        ),
    )
    # An unstated sample size defaults to a full season *only* when there is
    # some real baseline to have accumulated it from. A record with neither a
    # games-played count nor any baseline signal (a true zero-stat rookie) is
    # not "a full season of nothing" -- it's simply unobserved, and should
    # shrink fully to the positional anchor rather than being nearly ignored.
    default_sample = REGULAR_SEASON_GAMES if base > 0 else 0.0
    sample = clamp(
        _finite(row.get("games_played"), _finite(row.get("sample_size"), default_sample)),
        0.0,
        float(REGULAR_SEASON_GAMES),
    )
    denominator = sample + prior_strength
    estimate = (sample * base + prior_strength * anchor) / denominator if denominator else base
    return _result(
        "regression_to_mean",
        row,
        estimate,
        base_projection=round(base, 3),
        position_mean=round(anchor, 3),
        sample_size=round(sample, 2),
        prior_strength=round(prior_strength, 2),
        player_weight=round(sample / denominator, 4) if denominator else 1.0,
    )


def breakout_bust_probability_model(player: Any) -> dict[str, Any]:
    """Estimate bounded breakout and bust probabilities from role and risk."""
    row = _mapping(player)
    position = _position(row)
    base = _raw_projection(row)
    usage = compute_usage_rate(row)
    age = _finite(row.get("age"), 26.0)
    experience = _finite(row.get("years_experience"), _finite(row.get("experience"), 4.0))
    adp = max(1.0, _finite(row.get("adp"), 180.0))
    trend = clamp(_finite(row.get("momentum_score"), _finite(row.get("trend_score"), 0.0)) / 100.0, -1.0, 1.0)
    volatility = _volatility(row, base)
    confidence = _probability(row.get("projection_confidence"), 0.62)
    health = INJURY_MULTIPLIERS.get(str(row.get("injury_status") or "").upper(), 0.92)

    youth_ceiling = POSITION_YOUTH_CEILING.get(position, 29.0)
    decline_onset = POSITION_DECLINE_ONSET.get(position, 29.0)
    decline_span = POSITION_DECLINE_SPAN.get(position, 7.0)
    youth = clamp((youth_ceiling - age) / 8.0, 0.0, 1.0)
    opportunity_growth = _probability(row.get("opportunity_growth"), max(0.0, trend))
    market_discount = clamp((adp - 36.0) / 144.0, 0.0, 1.0)
    breakout = clamp(
        0.05
        + 0.24 * usage
        + 0.18 * youth
        + 0.17 * opportunity_growth
        + 0.13 * market_discount
        + 0.10 * max(trend, 0.0)
        - 0.10 * max(experience - 7.0, 0.0) / 8.0,
        0.01,
        0.92,
    )
    bust = clamp(
        0.04
        + 0.30 * volatility
        + 0.24 * (1.0 - health)
        + 0.18 * (1.0 - confidence)
        + 0.12 * max(-trend, 0.0)
        + 0.07 * clamp((age - decline_onset) / decline_span, 0.0, 1.0),
        0.01,
        0.92,
    )
    return {
        "model": "breakout_bust_probability",
        "player_id": _identity(row),
        "breakout_probability": round(breakout, 4),
        "bust_probability": round(bust, 4),
        "components": {
            "usage_rate": round(usage, 4),
            "youth": round(youth, 4),
            "opportunity_growth": round(opportunity_growth, 4),
            "market_discount": round(market_discount, 4),
            "volatility": round(volatility, 4),
            "confidence": round(confidence, 4),
            "health_multiplier": round(health, 4),
        },
    }


def compute_final_projection(
    player_id: Any,
    player_data: Any = None,
    *,
    position_means: Mapping[str, Any] | None = None,
    scoring_mode: str = "ppr",
    risk_tolerance: str = "balanced",
    model_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the auditable ensemble projection for one player.

    ``player_id`` may itself be a player mapping. When it is an id/name,
    ``player_data`` can be a sequence or lookup mapping; otherwise the
    configured registry and offline normalized loader are consulted.
    """
    player = _resolve_player(player_id, player_data)
    baseline = _raw_projection(player, scoring_mode)
    historical = weighted_historical_model(player, scoring_mode=scoring_mode)
    usage = usage_based_projection_model(player, base_projection=baseline, scoring_mode=scoring_mode)
    matchup = matchup_based_projection_model(player, base_projection=baseline, scoring_mode=scoring_mode)
    position = _position(player)
    position_mean = None
    if position_means is not None and position in position_means:
        position_mean = _finite(position_means[position])
    regression = regression_to_mean_model(
        player,
        base_projection=baseline,
        position_mean=position_mean,
        prior_strength=POSITION_PRIOR_STRENGTH.get(position, 6.0),
        scoring_mode=scoring_mode,
    )
    market = max(0.0, _finite(player.get("market_projection"), baseline))

    configured_weights = dict(POSITION_MODEL_WEIGHTS.get(position, MODEL_WEIGHTS))
    if model_weights:
        for key, value in model_weights.items():
            if key in configured_weights:
                configured_weights[key] = max(0.0, _finite(value))
    total_weight = sum(configured_weights.values())
    if total_weight <= 0:
        raise ValueError("model_weights must contain at least one positive weight")
    configured_weights = {key: value / total_weight for key, value in configured_weights.items()}
    estimates = {
        "historical": historical["estimate"],
        "usage": usage["estimate"],
        "matchup": matchup["estimate"],
        "regression": regression["estimate"],
        "market": market,
    }

    # A model with no real input still returns a value -- it falls back to
    # the shared baseline -- so including it at full weight would silently
    # double-count the baseline instead of contributing independent signal.
    # Drop models with no genuine signal and redistribute their weight
    # across the ones that do have something to say. Usage and matchup are
    # purely multiplicative against the baseline (``base * multiplier``), so
    # a zero baseline (a true rookie with no stat-derived projection) forces
    # them to exactly 0 no matter how real their multiplier is -- that's a
    # degenerate result, not a genuine "zero projection" signal. Regression
    # is exempt: it shrinks toward a positional anchor and stays informative
    # even at a zero baseline.
    signal_present = {
        "historical": historical["components"].get("samples", 0) > 0,
        "usage": usage["components"].get("base_projection", 0.0) > 0,
        "matchup": (
            matchup["components"].get("base_projection", 0.0) > 0
            and (
                abs(matchup["components"].get("matchup_multiplier", 1.0) - 1.0) > 1e-9
                or abs(matchup["components"].get("weather_multiplier", 1.0) - 1.0) > 1e-9
                or abs(matchup["components"].get("pace_multiplier", 1.0) - 1.0) > 1e-9
            )
        ),
        "regression": True,
        "market": player.get("market_projection") is not None,
    }
    active_weights = {key: weight for key, weight in configured_weights.items() if signal_present.get(key, True) and weight > 0}
    if not active_weights:
        active_weights = dict(configured_weights)
    active_total = sum(active_weights.values())
    active_weights = {key: weight / active_total for key, weight in active_weights.items()}
    ensemble = sum(estimates[key] * active_weights.get(key, 0.0) for key in estimates)
    injury = injury_adjusted_projection_model(player, base_projection=ensemble, scoring_mode=scoring_mode)
    volatility = volatility_adjusted_projection_model(
        player,
        base_projection=injury["estimate"],
        risk_tolerance=risk_tolerance,
        scoring_mode=scoring_mode,
    )
    probabilities = breakout_bust_probability_model(player)

    probability_multiplier = clamp(
        1.0 + 0.10 * probabilities["breakout_probability"] - 0.08 * probabilities["bust_probability"],
        0.90,
        1.08,
    )
    final_projection = max(0.0, volatility["estimate"] * probability_multiplier)

    # The five sub-models' own spread is a free uncertainty signal: when they
    # agree, the ensemble should be more confident than the inputs alone
    # suggest; when they diverge widely, less so.
    active_estimates = [estimates[key] for key in active_weights]
    disagreement = (
        clamp(statistics.pstdev(active_estimates) / max(ensemble, 1.0), 0.0, 1.0)
        if len(active_estimates) >= 2 and ensemble > 0
        else 0.0
    )
    confidence = _probability(player.get("projection_confidence"), 0.62)
    evidence = clamp(_finite(historical["components"].get("samples")) / REGULAR_SEASON_GAMES, 0.0, 1.0)
    confidence = clamp(
        0.58 * confidence
        + 0.15 * evidence
        + 0.12 * (1.0 - volatility["volatility"])
        + 0.15 * (1.0 - disagreement),
        0.05,
        0.99,
    )

    spread = max(final_projection * (0.10 + 0.55 * volatility["volatility"]), 0.5 if final_projection else 0.0)
    # Outcomes aren't symmetric: a bust-prone player's downside is a deeper
    # floor cut, and a breakout-prone player's upside is a taller ceiling --
    # reuse the probabilities already computed above instead of a fixed split.
    floor_ratio = 0.45 + 0.10 * probabilities["bust_probability"]
    ceiling_ratio = 0.65 + 0.20 * probabilities["breakout_probability"]
    floor = max(0.0, min(volatility["floor"], final_projection - spread * floor_ratio))
    ceiling = max(final_projection, max(volatility["ceiling"], final_projection + spread * ceiling_ratio))

    # Missed-games risk is a floor event (the bad outcome is "got hurt and
    # played little"), not a median one -- the injury model's availability
    # already discounted the ensemble's center, so apply a second, floor-only
    # discount instead of double-counting it against the median.
    games_availability = clamp(
        _finite(injury["components"].get("expected_games"), REGULAR_SEASON_GAMES) / REGULAR_SEASON_GAMES, 0.0, 1.0
    )
    floor = floor * (0.55 + 0.45 * games_availability)

    return {
        "player_id": _identity(player),
        "name": str(player.get("name") or player.get("player_name") or _identity(player)),
        "position": position,
        "team": str(player.get("team") or player.get("nfl_team") or "").upper(),
        "scoring_mode": scoring_mode,
        "final_projection": round(final_projection, 2),
        "projection": round(final_projection, 2),
        "expected_fantasy_points": round(final_projection, 2),
        "floor": round(floor, 2),
        "median": round(final_projection, 2),
        "ceiling": round(ceiling, 2),
        "confidence": round(confidence, 4),
        "projection_confidence": round(confidence, 4),
        "volatility": round(volatility["volatility"], 4),
        "breakout_probability": probabilities["breakout_probability"],
        "bust_probability": probabilities["bust_probability"],
        "health_multiplier": injury["components"]["multiplier"],
        "matchup_multiplier": matchup["components"]["multiplier"],
        "usage_rate": usage["components"]["usage_rate"],
        "model_weights": {key: round(value, 4) for key, value in active_weights.items()},
        "components": {
            "baseline": round(baseline, 3),
            "ensemble_before_risk": round(ensemble, 3),
            "probability_multiplier": round(probability_multiplier, 4),
            "historical": historical,
            "usage": usage,
            "matchup": matchup,
            "regression": regression,
            "market": {"estimate": round(market, 3)},
            "injury": injury,
            "volatility": volatility,
            "probabilities": probabilities,
            "configured_model_weights": {key: round(value, 4) for key, value in configured_weights.items()},
            "active_models": sorted(active_weights),
            "disagreement": round(disagreement, 4),
            "games_availability": round(games_availability, 4),
        },
    }


class ProjectionEngine:
    """Reusable projection service backed by an immutable caller-owned pool."""

    def __init__(
        self,
        players: Iterable[Any] | Mapping[str, Any] | None = None,
        *,
        scoring_mode: str = "ppr",
        position_means: Mapping[str, Any] | None = None,
    ) -> None:
        self.scoring_mode = scoring_mode
        self.position_means = dict(position_means or {})
        if isinstance(players, Mapping):
            self.players: Any = dict(players)
        else:
            self.players = [_mapping(player) for player in (players or [])]

    def compute_final_projection(self, player_id: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("scoring_mode", self.scoring_mode)
        kwargs.setdefault("position_means", self.position_means)
        return compute_final_projection(player_id, self.players, **kwargs)

    def project_all(self, **kwargs: Any) -> list[dict[str, Any]]:
        sources = list(self.players.values()) if isinstance(self.players, Mapping) else list(self.players)
        results = [self.compute_final_projection(source, **kwargs) for source in sources]
        results.sort(key=lambda result: result["final_projection"], reverse=True)
        return results


# Discoverable aliases for callers that prefer shorter model names.
weighted_historical_projection = weighted_historical_model
usage_based_projection = usage_based_projection_model
matchup_based_projection = matchup_based_projection_model
volatility_adjusted_projection = volatility_adjusted_projection_model
injury_adjusted_projection = injury_adjusted_projection_model
regression_to_mean_projection = regression_to_mean_model
breakout_bust_model = breakout_bust_probability_model


__all__ = [
    "INJURY_MULTIPLIERS",
    "MODEL_WEIGHTS",
    "POSITION_REPLACEMENT_BASELINES",
    "POSITION_USAGE_BASELINES",
    "ProjectionEngine",
    "breakout_bust_model",
    "breakout_bust_probability_model",
    "compute_final_projection",
    "compute_usage_rate",
    "configure_projection_data",
    "injury_adjusted_projection",
    "injury_adjusted_projection_model",
    "matchup_based_projection",
    "matchup_based_projection_model",
    "regression_to_mean_model",
    "regression_to_mean_projection",
    "usage_based_projection",
    "usage_based_projection_model",
    "volatility_adjusted_projection",
    "volatility_adjusted_projection_model",
    "weighted_historical_model",
    "weighted_historical_projection",
]
