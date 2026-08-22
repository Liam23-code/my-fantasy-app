"""Advanced waiver-wire opportunity, trend, and priority analytics.

The engine ranks what a player can become, not merely what the player scored
last week.  Opportunity, changing usage, matchup, uncertainty, roster need,
and the season projection are independently exposed so a recommendation can
be explained and audited.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .scarcity_engine import (
    DEFAULT_ROSTER_SLOTS,
    _as_mapping,
    _clamp,
    _coerce_players,
    _finite_number,
    _projection,
)
from .trade_engine import INACTIVE_STATUSES, team_need_score


def _normalize_score(value: Any, *, default: float | None = None) -> float | None:
    number = _finite_number(value)
    if number is None:
        return default
    if 0.0 <= number <= 1.0:
        number *= 100.0
    return _clamp(number, 0.0, 100.0)


def _normalize_share(value: Any) -> float | None:
    number = _finite_number(value)
    if number is None:
        return None
    if abs(number) <= 1.0:
        number *= 100.0
    return _clamp(number, 0.0, 100.0)


def _single_player(player: Any) -> dict[str, Any]:
    rows = _coerce_players(player)
    if len(rows) != 1:
        raise ValueError("player must resolve to exactly one player")
    return rows[0]


def _weighted_available(components: Sequence[tuple[float | None, float]]) -> tuple[float, float]:
    available = [(float(value), weight) for value, weight in components if value is not None and weight > 0]
    if not available:
        return 50.0, 0.0
    total_weight = sum(weight for _, weight in available)
    score = sum(value * weight for value, weight in available) / total_weight
    configured_weight = sum(weight for _, weight in components)
    return _clamp(score, 0.0, 100.0), _clamp(total_weight / max(configured_weight, 1e-9), 0.0, 1.0)


def _entry_usage(entry: Any) -> float | None:
    direct = _finite_number(entry)
    if direct is not None:
        return _normalize_score(direct)
    try:
        row = _as_mapping(entry, label="usage history entry")
    except TypeError:
        return None
    direct_score = _normalize_score(row.get("usage_rate", row.get("usage_score")))
    snap = _normalize_share(row.get("snap_share", row.get("snap_pct")))
    route = _normalize_share(row.get("route_participation", row.get("route_share")))
    target = _normalize_share(row.get("target_share"))
    carry = _normalize_share(row.get("carry_share", row.get("rush_share")))
    opportunity_share = _normalize_share(row.get("opportunity_share"))
    touches = _finite_number(row.get("opportunities", row.get("touches")))
    if touches is None:
        targets = _finite_number(row.get("targets"), 0.0) or 0.0
        carries = _finite_number(row.get("carries", row.get("rush_attempts")), 0.0) or 0.0
        touches = targets + carries if targets or carries else None
    volume = _clamp(touches / 25.0 * 100.0, 0.0, 100.0) if touches is not None else None
    score, _ = _weighted_available(
        [
            (direct_score, 0.35),
            (snap, 0.2),
            (route, 0.12),
            (opportunity_share, 0.18),
            (target, 0.08),
            (carry, 0.08),
            (volume, 0.14),
        ]
    )
    return score


def _usage_history(row: Mapping[str, Any]) -> list[float]:
    raw: Any = None
    for key in ("usage_history", "weekly_usage", "recent_usage", "weekly_stats", "game_log", "history"):
        if row.get(key) is not None:
            raw = row[key]
            break
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        def sort_key(item: Any) -> tuple[int, float | str]:
            number = _finite_number(item)
            return (0, number) if number is not None else (1, str(item))

        entries = [raw[key] for key in sorted(raw, key=sort_key)]
    elif isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        entries = list(raw)
    else:
        entries = [raw]
    values = [_entry_usage(entry) for entry in entries]
    return [float(value) for value in values if value is not None]


def usage_trend(player: Any) -> dict[str, Any]:
    """Measure recent usage level and linear direction on a 0--100 scale."""

    row = _single_player(player)
    history = _usage_history(row)
    if not history:
        current = _entry_usage(row)
        history = [current] if current is not None else []
    if not history:
        return {
            "player_id": row["player_id"],
            "score": 50.0,
            "direction": "flat",
            "slope": 0.0,
            "change": 0.0,
            "current_usage": 50.0,
            "previous_usage": 50.0,
            "sample_size": 0,
            "data_coverage": 0.0,
        }

    count = len(history)
    current = sum(history[-3:]) / min(3, count)
    previous_slice = history[:-3] if count > 3 else history[: max(1, count // 2)]
    previous = sum(previous_slice) / len(previous_slice) if previous_slice else current
    if count >= 2:
        x_mean = (count - 1) / 2.0
        y_mean = sum(history) / count
        denominator = sum((index - x_mean) ** 2 for index in range(count))
        slope = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(history)) / denominator if denominator else 0.0
    else:
        slope = 0.0
    change = current - previous
    score = _clamp(50.0 + slope * 4.0 + change * 0.7 + (current - 50.0) * 0.25, 0.0, 100.0)
    direction = "up" if slope >= 1.5 or change >= 5.0 else "down" if slope <= -1.5 or change <= -5.0 else "flat"
    return {
        "player_id": row["player_id"],
        "score": round(score, 2),
        "direction": direction,
        "slope": round(slope, 3),
        "change": round(change, 3),
        "current_usage": round(current, 2),
        "previous_usage": round(previous, 2),
        "sample_size": count,
        "data_coverage": round(min(1.0, count / 6.0), 3),
    }


def opportunity_score(player: Any) -> dict[str, Any]:
    """Estimate present and near-future role opportunity."""

    row = _single_player(player)
    direct = _normalize_score(row.get("raw_opportunity_score", row.get("opportunity_score")))
    snap = _normalize_share(row.get("snap_share", row.get("snap_pct")))
    route = _normalize_share(row.get("route_participation", row.get("route_share")))
    opportunity_share = _normalize_share(row.get("opportunity_share"))
    target_share = _normalize_share(row.get("target_share"))
    carry_share = _normalize_share(row.get("carry_share", row.get("rush_share")))
    red_zone_share = _normalize_share(row.get("red_zone_share", row.get("red_zone_opportunity_share")))
    touches = _finite_number(row.get("opportunities", row.get("touches")))
    if touches is None:
        targets = _finite_number(row.get("targets"), 0.0) or 0.0
        carries = _finite_number(row.get("carries", row.get("rush_attempts")), 0.0) or 0.0
        touches = targets + carries if targets or carries else None
    volume = _clamp(touches / 25.0 * 100.0, 0.0, 100.0) if touches is not None else None
    depth_rank = _finite_number(row.get("depth_chart_rank", row.get("depth_rank")))
    depth_score = _clamp(110.0 - 30.0 * depth_rank, 0.0, 100.0) if depth_rank is not None and depth_rank > 0 else None
    vacated = _normalize_share(row.get("vacated_opportunity_share", row.get("vacated_targets_pct")))
    role_change = _normalize_score(row.get("role_change_score"))
    if row.get("starter_ahead_injured") is True or row.get("depth_chart_opportunity") is True:
        role_change = max(role_change or 0.0, 85.0)

    score, coverage = _weighted_available(
        [
            (direct, 0.25),
            (snap, 0.17),
            (route, 0.1),
            (opportunity_share, 0.15),
            (target_share, 0.08),
            (carry_share, 0.08),
            (red_zone_share, 0.07),
            (volume, 0.15),
            (depth_score, 0.08),
            (vacated, 0.08),
            (role_change, 0.1),
        ]
    )
    return {
        "player_id": row["player_id"],
        "score": round(score, 2),
        "data_coverage": round(coverage, 3),
        "components": {
            "snap_share": round(snap, 2) if snap is not None else None,
            "route_participation": round(route, 2) if route is not None else None,
            "opportunity_share": round(opportunity_share, 2) if opportunity_share is not None else None,
            "target_share": round(target_share, 2) if target_share is not None else None,
            "carry_share": round(carry_share, 2) if carry_share is not None else None,
            "red_zone_share": round(red_zone_share, 2) if red_zone_share is not None else None,
            "volume": round(volume, 2) if volume is not None else None,
            "depth_chart": round(depth_score, 2) if depth_score is not None else None,
            "vacated_opportunity": round(vacated, 2) if vacated is not None else None,
            "role_change": round(role_change, 2) if role_change is not None else None,
        },
    }


def _efficiency_score(row: Mapping[str, Any]) -> tuple[float, float]:
    direct = _normalize_score(row.get("efficiency_score"))
    yards_per_touch = _finite_number(row.get("yards_per_touch"))
    yards_per_route = _finite_number(row.get("yards_per_route_run", row.get("yards_per_route")))
    fantasy_per_touch = _finite_number(row.get("fantasy_points_per_touch", row.get("points_per_opportunity")))
    target_rate = _normalize_share(row.get("targets_per_route", row.get("target_rate")))
    ypt_score = _clamp(yards_per_touch / 7.0 * 100.0, 0.0, 100.0) if yards_per_touch is not None else None
    yprr_score = _clamp(yards_per_route / 3.0 * 100.0, 0.0, 100.0) if yards_per_route is not None else None
    fpt_score = _clamp(fantasy_per_touch / 1.5 * 100.0, 0.0, 100.0) if fantasy_per_touch is not None else None
    return _weighted_available([(direct, 0.45), (ypt_score, 0.2), (yprr_score, 0.2), (fpt_score, 0.2), (target_rate, 0.12)])


def breakout_probability(player: Any) -> dict[str, Any]:
    """Estimate breakout probability from role, trend, efficiency, and age."""

    row = _single_player(player)
    explicit = _normalize_score(row.get("breakout_probability", row.get("breakout_prob")))
    opportunity = opportunity_score(row)
    trend = usage_trend(row)
    efficiency, efficiency_coverage = _efficiency_score(row)
    age = _finite_number(row.get("age"))
    experience = _finite_number(row.get("experience", row.get("years_exp")))
    age_score = None
    if age is not None:
        age_score = 85.0 if age <= 23 else 75.0 if age <= 25 else 55.0 if age <= 27 else 30.0 if age <= 30 else 15.0
    if experience is not None:
        experience_score = 85.0 if experience <= 1 else 70.0 if experience <= 3 else 45.0 if experience <= 6 else 25.0
        age_score = (age_score + experience_score) / 2.0 if age_score is not None else experience_score
    projection_delta = _finite_number(row.get("projection_delta", row.get("recent_projection_delta")))
    projection = _projection(row)
    delta_score = _clamp(50.0 + projection_delta / max(projection, 1.0) * 200.0, 0.0, 100.0) if projection_delta is not None else None
    confidence = _normalize_score(row.get("projection_confidence", row.get("confidence")))
    score, coverage = _weighted_available(
        [
            (explicit, 0.3),
            (opportunity["score"], 0.28),
            (trend["score"], 0.2),
            (efficiency, 0.18 if efficiency_coverage else 0.0),
            (age_score, 0.1),
            (delta_score, 0.14),
            (confidence, 0.05),
        ]
    )
    # Low-coverage estimates shrink toward a sober 50% prior.
    adjusted = 50.0 + (score - 50.0) * (0.55 + 0.45 * coverage)
    return {
        "player_id": row["player_id"],
        "probability": round(_clamp(adjusted / 100.0, 0.0, 1.0), 4),
        "score": round(_clamp(adjusted, 0.0, 100.0), 2),
        "data_coverage": round(coverage, 3),
        "components": {
            "opportunity": opportunity["score"],
            "usage_trend": trend["score"],
            "efficiency": round(efficiency, 2),
            "age_curve": round(age_score, 2) if age_score is not None else None,
            "projection_delta": round(delta_score, 2) if delta_score is not None else None,
        },
    }


def _week_row(row: Mapping[str, Any], week: int | None) -> dict[str, Any]:
    if week is None:
        return dict(row)
    for key in ("schedule", "weekly_matchups", "matchups", "weekly_projections"):
        schedule = row.get(key)
        if not isinstance(schedule, Mapping):
            continue
        candidate = schedule.get(week, schedule.get(str(week)))
        if isinstance(candidate, Mapping):
            return {**row, **candidate}
        if isinstance(candidate, str):
            return {**row, "opponent": candidate}
    return dict(row)


def matchup_advantage(player: Any, week: int | None = None) -> dict[str, Any]:
    """Score the scheduled opponent from 0 (worst) to 100 (best)."""

    if week is not None and (isinstance(week, bool) or int(week) != week or not 1 <= int(week) <= 18):
        raise ValueError("week must be an integer from 1 through 18")
    base = _single_player(player)
    row = _week_row(base, int(week) if week is not None else None)
    bye = week is not None and _finite_number(row.get("bye_week", row.get("bye"))) == float(week)
    opponent = str(row.get("opponent", row.get("opp", "")) or "").strip().upper()
    if bye or opponent in {"BYE", "OFF"}:
        return {
            "player_id": base["player_id"],
            "week": week,
            "opponent": "BYE",
            "score": 0.0,
            "label": "bye",
            "data_coverage": 1.0,
            "components": {},
        }

    direct = _normalize_score(row.get("weekly_matchup_score", row.get("matchup_score")))
    defense_rank = _finite_number(row.get("defense_rank_vs_position", row.get("opponent_defense_rank")))
    rank_score = _clamp((defense_rank - 1.0) / 31.0 * 100.0, 0.0, 100.0) if defense_rank is not None else None
    schedule_difficulty = _normalize_score(row.get("schedule_difficulty", row.get("matchup_difficulty")))
    difficulty_score = 100.0 - schedule_difficulty if schedule_difficulty is not None else None
    adjustment = _finite_number(row.get("defensive_adjustment", row.get("matchup_multiplier")))
    adjustment_score = _clamp(50.0 + (adjustment - 1.0) * 200.0, 0.0, 100.0) if adjustment is not None else None
    allowed = _finite_number(row.get("opponent_points_allowed", row.get("fantasy_points_allowed")))
    league_average = _finite_number(row.get("league_average_points_allowed"))
    allowed_score = (
        _clamp(50.0 + (allowed / league_average - 1.0) * 100.0, 0.0, 100.0)
        if allowed is not None and league_average is not None and league_average > 0
        else None
    )
    score, coverage = _weighted_available(
        [(direct, 0.35), (rank_score, 0.25), (difficulty_score, 0.2), (adjustment_score, 0.15), (allowed_score, 0.15)]
    )
    label = "elite" if score >= 75 else "favorable" if score >= 60 else "neutral" if score >= 40 else "difficult" if score >= 25 else "avoid"
    return {
        "player_id": base["player_id"],
        "week": week,
        "opponent": opponent or None,
        "score": round(score, 2),
        "label": label,
        "data_coverage": round(coverage, 3),
        "components": {
            "direct_matchup": round(direct, 2) if direct is not None else None,
            "defense_rank": round(rank_score, 2) if rank_score is not None else None,
            "schedule_difficulty": round(difficulty_score, 2) if difficulty_score is not None else None,
            "defensive_adjustment": round(adjustment_score, 2) if adjustment_score is not None else None,
            "points_allowed": round(allowed_score, 2) if allowed_score is not None else None,
        },
    }


def _point_history(row: Mapping[str, Any]) -> list[float]:
    raw = row.get("weekly_points", row.get("points_history", row.get("fantasy_points_history")))
    if raw is None:
        weekly = row.get("weekly_projections")
        if isinstance(weekly, Mapping):
            raw = [weekly[key] for key in sorted(weekly, key=lambda value: (_finite_number(value) is None, _finite_number(value) or 0.0, str(value)))]
    if raw is None:
        return []
    entries = list(raw.values()) if isinstance(raw, Mapping) else list(raw) if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)) else [raw]
    points: list[float] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            entry = entry.get("points", entry.get("projection"))
        number = _finite_number(entry)
        if number is not None:
            points.append(number)
    return points


def volatility_profile(player: Any) -> dict[str, Any]:
    """Return a 0--100 volatility score and its observable drivers."""

    row = _single_player(player)
    projection = _projection(row)
    direct = _finite_number(row.get("volatility", row.get("projection_volatility")))
    if direct is not None:
        if direct <= 1.0:
            direct_score = direct * 100.0
        else:
            direct_score = _clamp(direct / max(projection, 1.0) * 100.0, 0.0, 100.0)
    else:
        direct_score = None
    history = _point_history(row)
    history_score = None
    if len(history) >= 2:
        mean = statistics.fmean(history)
        history_score = _clamp(statistics.pstdev(history) / max(abs(mean), 1.0) * 100.0, 0.0, 100.0)
    floor = _finite_number(row.get("floor"))
    ceiling = _finite_number(row.get("ceiling"))
    range_score = (
        _clamp((ceiling - floor) / max(2.0 * projection, 1.0) * 100.0, 0.0, 100.0)
        if floor is not None and ceiling is not None and ceiling >= floor
        else None
    )
    confidence = _normalize_score(row.get("projection_confidence", row.get("confidence")))
    uncertainty_score = 100.0 - confidence if confidence is not None else None
    score, coverage = _weighted_available(
        [(direct_score, 0.35), (history_score, 0.35), (range_score, 0.2), (uncertainty_score, 0.15)]
    )
    label = "low" if score < 30 else "moderate" if score < 55 else "high" if score < 75 else "extreme"
    return {
        "player_id": row["player_id"],
        "score": round(score, 2),
        "label": label,
        "coefficient_of_variation": round(history_score / 100.0, 4) if history_score is not None else None,
        "sample_size": len(history),
        "data_coverage": round(coverage, 3),
        "components": {
            "declared_volatility": round(direct_score, 2) if direct_score is not None else None,
            "observed_variation": round(history_score, 2) if history_score is not None else None,
            "projection_range": round(range_score, 2) if range_score is not None else None,
            "confidence_uncertainty": round(uncertainty_score, 2) if uncertainty_score is not None else None,
        },
    }


def rank_waiver_priority(
    players: Any,
    *,
    team: Any = None,
    week: int | None = None,
    roster_requirements: Any = None,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    """Rank free agents by opportunity, upside, trend, matchup, and team fit."""

    if week is not None and (isinstance(week, bool) or int(week) != week or not 1 <= int(week) <= 18):
        raise ValueError("week must be an integer from 1 through 18")
    if max_results is not None and (isinstance(max_results, bool) or int(max_results) != max_results or max_results < 0):
        raise ValueError("max_results must be a non-negative integer or None")
    normalized = _coerce_players(players)
    if not normalized:
        return []
    projections = [player["projection"] for player in normalized]
    low_projection, high_projection = min(projections), max(projections)
    team_needs = team_need_score(team or [], roster_requirements=roster_requirements or DEFAULT_ROSTER_SLOTS)["by_position"]
    rows: list[dict[str, Any]] = []
    for player in normalized:
        opportunity = opportunity_score(player)
        breakout = breakout_probability(player)
        usage = usage_trend(player)
        matchup = matchup_advantage(player, week)
        volatility = volatility_profile(player)
        projection_score = (
            50.0
            if high_projection == low_projection
            else 100.0 * (player["projection"] - low_projection) / (high_projection - low_projection)
        )
        need = team_needs.get(player["position"], 35.0)
        # Waiver targets benefit from upside, but extreme volatility is a poor
        # claim profile.  The curve peaks around 60/100 volatility.
        volatility_fit = _clamp(100.0 - abs(volatility["score"] - 60.0) * 1.5, 0.0, 100.0)
        raw_score = (
            0.24 * opportunity["score"]
            + 0.20 * breakout["score"]
            + 0.18 * usage["score"]
            + 0.14 * matchup["score"]
            + 0.12 * projection_score
            + 0.08 * need
            + 0.04 * volatility_fit
        )
        status = str(player.get("injury_status", player.get("status", "")) or "").strip().upper()
        injury_factor = 0.35 if status in INACTIVE_STATUSES else 0.7 if status == "DOUBTFUL" else 0.9 if status == "QUESTIONABLE" else 1.0
        bye_factor = 0.75 if matchup["label"] == "bye" else 1.0
        priority = _clamp(raw_score * injury_factor * bye_factor, 0.0, 100.0)
        coverage = statistics.fmean(
            [opportunity["data_coverage"], breakout["data_coverage"], usage["data_coverage"], matchup["data_coverage"], volatility["data_coverage"]]
        )
        rows.append(
            {
                **player,
                "waiver_priority_score": round(priority, 2),
                "priority_score": round(priority, 2),
                "opportunity_score": opportunity["score"],
                "breakout_probability": breakout["probability"],
                "breakout_score": breakout["score"],
                "usage_trend_score": usage["score"],
                "usage_trend_direction": usage["direction"],
                "matchup_advantage": matchup["score"],
                "matchup_label": matchup["label"],
                "volatility_score": volatility["score"],
                "volatility_label": volatility["label"],
                "projection_score": round(projection_score, 2),
                "team_need_score": round(need, 2),
                "confidence": round(coverage, 3),
                "suggested_faab_pct": round(_clamp(priority * 0.35, 0.0, 35.0), 1),
                "rationale": (
                    f"Opportunity {opportunity['score']:.0f}/100; breakout {breakout['score']:.0f}/100; "
                    f"usage {usage['direction']} ({usage['score']:.0f}/100); "
                    f"matchup {matchup['label']}; volatility {volatility['label']}."
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["waiver_priority_score"],
            -row["opportunity_score"],
            -row["projection"],
            row["player_id"].casefold(),
        )
    )
    limit = len(rows) if max_results is None else int(max_results)
    selected = rows[:limit]
    for rank, row in enumerate(selected, start=1):
        row["waiver_rank"] = rank
        row["rank"] = rank
    return selected


def compute_waiver_priority(
    players: Any,
    *,
    team: Any = None,
    week: int | None = None,
    roster_requirements: Any = None,
    max_results: int | None = None,
) -> dict[str, Any]:
    """Return waiver rankings in the unified metric-envelope shape."""

    results = rank_waiver_priority(
        players,
        team=team,
        week=week,
        roster_requirements=roster_requirements,
        max_results=max_results,
    )
    return {
        "metric": "waiver_priority",
        "results": results,
        "by_player": {row["player_id"]: row for row in results},
        "metadata": {
            "player_count": len(results),
            "week": week,
            "method": "opportunity, breakout, usage, matchup, projection, need, and volatility",
        },
    }


# Public verb variants for straightforward product integration.
compute_opportunity_score = opportunity_score
compute_breakout_probability = breakout_probability
compute_usage_trend = usage_trend
compute_matchup_advantage = matchup_advantage
compute_volatility_profile = volatility_profile
waiver_priority_ranking = rank_waiver_priority
rank_waivers = rank_waiver_priority
waiver_priority = rank_waiver_priority
compute_waiver_priority_ranking = rank_waiver_priority
calculate_opportunity_score = opportunity_score
calculate_breakout_probability = breakout_probability
calculate_usage_trend = usage_trend
calculate_matchup_advantage = matchup_advantage
calculate_volatility_profile = volatility_profile


__all__ = [
    "breakout_probability",
    "calculate_breakout_probability",
    "calculate_matchup_advantage",
    "calculate_opportunity_score",
    "calculate_usage_trend",
    "calculate_volatility_profile",
    "compute_breakout_probability",
    "compute_matchup_advantage",
    "compute_opportunity_score",
    "compute_usage_trend",
    "compute_volatility_profile",
    "compute_waiver_priority",
    "compute_waiver_priority_ranking",
    "matchup_advantage",
    "opportunity_score",
    "rank_waiver_priority",
    "rank_waivers",
    "usage_trend",
    "volatility_profile",
    "waiver_priority_ranking",
    "waiver_priority",
]
