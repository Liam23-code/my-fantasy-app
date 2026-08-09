"""Fuse independent NBA models into one explainable statline projection."""
from __future__ import annotations

from math import isfinite
from typing import Any

from modules.context_engine import get_player_context
from modules.data_quality import (
    safe_dict,
    safe_get,
    safe_list,
    safe_number,
    safe_scalar_to_dict,
)
from modules.nba_cache import (
    ENHANCED_FALLBACK_WARNING,
    FALLBACK_WARNING,
    select_fallback_minutes,
    weighted_available_average,
)
from modules.projections import latest_season, project_player_statline

STATS = ("points", "rebounds", "assists", "pra")
MODEL_VALUE_KEYS = (
    "minutes_projection",
    "pace_projection",
    "difficulty_score",
    "value",
)


def normalize_model_output(x: Any) -> dict[str, Any]:
    """Return a stable numeric model envelope for mappings and scalars."""
    payload = safe_dict(x).copy()
    if not payload:
        payload = safe_scalar_to_dict(x)
    raw_value = None
    for key in MODEL_VALUE_KEYS:
        candidate = safe_get(payload, key)
        if candidate is not None:
            raw_value = candidate
            break
    payload["value"] = safe_number(raw_value)
    payload["confidence"] = max(
        0.0,
        min(100.0, safe_number(safe_get(payload, "confidence"), 50.0)),
    )
    payload["details"] = safe_dict(safe_get(payload, "details"))
    return payload


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) and number > 0 else None


def _nested_value(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    value = safe_get(payload, key)
    if value is not None:
        return value
    return safe_get(safe_dict(safe_get(payload, "details")), key, default)


def _component_confidence(payload: dict[str, Any], default: float = 50.0) -> float:
    return max(
        0.0,
        min(100.0, safe_number(safe_get(payload, "confidence"), default)),
    )


def _normalized_weights(scores: dict[str, float]) -> dict[str, float]:
    """Normalize reliability inputs to four weights that sum exactly to one."""
    positive_scores = {
        key: max(0.0, safe_number(value)) for key, value in scores.items()
    }
    total = sum(positive_scores.values())
    if total == 0:
        return {key: 1.0 / len(positive_scores) for key in positive_scores}
    return {key: value / total for key, value in positive_scores.items()}


def _profile_value(
    profile: dict[str, Any],
    window: str,
    stat: str,
    fallback: Any = None,
) -> Any:
    return safe_get(safe_dict(safe_get(profile, window)), stat, fallback)


def fuse_projection(
    player_name: str,
    opponent_team: str,
    season: str | None = None,
) -> dict[str, Any]:
    """Fuse projections with reliability-weighted, fully expressive factors."""
    season = season or latest_season()
    base = safe_dict(project_player_statline(player_name, opponent_team, season))
    base_statline = safe_dict(safe_get(base, "projected_statline"))
    if not base_statline:
        raise ValueError("Base projection returned an invalid statline payload.")

    context = safe_dict(get_player_context(player_name, opponent_team, season))
    components = safe_dict(safe_get(context, "components"))
    minutes = normalize_model_output(safe_get(components, "minutes"))
    pace = normalize_model_output(safe_get(components, "pace"))
    matchup = normalize_model_output(safe_get(components, "matchup"))
    availability = safe_dict(safe_get(components, "availability")).copy()
    if not availability:
        availability = safe_scalar_to_dict(safe_get(components, "availability"))

    profile = safe_dict(safe_get(base, "feature_summary"))
    season_profile = safe_dict(safe_get(profile, "season_average"))
    if not season_profile:
        season_profile = safe_dict(safe_get(base, "season_averages"))
    recent_10_profile = safe_dict(safe_get(profile, "last_10"))
    recent_5_profile = safe_dict(safe_get(profile, "last_5"))
    base_recent_10 = safe_dict(safe_get(base, "recent_10_game_averages"))
    base_recent_5 = safe_dict(safe_get(base, "recent_5_game_averages"))

    rolling_minutes = safe_get(minutes, "rolling_minutes")
    rolling_minutes_dict = safe_dict(rolling_minutes)
    rolling_last_10 = (
        safe_get(rolling_minutes_dict, "last_10")
        if rolling_minutes_dict
        else rolling_minutes
    )
    rolling_last_5 = (
        safe_get(rolling_minutes_dict, "last_5")
        if rolling_minutes_dict
        else None
    )
    season_minutes = _nested_value(
        minutes,
        "season_average_minutes",
        season_profile.get("minutes"),
    )
    last_10_minutes = _positive(rolling_last_10) or _positive(
        recent_10_profile.get("minutes")
    )
    last_5_minutes = _positive(rolling_last_5) or _positive(
        recent_5_profile.get("minutes")
    )
    projected_minutes = _positive(
        safe_get(minutes, "minutes_projection", safe_get(minutes, "value"))
    )
    try:
        baseline_minutes, minutes_source = select_fallback_minutes(
            season_minutes, last_10_minutes, last_5_minutes
        )
    except ValueError:
        if projected_minutes is None:
            raise ValueError(
                "Fusion requires season, last-10, last-5, or projected minutes."
            )
        baseline_minutes = projected_minutes
        minutes_source = "model_projection"
    if projected_minutes is None:
        projected_minutes = baseline_minutes
    minutes_factor = projected_minutes / baseline_minutes
    minutes_quality = safe_number(
        _nested_value(minutes, "fallback_minutes_quality"),
        100.0 if minutes_source == "season_average" else 76.0,
    )

    usage_profile = safe_dict(safe_get(profile, "usage"))
    season_usage = _positive(
        safe_get(usage_profile, "season", season_profile.get("usage"))
    )
    last_10_usage = _positive(
        safe_get(usage_profile, "last_10", recent_10_profile.get("usage"))
    )
    last_5_usage = _positive(
        safe_get(usage_profile, "last_5", recent_5_profile.get("usage"))
    )
    if any(value is not None for value in (season_usage, last_10_usage, last_5_usage)):
        blended_usage, usage_coverage = weighted_available_average([
            (season_usage, .50),
            (last_10_usage, .30),
            (last_5_usage, .20),
        ])
        usage_baseline = season_usage or blended_usage
    else:
        blended_usage, usage_baseline, usage_coverage = 1.0, 1.0, 0.5
    opponent_usage_factor = safe_number(
        _nested_value(matchup, "opponent_usage_factor"), 1.0
    )
    league_pace = _positive(_nested_value(pace, "league_average_pace"))
    team_pace = _positive(_nested_value(pace, "team_pace_value"))
    opponent_pace = _positive(_nested_value(pace, "opponent_pace_value"))
    explicit_pace_factor = _positive(_nested_value(pace, "pace_factor"))
    if explicit_pace_factor is not None:
        pace_factor = explicit_pace_factor
    elif league_pace and team_pace and opponent_pace:
        pace_factor = (team_pace + opponent_pace) / 2.0 / league_pace
    else:
        pace_factor = safe_number(
            safe_get(pace, "pace_projection", safe_get(pace, "value")),
            100.0,
        ) / 100.0
    adjusted_usage = blended_usage * opponent_usage_factor
    adjusted_usage *= pace_factor
    usage_factor = adjusted_usage / usage_baseline
    usage_quality = safe_number(
        safe_get(usage_profile, "quality"), usage_coverage * 100.0
    )

    efficiency_profile = safe_dict(safe_get(profile, "efficiency"))
    season_ts = _positive(
        safe_get(
            efficiency_profile,
            "season",
            season_profile.get("true_shooting_pct"),
        )
    )
    last_10_ts = _positive(
        safe_get(
            efficiency_profile,
            "last_10",
            recent_10_profile.get("true_shooting_pct"),
        )
    )
    last_5_ts = _positive(
        safe_get(
            efficiency_profile,
            "last_5",
            recent_5_profile.get("true_shooting_pct"),
        )
    )
    if any(value is not None for value in (season_ts, last_10_ts, last_5_ts)):
        blended_ts, efficiency_coverage = weighted_available_average([
            (season_ts, .50),
            (last_10_ts, .30),
            (last_5_ts, .20),
        ])
        efficiency_baseline = season_ts or blended_ts
    else:
        blended_ts, efficiency_baseline, efficiency_coverage = 1.0, 1.0, 0.5
    opponent_def_eff_factor = safe_number(
        _nested_value(matchup, "opponent_def_eff_factor"), 1.0
    )
    adjusted_ts = blended_ts * opponent_def_eff_factor
    efficiency_factor = adjusted_ts / efficiency_baseline
    efficiency_quality = safe_number(
        safe_get(efficiency_profile, "quality"),
        efficiency_coverage * 100.0,
    )

    difficulty = safe_number(
        safe_get(matchup, "difficulty_score", safe_get(matchup, "value")),
        50.0,
    )
    matchup_factor = 1.0 + (50.0 - difficulty) / 500.0
    injury_factor = 1.0 - safe_number(
        safe_get(availability, "impact_score")
    ) / 100.0
    context_factor = pace_factor * matchup_factor * injury_factor
    context_confidence = (
        _component_confidence(pace)
        + _component_confidence(matchup)
        + max(
            0.0,
            min(100.0, safe_number(context.get("role_stability"), 50.0)),
        )
    ) / 3.0

    weights = _normalized_weights({
        "weight_minutes": _component_confidence(minutes),
        "weight_usage": usage_quality,
        "weight_efficiency": efficiency_quality,
        "weight_context": context_confidence,
    })
    context["components"] = {
        "minutes": minutes,
        "pace": pace,
        "matchup": matchup,
        "availability": availability,
    }

    final: dict[str, float] = {}
    ranges: dict[str, dict[str, float]] = {}
    breakdown: dict[str, dict[str, float]] = {}
    usage_sensitivity = {"points": 1.0, "rebounds": .15, "assists": .55}
    efficiency_sensitivity = {"points": 1.0, "rebounds": .10, "assists": .10}

    for stat in ("points", "rebounds", "assists"):
        raw = safe_number(safe_get(base_statline, stat))
        recent_10 = safe_number(
            safe_get(base_recent_10, stat),
            safe_get(recent_10_profile, stat, raw),
        )
        recent_5 = safe_number(
            safe_get(base_recent_5, stat),
            safe_get(recent_5_profile, stat, recent_10),
        )
        season_average = safe_number(season_profile.get(stat), raw)
        recent_context = 0.50 * raw + 0.30 * recent_10 + 0.20 * recent_5

        minutes_candidate = raw * minutes_factor
        usage_candidate = raw * (
            1.0 + (usage_factor - 1.0) * usage_sensitivity[stat]
        )
        efficiency_candidate = raw * (
            1.0 + (efficiency_factor - 1.0) * efficiency_sensitivity[stat]
        )
        context_candidate = recent_context * context_factor
        weighted_sum = (
            weights["weight_minutes"] * minutes_candidate
            + weights["weight_usage"] * usage_candidate
            + weights["weight_efficiency"] * efficiency_candidate
            + weights["weight_context"] * context_candidate
        )
        value = (weighted_sum + season_average) / 2.0

        source_range = safe_dict(
            safe_get(safe_dict(safe_get(base, "confidence_ranges")), stat)
        )
        source_low = safe_number(source_range.get("low"), raw)
        source_high = safe_number(source_range.get("high"), raw)
        half_width = abs(source_high - source_low) / 2.0
        combined_reliability = sum(
            weights[key]
            * {
                "weight_minutes": _component_confidence(minutes),
                "weight_usage": usage_quality,
                "weight_efficiency": efficiency_quality,
                "weight_context": context_confidence,
            }[key]
            for key in weights
        ) / 100.0
        uncertainty = half_width * (2.0 - combined_reliability)

        final[stat] = round(value, 1)
        ranges[stat] = {
            "low": round(value - uncertainty, 1),
            "high": round(value + uncertainty, 1),
        }
        breakdown[stat] = {
            "base_regression": round(raw, 3),
            "minutes_candidate": round(minutes_candidate, 3),
            "usage_candidate": round(usage_candidate, 3),
            "efficiency_candidate": round(efficiency_candidate, 3),
            "context_candidate": round(context_candidate, 3),
            "weighted_sum": round(weighted_sum, 3),
            "season_average": round(season_average, 3),
            "smoothed_projection": round(value, 3),
        }

    final["pra"] = round(
        final["points"] + final["rebounds"] + final["assists"], 1
    )
    pra_uncertainty = sum(
        (ranges[stat]["high"] - ranges[stat]["low"]) / 2.0
        for stat in ("points", "rebounds", "assists")
    )
    ranges["pra"] = {
        "low": round(final["pra"] - pra_uncertainty, 1),
        "high": round(final["pra"] + pra_uncertainty, 1),
    }

    base_fallback = safe_dict(safe_get(base, "fallback_status"))
    base_warnings = [str(item) for item in safe_list(safe_get(base, "warnings"))]
    fallback_active = bool(safe_get(base_fallback, "active")) or any(
        warning in {FALLBACK_WARNING, ENHANCED_FALLBACK_WARNING}
        for warning in base_warnings
    )
    fallback_status = {
        "active": fallback_active,
        "source": safe_get(base_fallback, "source", "live"),
        "warning": ENHANCED_FALLBACK_WARNING if fallback_active else "",
    }
    input_breakdown = {
        "minutes": {
            "season_average": _positive(season_minutes),
            "last_10": _positive(last_10_minutes),
            "last_5": _positive(last_5_minutes),
            "selected_baseline": round(baseline_minutes, 3),
            "selected_source": minutes_source,
            "projection": round(projected_minutes, 3),
            "factor": round(minutes_factor, 4),
            "quality": round(minutes_quality, 1),
        },
        "usage": {
            "season_average": season_usage,
            "last_10": last_10_usage,
            "last_5": last_5_usage,
            "blended": round(blended_usage, 3),
            "opponent_usage_factor": round(opponent_usage_factor, 4),
            "pace_factor": round(pace_factor, 4),
            "adjusted": round(adjusted_usage, 3),
            "factor": round(usage_factor, 4),
            "quality": round(usage_quality, 1),
        },
        "efficiency": {
            "season_average_ts_pct": season_ts,
            "last_10_ts_pct": last_10_ts,
            "last_5_ts_pct": last_5_ts,
            "blended_ts_pct": round(blended_ts, 3),
            "opponent_def_eff_factor": round(opponent_def_eff_factor, 4),
            "adjusted_ts_pct": round(adjusted_ts, 3),
            "factor": round(efficiency_factor, 4),
            "quality": round(efficiency_quality, 1),
        },
        "context": {
            "pace_factor": round(pace_factor, 4),
            "matchup_factor": round(matchup_factor, 4),
            "injury_factor": round(injury_factor, 4),
            "combined_factor": round(context_factor, 4),
            "quality": round(context_confidence, 1),
        },
    }
    drivers = [
        f"Minutes factor: {minutes_factor:.3f}",
        f"Usage factor: {usage_factor:.3f}",
        f"Efficiency factor: {efficiency_factor:.3f}",
        f"Pace factor: {pace_factor:.3f}",
        f"Matchup factor: {matchup_factor:.3f}",
        f"Availability factor: {injury_factor:.3f}",
        "Fusion weights: "
        + ", ".join(
            f"{key.replace('weight_', '')} {value:.1%}"
            for key, value in weights.items()
        ),
    ]
    return {
        "player": base.get("player", player_name),
        "team": context.get("team", ""),
        "opponent": opponent_team,
        "season": season,
        "final_projection": final,
        "confidence_range": ranges,
        "driver_breakdown": breakdown,
        "fusion_weights": {key: round(value, 4) for key, value in weights.items()},
        "input_breakdown": input_breakdown,
        "fallback_status": fallback_status,
        "key_drivers": drivers,
        "base_projection": base,
        "context": context,
    }