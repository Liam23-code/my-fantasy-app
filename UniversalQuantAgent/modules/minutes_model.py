"""Explainable NBA minutes projection built from recent role and game context."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modules.data_quality import (
    coerce_numeric,
    parse_minutes,
    safe_dict,
    safe_get,
    safe_number,
    window_average,
)
from modules.injury_parser import get_player_availability, load_injury_data_from_file
from modules.nba_cache import FALLBACK_WARNING, select_fallback_minutes
from modules.projections import (
    _find_player,
    _find_team,
    _game_log_with_fuzzy_fallback,
    latest_season,
)


def _game_data(
    player_name: str, season: str
) -> tuple[dict, pd.DataFrame, list[str]]:
    player = _find_player(player_name)
    resolved, raw, warnings = _game_log_with_fuzzy_fallback(
        player_name, player, season
    )
    games = coerce_numeric(raw).copy()
    if "minutes" in games:
        source_minutes = games["minutes"]
    else:
        source_minutes = games.get("min", pd.Series(np.nan, index=games.index))
    games["minutes"] = source_minutes.map(parse_minutes)
    if "game_date" in games:
        games["game_date"] = pd.to_datetime(
            games["game_date"], errors="coerce"
        )
        games = games.sort_values("game_date")
    return resolved, games, warnings


def _blowout_risk(team_id: int | None, opponent_id: int, season: str) -> float:
    try:
        from modules.nba_advanced import fetch_league_team_stats

        table = fetch_league_team_stats(season)
        net = table.set_index("TEAM_ID")["NET_RATING"]
        difference = abs(
            float(net.get(team_id, 0)) - float(net.get(opponent_id, 0))
        )
        return round(min(100.0, difference * 4.0), 1)
    except Exception:
        return 35.0


def project_minutes(
    player_name: str,
    opponent_team: str,
    season: str | None = None,
) -> dict[str, Any]:
    """Project minutes without imposing an artificial output floor or ceiling.

    During fallback operation, the baseline is selected in this strict order:
    season average, last 10, then last 5. A missing value never becomes zero.
    Context adjustments remain visible so callers can distinguish the selected
    baseline from the final expected-minutes result.
    """
    season = season or latest_season()
    opponent = _find_team(opponent_team)
    player, games, warnings = _game_data(player_name, season)
    minutes = pd.to_numeric(games["minutes"], errors="coerce")
    minutes = minutes.where(minutes > 0).dropna()
    if minutes.empty:
        raise ValueError(
            f"No season, last-10, or last-5 minutes are available for "
            f"{player_name} in {season}."
        )

    season_average = safe_number(minutes.mean(), np.nan)
    last_10 = window_average(minutes, 10, np.nan)
    last_5 = window_average(minutes, 5, np.nan)
    fallback_minutes, fallback_source = select_fallback_minutes(
        season_average, last_10, last_5
    )
    fallback_active = FALLBACK_WARNING in warnings

    if fallback_active:
        baseline_minutes = fallback_minutes
    else:
        baseline_minutes = (
            0.50 * season_average + 0.30 * last_10 + 0.20 * last_5
        )

    role_change = last_5 - last_10
    variation = safe_number(minutes.tail(10).std(ddof=0))
    stability = 1.0 - variation / 12.0
    projection = baseline_minutes + role_change * 0.35

    availability = safe_dict(get_player_availability(player["full_name"], load_injury_data_from_file()))
    availability.setdefault("availability", "ACTIVE")
    availability.setdefault("impact_score", 0.0)
    injury_factor = 1.0 - safe_number(
        safe_get(availability, "impact_score")
    ) / 100.0
    projection *= injury_factor

    team_id = (
        int(games["team_id"].dropna().iloc[-1])
        if "team_id" in games and games["team_id"].notna().any()
        else None
    )
    blowout = _blowout_risk(team_id, opponent["id"], season)
    if blowout > 50.0:
        projection *= 1.0 - (blowout - 50.0) / 1000.0

    back_to_back = False
    if "game_date" in games and games["game_date"].notna().any():
        rest_days = (
            pd.Timestamp.now().normalize()
            - games["game_date"].dropna().max().normalize()
        ).days
        back_to_back = rest_days <= 1
        if back_to_back:
            projection -= 0.6

    spread = safe_number(minutes.tail(10).std(ddof=0))
    source_quality = {
        "season_average": 92.0,
        "last_10": 76.0,
        "last_5": 62.0,
    }[fallback_source]
    fallback_quality = source_quality if fallback_active else 100.0
    confidence = 60.0 + stability * 30.0 - variation * 2.0
    confidence = round(max(0.0, min(100.0, confidence)), 1)

    drivers = [
        f"Season-average minutes: {season_average:.1f}.",
        f"Last-five minutes: {last_5:.1f}; last-ten: {last_10:.1f}.",
        f"Selected baseline: {baseline_minutes:.1f} ({fallback_source}).",
        f"Recent role change: {role_change:+.1f} minutes.",
        f"Rotation stability proxy: {stability * 100:.0f}%.",
    ]
    if safe_get(availability, "availability", "ACTIVE") != "ACTIVE":
        drivers.append(
            f"Injury status is {safe_get(availability, 'availability')}."
        )
    if back_to_back:
        drivers.append("Back-to-back scheduling reduces the estimate by 0.6.")
    if blowout >= 60:
        drivers.append("Elevated blowout risk can shorten starter minutes.")

    value = round(projection, 1)
    confidence_low = round(projection - 1.645 * spread, 1)
    confidence_high = round(projection + 1.645 * spread, 1)
    fallback_status = {
        "active": fallback_active,
        "source": fallback_source if fallback_active else "live_blend",
        "quality": round(fallback_quality, 1),
    }
    details = {
        "season_average_minutes": round(season_average, 1),
        "rolling_minutes": {
            "last_5": round(last_5, 1),
            "last_10": round(last_10, 1),
        },
        "fallback_minutes": round(fallback_minutes, 1),
        "fallback_minutes_source": fallback_source,
        "fallback_minutes_quality": round(fallback_quality, 1),
        "fallback_status": fallback_status,
        "role_change": round(role_change, 1),
        "back_to_back": back_to_back,
        "coach_tendency_score": round(stability * 100, 1),
        "blowout_risk": blowout,
        "availability": availability,
        "key_drivers": drivers,
        "warnings": warnings,
    }
    return {
        "value": value,
        "confidence": confidence,
        "details": details,
        "player": player["full_name"],
        "opponent": opponent["abbreviation"],
        "season": season,
        "minutes_projection": value,
        "confidence_low": confidence_low,
        "confidence_high": confidence_high,
        **details,
    }