"""Matchup pace projection using season, opponent, league, and recent rates."""
from __future__ import annotations

from typing import Any

import pandas as pd

from modules.data_quality import (
    pace_fallback,
    safe_dict,
    safe_get,
    safe_list,
    safe_number,
)
from modules.injury_parser import load_injury_data_from_file
from modules.nba_advanced import fetch_league_team_stats, find_team, latest_season
from modules.nba_cache import collect_warnings


def _row(table: pd.DataFrame, team_id: int) -> pd.Series:
    rows = table[table["TEAM_ID"] == team_id]
    if rows.empty:
        raise ValueError("No team pace data was returned.")
    return rows.iloc[0]


def project_pace(
    team_one: str,
    team_two: str,
    season: str | None = None,
) -> dict[str, Any]:
    """Blend team, opponent, league, and last-ten pace transparently."""
    season = season or latest_season()
    first, second = find_team(team_one), find_team(team_two)
    season_table = fetch_league_team_stats(season)
    try:
        recent_table = fetch_league_team_stats(season, last_n_games=10)
    except Exception:
        recent_table = season_table

    first_row = _row(season_table, first["id"])
    second_row = _row(season_table, second["id"])
    first_recent = _row(recent_table, first["id"])
    second_recent = _row(recent_table, second["id"])
    warnings = collect_warnings(season_table, recent_table)

    league_average = pace_fallback(
        pd.to_numeric(season_table.get("PACE"), errors="coerce").mean(),
        league_average=100.0,
    )
    team_pace = pace_fallback(first_row.get("PACE"), league_average)
    opponent_pace = pace_fallback(second_row.get("PACE"), league_average)
    recent_team_pace = pace_fallback(
        first_recent.get("PACE"), team_pace, league_average
    )
    recent_opponent_pace = pace_fallback(
        second_recent.get("PACE"), opponent_pace, league_average
    )

    # The season blend retains an explicit league anchor. The matchup factor is
    # the requested possessions ratio and is not clipped or otherwise bounded.
    season_blend = (
        0.40 * team_pace
        + 0.40 * opponent_pace
        + 0.20 * league_average
    )
    recent_matchup = (recent_team_pace + recent_opponent_pace) / 2.0
    projection = 0.65 * season_blend + 0.35 * recent_matchup
    pace_factor = (
        (team_pace + opponent_pace) / 2.0 / league_average
    )

    try:
        report = safe_list(load_injury_data_from_file())
    except Exception:
        report = []
    team_codes = {first["abbreviation"], second["abbreviation"]}
    affected = sum(
        safe_get(safe_dict(row), "status") in {"OUT", "QUESTIONABLE"}
        and safe_get(safe_dict(row), "team") in team_codes
        for row in report
    )
    injury_adjustment = -0.15 * affected
    projection += injury_adjustment

    disagreement = abs(team_pace - opponent_pace)
    recent_disagreement = abs(season_blend - recent_matchup)
    confidence = 92.0 - disagreement * 3.0 - recent_disagreement * 2.0
    if warnings:
        confidence -= 12.0
    confidence = round(max(0.0, min(100.0, confidence)), 1)

    league_median = pace_fallback(
        pd.to_numeric(season_table.get("PACE"), errors="coerce").median(),
        league_average,
    )
    style = (
        "fast"
        if projection >= league_median + 1
        else "slow"
        if projection < league_median - 1
        else "balanced"
    )
    value = round(projection, 1)
    details = {
        "team_pace": {
            first["abbreviation"]: round(team_pace, 1),
            second["abbreviation"]: round(opponent_pace, 1),
        },
        "team_pace_value": round(team_pace, 3),
        "opponent_pace_value": round(opponent_pace, 3),
        "league_average_pace": round(league_average, 3),
        "season_blend": round(season_blend, 3),
        "recent_pace": round(recent_matchup, 1),
        "pace_factor": round(pace_factor, 4),
        "matchup_style": style,
        "injury_adjustment": round(injury_adjustment, 2),
        "warnings": warnings,
        "explanation": (
            "Pace blends 40% team, 40% opponent, and 20% league pace, "
            "then incorporates both teams' last-ten pace."
        ),
    }
    return {
        "value": value,
        "confidence": confidence,
        "details": details,
        "teams": [first["abbreviation"], second["abbreviation"]],
        "season": season,
        "pace_projection": value,
        "confidence_low": round(projection - 2.0, 1),
        "confidence_high": round(projection + 2.0, 1),
        **details,
    }