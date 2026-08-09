"""Explainable player/opponent defensive difficulty model."""
from __future__ import annotations

from typing import Any

import pandas as pd

from modules.data_quality import (
    coerce_numeric,
    defense_fallback,
    safe_dict,
    safe_get,
    safe_list,
    safe_number,
)
from modules.injury_model import fetch_injury_report
from modules.nba_advanced import fetch_league_team_stats, find_team, latest_season
from modules.nba_cache import collect_warnings
from modules.projections import _find_player, _game_log_with_fuzzy_fallback

# These modest team/role adjustments are only used when the public table lacks
# the position-proxy columns. Dynamic league percentiles take precedence.
POSITIONAL_TEAM_FALLBACKS = {
    "BOS": {"interior": 4.0, "primary ball handler": 2.0, "wing": 3.0},
    "MIL": {"interior": 1.0, "primary ball handler": 4.0, "wing": 2.0},
    "MIA": {"interior": 3.0, "primary ball handler": 3.0, "wing": 3.0},
    "MIN": {"interior": 5.0, "primary ball handler": 2.0, "wing": 3.0},
    "ORL": {"interior": 3.0, "primary ball handler": 3.0, "wing": 4.0},
}


def _percentile_adjustment(
    league: pd.DataFrame,
    row_index: Any,
    columns: list[str],
    scale: float,
) -> float | None:
    percentiles = []
    for column in columns:
        if column not in league:
            continue
        numeric = pd.to_numeric(league[column], errors="coerce")
        if numeric.notna().sum() < 2 or pd.isna(numeric.loc[row_index]):
            continue
        percentiles.append(float(numeric.rank(pct=True).loc[row_index]))
    if not percentiles:
        return None
    return (sum(percentiles) / len(percentiles) - 0.5) * scale


def project_matchup_difficulty(
    player_name: str,
    opponent_team: str,
    season: str | None = None,
) -> dict[str, Any]:
    """Score defensive difficulty from 0-100; higher is more difficult."""
    season = season or latest_season()
    opponent = find_team(opponent_team)
    league = fetch_league_team_stats(season)
    rows = league[league["TEAM_ID"] == opponent["id"]]
    if rows.empty:
        raise ValueError(f"No defensive data found for {opponent_team}.")
    row = rows.iloc[0]
    warnings: list[str] = collect_warnings(league)

    league_defense = defense_fallback(
        pd.to_numeric(league.get("DEF_RATING"), errors="coerce").mean(),
        league_average=110.0,
    )
    defense = defense_fallback(
        row.get("DEF_RATING"), league_defense, league_defense
    )
    opponent_def_eff_factor = defense / league_defense
    opponent_usage_factor = 1.0 + (opponent_def_eff_factor - 1.0) * 0.35

    # Lower defensive rating means better defense and therefore greater
    # difficulty. This ranking is the core 0-100 score before context.
    defense_ranks = pd.to_numeric(
        league["DEF_RATING"], errors="coerce"
    ).rank(method="average", ascending=False, pct=True)
    difficulty = float(defense_ranks.loc[row.name]) * 100.0

    history = None
    player_role = "wing"
    try:
        player = _find_player(player_name)
        resolved, raw, player_warnings = _game_log_with_fuzzy_fallback(
            player_name, player, season
        )
        warnings.extend(player_warnings)
        games = coerce_numeric(raw)
        assists = safe_number(
            pd.to_numeric(
                games.get("ast", pd.Series(dtype=float)), errors="coerce"
            ).mean()
        )
        rebounds = safe_number(
            pd.to_numeric(
                games.get("reb", pd.Series(dtype=float)), errors="coerce"
            ).mean()
        )
        player_role = (
            "primary ball handler"
            if assists >= 6
            else "interior"
            if rebounds >= 8
            else "wing"
        )
        if "matchup" in games:
            versus = games[
                games["matchup"].astype(str).str.contains(
                    opponent["abbreviation"], case=False, na=False
                )
            ]
            if not versus.empty and "pts" in versus:
                history = round(
                    safe_number(
                        pd.to_numeric(versus["pts"], errors="coerce").mean()
                    ),
                    1,
                )
                overall = safe_number(
                    pd.to_numeric(games["pts"], errors="coerce").mean()
                )
                difficulty += (overall - history) * 1.5
    except Exception as exc:
        warnings.append(f"Matchup history unavailable: {exc}")

    role_columns = {
        "interior": ["DREB", "BLK"],
        "primary ball handler": ["STL"],
        "wing": ["STL", "BLK"],
    }[player_role]
    role_scale = {
        "interior": 18.0,
        "primary ball handler": 14.0,
        "wing": 12.0,
    }[player_role]
    dynamic_position = _percentile_adjustment(
        league, row.name, role_columns, role_scale
    )
    position_source = "league_percentiles"
    if dynamic_position is None:
        dynamic_position = POSITIONAL_TEAM_FALLBACKS.get(
            opponent["abbreviation"], {}
        ).get(player_role, 0.0)
        position_source = "team_role_fallback"
    position_adjustment = float(dynamic_position)

    league_pace = safe_number(
        pd.to_numeric(league.get("PACE"), errors="coerce").mean(), 100.0
    )
    opponent_pace = safe_number(row.get("PACE"), league_pace)
    pace_adjustment = (league_pace - opponent_pace) * 0.7

    try:
        report = safe_list(fetch_injury_report())
    except Exception:
        report = []
    opponent_absences = [
        safe_dict(item)
        for item in report
        if safe_get(safe_dict(item), "team") == opponent["abbreviation"]
        and safe_get(safe_dict(item), "status") in {"OUT", "QUESTIONABLE"}
    ]
    injury_adjustment = -len(opponent_absences) * 2.0

    difficulty += position_adjustment + pace_adjustment + injury_adjustment
    difficulty = max(0.0, min(100.0, difficulty))
    position_defense_proxy = max(
        0.0, min(100.0, difficulty - pace_adjustment - injury_adjustment)
    )
    label = (
        "difficult"
        if difficulty >= 65
        else "favorable"
        if difficulty <= 35
        else "neutral"
    )
    explanation = (
        f"{opponent['abbreviation']} is a {label} matchup using its "
        f"{defense:.1f} defensive rating and a {position_adjustment:+.1f} "
        f"{player_role} adjustment."
    )
    if history is not None:
        explanation += (
            f" The player averaged {history:.1f} points in available "
            "matchup history."
        )

    confidence = 70.0
    if history is not None:
        confidence += 12.0
    if position_source == "league_percentiles":
        confidence += 8.0
    confidence -= len(warnings) * 5.0
    confidence = round(max(0.0, min(100.0, confidence)), 1)
    value = round(difficulty, 1)
    details = {
        "opponent_defensive_rating": round(defense, 2),
        "league_average_defensive_rating": round(league_defense, 2),
        "opponent_def_eff_factor": round(opponent_def_eff_factor, 4),
        "opponent_usage_factor": round(opponent_usage_factor, 4),
        "matchup_history_points": history,
        "player_role": player_role,
        "position_adjustment": round(position_adjustment, 2),
        "position_adjustment_source": position_source,
        "position_defense_proxy": round(position_defense_proxy, 1),
        "pace_adjustment": round(pace_adjustment, 2),
        "injury_adjustment": round(injury_adjustment, 1),
        "explanation": explanation,
        "warnings": list(dict.fromkeys(warnings)),
    }
    return {
        "value": value,
        "confidence": confidence,
        "details": details,
        "player": player_name,
        "opponent": opponent["abbreviation"],
        "season": season,
        "difficulty_score": value,
        **details,
    }