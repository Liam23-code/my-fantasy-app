"""Real recent-vs-season trend signals: team pace delta, player usage-rate delta.

An additive enrichment layer -- it does not modify modules/pace_model.py or
modules/minutes_model.py, which already compute their own real
season/recent blends for their specific purposes (a matchup's blended pace
projection; a player's role-change-adjusted minutes). This module answers
a narrower, standalone question directly useful to prop confidence: is
this team's pace, or this player's usage rate, trending up or down
relative to their own season baseline right now? Every number is real,
already-published ``nba_api`` data (a season table vs. its own
``last_n_games=10`` table, the same season/recent-split convention
modules/pace_model.py already uses for team pace); nothing here is
fabricated or forecast.
"""
from __future__ import annotations

from typing import Any

from betting.cache_utils import ttl_cache

from modules.data_quality import safe_number
from modules.nba_advanced import fetch_league_team_stats, find_team, latest_season

#: pace_delta / usage_delta magnitude below which a change is noise, not a
#: real trend worth flagging -- matches pace_model.py's own "balanced" band
#: width (+/-1 pace point) for team pace; usage rate gets a slightly wider
#: band since single-game usage swings more than pace does.
_PACE_TREND_THRESHOLD = 1.0
_USAGE_TREND_THRESHOLD_PCT = 1.5

#: These are display overlays computed on demand for potentially many rows
#: in a table (see the Betting Engine page) -- cache them so scrolling or
#: re-rendering the same table doesn't re-issue a live call per row.
_TREND_CACHE_SECONDS = 300


@ttl_cache(_TREND_CACHE_SECONDS)
def team_pace_trend(team: str, season: str | None = None) -> dict[str, Any]:
    """Real season vs. real last-10-games pace for one team, and the delta between them."""
    season = season or latest_season()
    resolved = find_team(team)
    season_table = fetch_league_team_stats(season)
    try:
        recent_table = fetch_league_team_stats(season, last_n_games=10)
    except Exception:
        recent_table = season_table

    season_row = season_table[season_table["TEAM_ID"] == resolved["id"]]
    if season_row.empty:
        raise ValueError(f"No season pace data available for {team!r}.")
    season_pace = safe_number(season_row.iloc[0].get("PACE"))

    recent_row = recent_table[recent_table["TEAM_ID"] == resolved["id"]]
    recent_pace = safe_number(recent_row.iloc[0].get("PACE"), season_pace) if not recent_row.empty else season_pace

    delta = round(recent_pace - season_pace, 2)
    return {
        "team": resolved["abbreviation"],
        "season": season,
        "season_pace": round(season_pace, 2),
        "recent_10_pace": round(recent_pace, 2),
        "pace_delta": delta,
        "trend": "accelerating" if delta > _PACE_TREND_THRESHOLD else "slowing" if delta < -_PACE_TREND_THRESHOLD else "stable",
    }


@ttl_cache(_TREND_CACHE_SECONDS)
def player_usage_trend(player_name: str, season: str | None = None) -> dict[str, Any]:
    """Real season vs. real last-10-games usage rate for one player, and the delta between them."""
    from nba_api.stats.endpoints import leaguedashplayerstats

    from modules.projections import _find_player

    season = season or latest_season()
    player = _find_player(player_name)

    season_frame = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame", timeout=15
    ).get_data_frames()[0]
    season_row = season_frame[season_frame["PLAYER_ID"] == player["id"]]
    if season_row.empty:
        raise ValueError(f"No season usage-rate data available for {player_name!r}.")
    season_usage = safe_number(season_row.iloc[0].get("USG_PCT")) * 100.0

    try:
        recent_frame = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame",
            last_n_games=10, timeout=15,
        ).get_data_frames()[0]
        recent_row = recent_frame[recent_frame["PLAYER_ID"] == player["id"]]
        recent_usage = safe_number(recent_row.iloc[0].get("USG_PCT")) * 100.0 if not recent_row.empty else season_usage
    except Exception:
        recent_usage = season_usage

    delta = round(recent_usage - season_usage, 2)
    return {
        "player": player["full_name"],
        "season": season,
        "season_usage_pct": round(season_usage, 2),
        "recent_10_usage_pct": round(recent_usage, 2),
        "usage_delta": delta,
        "trend": "rising" if delta > _USAGE_TREND_THRESHOLD_PCT else "falling" if delta < -_USAGE_TREND_THRESHOLD_PCT else "stable",
    }
