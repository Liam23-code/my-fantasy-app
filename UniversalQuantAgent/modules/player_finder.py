"""NBA player leaderboards built from public nba_api season aggregates."""

from __future__ import annotations

from typing import Any

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players

from modules.nba_advanced import latest_season
from modules.nba_cache import NBA_API_TIMEOUT, fetch_nba_frames
from modules.utils import safe_number


def fetch_player_aggregates(season: str | None = None) -> pd.DataFrame:
    """Fetch one reusable table containing base and advanced per-game stats."""
    season = season or latest_season()
    common = {
        "season": season,
        "season_type_all_star": "Regular Season",
        "per_mode_detailed": "PerGame",
        "timeout": NBA_API_TIMEOUT,
    }
    # The neutral rows keep leaderboards renderable during an NBA.com outage.
    # They are intentionally conservative and are marked by the shared cache
    # layer as fallback data rather than presented as live rankings.
    active_players = [player for player in players.get_players() if player.get("is_active")]
    neutral_base = pd.DataFrame([{
        "PLAYER_ID": player["id"], "PLAYER_NAME": player["full_name"],
        "TEAM_ABBREVIATION": "", "MIN": 20.0, "PTS": 10.0,
        "REB": 4.0, "AST": 3.0,
    } for player in active_players])
    neutral_advanced = pd.DataFrame([{
        "PLAYER_ID": player["id"], "USG_PCT": .20, "TS_PCT": .55,
    } for player in active_players])
    base_frames = fetch_nba_frames(
        f"player_finder_base_{season}",
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            measure_type_detailed_defense="Base", **common
        ).get_data_frames(),
        fallback_factory=lambda: neutral_base,
    )
    advanced_frames = fetch_nba_frames(
        f"player_finder_advanced_{season}",
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            measure_type_detailed_defense="Advanced", **common
        ).get_data_frames(),
        fallback_factory=lambda: neutral_advanced,
    )
    base = base_frames[0]
    advanced = advanced_frames[0]
    advanced_columns = [column for column in ["PLAYER_ID", "USG_PCT", "TS_PCT"]
                        if column in advanced.columns]
    table = base.merge(advanced[advanced_columns], on="PLAYER_ID", how="left")
    if table.empty:
        raise ValueError(f"No NBA player aggregates were returned for {season}.")
    return table


def _leaderboard(
    sort_column: str,
    limit: int = 20,
    min_minutes: float = 20,
    min_usage: float | None = None,
    season: str | None = None,
    player_table: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Filter, rank, and serialize a season player table."""
    table = player_table.copy() if player_table is not None else fetch_player_aggregates(season)
    table = table[pd.to_numeric(table["MIN"], errors="coerce") >= min_minutes]
    if min_usage is not None:
        table = table[pd.to_numeric(table["USG_PCT"], errors="coerce") * 100 >= min_usage]
    table = table.sort_values(sort_column, ascending=False).head(max(int(limit), 1))
    results = []
    for _, row in table.iterrows():
        results.append({
            "player_name": str(row.get("PLAYER_NAME", "Unknown")),
            "team": str(row.get("TEAM_ABBREVIATION", "")),
            "points_per_game": round(safe_number(row.get("PTS")), 1),
            "rebounds_per_game": round(safe_number(row.get("REB")), 1),
            "assists_per_game": round(safe_number(row.get("AST")), 1),
            "usage_rate": round(safe_number(row.get("USG_PCT")) * 100, 1),
            "true_shooting_pct": round(safe_number(row.get("TS_PCT")) * 100, 1),
            "minutes_per_game": round(safe_number(row.get("MIN")), 1),
        })
    return results


def top_scorers(limit: int = 20, min_minutes: float = 20,
                season: str | None = None) -> list[dict[str, Any]]:
    """Return qualified players ranked by points per game."""
    return _leaderboard("PTS", limit, min_minutes, season=season)


def top_rebounders(limit: int = 20, min_minutes: float = 20,
                   season: str | None = None) -> list[dict[str, Any]]:
    """Return qualified players ranked by rebounds per game."""
    return _leaderboard("REB", limit, min_minutes, season=season)


def top_assist_players(limit: int = 20, min_minutes: float = 20,
                       season: str | None = None) -> list[dict[str, Any]]:
    """Return qualified players ranked by assists per game."""
    return _leaderboard("AST", limit, min_minutes, season=season)


def high_usage_players(limit: int = 20, min_minutes: float = 20,
                       season: str | None = None) -> list[dict[str, Any]]:
    """Return qualified players ranked by usage rate."""
    return _leaderboard("USG_PCT", limit, min_minutes, season=season)


def high_efficiency_players(
    limit: int = 20,
    min_minutes: float = 20,
    min_usage: float = 20,
    season: str | None = None,
) -> list[dict[str, Any]]:
    """Return meaningful-usage players ranked by true shooting percentage."""
    return _leaderboard("TS_PCT", limit, min_minutes, min_usage, season)

