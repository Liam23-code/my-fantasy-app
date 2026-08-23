"""Precomputed real per-game rate table for NBA players.

One row per player: real per-game points/rebounds/assists/PRA, real season
usage rate, and pace-adjusted scoring (real points-per-game normalized to
league-average pace, so a player on a fast-paced team and a player on a
slow-paced team are comparable on a common basis -- a standard, transparent
normalization, not a fabricated adjustment). Every number traces to the
NBA's own stats API (the same real, already-established data source
:mod:`modules.nba_props_generator` uses); nothing here is fabricated.

Built for the "precomputed player-rate table" performance objective: one
cached table computation serves every consumer (the UI's comparison panel,
trend overlays, future callers) instead of each recomputing its own real
per-player rates.
"""
from __future__ import annotations

from typing import Any

from betting.cache_utils import ttl_cache

from modules.nba_advanced import latest_season
from modules.nba_props_generator import _MIN_GAMES_PLAYED, _MIN_MINUTES_PER_GAME, _fetch_base_player_stats
from modules.sportsbook_parser import normalize_player_name, normalize_team_name

_TABLE_CACHE_SECONDS = 300


@ttl_cache(_TABLE_CACHE_SECONDS)
def _fetch_advanced_player_stats(season: str) -> Any:
    """Real season usage rate and pace-of-play per player (one live call, deduped by max GP)."""
    from nba_api.stats.endpoints import leaguedashplayerstats

    frame = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame", timeout=30
    ).get_data_frames()[0]
    return frame.sort_values("GP", ascending=False).drop_duplicates(subset="PLAYER_ID", keep="first")


@ttl_cache(_TABLE_CACHE_SECONDS)
def build_player_rate_table(season: str | None = None, *, pool_size: int = 300) -> list[dict[str, Any]]:
    """Real per-game rate table for the ``pool_size`` most-used real players.

    Reuses :func:`modules.nba_props_generator._fetch_base_player_stats` (the
    same real, cached fetch the prop generator uses) rather than making a
    second live call for the same raw stats; only usage rate and pace need
    a separate ``measure_type_detailed_defense="Advanced"`` fetch, since
    that measure type doesn't carry the raw counting stats the base one does.
    """
    season = season or latest_season()
    base = _fetch_base_player_stats(season)
    base = base[(base["GP"] >= _MIN_GAMES_PLAYED) & (base["MIN"] / base["GP"] >= _MIN_MINUTES_PER_GAME)]
    base = base.sort_values("MIN", ascending=False).head(pool_size)

    advanced = _fetch_advanced_player_stats(season)
    advanced_by_id = advanced.set_index("PLAYER_ID") if not advanced.empty else advanced
    league_average_pace = float(advanced["PACE"].mean()) if "PACE" in advanced.columns and not advanced.empty else None

    rows: list[dict[str, Any]] = []
    for _, player in base.iterrows():
        games = float(player["GP"])
        player_id = player["PLAYER_ID"]
        name = normalize_player_name(str(player["PLAYER_NAME"]))
        team_raw = str(player["TEAM_ABBREVIATION"])
        team = "" if team_raw == "TOT" else normalize_team_name(team_raw)

        points_per_game = float(player["PTS"]) / games
        rebounds_per_game = float(player["REB"]) / games
        assists_per_game = float(player["AST"]) / games

        usage_pct: float | None = None
        player_pace: float | None = None
        pace_adjusted_points: float | None = None
        if not advanced.empty and player_id in advanced_by_id.index:
            adv_row = advanced_by_id.loc[player_id]
            usage_pct = round(float(adv_row["USG_PCT"]) * 100.0, 2)
            player_pace = round(float(adv_row["PACE"]), 2)
            if player_pace and league_average_pace:
                pace_adjusted_points = round(points_per_game * (league_average_pace / player_pace), 2)

        rows.append(
            {
                "player_name": name,
                "team": team,
                "games_played": int(games),
                "points_per_game": round(points_per_game, 2),
                "rebounds_per_game": round(rebounds_per_game, 2),
                "assists_per_game": round(assists_per_game, 2),
                "pra_per_game": round(points_per_game + rebounds_per_game + assists_per_game, 2),
                "usage_pct": usage_pct,
                "player_pace": player_pace,
                "pace_adjusted_points_per_game": pace_adjusted_points,
                "basis": f"{season} real per-game rate ({games:g} games)",
            }
        )
    return rows
