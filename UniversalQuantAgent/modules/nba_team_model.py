"""Team-level NBA scoring model: real game results -> expected points, margin/total volatility.

Mirrors ``fantasy_engine/betting/team_model.py``'s NFL shape and returns the
same ``{"points_scored_avg", "points_allowed_avg", "games_played"}`` row
shape, so :func:`betting.team_model.project_game` (a pure, sport-agnostic
blend of "your scoring rate vs. opponent's rate allowed") can be reused
as-is for NBA -- no need to reimplement it.

Every number traces to real, already-played games this season
(``nba_api``'s league game log -- already a first-class dependency; see
modules/nba_advanced.py). Unlike the NFL side, which uses a published
scoring-margin volatility constant, the margin/total standard deviations
here are computed directly from this season's real completed games, since
we already have the real per-game data available -- one fewer number taken
on faith.
"""
from __future__ import annotations

import statistics
from typing import Any

from modules.nba_advanced import latest_season


def team_scoring_by_game(season: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Real per-team-per-game points scored and allowed, from this season's completed games.

    Returns ``{team: [{"game_id", "opponent", "points_scored", "points_allowed"}, ...]}``.
    Fails soft to ``{}`` if the league game log can't be loaded -- an
    optional enrichment layer, matching the fail-soft convention already
    used by modules/pace_model.py and modules/nba_advanced.py.
    """
    from nba_api.stats.endpoints import leaguegamelog

    season = season or latest_season()
    try:
        frame = leaguegamelog.LeagueGameLog(
            season=season, season_type_all_star="Regular Season",
            player_or_team_abbreviation="T", sorter="DATE", direction="ASC", timeout=15,
        ).get_data_frames()[0]
    except Exception:
        return {}
    if frame.empty:
        return {}

    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in frame.to_dict("records"):
        by_game.setdefault(str(row.get("GAME_ID")), []).append(row)

    by_team: dict[str, list[dict[str, Any]]] = {}
    for game_id, rows in by_game.items():
        if len(rows) != 2:
            continue  # an in-progress or malformed game record -- skip rather than guess
        first, second = rows
        pairs = ((first, second), (second, first))
        for team_row, opponent_row in pairs:
            team = str(team_row.get("TEAM_ABBREVIATION") or "").strip().upper()
            opponent = str(opponent_row.get("TEAM_ABBREVIATION") or "").strip().upper()
            scored, allowed = team_row.get("PTS"), opponent_row.get("PTS")
            if not team or not opponent or scored is None or allowed is None:
                continue
            by_team.setdefault(team, []).append(
                {
                    "game_id": game_id,
                    "opponent": opponent,
                    "points_scored": float(scored),
                    "points_allowed": float(allowed),
                }
            )
    return by_team


def team_scoring_averages(season: str | None = None) -> dict[str, dict[str, float]]:
    """Real average points scored/allowed per game for every team that's actually played.

    Same shape as ``betting.team_model.team_scoring_averages`` --
    ``{team: {"points_scored_avg", "points_allowed_avg", "games_played"}}``
    -- so ``betting.team_model.project_game`` works unmodified for NBA.
    """
    by_team = team_scoring_by_game(season)
    averages: dict[str, dict[str, float]] = {}
    for team, games in by_team.items():
        if not games:
            continue
        averages[team] = {
            "points_scored_avg": round(statistics.fmean(g["points_scored"] for g in games), 2),
            "points_allowed_avg": round(statistics.fmean(g["points_allowed"] for g in games), 2),
            "games_played": len(games),
        }
    return averages


def real_margin_and_total_volatility(season: str | None = None) -> dict[str, float]:
    """Real observed standard deviation of NBA game margins and totals this season.

    Computed directly from every real completed game rather than assuming a
    published constant -- there are two rows per game in the underlying log
    (see :func:`team_scoring_by_game`), so each game is counted once here by
    reading one team's perspective per game_id.
    """
    by_team = team_scoring_by_game(season)
    seen_games: set[str] = set()
    margins: list[float] = []
    totals: list[float] = []
    for games in by_team.values():
        for game in games:
            game_id = game["game_id"]
            if game_id in seen_games:
                continue
            seen_games.add(game_id)
            margins.append(game["points_scored"] - game["points_allowed"])
            totals.append(game["points_scored"] + game["points_allowed"])
    if len(margins) < 2:
        # Not enough real games yet (e.g. very early season) -- these are
        # widely-cited approximate NBA figures, used only until enough real
        # games exist to compute our own, and never presented as "real" in
        # the way the computed branch above is.
        return {"margin_stdev": 12.0, "total_stdev": 11.5, "games_sampled": len(margins), "basis": "fallback_estimate"}
    return {
        "margin_stdev": round(statistics.stdev(margins), 2),
        "total_stdev": round(statistics.stdev(totals), 2),
        "games_sampled": len(margins),
        "basis": f"{season or latest_season()} real completed games",
    }
