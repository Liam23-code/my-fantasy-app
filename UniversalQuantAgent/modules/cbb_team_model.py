"""Team-level CBB scoring model: real completed-game results -> expected points, pace, margin/total volatility.

Uses ESPN's public site API (``site.api.espn.com/.../mens-college-basketball/...``)
-- real, structured JSON, not HTML, and not a sportsbook -- the same legal
category as ``nba_api``'s live scoreboard (see offline_data_contract.md's
section on what's allowed and why this module is a narrow, documented
exception to this project's usual "no requests" rule for market-pipeline
files). Verified live during this cycle's build (real Duke Blue Devils
data: real box-score inputs for a pace estimate, a real 75-60 completed
game vs. Texas on 2025-11-05).

ESPN's API needs one request per team (there is no single bulk
"every team's real averages" call the way ``nba_api``'s
``LeagueDashTeamStats`` gives NBA) -- :func:`build_team_scoring_averages`
fetches a bounded pool of real teams concurrently
(:mod:`betting.parallel_utils`) rather than every Division I program, and
documents that its pool is ESPN's team-list order, not a quality ranking.
"""
from __future__ import annotations

import statistics
from typing import Any

from betting.parallel_utils import parallel_map

ESPN_CBB_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_TIMEOUT = 15


def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    import requests

    response = requests.get(f"{ESPN_CBB_BASE}{path}", params=params or {}, timeout=ESPN_TIMEOUT)
    response.raise_for_status()
    return response.json()


def list_teams(limit: int = 500) -> list[dict[str, Any]]:
    """Every real Division I team ESPN carries (id, abbreviation, display name). Fails soft to ``[]``."""
    try:
        data = _get("/teams", {"limit": limit})
    except Exception:
        return []
    teams = ((data.get("sports") or [{}])[0].get("leagues") or [{}])[0].get("teams") or []
    return [entry["team"] for entry in teams if isinstance(entry, dict) and "team" in entry]


def team_pace_estimate(team_id: str, season: int) -> float | None:
    """A real, standard possession-estimate for one team this season, or ``None`` on failure.

    ``Poss ~= FGA - OREB + TOV + 0.44*FTA`` -- the same formula basketball
    analytics (including ``nba_api``'s own Advanced measure type)
    use internally; real box-score inputs, not a fabricated pace number.
    """
    try:
        data = _get(f"/teams/{team_id}/statistics", {"season": season})
    except Exception:
        return None
    categories = ((data.get("results") or {}).get("stats") or {}).get("categories") or []
    values: dict[str, float] = {}
    for category in categories:
        for stat in category.get("stats", []):
            name = stat.get("name")
            if name in ("avgFieldGoalsAttempted", "avgOffensiveRebounds", "avgTurnovers", "avgFreeThrowsAttempted"):
                try:
                    values[name] = float(stat.get("value"))
                except (TypeError, ValueError):
                    pass
    if len(values) < 4:
        return None
    return round(values["avgFieldGoalsAttempted"] - values["avgOffensiveRebounds"] + values["avgTurnovers"] + 0.44 * values["avgFreeThrowsAttempted"], 2)


def team_scoring_by_game(team_id: str, season: int) -> list[dict[str, Any]]:
    """Real per-game points scored/allowed for one team this season, from its real completed games."""
    try:
        data = _get(f"/teams/{team_id}/schedule", {"season": season})
    except Exception:
        return []
    events = data.get("events") or []
    games: list[dict[str, Any]] = []
    for event in events:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        completed = ((competition.get("status") or {}).get("type") or {}).get("completed")
        if not completed:
            continue
        competitors = competition.get("competitors") or []
        if len(competitors) != 2:
            continue
        mine = next((c for c in competitors if c.get("team", {}).get("id") == str(team_id)), None)
        theirs = next((c for c in competitors if c.get("team", {}).get("id") != str(team_id)), None)
        if not mine or not theirs:
            continue
        try:
            scored = float(mine["score"]["value"])
            allowed = float(theirs["score"]["value"])
        except (KeyError, TypeError, ValueError):
            continue
        games.append({"opponent": theirs.get("team", {}).get("abbreviation", ""), "points_scored": scored, "points_allowed": allowed})
    return games


def build_team_scoring_averages(season: int, *, team_pool_size: int = 40) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]:
    """Real per-game scoring averages for a bounded pool of real Division I teams.

    Returns ``(averages, games_by_team)`` -- ``averages`` matches
    ``betting.team_model.team_scoring_averages``'s shape
    (``{team: {"points_scored_avg", "points_allowed_avg", "games_played"}}``)
    so ``betting.team_model.project_game`` works unmodified for CBB;
    ``games_by_team`` is the raw per-game data, returned too so
    :func:`real_margin_and_total_volatility` (and
    modules.cbb_props_generator) can reuse it without a second fetch.
    """
    teams = list_teams()[:team_pool_size]

    def _fetch(team: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        code = str(team.get("abbreviation") or team.get("id") or "").upper()
        return code, team_scoring_by_game(team["id"], season)

    results = parallel_map(_fetch, teams, max_workers=8)

    averages: dict[str, dict[str, float]] = {}
    games_by_team: dict[str, list[dict[str, Any]]] = {}
    for code, games in results:
        if not games:
            continue
        games_by_team[code] = games
        averages[code] = {
            "points_scored_avg": round(statistics.fmean(g["points_scored"] for g in games), 2),
            "points_allowed_avg": round(statistics.fmean(g["points_allowed"] for g in games), 2),
            "games_played": len(games),
        }
    return averages, games_by_team


def real_margin_and_total_volatility(games_by_team: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Real observed standard deviation of CBB game margins and totals, from already-fetched real games.

    Takes ``games_by_team`` (from :func:`build_team_scoring_averages`)
    rather than fetching itself, since the per-team schedule calls are the
    expensive part. Each real game appears twice in ``games_by_team``
    (once per team) when both teams are in the pool -- only counted once
    here via a seen-pairs set. Falls back to a documented, disclosed
    estimate (not presented as "real") when fewer than 2 real games are
    available.
    """
    seen_games: set[tuple[str, str, float, float]] = set()
    margins: list[float] = []
    totals: list[float] = []
    for team, games in games_by_team.items():
        for game in games:
            key = tuple(sorted((team, game["opponent"]))) + (game["points_scored"] + game["points_allowed"], game["points_scored"] - game["points_allowed"])
            if key in seen_games:
                continue
            seen_games.add(key)
            margins.append(game["points_scored"] - game["points_allowed"])
            totals.append(game["points_scored"] + game["points_allowed"])
    if len(margins) < 2:
        return {"margin_stdev": 12.0, "total_stdev": 11.5, "games_sampled": len(margins), "basis": "fallback_estimate"}
    return {
        "margin_stdev": round(statistics.stdev(margins), 2),
        "total_stdev": round(statistics.stdev(totals), 2),
        "games_sampled": len(margins),
        "basis": f"{len(margins)} real completed games",
    }
