"""Team-level CFB scoring model: real completed-game results -> expected points, pace, margin/total volatility.

**Requires a College Football Data API key** (a free, registered key from
collegefootballdata.com -- set the ``CFBD_API_KEY`` environment variable).
Every function here returns an empty/neutral result when no key is
configured, rather than raising -- "no key yet" is a normal state, not an
error, matching this codebase's other fail-closed loaders.

Built without access to a live key (see offline_data_contract.md and
cfb_pipeline.md for why): field names below are written defensively,
tolerating both the API's documented camelCase (``homePoints``) and the
snake_case (``home_points``) an older client or a hand-built mirror might
use, via the same alias-lookup pattern already used throughout this
codebase's loaders (see ``_first``). **This integration has not been
verified against live data and should be checked against a real response
once a key is available** -- see cfb_pipeline.md.

Uses ``requests`` directly rather than the official ``cfbd`` PyPI client:
that client pins ``pydantic<2``, which conflicts with this project's
pydantic 2.x (required by FastAPI) and risks breaking the running app if
installed. A handful of GET calls to a public JSON API doesn't need a full
generated SDK. This is a narrow, deliberate, documented exception to this
project's usual "no requests" rule for market-pipeline files -- see
offline_data_contract.md's section on what's allowed and why.
"""
from __future__ import annotations

import os
import statistics
from typing import Any

CFBD_BASE_URL = "https://api.collegefootballdata.com"
CFBD_TIMEOUT = 15


def _api_key() -> str | None:
    return os.environ.get("CFBD_API_KEY") or None


def _get(path: str, params: dict[str, Any]) -> Any:
    """One authenticated GET against the CFBD API. Raises on any failure -- callers catch it."""
    import requests

    key = _api_key()
    if not key:
        raise RuntimeError("CFBD_API_KEY is not set")
    response = requests.get(
        f"{CFBD_BASE_URL}{path}", params=params, headers={"Authorization": f"Bearer {key}"}, timeout=CFBD_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def team_scoring_by_game(season: int) -> dict[str, list[dict[str, Any]]]:
    """Real per-team-per-game points scored and allowed, from CFBD's real completed games this season.

    Returns ``{team: [{"opponent", "points_scored", "points_allowed"}, ...]}``.
    Fails soft to ``{}`` if no API key is configured or the request fails
    for any reason -- an optional enrichment layer, not a hard dependency.
    """
    try:
        games = _get("/games", {"year": season, "seasonType": "regular"})
    except Exception:
        return {}
    if not isinstance(games, list):
        return {}

    by_team: dict[str, list[dict[str, Any]]] = {}
    for game in games:
        if not isinstance(game, dict):
            continue
        home = _first(game, "homeTeam", "home_team")
        away = _first(game, "awayTeam", "away_team")
        home_points = _first(game, "homePoints", "home_points")
        away_points = _first(game, "awayPoints", "away_points")
        if not home or not away or home_points is None or away_points is None:
            continue
        home, away = str(home).strip().upper(), str(away).strip().upper()
        try:
            home_points, away_points = float(home_points), float(away_points)
        except (TypeError, ValueError):
            continue
        by_team.setdefault(home, []).append({"opponent": away, "points_scored": home_points, "points_allowed": away_points})
        by_team.setdefault(away, []).append({"opponent": home, "points_scored": away_points, "points_allowed": home_points})
    return by_team


def team_scoring_averages(season: int) -> dict[str, dict[str, float]]:
    """Real average points scored/allowed per game for every team that's actually played.

    Same shape as ``betting.team_model.team_scoring_averages`` --
    ``{team: {"points_scored_avg", "points_allowed_avg", "games_played"}}``
    -- so ``betting.team_model.project_game`` works unmodified for CFB.
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


def real_margin_and_total_volatility(season: int) -> dict[str, float]:
    """Real observed standard deviation of CFB game margins and totals this season.

    Computed directly from every real completed game rather than assuming
    a published constant -- college football's real week-to-week variance
    is well known to run much higher than the NFL's (blowouts and upsets
    are both far more common against a much wider real talent gap between
    programs), so reusing an NFL constant here would be a real modeling
    error. Falls back to a documented, disclosed estimate (not presented
    as "real") only when fewer than 2 real games are available.
    """
    by_team = team_scoring_by_game(season)
    seen_games: set[tuple[str, str]] = set()
    margins: list[float] = []
    totals: list[float] = []
    for team, games in by_team.items():
        for game in games:
            pair = tuple(sorted((team, game["opponent"])))
            if pair in seen_games:
                continue
            seen_games.add(pair)
            margins.append(game["points_scored"] - game["points_allowed"])
            totals.append(game["points_scored"] + game["points_allowed"])
    if len(margins) < 2:
        # A widely-cited approximate range for real CFB scoring variance,
        # used only until enough real games exist to compute our own --
        # see modules.nba_team_model's identical fallback-disclosure pattern.
        return {"margin_stdev": 21.0, "total_stdev": 17.0, "games_sampled": len(margins), "basis": "fallback_estimate"}
    return {
        "margin_stdev": round(statistics.stdev(margins), 2),
        "total_stdev": round(statistics.stdev(totals), 2),
        "games_sampled": len(margins),
        "basis": f"{season} real completed games",
    }


def fetch_week_games(season: int, week: int) -> list[dict[str, Any]]:
    """Real scheduled matchups for one real week (played or not) -- CFB is weekly, like the NFL, not daily like NBA/CBB.

    Returns ``[{"home_team", "away_team"}, ...]`` for the Money Lines tab's
    "which games to evaluate" input, mirroring how the NFL side's default
    odds file covers one real week at a time. Fails soft to ``[]`` when no
    key is configured or the request fails.
    """
    try:
        games = _get("/games", {"year": season, "week": week, "seasonType": "regular"})
    except Exception:
        return []
    if not isinstance(games, list):
        return []
    results = []
    for game in games:
        if not isinstance(game, dict):
            continue
        home = _first(game, "homeTeam", "home_team")
        away = _first(game, "awayTeam", "away_team")
        if not home or not away:
            continue
        results.append({"home_team": str(home).strip().upper(), "away_team": str(away).strip().upper()})
    return results


def team_pace(season: int, team: str) -> dict[str, Any] | None:
    """Real plays-per-game for one team this season, or ``None`` if unavailable.

    Best-effort: CFBD's advanced-stats schema for offensive play counts has
    not been verified live (see module docstring). Returns ``None`` rather
    than a guessed number when the expected field isn't present, so a
    caller never silently treats a missing pace figure as zero.
    """
    try:
        rows = _get("/stats/season/advanced", {"year": season, "team": team})
    except Exception:
        return None
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0]
    offense = row.get("offense") if isinstance(row, dict) else None
    plays = _first(offense or {}, "plays") if isinstance(offense, dict) else None
    games = _first(row, "games") if isinstance(row, dict) else None
    if plays is None or not games:
        return None
    try:
        plays_per_game = float(plays) / float(games)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return {"team": str(team).strip().upper(), "season": season, "plays_per_game": round(plays_per_game, 1), "games": int(games)}
