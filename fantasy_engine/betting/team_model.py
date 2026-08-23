"""Team-level projection model: real scoring history -> expected points, pace, opponent strength.

Every number traces to the real NFL schedule (final scores of games
already played this season) -- the same offline ``nflreadpy.load_schedules``
source :mod:`fantasy.data_loader` already uses for DST scoring and bye
weeks. No network access beyond that already-established, cached local
data; no fabricated ratings.
"""

from __future__ import annotations

import statistics
from typing import Any

from .cache_utils import ttl_cache

#: A real NFL week's schedule and results don't change within a session --
#: this is not on any live per-request path today (see the NBA side's
#: modules/nba_schedule.py for the "changes every day" case that genuinely
#: needs a short TTL); a long TTL here just avoids redundant nflreadpy
#: calls if this function is called repeatedly (e.g. once per generated
#: game row) within one generation run.
_SCHEDULE_CACHE_SECONDS = 3600


def _require_nflreadpy() -> Any:
    try:
        import nflreadpy
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("nflreadpy is not installed; team scoring data cannot be loaded") from error
    return nflreadpy


@ttl_cache(_SCHEDULE_CACHE_SECONDS)
def team_scoring_by_week(season: int) -> dict[str, list[dict[str, Any]]]:
    """Real per-team-per-week points scored and allowed, from the schedule.

    Returns ``{team: [{"week", "opponent", "points_scored", "points_allowed"}, ...]}``,
    each team's played weeks in order. Fails soft to ``{}`` if the schedule
    can't be loaded -- an optional enrichment layer, like the equivalent
    fail-soft loaders in :mod:`fantasy.data_loader`.
    """
    try:
        nflreadpy = _require_nflreadpy()
        schedules = nflreadpy.load_schedules(seasons=[season]).to_pandas()
    except Exception:
        return {}
    if schedules.empty or "game_type" not in schedules.columns:
        return {}
    reg = schedules[schedules["game_type"] == "REG"]
    if reg.empty:
        return {}

    by_team: dict[str, list[dict[str, Any]]] = {}
    for row in reg.to_dict("records"):
        week = row.get("week")
        home, away = row.get("home_team"), row.get("away_team")
        home_score, away_score = row.get("home_score"), row.get("away_score")
        if week is None or not home or not away or home_score is None or away_score is None:
            continue
        week = int(week)
        home, away = str(home).strip().upper(), str(away).strip().upper()
        home_score, away_score = float(home_score), float(away_score)
        by_team.setdefault(home, []).append(
            {"week": week, "opponent": away, "points_scored": home_score, "points_allowed": away_score}
        )
        by_team.setdefault(away, []).append(
            {"week": week, "opponent": home, "points_scored": away_score, "points_allowed": home_score}
        )
    for weeks in by_team.values():
        weeks.sort(key=lambda entry: entry["week"])
    return by_team


def team_scoring_averages(season: int) -> dict[str, dict[str, float]]:
    """Real average points scored/allowed per game for every team that's actually played.

    Returns ``{team: {"points_scored_avg", "points_allowed_avg", "games_played"}}``.
    """
    by_team = team_scoring_by_week(season)
    averages: dict[str, dict[str, float]] = {}
    for team, weeks in by_team.items():
        if not weeks:
            continue
        averages[team] = {
            "points_scored_avg": round(statistics.fmean(w["points_scored"] for w in weeks), 2),
            "points_allowed_avg": round(statistics.fmean(w["points_allowed"] for w in weeks), 2),
            "games_played": len(weeks),
        }
    return averages


def project_game(
    home_team: str, away_team: str, *, averages: dict[str, dict[str, float]], league_average_points: float | None = None
) -> dict[str, Any]:
    """Expected points for both teams in one matchup, from real season-to-date scoring.

    Each team's expected points = the average of (their own real scoring
    rate, the opponent's real rate of points allowed) -- the standard,
    simple "offense vs. opponent's defense" blend used as a baseline team
    total model; not a claim of precise point-spread accuracy, a
    transparent starting estimate any downstream model can refine.
    ``league_average_points`` backfills a team with no games played yet
    (e.g. very early season) so the model degrades to a neutral estimate
    rather than raising.
    """
    home_team, away_team = home_team.strip().upper(), away_team.strip().upper()
    if league_average_points is None:
        pool = [row["points_scored_avg"] for row in averages.values()]
        league_average_points = round(statistics.fmean(pool), 2) if pool else 21.0

    def _team_rates(team: str) -> tuple[float, float]:
        row = averages.get(team)
        if row is None:
            return league_average_points, league_average_points
        return row["points_scored_avg"], row["points_allowed_avg"]

    home_scored, home_allowed = _team_rates(home_team)
    away_scored, away_allowed = _team_rates(away_team)

    home_expected = round((home_scored + away_allowed) / 2.0, 2)
    away_expected = round((away_scored + home_allowed) / 2.0, 2)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_expected_points": home_expected,
        "away_expected_points": away_expected,
        "spread": round(home_expected - away_expected, 2),  # positive -> home favored
        "total": round(home_expected + away_expected, 2),
        "league_average_points": league_average_points,
    }
