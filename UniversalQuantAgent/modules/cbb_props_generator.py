"""Generate our own default CBB prop lines -- real per-game rates, never a sportsbook.

Uses ESPN's public college-basketball stats API (see modules/cbb_team_model.py
for the legal basis -- real JSON, not a sportsbook, not HTML scraping).
Real per-player season averages come directly from ESPN's own "Season
Averages" computation (points/rebounds/assists per game, minutes per
game, games played) -- no season-total/games-played division needed here,
unlike the NFL/NBA generators, since ESPN already reports the per-game
rate.

Not on the request path: a one-off generation-time tool (mirrors
modules/nba_props_generator.py), run directly to (re)populate
``data/cbb_props.json``:

    python -m modules.cbb_props_generator

ESPN requires one request per team roster and one per player's stats --
there is no single bulk "every player's real averages" call the way
``nba_api``'s ``LeagueDashPlayerStats`` gives NBA -- so this generator
covers a bounded, real team pool (ESPN's team-list order, not a quality
ranking; see modules.cbb_team_model's identical disclosure), fetched
concurrently (:mod:`betting.parallel_utils`) to keep generation time
reasonable. A full-league run is possible by raising ``team_pool_size``;
it just takes longer (each real per-player stats call is meaningfully
slower than a team-level call -- measured ~1-2s each during this cycle's
build).
"""
from __future__ import annotations

from typing import Any

from betting.parallel_utils import parallel_map

from modules.cbb_team_model import ESPN_CBB_BASE, ESPN_TIMEOUT, list_teams
from modules.college_sports_common import normalize_college_team_name, normalize_player_name

DATA_ROOT_NAME = "cbb_props.json"

#: Minimum real per-game rate before a market is worth offering -- same
#: rationale as the NFL/NBA generators' identical guards.
_MIN_RATE_BY_CATEGORY: dict[str, float] = {"points": 4.0, "rebounds": 2.0, "assists": 1.0, "PRA": 6.0}
_MIN_GAMES_PLAYED = 8
_MIN_MINUTES_PER_GAME = 8.0


def _round_to_half(value: float) -> float:
    """Nearest sportsbook-style X.5 line -- never a whole number (see NFL/NBA generators for why)."""
    return round(value - 0.5) + 0.5


def _get_roster(team_id: str, season: int) -> list[dict[str, Any]]:
    import requests

    try:
        response = requests.get(f"{ESPN_CBB_BASE}/teams/{team_id}/roster", params={"season": season}, timeout=ESPN_TIMEOUT)
        response.raise_for_status()
        return response.json().get("athletes") or []
    except Exception:
        return []


def _get_player_season_averages(athlete_id: str) -> dict[str, float] | None:
    import requests

    try:
        response = requests.get(
            f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/mens-college-basketball/athletes/{athlete_id}/stats",
            timeout=ESPN_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None
    categories = data.get("categories") or []
    if not categories:
        return None
    season_averages = categories[0]  # "Season Averages" is always the first category ESPN returns
    labels = season_averages.get("labels") or []
    statistics_rows = season_averages.get("statistics") or []
    if not statistics_rows or not labels:
        return None
    values = statistics_rows[0].get("stats") or []
    if len(values) != len(labels):
        return None
    by_label = dict(zip(labels, values))
    try:
        return {
            "games_played": float(by_label.get("GP", 0) or 0),
            "minutes": float(by_label.get("MIN", 0) or 0),
            "points": float(by_label.get("PTS", 0) or 0),
            "rebounds": float(by_label.get("REB", 0) or 0),
            "assists": float(by_label.get("AST", 0) or 0),
        }
    except (TypeError, ValueError):
        return None


def generate_default_props(season: int, *, team_pool_size: int = 20, max_workers: int = 16) -> list[dict[str, Any]]:
    """Real per-game-rate prop lines for a bounded pool of real Division I players.

    A player's line is their real season-average stat, exactly as ESPN
    itself reports it -- not a season-total/games division (ESPN already
    computes the per-game rate). Filtered to rotation players
    (``_MIN_GAMES_PLAYED``, ``_MIN_MINUTES_PER_GAME``) so a deep-bench
    garbage-time stat line doesn't become a "real" market.
    """
    teams = list_teams()[:team_pool_size]

    def _fetch_roster(team: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        code = str(team.get("abbreviation") or team.get("id") or "").upper()
        return code, _get_roster(team["id"], season)

    rosters = parallel_map(_fetch_roster, teams, max_workers=max_workers)

    player_team: dict[str, str] = {}
    all_athletes: list[dict[str, Any]] = []
    for code, athletes in rosters:
        for athlete in athletes:
            player_team[athlete["id"]] = code
            all_athletes.append(athlete)

    def _fetch_stats(athlete: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float] | None]:
        return athlete, _get_player_season_averages(athlete["id"])

    stat_results = parallel_map(_fetch_stats, all_athletes, max_workers=max_workers)

    rows: list[dict[str, Any]] = []
    for athlete, stats in stat_results:
        if not stats:
            continue
        games = stats["games_played"]
        if games < _MIN_GAMES_PLAYED or games <= 0:
            continue
        if stats["minutes"] < _MIN_MINUTES_PER_GAME:
            continue
        name = normalize_player_name(str(athlete.get("displayName") or athlete.get("fullName") or ""))
        team = normalize_college_team_name(player_team.get(athlete["id"], ""))
        if not name:
            continue
        rates = {
            "points": stats["points"],
            "rebounds": stats["rebounds"],
            "assists": stats["assists"],
            "PRA": stats["points"] + stats["rebounds"] + stats["assists"],
        }
        for category, rate in rates.items():
            if rate < _MIN_RATE_BY_CATEGORY[category]:
                continue
            rows.append(
                {
                    "player_name": name,
                    "team": team,
                    "category": category,
                    "line": _round_to_half(rate),
                    "over_price": -110.0,
                    "under_price": -110.0,
                    "sportsbook": "default",
                    "basis": f"{season} real per-game rate ({games:g} games, ESPN season averages)",
                }
            )
    return rows


def write_default_props_file(season: int, *, team_pool_size: int = 20, path: Any = None) -> int:
    """Generate and write ``data/cbb_props.json``. Returns the number of rows written."""
    import json
    from pathlib import Path

    rows = generate_default_props(season, team_pool_size=team_pool_size)
    target = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "data" / DATA_ROOT_NAME
    payload = {
        "note": (
            "Real per-game rates from ESPN's public college-basketball stats API (see "
            "modules/cbb_props_generator.py) -- never a sportsbook. Upload a file from the app to "
            "override or add to these."
        ),
        "generated_by": "modules.cbb_props_generator",
        "source_season": season,
        "team_pool_size": team_pool_size,
        "props": rows,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    import datetime

    current_year = datetime.date.today().year
    count = write_default_props_file(current_year)
    print(f"Wrote {count} prop lines for season {current_year}")
