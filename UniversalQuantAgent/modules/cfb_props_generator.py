"""Generate our own default CFB prop lines -- real per-game rates, never a sportsbook.

**Requires a College Football Data API key** (set ``CFBD_API_KEY``). Returns
an empty list, and ``write_default_props_file`` writes an empty file with a
note explaining why, when no key is configured -- matching every other
"not yet available" state in this codebase (never fabricates a line).

Not on the request path: a one-off generation-time tool (mirrors
modules/nba_props_generator.py), run directly to populate
``data/cfb_props.json`` once a key exists:

    CFBD_API_KEY=... python -m modules.cfb_props_generator

**Built without access to a live key -- not yet verified against real
data.** See modules/cfb_team_model.py's docstring for why ``requests`` is
used directly here rather than the official ``cfbd`` client, and
cfb_pipeline.md for the verification this integration still needs.
"""
from __future__ import annotations

from typing import Any

from modules.cfb_team_model import CFBD_BASE_URL, CFBD_TIMEOUT, _api_key, _first, team_scoring_by_game
from modules.college_sports_common import normalize_college_team_name, normalize_player_name

DATA_ROOT_NAME = "cfb_props.json"

#: category -> (CFBD stat category, CFBD statType for yards, CFBD statType for TDs).
_STAT_CATEGORIES: dict[str, tuple[str, str, str]] = {
    "passing_yards": ("passing", "YDS", ""),
    "passing_tds": ("passing", "", "TD"),
    "rushing_yards": ("rushing", "YDS", ""),
    "rushing_tds": ("rushing", "", "TD"),
    "receiving_yards": ("receiving", "YDS", ""),
    "receiving_tds": ("receiving", "", "TD"),
}

#: Minimum real per-game rate before a market is worth offering -- same
#: rationale as fantasy_engine/betting/odds_generator.py's identical guard.
_MIN_RATE_BY_CATEGORY: dict[str, float] = {
    "passing_yards": 50.0,
    "passing_tds": 0.1,
    "rushing_yards": 10.0,
    "rushing_tds": 0.1,
    "receiving_yards": 10.0,
    "receiving_tds": 0.1,
}

_MIN_GAMES_PLAYED = 4


def _round_to_half(value: float) -> float:
    """Nearest sportsbook-style X.5 line -- never a whole number (see NFL/NBA generators for why)."""
    return round(value - 0.5) + 0.5


def _fetch_player_season_stats(season: int) -> list[dict[str, Any]]:
    import requests

    key = _api_key()
    if not key:
        return []
    try:
        response = requests.get(
            f"{CFBD_BASE_URL}/stats/player/season",
            params={"year": season},
            headers={"Authorization": f"Bearer {key}"},
            timeout=CFBD_TIMEOUT,
        )
        response.raise_for_status()
        rows = response.json()
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def generate_default_props(season: int) -> list[dict[str, Any]]:
    """Real per-game-rate prop lines from real season player stats. Empty if no CFBD_API_KEY is set.

    CFBD returns one row per (player, category, statType) rather than one
    full season line per player, so rows are grouped by player before a
    line can be computed. Games-played uses the player's team's real games
    played this season (from team_scoring_by_game) as a proxy -- CFBD's
    player-season endpoint does not directly carry individual games-played,
    and most players who accumulate a meaningful season stat line play
    close to their full team schedule; a documented simplification, not a
    fabricated number.
    """
    if not _api_key():
        return []
    rows = _fetch_player_season_stats(season)
    if not rows:
        return []

    team_games = {team: len(games) for team, games in team_scoring_by_game(season).items()}

    # (player, team) -> {(category, statType): value}
    by_player: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        player = _first(row, "player")
        team = _first(row, "team")
        category = _first(row, "category")
        stat_type = _first(row, "statType", "stat_type")
        stat_value = _first(row, "stat")
        if not player or not team or not category or stat_type is None or stat_value is None:
            continue
        try:
            value = float(stat_value)
        except (TypeError, ValueError):
            continue
        key = (normalize_player_name(str(player)), normalize_college_team_name(str(team)))
        by_player.setdefault(key, {})[(str(category), str(stat_type))] = value

    output: list[dict[str, Any]] = []
    for (player, team), stats in by_player.items():
        games = team_games.get(team, 0)
        if games < _MIN_GAMES_PLAYED:
            continue
        for market_category, (cfbd_category, yards_stat, tds_stat) in _STAT_CATEGORIES.items():
            stat_type = yards_stat or tds_stat
            value = stats.get((cfbd_category, stat_type))
            if value is None:
                continue
            rate = value / games
            if rate < _MIN_RATE_BY_CATEGORY[market_category]:
                continue
            output.append(
                {
                    "player_name": player,
                    "team": team,
                    "category": market_category,
                    "line": _round_to_half(rate),
                    "over_price": -110.0,
                    "under_price": -110.0,
                    "sportsbook": "default",
                    "basis": f"{season} real per-game rate ({games:g} team games)",
                }
            )
    return output


def write_default_props_file(season: int, *, path: Any = None) -> int:
    """Generate and write ``data/cfb_props.json``. Returns the number of rows written (0 if no key configured)."""
    import json
    from pathlib import Path

    rows = generate_default_props(season)
    target = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "data" / DATA_ROOT_NAME
    if rows:
        note = (
            "Our own default lines -- real per-game rates from the College Football Data API, "
            "never a sportsbook. Upload a file from the app to override or add to them."
        )
    else:
        note = (
            "Empty -- no CFBD_API_KEY was configured when this file was last generated. "
            "Set the CFBD_API_KEY environment variable and re-run `python -m modules.cfb_props_generator`."
        )
    payload = {"note": note, "generated_by": "modules.cfb_props_generator", "source_season": season, "props": rows}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    import datetime

    current_year = datetime.date.today().year
    count = write_default_props_file(current_year)
    print(f"Wrote {count} prop lines for season {current_year}" + ("" if count else " (no CFBD_API_KEY configured -- file written empty)"))
