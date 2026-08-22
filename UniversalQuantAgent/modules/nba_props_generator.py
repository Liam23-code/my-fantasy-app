"""Generate our own default NBA prop lines -- real per-game rates, never a sportsbook.

A one-off generation-time tool (mirrors
``fantasy_engine/betting/odds_generator.py``'s NFL equivalent): it makes a
real, legal, live call to the NBA's own stats API (``nba_api`` -- already a
first-class dependency of this app; see modules/nba_advanced.py,
modules/projections.py) to compute real per-game rates for real players
from a real, completed season, then writes them to ``data/nba_props.json``
as this app's own default lines.

Not imported by any Streamlit page and not on the request-time path -- run
it directly (``python -m modules.nba_props_generator``) to regenerate the
default file after a season completes. The live app only ever reads the
resulting static file via :mod:`modules.nba_props_loader`; it never calls
``nba_api`` for props at request time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.nba_advanced import latest_season
from modules.sportsbook_parser import normalize_player_name, normalize_team_name

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PROPS_PATH = DATA_ROOT / "nba_props.json"

#: category -> minimum real per-game rate before a market is worth offering,
#: checked against the *raw rate*, not the rounded line -- a rounded line is
#: always >= 0.5 by construction (see _round_to_half), so checking it can
#: never filter anything. Without this, e.g. a center who averages 0.1
#: 3-pointers/game (a real stat, but not a real market) still gets a "3PM"
#: line, and a model comparing a real ~0-mean distribution to that line
#: reports a huge, meaningless "edge". Same rationale as the NFL generator's
#: _MIN_RATE_BY_MARKET.
_MIN_RATE_BY_CATEGORY: dict[str, float] = {
    "points": 8.0,
    "rebounds": 3.0,
    "assists": 2.0,
    "PRA": 12.0,
    "3PM": 0.5,
}

#: Below this many real games, a season-total/games rate is too noisy to
#: trust as a default line.
_MIN_GAMES_PLAYED = 15

#: Below this many real minutes/game, a player isn't a rotation piece --
#: their rate is either garbage-time noise or a stat line no one would
#: realistically price a prop on.
_MIN_MINUTES_PER_GAME = 15.0


def _round_to_half(value: float) -> float:
    """Nearest sportsbook-style X.5 line -- never a whole number.

    A real two-way prop line always lands on a half point specifically so
    it can never push; ``round(value * 2) / 2`` doesn't guarantee that (a
    value like 0.9 rounds to 1.0, not 0.5 or 1.5). This always returns
    ``n + 0.5`` for the integer ``n`` closest to ``value - 0.5``.
    """
    return round(value - 0.5) + 0.5


def generate_default_props(season: str | None = None, *, pool_size: int = 175) -> list[dict[str, Any]]:
    """Real per-game-rate prop lines for the ``pool_size`` most-used real players.

    A player's line is their real season-total stat (points, rebounds,
    assists, 3-pointers made) divided by real games played, rounded to a
    sportsbook-style half point. Makes one live call to the NBA's own stats
    API -- not a sportsbook, and nothing here is fabricated.
    """
    from nba_api.stats.endpoints import leaguedashplayerstats

    season = season or latest_season()
    frame = leaguedashplayerstats.LeagueDashPlayerStats(season=season, timeout=30).get_data_frames()[0]

    # A player traded mid-season has one row per team plus a combined "TOT"
    # row; the combined row always has the highest GP for that player, so
    # keeping max-GP-per-player selects it automatically without needing to
    # special-case the "TOT" label.
    frame = frame.sort_values("GP", ascending=False).drop_duplicates(subset="PLAYER_ID", keep="first")
    frame = frame[(frame["GP"] >= _MIN_GAMES_PLAYED) & (frame["MIN"] / frame["GP"] >= _MIN_MINUTES_PER_GAME)]
    frame = frame.sort_values("MIN", ascending=False).head(pool_size)

    rows: list[dict[str, Any]] = []
    for _, player in frame.iterrows():
        games = float(player["GP"])
        name = normalize_player_name(str(player["PLAYER_NAME"]))
        team_raw = str(player["TEAM_ABBREVIATION"])
        team = "" if team_raw == "TOT" else normalize_team_name(team_raw)
        rates = {
            "points": float(player["PTS"]) / games,
            "rebounds": float(player["REB"]) / games,
            "assists": float(player["AST"]) / games,
            "PRA": (float(player["PTS"]) + float(player["REB"]) + float(player["AST"])) / games,
            "3PM": float(player["FG3M"]) / games,
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
                    "basis": f"{season} real per-game rate ({games:g} games)",
                }
            )
    return rows


def write_default_props_file(season: str | None = None, *, pool_size: int = 175, path: Path | None = None) -> int:
    """Generate and write ``data/nba_props.json``. Returns the number of rows written."""
    season = season or latest_season()
    rows = generate_default_props(season, pool_size=pool_size)
    payload = {
        "note": (
            "Our own default lines -- real per-game rates from the NBA's own stats API, "
            "never a sportsbook. Upload a file from the app to override or add to them."
        ),
        "generated_by": "modules.nba_props_generator",
        "source_season": season,
        "props": rows,
    }
    target = path or DEFAULT_PROPS_PATH
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    count = write_default_props_file()
    print(f"Wrote {count} prop lines to {DEFAULT_PROPS_PATH}")
