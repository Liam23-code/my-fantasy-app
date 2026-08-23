"""Real, live NBA schedule data -- today's games, no odds.

Uses nba_api's public live-scoreboard service (the same real-time
schedule/score feed the rest of this app already treats as a first-class
dependency -- see modules/nba_advanced.py, modules/sports.py) to answer
"which NBA games are today, and who's playing whom." This is live
game-data ingestion, which is explicitly allowed; it is not sportsbook-odds
ingestion, which remains permanently disabled -- see
modules/sportsbook_scraper_disabled.py for what stays off-limits.

The underlying feed also carries a partner-sportsbook odds field
(``pbOdds``) on each game. That field is never read here -- only team and
timing fields are.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from betting.cache_utils import ttl_cache

from modules.sportsbook_parser import normalize_team_name

#: Today's schedule can change within a session (a postponement, a
#: same-day addition) but not every request needs a fresh live call --
#: 60s balances freshness against hammering the live feed on every rerun.
_SCHEDULE_CACHE_SECONDS = 60


def _team_code(side: dict[str, Any]) -> str:
    return normalize_team_name(str(side.get("teamTricode") or side.get("teamAbbreviation") or ""))


def _from_live_scoreboard() -> list[dict[str, Any]]:
    """Today's games from nba_api's live scoreboard endpoint."""
    from nba_api.live.nba.endpoints import scoreboard

    payload = scoreboard.ScoreBoard().get_dict()
    games = ((payload or {}).get("scoreboard") or {}).get("games") or []
    results = []
    for game in games:
        home, away = game.get("homeTeam") or {}, game.get("awayTeam") or {}
        home_code, away_code = _team_code(home), _team_code(away)
        if home_code and away_code:
            results.append(
                {"home_team": home_code, "away_team": away_code, "start_time": str(game.get("gameTimeUTC") or "")}
            )
    return results


def _from_stats_scoreboard(game_date: date) -> list[dict[str, Any]]:
    """A specific date's games from nba_api's stats-scoreboard endpoint (numeric team ids -> abbreviations)."""
    from nba_api.stats.endpoints import scoreboardv2
    from nba_api.stats.static import teams

    team_codes = {team["id"]: team["abbreviation"] for team in teams.get_teams()}
    frames = scoreboardv2.ScoreboardV2(game_date=game_date.isoformat()).get_data_frames()
    header = frames[0] if frames else None
    if header is None or header.empty:
        return []
    results = []
    for _, row in header.iterrows():
        home_code = normalize_team_name(str(team_codes.get(row.get("HOME_TEAM_ID"), "")))
        away_code = normalize_team_name(str(team_codes.get(row.get("VISITOR_TEAM_ID"), "")))
        if home_code and away_code:
            results.append({"home_team": home_code, "away_team": away_code, "start_time": str(row.get("GAME_DATE_EST") or "")})
    return results


@ttl_cache(_SCHEDULE_CACHE_SECONDS)
def fetch_todays_games(game_date: date | None = None) -> list[dict[str, Any]]:
    """Real home/away matchups for ``game_date`` (default: today), via the NBA's live schedule feed.

    Returns ``[]`` on any failure (network unavailable, off-season, no
    games scheduled) rather than raising -- an empty slate is a normal
    state, not an error, matching the injury/odds loaders' convention.
    Cached for :data:`_SCHEDULE_CACHE_SECONDS` -- every caller (this
    module's own tests aside) shares one cache, so `daily_slate.py`,
    `props.py`, `recommendations.py`, and the Streamlit page don't each
    make their own live call within the same short window.
    """
    try:
        if game_date is None or game_date == date.today():
            return _from_live_scoreboard()
        return _from_stats_scoreboard(game_date)
    except Exception:
        return []
