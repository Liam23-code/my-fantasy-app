"""Real, live CBB schedule data -- today's games, no odds.

Uses ESPN's public scoreboard API (real, structured JSON -- not a
sportsbook, not HTML scraping; see modules/cbb_team_model.py for the full
legal-basis discussion and offline_data_contract.md for what this
exception does and does not permit). Mirrors modules/nba_schedule.py's
shape and fail-closed convention exactly.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from betting.cache_utils import ttl_cache

from modules.cbb_team_model import ESPN_CBB_BASE, ESPN_TIMEOUT
from modules.college_sports_common import normalize_college_team_name

#: Today's real schedule can change within a session; short TTL balances
#: freshness against re-fetching on every rerun -- same rationale as
#: modules/nba_schedule.py's identical constant.
_SCHEDULE_CACHE_SECONDS = 60


def _fetch_scoreboard(game_date: date | None) -> list[dict[str, Any]]:
    import requests

    params = {}
    if game_date is not None:
        params["dates"] = game_date.strftime("%Y%m%d")
    response = requests.get(f"{ESPN_CBB_BASE}/scoreboard", params=params, timeout=ESPN_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    results = []
    for event in data.get("events") or []:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competitors = competitions[0].get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_code = normalize_college_team_name(str((home.get("team") or {}).get("abbreviation") or ""))
        away_code = normalize_college_team_name(str((away.get("team") or {}).get("abbreviation") or ""))
        if not home_code or not away_code:
            continue
        results.append({"home_team": home_code, "away_team": away_code, "start_time": str(event.get("date") or "")})
    return results


@ttl_cache(_SCHEDULE_CACHE_SECONDS)
def fetch_todays_games(game_date: date | None = None) -> list[dict[str, Any]]:
    """Real home/away matchups for ``game_date`` (default: today), via ESPN's live scoreboard.

    Returns ``[]`` on any failure (network unavailable, off-season, no
    games scheduled) rather than raising -- an empty slate is a normal
    state, not an error, matching every other loader in this codebase.
    """
    try:
        return _fetch_scoreboard(game_date)
    except Exception:
        return []
