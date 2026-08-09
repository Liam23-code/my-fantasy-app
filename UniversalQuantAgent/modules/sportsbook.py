"""Free, best-effort NBA player-prop ingestion.

Sportsbook pages change often and can be geofenced.  This module therefore uses
configurable JSON endpoints first, then inspects public HTML as a fallback.  It
returns an empty list when no real line can be verified; it never invents odds.
Set UQA_<BOOK>_PROPS_URL to a permitted public/partner feed when available.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
import re
import unicodedata
from typing import Any

import requests
from bs4 import BeautifulSoup
from modules.data_quality import normalize_text, safe_number

BOOKS = {
    "DraftKings": ("UQA_DRAFTKINGS_PROPS_URL", "https://sportsbook.draftkings.com/leagues/basketball/nba"),
    "FanDuel": ("UQA_FANDUEL_PROPS_URL", "https://sportsbook.fanduel.com/navigation/nba"),
    "BetMGM": ("UQA_BETMGM_PROPS_URL", "https://www.betmgm.com/en/sports/basketball-7/betting/usa-9/nba-6004"),
    "Caesars": ("UQA_CAESARS_PROPS_URL", "https://sportsbook.caesars.com/us/co/bet/basketball/events/all"),
    "ESPN BET": ("UQA_ESPNBET_PROPS_URL", "https://espnbet.com/sport/basketball/organization/united-states/competition/nba"),
}

TEAM_ALIASES = {
    "ATLANTA HAWKS":"ATL","BOSTON CELTICS":"BOS","BROOKLYN NETS":"BKN","CHARLOTTE HORNETS":"CHA",
    "CHICAGO BULLS":"CHI","CLEVELAND CAVALIERS":"CLE","DALLAS MAVERICKS":"DAL","DENVER NUGGETS":"DEN",
    "DETROIT PISTONS":"DET","GOLDEN STATE WARRIORS":"GSW","HOUSTON ROCKETS":"HOU","INDIANA PACERS":"IND",
    "LA CLIPPERS":"LAC","LOS ANGELES CLIPPERS":"LAC","LOS ANGELES LAKERS":"LAL","MEMPHIS GRIZZLIES":"MEM",
    "MIAMI HEAT":"MIA","MILWAUKEE BUCKS":"MIL","MINNESOTA TIMBERWOLVES":"MIN","NEW ORLEANS PELICANS":"NOP",
    "NEW YORK KNICKS":"NYK","OKLAHOMA CITY THUNDER":"OKC","ORLANDO MAGIC":"ORL","PHILADELPHIA 76ERS":"PHI",
    "PHOENIX SUNS":"PHX","PORTLAND TRAIL BLAZERS":"POR","SACRAMENTO KINGS":"SAC","SAN ANTONIO SPURS":"SAS",
    "TORONTO RAPTORS":"TOR","UTAH JAZZ":"UTA","WASHINGTON WIZARDS":"WAS",
}
for full, abbr in list(TEAM_ALIASES.items()):
    TEAM_ALIASES[full.split()[-1]] = abbr
    TEAM_ALIASES[abbr] = abbr

CATEGORY_WORDS = {
    "points":"points", "pts":"points", "rebounds":"rebounds", "rebs":"rebounds",
    "assists":"assists", "asts":"assists", "pra":"PRA", "points rebounds assists":"PRA",
    "3pm":"3PM", "three pointers made":"3PM", "3 pointers made":"3PM",
}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def normalize_player_name(value: str) -> str:
    """Normalize accents, suffixes, and whitespace without losing readability."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"\b(JR|SR|II|III|IV)\.?\b", "", text, flags=re.I)
    return " ".join(re.sub(r"[^A-Za-z .'-]", " ", text).split()).title()

def normalize_team_name(value: str) -> str:
    key = normalize_text(value).upper()
    return TEAM_ALIASES.get(key, key[:3])

def normalize_category(value: str) -> str | None:
    key = " ".join(re.sub(r"[^a-z0-9]", " ", str(value).lower()).split())
    for phrase, category in sorted(CATEGORY_WORDS.items(), key=lambda item: -len(item[0])):
        if phrase in key:
            return category
    return None

def _number(value: Any) -> float | None:
    try:
        result = safe_number(value, -1)
        return result if 0 <= result <= 100 else None
    except (TypeError, ValueError):
        return None

def _first(record: dict, *keys: str) -> Any:
    lowered = {str(k).lower(): v for k, v in record.items()}
    return next((lowered[k.lower()] for k in keys if k.lower() in lowered), None)

def _walk_records(node: Any, sportsbook: str, output: list[dict],
                  inherited_category: str | None = None,
                  inherited_player: str | None = None,
                  inherited_team: str = "") -> None:
    """Recursively recognize nested market-feed records from different books."""
    if isinstance(node, list):
        for item in node:
            _walk_records(item, sportsbook, output, inherited_category,
                          inherited_player, inherited_team)
        return
    if not isinstance(node, dict):
        return
    label = _first(node, "marketName", "market_name", "market", "description", "label", "statType")
    category = normalize_category(str(label or "")) or inherited_category
    player_value = _first(node, "playerName", "player_name", "participantName", "athleteName")
    player = normalize_player_name(str(player_value)) if player_value else inherited_player
    team_value = _first(node, "team", "teamName", "teamAbbreviation")
    team = normalize_team_name(str(team_value)) if team_value else inherited_team
    line = _number(_first(node, "line", "handicap", "points", "total", "value"))
    if category and player and line is not None:
        output.append({
            "player_name": player, "team": team, "category": category,
            "line": round(line, 2), "sportsbook": sportsbook,
            "timestamp": str(_first(node, "timestamp", "updatedAt", "lastUpdated") or _now()),
        })
    for value in node.values():
        if isinstance(value, (dict, list)):
            _walk_records(value, sportsbook, output, category, player, team)
def _parse_html(text: str, sportsbook: str) -> list[dict]:
    results: list[dict] = []
    soup = BeautifulSoup(text, "html.parser")
    for script in soup.find_all("script"):
        raw = script.string or script.get_text()
        if raw.strip().startswith(("{", "[")):
            try:
                _walk_records(json.loads(raw), sportsbook, results)
            except (json.JSONDecodeError, TypeError):
                pass
    # Simple fallback for accessible, server-rendered market rows.
    pattern = re.compile(r"([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3})\s+(Points|Rebounds|Assists|PRA|3PM)\s+(\d+(?:\.5)?)", re.I)
    for player, category, line in pattern.findall(soup.get_text(" ", strip=True)):
        results.append({"player_name":normalize_player_name(player), "team":"",
                        "category":normalize_category(category), "line":float(line),
                        "sportsbook":sportsbook, "timestamp":_now()})
    return results

def _fetch_book(sportsbook: str, timeout: int = 12) -> list[dict]:
    env_name, public_url = BOOKS[sportsbook]
    configured_url = os.getenv(env_name)
    # A failed configured JSON feed falls back to the book's public HTML page.
    urls = list(dict.fromkeys(url for url in (configured_url, public_url) if url))
    results: list[dict] = []
    for url in urls:
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0 UniversalQuantAgent/1.0"})
            response.raise_for_status()
            try:
                _walk_records(response.json(), sportsbook, results)
            except (requests.JSONDecodeError, ValueError):
                results.extend(_parse_html(response.text, sportsbook))
            if results:
                break
        except requests.RequestException:
            continue
    unique = {(r["player_name"], r["category"], r["line"]): r for r in results if r.get("category")}
    return list(unique.values())

def fetch_draftkings_props() -> list[dict]: return _fetch_book("DraftKings")
def fetch_fanduel_props() -> list[dict]: return _fetch_book("FanDuel")
def fetch_betmgm_props() -> list[dict]: return _fetch_book("BetMGM")
def fetch_caesars_props() -> list[dict]: return _fetch_book("Caesars")
def fetch_espnbet_props() -> list[dict]: return _fetch_book("ESPN BET")

def fetch_all_sportsbook_props() -> list[dict]:
    """Fetch all five books independently so one outage cannot stop the slate."""
    results: list[dict] = []
    for function in (fetch_draftkings_props, fetch_fanduel_props, fetch_betmgm_props, fetch_caesars_props, fetch_espnbet_props):
        results.extend(function())
    return results

def fetch_daily_games(game_date: date | None = None) -> list[dict]:
    """Read today's NBA schedule from ESPN's public scoreboard endpoint."""
    day = game_date or date.today()
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={day:%Y%m%d}"
    try:
        data = requests.get(url, timeout=12, headers={"User-Agent":"UniversalQuantAgent/1.0"}).json()
    except (requests.RequestException, ValueError):
        return []
    games = []
    for event in data.get("events", []):
        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        games.append({"home_team":normalize_team_name(home.get("team", {}).get("abbreviation", "")),
                      "away_team":normalize_team_name(away.get("team", {}).get("abbreviation", "")),
                      "start_time":event.get("date", "")})
    return games
