"""Safe, offline parsing/normalization logic for sportsbook-shaped data.

Everything in this module is a pure transform: it takes a JSON payload, an
HTML string, or a raw label that the *caller* already obtained (from
``odds.json``, a user upload, or -- historically -- a live fetch) and turns
it into the app's normalized market/team/category vocabulary. Nothing here
makes a network request, and none of it depends on any function that does.

This is the "good half" of the former ``modules/sportsbook.py``: the
network-fetching half now lives, permanently disabled, in
``modules/sportsbook_scraper_disabled.py``. See that module's header for why.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import unicodedata
from typing import Any

from modules.data_quality import normalize_text, safe_number

TEAM_ALIASES = {
    "ATLANTA HAWKS": "ATL", "BOSTON CELTICS": "BOS", "BROOKLYN NETS": "BKN", "CHARLOTTE HORNETS": "CHA",
    "CHICAGO BULLS": "CHI", "CLEVELAND CAVALIERS": "CLE", "DALLAS MAVERICKS": "DAL", "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET", "GOLDEN STATE WARRIORS": "GSW", "HOUSTON ROCKETS": "HOU", "INDIANA PACERS": "IND",
    "LA CLIPPERS": "LAC", "LOS ANGELES CLIPPERS": "LAC", "LOS ANGELES LAKERS": "LAL", "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA", "MILWAUKEE BUCKS": "MIL", "MINNESOTA TIMBERWOLVES": "MIN", "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK", "OKLAHOMA CITY THUNDER": "OKC", "ORLANDO MAGIC": "ORL", "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHX", "PORTLAND TRAIL BLAZERS": "POR", "SACRAMENTO KINGS": "SAC", "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR", "UTAH JAZZ": "UTA", "WASHINGTON WIZARDS": "WAS",
}
for _full, _abbr in list(TEAM_ALIASES.items()):
    TEAM_ALIASES[_full.split()[-1]] = _abbr
    TEAM_ALIASES[_abbr] = _abbr

CATEGORY_WORDS = {
    "points": "points", "pts": "points", "rebounds": "rebounds", "rebs": "rebounds",
    "assists": "assists", "asts": "assists", "pra": "PRA", "points rebounds assists": "PRA",
    "3pm": "3PM", "three pointers made": "3PM", "3 pointers made": "3PM",
}


def now_iso() -> str:
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
    """Map a sportsbook-style market label (e.g. ``"DK_RUSH_YDS"``-ish free text) to an internal market id."""
    key = " ".join(re.sub(r"[^a-z0-9]", " ", str(value).lower()).split())
    for phrase, category in sorted(CATEGORY_WORDS.items(), key=lambda item: -len(item[0])):
        if phrase in key:
            return category
    return None


def price_to_implied_probability(price: float) -> float:
    """American odds -> raw (vig-included) implied win probability."""
    price = float(price)
    if price == 0:
        raise ValueError("American odds price cannot be 0")
    decimal_odds = 1.0 + price / 100.0 if price > 0 else 1.0 + 100.0 / abs(price)
    return 1.0 / decimal_odds


def _bounded_number(value: Any) -> float | None:
    """A line value in the sane 0-100 range, or None -- guards against parsing garbage as a real line."""
    try:
        result = safe_number(value, -1)
        return result if 0 <= result <= 100 else None
    except (TypeError, ValueError):
        return None


def _first(record: dict, *keys: str) -> Any:
    lowered = {str(k).lower(): v for k, v in record.items()}
    return next((lowered[k.lower()] for k in keys if k.lower() in lowered), None)


def walk_market_records(
    node: Any,
    source: str,
    output: list[dict],
    *,
    inherited_category: str | None = None,
    inherited_player: str | None = None,
    inherited_team: str = "",
) -> None:
    """Recursively recognize nested market records in an already-obtained JSON payload.

    ``node`` is data the caller already has in memory -- parsed from
    ``odds.json``, a user upload, or any other structured source. This
    function performs no I/O of its own.
    """
    if isinstance(node, list):
        for item in node:
            walk_market_records(
                item, source, output,
                inherited_category=inherited_category, inherited_player=inherited_player, inherited_team=inherited_team,
            )
        return
    if not isinstance(node, dict):
        return
    label = _first(node, "marketName", "market_name", "market", "description", "label", "statType")
    category = normalize_category(str(label or "")) or inherited_category
    player_value = _first(node, "playerName", "player_name", "participantName", "athleteName")
    player = normalize_player_name(str(player_value)) if player_value else inherited_player
    team_value = _first(node, "team", "teamName", "teamAbbreviation")
    team = normalize_team_name(str(team_value)) if team_value else inherited_team
    line = _bounded_number(_first(node, "line", "handicap", "points", "total", "value"))
    if category and player and line is not None:
        output.append(
            {
                "player_name": player,
                "team": team,
                "category": category,
                "line": round(line, 2),
                "sportsbook": source,
                "timestamp": str(_first(node, "timestamp", "updatedAt", "lastUpdated") or now_iso()),
            }
        )
    for value in node.values():
        if isinstance(value, (dict, list)):
            walk_market_records(value, source, output, inherited_category=category, inherited_player=player, inherited_team=team)


def parse_market_json(payload: Any, source: str) -> list[dict]:
    """Parse an already-obtained JSON payload (e.g. from a user-uploaded file) into normalized market rows."""
    results: list[dict] = []
    walk_market_records(payload, source, results)
    unique = {(r["player_name"], r["category"], r["line"]): r for r in results if r.get("category")}
    return list(unique.values())


def parse_market_json_text(text: str, source: str) -> list[dict]:
    """Parse a JSON string (e.g. the contents of an uploaded ``.json`` file)."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    return parse_market_json(payload, source)
