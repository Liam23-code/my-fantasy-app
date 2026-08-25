"""Shared, sport-specific normalization for the MLB loaders (props, odds, injuries, lineups).

MLB is a fixed 30-team league (like NBA), so -- unlike CFB/CBB's college
punctuation-strip normalizer (modules/college_sports_common.py) -- team
names are normalized through a fixed alias table, mirroring
modules/sportsbook_parser.py's NBA table exactly. Player-name
normalization has no sport in it at all, so it's re-exported directly
from modules.sportsbook_parser rather than redefined.
"""
from __future__ import annotations

import re

from modules.sportsbook_parser import normalize_player_name

__all__ = ["normalize_player_name", "normalize_team_name", "normalize_category", "TEAM_ALIASES", "CATEGORY_WORDS", "STAT_CATEGORIES"]

#: The 7 stat categories this engine models -- see mlb_season_model.py.
STAT_CATEGORIES = ("hits", "home_runs", "rbi", "total_bases", "strikeouts", "walks", "stolen_bases")

TEAM_ALIASES = {
    "ARIZONA DIAMONDBACKS": "ARI", "ATLANTA BRAVES": "ATL", "BALTIMORE ORIOLES": "BAL", "BOSTON RED SOX": "BOS",
    "CHICAGO CUBS": "CHC", "CHICAGO WHITE SOX": "CWS", "CINCINNATI REDS": "CIN", "CLEVELAND GUARDIANS": "CLE",
    "COLORADO ROCKIES": "COL", "DETROIT TIGERS": "DET", "HOUSTON ASTROS": "HOU", "KANSAS CITY ROYALS": "KC",
    "LOS ANGELES ANGELS": "LAA", "LOS ANGELES DODGERS": "LAD", "MIAMI MARLINS": "MIA", "MILWAUKEE BREWERS": "MIL",
    "MINNESOTA TWINS": "MIN", "NEW YORK METS": "NYM", "NEW YORK YANKEES": "NYY", "ATHLETICS": "ATH",
    "OAKLAND ATHLETICS": "ATH", "PHILADELPHIA PHILLIES": "PHI", "PITTSBURGH PIRATES": "PIT", "SAN DIEGO PADRES": "SD",
    "SEATTLE MARINERS": "SEA", "SAN FRANCISCO GIANTS": "SF", "ST LOUIS CARDINALS": "STL", "TAMPA BAY RAYS": "TB",
    "TEXAS RANGERS": "TEX", "TORONTO BLUE JAYS": "TOR", "WASHINGTON NATIONALS": "WSH",
}
# A short, last-word alias (e.g. "ROCKIES" -> COL) is added for every team
# whose last word isn't already claimed -- "SOX" is genuinely ambiguous
# between the Red Sox and White Sox, so whichever team is inserted first
# above keeps it; the other stays reachable only by its full name or its
# own 2-3 letter abbreviation. First-inserted-wins, not overwritten.
for _full, _abbr in list(TEAM_ALIASES.items()):
    _last = _full.split()[-1]
    if _last not in TEAM_ALIASES:
        TEAM_ALIASES[_last] = _abbr
    TEAM_ALIASES[_abbr] = _abbr
del _full, _abbr, _last

#: Sportsbook-style market label -> our internal category id (see
#: STAT_CATEGORIES). Deliberately no single-letter or common-English-word
#: aliases ("a", "h", "so") -- unlike "hr"/"rbi"/"tb"/"bb"/"sb" (not real
#: English words), a bare "a" or "so" is a real word that can appear as a
#: standalone token inside unrelated text and false-positive match via
#: normalize_category's word-boundary check.
CATEGORY_WORDS = {
    "hits": "hits", "base hits": "hits",
    "home runs": "home_runs", "home run": "home_runs", "hr": "home_runs", "hrs": "home_runs",
    "rbi": "rbi", "rbis": "rbi", "runs batted in": "rbi",
    "total bases": "total_bases", "tb": "total_bases",
    "strikeouts": "strikeouts", "pitcher strikeouts": "strikeouts", "ks": "strikeouts",
    "walks": "walks", "bb": "walks", "base on balls": "walks",
    "stolen bases": "stolen_bases", "sb": "stolen_bases", "stolen base": "stolen_bases",
}


def normalize_team_name(value: str) -> str:
    """A real team name/city/nickname -> its 2-3 letter code, or the first 3 uppercased characters if unrecognized."""
    key = re.sub(r"[^A-Za-z0-9 ]", "", str(value or "")).strip().upper()
    key = " ".join(key.split())
    return TEAM_ALIASES.get(key, key[:3])


def normalize_category(value: str) -> str | None:
    """A sportsbook-style market label -> our internal category id, or ``None`` if unrecognized."""
    key = " ".join(re.sub(r"[^a-z0-9]", " ", str(value).lower()).split())
    for phrase, category in sorted(CATEGORY_WORDS.items(), key=lambda item: -len(item[0])):
        if phrase == key or f" {phrase} " in f" {key} ":
            return category
    return None
