"""Shared, sport-specific normalization for the NHL loaders (props, odds, injuries).

NHL is a fixed 32-team league (like NBA/MLB), so team names are
normalized through a fixed alias table, mirroring
modules/sportsbook_parser.py's NBA table and modules/mlb_common.py's MLB
table exactly. Player-name normalization has no sport in it at all, so
it's re-exported directly from modules.sportsbook_parser rather than
redefined.
"""
from __future__ import annotations

import re

from modules.sportsbook_parser import normalize_player_name

__all__ = ["normalize_player_name", "normalize_team_name", "normalize_category", "TEAM_ALIASES", "CATEGORY_WORDS", "STAT_CATEGORIES"]

#: The 4 stat categories this lightweight engine models.
STAT_CATEGORIES = ("shots", "goals", "assists", "saves")

TEAM_ALIASES = {
    "ANAHEIM DUCKS": "ANA", "BOSTON BRUINS": "BOS", "BUFFALO SABRES": "BUF", "CALGARY FLAMES": "CGY",
    "CAROLINA HURRICANES": "CAR", "CHICAGO BLACKHAWKS": "CHI", "COLORADO AVALANCHE": "COL",
    "COLUMBUS BLUE JACKETS": "CBJ", "DALLAS STARS": "DAL", "DETROIT RED WINGS": "DET", "EDMONTON OILERS": "EDM",
    "FLORIDA PANTHERS": "FLA", "LOS ANGELES KINGS": "LAK", "MINNESOTA WILD": "MIN", "MONTREAL CANADIENS": "MTL",
    "NASHVILLE PREDATORS": "NSH", "NEW JERSEY DEVILS": "NJD", "NEW YORK ISLANDERS": "NYI", "NEW YORK RANGERS": "NYR",
    "OTTAWA SENATORS": "OTT", "PHILADELPHIA FLYERS": "PHI", "PITTSBURGH PENGUINS": "PIT", "SAN JOSE SHARKS": "SJS",
    "SEATTLE KRAKEN": "SEA", "ST LOUIS BLUES": "STL", "TAMPA BAY LIGHTNING": "TBL", "TORONTO MAPLE LEAFS": "TOR",
    "UTAH MAMMOTH": "UTA", "VANCOUVER CANUCKS": "VAN", "VEGAS GOLDEN KNIGHTS": "VGK", "WASHINGTON CAPITALS": "WSH",
    "WINNIPEG JETS": "WPG",
}
# Short, last-word aliases, first-inserted-wins on any collision -- see
# modules/mlb_common.py's identical guarded loop for why this is safer
# than an unconditional overwrite.
for _full, _abbr in list(TEAM_ALIASES.items()):
    _last = _full.split()[-1]
    if _last not in TEAM_ALIASES:
        TEAM_ALIASES[_last] = _abbr
    TEAM_ALIASES[_abbr] = _abbr
del _full, _abbr, _last

#: Deliberately no single-letter aliases ("g", "a") -- see
#: modules/mlb_common.py's identical note; "a" in particular is a common
#: standalone English word that can false-positive match unrelated text.
CATEGORY_WORDS = {
    "shots": "shots", "shots on goal": "shots", "sog": "shots",
    "goals": "goals",
    "assists": "assists", "asts": "assists",
    "saves": "saves", "goalie saves": "saves", "sv": "saves",
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
