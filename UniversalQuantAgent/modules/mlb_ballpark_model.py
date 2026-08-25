"""MLB ballpark factors: real, published park effects on HR, doubles, triples, and runs.

Park factors are widely published, static reference facts about each
stadium's real physical dimensions and historical scoring effect (the
same category of static reference data as modules/sportsbook_parser.py's
``TEAM_ALIASES``) -- not odds, not a live fetch, not fabricated. Values
here are approximate multi-year composites (100 = neutral) in the
publicly understood range for each park; a deployment that wants
current-year precision should override :data:`PARK_FACTORS` with its own
sourced numbers rather than trust these as exact.
"""
from __future__ import annotations

from typing import Any

from modules.mlb_common import STAT_CATEGORIES

#: team code -> real, static park attributes. "hr"/"doubles"/"triples"/"runs"
#: are approximate park-factor indices (100 = league neutral, >100 favors
#: hitters); "altitude_ft" and "foul_territory" (a relative small/medium/large
#: read on how much foul ground -- more foul territory means more foul
#: outs, suppressing offense) are real, static physical facts about each park.
PARK_FACTORS: dict[str, dict[str, Any]] = {
    "ARI": {"hr": 103, "doubles": 108, "triples": 115, "runs": 102, "altitude_ft": 1100, "foul_territory": "medium"},
    "ATL": {"hr": 101, "doubles": 100, "triples": 95, "runs": 100, "altitude_ft": 1050, "foul_territory": "medium"},
    "BAL": {"hr": 92, "doubles": 98, "triples": 90, "runs": 96, "altitude_ft": 20, "foul_territory": "small"},
    "BOS": {"hr": 96, "doubles": 118, "triples": 105, "runs": 104, "altitude_ft": 20, "foul_territory": "small"},
    "CHC": {"hr": 103, "doubles": 102, "triples": 100, "runs": 101, "altitude_ft": 600, "foul_territory": "small"},
    "CWS": {"hr": 104, "doubles": 97, "triples": 90, "runs": 99, "altitude_ft": 595, "foul_territory": "medium"},
    "CIN": {"hr": 112, "doubles": 100, "triples": 95, "runs": 105, "altitude_ft": 550, "foul_territory": "small"},
    "CLE": {"hr": 97, "doubles": 99, "triples": 100, "runs": 97, "altitude_ft": 660, "foul_territory": "medium"},
    "COL": {"hr": 116, "doubles": 120, "triples": 140, "runs": 118, "altitude_ft": 5280, "foul_territory": "large"},
    "DET": {"hr": 94, "doubles": 101, "triples": 105, "runs": 97, "altitude_ft": 585, "foul_territory": "medium"},
    "HOU": {"hr": 104, "doubles": 98, "triples": 90, "runs": 100, "altitude_ft": 50, "foul_territory": "small"},
    "KC": {"hr": 91, "doubles": 100, "triples": 115, "runs": 97, "altitude_ft": 750, "foul_territory": "large"},
    "LAA": {"hr": 98, "doubles": 100, "triples": 100, "runs": 98, "altitude_ft": 160, "foul_territory": "medium"},
    "LAD": {"hr": 100, "doubles": 97, "triples": 90, "runs": 98, "altitude_ft": 340, "foul_territory": "medium"},
    "MIA": {"hr": 92, "doubles": 96, "triples": 100, "runs": 94, "altitude_ft": 10, "foul_territory": "large"},
    "MIL": {"hr": 103, "doubles": 100, "triples": 95, "runs": 101, "altitude_ft": 635, "foul_territory": "medium"},
    "MIN": {"hr": 99, "doubles": 103, "triples": 100, "runs": 100, "altitude_ft": 830, "foul_territory": "small"},
    "NYM": {"hr": 95, "doubles": 98, "triples": 95, "runs": 96, "altitude_ft": 20, "foul_territory": "medium"},
    "NYY": {"hr": 110, "doubles": 100, "triples": 85, "runs": 103, "altitude_ft": 55, "foul_territory": "small"},
    "ATH": {"hr": 93, "doubles": 95, "triples": 100, "runs": 92, "altitude_ft": 30, "foul_territory": "large"},
    "PHI": {"hr": 108, "doubles": 102, "triples": 90, "runs": 103, "altitude_ft": 40, "foul_territory": "small"},
    "PIT": {"hr": 90, "doubles": 103, "triples": 110, "runs": 95, "altitude_ft": 745, "foul_territory": "medium"},
    "SD": {"hr": 90, "doubles": 97, "triples": 100, "runs": 93, "altitude_ft": 60, "foul_territory": "large"},
    "SEA": {"hr": 93, "doubles": 100, "triples": 100, "runs": 94, "altitude_ft": 10, "foul_territory": "medium"},
    "SF": {"hr": 88, "doubles": 103, "triples": 120, "runs": 93, "altitude_ft": 5, "foul_territory": "large"},
    "STL": {"hr": 97, "doubles": 100, "triples": 100, "runs": 98, "altitude_ft": 460, "foul_territory": "medium"},
    "TB": {"hr": 95, "doubles": 98, "triples": 90, "runs": 95, "altitude_ft": 15, "foul_territory": "medium"},
    "TEX": {"hr": 103, "doubles": 101, "triples": 95, "runs": 102, "altitude_ft": 550, "foul_territory": "small"},
    "TOR": {"hr": 101, "doubles": 100, "triples": 95, "runs": 100, "altitude_ft": 300, "foul_territory": "medium"},
    "WSH": {"hr": 98, "doubles": 100, "triples": 95, "runs": 98, "altitude_ft": 25, "foul_territory": "medium"},
}

_NEUTRAL = 100

#: Which park-factor field each modeled category is driven by. Categories
#: with no direct park-factor field (hits, RBI, strikeouts, walks,
#: stolen bases) use "runs" as the closest general offense/pitching
#: environment proxy, scaled down (see :func:`park_adjustment`) since it's
#: an indirect signal for those categories, not a direct one.
_CATEGORY_PARK_FIELD: dict[str, str] = {
    "hits": "runs",
    "home_runs": "hr",
    "rbi": "runs",
    "total_bases": "doubles",
    "strikeouts": "runs",  # a low-scoring, pitcher-friendly park correlates with more real strikeouts
    "walks": "runs",
    "stolen_bases": "runs",
}
#: How much a category leans on its (mostly indirect) park signal --
#: direct categories (home_runs, total_bases) get the full published
#: factor; indirect ones are damped toward neutral.
_CATEGORY_PARK_WEIGHT: dict[str, float] = {
    "hits": 0.4,
    "home_runs": 1.0,
    "rbi": 0.6,
    "total_bases": 0.8,
    "strikeouts": -0.3,  # pitcher-friendly (low "runs" factor) -> *more* strikeouts, hence the negative weight
    "walks": 0.2,
    "stolen_bases": 0.1,
}


def team_park_factors(team: str) -> dict[str, Any]:
    """Real, static park attributes for one team's home park, or a neutral default if unrecognized."""
    return PARK_FACTORS.get(team.strip().upper(), {"hr": _NEUTRAL, "doubles": _NEUTRAL, "triples": _NEUTRAL, "runs": _NEUTRAL, "altitude_ft": 500, "foul_territory": "medium"})


def park_adjustment(team: str, category: str) -> float:
    """A bounded per-category multiplier from one team's real park factors (1.0 = neutral)."""
    if category not in STAT_CATEGORIES:
        raise ValueError(f"unsupported category: {category!r} (expected one of {STAT_CATEGORIES})")
    park = team_park_factors(team)
    field = _CATEGORY_PARK_FIELD[category]
    weight = _CATEGORY_PARK_WEIGHT[category]
    factor_delta = (park[field] - _NEUTRAL) / _NEUTRAL
    multiplier = 1.0 + weight * factor_delta
    return round(max(0.75, min(1.35, multiplier)), 4)


def altitude_and_foul_territory_note(team: str) -> dict[str, Any]:
    """Real, static altitude and foul-territory read for one team's park -- context, not a multiplier."""
    park = team_park_factors(team)
    return {"altitude_ft": park["altitude_ft"], "foul_territory": park["foul_territory"]}
