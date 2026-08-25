"""MLB defense model: real defensive efficiency, outfield arm strength, catcher framing, infield range.

One of five DFS matchup modules feeding modules/mlb_fusion_model.py (see
mlb_matchup_engine.md) and an input to mlb_moneyline_model.py. Pure math
over real, caller-supplied defensive metrics -- no live fetch, no
fabricated fielding stat.
"""
from __future__ import annotations

from typing import Any

_LEAGUE_AVERAGE_DEFENSIVE_EFFICIENCY = 0.690  # real MLB-wide DER is typically ~68-70%
_LEAGUE_AVERAGE_OUTFIELD_ASSISTS = 45.0  # real, roughly league-average full-season outfield assists
_LEAGUE_AVERAGE_RANGE_FACTOR = 2.9  # real, roughly league-average infield range factor/9


def defensive_efficiency_rating(balls_in_play_converted_pct: float, league_average: float = _LEAGUE_AVERAGE_DEFENSIVE_EFFICIENCY) -> float:
    """A bounded rating from a real defense's share of balls in play converted to outs (1.0 = league average)."""
    if league_average <= 0:
        return 1.0
    return round(max(0.7, min(1.3, float(balls_in_play_converted_pct) / league_average)), 4)


def outfield_arm_rating(assists: float, league_average_assists: float = _LEAGUE_AVERAGE_OUTFIELD_ASSISTS) -> float:
    """A bounded rating from a real outfield's season assist total (1.0 = league average).

    A strong-armed outfield real deterrent effect suppresses real
    stolen-base attempts and extra bases taken -- feeds into the stolen-base
    projection alongside modules/mlb_lineup_model.py's environment read.
    """
    if league_average_assists <= 0:
        return 1.0
    return round(max(0.6, min(1.5, float(assists) / league_average_assists)), 4)


def catcher_framing_rating(framing_runs: float) -> float:
    """A bounded rating from a real catcher's framing-runs-above-average (0 = league average).

    A catcher who reliably steals real strikes at the edges of the zone
    real-ly inflates the pitching staff's strikeout rate -- a well-
    documented Statcast-era effect (framing_runs typically ranges roughly
    -15 to +15 over a season).
    """
    return round(max(0.85, min(1.15, 1.0 + float(framing_runs) / 100.0)), 4)


def infield_range_rating(range_factor: float, league_average: float = _LEAGUE_AVERAGE_RANGE_FACTOR) -> float:
    """A bounded rating from a real infield's range factor per 9 innings (1.0 = league average)."""
    if league_average <= 0:
        return 1.0
    return round(max(0.7, min(1.3, float(range_factor) / league_average)), 4)


def composite_defense_rating(components: dict[str, Any]) -> dict[str, Any]:
    """Combine every real defensive signal into one composite 0-1+ team-defense rating.

    ``components``: ``{"balls_in_play_converted_pct", "outfield_assists",
    "catcher_framing_runs", "infield_range_factor"}``. Missing pieces fall
    back to a neutral (league-average) read rather than raising.
    """
    efficiency = defensive_efficiency_rating(components.get("balls_in_play_converted_pct", _LEAGUE_AVERAGE_DEFENSIVE_EFFICIENCY))
    arm = outfield_arm_rating(components.get("outfield_assists", _LEAGUE_AVERAGE_OUTFIELD_ASSISTS))
    framing = catcher_framing_rating(components.get("catcher_framing_runs", 0.0))
    range_rating = infield_range_rating(components.get("infield_range_factor", _LEAGUE_AVERAGE_RANGE_FACTOR))

    composite = 0.4 * efficiency + 0.2 * arm + 0.2 * framing + 0.2 * range_rating
    return {
        "composite_rating": round(max(0.5, min(1.4, composite)), 4),
        "defensive_efficiency_rating": efficiency,
        "outfield_arm_rating": arm,
        "catcher_framing_rating": framing,
        "infield_range_rating": range_rating,
    }
