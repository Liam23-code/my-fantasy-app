"""Cross-sport parlay support: NFL, NBA, CFB, CBB, MLB, and NHL legs combined in one bet.

Every sport's parlay engine already shares the same leg shape and the same
generic combinatorics (:mod:`betting.parlay_engine`'s
``parlay_decimal_odds``, ``correlation_adjusted_probability``,
``_parlay_ev``, ``_risk_tier`` -- reused directly here too, not
duplicated). CFB's and CBB's own correlation detectors
(``modules.cfb_parlay_engine.detect_correlations``,
``modules.cbb_parlay_engine.detect_correlations``) are themselves
re-exports of NFL's and NBA's respectively (see those modules'
docstrings) -- football and basketball correlation patterns don't change
between the pro and college game, so this module's dispatch table maps
CFB/CBB to the same underlying detector functions NFL/NBA already use,
not a fourth and fifth copy. MLB and NHL each needed genuinely new
correlation-pattern code instead (baseball and hockey correlations are
real, different phenomena -- see modules.mlb_parlay_engine and
modules.nhl_parlay_engine's own docstrings).

The only genuinely new piece a cross-sport parlay needs is routing: each
sport's own correlation-pattern detector must only ever run against leg
pairs from *its own* sport -- an NFL quarterback and an NBA center (or an
MLB and an NHL player) are not correlated by any pattern any detector
models. This module partitions legs by sport before handing each
partition to its own detector, so that ambiguity never arises.
"""
from __future__ import annotations

from typing import Any, Callable

from betting.parlay_engine import (
    _parlay_ev,
    _risk_tier,
    correlation_adjusted_probability,
    make_leg as nfl_make_leg,
    detect_correlations as nfl_detect_correlations,
    parlay_decimal_odds,
)

from modules.cbb_parlay_engine import make_leg as cbb_make_leg, detect_correlations as cbb_detect_correlations
from modules.cfb_parlay_engine import make_leg as cfb_make_leg, detect_correlations as cfb_detect_correlations
from modules.mlb_parlay_engine import make_leg as mlb_make_leg, mlb_detect_correlations
from modules.nba_parlay_engine import make_leg as nba_make_leg, nba_detect_correlations
from modules.nhl_parlay_engine import make_leg as nhl_make_leg, nhl_detect_correlations

_MAKE_LEG_BY_SPORT: dict[str, Callable[..., dict[str, Any]]] = {
    "NFL": nfl_make_leg,
    "NBA": nba_make_leg,
    "CFB": cfb_make_leg,
    "CBB": cbb_make_leg,
    "MLB": mlb_make_leg,
    "NHL": nhl_make_leg,
}
_DETECT_CORRELATIONS_BY_SPORT: dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = {
    "NFL": nfl_detect_correlations,
    "NBA": nba_detect_correlations,
    "CFB": cfb_detect_correlations,
    "CBB": cbb_detect_correlations,
    "MLB": mlb_detect_correlations,
    "NHL": nhl_detect_correlations,
}
_VALID_SPORTS = frozenset(_MAKE_LEG_BY_SPORT)


def make_unified_leg(sport: str, **kwargs: Any) -> dict[str, Any]:
    """One parlay leg tagged with its sport -- all six sports' legs share the same underlying shape.

    ``sport`` must be one of ``"NFL"``, ``"NBA"``, ``"CFB"``, ``"CBB"``,
    ``"MLB"``, ``"NHL"``; every other keyword is forwarded to that
    sport's own ``make_leg`` (identical fields in every case).
    """
    if sport not in _VALID_SPORTS:
        raise ValueError(f"sport must be one of {sorted(_VALID_SPORTS)}, got {sport!r}")
    leg = _MAKE_LEG_BY_SPORT[sport](**kwargs)
    return {**leg, "sport": sport}


def detect_cross_sport_correlations(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Correlation findings across a mixed multi-sport leg list.

    Each sport's own pattern detector only ever runs against pairs drawn
    from that same sport's legs. A cross-sport pair is never checked by
    any detector and so never produces a finding -- correct, since the two
    events genuinely are independent; such a pair simply combines via the
    naive (independent) probability product downstream, with no positive
    or negative adjustment.
    """
    findings: list[dict[str, Any]] = []
    for sport, detector in _DETECT_CORRELATIONS_BY_SPORT.items():
        indices = [i for i, leg in enumerate(legs) if leg.get("sport") == sport]
        if len(indices) < 2:
            continue
        sport_legs = [legs[i] for i in indices]
        for finding in detector(sport_legs):
            local_a, local_b = finding["legs"]
            findings.append({**finding, "legs": (indices[local_a], indices[local_b])})
    return findings


def evaluate_cross_sport_parlay(legs: list[dict[str, Any]], *, stake: float = 100.0) -> dict[str, Any]:
    """Full evaluation of a parlay mixing legs from any of the six sports.

    Same output shape as any single sport's own ``evaluate_parlay`` (with
    an added ``"sports"`` field listing which sports are represented),
    built from the same shared, generic building blocks -- no separate
    cross-sport EV/risk math exists to drift out of sync with any sport's
    own engine.
    """
    if len(legs) < 2:
        raise ValueError("a parlay needs at least 2 legs")
    unknown = [leg.get("sport") for leg in legs if leg.get("sport") not in _VALID_SPORTS]
    if unknown:
        raise ValueError(f"every leg must be tagged with a sport in {sorted(_VALID_SPORTS)} (see make_unified_leg)")

    correlations = detect_cross_sport_correlations(legs)
    probability_result = correlation_adjusted_probability(legs, correlations=correlations)
    decimal_odds = parlay_decimal_odds(legs)
    payout_if_win = round(stake * (decimal_odds - 1.0), 2)
    naive_ev = round(_parlay_ev(probability_result["naive_probability"], decimal_odds, stake), 2)
    adjusted_ev = round(_parlay_ev(probability_result["adjusted_probability"], decimal_odds, stake), 2)

    average_confidence = sum(leg.get("confidence", 0.7) for leg in legs) / len(legs)
    confidence = round(max(0.0, min(1.0, average_confidence * (1.0 - 0.04 * (len(legs) - 2)))), 4)

    return {
        "legs": [leg["description"] for leg in legs],
        "sports": sorted({leg["sport"] for leg in legs}),
        "num_legs": len(legs),
        "decimal_odds": round(decimal_odds, 4),
        "payout_per_100_stake": payout_if_win,
        "naive_hit_probability": probability_result["naive_probability"],
        "adjusted_hit_probability": probability_result["adjusted_probability"],
        "correlations_detected": probability_result["correlations_detected"],
        "naive_ev": naive_ev,
        "adjusted_ev": adjusted_ev,
        "confidence": confidence,
        "risk_tier": _risk_tier(len(legs), probability_result["adjusted_probability"]),
    }
