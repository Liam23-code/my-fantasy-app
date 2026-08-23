"""Cross-sport parlay support: NFL and NBA legs combined in one bet.

Both sports' parlay engines already share the same leg shape and the same
generic combinatorics (:mod:`betting.parlay_engine`'s
``parlay_decimal_odds``, ``correlation_adjusted_probability``,
``_parlay_ev``, ``_risk_tier`` -- reused directly here too, not
duplicated). The only genuinely new piece a cross-sport parlay needs is
routing: each sport's own correlation-pattern detector
(:func:`betting.parlay_engine.detect_correlations`,
:func:`modules.nba_parlay_engine.nba_detect_correlations`) must only ever
run against leg pairs from *its own* sport -- an NFL quarterback and an
NBA center are not correlated by any pattern either detector models, and
running NFL's detector against an NBA leg (or vice versa) would either
find nothing (harmless) or, worse, misfire on a coincidental field-name
match (e.g. NBA's "points" market happening to share a string with some
future NFL market). This module partitions legs by sport before handing
each partition to its own detector, so that ambiguity never arises.
"""
from __future__ import annotations

from typing import Any

from betting.parlay_engine import (
    _parlay_ev,
    _risk_tier,
    correlation_adjusted_probability,
    detect_correlations as nfl_detect_correlations,
    make_leg as nfl_make_leg,
    parlay_decimal_odds,
)

from modules.nba_parlay_engine import make_leg as nba_make_leg, nba_detect_correlations

_VALID_SPORTS = frozenset({"NFL", "NBA"})


def make_unified_leg(sport: str, **kwargs: Any) -> dict[str, Any]:
    """One parlay leg tagged with its sport -- NFL and NBA legs share the same underlying shape.

    ``sport`` must be ``"NFL"`` or ``"NBA"``; every other keyword is
    forwarded to that sport's own ``make_leg`` (identical fields either
    way -- see ``betting.parlay_engine.make_leg`` /
    ``modules.nba_parlay_engine.make_leg``).
    """
    if sport not in _VALID_SPORTS:
        raise ValueError(f"sport must be one of {sorted(_VALID_SPORTS)}, got {sport!r}")
    leg = (nfl_make_leg if sport == "NFL" else nba_make_leg)(**kwargs)
    return {**leg, "sport": sport}


def detect_cross_sport_correlations(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Correlation findings across a mixed NFL+NBA leg list.

    Each sport's own pattern detector only ever runs against pairs drawn
    from that same sport's legs. A cross-sport pair (one NFL leg, one NBA
    leg) is never checked by either detector and so never produces a
    finding -- correct, since the two events genuinely are independent;
    such a pair simply combines via the naive (independent) probability
    product downstream, with no positive or negative adjustment.
    """
    nfl_indices = [i for i, leg in enumerate(legs) if leg.get("sport") == "NFL"]
    nba_indices = [i for i, leg in enumerate(legs) if leg.get("sport") == "NBA"]

    findings: list[dict[str, Any]] = []
    if len(nfl_indices) >= 2:
        nfl_legs = [legs[i] for i in nfl_indices]
        for finding in nfl_detect_correlations(nfl_legs):
            local_a, local_b = finding["legs"]
            findings.append({**finding, "legs": (nfl_indices[local_a], nfl_indices[local_b])})
    if len(nba_indices) >= 2:
        nba_legs = [legs[i] for i in nba_indices]
        for finding in nba_detect_correlations(nba_legs):
            local_a, local_b = finding["legs"]
            findings.append({**finding, "legs": (nba_indices[local_a], nba_indices[local_b])})
    return findings


def evaluate_cross_sport_parlay(legs: list[dict[str, Any]], *, stake: float = 100.0) -> dict[str, Any]:
    """Full evaluation of a parlay mixing NFL and NBA legs.

    Same output shape as either sport's own ``evaluate_parlay`` (with an
    added ``"sports"`` field listing which sports are represented), built
    from the same shared, generic building blocks -- no separate cross-sport
    EV/risk math exists to drift out of sync with either sport's own engine.
    """
    if len(legs) < 2:
        raise ValueError("a parlay needs at least 2 legs")
    unknown = [leg.get("sport") for leg in legs if leg.get("sport") not in _VALID_SPORTS]
    if unknown:
        raise ValueError("every leg must be tagged with sport='NFL' or sport='NBA' (see make_unified_leg)")

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
