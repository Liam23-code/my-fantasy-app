"""NBA parlay engine: NBA-specific correlation patterns, generic parlay math reused from NFL.

``betting.parlay_engine``'s leg construction, combined-odds, EV, and
risk-tier math (:func:`~betting.parlay_engine.make_leg`,
:func:`~betting.parlay_engine.parlay_decimal_odds`,
:func:`~betting.parlay_engine.correlation_adjusted_probability`) is
entirely sport-agnostic -- it operates on generic leg dicts and has no NFL
content in it, so it's reused directly rather than duplicated. Its
built-in *pattern detector* (:func:`~betting.parlay_engine.detect_correlations`)
does carry two genuinely sport-specific pieces: NFL stat-category sets
(QB/pass-catcher/rush stacks) and one fully generic pattern (a favored
moneyline side correlating with the game total) that applies to any sport
using the market names ``"moneyline"``/``"total"`` -- which NBA legs built
here do too, so that pattern still fires unmodified.

What NBA needs of its own: prop-market correlation. NBA's own market
overlap is structural rather than shared-role -- ``PRA`` is literally
``points + rebounds + assists``, so a leg on a player's points and a leg on
the same player's PRA are not independent the way two unrelated stats
would be. That pattern (and a same-team, same-direction scoring stack) is
what :func:`nba_detect_correlations` adds on top of the generic detector.
"""
from __future__ import annotations

from typing import Any

from betting.parlay_engine import (
    _parlay_ev,
    _risk_tier,
    correlation_adjusted_probability,
    detect_correlations as _nfl_style_correlations,
    make_leg,
    parlay_decimal_odds,
)

#: category -> the other categories it structurally overlaps with (PRA
#: contains all three; points/rebounds/assists overlap with PRA but not
#: with each other -- a points leg and a rebounds leg on the same player
#: are not driven by the same underlying box-score number).
_OVERLAPPING_CATEGORIES: dict[str, frozenset[str]] = {
    "points": frozenset({"PRA"}),
    "rebounds": frozenset({"PRA"}),
    "assists": frozenset({"PRA"}),
    "PRA": frozenset({"points", "rebounds", "assists"}),
}

_OVER_LIKE_SIDES = frozenset({"over", "home", "away"})

#: Same fixed, disclosed, capped adjustment as the NFL engine -- see
#: betting.parlay_engine's module docstring for why this is a modest
#: constant and not a fitted coefficient.
CORRELATION_ADJUSTMENT = 0.12


def _same_direction(leg_a: dict[str, Any], leg_b: dict[str, Any]) -> bool:
    side_a, side_b = leg_a.get("side"), leg_b.get("side")
    return side_a is not None and side_a == side_b and side_a in _OVER_LIKE_SIDES


def _nba_pair_correlation(leg_a: dict[str, Any], leg_b: dict[str, Any]) -> dict[str, Any] | None:
    market_a, market_b = leg_a.get("market"), leg_b.get("market")
    same_player = leg_a.get("player_id") and leg_a.get("player_id") == leg_b.get("player_id")
    same_team = leg_a.get("team") and leg_a.get("team") == leg_b.get("team")

    if same_player and _same_direction(leg_a, leg_b):
        if market_b in _OVERLAPPING_CATEGORIES.get(market_a, frozenset()):
            return {
                "kind": "overlapping_stat_categories",
                "direction": "positive",
                "note": f"same player, {market_a} is structurally part of {market_b} (or vice versa)",
            }

    if same_team and not same_player and market_a == "points" and market_b == "points" and _same_direction(leg_a, leg_b):
        return {
            "kind": "teammate_scoring_stack",
            "direction": "positive",
            "note": "same team, same game pace/script drives both players' scoring",
        }

    return None


def nba_detect_correlations(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All correlated leg pairs in an NBA parlay: NFL's generic moneyline/total pattern plus NBA's own prop patterns."""
    from itertools import combinations

    findings = list(_nfl_style_correlations(legs))
    covered = {finding["legs"] for finding in findings}
    for i, j in combinations(range(len(legs)), 2):
        if (i, j) in covered:
            continue
        finding = _nba_pair_correlation(legs[i], legs[j])
        if finding:
            findings.append({"legs": (i, j), **finding})
    return findings


def evaluate_parlay(legs: list[dict[str, Any]], *, stake: float = 100.0) -> dict[str, Any]:
    """Full evaluation of one NBA parlay: correlation-adjusted hit probability, payout, EV, confidence, risk tier.

    Same output shape as ``betting.parlay_engine.evaluate_parlay`` --
    reuses its generic building blocks, swapping in
    :func:`nba_detect_correlations` for the pattern-detection step.
    """
    if len(legs) < 2:
        raise ValueError("a parlay needs at least 2 legs")

    correlations = nba_detect_correlations(legs)
    probability_result = correlation_adjusted_probability(legs, correlations=correlations)
    decimal_odds = parlay_decimal_odds(legs)
    payout_if_win = round(stake * (decimal_odds - 1.0), 2)
    naive_ev = round(_parlay_ev(probability_result["naive_probability"], decimal_odds, stake), 2)
    adjusted_ev = round(_parlay_ev(probability_result["adjusted_probability"], decimal_odds, stake), 2)

    average_confidence = sum(leg.get("confidence", 0.7) for leg in legs) / len(legs)
    confidence = round(max(0.0, min(1.0, average_confidence * (1.0 - 0.04 * (len(legs) - 2)))), 4)

    return {
        "legs": [leg["description"] for leg in legs],
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


def rank_parlays(candidate_leg_sets: list[list[dict[str, Any]]], *, stake: float = 100.0) -> list[dict[str, Any]]:
    """Evaluate multiple candidate NBA parlays and rank by adjusted EV, then confidence."""
    evaluated = [evaluate_parlay(legs, stake=stake) for legs in candidate_leg_sets]
    evaluated.sort(key=lambda parlay: (-parlay["adjusted_ev"], -parlay["confidence"]))
    return evaluated


__all__ = ["make_leg", "nba_detect_correlations", "evaluate_parlay", "rank_parlays", "CORRELATION_ADJUSTMENT"]
