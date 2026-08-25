"""NHL parlay engine: real hockey correlation patterns, generic parlay math reused from NFL.

``betting.parlay_engine``'s leg construction, combined odds, and EV/risk
math is sport-agnostic and reused directly (see betting_engine.md's
"shared, not duplicated" section). NHL is a lightweight secondary sport in
this engine (see nhl_pipeline.md), so it adds two real, well-established
correlation patterns rather than MLB's four:

* **goals <-> assists** (same player, same game): a real multi-point game
  structurally links a player's own goal and assist totals -- the same
  "overlapping stat category" relationship modules/nba_parlay_engine.py's
  PRA pattern and modules/mlb_parlay_engine.py's HR/total-bases pattern
  both model for their sports.
* **teammate goal stack** (same team, same game, different players): two
  teammates' goal totals both real-ly depend on the same game's pace and
  power-play opportunities -- the same "teammate stack, same shared
  cause" relationship modules/nba_parlay_engine.py's teammate-scoring
  pattern models for basketball.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from betting.parlay_engine import (
    _parlay_ev,
    _risk_tier,
    correlation_adjusted_probability,
    detect_correlations as _generic_detect_correlations,
    make_leg,
    parlay_decimal_odds,
)

_OVER_LIKE_SIDES = frozenset({"over", "home", "away"})
CORRELATION_ADJUSTMENT = 0.12


def _same_direction(leg_a: dict[str, Any], leg_b: dict[str, Any]) -> bool:
    side_a, side_b = leg_a.get("side"), leg_b.get("side")
    return side_a is not None and side_a == side_b and side_a in _OVER_LIKE_SIDES


def _nhl_pair_correlation(leg_a: dict[str, Any], leg_b: dict[str, Any]) -> dict[str, Any] | None:
    market_a, market_b = leg_a.get("market"), leg_b.get("market")
    same_player = leg_a.get("player_id") and leg_a.get("player_id") == leg_b.get("player_id")
    same_team = leg_a.get("team") and leg_a.get("team") == leg_b.get("team")

    if same_player and _same_direction(leg_a, leg_b) and {market_a, market_b} == {"goals", "assists"}:
        return {"kind": "goals_and_assists", "direction": "positive", "note": "same player, a multi-point game structurally links goal and assist totals"}

    if same_team and not same_player and market_a == "goals" and market_b == "goals" and _same_direction(leg_a, leg_b):
        return {"kind": "teammate_goal_stack", "direction": "positive", "note": "same team, same game pace and power-play opportunities drive both players' goal totals"}

    return None


def nhl_detect_correlations(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All correlated leg pairs in an NHL parlay: the generic moneyline/total pattern plus NHL's own prop patterns."""
    findings = list(_generic_detect_correlations(legs))
    covered = {finding["legs"] for finding in findings}
    for i, j in combinations(range(len(legs)), 2):
        if (i, j) in covered:
            continue
        finding = _nhl_pair_correlation(legs[i], legs[j])
        if finding:
            findings.append({"legs": (i, j), **finding})
    return findings


def evaluate_parlay(legs: list[dict[str, Any]], *, stake: float = 100.0) -> dict[str, Any]:
    """Full evaluation of one NHL parlay: correlation-adjusted hit probability, payout, EV, confidence, risk tier."""
    if len(legs) < 2:
        raise ValueError("a parlay needs at least 2 legs")

    correlations = nhl_detect_correlations(legs)
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
    """Evaluate multiple candidate NHL parlays and rank by adjusted EV, then confidence."""
    evaluated = [evaluate_parlay(legs, stake=stake) for legs in candidate_leg_sets]
    evaluated.sort(key=lambda parlay: (-parlay["adjusted_ev"], -parlay["confidence"]))
    return evaluated


__all__ = ["make_leg", "nhl_detect_correlations", "evaluate_parlay", "rank_parlays", "CORRELATION_ADJUSTMENT"]
