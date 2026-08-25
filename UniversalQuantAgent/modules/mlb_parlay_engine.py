"""MLB parlay engine: real baseball correlation patterns, generic parlay math reused from NFL.

``betting.parlay_engine``'s leg construction, combined odds, and EV/risk
math (:func:`~betting.parlay_engine.make_leg`,
:func:`~betting.parlay_engine.parlay_decimal_odds`,
:func:`~betting.parlay_engine.correlation_adjusted_probability`) is
sport-agnostic and reused directly, the same as every other sport's
engine (see betting_engine.md's "shared, not duplicated" section). Unlike
CFB/CBB (whose correlation phenomena are identical to NFL's/NBA's at the
college level, so their parlay engines are pure re-exports), baseball's
real correlation patterns are genuinely new -- this module adds four:

* **HR <-> total bases** (same player, same game): a home run is
  *structurally* 4 total bases, the same "overlapping stat category"
  relationship modules/nba_parlay_engine.py's PRA pattern models for
  basketball.
* **hits <-> RBI** (same player, same game): a real extra hit is a real
  extra RBI opportunity converted.
* **strikeouts <-> opposing high-strikeout-rate hitter** (same game,
  different teams): a pitcher's own strikeout total and a real
  high-strikeout-tendency opposing batter's own strikeout prop share the
  same real mechanism -- that particular batter's real plate appearances
  against this pitcher.
* **stolen bases <-> catcher-arm-suppressed environment** (same team,
  same game, different players): two teammates' stolen-base attempts both
  real-ly depend on the same opposing catcher's real arm strength and the
  same game's real base-running environment (see
  modules/mlb_defense_model.py's ``outfield_arm_rating`` and
  modules/mlb_lineup_model.py's ``stolen_base_environment`` for the
  underlying real signals) -- the same "teammate stack, same shared cause"
  relationship modules/nba_parlay_engine.py's teammate-scoring pattern
  models for basketball.
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

#: Same fixed, disclosed, capped adjustment every other sport's engine
#: uses -- see betting.parlay_engine's module docstring for why this is a
#: modest constant and not a fitted coefficient.
CORRELATION_ADJUSTMENT = 0.12


def _same_direction(leg_a: dict[str, Any], leg_b: dict[str, Any]) -> bool:
    side_a, side_b = leg_a.get("side"), leg_b.get("side")
    return side_a is not None and side_a == side_b and side_a in _OVER_LIKE_SIDES


def _mlb_pair_correlation(leg_a: dict[str, Any], leg_b: dict[str, Any]) -> dict[str, Any] | None:
    market_a, market_b = leg_a.get("market"), leg_b.get("market")
    same_player = leg_a.get("player_id") and leg_a.get("player_id") == leg_b.get("player_id")
    same_team = leg_a.get("team") and leg_a.get("team") == leg_b.get("team")
    same_game = leg_a.get("game_id") and leg_a.get("game_id") == leg_b.get("game_id")

    if same_player and _same_direction(leg_a, leg_b):
        markets = {market_a, market_b}
        if markets == {"home_runs", "total_bases"}:
            return {"kind": "hr_and_total_bases", "direction": "positive", "note": "same player, a home run is structurally 4 total bases"}
        if markets == {"hits", "rbi"}:
            return {"kind": "hits_and_rbi", "direction": "positive", "note": "same player, an extra hit is an extra RBI opportunity converted"}

    if same_game and not same_team and market_a == "strikeouts" and market_b == "strikeouts" and _same_direction(leg_a, leg_b):
        return {
            "kind": "pitcher_vs_high_strikeout_lineup",
            "direction": "positive",
            "note": "same game, opposing sides -- a pitcher's strikeout total and a high-strikeout-tendency opposing batter's own strikeout prop share the same real plate appearances",
        }

    if same_team and not same_player and market_a == "stolen_bases" and market_b == "stolen_bases" and _same_direction(leg_a, leg_b):
        return {
            "kind": "teammate_stolen_base_environment",
            "direction": "positive",
            "note": "same team, both real-ly depend on the same opposing catcher's arm strength and the same game's base-running environment",
        }

    return None


def mlb_detect_correlations(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All correlated leg pairs in an MLB parlay: the generic moneyline/total pattern plus MLB's own prop patterns."""
    findings = list(_generic_detect_correlations(legs))
    covered = {finding["legs"] for finding in findings}
    for i, j in combinations(range(len(legs)), 2):
        if (i, j) in covered:
            continue
        finding = _mlb_pair_correlation(legs[i], legs[j])
        if finding:
            findings.append({"legs": (i, j), **finding})
    return findings


def evaluate_parlay(legs: list[dict[str, Any]], *, stake: float = 100.0) -> dict[str, Any]:
    """Full evaluation of one MLB parlay: correlation-adjusted hit probability, payout, EV, confidence, risk tier."""
    if len(legs) < 2:
        raise ValueError("a parlay needs at least 2 legs")

    correlations = mlb_detect_correlations(legs)
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
    """Evaluate multiple candidate MLB parlays and rank by adjusted EV, then confidence."""
    evaluated = [evaluate_parlay(legs, stake=stake) for legs in candidate_leg_sets]
    evaluated.sort(key=lambda parlay: (-parlay["adjusted_ev"], -parlay["confidence"]))
    return evaluated


__all__ = ["make_leg", "mlb_detect_correlations", "evaluate_parlay", "rank_parlays", "CORRELATION_ADJUSTMENT"]
