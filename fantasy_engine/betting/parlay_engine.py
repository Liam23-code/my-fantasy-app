"""Parlay engine: leg construction, correlation-aware hit probability, EV ranking.

A parlay's true joint probability is not simply the product of its legs'
individual probabilities when the legs are correlated -- the same game
script drives a QB's passing yards and his WR's receiving yards, for
example. This module detects known correlation *patterns* (it does not fit
correlation coefficients from data we don't have offline) and applies a
modest, capped, always-disclosed adjustment rather than presenting a false
precision the underlying data can't support. Every leg's own probability
must already come from :mod:`betting.prop_model` or
:mod:`betting.moneyline_model`; this module only combines legs, it never
invents one.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .odds_math import american_to_decimal

_QB_MARKETS = frozenset({"passing_yards", "passing_tds"})
_PASS_CATCHER_MARKETS = frozenset({"receiving_yards", "receiving_tds", "receptions"})
_RUSH_VOLUME_MARKETS = frozenset({"rushing_yards"})
_RUSH_SCORE_MARKETS = frozenset({"rushing_tds"})
_OVER_LIKE_SIDES = frozenset({"over", "home", "away"})  # "favored to win" sides, for same-direction checks

#: Modest, fixed, capped relative adjustment applied per detected
#: correlated pair -- not a fitted coefficient (we have no offline source
#: of real joint-outcome data to calibrate one against). Kept small and
#: symmetric on purpose: the point is to flag and roughly account for
#: correlation, not to claim precision the data can't support.
CORRELATION_ADJUSTMENT = 0.12


def make_leg(
    *,
    description: str,
    model_probability: float,
    price: float,
    player_id: str | None = None,
    team: str | None = None,
    game_id: str | None = None,
    market: str | None = None,
    side: str | None = None,
    confidence: float = 0.7,
) -> dict[str, Any]:
    """One parlay leg: a single prop or game-market side, already evaluated by another model."""
    return {
        "description": description,
        "model_probability": float(model_probability),
        "price": float(price),
        "player_id": player_id,
        "team": team,
        "game_id": game_id,
        "market": market,
        "side": side,
        "confidence": float(confidence),
    }


def _same_direction(leg_a: dict[str, Any], leg_b: dict[str, Any]) -> bool:
    side_a, side_b = leg_a.get("side"), leg_b.get("side")
    return side_a is not None and side_a == side_b and side_a in _OVER_LIKE_SIDES


def _pair_correlation(leg_a: dict[str, Any], leg_b: dict[str, Any]) -> dict[str, Any] | None:
    """One correlation finding for a leg pair, or ``None`` if no known pattern matches."""
    market_a, market_b = leg_a.get("market"), leg_b.get("market")

    same_team = leg_a.get("team") and leg_a.get("team") == leg_b.get("team")
    same_player = leg_a.get("player_id") and leg_a.get("player_id") == leg_b.get("player_id")
    same_game = leg_a.get("game_id") and leg_a.get("game_id") == leg_b.get("game_id")

    if same_team and _same_direction(leg_a, leg_b):
        if (market_a in _QB_MARKETS and market_b in _PASS_CATCHER_MARKETS) or (
            market_b in _QB_MARKETS and market_a in _PASS_CATCHER_MARKETS
        ):
            return {
                "kind": "qb_pass_catcher_stack",
                "direction": "positive",
                "note": "same team, same passing-game script drives both legs",
            }

    if same_player and _same_direction(leg_a, leg_b):
        if (market_a in _RUSH_VOLUME_MARKETS and market_b in _RUSH_SCORE_MARKETS) or (
            market_b in _RUSH_VOLUME_MARKETS and market_a in _RUSH_SCORE_MARKETS
        ):
            return {
                "kind": "rb_volume_and_touchdown",
                "direction": "positive",
                "note": "same player, shared rushing volume drives both legs",
            }

    if same_game and market_a == "moneyline" and market_b == "total":
        favored = leg_a if leg_a.get("side") in {"home", "away"} else None
        if favored is not None and leg_b.get("side") == "over":
            return {"kind": "favorite_and_game_total_over", "direction": "positive", "note": "a favored team correlates with a faster-paced, higher-scoring game"}
        if favored is not None and leg_b.get("side") == "under":
            return {"kind": "favorite_and_game_total_under", "direction": "positive", "note": "a dominant favorite (defensive control) correlates with a lower-scoring game"}
    if same_game and market_b == "moneyline" and market_a == "total":
        return _pair_correlation(leg_b, leg_a)

    return None


def detect_correlations(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All correlated leg pairs found in a parlay, with their leg indices."""
    findings = []
    for i, j in combinations(range(len(legs)), 2):
        finding = _pair_correlation(legs[i], legs[j])
        if finding:
            findings.append({"legs": (i, j), **finding})
    return findings


def naive_joint_probability(legs: list[dict[str, Any]]) -> float:
    """Product of independent leg probabilities -- the textbook (uncorrelated) parlay probability."""
    probability = 1.0
    for leg in legs:
        probability *= leg["model_probability"]
    return probability


def correlation_adjusted_probability(legs: list[dict[str, Any]], *, correlations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Naive joint probability, nudged for detected correlation patterns.

    Every detected pair applies :data:`CORRELATION_ADJUSTMENT` as a relative
    bump (positive correlation) or discount (negative correlation) to the
    naive product. The result is capped at the smallest individual leg's own
    probability -- a joint probability can never exceed any one leg's
    probability, regardless of correlation strength.
    """
    naive = naive_joint_probability(legs)
    correlations = correlations if correlations is not None else detect_correlations(legs)
    adjusted = naive
    for finding in correlations:
        factor = 1.0 + CORRELATION_ADJUSTMENT if finding["direction"] == "positive" else 1.0 - CORRELATION_ADJUSTMENT
        adjusted *= factor
    ceiling = min((leg["model_probability"] for leg in legs), default=1.0)
    adjusted = max(0.0, min(adjusted, ceiling))
    return {"naive_probability": round(naive, 6), "adjusted_probability": round(adjusted, 6), "correlations_detected": correlations}


def parlay_decimal_odds(legs: list[dict[str, Any]]) -> float:
    """Combined decimal odds for a parlay: the product of each leg's own decimal odds."""
    combined = 1.0
    for leg in legs:
        combined *= american_to_decimal(leg["price"])
    return combined


def _parlay_ev(probability: float, decimal_odds: float, stake: float) -> float:
    payout_if_win = stake * (decimal_odds - 1.0)
    return probability * payout_if_win - (1.0 - probability) * stake


def _risk_tier(num_legs: int, hit_probability: float) -> str:
    if num_legs <= 2 and hit_probability >= 0.35:
        return "low"
    if num_legs <= 4 and hit_probability >= 0.15:
        return "medium"
    return "high"


def evaluate_parlay(legs: list[dict[str, Any]], *, stake: float = 100.0) -> dict[str, Any]:
    """Full evaluation of one parlay: correlation-adjusted hit probability, payout, EV, confidence, risk tier."""
    if len(legs) < 2:
        raise ValueError("a parlay needs at least 2 legs")

    probability_result = correlation_adjusted_probability(legs)
    decimal_odds = parlay_decimal_odds(legs)
    payout_if_win = round(stake * (decimal_odds - 1.0), 2)
    naive_ev = round(_parlay_ev(probability_result["naive_probability"], decimal_odds, stake), 2)
    adjusted_ev = round(_parlay_ev(probability_result["adjusted_probability"], decimal_odds, stake), 2)

    # More legs compounds uncertainty even beyond each leg's own confidence --
    # a mild per-leg penalty on top of the average, floored at zero.
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
    """Evaluate multiple candidate parlays and rank by adjusted EV, then confidence."""
    evaluated = [evaluate_parlay(legs, stake=stake) for legs in candidate_leg_sets]
    evaluated.sort(key=lambda parlay: (-parlay["adjusted_ev"], -parlay["confidence"]))
    return evaluated
