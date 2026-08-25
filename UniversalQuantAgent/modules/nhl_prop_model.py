"""NHL player prop model: real per-game rates -> over/under probability, edge, EV, risk.

The contract-required evaluator -- like CFB/CBB/MLB, NHL has no
pre-existing rich per-player projection system to wrap, so this prices
whatever real per-game rate a prop row's own ``"line"`` already carries
directly (see modules/mlb_prop_model.py's identical rationale).

Two of the four modeled categories (goals, assists) are low-mean,
right-skewed counts a Gaussian badly fits -- the same case
fantasy_engine/betting/prop_model.py's own ``_COUNT_MARKETS`` documents
for NFL touchdowns/receptions -- and are modeled as Poisson. Shots and
goalie saves have a meaningfully higher real per-game mean (shots ~2-3,
saves ~25-30) where the normal approximation is a reasonable fit, so they
keep the same Gaussian-with-disclosed-CV treatment CFB/CBB/MLB's simpler
categories use.
"""
from __future__ import annotations

import math
from typing import Any

from betting.odds_math import edge_vs_fair, expected_value, remove_vig_two_way
from betting.parallel_utils import parallel_ev_map
from betting.prop_model import _risk_tier

#: goals/assists are low-mean, discrete, right-skewed -- Poisson, not
#: Gaussian (see module docstring). shots/saves get a Gaussian with a
#: disclosed coefficient of variation instead.
_COUNT_CATEGORIES = frozenset({"goals", "assists"})
#: Base coefficient of variation for the Gaussian categories -- shots and
#: saves both vary game-to-game with matchup/pace, wider than a typical
#: NFL yardage market's CV range but narrower than a pure count stat.
_BASE_CV = 0.45
_MIN_STDEV = 0.5


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam), by direct PMF summation."""
    if lam <= 0.0:
        return 1.0
    if k < 0:
        return 0.0
    total = math.exp(-lam)
    term = total
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(1.0, total)


def over_under_probability(line: float, mean: float, *, category: str) -> dict[str, Any]:
    """Probability a real NHL stat lands over/under ``line``, Poisson for counts, Gaussian otherwise."""
    line = float(line)
    if category in _COUNT_CATEGORIES:
        probability_under = round(_poisson_cdf(math.floor(line), mean), 4)
        cv = round(math.sqrt(mean) / mean, 4) if mean > 0 else None
        family = "poisson"
    else:
        stdev = max(mean * _BASE_CV, _MIN_STDEV)
        z = (line - mean) / stdev
        probability_under = round(_normal_cdf(z), 4)
        cv = round(stdev / mean, 4) if mean > 0 else None
        family = "gaussian"
    probability_over = round(1.0 - probability_under, 4)
    return {"line": line, "mean": round(mean, 4), "cv": cv, "family": family, "probability_over": probability_over, "probability_under": probability_under}


def evaluate_prop(prop_odds: dict[str, Any]) -> dict[str, Any]:
    """Full evaluation of one real NHL prop line: probability, market-fair probability, edge, EV, risk.

    ``prop_odds`` is one entry from :func:`modules.nhl_props_loader.unified_props`
    -- carries its own real per-game-rate ``"line"`` as the model's mean.
    """
    category = prop_odds.get("category")
    mean = float(prop_odds["line"])
    over_price = float(prop_odds.get("over_price") or -110.0)
    under_price = float(prop_odds.get("under_price") or -110.0)
    distribution = over_under_probability(mean, mean, category=category)

    fair_over, fair_under = remove_vig_two_way(over_price, under_price)
    edge_over = edge_vs_fair(distribution["probability_over"], over_price, under_price, side="a")
    edge_under = edge_vs_fair(distribution["probability_under"], over_price, under_price, side="b")
    ev_over = expected_value(distribution["probability_over"], over_price)
    ev_under = expected_value(distribution["probability_under"], under_price)
    recommended_side = "over" if edge_over >= edge_under else "under"

    return {
        "player_name": prop_odds.get("player_name"),
        "team": prop_odds.get("team"),
        "category": category,
        "line": mean,
        "over_price": over_price,
        "under_price": under_price,
        "model_probability_over": distribution["probability_over"],
        "model_probability_under": distribution["probability_under"],
        "market_fair_probability_over": round(fair_over, 4),
        "market_fair_probability_under": round(fair_under, 4),
        "edge_over": round(edge_over, 4),
        "edge_under": round(edge_under, 4),
        "ev_over": round(ev_over, 2),
        "ev_under": round(ev_under, 2),
        "recommended_side": recommended_side,
        "recommended_edge": round(edge_over if recommended_side == "over" else edge_under, 4),
        "recommended_ev": round(ev_over if recommended_side == "over" else ev_under, 2),
        "risk_tier": _risk_tier(distribution["cv"]),
        "basis": prop_odds.get("basis"),
        "odds_source": prop_odds.get("sportsbook"),
    }


def evaluate_props(props: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate every loaded NHL prop line. Each row's math is pure and independent -- see betting.parallel_utils."""
    rows = [row for row in parallel_ev_map(evaluate_prop, props) if row is not None]
    rows.sort(key=lambda row: -abs(row["recommended_edge"]))
    return rows
