"""MLB player prop model: real per-game rates -> over/under probability, edge, EV, risk.

The contract-required evaluator (see modules/unified_betting_contract.py)
-- like CFB/CBB, MLB has no pre-existing rich per-player projection system
to wrap, so this prices whatever real per-game rate a prop row's own
``"line"`` already carries directly, the same self-contained pattern
modules/cbb_prop_model.py uses. The richer season/matchup/fusion layers
(modules/mlb_season_model.py, mlb_fusion_model.py, and the four matchup
modules) are an optional overlay that can inform what a user trusts enough
to put in that line -- they're not required to price a prop at all.

All 7 modeled categories (see modules/mlb_common.py's ``STAT_CATEGORIES``)
are low-mean, non-negative per-game counts (a typical hits line is ~1.0,
a home-run line is ~0.3) -- the same shape fantasy_engine/betting/prop_model.py's
own ``_COUNT_MARKETS`` (touchdowns, receptions) already models as Poisson
rather than Gaussian, and for the same reason: a Gaussian with
stdev = mean * cv badly understates a low-mean count stat's real spread.
Every MLB category here uses that same Poisson treatment; ``betting.prop_model``'s
own count-market math isn't reused directly only because it's private to
that module, but the approach and its risk-tier classification
(``betting.prop_model._risk_tier``, reused directly) are identical.
"""
from __future__ import annotations

import math
from typing import Any

from betting.odds_math import edge_vs_fair, expected_value, remove_vig_two_way
from betting.parallel_utils import parallel_ev_map
from betting.prop_model import _risk_tier


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam), by direct PMF summation (stable at MLB counting-stat magnitudes)."""
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


def over_under_probability(line: float, mean: float) -> dict[str, Any]:
    """Probability a real MLB stat lands over/under ``line``, from a Poisson distribution around the real mean."""
    line = float(line)
    probability_under = round(_poisson_cdf(math.floor(line), mean), 4)
    probability_over = round(1.0 - probability_under, 4)
    cv = round((math.sqrt(mean) / mean), 4) if mean > 0 else None
    return {"line": line, "mean": round(mean, 4), "cv": cv, "probability_over": probability_over, "probability_under": probability_under}


def evaluate_prop(prop_odds: dict[str, Any], *, matchup_multiplier: float = 1.0) -> dict[str, Any]:
    """Full evaluation of one real MLB prop line: probability, market-fair probability, edge, EV, risk.

    ``prop_odds`` is one entry from :func:`modules.mlb_props_loader.unified_props`
    -- carries its own real per-game-rate ``"line"`` as the model's mean.
    ``matchup_multiplier``, if known (e.g. from
    :func:`modules.mlb_fusion_model.fuse_projection`), adjusts that mean
    before pricing; the default 1.0 prices the line exactly as given, the
    same contract CFB/CBB's optional context parameters follow.
    """
    mean = float(prop_odds["line"]) * float(matchup_multiplier)
    over_price = float(prop_odds.get("over_price") or -110.0)
    under_price = float(prop_odds.get("under_price") or -110.0)
    distribution = over_under_probability(mean, mean)

    fair_over, fair_under = remove_vig_two_way(over_price, under_price)
    edge_over = edge_vs_fair(distribution["probability_over"], over_price, under_price, side="a")
    edge_under = edge_vs_fair(distribution["probability_under"], over_price, under_price, side="b")
    ev_over = expected_value(distribution["probability_over"], over_price)
    ev_under = expected_value(distribution["probability_under"], under_price)
    recommended_side = "over" if edge_over >= edge_under else "under"

    return {
        "player_name": prop_odds.get("player_name"),
        "team": prop_odds.get("team"),
        "category": prop_odds.get("category"),
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


def evaluate_props(props: list[dict[str, Any]], *, matchup_multipliers: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Evaluate every loaded MLB prop line. Each row's math is pure and independent -- see betting.parallel_utils."""
    matchup_multipliers = matchup_multipliers or {}

    def _evaluate(prop_odds: dict[str, Any]) -> dict[str, Any]:
        return evaluate_prop(prop_odds, matchup_multiplier=matchup_multipliers.get(prop_odds.get("player_name"), 1.0))

    rows = [row for row in parallel_ev_map(_evaluate, props) if row is not None]
    rows.sort(key=lambda row: -abs(row["recommended_edge"]))
    return rows
