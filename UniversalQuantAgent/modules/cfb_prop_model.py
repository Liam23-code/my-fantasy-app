"""CFB player prop model: real per-game rates -> over/under probability, edge, EV, risk.

CFB has no pre-existing rich per-player projection system the way NBA does
(fuse_projection/reliability were built for fantasy football years before
any of this betting work started) -- there is nothing analogous to wrap
for CFB, so this mirrors ``fantasy_engine/betting/prop_model.py``'s
from-scratch structure directly: a stat is modeled as approximately
Gaussian around a real per-game rate, with a disclosed coefficient-of-
variation standard in for volatility (real college football week-to-week
variance runs meaningfully higher than the NFL's -- larger real talent
gaps between programs mean both blowouts and upsets are more common; see
_CV below). ``betting.odds_math`` and ``betting.prop_model``'s risk-tier
classification are reused directly (both are fully sport-agnostic; see
betting_engine.md's "shared, not duplicated" section).
"""
from __future__ import annotations

import math
from typing import Any

from betting.odds_math import edge_vs_fair, expected_value, remove_vig_two_way
from betting.parallel_utils import parallel_ev_map
from betting.prop_model import _risk_tier

#: A single, disclosed coefficient of variation (stdev / mean) for every
#: CFB prop category -- not tuned per player (no per-player game-log
#: volatility source has been verified live yet; see cfb_pipeline.md).
#: Meaningfully wider than fantasy_engine/betting/prop_model.py's NFL
#: CV_LOW/CV_HIGH range (0.28-0.65), reflecting CFB's real higher
#: game-to-game variance.
_CFB_CV = 0.55

_MIN_STDEV = 1.0


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def over_under_probability(line: float, mean: float, *, cv: float = _CFB_CV) -> dict[str, Any]:
    """Probability a real CFB stat lands over/under ``line``, from a Gaussian around the real per-game mean."""
    stdev = max(mean * cv, _MIN_STDEV)
    z = (float(line) - mean) / stdev
    probability_under = round(_normal_cdf(z), 4)
    probability_over = round(1.0 - probability_under, 4)
    return {"line": float(line), "mean": mean, "stdev": round(stdev, 2), "probability_over": probability_over, "probability_under": probability_under}


def evaluate_prop(prop_odds: dict[str, Any]) -> dict[str, Any]:
    """Full evaluation of one real CFB prop line: probability, market-fair probability, edge, EV, risk.

    ``prop_odds`` is one entry from :func:`modules.cfb_props_loader.unified_props`
    -- carries its own real per-game-rate ``"line"`` (from
    modules.cfb_props_generator, or a user upload) as the model's mean,
    plus real ``"over_price"``/``"under_price"``.
    """
    mean = float(prop_odds["line"])
    over_price = float(prop_odds.get("over_price") or -110.0)
    under_price = float(prop_odds.get("under_price") or -110.0)
    distribution = over_under_probability(mean, mean)

    fair_over, fair_under = remove_vig_two_way(over_price, under_price)
    edge_over = edge_vs_fair(distribution["probability_over"], over_price, under_price, side="a")
    edge_under = edge_vs_fair(distribution["probability_under"], over_price, under_price, side="b")
    ev_over = expected_value(distribution["probability_over"], over_price)
    ev_under = expected_value(distribution["probability_under"], under_price)
    recommended_side = "over" if edge_over >= edge_under else "under"
    cv = distribution["stdev"] / mean if mean > 0 else None

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
        "risk_tier": _risk_tier(cv),
        "basis": prop_odds.get("basis"),
        "odds_source": prop_odds.get("sportsbook"),
    }


def evaluate_props(props: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate every loaded CFB prop line. Each row's math is pure and independent -- see betting.parallel_utils."""
    rows = [row for row in parallel_ev_map(evaluate_prop, props) if row is not None]
    rows.sort(key=lambda row: -abs(row["recommended_edge"]))
    return rows
