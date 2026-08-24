"""CBB player prop model: real per-game rates -> over/under probability, edge, EV, risk.

Like CFB (see modules/cfb_prop_model.py's docstring), CBB has no
pre-existing rich per-player projection system to wrap -- NBA's
fuse_projection/reliability infrastructure was built for fantasy
basketball years before any of this betting work started, and has no CBB
equivalent. This mirrors ``fantasy_engine/betting/prop_model.py``'s
from-scratch Gaussian structure, reusing ``betting.odds_math`` and
``betting.prop_model``'s risk-tier classification directly (both fully
sport-agnostic).

One real signal CFB's equivalent doesn't have: ESPN's real per-player
``minutes`` average (see modules/cbb_props_generator.py), which
:func:`minutes_volatility_multiplier` uses to widen the assumed
coefficient of variation for a low-minutes player -- a real, disclosed
proxy (less playing time really does mean a noisier per-game rate) rather
than a single flat constant for every player regardless of role.
"""
from __future__ import annotations

import math
from typing import Any

from betting.odds_math import edge_vs_fair, expected_value, remove_vig_two_way
from betting.parallel_utils import parallel_ev_map
from betting.prop_model import _risk_tier

#: Base coefficient of variation (stdev / mean) for a full-rotation CBB
#: player -- meaningfully wider than fantasy_engine/betting/prop_model.py's
#: NFL CV_LOW/CV_HIGH range (0.28-0.65) but narrower than CFB's flat 0.55
#: (modules/cfb_prop_model.py), since CBB has a real per-player minutes
#: signal to scale it with rather than one number for every player.
_BASE_CV = 0.35
#: Below this many real minutes/game, treat the rate as proportionally
#: noisier -- a bench player's per-game counting stat swings more, in real
#: relative terms, than a starter's.
_FULL_ROTATION_MINUTES = 24.0
_MIN_STDEV = 0.5


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def minutes_volatility_multiplier(minutes_per_game: float | None) -> float:
    """Real minutes-per-game -> a CV multiplier, >= 1.0 (never narrows the base CV, only widens it)."""
    if not minutes_per_game or minutes_per_game <= 0:
        return 1.5
    return max(1.0, min(2.0, _FULL_ROTATION_MINUTES / minutes_per_game))


def over_under_probability(line: float, mean: float, *, cv: float) -> dict[str, Any]:
    """Probability a real CBB stat lands over/under ``line``, from a Gaussian around the real per-game mean."""
    stdev = max(mean * cv, _MIN_STDEV)
    z = (float(line) - mean) / stdev
    probability_under = round(_normal_cdf(z), 4)
    probability_over = round(1.0 - probability_under, 4)
    return {"line": float(line), "mean": mean, "stdev": round(stdev, 2), "probability_over": probability_over, "probability_under": probability_under}


def evaluate_prop(prop_odds: dict[str, Any], *, minutes_per_game: float | None = None) -> dict[str, Any]:
    """Full evaluation of one real CBB prop line: probability, market-fair probability, edge, EV, risk.

    ``prop_odds`` is one entry from :func:`modules.cbb_props_loader.unified_props`
    -- carries its own real per-game-rate ``"line"`` as the model's mean.
    ``minutes_per_game``, if known, widens the assumed variance for a
    low-minutes player (see :func:`minutes_volatility_multiplier`).
    """
    mean = float(prop_odds["line"])
    over_price = float(prop_odds.get("over_price") or -110.0)
    under_price = float(prop_odds.get("under_price") or -110.0)
    cv = _BASE_CV * minutes_volatility_multiplier(minutes_per_game)
    distribution = over_under_probability(mean, mean, cv=cv)

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
        "risk_tier": _risk_tier(cv),
        "basis": prop_odds.get("basis"),
        "odds_source": prop_odds.get("sportsbook"),
    }


def evaluate_props(props: list[dict[str, Any]], *, minutes_by_player: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Evaluate every loaded CBB prop line. Each row's math is pure and independent -- see betting.parallel_utils."""
    minutes_by_player = minutes_by_player or {}

    def _evaluate(prop_odds: dict[str, Any]) -> dict[str, Any]:
        return evaluate_prop(prop_odds, minutes_per_game=minutes_by_player.get(prop_odds.get("player_name")))

    rows = [row for row in parallel_ev_map(_evaluate, props) if row is not None]
    rows.sort(key=lambda row: -abs(row["recommended_edge"]))
    return rows
