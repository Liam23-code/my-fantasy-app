"""Price-aware NBA prop evaluation: probability, edge, and dollar EV on top of the existing projection engine.

NBA's player-projection pipeline (``fuse_projection``, ``get_reliability_score``,
``modules.props.compare_props``) already produces a real per-player
projection and an uncertainty band for it -- richer, real signal than the
Gaussian/Poisson model NFL's simpler ``betting.prop_model`` builds from raw
season rates alone. This module does not replace or duplicate any of that:
it is a thin layer that takes ``compare_props``/``recommend_props``'s own
output and the matching real over/under prices (from
:mod:`modules.nba_props_loader`) and computes what a *priced* bet actually
implies -- win probability, the market's de-vigged fair probability, edge,
and dollar expected value -- using :mod:`betting.odds_math`'s sport-agnostic
American-odds math directly (no NBA-specific math needed; odds math has no
sport in it).

Every ``compare_props``/``recommend_props`` row already carries a
``confidence_low``/``confidence_high`` band. This module treats that band
as this app's own established ~90% interval convention (z = 1.645), the
same convention :mod:`modules.minutes_model` and
``modules.props._three_projection`` already use -- an approximation, not a
rigorously fitted Gaussian, and disclosed as such rather than presented as
more precise than it is.
"""
from __future__ import annotations

import math
from typing import Any

from betting.odds_math import edge_vs_fair, expected_value, remove_vig_two_way

from modules.sportsbook_parser import normalize_player_name

#: This app's own established confidence-band convention (see
#: modules/minutes_model.py, modules/props.py::_three_projection): treat
#: confidence_low/high as an approximate 90% interval.
_CONFIDENCE_Z = 1.645

#: Below this stdev, a line that lands past the confidence band produces a
#: probability so close to 0/1 that the resulting "edge" is more a
#: statement about the interval's width than a real signal -- floor it so
#: an unusually tight band can't manufacture a false-precision result.
_MIN_STDEV = 0.05


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _stdev_from_confidence_band(confidence_low: float, confidence_high: float) -> float:
    width = max(0.0, float(confidence_high) - float(confidence_low))
    return max(width / (2.0 * _CONFIDENCE_Z), _MIN_STDEV)


def price_prop_comparison(comparison_row: dict[str, Any], prop_odds: dict[str, Any]) -> dict[str, Any]:
    """Add probability, market-fair probability, edge, and EV to one ``compare_props`` row.

    ``comparison_row`` is one entry from :func:`modules.props.compare_props`
    / :func:`modules.recommendations.recommend_props`'s output (has
    ``"minutes_adjusted_projection"``, ``"confidence_low"``,
    ``"confidence_high"``, ``"sportsbook_line"``). ``prop_odds`` is the
    matching row from :func:`modules.nba_props_loader.unified_props` for
    the same player + category (has ``"over_price"``, ``"under_price"``).
    """
    projection = float(comparison_row["minutes_adjusted_projection"])
    stdev = _stdev_from_confidence_band(comparison_row["confidence_low"], comparison_row["confidence_high"])
    line = float(comparison_row["sportsbook_line"])
    over_price = float(prop_odds.get("over_price") or -110.0)
    under_price = float(prop_odds.get("under_price") or -110.0)

    z = (line - projection) / stdev
    probability_under = _normal_cdf(z)
    probability_over = 1.0 - probability_under

    fair_over, fair_under = remove_vig_two_way(over_price, under_price)
    edge_over = edge_vs_fair(probability_over, over_price, under_price, side="a")
    edge_under = edge_vs_fair(probability_under, over_price, under_price, side="b")
    ev_over = expected_value(probability_over, over_price)
    ev_under = expected_value(probability_under, under_price)
    recommended_side = "over" if edge_over >= edge_under else "under"

    return {
        **comparison_row,
        "over_price": over_price,
        "under_price": under_price,
        "model_probability_over": round(probability_over, 4),
        "model_probability_under": round(probability_under, 4),
        "market_fair_probability_over": round(fair_over, 4),
        "market_fair_probability_under": round(fair_under, 4),
        "probability_edge_over": round(edge_over, 4),
        "probability_edge_under": round(edge_under, 4),
        "ev_over": round(ev_over, 2),
        "ev_under": round(ev_under, 2),
        "recommended_priced_side": recommended_side,
        "recommended_priced_ev": round(ev_over if recommended_side == "over" else ev_under, 2),
    }


def index_props_by_player_and_category(props: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Key a :func:`modules.nba_props_loader.unified_props` list by ``(normalized player name, category)``."""
    return {(normalize_player_name(row["player_name"]), row["category"]): row for row in props}


def price_aware_evaluations(
    comparison_rows: list[dict[str, Any]], props_by_key: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Price every ``compare_props``/``recommend_props`` row that has a matching real price.

    Matches by ``(normalized player name, category)`` -- robust to whether
    a comparison row's own ``"player"`` field happens to be nba_api's raw
    roster name or an already-normalized one, since normalization is
    idempotent. A comparison row whose player+category isn't in
    ``props_by_key`` is skipped, not an error -- not every prop line
    necessarily carries a real price yet.
    """
    priced = []
    for row in comparison_rows:
        key = (normalize_player_name(row["player"]), row["category"])
        prop_odds = props_by_key.get(key)
        if prop_odds is None:
            continue
        priced.append(price_prop_comparison(row, prop_odds))
    priced.sort(key=lambda row: -abs(row["recommended_priced_ev"]))
    return priced
