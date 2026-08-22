"""Offline, deterministic sports-betting analytics engine.

Every number in this package traces back to either our own Quant Engine's
projections (:mod:`quant`, :mod:`projections`) or odds the caller supplied
directly -- our default ``odds.json`` or a user-uploaded file. Nothing here
ever fetches a network resource; :mod:`betting.odds_loader` only ever reads
local files or in-memory data.
"""

from __future__ import annotations

from .odds_loader import OddsLoadError, load_default_odds, load_uploaded_odds, merge_odds, unified_odds
from .odds_math import (
    american_to_decimal,
    decimal_to_american,
    edge,
    edge_vs_fair,
    expected_value,
    fair_price_from_probability,
    hold,
    implied_probability,
    remove_vig_two_way,
)
from .moneyline_model import evaluate_game, evaluate_games, fair_moneyline, win_probability_from_spread
from .parlay_engine import (
    correlation_adjusted_probability,
    detect_correlations,
    evaluate_parlay,
    make_leg,
    naive_joint_probability,
    parlay_decimal_odds,
    rank_parlays,
)
from .prop_model import evaluate_prop, evaluate_props, over_under_probability, stat_distribution
from .team_model import project_game, team_scoring_averages, team_scoring_by_week

__all__ = [
    "OddsLoadError",
    "american_to_decimal",
    "correlation_adjusted_probability",
    "decimal_to_american",
    "detect_correlations",
    "edge",
    "edge_vs_fair",
    "evaluate_game",
    "evaluate_games",
    "evaluate_parlay",
    "evaluate_prop",
    "evaluate_props",
    "expected_value",
    "fair_moneyline",
    "fair_price_from_probability",
    "hold",
    "implied_probability",
    "load_default_odds",
    "load_uploaded_odds",
    "make_leg",
    "merge_odds",
    "naive_joint_probability",
    "over_under_probability",
    "parlay_decimal_odds",
    "project_game",
    "rank_parlays",
    "remove_vig_two_way",
    "stat_distribution",
    "team_scoring_averages",
    "team_scoring_by_week",
    "unified_odds",
    "win_probability_from_spread",
]
