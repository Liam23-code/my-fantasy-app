"""CBB parlay engine: re-exports modules.nba_parlay_engine directly -- no new logic needed.

CBB's requested correlation patterns -- PRA overlapping points/rebounds/
assists, and teammate scoring stacks -- are the *exact same* basketball
patterns already implemented in ``modules.nba_parlay_engine`` for NBA
(``overlapping_stat_categories``, ``teammate_scoring_stack``). Those
patterns key off market names (``"points"``, ``"rebounds"``, ``"assists"``,
``"PRA"``) and leg roles (``team``, ``player_id``, ``side``), none of
which are NBA-specific -- a CBB leg built with the same market/role
vocabulary triggers them identically. See modules/cfb_parlay_engine.py for
the same reasoning applied to football, and betting_engine.md's "shared,
not duplicated" section.
"""
from __future__ import annotations

from modules.nba_parlay_engine import (
    CORRELATION_ADJUSTMENT,
    evaluate_parlay,
    make_leg,
    nba_detect_correlations as detect_correlations,
    rank_parlays,
)

__all__ = ["make_leg", "detect_correlations", "evaluate_parlay", "rank_parlays", "CORRELATION_ADJUSTMENT"]
