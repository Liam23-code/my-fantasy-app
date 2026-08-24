"""CFB parlay engine: re-exports betting.parlay_engine directly -- no new logic needed.

Unlike NBA (whose prop-market structure -- PRA overlapping points/
rebounds/assists -- has no NFL equivalent, so it needed its own
correlation patterns; see modules/nba_parlay_engine.py), CFB's requested
correlation patterns are QB passing <-> WR receiving and RB rushing <->
team total -- the *exact same* football role-based patterns already
implemented in ``betting.parlay_engine`` for NFL
(``qb_pass_catcher_stack``, ``rb_volume_and_game_total``). Those patterns
key off market names (``"passing_yards"``, ``"receiving_yards"``,
``"rushing_yards"``, ``"total"``) and leg roles (``team``, ``player_id``,
``side``), none of which are NFL-specific -- a CFB leg built with the same
market/role vocabulary triggers them identically. Writing a second copy of
this logic for CFB would just be the same code with a new docstring, so
this module re-exports the original instead (see betting_engine.md's
"shared, not duplicated" section, and modules/parallel_utils.py for the
same re-export pattern used elsewhere in this codebase).

If CFB ever needs a genuinely football-but-not-NFL-specific pattern (none
identified yet), add it to ``betting.parlay_engine`` directly rather than
forking a CFB-only copy -- NFL legs built with matching market/role
vocabulary would benefit from it too.
"""
from __future__ import annotations

from betting.parlay_engine import (
    correlation_adjusted_probability,
    detect_correlations,
    evaluate_parlay,
    make_leg,
    naive_joint_probability,
    parlay_decimal_odds,
    rank_parlays,
)

__all__ = [
    "make_leg",
    "detect_correlations",
    "correlation_adjusted_probability",
    "naive_joint_probability",
    "parlay_decimal_odds",
    "evaluate_parlay",
    "rank_parlays",
]
