"""Re-exports :mod:`betting.parallel_utils` -- see that module for the real implementation and docs.

Genuinely sport-agnostic infrastructure lives once, in
``fantasy_engine/betting/`` (alongside ``odds_math``, ``cache_utils``), and
is imported directly rather than duplicated -- see betting_engine.md. This
module exists only so NBA-side code can write the more locally-obvious
``from modules.parallel_utils import parallel_map``.
"""
from __future__ import annotations

from betting.parallel_utils import parallel_ev_map, parallel_map

__all__ = ["parallel_map", "parallel_ev_map"]
