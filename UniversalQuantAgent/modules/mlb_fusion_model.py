"""MLB fusion model: blend the season-average baseline with the DFS matchup layer into a final projection.

The third layer of the three-layer MLB engine (see mlb_pipeline.md /
mlb_fusion_model.md): modules/mlb_season_model.py's long-term regression
baseline, adjusted by the four matchup multipliers computed separately by
modules/mlb_batter_vs_pitcher.py, modules/mlb_ballpark_model.py,
modules/mlb_bullpen_model.py (as a pitcher-quality read), and
modules/mlb_lineup_model.py. This module does not call any of those four
itself -- each stays independently testable, and this module only
combines already-computed scalars, the same "small composable pure
functions" pattern modules/cbb_moneyline_model.py's ``evaluate_game``
already uses for an already-computed volatility dict.

This is an optional overlay, not part of the unified betting-engine
contract (see modules/unified_betting_contract.py's docstring and
betting_engine.md) -- modules/mlb_prop_model.py, the contract-required
evaluator, prices whatever "line" a prop row already carries (default or
uploaded) directly, the same as CFB/CBB. A power user (or a future
generator) can run this fusion model to arrive at the number they trust
enough to put in that line, exactly the role NBA's
``modules.nba_matchup_engine.compute_matchup_context`` plays for NBA
props today.

## The weighting scheme

Six named weights, as specified for this engine:

* **matchup_difficulty**, **park_factor**, **pitcher_quality**,
  **lineup_protection** -- the *relative importance* of each of the four
  matchup-layer multipliers when they're blended into one combined
  adjustment (``adjustment_delta_raw`` below). Each of the four multiplier
  inputs is centered at 1.0 (neutral); a multiplier's distance from 1.0,
  scaled by its weight, is its contribution to the blended adjustment.
* **reliability**, **sample_size** -- together they set how much of that
  blended adjustment is actually trusted enough to move the season
  baseline (``confidence_scale`` below). A player with a highly reliable,
  large-sample season baseline (see modules/mlb_season_model.py) is
  adjusted by nearly the full matchup signal; a player with a thin season
  sample gets a damped adjustment, since stacking an uncertain matchup
  read on top of an already-uncertain baseline would compound noise
  rather than add real information. This is a documented modeling choice,
  not a fitted coefficient -- a deployment with a different view of how
  much to trust matchup context over season data can override
  :data:`WEIGHTS`.
"""
from __future__ import annotations

from typing import Any

from betting.prop_model import _risk_tier

from modules.mlb_common import STAT_CATEGORIES

WEIGHTS: dict[str, float] = {
    "reliability": 0.6,
    "sample_size": 0.4,
    "matchup_difficulty": 0.40,
    "park_factor": 0.25,
    "pitcher_quality": 0.25,
    "lineup_protection": 0.10,
}

#: Real games at which sample-size confidence (distinct from, but related
#: to, the season model's own per-category reliability) is treated as full.
_SAMPLE_SIZE_FULL_GAMES = 60.0


def sample_size_confidence(games_played: int) -> float:
    """0-1 confidence from raw games played alone, independent of any one category's stabilization point."""
    return round(max(0.0, min(1.0, games_played / _SAMPLE_SIZE_FULL_GAMES)), 4)


def combined_matchup_adjustment(
    *,
    matchup_difficulty_multiplier: float = 1.0,
    park_factor_multiplier: float = 1.0,
    pitcher_quality_multiplier: float = 1.0,
    lineup_protection_multiplier: float = 1.0,
) -> float:
    """The four matchup-layer multipliers, blended by their relative :data:`WEIGHTS` into one signed delta."""
    return (
        WEIGHTS["matchup_difficulty"] * (matchup_difficulty_multiplier - 1.0)
        + WEIGHTS["park_factor"] * (park_factor_multiplier - 1.0)
        + WEIGHTS["pitcher_quality"] * (pitcher_quality_multiplier - 1.0)
        + WEIGHTS["lineup_protection"] * (lineup_protection_multiplier - 1.0)
    )


def fuse_projection(
    season_baseline: dict[str, dict[str, Any]],
    *,
    matchup_difficulty_multiplier: float = 1.0,
    park_factor_by_category: dict[str, float] | None = None,
    pitcher_quality_multiplier: float = 1.0,
    lineup_protection_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Final per-category projection: the season baseline, adjusted by a reliability-scaled matchup signal.

    ``season_baseline`` is one player's output from
    :func:`modules.mlb_season_model.project_season_baseline`.
    ``park_factor_by_category`` (category -> multiplier, from
    :func:`modules.mlb_ballpark_model.park_adjustment`) is the one
    matchup input that's genuinely per-category; the other three apply
    uniformly across every category for this player/game (a documented
    simplification -- see module docstring).
    """
    park_factor_by_category = park_factor_by_category or {}
    projections: dict[str, Any] = {}

    for category in STAT_CATEGORIES:
        baseline = season_baseline.get(category, {"stabilized_mean": 0.0, "reliability": 0.05, "games_played": 0})
        stabilized_mean = float(baseline["stabilized_mean"])
        reliability = float(baseline["reliability"])
        games_played = int(baseline.get("games_played", 0))

        adjustment_delta_raw = combined_matchup_adjustment(
            matchup_difficulty_multiplier=matchup_difficulty_multiplier,
            park_factor_multiplier=park_factor_by_category.get(category, 1.0),
            pitcher_quality_multiplier=pitcher_quality_multiplier,
            lineup_protection_multiplier=lineup_protection_multiplier,
        )
        sample_confidence = sample_size_confidence(games_played)
        confidence_scale = WEIGHTS["reliability"] * reliability + WEIGHTS["sample_size"] * sample_confidence
        final_mean = max(0.0, stabilized_mean * (1.0 + adjustment_delta_raw * confidence_scale))

        overall_confidence = round(0.5 * reliability + 0.5 * confidence_scale, 4)
        pseudo_cv = max(0.05, 1.0 - overall_confidence)

        projections[category] = {
            "projection": round(final_mean, 4),
            "season_baseline": round(stabilized_mean, 4),
            "adjustment_delta": round(adjustment_delta_raw * confidence_scale, 4),
            "confidence": overall_confidence,
            "risk_tier": _risk_tier(pseudo_cv),
        }
    return projections
