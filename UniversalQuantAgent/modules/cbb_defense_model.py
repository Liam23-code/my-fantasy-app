"""Real CBB opponent defensive difficulty: a percentile over real points allowed, plus real pace context.

Part of the matchup-aware betting engine (see matchup_engine.md), extending
the NBA-first pattern to CBB. Reuses ``modules.cbb_team_model``'s already-
fetched real per-game scoring (``build_team_scoring_averages``) rather
than a new ESPN call -- the same "don't invent a second unverified fetch"
rationale as ``modules.cfb_defense_model``.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def matchup_difficulty_score(opponent_team: str, averages: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Real, percentile-ranked defensive difficulty (0-100, higher = tougher) from real points allowed.

    ``averages`` is already-fetched output from
    ``modules.cbb_team_model.build_team_scoring_averages`` -- this module
    takes it as an argument rather than fetching it again, since CBB's
    per-team fetch (a real per-team schedule call) is the expensive part
    and the caller has almost always already paid it (see
    ``modules.cbb_team_model.real_margin_and_total_volatility``'s
    identical "reuse the already-fetched pool" pattern).
    """
    if opponent_team not in averages or len(averages) < 2:
        return {"opponent": opponent_team, "difficulty_score": 50.0, "basis": "no_real_data"}

    allowed = pd.Series({team: -values["points_allowed_avg"] for team, values in averages.items()})
    percentile = float(allowed.rank(pct=True).loc[opponent_team]) * 100.0
    return {
        "opponent": opponent_team,
        "difficulty_score": round(percentile, 1),
        "points_allowed_avg": averages[opponent_team]["points_allowed_avg"],
        "games_sampled": averages[opponent_team]["games_played"],
        "basis": "real_points_allowed",
    }
