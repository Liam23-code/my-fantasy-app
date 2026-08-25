"""Real CFB opponent defensive difficulty: a percentile over real points allowed.

Part of the matchup-aware betting engine (see matchup_engine.md), extending
the NBA-first pattern to CFB. Unlike NBA (real player-tracking defense
data) and NFL (real per-category yards allowed), CFB has no real
per-category defensive stat verified live yet under this build's
``CFBD_API_KEY``-gated access (see cfb_pipeline.md) -- so this module
builds the one real signal that *is* already fetched and verified,
``modules.cfb_team_model.team_scoring_averages``'s real
``points_allowed_avg``, into a percentile difficulty score, rather than
inventing an unverified new endpoint call. Fails soft to a neutral score
with no key configured, matching every other CFB module's contract.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from modules.cfb_team_model import team_scoring_averages


def matchup_difficulty_score(opponent_team: str, season: int) -> dict[str, Any]:
    """Real, percentile-ranked defensive difficulty (0-100, higher = tougher) from real points allowed this season."""
    averages = team_scoring_averages(season)
    if opponent_team not in averages or len(averages) < 2:
        return {"opponent": opponent_team, "season": season, "difficulty_score": 50.0, "basis": "no_real_data"}

    allowed = pd.Series({team: -values["points_allowed_avg"] for team, values in averages.items()})
    percentile = float(allowed.rank(pct=True).loc[opponent_team]) * 100.0
    return {
        "opponent": opponent_team,
        "season": season,
        "difficulty_score": round(percentile, 1),
        "points_allowed_avg": averages[opponent_team]["points_allowed_avg"],
        "games_sampled": averages[opponent_team]["games_played"],
        "basis": "real_points_allowed",
    }
