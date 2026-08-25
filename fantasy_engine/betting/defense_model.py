"""Real, position-group-aware NFL opponent defensive difficulty: pass defense vs. rush defense.

Part of the matchup-aware betting engine (see matchup_engine.md), extending
the NBA-first pattern in ``modules.nba_defense_model`` (UniversalQuantAgent)
to NFL with NFL's own real data source. ``betting.team_model``'s existing
``team_scoring_averages``/``project_game`` already model real *points*
scored/allowed -- this module adds a real, more granular signal on top
without touching that file: real per-game passing and rushing yards
*allowed*, from ``nflreadpy.load_team_stats`` (already a first-class
dependency -- see ``betting/odds_generator.py``'s ``load_schedules`` use).

This module intentionally does **not** build individual CB-vs-WR coverage
matchups or a weather signal -- see architecture.md's "what's not built"
convention: no legal offline NFL injury pipeline exists yet in this repo
(``nflreadpy.load_injuries`` is a real *live* feed, and injury data is
restricted to our own file or a user upload -- offline_data_contract.md
rule 1 -- the same rule that already keeps NBA/CFB/CBB injuries
offline-only), and no legal offline weather source was identified. Both
are disclosed gaps, not silent omissions -- see matchup_engine.md.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from betting.cache_utils import ttl_cache

_DEFENSE_CACHE_SECONDS = 60 * 60

#: Real position-group role -> which allowed-yardage column matters most,
#: and how heavily -- mirrors modules.nba_defense_model's role-weighting
#: shape (interior/perimeter there, pass/rush here).
_ROLE_WEIGHTS = {
    "passer": {"passing": 1.0, "rushing": 0.0},
    "receiver": {"passing": 1.0, "rushing": 0.0},
    "rusher": {"passing": 0.0, "rushing": 1.0},
}


@ttl_cache(_DEFENSE_CACHE_SECONDS)
def _fetch_team_stats(season: int) -> pd.DataFrame:
    import nflreadpy

    frame = nflreadpy.load_team_stats(seasons=[season])
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


def team_yards_allowed(season: int) -> dict[str, dict[str, float]]:
    """Real per-team-per-game passing/rushing yards allowed this season.

    Aggregates every real game's offense row by ``opponent_team`` (what
    the *opponent* produced against this team, the same "group by the
    other side" pattern ``modules.cfb_team_model.team_scoring_by_game``
    already uses for points). Returns
    ``{team: {"passing_yards_allowed_avg", "rushing_yards_allowed_avg", "games_played"}}``.
    Fails soft to ``{}`` on any provider error.
    """
    try:
        frame = _fetch_team_stats(season)
    except Exception:
        return {}
    if frame.empty or "opponent_team" not in frame:
        return {}
    grouped = frame.groupby("opponent_team")
    return {
        str(team): {
            "passing_yards_allowed_avg": round(float(pd.to_numeric(rows["passing_yards"], errors="coerce").mean()), 1),
            "rushing_yards_allowed_avg": round(float(pd.to_numeric(rows["rushing_yards"], errors="coerce").mean()), 1),
            "games_played": int(len(rows)),
        }
        for team, rows in grouped
    }


def matchup_difficulty_score(player_role: str, opponent_team: str, season: int) -> dict[str, Any]:
    """Real, role-weighted defensive difficulty (0-100, higher = tougher) for one NFL opponent.

    ``player_role`` is one of ``"passer"``, ``"receiver"``, ``"rusher"``
    -- a QB/WR's real difficulty comes from real yards allowed through
    the air; a RB's comes from real yards allowed on the ground. Fewer
    real yards allowed is a *tougher* defense, so the percentile is built
    from ``-yards_allowed`` (higher percentile = allows less = tougher).
    """
    weights = _ROLE_WEIGHTS.get(player_role, _ROLE_WEIGHTS["receiver"])
    allowed = team_yards_allowed(season)
    if opponent_team not in allowed or len(allowed) < 2:
        return {"opponent": opponent_team, "season": season, "player_role": player_role, "difficulty_score": 50.0, "basis": "no_real_data"}

    pass_series = pd.Series({team: -values["passing_yards_allowed_avg"] for team, values in allowed.items()})
    rush_series = pd.Series({team: -values["rushing_yards_allowed_avg"] for team, values in allowed.items()})
    pass_percentile = float(pass_series.rank(pct=True).loc[opponent_team]) * 100.0
    rush_percentile = float(rush_series.rank(pct=True).loc[opponent_team]) * 100.0

    difficulty = pass_percentile * weights["passing"] + rush_percentile * weights["rushing"]
    return {
        "opponent": opponent_team,
        "season": season,
        "player_role": player_role,
        "difficulty_score": round(difficulty, 1),
        "pass_defense_percentile": round(pass_percentile, 1),
        "rush_defense_percentile": round(rush_percentile, 1),
        "games_sampled": allowed[opponent_team]["games_played"],
        "basis": "real_yards_allowed",
    }
