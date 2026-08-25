"""MLB season-average model: long-term regression baselines for the 7 modeled stat categories.

The baseline layer of the three-layer MLB engine (see mlb_pipeline.md):
season model -> DFS matchup model -> fusion model. Pure, offline math over
a per-player game log the caller already has (an uploaded file, or a
future generator) -- this module fetches nothing and fabricates nothing.

Two real, well-established sabermetric techniques, applied generically
rather than fit to any one dataset:

* **Regression to the mean via stabilization points** -- a counting stat's
  observed per-game rate is only partially reliable until enough games
  have accumulated; :func:`stabilize` blends the observed rate with a
  league-average prior, weighted by real sample size against a disclosed,
  published stabilization point per category (see
  :data:`STABILIZATION_GAMES`; the classic reference is Russell Carleton's
  sabermetric stabilization-point research). This is the same idea
  fantasy_engine/betting/prop_model.py's ``_FULL_SAMPLE_GAMES`` and
  modules/cbb_prop_model.py's minutes-based widening both already apply,
  generalized here into an explicit blend rather than only a variance
  widener.
* **Park-neutral projection** -- a raw rate is scaled by the batter's home
  park factor so two players with the same real talent but different home
  parks aren't compared on an uneven baseline; the matchup layer
  (mlb_ballpark_model.py) re-applies the *opposing* park's factor on top
  of this neutral baseline for a specific game.
"""
from __future__ import annotations

import statistics
from typing import Any

from modules.mlb_common import STAT_CATEGORIES

#: Real games needed before a category's observed per-game rate is
#: considered fully reliable (stabilization point, in games rather than
#: plate appearances/innings -- consistent with the rest of this codebase's
#: games-based sample-size convention, e.g. fantasy_engine's
#: ``_FULL_SAMPLE_GAMES``). Counting stats that occur less often per game
#: (home runs, stolen bases) take longer to stabilize than ones that occur
#: almost every game (hits, strikeouts allowed by a starter).
STABILIZATION_GAMES: dict[str, float] = {
    "hits": 30.0,
    "home_runs": 55.0,
    "rbi": 45.0,
    "total_bases": 35.0,
    "strikeouts": 15.0,  # pitcher Ks/start -- occurs every appearance, stabilizes fastest
    "walks": 40.0,
    "stolen_bases": 60.0,  # rare, situational -- slowest to stabilize
}

#: Full-sample reliability floor/ceiling mirrors betting.prop_model's
#: volatility-scaling convention (never fully 0 or 1).
_MIN_RELIABILITY = 0.05
_MAX_RELIABILITY = 0.98


def rolling_average(values: list[float], window: int = 15) -> float:
    """Average of the most recent ``window`` real per-game values (fewer than ``window`` -> all of them)."""
    if not values:
        return 0.0
    recent = values[-window:]
    return round(statistics.fmean(recent), 4)


def reliability_score(games_played: int, *, category: str) -> float:
    """0-1 real-sample-size reliability for one category, from its own stabilization point."""
    stabilization_point = STABILIZATION_GAMES.get(category, 40.0)
    if games_played <= 0:
        return _MIN_RELIABILITY
    raw = games_played / (games_played + stabilization_point)
    return round(max(_MIN_RELIABILITY, min(_MAX_RELIABILITY, raw)), 4)


def stabilize(raw_rate: float, league_mean: float, *, games_played: int, category: str) -> float:
    """Blend an observed per-game rate with the league-average prior, weighted by real sample size.

    ``reliability`` (see :func:`reliability_score`) is literally the
    regression weight: a player with few real games leans heavily on the
    league prior; a player past the stabilization point is trusted almost
    entirely on their own real rate.
    """
    reliability = reliability_score(games_played, category=category)
    return round(reliability * raw_rate + (1.0 - reliability) * league_mean, 4)


def park_neutral_rate(raw_rate: float, park_factor: float) -> float:
    """A raw per-game rate, scaled to remove the player's *home* park's real effect (100 = neutral)."""
    if park_factor <= 0:
        return round(raw_rate, 4)
    return round(raw_rate * (100.0 / park_factor), 4)


def _category_rate(game_log: list[dict[str, Any]], category: str) -> list[float]:
    return [float(game.get(category) or 0.0) for game in game_log]


def project_season_baseline(
    player: dict[str, Any], *, league_means: dict[str, float] | None = None, home_park_factor: float = 100.0
) -> dict[str, Any]:
    """Long-term regression baseline for every modeled category, from one player's real game log.

    ``player`` carries ``"game_log"`` -- a list of real per-game stat
    dicts (one entry per category in :data:`modules.mlb_common.STAT_CATEGORIES`
    per game); ``league_means`` (category -> league-average per-game rate)
    defaults to a neutral 0.0 prior when not supplied -- this module never
    fabricates a league average of its own. Returns
    ``{category: {"raw_mean", "reliability", "stabilized_mean", "park_neutral_mean", "games_played"}}``.
    """
    game_log = player.get("game_log") or []
    games_played = len(game_log)
    league_means = league_means or {}

    baseline: dict[str, Any] = {}
    for category in STAT_CATEGORIES:
        values = _category_rate(game_log, category)
        raw_mean = rolling_average(values, window=len(values) or 1) if values else 0.0
        league_mean = float(league_means.get(category, raw_mean))
        stabilized = stabilize(raw_mean, league_mean, games_played=games_played, category=category)
        baseline[category] = {
            "raw_mean": raw_mean,
            "reliability": reliability_score(games_played, category=category),
            "stabilized_mean": stabilized,
            "park_neutral_mean": park_neutral_rate(stabilized, home_park_factor),
            "games_played": games_played,
        }
    return baseline
