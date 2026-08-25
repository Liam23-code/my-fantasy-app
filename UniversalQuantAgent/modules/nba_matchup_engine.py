"""Matchup-aware NBA betting context: combines the four real signals in matchup_engine.md into one bounded adjustment.

This module contains no new betting logic of its own -- it composes four
already-real, already-built signals (opponent-specific history, real
defensive difficulty, real opponent-injury context, real teammate on/off
swings) into one disclosed context dict, then applies a small, bounded,
capped adjustment to an already-priced NBA prop
(:func:`modules.nba_prop_model.price_prop_comparison`, called directly,
not reimplemented) exactly the way ``modules.fusion_model.fuse_projection``
already blends its own ``matchup_factor`` -- same scale, same rationale
(see that module's ``context_factor`` line): a modest, disclosed nudge,
never a large override, and never applied unless the underlying real data
was actually available.

Nothing here touches ``fuse_projection``, ``modules.matchup_model``, or
``modules.nba_prop_model`` -- this is a new, additive layer on top of the
existing pricing pipeline, called only from the pages that opt into it.
"""
from __future__ import annotations

from typing import Any

from modules.nba_advanced import latest_season
from modules.nba_defense_model import matchup_difficulty_score
from modules.nba_injury_impact import opponent_injury_context
from modules.nba_lineup_model import team_on_off_impact
from modules.nba_matchup_history import MIN_RELIABLE_GAMES, games_vs_opponent
from modules.nba_prop_model import price_prop_comparison

#: A real, absent rim defender modestly lowers effective difficulty for
#: an interior player -- fixed, disclosed points off the 0-100 difficulty
#: score, not a fitted coefficient (see betting_engine_advanced.md's
#: CORRELATION_ADJUSTMENT for the same "no real joint-outcome data to fit
#: one against" rationale).
_RIM_DEFENDER_OUT_RELIEF = 10.0

#: Caps how much real opponent-specific history can shift the model's
#: projection, in the same units as the stat itself -- a real 15-point
#: outlier game shouldn't single-handedly swing the projection more than
#: this, even with a reliable sample.
_MAX_HISTORY_SHIFT = 3.0


def compute_matchup_context(
    player_name: str,
    team: str,
    opponent_team: str,
    season: str | None = None,
    *,
    player_role: str = "wing",
    stat: str = "pts",
) -> dict[str, Any]:
    """Real, disclosed matchup context for one player against one opponent -- the four signals in matchup_engine.md.

    Returns ``{"defense", "history", "opponent_injuries", "team_lineup",
    "context_multiplier", "history_shift", "warnings"}``.
    ``context_multiplier`` is bounded to roughly [0.85, 1.15] -- the same
    modest-adjustment philosophy ``fusion_model.py``'s own
    ``matchup_factor`` already uses, applied here to genuinely new real
    signals (rim/perimeter defense, opponent absences) that
    ``modules.matchup_model`` doesn't compute.
    """
    season = season or latest_season()
    warnings: list[str] = []

    try:
        defense = matchup_difficulty_score(player_role, opponent_team, season)
    except Exception as exc:
        defense = {"difficulty_score": 50.0, "basis": "unavailable"}
        warnings.append(f"Defense model unavailable: {exc}")

    try:
        history = games_vs_opponent(player_name, opponent_team, season)
    except Exception as exc:
        history = {"games_sampled": 0, "delta_vs_season": {}}
        warnings.append(f"Matchup history unavailable: {exc}")

    try:
        opponent_injuries = opponent_injury_context(opponent_team, season)
    except Exception as exc:
        opponent_injuries = {"absences": [], "severity": 0.0, "rim_defender_out": None}
        warnings.append(f"Opponent injury context unavailable: {exc}")

    try:
        team_lineup = team_on_off_impact(team, season)
    except Exception as exc:
        team_lineup = {"players": []}
        warnings.append(f"Team lineup context unavailable: {exc}")

    effective_difficulty = defense.get("difficulty_score", 50.0)
    if player_role == "interior" and opponent_injuries.get("rim_defender_out"):
        effective_difficulty = max(0.0, effective_difficulty - _RIM_DEFENDER_OUT_RELIEF)

    # Same scale fusion_model.py's own matchup_factor already uses for its
    # difficulty_score -> factor conversion (see that module's docstring).
    context_multiplier = 1.0 + (50.0 - effective_difficulty) / 500.0

    games_sampled = history.get("games_sampled", 0)
    raw_shift = history.get("delta_vs_season", {}).get(stat, 0.0) if games_sampled >= MIN_RELIABLE_GAMES else 0.0
    history_shift = max(-_MAX_HISTORY_SHIFT, min(_MAX_HISTORY_SHIFT, raw_shift))

    return {
        "player": player_name,
        "team": team,
        "opponent": opponent_team,
        "season": season,
        "defense": defense,
        "history": history,
        "opponent_injuries": opponent_injuries,
        "team_lineup": team_lineup,
        "context_multiplier": round(context_multiplier, 4),
        "history_shift": round(history_shift, 2),
        "warnings": warnings,
    }


def matchup_adjusted_evaluation(comparison_row: dict[str, Any], prop_odds: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Re-price one already-compared NBA prop with :func:`compute_matchup_context`'s real, bounded adjustment.

    Builds an adjusted copy of ``comparison_row`` (projection nudged by
    ``context["context_multiplier"]`` and ``context["history_shift"]``,
    confidence band scaled proportionally so it stays centered) and prices
    it with :func:`modules.nba_prop_model.price_prop_comparison` directly
    -- the pricing math itself is not duplicated, only the input to it.
    """
    base_projection = float(comparison_row["minutes_adjusted_projection"])
    adjusted_projection = base_projection * context["context_multiplier"] + context["history_shift"]
    scale = adjusted_projection / base_projection if base_projection else 1.0

    adjusted_row = {
        **comparison_row,
        "minutes_adjusted_projection": round(adjusted_projection, 2),
        "confidence_low": round(float(comparison_row["confidence_low"]) * scale, 2),
        "confidence_high": round(float(comparison_row["confidence_high"]) * scale, 2),
    }
    priced = price_prop_comparison(adjusted_row, prop_odds)
    return {
        **priced,
        "pre_matchup_projection": round(base_projection, 2),
        "matchup_context_multiplier": context["context_multiplier"],
        "matchup_history_shift": context["history_shift"],
    }
