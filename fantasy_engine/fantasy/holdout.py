"""Holdout-season evaluation: score drafted rosters on a season the model never saw.

Usage::

    from fantasy.holdout import evaluate_draft_on_holdout

    result = evaluate_draft_on_holdout(draft_season=2024, holdout_season=2025, seeds=range(20))
    print(result["user_mean"], result["bot_mean"], result["user_win_rate"])

Why this module exists
----------------------
Scoring a simulated draft with the same season's actuals that produced the
projections is circular: :mod:`fantasy.user_brain` selects players by VORP
derived from season *Y*, so grading it on season *Y* measures how well it
optimized its own input, not whether the roster was actually good. That
inflates the apparent edge -- an earlier in-sample run showed the user team
beating the bot average by ~800 points, which is not a credible measure of
draft skill.

Here the draft is run on season ``draft_season`` projections and the
resulting rosters are scored on ``holdout_season`` actuals, which no part of
the drafting logic could see. Lamar Jackson projected 437 points on a 2024
basis and actually scored 213 in 2025; Joe Burrow 394 -> 140. A model that
looks strong in-sample and mediocre out-of-sample is exactly what this is
built to reveal.

ADP leakage: fixed, but only when you ask for it
------------------------------------------------
Earlier versions of this module carried an unavoidable lookahead leak: the
only ADP available was the present-day market, which was formed *with*
knowledge of how the holdout season turned out. That is no longer true --
:mod:`fantasy.historical_adp` fetches genuine pre-season snapshots (real
drafts measured that August, verified live). Pass ``adp_season`` to use one:

    evaluate_draft_on_holdout(draft_season=2024, adp_season=2025, holdout_season=2025)

drafts on 2024-actuals projections plus the market's *August 2025* view, then
scores on 2025 actuals -- exactly the information a real drafter had, and
nothing more. The returned ``leak_free`` flag records whether this was done;
**omitting ``adp_season`` leaves the old leak in place**, so a run without it
should not be described as clean.

One remaining limitation
------------------------
**Players absent from the holdout season score 0**, not dropped. Roughly
   23% of a season's pool has no following-season production (retirement,
   injury, leaving the league). Dropping them would silently flatter whichever
   team drafted more of them; scoring them 0 treats "drafted someone who never
   played" as the real cost it is.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from typing import Any

from fantasy.data_loader import load_real_projections
from fantasy.draft import simulate_draft

#: Points credited to a drafted player with no holdout-season production.
MISSING_PLAYER_POINTS = 0.0


def holdout_points_by_player_id(holdout_season: int, scoring_mode: str = "ppr") -> dict[str, float]:
    """Actual fantasy points each player scored in ``holdout_season``.

    Keyed by nflverse ``player_id``, which is stable across seasons -- the
    same identifier the draft pool uses, so no name matching is involved.
    """
    return {
        player["player_id"]: float(player["projection"])
        for player in load_real_projections(season=holdout_season, scoring_mode=scoring_mode)
        if player.get("player_id")
    }


def score_roster_on_holdout(
    roster: list[dict[str, Any]],
    holdout_points: dict[str, float],
) -> dict[str, Any]:
    """Score one drafted roster against holdout-season actuals.

    Returns ``{"total", "scored", "missing", "players"}``. ``missing`` counts
    roster spots with no holdout production -- credited
    :data:`MISSING_PLAYER_POINTS` rather than excluded, so a roster full of
    players who never took the field is correctly worth almost nothing.
    """
    players: list[dict[str, Any]] = []
    total = 0.0
    missing = 0
    for player in roster:
        player_id = player.get("player_id")
        if player_id in holdout_points:
            points = holdout_points[player_id]
        else:
            points = MISSING_PLAYER_POINTS
            missing += 1
        total += points
        players.append({"name": player.get("name"), "position": player.get("position"), "holdout_points": points})
    return {
        "total": round(total, 2),
        "scored": len(roster) - missing,
        "missing": missing,
        "players": players,
    }


def evaluate_draft_on_holdout(
    draft_season: int,
    holdout_season: int,
    seeds: Iterable[int] = range(10),
    n_teams: int = 12,
    num_rounds: int = 14,
    user_draft_slot: int = 6,
    scoring_mode: str = "ppr",
    league_settings: dict[str, Any] | None = None,
    adp_season: int | None = None,
) -> dict[str, Any]:
    """Draft on ``draft_season`` projections, score on ``holdout_season`` actuals.

    ``num_rounds`` defaults to 14 rather than 15 because a 15-round draft
    cannot fit inside :data:`fantasy.models.ROSTER_POSITION_LIMITS` when no
    DST data exists -- see the capacity warning in
    :func:`fantasy.draft.simulate_draft`. Using 14 keeps every team inside its
    caps so the comparison isn't distorted by forced overflow picks.

    Returns per-seed results plus aggregates: ``user_mean``, ``bot_mean``,
    ``user_win_rate``, and ``mean_margin``.
    """
    settings = league_settings or {
        "n_teams": n_teams,
        "scoring_mode": scoring_mode,
        "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 0, "K": 1, "BENCH": 6},
        "flex_eligible": ["RB", "WR", "TE"],
    }

    draft_pool = load_real_projections(season=draft_season, scoring_mode=scoring_mode)
    adp_coverage: dict[str, Any] | None = None
    if adp_season is not None:
        # Swap the present-day fused ADP for that season's real pre-season
        # snapshot -- this is what makes the evaluation leak-free.
        from fantasy.historical_adp import apply_historical_adp, historical_adp_coverage

        adp_coverage = historical_adp_coverage(draft_pool, adp_season)
        draft_pool = apply_historical_adp(draft_pool, season=adp_season, scoring=scoring_mode, teams=n_teams)
    holdout_points = holdout_points_by_player_id(holdout_season, scoring_mode=scoring_mode)
    user_team_name = f"Team {user_draft_slot}"

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        draft = simulate_draft(
            draft_pool,
            settings,
            num_teams=n_teams,
            num_rounds=num_rounds,
            user_draft_slot=user_draft_slot,
            seed=seed,
        )
        scored = {team: score_roster_on_holdout(roster, holdout_points) for team, roster in draft["rosters"].items()}
        user_total = scored[user_team_name]["total"]
        bot_totals = [result["total"] for team, result in scored.items() if team != user_team_name]
        per_seed.append(
            {
                "seed": seed,
                "user_total": user_total,
                "bot_mean": round(statistics.fmean(bot_totals), 2) if bot_totals else 0.0,
                "bot_best": max(bot_totals) if bot_totals else 0.0,
                "user_missing": scored[user_team_name]["missing"],
                "user_beat_bot_mean": bool(bot_totals) and user_total > statistics.fmean(bot_totals),
                "user_rank": 1 + sum(1 for total in bot_totals if total > user_total),
            }
        )

    if not per_seed:
        return {"per_seed": [], "user_mean": 0.0, "bot_mean": 0.0, "user_win_rate": 0.0, "mean_margin": 0.0}

    user_totals = [row["user_total"] for row in per_seed]
    bot_means = [row["bot_mean"] for row in per_seed]
    return {
        "draft_season": draft_season,
        "holdout_season": holdout_season,
        "adp_season": adp_season,
        "adp_coverage": adp_coverage,
        "leak_free": adp_season is not None and adp_season <= holdout_season,
        "per_seed": per_seed,
        "user_mean": round(statistics.fmean(user_totals), 2),
        "bot_mean": round(statistics.fmean(bot_means), 2),
        "mean_margin": round(statistics.fmean([u - b for u, b in zip(user_totals, bot_means, strict=True)]), 2),
        "user_win_rate": round(sum(1 for row in per_seed if row["user_beat_bot_mean"]) / len(per_seed), 3),
        "mean_user_rank": round(statistics.fmean([row["user_rank"] for row in per_seed]), 2),
        "n_teams": n_teams,
    }
