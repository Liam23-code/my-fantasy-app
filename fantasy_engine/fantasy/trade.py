"""Trade analyzer with a Monte Carlo rest-of-season simulation.

Usage::

    from fantasy.trade import evaluate_trade

    result = evaluate_trade(
        team_a_players=["Saquon Barkley"],
        team_b_players=["Puka Nacua", "Bijan Robinson"],
        league_settings=league_settings,
        projections=full_player_pool,  # only needed if a side lists bare names/ids
    )

``team_a_players`` is what Team A gives up (i.e. what Team B receives), and
``team_b_players`` is what Team B gives up (i.e. what Team A receives) --
the standard "each side lists what they're sending" trade convention.
``fair_value`` is the deterministic (median-based) point swing over the
remaining season in favor of Team A; ``win_prob_delta`` comes from the Monte
Carlo simulation and is Team A's probability of ending up net ahead on the
trade, minus the 0.5 break-even baseline.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fantasy.adapter import normalize_projection
from fantasy.draft import _position_need_multiplier
from fantasy.models import LeagueSettings
from fantasy.scoring import calculate_fantasy_points
from fantasy.utils import safe_float

DEFAULT_WEEKS_REMAINING = 10
DEFAULT_MONTE_CARLO_ITERATIONS = 5000
FAIR_TRADE_THRESHOLD_PCT = 0.05  # net swing within 5% of total value swapped is "fair"


def _coerce_league_settings(league_settings: dict[str, Any] | LeagueSettings) -> LeagueSettings:
    if isinstance(league_settings, LeagueSettings):
        return league_settings
    return LeagueSettings(**(league_settings or {}))


def _build_lookup(projections: list[Any] | None) -> dict[str, dict[str, Any]]:
    if not projections:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for source in projections:
        canonical = normalize_projection(source)
        if canonical["player_id"]:
            lookup[canonical["player_id"]] = canonical
        if canonical["name"]:
            lookup.setdefault(canonical["name"].strip().lower(), canonical)
    return lookup


def _resolve_side(players: list[Any], lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    resolved = []
    for player in players:
        if isinstance(player, str):
            key = player if player in lookup else player.strip().lower()
            if key not in lookup:
                raise ValueError(f"Could not resolve player {player!r}; pass it in `projections` for lookup.")
            resolved.append(lookup[key])
        else:
            resolved.append(normalize_projection(player))
    return resolved


def _weekly_distribution(canonical: dict[str, Any], settings: LeagueSettings) -> tuple[float, float]:
    """Return (mean, std) of one player's weekly fantasy points."""
    mean = calculate_fantasy_points(canonical, mode=settings.scoring_mode, custom_rules=settings.custom_rules)["total_points"]
    floor, ceiling = canonical.get("floor"), canonical.get("ceiling")
    if floor is not None and ceiling is not None and ceiling > floor:
        # Treat [floor, ceiling] as a rough 5th-95th percentile band (+/-1.645 SD).
        std = (safe_float(ceiling) - safe_float(floor)) / 3.29
    else:
        std = max(abs(mean) * 0.35, 1.0)
    return mean, max(std, 0.1)


def _season_totals(
    players: list[dict[str, Any]],
    settings: LeagueSettings,
    weeks_remaining: int,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    totals = np.zeros(iterations)
    for player in players:
        weekly_mean, weekly_std = _weekly_distribution(player, settings)
        season_mean = weekly_mean * weeks_remaining
        season_std = weekly_std * (weeks_remaining**0.5)
        totals += rng.normal(loc=season_mean, scale=season_std, size=iterations)
    return totals


def _need_context(roster: list[dict[str, Any]] | None, settings: LeagueSettings, received: list[dict[str, Any]]) -> str | None:
    if roster is None or not received:
        return None
    position_counts: dict[str, int] = {}
    for player in roster:
        position = str(player.get("position", "")).strip().upper()
        position_counts[position] = position_counts.get(position, 0) + 1
    boosted = [p["name"] for p in received if _position_need_multiplier(position_counts, settings, p["position"]) > 1.0]
    if boosted:
        return f"Addresses a roster need at: {', '.join(boosted)}."
    return "Does not address a starting roster need for this side."


def evaluate_trade(
    team_a_players: list[Any],
    team_b_players: list[Any],
    league_settings: dict[str, Any] | LeagueSettings,
    projections: list[Any] | None = None,
    monte_carlo_iterations: int = DEFAULT_MONTE_CARLO_ITERATIONS,
    weeks_remaining: int = DEFAULT_WEEKS_REMAINING,
    team_a_roster: list[dict[str, Any]] | None = None,
    team_b_roster: list[dict[str, Any]] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a proposed trade with a Monte Carlo rest-of-season simulation.

    ``team_a_players``/``team_b_players`` entries may be full projection
    dicts/objects, or bare player id/name strings resolved against
    ``projections`` (a pool of every relevant player's projection). Rosters
    are optional context used only to note whether the trade fills a
    starting need for either side.
    """
    if monte_carlo_iterations < 1:
        raise ValueError("monte_carlo_iterations must be >= 1")
    if weeks_remaining < 1:
        raise ValueError("weeks_remaining must be >= 1")

    settings = _coerce_league_settings(league_settings)
    lookup = _build_lookup(projections)
    gives_a = _resolve_side(team_a_players, lookup)  # what A sends away (B receives)
    gives_b = _resolve_side(team_b_players, lookup)  # what B sends away (A receives)

    a_receives_mean = sum(_weekly_distribution(p, settings)[0] for p in gives_b) * weeks_remaining
    a_gives_mean = sum(_weekly_distribution(p, settings)[0] for p in gives_a) * weeks_remaining
    fair_value = round(a_receives_mean - a_gives_mean, 2)

    rng = np.random.default_rng(seed)
    a_receives_sim = _season_totals(gives_b, settings, weeks_remaining, monte_carlo_iterations, rng)
    a_gives_sim = _season_totals(gives_a, settings, weeks_remaining, monte_carlo_iterations, rng)
    a_net = a_receives_sim - a_gives_sim
    a_win_probability = float(np.mean(a_net > 0))
    win_prob_delta = round(a_win_probability - 0.5, 4)

    total_value_swapped = max(a_receives_mean, a_gives_mean, 1.0)
    if abs(fair_value) / total_value_swapped <= FAIR_TRADE_THRESHOLD_PCT:
        recommendation = "Fair trade: value is roughly even for both sides."
    elif fair_value > 0:
        recommendation = "Favors Team A: they receive more projected value than they give up."
    else:
        recommendation = "Favors Team B: they receive more projected value than they give up."

    rationale = [
        f"Team A gives up {', '.join(p['name'] for p in gives_a) or 'nothing'} "
        f"({a_gives_mean:.1f} pts over {weeks_remaining} weeks) and receives "
        f"{', '.join(p['name'] for p in gives_b) or 'nothing'} ({a_receives_mean:.1f} pts).",
        f"Net swing: {fair_value:+.1f} pts over the remaining season in favor of "
        f"{'Team A' if fair_value > 0 else 'Team B' if fair_value < 0 else 'neither side'}.",
        f"Monte Carlo ({monte_carlo_iterations} simulations): Team A ends up ahead in "
        f"{a_win_probability * 100:.1f}% of simulated outcomes.",
    ]
    a_need_note = _need_context(team_a_roster, settings, gives_b)
    if a_need_note:
        rationale.append(f"Team A: {a_need_note}")
    b_need_note = _need_context(team_b_roster, settings, gives_a)
    if b_need_note:
        rationale.append(f"Team B: {b_need_note}")

    return {
        "fair_value": fair_value,
        "recommendation": recommendation,
        "win_prob_delta": win_prob_delta,
        "rationale": rationale,
        "team_a_receives_points": round(a_receives_mean, 2),
        "team_a_gives_points": round(a_gives_mean, 2),
        "team_a_win_probability": round(a_win_probability, 4),
        "weeks_remaining": weeks_remaining,
        "monte_carlo_iterations": monte_carlo_iterations,
    }
