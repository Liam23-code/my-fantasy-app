"""Draft assistant: value-over-replacement ranking, cheat sheets, and picks.

Usage::

    from fantasy.draft import rank_players_for_draft, generate_cheatsheet, suggest_picks

    ranked = rank_players_for_draft(projections, league_settings)
    cheatsheet = generate_cheatsheet(projections, league_settings, top_n=150)
    choice = suggest_picks(draft_state, available_players, {"risk_tolerance": "balanced"})

Ranking is value-over-replacement (VOR): every player's projected fantasy
points (scored under the league's own scoring mode, not whatever the
projection source's default mode was) minus the points of the last plausible
starter at their position, once FLEX-eligible depth is accounted for. This is
what makes a PPR league rank pass-catching backs differently from a standard
league using the exact same raw projections.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fantasy.adapter import normalize_projection
from fantasy.models import LeagueSettings, RosterRequirements
from fantasy.scoring import calculate_fantasy_points
from fantasy.utils import safe_float

RISK_WEIGHTS = {
    "safe": {"floor": 0.6, "median": 0.4, "ceiling": 0.0},
    "balanced": {"floor": 0.15, "median": 0.7, "ceiling": 0.15},
    "boom_bust": {"floor": 0.0, "median": 0.4, "ceiling": 0.6},
}


def _coerce_league_settings(league_settings: dict[str, Any] | LeagueSettings) -> LeagueSettings:
    if isinstance(league_settings, LeagueSettings):
        return league_settings
    return LeagueSettings(**(league_settings or {}))


def _score_player(canonical: dict[str, Any], settings: LeagueSettings) -> float:
    result = calculate_fantasy_points(canonical, mode=settings.scoring_mode, custom_rules=settings.custom_rules)
    return result["total_points"]


def _volatility(canonical: dict[str, Any], points: float) -> float:
    floor = canonical.get("floor")
    ceiling = canonical.get("ceiling")
    if floor is None or ceiling is None:
        return 0.0
    spread = safe_float(ceiling) - safe_float(floor)
    return round(spread / points, 3) if points > 0 else 0.0


def _build_rationale(player: dict[str, Any], replacement_points: float) -> str:
    parts = [f"{player['vor']:+.1f} pts over positional replacement ({player['position']} rank {player['position_rank']})"]
    drivers = player.get("drivers") or []
    if drivers:
        parts.append(f"driven by {', '.join(drivers[:3])}")
    if player.get("volatility", 0) >= 0.9:
        parts.append("high week-to-week volatility (boom/bust profile)")
    elif player.get("volatility", 0) and player["volatility"] <= 0.35:
        parts.append("stable, low-volatility floor")
    if player.get("injury_status") in {"OUT", "DOUBTFUL", "IR"}:
        parts.append(f"elevated injury risk ({player['injury_status']})")
    elif player.get("injury_status") == "QUESTIONABLE":
        parts.append("mild injury risk (questionable)")
    if player.get("floor") is not None and player.get("ceiling") is not None:
        parts.append(f"confidence band {player['floor']:.1f}-{player['ceiling']:.1f} pts")
    return "; ".join(parts)


def _replacement_levels(
    scored_players: list[dict[str, Any]],
    settings: LeagueSettings,
    n_teams: int,
) -> dict[str, float]:
    """Compute the replacement-level (worst-starter) point total per position.

    Dedicated starters are filled first (``n_teams * starters_at_position``),
    then FLEX slots are filled from whichever flex-eligible players are left,
    pooled together and sorted by points. Replacement level for a position is
    the points of the next player at that position who did not make either a
    dedicated or FLEX starting slot.
    """
    by_position: dict[str, list[dict[str, Any]]] = {}
    for player in scored_players:
        by_position.setdefault(player["position"], []).append(player)
    for players in by_position.values():
        players.sort(key=lambda p: p["points"], reverse=True)

    starters = settings.roster_requirements.starting_slots()
    flex_count = starters.pop("FLEX", 0) * n_teams
    dedicated_counts = {position: count * n_teams for position, count in starters.items()}

    flex_pool: list[dict[str, Any]] = []
    for position in settings.flex_eligible:
        players = by_position.get(position, [])
        dedicated = dedicated_counts.get(position, 0)
        flex_pool.extend(players[dedicated:])
    flex_pool.sort(key=lambda p: p["points"], reverse=True)
    flex_starters = flex_pool[:flex_count]
    flex_starter_ids = {id(player) for player in flex_starters}

    replacement: dict[str, float] = {}
    for position, players in by_position.items():
        dedicated = dedicated_counts.get(position, 0)
        starter_count = dedicated
        if position in settings.flex_eligible:
            starter_count += sum(1 for player in players[dedicated:] if id(player) in flex_starter_ids)
        # `players` is never empty here: a position only appears in
        # `by_position` because at least one scored player was grouped into
        # it, so the "no starters made it, no players left either" case
        # cannot occur -- the worst-case fallback is simply the last player.
        if starter_count < len(players):
            replacement[position] = players[starter_count]["points"]
        else:
            replacement[position] = players[-1]["points"]
    return replacement


def rank_players_for_draft(
    projections: list[dict[str, Any]],
    league_settings: dict[str, Any] | LeagueSettings,
    roster_requirements: dict[str, Any] | None = None,
    n_teams: int | None = None,
) -> list[dict[str, Any]]:
    """Rank a pool of projections by value-over-replacement for a draft.

    ``roster_requirements``/``n_teams`` override the corresponding fields on
    ``league_settings`` when provided, so callers can reuse one league
    settings object across differently-sized mock drafts.
    """
    settings = _coerce_league_settings(league_settings)
    if roster_requirements is not None:
        settings = settings.model_copy(update={"roster_requirements": RosterRequirements(**roster_requirements)})
    if n_teams is not None:
        settings = settings.model_copy(update={"n_teams": n_teams})

    scored: list[dict[str, Any]] = []
    for source in projections:
        canonical = normalize_projection(source)
        points = _score_player(canonical, settings)
        scored.append({**canonical, "points": round(points, 2)})

    replacement_levels = _replacement_levels(scored, settings, settings.n_teams)

    ranked: list[dict[str, Any]] = []
    for player in scored:
        replacement_points = replacement_levels.get(player["position"], 0.0)
        ranked.append(
            {
                **player,
                "vor": round(player["points"] - replacement_points, 2),
                "replacement_points": round(replacement_points, 2),
                "volatility": _volatility(player, player["points"]),
            }
        )

    position_rank: dict[str, int] = {}
    for player in sorted(ranked, key=lambda p: p["points"], reverse=True):
        position_rank[player["position"]] = position_rank.get(player["position"], 0) + 1
        player["position_rank"] = position_rank[player["position"]]

    ranked.sort(key=lambda p: p["vor"], reverse=True)
    for index, player in enumerate(ranked, start=1):
        player["overall_rank"] = index
        player["rationale"] = _build_rationale(player, player["replacement_points"])

    return ranked


def generate_cheatsheet(
    projections: list[dict[str, Any]],
    league_settings: dict[str, Any] | LeagueSettings,
    roster_requirements: dict[str, Any] | None = None,
    top_n: int = 200,
) -> list[dict[str, Any]]:
    """Return the top ``top_n`` players in draft-board order."""
    ranked = rank_players_for_draft(projections, league_settings, roster_requirements)
    return ranked[:top_n]


def _position_need_multiplier(
    roster_position_counts: dict[str, int],
    settings: LeagueSettings,
    position: str,
) -> float:
    starters = settings.roster_requirements.starting_slots()
    required = float(starters.get(position, 0))
    if position in settings.flex_eligible:
        flex_share = starters.get("FLEX", 0) / max(len(settings.flex_eligible), 1)
        required += flex_share
    have = roster_position_counts.get(position, 0)
    if have < required:
        return 1.25
    if have < required + 1:
        return 1.0
    return 0.85


def suggest_picks(
    current_draft_state: dict[str, Any],
    available_players: list[dict[str, Any]],
    user_preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recommend the best available pick plus a short list of alternatives.

    ``current_draft_state`` expects ``{"league_settings": {...}, "my_roster":
    [ {"position": "RB", ...}, ... ]}``. ``user_preferences`` supports
    ``risk_tolerance`` (``"safe"``/``"balanced"``/``"boom_bust"``) and an
    optional ``position_priority`` list that nudges the composite score.
    """
    preferences = user_preferences or {}
    risk_tolerance = preferences.get("risk_tolerance", "balanced")
    if risk_tolerance not in RISK_WEIGHTS:
        raise ValueError(f"Unknown risk_tolerance {risk_tolerance!r}; expected one of {sorted(RISK_WEIGHTS)}")
    weights = RISK_WEIGHTS[risk_tolerance]

    settings = _coerce_league_settings(current_draft_state.get("league_settings", {}))
    my_roster = current_draft_state.get("my_roster", [])
    roster_position_counts: dict[str, int] = {}
    for player in my_roster:
        position = str(player.get("position", "")).strip().upper()
        roster_position_counts[position] = roster_position_counts.get(position, 0) + 1

    ranked = rank_players_for_draft(available_players, settings)
    priority_positions = {p.strip().upper() for p in preferences.get("position_priority", [])}

    for player in ranked:
        need_multiplier = _position_need_multiplier(roster_position_counts, settings, player["position"])
        raw_median, raw_floor, raw_ceiling = player.get("median"), player.get("floor"), player.get("ceiling")
        median = raw_median if raw_median is not None else player["points"]
        floor = raw_floor if raw_floor is not None else player["points"]
        ceiling = raw_ceiling if raw_ceiling is not None else player["points"]
        risk_adjusted_points = weights["floor"] * floor + weights["median"] * median + weights["ceiling"] * ceiling
        priority_boost = 1.1 if player["position"] in priority_positions else 1.0
        base_value = player["vor"] if player["vor"] > 0 else player["points"]
        player["pick_score"] = round(base_value * need_multiplier * priority_boost * (risk_adjusted_points / median if median else 1.0), 2)
        player["need_multiplier"] = need_multiplier

    ranked.sort(key=lambda p: p["pick_score"], reverse=True)
    if not ranked:
        return {"best_pick": None, "alternatives": []}

    best_pick = ranked[0]
    alternatives = ranked[1:5]
    return {"best_pick": best_pick, "alternatives": alternatives}


def simulate_draft(
    projections: list[Any],
    league_settings: dict[str, Any] | LeagueSettings,
    rounds: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate a snake mock draft (round 1 goes 1..N, round 2 goes N..1, etc.).

    Each pick samples from the top 3 remaining players weighted by VOR
    (softmax-like, via ``np.random.Generator.choice``) rather than always
    taking the single best player, so the simulation isn't perfectly
    deterministic team-to-team -- but ``seed`` makes a given run fully
    reproducible. ``rounds`` defaults to one team's total roster size
    (starters + bench + IR + taxi).
    """
    settings = _coerce_league_settings(league_settings)
    if rounds is None:
        rounds = sum(settings.roster_requirements.model_dump().values())
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    remaining = rank_players_for_draft(projections, settings)
    rng = np.random.default_rng(seed)
    rosters: dict[str, list[dict[str, Any]]] = {f"Team {team}": [] for team in range(1, settings.n_teams + 1)}
    picks: list[dict[str, Any]] = []
    overall_pick = 0

    for round_number in range(1, rounds + 1):
        order = range(1, settings.n_teams + 1) if round_number % 2 == 1 else range(settings.n_teams, 0, -1)
        for team_number in order:
            if not remaining:
                break
            overall_pick += 1
            pool = remaining[: min(3, len(remaining))]
            weights = np.array([max(candidate["vor"], 0.01) for candidate in pool])
            weights = weights / weights.sum()
            choice_index = int(rng.choice(len(pool), p=weights))
            player = remaining.pop(choice_index)
            team_name = f"Team {team_number}"
            rosters[team_name].append(player)
            picks.append(
                {
                    "overall_pick": overall_pick,
                    "round": round_number,
                    "team": team_name,
                    "player_id": player["player_id"],
                    "name": player["name"],
                    "position": player["position"],
                    "vor": player["vor"],
                }
            )

    return {"picks": picks, "rosters": rosters, "n_teams": settings.n_teams, "rounds": rounds, "seed": seed}
