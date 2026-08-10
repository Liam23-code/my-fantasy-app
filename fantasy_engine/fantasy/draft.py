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

from collections import Counter
from typing import Any

import numpy as np

from fantasy.adapter import normalize_projection
from fantasy.assistant import picks_between
from fantasy.data_loader import is_synthetic
from fantasy.models import (
    ROSTER_POSITION_LIMITS,
    LeagueSettings,
    RosterRequirements,
    roster_cap_reached,
    roster_min_required,
)
from fantasy.room_brain import is_round_one_chalk, room_brain_weight, room_candidate_pool, round_one_pick
from fantasy.scoring import calculate_fantasy_points
from fantasy.user_brain import user_brain_pick
from fantasy.utils import safe_float

RISK_WEIGHTS = {
    "safe": {"floor": 0.6, "median": 0.4, "ceiling": 0.0},
    "balanced": {"floor": 0.15, "median": 0.7, "ceiling": 0.15},
    "boom_bust": {"floor": 0.0, "median": 0.4, "ceiling": 0.6},
}

#: Skill positions that can plausibly form a draft "run" -- K/DST are excluded
#: even though their ADPs cluster tightly, since there is only ever one
#: startable slot at either and a cluster there isn't a run in the fantasy
#: sense.
RUN_POSITIONS = ("QB", "RB", "WR", "TE")

#: How many candidates a simulated team weighs before picking. Wide enough
#: that the need/run bias below can actually reorder who gets taken -- with
#: only the single best player considered, a bias multiplier has nothing to
#: act on.
CANDIDATE_POOL_SIZE = 5

#: Weight multiplier for a candidate whose position is below the team's
#: :data:`fantasy.models.ROSTER_POSITION_LIMITS` minimum.
NEED_BOOST = 1.3

#: Weight multiplier for a candidate whose position is in an active run.
#: Section-3 fix: 1.6 wasn't a strong enough pull to reliably show up as a
#: visible run in a 12-team board; teams now follow a run far more eagerly.
RUN_BOOST = 3.0

#: An ADP cluster is treated as "active" starting this many picks before its
#: earliest ADP (teams start anticipating the run before it technically
#: begins) through this many picks after its latest ADP (stragglers still get
#: pulled along).
RUN_LOOKAHEAD_PICKS = 3
RUN_TRAILING_PICKS = 2

#: Positions that never get a 2nd copy, even in a roster-cap emergency -- a
#: backup kicker or DST is never realistic roster construction, unlike a
#: 5th/6th RB or WR, which real benches carry all the time.
NEVER_EXCEED_CAP_POSITIONS = frozenset({"K", "DST"})


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


def build_draft_order(num_teams: int, num_rounds: int, snake: bool = True) -> list[dict[str, int]]:
    """Lay out every pick in the draft, in order.

    Returns one entry per pick as ``{"overall_pick", "round", "team_number",
    "slot_in_round"}``. A snake draft reverses on even rounds (1..N, N..1,
    1..N); a linear draft runs 1..N every round.
    """
    if num_teams < 1:
        raise ValueError("num_teams must be >= 1")
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")

    order: list[dict[str, int]] = []
    overall_pick = 0
    for round_number in range(1, num_rounds + 1):
        teams = range(1, num_teams + 1)
        if snake and round_number % 2 == 0:
            teams = range(num_teams, 0, -1)
        for slot, team_number in enumerate(teams, start=1):
            overall_pick += 1
            order.append(
                {
                    "overall_pick": overall_pick,
                    "round": round_number,
                    "team_number": team_number,
                    "slot_in_round": slot,
                }
            )
    return order


def draft_projection(player: dict[str, Any]) -> float:
    """The player's projected points, preferring a real precomputed projection."""
    for key in ("projection", "expected_fantasy_points"):
        value = player.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), 2)
    return round(safe_float(player.get("points")), 2)


def _describe_cluster(position: str, entries: list[tuple[float, str]]) -> dict[str, Any]:
    return {
        "position": position,
        "start_adp": round(entries[0][0], 1),
        "end_adp": round(entries[-1][0], 1),
        "size": len(entries),
        "players": [name for _, name in entries],
    }


def identify_adp_clusters(
    players: list[dict[str, Any]],
    max_gap: float = 6.0,
    min_size: int = 3,
) -> list[dict[str, Any]]:
    """Find same-position ADP clusters -- the tight bunches that trigger a run.

    A cluster is a maximal group of same-position players, sorted by ADP,
    where each consecutive pair is within ``max_gap`` picks of each other.
    This is the real "cluster then a cliff" pattern: once a draft reaches a
    tight bunch of, say, five TEs bunched between picks 45-60, teams tend to
    keep taking TEs rather than risk the cliff in value right after the
    cluster ends. Only :data:`RUN_POSITIONS` are considered; players missing
    an ``adp`` are ignored (no real signal to cluster on).

    Returns clusters sorted by ``start_adp``, each as ``{"position",
    "start_adp", "end_adp", "size", "players"}``.
    """
    by_position: dict[str, list[tuple[float, str]]] = {}
    for player in players:
        position = str(player.get("position", "")).strip().upper()
        if position not in RUN_POSITIONS:
            continue
        adp = player.get("adp")
        if adp is None:
            continue
        by_position.setdefault(position, []).append((safe_float(adp), str(player.get("name", ""))))

    clusters: list[dict[str, Any]] = []
    for position, entries in by_position.items():
        entries.sort(key=lambda entry: entry[0])
        current: list[tuple[float, str]] = []
        for entry in entries:
            if current and entry[0] - current[-1][0] > max_gap:
                if len(current) >= min_size:
                    clusters.append(_describe_cluster(position, current))
                current = []
            current.append(entry)
        if len(current) >= min_size:
            clusters.append(_describe_cluster(position, current))

    clusters.sort(key=lambda cluster: cluster["start_adp"])
    return clusters


def active_run_position(overall_pick: int, clusters: list[dict[str, Any]]) -> str | None:
    """The position an ADP cluster is actively driving a run on, if any.

    When multiple clusters overlap at the same pick (rare), the one with the
    earliest ``start_adp`` wins -- ``clusters`` is kept sorted that way.
    """
    for cluster in clusters:
        if cluster["start_adp"] - RUN_LOOKAHEAD_PICKS <= overall_pick <= cluster["end_adp"] + RUN_TRAILING_PICKS:
            return cluster["position"]
    return None


def _room_pick(
    pool: list[dict[str, Any]],
    team_counts: Counter,
    overall_pick: int,
    round_number: int,
    active_run: str | None,
    tolerance_scale: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Weighted room-brain choice among ``pool`` for a non-round-1 pick.

    Verbatim extraction of what the ``simulate_draft`` loop always did for a
    non-user team past round 1 -- moved here unchanged so :func:`room_team_pick`
    (used by the live, pick-by-pick draft) and ``simulate_draft`` (batch mode)
    share one implementation instead of two copies that could quietly drift.
    """
    weights = []
    for candidate in pool:
        scale = tolerance_scale.get(candidate["position"], 1.0)
        weight = room_brain_weight(candidate, overall_pick, round_number, scale)
        if team_counts.get(candidate["position"], 0) < roster_min_required(candidate["position"]):
            weight *= NEED_BOOST
        if active_run and candidate["position"] == active_run:
            weight *= RUN_BOOST
        weights.append(weight)
    weights_array = np.array(weights)
    weights_array = weights_array / weights_array.sum()
    choice_index = int(rng.choice(len(pool), p=weights_array))
    return pool[choice_index]


def room_team_pick(
    eligible: list[dict[str, Any]],
    team_counts: Counter,
    overall_pick: int,
    round_number: int,
    active_run: str | None,
    tolerance_scale: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """One non-user team's pick -- the exact decision :func:`simulate_draft` makes.

    Covers both branches a room team can hit: round-1 chalk
    (:func:`fantasy.room_brain.is_round_one_chalk`) and the weighted pick past
    it (:func:`_room_pick`). Public and reusable so a live, pausable,
    pick-by-pick draft (see :mod:`fantasy.live_draft`) can drive bot teams with
    the identical, already-validated room model instead of a second
    implementation -- only the *loop* differs (resumable vs. run-to-completion),
    never the decision itself.
    """
    pool = room_candidate_pool(eligible, CANDIDATE_POOL_SIZE)
    if is_round_one_chalk(round_number):
        return round_one_pick(pool[: min(2, len(pool))], rng)
    return _room_pick(pool, team_counts, overall_pick, round_number, active_run, tolerance_scale, rng)


def simulate_draft(
    projections: list[Any],
    league_settings: dict[str, Any] | LeagueSettings,
    rounds: int | None = None,
    seed: int | None = None,
    num_teams: int | None = None,
    num_rounds: int | None = None,
    snake: bool = True,
    user_draft_slot: int | None = None,
) -> dict[str, Any]:
    """Simulate a mock draft over a pool of **real** players only.

    The draft order comes from :func:`build_draft_order`, so ``num_teams`` /
    ``num_rounds`` / ``snake`` fully describe it -- set ``snake=False`` for a
    linear draft where every round runs 1..N. ``user_draft_slot`` marks one
    team as yours so its picks come back flagged.

    Synthetic players are filtered out before the draft starts: a mock draft
    board full of fabricated names is worse than a short one.

    Every simulated team drafts under the same roster-construction rules
    (:data:`fantasy.models.ROSTER_POSITION_LIMITS`): a team already at the
    cap for a position (e.g. its 2nd QB in a 1-2 QB league) will not draft a
    3rd, and a team still below the minimum at a position is weighted toward
    filling it (``NEED_BOOST``). :func:`identify_adp_clusters` is run once on
    the pool up front, and any pick landing inside an active cluster's range
    biases that team toward the clustered position (``RUN_BOOST``).

    Two different decision models drive picks after round 1 (round 1 itself
    is pure market/ADP order for every team -- see :func:`fantasy.room_brain.is_round_one_chalk`):

    * ``user_draft_slot``'s team picks via :func:`fantasy.user_brain.user_brain_pick`
      -- the same multi-signal engine (VORP, ADP timing, scarcity, need, risk,
      byes, stacking) behind the live Draft Assistant tab.
    * Every other team picks via :func:`fantasy.room_brain.room_brain_weight`
      -- a simpler ADP-anchored heuristic with positional tendencies (RB-heavy
      early, WR waves, TE/QB patience) and early-round risk aversion, modeling
      a typical draft room rather than expert judgment.

    This is what "the user's team always drafts from the stronger model"
    means in practice: every pick after round 1 goes through a materially
    richer decision process than the room's, not a rigged or guaranteed
    outcome for any single simulated draft.

    Returns a dict whose ``by_round`` key holds the round-by-round board::

        {1: [{"pick": 1, "team": "Team 1", "player_name": "...",
              "position": "RB", "projection": 428.6, "is_user_pick": False,
              "position_run": None, "adp": 8.5, "vor": 279.9,
              "remaining_at_position": 42}, ...],
         2: [...]}

    ``picks`` (a flat list), ``rosters``, ``n_teams``, ``rounds``, ``seed``,
    and ``position_runs`` (the clusters :func:`identify_adp_clusters` found)
    are also returned for callers that want the whole draft at once.
    """
    settings = _coerce_league_settings(league_settings)
    if num_teams is not None:
        settings = settings.model_copy(update={"n_teams": int(num_teams)})
    if num_rounds is not None:
        rounds = int(num_rounds)
    if rounds is None:
        rounds = sum(settings.roster_requirements.model_dump().values())
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if user_draft_slot is not None and not 1 <= user_draft_slot <= settings.n_teams:
        raise ValueError(f"user_draft_slot must be between 1 and {settings.n_teams}")

    warnings: list[str] = []
    real_projections = [player for player in (projections or []) if not is_synthetic(player)]
    dropped = len(projections or []) - len(real_projections)
    if dropped:
        warnings.append(f"Excluded {dropped} synthetic player(s) from the draft pool.")
    if not real_projections:
        warnings.append("No real players were available to draft.")

    remaining = rank_players_for_draft(real_projections, settings) if real_projections else []
    requested_picks = settings.n_teams * rounds
    if len(remaining) < requested_picks:
        warnings.append(
            f"Pool has {len(remaining)} real players but {requested_picks} picks were requested "
            f"({settings.n_teams} teams x {rounds} rounds); the draft will end early."
        )

    # A team can only stay inside ROSTER_POSITION_LIMITS if the caps, summed
    # over positions that actually have players available, cover every round.
    # They often don't: DST carries a (1, 1) cap but nflverse has no DST data
    # at all, so its slot can never absorb a pick. Say so up front rather than
    # letting each team silently overflow on its last pick.
    positions_available = {candidate["position"] for candidate in remaining}
    fillable_within_caps = sum(
        limits[1] for position, limits in ROSTER_POSITION_LIMITS.items() if position in positions_available
    )
    if remaining and rounds > fillable_within_caps:
        warnings.append(
            f"{rounds} rounds exceeds the {fillable_within_caps} roster spots that fit inside "
            f"ROSTER_POSITION_LIMITS for the positions actually available "
            f"({', '.join(sorted(positions_available))}); teams will exceed a position cap on "
            f"{rounds - fillable_within_caps} pick(s) each."
        )

    clusters = identify_adp_clusters(remaining)
    # Local import: fantasy.mock_data_ingestion imports fantasy.draft at module
    # load time (it re-exports identify_adp_clusters/RUN_POSITIONS), so this
    # cannot be a top-level import here without a real cycle. By the time this
    # function actually runs, fantasy.draft has already finished loading.
    from fantasy.mock_data_ingestion import reach_tolerance_scale

    tolerance_scale = reach_tolerance_scale(remaining)

    rng = np.random.default_rng(seed)
    rosters: dict[str, list[dict[str, Any]]] = {f"Team {team}": [] for team in range(1, settings.n_teams + 1)}
    user_team = f"Team {user_draft_slot}" if user_draft_slot else None
    by_round: dict[int, list[dict[str, Any]]] = {}
    picks: list[dict[str, Any]] = []
    cap_emergencies = 0
    cap_relaxations = 0

    for slot in build_draft_order(settings.n_teams, rounds, snake=snake):
        if not remaining:
            break
        team_name = f"Team {slot['team_number']}"
        team_counts = Counter(candidate["position"] for candidate in rosters[team_name])

        eligible = [candidate for candidate in remaining if not roster_cap_reached(team_counts, candidate["position"])]
        if not eligible:
            # Emergency stage 1: every real skill position (QB/RB/WR/TE) is
            # capped for this team -- this is expected, not rare, once the
            # sum of real-position maxes falls short of the round count (no
            # real DST data exists at all, so its "slot" never actually
            # absorbs a pick). Take extra bench depth at a skill position
            # rather than stalling, but K/DST never get a 2nd copy even here --
            # a backup kicker is never realistic roster construction.
            cap_relaxations += 1
            eligible = [candidate for candidate in remaining if candidate["position"] not in NEVER_EXCEED_CAP_POSITIONS]
            if not eligible:
                # Emergency stage 2: truly nothing left but a capped-out K/DST.
                # This should be rare -- tracked and reported below.
                cap_emergencies += 1
                eligible = remaining
        # Room teams choose off the consensus board (ADP order), not our own
        # VORP order -- see fantasy.room_brain.room_candidate_pool for why
        # `eligible[:N]` (VORP-sorted) made high-ADP/negative-VOR players
        # effectively undraftable.
        pool = room_candidate_pool(eligible, CANDIDATE_POOL_SIZE)
        active_run = active_run_position(slot["overall_pick"], clusters)

        if is_round_one_chalk(slot["round"]):
            # Round 1 is market order for every team, room and user alike --
            # there is nothing yet for scarcity/need/risk reasoning to bite on.
            player = round_one_pick(pool[: min(2, len(pool))], rng)
        elif team_name == user_team and user_draft_slot is not None:
            gap = picks_between(settings.n_teams, user_draft_slot, slot["round"], snake=snake)
            picked = user_brain_pick(eligible, settings, rosters[team_name], slot["overall_pick"], gap)
            player = picked if picked is not None else pool[0]
        else:
            player = _room_pick(pool, team_counts, slot["overall_pick"], slot["round"], active_run, tolerance_scale, rng)
        # How many same-position players were still in the pool right before
        # this pick -- a real, honestly-computed "how thin was this position at
        # this moment" scarcity proxy for the round-by-round board.
        remaining_at_position = sum(1 for candidate in remaining if candidate["position"] == player["position"])
        remaining.pop(next(i for i, candidate in enumerate(remaining) if candidate is player))

        rosters[team_name].append(player)
        is_user_pick = team_name == user_team
        projection = draft_projection(player)
        adp = player.get("adp")

        by_round.setdefault(slot["round"], []).append(
            {
                "pick": slot["overall_pick"],
                "round": slot["round"],
                "team": team_name,
                "player_name": player["name"],
                "position": player["position"],
                "projection": projection,
                "is_user_pick": is_user_pick,
                "player_id": player["player_id"],
                "position_run": active_run,
                "adp": adp,
                "vor": player["vor"],
                "remaining_at_position": remaining_at_position,
            }
        )
        picks.append(
            {
                "overall_pick": slot["overall_pick"],
                "round": slot["round"],
                "team": team_name,
                "player_id": player["player_id"],
                "name": player["name"],
                "position": player["position"],
                "vor": player["vor"],
                "projection": projection,
                "adp": adp,
                "remaining_at_position": remaining_at_position,
                "is_user_pick": is_user_pick,
                "position_run": active_run,
            }
        )

    if cap_relaxations:
        warnings.append(
            f"{cap_relaxations} pick(s) exceeded a skill-position cap (RB/WR/TE/QB bench depth) because every "
            "uncapped position was exhausted for that team; K/DST caps were still never exceeded."
        )
    if cap_emergencies:
        warnings.append(
            f"{cap_emergencies} pick(s) had to exceed a roster cap because every remaining real player was "
            "already capped out for that team -- a rare, late-draft emergency, not routine cap enforcement."
        )

    return {
        "by_round": by_round,
        "picks": picks,
        "rosters": rosters,
        "n_teams": settings.n_teams,
        "rounds": rounds,
        "seed": seed,
        "snake": snake,
        "user_team": user_team,
        "user_picks": [pick for pick in picks if pick["is_user_pick"]],
        "cap_emergencies": cap_emergencies,
        "cap_relaxations": cap_relaxations,
        "warnings": warnings,
        "position_runs": clusters,
    }
