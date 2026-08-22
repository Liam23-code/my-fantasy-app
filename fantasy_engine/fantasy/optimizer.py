"""Weekly lineup optimizer and start/sit advice.

Usage::

    from fantasy.optimizer import optimize_lineup, start_sit_advice

    lineup = optimize_lineup(roster, week_projections, league_settings)
    advice = start_sit_advice(roster, week_projections, league_settings)

The optimizer is an integer program: maximize total projected points subject
to exact-fill requirements for dedicated position slots and a combined
FLEX-pool constraint across flex-eligible positions, plus optional per-team
and locked/excluded player constraints. When PuLP is unavailable (or fails to
find a solution for any reason), a greedy heuristic produces a very close --
usually identical, for realistic rosters -- answer instead, so the feature
never hard-depends on a working MILP solver being installed.
"""

from __future__ import annotations

from typing import Any

from fantasy.adapter import normalize_projection
from fantasy.models import LeagueSettings, Roster, RosterPlayer
from fantasy.projections import projected_or_scored

try:
    import pulp

    PULP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via force_greedy in tests
    PULP_AVAILABLE = False

INACTIVE_STATUSES = {"OUT", "IR", "DOUBTFUL"}


def _coerce_roster(roster: Any) -> Roster:
    if isinstance(roster, Roster):
        return roster
    if isinstance(roster, dict):
        return Roster(**roster)
    if isinstance(roster, list):
        return Roster(team_name="", players=[p if isinstance(p, RosterPlayer) else RosterPlayer(**p) for p in roster])
    raise TypeError(f"Cannot coerce {type(roster).__name__} into a Roster")


def _coerce_league_settings(league_settings: Any) -> LeagueSettings:
    if isinstance(league_settings, LeagueSettings):
        return league_settings
    return LeagueSettings(**(league_settings or {}))


def _score_candidates(
    players: list[RosterPlayer],
    week_projections: list[Any],
    settings: LeagueSettings,
) -> dict[str, dict[str, Any]]:
    """Score every rostered player, matching by player_id then by name."""
    normalized = [normalize_projection(p) for p in week_projections]
    by_id = {p["player_id"]: p for p in normalized if p["player_id"]}
    by_name = {p["name"].strip().lower(): p for p in normalized if p["name"]}

    scored: dict[str, dict[str, Any]] = {}
    for player in players:
        canonical = by_id.get(player.player_id) or by_name.get(player.name.strip().lower())
        # Prefers a real projection over the raw stat line (see
        # fantasy.projections.projected_or_scored). `week_projections` is named
        # for what it should hold -- whatever cadence the caller supplies,
        # weekly or season, is what the lineup is optimized on; what must not
        # happen is optimizing on last season's box score while the draft board
        # next to it shows a forward projection.
        points = projected_or_scored(canonical, settings)
        scored[player.player_id] = {
            "player_id": player.player_id,
            "name": player.name,
            "position": player.position,
            "nfl_team": player.nfl_team,
            "injury_status": player.injury_status,
            "points": round(points, 2),
            "has_projection": canonical is not None,
        }
    return scored


def _solve_ilp(
    candidates: list[dict[str, Any]],
    settings: LeagueSettings,
    dedicated: dict[str, int],
    flex_count: int,
    max_players_per_team: int | None,
    locked: set[str],
    excluded: set[str],
) -> set[str] | None:
    """Return the set of started player_ids, or None if no solution was found."""
    problem = pulp.LpProblem("lineup_optimizer", pulp.LpMaximize)
    start_vars = {c["player_id"]: pulp.LpVariable(f"start_{c['player_id']}", cat="Binary") for c in candidates}

    problem += pulp.lpSum(c["points"] * start_vars[c["player_id"]] for c in candidates)

    by_position: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_position.setdefault(candidate["position"], []).append(candidate)

    flex_eligible = set(settings.flex_eligible)
    for position, required in dedicated.items():
        pool = by_position.get(position, [])
        exact = min(required, len(pool))
        if position in flex_eligible:
            problem += pulp.lpSum(start_vars[c["player_id"]] for c in pool) >= exact
        else:
            problem += pulp.lpSum(start_vars[c["player_id"]] for c in pool) == exact

    if flex_eligible:
        flex_pool = [c for pos in flex_eligible for c in by_position.get(pos, [])]
        dedicated_in_flex_positions = sum(min(dedicated.get(pos, 0), len(by_position.get(pos, []))) for pos in flex_eligible)
        target = min(dedicated_in_flex_positions + flex_count, len(flex_pool))
        if flex_pool:
            problem += pulp.lpSum(start_vars[c["player_id"]] for c in flex_pool) == target

    if max_players_per_team:
        by_team: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            if candidate["nfl_team"]:
                by_team.setdefault(candidate["nfl_team"], []).append(candidate)
        for team_players in by_team.values():
            problem += pulp.lpSum(start_vars[c["player_id"]] for c in team_players) <= max_players_per_team

    for player_id in locked:
        if player_id in start_vars:
            problem += start_vars[player_id] == 1
    for player_id in excluded:
        if player_id in start_vars:
            problem += start_vars[player_id] == 0

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None
    return {player_id for player_id, var in start_vars.items() if pulp.value(var) and pulp.value(var) > 0.5}


def _solve_greedy(
    candidates: list[dict[str, Any]],
    settings: LeagueSettings,
    dedicated: dict[str, int],
    flex_count: int,
    max_players_per_team: int | None,
    locked: set[str],
    excluded: set[str],
) -> set[str]:
    """Heuristic fallback: fill dedicated slots best-first, then FLEX best-first.

    Not guaranteed globally optimal under a binding per-team cap (a true
    optimum can require swapping an already-placed player to free up team
    capacity for a better one elsewhere), but it matches the ILP result for
    the common case where the team cap isn't the binding constraint.
    """
    pool = [c for c in candidates if c["player_id"] not in excluded]
    by_position: dict[str, list[dict[str, Any]]] = {}
    for candidate in pool:
        by_position.setdefault(candidate["position"], []).append(candidate)
    for players in by_position.values():
        players.sort(key=lambda c: c["points"], reverse=True)

    started: set[str] = set()
    team_counts: dict[str, int] = {}

    def _team_room(candidate: dict[str, Any]) -> bool:
        team = candidate["nfl_team"]
        if not max_players_per_team or not team:
            return True
        return team_counts.get(team, 0) < max_players_per_team

    def _take(candidate: dict[str, Any]) -> None:
        started.add(candidate["player_id"])
        if candidate["nfl_team"]:
            team_counts[candidate["nfl_team"]] = team_counts.get(candidate["nfl_team"], 0) + 1

    for player_id in locked:
        locked_candidate: dict[str, Any] | None = next((c for c in pool if c["player_id"] == player_id), None)
        if locked_candidate:
            _take(locked_candidate)

    used_ids: set[str] = set(started)
    flex_eligible = list(settings.flex_eligible)
    for position, required in dedicated.items():
        already = sum(1 for c in by_position.get(position, []) if c["player_id"] in used_ids)
        for candidate in by_position.get(position, []):
            if already >= required:
                break
            if candidate["player_id"] in used_ids:
                continue
            if not _team_room(candidate):
                continue
            _take(candidate)
            used_ids.add(candidate["player_id"])
            already += 1

    if flex_eligible:
        flex_pool = sorted(
            (c for pos in flex_eligible for c in by_position.get(pos, []) if c["player_id"] not in used_ids),
            key=lambda c: c["points"],
            reverse=True,
        )
        already_flex = 0
        for candidate in flex_pool:
            if already_flex >= flex_count:
                break
            if not _team_room(candidate):
                continue
            _take(candidate)
            used_ids.add(candidate["player_id"])
            already_flex += 1

    return started


def optimize_lineup(
    roster: Any,
    week_projections: list[Any],
    league_settings: Any,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose the point-maximizing legal starting lineup for one week.

    ``constraints`` supports ``max_players_per_team`` (overrides
    ``league_settings``), ``locked_player_ids`` (forced starters), and
    ``excluded_player_ids`` (forced bench, e.g. known inactives) plus
    ``solver`` (``"auto"`` (default), ``"ilp"``, or ``"greedy"``).
    """
    constraints = constraints or {}
    settings = _coerce_league_settings(league_settings)
    parsed_roster = _coerce_roster(roster)

    locked = set(constraints.get("locked_player_ids", []))
    excluded = set(constraints.get("excluded_player_ids", []))
    max_players_per_team = constraints.get("max_players_per_team", settings.max_players_per_nfl_team)
    solver_choice = constraints.get("solver", "auto")

    # IR/TAXI players are structurally unavailable and dropped entirely. OUT/
    # DOUBTFUL/IR-status players are still scored and can appear on the bench
    # (so start_sit_advice can explain why they're sitting) but are forced
    # out of the solver's selection unless a caller explicitly locks them in.
    eligible_players = [p for p in parsed_roster.players if p.slot not in {"IR", "TAXI"}]
    scored = _score_candidates(eligible_players, week_projections, settings)
    candidates = list(scored.values())
    auto_excluded = {p.player_id for p in eligible_players if p.injury_status in INACTIVE_STATUSES} - locked
    excluded = excluded | auto_excluded

    dedicated = {k: v for k, v in settings.roster_requirements.starting_slots().items() if k != "FLEX"}
    flex_count = settings.roster_requirements.starting_slots().get("FLEX", 0)

    started_ids: set[str] | None = None
    solver_used = "greedy"
    if solver_choice in {"auto", "ilp"} and PULP_AVAILABLE and candidates:
        try:
            started_ids = _solve_ilp(candidates, settings, dedicated, flex_count, max_players_per_team, locked, excluded)
            if started_ids is not None:
                solver_used = "ilp"
        except Exception:
            started_ids = None
    if started_ids is None:
        if solver_choice == "ilp":
            raise RuntimeError("ILP solver unavailable or infeasible and solver='ilp' forbids the greedy fallback.")
        started_ids = _solve_greedy(candidates, settings, dedicated, flex_count, max_players_per_team, locked, excluded)
        solver_used = "greedy"

    starters, bench = [], []
    flex_positions = set(settings.flex_eligible)
    dedicated_filled: dict[str, int] = {}
    for candidate in sorted(candidates, key=lambda c: c["points"], reverse=True):
        if candidate["player_id"] not in started_ids:
            bench.append({**candidate, "slot": "BENCH"})
            continue
        position = candidate["position"]
        if position in flex_positions and dedicated_filled.get(position, 0) >= dedicated.get(position, 0):
            slot = "FLEX"
        else:
            slot = position
            dedicated_filled[position] = dedicated_filled.get(position, 0) + 1
        starters.append({**candidate, "slot": slot})

    unfilled_slots = [
        position for position, required in dedicated.items() if sum(1 for c in starters if c["position"] == position and c["slot"] == position) < required
    ]

    return {
        "starters": starters,
        "bench": bench,
        "total_points": round(sum(c["points"] for c in starters), 2),
        "solver": solver_used,
        "unfilled_slots": unfilled_slots,
        "warnings": [f"Could not fill all {position} slots; not enough eligible players." for position in unfilled_slots],
    }


def start_sit_advice(
    roster: Any,
    week_projections: list[Any],
    league_settings: Any | None = None,
) -> list[dict[str, Any]]:
    """Per-player start/bench verdicts with the point swing and a reason."""
    settings = _coerce_league_settings(league_settings)
    lineup = optimize_lineup(roster, week_projections, settings)
    flex_eligible = set(settings.flex_eligible)

    def _bench_alternatives(starter: dict[str, Any]) -> list[dict[str, Any]]:
        # A starter parked in the FLEX slot competes with any flex-eligible
        # bench player, not just ones at its own exact position.
        if starter["slot"] == "FLEX":
            return [c for c in lineup["bench"] if c["position"] in flex_eligible]
        return [c for c in lineup["bench"] if c["position"] == starter["position"]]

    def _starter_mates(bench_player: dict[str, Any]) -> list[dict[str, Any]]:
        mates = {c["player_id"]: c for c in lineup["starters"] if c["position"] == bench_player["position"]}
        if bench_player["position"] in flex_eligible:
            mates.update({c["player_id"]: c for c in lineup["starters"] if c["slot"] == "FLEX"})
        return list(mates.values())

    advice: list[dict[str, Any]] = []
    for player in lineup["starters"]:
        bench_alternatives = _bench_alternatives(player)
        if bench_alternatives:
            best_alt = max(bench_alternatives, key=lambda c: c["points"])
            delta = round(player["points"] - best_alt["points"], 2)
            reason = f"Best {player['slot']} option; +{delta:.1f} pts over next best bench alternative ({best_alt['name']})"
        else:
            delta = round(player["points"], 2)
            reason = f"Locked starter at {player['slot']}; no bench alternative available."
        advice.append({"player": player["name"], "position": player["position"], "start_or_bench": "start", "delta_points": delta, "reason": reason})

    for player in lineup["bench"]:
        if player.get("injury_status") in INACTIVE_STATUSES:
            advice.append(
                {
                    "player": player["name"],
                    "position": player["position"],
                    "start_or_bench": "bench",
                    "delta_points": 0.0,
                    "reason": f"Ruled {player['injury_status']}; must bench regardless of projection.",
                }
            )
            continue
        starter_mates = _starter_mates(player)
        if starter_mates:
            weakest = min(starter_mates, key=lambda c: c["points"])
            delta = round(player["points"] - weakest["points"], 2)
            if delta > 0:
                reason = f"Would net +{delta:.1f} pts swapping in for {weakest['name']} ({weakest['slot']})"
            else:
                reason = f"Correctly benched: {delta:.1f} pts below {weakest['name']} ({weakest['slot']})"
        else:
            delta = 0.0
            reason = "No eligible starting slot for this position this week."
        advice.append({"player": player["name"], "position": player["position"], "start_or_bench": "bench", "delta_points": delta, "reason": reason})

    return advice
