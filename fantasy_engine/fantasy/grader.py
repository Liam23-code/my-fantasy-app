"""grader: score a drafted team, live during the draft and again at the end.

Usage::

    from fantasy.grader import grade_team, grade_position_group, grade_overall_team

    report = grade_team(my_roster, board, league_settings, picks=my_picks,
                        all_rosters=state["rosters"])
    report["overall"]["score"]          # 0-100
    report["overall"]["grade"]          # "B+"
    report["positions"][0]              # {"position": "RB", "score": 78.4, ...}
    report["best_pick"], report["worst_pick"]

Everything here is measured against the league the team is actually in, not
against an absolute scale. "Good at RB" has no meaning on its own -- 380 RB
points is a league-winning room in one league and a below-average one in
another. So every positional score is an edge over *this* league's average
team at *this* position, computed from the player pool itself:

    the average team's Nth starter at a position
        = the mean of the players ranked [N*n_teams, (N+1)*n_teams) there

With 12 teams, the average RB1 is the mean of RBs 1-12, the average RB2 the
mean of RBs 13-24, and a team's RB grade is how far its own top two sit above
or below that sum. No hard-coded "elite RB = 300 points" anywhere.

Three signals move a positional score (see :func:`grade_position_group`):
projected points versus that league average, positional scarcity
(:data:`fantasy.assistant.POSITION_SCARCITY_BIAS` -- the same bias the draft
assistant ranks with, so the grader and the recommender agree about which
positions are hard to replace), and ADP value actually captured at that
position. The overall grade (:func:`grade_overall_team`) then weighs projected
points, positional balance, ADP value, and risk.

Scores are bounded 0-100 through ``tanh``, deliberately: a linear scale lets
one absurd position swing an overall grade to 0 or 100, and a team that is
merely *very* good should not be indistinguishable from a perfect one.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from fantasy import player_status as _status
from fantasy.assistant import POSITION_SCARCITY_BIAS, replacement_levels
from fantasy.models import LeagueSettings
from fantasy.projections import projected_points
from fantasy.scoring import calculate_fantasy_points
from fantasy.utils import clamp, safe_float

#: How the four overall components combine. Projected points dominates
#: because it is the only component measuring the thing that actually wins a
#: fantasy season; the other three describe *how safely* those points were
#: acquired and how well they are distributed across startable slots.
OVERALL_WEIGHTS: dict[str, float] = {
    "projected_points": 0.40,
    "positional_balance": 0.25,
    "value_vs_adp": 0.20,
    "risk_profile": 0.15,
}

#: Fraction of the league-average total treated as one "unit" of edge. A team
#: 35% above the average starter total at a position scores ~88 there; one at
#: the average scores exactly 50.
GRADE_SPREAD_FRACTION = 0.35

#: How much each signal moves a positional score inside the tanh.
POINTS_EDGE_WEIGHT = 0.85
ADP_EDGE_WEIGHT = 0.30

#: Rounds of cumulative ADP value that count as one full unit of edge. Two
#: rounds of value across a position group is a genuinely well-timed draft.
ADP_VALUE_ROUNDS_SCALE = 2.0

#: Points removed from the risk score per rostered player by injury status.
RISK_INJURY_PENALTY: dict[str, float] = {
    "OUT": 14.0,
    "IR": 18.0,
    "DOUBTFUL": 9.0,
    "QUESTIONABLE": 4.0,
}

#: Ceiling-minus-floor spread, relative to projection, treated as normal
#: week-to-week variance. Rosters above this are boom/bust and lose up to
#: :data:`RISK_VOLATILITY_PENALTY` points.
RISK_VOLATILITY_REFERENCE = 0.80
RISK_VOLATILITY_PENALTY = 10.0

#: Points removed per starting slot the roster cannot fill at all.
RISK_UNFILLED_SLOT_PENALTY = 12.0

#: How deep into the draft the points-per-pick slope is measured, in rounds.
#: Used to convert "picks of ADP value" into points for best/worst pick.
VALUE_CURVE_ROUNDS = 8

#: Letter bands anchored to what the 0-100 scores here actually *mean*: every
#: component is centred so that a genuinely league-average team lands near 50
#: (see :func:`_bounded_score` and :func:`_risk_profile_score`). An average
#: team is a C, not an F; a team a full :data:`GRADE_SPREAD_FRACTION` above its
#: league across the board is a B; two of those, an A.
_LETTER_GRADES: tuple[tuple[float, str], ...] = (
    (90.0, "A+"),
    (83.0, "A"),
    (78.0, "A-"),
    (72.0, "B+"),
    (65.0, "B"),
    (59.0, "B-"),
    (53.0, "C+"),
    (46.0, "C"),
    (41.0, "C-"),
    (35.0, "D+"),
    (30.0, "D"),
    (25.0, "D-"),
)


def letter_grade(score: float) -> str:
    """Map a 0-100 score onto a letter grade, anchored so ~50 is a C."""
    for threshold, letter in _LETTER_GRADES:
        if score >= threshold:
            return letter
    return "F"


def _bounded_score(edge: float) -> float:
    """Map an unbounded edge onto 0-100, with 0 edge landing exactly at 50."""
    return round(50.0 + 50.0 * math.tanh(edge), 1)


def _coerce_settings(scoring_model: Any) -> LeagueSettings:
    """Accept league settings, a settings dict, or a bare scoring-mode string."""
    if isinstance(scoring_model, LeagueSettings):
        return scoring_model
    if isinstance(scoring_model, Mapping):
        return LeagueSettings(**dict(scoring_model))
    if isinstance(scoring_model, str) and scoring_model.strip():
        return LeagueSettings(scoring_mode=scoring_model.strip().lower())
    return LeagueSettings()


def _position_of(player: Mapping[str, Any]) -> str:
    return str((player or {}).get("position", "")).strip().upper()


def _points_of(player: Mapping[str, Any], settings: LeagueSettings) -> float:
    """Projected points, preferring a real forward projection over raw stats.

    Mirrors :func:`fantasy.assistant.suggest_draft_picks`'s own preference
    order, so a grade and a recommendation are always talking about the same
    number for the same player.
    """
    precomputed = projected_points(player, settings)
    if precomputed is not None:
        return precomputed
    scored = calculate_fantasy_points(dict(player), mode=settings.scoring_mode, custom_rules=settings.custom_rules)
    return float(scored["total_points"])


def _identity(player: Mapping[str, Any]) -> str:
    return str((player or {}).get("player_id") or (player or {}).get("id") or "")


def build_universe(
    roster: list[dict[str, Any]] | None,
    board: list[dict[str, Any]] | None,
    all_rosters: dict[str, list[dict[str, Any]]] | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Every player the league average should be measured over, de-duplicated.

    Passing ``all_rosters`` (e.g. :mod:`fantasy.live_draft`'s ``state["rosters"]``)
    makes the average a *real* league average over drafted teams. Without it,
    the universe is your roster plus whoever is still on the board, which
    still works but understates the league mid-draft, since the other teams'
    picks have left the board without joining the universe.
    """
    universe: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(players: Any) -> None:
        for player in players or []:
            if not isinstance(player, dict):
                continue
            key = _identity(player) or f"anon:{id(player)}"
            if key in seen:
                continue
            seen.add(key)
            universe.append(player)

    _add(roster)
    if isinstance(all_rosters, Mapping):
        for team_players in all_rosters.values():
            _add(team_players)
    else:
        _add(all_rosters)
    _add(board)
    return universe


def _dedicated_slots(settings: LeagueSettings) -> dict[str, int]:
    slots = settings.roster_requirements.starting_slots()
    slots.pop("FLEX", None)
    return {position: int(count) for position, count in slots.items() if count > 0}


def _flex_slots(settings: LeagueSettings) -> int:
    return int(settings.roster_requirements.starting_slots().get("FLEX", 0))


def _flex_eligible(settings: LeagueSettings) -> list[str]:
    return [str(position).strip().upper() for position in settings.flex_eligible]


def _unfilled_starting_slots(roster: list[dict[str, Any]] | None, settings: LeagueSettings) -> int:
    """How many starting-lineup slots (dedicated + FLEX) the roster has not filled.

    Drives the mid-draft grading accommodations: they stay on while any slot is
    still open -- K and DST included -- and switch off one slot at a time as
    those slots fill, so the grade never lurches when the roster count happens
    to cross the starting-lineup size while streamer slots are still empty.
    """
    counts: dict[str, int] = {}
    for player in roster or []:
        position = _position_of(player)
        if position:
            counts[position] = counts.get(position, 0) + 1

    eligible = set(_flex_eligible(settings))
    unfilled = 0
    flex_surplus = 0
    for position, required in _dedicated_slots(settings).items():
        have = counts.get(position, 0)
        unfilled += max(0, required - have)
        if position in eligible:
            flex_surplus += max(0, have - required)
    unfilled += max(0, _flex_slots(settings) - flex_surplus)
    return unfilled


def _sorted_points(
    players: list[dict[str, Any]],
    settings: LeagueSettings,
    positions: list[str] | str,
) -> list[float]:
    wanted = {positions} if isinstance(positions, str) else set(positions)
    return sorted(
        (_points_of(player, settings) for player in players if _position_of(player) in wanted),
        reverse=True,
    )


def _average_team_block(ordered: list[float], index: int, n_teams: int) -> float:
    """Mean of the ``index``-th block of ``n_teams`` players -- the average team's Nth starter."""
    if not ordered:
        return 0.0
    block = ordered[index * n_teams : (index + 1) * n_teams]
    if not block:
        # The league is thinner at this position than its own starting
        # requirement; the worst available player is the honest floor.
        return ordered[-1]
    return sum(block) / len(block)


def league_average_starters(
    position: str,
    universe: list[dict[str, Any]],
    scoring_model: Any = None,
    *,
    slots_override: int | None = None,
) -> float:
    """Projected points the *average* team in this league starts at ``position``.

    ``position`` may be ``"FLEX"``, in which case the flex-eligible pool is
    measured after every dedicated starting slot at those positions has been
    filled league-wide -- the same accounting
    :func:`fantasy.draft._replacement_levels` uses for replacement level.

    ``slots_override`` measures the average over only the top ``N`` starters
    instead of the league's full requirement -- used mid-draft to compare a
    half-filled group against a like-sized slice of the average team rather
    than against slots the roster has not reached yet.
    """
    settings = _coerce_settings(scoring_model)
    position = str(position).strip().upper()
    n_teams = max(1, int(settings.n_teams))

    if position == "FLEX":
        flex_slots = _flex_slots(settings) if slots_override is None else int(slots_override)
        if flex_slots <= 0:
            return 0.0
        dedicated = _dedicated_slots(settings)
        eligible = _flex_eligible(settings)
        pool: list[float] = []
        for eligible_position in eligible:
            ranked = _sorted_points(universe, settings, eligible_position)
            pool.extend(ranked[dedicated.get(eligible_position, 0) * n_teams :])
        pool.sort(reverse=True)
        return round(sum(_average_team_block(pool, k, n_teams) for k in range(flex_slots)), 2)

    slots = _dedicated_slots(settings).get(position, 0) if slots_override is None else int(slots_override)
    if slots <= 0:
        return 0.0
    ordered = _sorted_points(universe, settings, position)
    return round(sum(_average_team_block(ordered, k, n_teams) for k in range(slots)), 2)


def _my_starters(
    position: str,
    roster: list[dict[str, Any]] | None,
    settings: LeagueSettings,
) -> list[dict[str, Any]]:
    """The players on ``roster`` who actually occupy this position's starting slots."""
    position = str(position).strip().upper()
    players = list(roster or [])

    if position == "FLEX":
        flex_slots = _flex_slots(settings)
        if flex_slots <= 0:
            return []
        dedicated = _dedicated_slots(settings)
        leftovers: list[dict[str, Any]] = []
        for eligible_position in _flex_eligible(settings):
            same = sorted(
                (p for p in players if _position_of(p) == eligible_position),
                key=lambda p: _points_of(p, settings),
                reverse=True,
            )
            leftovers.extend(same[dedicated.get(eligible_position, 0) :])
        leftovers.sort(key=lambda p: _points_of(p, settings), reverse=True)
        return leftovers[:flex_slots]

    slots = _dedicated_slots(settings).get(position, 0)
    if slots <= 0:
        return []
    same = sorted(
        (p for p in players if _position_of(p) == position),
        key=lambda p: _points_of(p, settings),
        reverse=True,
    )
    return same[:slots]


def _adp_value_picks(picks: list[dict[str, Any]] | None, positions: set[str] | None = None) -> float:
    """Cumulative picks of ADP value: positive means taken later than the market expected."""
    total = 0.0
    for pick in picks or []:
        if not isinstance(pick, dict):
            continue
        if positions is not None and str(pick.get("position", "")).strip().upper() not in positions:
            continue
        adp = pick.get("adp")
        overall = pick.get("overall_pick") or pick.get("pick")
        if adp is None or overall is None:
            continue
        total += safe_float(overall) - safe_float(adp)
    return round(total, 1)


def grade_position_group(
    position: str,
    roster: list[dict[str, Any]] | None,
    board: list[dict[str, Any]] | None,
    scoring_model: Any = None,
    picks: list[dict[str, Any]] | None = None,
    all_rosters: dict[str, list[dict[str, Any]]] | None = None,
    universe: list[dict[str, Any]] | None = None,
    *,
    filled_slot_basis: bool = False,
) -> dict[str, Any]:
    """Grade one position group 0-100 against this league's average team.

    Three real signals move the score, exactly as the module docstring
    describes:

    * **Projected points** -- the roster's starters at this position versus
      :func:`league_average_starters`, standardized by
      :data:`GRADE_SPREAD_FRACTION` of that average.
    * **Scarcity** -- the edge is multiplied by
      :data:`fantasy.assistant.POSITION_SCARCITY_BIAS`, so being a hundred
      points clear at TE (a thin position, hard to fix later) grades higher
      than the same margin at WR (deep, fixable on waivers).
    * **ADP value** -- cumulative picks of value captured at this position,
      scaled by :data:`ADP_VALUE_ROUNDS_SCALE`. Requires ``picks``; without it
      this term is simply zero rather than guessed.

    ``position`` accepts ``"FLEX"`` alongside the real positions. Returns a
    dict with the score, letter grade, both point totals, the starters it
    graded, and a plain-language rationale. A position the league does not
    start at all scores 50 (neutral) with ``"applicable": False``.
    """
    settings = _coerce_settings(scoring_model)
    position = str(position).strip().upper()
    pool = universe if universe is not None else build_universe(roster, board, all_rosters)

    slots = _flex_slots(settings) if position == "FLEX" else _dedicated_slots(settings).get(position, 0)
    depth_positions = set(_flex_eligible(settings)) if position == "FLEX" else {position}
    depth = sum(1 for player in (roster or []) if _position_of(player) in depth_positions)

    if slots <= 0:
        return {
            "position": position,
            "applicable": False,
            "score": 50.0,
            "grade": letter_grade(50.0),
            "starter_slots": 0,
            "starters": [],
            "starter_points": 0.0,
            "league_average_points": 0.0,
            "points_vs_average": 0.0,
            "depth": depth,
            "scarcity_weight": round(POSITION_SCARCITY_BIAS.get(position, 1.0), 2),
            "adp_value_picks": 0.0,
            "rationale": f"{position} is not a starting slot in this league.",
        }

    starters = _my_starters(position, roster, settings)
    starter_points = round(sum(_points_of(player, settings) for player in starters), 2)
    # Mid-draft, a group with some -- but not all -- of its starting slots
    # filled is measured against a like-sized slice of the average team, so a
    # one-of-two WR room is not scored as if its WR2 slot were a zero. A group
    # with *no* starters yet (K/DST before the streamer rounds) is a neutral 50
    # -- "still to draft", not a failure -- so it never drags the live panel's
    # positional chart or its weakest-group callout.
    if filled_slot_basis and not starters:
        average_points = league_average_starters(position, pool, settings)
        return {
            "position": position,
            "applicable": True,
            "score": 50.0,
            "grade": letter_grade(50.0),
            "starter_slots": slots,
            "starters": [],
            "starter_points": 0.0,
            "league_average_points": average_points,
            "points_vs_average": round(0.0 - average_points, 2),
            "depth": depth,
            "unfilled_slots": slots,
            "scarcity_weight": round(POSITION_SCARCITY_BIAS.get(position, 1.0), 2),
            "adp_value_picks": 0.0,
            "rationale": f"0/{slots} starting {position} slot(s) filled -- still to draft.",
        }
    basis_slots = len(starters) if (filled_slot_basis and 0 < len(starters) < slots) else None
    average_points = league_average_starters(position, pool, settings, slots_override=basis_slots)
    delta = round(starter_points - average_points, 2)

    spread = max(1.0, GRADE_SPREAD_FRACTION * average_points)
    scarcity_weight = POSITION_SCARCITY_BIAS.get(position, 1.0)
    points_edge = (delta / spread) * scarcity_weight

    positions_for_adp = depth_positions if position == "FLEX" else {position}
    adp_value = _adp_value_picks(picks, positions_for_adp) if picks else 0.0
    adp_edge = adp_value / max(1.0, settings.n_teams * ADP_VALUE_ROUNDS_SCALE)

    score = _bounded_score(POINTS_EDGE_WEIGHT * points_edge + ADP_EDGE_WEIGHT * adp_edge)

    missing = max(0, slots - len(starters))
    parts = [
        f"{len(starters)}/{slots} starting {position} slot(s) filled",
        f"{starter_points:,.0f} pts vs a {average_points:,.0f}-pt league average ({delta:+,.0f})",
    ]
    if scarcity_weight != 1.0:
        thin = "scarce" if scarcity_weight > 1.0 else "deep"
        parts.append(f"{position} is a {thin} position (x{scarcity_weight:.2f})")
    if adp_value:
        direction = "gained" if adp_value > 0 else "paid"
        parts.append(f"{abs(adp_value):.0f} picks of ADP value {direction} here")
    if missing:
        parts.append(f"{missing} slot(s) still empty")
    if depth > slots:
        parts.append(f"{depth - slots} of bench depth behind them")

    return {
        "position": position,
        "applicable": True,
        "score": score,
        "grade": letter_grade(score),
        "starter_slots": slots,
        "starters": [
            {
                "player_id": player.get("player_id") or player.get("id"),
                "name": player.get("name"),
                "position": _position_of(player),
                "team": player.get("team", ""),
                "projection": round(_points_of(player, settings), 2),
                "adp": round(safe_float(player["adp"]), 1) if player.get("adp") is not None else None,
            }
            for player in starters
        ],
        "starter_points": starter_points,
        "league_average_points": average_points,
        "points_vs_average": delta,
        "depth": depth,
        "unfilled_slots": missing,
        "scarcity_weight": round(scarcity_weight, 2),
        "adp_value_picks": adp_value,
        "rationale": "; ".join(parts),
    }


def _graded_positions(settings: LeagueSettings) -> list[str]:
    """Every group worth grading, dedicated positions first and FLEX last."""
    positions = list(_dedicated_slots(settings))
    positions.sort(key=lambda p: (p not in {"QB", "RB", "WR", "TE"}, p))
    if _flex_slots(settings) > 0:
        positions.append("FLEX")
    return positions


def _volatility_of(player: Mapping[str, Any], projection: float) -> float:
    floor, ceiling = player.get("floor"), player.get("ceiling")
    if floor is None or ceiling is None or projection <= 0:
        return 0.0
    return max(0.0, (safe_float(ceiling) - safe_float(floor)) / projection)


def _risk_profile_score(
    roster: list[dict[str, Any]] | None,
    position_grades: list[dict[str, Any]],
    settings: LeagueSettings,
    *,
    suppress_unfilled: bool = False,
) -> tuple[float, list[str]]:
    """0-100 risk score, centred on 50 = ordinary risk.

    A clean roster -- no injury designations, normal variance, every slot it
    is *expected* to have filled -- sits at 50, the same neutral point the
    other three components use, rather than a lone 100 that dragged the whole
    scale off centre. Injuries, boom/bust variance and (unless
    ``suppress_unfilled``) unfilled starting slots push it down from there.
    ``suppress_unfilled`` is set mid-draft, when empty slots are draft
    progress rather than a roster hole.
    """
    players = list(roster or [])
    score = 50.0
    notes: list[str] = []

    injured = 0
    for player in players:
        status = str(player.get("injury_status") or "").strip().upper()
        penalty = RISK_INJURY_PENALTY.get(status, 0.0)
        if penalty:
            injured += 1
            score -= penalty
    if injured:
        notes.append(f"{injured} rostered player(s) carry an injury designation")

    volatilities = [
        _volatility_of(player, _points_of(player, settings)) for player in players if player.get("floor") is not None and player.get("ceiling") is not None
    ]
    if volatilities:
        mean_volatility = statistics.mean(volatilities)
        excess = clamp((mean_volatility - RISK_VOLATILITY_REFERENCE) / RISK_VOLATILITY_REFERENCE, 0.0, 1.0)
        if excess > 0:
            score -= RISK_VOLATILITY_PENALTY * excess
            notes.append(f"boom/bust roster (mean ceiling-floor spread {mean_volatility:.2f}x projection)")

    unfilled = sum(int(group.get("unfilled_slots", 0) or 0) for group in position_grades)
    if unfilled and not suppress_unfilled:
        score -= RISK_UNFILLED_SLOT_PENALTY * unfilled
        notes.append(f"{unfilled} starting slot(s) unfilled")
    elif unfilled:
        notes.append(f"{unfilled} starting slot(s) still to draft")

    if not notes:
        notes.append("no injury designations, normal variance, every starting slot filled")
    return round(clamp(score, 0.0, 100.0), 1), notes


def grade_overall_team(
    roster: list[dict[str, Any]] | None,
    board: list[dict[str, Any]] | None,
    scoring_model: Any = None,
    position_grades: list[dict[str, Any]] | None = None,
    picks: list[dict[str, Any]] | None = None,
    all_rosters: dict[str, list[dict[str, Any]]] | None = None,
    universe: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Overall team strength 0-100, from four separately-reported components.

    Weighted by :data:`OVERALL_WEIGHTS`:

    ``projected_points`` (0.40)
        Total starting-lineup projection versus the league's average starting
        lineup, standardized the same way a positional grade is.
    ``positional_balance`` (0.25)
        Half the mean positional score, half the *worst* one -- so a roster
        with one gaping hole cannot hide behind three strong groups, which is
        exactly how a real fantasy season punishes it.
    ``value_vs_adp`` (0.20)
        Cumulative picks of ADP value captured across the whole draft.
    ``risk_profile`` (0.15)
        Injury designations, boom/bust variance, and unfilled starting slots.

    ``position_grades`` is computed when not supplied. Returns the score, its
    letter grade, every component, and the weights used -- nothing is hidden
    inside a single opaque number.
    """
    settings = _coerce_settings(scoring_model)
    pool = universe if universe is not None else build_universe(roster, board, all_rosters)

    # Mid-draft a roster is half-built by construction: most starting slots are
    # still empty. Grading it against a *full* league starting lineup buries the
    # score at F for most of the draft -- noise, not signal. So whenever a live
    # draft is in progress (``picks`` supplied) and any starting slot is still
    # open (K and DST included), score only the slots actually filled and treat
    # the empty ones as draft progress. Keyed on unfilled slots, not roster
    # count, so the grade never lurches when the count crosses the starting-
    # lineup size while streamer slots are still to draft. A complete lineup, or
    # a direct call with no ``picks`` at all, is graded exactly as before.
    mid_draft = picks is not None and _unfilled_starting_slots(roster, settings) > 0

    groups = (
        position_grades
        if position_grades is not None
        else [
            grade_position_group(
                position, roster, board, settings, picks=picks, universe=pool, filled_slot_basis=mid_draft
            )
            for position in _graded_positions(settings)
        ]
    )
    applicable = [group for group in groups if group.get("applicable")]
    scored_groups = [group for group in applicable if group.get("starters")] if mid_draft else applicable

    starter_points = round(sum(group["starter_points"] for group in applicable), 2)
    average_points = round(sum(group["league_average_points"] for group in applicable), 2)
    basis_starter_points = round(sum(group["starter_points"] for group in scored_groups), 2)
    basis_average_points = round(sum(group["league_average_points"] for group in scored_groups), 2)
    spread = max(1.0, GRADE_SPREAD_FRACTION * basis_average_points)
    points_score = _bounded_score((basis_starter_points - basis_average_points) / spread)

    # Report the point totals / delta / rationale over the same basis the score
    # uses, so a fine mid-draft team never shows a "C" beside a caption claiming
    # it is hundreds of points below average (that gap is almost entirely the
    # streamer slots it has not reached yet).
    reported_average_points = basis_average_points if mid_draft else average_points
    reported_delta = round(basis_starter_points - basis_average_points, 2) if mid_draft else round(starter_points - average_points, 2)

    balance_groups = scored_groups if mid_draft else applicable
    if balance_groups:
        scores = [group["score"] for group in balance_groups]
        balance_score = round(0.5 * statistics.mean(scores) + 0.5 * min(scores), 1)
    else:
        balance_score = 50.0

    total_adp_value = _adp_value_picks(picks) if picks else 0.0
    value_score = _bounded_score(total_adp_value / max(1.0, settings.n_teams * ADP_VALUE_ROUNDS_SCALE))

    risk_score, risk_notes = _risk_profile_score(roster, groups, settings, suppress_unfilled=mid_draft)

    components = {
        "projected_points": points_score,
        "positional_balance": balance_score,
        "value_vs_adp": value_score,
        "risk_profile": risk_score,
    }
    score = round(sum(OVERALL_WEIGHTS[key] * value for key, value in components.items()), 1)

    return {
        "score": score,
        "grade": letter_grade(score),
        "components": components,
        "weights": dict(OVERALL_WEIGHTS),
        "starter_points": starter_points,
        "league_average_points": reported_average_points,
        "points_vs_average": reported_delta,
        "adp_value_picks": total_adp_value,
        "risk_notes": risk_notes,
        "rationale": (
            f"{basis_starter_points:,.0f} projected starting points vs a {reported_average_points:,.0f}-pt league "
            f"average ({reported_delta:+,.0f}){' (filled slots)' if mid_draft else ''}; balance {balance_score:.0f}/100, "
            f"ADP value {total_adp_value:+.0f} picks, risk {risk_score:.0f}/100"
        ),
    }


def _points_per_pick(universe: list[dict[str, Any]], settings: LeagueSettings) -> float:
    """Average projected points a draft slot is worth, over the early rounds.

    Measured off the pool's own value curve -- best player to the player
    :data:`VALUE_CURVE_ROUNDS` rounds down the board -- rather than assumed, so
    "one pick of ADP value" converts into points on the scale this particular
    league and pool actually produce. A 12-team league with a deep pool prices
    a pick at a few points; a shallow one prices it much higher, correctly.

    Ranked by projection rather than by ADP on purpose. The question being
    priced is "what is one rank of board position worth", and the value curve
    answers it directly; ordering by ADP would fold the market's own
    mispricings into a number whose entire job is to convert *away* from
    market picks and into points.
    """
    curve = sorted((_points_of(player, settings) for player in universe), reverse=True)
    horizon = min(len(curve) - 1, int(settings.n_teams) * VALUE_CURVE_ROUNDS)
    if horizon <= 0:
        return 0.0
    return max(0.0, (curve[0] - curve[horizon]) / horizon)


def _pick_report(
    picks: list[dict[str, Any]] | None,
    roster: list[dict[str, Any]] | None,
    universe: list[dict[str, Any]],
    settings: LeagueSettings,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Per-pick value, plus the best and worst pick of the draft.

    A pick is scored as its points over positional replacement plus the ADP
    value it captured, converted to points at the pool's own measured
    :func:`_points_per_pick` slope. When no ``picks`` are supplied there are no
    pick numbers to measure timing against, so the ranking falls back to pure
    value over replacement.
    """
    levels = replacement_levels(universe, settings)
    slope = _points_per_pick(universe, settings)
    by_id = {_identity(player): player for player in (roster or []) if _identity(player)}

    entries: list[dict[str, Any]] = []
    source = picks if picks else [dict(player, overall_pick=None) for player in (roster or [])]
    for pick in source:
        if not isinstance(pick, dict):
            continue
        identity = _identity(pick)
        player = by_id.get(identity, pick)
        position = _position_of(pick) or _position_of(player)
        projection = _points_of(player, settings)
        vorp = projection - levels.get(position, 0.0)

        adp = pick.get("adp") if pick.get("adp") is not None else player.get("adp")
        overall = pick.get("overall_pick") or pick.get("pick")
        value_picks = round(safe_float(overall) - safe_float(adp), 1) if adp is not None and overall is not None else None
        value_points = round(slope * value_picks, 2) if value_picks is not None else 0.0

        entries.append(
            {
                "player_id": pick.get("player_id") or player.get("player_id"),
                "name": pick.get("name") or player.get("name"),
                "position": position,
                "round": pick.get("round"),
                "overall_pick": overall,
                "adp": round(safe_float(adp), 1) if adp is not None else None,
                "projection": round(projection, 2),
                "vorp": round(vorp, 2),
                "value_picks": value_picks,
                "value_points": value_points,
                "pick_score": round(vorp + value_points, 2),
            }
        )

    if not entries:
        return [], None, None
    ordered = sorted(entries, key=lambda entry: entry["pick_score"], reverse=True)
    return entries, ordered[0], ordered[-1]


def grade_team(
    roster: list[dict[str, Any]] | None,
    board: list[dict[str, Any]] | None,
    scoring_model: Any = None,
    picks: list[dict[str, Any]] | None = None,
    all_rosters: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """The full team grade -- every positional group, the overall score, and the picks.

    Safe to call after every single pick: it is pure computation over the
    state you already hold, holds no state of its own, and returns a complete
    report whether the roster has one player or fifteen. That is what makes it
    usable both as a live panel during the draft and as the final report after
    it.

    ``scoring_model`` accepts a :class:`fantasy.models.LeagueSettings`, the
    plain settings dict a UI keeps in session state, or a bare scoring-mode
    string (``"ppr"``) when nothing richer is available.

    ``picks`` should be the user's own pick records (:mod:`fantasy.live_draft`'s
    ``state["picks"]`` filtered to ``is_user_pick``), which is what supplies
    pick numbers for the ADP-value and best/worst-pick sections. ``all_rosters``
    should be ``state["rosters"]`` so the league average is measured over the
    real league rather than the board alone.

    Returns ``{"overall", "positions", "best_pick", "worst_pick",
    "value_vs_adp", "picks", "roster_size", "graded_at_pick"}``. An empty
    roster is a normal state, not an error: with ``picks`` supplied (a live
    draft that has not started) it grades a neutral ~50; with ``picks=None`` it
    grades out at the league average's floor with every slot flagged unfilled.
    """
    settings = _coerce_settings(scoring_model)

    # Live player-status overlay (no-op until the status file is refreshed).
    # An OUT / HOLDOUT / SUSPENDED player cannot be started, so the room
    # ignores them entirely; DOUBTFUL / QUESTIONABLE stay on the roster with a
    # canonical ``injury_status`` and lift the risk component through the
    # existing RISK_INJURY_PENALTY.
    roster_size = len(roster or [])
    if _status.has_status_data():
        roster = _status.overlay_pool_status(
            [player for player in (roster or []) if not _status.is_unavailable(player)]
        )
        board = _status.overlay_pool_status([player for player in (board or []) if not _status.is_out(player)])

    universe = build_universe(roster, board, all_rosters)

    # The live panel calls this after every pick; while a draft is in progress
    # (``picks`` supplied) and any starting slot is still open (K and DST
    # included) the roster is graded on the slots it *has* filled, so the score
    # climbs with the draft instead of sitting at F until it is over. Keyed on
    # unfilled slots, not roster count, so it never lurches when the count
    # crosses the starting-lineup size. Before the first pick this makes an
    # empty live roster read as a neutral ~50/C rather than F; a direct call
    # with ``picks=None`` keeps the legacy full-lineup grading.
    mid_draft = picks is not None and _unfilled_starting_slots(roster, settings) > 0

    positions = [
        grade_position_group(
            position, roster, board, settings, picks=picks, universe=universe, filled_slot_basis=mid_draft
        )
        for position in _graded_positions(settings)
    ]
    overall = grade_overall_team(
        roster,
        board,
        settings,
        position_grades=positions,
        picks=picks,
        universe=universe,
    )
    pick_entries, best_pick, worst_pick = _pick_report(picks, roster, universe, settings)

    total_value_picks = _adp_value_picks(picks) if picks else 0.0
    total_value_points = round(sum(entry["value_points"] for entry in pick_entries), 2)

    return {
        "overall": overall,
        "positions": positions,
        "best_pick": best_pick,
        "worst_pick": worst_pick,
        "value_vs_adp": {
            "total_picks": total_value_picks,
            "total_points": total_value_points,
            "picks_with_adp": sum(1 for entry in pick_entries if entry["value_picks"] is not None),
        },
        "picks": pick_entries,
        "roster_size": roster_size,
        "graded_at_pick": max(
            (safe_float(entry["overall_pick"]) for entry in pick_entries if entry["overall_pick"]),
            default=None,
        ),
    }
