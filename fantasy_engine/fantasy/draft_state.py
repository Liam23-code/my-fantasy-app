"""draft_state: a compact, read-only snapshot of "where this draft stands".

Everything a live recommendation needs beyond the raw board and roster:
which positions are still a genuine need, how sharp the value drop-off is at
each position *right now*, whether a positional run is on, and whether it is
finally late enough that a kicker or defense is a reasonable pick.

Pure and deterministic. It reads the board, the roster and the pick log and
returns plain dicts / sets / ints; it holds no state, does no I/O and never
mutates its inputs, so a UI can rebuild it on every rerun for free. It is a
leaf module: the two non-trivial helpers it borrows from
:mod:`fantasy.assistant` (``_projection_of`` and ``_scarcity_by_position``,
so the numbers here match the ones the recommender ranks with) are imported
lazily inside the functions that use them, so importing this module never
drags in the whole assistant.

Usage::

    from fantasy.draft_state import build_draft_state, from_live_state

    ds = from_live_state(state)          # straight off a fantasy.live_draft state
    get_best_pick_for_round(..., draft_state=ds, weights={"dropoff": 0.6})
"""

from __future__ import annotations

from collections import Counter
from math import ceil
from typing import Any

from fantasy.models import LeagueSettings, roster_cap_reached

#: Streamer positions -- one starter, no FLEX outlet, so a second one is close
#: to dead weight. The late-round logic and the recommender's round-band
#: multiplier both key off this set.
STREAMER_POSITIONS: frozenset[str] = frozenset({"K", "DST"})

#: How many of the most recent picks count as "recent" for run detection.
RUN_WINDOW = 8

#: A position is "running" once it takes at least this share of the trailing
#: window *and* at least this many of those picks.
RUN_MIN_SHARE = 0.5
RUN_MIN_COUNT = 3

#: Rounds of slack before the draft is "late enough" to spend a pick on a
#: streamer: once ``rounds_remaining <= (real-position starting slots still
#: unfilled) + this buffer``, K/DST are cleared to be drafted.
LATE_ROUND_BUFFER = 2


def _coerce_settings(league_settings: Any) -> LeagueSettings:
    if isinstance(league_settings, LeagueSettings):
        return league_settings
    return LeagueSettings(**(league_settings or {}))


def _position_of(player: Any) -> str:
    return str((player or {}).get("position", "")).strip().upper()


def _roster_counts(my_roster: list[dict[str, Any]] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in my_roster or []:
        position = _position_of(player)
        if position:
            counts[position] = counts.get(position, 0) + 1
    return counts


def neutral_state() -> dict[str, Any]:
    """The snapshot that changes no recommendation -- used before a draft
    starts, off-turn, and whenever the board is empty."""
    return {
        "roster_counts": {},
        "needs": {},
        "dropoff_by_pos": {},
        "active_run": None,
        "run_pressure": {},
        "late_round_ok_positions": set(),
        "rounds_remaining": None,
    }


def trailing_run(
    picks: list[dict[str, Any]] | None,
    window: int = RUN_WINDOW,
) -> tuple[str | None, dict[str, float]]:
    """The position on a run over the last ``window`` picks, plus every
    position's share of that window. ``(None, {})`` when there are no picks.

    Deliberately measured from the picks that actually happened, not from a
    pre-draft ADP clustering: a run that emerges live (five of the last seven
    picks were WR) is exactly what "reactive" pick logic has to see.
    """
    recent = [p for p in (picks or []) if isinstance(p, dict) and _position_of(p)][-int(window):]
    if not recent:
        return None, {}
    counts = Counter(_position_of(p) for p in recent)
    share = {position: round(count / len(recent), 3) for position, count in counts.items()}
    top_position, top_count = counts.most_common(1)[0]
    on_run = top_count >= RUN_MIN_COUNT and (top_count / len(recent)) >= RUN_MIN_SHARE
    return (top_position if on_run else None), share


def positional_dropoff(
    board: list[dict[str, Any]] | None,
    league_settings: Any = None,
) -> dict[str, float]:
    """Points lost by passing on the best player left at each position -- the
    gap from the best available to the next one down.

    A large gap is a tier cliff: take him now or take a meaningfully worse
    player at that position later. Positions with one player left (or none)
    have no measurable drop and report ``0.0``.
    """
    from fantasy.assistant import _projection_of  # match the recommender's number

    settings = _coerce_settings(league_settings)
    by_position: dict[str, list[float]] = {}
    for player in board or []:
        position = _position_of(player)
        if position:
            by_position.setdefault(position, []).append(_projection_of(player, settings))

    dropoff: dict[str, float] = {}
    for position, projections in by_position.items():
        if len(projections) < 2:
            dropoff[position] = 0.0
            continue
        projections.sort(reverse=True)
        dropoff[position] = round(max(0.0, projections[0] - projections[1]), 2)
    return dropoff


def roster_needs(
    my_roster: list[dict[str, Any]] | None,
    league_settings: Any = None,
) -> dict[str, dict[str, Any]]:
    """Per starting position (FLEX included): how many you have, how many you
    start, whether it still fills a need, how many slots are empty, and
    whether roster construction limits are already maxed."""
    settings = _coerce_settings(league_settings)
    counts = _roster_counts(my_roster)
    slots = settings.roster_requirements.starting_slots()
    flex_slots = int(slots.pop("FLEX", 0) or 0)
    flex_eligible = {str(position).strip().upper() for position in settings.flex_eligible}

    needs: dict[str, dict[str, Any]] = {}
    flex_surplus = 0
    for position, slot_count in slots.items():
        position = str(position).strip().upper()
        required = int(slot_count or 0)
        have = counts.get(position, 0)
        unfilled = max(0, required - have)
        needs[position] = {
            "have": have,
            "starting_slots": required,
            "unfilled": unfilled,
            "label": "Fills need" if unfilled else "Depth",
            "capped": roster_cap_reached(counts, position),
        }
        if position in flex_eligible:
            flex_surplus += max(0, have - required)

    if flex_slots:
        flex_unfilled = max(0, flex_slots - flex_surplus)
        needs["FLEX"] = {
            "have": flex_surplus,
            "starting_slots": flex_slots,
            "unfilled": flex_unfilled,
            "label": "Fills need" if flex_unfilled else "Depth",
            "capped": False,
        }
    return needs


def late_round_streamer_flags(
    needs: dict[str, dict[str, Any]],
    rounds_remaining: int | None,
) -> set[str]:
    """Which streamer positions it is finally reasonable to draft.

    Empty until the draft is close enough to the end that every real-position
    starting slot still open could be filled and still leave room for a K and
    a DST. A position already at its roster-construction cap is not returned.
    """
    if rounds_remaining is None:
        return set()
    unfilled_real = sum(
        int(info.get("unfilled", 0) or 0)
        for position, info in needs.items()
        if position not in STREAMER_POSITIONS
    )
    if rounds_remaining <= unfilled_real + LATE_ROUND_BUFFER:
        return {
            position
            for position in STREAMER_POSITIONS
            if not needs.get(position, {}).get("capped")
        }
    return set()


def build_draft_state(
    board: list[dict[str, Any]] | None,
    my_roster: list[dict[str, Any]] | None,
    league_settings: Any = None,
    *,
    picks: list[dict[str, Any]] | None = None,
    current_pick_overall: int | None = None,
    picks_until_next: int | None = None,
    num_rounds: int | None = None,
    n_teams: int | None = None,
) -> dict[str, Any]:
    """Assemble the snapshot. Safe on empty input -- an empty board returns
    :func:`neutral_state`, which folds into a recommendation as a no-op."""
    if not board:
        return neutral_state()

    settings = _coerce_settings(league_settings)
    teams = int(n_teams or settings.n_teams or 12)

    needs = roster_needs(my_roster, settings)
    dropoff_by_pos = positional_dropoff(board, settings)
    active_run, run_pressure = trailing_run(picks)

    rounds_remaining: int | None = None
    if num_rounds and current_pick_overall and teams >= 1:
        current_round = max(1, ceil(int(current_pick_overall) / teams))
        rounds_remaining = max(0, int(num_rounds) - current_round + 1)

    return {
        "roster_counts": _roster_counts(my_roster),
        "needs": needs,
        "dropoff_by_pos": dropoff_by_pos,
        "active_run": active_run,
        "run_pressure": run_pressure,
        "late_round_ok_positions": late_round_streamer_flags(needs, rounds_remaining),
        "rounds_remaining": rounds_remaining,
    }


def from_live_state(
    state: dict[str, Any] | None,
    *,
    current_pick_overall: int | None = None,
    picks_until_next: int | None = None,
) -> dict[str, Any]:
    """Build a snapshot straight from a :mod:`fantasy.live_draft` state dict.

    Read-only: it never mutates ``state``. Pass ``current_pick_overall`` /
    ``picks_until_next`` when the caller has already computed a
    ``user_turn_context`` (the live page has); otherwise this derives them from
    ``state["order"]`` directly (without going through ``user_turn_context``,
    which normalizes the board in place). Returns :func:`neutral_state` off-turn
    or before a draft has started.
    """
    if not state:
        return neutral_state()
    board = state.get("remaining") or []
    if not board:
        return neutral_state()

    user_team = state.get("user_team")
    my_roster = list((state.get("rosters") or {}).get(user_team) or [])

    if current_pick_overall is None and picks_until_next is None and state.get("awaiting_user_pick"):
        order = state.get("order") or []
        index = int(state.get("order_index", 0) or 0)
        if 0 <= index < len(order):
            slot = order[index]
            current_pick_overall = slot.get("overall_pick")
            following = next(
                (
                    later
                    for later in order[index + 1 :]
                    if f"Team {later.get('team_number')}" == user_team
                ),
                None,
            )
            if following is not None and current_pick_overall is not None:
                picks_until_next = following["overall_pick"] - slot["overall_pick"] - 1

    return build_draft_state(
        board,
        my_roster,
        state.get("league_settings"),
        picks=state.get("picks"),
        current_pick_overall=current_pick_overall,
        picks_until_next=picks_until_next,
        num_rounds=state.get("rounds"),
        n_teams=state.get("n_teams"),
    )
