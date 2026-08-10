"""live_draft: turn-by-turn draft state for a human-in-the-loop mock draft.

Usage::

    from fantasy.live_draft import start_live_draft, draft_for_user, user_turn_context

    state = start_live_draft(
        projections, league_settings,
        num_teams=12, num_rounds=15, snake=True, user_draft_slot=5, seed=42,
    )
    # Bot teams ahead of you have already picked. state["awaiting_user_pick"]
    # is True once it's genuinely your turn.
    context = user_turn_context(state)
    # context["board"] / context["my_roster"] / context["overall_pick"] feed
    # fantasy.assistant.get_best_pick_for_round for live recommendations.

    state = draft_for_user(state, player_id=chosen["player_id"])
    # Your pick is recorded, then every bot pick up to your NEXT turn (or the
    # end of the draft) resolves automatically. state["picks"] now holds all
    # of it -- diff its length against what it was before this call to show
    # "what just happened" in a UI.

Why this exists rather than reusing :func:`fantasy.draft.simulate_draft`
--------------------------------------------------------------------------
``simulate_draft`` plays an entire mock draft in one call, using
:func:`fantasy.user_brain.user_brain_pick` to choose *for* the user. That's
the right tool for "show me how a draft could go." It is the wrong tool for
"let me actually make my own picks and see the room react" -- there is no
pause point in a single function call.

This module is the pause point. Every non-user pick still goes through
:func:`fantasy.draft.room_team_pick` -- the exact, already-validated room
model ``simulate_draft`` uses -- so the *bots* behave identically in both
modes. Only the *human's* pick is different: chosen live, not by
``user_brain_pick``, and every recommendation surface (see
:func:`fantasy.assistant.get_best_pick_for_round`) recomputes from the state
as it actually stands after your real choices, not a pre-planned script.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from fantasy.data_loader import is_synthetic
from fantasy.draft import (
    NEVER_EXCEED_CAP_POSITIONS,
    active_run_position,
    build_draft_order,
    draft_projection,
    identify_adp_clusters,
    rank_players_for_draft,
    room_team_pick,
)
from fantasy.models import ROSTER_POSITION_LIMITS, LeagueSettings, roster_cap_reached

_CAP_RELAXATION_WARNING = (
    "{count} pick(s) exceeded a skill-position cap (RB/WR/TE/QB bench depth) because every "
    "uncapped position was exhausted for that team; K/DST caps were still never exceeded."
)
_CAP_EMERGENCY_WARNING = (
    "{count} pick(s) had to exceed a roster cap because every remaining real player was "
    "already capped out for that team -- a rare, late-draft emergency, not routine cap enforcement."
)


def _coerce_league_settings(league_settings: dict[str, Any] | LeagueSettings) -> LeagueSettings:
    if isinstance(league_settings, LeagueSettings):
        return league_settings
    return LeagueSettings(**(league_settings or {}))


def start_live_draft(
    projections: list[dict[str, Any]],
    league_settings: dict[str, Any] | LeagueSettings,
    num_teams: int | None = None,
    num_rounds: int | None = None,
    snake: bool = True,
    user_draft_slot: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Start a live draft and fast-forward through any bot picks before your first turn.

    Synthetic players are dropped up front, exactly as :func:`fantasy.draft.simulate_draft`
    does -- a live draft board full of fabricated names is worse than a short one.

    Returns a plain-dict state (safe to hold in ``st.session_state`` -- nothing
    in it needs JSON serialization, it just needs to survive in memory for the
    session). Pass it to :func:`user_turn_context` to read what to show, and
    to :func:`draft_for_user` to record a pick.
    """
    settings = _coerce_league_settings(league_settings)
    if num_teams is not None:
        settings = settings.model_copy(update={"n_teams": int(num_teams)})
    rounds = int(num_rounds) if num_rounds is not None else sum(settings.roster_requirements.model_dump().values())
    if rounds < 1:
        raise ValueError("num_rounds must be >= 1")
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
    # matches the identical local import simulate_draft already uses.
    from fantasy.mock_data_ingestion import reach_tolerance_scale

    tolerance_scale = reach_tolerance_scale(remaining)

    state: dict[str, Any] = {
        "league_settings": settings,
        "n_teams": settings.n_teams,
        "rounds": rounds,
        "snake": snake,
        "user_draft_slot": user_draft_slot,
        "user_team": f"Team {user_draft_slot}" if user_draft_slot else None,
        "seed": seed,
        "order": build_draft_order(settings.n_teams, rounds, snake=snake),
        "order_index": 0,
        "remaining": remaining,
        "rosters": {f"Team {team}": [] for team in range(1, settings.n_teams + 1)},
        "picks": [],
        "by_round": {},
        "clusters": clusters,
        "tolerance_scale": tolerance_scale,
        "rng": np.random.default_rng(seed),
        "cap_emergencies": 0,
        "cap_relaxations": 0,
        "warnings": warnings,
        "awaiting_user_pick": False,
        "is_complete": False,
    }
    _advance_bot_picks(state)
    return state


def _resolve_eligible(state: dict[str, Any], team_name: str) -> tuple[list[dict[str, Any]], Counter]:
    """Eligible pool for one team's pick, with the same cap-relaxation ladder ``simulate_draft`` uses."""
    team_counts = Counter(candidate["position"] for candidate in state["rosters"][team_name])
    eligible = [
        candidate for candidate in state["remaining"] if not roster_cap_reached(team_counts, candidate["position"])
    ]
    if not eligible:
        # Emergency stage 1: every skill position is capped for this team --
        # take extra bench depth rather than stall. K/DST never get a 2nd copy
        # even here, see NEVER_EXCEED_CAP_POSITIONS.
        state["cap_relaxations"] += 1
        eligible = [
            candidate for candidate in state["remaining"] if candidate["position"] not in NEVER_EXCEED_CAP_POSITIONS
        ]
        if not eligible:
            # Emergency stage 2: truly nothing left but a capped-out K/DST.
            state["cap_emergencies"] += 1
            eligible = state["remaining"]
    return eligible, team_counts


def _apply_pick(state: dict[str, Any], team_name: str, player: dict[str, Any], slot: dict[str, int], active_run: str | None) -> None:
    """Record one resolved pick -- identical bookkeeping to ``simulate_draft``'s loop body."""
    remaining_at_position = sum(1 for candidate in state["remaining"] if candidate["position"] == player["position"])
    state["remaining"].pop(next(i for i, candidate in enumerate(state["remaining"]) if candidate is player))
    state["rosters"][team_name].append(player)

    is_user_pick = team_name == state["user_team"]
    projection = draft_projection(player)
    adp = player.get("adp")

    state["by_round"].setdefault(slot["round"], []).append(
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
    state["picks"].append(
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


def _advance_bot_picks(state: dict[str, Any]) -> None:
    """Resolve every consecutive bot pick from ``order_index`` on, stopping at your turn or the end."""
    while state["order_index"] < len(state["order"]):
        if not state["remaining"]:
            break
        slot = state["order"][state["order_index"]]
        team_name = f"Team {slot['team_number']}"
        if team_name == state["user_team"]:
            state["awaiting_user_pick"] = True
            return
        eligible, team_counts = _resolve_eligible(state, team_name)
        active_run = active_run_position(slot["overall_pick"], state["clusters"])
        player = room_team_pick(
            eligible, team_counts, slot["overall_pick"], slot["round"], active_run, state["tolerance_scale"], state["rng"]
        )
        _apply_pick(state, team_name, player, slot, active_run)
        state["order_index"] += 1

    state["awaiting_user_pick"] = False
    # This tail only ever runs once per state: draft_for_user refuses to run
    # again once awaiting_user_pick is False, so there's no call path that
    # reaches here twice. The "already added" checks are a guard against a
    # future refactor breaking that invariant, not a sign it happens today.
    state["is_complete"] = True
    if state["cap_relaxations"] and not any("skill-position cap" in warning for warning in state["warnings"]):
        state["warnings"].append(_CAP_RELAXATION_WARNING.format(count=state["cap_relaxations"]))
    if state["cap_emergencies"] and not any("late-draft emergency" in warning for warning in state["warnings"]):
        state["warnings"].append(_CAP_EMERGENCY_WARNING.format(count=state["cap_emergencies"]))


def user_turn_context(state: dict[str, Any]) -> dict[str, Any] | None:
    """Everything a UI needs to recommend and render your pending pick.

    Returns ``None`` when it isn't currently your turn (including when the
    draft has finished) -- a normal state for a caller to check, not an error.
    ``board`` is already "everyone still undrafted," ready to hand straight to
    :func:`fantasy.assistant.get_best_pick_for_round` as its ``board`` argument.
    """
    if not state.get("awaiting_user_pick"):
        return None
    slot = state["order"][state["order_index"]]
    next_user_slot = next(
        (s for s in state["order"][state["order_index"] + 1 :] if f"Team {s['team_number']}" == state["user_team"]),
        None,
    )
    picks_until_next = (next_user_slot["overall_pick"] - slot["overall_pick"] - 1) if next_user_slot else None
    return {
        "overall_pick": slot["overall_pick"],
        "round": slot["round"],
        "my_roster": list(state["rosters"][state["user_team"]]),
        "board": list(state["remaining"]),
        "picks_until_next": picks_until_next,
    }


def draft_for_user(state: dict[str, Any], player_id: Any) -> dict[str, Any]:
    """Record your pick for the current slot, then fast-forward to your next turn.

    Mutates and returns ``state`` (also mutated in place -- the return value
    is for chaining/reassignment convenience, e.g. ``state = draft_for_user(...)``).
    Every bot pick made while fast-forwarding is appended to ``state["picks"]``
    in order, so a caller can diff its length before/after this call to show
    exactly what the room did in between your turns.

    Raises :class:`ValueError` if it isn't currently your turn, or if
    ``player_id`` doesn't match anyone still on the board -- both are
    programming errors in the caller (the UI should only ever offer players
    from :func:`user_turn_context`'s own ``board``), not data conditions to
    silently paper over.
    """
    if not state.get("awaiting_user_pick"):
        raise ValueError("It is not currently your turn to pick.")
    slot = state["order"][state["order_index"]]
    team_name = state["user_team"]
    player = next((p for p in state["remaining"] if str(p.get("player_id")) == str(player_id)), None)
    if player is None:
        raise ValueError(f"Player {player_id!r} is not available on the board.")

    active_run = active_run_position(slot["overall_pick"], state["clusters"])
    _apply_pick(state, team_name, player, slot, active_run)
    state["order_index"] += 1
    state["awaiting_user_pick"] = False
    _advance_bot_picks(state)
    return state
