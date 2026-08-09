"""user_brain: the user's own analytics-driven pick during a simulated draft.

Usage::

    from fantasy.user_brain import user_brain_pick

    player = user_brain_pick(
        remaining_pool, league_settings, my_roster,
        next_pick_overall=17, picks_until_next=8,
    )

This wraps :func:`fantasy.assistant.suggest_draft_picks` -- the exact engine
behind the live Draft Assistant tab -- and takes its #1 suggestion. Every
signal lives in ``suggest_draft_picks`` itself (VORP value, ADP timing with
steal/reach, position-biased scarcity, roster need and hard caps, risk
moderation, bye-week awareness, QB/WR stacking, same-team limits,
must-draft-now availability); nothing is duplicated here. An improvement to
the assistant improves the simulated user team for free, and the two can
never quietly drift apart.

"Always outperforms room_brain" (see :mod:`fantasy.room_brain`) means the
user's team always drafts from this fuller, multi-signal model while every
other team uses a simpler ADP-anchored heuristic -- a real, structural,
statistically-verifiable advantage. It is not a guaranteed win forced onto any
one simulated draft: this module does not rig outcomes, inflate the user's
players, or handicap the room's.
"""

from __future__ import annotations

from typing import Any

from fantasy.assistant import suggest_draft_picks
from fantasy.models import LeagueSettings


def user_brain_pick(
    pool: list[dict[str, Any]],
    league_settings: dict[str, Any] | LeagueSettings,
    my_roster: list[dict[str, Any]],
    next_pick_overall: int,
    picks_until_next: int | None,
) -> dict[str, Any] | None:
    """The single best real player object from ``pool`` for the user's team right now.

    Returns ``None`` when ``pool`` has no valid (real, roster-eligible)
    player left -- callers should fall back to their own emergency pick
    rather than treat this as an error, the same way ``suggest_draft_picks``
    returning ``[]`` is a normal, renderable empty state elsewhere.
    """
    suggestions = suggest_draft_picks(
        pool,
        league_settings,
        my_roster=my_roster,
        next_pick_overall=next_pick_overall,
        picks_until_next=picks_until_next,
        limit=1,
    )
    if not suggestions:
        return None
    best_id = suggestions[0]["player_id"]
    return next((player for player in pool if (player.get("player_id") or player.get("id")) == best_id), None)
