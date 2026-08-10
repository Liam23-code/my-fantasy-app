"""Tests for the turn-by-turn live draft engine."""

from __future__ import annotations

import pytest

from fantasy.draft import simulate_draft
from fantasy.live_draft import draft_for_user, start_live_draft, user_turn_context

# A small multi-position pool, ADP-spread within each position, so recommendations
# and roster-need reasoning have real signal to work with (unlike test_draft.py's
# single-position pools, which are enough for room_brain/order tests but not for
# proving the user's own choices actually change what happens next).
LIVE_SETTINGS = {
    "n_teams": 4,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 1, "WR": 1, "TE": 0, "FLEX": 1, "DST": 0, "K": 0, "BENCH": 4},
    "flex_eligible": ["RB", "WR", "TE"],
}


def _live_pool(rb=16, wr=16, qb=8, te=8):
    players = []
    for i in range(rb):
        players.append({"player_id": f"rb{i}", "name": f"RB{i}", "position": "RB", "team": "SF", "rushing_yards": 100 - i * 3, "adp": float(i * 2 + 1)})
    for i in range(wr):
        players.append({"player_id": f"wr{i}", "name": f"WR{i}", "position": "WR", "team": "KC", "receiving_yards": 95 - i * 3, "adp": float(i * 2 + 2)})
    for i in range(qb):
        players.append({"player_id": f"qb{i}", "name": f"QB{i}", "position": "QB", "team": "BUF", "passing_yards": 250 - i * 5, "adp": float(i * 4 + 15)})
    for i in range(te):
        players.append({"player_id": f"te{i}", "name": f"TE{i}", "position": "TE", "team": "DAL", "receiving_yards": 60 - i * 3, "adp": float(i * 4 + 18)})
    return players


def _play_out(state, chooser=None):
    """Drive a live draft to completion, picking recs[0] (or chooser's choice) every turn."""
    from fantasy.assistant import get_best_pick_for_round

    while not state["is_complete"]:
        context = user_turn_context(state)
        assert context is not None
        recs = get_best_pick_for_round(
            context["round"], context["my_roster"], context["board"], LIVE_SETTINGS,
            current_pick_overall=context["overall_pick"], picks_until_next=context["picks_until_next"], limit=5,
        )
        if chooser is not None:
            player_id = chooser(context, recs)
        elif recs:
            player_id = recs[0]["player_id"]
        else:
            player_id = context["board"][0]["player_id"]
        state = draft_for_user(state, player_id)
    return state


# --- starting a live draft ---------------------------------------------------


def test_start_live_draft_resolves_bots_ahead_of_the_users_slot():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=3, user_draft_slot=3, seed=1)
    assert state["awaiting_user_pick"] is True
    assert state["order_index"] == 2  # picks 1 and 2 (Team 1, Team 2) resolved
    assert len(state["picks"]) == 2
    assert all(pick["team"] != "Team 3" for pick in state["picks"])


def test_start_live_draft_first_slot_needs_no_bot_picks_first():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=3, user_draft_slot=1, seed=1)
    assert state["awaiting_user_pick"] is True
    assert state["order_index"] == 0
    assert state["picks"] == []


def test_start_live_draft_without_a_user_slot_runs_to_completion():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=2, user_draft_slot=None, seed=1)
    assert state["is_complete"] is True
    assert state["awaiting_user_pick"] is False
    assert len(state["picks"]) == LIVE_SETTINGS["n_teams"] * 2


def test_start_live_draft_rejects_invalid_rounds():
    with pytest.raises(ValueError):
        start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=0, user_draft_slot=1)


def test_start_live_draft_rejects_a_user_slot_outside_the_league():
    with pytest.raises(ValueError):
        start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=2, user_draft_slot=99)


def test_start_live_draft_excludes_synthetic_players_and_warns():
    pool = _live_pool() + [
        {"player_id": f"synthetic:rb:{i}", "name": f"Synthetic RB {i}", "position": "RB", "rushing_yards": 999}
        for i in range(5)
    ]
    state = start_live_draft(pool, LIVE_SETTINGS, num_rounds=2, user_draft_slot=1, seed=1)
    assert all(not pick["player_id"].startswith("synthetic") for pick in state["picks"])
    assert any("synthetic" in warning for warning in state["warnings"])


# --- user_turn_context --------------------------------------------------------


def test_user_turn_context_is_none_when_the_draft_has_finished():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=1, user_draft_slot=None, seed=1)
    assert state["is_complete"]
    assert user_turn_context(state) is None


def test_user_turn_context_board_excludes_already_drafted_players():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=3, user_draft_slot=3, seed=1)
    context = user_turn_context(state)
    drafted_ids = {pick["player_id"] for pick in state["picks"]}
    board_ids = {player["player_id"] for player in context["board"]}
    assert not (drafted_ids & board_ids)


def test_user_turn_context_reports_the_correct_pick_number_and_gap():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=3, user_draft_slot=3, seed=1)
    context = user_turn_context(state)
    # 4-team snake, slot 3, round 1: overall pick 3. Next user turn is round 2
    # slot (n_teams - 3 + 1) = 2 -> overall pick 4 + 2 = 6. Gap = 6 - 3 - 1 = 2.
    assert context["overall_pick"] == 3
    assert context["round"] == 1
    assert context["picks_until_next"] == 2


# --- draft_for_user ------------------------------------------------------------


def test_draft_for_user_records_the_pick_under_the_user_team():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=3, user_draft_slot=1, seed=1)
    context = user_turn_context(state)
    target = context["board"][0]
    state = draft_for_user(state, target["player_id"])
    mine = [pick for pick in state["picks"] if pick["is_user_pick"]]
    assert len(mine) == 1
    assert mine[0]["player_id"] == target["player_id"]
    assert mine[0]["team"] == "Team 1"


def test_draft_for_user_advances_through_bot_picks_to_the_next_user_turn():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=3, user_draft_slot=1, seed=1)
    context = user_turn_context(state)
    before = len(state["picks"])
    state = draft_for_user(state, context["board"][0]["player_id"])
    new_picks = state["picks"][before:]
    # My pick, then every bot pick up to (but not including) my next turn.
    assert new_picks[0]["is_user_pick"] is True
    assert all(not pick["is_user_pick"] for pick in new_picks[1:])
    assert state["awaiting_user_pick"] is True or state["is_complete"] is True


def test_draft_for_user_rejects_when_it_is_not_the_users_turn():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=1, user_draft_slot=None, seed=1)
    assert state["is_complete"]
    with pytest.raises(ValueError):
        draft_for_user(state, "rb0")


def test_draft_for_user_rejects_an_unknown_player_id():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=2, user_draft_slot=1, seed=1)
    with pytest.raises(ValueError):
        draft_for_user(state, "does-not-exist")


def test_draft_for_user_rejects_a_player_already_drafted():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=3, user_draft_slot=1, seed=1)
    context = user_turn_context(state)
    target_id = context["board"][0]["player_id"]
    state = draft_for_user(state, target_id)
    if state["awaiting_user_pick"]:
        with pytest.raises(ValueError):
            draft_for_user(state, target_id)  # already off the board


# --- full-draft integrity -----------------------------------------------------


def test_live_draft_never_drafts_the_same_player_twice():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=4, user_draft_slot=2, seed=3)
    state = _play_out(state)
    ids = [pick["player_id"] for pick in state["picks"]]
    assert len(ids) == len(set(ids))
    assert len(ids) == LIVE_SETTINGS["n_teams"] * 4


def test_live_draft_gives_the_user_exactly_one_pick_per_round():
    state = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=4, user_draft_slot=2, seed=3)
    state = _play_out(state)
    mine = [pick for pick in state["picks"] if pick["is_user_pick"]]
    assert len(mine) == 4
    assert sorted(pick["round"] for pick in mine) == [1, 2, 3, 4]


# --- reactivity: your actual choice changes what happens next -----------------


def test_live_draft_recommendations_reflect_the_users_real_roster_not_a_script():
    """Taking a WR instead of an available RB must change the next turn's context."""
    state_rb = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=2, user_draft_slot=1, seed=9)
    context = user_turn_context(state_rb)
    rb_choice = next(p for p in context["board"] if p["position"] == "RB")
    wr_choice = next(p for p in context["board"] if p["position"] == "WR")

    state_rb = draft_for_user(state_rb, rb_choice["player_id"])
    state_wr = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=2, user_draft_slot=1, seed=9)
    state_wr = draft_for_user(state_wr, wr_choice["player_id"])

    roster_rb = [p["position"] for p in user_turn_context(state_rb)["my_roster"]]
    roster_wr = [p["position"] for p in user_turn_context(state_wr)["my_roster"]]
    assert roster_rb == ["RB"]
    assert roster_wr == ["WR"]
    assert roster_rb != roster_wr


def test_live_draft_bot_picks_before_the_users_first_turn_match_simulate_draft():
    """Bots must behave identically in both engines -- same room_team_pick, same seed."""
    batch = simulate_draft(_live_pool(), LIVE_SETTINGS, rounds=3, seed=5, user_draft_slot=3)
    live = start_live_draft(_live_pool(), LIVE_SETTINGS, num_rounds=3, user_draft_slot=3, seed=5)

    batch_before_user = [pick for pick in batch["picks"] if pick["overall_pick"] < 3]
    live_before_user = live["picks"]
    assert [p["player_id"] for p in batch_before_user] == [p["player_id"] for p in live_before_user]


# --- edge cases ------------------------------------------------------------


def test_live_draft_falls_back_gracefully_when_every_remaining_position_is_capped():
    """When recommendations empty out (every remaining position roster-capped),
    the raw board must still let the user draft someone -- this is a real,
    reachable state, not a dead end."""
    tiny_settings = {
        "n_teams": 2,
        "scoring_mode": "ppr",
        "roster_requirements": {"QB": 0, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 6},
        "flex_eligible": [],
    }
    pool = [{"player_id": f"rb{i}", "name": f"RB{i}", "position": "RB", "rushing_yards": 100 - i, "adp": float(i + 1)} for i in range(10)]
    state = start_live_draft(pool, tiny_settings, num_rounds=5, user_draft_slot=1, seed=1)
    state = _play_out(state)
    assert state["is_complete"]
    mine = [pick for pick in state["picks"] if pick["is_user_pick"]]
    assert len(mine) == 5  # kept drafting via the board fallback even once recs ran dry
