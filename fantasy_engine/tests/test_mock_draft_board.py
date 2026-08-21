"""Integrity tests for batch and live mock-draft boards."""

from __future__ import annotations

import pytest

from fantasy.draft import (
    ensure_unique_board_state,
    finalize_user_team,
    rank_players_for_draft,
    remove_player_from_board,
    simulate_draft,
    validate_board_integrity,
)
from fantasy.live_draft import override_draft_for_user, start_live_draft, user_turn_context

SETTINGS = {
    "n_teams": 4,
    "scoring_mode": "ppr",
    "roster_requirements": {
        "QB": 0,
        "RB": 1,
        "WR": 1,
        "TE": 0,
        "FLEX": 0,
        "DST": 0,
        "K": 0,
        "BENCH": 2,
    },
    "flex_eligible": ["RB", "WR", "TE"],
}


def _player(index: int, position: str | None = None, *, adp: float | None = None) -> dict:
    resolved_position = position or ("RB" if index % 2 == 0 else "WR")
    row = {
        "player_id": f"p{index}",
        "name": f"Player {index}",
        "position": resolved_position,
        "team": f"T{index % 8}",
        "adp": float(index + 1) if adp is None else adp,
    }
    if resolved_position == "RB":
        row["rushing_yards"] = 1_200 - index * 10
        row["receptions"] = 30
    else:
        row["receiving_yards"] = 1_200 - index * 10
        row["receptions"] = 60
    return row


def _pool(size: int = 32) -> list[dict]:
    return [_player(index) for index in range(size)]


def test_unique_board_deduplicates_and_orders_by_consensus_adp():
    board = [
        _player(1, adp=28),
        _player(2, adp=14),
        {**_player(1, adp=4), "name": "Duplicate Player 1"},
        _player(3, adp=9),
    ]

    returned = ensure_unique_board_state(board)

    assert returned is board
    assert [player["player_id"] for player in board] == ["p1", "p3", "p2"]
    assert [player["adp"] for player in board] == [4, 9, 14]
    assert validate_board_integrity(board)


def test_remove_player_tombstone_prevents_stale_state_from_reinserting_player():
    removed = _player(1, adp=1)
    state = {"remaining": [removed, _player(2, adp=2)], "picks": [], "rosters": {}}

    returned = remove_player_from_board(state, "p1")
    state["remaining"].append(dict(removed))
    ensure_unique_board_state(state)

    assert returned is state
    assert state["removed_player_ids"] == ["p1"]
    assert [player["player_id"] for player in state["remaining"]] == ["p2"]
    assert validate_board_integrity(state)


def test_board_normalization_excludes_drafted_players_and_duplicate_copies():
    state = {
        "remaining": [_player(3, adp=30), _player(1, adp=10), _player(1, adp=1)],
        "picks": [{"player_id": "p3"}],
        "rosters": {},
    }

    ensure_unique_board_state(state)

    assert [player["player_id"] for player in state["remaining"]] == ["p1"]
    assert state["remaining"][0]["adp"] == 1
    assert validate_board_integrity(state)


@pytest.mark.parametrize(
    ("position_filter", "expected"),
    [
        ("RB", {"RB"}),
        ("wr", {"WR"}),
        ("FLEX", {"RB", "WR", "TE"}),
        ("ALL", {"QB", "RB", "WR", "TE"}),
    ],
)
def test_position_filtering_is_case_insensitive_and_supports_flex(position_filter, expected):
    board = [_player(1, "QB"), _player(2, "RB"), _player(3, "WR"), _player(4, "TE")]

    ensure_unique_board_state(board, position_filter=position_filter)

    assert {player["position"] for player in board} == expected
    assert validate_board_integrity(board, position_filter=position_filter)


def test_validate_board_integrity_reports_duplicates_order_and_drafted_overlap():
    duplicate = _player(1, adp=20)
    state = {
        "remaining": [duplicate, _player(2, adp=10), dict(duplicate)],
        "picks": [{"player_id": "p2"}],
        "rosters": {},
    }

    assert not validate_board_integrity(state)
    with pytest.raises(ValueError, match="duplicate|drafted|ADP"):
        validate_board_integrity(state, raise_on_error=True)


def test_ranked_board_never_counts_duplicate_player_rows_twice():
    source = _player(1)
    ranked = rank_players_for_draft([source, dict(source), _player(2)], SETTINGS)

    assert [player["player_id"] for player in ranked].count("p1") == 1
    assert len(ranked) == 2


def test_batch_mock_draft_removes_every_drafted_player_from_remaining_board():
    pool = _pool()
    pool.insert(0, {**pool[0], "adp": 99})

    result = simulate_draft(pool, SETTINGS, rounds=3, seed=7, user_draft_slot=2)

    drafted_ids = {pick["player_id"] for pick in result["picks"]}
    remaining_ids = [player["player_id"] for player in result["remaining"]]
    assert drafted_ids.isdisjoint(remaining_ids)
    assert len(remaining_ids) == len(set(remaining_ids))
    assert validate_board_integrity(result["remaining"])
    assert any("duplicate" in warning for warning in result["warnings"])


def test_live_board_is_adp_ordered_and_position_filter_does_not_mutate_full_board():
    state = start_live_draft(_pool(), SETTINGS, num_rounds=3, user_draft_slot=3, seed=4)
    full_ids = [player["player_id"] for player in state["remaining"]]

    context = user_turn_context(state, position_filter="RB")

    assert context is not None
    assert context["board"]
    assert all(player["position"] == "RB" for player in context["board"])
    assert [player["player_id"] for player in state["remaining"]] == full_ids
    assert validate_board_integrity(state)


def test_override_reclaims_player_atomically_without_duplicate_or_reappearance():
    state = start_live_draft(_pool(), SETTINGS, num_rounds=3, user_draft_slot=3, seed=11)
    target = state["picks"][0]

    override_draft_for_user(state, target["player_id"])

    all_rostered_ids = [
        player["player_id"]
        for roster in state["rosters"].values()
        for player in roster
    ]
    remaining_ids = [player["player_id"] for player in state["remaining"]]
    user_ids = {player["player_id"] for player in state["rosters"][state["user_team"]]}
    assert target["player_id"] in user_ids
    assert target["player_id"] not in remaining_ids
    assert len(all_rostered_ids) == len(set(all_rostered_ids))
    assert set(all_rostered_ids).isdisjoint(remaining_ids)
    assert validate_board_integrity(state)


def test_finalize_handoff_requires_explicit_save_and_does_not_call_persistence(monkeypatch):
    import fantasy.my_team_manager as manager

    def unexpected_save(*_args, **_kwargs):
        raise AssertionError("mock draft completion must not persist a managed team")

    monkeypatch.setattr(manager, "save_user_team", unexpected_save)
    roster = [_player(1)]

    handoff = finalize_user_team(roster)

    assert handoff["saved"] is False
    assert handoff["requires_explicit_save"] is True
    assert handoff["roster"] == roster
    assert handoff["redirect_page"] is None
    assert handoff["recommended_page"] == "pages/26_Fantasy_Saved_Teams.py"


def test_simulated_user_draft_returns_unsaved_handoff():
    result = simulate_draft(_pool(), SETTINGS, rounds=2, seed=5, user_draft_slot=1)

    assert result["my_team_handoff"]["saved"] is False
    assert result["my_team_handoff"]["requires_explicit_save"] is True
    assert result["redirect_page"] is None
    assert result["redirect_tab"] is None
