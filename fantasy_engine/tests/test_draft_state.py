"""Tests for fantasy.draft_state -- the read-only "where the draft stands" snapshot."""

from __future__ import annotations

from fantasy.draft_state import (
    LATE_ROUND_BUFFER,
    STREAMER_POSITIONS,
    build_draft_state,
    from_live_state,
    late_round_streamer_flags,
    neutral_state,
    positional_dropoff,
    roster_needs,
    trailing_run,
)

SETTINGS = {
    "n_teams": 12,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1, "BENCH": 6},
    "flex_eligible": ["RB", "WR", "TE"],
}


def _p(pid, position, projection, adp=None):
    player = {
        "player_id": pid,
        "name": pid.upper(),
        "position": position,
        "team": "SF",
        "projection": float(projection),
        "expected_fantasy_points": float(projection),
    }
    if adp is not None:
        player["adp"] = float(adp)
    return player


def _pick(position):
    return {"position": position, "name": f"{position} guy", "player_id": f"id-{position}"}


# --- neutral / empty --------------------------------------------------------


def test_empty_board_returns_a_neutral_snapshot():
    assert build_draft_state([], [], SETTINGS) == neutral_state()


def test_neutral_snapshot_has_every_key_a_recommender_reads():
    state = neutral_state()
    assert set(state) == {
        "roster_counts",
        "needs",
        "dropoff_by_pos",
        "active_run",
        "run_pressure",
        "late_round_ok_positions",
        "rounds_remaining",
    }
    assert state["active_run"] is None
    assert state["run_pressure"] == {}
    assert state["late_round_ok_positions"] == set()


def test_from_live_state_tolerates_none_and_a_bare_dict():
    assert from_live_state(None) == neutral_state()
    assert from_live_state({}) == neutral_state()
    assert from_live_state({"remaining": []}) == neutral_state()


# --- trailing run ---------------------------------------------------------


def test_trailing_run_flags_a_position_that_dominates_the_window():
    picks = [_pick("RB"), _pick("WR"), _pick("WR"), _pick("WR"), _pick("WR"), _pick("QB")]
    active, share = trailing_run(picks, window=6)
    assert active == "WR"
    assert share["WR"] == round(4 / 6, 3)


def test_trailing_run_is_quiet_when_no_position_dominates():
    picks = [_pick("RB"), _pick("WR"), _pick("QB"), _pick("TE"), _pick("RB"), _pick("WR")]
    active, _ = trailing_run(picks, window=6)
    assert active is None


def test_trailing_run_handles_no_picks():
    assert trailing_run(None) == (None, {})
    assert trailing_run([]) == (None, {})


def test_trailing_run_only_reads_the_last_window():
    picks = [_pick("WR")] * 10 + [_pick("RB")] * 4
    active, _ = trailing_run(picks, window=4)
    assert active == "RB"


# --- positional dropoff -------------------------------------------------------


def test_positional_dropoff_is_the_gap_to_the_next_player():
    board = [_p("rb1", "RB", 250), _p("rb2", "RB", 200), _p("rb3", "RB", 190), _p("wr1", "WR", 180)]
    dropoff = positional_dropoff(board, SETTINGS)
    assert dropoff["RB"] == 50.0  # 250 -> 200
    assert dropoff["WR"] == 0.0  # only one WR left


def test_positional_dropoff_never_negative_and_zero_for_thin_positions():
    board = [_p("te1", "TE", 120)]
    assert positional_dropoff(board, SETTINGS)["TE"] == 0.0


# --- roster needs -----------------------------------------------------------


def test_roster_needs_counts_unfilled_starting_slots():
    needs = roster_needs([_p("rb1", "RB", 250)], SETTINGS)
    assert needs["RB"]["unfilled"] == 1  # 2 starters, 1 rostered
    assert needs["QB"]["unfilled"] == 1
    assert needs["RB"]["label"] == "Fills need"
    assert "FLEX" in needs


def test_roster_needs_marks_depth_once_a_slot_is_covered():
    roster = [_p("qb1", "QB", 300), _p("qb2", "QB", 280)]
    needs = roster_needs(roster, SETTINGS)
    assert needs["QB"]["unfilled"] == 0
    assert needs["QB"]["label"] == "Depth"


# --- late-round streamer flags ---------------------------------------------


def test_streamers_are_not_ok_early():
    # Empty roster -> 7 real starting slots open, so flags stay shut until
    # rounds_remaining <= 7 + LATE_ROUND_BUFFER.
    needs = roster_needs([], SETTINGS)
    assert late_round_streamer_flags(needs, rounds_remaining=7 + LATE_ROUND_BUFFER + 1) == set()


def test_streamers_open_up_when_the_draft_is_nearly_done():
    needs = roster_needs([], SETTINGS)
    assert late_round_streamer_flags(needs, rounds_remaining=7 + LATE_ROUND_BUFFER) == set(STREAMER_POSITIONS)


def test_streamers_open_sooner_once_real_starters_are_filled():
    roster = [
        _p("qb1", "QB", 300), _p("rb1", "RB", 250), _p("rb2", "RB", 240),
        _p("wr1", "WR", 230), _p("wr2", "WR", 220), _p("te1", "TE", 150),
        _p("rb3", "RB", 200),  # fills FLEX
    ]
    needs = roster_needs(roster, SETTINGS)
    assert late_round_streamer_flags(needs, rounds_remaining=LATE_ROUND_BUFFER) == set(STREAMER_POSITIONS)
    assert late_round_streamer_flags(needs, rounds_remaining=LATE_ROUND_BUFFER + 1) == set()


def test_late_round_flags_need_a_round_count():
    assert late_round_streamer_flags(roster_needs([], SETTINGS), rounds_remaining=None) == set()


# --- build_draft_state end to end -----------------------------------------


def test_build_draft_state_assembles_run_dropoff_and_rounds_remaining():
    board = [_p("rb1", "RB", 250, adp=20), _p("rb2", "RB", 195, adp=40), _p("wr1", "WR", 240, adp=22)]
    picks = [_pick("WR"), _pick("WR"), _pick("WR"), _pick("RB")]
    state = build_draft_state(
        board,
        [_p("qb1", "QB", 300)],
        SETTINGS,
        picks=picks,
        current_pick_overall=25,
        picks_until_next=20,
        num_rounds=15,
    )
    assert state["active_run"] == "WR"
    assert state["dropoff_by_pos"]["RB"] == 55.0
    assert state["rounds_remaining"] == 15 - 3 + 1  # pick 25, 12 teams -> round 3
    assert state["roster_counts"] == {"QB": 1}
