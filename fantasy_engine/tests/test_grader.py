"""Tests for the live team grader.

Every score here is an edge over *this* league's average team, so the tests are
built around pools where that average is hand-computable: with ``n_teams=2``,
the average team's RB1 is simply the mean of the two best RBs.
"""

from __future__ import annotations

import pytest

from fantasy.grader import (
    OVERALL_WEIGHTS,
    build_universe,
    grade_overall_team,
    grade_position_group,
    grade_team,
    league_average_starters,
    letter_grade,
)

SETTINGS = {
    "n_teams": 2,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 3},
    "flex_eligible": ["RB", "WR", "TE"],
}

FLEX_SETTINGS = {
    "n_teams": 2,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 0, "RB": 1, "WR": 1, "TE": 0, "FLEX": 1, "DST": 0, "K": 0, "BENCH": 2},
    "flex_eligible": ["RB", "WR"],
}

SCARCITY_SETTINGS = {
    "n_teams": 2,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 0, "RB": 0, "WR": 1, "TE": 1, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 2},
    "flex_eligible": ["RB", "WR", "TE"],
}


def _p(player_id, position, projection, adp=None, **extra):
    """One pool player carrying an explicit projection, as a projected pool does."""
    player = {
        "player_id": player_id,
        "name": player_id.upper(),
        "position": position,
        "team": "SF",
        "projection": float(projection),
        "expected_fantasy_points": float(projection),
        "projection_season": 2026,
    }
    if adp is not None:
        player["adp"] = float(adp)
    player.update(extra)
    return player


def _pick(player_id, position, overall_pick, adp, projection):
    return {
        "player_id": player_id,
        "name": player_id.upper(),
        "position": position,
        "overall_pick": overall_pick,
        "round": (overall_pick - 1) // 2 + 1,
        "adp": float(adp),
        "projection": float(projection),
        "is_user_pick": True,
    }


def _group(report, position):
    return next(group for group in report["positions"] if group["position"] == position)


# --- letter grades ------------------------------------------------------------


def test_letter_grade_covers_the_whole_scale():
    assert letter_grade(100.0) == "A+"
    assert letter_grade(93.0) == "A"
    assert letter_grade(83.0) == "B"
    assert letter_grade(73.0) == "C"
    assert letter_grade(63.0) == "D"
    assert letter_grade(12.0) == "F"


# --- league average -----------------------------------------------------------


def test_league_average_starter_is_the_mean_of_the_first_block_of_teams():
    """With 2 teams, the average RB1 is the mean of the two best RBs."""
    universe = [_p("rb1", "RB", 200), _p("rb2", "RB", 180), _p("rb3", "RB", 160), _p("rb4", "RB", 140)]
    assert league_average_starters("RB", universe, SETTINGS) == pytest.approx(190.0)


def test_league_average_falls_back_to_the_worst_player_when_the_pool_is_thin():
    """A league thinner than its own starting requirement still gets a real number."""
    universe = [_p("rb1", "RB", 200)]
    assert league_average_starters("RB", universe, SETTINGS) == pytest.approx(200.0)


def test_league_average_is_zero_for_a_position_the_league_does_not_start():
    universe = [_p("te1", "TE", 150), _p("te2", "TE", 120)]
    assert league_average_starters("TE", universe, SETTINGS) == 0.0


# --- build_universe -----------------------------------------------------------


def test_build_universe_merges_rosters_and_board_without_duplicates():
    roster = [_p("rb1", "RB", 200)]
    other = {"Team 2": [_p("rb2", "RB", 180)]}
    board = [_p("rb1", "RB", 200), _p("rb3", "RB", 160)]
    universe = build_universe(roster, board, other)
    assert [player["player_id"] for player in universe] == ["rb1", "rb2", "rb3"]


# --- positional grades --------------------------------------------------------


def test_a_perfectly_average_position_group_scores_exactly_fifty():
    roster = [_p("rb1", "RB", 190)]
    board = [_p("rb2", "RB", 190), _p("rb3", "RB", 190), _p("rb4", "RB", 190)]
    group = grade_position_group("RB", roster, board, SETTINGS)
    assert group["league_average_points"] == pytest.approx(190.0)
    assert group["points_vs_average"] == pytest.approx(0.0)
    assert group["score"] == pytest.approx(50.0)


def test_a_stronger_position_group_outscores_a_weaker_one():
    board = [_p("rb2", "RB", 180), _p("rb3", "RB", 160), _p("rb4", "RB", 140)]
    strong = grade_position_group("RB", [_p("rb1", "RB", 200)], board, SETTINGS)
    weak = grade_position_group("RB", [_p("rb4", "RB", 140)], board + [_p("rb1", "RB", 200)], SETTINGS)
    assert strong["score"] > 50.0 > weak["score"]
    assert strong["points_vs_average"] > 0 > weak["points_vs_average"]


def test_scores_stay_inside_zero_to_one_hundred_for_an_absurd_roster():
    board = [_p("rb2", "RB", 10), _p("rb3", "RB", 10), _p("rb4", "RB", 10)]
    group = grade_position_group("RB", [_p("rb1", "RB", 100_000)], board, SETTINGS)
    assert 0.0 <= group["score"] <= 100.0


def test_the_same_margin_grades_higher_at_a_scarcer_position():
    """A hundred points clear at TE is worth more than at WR -- TE is harder to fix."""
    roster = [_p("wr1", "WR", 200), _p("te1", "TE", 200)]
    board = [
        _p("wr2", "WR", 160),
        _p("wr3", "WR", 140),
        _p("wr4", "WR", 120),
        _p("te2", "TE", 160),
        _p("te3", "TE", 140),
        _p("te4", "TE", 120),
    ]
    wr = grade_position_group("WR", roster, board, SCARCITY_SETTINGS)
    te = grade_position_group("TE", roster, board, SCARCITY_SETTINGS)
    assert wr["points_vs_average"] == pytest.approx(te["points_vs_average"])
    assert te["score"] > wr["score"]
    assert te["scarcity_weight"] > wr["scarcity_weight"]


def test_adp_value_captured_at_a_position_raises_its_grade():
    roster = [_p("rb1", "RB", 190)]
    board = [_p("rb2", "RB", 190), _p("rb3", "RB", 190), _p("rb4", "RB", 190)]
    neutral = grade_position_group("RB", roster, board, SETTINGS)
    steal = grade_position_group("RB", roster, board, SETTINGS, picks=[_pick("rb1", "RB", overall_pick=20, adp=4.0, projection=190)])
    assert steal["adp_value_picks"] == pytest.approx(16.0)
    assert steal["score"] > neutral["score"]


def test_a_position_the_league_does_not_start_is_reported_as_not_applicable():
    group = grade_position_group("TE", [_p("te1", "TE", 150)], [], SETTINGS)
    assert group["applicable"] is False
    assert group["score"] == 50.0
    assert "not a starting slot" in group["rationale"]


def test_an_unfilled_starting_slot_is_reported():
    group = grade_position_group("RB", [], [_p("rb2", "RB", 190)], SETTINGS)
    assert group["unfilled_slots"] == 1
    assert group["starters"] == []
    assert "still empty" in group["rationale"]


def test_bench_depth_beyond_the_starting_slots_is_counted():
    roster = [_p("rb1", "RB", 200), _p("rb5", "RB", 90), _p("rb6", "RB", 80)]
    group = grade_position_group("RB", roster, [_p("rb2", "RB", 180)], SETTINGS)
    assert group["starter_slots"] == 1
    assert group["depth"] == 3
    assert len(group["starters"]) == 1


def test_flex_is_graded_from_whoever_is_left_after_the_dedicated_slots():
    roster = [_p("rb1", "RB", 200), _p("rb2", "RB", 180), _p("wr1", "WR", 150)]
    board = [_p("rb3", "RB", 120), _p("rb4", "RB", 110), _p("wr2", "WR", 100), _p("wr3", "WR", 90)]
    flex = grade_position_group("FLEX", roster, board, FLEX_SETTINGS)
    assert [starter["player_id"] for starter in flex["starters"]] == ["rb2"]
    assert flex["starter_points"] == pytest.approx(180.0)
    assert flex["applicable"] is True


# --- overall grade ------------------------------------------------------------


def test_overall_weights_sum_to_one():
    assert sum(OVERALL_WEIGHTS.values()) == pytest.approx(1.0)


def test_overall_grade_reports_all_four_components():
    roster = [_p("qb1", "QB", 300), _p("rb1", "RB", 200), _p("wr1", "WR", 190)]
    board = [_p("qb2", "QB", 260), _p("rb2", "RB", 180), _p("wr2", "WR", 170)]
    overall = grade_overall_team(roster, board, SETTINGS)
    assert set(overall["components"]) == set(OVERALL_WEIGHTS)
    assert 0.0 <= overall["score"] <= 100.0
    assert overall["grade"] == letter_grade(overall["score"])


def test_positional_balance_punishes_a_roster_with_one_hole():
    """Two strong groups plus an empty one must grade below three even ones."""
    board = [
        _p("qb2", "QB", 200),
        _p("qb3", "QB", 190),
        _p("rb2", "RB", 200),
        _p("rb3", "RB", 190),
        _p("wr2", "WR", 200),
        _p("wr3", "WR", 190),
    ]
    even = grade_overall_team([_p("qb1", "QB", 210), _p("rb1", "RB", 210), _p("wr1", "WR", 210)], board, SETTINGS)
    lopsided = grade_overall_team([_p("qb1", "QB", 320), _p("rb1", "RB", 320)], board, SETTINGS)
    assert lopsided["components"]["positional_balance"] < even["components"]["positional_balance"]


def test_injury_designations_lower_the_risk_component():
    board = [_p("rb2", "RB", 190), _p("wr2", "WR", 190), _p("qb2", "QB", 190)]
    healthy = grade_overall_team([_p("rb1", "RB", 200)], board, SETTINGS)
    injured = grade_overall_team([_p("rb1", "RB", 200, injury_status="OUT")], board, SETTINGS)
    assert injured["components"]["risk_profile"] < healthy["components"]["risk_profile"]
    assert any("injury" in note for note in injured["risk_notes"])


def test_unfilled_starting_slots_lower_the_risk_component():
    board = [_p("rb2", "RB", 190), _p("wr2", "WR", 190), _p("qb2", "QB", 190)]
    full = grade_overall_team([_p("qb1", "QB", 190), _p("rb1", "RB", 190), _p("wr1", "WR", 190)], board, SETTINGS)
    partial = grade_overall_team([_p("rb1", "RB", 190)], board, SETTINGS)
    assert partial["components"]["risk_profile"] < full["components"]["risk_profile"]


def test_capturing_adp_value_across_the_draft_raises_the_value_component():
    roster = [_p("rb1", "RB", 190), _p("wr1", "WR", 190)]
    board = [_p("rb2", "RB", 190), _p("wr2", "WR", 190)]
    reached = grade_overall_team(
        roster,
        board,
        SETTINGS,
        picks=[_pick("rb1", "RB", 2, 30.0, 190), _pick("wr1", "WR", 3, 40.0, 190)],
    )
    stole = grade_overall_team(
        roster,
        board,
        SETTINGS,
        picks=[_pick("rb1", "RB", 30, 2.0, 190), _pick("wr1", "WR", 40, 3.0, 190)],
    )
    assert stole["components"]["value_vs_adp"] > reached["components"]["value_vs_adp"]


# --- the full report ----------------------------------------------------------


def _deep_pool():
    """A board deep enough to have a real value curve, with ADP tracking projection."""
    players = [_p(f"rb{i + 1}", "RB", 220 - i * 12, adp=1 + i * 3) for i in range(8)]
    players += [_p(f"wr{i + 1}", "WR", 210 - i * 11, adp=2 + i * 3) for i in range(8)]
    players += [_p(f"qb{i + 1}", "QB", 300 - i * 14, adp=20 + i * 6) for i in range(6)]
    return players


def _full_report():
    """A three-pick draft: one clear steal, one clear reach, one taken on time."""
    pool = {player["player_id"]: player for player in _deep_pool()}
    mine = ["rb1", "wr8", "qb1"]
    roster = [pool[player_id] for player_id in mine]
    board = [player for player_id, player in pool.items() if player_id not in mine]
    picks = [
        _pick("wr8", "WR", overall_pick=2, adp=23.0, projection=133),  # 21-pick reach on the worst WR
        _pick("rb1", "RB", overall_pick=18, adp=1.0, projection=220),  # the best RB, 17 picks late
        _pick("qb1", "QB", overall_pick=20, adp=20.0, projection=300),  # exactly on time
    ]
    return grade_team(roster, board, SETTINGS, picks=picks)


def test_grade_team_returns_the_whole_report():
    report = _full_report()
    assert set(report) >= {"overall", "positions", "best_pick", "worst_pick", "value_vs_adp", "picks", "roster_size"}
    assert report["roster_size"] == 3
    assert {group["position"] for group in report["positions"]} == {"QB", "RB", "WR"}


def test_grade_team_finds_the_best_and_worst_pick():
    report = _full_report()
    assert report["best_pick"]["player_id"] == "rb1"  # the best RB, taken 17 picks late
    assert report["worst_pick"]["player_id"] == "wr8"  # the worst WR, taken 21 picks early
    assert report["best_pick"]["pick_score"] > report["worst_pick"]["pick_score"]
    assert report["best_pick"]["value_picks"] == pytest.approx(17.0)
    assert report["worst_pick"]["value_picks"] == pytest.approx(-21.0)


def test_a_pick_of_adp_value_is_priced_off_the_pools_own_value_curve():
    report = _full_report()
    steal = report["best_pick"]
    assert steal["value_points"] > 0
    assert steal["pick_score"] == pytest.approx(steal["vorp"] + steal["value_points"])


def test_grade_team_totals_the_value_gained_against_adp():
    report = _full_report()
    # (2-23) + (18-1) + (20-20) = -4
    assert report["value_vs_adp"]["total_picks"] == pytest.approx(-4.0)
    assert report["value_vs_adp"]["picks_with_adp"] == 3


def test_grade_team_tracks_the_pick_it_was_graded_at():
    assert _full_report()["graded_at_pick"] == pytest.approx(20.0)


def test_grade_team_handles_an_empty_roster_as_a_normal_state():
    board = [_p("rb2", "RB", 190), _p("wr2", "WR", 190), _p("qb2", "QB", 190)]
    report = grade_team([], board, SETTINGS)
    assert report["roster_size"] == 0
    assert report["best_pick"] is None
    assert all(group["unfilled_slots"] == 1 for group in report["positions"])
    assert report["overall"]["score"] < 50.0


def test_grade_team_accepts_a_bare_scoring_mode_string():
    report = grade_team([_p("rb1", "RB", 200)], [_p("rb2", "RB", 180)], "ppr")
    assert 0.0 <= report["overall"]["score"] <= 100.0


def test_grade_team_measures_the_league_average_over_real_rosters_when_given_them():
    """Mid-draft, other teams' picks have left the board -- all_rosters puts them back."""
    roster = [_p("rb1", "RB", 200)]
    board = [_p("rb4", "RB", 100)]
    without = grade_team(roster, board, SETTINGS)
    with_league = grade_team(roster, board, SETTINGS, all_rosters={"Team 2": [_p("rb2", "RB", 260)]})
    assert _group(with_league, "RB")["league_average_points"] > _group(without, "RB")["league_average_points"]
    assert _group(with_league, "RB")["score"] < _group(without, "RB")["score"]


def test_grade_team_is_callable_after_every_pick_without_carrying_state():
    """The live panel calls this on every rerun; two identical calls must agree."""
    first = _full_report()
    second = _full_report()
    assert first["overall"]["score"] == second["overall"]["score"]
    assert [group["score"] for group in first["positions"]] == [group["score"] for group in second["positions"]]
