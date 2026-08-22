"""Unit tests for fantasy.optimizer."""

from __future__ import annotations

import pytest

from fantasy import optimizer as optimizer_module
from fantasy.models import Roster
from fantasy.optimizer import optimize_lineup, start_sit_advice

SIMPLE_SETTINGS = {
    "n_teams": 10,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 0, "K": 0, "BENCH": 3},
    "flex_eligible": ["RB", "WR", "TE"],
}


def _player(pid, name, position, team="AAA", injury_status=None, slot="BENCH"):
    return {"player_id": pid, "name": name, "position": position, "nfl_team": team, "slot": slot, "injury_status": injury_status}


def _proj(pid, name, position, points_field, value):
    return {"player_id": pid, "name": name, "position": position, points_field: value}


@pytest.fixture
def sample_roster():
    return [
        _player("qb1", "QB1", "QB"),
        _player("rb1", "RB1", "RB"),
        _player("rb2", "RB2", "RB"),
        _player("rb3", "RB3", "RB"),
        _player("wr1", "WR1", "WR"),
        _player("wr2", "WR2", "WR"),
        _player("wr3", "WR3", "WR"),
        _player("te1", "TE1", "TE"),
        _player("te2", "TE2", "TE"),
    ]


@pytest.fixture
def sample_projections():
    # rushing_yards / receiving_yards divided by 10 -> exact points (ppr, no bonus since <100).
    return [
        _proj("qb1", "QB1", "QB", "passing_yards", 200),  # 8.0 pts
        _proj("rb1", "RB1", "RB", "rushing_yards", 90),  # 9.0
        _proj("rb2", "RB2", "RB", "rushing_yards", 70),  # 7.0
        _proj("rb3", "RB3", "RB", "rushing_yards", 30),  # 3.0
        _proj("wr1", "WR1", "WR", "receiving_yards", 95),  # 9.5
        _proj("wr2", "WR2", "WR", "receiving_yards", 60),  # 6.0
        _proj("wr3", "WR3", "WR", "receiving_yards", 20),  # 2.0
        _proj("te1", "TE1", "TE", "receiving_yards", 80),  # 8.0
        _proj("te2", "TE2", "TE", "receiving_yards", 10),  # 1.0
    ]


def test_optimize_lineup_fills_dedicated_slots_with_best_players(sample_roster, sample_projections):
    lineup = optimize_lineup(sample_roster, sample_projections, SIMPLE_SETTINGS)
    starter_names = {s["name"] for s in lineup["starters"]}
    assert "QB1" in starter_names
    assert {"RB1", "RB2"}.issubset(starter_names)  # top 2 RBs are dedicated starters
    assert {"WR1", "WR2"}.issubset(starter_names)  # top 2 WRs are dedicated starters
    assert "TE1" in starter_names  # best TE is the dedicated starter


def test_optimize_lineup_flex_goes_to_best_remaining_flex_eligible_player(sample_roster, sample_projections):
    lineup = optimize_lineup(sample_roster, sample_projections, SIMPLE_SETTINGS)
    flex_starter = next(s for s in lineup["starters"] if s["slot"] == "FLEX")
    # Remaining flex-eligible pool after dedicated slots: RB3(3.0), WR3(2.0), TE2(1.0) -> RB3 wins.
    assert flex_starter["name"] == "RB3"


def test_optimize_lineup_total_points_is_sum_of_starters(sample_roster, sample_projections):
    lineup = optimize_lineup(sample_roster, sample_projections, SIMPLE_SETTINGS)
    assert lineup["total_points"] == pytest.approx(sum(s["points"] for s in lineup["starters"]))
    assert lineup["total_points"] == pytest.approx(8.0 + 9.0 + 7.0 + 9.5 + 6.0 + 8.0 + 3.0)


def test_optimize_lineup_bench_gets_worst_players(sample_roster, sample_projections):
    lineup = optimize_lineup(sample_roster, sample_projections, SIMPLE_SETTINGS)
    bench_names = {b["name"] for b in lineup["bench"]}
    assert bench_names == {"WR3", "TE2"}


def test_optimize_lineup_excludes_injured_players_by_default(sample_roster, sample_projections):
    roster = list(sample_roster)
    roster[1] = _player("rb1", "RB1", "RB", injury_status="OUT")  # was the best RB
    lineup = optimize_lineup(roster, sample_projections, SIMPLE_SETTINGS)
    starter_names = {s["name"] for s in lineup["starters"]}
    assert "RB1" not in starter_names
    assert "RB2" in starter_names  # next best RB starts instead


def test_optimize_lineup_respects_locked_player_override_for_injury(sample_roster, sample_projections):
    roster = list(sample_roster)
    roster[1] = _player("rb1", "RB1", "RB", injury_status="OUT")
    lineup = optimize_lineup(roster, sample_projections, SIMPLE_SETTINGS, constraints={"locked_player_ids": ["rb1"]})
    starter_names = {s["name"] for s in lineup["starters"]}
    assert "RB1" in starter_names


def test_optimize_lineup_excluded_player_never_starts(sample_roster, sample_projections):
    lineup = optimize_lineup(sample_roster, sample_projections, SIMPLE_SETTINGS, constraints={"excluded_player_ids": ["rb1"]})
    starter_names = {s["name"] for s in lineup["starters"]}
    assert "RB1" not in starter_names
    assert "RB2" in starter_names
    assert "RB3" in starter_names  # both remaining RBs must start (dedicated=2)


def test_optimize_lineup_max_players_per_team_constraint(sample_projections):
    stacked_roster = [
        _player("qb1", "QB1", "QB", team="AAA"),
        _player("rb1", "RB1", "RB", team="AAA"),
        _player("rb2", "RB2", "RB", team="AAA"),
        _player("rb3", "RB3", "RB", team="BBB"),
        _player("wr1", "WR1", "WR", team="AAA"),
        _player("wr2", "WR2", "WR", team="AAA"),
        _player("wr3", "WR3", "WR", team="BBB"),
        _player("te1", "TE1", "TE", team="AAA"),
        _player("te2", "TE2", "TE", team="BBB"),
    ]
    lineup = optimize_lineup(stacked_roster, sample_projections, SIMPLE_SETTINGS, constraints={"max_players_per_team": 3})
    team_counts: dict[str, int] = {}
    for starter in lineup["starters"]:
        team_counts[starter["nfl_team"]] = team_counts.get(starter["nfl_team"], 0) + 1
    assert all(count <= 3 for count in team_counts.values())


def test_optimize_lineup_reports_unfilled_slots_when_roster_too_thin():
    thin_roster = [_player("qb1", "QB1", "QB")]
    thin_projections = [_proj("qb1", "QB1", "QB", "passing_yards", 200)]
    lineup = optimize_lineup(thin_roster, thin_projections, SIMPLE_SETTINGS)
    assert "RB" in lineup["unfilled_slots"]
    assert any("RB" in warning for warning in lineup["warnings"])


def test_optimize_lineup_ilp_and_greedy_agree_on_total_points(sample_roster, sample_projections):
    ilp_lineup = optimize_lineup(sample_roster, sample_projections, SIMPLE_SETTINGS, constraints={"solver": "ilp"})
    greedy_lineup = optimize_lineup(sample_roster, sample_projections, SIMPLE_SETTINGS, constraints={"solver": "greedy"})
    assert ilp_lineup["solver"] == "ilp"
    assert greedy_lineup["solver"] == "greedy"
    assert ilp_lineup["total_points"] == pytest.approx(greedy_lineup["total_points"])


def test_optimize_lineup_falls_back_to_greedy_when_pulp_unavailable(sample_roster, sample_projections, monkeypatch):
    monkeypatch.setattr(optimizer_module, "PULP_AVAILABLE", False)
    lineup = optimize_lineup(sample_roster, sample_projections, SIMPLE_SETTINGS)
    assert lineup["solver"] == "greedy"
    assert lineup["total_points"] > 0


def test_optimize_lineup_forced_ilp_raises_if_infeasible(sample_roster, sample_projections, monkeypatch):
    monkeypatch.setattr(optimizer_module, "PULP_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        optimize_lineup(sample_roster, sample_projections, SIMPLE_SETTINGS, constraints={"solver": "ilp"})


def test_optimize_lineup_accepts_roster_model_and_dict(sample_roster, sample_projections):
    model_lineup = optimize_lineup(Roster(team_name="T", players=sample_roster), sample_projections, SIMPLE_SETTINGS)
    dict_lineup = optimize_lineup({"team_name": "T", "players": sample_roster}, sample_projections, SIMPLE_SETTINGS)
    assert model_lineup["total_points"] == pytest.approx(dict_lineup["total_points"])


def test_optimize_lineup_rejects_unsupported_roster_type():
    with pytest.raises(TypeError):
        optimize_lineup(12345, [], SIMPLE_SETTINGS)


def test_solve_ilp_exception_falls_back_to_greedy(sample_roster, sample_projections, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(optimizer_module, "_solve_ilp", _boom)
    lineup = optimize_lineup(sample_roster, sample_projections, SIMPLE_SETTINGS)
    assert lineup["solver"] == "greedy"
    assert lineup["total_points"] > 0


def test_greedy_solver_honors_locked_players_directly(sample_roster, sample_projections):
    lineup = optimize_lineup(
        sample_roster,
        sample_projections,
        SIMPLE_SETTINGS,
        constraints={"solver": "greedy", "locked_player_ids": ["rb3"]},
    )
    starter_names = {s["name"] for s in lineup["starters"]}
    assert "RB3" in starter_names  # locked in directly by the greedy solver's own locked-player pass


def test_greedy_solver_does_not_double_count_a_locked_top_ranked_player(sample_roster, sample_projections):
    # Locking RB1 (already the #1 RB by points) must not cause the dedicated
    # fill loop to count it twice when it re-encounters RB1 in sorted order.
    lineup = optimize_lineup(
        sample_roster,
        sample_projections,
        SIMPLE_SETTINGS,
        constraints={"solver": "greedy", "locked_player_ids": ["rb1"]},
    )
    rb_starters = [s for s in lineup["starters"] if s["position"] == "RB" and s["slot"] == "RB"]
    assert len(rb_starters) == 2
    assert {s["name"] for s in rb_starters} == {"RB1", "RB2"}


def test_start_sit_advice_no_eligible_slot_for_position():
    settings = {
        "n_teams": 1,
        "roster_requirements": {"QB": 1, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 1},
        "flex_eligible": [],
    }
    roster = [_player("qb1", "QB1", "QB"), _player("k1", "K1", "K")]
    projections = [_proj("qb1", "QB1", "QB", "passing_yards", 200), _proj("k1", "K1", "K", "passing_yards", 0)]
    advice = start_sit_advice(roster, projections, settings)
    k1_advice = next(a for a in advice if a["player"] == "K1")
    assert k1_advice["start_or_bench"] == "bench"
    assert k1_advice["delta_points"] == 0.0
    assert "No eligible starting slot" in k1_advice["reason"]


def test_start_sit_advice_flags_a_positive_swap_under_binding_team_cap(monkeypatch):
    # The greedy fallback isn't guaranteed optimal under a binding per-team
    # cap (documented in optimizer._solve_greedy): filling RB before WR can
    # block a higher-scoring same-team WR out of both its dedicated slot and
    # FLEX, leaving a genuinely better bench option undetected by the solver.
    # start_sit_advice should still surface that swap opportunity to the user.
    monkeypatch.setattr(optimizer_module, "PULP_AVAILABLE", False)
    settings = {
        "n_teams": 1,
        "roster_requirements": {"QB": 0, "RB": 1, "WR": 1, "TE": 0, "FLEX": 1, "DST": 0, "K": 0, "BENCH": 1},
        "flex_eligible": ["RB", "WR"],
        "max_players_per_nfl_team": 1,
    }
    roster = [
        _player("rb1", "RB1", "RB", team="A"),
        _player("wr1", "WR1", "WR", team="A"),
        _player("rb2", "RB2", "RB", team="B"),
        _player("wr2", "WR2", "WR", team="B"),
    ]
    projections = [
        _proj("rb1", "RB1", "RB", "rushing_yards", 100),
        _proj("wr1", "WR1", "WR", "receiving_yards", 90),
        _proj("rb2", "RB2", "RB", "rushing_yards", 50),
        _proj("wr2", "WR2", "WR", "receiving_yards", 10),
    ]
    advice = start_sit_advice(roster, projections, settings)
    by_name = {a["player"]: a for a in advice}
    assert by_name["WR1"]["start_or_bench"] == "bench"
    assert by_name["WR1"]["delta_points"] > 0
    assert "Would net" in by_name["WR1"]["reason"]


def test_optimize_lineup_missing_projection_scores_zero_not_error(sample_roster):
    lineup = optimize_lineup(sample_roster, [], SIMPLE_SETTINGS)
    assert lineup["total_points"] == 0.0
    assert len(lineup["starters"]) > 0  # slots still fill, just with zero-point players


def test_start_sit_advice_marks_starters_and_bench(sample_roster, sample_projections):
    advice = start_sit_advice(sample_roster, sample_projections, SIMPLE_SETTINGS)
    by_name = {a["player"]: a for a in advice}
    assert by_name["RB1"]["start_or_bench"] == "start"
    assert by_name["WR3"]["start_or_bench"] == "bench"
    assert len(advice) == len(sample_roster)


def test_start_sit_advice_bench_delta_reflects_gap_to_weakest_starter(sample_roster, sample_projections):
    advice = start_sit_advice(sample_roster, sample_projections, SIMPLE_SETTINGS)
    by_name = {a["player"]: a for a in advice}
    # WR3 (2.0 pts) benched behind flex starter RB3 (3.0 pts): correctly benched, negative delta.
    assert by_name["WR3"]["delta_points"] < 0
    assert "Correctly benched" in by_name["WR3"]["reason"]


def test_start_sit_advice_flags_ruled_out_players_regardless_of_projection(sample_roster, sample_projections):
    roster = list(sample_roster)
    roster[-1] = _player("te2", "TE2", "TE", injury_status="OUT")
    projections = list(sample_projections)
    projections[-1] = _proj("te2", "TE2", "TE", "receiving_yards", 200)  # would otherwise be a great start
    advice = start_sit_advice(roster, projections, SIMPLE_SETTINGS)
    te2_advice = next(a for a in advice if a["player"] == "TE2")
    assert te2_advice["start_or_bench"] == "bench"
    assert "Ruled OUT" in te2_advice["reason"]


def test_start_sit_advice_default_league_settings_when_none(sample_roster, sample_projections):
    advice = start_sit_advice(sample_roster, sample_projections, None)
    assert len(advice) == len(sample_roster)


def test_start_sit_advice_flex_starter_compares_against_all_flex_eligible_bench():
    # QB-only + one flex-eligible starter scenario: the FLEX starter should be
    # compared to every flex-eligible bench player, not just same-position ones.
    settings = {
        "n_teams": 10,
        "roster_requirements": {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 1, "DST": 0, "K": 0, "BENCH": 2},
        "flex_eligible": ["RB", "WR"],
    }
    roster = [_player("rb1", "RB1", "RB"), _player("wr1", "WR1", "WR")]
    projections = [_proj("rb1", "RB1", "RB", "rushing_yards", 90), _proj("wr1", "WR1", "WR", "receiving_yards", 40)]
    advice = start_sit_advice(roster, projections, settings)
    by_name = {a["player"]: a for a in advice}
    assert by_name["RB1"]["start_or_bench"] == "start"
    assert by_name["WR1"]["start_or_bench"] == "bench"
    assert "RB1" in by_name["WR1"]["reason"]


# --- basis: the lineup is solved on projections, not prior-season box scores ---

ONE_RB_SETTINGS = {
    "n_teams": 10,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 0, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 1},
    "flex_eligible": [],
}


def test_lineup_starts_the_better_projection_not_the_better_box_score():
    """A back coming off an injured season outstarts one whose raw line was better."""
    roster = [_player("hurt", "Hurt Star", "RB"), _player("filler", "Healthy Filler", "RB")]
    projections = [
        # 40 raw points, but projected for a full season back in the lead role.
        {"player_id": "hurt", "name": "Hurt Star", "position": "RB",
         "rushing_yards": 400, "projection": 240.0, "scoring_mode": "ppr"},
        # A better box score last year, and a worse outlook.
        {"player_id": "filler", "name": "Healthy Filler", "position": "RB",
         "rushing_yards": 900, "projection": 120.0, "scoring_mode": "ppr"},
    ]
    lineup = optimize_lineup(roster, projections, ONE_RB_SETTINGS)
    assert [starter["name"] for starter in lineup["starters"]] == ["Hurt Star"]
    assert lineup["total_points"] == pytest.approx(240.0)
