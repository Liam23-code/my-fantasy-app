"""Unit tests for fantasy.trade."""
from __future__ import annotations

import pytest

from fantasy.trade import evaluate_trade

SETTINGS = {"n_teams": 12, "scoring_mode": "ppr", "roster_requirements": {"RB": 2, "WR": 2}, "flex_eligible": ["RB", "WR", "TE"]}


def _player(pid, name, position, points_field, value, **extra):
    return {"player_id": pid, "name": name, "position": position, points_field: value, **extra}


def test_fair_value_matches_hand_computed_weekly_point_difference():
    # A gives up a 5.0/wk player, receives a 9.0/wk player -> +4.0/wk * weeks favors A.
    team_a_gives = [_player("a", "Give Guy", "RB", "rushing_yards", 50)]  # 5.0 pts/wk
    team_b_gives = [_player("b", "Receive Guy", "RB", "rushing_yards", 90)]  # 9.0 pts/wk
    result = evaluate_trade(team_a_gives, team_b_gives, SETTINGS, weeks_remaining=10, monte_carlo_iterations=200, seed=1)
    assert result["fair_value"] == pytest.approx((9.0 - 5.0) * 10, abs=0.01)
    assert result["team_a_receives_points"] == pytest.approx(9.0 * 10, abs=0.01)
    assert result["team_a_gives_points"] == pytest.approx(5.0 * 10, abs=0.01)


def test_positive_fair_value_favors_team_a_with_supporting_win_probability():
    team_a_gives = [_player("a", "Give Guy", "RB", "rushing_yards", 20)]
    team_b_gives = [_player("b", "Receive Guy", "RB", "rushing_yards", 150)]
    result = evaluate_trade(team_a_gives, team_b_gives, SETTINGS, weeks_remaining=8, monte_carlo_iterations=2000, seed=1)
    assert result["fair_value"] > 0
    assert "Favors Team A" in result["recommendation"]
    assert result["win_prob_delta"] > 0
    assert result["team_a_win_probability"] > 0.5


def test_negative_fair_value_favors_team_b():
    team_a_gives = [_player("a", "Give Guy", "RB", "rushing_yards", 150)]
    team_b_gives = [_player("b", "Receive Guy", "RB", "rushing_yards", 20)]
    result = evaluate_trade(team_a_gives, team_b_gives, SETTINGS, weeks_remaining=8, monte_carlo_iterations=2000, seed=1)
    assert result["fair_value"] < 0
    assert "Favors Team B" in result["recommendation"]
    assert result["win_prob_delta"] < 0


def test_roughly_equal_value_trade_is_labeled_fair():
    team_a_gives = [_player("a", "A Guy", "WR", "receiving_yards", 80)]
    team_b_gives = [_player("b", "B Guy", "WR", "receiving_yards", 82)]
    result = evaluate_trade(team_a_gives, team_b_gives, SETTINGS, weeks_remaining=10, monte_carlo_iterations=200, seed=1)
    assert "Fair trade" in result["recommendation"]
    assert abs(result["win_prob_delta"]) < 0.2


def test_bare_name_strings_resolved_via_projections_lookup():
    pool = [
        _player("a", "Give Guy", "RB", "rushing_yards", 50),
        _player("b", "Receive Guy", "RB", "rushing_yards", 90),
    ]
    result = evaluate_trade(["Give Guy"], ["Receive Guy"], SETTINGS, projections=pool, monte_carlo_iterations=200, seed=1)
    assert result["team_a_gives_points"] == pytest.approx(5.0 * result["weeks_remaining"], abs=0.01)
    assert result["team_a_receives_points"] == pytest.approx(9.0 * result["weeks_remaining"], abs=0.01)


def test_bare_name_not_found_in_projections_raises_value_error():
    pool = [_player("a", "Give Guy", "RB", "rushing_yards", 50)]
    with pytest.raises(ValueError):
        evaluate_trade(["Give Guy"], ["Nonexistent Player"], SETTINGS, projections=pool)


def test_bare_name_without_any_projections_raises_value_error():
    with pytest.raises(ValueError):
        evaluate_trade(["Some Player"], [_player("b", "B", "RB", "rushing_yards", 50)], SETTINGS)


def test_multi_for_one_trade_supported():
    team_a_gives = [_player("a", "Star Player", "RB", "rushing_yards", 150)]
    team_b_gives = [
        _player("b1", "Depth 1", "WR", "receiving_yards", 60),
        _player("b2", "Depth 2", "WR", "receiving_yards", 60),
    ]
    result = evaluate_trade(team_a_gives, team_b_gives, SETTINGS, weeks_remaining=10, monte_carlo_iterations=200, seed=1)
    assert result["team_a_receives_points"] == pytest.approx(12.0 * 10, abs=0.01)


def test_same_seed_is_fully_reproducible():
    team_a_gives = [_player("a", "A", "RB", "rushing_yards", 60)]
    team_b_gives = [_player("b", "B", "RB", "rushing_yards", 65)]
    first = evaluate_trade(team_a_gives, team_b_gives, SETTINGS, monte_carlo_iterations=500, seed=42)
    second = evaluate_trade(team_a_gives, team_b_gives, SETTINGS, monte_carlo_iterations=500, seed=42)
    assert first["team_a_win_probability"] == second["team_a_win_probability"]


def test_invalid_monte_carlo_iterations_raises():
    with pytest.raises(ValueError):
        evaluate_trade([_player("a", "A", "RB", "rushing_yards", 50)], [_player("b", "B", "RB", "rushing_yards", 50)], SETTINGS, monte_carlo_iterations=0)


def test_invalid_weeks_remaining_raises():
    with pytest.raises(ValueError):
        evaluate_trade([_player("a", "A", "RB", "rushing_yards", 50)], [_player("b", "B", "RB", "rushing_yards", 50)], SETTINGS, weeks_remaining=0)


def test_rationale_mentions_players_points_and_monte_carlo():
    team_a_gives = [_player("a", "Give Guy", "RB", "rushing_yards", 50)]
    team_b_gives = [_player("b", "Receive Guy", "RB", "rushing_yards", 90)]
    result = evaluate_trade(team_a_gives, team_b_gives, SETTINGS, monte_carlo_iterations=200, seed=1)
    combined_rationale = " ".join(result["rationale"])
    assert "Give Guy" in combined_rationale
    assert "Receive Guy" in combined_rationale
    assert "Monte Carlo" in combined_rationale
    assert "Net swing" in combined_rationale


def test_roster_context_notes_need_when_provided():
    team_a_gives = [_player("a", "Give Guy", "WR", "receiving_yards", 50)]
    team_b_gives = [_player("b", "Receive RB", "RB", "rushing_yards", 90)]
    settings = {"n_teams": 1, "roster_requirements": {"RB": 1, "WR": 0, "FLEX": 0}, "flex_eligible": []}
    result = evaluate_trade(
        team_a_gives, team_b_gives, settings,
        team_a_roster=[],  # Team A has zero RBs -> receiving one fills a need
        monte_carlo_iterations=200, seed=1,
    )
    combined_rationale = " ".join(result["rationale"])
    assert "Addresses a roster need" in combined_rationale
    assert "Receive RB" in combined_rationale


def test_roster_context_notes_no_need_when_position_already_full():
    team_a_gives = [_player("a", "Give Guy", "WR", "receiving_yards", 50)]
    team_b_gives = [_player("b", "Receive RB", "RB", "rushing_yards", 90)]
    settings = {"n_teams": 1, "roster_requirements": {"RB": 1, "WR": 0, "FLEX": 0}, "flex_eligible": []}
    result = evaluate_trade(
        team_a_gives, team_b_gives, settings,
        team_a_roster=[{"position": "RB"}, {"position": "RB"}],  # already well-stocked at RB
        monte_carlo_iterations=200, seed=1,
    )
    combined_rationale = " ".join(result["rationale"])
    assert "Does not address a starting roster need" in combined_rationale


def test_team_b_roster_context_also_reported():
    team_a_gives = [_player("a", "Give RB", "RB", "rushing_yards", 90)]
    team_b_gives = [_player("b", "Give WR", "WR", "receiving_yards", 50)]
    settings = {"n_teams": 1, "roster_requirements": {"RB": 1, "WR": 0, "FLEX": 0}, "flex_eligible": []}
    result = evaluate_trade(
        team_a_gives, team_b_gives, settings,
        team_b_roster=[],  # Team B has zero RBs -> receiving one fills a need
        monte_carlo_iterations=200, seed=1,
    )
    combined_rationale = " ".join(result["rationale"])
    assert "Team B: Addresses a roster need" in combined_rationale


def test_floor_and_ceiling_widen_simulation_variance_without_changing_fair_value():
    tight = [_player("a", "Tight", "RB", "rushing_yards", 60, floor=55, ceiling=65)]
    wide = [_player("a", "Wide", "RB", "rushing_yards", 60, floor=10, ceiling=110)]
    other_side = [_player("b", "B", "RB", "rushing_yards", 60)]
    tight_result = evaluate_trade(tight, other_side, SETTINGS, monte_carlo_iterations=3000, seed=7)
    wide_result = evaluate_trade(wide, other_side, SETTINGS, monte_carlo_iterations=3000, seed=7)
    assert tight_result["fair_value"] == pytest.approx(wide_result["fair_value"], abs=0.01)
    # Wider band means outcomes are less predictable, pulling the win probability closer to 50/50.
    assert abs(wide_result["win_prob_delta"]) <= abs(tight_result["win_prob_delta"]) + 1e-9


def test_accepts_league_settings_model_instance():
    from fantasy.models import LeagueSettings

    result = evaluate_trade(
        [_player("a", "A", "RB", "rushing_yards", 50)],
        [_player("b", "B", "RB", "rushing_yards", 50)],
        LeagueSettings(**SETTINGS),
        monte_carlo_iterations=100,
    )
    assert "recommendation" in result
