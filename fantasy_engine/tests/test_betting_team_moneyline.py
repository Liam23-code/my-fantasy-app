"""Tests for the team-level projection model and the moneyline/spread/total model."""

from __future__ import annotations

import pytest

from betting.moneyline_model import evaluate_game, fair_moneyline, win_probability_from_spread
from betting.team_model import project_game, team_scoring_averages, team_scoring_by_week


def _averages(**teams) -> dict[str, dict[str, float]]:
    return {team: {"points_scored_avg": scored, "points_allowed_avg": allowed, "games_played": 8} for team, (scored, allowed) in teams.items()}


# --- team_model --------------------------------------------------------------


def test_project_game_favors_better_offense_and_worse_defense():
    averages = _averages(AAA=(28.0, 15.0), BBB=(15.0, 28.0))
    projection = project_game("AAA", "BBB", averages=averages)
    assert projection["home_expected_points"] > projection["away_expected_points"]
    assert projection["spread"] > 0


def test_project_game_evenly_matched_teams_gives_near_zero_spread():
    averages = _averages(AAA=(21.0, 21.0), BBB=(21.0, 21.0))
    projection = project_game("AAA", "BBB", averages=averages)
    assert projection["spread"] == pytest.approx(0.0)


def test_project_game_total_is_sum_of_expected_points():
    averages = _averages(AAA=(24.0, 20.0), BBB=(18.0, 22.0))
    projection = project_game("AAA", "BBB", averages=averages)
    assert projection["total"] == pytest.approx(projection["home_expected_points"] + projection["away_expected_points"])


def test_project_game_unknown_team_falls_back_to_league_average_not_error():
    averages = _averages(AAA=(24.0, 20.0))
    projection = project_game("AAA", "ZZZ", averages=averages, league_average_points=22.0)
    assert projection["away_expected_points"] > 0  # didn't crash, produced a real number


def test_team_scoring_averages_real_data_covers_all_32_teams():
    averages = team_scoring_averages(2025)
    assert len(averages) == 32
    for row in averages.values():
        assert row["games_played"] > 0
        assert row["points_scored_avg"] >= 0
        assert row["points_allowed_avg"] >= 0


def test_team_scoring_by_week_is_symmetric():
    # Whatever team A scored against B in a given week, B allowed exactly that much.
    by_week = team_scoring_by_week(2025)
    kc_week1 = next(w for w in by_week["KC"] if w["week"] == 1)
    opponent_week1 = next(w for w in by_week[kc_week1["opponent"]] if w["week"] == 1)
    assert opponent_week1["points_allowed"] == kc_week1["points_scored"]
    assert opponent_week1["points_scored"] == kc_week1["points_allowed"]


# --- moneyline_model -----------------------------------------------------------


def test_win_probability_from_spread_zero_is_a_coin_flip():
    assert win_probability_from_spread(0.0) == pytest.approx(0.5)


def test_win_probability_from_spread_increases_with_spread():
    assert win_probability_from_spread(7.0) > win_probability_from_spread(3.0) > win_probability_from_spread(0.0)


def test_win_probability_from_spread_bounded():
    assert 0.0 < win_probability_from_spread(-50.0) < 1.0
    assert 0.0 < win_probability_from_spread(50.0) < 1.0


def test_fair_moneyline_probabilities_sum_to_one():
    averages = _averages(AAA=(24.0, 20.0), BBB=(20.0, 24.0))
    fair = fair_moneyline("AAA", "BBB", averages=averages)
    assert fair["home_win_probability"] + fair["away_win_probability"] == pytest.approx(1.0)


def test_evaluate_game_moneyline_edge_favors_side_model_disagrees_with_market_on():
    averages = _averages(AAA=(28.0, 14.0), BBB=(14.0, 28.0))  # AAA heavily favored by the model
    # Market prices BBB (the underdog per our model) as the favorite -- a big model/market disagreement.
    game_odds = {"home_team": "AAA", "away_team": "BBB", "moneyline": {"home": 120, "away": -140}}
    result = evaluate_game(game_odds, averages=averages)
    assert result["moneyline"]["recommended_side"] == "home"
    assert result["moneyline"]["home_edge"] > 0


def test_evaluate_game_total_present_only_when_odds_supply_it():
    averages = _averages(AAA=(24.0, 20.0), BBB=(20.0, 24.0))
    game_odds = {"home_team": "AAA", "away_team": "BBB", "moneyline": {"home": -110, "away": -110}}
    result = evaluate_game(game_odds, averages=averages)
    assert "total" not in result
    assert "moneyline" in result


def test_evaluate_game_handles_no_odds_at_all_gracefully():
    averages = _averages(AAA=(24.0, 20.0), BBB=(20.0, 24.0))
    result = evaluate_game({"home_team": "AAA", "away_team": "BBB"}, averages=averages)
    assert "moneyline" not in result
    assert "total" not in result
    assert result["model"]["home_win_probability"] is not None
