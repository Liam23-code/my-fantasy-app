"""Tests for the parlay engine: leg construction, correlation detection, EV ranking."""

from __future__ import annotations

import pytest

from betting.parlay_engine import (
    correlation_adjusted_probability,
    detect_correlations,
    evaluate_parlay,
    make_leg,
    naive_joint_probability,
    parlay_decimal_odds,
    rank_parlays,
)


def _leg(**overrides) -> dict:
    base = {"description": "leg", "model_probability": 0.55, "price": -110, "confidence": 0.7}
    base.update(overrides)
    return make_leg(**base)


# --- correlation detection ---------------------------------------------------


def test_qb_pass_catcher_same_team_same_direction_is_correlated():
    qb = _leg(team="KC", market="passing_yards", side="over")
    wr = _leg(team="KC", market="receiving_yards", side="over")
    findings = detect_correlations([qb, wr])
    assert len(findings) == 1
    assert findings[0]["kind"] == "qb_pass_catcher_stack"
    assert findings[0]["direction"] == "positive"


def test_qb_pass_catcher_opposite_direction_is_not_correlated():
    qb = _leg(team="KC", market="passing_yards", side="over")
    wr = _leg(team="KC", market="receiving_yards", side="under")
    assert detect_correlations([qb, wr]) == []


def test_qb_pass_catcher_different_teams_is_not_correlated():
    qb = _leg(team="KC", market="passing_yards", side="over")
    wr = _leg(team="BUF", market="receiving_yards", side="over")
    assert detect_correlations([qb, wr]) == []


def test_same_player_rush_volume_and_touchdown_is_correlated():
    volume = _leg(player_id="p1", market="rushing_yards", side="over")
    score = _leg(player_id="p1", market="rushing_tds", side="over")
    findings = detect_correlations([volume, score])
    assert findings[0]["kind"] == "rb_volume_and_touchdown"


def test_favorite_and_total_over_same_game_is_correlated():
    favorite = _leg(game_id="g1", market="moneyline", side="home")
    total = _leg(game_id="g1", market="total", side="over")
    findings = detect_correlations([favorite, total])
    assert findings[0]["kind"] == "favorite_and_game_total_over"


def test_rb_rushing_volume_and_game_total_over_is_correlated():
    volume = _leg(game_id="g1", player_id="p1", market="rushing_yards", side="over")
    total = _leg(game_id="g1", market="total", side="over")
    findings = detect_correlations([volume, total])
    assert findings[0]["kind"] == "rb_volume_and_game_total"
    assert findings[0]["direction"] == "positive"


def test_rb_rushing_volume_and_game_total_finds_pattern_regardless_of_leg_order():
    volume = _leg(game_id="g1", player_id="p1", market="rushing_yards", side="over")
    total = _leg(game_id="g1", market="total", side="over")
    assert detect_correlations([total, volume])[0]["kind"] == "rb_volume_and_game_total"


def test_rb_rushing_under_and_total_over_is_not_correlated():
    # Different directions -- the shared-game-script rationale only holds
    # when both legs point the same way.
    volume = _leg(game_id="g1", player_id="p1", market="rushing_yards", side="under")
    total = _leg(game_id="g1", market="total", side="over")
    assert detect_correlations([volume, total]) == []


def test_rb_rushing_volume_different_games_is_not_correlated():
    volume = _leg(game_id="g1", player_id="p1", market="rushing_yards", side="over")
    total = _leg(game_id="g2", market="total", side="over")
    assert detect_correlations([volume, total]) == []


def test_unrelated_legs_are_not_correlated():
    a = _leg(team="KC", market="passing_yards", side="over")
    b = _leg(team="BUF", market="rushing_yards", side="under")
    assert detect_correlations([a, b]) == []


def test_three_leg_parlay_finds_all_pairs():
    qb = _leg(team="KC", market="passing_yards", side="over")
    wr = _leg(team="KC", market="receiving_yards", side="over")
    unrelated = _leg(team="SF", market="rushing_yards", side="over")
    findings = detect_correlations([qb, wr, unrelated])
    assert len(findings) == 1
    assert findings[0]["legs"] == (0, 1)


# --- probability math ---------------------------------------------------------


def test_naive_joint_probability_is_the_product():
    legs = [_leg(model_probability=0.5), _leg(model_probability=0.4)]
    assert naive_joint_probability(legs) == pytest.approx(0.2)


def test_correlation_adjustment_increases_probability_for_positive_correlation():
    qb = _leg(team="KC", market="passing_yards", side="over", model_probability=0.55)
    wr = _leg(team="KC", market="receiving_yards", side="over", model_probability=0.52)
    result = correlation_adjusted_probability([qb, wr])
    assert result["adjusted_probability"] > result["naive_probability"]
    assert len(result["correlations_detected"]) == 1


def test_correlation_adjustment_never_exceeds_the_smallest_leg_probability():
    # Even with a big adjustment, joint probability can't exceed any single leg's.
    qb = _leg(team="KC", market="passing_yards", side="over", model_probability=0.9)
    wr = _leg(team="KC", market="receiving_yards", side="over", model_probability=0.85)
    result = correlation_adjusted_probability([qb, wr])
    assert result["adjusted_probability"] <= min(0.9, 0.85)


def test_uncorrelated_legs_adjusted_equals_naive():
    a = _leg(model_probability=0.5)
    b = _leg(model_probability=0.4)
    result = correlation_adjusted_probability([a, b])
    assert result["adjusted_probability"] == result["naive_probability"]


# --- payout / EV ---------------------------------------------------------------


def test_parlay_decimal_odds_is_product_of_leg_decimal_odds():
    legs = [_leg(price=-110), _leg(price=-110)]
    # -110 decimal = 1.9090909...
    assert parlay_decimal_odds(legs) == pytest.approx(1.9090909 * 1.9090909, rel=1e-4)


def test_evaluate_parlay_requires_at_least_two_legs():
    with pytest.raises(ValueError):
        evaluate_parlay([_leg()])


def test_evaluate_parlay_positive_ev_when_true_probability_beats_market():
    legs = [_leg(model_probability=0.60, price=-110), _leg(model_probability=0.60, price=-110)]
    result = evaluate_parlay(legs)
    assert result["adjusted_ev"] > 0


def test_evaluate_parlay_confidence_bounded():
    legs = [_leg(confidence=0.9), _leg(confidence=0.8), _leg(confidence=0.7), _leg(confidence=0.6)]
    result = evaluate_parlay(legs)
    assert 0.0 <= result["confidence"] <= 1.0


def test_evaluate_parlay_more_legs_reduces_confidence_all_else_equal():
    two_legs = evaluate_parlay([_leg(confidence=0.8), _leg(confidence=0.8)])
    four_legs = evaluate_parlay([_leg(confidence=0.8)] * 4)
    assert four_legs["confidence"] < two_legs["confidence"]


def test_evaluate_parlay_risk_tier_present():
    result = evaluate_parlay([_leg(), _leg()])
    assert result["risk_tier"] in {"low", "medium", "high"}


# --- ranking -----------------------------------------------------------------


def test_rank_parlays_sorted_by_ev_descending():
    strong = [_leg(model_probability=0.65, price=120), _leg(model_probability=0.65, price=120)]
    weak = [_leg(model_probability=0.30, price=-110), _leg(model_probability=0.30, price=-110)]
    ranked = rank_parlays([weak, strong])
    assert ranked[0]["adjusted_ev"] >= ranked[1]["adjusted_ev"]


def test_rank_parlays_deterministic():
    a_legs = [_leg(model_probability=0.5), _leg(model_probability=0.5)]
    b_legs = [_leg(model_probability=0.4), _leg(model_probability=0.4)]
    first = rank_parlays([a_legs, b_legs])
    second = rank_parlays([a_legs, b_legs])
    assert first == second
