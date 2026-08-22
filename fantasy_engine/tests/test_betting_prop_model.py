"""Tests for the player prop model: distributions, probabilities, and full evaluation."""

from __future__ import annotations

import math

import pytest

from betting.odds_generator import _round_to_half, generate_default_props
from betting.prop_model import evaluate_prop, evaluate_props, over_under_probability, stat_distribution


def _player(**overrides) -> dict:
    player = {
        "player_id": "p1",
        "name": "Test Player",
        "position": "RB",
        "team": "KC",
        "games_played": 16,
        "rushing_yards": 960.0,  # 60/game
        "rushing_tds": 8.0,  # 0.5/game
        "receptions": 32.0,  # 2/game
        "receiving_yards": 240.0,
        "projection": 200.0,
        "expected_fantasy_points": 200.0,
        "floor": 140.0,
        "median": 200.0,
        "ceiling": 280.0,
        "adp": 24.0,
    }
    player.update(overrides)
    return player


# --- stat_distribution: family selection -----------------------------------


def test_yardage_market_uses_gaussian_family():
    dist = stat_distribution(_player(), "rushing_yards")
    assert dist["family"] == "gaussian"
    assert dist["mean"] == pytest.approx(60.0)


def test_td_market_uses_poisson_family():
    dist = stat_distribution(_player(), "rushing_tds")
    assert dist["family"] == "poisson"
    assert dist["mean"] == pytest.approx(0.5)


def test_reception_market_uses_poisson_family():
    dist = stat_distribution(_player(), "receptions")
    assert dist["family"] == "poisson"


def test_poisson_stdev_is_sqrt_of_mean_not_mean_times_cv():
    dist = stat_distribution(_player(rushing_tds=8.0, games_played=16), "rushing_tds")
    assert dist["stdev"] == pytest.approx(math.sqrt(0.5), rel=1e-3)


def test_unsupported_market_raises():
    with pytest.raises(ValueError):
        stat_distribution(_player(), "not_a_real_market")


def test_zero_games_played_raises():
    with pytest.raises(ValueError):
        stat_distribution(_player(games_played=0), "rushing_yards")


# --- over_under_probability: correctness ------------------------------------


def test_poisson_over_under_matches_closed_form_for_zero_boundary():
    # mean=0.5 TDs/game, line=0.5 -> P(under) = P(X=0) = exp(-0.5).
    result = over_under_probability(_player(rushing_tds=8.0, games_played=16), "rushing_tds", 0.5)
    assert result["probability_under"] == pytest.approx(math.exp(-0.5), rel=1e-3)


def test_gaussian_over_under_symmetric_at_the_mean():
    player = _player(rushing_yards=960.0, games_played=16)  # mean = 60.0
    result = over_under_probability(player, "rushing_yards", 60.0)
    assert result["probability_over"] == pytest.approx(0.5, abs=0.01)


def test_probabilities_always_sum_to_one():
    for market, line in (("rushing_yards", 55.5), ("rushing_tds", 0.5), ("receptions", 2.5)):
        result = over_under_probability(_player(), market, line)
        assert result["probability_over"] + result["probability_under"] == pytest.approx(1.0)


# --- evaluate_prop / evaluate_props: full pipeline --------------------------


def test_evaluate_prop_returns_bounded_probabilities_and_consistent_recommendation():
    prop_odds = {"market": "rushing_yards", "line": 55.5, "over_price": -110, "under_price": -110, "source": "default"}
    result = evaluate_prop(_player(), prop_odds)
    assert 0.0 <= result["model_probability_over"] <= 1.0
    assert 0.0 <= result["model_probability_under"] <= 1.0
    assert result["recommended_side"] in {"over", "under"}
    assert result["risk_tier"] in {"low", "medium", "high"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_evaluate_prop_ev_sign_matches_edge_sign_direction():
    # A line well below the model's mean should favor "over" with positive edge.
    prop_odds = {"market": "rushing_yards", "line": 30.0, "over_price": -110, "under_price": -110, "source": "default"}
    result = evaluate_prop(_player(), prop_odds)  # mean = 60.0, line = 30.0 -> over is heavily favored
    assert result["recommended_side"] == "over"
    assert result["recommended_edge"] > 0
    assert result["recommended_ev"] > 0


def test_evaluate_props_skips_players_not_in_pool():
    odds = {
        "player_props": {
            "p1:rushing_yards": {"player_id": "p1", "market": "rushing_yards", "line": 55.5, "over_price": -110, "under_price": -110},
            "unknown:rushing_yards": {"player_id": "unknown", "market": "rushing_yards", "line": 10.0, "over_price": -110, "under_price": -110},
        }
    }
    results = evaluate_props({"p1": _player()}, odds)
    assert len(results) == 1
    assert results[0]["player_id"] == "p1"


def test_evaluate_props_skips_unmodeled_markets():
    odds = {"player_props": {"p1:kick_return_yards": {"player_id": "p1", "market": "kick_return_yards", "line": 10.0, "over_price": -110, "under_price": -110}}}
    assert evaluate_props({"p1": _player()}, odds) == []


def test_evaluate_props_sorted_by_edge_descending():
    results = evaluate_props(
        {"p1": _player()},
        {
            "player_props": {
                "p1:rushing_yards": {"player_id": "p1", "market": "rushing_yards", "line": 30.0, "over_price": -110, "under_price": -110},
                "p1:receptions": {"player_id": "p1", "market": "receptions", "line": 2.0, "over_price": -110, "under_price": -110},
            }
        },
    )
    edges = [r["recommended_edge"] for r in results]
    assert edges == sorted(edges, reverse=True)


# --- odds_generator: line quality -------------------------------------------


def test_round_to_half_never_returns_a_whole_number():
    for value in (0.0, 0.294, 0.5, 0.9, 1.0, 1.5, 60.0, 60.4, 99.9):
        line = _round_to_half(value)
        assert not float(line).is_integer(), f"{value} rounded to whole number {line}"


def test_generate_default_props_has_no_whole_number_lines():
    rows = generate_default_props(season=2025, top_n_per_position=5)
    whole = [row for row in rows if float(row["line"]).is_integer()]
    assert whole == []


def test_generate_default_props_filters_near_zero_rate_markets():
    # A WR's real rushing_yards rate is almost always near zero -- that
    # market should not appear for most receivers even though it's in
    # _MARKETS_BY_POSITION for the rare gadget-play WR who does carry it.
    rows = generate_default_props(season=2025, top_n_per_position=15)
    wr_rushing = [row for row in rows if row["market"] == "rushing_yards" and row["team"]]
    # Every row that made it through must reflect a real, non-trivial rate --
    # spot check none of them have an absurdly low implied per-game rate.
    for row in wr_rushing:
        assert row["line"] >= 2.5


def test_generate_default_props_is_deterministic():
    a = generate_default_props(season=2025, top_n_per_position=10)
    b = generate_default_props(season=2025, top_n_per_position=10)
    assert a == b
