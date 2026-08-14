"""Tests for the forward-projection layer.

The point of every test here is that a "projection" is a statement about the
*next* season, not a relabelled copy of the last one -- so the assertions are
mostly about the number having genuinely moved, and moved for a stated reason.
"""

from __future__ import annotations

import datetime as dt

import pytest

from fantasy.data_loader import latest_completed_season
from fantasy.models import LeagueSettings
from fantasy.projections import (
    MARKET_RECONCILIATION_WEIGHT,
    load_forward_projections,
    project_forward,
    projected_points,
    projection_season_label,
    upcoming_season,
)


def _rb(player_id, points, games, adp=None, **extra):
    """One prior-season RB line, in load_real_projections' shape."""
    player = {
        "player_id": player_id,
        "name": player_id.upper(),
        "position": "RB",
        "team": "SF",
        "season": 2025,
        "projection": float(points),
        "expected_fantasy_points": float(points),
        "games_played": int(games),
        "points_per_game": round(points / games, 2),
        "scoring_mode": "ppr",
    }
    if adp is not None:
        player["adp"] = float(adp)
    player.update(extra)
    return player


def _pool():
    """Three RBs at the same rate but very different availability, plus two WRs."""
    return [
        _rb("rb_full", 255.0, 17),  # 15.0 ppg over a full season
        _rb("rb_short", 90.0, 6),  # 15.0 ppg, but only six games
        _rb("rb_low", 102.0, 17),  # 6.0 ppg over a full season
        {
            "player_id": "wr1",
            "name": "WR1",
            "position": "WR",
            "team": "KC",
            "season": 2025,
            "projection": 240.0,
            "expected_fantasy_points": 240.0,
            "games_played": 17,
            "points_per_game": 14.12,
            "scoring_mode": "ppr",
        },
        {
            "player_id": "wr2",
            "name": "WR2",
            "position": "WR",
            "team": "KC",
            "season": 2025,
            "projection": 170.0,
            "expected_fantasy_points": 170.0,
            "games_played": 17,
            "points_per_game": 10.0,
            "scoring_mode": "ppr",
        },
    ]


def _by_id(players):
    return {player["player_id"]: player for player in players}


# --- season helpers -----------------------------------------------------------


def test_upcoming_season_is_one_past_the_last_completed_one():
    assert upcoming_season(dt.date(2026, 8, 1)) == latest_completed_season(dt.date(2026, 8, 1)) + 1
    assert upcoming_season(dt.date(2026, 8, 1)) == 2026
    # Before March the previous season is still being finished, so the season
    # being drafted for is the one about to start that autumn.
    assert upcoming_season(dt.date(2026, 1, 15)) == 2025


def test_projection_season_label_reads_as_a_ui_label():
    assert projection_season_label(2026) == "2026 Projections"
    assert projection_season_label().endswith("Projections")


# --- projected_points ---------------------------------------------------------


def test_projected_points_prefers_a_precomputed_projection():
    player = _rb("rb", 255.0, 17)
    assert projected_points(player, {"scoring_mode": "ppr"}) == pytest.approx(255.0)


def test_projected_points_refuses_a_projection_from_another_scoring_mode():
    """A PPR total must never be reported as a standard-league number."""
    player = _rb("rb", 255.0, 17)
    assert projected_points(player, {"scoring_mode": "standard"}) is None


def test_projected_points_refuses_a_baked_projection_under_custom_rules():
    player = _rb("rb", 255.0, 17)
    settings = LeagueSettings(scoring_mode="ppr", custom_rules={"multipliers": {"receptions": 2.0}})
    assert projected_points(player, settings) is None


def test_projected_points_returns_none_for_a_bare_stat_line():
    assert projected_points({"player_id": "x", "position": "RB", "rushing_yards": 900}) is None


def test_projected_points_accepts_a_pool_with_no_scoring_mode_recorded():
    """A ranked board loses `scoring_mode` through the adapter; it must still score."""
    assert projected_points({"expected_fantasy_points": 120.0}, {"scoring_mode": "ppr"}) == pytest.approx(120.0)


# --- project_forward ----------------------------------------------------------


def test_project_forward_leaves_the_input_pool_untouched():
    pool = _pool()
    before = [dict(player) for player in pool]
    project_forward(pool, target_season=2026)
    assert pool == before


def test_project_forward_stamps_the_target_season_on_every_player():
    projected = project_forward(_pool(), target_season=2026)
    assert {player["projection_season"] for player in projected} == {2026}
    assert {player["season"] for player in projected} == {2026}
    assert all(player["prior_season"] == 2025 for player in projected)
    assert all("2026 projection from 2025 actuals" in player["projection_basis"] for player in projected)


def test_project_forward_scales_a_short_season_up_to_a_full_one():
    """Six games at 15 ppg is not a 90-point player -- that is the core fix."""
    projected = _by_id(project_forward(_pool(), target_season=2026))
    short = projected["rb_short"]
    assert short["prior_season_points"] == pytest.approx(90.0)
    assert short["projection"] > 90.0
    assert short["expected_games"] > 6


def test_project_forward_regresses_availability_rather_than_replaying_it():
    projected = _by_id(project_forward(_pool(), target_season=2026))
    # A full-season player is not assumed to repeat 17 games...
    assert projected["rb_full"]["expected_games"] < 17
    # ...and a six-game player is not assumed to repeat six either.
    assert projected["rb_short"]["expected_games"] > 6
    assert projected["rb_short"]["expected_games"] < projected["rb_full"]["expected_games"]


def test_project_forward_regresses_a_small_sample_rate_toward_the_position():
    """A three-game outlier must not project as a full season of that rate."""
    pool = _pool() + [_rb("rb_spike", 90.0, 3)]  # 30.0 ppg on three games
    projected = _by_id(project_forward(pool, target_season=2026))
    spike = projected["rb_spike"]
    assert spike["points_per_game"] < 30.0
    # Still ahead of the position's median rate -- regressed, not erased.
    assert spike["points_per_game"] > projected["rb_low"]["points_per_game"]


def test_project_forward_without_market_weight_is_pure_rate_projection():
    pool = [_rb("rb_low", 102.0, 17, adp=1.0), _rb("rb_full", 255.0, 17, adp=100.0)]
    rate_only = _by_id(project_forward(pool, target_season=2026, market_weight=0.0))
    assert rate_only["rb_full"]["projection"] > rate_only["rb_low"]["projection"]


def test_market_reconciliation_lifts_a_player_the_market_rates_far_higher():
    """ADP is forward-looking information the box score cannot see."""
    pool = [_rb("rb_low", 102.0, 17, adp=1.0), _rb("rb_full", 255.0, 17, adp=100.0)]
    rate_only = _by_id(project_forward(pool, target_season=2026, market_weight=0.0))
    blended = _by_id(project_forward(pool, target_season=2026, market_weight=0.6))
    assert blended["rb_low"]["projection"] > rate_only["rb_low"]["projection"]
    assert blended["rb_full"]["projection"] < rate_only["rb_full"]["projection"]


def test_market_reconciliation_leaves_players_with_no_adp_on_their_rate():
    pool = [_rb("rb_full", 255.0, 17, adp=2.0), _rb("rb_short", 90.0, 6)]
    rate_only = _by_id(project_forward(pool, target_season=2026, market_weight=0.0))
    blended = _by_id(project_forward(pool, target_season=2026, market_weight=0.9))
    assert blended["rb_short"]["projection"] == pytest.approx(rate_only["rb_short"]["projection"])


def test_project_forward_rescales_the_confidence_band_with_the_projection():
    pool = [_rb("rb_short", 90.0, 6, floor=50.0, median=88.0, ceiling=140.0)]
    projected = project_forward(pool, target_season=2026)[0]
    ratio = projected["projection"] / 90.0
    assert projected["floor"] == pytest.approx(50.0 * ratio, rel=1e-3)
    assert projected["ceiling"] == pytest.approx(140.0 * ratio, rel=1e-3)
    assert projected["floor"] < projected["median"] < projected["ceiling"]


def test_project_forward_reports_lower_confidence_for_a_smaller_sample():
    projected = _by_id(project_forward(_pool(), target_season=2026))
    assert projected["rb_short"]["projection_confidence"] < projected["rb_full"]["projection_confidence"]
    assert 0.0 <= projected["rb_short"]["projection_confidence"] <= 1.0


def test_project_forward_reports_higher_confidence_when_the_market_has_an_opinion():
    pool = [_rb("with_adp", 200.0, 17, adp=10.0), _rb("no_adp", 200.0, 17)]
    projected = _by_id(project_forward(pool, target_season=2026))
    assert projected["with_adp"]["projection_confidence"] > projected["no_adp"]["projection_confidence"]


def test_project_forward_returns_players_sorted_best_first():
    projected = project_forward(_pool(), target_season=2026)
    values = [player["projection"] for player in projected]
    assert values == sorted(values, reverse=True)


def test_project_forward_handles_an_empty_pool():
    assert project_forward([], target_season=2026) == []


def test_project_forward_survives_a_player_with_no_prior_points():
    pool = [{"player_id": "x", "name": "X", "position": "TE", "team": "DAL", "season": 2025}]
    projected = project_forward(pool, target_season=2026)
    assert projected[0]["projection_season"] == 2026
    assert projected[0]["projection"] >= 0.0


def test_project_forward_defaults_to_the_upcoming_season():
    projected = project_forward(_pool())
    assert projected[0]["projection_season"] == upcoming_season()


def test_market_reconciliation_weight_is_a_blend_not_a_substitution():
    """The constant must stay a blend -- at 1.0 it would double-count ADP."""
    assert 0.0 < MARKET_RECONCILIATION_WEIGHT < 1.0


def test_load_forward_projections_projects_what_the_loader_returned(monkeypatch):
    """The loader is wrapped, not reimplemented -- and its output is projected forward."""
    captured: dict = {}

    def _fake_load(**kwargs):
        captured.update(kwargs)
        return _pool()

    monkeypatch.setattr("fantasy.projections.load_real_projections", _fake_load)
    players = load_forward_projections(season=2025, target_season=2026, scoring_mode="ppr")

    assert captured["season"] == 2025
    assert captured["scoring_mode"] == "ppr"
    assert {player["projection_season"] for player in players} == {2026}
