"""Tests for persistent weekly management of a drafted fantasy team."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import fantasy.my_team_manager as manager
from fantasy.draft import simulate_draft

WEEK = 4
SETTINGS = {
    "n_teams": 1,
    "scoring_mode": "ppr",
    "roster_requirements": {
        "QB": 0,
        "RB": 1,
        "WR": 0,
        "TE": 0,
        "FLEX": 0,
        "DST": 0,
        "K": 0,
        "BENCH": 2,
    },
    "flex_eligible": ["RB", "WR", "TE"],
}


def _weekly(points: float, confidence: float = 0.8, bye_week: int | None = None) -> dict:
    return {
        week: {
            "points": 0.0 if week == bye_week else points,
            "confidence": 1.0 if week == bye_week else confidence,
        }
        for week in range(1, 19)
    }


def _player(
    player_id: str,
    name: str,
    projection: float,
    weekly_points: float,
    *,
    slot: str = "BENCH",
    confidence: float = 0.8,
    bye_week: int | None = None,
    injury_status: str | None = None,
) -> dict:
    return {
        "player_id": player_id,
        "name": name,
        "position": "RB",
        "team": "DEN",
        "slot": slot,
        "projection": projection,
        "expected_fantasy_points": projection,
        "projection_confidence": confidence,
        "scoring_mode": "ppr",
        "bye_week": bye_week,
        "injury_status": injury_status,
        "weekly_projection": _weekly(weekly_points, confidence, bye_week),
    }


@pytest.fixture
def team() -> dict:
    return {
        "league_settings": SETTINGS,
        "players": [
            _player("low", "Low Starter", 170.0, 10.0, slot="RB", confidence=0.65),
            _player("high", "High Bench", 255.0, 20.0, slot="BENCH", confidence=0.90),
        ],
    }


def test_team_saving_and_loading_round_trip_without_shape_changes(tmp_path, monkeypatch, team):
    path = tmp_path / "data" / "user_team.json"
    monkeypatch.setattr(manager, "USER_TEAM_PATH", path)

    saved = manager.save_user_team(team)

    assert path.exists()
    assert manager.load_user_team() == saved
    assert manager.load_user_team()["players"][0]["name"] == "Low Starter"


def test_load_user_team_returns_empty_list_when_no_save_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "USER_TEAM_PATH", tmp_path / "missing" / "user_team.json")
    assert manager.load_user_team() == []


def test_weekly_team_projection_uses_optimized_weekly_lineup(team):
    result = manager.weekly_team_projection(team, WEEK)

    assert result["week"] == WEEK
    assert result["total_points"] == pytest.approx(20.0)
    assert [player["name"] for player in result["starters"]] == ["High Bench"]
    assert result["confidence"] == pytest.approx(0.90)
    assert result["quant_total_points"] == pytest.approx(20.0)
    assert 0.0 <= result["quant_confidence"] <= 1.0
    assert result["starters"][0]["rarity_tier"]
    assert result["starters"][0]["trend_direction"] in {"up", "down", "flat"}


def test_weekly_team_projection_falls_back_when_quant_enrichment_is_unavailable(team, monkeypatch):
    monkeypatch.setattr(manager, "_quant_by_player", lambda *args, **kwargs: {})

    result = manager.weekly_team_projection(team, WEEK)

    assert result["quant_total_points"] == result["total_points"] == pytest.approx(20.0)
    assert result["quant_confidence"] == result["confidence"] == pytest.approx(0.90)


def test_lineup_recommendations_swap_the_better_bench_player_in(team):
    swaps = manager.recommend_lineup_swaps(team, WEEK)

    assert swaps
    assert swaps[0]["start"] == "High Bench"
    assert swaps[0]["bench"] == "Low Starter"
    assert swaps[0]["projected_gain"] == pytest.approx(10.0)
    assert swaps[0]["quant_projected_gain"] == pytest.approx(10.0)
    assert "momentum_score" in swaps[0]


def test_waiver_recommendation_adds_an_upgrade_and_names_the_drop(team):
    waiver_pool = [_player("waiver", "Waiver Upgrade", 300.0, 25.0, confidence=0.85)]

    recommendations = manager.recommend_add_drop(team, WEEK, waiver_pool)

    assert recommendations
    assert recommendations[0]["add"] == "Waiver Upgrade"
    assert recommendations[0]["drop"] == "Low Starter"
    assert recommendations[0]["weekly_gain"] == pytest.approx(15.0)
    assert 0.0 <= recommendations[0]["quant_priority_score"] <= 100.0
    assert 0.0 <= recommendations[0]["breakout_probability"] <= 1.0
    assert recommendations[0]["rarity_tier"]


def test_trade_recommendation_targets_rest_of_season_upgrade(team):
    trade_pool = [_player("target", "Trade Target", 340.0, 27.0, confidence=0.88)]

    recommendations = manager.recommend_trades(team, WEEK, trade_pool)

    assert recommendations
    assert recommendations[0]["trade_for"] == "Trade Target"
    assert recommendations[0]["offer"] == "Low Starter"
    assert recommendations[0]["rest_of_season_gain"] > 0
    assert 0.0 <= recommendations[0]["quant_trade_score"] <= 100.0
    assert "fairness_score" in recommendations[0]
    assert recommendations[0]["quant"]["target"]["rarity_tier"]


def test_bench_vs_start_decision_sits_a_player_on_bye():
    player = _player("bye", "Bye Player", 250.0, 18.0, bye_week=WEEK)
    decision = manager.bench_vs_start_decision(player, WEEK)

    assert decision["decision"] == "sit"
    assert decision["points"] == 0.0
    assert decision["opponent"] == "BYE"
    assert decision["quant_decision"] == "sit"
    assert decision["quant_projected_points"] == 0.0


def test_team_confidence_curve_contains_points_and_confidence_for_all_weeks(team):
    curve = manager.team_confidence_curve(team)

    assert set(curve) == set(range(1, 19))
    assert curve[WEEK]["points"] == pytest.approx(20.0)
    assert curve[WEEK]["confidence"] == pytest.approx(0.90)
    assert all(0.0 <= value["confidence"] <= 1.0 for value in curve.values())
    assert curve[WEEK]["quant_points"] == pytest.approx(20.0)
    assert all(0.0 <= value["quant_confidence"] <= 1.0 for value in curve.values())
    assert all(0.0 <= value["volatility"] <= 1.0 for value in curve.values())


def test_team_health_reports_inactive_players(team):
    team["players"][0]["injury_status"] = "OUT"
    health = manager.team_health_status(team)

    assert health["status"] in {"watch", "critical"}
    assert health["available_players"] == 1
    assert health["issues"][0]["player"] == "Low Starter"
    assert 0.0 <= health["quant_health_score"] <= 100.0
    assert health["quant_players"]["low"]["health_multiplier"] == 0.0


def test_completed_user_draft_requires_an_explicit_team_save(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "USER_TEAM_PATH", tmp_path / "user_team.json")
    pool = [
        _player("one", "Back One", 250.0, 16.0),
        _player("two", "Back Two", 220.0, 14.0),
    ]
    settings = {**SETTINGS, "n_teams": 2}

    result = simulate_draft(pool, settings, rounds=1, seed=1, user_draft_slot=1)

    handoff = result["my_team_handoff"]
    assert handoff["saved"] is False
    assert handoff["requires_explicit_save"] is True
    assert handoff["recommended_page"] == "pages/26_Fantasy_Saved_Teams.py"
    assert result["redirect_page"] is None
    assert result["redirect_tab"] is None
    assert not manager.USER_TEAM_PATH.exists()


def test_saved_team_and_my_team_surfaces_have_separate_responsibilities():
    workspace = Path(__file__).resolve().parents[2]
    season_tools = workspace / "UniversalQuantAgent" / "app" / "pages" / "27_Fantasy_Season_Tools.py"
    saved_teams_page = workspace / "UniversalQuantAgent" / "app" / "pages" / "26_Fantasy_Saved_Teams.py"
    my_team_page = workspace / "UniversalQuantAgent" / "app" / "pages" / "28_Fantasy_My_Team.py"

    season_source = season_tools.read_text(encoding="utf-8")
    saved_source = saved_teams_page.read_text(encoding="utf-8")
    my_team_source = my_team_page.read_text(encoding="utf-8")
    for source in (season_source, saved_source, my_team_source):
        ast.parse(source)

    for name in (
        "weekly_team_projection",
        "recommend_add_drop",
        "recommend_trades",
        "recommend_lineup_swaps",
        "team_confidence_curve",
    ):
        assert name in my_team_source

    for name in ("list_saved_teams", "create_new_team_save", "delete_team_save"):
        assert name in saved_source

    for name in (
        "weekly_team_projection",
        "recommend_add_drop",
        "recommend_trades",
        "recommend_lineup_swaps",
        "team_confidence_curve",
    ):
        assert name not in season_source
    assert '"My Team"' not in season_source
