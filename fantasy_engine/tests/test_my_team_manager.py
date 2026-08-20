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


def test_lineup_recommendations_swap_the_better_bench_player_in(team):
    swaps = manager.recommend_lineup_swaps(team, WEEK)

    assert swaps
    assert swaps[0]["start"] == "High Bench"
    assert swaps[0]["bench"] == "Low Starter"
    assert swaps[0]["projected_gain"] == pytest.approx(10.0)


def test_waiver_recommendation_adds_an_upgrade_and_names_the_drop(team):
    waiver_pool = [_player("waiver", "Waiver Upgrade", 300.0, 25.0, confidence=0.85)]

    recommendations = manager.recommend_add_drop(team, WEEK, waiver_pool)

    assert recommendations
    assert recommendations[0]["add"] == "Waiver Upgrade"
    assert recommendations[0]["drop"] == "Low Starter"
    assert recommendations[0]["weekly_gain"] == pytest.approx(15.0)


def test_trade_recommendation_targets_rest_of_season_upgrade(team):
    trade_pool = [_player("target", "Trade Target", 340.0, 27.0, confidence=0.88)]

    recommendations = manager.recommend_trades(team, WEEK, trade_pool)

    assert recommendations
    assert recommendations[0]["trade_for"] == "Trade Target"
    assert recommendations[0]["offer"] == "Low Starter"
    assert recommendations[0]["rest_of_season_gain"] > 0


def test_bench_vs_start_decision_sits_a_player_on_bye():
    player = _player("bye", "Bye Player", 250.0, 18.0, bye_week=WEEK)
    decision = manager.bench_vs_start_decision(player, WEEK)

    assert decision["decision"] == "sit"
    assert decision["points"] == 0.0
    assert decision["opponent"] == "BYE"


def test_team_confidence_curve_contains_points_and_confidence_for_all_weeks(team):
    curve = manager.team_confidence_curve(team)

    assert set(curve) == set(range(1, 19))
    assert curve[WEEK]["points"] == pytest.approx(20.0)
    assert curve[WEEK]["confidence"] == pytest.approx(0.90)
    assert all(0.0 <= value["confidence"] <= 1.0 for value in curve.values())


def test_team_health_reports_inactive_players(team):
    team["players"][0]["injury_status"] = "OUT"
    health = manager.team_health_status(team)

    assert health["status"] in {"watch", "critical"}
    assert health["available_players"] == 1
    assert health["issues"][0]["player"] == "Low Starter"


def test_completed_user_draft_saves_roster_and_returns_redirect(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "USER_TEAM_PATH", tmp_path / "user_team.json")
    pool = [
        _player("one", "Back One", 250.0, 16.0),
        _player("two", "Back Two", 220.0, 14.0),
    ]
    settings = {**SETTINGS, "n_teams": 2}

    result = simulate_draft(pool, settings, rounds=1, seed=1, user_draft_slot=1)

    assert result["my_team_handoff"]["saved"] is True
    assert result["redirect_page"] == "pages/28_Fantasy_My_Team.py"
    assert result["redirect_tab"] == "My Team"
    assert len(manager.load_user_team()) == 1


def test_both_my_team_ui_surfaces_parse_and_use_the_manager():
    workspace = Path(__file__).resolve().parents[2]
    season_tools = workspace / "UniversalQuantAgent" / "app" / "pages" / "27_Fantasy_Season_Tools.py"
    my_team_page = workspace / "UniversalQuantAgent" / "app" / "pages" / "28_Fantasy_My_Team.py"

    for page in (season_tools, my_team_page):
        source = page.read_text(encoding="utf-8")
        ast.parse(source)
        assert "weekly_team_projection" in source
        assert "recommend_add_drop" in source
        assert "recommend_trades" in source
        assert "recommend_lineup_swaps" in source
        assert "team_confidence_curve" in source
    assert '"My Team"' in season_tools.read_text(encoding="utf-8")
