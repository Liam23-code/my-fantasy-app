"""Unit tests for fantasy.pipeline."""

from __future__ import annotations

import json

import pytest

from fantasy.models import LeagueSettings
from fantasy.pipeline import diff_snapshots, load_snapshot, persist_snapshot, update_weekly

SETTINGS = {
    "n_teams": 4,
    "scoring_mode": "ppr",
    "roster_requirements": {"RB": 1, "WR": 1, "FLEX": 0, "QB": 0, "TE": 0, "DST": 0, "K": 0, "BENCH": 0},
    "flex_eligible": [],
}


def _player(pid, name, position, points_field, value, **extra):
    return {"player_id": pid, "name": name, "position": position, points_field: value, **extra}


def _week1_projections():
    return [
        _player("rb1", "RB1", "RB", "rushing_yards", 80),
        _player("wr1", "WR1", "WR", "receiving_yards", 60),
    ]


def test_update_weekly_returns_ranked_players_and_snapshot(tmp_path):
    result = update_weekly(_week1_projections, SETTINGS, week=1, snapshot_dir=tmp_path)
    assert len(result["ranked_players"]) == 2
    assert result["movers"] == []  # no prior week to diff against
    assert result["snapshot_path"] == str(tmp_path / "week_01.json")
    assert (tmp_path / "week_01.json").exists()
    assert "lineup" not in result
    assert "waiver_recommendations" not in result


def test_update_weekly_includes_lineup_and_advice_when_roster_given(tmp_path):
    roster = [{"player_id": "rb1", "name": "RB1", "position": "RB", "slot": "RB"}]
    result = update_weekly(_week1_projections, SETTINGS, week=1, my_roster=roster, snapshot_dir=tmp_path)
    assert "lineup" in result
    assert "start_sit_advice" in result
    assert result["lineup"]["starters"][0]["name"] == "RB1"


def test_update_weekly_includes_waiver_recommendations_when_source_given(tmp_path):
    def free_agents():
        return [_player("fa1", "Free Agent", "WR", "receiving_yards", 40)]

    result = update_weekly(_week1_projections, SETTINGS, week=1, available_players_source=free_agents, snapshot_dir=tmp_path)
    assert "waiver_recommendations" in result
    assert result["waiver_recommendations"][0]["name"] == "Free Agent"


def test_projection_and_waiver_sources_are_each_called_exactly_once(tmp_path):
    projection_calls = []
    waiver_calls = []

    def projection_source():
        projection_calls.append(1)
        return _week1_projections()

    def waiver_source():
        waiver_calls.append(1)
        return [_player("fa1", "FA", "WR", "receiving_yards", 30)]

    update_weekly(projection_source, SETTINGS, week=1, available_players_source=waiver_source, snapshot_dir=tmp_path)
    assert len(projection_calls) == 1
    assert len(waiver_calls) == 1


def test_movers_reflect_point_deltas_between_consecutive_weeks(tmp_path):
    update_weekly(_week1_projections, SETTINGS, week=1, snapshot_dir=tmp_path)

    def week2_projections():
        return [
            _player("rb1", "RB1", "RB", "rushing_yards", 120),  # improved
            _player("wr1", "WR1", "WR", "receiving_yards", 60),  # unchanged
        ]

    result = update_weekly(week2_projections, SETTINGS, week=2, snapshot_dir=tmp_path)
    movers = {m["player_id"]: m for m in result["movers"]}
    assert "rb1" in movers
    assert movers["rb1"]["delta"] > 0
    assert "wr1" not in movers  # unchanged points -> not a mover


def test_update_weekly_accepts_league_settings_model_instance(tmp_path):
    result = update_weekly(_week1_projections, LeagueSettings(**SETTINGS), week=1, snapshot_dir=tmp_path)
    assert len(result["ranked_players"]) == 2


def test_persist_and_load_snapshot_round_trip(tmp_path):
    players = [{"player_id": "a", "name": "A", "position": "RB", "points": 12.5}]
    path = persist_snapshot(1, players, snapshot_dir=tmp_path)
    loaded = load_snapshot(1, snapshot_dir=tmp_path)
    assert loaded == {"a": {"name": "A", "position": "RB", "points": 12.5}}
    assert json.loads(path.read_text(encoding="utf-8")) == loaded


def test_load_snapshot_missing_week_returns_none(tmp_path):
    assert load_snapshot(99, snapshot_dir=tmp_path) is None


def test_diff_snapshots_with_no_previous_returns_empty_list():
    assert diff_snapshots({"a": {"name": "A", "position": "RB", "points": 10}}, None) == []


def test_diff_snapshots_ignores_players_missing_from_previous():
    current = {"a": {"name": "A", "position": "RB", "points": 10}, "b": {"name": "B", "position": "WR", "points": 5}}
    previous = {"a": {"name": "A", "position": "RB", "points": 8}}
    movers = diff_snapshots(current, previous)
    assert len(movers) == 1
    assert movers[0]["player_id"] == "a"
    assert movers[0]["delta"] == pytest.approx(2.0)


def test_diff_snapshots_sorted_by_absolute_delta_descending():
    current = {"a": {"name": "A", "position": "RB", "points": 10}, "b": {"name": "B", "position": "WR", "points": 5}}
    previous = {"a": {"name": "A", "position": "RB", "points": 9}, "b": {"name": "B", "position": "WR", "points": 15}}
    movers = diff_snapshots(current, previous)
    assert [m["player_id"] for m in movers] == ["b", "a"]  # |-10| > |1|
