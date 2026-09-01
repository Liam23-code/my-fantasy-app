"""betting.prop_model honours the live player-status overlay."""

from __future__ import annotations

import json

import pytest

from betting.prop_model import evaluate_prop, evaluate_props
from fantasy import player_status as ps


@pytest.fixture()
def status(tmp_path, monkeypatch):
    path = tmp_path / "player_status.json"
    monkeypatch.setattr(ps, "STATUS_PATH", path)
    ps.clear_cache()

    def _set(mapping):
        path.write_text(json.dumps(mapping), encoding="utf-8")
        ps.clear_cache()

    yield _set
    ps.clear_cache()


def _player(pid="p1", name="Test Player", **overrides):
    player = {
        "player_id": pid,
        "name": name,
        "position": "RB",
        "team": "KC",
        "games_played": 16,
        "rushing_yards": 960.0,   # 60 / game
        "rushing_tds": 8.0,
        "receptions": 32.0,
        "receiving_yards": 240.0,
        "projection": 200.0,
        "expected_fantasy_points": 200.0,
        "floor": 140.0,
        "median": 200.0,
        "ceiling": 280.0,
    }
    player.update(overrides)
    return player


def _prop(pid="p1", market="rushing_yards", line=55.5):
    return {"player_id": pid, "market": market, "line": line, "over_price": -110, "under_price": -110}


# --- evaluate_props: OUT is removed entirely ------------------------------------


def test_out_players_are_removed_from_the_prop_board(status):
    status({"p1": {"status": "OUT"}})
    odds = {"player_props": {"p1:rushing_yards": _prop("p1"), "p2:rushing_yards": _prop("p2")}}
    results = evaluate_props({"p1": _player("p1"), "p2": _player("p2", "Healthy Guy")}, odds)
    assert {r["player_id"] for r in results} == {"p2"}


def test_no_overlay_leaves_the_board_unchanged():
    odds = {"player_props": {"p1:rushing_yards": _prop("p1")}}
    results = evaluate_props({"p1": _player("p1")}, odds)
    assert [r["player_id"] for r in results] == ["p1"]


# --- evaluate_prop: projection zeroed / trimmed -------------------------------


@pytest.mark.parametrize("flag", ["HOLDOUT", "SUSPENDED"])
def test_holdout_and_suspended_zero_the_projection(status, flag):
    status({"p1": {"status": flag}})
    result = evaluate_prop(_player("p1"), _prop("p1", line=55.5))
    assert result["status"] == flag
    assert result["distribution"]["mean"] == 0.0
    assert result["model_probability_over"] == 0.0  # no chance of clearing a positive line
    assert result["model_probability_under"] == 1.0


def test_doubtful_trims_the_projection_but_keeps_the_prop(status):
    healthy = evaluate_prop(_player("p1"), _prop("p1", line=55.5))
    status({"p1": {"status": "DOUBTFUL"}})
    shaky = evaluate_prop(_player("p1"), _prop("p1", line=55.5))
    assert shaky["status"] == "DOUBTFUL"
    assert 0.0 < shaky["distribution"]["mean"] < healthy["distribution"]["mean"]
    assert shaky["model_probability_over"] < healthy["model_probability_over"]


def test_questionable_is_a_mild_trim(status):
    healthy = evaluate_prop(_player("p1"), _prop("p1", line=55.5))
    status({"p1": {"status": "QUESTIONABLE"}})
    q = evaluate_prop(_player("p1"), _prop("p1", line=55.5))
    assert healthy["distribution"]["mean"] * 0.80 < q["distribution"]["mean"] < healthy["distribution"]["mean"]


def test_healthy_player_is_untouched(status):
    status({"other": {"status": "OUT"}})
    healthy = evaluate_prop(_player("p1"), _prop("p1", line=55.5))
    assert healthy["status"] == "HEALTHY"
    assert healthy["distribution"]["mean"] == pytest.approx(60.0)
