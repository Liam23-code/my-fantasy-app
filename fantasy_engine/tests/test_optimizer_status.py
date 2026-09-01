"""fantasy.optimizer honours the live player-status overlay (DFS pool)."""

from __future__ import annotations

import json

import pytest

from fantasy import player_status as ps
from fantasy.optimizer import optimize_lineup

SETTINGS = {
    "n_teams": 10,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 0, "K": 0, "BENCH": 3},
    "flex_eligible": ["RB", "WR", "TE"],
}


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


def _player(pid, name, position, team="AAA"):
    return {"player_id": pid, "name": name, "position": position, "nfl_team": team, "slot": "BENCH", "injury_status": None}


def _proj(pid, name, position, field, value):
    return {"player_id": pid, "name": name, "position": position, field: value}


@pytest.fixture()
def roster():
    return [
        _player("qb1", "QB1", "QB"),
        _player("rb1", "RB1", "RB"), _player("rb2", "RB2", "RB"), _player("rb3", "RB3", "RB"),
        _player("wr1", "WR1", "WR"), _player("wr2", "WR2", "WR"), _player("wr3", "WR3", "WR"),
        _player("te1", "TE1", "TE"), _player("te2", "TE2", "TE"),
    ]


@pytest.fixture()
def projections():
    return [
        _proj("qb1", "QB1", "QB", "passing_yards", 200),  # 8.0
        _proj("rb1", "RB1", "RB", "rushing_yards", 90),   # 9.0  (best RB)
        _proj("rb2", "RB2", "RB", "rushing_yards", 70),   # 7.0
        _proj("rb3", "RB3", "RB", "rushing_yards", 30),   # 3.0
        _proj("wr1", "WR1", "WR", "receiving_yards", 95), # 9.5
        _proj("wr2", "WR2", "WR", "receiving_yards", 60), # 6.0
        _proj("wr3", "WR3", "WR", "receiving_yards", 20), # 2.0
        _proj("te1", "TE1", "TE", "receiving_yards", 80), # 8.0
        _proj("te2", "TE2", "TE", "receiving_yards", 10), # 1.0
    ]


def _names(lineup, key):
    return {row["name"] for row in lineup[key]}


def test_no_overlay_is_unchanged(roster, projections):
    lineup = optimize_lineup(roster, projections, SETTINGS)
    assert "RB1" in _names(lineup, "starters")


@pytest.mark.parametrize("flag", ["OUT", "HOLDOUT", "SUSPENDED"])
def test_unavailable_players_are_dropped_from_the_pool(status, roster, projections, flag):
    status({"rb1": {"status": flag}})
    lineup = optimize_lineup(roster, projections, SETTINGS)
    assert "RB1" not in _names(lineup, "starters")
    assert "RB2" in _names(lineup, "starters")  # next best RB starts instead
    # RB1 is benched with a zeroed projection
    rb1 = next(row for row in lineup["bench"] if row["name"] == "RB1")
    assert rb1["points"] == 0.0
    assert rb1["status"] == flag


def test_an_explicit_lock_overrides_the_overlay(status, roster, projections):
    status({"rb1": {"status": "OUT"}})
    lineup = optimize_lineup(roster, projections, SETTINGS, constraints={"locked_player_ids": ["rb1"]})
    assert "RB1" in _names(lineup, "starters")
    rb1 = next(row for row in lineup["starters"] if row["name"] == "RB1")
    assert rb1["points"] == pytest.approx(9.0)  # projection not zeroed for a locked player


def test_doubtful_downweights_the_projection_but_keeps_the_player_eligible(status, roster, projections):
    healthy = optimize_lineup(roster, projections, SETTINGS)
    rb1_healthy = next(r for r in healthy["starters"] if r["name"] == "RB1")

    status({"rb1": {"status": "DOUBTFUL"}})
    shaky = optimize_lineup(roster, projections, SETTINGS)
    rb1_shaky = next(
        r for r in shaky["starters"] + shaky["bench"] if r["name"] == "RB1"
    )
    # still selectable (not force-excluded), but on a trimmed projection
    assert rb1_shaky["status"] == "DOUBTFUL"
    assert 0.0 < rb1_shaky["points"] < rb1_healthy["points"]
    assert rb1_shaky["points"] == pytest.approx(9.0 * 0.45, abs=0.01)
