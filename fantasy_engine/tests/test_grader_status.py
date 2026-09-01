"""fantasy.grader honours the live player-status overlay in grade_team."""

from __future__ import annotations

import json

import pytest

from fantasy import player_status as ps
from fantasy.grader import grade_team

SETTINGS = {
    "n_teams": 2,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 3},
    "flex_eligible": [],
}


def _p(pid, position, projection, **extra):
    player = {
        "player_id": pid,
        "name": pid.replace("-", " ").title(),
        "position": position,
        "team": "SF",
        "projection": float(projection),
        "expected_fantasy_points": float(projection),
    }
    player.update(extra)
    return player


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


def _group(report, position):
    return next(g for g in report["positions"] if g["position"] == position)


def test_out_player_is_ignored_by_the_room(status):
    board = [_p("rb-b", "RB", 180), _p("wr-b", "WR", 180), _p("qb-b", "QB", 180)]
    roster = [_p("rb-1", "RB", 240), _p("wr-1", "WR", 200), _p("qb-1", "QB", 300)]

    healthy = grade_team(roster, board, SETTINGS)
    status({"rb-1": {"status": "OUT"}})
    with_out = grade_team(roster, board, SETTINGS)

    # The OUT RB no longer counts as a starter, so the RB room's points drop.
    assert _group(with_out, "RB")["starters"] == []
    assert _group(with_out, "RB")["starter_points"] == 0.0
    assert _group(with_out, "RB")["score"] < _group(healthy, "RB")["score"]
    # roster_size still reflects what the manager actually drafted.
    assert with_out["roster_size"] == 3


def test_questionable_player_lifts_the_risk_component(status):
    board = [_p("rb-b", "RB", 180), _p("wr-b", "WR", 180), _p("qb-b", "QB", 180)]
    roster = [_p("rb-1", "RB", 240), _p("wr-1", "WR", 200), _p("qb-1", "QB", 300)]

    healthy = grade_team(roster, board, SETTINGS)
    status({"wr-1": {"status": "QUESTIONABLE"}})
    shaky = grade_team(roster, board, SETTINGS)

    assert shaky["overall"]["components"]["risk_profile"] < healthy["overall"]["components"]["risk_profile"]
    assert any("injury" in note for note in shaky["overall"]["risk_notes"])


def test_holdout_player_is_removed_from_the_room(status):
    board = [_p("rb-b", "RB", 180), _p("wr-b", "WR", 180), _p("qb-b", "QB", 180)]
    roster = [_p("rb-1", "RB", 240), _p("wr-1", "WR", 200), _p("qb-1", "QB", 300)]
    status({"qb-1": {"status": "HOLDOUT"}})
    report = grade_team(roster, board, SETTINGS)
    assert _group(report, "QB")["starters"] == []
    assert report["roster_size"] == 3


def test_no_status_file_leaves_grading_unchanged():
    board = [_p("rb-b", "RB", 180), _p("wr-b", "WR", 180), _p("qb-b", "QB", 180)]
    roster = [_p("rb-1", "RB", 240, injury_status="OUT"), _p("wr-1", "WR", 200), _p("qb-1", "QB", 300)]
    # With no overlay, an OUT pool row still counts (and dings risk) exactly as before.
    report = grade_team(roster, board, SETTINGS)
    assert _group(report, "RB")["starters"] != []
