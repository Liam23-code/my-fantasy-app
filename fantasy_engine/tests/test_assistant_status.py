"""fantasy.assistant honours the live player-status overlay in get_best_pick_for_round."""

from __future__ import annotations

import json

import pytest

from fantasy import player_status as ps
from fantasy.assistant import get_best_pick_for_round

SETTINGS = {
    "n_teams": 12,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1, "BENCH": 6},
    "flex_eligible": ["RB", "WR", "TE"],
}


def _p(pid, position, projection, adp):
    return {
        "player_id": pid,
        "name": pid.replace("-", " ").title(),
        "position": position,
        "team": "SF",
        "projection": float(projection),
        "expected_fantasy_points": float(projection),
        "adp": float(adp),
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


def _pool():
    # A few RBs bunched around pick 20 so ADP-proximity does not decide the
    # order, plus filler so the position has a real (positive) replacement level.
    return [
        _p("rb-healthy", "RB", 240.0, 19.0),
        _p("rb-out", "RB", 250.0, 20.0),
        _p("rb-holdout", "RB", 260.0, 21.0),
        _p("wr-doubtful", "WR", 235.0, 20.0),
        _p("wr-fine", "WR", 200.0, 20.0),
    ] + [_p(f"rb-fill{i}", "RB", 210.0 - i * 6, 40.0 + i) for i in range(18)]


def test_out_players_never_appear_in_recommendations(status):
    status({"rb-out": {"status": "OUT"}})
    picks = get_best_pick_for_round(2, [], _pool(), SETTINGS, current_pick_overall=20, picks_until_next=22, limit=20)
    names = {entry["player_id"] for entry in picks}
    assert "rb-out" not in names
    assert "rb-healthy" in names  # the rest are untouched


def test_out_flag_from_the_pool_row_alone_is_ignored_until_the_overlay_exists():
    # No status file at all -> the OUT rb-out (via its own injury_status) still shows,
    # exactly as before this feature. (Overlay is opt-in.)
    pool = _pool()
    pool[1]["injury_status"] = "OUT"
    picks = get_best_pick_for_round(2, [], pool, SETTINGS, current_pick_overall=20, picks_until_next=22, limit=20)
    assert "rb-out" in {entry["player_id"] for entry in picks}


def test_holdout_is_zeroed_and_sinks_to_the_bottom(status):
    status({"rb-holdout": {"status": "HOLDOUT"}})
    picks = get_best_pick_for_round(2, [], _pool(), SETTINGS, current_pick_overall=20, picks_until_next=22, limit=20)
    holdout = next(e for e in picks if e["player_id"] == "rb-holdout")
    assert holdout["projection"] == 0.0
    assert holdout["vorp"] <= 0
    assert holdout["status"] == "HOLDOUT"
    # ...and it ranks below the healthy RB it out-projected before the overlay.
    order = [e["player_id"] for e in picks]
    assert order.index("rb-holdout") > order.index("rb-healthy")


def test_doubtful_is_downweighted_but_still_listed(status):
    baseline = get_best_pick_for_round(
        2, [], _pool(), SETTINGS, current_pick_overall=20, picks_until_next=22, limit=20
    )
    base_wr = next(e for e in baseline if e["player_id"] == "wr-doubtful")

    status({"wr-doubtful": {"status": "DOUBTFUL"}})
    picks = get_best_pick_for_round(2, [], _pool(), SETTINGS, current_pick_overall=20, picks_until_next=22, limit=20)
    wr = next(e for e in picks if e["player_id"] == "wr-doubtful")
    assert wr["status"] == "DOUBTFUL"
    assert wr["risk_multiplier"] < base_wr["risk_multiplier"]  # existing injury multiplier now bites
    assert wr["scoring_value"] < base_wr["scoring_value"]


def test_every_recommendation_row_carries_a_status_key():
    picks = get_best_pick_for_round(2, [], _pool(), SETTINGS, current_pick_overall=20, picks_until_next=22)
    assert picks and all(entry["status"] == "HEALTHY" for entry in picks)
