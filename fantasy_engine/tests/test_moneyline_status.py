"""betting: the moneyline model's opt-in team_status_penalty seam, and the
player_status_utils helper that computes it from the live overlay."""

from __future__ import annotations

import json

import pytest

from betting import player_status_utils as status_utils
from betting.moneyline_model import evaluate_game, fair_moneyline
from fantasy import player_status as ps

AVERAGES = {
    "KC": {"points_scored_avg": 28.0, "points_allowed_avg": 20.0, "games_played": 6},
    "DEN": {"points_scored_avg": 18.0, "points_allowed_avg": 24.0, "games_played": 6},
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


# --- the seam is a pure no-op by default -----------------------------------


def test_no_penalty_is_byte_identical_to_the_historical_result():
    base = fair_moneyline("KC", "DEN", averages=AVERAGES)
    assert fair_moneyline("KC", "DEN", averages=AVERAGES, team_status_penalty=None) == base
    assert fair_moneyline("KC", "DEN", averages=AVERAGES, team_status_penalty={}) == base
    assert fair_moneyline("KC", "DEN", averages=AVERAGES, team_status_penalty={"KC": 0.0}) == base


def test_a_penalty_shades_the_favoured_team_toward_a_pick_em():
    base = fair_moneyline("KC", "DEN", averages=AVERAGES)
    shaded = fair_moneyline("KC", "DEN", averages=AVERAGES, team_status_penalty={"KC": 7.0})
    assert shaded["spread"] == round(base["spread"] - 7.0, 2)
    assert shaded["total"] == round(base["total"] - 7.0, 2)
    assert shaded["home_win_probability"] < base["home_win_probability"]
    assert shaded["status_penalty"] == {"home": 7.0, "away": 0.0}


def test_evaluate_game_threads_the_penalty_through():
    game = {
        "home_team": "KC", "away_team": "DEN", "game_id": "g1",
        "moneyline": {"home": -200, "away": 170},
    }
    base = evaluate_game(game, averages=AVERAGES)
    shaded = evaluate_game(game, averages=AVERAGES, team_status_penalty={"KC": 10.0})
    assert shaded["model"]["home_win_probability"] < base["model"]["home_win_probability"]
    # the KC moneyline edge moves once its win probability drops
    assert shaded["moneyline"]["home_edge"] < base["moneyline"]["home_edge"]


# --- team_scoring_penalty: derive the map from live status -----------------


def test_team_scoring_penalty_is_empty_without_an_overlay():
    roster = {"KC": [{"player_id": "qb", "position": "QB"}]}
    assert status_utils.team_scoring_penalty(roster) == {}


def test_team_scoring_penalty_sums_position_weights_by_status(status):
    status(
        {
            "kc-qb": {"status": "OUT"},
            "kc-wr": {"status": "DOUBTFUL"},
            "kc-te": {"status": "QUESTIONABLE"},
            "den-rb": {"status": "HEALTHY"},
        }
    )
    roster = {
        "KC": [
            {"player_id": "kc-qb", "position": "QB"},    # OUT  -> 7.0 * 1.0
            {"player_id": "kc-wr", "position": "WR"},    # DBT  -> 2.5 * 0.5
            {"player_id": "kc-te", "position": "TE"},    # Q    -> 1.5 * 0.15
        ],
        "DEN": [{"player_id": "den-rb", "position": "RB"}],  # healthy -> nothing
    }
    penalty = status_utils.team_scoring_penalty(roster)
    # QB OUT (7.0) + WR doubtful (2.5*0.5=1.25) + TE questionable (1.5*0.15=0.225) ~= 8.48
    assert 8.4 < penalty["KC"] < 8.55
    assert "DEN" not in penalty


def test_team_scoring_penalty_feeds_fair_moneyline(status):
    status({"kc-qb": {"status": "OUT"}})
    roster = {"KC": [{"player_id": "kc-qb", "position": "QB"}]}
    penalty = status_utils.team_scoring_penalty(roster)
    shaded = fair_moneyline("KC", "DEN", averages=AVERAGES, team_status_penalty=penalty)
    base = fair_moneyline("KC", "DEN", averages=AVERAGES)
    assert shaded["spread"] < base["spread"]
