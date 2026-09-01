"""Shared fixtures for the fantasy engine test suite.

The six named players mirror the validation matrix used in the NFL
projection engine (``UniversalQuantAgent/tests/test_nfl_engine.py``): a
rushing QB, a rushing+passing QB, a receiving-work RB, a dual-threat RB, a
WR, and a receiving-only TE. Keeping the same names makes it easy to eyeball
that fantasy scores track the underlying projections sensibly.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_player_status(tmp_path_factory, monkeypatch):
    """Point ``fantasy.player_status`` at an empty per-test file so the engine
    suite never depends on whatever a local ``player_status.json`` happens to
    hold (a user may have hit the Refresh button). Tests that exercise the
    overlay re-point it at their own temp file on top of this."""
    from fantasy import player_status

    empty = tmp_path_factory.mktemp("player_status") / "player_status.json"
    empty.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(player_status, "STATUS_PATH", empty)
    player_status.clear_cache()
    yield
    player_status.clear_cache()


@pytest.fixture
def lamar_jackson_projection() -> dict:
    """Rushing QB: real rushing/passing TD volume, zero receiving."""
    return {
        "player_id": "nfl:player:00-0034796",
        "name": "Lamar Jackson",
        "position": "QB",
        "team": "BAL",
        "opponent": "CLE",
        "season": 2026,
        "passing_yards": 245.0,
        "passing_tds": 1.8,
        "interceptions": 0.4,
        "rushing_yards": 65.0,
        "rushing_tds": 0.5,
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_tds": 0.0,
        "fumbles_lost": 0.05,
        "floor": 8.2,
        "median": 15.6,
        "ceiling": 28.4,
        "drivers": ["pace", "red_zone_usage", "injury_risk"],
    }


@pytest.fixture
def josh_allen_projection() -> dict:
    """Passing + rushing QB."""
    return {
        "player_id": "nfl:player:00-0034857",
        "name": "Josh Allen",
        "position": "QB",
        "team": "BUF",
        "opponent": "KC",
        "season": 2026,
        "passing_yards": 219.5,
        "passing_tds": 1.65,
        "interceptions": 0.35,
        "rushing_yards": 31.2,
        "rushing_tds": 0.71,
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_tds": 0.0,
        "fumbles_lost": 0.1,
        "floor": 14.0,
        "median": 22.5,
        "ceiling": 34.0,
        "drivers": ["pace", "goal_line_role"],
    }


@pytest.fixture
def saquon_barkley_projection() -> dict:
    """RB1 with a real receiving role."""
    return {
        "player_id": "nfl:player:00-0034844",
        "name": "Saquon Barkley",
        "position": "RB",
        "team": "PHI",
        "opponent": "DAL",
        "season": 2026,
        "passing_yards": 0.0,
        "passing_tds": 0.0,
        "interceptions": 0.0,
        "rushing_yards": 125.3,
        "rushing_tds": 0.81,
        "receptions": 2.1,
        "receiving_yards": 17.4,
        "receiving_tds": 0.12,
        "fumbles_lost": 0.05,
        "floor": 12.0,
        "median": 21.9,
        "ceiling": 34.5,
        "drivers": ["volume", "red_zone_share", "matchup"],
    }


@pytest.fixture
def cmc_projection() -> dict:
    """Dual-threat RB1."""
    return {
        "player_id": "nfl:player:00-0033280",
        "name": "Christian McCaffrey",
        "position": "RB",
        "team": "SF",
        "opponent": "SEA",
        "season": 2026,
        "passing_yards": 0.0,
        "passing_tds": 0.0,
        "interceptions": 0.0,
        "rushing_yards": 91.2,
        "rushing_tds": 0.85,
        "receptions": 3.9,
        "receiving_yards": 32.5,
        "receiving_tds": 0.42,
        "fumbles_lost": 0.05,
        "floor": 15.0,
        "median": 24.0,
        "ceiling": 38.0,
        "drivers": ["target_share", "goal_line_role"],
    }


@pytest.fixture
def puka_nacua_projection() -> dict:
    """WR, receiving only."""
    return {
        "player_id": "nfl:player:00-0039075",
        "name": "Puka Nacua",
        "position": "WR",
        "team": "LA",
        "opponent": "SF",
        "season": 2026,
        "passing_yards": 0.0,
        "passing_tds": 0.0,
        "interceptions": 0.0,
        "rushing_yards": 0.0,
        "rushing_tds": 0.0,
        "receptions": 6.5,
        "receiving_yards": 92.0,
        "receiving_tds": 0.43,
        "fumbles_lost": 0.02,
        "floor": 9.0,
        "median": 17.5,
        "ceiling": 27.0,
        "drivers": ["target_share", "air_yards"],
    }


@pytest.fixture
def travis_kelce_projection() -> dict:
    """TE, receiving only."""
    return {
        "player_id": "nfl:player:00-0030506",
        "name": "Travis Kelce",
        "position": "TE",
        "team": "KC",
        "opponent": "DEN",
        "season": 2026,
        "passing_yards": 0.0,
        "passing_tds": 0.0,
        "interceptions": 0.0,
        "rushing_yards": 0.0,
        "rushing_tds": 0.0,
        "receptions": 5.6,
        "receiving_yards": 47.4,
        "receiving_tds": 0.17,
        "fumbles_lost": 0.01,
        "floor": 4.0,
        "median": 10.5,
        "ceiling": 18.0,
        "drivers": ["red_zone_targets"],
    }


@pytest.fixture
def named_player_projections(
    lamar_jackson_projection,
    josh_allen_projection,
    saquon_barkley_projection,
    cmc_projection,
    puka_nacua_projection,
    travis_kelce_projection,
) -> list[dict]:
    return [
        lamar_jackson_projection,
        josh_allen_projection,
        saquon_barkley_projection,
        cmc_projection,
        puka_nacua_projection,
        travis_kelce_projection,
    ]


@pytest.fixture
def league_settings() -> dict:
    return {
        "n_teams": 12,
        "scoring_mode": "ppr",
        "roster_requirements": {
            "QB": 1,
            "RB": 2,
            "WR": 2,
            "TE": 1,
            "FLEX": 1,
            "DST": 1,
            "K": 1,
            "BENCH": 6,
        },
        "flex_eligible": ["RB", "WR", "TE"],
        "max_players_per_nfl_team": 8,
    }
