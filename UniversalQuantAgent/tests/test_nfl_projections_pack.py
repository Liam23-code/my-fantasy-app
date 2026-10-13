"""``project_nfl_player(pack_context=...)``: opt-in upgrade-pack adjustment of the per-game line. Offline."""
from __future__ import annotations

import copy

import pytest

from modules import nfl_projections as np_mod
from modules import nfl_upgrade_pipeline

RAW = {
    "QB": {"games": 16, "pass_attempts": 560, "pass_yards": 4000, "pass_tds": 28, "ints": 10, "rush_attempts": 60,
           "rush_yards": 300, "rush_tds": 3, "red_zone_share": 50},
    "RB": {"games": 16, "rush_attempts": 280, "rush_yards": 1300, "rush_tds": 9, "targets": 60, "receptions": 48,
           "rec_yards": 380, "rec_tds": 2, "red_zone_share": 50},
    "WR": {"games": 16, "targets": 130, "receptions": 88, "rec_yards": 1200, "rec_tds": 8, "rush_attempts": 0, "red_zone_share": 50},
    "TE": {"games": 16, "targets": 90, "receptions": 62, "rec_yards": 700, "rec_tds": 5, "rush_attempts": 0, "red_zone_share": 50},
    "DEF": {"games": 16, "pressure_rate": 30.0, "sack_probability": 7.0, "epa_allowed": -0.05},
}

BAD_PROTECTION = {
    "offense": {"ol_pass_block_win_rate": 0.62, "ol_continuity": 0.65, "ol_injuries": 2.0, "ol_run_block_win_rate": 0.64},
    "opponent_defense": {"opp_pressure_rate": 0.38, "opp_blitz_rate": 0.34, "opp_pass_rush_win_rate": 0.36,
                         "opp_simulated_pressure_success": 0.33, "opp_run_defense_grade": 88.0, "opp_stacked_box_rate": 0.31},
    "qb": {"qb_time_to_throw": 3.1, "qb_pressure_epa": -0.8, "qb_sack_avoidance": 0.90, "qb_release_time": 2.6},
    "game": {"vegas_spread": 3.0, "vegas_total": 41.0},
}
GOOD_SPOT = {
    "offense": {"team_epa_per_play": 0.15, "explosive_pass_rate": 0.115, "explosive_run_rate": 0.15,
                "red_zone_efficiency": 0.68, "ol_pass_block_win_rate": 0.78, "ol_run_block_win_rate": 0.76},
    "opponent_defense": {"opp_def_epa_per_play": 0.11, "opp_defense_rank": 31, "opp_pressure_rate": 0.24,
                         "opp_coverage_grade": 8.0, "opp_explosive_play_prevention": 6.0},
    "game": {"vegas_spread": -3.0, "vegas_total": 52.0},
}


def fake_analysis(position: str):
    def analyze(player_name, opponent_team=None, season=None, mode="Adjusted", comparison_mode="League"):
        return {
            "player": player_name, "player_id": f"id-{position}", "team": "BUF", "position": position, "season": 2025,
            "raw_stats": dict(RAW[position]), "context_factor": 1.0, "volatility_score": 40.0,
            "matchup": {"difficulty": 50, "team": opponent_team}, "weather": {"factor": 1.0}, "pace": {},
            "source": "unit-test", "warnings": [],
        }
    return analyze


@pytest.fixture
def project(monkeypatch):
    def run(position, pack_context=None, **kw):
        monkeypatch.setattr(np_mod, "analyze_nfl_player", fake_analysis(position))
        return np_mod.project_nfl_player("Test Player", "NYJ", 2025, "Adjusted", "League", pack_context=pack_context, **kw)

    return run


# --- default off -----------------------------------------------------------------------------------
@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "DEF"])
def test_default_is_byte_identical_and_has_no_pack_key(monkeypatch, position):
    monkeypatch.setattr(np_mod, "analyze_nfl_player", fake_analysis(position))
    without = np_mod.project_nfl_player("Test Player", "NYJ", 2025)
    explicit = np_mod.project_nfl_player("Test Player", "NYJ", 2025, pack_context=None)
    assert without == explicit
    assert "upgrade_pack" not in without


# --- direction -------------------------------------------------------------------------------------
def test_bad_protection_cuts_a_qb_and_the_fantasy_points_follow(project):
    base = project("QB")
    packed = project("QB", BAD_PROTECTION)
    assert packed["upgrade_pack"]["applied"] is True and packed["upgrade_pack"]["neutral"] is False
    assert packed["projection"]["passing_yards"] < base["projection"]["passing_yards"]
    assert packed["projection"]["passing_tds"] < base["projection"]["passing_tds"]
    assert packed["projection"]["interceptions"] > base["projection"]["interceptions"]
    assert packed["projection"]["expected_fantasy_points"] < base["projection"]["expected_fantasy_points"]
    assert packed["confidence"]["low"] < base["confidence"]["low"]
    # the headline fantasy number is recomputed from the adjusted line, not carried over
    p = packed["projection"]
    expected = p["passing_yards"] * 0.04 + p["passing_tds"] * 4 - p["interceptions"] * 2 + p["rushing_yards"] * 0.1 + p["rushing_tds"] * 6
    assert p["expected_fantasy_points"] == pytest.approx(expected, abs=0.02)
    assert any(d.startswith("Upgrade pack:") for d in packed["drivers"])


def test_a_good_spot_lifts_a_qb(project):
    base, packed = project("QB"), project("QB", GOOD_SPOT)
    assert packed["projection"]["passing_yards"] > base["projection"]["passing_yards"]


@pytest.mark.parametrize("position, field", [("WR", "receiving_yards"), ("TE", "receiving_yards"), ("RB", "rushing_yards")])
def test_collapse_cuts_the_positions_own_yards(project, position, field):
    base, packed = project(position), project(position, BAD_PROTECTION)
    assert packed["projection"][field] < base["projection"][field]
    assert packed["upgrade_pack"]["applied"] is True
    assert packed["position"] == position


def test_an_rb_receiving_line_moves_only_through_volume_passthrough(project):
    base, packed = project("RB"), project("RB", GOOD_SPOT)
    ratio = packed["projection"]["receiving_yards"] / base["projection"]["receiving_yards"]
    carries_ratio = packed["projection"]["carries"] / base["projection"]["carries"]
    assert ratio == pytest.approx(packed["projection"]["receptions"] / base["projection"]["receptions"], rel=0.05)
    assert carries_ratio != pytest.approx(1.0, abs=1e-9) or ratio != pytest.approx(1.0, abs=1e-9)


# --- neutrality / safety ---------------------------------------------------------------------------
@pytest.mark.parametrize("context", [{}, {"offense": {}}, {"qb_time_to_throw": "n/a"}])
def test_a_context_without_usable_features_changes_nothing(project, context):
    base = project("QB")
    packed = project("QB", context)
    assert packed["upgrade_pack"]["neutral"] is True and packed["upgrade_pack"]["applied"] is False
    assert packed["projection"] == base["projection"]
    assert packed["confidence"] == base["confidence"]
    assert packed["drivers"] == base["drivers"]
    assert packed["upgrade_pack"]["explanations"]


def test_defenses_are_not_adjusted_and_say_why(project):
    base, packed = project("DEF"), project("DEF", BAD_PROTECTION)
    assert packed["projection"] == base["projection"]
    assert packed["upgrade_pack"]["applied"] is False and packed["upgrade_pack"]["warnings"]


def test_the_context_is_not_mutated(project):
    ctx = copy.deepcopy(BAD_PROTECTION)
    project("WR", ctx)
    assert ctx == BAD_PROTECTION


def test_a_defect_inside_the_pack_leaves_the_projection_unchanged(monkeypatch, project):
    base = project("QB")

    def boom(*args, **kwargs):
        raise RuntimeError("pack exploded")

    monkeypatch.setattr(nfl_upgrade_pipeline, "project_with_pack", boom)
    packed = project("QB", BAD_PROTECTION)
    assert packed["projection"] == base["projection"]
    assert packed["upgrade_pack"]["applied"] is False
    assert any("could not be applied" in w for w in packed["upgrade_pack"]["warnings"])


def test_the_audit_names_stages_in_the_briefs_order(project):
    packed = project("WR", {**BAD_PROTECTION, **{"opponent_defense": {**BAD_PROTECTION["opponent_defense"], **GOOD_SPOT["opponent_defense"]}}})
    stages = packed["upgrade_pack"]["stage_order"]
    assert stages.index("pass_game_collapse") < stages.index("explosion") < stages.index("explosive_play")
    assert stages.index("explosive_play") < stages.index("red_zone_funnel") < stages.index("game_script")
    assert set(packed["upgrade_pack"]["signals"]) >= {"collapse_probability", "explosion_probability", "positive_script_prob"}
