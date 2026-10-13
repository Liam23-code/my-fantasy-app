"""The NFL upgrade pack's explosion stage inside ``modules.nfl_moneyline_analysis.analyze_game``.

Offline: hand-built ratings / QB context (reused from the engine's own test module) and hand-built feature
contexts. The default (no ``pack_contexts``) must be byte-identical to the pre-pack analysis.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import test_nfl_moneyline_analysis as base  # noqa: E402  (fixtures: make_ratings, make_qb_context)

from modules import nfl_moneyline_analysis as m  # noqa: E402
from modules import nfl_upgrade_pipeline  # noqa: E402

GAME = {"game_id": "2026_05_NYJ_BUF", "week": 5, "home_team": "BUF", "away_team": "NYJ",
        "kickoff_label": "Sun Oct 11 - 1:00 PM ET", "home_qb": None, "away_qb": None}

#: An elite offence against a bottom-five defence (what the explosion model is built to flag).
ELITE_VS_WEAK = {
    "offense": {
        "team_epa_per_play": 0.15, "qb_efficiency": 0.25, "explosive_pass_rate": 0.115, "explosive_run_rate": 0.15,
        "red_zone_efficiency": 0.68, "ol_pass_block_win_rate": 0.78, "ol_run_block_win_rate": 0.76,
        "pace_seconds_per_play": 31.0, "wr1_separation": 3.5, "rb_yards_after_contact": 2.7,
    },
    "opponent_defense": {
        "opp_def_epa_per_play": 0.11, "opp_defense_rank": 31, "opp_pressure_rate": 0.24, "opp_coverage_grade": 8.0,
        "opp_explosive_play_prevention": 6.0,
    },
    "qb": {"qb_time_to_throw": 2.5, "qb_pressure_epa": -0.2, "qb_sack_avoidance": 0.96, "qb_release_time": 2.2},
    "game": {"vegas_spread": -7.0, "vegas_total": 48.0},
}
WEAK_VS_STRONG = {
    "offense": {"team_epa_per_play": -0.12, "qb_efficiency": -0.15, "explosive_pass_rate": 0.055, "explosive_run_rate": 0.08,
                "red_zone_efficiency": 0.45},
    "opponent_defense": {"opp_def_epa_per_play": -0.12, "opp_defense_rank": 2, "opp_pressure_rate": 0.36,
                         "opp_coverage_grade": 94.0, "opp_explosive_play_prevention": 92.0},
}
LEAGUE_AVERAGE = {
    "offense": {"team_epa_per_play": 0.0, "qb_efficiency": 0.04, "explosive_pass_rate": 0.084, "explosive_run_rate": 0.116,
                "red_zone_efficiency": 0.57, "pace_seconds_per_play": 32.7},
    "opponent_defense": {"opp_def_epa_per_play": 0.0, "opp_defense_rank": 16.5, "opp_pressure_rate": 0.30,
                         "opp_coverage_grade": 50.0, "opp_explosive_play_prevention": 50.0},
}


def run(**kwargs):
    return m.analyze_game(dict(GAME), base.make_ratings(), base.make_qb_context(), **kwargs)


def without(result, *keys):
    return {k: v for k, v in result.items() if k not in keys}


@pytest.fixture(scope="module")
def plain():
    return run()


# --- default off -----------------------------------------------------------------------------------
def test_without_pack_contexts_the_analysis_is_exactly_the_old_one(plain):
    assert "pack" not in plain
    assert run(pack_contexts=None) == plain
    titles = [item["title"] for item in plain["reasoning"]]
    assert not any("explosion" in t.lower() or "collapse" in t.lower() for t in titles)
    assert "pack_spread" not in plain["scoring"]["adjustments"]


def test_an_empty_mapping_reports_but_changes_no_number(plain):
    result = run(pack_contexts={})
    assert result["pack"]["applied"] is False and result["pack"]["has_signal"] is False
    assert without(result, "pack") == plain


def test_a_league_average_matchup_is_an_exact_no_op(plain):
    result = run(pack_contexts={"home": LEAGUE_AVERAGE, "away": LEAGUE_AVERAGE})
    assert result["pack"]["applied"] is False
    assert without(result, "pack") == plain
    assert result["pack"]["home"]["points_delta"] == 0.0 and result["pack"]["home"]["factor"] == 1.0


# --- direction ------------------------------------------------------------------------------------
def test_an_elite_offence_against_a_weak_defence_raises_its_points_spread_and_win_probability(plain):
    result = run(pack_contexts={"home": ELITE_VS_WEAK})
    pack = result["pack"]
    assert pack["applied"] is True and pack["home"]["applied"] is True and pack["away"]["applied"] is False
    assert pack["home"]["points_after"] > pack["home"]["points_before"]
    assert pack["home"]["explosion_probability"] > pack["home"]["reference_probability"]
    assert result["scoring"]["home_expected_points"] > plain["scoring"]["home_expected_points"]
    assert result["scoring"]["away_expected_points"] == plain["scoring"]["away_expected_points"]
    assert result["scoring"]["model_spread_home"] > plain["scoring"]["model_spread_home"]
    assert result["scoring"]["model_total"] > plain["scoring"]["model_total"]
    assert result["probabilities"]["home_win"] > plain["probabilities"]["home_win"]
    assert result["probabilities"]["home_win"] + result["probabilities"]["away_win"] == pytest.approx(1.0, abs=2e-4)


def test_the_same_context_on_the_away_side_moves_the_spread_toward_away(plain):
    result = run(pack_contexts={"away": ELITE_VS_WEAK})
    assert result["pack"]["away"]["applied"] is True and result["pack"]["home"]["applied"] is False
    assert result["scoring"]["model_spread_home"] < plain["scoring"]["model_spread_home"]
    assert result["probabilities"]["away_win"] > plain["probabilities"]["away_win"]


def test_a_weak_offence_against_a_strong_defence_loses_points(plain):
    result = run(pack_contexts={"home": WEAK_VS_STRONG})
    assert result["pack"]["home"]["points_delta"] < 0
    assert result["scoring"]["home_expected_points"] < plain["scoring"]["home_expected_points"]
    assert result["probabilities"]["home_win"] < plain["probabilities"]["home_win"]


def test_two_sided_contexts_move_both_teams_independently(plain):
    result = run(pack_contexts={"home": ELITE_VS_WEAK, "away": WEAK_VS_STRONG})
    pack = result["pack"]
    assert pack["home"]["points_delta"] > 0 > pack["away"]["points_delta"]
    expected_spread = round(result["scoring"]["home_expected_points"] - result["scoring"]["away_expected_points"], 2)
    assert result["scoring"]["model_spread_home"] == pytest.approx(expected_spread, abs=0.011)


def test_the_points_move_by_exactly_the_briefs_re_centred_formula(plain):
    result = run(pack_contexts={"home": ELITE_VS_WEAK})
    info = result["pack"]["home"]
    p, ref, c = info["explosion_probability"], info["reference_probability"], m.PACK_POINTS_COEFFICIENT
    expected = plain["scoring"]["home_expected_points"] * (1 + p * c) / (1 + ref * c)
    assert info["points_after"] == pytest.approx(expected, abs=0.011)
    assert c == 0.30


# --- audit / reasoning ------------------------------------------------------------------------------
def test_adjustments_reconcile_with_the_spread_and_total(plain):
    result = run(pack_contexts={"home": ELITE_VS_WEAK, "away": WEAK_VS_STRONG})
    adj = result["scoring"]["adjustments"]
    pack = result["pack"]
    assert adj["pack_spread"] == pytest.approx(pack["spread_after"] - pack["spread_before"], abs=0.011)
    assert adj["pack_total"] == pytest.approx(pack["total_after"] - pack["total_before"], abs=0.011)
    assert result["scoring"]["model_spread_home"] == pytest.approx(plain["scoring"]["model_spread_home"] + adj["pack_spread"], abs=0.02)


def test_reasoning_gains_two_cited_items_only_when_the_pack_moved_points(plain):
    applied = run(pack_contexts={"home": ELITE_VS_WEAK})
    extra = [i for i in applied["reasoning"] if i not in plain["reasoning"]]
    titles = [i["title"] for i in applied["reasoning"]]
    assert any(t.startswith("Matchup explosion") for t in titles)
    assert any(t == "Collapse risk is context, not a points input" for t in titles)
    assert len(applied["reasoning"]) >= len(plain["reasoning"]) + 2 or len(extra) >= 2
    explosion_item = next(i for i in applied["reasoning"] if i["title"].startswith("Matchup explosion"))
    assert "explosion chance" in explosion_item["detail"] and "priors" not in explosion_item["title"]
    neutral = run(pack_contexts={"home": LEAGUE_AVERAGE})
    assert not any("Matchup explosion" in i["title"] for i in neutral["reasoning"])


def test_the_pack_block_is_honest_about_its_limits():
    result = run(pack_contexts={"home": ELITE_VS_WEAK})
    basis = result["pack"]["basis"].lower()
    assert "priors" in basis and "proxies" in basis and "no collapse coefficient" in basis
    assert result["pack"]["coefficient"] == 0.30 and result["pack"]["recentered"] is True
    assert result["pack"]["stage"] == "explosion"
    assert any("collapse" in i["detail"].lower() for i in result["reasoning"])


# --- robustness --------------------------------------------------------------------------------------
def test_contexts_with_no_usable_features_leave_the_analysis_unchanged_and_say_so(plain):
    result = run(pack_contexts={"home": {"offense": {}, "game": {"vegas_spread": -3.0}}, "away": {"qb": {"x": "n/a"}}})
    assert result["pack"]["applied"] is False
    assert without(result, "pack") == plain
    assert result["pack"]["warnings"]


@pytest.mark.parametrize("bad", ["nope", 7, [1, 2], object()])
def test_a_non_mapping_pack_contexts_is_reported_not_raised(plain, bad):
    result = run(pack_contexts=bad)
    assert without(result, "pack") == plain
    assert result["pack"]["applied"] is False and result["pack"]["warnings"]


def test_a_defect_inside_the_pack_cannot_take_the_analysis_down(monkeypatch, plain):
    def boom(*args, **kwargs):
        raise RuntimeError("pack exploded")

    monkeypatch.setattr(nfl_upgrade_pipeline, "compute_game_signals", boom)
    result = run(pack_contexts={"home": ELITE_VS_WEAK, "away": ELITE_VS_WEAK})
    assert without(result, "pack") == plain
    assert any("could not be applied" in w for w in result["pack"]["warnings"] + result["pack"]["home"]["warnings"])


def test_contexts_are_never_mutated_and_results_are_deterministic():
    contexts = {"home": copy.deepcopy(ELITE_VS_WEAK), "away": copy.deepcopy(WEAK_VS_STRONG)}
    snapshot = copy.deepcopy(contexts)
    first = run(pack_contexts=contexts)
    assert contexts == snapshot
    assert run(pack_contexts=contexts) == first


def test_everything_stays_finite_with_the_pack_on():
    result = run(pack_contexts={"home": ELITE_VS_WEAK, "away": WEAK_VS_STRONG})

    def walk(value, path="result"):
        if isinstance(value, float):
            assert math.isfinite(value), path
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}")
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")

    walk(result)


def test_expected_points_never_drop_below_the_engine_floor():
    crushing = copy.deepcopy(WEAK_VS_STRONG)
    result = run(pack_contexts={"home": crushing, "away": crushing})
    assert result["scoring"]["home_expected_points"] >= m._MIN_TEAM_POINTS
    assert result["scoring"]["away_expected_points"] >= m._MIN_TEAM_POINTS


def test_public_surface_exports_the_pack_helpers():
    for name in ("PACK_BASIS", "PACK_POINTS_COEFFICIENT", "apply_matchup_pack"):
        assert name in m.__all__ and hasattr(m, name)
