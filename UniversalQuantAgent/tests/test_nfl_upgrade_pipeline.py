"""Tests for ``modules.nfl_upgrade_pipeline`` -- order, mapping, neutrality, audit and guard behaviour."""
from __future__ import annotations

import copy
import logging
import math

import pytest

from modules import nfl_upgrade_pipeline as pipe
from modules import (coverage_mismatch_model, explosion_model, game_script_model, pass_game_collapse,
                     qb_collapse_model, red_zone_funnel_model, run_game_collapse)

BAD_PROTECTION = {
    "offense": {"ol_pass_block_win_rate": 0.62, "ol_continuity": 0.65, "ol_injuries": 2.0, "ol_run_block_win_rate": 0.64},
    "opponent_defense": {
        "opp_pressure_rate": 0.38, "opp_blitz_rate": 0.34, "opp_pass_rush_win_rate": 0.36,
        "opp_simulated_pressure_success": 0.33, "opp_run_defense_grade": 88.0, "opp_stacked_box_rate": 0.31,
    },
    "qb": {"qb_time_to_throw": 3.1, "qb_pressure_epa": -0.8, "qb_sack_avoidance": 0.90, "qb_release_time": 2.6},
    "game": {"vegas_spread": 3.0, "vegas_total": 41.0},
}
GOOD_SPOT = {
    "offense": {
        "team_epa_per_play": 0.15, "explosive_pass_rate": 0.115, "explosive_run_rate": 0.15, "red_zone_efficiency": 0.68,
        "ol_pass_block_win_rate": 0.78, "ol_continuity": 0.95, "ol_injuries": 0.0, "ol_run_block_win_rate": 0.76,
        "pace_seconds_per_play": 31.0, "wr1_separation": 3.5, "rb_yards_after_contact": 2.7, "qb_efficiency": 0.25,
        "team_rz_pass_rate": 0.60,
    },
    "opponent_defense": {
        "opp_def_epa_per_play": 0.11, "opp_defense_rank": 31, "opp_pressure_rate": 0.24, "opp_coverage_grade": 8.0,
        "opp_explosive_play_prevention": 6.0, "opp_rz_pass_defense_grade": 10.0, "opp_rz_run_defense_grade": 90.0,
        "opp_run_defense_grade": 20.0,
    },
    "qb": {"qb_time_to_throw": 2.5, "qb_pressure_epa": -0.2, "qb_sack_avoidance": 0.96, "qb_release_time": 2.2},
    "game": {"vegas_spread": -10.0, "vegas_total": 52.0},
}
QB_LINE = {"passing_yards": 250.0, "completion_pct": 0.65, "passing_tds": 1.8, "interceptions": 0.7, "attempts": 35.0,
           "rushing_yards": 18.0, "team_points": 24.0}
WR_LINE = {"receiving_yards": 70.0, "receiving_tds": 0.5, "receptions": 5.5, "targets": 8.0, "rushing_yards": 0.0}
TE_LINE = {"receiving_yards": 45.0, "receiving_tds": 0.35, "receptions": 4.0, "targets": 6.0}
RB_LINE = {"rushing_yards": 70.0, "rushing_tds": 0.5, "carries": 15.0, "receiving_yards": 22.0, "receptions": 2.5}

ORDER_INDEX = {"qb_collapse": 0, "pass_game_collapse": 0, "run_game_collapse": 0, "explosion": 1, "coverage_mismatch": 2,
               "explosive_play": 3, "red_zone_funnel": 4, "game_script": 5, "volume_passthrough": 6}


def run(line, position, context, **kwargs):
    return pipe.project_with_pack(line, position=position, context=context, **kwargs)


# --- inputs ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [("qb", "QB"), (" wr ", "WR"), ("HB", "RB"), ("FB", "RB"), ("te", "TE"),
                                           ("K", None), (None, None), ("", None), (7, None)])
def test_normalize_position(raw, expected):
    assert pipe.normalize_position(raw) == expected


def test_model_inputs_merges_sections_later_wins_and_none_never_erases():
    ctx = {"offense": {"a": 1, "b": 2}, "opponent_defense": {"b": 20, "c": None}, "qb": {"d": 4}, "game": {"e": 5},
           "player": {"a": 10}}
    assert pipe.model_inputs(ctx) == {"a": 10, "b": 20, "d": 4, "e": 5}
    assert pipe.model_inputs(ctx, player_features={"d": 40}, overrides={"e": 50, "b": None}) == {"a": 10, "b": 20, "d": 40, "e": 50}


def test_model_inputs_accepts_an_already_flat_mapping_and_empty_inputs():
    assert pipe.model_inputs({"ol_injuries": 1.0, "qb_time_to_throw": 3.0}) == {"ol_injuries": 1.0, "qb_time_to_throw": 3.0}
    assert pipe.model_inputs(None) == {} and pipe.model_inputs({}) == {}


# --- signals --------------------------------------------------------------------------------------
def test_empty_context_is_neutral_everywhere():
    s = pipe.compute_game_signals(None, position="QB")
    probs = s["probabilities"]
    assert set(pipe.SIGNAL_KEYS) <= set(probs)
    assert s["has_signal"] is False
    refs = {k: v for res in s["results"].values() for k, v in (res.get("references") or {}).items()}
    for key in ("collapse_probability", "pass_game_collapse", "run_collapse", "explosion_probability"):
        assert probs[key] == pytest.approx(refs[key])
    assert probs["coverage_mismatch_factor"] == 0.0 and probs["rz_pass_funnel"] == 0.0 and probs["rz_run_funnel"] == 0.0
    assert probs["positive_script_prob"] + probs["negative_script_prob"] + probs["neutral_script_prob"] == pytest.approx(1.0, abs=1e-3)


def test_qb_collapse_probability_feeds_pass_game_collapse_and_script_feeds_run_collapse():
    s = pipe.compute_game_signals(BAD_PROTECTION, position="WR")
    qb_p = s["results"]["qb_collapse"]["collapse_probability"]
    expected_pgc = pass_game_collapse.compute_pass_game_collapse({**s["inputs"], "qb_collapse_probability": qb_p})
    assert s["results"]["pass_game_collapse"]["pass_game_collapse"] == expected_pgc["pass_game_collapse"]
    trailing = s["results"]["game_script"]["trailing_script_prob"]
    expected_rgc = run_game_collapse.compute_run_game_collapse({**s["inputs"], "trailing_script_prob": trailing})
    assert s["results"]["run_game_collapse"]["run_collapse"] == expected_rgc["run_collapse"]
    assert qb_p > qb_collapse_model.compute_qb_collapse({})["collapse_probability"]  # a bad spot raises it


def test_a_trailing_underdog_raises_run_collapse_versus_a_leading_favourite():
    dog = pipe.compute_game_signals({"offense": {"ol_run_block_win_rate": 0.70}, "game": {"vegas_spread": 10.0}}, position="RB")
    fav = pipe.compute_game_signals({"offense": {"ol_run_block_win_rate": 0.70}, "game": {"vegas_spread": -10.0}}, position="RB")
    assert dog["probabilities"]["run_collapse"] > fav["probabilities"]["run_collapse"]


def test_wr_role_selects_the_coverage_model_role():
    wr = pipe.compute_game_signals({"te_vs_lb_coverage_grade": 10.0, "slot_wr_vs_nickel_grade": 90.0}, position="WR", wr_role="slot")
    te = pipe.compute_game_signals({"te_vs_lb_coverage_grade": 10.0, "slot_wr_vs_nickel_grade": 90.0}, position="TE")
    assert wr["results"]["coverage_mismatch"]["role"] == "slot" and te["results"]["coverage_mismatch"]["role"] == "te"
    assert wr["probabilities"]["coverage_mismatch_factor"] > 0 > te["probabilities"]["coverage_mismatch_factor"]


# --- neutrality -----------------------------------------------------------------------------------
@pytest.mark.parametrize("position, line", [("QB", QB_LINE), ("RB", RB_LINE), ("WR", WR_LINE), ("TE", TE_LINE)])
@pytest.mark.parametrize("context", [None, {}, {"offense": {}, "game": {}}, {"qb_time_to_throw": "n/a", "ol_injuries": float("nan")}])
def test_no_features_returns_the_line_exactly(position, line, context):
    result = run(line, position, context)
    assert result["line"] == line
    assert result["neutral"] is True and result["summary"] == {} and result["guard"] == []
    assert any("unchanged" in text for text in result["explanations"])


@pytest.mark.parametrize("position, line", [("QB", QB_LINE), ("RB", RB_LINE), ("WR", WR_LINE), ("TE", TE_LINE)])
def test_league_average_features_are_a_no_op(position, line):
    # every feature supplied at its model's own league reference == neutral
    ref_inputs = {}
    for mod, factors in ((qb_collapse_model, qb_collapse_model.FACTORS), (explosion_model, explosion_model.FACTORS),
                         (run_game_collapse, run_game_collapse.FACTORS)):
        for f in factors:
            if f.key not in ("trailing_script_prob", "ol_health"):
                ref_inputs[f.key] = f.mean
    result = run(line, position, ref_inputs, guard=False)
    for key, value in line.items():
        assert result["line"][key] == pytest.approx(value, rel=0.02), (position, key)


# --- direction and mapping ------------------------------------------------------------------------
def test_qb_collapse_context_cuts_yards_completion_and_tds_and_raises_interceptions():
    out = run(QB_LINE, "QB", BAD_PROTECTION, propagate_volume=False)["line"]
    assert out["passing_yards"] < QB_LINE["passing_yards"]
    assert out["completion_pct"] < QB_LINE["completion_pct"] <= 1.0
    assert out["passing_tds"] < QB_LINE["passing_tds"]
    assert out["interceptions"] > QB_LINE["interceptions"]
    assert out["rushing_yards"] == QB_LINE["rushing_yards"]       # not in the brief: untouched
    assert out["team_points"] <= QB_LINE["team_points"] * 1.0001  # collapse never raises points


def test_pass_game_collapse_cuts_wr_and_te_but_run_collapse_does_not_touch_them():
    wr = run(WR_LINE, "WR", BAD_PROTECTION, propagate_volume=False)
    assert wr["line"]["receiving_yards"] < WR_LINE["receiving_yards"]
    assert wr["line"]["receiving_tds"] < WR_LINE["receiving_tds"]
    assert wr["line"]["receptions"] == WR_LINE["receptions"]     # the brief moves yards and TD rate only
    te = run(TE_LINE, "TE", BAD_PROTECTION, propagate_volume=False)
    assert te["line"]["receiving_yards"] < TE_LINE["receiving_yards"]
    assert {a["stage"] for a in te["adjustments"]} <= {"pass_game_collapse", "explosion", "coverage_mismatch",
                                                      "explosive_play", "red_zone_funnel", "game_script"}


def test_run_collapse_cuts_rb_rushing_not_receiving():
    rb = run(RB_LINE, "RB", {"offense": {"ol_run_block_win_rate": 0.60, "ol_injuries": 2.0},
                              "opponent_defense": {"opp_run_defense_grade": 90.0, "opp_stacked_box_rate": 0.32},
                              "game": {"vegas_spread": 9.0}}, propagate_volume=False)["line"]
    assert rb["rushing_yards"] < RB_LINE["rushing_yards"] and rb["rushing_tds"] < RB_LINE["rushing_tds"]
    assert rb["receiving_yards"] == RB_LINE["receiving_yards"]


def test_good_spot_lifts_everyone_and_team_points():
    qb = run(QB_LINE, "QB", GOOD_SPOT, propagate_volume=False)["line"]
    assert qb["passing_yards"] > QB_LINE["passing_yards"] and qb["team_points"] > QB_LINE["team_points"]
    wr = run(WR_LINE, "WR", GOOD_SPOT, propagate_volume=False)["line"]
    assert wr["receiving_yards"] > WR_LINE["receiving_yards"]
    rb = run(RB_LINE, "RB", GOOD_SPOT, propagate_volume=False)["line"]
    assert rb["rushing_yards"] > RB_LINE["rushing_yards"] * 0.95  # explosion + run block help; heavy-favourite script may trim carries


def test_team_points_move_only_through_explosion():
    signals = {"results": {"qb_collapse": {"collapse_probability": 0.9, "references": {"collapse_probability": 0.16}}}}
    out = pipe.apply_upgrade_pack({"passing_yards": 250.0, "team_points": 24.0}, signals, position="QB")["line"]
    assert out["team_points"] == 24.0 and out["passing_yards"] < 250.0


# --- order ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("position, line, expected", [
    ("QB", QB_LINE, ["qb_collapse", "explosion", "explosive_play", "red_zone_funnel", "game_script", "volume_passthrough"]),
    ("WR", WR_LINE, ["pass_game_collapse", "explosion", "coverage_mismatch", "explosive_play", "red_zone_funnel",
                     "game_script", "volume_passthrough"]),
    ("TE", TE_LINE, ["pass_game_collapse", "explosion", "coverage_mismatch", "explosive_play", "red_zone_funnel",
                     "game_script", "volume_passthrough"]),
    ("RB", RB_LINE, ["run_game_collapse", "explosion", "explosive_play", "red_zone_funnel", "game_script", "volume_passthrough"]),
])
def test_stage_order_is_the_briefs_order(position, line, expected):
    result = run(line, position, {**BAD_PROTECTION, **{"opponent_defense": {**BAD_PROTECTION["opponent_defense"], **GOOD_SPOT["opponent_defense"]}}})
    assert result["stage_order"] == expected
    indexes = [ORDER_INDEX[a["stage"]] for a in result["adjustments"]]
    assert indexes == sorted(indexes), "an adjustment was applied out of the brief's order"
    assert indexes, "the context should have moved something"


def test_collapse_is_applied_before_explosion_multiplicatively_in_literal_mode():
    ctx = {**GOOD_SPOT, "qb": BAD_PROTECTION["qb"], "offense": {**GOOD_SPOT["offense"], **BAD_PROTECTION["offense"]}}
    result = run(QB_LINE, "QB", ctx, recenter=False, propagate_volume=False, guard=False)
    p = result["signals"]["probabilities"]
    expected = 250.0 * (1 - p["collapse_probability"] * 0.65) * (1 + p["explosion_probability"] * 0.45)
    assert result["line"]["passing_yards"] == pytest.approx(expected, rel=1e-4)
    assert result["line"]["interceptions"] == pytest.approx(0.7 * (1 + p["collapse_probability"] * 0.40), rel=1e-4)
    assert result["line"]["passing_tds"] == pytest.approx(
        1.8 * (1 - p["collapse_probability"] * 0.35) * (1 + p["rz_pass_funnel"] * 0.35), rel=1e-4)
    attempts = 35.0 * (1 + p["positive_script_prob"] * 0.20 - p["negative_script_prob"] * 0.25)
    assert result["line"]["attempts"] == pytest.approx(attempts, rel=1e-4)


# --- volume pass-through --------------------------------------------------------------------------
def test_volume_passthrough_scales_dependent_stats_by_the_script_factors():
    ctx = {"game": {"vegas_spread": -12.0, "vegas_total": 47.0}}
    on = run({**QB_LINE, **{"receiving_yards": 0.0}}, "QB", ctx, propagate_volume=True)
    off = run(QB_LINE, "QB", ctx, propagate_volume=False)
    factor = on["line"]["attempts"] / QB_LINE["attempts"]
    assert factor < 1.0                                           # a heavy favourite throws less
    assert off["line"]["passing_yards"] == QB_LINE["passing_yards"]  # script alone leaves yards untouched
    assert on["line"]["passing_yards"] == pytest.approx(QB_LINE["passing_yards"] * factor, rel=1e-3)
    assert on["line"]["passing_tds"] == pytest.approx(QB_LINE["passing_tds"] * factor, rel=1e-3)
    assert any(a["stage"] == "volume_passthrough" for a in on["adjustments"])
    assert "volume_passthrough" not in off["stage_order"]
    rb = run(RB_LINE, "RB", ctx)["line"]
    assert rb["rushing_yards"] > RB_LINE["rushing_yards"] and rb["carries"] > RB_LINE["carries"]  # a favourite runs more


# --- guard ----------------------------------------------------------------------------------------
def test_cumulative_guard_limits_a_compounded_boost_and_says_so():
    signals = {"results": {
        "explosion": {"explosion_probability": 1.0, "references": {"explosion_probability": 0.0}},
        "coverage_mismatch": {"coverage_mismatch_factor": 1.0, "references": {"coverage_mismatch_factor": 0.0}},
        "explosive_play": {"explosive_play_prob_pass": 1.0, "explosive_play_prob_run": 1.0,
                           "references": {"explosive_play_prob_pass": 0.0, "explosive_play_prob_run": 0.0}},
    }}
    guarded = pipe.apply_upgrade_pack({"receiving_yards": 60.0}, signals, position="WR", propagate_volume=False)
    assert guarded["line"]["receiving_yards"] == pytest.approx(120.0)   # 2.0x ceiling (unguarded 60*1.4*1.25+18 = 123)
    assert guarded["guard"] and guarded["guard"][0]["field"] == "receiving_yards" and guarded["warnings"]
    free = pipe.apply_upgrade_pack({"receiving_yards": 60.0}, signals, position="WR", propagate_volume=False, guard=False)
    assert free["line"]["receiving_yards"] == pytest.approx(60 * 1.4 * 1.25 + 18.0)


# --- completions ----------------------------------------------------------------------------------
def test_completions_and_completion_pct_stay_consistent():
    line = {"passing_yards": 250.0, "attempts": 30.0, "completions": 20.0}
    result = run(line, "QB", BAD_PROTECTION, propagate_volume=False)
    assert "completion_pct" not in result["line"]                    # the engine line never had one; none invented
    assert result["line"]["completions"] / result["line"]["attempts"] < 20.0 / 30.0
    with_pct = run({**line, "completion_pct": 20 / 30}, "QB", BAD_PROTECTION, propagate_volume=False)["line"]
    assert with_pct["completion_pct"] < 20 / 30
    assert with_pct["completions"] == pytest.approx(with_pct["completion_pct"] * with_pct["attempts"])


# --- audit, safety ----------------------------------------------------------------------------------
def test_inputs_are_never_mutated_and_results_are_deterministic():
    line, ctx = copy.deepcopy(QB_LINE), copy.deepcopy(BAD_PROTECTION)
    first = run(line, "QB", ctx)
    assert line == QB_LINE and ctx == BAD_PROTECTION
    second = run(line, "QB", ctx)
    assert first["line"] == second["line"] and first["adjustments"] == second["adjustments"]
    first["line"]["passing_yards"] = -1.0
    assert run(line, "QB", ctx)["line"]["passing_yards"] != -1.0


def test_unknown_position_returns_the_line_untouched_with_a_warning():
    result = pipe.apply_upgrade_pack({"passing_yards": 200.0}, pipe.compute_game_signals(BAD_PROTECTION), position="K")
    assert result["line"] == {"passing_yards": 200.0} and result["warnings"] and result["neutral"]


def test_audit_rows_have_the_documented_shape():
    result = run(QB_LINE, "QB", BAD_PROTECTION)
    for row in result["adjustments"]:
        assert {"stage", "field", "applied", "before", "after"} <= set(row)
        assert row["applied"] is True and row["before"] >= 0 and row["after"] >= 0
    assert set(result["summary"]) >= {"passing_yards"}
    entry = result["summary"]["passing_yards"]
    assert entry["after"] == pytest.approx(result["line"]["passing_yards"], abs=1e-3)
    assert entry["factor"] == pytest.approx(entry["after"] / entry["before"], abs=1e-3)


def test_garbage_in_context_never_raises_and_stays_finite():
    junk = {"offense": {"ol_pass_block_win_rate": "abc", "ol_injuries": [], "ol_continuity": float("inf")},
            "opponent_defense": {"opp_pressure_rate": -3.0, "opp_blitz_rate": 900}, "qb": None, "game": {"vegas_spread": "x"},
            "player": {"wr_separation": object()}}
    for position, line in (("QB", QB_LINE), ("WR", WR_LINE), ("RB", RB_LINE), ("TE", TE_LINE)):
        out = run(line, position, junk)
        assert all(math.isfinite(v) for v in out["line"].values() if isinstance(v, float))


def test_debug_logging_names_the_probabilities(caplog):
    with caplog.at_level(logging.DEBUG, logger="nfl_pack"):
        run(QB_LINE, "QB", BAD_PROTECTION)
    text = " ".join(record.getMessage() for record in caplog.records)
    assert "collapse_probability=" in text           # the QB model's own debug line
    assert "signals pos=QB" in text and "stage" in text


# --- team-level read and explanations --------------------------------------------------------------
def test_team_signals_expose_probabilities_references_and_the_spread():
    t = pipe.team_signals(BAD_PROTECTION)
    assert t["has_signal"] is True and t["vegas_spread"] == 3.0
    assert set(pipe.SIGNAL_KEYS) <= set(t["probabilities"])
    assert set(pipe.SIGNAL_KEYS) <= set(t["references"])
    assert t["probabilities"]["collapse_probability"] > t["references"]["collapse_probability"]
    empty = pipe.team_signals(None)
    assert empty["has_signal"] is False and empty["vegas_spread"] is None


def test_explanations_name_signals_with_league_average_comparison():
    lines = run(QB_LINE, "QB", BAD_PROTECTION)["explanations"]
    assert any(line.startswith("QB collapse risk:") and "league average" in line for line in lines)
    assert any(line.startswith("Game script:") for line in lines)
    wr_lines = run(WR_LINE, "WR", BAD_PROTECTION)["explanations"]
    assert any(line.startswith("Pass-game collapse risk:") for line in wr_lines)
    assert not any(line.startswith("Run-game collapse risk:") for line in wr_lines)


def test_game_script_flip_constant_is_honoured_by_the_pipeline(monkeypatch):
    ctx = {"game": {"vegas_spread": -12.0}}
    normal = run(QB_LINE, "QB", ctx, propagate_volume=False)["line"]["attempts"]
    monkeypatch.setattr(game_script_model, "POSITIVE_SCRIPT_IS_PASS_HEAVY", False)
    flipped = run(QB_LINE, "QB", ctx, propagate_volume=False)["line"]["attempts"]
    assert normal < QB_LINE["attempts"] and flipped != normal   # the flag changes which side each coefficient rides


def test_stage_order_constant_matches_the_documented_order():
    assert pipe.STAGE_ORDER == ("collapse", "explosion", "coverage_mismatch", "explosive_play", "red_zone_funnel", "game_script")
    assert red_zone_funnel_model is not None and coverage_mismatch_model is not None
