"""Tests for the explosion model of the NFL model upgrade pack (offline, pure)."""
from __future__ import annotations

import ast
import copy
import json
import logging
import math
from pathlib import Path

import pytest

from modules import explosion_model as m
from modules import nfl_pack_common as c

#: The brief's inputs, split by the side of the ball they describe.
BRIEF_OFFENSE_KEYS = (
    "team_epa_per_play",
    "qb_efficiency",
    "explosive_pass_rate",
    "explosive_run_rate",
    "wr1_separation",
    "rb_yards_after_contact",
    "red_zone_efficiency",
    "ol_health",
    "pace_seconds_per_play",
)
BRIEF_DEFENSE_KEYS = (
    "opp_def_epa_per_play",
    "opp_defense_rank",
    "opp_pressure_rate",
    "opp_coverage_grade",
    "opp_explosive_play_prevention",
)
BRIEF_KEYS = BRIEF_OFFENSE_KEYS + BRIEF_DEFENSE_KEYS
#: Direction of each input's effect on explosion, from the brief / contract section 4
#: (+1: a higher value is more explosive; defence keys: a *weaker* defence is more explosive).
DIRECTIONS = {
    "team_epa_per_play": +1,
    "qb_efficiency": +1,
    "explosive_pass_rate": +1,
    "explosive_run_rate": +1,
    "wr1_separation": +1,
    "rb_yards_after_contact": +1,
    "red_zone_efficiency": +1,
    "ol_health": +1,
    "pace_seconds_per_play": -1,
    "opp_def_epa_per_play": +1,
    "opp_defense_rank": +1,
    "opp_pressure_rate": -1,
    "opp_coverage_grade": -1,
    "opp_explosive_play_prevention": -1,
}
CONTRACT_KEYS = (
    "model", "explosion_probability", "probability", "reference_probability", "references", "z_score",
    "coverage", "inputs_used", "inputs_defaulted", "contributions", "drivers", "basis",
)
FACTOR_BY_KEY = {f.key: f for f in m.FACTORS}
PROJECTION = {"qb_yards": 260.0, "wr_yards": 85.0, "rb_yards": 70.0, "team_points": 24.0}

# Realistic profiles (absolute numbers, written independently of the module's references).
ELITE_OFFENSE = {
    "team_epa_per_play": 0.14, "qb_efficiency": 0.30, "explosive_pass_rate": 0.115, "explosive_run_rate": 0.15,
    "wr1_separation": 3.5, "rb_yards_after_contact": 2.7, "red_zone_efficiency": 0.68, "ol_health": 1.0,
    "pace_seconds_per_play": 31.0,
}
AVERAGE_OFFENSE = {
    "team_epa_per_play": 0.0, "qb_efficiency": 0.04, "explosive_pass_rate": 0.084, "explosive_run_rate": 0.116,
    "wr1_separation": 2.84, "rb_yards_after_contact": 2.05, "red_zone_efficiency": 0.57, "ol_health": 0.93,
    "pace_seconds_per_play": 32.7,
}
WEAK_OFFENSE = {
    "team_epa_per_play": -0.14, "qb_efficiency": -0.15, "explosive_pass_rate": 0.055, "explosive_run_rate": 0.08,
    "wr1_separation": 2.2, "rb_yards_after_contact": 1.5, "red_zone_efficiency": 0.45, "ol_health": 0.6,
    "pace_seconds_per_play": 34.2,
}
BOTTOM_FIVE_DEFENSE = {
    "opp_def_epa_per_play": 0.11, "opp_defense_rank": 31, "opp_pressure_rate": 0.24, "opp_coverage_grade": 8.0,
    "opp_explosive_play_prevention": 6.0,
}
AVERAGE_DEFENSE = {
    "opp_def_epa_per_play": 0.0, "opp_defense_rank": 16.5, "opp_pressure_rate": 0.30, "opp_coverage_grade": 50.0,
    "opp_explosive_play_prevention": 50.0,
}
ELITE_DEFENSE = {
    "opp_def_epa_per_play": -0.12, "opp_defense_rank": 2, "opp_pressure_rate": 0.36, "opp_coverage_grade": 94.0,
    "opp_explosive_play_prevention": 92.0,
}


def neutral() -> dict:
    """Every input exactly at its league reference."""
    return {f.key: f.mean for f in m.FACTORS}


def _physical(factor: c.Factor, value: float) -> float:
    if factor.kind == "rate":
        return min(max(value, 0.0), 1.0)
    if factor.key in ("opp_coverage_grade", "opp_explosive_play_prevention"):
        return min(max(value, 0.0), 100.0)
    if factor.key == "opp_defense_rank":
        return min(max(value, 1.0), 32.0)
    return value


def at_sd(k_off: float, k_def: float | None = None, keys=None) -> dict:
    """Inputs ``k`` league SDs from average in the *explosive* direction (offence side ``k_off``,
    defence side ``k_def``), kept inside their physical range.  Directions come from the BRIEF table
    above, not from the module under test."""
    k_def = k_off if k_def is None else k_def
    out = {}
    for key, direction in DIRECTIONS.items():
        if keys is not None and key not in keys:
            continue
        factor = FACTOR_BY_KEY[key]
        k = k_off if key in BRIEF_OFFENSE_KEYS else k_def
        out[key] = _physical(factor, factor.mean + k * direction * factor.sd)
    return out


def p_of(inputs) -> float:
    return m.compute_explosion(inputs)["explosion_probability"]


def scenario(offense: dict, defense: dict) -> float:
    return p_of({**offense, **defense})


def fake_result(p: float, reference: float = m.BASE_RATE) -> dict:
    return {"explosion_probability": p, "probability": p, "reference_probability": reference,
            "references": {"explosion_probability": reference}}


# --- constants and structure --------------------------------------------------
def test_constants_are_the_briefs_coefficients_and_a_measured_base_rate():
    assert m.QB_YARDS_COEFFICIENT == 0.45
    assert m.WR_YARDS_COEFFICIENT == 0.40
    assert m.RB_YARDS_COEFFICIENT == 0.35
    assert m.TEAM_POINTS_COEFFICIENT == 0.30
    assert m.ADJUSTED_FIELDS == (("qb_yards", 0.45), ("wr_yards", 0.40), ("rb_yards", 0.35), ("team_points", 0.30))
    assert m.OUTPUT_KEY == "explosion_probability"
    assert m.EVENT_THRESHOLD_POINTS == 35
    assert m.BASE_RATE == pytest.approx(437 / 3742, abs=0.001)  # measured, nflverse REG 2019-2025
    assert 0.08 <= m.BASE_RATE <= 0.20  # a recognisable, reasonably common event
    assert m.GAIN > 0 and m.INTERACTION > 0
    assert 0.5 < m.CEILING < 0.9  # the brief's "never exceeds a stated ceiling (< 0.9)"


def test_event_definition_states_the_threshold_the_count_and_the_base_rate():
    assert "35" in m.EVENT_DEFINITION and "points" in m.EVENT_DEFINITION
    doc = m.__doc__
    assert "437 of 3,742" in doc and "11.68" in doc and "EVENT DEFINITION" in doc
    assert "HONEST LIMITS" in doc and "priors, not fitted" in doc
    assert "437 of 3,742" in m.BASIS


def test_factors_are_exactly_the_brief_inputs_with_the_stated_directions():
    assert {f.key for f in m.FACTORS} == set(BRIEF_KEYS) and len(m.FACTORS) == 14
    assert {f.key: f.direction for f in m.FACTORS} == DIRECTIONS
    assert {f.key for f in m.OFFENSE_FACTORS} == set(BRIEF_OFFENSE_KEYS)
    assert {f.key for f in m.DEFENSE_FACTORS} == set(BRIEF_DEFENSE_KEYS)
    assert m.OFFENSE_KEYS == tuple(f.key for f in m.OFFENSE_FACTORS)
    assert m.DEFENSE_KEYS == tuple(f.key for f in m.DEFENSE_FACTORS)
    for f in m.FACTORS:
        assert isinstance(f, c.Factor)
        assert f.weight > 0 and f.sd > 0 and f.label


def test_every_reference_states_its_feed_and_uses_the_right_units():
    for f in m.FACTORS:
        assert f.source and f.source != "prior"  # every reference was measured; none is a bare prior
        assert "nflverse" in f.source or "derived" in f.source
        if f.kind == "rate":
            assert 0.0 < f.mean < 1.0 and f.sd < 0.5  # a fraction, not a percent
    for key in ("opp_coverage_grade", "opp_explosive_play_prevention"):
        assert FACTOR_BY_KEY[key].mean == 50.0 and 28.0 <= FACTOR_BY_KEY[key].sd <= 29.5  # uniform percentile
    assert FACTOR_BY_KEY["opp_defense_rank"].mean == 16.5  # 1..32 uniform


def test_weights_are_ordered_by_causal_importance():
    off = {f.key: f.weight for f in m.OFFENSE_FACTORS}
    dfn = {f.key: f.weight for f in m.DEFENSE_FACTORS}
    assert max(off, key=off.get) == "team_epa_per_play"  # the best single summary of the offence
    assert off["team_epa_per_play"] > off["qb_efficiency"] > off["explosive_pass_rate"] > off["red_zone_efficiency"]
    assert off["explosive_pass_rate"] > off["explosive_run_rate"]  # passing chunks are the bigger channel
    assert min(off, key=off.get) == "pace_seconds_per_play"  # a small effect
    assert max(dfn, key=dfn.get) == "opp_def_epa_per_play"
    assert dfn["opp_def_epa_per_play"] > dfn["opp_explosive_play_prevention"] > dfn["opp_coverage_grade"]
    assert min(dfn, key=dfn.get) == "opp_defense_rank"  # kept light: its job is the gate


def test_module_is_pure_stdlib_plus_the_pack_helpers():
    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported <= {"__future__", "logging", "math", "typing", "modules.nfl_pack_common"}, imported
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4", "pandas", "nflreadpy"}
    assert not ({name.split(".")[0] for name in imported} & banned)


def test_logger_is_the_models_child_of_the_pack_logger():
    assert m.LOG.name == "nfl_pack.explosion"
    assert m.LOG.parent is c.LOG or m.LOG.name.startswith(c.LOG.name + ".")


# --- the apply formulas (the brief, verbatim) -----------------------------------
@pytest.mark.parametrize("p", [0.0, 0.05, 0.117, 0.25, 0.5, 0.8, 1.0])
def test_literal_formula_equals_the_brief_for_several_probabilities(p):
    new, adjustments = m.apply_explosion(PROJECTION, fake_result(p), recenter=False)
    assert new["qb_yards"] == pytest.approx(260.0 * (1 + p * 0.45))
    assert new["wr_yards"] == pytest.approx(85.0 * (1 + p * 0.40))
    assert new["rb_yards"] == pytest.approx(70.0 * (1 + p * 0.35))
    assert new["team_points"] == pytest.approx(24.0 * (1 + p * 0.30))
    assert [a["field"] for a in adjustments] == ["qb_yards", "wr_yards", "rb_yards", "team_points"]
    assert all(a["applied"] and a["recentered"] is False and a["direction"] == "boost" for a in adjustments)
    assert {a["stage"] for a in adjustments} == {"explosion"}


def test_recenter_is_an_exact_noop_at_the_reference():
    new, adjustments = m.apply_explosion(PROJECTION, fake_result(m.BASE_RATE))
    for key, before in PROJECTION.items():
        assert new[key] == pytest.approx(before, rel=1e-12)
    assert all(a["factor"] == 1.0 for a in adjustments)
    # ... whereas the literal formula at the same probability is a real boost
    literal, _ = m.apply_explosion(PROJECTION, fake_result(m.BASE_RATE), recenter=False)
    assert literal["qb_yards"] > PROJECTION["qb_yards"] * 1.04


def test_recenter_divides_each_multiplier_by_the_formula_at_the_reference():
    p, ref = 0.40, m.BASE_RATE
    new, _ = m.apply_explosion(PROJECTION, fake_result(p, ref))
    assert new["qb_yards"] == pytest.approx(260.0 * (1 + p * 0.45) / (1 + ref * 0.45))
    assert new["wr_yards"] == pytest.approx(85.0 * (1 + p * 0.40) / (1 + ref * 0.40))
    assert new["rb_yards"] == pytest.approx(70.0 * (1 + p * 0.35) / (1 + ref * 0.35))
    assert new["team_points"] == pytest.approx(24.0 * (1 + p * 0.30) / (1 + ref * 0.30))
    # below the reference the same projection is cut
    low, _ = m.apply_explosion(PROJECTION, fake_result(0.03, ref))
    assert all(low[k] < PROJECTION[k] for k in PROJECTION)


def test_apply_reads_the_reference_from_the_result_not_from_the_constant():
    new, _ = m.apply_explosion(PROJECTION, fake_result(0.30, reference=0.30))
    assert new == pytest.approx(PROJECTION)
    only_alias = {"explosion_probability": 0.30, "reference_probability": 0.30}
    new, _ = m.apply_explosion(PROJECTION, only_alias)
    assert new == pytest.approx(PROJECTION)
    new, _ = m.apply_explosion(PROJECTION, {"explosion_probability": m.BASE_RATE})  # no reference at all
    assert new == pytest.approx(PROJECTION)


def test_apply_never_mutates_the_input_and_returns_a_new_dict():
    projection = copy.deepcopy(PROJECTION)
    snapshot = copy.deepcopy(projection)
    result = fake_result(0.5)
    result_snapshot = copy.deepcopy(result)
    new, adjustments = m.apply_explosion(projection, result)
    assert projection == snapshot and result == result_snapshot
    assert new is not projection and new != projection
    assert isinstance(adjustments, list) and len(adjustments) == 4


def test_missing_field_is_recorded_as_not_applied_not_raised():
    new, adjustments = m.apply_explosion({"qb_yards": 250.0}, fake_result(0.5))
    by_field = {a["field"]: a for a in adjustments}
    assert by_field["qb_yards"]["applied"] is True
    for field in ("wr_yards", "rb_yards", "team_points"):
        assert by_field[field]["applied"] is False and "missing" in by_field[field]["reason"]
        assert field not in new  # nothing is invented


def test_non_numeric_field_is_skipped_and_left_alone():
    new, adjustments = m.apply_explosion(
        {"qb_yards": "n/a", "wr_yards": None, "rb_yards": True, "team_points": float("nan")}, fake_result(0.5)
    )
    assert new["qb_yards"] == "n/a" and new["wr_yards"] is None and new["rb_yards"] is True
    assert math.isnan(new["team_points"])
    assert not any(a["applied"] for a in adjustments)


def test_apply_copies_unrelated_keys_through_untouched():
    projection = dict(PROJECTION, qb_td_prob=0.55, qb_completion_pct=0.64, wr_td_prob=0.4, player="Some QB")
    new, adjustments = m.apply_explosion(projection, fake_result(0.6))
    for key in ("qb_td_prob", "qb_completion_pct", "wr_td_prob", "player"):
        assert new[key] == projection[key]  # explosion moves yards and points only
    assert {a["field"] for a in adjustments} == {"qb_yards", "wr_yards", "rb_yards", "team_points"}


def test_coefficients_rank_qb_above_wr_above_rb_above_points():
    new, _ = m.apply_explosion(dict.fromkeys(PROJECTION, 100.0), fake_result(0.6), recenter=False)
    assert new["qb_yards"] > new["wr_yards"] > new["rb_yards"] > new["team_points"] > 100.0


def test_a_larger_probability_never_gives_a_smaller_projection():
    lo, _ = m.apply_explosion(PROJECTION, fake_result(0.2))
    hi, _ = m.apply_explosion(PROJECTION, fake_result(0.4))
    assert all(hi[k] > lo[k] for k in PROJECTION)


@pytest.mark.parametrize("bad", [None, {}, {"explosion_probability": "x"}, {"probability": float("nan")}, "nope", 7])
def test_invalid_result_is_a_recorded_noop(bad):
    new, adjustments = m.apply_explosion(PROJECTION, bad)
    assert new == PROJECTION
    assert len(adjustments) == 4 and not any(a["applied"] for a in adjustments)
    assert all("explosion probability" in a["reason"] for a in adjustments)


def test_apply_falls_back_to_probability_alias_and_clamps_to_unit_interval():
    new, _ = m.apply_explosion({"qb_yards": 100.0}, {"probability": 1.7}, recenter=False)
    assert new["qb_yards"] == pytest.approx(100.0 * (1 + 1.0 * 0.45))
    new, _ = m.apply_explosion({"qb_yards": 100.0}, {"explosion_probability": -0.3}, recenter=False)
    assert new["qb_yards"] == pytest.approx(100.0)
    both, _ = m.apply_explosion({"qb_yards": 100.0}, {"explosion_probability": 0.3, "probability": 0.9}, recenter=False)
    assert both["qb_yards"] == pytest.approx(100.0 * (1 + 0.3 * 0.45))  # the named key wins over the alias


def test_non_mapping_projection_is_a_recorded_noop():
    new, adjustments = m.apply_explosion(None, fake_result(0.5))
    assert new == {} and len(adjustments) == 4 and not any(a["applied"] for a in adjustments)


def test_string_numbers_in_the_projection_are_read_as_numbers():
    new, adjustments = m.apply_explosion({"qb_yards": "250"}, fake_result(0.5), recenter=False)
    assert new["qb_yards"] == pytest.approx(250.0 * (1 + 0.5 * 0.45))
    assert adjustments[0]["applied"] is True


def test_end_to_end_neutral_is_a_noop_and_a_mismatch_boosts_every_field():
    neutral_new, neutral_adj = m.apply_explosion(PROJECTION, m.compute_explosion({}))
    assert neutral_new == pytest.approx(PROJECTION) and all(a["factor"] == 1.0 for a in neutral_adj)
    boosted, adjustments = m.apply_explosion(PROJECTION, m.compute_explosion({**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE}))
    assert all(boosted[k] > PROJECTION[k] for k in PROJECTION)
    factors = {a["field"]: a["factor"] for a in adjustments}
    assert factors["qb_yards"] > factors["wr_yards"] > factors["rb_yards"] > factors["team_points"] > 1.0
    cut, _ = m.apply_explosion(PROJECTION, m.compute_explosion({**WEAK_OFFENSE, **ELITE_DEFENSE}))
    assert all(cut[k] < PROJECTION[k] for k in PROJECTION)  # the recentred boost is a haircut below the base rate


def test_end_to_end_literal_formula_uses_the_computed_probability():
    result = m.compute_explosion({**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE})
    p = result["explosion_probability"]
    literal, _ = m.apply_explosion(PROJECTION, result, recenter=False)
    assert literal["qb_yards"] == pytest.approx(260.0 * (1 + p * 0.45))
    assert literal["team_points"] == pytest.approx(24.0 * (1 + p * 0.30))


# --- compute: contract ---------------------------------------------------------
def test_result_carries_every_contract_key_and_the_extras():
    result = m.compute_explosion({**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE})
    for key in CONTRACT_KEYS:
        assert key in result, key
    assert result["model"] == "explosion"
    assert result["probability"] == result["explosion_probability"]
    assert result["references"] == {"explosion_probability": result["reference_probability"]}
    assert result["reference_probability"] == m.BASE_RATE
    assert 0.0 <= result["explosion_probability"] <= 1.0 and 0.0 <= result["coverage"] <= 1.0
    for key in ("offense_strength", "defense_weakness", "bottom_five_gate", "conjunction", "z_raw", "ceiling",
                "derived_inputs", "offense_coverage", "defense_coverage", "event", "gate_rank", "gate_rank_source"):
        assert key in result, key
    assert result["event"] == m.EVENT_DEFINITION and result["ceiling"] == m.CEILING
    assert isinstance(result["drivers"], list) and all(isinstance(d, str) for d in result["drivers"])
    json.dumps(result)  # the pages serialise it


def test_neutral_in_reference_out():
    empty = m.compute_explosion({})
    none = m.compute_explosion(None)
    average = m.compute_explosion(neutral())
    assert empty["explosion_probability"] == none["explosion_probability"] == average["explosion_probability"]
    assert empty["explosion_probability"] == empty["references"]["explosion_probability"] == m.BASE_RATE
    assert average["z_score"] == pytest.approx(0.0, abs=1e-6)
    assert average["offense_strength"] == pytest.approx(0.0, abs=1e-6)
    assert average["defense_weakness"] == pytest.approx(0.0, abs=1e-6)
    assert average["conjunction"] == 0.0
    assert average["coverage"] == 1.0 and empty["coverage"] == 0.0
    assert empty["inputs_defaulted"] == [f.key for f in m.FACTORS] and empty["inputs_used"] == []


def test_used_and_defaulted_partition_the_fourteen_inputs():
    result = m.compute_explosion({"team_epa_per_play": 0.1, "opp_defense_rank": 30, "ol_injuries": 1})
    assert set(result["inputs_used"]) == {"team_epa_per_play", "opp_defense_rank", "ol_health"}
    assert set(result["inputs_used"]) | set(result["inputs_defaulted"]) == set(BRIEF_KEYS)
    assert not set(result["inputs_used"]) & set(result["inputs_defaulted"])
    assert result["contributions"]["team_epa_per_play"]["side"] == "offense"
    assert result["contributions"]["opp_defense_rank"]["side"] == "defense"


def test_basis_declares_priors_and_names_the_event_and_the_feed():
    basis = m.compute_explosion({})["basis"]
    assert "priors, not fitted" in basis and "nflverse" in basis
    assert "35+ points" in basis and "11.68%" in basis
    assert "proxies" in basis  # grades and OL health are disclosed as proxies


def test_drivers_name_the_inputs_that_moved_the_score():
    result = m.compute_explosion({**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE})
    text = " ".join(result["drivers"])
    assert result["drivers"] and "EPA" in text and "raises the score" in text
    assert m.compute_explosion({})["drivers"] == []


# --- compute: the conjunction ---------------------------------------------------
def test_average_offence_against_average_defence_is_exactly_the_base_rate():
    result = m.compute_explosion({**AVERAGE_OFFENSE, **AVERAGE_DEFENSE})
    assert result["explosion_probability"] == pytest.approx(m.BASE_RATE, abs=0.003)  # rounded literals, so near
    exact = m.compute_explosion(neutral())
    assert exact["explosion_probability"] == m.BASE_RATE
    assert exact["offense_strength"] == pytest.approx(0.0, abs=1e-9) and exact["defense_weakness"] == pytest.approx(0.0, abs=1e-9)


def test_scenario_matrix_is_ordered_the_way_football_says():
    elite_v_bottom5 = scenario(ELITE_OFFENSE, BOTTOM_FIVE_DEFENSE)
    elite_v_avg = scenario(ELITE_OFFENSE, AVERAGE_DEFENSE)
    avg_v_bottom5 = scenario(AVERAGE_OFFENSE, BOTTOM_FIVE_DEFENSE)
    avg_v_avg = scenario(AVERAGE_OFFENSE, AVERAGE_DEFENSE)
    avg_v_elite = scenario(AVERAGE_OFFENSE, ELITE_DEFENSE)
    weak_v_elite = scenario(WEAK_OFFENSE, ELITE_DEFENSE)
    assert elite_v_bottom5 > elite_v_avg > avg_v_avg > avg_v_elite > weak_v_elite
    assert elite_v_bottom5 > avg_v_bottom5 > avg_v_avg
    assert elite_v_bottom5 > 0.35  # a conjunction of both advantages is a real probability, not a nudge
    assert elite_v_bottom5 > elite_v_avg + 0.10 and elite_v_bottom5 > avg_v_bottom5 + 0.20
    assert weak_v_elite < 0.05


def test_elite_vs_elite_and_weak_vs_weak_stay_near_the_base_rate_at_one_sd():
    elite_v_elite = p_of(at_sd(+1.0, -1.0))
    weak_v_weak = p_of(at_sd(-1.0, +1.0))
    elite_v_weak = p_of(at_sd(+1.0, +1.0))
    assert abs(elite_v_elite - m.BASE_RATE) < 0.04
    assert abs(weak_v_weak - m.BASE_RATE) < 0.04
    # ... and both are a small fraction of what the conjunction does
    assert elite_v_weak - m.BASE_RATE > 3.0 * max(abs(elite_v_elite - m.BASE_RATE), abs(weak_v_weak - m.BASE_RATE))


def test_elite_vs_elite_and_weak_vs_weak_stay_near_the_base_rate_for_realistic_profiles():
    elite_v_elite = scenario(ELITE_OFFENSE, ELITE_DEFENSE)
    weak_v_bottom5 = scenario(WEAK_OFFENSE, BOTTOM_FIVE_DEFENSE)
    elite_v_bottom5 = scenario(ELITE_OFFENSE, BOTTOM_FIVE_DEFENSE)
    lift = elite_v_bottom5 - m.BASE_RATE
    assert abs(elite_v_elite - m.BASE_RATE) < 0.08 and abs(weak_v_bottom5 - m.BASE_RATE) < 0.08
    assert abs(elite_v_elite - m.BASE_RATE) < 0.30 * lift
    assert abs(weak_v_bottom5 - m.BASE_RATE) < 0.30 * lift
    # the tilt is towards the offence (the more reliable rating): elite-vs-elite is above, weak-vs-weak below
    assert weak_v_bottom5 < m.BASE_RATE < elite_v_elite


def test_one_sided_advantages_move_the_probability_only_moderately():
    elite_only = p_of(at_sd(+1.0, 0.0))
    weak_defence_only = p_of(at_sd(0.0, +1.0))
    both = p_of(at_sd(+1.0, +1.0))
    assert m.BASE_RATE < weak_defence_only < both
    assert m.BASE_RATE < elite_only < both
    assert both > 0.25 and elite_only < 0.22 and weak_defence_only < 0.20


def test_product_term_fires_only_when_both_sides_point_towards_an_explosion():
    exactly_average = neutral()  # the rounded AVERAGE_* literals sit a hair off the references
    average_defense = {k: exactly_average[k] for k in BRIEF_DEFENSE_KEYS}
    average_offense = {k: exactly_average[k] for k in BRIEF_OFFENSE_KEYS}
    both = m.compute_explosion({**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE})
    elite_only = m.compute_explosion({**ELITE_OFFENSE, **average_defense})
    weak_only = m.compute_explosion({**average_offense, **BOTTOM_FIVE_DEFENSE})
    cancelled = m.compute_explosion({**ELITE_OFFENSE, **ELITE_DEFENSE})
    assert both["conjunction"] > 0.5 and both["z_raw"] == pytest.approx(
        both["offense_strength"] + both["defense_weakness"] + both["conjunction"], abs=2e-3)
    assert elite_only["conjunction"] == 0.0 and weak_only["conjunction"] == 0.0
    assert cancelled["conjunction"] == 0.0 and cancelled["defense_weakness"] < 0 < cancelled["offense_strength"]


def test_interaction_constant_is_wired_into_the_probability(monkeypatch):
    average_defense = {k: v for k, v in neutral().items() if k in BRIEF_DEFENSE_KEYS}
    mismatch = {**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE}
    one_sided = {**ELITE_OFFENSE, **average_defense}
    with_term, one_sided_with = p_of(mismatch), p_of(one_sided)
    monkeypatch.setattr(m, "INTERACTION", 0.0)
    assert p_of(mismatch) < with_term - 0.03  # the conjunction is worth several points of probability
    assert p_of(one_sided) == one_sided_with  # a one-sided matchup has no product term to switch off


def test_bottom_five_gate_shape():
    assert m.bottom_five_gate(None) == 0.0
    assert m.bottom_five_gate(1) == 0.0 and m.bottom_five_gate(16) == 0.0 and m.bottom_five_gate(24) == 0.0
    assert m.bottom_five_gate(26) == pytest.approx(0.5)
    assert m.bottom_five_gate(28) == 1.0 and m.bottom_five_gate(32) == 1.0 and m.bottom_five_gate(99) == 1.0
    ranks = list(range(1, 33))
    gates = [m.bottom_five_gate(rank) for rank in ranks]
    assert gates == sorted(gates)  # monotone
    assert m.BOTTOM_FIVE_RANK == 28 and m.N_TEAMS == 32  # ranks 28-32 are the bottom five of 32


def test_gate_scales_the_product_term_from_the_floor_to_full_strength():
    closed = m.combine_scores(1.0, 1.0, gate=0.0)
    half = m.combine_scores(1.0, 1.0, gate=0.5)
    opened = m.combine_scores(1.0, 1.0, gate=1.0)
    assert closed[2] == pytest.approx(m.INTERACTION * m.GATE_FLOOR)
    assert opened[2] == pytest.approx(m.INTERACTION)
    assert closed[2] < half[2] < opened[2]
    assert closed[0] < half[0] < opened[0]  # and so is the capped score
    # an out-of-range gate is clamped, not extrapolated
    assert m.combine_scores(1.0, 1.0, gate=7.0)[2] == pytest.approx(m.INTERACTION)
    assert m.combine_scores(1.0, 1.0, gate=-3.0)[2] == pytest.approx(m.INTERACTION * m.GATE_FLOOR)


def test_a_bottom_five_defence_gets_the_full_product_term_and_a_middling_one_does_not(monkeypatch):
    middling = {**BOTTOM_FIVE_DEFENSE, "opp_defense_rank": 20}
    bottom = {**BOTTOM_FIVE_DEFENSE, "opp_defense_rank": 31}
    assert m.compute_explosion({**ELITE_OFFENSE, **middling})["bottom_five_gate"] == 0.0
    assert m.compute_explosion({**ELITE_OFFENSE, **bottom})["bottom_five_gate"] == 1.0
    base_middling = p_of({**ELITE_OFFENSE, **middling})
    base_bottom = p_of({**ELITE_OFFENSE, **bottom})
    monkeypatch.setattr(m, "GATE_FLOOR", 1.0)  # gate switched off: the product term is always full strength
    assert p_of({**ELITE_OFFENSE, **bottom}) == pytest.approx(base_bottom, abs=1e-12)  # already full strength
    assert p_of({**ELITE_OFFENSE, **middling}) > base_middling + 0.01  # the gate was costing it


def test_without_a_supplied_rank_the_gate_uses_a_rank_estimated_from_epa_allowed_and_says_so():
    bad = m.compute_explosion({"opp_def_epa_per_play": 0.15})
    assert bad["gate_rank_source"] == "estimated_from_opp_def_epa_per_play"
    assert bad["gate_rank"] >= 28 and bad["bottom_five_gate"] == 1.0
    assert bad["derived_inputs"]["opp_defense_rank"]["from"] == "opp_def_epa_per_play"
    assert "opp_defense_rank" not in bad["inputs_used"]  # the estimate is not counted as a second factor
    assert "opp_defense_rank" not in bad["contributions"]
    good = m.compute_explosion({"opp_def_epa_per_play": -0.15})
    assert good["gate_rank"] <= 5 and good["bottom_five_gate"] == 0.0
    supplied = m.compute_explosion({"opp_def_epa_per_play": 0.15, "opp_defense_rank": 10})
    assert supplied["gate_rank_source"] == "supplied" and supplied["gate_rank"] == 10.0
    assert supplied["bottom_five_gate"] == 0.0 and "opp_defense_rank" not in supplied["derived_inputs"]
    none = m.compute_explosion({})
    assert none["gate_rank"] is None and none["gate_rank_source"] is None and none["bottom_five_gate"] == 0.0


def test_probability_never_exceeds_the_stated_ceiling_even_with_absurd_inputs():
    extreme = {
        "team_epa_per_play": 9.0, "qb_efficiency": 9.0, "explosive_pass_rate": 1.0, "explosive_run_rate": 1.0,
        "wr1_separation": 40.0, "rb_yards_after_contact": 40.0, "red_zone_efficiency": 1.0, "ol_health": 1.0,
        "pace_seconds_per_play": -50.0, "opp_def_epa_per_play": 9.0, "opp_defense_rank": 32,
        "opp_pressure_rate": 0.0, "opp_coverage_grade": 0.0, "opp_explosive_play_prevention": 0.0,
    }
    result = m.compute_explosion(extreme)
    assert result["explosion_probability"] <= m.CEILING < 0.9
    assert result["explosion_probability"] > 0.6  # it does approach it
    assert result["explosion_probability"] > scenario(ELITE_OFFENSE, BOTTOM_FIVE_DEFENSE)
    # the algebraic limit of the soft cap is the ceiling itself
    limit = c.sigmoid(c.logit(m.BASE_RATE) + m.GAIN * m.Z_CAP)
    assert limit == pytest.approx(m.CEILING, abs=1e-9)
    assert m.explosion_probability_from_scores(1e6, 1e6, 1.0) == pytest.approx(m.CEILING, abs=1e-6)


def test_probability_has_a_floor_above_zero():
    result = m.compute_explosion({**at_sd(-10.0)})
    assert 0.0 < result["explosion_probability"] < 0.05 < m.BASE_RATE
    assert m.explosion_probability_from_scores(-1e6, -1e6, 0.0) >= 0.0


def test_combine_scores_is_monotone_capped_and_has_slope_one_at_zero():
    assert m.combine_scores(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)
    assert m.combine_scores(0.01, 0.0)[0] == pytest.approx(0.01, rel=1e-3)
    previous = -1e9
    for off in (-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        z_capped, z_raw, conjunction = m.combine_scores(off, 0.8, 1.0)
        assert z_capped > previous and z_capped < m.Z_CAP
        assert conjunction == (0.0 if off <= 0 else pytest.approx(m.INTERACTION * off * 0.8))
        previous = z_capped
    assert m.combine_scores(500.0, 500.0, 1.0)[0] == pytest.approx(m.Z_CAP, abs=1e-6)
    assert m.combine_scores(-5.0, 4.0, 1.0)[2] == 0.0  # a mixed matchup has no product term


def test_compute_agrees_with_the_score_pair_helper():
    result = m.compute_explosion({**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE})
    helper = m.explosion_probability_from_scores(
        result["offense_strength"], result["defense_weakness"], result["bottom_five_gate"])
    assert helper == pytest.approx(result["explosion_probability"], abs=2e-3)
    assert result["z_score"] < result["z_raw"]  # the soft cap bites at this strength
    assert result["z_score"] == pytest.approx(m.Z_CAP * math.tanh(result["z_raw"] / m.Z_CAP), abs=2e-3)


def test_sub_scores_have_the_expected_signs():
    elite = m.compute_explosion(ELITE_OFFENSE)
    weak = m.compute_explosion(WEAK_OFFENSE)
    soft_d = m.compute_explosion(BOTTOM_FIVE_DEFENSE)
    hard_d = m.compute_explosion(ELITE_DEFENSE)
    assert elite["offense_strength"] > 1.0 > 0.0 > weak["offense_strength"]
    assert elite["defense_weakness"] == 0.0 and weak["defense_weakness"] == 0.0  # nothing on that side supplied
    assert soft_d["defense_weakness"] > 0.5 > 0.0 > hard_d["defense_weakness"]
    assert soft_d["offense_strength"] == 0.0 and hard_d["offense_strength"] == 0.0


# --- compute: gain, monotonicity, bounds -----------------------------------------
def test_gain_anchors_all_plus_one_and_two_sd():
    plus1 = p_of(at_sd(1.0))
    plus2 = p_of(at_sd(2.0))
    clip = p_of(at_sd(3.0))
    minus1 = p_of(at_sd(-1.0))
    assert 0.24 <= plus1 <= 0.32  # the documented ~0.28
    assert 0.46 <= plus2 <= 0.60  # the documented ~0.54, well below 0.95
    assert plus2 < clip <= m.CEILING < 0.95
    assert 0.03 <= minus1 <= 0.08  # the documented ~0.05
    assert minus1 < m.BASE_RATE < plus1 < plus2


@pytest.mark.parametrize("background_name", ["neutral", "elite_vs_bottom_five", "weak_vs_elite"])
@pytest.mark.parametrize("key", BRIEF_KEYS)
def test_each_input_is_monotone_in_its_stated_direction(key, background_name):
    backgrounds = {
        "neutral": neutral(),
        "elite_vs_bottom_five": {**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE},  # product term active
        "weak_vs_elite": {**WEAK_OFFENSE, **ELITE_DEFENSE},  # product term dormant
    }
    ps = []
    for k in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3):
        inputs = dict(backgrounds[background_name])
        inputs.update(at_sd(k, k, keys={key}))
        ps.append(p_of(inputs))
    assert ps == sorted(ps), (key, background_name, ps)
    assert ps[-1] > ps[0], (key, background_name, ps)


@pytest.mark.parametrize("key", BRIEF_KEYS)
def test_a_raw_value_in_the_explosive_direction_beats_the_average(key):
    factor = FACTOR_BY_KEY[key]
    better, worse = neutral(), neutral()
    better[key] = _physical(factor, factor.mean + 1.5 * DIRECTIONS[key] * factor.sd)
    worse[key] = _physical(factor, factor.mean - 1.5 * DIRECTIONS[key] * factor.sd)
    assert p_of(worse) < m.BASE_RATE < p_of(better), key


def test_every_brief_input_is_accepted_alone_and_moves_the_output():
    for key in BRIEF_KEYS:
        up = m.compute_explosion(at_sd(2.0, 2.0, keys={key}))
        down = m.compute_explosion(at_sd(-2.0, -2.0, keys={key}))
        assert up["inputs_used"] == [key] and down["inputs_used"] == [key], key
        assert up["explosion_probability"] > m.BASE_RATE > down["explosion_probability"], key


@pytest.mark.parametrize(
    "alias,canonical,value",
    [
        ("wr_separation", "wr1_separation", 3.4),
        ("off_epa_per_play", "team_epa_per_play", 0.11),
        ("offensive_epa_per_play", "team_epa_per_play", 0.11),
        ("qb_epa_per_dropback", "qb_efficiency", 0.25),
        ("opp_def_rank", "opp_defense_rank", 30),
        ("def_epa_per_play_allowed", "opp_def_epa_per_play", 0.08),
        ("defensive_epa_per_play", "opp_def_epa_per_play", 0.08),
        ("opp_explosive_prevention", "opp_explosive_play_prevention", 12.0),
    ],
)
def test_aliases_are_equivalent_to_the_canonical_key(alias, canonical, value):
    via_alias = m.compute_explosion({alias: value})
    direct = m.compute_explosion({canonical: value})
    assert via_alias["explosion_probability"] == direct["explosion_probability"] != m.BASE_RATE
    assert via_alias["inputs_used"] == [canonical]


def test_scores_stay_bounded_across_a_grid_of_inputs():
    for k_off in (-6, -3, -1, 0, 1, 3, 6):
        for k_def in (-6, -3, -1, 0, 1, 3, 6):
            result = m.compute_explosion(at_sd(k_off, k_def))
            assert 0.0 <= result["explosion_probability"] <= m.CEILING + 1e-9
            assert math.isfinite(result["z_score"]) and 0.0 <= result["coverage"] <= 1.0


# --- compute: missing data and bad data -------------------------------------------
def test_missing_data_shrinks_toward_the_base_rate():
    full = at_sd(+1.0)
    two = {k: full[k] for k in ("team_epa_per_play", "opp_def_epa_per_play")}
    seven = {k: full[k] for k in list(full)[::2]}
    p_two, p_seven, p_all = p_of(two), p_of(seven), p_of(full)
    assert m.BASE_RATE < p_two < p_seven < p_all
    low = at_sd(-1.0)
    p_two_low = p_of({k: low[k] for k in two})
    assert p_of(low) < p_two_low < m.BASE_RATE
    assert m.compute_explosion(two)["coverage"] < m.compute_explosion(seven)["coverage"] < m.compute_explosion(full)["coverage"]
    assert m.compute_explosion(full)["coverage"] == 1.0


def test_one_side_only_input_cannot_trigger_the_product_term():
    result = m.compute_explosion(at_sd(2.0, 2.0, keys=set(BRIEF_OFFENSE_KEYS)))
    assert result["defense_weakness"] == 0.0 and result["conjunction"] == 0.0
    assert result["offense_coverage"] == 1.0 and result["defense_coverage"] == 0.0


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf"), "n/a", "", "abc", True, False,
                                 [], {}, object()])
@pytest.mark.parametrize("key", ["team_epa_per_play", "opp_defense_rank", "red_zone_efficiency", "opp_coverage_grade"])
def test_unusable_values_are_treated_as_missing(key, bad):
    result = m.compute_explosion({key: bad})
    assert result["explosion_probability"] == m.BASE_RATE
    assert key in result["inputs_defaulted"] and result["inputs_used"] == []
    assert result["coverage"] == 0.0


def test_a_bad_value_does_not_disturb_the_good_ones():
    good = {"team_epa_per_play": 0.12, "opp_def_epa_per_play": 0.07}
    noisy = dict(good, qb_efficiency="n/a", opp_pressure_rate=float("nan"), wr1_separation=None, ol_health=True)
    assert p_of(noisy) == p_of(good)
    assert set(m.compute_explosion(noisy)["inputs_used"]) == set(good)


@pytest.mark.parametrize("not_a_mapping", [None, [], [("team_epa_per_play", 0.2)], "team_epa_per_play", 7, 3.5, object()])
def test_non_mapping_inputs_are_neutral_not_errors(not_a_mapping):
    assert m.compute_explosion(not_a_mapping)["explosion_probability"] == m.BASE_RATE


def test_objects_exposing_items_such_as_a_pandas_row_are_read():
    class Row:
        def __init__(self, data):
            self._data = data

        def items(self):
            return self._data.items()

    data = {"team_epa_per_play": 0.12, "opp_def_epa_per_play": 0.07}
    assert p_of(Row(data)) == p_of(data) != m.BASE_RATE

    class Broken:
        def items(self):
            raise RuntimeError("boom")

    assert p_of(Broken()) == m.BASE_RATE


def test_percent_and_fraction_rates_are_equivalent():
    for key, fraction, percent in (
        ("explosive_pass_rate", 0.12, 12.0),
        ("red_zone_efficiency", 0.70, 70.0),
        ("opp_pressure_rate", 0.26, 26.0),
        ("explosive_run_rate", 0.15, 15.0),
        ("ol_health", 0.80, 80.0),
    ):
        as_fraction = p_of({key: fraction})
        assert as_fraction == p_of({key: percent}) == p_of({key: f"{percent:g}%"}) == p_of({key: str(percent)})
        assert as_fraction != m.BASE_RATE


def test_rates_out_of_range_are_rejected_as_data_errors_not_clamped():
    for key in ("explosive_pass_rate", "red_zone_efficiency", "opp_pressure_rate", "ol_health"):
        for bad in (-0.2, -5.0, 150.0, 1e9):
            result = m.compute_explosion({key: bad})
            assert result["explosion_probability"] == m.BASE_RATE and key in result["inputs_defaulted"], (key, bad)


def test_huge_values_are_clipped_at_three_sd_not_amplified():
    factor = FACTOR_BY_KEY["team_epa_per_play"]
    at_clip = p_of({"team_epa_per_play": factor.mean + 3 * factor.sd})
    assert p_of({"team_epa_per_play": 1e12}) == p_of({"team_epa_per_play": 50.0}) == at_clip
    assert p_of({"team_epa_per_play": -1e12}) == p_of({"team_epa_per_play": factor.mean - 3 * factor.sd})
    pace = FACTOR_BY_KEY["pace_seconds_per_play"]
    assert p_of({"pace_seconds_per_play": -1e9}) == p_of({"pace_seconds_per_play": pace.mean - 3 * pace.sd})
    result = m.compute_explosion({"team_epa_per_play": 1e12})
    assert math.isfinite(result["z_score"]) and result["explosion_probability"] < 0.2


def test_grades_and_ranks_are_clamped_to_their_scales_in_a_monotone_way():
    assert p_of({"opp_coverage_grade": -40}) == p_of({"opp_coverage_grade": 0})
    assert p_of({"opp_coverage_grade": 400}) == p_of({"opp_coverage_grade": 100})
    assert p_of({"opp_explosive_play_prevention": -1}) == p_of({"opp_explosive_play_prevention": 0})
    assert p_of({"opp_explosive_play_prevention": 1e6}) == p_of({"opp_explosive_play_prevention": 100})
    assert p_of({"opp_defense_rank": -3}) == p_of({"opp_defense_rank": 1}) == p_of({"opp_defense_rank": 0})
    assert p_of({"opp_defense_rank": 99}) == p_of({"opp_defense_rank": 32})
    # a worse-than-possible coverage grade is no better than the worst real one
    assert p_of({"opp_coverage_grade": -40}) > m.BASE_RATE
    assert p_of({"opp_coverage_grade": 400}) < m.BASE_RATE


def test_numeric_strings_are_read_as_numbers():
    assert p_of({"opp_defense_rank": "31"}) == p_of({"opp_defense_rank": 31}) > m.BASE_RATE
    assert p_of({"team_epa_per_play": " 0.12 "}) == p_of({"team_epa_per_play": 0.12})


def test_ol_health_is_derived_from_ol_injuries_when_only_that_is_given():
    healthy = m.compute_explosion({"ol_injuries": 0})
    banged_up = m.compute_explosion({"ol_injuries": 3})
    assert healthy["explosion_probability"] > m.BASE_RATE > banged_up["explosion_probability"]
    assert banged_up["explosion_probability"] == p_of({"ol_health": 0.4})  # 1 - 3/5
    assert p_of({"ol_injuries": 2.5}) == p_of({"ol_health": 0.5})  # a questionable starter counts as half
    assert banged_up["inputs_used"] == ["ol_health"] and "ol_health" not in banged_up["inputs_defaulted"]
    derived = banged_up["derived_inputs"]["ol_health"]
    assert derived["from"] == "ol_injuries" and derived["value"] == pytest.approx(0.4)
    assert banged_up["contributions"]["ol_health"]["via"] == "ol_injuries"
    assert m.compute_explosion({"ol_health": 0.4})["derived_inputs"] == {}


def test_ol_injuries_out_of_range_is_clamped_to_zero_to_five():
    assert p_of({"ol_injuries": 99}) == p_of({"ol_health": 0.0}) == p_of({"ol_injuries": 5})
    assert p_of({"ol_injuries": -4}) == p_of({"ol_health": 1.0}) == p_of({"ol_injuries": 0})
    ps = [p_of({"ol_injuries": n}) for n in (0, 1, 2, 3, 4, 5)]
    assert ps == sorted(ps, reverse=True)  # never better with more starters out ...
    assert ps[0] > ps[1] > ps[2]  # ... and strictly worse for the first two


def test_ol_health_saturates_at_the_three_sd_clip_after_about_two_starters_out():
    # Injuries are rare and skewed (0.37 starters out per game, sd 0.53 -> health sd 0.106), so the pack's
    # +/-3 SD clip bites at ~1.96 starters out.  Documented limit: 3, 4 and 5 out score like 2 out.
    two = m.compute_explosion({"ol_injuries": 2})
    five = m.compute_explosion({"ol_injuries": 5})
    assert two["contributions"]["ol_health"]["z"] == -3.0 == five["contributions"]["ol_health"]["z"]
    assert two["explosion_probability"] == five["explosion_probability"]
    assert "saturates" in m.__doc__


def test_an_explicit_ol_health_wins_and_a_broken_one_falls_back_to_injuries():
    both = m.compute_explosion({"ol_health": 1.0, "ol_injuries": 4})
    assert both["explosion_probability"] == p_of({"ol_health": 1.0})
    assert both["derived_inputs"] == {}
    broken = m.compute_explosion({"ol_health": "n/a", "ol_injuries": 4})
    assert broken["explosion_probability"] == p_of({"ol_health": 0.2})
    assert broken["derived_inputs"]["ol_health"]["from"] == "ol_injuries"
    unusable = m.compute_explosion({"ol_injuries": "n/a"})
    assert unusable["explosion_probability"] == m.BASE_RATE and "ol_health" in unusable["inputs_defaulted"]


# --- compute: purity, determinism, logging ----------------------------------------
def test_compute_does_not_mutate_the_input_mapping():
    inputs = {**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE, "opp_coverage_grade": 140.0, "opp_defense_rank": 99,
              "ol_injuries": 2, "ol_health": None, "extra_key": {"nested": [1, 2]}}
    snapshot = copy.deepcopy(inputs)
    m.compute_explosion(inputs)
    assert inputs == snapshot


def test_compute_is_deterministic_and_keeps_no_state_between_calls():
    inputs = {**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE}
    first = m.compute_explosion(inputs)
    first["inputs_used"].append("poison")
    first["contributions"].clear()
    second = m.compute_explosion(inputs)
    third = m.compute_explosion(copy.deepcopy(inputs))
    assert second == third
    assert "poison" not in second["inputs_used"] and second["contributions"]
    assert m.compute_explosion({})["explosion_probability"] == m.BASE_RATE  # unaffected by earlier calls


def test_probability_is_logged_at_debug_on_the_model_logger(caplog):
    with caplog.at_level(logging.DEBUG, logger="nfl_pack"):
        result = m.compute_explosion({**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE})
    records = [rec for rec in caplog.records if rec.name == "nfl_pack.explosion"]
    assert records and records[0].levelno == logging.DEBUG
    message = records[0].getMessage()
    assert "explosion_probability" in message
    assert f"{result['explosion_probability']:.4f}" in message
    assert "offense=" in message and "defense=" in message and "gate=" in message


def test_nothing_is_logged_above_debug(caplog):
    with caplog.at_level(logging.INFO, logger="nfl_pack"):
        m.compute_explosion({**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE})
        m.apply_explosion(PROJECTION, fake_result(0.4))
    assert not [rec for rec in caplog.records if rec.name.startswith("nfl_pack")]


# --- realistic scenarios -----------------------------------------------------------
def test_offensive_line_injuries_take_the_edge_off_a_great_matchup():
    healthy = scenario({**ELITE_OFFENSE, "ol_health": 1.0}, BOTTOM_FIVE_DEFENSE)
    one_out = scenario({**ELITE_OFFENSE, "ol_health": 0.8}, BOTTOM_FIVE_DEFENSE)
    two_out = scenario({**ELITE_OFFENSE, "ol_health": 0.6}, BOTTOM_FIVE_DEFENSE)
    four_out = scenario({**ELITE_OFFENSE, "ol_health": 0.2}, BOTTOM_FIVE_DEFENSE)
    assert healthy > one_out > two_out >= four_out
    assert healthy - two_out > 0.03  # two starters out is a real dent ...
    assert four_out > scenario(AVERAGE_OFFENSE, AVERAGE_DEFENSE)  # ... but a great offence stays above average


def test_pace_helps_but_is_a_minor_lever():
    fast = scenario({**ELITE_OFFENSE, "pace_seconds_per_play": 30.5}, BOTTOM_FIVE_DEFENSE)
    slow = scenario({**ELITE_OFFENSE, "pace_seconds_per_play": 34.5}, BOTTOM_FIVE_DEFENSE)
    epa_down = scenario({**ELITE_OFFENSE, "team_epa_per_play": 0.0}, BOTTOM_FIVE_DEFENSE)
    assert fast > slow
    assert (fast - slow) < (scenario(ELITE_OFFENSE, BOTTOM_FIVE_DEFENSE) - epa_down)  # EPA/play matters far more


def test_a_defence_that_pressures_the_passer_is_harder_to_explode_on():
    base = {**ELITE_OFFENSE, **BOTTOM_FIVE_DEFENSE}
    soft = p_of({**base, "opp_pressure_rate": 0.24})
    hard = p_of({**base, "opp_pressure_rate": 0.36})
    assert soft > hard


def test_qb_and_team_epa_each_carry_weight_but_team_epa_carries_more():
    only_team = p_of({"team_epa_per_play": 0.14})
    only_qb = p_of({"qb_efficiency": 0.30})
    assert only_team > m.BASE_RATE and only_qb > m.BASE_RATE
    # equal-SD moves: team EPA/play (weight 3.0) outweighs QB EPA/dropback (weight 2.5)
    team = FACTOR_BY_KEY["team_epa_per_play"]
    qb = FACTOR_BY_KEY["qb_efficiency"]
    assert p_of({"team_epa_per_play": team.mean + 1.5 * team.sd}) > p_of({"qb_efficiency": qb.mean + 1.5 * qb.sd})


def test_explosion_probability_is_a_proper_probability_for_every_realistic_matchup():
    offenses = (ELITE_OFFENSE, AVERAGE_OFFENSE, WEAK_OFFENSE, {})
    defenses = (BOTTOM_FIVE_DEFENSE, AVERAGE_DEFENSE, ELITE_DEFENSE, {})
    for off in offenses:
        for dfn in defenses:
            p = scenario(off, dfn)
            assert 0.0 < p < m.CEILING + 1e-9
