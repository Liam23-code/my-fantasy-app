"""Tests for the run-game collapse model of the NFL model upgrade pack (offline, pure)."""
from __future__ import annotations

import ast
import copy
import logging
import math
import random
from pathlib import Path

import pytest

from modules import nfl_pack_common as c
from modules import run_game_collapse as m

#: the seven inputs the brief names, in canonical form
BRIEF_KEYS = (
    "ol_run_block_win_rate",
    "ol_injuries",
    "opp_run_defense_grade",
    "opp_stacked_box_rate",
    "trailing_script_prob",
    "rb_yards_after_contact",
    "rb_breakaway_rate",
)
#: direction of each input's effect on run-game collapse risk (contract section 4)
DIRECTIONS = {
    "ol_run_block_win_rate": -1,
    "ol_injuries": +1,
    "opp_run_defense_grade": +1,
    "opp_stacked_box_rate": +1,
    "trailing_script_prob": +1,
    "rb_yards_after_contact": -1,
    "rb_breakaway_rate": -1,
}
#: the brief's wording -> canonical key
ALIASES = {
    "negative_game_script_prob": "trailing_script_prob",
    "offensive_line_run_block_win_rate": "ol_run_block_win_rate",
    "run_block_win_rate": "ol_run_block_win_rate",
    "offensive_line_injuries": "ol_injuries",
    "ol_injury_count": "ol_injuries",
    "opponent_run_defense_grade": "opp_run_defense_grade",
    "stacked_box_rate": "opp_stacked_box_rate",
}
FACTOR_BY_KEY = {f.key: f for f in m.FACTORS}
CONTRACT_KEYS = (
    "model", "run_collapse", "probability", "reference_probability", "references", "z_score",
    "coverage", "inputs_used", "inputs_defaulted", "contributions", "drivers", "basis",
)
PROJECTION = {"rb_yards": 78.0, "rb_td_prob": 0.42, "rb_attempts": 16.0}
#: physical windows used to keep synthetic "k SD" inputs on the real scale of each key
WINDOW = {"ol_injuries": (0.0, 5.0), "opp_run_defense_grade": (0.0, 100.0), "rb_yards_after_contact": (0.0, 10.0)}


def neutral() -> dict:
    return {f.key: f.mean for f in m.FACTORS}


def at_sd(k: float, keys=None) -> dict:
    """Inputs ``k`` league SDs from average in the *risk* direction, kept inside their physical range."""
    out = {}
    for f in m.FACTORS:
        if keys is not None and f.key not in keys:
            continue
        value = f.mean + k * f.direction * f.sd
        if f.kind == "rate":
            value = min(max(value, 0.0), 1.0)
        if f.key in WINDOW:
            value = min(max(value, WINDOW[f.key][0]), WINDOW[f.key][1])
        out[f.key] = value
    return out


def p_of(inputs) -> float:
    return m.compute_run_game_collapse(inputs)["run_collapse"]


def fake_result(p: float, reference: float = m.BASE_RATE) -> dict:
    return {"run_collapse": p, "probability": p, "reference_probability": reference,
            "references": {"run_collapse": reference}}


# --- constants and structure --------------------------------------------------
def test_constants_are_the_briefs_coefficients_and_a_measured_base_rate():
    assert m.RB_YARDS_COEFFICIENT == 0.50
    assert m.RB_TD_PROB_COEFFICIENT == 0.35
    assert 0.08 <= m.BASE_RATE <= 0.20  # the measured 0.1761, rounded; a recognisable, common event
    assert m.BASE_RATE == 0.176
    assert 0 < m.GAIN < 2
    assert m.OUTPUT_KEY == "run_collapse" and m.MODEL_NAME == "run_game_collapse"
    assert "17.6%" in m.EVENT_DEFINITION and "3,742" in m.EVENT_DEFINITION
    assert "60%" in m.EVENT_DEFINITION and "EPA" in m.EVENT_DEFINITION


def test_factors_are_exactly_the_seven_brief_inputs_with_the_stated_directions():
    assert {f.key for f in m.FACTORS} == set(BRIEF_KEYS) and len(m.FACTORS) == 7
    assert {f.key: f.direction for f in m.FACTORS} == DIRECTIONS
    for f in m.FACTORS:
        assert isinstance(f, c.Factor)
        # every reference states where it came from: a feed + seasons, or an explicit prior
        assert f.source.startswith("prior") or "nflverse" in f.source or "PFR" in f.source, f.key
        if f.kind == "rate":
            assert 0.0 < f.mean < 1.0 and f.sd < 0.5  # rates are fractions, not percents


def test_inputs_without_a_free_feed_are_labelled_as_priors():
    assert FACTOR_BY_KEY["ol_run_block_win_rate"].source.startswith("prior")  # tracking-based, no free feed
    assert FACTOR_BY_KEY["opp_run_defense_grade"].source.startswith("prior")  # a percentile by construction
    grade = FACTOR_BY_KEY["opp_run_defense_grade"]
    assert (grade.mean, grade.sd) == (50.0, 28.9)  # uniform percentile
    for key in ("trailing_script_prob", "opp_stacked_box_rate", "ol_injuries", "rb_yards_after_contact",
                "rb_breakaway_rate"):
        assert not FACTOR_BY_KEY[key].source.startswith("prior"), key  # measured from a named feed


def test_weights_follow_the_documented_causal_ordering():
    ordered = [f.key for f in sorted(m.FACTORS, key=lambda f: -f.weight)]
    assert ordered == [
        "ol_run_block_win_rate",   # blockers losing their assignments: proximate cause
        "trailing_script_prob",    # the volume lever
        "opp_run_defense_grade",   # front quality
        "opp_stacked_box_rate",    # the mechanism, partly a response to the offense
        "ol_injuries",             # lineup change, overlaps with the win rate
        "rb_yards_after_contact",  # mitigates
        "rb_breakaway_rate",       # highest variance, smallest signal
    ]
    assert len({f.weight for f in m.FACTORS}) == 7  # no ties: the ordering is a statement


# --- the apply formulas (the brief, verbatim) -----------------------------------
@pytest.mark.parametrize("p", [0.0, 0.05, 0.176, 0.34, 0.6, 1.0])
def test_literal_formula_equals_the_brief_for_several_probabilities(p):
    new, adjustments = m.apply_run_game_collapse(PROJECTION, fake_result(p), recenter=False)
    assert new["rb_yards"] == pytest.approx(78.0 * (1 - p * 0.50))
    assert new["rb_td_prob"] == pytest.approx(0.42 * (1 - p * 0.35))
    assert [a["field"] for a in adjustments] == ["rb_yards", "rb_td_prob"]
    assert all(a["applied"] and a["recentered"] is False and a["direction"] == "haircut" for a in adjustments)
    assert [a["coefficient"] for a in adjustments] == [0.50, 0.35]
    assert all(a["stage"] == "run_game_collapse" for a in adjustments)


def test_recenter_is_an_exact_noop_at_the_reference():
    new, adjustments = m.apply_run_game_collapse(PROJECTION, fake_result(m.BASE_RATE))
    for key, before in PROJECTION.items():
        assert new[key] == pytest.approx(before, rel=1e-12)
    assert all(a["factor"] == 1.0 for a in adjustments)
    # ... whereas the literal formula at the same probability is a real haircut
    literal, _ = m.apply_run_game_collapse(PROJECTION, fake_result(m.BASE_RATE), recenter=False)
    assert literal["rb_yards"] < PROJECTION["rb_yards"]
    assert literal["rb_td_prob"] < PROJECTION["rb_td_prob"]


def test_recenter_divides_each_multiplier_by_the_formula_at_the_reference():
    p, ref = 0.40, m.BASE_RATE
    new, _ = m.apply_run_game_collapse(PROJECTION, fake_result(p, ref))
    assert new["rb_yards"] == pytest.approx(78.0 * (1 - p * 0.50) / (1 - ref * 0.50))
    assert new["rb_td_prob"] == pytest.approx(0.42 * (1 - p * 0.35) / (1 - ref * 0.35))


def test_below_reference_collapse_probability_lifts_the_projection_when_recentered():
    new, _ = m.apply_run_game_collapse(PROJECTION, fake_result(0.05))
    assert new["rb_yards"] > PROJECTION["rb_yards"]
    assert new["rb_td_prob"] > PROJECTION["rb_td_prob"]
    assert new["rb_yards"] == pytest.approx(78.0 * (1 - 0.05 * 0.5) / (1 - m.BASE_RATE * 0.5))


def test_apply_reads_the_reference_from_the_result():
    new, _ = m.apply_run_game_collapse(PROJECTION, fake_result(0.30, reference=0.30))
    assert new == pytest.approx(PROJECTION)
    # a bare reference_probability (no `references` map) is honoured too
    bare = {"run_collapse": 0.30, "reference_probability": 0.30}
    assert m.apply_run_game_collapse(PROJECTION, bare)[0] == pytest.approx(PROJECTION)


def test_apply_without_a_reference_in_the_result_uses_base_rate():
    new, _ = m.apply_run_game_collapse(PROJECTION, {"run_collapse": m.BASE_RATE})
    assert new == pytest.approx(PROJECTION)


def test_apply_never_mutates_the_input_and_returns_a_new_dict():
    projection = copy.deepcopy(PROJECTION)
    snapshot = copy.deepcopy(projection)
    result = fake_result(0.5)
    result_snapshot = copy.deepcopy(result)
    new, adjustments = m.apply_run_game_collapse(projection, result)
    assert projection == snapshot and result == result_snapshot
    assert new is not projection and new != projection
    assert isinstance(adjustments, list)


def test_missing_field_is_recorded_as_not_applied_not_raised():
    new, adjustments = m.apply_run_game_collapse({"rb_yards": 78.0}, fake_result(0.5))
    by_field = {a["field"]: a for a in adjustments}
    assert by_field["rb_yards"]["applied"] is True
    assert by_field["rb_td_prob"]["applied"] is False and "missing" in by_field["rb_td_prob"]["reason"]
    assert "rb_td_prob" not in new  # nothing is invented
    empty, notes = m.apply_run_game_collapse({}, fake_result(0.5))
    assert empty == {} and [n["applied"] for n in notes] == [False, False]


def test_non_numeric_field_is_skipped_and_left_alone():
    new, adjustments = m.apply_run_game_collapse({"rb_yards": "n/a", "rb_td_prob": None}, fake_result(0.5))
    assert new == {"rb_yards": "n/a", "rb_td_prob": None}
    assert not any(a["applied"] for a in adjustments)


def test_apply_touches_only_rb_yards_and_rb_td_prob():
    projection = dict(PROJECTION, wr_yards=88.0, qb_yards=240.0, te_td_prob=0.2, team_points=24.5, player="Some RB",
                      rb_receiving_yards=31.0)
    new, adjustments = m.apply_run_game_collapse(projection, fake_result(0.6), recenter=False)
    assert {a["field"] for a in adjustments} == {"rb_yards", "rb_td_prob"}
    for key in ("rb_attempts", "wr_yards", "qb_yards", "te_td_prob", "team_points", "player", "rb_receiving_yards"):
        assert new[key] == projection[key], key  # carries, receiving, other positions: untouched
    assert new["rb_yards"] != projection["rb_yards"]


def test_rb_td_prob_is_capped_on_fraction_and_percent_scales_but_yards_are_not():
    # p=0 re-centered is a boost of 1/(1 - 0.176*0.35) = 1.066: enough to push 0.99 / 99.0 past the cap
    fraction, _ = m.apply_run_game_collapse({"rb_td_prob": 0.99}, fake_result(0.0))
    percent, _ = m.apply_run_game_collapse({"rb_td_prob": 99.0}, fake_result(0.0))
    assert fraction["rb_td_prob"] == 1.0
    assert percent["rb_td_prob"] == 100.0
    uncapped, _ = m.apply_run_game_collapse({"rb_td_prob": 65.0}, fake_result(0.0))
    assert uncapped["rb_td_prob"] == pytest.approx(65.0 / (1 - m.BASE_RATE * 0.35))
    # yards are not a probability: 0.99 x 1.096 must stay 1.0855, not be clipped to 1.0
    yards, _ = m.apply_run_game_collapse({"rb_yards": 0.99}, fake_result(0.0))
    assert yards["rb_yards"] == pytest.approx(0.99 / (1 - m.BASE_RATE * 0.5))
    assert yards["rb_yards"] > 1.0


def test_a_cap_never_pushes_a_value_against_the_direction_of_the_adjustment():
    # 1.2 is a malformed probability; a boost (p below the reference) must not "cap" it down to 1.0
    new, adjustments = m.apply_run_game_collapse({"rb_td_prob": 1.2}, fake_result(0.0))
    assert new["rb_td_prob"] >= 1.2
    assert adjustments[-1]["applied"]


@pytest.mark.parametrize("bad", [None, {}, {"run_collapse": "x"}, {"probability": float("nan")}, "nope", 7,
                                 {"run_collapse": -0.2}, {"run_collapse": True}, {"run_collapse": 1e9}])
def test_invalid_result_is_a_recorded_noop(bad):
    new, adjustments = m.apply_run_game_collapse(PROJECTION, bad)
    assert new == PROJECTION
    assert len(adjustments) == 2 and not any(a["applied"] for a in adjustments)
    assert all("run_collapse" in a["reason"] for a in adjustments)


def test_apply_falls_back_to_the_probability_alias_and_reads_percent_scale():
    new, _ = m.apply_run_game_collapse({"rb_yards": 100.0}, {"run_collapse": "x", "probability": 0.4}, recenter=False)
    assert new["rb_yards"] == pytest.approx(100.0 * (1 - 0.4 * 0.5))
    pct, _ = m.apply_run_game_collapse({"rb_yards": 100.0}, {"run_collapse": 30.0}, recenter=False)  # 30 -> 0.30
    assert pct["rb_yards"] == pytest.approx(100.0 * (1 - 0.30 * 0.5))
    one, _ = m.apply_run_game_collapse({"rb_yards": 100.0}, {"run_collapse": 1.0}, recenter=False)  # 1.0 = 100%
    assert one["rb_yards"] == pytest.approx(50.0)


def test_apply_tolerates_a_non_mapping_projection():
    new, adjustments = m.apply_run_game_collapse(None, fake_result(0.3))  # type: ignore[arg-type]
    assert new == {} and not any(a["applied"] for a in adjustments)


# --- compute: contract ---------------------------------------------------------
def test_result_carries_every_contract_key_and_the_extras():
    result = m.compute_run_game_collapse(at_sd(1.0))
    for key in CONTRACT_KEYS:
        assert key in result, key
    assert result["model"] == "run_game_collapse"
    assert result["probability"] == result["run_collapse"]
    assert result["references"] == {"run_collapse": result["reference_probability"]} == {"run_collapse": m.BASE_RATE}
    assert 0.0 <= result["run_collapse"] <= 1.0
    assert result["inputs_rejected"] == [] and result["inputs_derived"] == {}
    assert result["event"] == m.EVENT_DEFINITION
    assert set(result["multipliers"]) == {"rb_yards", "rb_td_prob"}
    assert set(result["contributions"]) == set(BRIEF_KEYS) and result["inputs_defaulted"] == []


def test_neutral_in_reference_out():
    empty = m.compute_run_game_collapse({})
    none = m.compute_run_game_collapse(None)
    average = m.compute_run_game_collapse(neutral())
    assert empty["run_collapse"] == none["run_collapse"] == average["run_collapse"]
    assert empty["run_collapse"] == empty["references"]["run_collapse"] == m.BASE_RATE
    assert average["z_score"] == pytest.approx(0.0, abs=1e-6)
    assert average["coverage"] == 1.0 and empty["coverage"] == 0.0
    assert empty["inputs_defaulted"] == [f.key for f in m.FACTORS] and empty["inputs_used"] == []


def test_basis_declares_priors_and_names_the_event_and_feed():
    basis = m.compute_run_game_collapse({})["basis"]
    assert "priors" in basis and "not fitted" in basis
    assert "nflverse" in basis
    assert "0.176" in basis and "3,742" in basis and "60%" in basis
    assert "0 of 7" in basis
    assert "3 of 7" in m.compute_run_game_collapse(dict(list(neutral().items())[:3]))["basis"]
    assert "weak" in basis  # the honest association caveat travels with every result


def test_multipliers_in_the_result_match_what_apply_does():
    result = m.compute_run_game_collapse(at_sd(1.0))
    p = result["run_collapse"]
    assert result["multipliers"]["rb_yards"]["literal"] == pytest.approx(1 - p * 0.5, abs=1e-4)
    assert result["multipliers"]["rb_td_prob"]["literal"] == pytest.approx(1 - p * 0.35, abs=1e-4)
    new, _ = m.apply_run_game_collapse({"rb_yards": 100.0, "rb_td_prob": 0.5}, result)
    assert new["rb_yards"] / 100.0 == pytest.approx(result["multipliers"]["rb_yards"]["recentered"], abs=1e-3)
    assert new["rb_td_prob"] / 0.5 == pytest.approx(result["multipliers"]["rb_td_prob"]["recentered"], abs=1e-3)


# --- compute: monotonicity, bounds, gain ----------------------------------------
@pytest.mark.parametrize("key", BRIEF_KEYS)
def test_each_input_is_monotone_in_its_stated_direction(key):
    ps = []
    for k in (-2, -1, 0, 1, 2):
        inputs = neutral()
        inputs.update(at_sd(k, keys={key}))
        ps.append(p_of(inputs))
    assert ps == sorted(ps), (key, ps)
    assert ps[-1] > ps[0]
    # raw direction check: a worse raw value (in the risk direction) beats the average one
    factor = FACTOR_BY_KEY[key]
    worse = neutral()
    worse[key] = factor.mean + DIRECTIONS[key] * factor.sd
    assert p_of(worse) > m.BASE_RATE
    better = neutral()
    better[key] = factor.mean - DIRECTIONS[key] * factor.sd
    if key == "ol_injuries":
        better[key] = 0.0  # cannot go below zero starters out
    assert p_of(better) < m.BASE_RATE


def test_probabilities_stay_in_unit_interval_and_below_ceiling_at_extremes():
    worst = m.compute_run_game_collapse(at_sd(10.0))  # clipped at 3 SD per input
    best = m.compute_run_game_collapse(at_sd(-10.0))
    assert 0.0 < best["run_collapse"] < m.BASE_RATE < worst["run_collapse"] <= 1.0
    assert worst["run_collapse"] < 0.95
    ceiling = c.logistic_probability(m.BASE_RATE, c.Z_CLIP, m.GAIN)  # every input at the 3-SD clip
    assert worst["run_collapse"] <= ceiling + 1e-9 and ceiling < 0.75


def test_gain_anchors_all_plus_one_and_two_sd():
    # the documented formula anchors (docstring): z = 1 -> 0.322, z = 2 -> 0.514, z = 3 (clip) -> 0.702
    assert c.logistic_probability(m.BASE_RATE, 1.0, m.GAIN) == pytest.approx(0.322, abs=0.002)
    assert c.logistic_probability(m.BASE_RATE, 2.0, m.GAIN) == pytest.approx(0.514, abs=0.002)
    assert c.logistic_probability(m.BASE_RATE, 3.0, m.GAIN) == pytest.approx(0.702, abs=0.002)
    assert c.logistic_probability(m.BASE_RATE, -1.0, m.GAIN) == pytest.approx(0.088, abs=0.002)
    # ... and through the real input path (grade held at its 0-100 maximum, injuries floored at 0)
    plus1, plus2, minus1 = p_of(at_sd(1.0)), p_of(at_sd(2.0)), p_of(at_sd(-1.0))
    assert plus1 == pytest.approx(0.322, abs=0.005)
    assert 0.45 <= plus2 <= 0.60  # "well below 0.95"
    assert 0.07 <= minus1 <= 0.11
    assert minus1 < m.BASE_RATE < plus1 < plus2 < 0.8


def test_per_input_effect_at_one_sd_is_ordered_by_weight():
    effects = {}
    for key in BRIEF_KEYS:
        inputs = neutral()
        inputs.update(at_sd(1.0, keys={key}))
        effects[key] = p_of(inputs) - m.BASE_RATE
    assert all(v > 0 for v in effects.values())
    by_effect = sorted(BRIEF_KEYS, key=lambda k: -effects[k])
    by_weight = [f.key for f in sorted(m.FACTORS, key=lambda f: -f.weight)]
    assert by_effect == by_weight
    assert max(effects, key=effects.get) == "ol_run_block_win_rate"
    assert min(effects, key=effects.get) == "rb_breakaway_rate"


def test_a_bad_stat_tracker_input_cannot_dominate_by_itself():
    # one input at the 3-SD clip is a bounded nudge: the model is a blend, not a single-factor switch
    for key in BRIEF_KEYS:
        inputs = neutral()
        inputs.update(at_sd(3.0, keys={key}))
        assert p_of(inputs) < 0.45, key


# --- compute: bad data ---------------------------------------------------------
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf"), "n/a", "", "  ", True, False, [], {},
                                 object()])
def test_garbage_values_are_missing_for_every_input(bad):
    for key in BRIEF_KEYS:
        result = m.compute_run_game_collapse({key: bad})
        assert result["inputs_used"] == [], (key, bad)
        assert result["run_collapse"] == m.BASE_RATE
        assert result["inputs_rejected"] == []  # unreadable is "absent", not "rejected"
        assert result["coverage"] == 0.0


def test_garbage_container_never_raises():
    for bad in (None, {}, [], "text", 7, ("a", "b"), [("ol_injuries", 2)], float("nan")):
        result = m.compute_run_game_collapse(bad)  # type: ignore[arg-type]
        assert result["run_collapse"] == m.BASE_RATE


@pytest.mark.parametrize(
    "key, value",
    [
        ("ol_injuries", -1),               # negative count
        ("ol_injuries", 6),                # more starters than a line has
        ("ol_injuries", 99),
        ("opp_run_defense_grade", -5),     # percentile below 0
        ("opp_run_defense_grade", 140),    # percentile above 100
        ("opp_run_defense_grade", "150%"),
        ("rb_yards_after_contact", -1.0),  # negative yards
        ("rb_yards_after_contact", 25),    # a cumulative total / feet passed by mistake
        ("ol_run_block_win_rate", -0.2),   # negative rate
        ("ol_run_block_win_rate", 150),    # over 100%
        ("trailing_script_prob", 1e9),     # huge
        ("opp_stacked_box_rate", "-5%"),
        ("rb_breakaway_rate", 101),
    ],
)
def test_impossible_values_are_rejected_not_clamped(key, value):
    result = m.compute_run_game_collapse({key: value})
    assert key not in result["inputs_used"]
    assert result["run_collapse"] == m.BASE_RATE  # not "max risk", not "min risk": just absent
    assert [r["key"] for r in result["inputs_rejected"]] == [key]
    assert result["inputs_rejected"][0]["reason"]


def test_extreme_but_possible_values_are_clipped_at_three_sd_not_rejected():
    five_out = m.compute_run_game_collapse({"ol_injuries": 5})
    four_out = m.compute_run_game_collapse({"ol_injuries": 4})
    assert five_out["contributions"]["ol_injuries"]["z"] == pytest.approx(c.Z_CLIP)
    assert five_out["run_collapse"] == four_out["run_collapse"] > m.BASE_RATE
    assert five_out["inputs_rejected"] == []
    # a back who gains nothing after contact is a 3-SD-bad input, not a data error
    nothing = m.compute_run_game_collapse({"rb_yards_after_contact": 0.0})
    assert nothing["contributions"]["rb_yards_after_contact"]["z"] == pytest.approx(c.Z_CLIP)
    assert nothing["inputs_rejected"] == [] and nothing["run_collapse"] > m.BASE_RATE
    # the grade's own endpoints are real: the very worst and very best run defense
    assert p_of({"opp_run_defense_grade": 0}) < m.BASE_RATE < p_of({"opp_run_defense_grade": 100})


@pytest.mark.parametrize(
    "key, as_fraction, as_percent, as_text",
    [
        ("ol_run_block_win_rate", 0.62, 62, "62%"),
        ("trailing_script_prob", 0.45, 45, "45%"),
        ("opp_stacked_box_rate", 0.30, 30, "30%"),
        ("rb_breakaway_rate", 0.03, 3, "3%"),
    ],
)
def test_percent_and_fraction_spellings_agree(key, as_fraction, as_percent, as_text):
    a, b, d = (m.compute_run_game_collapse({key: v}) for v in (as_fraction, as_percent, as_text))
    assert a["run_collapse"] == b["run_collapse"] == d["run_collapse"] != m.BASE_RATE
    assert a["inputs_used"] == b["inputs_used"] == d["inputs_used"] == [key]
    assert a["contributions"][key]["value"] == pytest.approx(as_fraction)


def test_rate_of_exactly_one_is_a_fraction_not_one_percent():
    # a breakaway rate of 1.0 is "every carry", far better than average; read as 1% it would raise risk
    assert p_of({"rb_breakaway_rate": 1.0}) < m.BASE_RATE
    assert p_of({"rb_breakaway_rate": 0.01}) > m.BASE_RATE


def test_grade_is_a_literal_percentile_and_a_percent_string_is_scaled():
    assert p_of({"opp_run_defense_grade": "78%"}) == p_of({"opp_run_defense_grade": 78}) > m.BASE_RATE
    assert p_of({"opp_run_defense_grade": "78"}) == p_of({"opp_run_defense_grade": 78.0})
    # documented trap: 0.8 is the 0.8th percentile (a very soft front), not "80%"
    assert p_of({"opp_run_defense_grade": 0.8}) < m.BASE_RATE


def test_missing_data_shrinks_toward_the_reference():
    bad = at_sd(1.5)
    two_keys = {"ol_run_block_win_rate": bad["ol_run_block_win_rate"], "trailing_script_prob": bad["trailing_script_prob"]}
    partial = m.compute_run_game_collapse(two_keys)
    full = m.compute_run_game_collapse(bad)
    assert partial["coverage"] < full["coverage"] == 1.0
    assert m.BASE_RATE < partial["run_collapse"] < full["run_collapse"]
    assert 0 < partial["z_score"] < full["z_score"]
    assert set(partial["inputs_defaulted"]) == set(BRIEF_KEYS) - set(two_keys)
    assert partial["coverage"] == pytest.approx((1.30 + 1.20) / sum(f.weight for f in m.FACTORS), abs=1e-3)


def test_deterministic_and_order_independent_and_non_mutating():
    inputs = at_sd(1.0)
    snapshot = copy.deepcopy(inputs)
    first = m.compute_run_game_collapse(inputs)
    second = m.compute_run_game_collapse(dict(reversed(list(inputs.items()))))
    assert first == second
    assert inputs == snapshot
    assert m.compute_run_game_collapse(inputs) == first


def test_outputs_are_rounded_and_finite():
    result = m.compute_run_game_collapse(at_sd(1.0))
    assert result["run_collapse"] == round(result["run_collapse"], 4)
    assert math.isfinite(result["z_score"]) and math.isfinite(result["coverage"])
    assert m.compute_run_game_collapse(at_sd(-10.0))["run_collapse"] > 0.0


def test_fuzz_compute_then_apply_never_raises_and_stays_bounded():
    rng = random.Random(20261011)
    pool = [None, float("nan"), float("inf"), "n/a", "", True, -3, -0.2, 0, 0.0, 0.01, 0.3, 0.7, 1, 1.0, 2, 5, 5.5,
            30, 62.5, 99, 100, 101, 1e9, "30%", "0.4", "1,204", [], {}]
    for _ in range(300):
        inputs = {}
        for f in m.FACTORS:
            roll = rng.random()
            if roll < 0.35:
                continue
            key = rng.choice((f.key, *f.aliases))
            inputs[key] = rng.choice(pool) if roll < 0.7 else f.mean + rng.uniform(-4, 4) * f.sd
        result = m.compute_run_game_collapse(inputs)
        p = result["run_collapse"]
        assert 0.0 <= p <= 1.0 and math.isfinite(result["z_score"])
        assert 0.0 <= result["coverage"] <= 1.0
        assert set(result["inputs_used"]) | set(result["inputs_defaulted"]) == set(BRIEF_KEYS)
        for recenter in (True, False):
            new, adjustments = m.apply_run_game_collapse(PROJECTION, result, recenter=recenter)
            assert all(math.isfinite(v) and v >= 0 for v in new.values())
            assert new["rb_attempts"] == PROJECTION["rb_attempts"]
            assert len(adjustments) == 2 and all(a["applied"] for a in adjustments)
            assert new["rb_td_prob"] <= 1.0


# --- compute: vocabulary -------------------------------------------------------
@pytest.mark.parametrize("key", BRIEF_KEYS)
def test_every_brief_named_input_is_accepted_and_used(key):
    factor = FACTOR_BY_KEY[key]
    value = at_sd(1.0, keys={key})[key]
    result = m.compute_run_game_collapse({key: value})
    assert result["inputs_used"] == [key]
    assert result["contributions"][key]["z"] == pytest.approx(1.0, abs=0.05)  # +1 SD in the risk direction
    assert result["run_collapse"] > m.BASE_RATE
    assert result["contributions"][key]["weight"] == factor.weight


@pytest.mark.parametrize("alias, key", sorted(ALIASES.items()))
def test_aliases_are_accepted_and_mean_the_same_thing(alias, key):
    value = at_sd(1.0, keys={key})[key]
    via_alias = m.compute_run_game_collapse({alias: value})
    canonical = m.compute_run_game_collapse({key: value})
    assert via_alias["inputs_used"] == [key]
    assert via_alias["run_collapse"] == canonical["run_collapse"] > m.BASE_RATE


def test_the_canonical_key_wins_over_an_alias():
    both = m.compute_run_game_collapse({"trailing_script_prob": 0.20, "negative_game_script_prob": 0.60})
    assert both["contributions"]["trailing_script_prob"]["value"] == 0.20
    assert both["run_collapse"] < m.BASE_RATE
    # an unusable canonical value falls through to a usable alias
    fallback = m.compute_run_game_collapse({"trailing_script_prob": "n/a", "negative_game_script_prob": 0.60})
    assert fallback["contributions"]["trailing_script_prob"]["value"] == 0.60


def test_negative_game_script_means_trailing_and_game_script_models_negative_script_is_not_accepted():
    # the brief's "negative game script" = the offense is TRAILING and abandons the run -> more collapse risk
    assert p_of({"negative_game_script_prob": 0.55}) > m.BASE_RATE > p_of({"negative_game_script_prob": 0.10})
    # game_script_model's negative_script_prob means run-heavy / leading (the opposite): ignored, never inverted
    ignored = m.compute_run_game_collapse({"negative_script_prob": 0.90, "leading_script_prob": 0.90})
    assert ignored["inputs_used"] == [] and ignored["run_collapse"] == m.BASE_RATE


def test_ol_health_is_converted_when_injuries_are_absent_and_never_overrides_them():
    derived = m.compute_run_game_collapse({"ol_health": 0.6})  # 40% less healthy = 2 starters out
    assert derived["inputs_derived"] == {"ol_injuries": "5 * (1 - ol_health)"}
    assert derived["contributions"]["ol_injuries"]["value"] == pytest.approx(2.0)
    assert derived["run_collapse"] == p_of({"ol_injuries": 2}) > m.BASE_RATE
    assert p_of({"ol_health": 60}) == derived["run_collapse"]  # percent spelling
    explicit = m.compute_run_game_collapse({"ol_injuries": 0, "ol_health": 0.2})
    assert explicit["contributions"]["ol_injuries"]["value"] == 0.0 and explicit["inputs_derived"] == {}
    bad = m.compute_run_game_collapse({"ol_health": 400})
    assert bad["inputs_rejected"][0]["key"] == "ol_health" and bad["inputs_used"] == []


def test_unrelated_keys_are_ignored():
    result = m.compute_run_game_collapse(
        {"ol_pass_block_win_rate": 0.1, "opp_pressure_rate": 0.5, "vegas_total": 55, "foo": "bar"}
    )
    assert result["inputs_used"] == [] and result["run_collapse"] == m.BASE_RATE


def test_drivers_name_the_inputs_that_moved_the_number():
    result = m.compute_run_game_collapse({"ol_run_block_win_rate": 0.60, "trailing_script_prob": 0.50})
    assert result["drivers"] and "OL run-block win rate" in result["drivers"][0]
    assert all("raises" in d for d in result["drivers"])


# --- scenarios -----------------------------------------------------------------
STONEWALL = {  # weak, depleted line; stout front that stacks the box; a likely-trailing offense; a back who creates little
    "ol_run_block_win_rate": 0.63, "ol_injuries": 2, "opp_run_defense_grade": 88, "opp_stacked_box_rate": 0.33,
    "trailing_script_prob": 0.46, "rb_yards_after_contact": 1.55, "rb_breakaway_rate": 0.020,
}
MAULING = {  # dominant healthy line, soft front, big favourite, explosive back
    "ol_run_block_win_rate": 0.77, "ol_injuries": 0, "opp_run_defense_grade": 15, "opp_stacked_box_rate": 0.13,
    "trailing_script_prob": 0.16, "rb_yards_after_contact": 2.55, "rb_breakaway_rate": 0.075,
}


def test_realistic_scenarios_order_as_football_says():
    stonewall, average, mauling = p_of(STONEWALL), p_of(neutral()), p_of(MAULING)
    assert mauling < average < stonewall
    assert mauling < 0.10 and stonewall > 0.30
    assert stonewall < 0.65  # bad is not certain: collapse is partly luck


def test_script_alone_can_hurt_a_good_run_game_but_not_as_much_as_everything_together():
    good_game = dict(MAULING, trailing_script_prob=0.55)  # a great run game that is expected to trail
    assert p_of(good_game) > p_of(MAULING)
    assert p_of(good_game) < p_of(STONEWALL)
    # a stout front hurts a good line, but a bad line against a stout front is worse still
    good_line_vs_stout = dict(MAULING, opp_run_defense_grade=90, opp_stacked_box_rate=0.32)
    bad_line_vs_stout = dict(good_line_vs_stout, ol_run_block_win_rate=0.62, ol_injuries=2)
    assert p_of(MAULING) < p_of(good_line_vs_stout) < p_of(bad_line_vs_stout)


def test_end_to_end_a_stonewalled_game_trims_the_projection_and_a_mauling_lifts_it():
    stone = m.compute_run_game_collapse(STONEWALL)
    maul = m.compute_run_game_collapse(MAULING)
    s_new, _ = m.apply_run_game_collapse(PROJECTION, stone)
    m_new, _ = m.apply_run_game_collapse(PROJECTION, maul)
    assert s_new["rb_yards"] < PROJECTION["rb_yards"] < m_new["rb_yards"]
    assert s_new["rb_td_prob"] < PROJECTION["rb_td_prob"] < m_new["rb_td_prob"]
    # a league-average game leaves the engine's number alone (default re-centering)
    neutral_new, _ = m.apply_run_game_collapse(PROJECTION, m.compute_run_game_collapse({}))
    assert neutral_new == pytest.approx(PROJECTION)
    # the brief's coefficient ordering: yards take the bigger haircut (0.50 vs 0.35)
    cut_yards = 1 - s_new["rb_yards"] / PROJECTION["rb_yards"]
    cut_td = 1 - s_new["rb_td_prob"] / PROJECTION["rb_td_prob"]
    assert cut_yards > cut_td > 0
    p, ref = stone["run_collapse"], stone["reference_probability"]
    assert s_new["rb_yards"] == pytest.approx(78.0 * (1 - p * 0.50) / (1 - ref * 0.50))
    # and the literal brief formula is a deeper cut than the re-centered one for the same p
    literal, _ = m.apply_run_game_collapse(PROJECTION, stone, recenter=False)
    assert literal["rb_yards"] < s_new["rb_yards"]


def test_worst_case_never_zeroes_the_back():
    worst = m.compute_run_game_collapse(at_sd(10.0))
    new, _ = m.apply_run_game_collapse(PROJECTION, worst, recenter=False)
    assert new["rb_yards"] >= PROJECTION["rb_yards"] * 0.60  # (1 - p*0.5) with p <= 0.70
    assert new["rb_td_prob"] >= PROJECTION["rb_td_prob"] * 0.70


# --- logging, docs and purity --------------------------------------------------
def test_run_collapse_is_logged_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="nfl_pack"):
        result = m.compute_run_game_collapse(STONEWALL)
        m.apply_run_game_collapse(PROJECTION, result)
    assert any(rec.name == "nfl_pack" and "run_collapse=" in rec.getMessage() for rec in caplog.records)
    own = [rec for rec in caplog.records if rec.name == "nfl_pack.run_game_collapse"]
    assert len(own) == 2 and all(rec.levelno == logging.DEBUG for rec in own)
    assert f"run_collapse={result['run_collapse']:.4f}" in own[0].getMessage()
    assert "rb_yards" in own[1].getMessage() and f"{result['run_collapse']:.4f}" in own[1].getMessage()
    assert m.LOG.name == "nfl_pack.run_game_collapse"


def test_nothing_is_logged_above_debug(caplog):
    with caplog.at_level(logging.INFO, logger="nfl_pack"):
        result = m.compute_run_game_collapse(STONEWALL)
        m.apply_run_game_collapse(PROJECTION, result)
    assert not [rec for rec in caplog.records if rec.name.startswith("nfl_pack")]


def test_module_docstring_keeps_its_honesty_promises():
    doc = m.__doc__
    flat = " ".join(doc.split()).lower()
    assert "HONEST LIMITS" in doc and "priors" in flat
    assert "do not read" in flat and "calibrated probability" in flat  # the weak-association warning
    assert f"{m.BASE_RATE * 100:.1f}" in doc  # the base rate in the event definition matches the constant
    assert "3,742" in doc and "0.176" in doc
    assert "negative_script_prob" in doc  # the deliberate non-alias is documented
    for key in BRIEF_KEYS:
        assert key in doc


def test_module_imports_only_stdlib_and_the_common_helpers():
    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "modules":
                imported.update(f"modules.{alias.name}" for alias in node.names)
            else:
                imported.add(node.module or "")
    allowed = {"__future__", "logging", "typing", "modules.nfl_pack_common"}
    assert imported <= allowed, imported - allowed
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4", "pandas", "numpy",
              "nflreadpy", "polars"}
    assert not {name.split(".")[0] for name in imported} & banned
