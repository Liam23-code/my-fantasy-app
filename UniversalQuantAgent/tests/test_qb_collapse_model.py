"""Tests for the QB collapse model of the NFL model upgrade pack (offline, pure)."""
from __future__ import annotations

import ast
import copy
import logging
import math
from pathlib import Path

import pytest

from modules import nfl_pack_common as c
from modules import qb_collapse_model as m

BRIEF_KEYS = (
    "ol_pass_block_win_rate",
    "ol_injuries",
    "ol_continuity",
    "opp_pressure_rate",
    "opp_blitz_rate",
    "opp_simulated_pressure_success",
    "opp_pass_rush_win_rate",
    "qb_time_to_throw",
    "qb_pressure_epa",
    "qb_scramble_rate",
    "qb_sack_avoidance",
    "qb_release_time",
)
#: direction of each input's effect on collapse risk, from the brief / contract section 4
DIRECTIONS = {
    "ol_pass_block_win_rate": -1,
    "ol_injuries": +1,
    "ol_continuity": -1,
    "opp_pressure_rate": +1,
    "opp_blitz_rate": +1,
    "opp_simulated_pressure_success": +1,
    "opp_pass_rush_win_rate": +1,
    "qb_time_to_throw": +1,
    "qb_pressure_epa": -1,
    "qb_scramble_rate": -1,
    "qb_sack_avoidance": -1,
    "qb_release_time": +1,
}
FACTOR_BY_KEY = {f.key: f for f in m.FACTORS}
CONTRACT_KEYS = (
    "model", "collapse_probability", "probability", "reference_probability", "references", "z_score",
    "coverage", "inputs_used", "inputs_defaulted", "contributions", "drivers", "basis",
)
PROJECTION = {"qb_yards": 250.0, "qb_completion_pct": 0.62, "qb_td_prob": 0.55, "qb_int_prob": 0.22}


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
        if f.key == "ol_injuries":
            value = min(max(value, 0.0), 5.0)
        out[f.key] = value
    return out


def p_of(inputs) -> float:
    return m.compute_qb_collapse(inputs)["collapse_probability"]


def fake_result(p: float, reference: float = m.BASE_RATE) -> dict:
    return {"collapse_probability": p, "probability": p, "reference_probability": reference,
            "references": {"collapse_probability": reference}}


# --- constants and structure --------------------------------------------------
def test_constants_are_the_briefs_coefficients_and_a_measured_base_rate():
    assert m.QB_YARDS_COEFFICIENT == 0.65
    assert m.QB_COMPLETION_PCT_COEFFICIENT == 0.20
    assert m.QB_TD_PROB_COEFFICIENT == 0.35
    assert m.QB_INT_PROB_COEFFICIENT == 0.40
    assert m.EVENT_EPA_THRESHOLD == -0.25 and m.EVENT_MIN_DROPBACKS == 20
    assert 0.12 <= m.BASE_RATE <= 0.18  # the measured 0.1586, rounded
    assert m.GAIN > 0
    assert m.OUTPUT_KEY == "collapse_probability"
    assert "0.1586" in m.EVENT_DEFINITION and "3,658" in m.EVENT_DEFINITION


def test_factors_are_exactly_the_twelve_brief_inputs_with_the_stated_directions():
    assert {f.key for f in m.FACTORS} == set(BRIEF_KEYS) and len(m.FACTORS) == 12
    assert {f.key: f.direction for f in m.FACTORS} == DIRECTIONS
    for f in m.FACTORS:
        assert isinstance(f, c.Factor)
        assert f.source and "nflverse" in f.source  # every reference states its feed
        if f.kind == "rate":
            assert 0.0 < f.mean < 1.0 and f.sd < 0.5  # rates are fractions, not percents


def test_scramble_rate_has_the_smallest_weight_and_is_protective():
    scramble = FACTOR_BY_KEY["qb_scramble_rate"]
    assert scramble.direction == -1
    assert all(scramble.weight < f.weight for f in m.FACTORS if f.key != "qb_scramble_rate")


def test_protection_and_opponent_pressure_outweigh_qb_skill_and_the_trio_does_not_outvote_protection():
    shares = m.CLUSTER_WEIGHT_SHARE
    assert sum(shares.values()) == pytest.approx(1.0, abs=0.01)
    assert shares["protection"] > shares["opponent_pressure"]  # the collinear rush cluster never out-votes the line
    assert m.PRESSURE_TRIO_WEIGHT_SHARE < shares["protection"]
    assert shares["protection"] + shares["opponent_pressure"] > 0.6  # they are the cause
    assert shares["qb"] < shares["protection"]
    assert max(m.FACTORS, key=lambda f: f.weight).key == "ol_pass_block_win_rate"
    assert {k for keys in m.CLUSTERS.values() for k in keys} == set(BRIEF_KEYS)


# --- the apply formulas (the brief, verbatim) -----------------------------------
@pytest.mark.parametrize("p", [0.0, 0.05, 0.16, 0.34, 0.6, 1.0])
def test_literal_formula_equals_the_brief_for_several_probabilities(p):
    new, adjustments = m.apply_qb_collapse(PROJECTION, fake_result(p), recenter=False)
    assert new["qb_yards"] == pytest.approx(250.0 * (1 - p * 0.65))
    assert new["qb_completion_pct"] == pytest.approx(0.62 * (1 - p * 0.20))
    assert new["qb_td_prob"] == pytest.approx(0.55 * (1 - p * 0.35))
    assert new["qb_int_prob"] == pytest.approx(0.22 * (1 + p * 0.40))
    assert [a["field"] for a in adjustments] == ["qb_yards", "qb_completion_pct", "qb_td_prob", "qb_int_prob"]
    assert all(a["applied"] and a["recentered"] is False for a in adjustments)
    assert [a["direction"] for a in adjustments] == ["haircut", "haircut", "haircut", "boost"]


def test_recenter_is_an_exact_noop_at_the_reference():
    new, adjustments = m.apply_qb_collapse(PROJECTION, fake_result(m.BASE_RATE))
    for key, before in PROJECTION.items():
        assert new[key] == pytest.approx(before, rel=1e-12)
    assert all(a["factor"] == 1.0 for a in adjustments)
    # ... whereas the literal formula at the same probability is a real haircut
    literal, _ = m.apply_qb_collapse(PROJECTION, fake_result(m.BASE_RATE), recenter=False)
    assert literal["qb_yards"] < PROJECTION["qb_yards"]


def test_recenter_divides_each_multiplier_by_the_formula_at_the_reference():
    p, ref = 0.40, m.BASE_RATE
    new, _ = m.apply_qb_collapse(PROJECTION, fake_result(p, ref))
    assert new["qb_yards"] == pytest.approx(250.0 * (1 - p * 0.65) / (1 - ref * 0.65))
    assert new["qb_completion_pct"] == pytest.approx(0.62 * (1 - p * 0.20) / (1 - ref * 0.20))
    assert new["qb_td_prob"] == pytest.approx(0.55 * (1 - p * 0.35) / (1 - ref * 0.35))
    assert new["qb_int_prob"] == pytest.approx(0.22 * (1 + p * 0.40) / (1 + ref * 0.40))


def test_apply_reads_the_reference_from_the_result():
    new, _ = m.apply_qb_collapse(PROJECTION, fake_result(0.30, reference=0.30))
    assert new == pytest.approx(PROJECTION)


def test_apply_never_mutates_the_input_and_returns_a_new_dict():
    projection = copy.deepcopy(PROJECTION)
    snapshot = copy.deepcopy(projection)
    result = fake_result(0.5)
    result_snapshot = copy.deepcopy(result)
    new, adjustments = m.apply_qb_collapse(projection, result)
    assert projection == snapshot and result == result_snapshot
    assert new is not projection and new != projection
    assert isinstance(adjustments, list)


def test_missing_field_is_recorded_as_not_applied_not_raised():
    new, adjustments = m.apply_qb_collapse({"qb_yards": 250.0}, fake_result(0.5))
    by_field = {a["field"]: a for a in adjustments}
    assert by_field["qb_yards"]["applied"] is True
    for field in ("qb_completion_pct", "qb_td_prob", "qb_int_prob"):
        assert by_field[field]["applied"] is False and "missing" in by_field[field]["reason"]
        assert field not in new  # nothing is invented


def test_non_numeric_field_is_skipped_and_left_alone():
    new, adjustments = m.apply_qb_collapse({"qb_yards": "n/a", "qb_td_prob": None}, fake_result(0.5))
    assert new == {"qb_yards": "n/a", "qb_td_prob": None}
    assert not any(a["applied"] for a in adjustments)


def test_apply_copies_unrelated_keys_through_untouched():
    projection = dict(PROJECTION, wr_yards=88.0, player="Some QB", team_points=24.5)
    new, adjustments = m.apply_qb_collapse(projection, fake_result(0.6))
    assert new["wr_yards"] == 88.0 and new["player"] == "Some QB" and new["team_points"] == 24.5
    assert {a["field"] for a in adjustments} == {"qb_yards", "qb_completion_pct", "qb_td_prob", "qb_int_prob"}


def test_completion_pct_is_capped_on_fraction_and_percent_scales():
    # p=0 re-centered is a boost of 1/(1 - 0.16*0.2) = 1.033: enough to push 0.99 / 99.0 past the cap
    fraction, _ = m.apply_qb_collapse({"qb_completion_pct": 0.99}, fake_result(0.0))
    percent, _ = m.apply_qb_collapse({"qb_completion_pct": 99.0}, fake_result(0.0))
    assert fraction["qb_completion_pct"] == 1.0
    assert percent["qb_completion_pct"] == 100.0
    uncapped, _ = m.apply_qb_collapse({"qb_completion_pct": 65.0}, fake_result(0.0))
    assert uncapped["qb_completion_pct"] == pytest.approx(65.0 / (1 - m.BASE_RATE * 0.20))


def test_touchdown_and_interception_probabilities_are_capped():
    boosted, _ = m.apply_qb_collapse({"qb_int_prob": 0.98, "qb_td_prob": 0.99}, fake_result(1.0))
    assert boosted["qb_int_prob"] == 1.0  # 0.98 * 1.4/1.064 would exceed 1
    cheap, _ = m.apply_qb_collapse({"qb_td_prob": 0.99}, fake_result(0.0))  # re-centered boost of 1/(1-0.16*0.35)
    assert cheap["qb_td_prob"] == 1.0
    percent, _ = m.apply_qb_collapse({"qb_int_prob": 98.0}, fake_result(1.0))
    assert percent["qb_int_prob"] == 100.0


def test_values_between_one_and_one_and_a_half_are_not_capped_so_a_boost_is_not_cancelled():
    # 1.2 can be a percent-scale interception rate (1.2%) or 1.3 an expected TD count: never a
    # fraction-scale probability.  Capping at 1.0 would cut it; capping at the value itself would
    # silently cancel the boost while the audit still says applied=True.
    boosted, adjustments = m.apply_qb_collapse({"qb_int_prob": 1.2}, fake_result(1.0))
    assert boosted["qb_int_prob"] == pytest.approx(1.2 * 1.4 / (1 + m.BASE_RATE * 0.40))
    assert adjustments[-1]["applied"] and adjustments[-1]["after"] > adjustments[-1]["before"]
    lifted, _ = m.apply_qb_collapse({"qb_td_prob": 1.3}, fake_result(0.0))  # re-centered boost, p below reference
    assert lifted["qb_td_prob"] == pytest.approx(1.3 / (1 - m.BASE_RATE * 0.35))
    cut, _ = m.apply_qb_collapse({"qb_td_prob": 1.3}, fake_result(0.6))  # a haircut in that band still bites
    assert cut["qb_td_prob"] < 1.3


def test_cap_boundaries_one_is_a_fraction_and_percent_scale_stops_at_one_hundred():
    at_one, _ = m.apply_qb_collapse({"qb_td_prob": 1.0}, fake_result(0.0))  # exactly 1.0 is a fraction: capped
    assert at_one["qb_td_prob"] == 1.0
    percent, _ = m.apply_qb_collapse({"qb_td_prob": 98.0}, fake_result(0.0))  # 1.5 < v: percent scale
    assert percent["qb_td_prob"] == 100.0
    already_over, _ = m.apply_qb_collapse({"qb_td_prob": 140.0}, fake_result(0.0))  # nonsense input: not reduced
    assert already_over["qb_td_prob"] >= 140.0


@pytest.mark.parametrize("bad", [None, {}, {"collapse_probability": "x"}, {"probability": float("nan")}, "nope"])
def test_invalid_result_is_a_recorded_noop(bad):
    new, adjustments = m.apply_qb_collapse(PROJECTION, bad)
    assert new == PROJECTION
    assert len(adjustments) == 4 and not any(a["applied"] for a in adjustments)
    assert all("collapse_probability" in a["reason"] for a in adjustments)


def test_apply_falls_back_to_probability_alias_and_clamps_to_unit_interval():
    new, _ = m.apply_qb_collapse({"qb_yards": 100.0}, {"probability": 1.7}, recenter=False)
    assert new["qb_yards"] == pytest.approx(100.0 * (1 - 1.0 * 0.65))
    new, _ = m.apply_qb_collapse({"qb_yards": 100.0}, {"collapse_probability": -0.3}, recenter=False)
    assert new["qb_yards"] == pytest.approx(100.0)


def test_apply_without_a_reference_in_the_result_uses_base_rate():
    new, _ = m.apply_qb_collapse(PROJECTION, {"collapse_probability": m.BASE_RATE})
    assert new == pytest.approx(PROJECTION)


def test_apply_falls_back_to_reference_probability_when_references_is_absent_or_malformed():
    base = {"collapse_probability": 0.30, "reference_probability": 0.30}
    for result in (base, dict(base, references="junk"), dict(base, references={"collapse_probability": None})):
        new, _ = m.apply_qb_collapse(PROJECTION, result)
        assert new == pytest.approx(PROJECTION), result  # reference 0.30 -> a no-op, not a BASE_RATE re-centre
    # ``references`` wins over the flat alias when both are present
    both = {"collapse_probability": 0.30, "reference_probability": 0.10, "references": {"collapse_probability": 0.30}}
    new, _ = m.apply_qb_collapse(PROJECTION, both)
    assert new == pytest.approx(PROJECTION)


# --- compute: contract ---------------------------------------------------------
def test_result_carries_every_contract_key_and_the_extras():
    result = m.compute_qb_collapse(at_sd(1.0))
    for key in CONTRACT_KEYS:
        assert key in result, key
    assert result["model"] == "qb_collapse"
    assert result["probability"] == result["collapse_probability"]
    assert result["references"] == {"collapse_probability": result["reference_probability"]}
    assert 0.0 <= result["collapse_probability"] <= 1.0
    assert result["inputs_rejected"] == [] and set(result["cluster_effects"]) == set(m.CLUSTERS)
    assert result["event"] == m.EVENT_DEFINITION


def test_neutral_in_reference_out():
    empty = m.compute_qb_collapse({})
    none = m.compute_qb_collapse(None)
    average = m.compute_qb_collapse(neutral())
    assert empty["collapse_probability"] == none["collapse_probability"] == average["collapse_probability"]
    assert empty["collapse_probability"] == empty["references"]["collapse_probability"] == m.BASE_RATE
    assert average["z_score"] == pytest.approx(0.0, abs=1e-6)
    assert average["coverage"] == 1.0 and empty["coverage"] == 0.0
    assert empty["inputs_defaulted"] == [f.key for f in m.FACTORS]


def test_basis_declares_priors_and_names_the_event_and_feed():
    basis = m.compute_qb_collapse({})["basis"]
    assert "PRIOR" in basis.upper() and "not fitted" in basis
    assert "EPA/dropback <= -0.25" in basis and "nflverse" in basis
    assert "0 of 12" in basis
    assert "6 of 12" in m.compute_qb_collapse(dict(list(neutral().items())[:6]))["basis"]


def test_cluster_effects_sum_to_the_z_score():
    result = m.compute_qb_collapse(at_sd(1.0))
    assert sum(result["cluster_effects"].values()) == pytest.approx(result["z_score"], abs=2e-3)
    assert result["cluster_effects"]["protection"] > 0


# --- compute: monotonicity, bounds, gain ---------------------------------------
@pytest.mark.parametrize("key", BRIEF_KEYS)
def test_each_input_is_monotone_in_its_stated_direction(key):
    ps = []
    for k in (-2, -1, 0, 1, 2):
        inputs = neutral()
        inputs.update(at_sd(k, keys={key}))
        ps.append(p_of(inputs))
    assert ps == sorted(ps), (key, ps)
    assert ps[-1] > ps[0]
    # and raw direction check: a worse raw value (in the risk direction) beats the average
    factor = FACTOR_BY_KEY[key]
    worse = neutral()
    worse[key] = factor.mean + DIRECTIONS[key] * factor.sd
    if factor.kind == "rate":
        worse[key] = min(max(worse[key], 0.0), 1.0)
    assert p_of(worse) > m.BASE_RATE


def test_probabilities_stay_in_unit_interval_and_below_ceiling_at_extremes():
    worst = m.compute_qb_collapse({k: v for k, v in at_sd(10.0).items()})  # clipped at 3 SD per input
    best = m.compute_qb_collapse(at_sd(-10.0))
    assert 0.0 <= best["collapse_probability"] < m.BASE_RATE < worst["collapse_probability"] <= 1.0
    assert worst["collapse_probability"] < 0.95
    assert best["collapse_probability"] > 0.0


def test_gain_anchors_all_plus_one_and_two_sd():
    plus1 = p_of(at_sd(1.0))
    plus2 = p_of(at_sd(2.0))
    minus1 = p_of(at_sd(-1.0))
    assert 0.30 <= plus1 <= 0.40  # the documented ~0.34
    assert 0.50 <= plus2 <= 0.70  # the documented ~0.58, well below 0.95
    assert 0.04 <= minus1 <= 0.10
    assert minus1 < m.BASE_RATE < plus1 < plus2 < 0.8


def test_scramble_rate_moves_the_probability_least_of_all_inputs():
    effects = {}
    for key in BRIEF_KEYS:
        inputs = neutral()
        inputs.update(at_sd(1.0, keys={key}))
        effects[key] = p_of(inputs) - m.BASE_RATE
    assert min(effects, key=effects.get) == "qb_scramble_rate"
    assert effects["qb_scramble_rate"] > 0  # still protective in the stated direction
    assert effects["ol_pass_block_win_rate"] == max(effects.values())


def test_pressure_cluster_alone_does_not_out_vote_protection_alone():
    protection = p_of(at_sd(2.0, keys=set(m.CLUSTERS["protection"])))
    trio = p_of(at_sd(2.0, keys=set(m.PRESSURE_TRIO)))
    opponent = p_of(at_sd(2.0, keys=set(m.CLUSTERS["opponent_pressure"])))
    qb = p_of(at_sd(2.0, keys=set(m.CLUSTERS["qb"])))
    assert trio < protection and opponent < protection
    assert qb < protection


def test_more_negative_pressure_epa_raises_collapse_risk():
    ordered = [p_of({"qb_pressure_epa": v}) for v in (-0.1, -0.45, -0.8, -1.2)]
    assert ordered == sorted(ordered) and ordered[0] < m.BASE_RATE < ordered[-1]


# --- compute: bad data ---------------------------------------------------------
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf"), "n/a", "", "  ", True, False, [], {}, object()])
def test_garbage_values_are_missing_for_every_input(bad):
    for key in BRIEF_KEYS:
        result = m.compute_qb_collapse({key: bad})
        assert result["inputs_used"] == [], (key, bad)
        assert result["collapse_probability"] == m.BASE_RATE
        assert result["inputs_rejected"] == []  # unreadable is "absent", not "rejected"


def test_garbage_container_never_raises():
    for bad in (None, {}, [], "text", 7, ("a", "b"), [("ol_continuity", 0.5)]):
        result = m.compute_qb_collapse(bad)
        assert 0.0 <= result["collapse_probability"] <= 1.0


def test_a_pandas_row_is_read_like_a_mapping_not_silently_neutral():
    pd = pytest.importorskip("pandas")
    values = at_sd(1.0)
    row = pd.Series(values)
    from_row = m.compute_qb_collapse(row)
    assert from_row["collapse_probability"] > m.BASE_RATE
    assert from_row == m.compute_qb_collapse(values)
    with_na = pd.Series({"ol_continuity": float("nan"), "opp_pressure_rate": 0.36, "qb_sack_avoidance": pd.NA})
    assert m.compute_qb_collapse(with_na)["inputs_used"] == ["opp_pressure_rate"]


def test_fractional_injury_counts_are_accepted_questionable_is_half_a_starter():
    p0, p_half, p1 = p_of({"ol_injuries": 0}), p_of({"ol_injuries": 0.5}), p_of({"ol_injuries": 1})
    assert p0 < p_half < p1
    assert m.compute_qb_collapse({"ol_injuries": 0.5})["inputs_rejected"] == []


@pytest.mark.parametrize(
    "key, value",
    [
        ("ol_injuries", -1),            # negative count
        ("ol_injuries", 99),            # more starters than a line has
        ("qb_time_to_throw", -2.7),     # negative seconds
        ("qb_time_to_throw", 0),        # zero seconds
        ("qb_time_to_throw", 2700),     # milliseconds passed as seconds
        ("qb_release_time", -0.1),
        ("qb_release_time", 9999),
        ("qb_pressure_epa", -45.0),     # a percent / scale error
        ("qb_sack_avoidance", 0.07),    # the sack RATE passed by mistake
        ("ol_continuity", -0.2),        # negative rate
        ("ol_continuity", 150),         # over 100%
        ("opp_pressure_rate", 1e9),     # huge
        ("opp_blitz_rate", "-5%"),
    ],
)
def test_impossible_values_are_rejected_not_clamped(key, value):
    result = m.compute_qb_collapse({key: value})
    assert key not in result["inputs_used"]
    assert result["collapse_probability"] == m.BASE_RATE  # not "max risk", not "min risk": just absent
    assert [r["key"] for r in result["inputs_rejected"]] == [key]
    assert result["inputs_rejected"][0]["reason"]


def test_extreme_but_possible_values_are_clipped_at_three_sd_not_rejected():
    five_out = m.compute_qb_collapse({"ol_injuries": 5})
    four_out = m.compute_qb_collapse({"ol_injuries": 4})
    assert five_out["contributions"]["ol_injuries"]["z"] == pytest.approx(c.Z_CLIP)
    assert five_out["collapse_probability"] == four_out["collapse_probability"] > m.BASE_RATE
    slow = m.compute_qb_collapse({"qb_time_to_throw": 6.0})
    assert slow["contributions"]["qb_time_to_throw"]["z"] == pytest.approx(c.Z_CLIP)
    assert slow["inputs_rejected"] == []


def test_percent_and_fraction_spellings_agree():
    reference = p_of({"ol_continuity": 0.70, "opp_pressure_rate": 0.36, "qb_sack_avoidance": 0.90})
    assert reference == p_of({"ol_continuity": 70, "opp_pressure_rate": 36, "qb_sack_avoidance": 90})
    assert reference == p_of({"ol_continuity": "70%", "opp_pressure_rate": "36.0%", "qb_sack_avoidance": "90"})
    assert reference > m.BASE_RATE


def test_rate_of_exactly_one_is_a_fraction_not_one_percent():
    # 1.0 sack avoidance = never sacked (a fraction), which is better than average, never "1%"
    assert p_of({"qb_sack_avoidance": 1.0}) < m.BASE_RATE


def test_missing_data_shrinks_toward_the_reference():
    bad = at_sd(1.5)
    two_keys = {"ol_pass_block_win_rate": bad["ol_pass_block_win_rate"], "opp_pressure_rate": bad["opp_pressure_rate"]}
    partial = m.compute_qb_collapse(two_keys)
    full = m.compute_qb_collapse(bad)
    assert partial["coverage"] < full["coverage"] == 1.0
    assert m.BASE_RATE < partial["collapse_probability"] < full["collapse_probability"]
    assert abs(partial["z_score"]) < abs(full["z_score"])
    assert set(partial["inputs_defaulted"]) == set(BRIEF_KEYS) - set(two_keys)


def test_deterministic_and_order_independent_and_non_mutating():
    inputs = at_sd(1.0)
    snapshot = copy.deepcopy(inputs)
    first = m.compute_qb_collapse(inputs)
    second = m.compute_qb_collapse(dict(reversed(list(inputs.items()))))
    assert first == second
    assert inputs == snapshot


# --- compute: vocabulary -------------------------------------------------------
@pytest.mark.parametrize("key", BRIEF_KEYS)
def test_every_brief_named_input_is_accepted_and_used(key):
    factor = FACTOR_BY_KEY[key]
    value = factor.mean + DIRECTIONS[key] * factor.sd
    if factor.kind == "rate":
        value = min(max(value, 0.0), 1.0)
    result = m.compute_qb_collapse({key: value})
    assert result["inputs_used"] == [key]
    assert result["contributions"][key]["z"] == pytest.approx(1.0, abs=0.05)  # +1 SD in the risk direction
    assert result["collapse_probability"] > m.BASE_RATE


def test_aliases_are_accepted_and_the_canonical_key_wins():
    via_alias = m.compute_qb_collapse({"time_to_throw": 3.2, "pass_block_win_rate": 0.6, "ol_injury_count": 2})
    assert set(via_alias["inputs_used"]) == {"qb_time_to_throw", "ol_pass_block_win_rate", "ol_injuries"}
    both = m.compute_qb_collapse({"qb_time_to_throw": 2.5, "time_to_throw": 3.5})
    assert both["contributions"]["qb_time_to_throw"]["value"] == 2.5


def test_ol_health_is_converted_when_injuries_are_absent_and_never_overrides_them():
    derived = m.compute_qb_collapse({"ol_health": 0.6})  # 40% healthy-less = 2 starters out
    assert derived["inputs_derived"] == {"ol_injuries": "5 * (1 - ol_health)"}
    assert derived["contributions"]["ol_injuries"]["value"] == pytest.approx(2.0)
    assert derived["collapse_probability"] == p_of({"ol_injuries": 2})
    explicit = m.compute_qb_collapse({"ol_injuries": 0, "ol_health": 0.2})
    assert explicit["contributions"]["ol_injuries"]["value"] == 0.0 and explicit["inputs_derived"] == {}
    assert m.compute_qb_collapse({"ol_health": 400})["inputs_rejected"][0]["key"] == "ol_health"


def test_unrelated_keys_are_ignored():
    result = m.compute_qb_collapse({"ol_run_block_win_rate": 0.1, "vegas_total": 55, "foo": "bar"})
    assert result["inputs_used"] == [] and result["collapse_probability"] == m.BASE_RATE


# --- scenarios -----------------------------------------------------------------
FORTRESS = {
    "ol_pass_block_win_rate": 0.78, "ol_injuries": 0, "ol_continuity": 0.92,
    "opp_pressure_rate": 0.25, "opp_pass_rush_win_rate": 0.25, "opp_simulated_pressure_success": 0.22,
    "opp_blitz_rate": 0.20,
    "qb_pressure_epa": -0.25, "qb_sack_avoidance": 0.955, "qb_time_to_throw": 2.6, "qb_release_time": 2.15,
    "qb_scramble_rate": 0.08,
}
DISASTER = {
    "ol_pass_block_win_rate": 0.60, "ol_injuries": 2, "ol_continuity": 0.70,
    "opp_pressure_rate": 0.36, "opp_pass_rush_win_rate": 0.36, "opp_simulated_pressure_success": 0.32,
    "opp_blitz_rate": 0.35,
    "qb_pressure_epa": -0.80, "qb_sack_avoidance": 0.89, "qb_time_to_throw": 3.1, "qb_release_time": 2.6,
    "qb_scramble_rate": 0.02,
}


def test_realistic_scenarios_order_as_football_says():
    fortress, average, disaster = p_of(FORTRESS), p_of(neutral()), p_of(DISASTER)
    assert fortress < average < disaster
    assert fortress < 0.08 and disaster > 0.30
    assert disaster < 0.60  # bad is not certain: collapse is partly luck


def test_protection_matters_even_for_a_pressure_proof_quarterback():
    qb_good = {k: v for k, v in FORTRESS.items() if k.startswith("qb_")}
    bad_line = dict(qb_good, ol_pass_block_win_rate=0.58, ol_injuries=3, ol_continuity=0.65)
    assert p_of(bad_line) > p_of(qb_good) and p_of(bad_line) > m.BASE_RATE * 0.8


def test_elite_pass_rush_raises_risk_for_the_same_line_and_quarterback():
    base = dict(neutral())
    rush = dict(base, opp_pressure_rate=0.37, opp_pass_rush_win_rate=0.37, opp_simulated_pressure_success=0.33, opp_blitz_rate=0.33)
    assert p_of(rush) > p_of(base)


def test_end_to_end_a_disaster_game_trims_the_projection_and_a_fortress_lifts_it():
    disaster = m.compute_qb_collapse(DISASTER)
    fortress = m.compute_qb_collapse(FORTRESS)
    d_new, _ = m.apply_qb_collapse(PROJECTION, disaster)
    f_new, _ = m.apply_qb_collapse(PROJECTION, fortress)
    assert d_new["qb_yards"] < PROJECTION["qb_yards"] < f_new["qb_yards"]
    assert d_new["qb_completion_pct"] < PROJECTION["qb_completion_pct"] < f_new["qb_completion_pct"]
    assert d_new["qb_td_prob"] < PROJECTION["qb_td_prob"] < f_new["qb_td_prob"]
    assert d_new["qb_int_prob"] > PROJECTION["qb_int_prob"] > f_new["qb_int_prob"]
    # a league-average game leaves the engine's number alone (default re-centering)
    neutral_new, _ = m.apply_qb_collapse(PROJECTION, m.compute_qb_collapse({}))
    assert neutral_new == pytest.approx(PROJECTION)
    # the coefficient ordering of the brief: yards take the biggest haircut, completion the smallest
    p, ref = disaster["collapse_probability"], disaster["reference_probability"]
    cut = {k: 1 - d_new[k] / PROJECTION[k] for k in ("qb_yards", "qb_completion_pct", "qb_td_prob")}
    assert cut["qb_yards"] > cut["qb_td_prob"] > cut["qb_completion_pct"] > 0
    assert d_new["qb_yards"] == pytest.approx(250.0 * (1 - p * 0.65) / (1 - ref * 0.65))


# --- logging and purity --------------------------------------------------------
def test_collapse_probability_is_logged_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="nfl_pack"):
        result = m.compute_qb_collapse(DISASTER)
        m.apply_qb_collapse(PROJECTION, result)
    messages = [rec.getMessage() for rec in caplog.records]
    assert any(rec.name == "nfl_pack" and "collapse_probability=" in rec.getMessage() for rec in caplog.records)
    own = [rec for rec in caplog.records if rec.name == "nfl_pack.qb_collapse"]
    assert len(own) == 2 and all(rec.levelno == logging.DEBUG for rec in own)
    assert f"collapse_probability={result['collapse_probability']:.4f}" in own[0].getMessage()
    assert "qb_yards" in own[1].getMessage() and f"{result['collapse_probability']:.4f}" in own[1].getMessage()
    assert messages  # something was emitted


def test_nothing_is_logged_above_debug(caplog):
    with caplog.at_level(logging.INFO, logger="nfl_pack"):
        m.compute_qb_collapse(DISASTER)
    assert not [rec for rec in caplog.records if rec.name.startswith("nfl_pack")]


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
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4", "pandas", "numpy", "nflreadpy"}
    assert not {name.split(".")[0] for name in imported} & banned


def test_a_missing_or_failing_clean_path_does_not_leak_float_artifacts():
    # outputs are rounded for display / JSON: no 0.15999999999 style numbers
    result = m.compute_qb_collapse(at_sd(1.0))
    assert result["collapse_probability"] == round(result["collapse_probability"], 4)
    assert math.isfinite(result["z_score"])
