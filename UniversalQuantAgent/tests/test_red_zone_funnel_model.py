"""Offline tests for the NFL upgrade pack's red-zone funnel model (``modules.red_zone_funnel_model``)."""
from __future__ import annotations

import ast
import copy
import logging
import math
from pathlib import Path

import pytest

from modules import red_zone_funnel_model as rz

FACTOR = {f.key: f for f in rz.FACTORS}
RUN_G = "opp_rz_run_defense_grade"
PASS_G = "opp_rz_pass_defense_grade"
RATE = "team_rz_pass_rate"
QB = "qb_rz_efficiency"
RB = "rb_rz_efficiency"
TE = "te_rz_target_share"

# A soft-pass / stout-run red-zone defense: funnels toward the pass.
PASS_FUNNEL_D = {RUN_G: 85.0, PASS_G: 15.0}
# The mirror image: funnels toward the run.
RUN_FUNNEL_D = {RUN_G: 15.0, PASS_G: 85.0}


def at(key: str, sd: float) -> float:
    """The factor's league mean plus ``sd`` standard deviations."""
    factor = FACTOR[key]
    return factor.mean + sd * factor.sd


def average_inputs() -> dict:
    return {f.key: f.mean for f in rz.FACTORS}


def funnels(inputs):
    out = rz.compute_red_zone_funnel(inputs)
    return out["rz_pass_funnel"], out["rz_run_funnel"]


def signed_result(f_pass: float, f_run: float) -> dict:
    return {"rz_pass_funnel": f_pass, "rz_run_funnel": f_run, "references": {"rz_pass_funnel": 0.0, "rz_run_funnel": 0.0}}


PROJECTION = {
    "qb_yards": 255.0,
    "qb_td_prob": 0.62,
    "rb_yards": 71.0,
    "rb_td_prob": 0.38,
    "te_yards": 44.0,
    "te_td_prob": 0.29,
    "wr_td_prob": 0.33,
    "player": "Test Player",
}


# --- constants and structure ---------------------------------------------------
def test_brief_coefficients_are_verbatim():
    assert rz.QB_TD_PROB_COEFFICIENT == 0.35
    assert rz.RB_TD_PROB_COEFFICIENT == 0.30
    assert rz.TE_TD_PROB_COEFFICIENT == 0.25


def test_public_constants_and_factor_table_are_well_formed():
    keys = [f.key for f in rz.FACTORS]
    assert len(keys) == len(set(keys))
    assert set(keys) == {RUN_G, PASS_G, RATE, QB, RB, TE}
    for f in rz.FACTORS:
        assert f.sd > 0 and f.weight > 0 and f.direction in (-1, 1) and f.source
        if f.kind == "rate":
            assert 0.0 < f.mean < 1.0 and f.sd < 0.5
    assert 0.0 < rz.BASE_RATE < 1.0
    assert rz.GAIN > 0
    # Causal order of the priors: contrast first, then team tendency, then efficiency, then TE share.
    assert FACTOR[RUN_G].weight == FACTOR[PASS_G].weight  # equal weights are what make the contrast antisymmetric
    assert FACTOR[RUN_G].weight >= FACTOR[RATE].weight >= FACTOR[QB].weight >= FACTOR[TE].weight
    # Grades are labelled as a prior (a property of a percentile scale), not as a measured quantity.
    assert FACTOR[RUN_G].source.startswith("prior") and FACTOR[PASS_G].source.startswith("prior")
    assert FACTOR[RUN_G].mean == 50.0


def test_basis_and_result_say_weights_are_priors():
    result = rz.compute_red_zone_funnel({})
    assert "priors" in result["basis"].lower() and "not fitted" in result["basis"].lower()
    assert "nflverse" in result["basis"]


# --- compute: neutral, contract keys ----------------------------------------------
@pytest.mark.parametrize("inputs", [None, {}, average_inputs()])
def test_neutral_in_reference_out(inputs):
    result = rz.compute_red_zone_funnel(inputs)
    assert result["rz_pass_funnel"] == 0.0
    assert result["rz_run_funnel"] == 0.0
    assert result["references"] == {"rz_pass_funnel": 0.0, "rz_run_funnel": 0.0}
    assert result["probability"] == result["reference_probability"] == rz.BASE_RATE
    assert result["pass_amplifier"] == 1.0 and result["run_amplifier"] == 1.0


def test_neutral_input_and_average_input_agree_with_references():
    base = rz.compute_red_zone_funnel({})
    avg = rz.compute_red_zone_funnel(average_inputs())
    for key in ("rz_pass_funnel", "rz_run_funnel"):
        assert base[key] == avg[key] == base["references"][key]
    assert math.copysign(1.0, base["rz_pass_funnel"]) == 1.0  # never -0.0


def test_result_carries_every_contract_key():
    result = rz.compute_red_zone_funnel({RUN_G: 70, PASS_G: 40})
    for key in (
        "model", "rz_pass_funnel", "rz_run_funnel", "probability", "reference_probability", "references",
        "z_score", "coverage", "inputs_used", "inputs_defaulted", "contributions", "drivers", "basis",
    ):
        assert key in result, key
    assert result["model"] == "red_zone_funnel"
    assert set(result["inputs_used"]) == {RUN_G, PASS_G}
    assert set(result["inputs_defaulted"]) == {RATE, QB, RB, TE}
    assert result["drivers"] and all(isinstance(line, str) for line in result["drivers"])


@pytest.mark.parametrize("bad", [[1, 2], "red zone", 7, 3.5, object(), ("a",), True])
def test_non_mapping_inputs_return_neutral_instead_of_raising(bad):
    result = rz.compute_red_zone_funnel(bad)
    assert funnels(bad) == (0.0, 0.0)
    assert result["coverage"] == 0.0 and result["inputs_used"] == []


def test_compute_does_not_mutate_inputs_and_is_deterministic():
    inputs = {RUN_G: 77.0, PASS_G: "31", RATE: "0.62", QB: None, "junk": object(), TE: float("nan")}
    snapshot = {k: v for k, v in inputs.items() if k != "junk"}
    first = rz.compute_red_zone_funnel(inputs)
    second = rz.compute_red_zone_funnel(dict(reversed(list(inputs.items()))))
    assert first == second
    assert {k: v for k, v in inputs.items() if k != "junk"}.keys() == snapshot.keys()
    assert inputs[RUN_G] == 77.0 and inputs[PASS_G] == "31" and inputs[TE] != inputs[TE]


# --- compute: design ----------------------------------------------------------------
def test_stout_run_soft_pass_defense_funnels_to_the_pass_and_the_mirror_flips():
    f_pass, f_run = funnels(PASS_FUNNEL_D)
    assert f_pass > 0.2 and f_run < -0.2
    m_pass, m_run = funnels(RUN_FUNNEL_D)
    assert m_pass < -0.2 and m_run > 0.2


@pytest.mark.parametrize("grade", [0.0, 10.0, 50.0, 90.0, 100.0])
def test_uniformly_strong_or_weak_defense_funnels_nothing(grade):
    assert funnels({RUN_G: grade, PASS_G: grade}) == (0.0, 0.0)


@pytest.mark.parametrize("run_g, pass_g", [(80, 20), (65, 40), (100, 0), (55, 50), (30, 75), (0, 100)])
def test_antisymmetry_of_the_grade_contrast(run_g, pass_g):
    forward = funnels({RUN_G: run_g, PASS_G: pass_g})
    swapped = funnels({RUN_G: pass_g, PASS_G: run_g})
    assert swapped[0] == -forward[0]
    assert swapped[1] == -forward[1]
    # With a neutral offense the two phases are exact mirrors of each other.
    assert forward[1] == -forward[0]


def test_antisymmetry_holds_with_an_offense_present():
    offense = {RATE: at(RATE, 1.0), QB: at(QB, 0.8), TE: at(TE, -0.5), RB: at(RB, 0.4)}
    forward = funnels({**offense, RUN_G: 80, PASS_G: 25})
    swapped = funnels({**offense, RUN_G: 25, PASS_G: 80})
    assert swapped[0] == -forward[0] and swapped[1] == -forward[1]


def test_grades_are_monotone_in_the_stated_direction():
    previous = None
    for grade in range(0, 101, 5):
        f_pass, f_run = funnels({RUN_G: grade, PASS_G: 50})
        if previous is not None:
            assert f_pass >= previous[0] and f_run <= previous[1]
        previous = (f_pass, f_run)
    assert funnels({RUN_G: 100, PASS_G: 50})[0] > funnels({RUN_G: 0, PASS_G: 50})[0]
    previous = None
    for grade in range(0, 101, 5):
        f_pass, f_run = funnels({RUN_G: 50, PASS_G: grade})
        if previous is not None:
            assert f_pass <= previous[0] and f_run >= previous[1]
        previous = (f_pass, f_run)
    assert funnels({RUN_G: 50, PASS_G: 0})[0] > funnels({RUN_G: 50, PASS_G: 100})[0]


def test_amplifier_inputs_scale_the_pass_funnel_in_the_stated_direction():
    baseline = funnels(PASS_FUNNEL_D)
    for key in (RATE, QB, TE):
        low = funnels({**PASS_FUNNEL_D, key: at(key, -1.5)})
        high = funnels({**PASS_FUNNEL_D, key: at(key, +1.5)})
        assert low[0] < baseline[0] < high[0], key
        assert low[0] > 0 and high[0] < 1  # damps and amplifies, never flips
    # The RB input belongs to the run phase and leaves the pass funnel alone.
    assert funnels({**PASS_FUNNEL_D, RB: at(RB, 2.0)})[0] == baseline[0]


def test_amplifier_inputs_scale_the_run_funnel_in_the_stated_direction():
    soft_run_d = RUN_FUNNEL_D  # soft run defense, stout pass defense -> positive run funnel
    baseline = funnels(soft_run_d)
    assert baseline[1] > 0
    assert funnels({**soft_run_d, RB: at(RB, 1.5)})[1] > baseline[1] > funnels({**soft_run_d, RB: at(RB, -1.5)})[1]
    # A pass-heavy team is less able to use a soft run defense; a run-heavy one more.
    assert funnels({**soft_run_d, RATE: at(RATE, 1.5)})[1] < baseline[1] < funnels({**soft_run_d, RATE: at(RATE, -1.5)})[1]
    # QB and TE inputs belong to the pass phase.
    assert funnels({**soft_run_d, QB: at(QB, 2.0), TE: at(TE, 2.0)})[1] == baseline[1]


def test_amplifiers_scale_the_magnitude_of_an_adverse_funnel_too():
    """Documented semantics: the offense inputs are monotone in |funnel|, so a NEGATIVE funnel gets more negative.

    A pass-stout / run-soft defense makes ``rz_pass_funnel`` negative; a pass-heavy offense with a hot QB and TE
    is more exposed to it (more negative), a run-first one less. The sign never flips either way.
    """
    adverse = RUN_FUNNEL_D
    baseline = funnels(adverse)[0]
    assert baseline < 0
    for key in (RATE, QB, TE):
        hot = funnels({**adverse, key: at(key, +1.5)})[0]
        cold = funnels({**adverse, key: at(key, -1.5)})[0]
        assert hot < baseline < cold < 0, key
    # Run phase, mirror image: PASS_FUNNEL_D makes rz_run_funnel negative; a run-first offense is more exposed.
    run_base = funnels(PASS_FUNNEL_D)[1]
    assert run_base < 0
    assert funnels({**PASS_FUNNEL_D, RB: at(RB, 1.5)})[1] < run_base < funnels({**PASS_FUNNEL_D, RB: at(RB, -1.5)})[1] < 0
    assert funnels({**PASS_FUNNEL_D, RATE: at(RATE, -1.5)})[1] < run_base < funnels({**PASS_FUNNEL_D, RATE: at(RATE, 1.5)})[1] < 0


def test_amplifiers_move_magnitude_never_sign_and_never_create_a_funnel_from_nothing():
    offense_extreme = {RATE: at(RATE, 3), QB: at(QB, 3), TE: at(TE, 3), RB: at(RB, 3)}
    assert funnels({**offense_extreme, RUN_G: 60, PASS_G: 60}) == (0.0, 0.0)
    assert funnels(offense_extreme) == (0.0, 0.0)
    for grades in (PASS_FUNNEL_D, RUN_FUNNEL_D):
        raw = funnels(grades)
        for sd in (-3, -1, 1, 3):
            amped = funnels({**grades, **{k: at(k, sd) for k in (RATE, QB, TE, RB)}})
            assert math.copysign(1, amped[0]) == math.copysign(1, raw[0])
            assert math.copysign(1, amped[1]) == math.copysign(1, raw[1])


def test_a_team_that_cannot_run_gains_little_from_a_soft_run_defense():
    average_offense = funnels(RUN_FUNNEL_D)[1]
    cannot_run = funnels({**RUN_FUNNEL_D, RATE: at(RATE, 2.0), RB: at(RB, -2.0)})[1]
    can_run = funnels({**RUN_FUNNEL_D, RATE: at(RATE, -2.0), RB: at(RB, 2.0)})[1]
    assert 0 < cannot_run < 0.7 * average_offense
    assert can_run > 1.2 * average_offense


def test_amplifier_values_are_bounded_and_ordered():
    top = rz.compute_red_zone_funnel({RATE: at(RATE, 3), QB: at(QB, 3), TE: at(TE, 3), RB: at(RB, -3)})
    bottom = rz.compute_red_zone_funnel({RATE: at(RATE, -3), QB: at(QB, -3), TE: at(TE, -3), RB: at(RB, 3)})
    lo, hi = 1 - rz.AMP_RANGE, 1 + rz.AMP_RANGE
    assert lo <= bottom["pass_amplifier"] < 1.0 < top["pass_amplifier"] <= hi
    assert lo <= top["run_amplifier"] < 1.0 < bottom["run_amplifier"] <= hi


def test_everything_in_the_favourable_direction_gives_a_strong_but_bounded_funnel():
    plus1 = {RUN_G: at(RUN_G, 1), PASS_G: at(PASS_G, -1), RATE: at(RATE, 1), QB: at(QB, 1), RB: at(RB, 1), TE: at(TE, 1)}
    plus2 = {RUN_G: 100.0, PASS_G: 0.0, RATE: at(RATE, 2), QB: at(QB, 2), RB: at(RB, 2), TE: at(TE, 2)}
    f1 = funnels(plus1)[0]
    f2 = funnels(plus2)[0]
    assert 0.30 < f1 < 0.60
    assert f1 < f2 < 0.85  # "well below" the +-1 ceiling


# --- compute: direct funnels, bad data, units ---------------------------------------------
def test_direct_funnels_take_precedence_over_grades_and_pass_through_at_a_neutral_offense():
    contradicting_grades = {RUN_G: 95.0, PASS_G: 5.0}  # grades say: funnel to the pass
    out = rz.compute_red_zone_funnel({**contradicting_grades, "opp_rz_pass_funnel": -0.5, "opp_rz_run_funnel": 0.25})
    assert out["rz_pass_funnel"] == pytest.approx(-0.5, abs=1e-4)
    assert out["rz_run_funnel"] == pytest.approx(0.25, abs=1e-4)
    assert out["contrast_source"] == "direct"
    assert set(out["inputs_overridden"]) == {RUN_G, PASS_G}
    assert "opp_rz_pass_funnel" in out["inputs_used"] and "opp_rz_run_funnel" in out["inputs_used"]


def test_a_single_direct_funnel_is_mirrored_onto_the_other_phase():
    out = rz.compute_red_zone_funnel({"opp_rz_pass_funnel": 0.4})
    assert out["rz_pass_funnel"] == pytest.approx(0.4, abs=1e-4)
    assert out["rz_run_funnel"] == pytest.approx(-0.4, abs=1e-4)
    out = rz.compute_red_zone_funnel({"opp_rz_run_funnel": 0.4})
    assert out["rz_run_funnel"] == pytest.approx(0.4, abs=1e-4)
    assert out["rz_pass_funnel"] == pytest.approx(-0.4, abs=1e-4)


def test_direct_funnel_is_monotone_and_amplified_by_the_offense():
    values = [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9]
    outs = [funnels({"opp_rz_pass_funnel": v})[0] for v in values]
    assert outs == sorted(outs)
    plain = funnels({"opp_rz_pass_funnel": 0.5})[0]
    hot = funnels({"opp_rz_pass_funnel": 0.5, RATE: at(RATE, 2), QB: at(QB, 2)})[0]
    assert 0.5 - 1e-3 < plain < 0.5 + 1e-3 and plain < hot < 1.0


@pytest.mark.parametrize("bad", [2.0, -1.5, 40, "n/a", None, float("nan"), float("inf"), True])
def test_out_of_range_or_unusable_direct_funnels_are_rejected_not_clamped(bad):
    out = rz.compute_red_zone_funnel({**RUN_FUNNEL_D, "opp_rz_pass_funnel": bad})
    reference = rz.compute_red_zone_funnel(RUN_FUNNEL_D)
    assert out["contrast_source"] == "grades"
    assert (out["rz_pass_funnel"], out["rz_run_funnel"]) == (reference["rz_pass_funnel"], reference["rz_run_funnel"])


def test_direct_funnel_extremes_stay_bounded():
    for v in (1.0, -1.0):
        out = rz.compute_red_zone_funnel({"opp_rz_pass_funnel": v, RATE: at(RATE, 3), QB: at(QB, 3), TE: at(TE, 3)})
        assert -1.0 <= out["rz_pass_funnel"] <= 1.0 and -1.0 <= out["rz_run_funnel"] <= 1.0
        assert math.copysign(1, out["rz_pass_funnel"]) == math.copysign(1, v)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "n/a", "", True, [], {}, -5.0, 101.0, 4000.0, "-3"])
def test_bad_or_out_of_range_grades_are_treated_as_missing(bad):
    out = rz.compute_red_zone_funnel({RUN_G: bad, PASS_G: bad, RATE: bad, QB: bad, RB: bad, TE: bad})
    assert (out["rz_pass_funnel"], out["rz_run_funnel"]) == (0.0, 0.0)
    assert out["inputs_used"] == [] and out["coverage"] == 0.0


@pytest.mark.parametrize(
    "run_text, pass_text",
    [("85%", "15%"), (" 85 % ", "15%"), ("85", "15"), (85, 15), (85.0, 15.0), ("85.0%", "15.0 %")],
)
def test_grades_written_as_percent_strings_mean_the_same_percentile(run_text, pass_text):
    # "85%" is the 85th percentile, not 0.85 (which common.to_float would otherwise produce).
    assert funnels({RUN_G: run_text, PASS_G: pass_text}) == funnels(PASS_FUNNEL_D)
    assert funnels({RUN_G: run_text, PASS_G: pass_text})[0] > 0.2


def test_fraction_scale_grades_are_not_rescaled():
    # Documented: grades are percentiles on 0-100 only. 0.85 is the 0.85th percentile, so a 0-1 pair is ~neutral
    # (it cannot be told apart from a genuine 1st-percentile pair).
    assert abs(funnels({RUN_G: 0.85, PASS_G: 0.15})[0]) < 0.01


@pytest.mark.parametrize("bad", ["-3%", "105%", "101", "n/a%", "%", "  "])
def test_out_of_range_or_malformed_percent_string_grades_are_rejected(bad):
    out = rz.compute_red_zone_funnel({RUN_G: bad, PASS_G: bad})
    assert (out["rz_pass_funnel"], out["rz_run_funnel"]) == (0.0, 0.0)
    assert out["inputs_used"] == [] and out["coverage"] == 0.0
    # ... and one rejected grade leaves the other to act alone (halved contrast), it does not poison it.
    one = rz.compute_red_zone_funnel({RUN_G: bad, PASS_G: 15})
    assert one["inputs_used"] == [PASS_G]
    assert 0 < one["rz_pass_funnel"] < funnels(PASS_FUNNEL_D)[0]


def test_one_bad_value_does_not_poison_the_good_ones():
    clean = rz.compute_red_zone_funnel({RUN_G: 85, PASS_G: 15})
    mixed = rz.compute_red_zone_funnel({RUN_G: 85, PASS_G: 15, RATE: "n/a", QB: float("nan"), TE: -0.3})
    assert (mixed["rz_pass_funnel"], mixed["rz_run_funnel"]) == (clean["rz_pass_funnel"], clean["rz_run_funnel"])
    assert set(mixed["inputs_used"]) == {RUN_G, PASS_G}


def test_percent_and_fraction_scale_rates_agree_and_percent_strings_parse():
    fraction = funnels({**PASS_FUNNEL_D, RATE: 0.64, QB: 0.30, TE: 0.25})
    percent = funnels({**PASS_FUNNEL_D, RATE: 64.0, QB: 30.0, TE: 25.0})
    strings = funnels({**PASS_FUNNEL_D, RATE: "64%", QB: "30%", TE: "25 %"})
    assert fraction == percent == strings
    assert fraction[0] > funnels(PASS_FUNNEL_D)[0]


def test_huge_but_valid_values_are_clipped_by_the_z_limit():
    extreme = funnels({**PASS_FUNNEL_D, RATE: 1.0, QB: 1.0, TE: 1.0})
    at_clip = funnels({**PASS_FUNNEL_D, RATE: at(RATE, 3), QB: at(QB, 3), TE: at(TE, 3)})
    assert extreme == at_clip  # nothing beyond +3 SD buys extra
    assert all(math.isfinite(v) and -1 <= v <= 1 for v in extreme)


def test_outputs_stay_bounded_across_a_dense_grid():
    for run_g in (0, 25, 50, 75, 100):
        for pass_g in (0, 25, 50, 75, 100):
            for sd in (-4, -1, 0, 1, 4):
                inputs = {RUN_G: run_g, PASS_G: pass_g, RATE: at(RATE, sd), QB: at(QB, sd), RB: at(RB, -sd), TE: at(TE, sd)}
                out = rz.compute_red_zone_funnel(inputs)
                assert -1.0 <= out["rz_pass_funnel"] <= 1.0
                assert -1.0 <= out["rz_run_funnel"] <= 1.0
                assert 0.0 <= out["probability"] <= 1.0


# --- compute: missing data ----------------------------------------------------------------
def test_missing_data_shrinks_toward_neutral():
    both = abs(funnels(PASS_FUNNEL_D)[0])
    one_grade = abs(funnels({RUN_G: PASS_FUNNEL_D[RUN_G]})[0])
    assert 0 < one_grade < both
    offense = {RATE: at(RATE, 2), QB: at(QB, 2), TE: at(TE, 2), RB: at(RB, 2)}
    partial = abs(funnels({**PASS_FUNNEL_D, RATE: offense[RATE]})[0])
    full = abs(funnels({**PASS_FUNNEL_D, **offense})[0])
    assert abs(funnels(PASS_FUNNEL_D)[0]) < partial < full


def test_coverage_reflects_observed_weight():
    assert rz.compute_red_zone_funnel({})["coverage"] == 0.0
    assert rz.compute_red_zone_funnel(average_inputs())["coverage"] == 1.0
    grades_only = rz.compute_red_zone_funnel(PASS_FUNNEL_D)["coverage"]
    direct_only = rz.compute_red_zone_funnel({"opp_rz_pass_funnel": 0.3})["coverage"]
    assert 0 < grades_only < 1
    assert grades_only == direct_only  # a supplied funnel fills the same slot the two grades fill
    assert rz.compute_red_zone_funnel({RUN_G: 80})["coverage"] < grades_only


def test_every_brief_named_input_is_accepted_and_changes_something():
    for key in (RUN_G, PASS_G, RATE, QB, RB, TE, "opp_rz_pass_funnel", "opp_rz_run_funnel"):
        out = rz.compute_red_zone_funnel({key: FACTOR[key].mean + 1.0 * FACTOR[key].sd if key in FACTOR else 0.5})
        assert key in out["inputs_used"], key
        assert out["coverage"] > 0, key
        assert key in out["contributions"], key
    # And the offense-only inputs are inert on their own: no contrast, no funnel.
    assert funnels({RATE: at(RATE, 2), QB: at(QB, 2), RB: at(RB, 2), TE: at(TE, 2)}) == (0.0, 0.0)


def test_drivers_explain_a_funnel_and_admit_when_there_is_no_contrast():
    funnelled = rz.compute_red_zone_funnel({**PASS_FUNNEL_D, RATE: 0.64})
    assert "pass" in funnelled["drivers"][0] and "funnels toward the pass" in funnelled["drivers"][0]
    assert any("amplifies the pass funnel" in line for line in funnelled["drivers"])
    offense_only = rz.compute_red_zone_funnel({RATE: 0.64, QB: 0.30})
    assert offense_only["contrast_source"] == "none"
    assert "No opponent red-zone defense contrast" in offense_only["drivers"][0]
    balanced = rz.compute_red_zone_funnel({RUN_G: 60, PASS_G: 60})
    assert "no funnel" in balanced["drivers"][0]


def test_contributions_label_the_phase_each_input_acts_on():
    out = rz.compute_red_zone_funnel({RUN_G: 70, PASS_G: 30, RATE: 0.62, QB: 0.3, RB: 0.2, TE: 0.2})
    c = out["contributions"]
    assert c[RUN_G]["phase"] == c[PASS_G]["phase"] == "contrast"
    assert c[RATE]["phase"] == "both" and c[RATE]["run_z"] == pytest.approx(-c[RATE]["z"])
    assert c[QB]["phase"] == c[TE]["phase"] == "pass" and c[RB]["phase"] == "run"
    assert all("value" in item and "weight" in item for item in c.values())


# --- compute: realistic scenarios ----------------------------------------------------------
def test_scenarios_rank_in_the_expected_order():
    pass_heavy_offense = {RATE: 0.62, QB: 0.29, TE: 0.22, RB: 0.13}
    run_heavy_offense = {RATE: 0.44, QB: 0.17, TE: 0.10, RB: 0.22}
    elite_vs_pass_funnel = funnels({**pass_heavy_offense, **PASS_FUNNEL_D})[0]
    average_vs_pass_funnel = funnels(PASS_FUNNEL_D)[0]
    run_team_vs_pass_funnel = funnels({**run_heavy_offense, **PASS_FUNNEL_D})[0]
    elite_vs_balanced = funnels({**pass_heavy_offense, RUN_G: 55, PASS_G: 55})[0]
    elite_vs_run_funnel = funnels({**pass_heavy_offense, **RUN_FUNNEL_D})[0]
    assert elite_vs_pass_funnel > average_vs_pass_funnel > run_team_vs_pass_funnel > elite_vs_balanced == 0.0
    assert elite_vs_balanced > elite_vs_run_funnel
    # Run phase: the same pass-heavy offense is hurt (negative) by the pass-funnel defense less than a
    # run-heavy offense that leans on the run phase that defense is stout against.
    assert funnels({**run_heavy_offense, **PASS_FUNNEL_D})[1] < funnels({**pass_heavy_offense, **PASS_FUNNEL_D})[1] < 0


def test_a_run_heavy_offense_against_a_soft_run_defense_gets_a_positive_run_funnel():
    out = rz.compute_red_zone_funnel({**RUN_FUNNEL_D, RATE: 0.45, RB: 0.21})
    assert out["rz_run_funnel"] > funnels(RUN_FUNNEL_D)[1] > 0
    assert out["rz_pass_funnel"] < 0
    assert out["probability"] < rz.BASE_RATE  # implied share of TDs through the air falls


def test_implied_pass_share_is_the_brief_formula_on_the_league_split():
    assert rz.implied_pass_td_share(0.0, 0.0) == rz.BASE_RATE
    for fp, fr in ((0.4, -0.3), (-0.5, 0.5), (1.0, -1.0), (0.2, 0.2)):
        p = rz.BASE_RATE * (1 + fp * 0.35)
        u = (1 - rz.BASE_RATE) * (1 + fr * 0.30)
        assert rz.implied_pass_td_share(fp, fr) == pytest.approx(p / (p + u))
    out = rz.compute_red_zone_funnel(PASS_FUNNEL_D)
    assert out["probability"] == pytest.approx(
        rz.implied_pass_td_share(out["rz_pass_funnel"], out["rz_run_funnel"]), abs=1e-4
    )
    assert out["probability"] > rz.BASE_RATE


# --- logging --------------------------------------------------------------------------------
def test_debug_log_is_emitted_on_the_pack_logger(caplog):
    with caplog.at_level(logging.DEBUG, logger="nfl_pack"):
        rz.compute_red_zone_funnel({**PASS_FUNNEL_D, RATE: 0.6})
    records = [rec for rec in caplog.records if rec.name == "nfl_pack.red_zone_funnel"]
    assert records and records[0].levelno == logging.DEBUG
    message = records[0].getMessage()
    assert "rz_pass_funnel=" in message and "rz_run_funnel=" in message


def test_no_log_output_when_debug_is_off(caplog):
    with caplog.at_level(logging.INFO, logger="nfl_pack"):
        rz.compute_red_zone_funnel(PASS_FUNNEL_D)
    assert not [rec for rec in caplog.records if rec.name.startswith("nfl_pack")]


# --- apply: the brief's formulas -------------------------------------------------------------
@pytest.mark.parametrize("f_pass, f_run", [(0.0, 0.0), (0.4, -0.4), (-0.7, 0.7), (1.0, -1.0), (-1.0, 1.0), (0.123, 0.456)])
def test_literal_formula_equality_with_the_brief(f_pass, f_run):
    adjusted, log = rz.apply_red_zone_funnel(PROJECTION, signed_result(f_pass, f_run), recenter=False)
    expected_qb = min(0.62 * (1 + f_pass * 0.35), 1.0)
    expected_rb = min(0.38 * (1 + f_run * 0.30), 1.0)
    expected_te = min(0.29 * (1 + f_pass * 0.25), 1.0)
    assert adjusted["qb_td_prob"] == pytest.approx(expected_qb)
    assert adjusted["rb_td_prob"] == pytest.approx(expected_rb)
    assert adjusted["te_td_prob"] == pytest.approx(expected_te)
    assert [e["field"] for e in log] == ["qb_td_prob", "rb_td_prob", "te_td_prob"]
    assert all(e["applied"] and e["stage"] == "red_zone_funnel" and e["recentered"] is False for e in log)


def test_each_field_rides_the_right_funnel_and_coefficient():
    only_pass, _ = rz.apply_red_zone_funnel(PROJECTION, signed_result(0.5, 0.0))
    assert only_pass["qb_td_prob"] == pytest.approx(0.62 * (1 + 0.5 * 0.35))
    assert only_pass["te_td_prob"] == pytest.approx(0.29 * (1 + 0.5 * 0.25))
    assert only_pass["rb_td_prob"] == 0.38
    only_run, _ = rz.apply_red_zone_funnel(PROJECTION, signed_result(0.0, 0.5))
    assert only_run["rb_td_prob"] == pytest.approx(0.38 * (1 + 0.5 * 0.30))
    assert only_run["qb_td_prob"] == 0.62 and only_run["te_td_prob"] == 0.29


def test_recenter_is_a_no_op_for_signed_factors_with_a_zero_reference():
    for f_pass, f_run in ((0.6, -0.2), (-0.9, 0.9), (0.0, 0.0)):
        on, _ = rz.apply_red_zone_funnel(PROJECTION, signed_result(f_pass, f_run), recenter=True)
        off, _ = rz.apply_red_zone_funnel(PROJECTION, signed_result(f_pass, f_run), recenter=False)
        assert on == off
    # ... and the default is recenter=True, giving the same numbers.
    default, _ = rz.apply_red_zone_funnel(PROJECTION, signed_result(0.6, -0.2))
    literal, _ = rz.apply_red_zone_funnel(PROJECTION, signed_result(0.6, -0.2), recenter=False)
    assert default == literal


def test_neutral_result_leaves_the_projection_untouched():
    neutral = rz.compute_red_zone_funnel({})
    adjusted, log = rz.apply_red_zone_funnel(PROJECTION, neutral)
    assert adjusted == PROJECTION
    assert all(e["applied"] and e["factor"] == 1.0 for e in log)


def test_apply_returns_a_new_dict_and_never_mutates_the_input():
    original = copy.deepcopy(PROJECTION)
    adjusted, _ = rz.apply_red_zone_funnel(PROJECTION, signed_result(0.8, -0.8))
    assert PROJECTION == original
    assert adjusted is not PROJECTION
    adjusted["qb_td_prob"] = -1
    assert PROJECTION == original


def test_only_the_three_td_fields_move_and_everything_else_is_copied_through():
    adjusted, _ = rz.apply_red_zone_funnel(PROJECTION, signed_result(0.9, 0.9))
    for key, value in PROJECTION.items():
        if key not in ("qb_td_prob", "rb_td_prob", "te_td_prob"):
            assert adjusted[key] == value
    assert set(adjusted) == set(PROJECTION)


def test_missing_or_non_numeric_fields_are_recorded_as_not_applied():
    projection = {"qb_td_prob": 0.5, "te_td_prob": "n/a", "qb_yards": 250.0}
    adjusted, log = rz.apply_red_zone_funnel(projection, signed_result(0.5, 0.5))
    by_field = {e["field"]: e for e in log}
    assert by_field["qb_td_prob"]["applied"] is True
    assert by_field["rb_td_prob"]["applied"] is False and "missing" in by_field["rb_td_prob"]["reason"]
    assert by_field["te_td_prob"]["applied"] is False
    assert adjusted["te_td_prob"] == "n/a" and "rb_td_prob" not in adjusted
    assert adjusted["qb_td_prob"] == pytest.approx(0.5 * (1 + 0.5 * 0.35))


def test_audit_entries_say_which_funnel_moved_which_field():
    _, log = rz.apply_red_zone_funnel(PROJECTION, signed_result(0.5, -0.25))
    by_field = {e["field"]: e for e in log}
    assert by_field["qb_td_prob"]["funnel"] == by_field["te_td_prob"]["funnel"] == "rz_pass_funnel"
    assert by_field["rb_td_prob"]["funnel"] == "rz_run_funnel"
    assert by_field["qb_td_prob"]["coefficient"] == 0.35 and by_field["rb_td_prob"]["coefficient"] == 0.30
    assert by_field["te_td_prob"]["coefficient"] == 0.25
    assert by_field["rb_td_prob"]["factor"] == pytest.approx(1 - 0.25 * 0.30, abs=1e-4)
    assert by_field["qb_td_prob"]["before"] == 0.62 and by_field["qb_td_prob"]["direction"] == "boost"


@pytest.mark.parametrize("bad_result", [{}, None, {"rz_pass_funnel": "n/a", "rz_run_funnel": None}, "nope", [1]])
def test_unusable_result_moves_nothing_and_says_why(bad_result):
    adjusted, log = rz.apply_red_zone_funnel(PROJECTION, bad_result)
    assert adjusted == PROJECTION
    assert len(log) == 3 and all(not e["applied"] and "rz_" in e["reason"] for e in log)


def test_out_of_range_funnel_in_a_result_is_clamped_to_the_signed_bounds():
    adjusted, _ = rz.apply_red_zone_funnel({"rb_td_prob": 0.3}, signed_result(0.0, 9.0), recenter=False)
    assert adjusted["rb_td_prob"] == pytest.approx(0.3 * 1.30)
    adjusted, _ = rz.apply_red_zone_funnel({"rb_td_prob": 0.3}, signed_result(0.0, -9.0), recenter=False)
    assert adjusted["rb_td_prob"] == pytest.approx(0.3 * 0.70)


def test_probability_fields_are_capped_at_one_on_the_fraction_scale():
    adjusted, log = rz.apply_red_zone_funnel({"qb_td_prob": 0.95, "te_td_prob": 0.9}, signed_result(1.0, 0.0))
    assert adjusted["qb_td_prob"] == 1.0 and adjusted["te_td_prob"] == 1.0
    assert log[0]["after"] == 1.0


def test_probability_fields_are_capped_at_one_hundred_on_the_percent_scale():
    adjusted, _ = rz.apply_red_zone_funnel({"qb_td_prob": 80.0, "rb_td_prob": 70.0, "te_td_prob": 99.0}, signed_result(1.0, 1.0))
    assert adjusted["qb_td_prob"] == 100.0  # 80 * 1.35 = 108
    assert adjusted["rb_td_prob"] == pytest.approx(70.0 * 1.30)  # 91, under the cap
    assert adjusted["te_td_prob"] == 100.0
    # A percent-scale value is not squashed down to 1.0 by mistake.
    mild, _ = rz.apply_red_zone_funnel({"qb_td_prob": 60.0}, signed_result(0.1, 0.0))
    assert mild["qb_td_prob"] == pytest.approx(60.0 * 1.035)


def test_a_negative_funnel_lowers_and_a_positive_one_raises():
    down, _ = rz.apply_red_zone_funnel(PROJECTION, signed_result(-0.6, -0.6))
    up, _ = rz.apply_red_zone_funnel(PROJECTION, signed_result(0.6, 0.6))
    for field in ("qb_td_prob", "rb_td_prob", "te_td_prob"):
        assert down[field] < PROJECTION[field] < up[field]
        assert down[field] > 0


def test_compute_then_apply_end_to_end():
    result = rz.compute_red_zone_funnel({**PASS_FUNNEL_D, RATE: 0.6, QB: 0.28})
    adjusted, log = rz.apply_red_zone_funnel(PROJECTION, result)
    assert adjusted["qb_td_prob"] > PROJECTION["qb_td_prob"]
    assert adjusted["te_td_prob"] > PROJECTION["te_td_prob"]
    assert adjusted["rb_td_prob"] < PROJECTION["rb_td_prob"]
    # The QB moves more than the TE for the same pass funnel (0.35 vs 0.25 coefficient).
    qb_gain = adjusted["qb_td_prob"] / PROJECTION["qb_td_prob"] - 1
    te_gain = adjusted["te_td_prob"] / PROJECTION["te_td_prob"] - 1
    assert qb_gain == pytest.approx(result["rz_pass_funnel"] * 0.35, abs=1e-6)
    assert te_gain == pytest.approx(result["rz_pass_funnel"] * 0.25, abs=1e-6)
    assert qb_gain > te_gain > 0


# --- module hygiene -----------------------------------------------------------------------
def test_module_imports_only_stdlib_and_the_pack_common():
    source = (Path(rz.__file__)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0] if node.level == 0 else "." + node.module)
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4", "pandas", "numpy", "nflreadpy", "polars"}
    assert not (imported & banned)
    allowed = {"__future__", "dataclasses", "logging", "math", "typing", "modules"}
    assert imported <= allowed, imported - allowed
    # ``from modules import nfl_pack_common`` is the only project import.
    project_imports = [
        (node.module, [a.name for a in node.names])
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "modules"
    ]
    assert all("nfl_pack_common" in (module + "." + ".".join(names)) for module, names in project_imports)


def test_an_expected_count_td_rate_is_boosted_not_cut_by_the_probability_cap():
    """Regression: the pipeline passes per-game TD RATES (expected counts such as 1.3), not only probabilities.

    A flat 1.0 ceiling for every value <= 1.5 turned a funnel BOOST of a 1.3 rate into a 23% cut.
    """
    result = {"rz_pass_funnel": 0.5, "rz_run_funnel": 0.5, "references": {"rz_pass_funnel": 0.0, "rz_run_funnel": 0.0}}
    out, _ = rz.apply_red_zone_funnel({"qb_td_prob": 1.3, "te_td_prob": 1.2}, result)
    assert out["qb_td_prob"] == pytest.approx(1.3 * (1 + 0.5 * 0.35))
    assert out["te_td_prob"] == pytest.approx(1.2 * (1 + 0.5 * 0.25))
    # a genuine probability is still held at 1.0, a percent-scale value at 100
    boosted, _ = rz.apply_red_zone_funnel({"rb_td_prob": 0.9}, {"rz_run_funnel": 1.0, "references": {"rz_run_funnel": 0.0}})
    assert boosted["rb_td_prob"] == 1.0
    pct, _ = rz.apply_red_zone_funnel({"qb_td_prob": 95.0}, {"rz_pass_funnel": 1.0, "references": {"rz_pass_funnel": 0.0}})
    assert pct["qb_td_prob"] == 100.0
