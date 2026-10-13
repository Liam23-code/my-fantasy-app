"""Tests for the game-script model of the NFL model upgrade pack (offline, pure).

The model maps a team's pre-game spread (and total) to the share of its snaps that
will be played pass-heavy (trailing by 7+), neutral or run-heavy (leading by 7+),
then applies the brief's two attempt formulas.  Every expected number below is
either the brief's formula written out by hand or a value measured on nflverse
2018-2025 (quoted in the module docstring), never read back from the module under
test.
"""
from __future__ import annotations

import ast
import copy
import logging
import math
from pathlib import Path
from types import MappingProxyType

import pytest

from modules import game_script_model as gs
from modules import nfl_pack_common as c

BRIEF_INPUTS = (
    "vegas_spread",
    "vegas_total",
    "team_epa_per_play",
    "opp_epa_per_play",
    "pace_seconds_per_play",
    "proe",
    "rroe",
)
CONTRACT_KEYS = (
    "model", "positive_script_prob", "negative_script_prob", "neutral_script_prob", "trailing_script_prob",
    "leading_script_prob", "probability", "reference_probability", "references", "z_score", "coverage",
    "inputs_used", "inputs_defaulted", "contributions", "drivers", "basis",
)
FACTOR_BY_KEY = {f.key: f for f in gs.FACTORS}
PROJECTION = {"qb_attempts": 36.0, "rb_attempts": 18.5}


def brief_qb(pos: float, neg: float) -> float:
    """The brief's multiplier, written out: qb_attempts *= (1 + pos*0.20 - neg*0.25)."""
    return 1.0 + pos * 0.20 - neg * 0.25


def brief_rb(pos: float, neg: float) -> float:
    """The brief's multiplier, written out: rb_attempts *= (1 + neg*0.30 - pos*0.20)."""
    return 1.0 + neg * 0.30 - pos * 0.20


def neutral() -> dict:
    return {f.key: f.mean for f in gs.FACTORS}


def shares(inputs) -> tuple[float, float, float]:
    out = gs.compute_game_script(inputs)
    return out["positive_script_prob"], out["neutral_script_prob"], out["negative_script_prob"]


def fake_result(pos: float, neg: float, refs: tuple[float, float] | None = None) -> dict:
    out = {"positive_script_prob": pos, "negative_script_prob": neg, "neutral_script_prob": 1.0 - pos - neg}
    if refs is not None:
        out["references"] = {"positive_script_prob": refs[0], "negative_script_prob": refs[1]}
    return out


# --- constants and structure --------------------------------------------------
def test_constants_are_the_briefs_coefficients_and_a_measured_base_rate():
    assert gs.QB_ATTEMPTS_POSITIVE_COEFFICIENT == 0.20
    assert gs.QB_ATTEMPTS_NEGATIVE_COEFFICIENT == 0.25
    assert gs.RB_ATTEMPTS_NEGATIVE_COEFFICIENT == 0.30
    assert gs.RB_ATTEMPTS_POSITIVE_COEFFICIENT == 0.20
    assert gs.SCRIPT_THRESHOLD_POINTS == 7
    assert gs.POSITIVE_SCRIPT_IS_PASS_HEAVY is True
    assert gs.GAIN > 0
    assert 0.28 <= gs.BASE_RATE <= 0.31  # trailing share at a pick'em, measured 0.2968
    assert "0.2968" in gs.EVENT_DEFINITION and "264,889" in gs.EVENT_DEFINITION
    assert gs.OUTPUT_KEYS == ("positive_script_prob", "negative_script_prob", "neutral_script_prob")
    assert gs.LOG.name == "nfl_pack.game_script"


def test_factors_cover_every_brief_input_with_a_real_reference_and_causal_weight_order():
    assert {f.key for f in gs.FACTORS} == set(BRIEF_INPUTS) and len(gs.FACTORS) == len(BRIEF_INPUTS)
    weights = [f.weight for f in gs.FACTORS]
    assert weights == sorted(weights, reverse=True)  # ordered by causal importance, spread first
    assert gs.FACTORS[0].key == "vegas_spread" and weights[0] > 3 * weights[1]
    for factor in gs.FACTORS:
        assert isinstance(factor, c.Factor)
        assert factor.sd > 0 and factor.weight > 0 and factor.direction in (-1, 1)
        assert factor.source != "prior" and "nflverse" in factor.source  # every reference is measured
    # the spread is the one factor whose league mean is exactly zero (each game appears once per side)
    assert FACTOR_BY_KEY["vegas_spread"].mean == 0.0
    # directions are stated relative to the pass-heavy share: better offense / faster pace / pass lean => ...
    assert FACTOR_BY_KEY["vegas_spread"].direction == +1  # underdog trails more
    assert FACTOR_BY_KEY["team_epa_per_play"].direction == -1  # better own offense trails less
    assert FACTOR_BY_KEY["opp_epa_per_play"].direction == +1
    assert FACTOR_BY_KEY["proe"].direction == +1 and FACTOR_BY_KEY["rroe"].direction == -1  # exact mirrors


def test_result_contains_every_contract_key_and_a_priors_not_fitted_basis():
    result = gs.compute_game_script({"vegas_spread": -3.5, "vegas_total": 47.5})
    for key in CONTRACT_KEYS:
        assert key in result, key
    assert result["model"] == "game_script"
    assert result["probability"] == result["positive_script_prob"]
    assert result["reference_probability"] == result["references"]["positive_script_prob"]
    assert set(result["references"]) == set(gs.OUTPUT_KEYS)
    basis = result["basis"].lower()
    assert "prior" in basis and "not fitted" in basis and "2018-2025" in basis
    assert result["drivers"] and "Spread" in result["drivers"][0]
    assert set(result["inputs_used"]) | set(result["inputs_defaulted"]) == set(BRIEF_INPUTS)
    assert not set(result["inputs_used"]) & set(result["inputs_defaulted"])


def test_module_is_pure_offline_and_imports_only_stdlib_and_common():
    tree = ast.parse(Path(gs.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= {"__future__", "logging", "math", "typing", "modules.nfl_pack_common"}, imported
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4", "pandas", "nflreadpy"}
    assert not {name.split(".")[0] for name in imported} & banned


# --- neutral in -> reference out ------------------------------------------------
def test_neutral_in_reference_out_for_every_way_of_saying_nothing():
    reference = gs.compute_game_script({})
    assert reference["positive_script_prob"] == reference["references"]["positive_script_prob"] == gs.BASE_RATE
    assert reference["negative_script_prob"] == reference["references"]["negative_script_prob"]
    assert reference["neutral_script_prob"] == reference["references"]["neutral_script_prob"]
    assert reference["z_score"] == 0.0 and reference["coverage"] == 0.0 and reference["margin_source"] == "none"
    for blank in (None, "not a mapping", 42, [], [("vegas_spread", -10)]):
        assert gs.compute_game_script(blank)["positive_script_prob"] == gs.BASE_RATE
    average = gs.compute_game_script(neutral())  # every input exactly at its league mean
    for key in gs.OUTPUT_KEYS:
        assert average[key] == reference[key] == reference["references"][key]
    assert average["coverage"] == 1.0 and average["inputs_defaulted"] == []


def test_reference_matches_the_measured_pickem_shares():
    # nflverse 2018-2025, team-games with spread in (-1.5, +1.5]: trailing 0.302 / neutral 0.516 / leading 0.182
    pos, neu, neg = shares({})
    assert pos == pytest.approx(0.302, abs=0.012)
    assert neu == pytest.approx(0.516, abs=0.012)
    assert neg == pytest.approx(0.182, abs=0.012)
    # the zero-point spread is genuinely NOT symmetric (the trailing team snaps the ball more often);
    # the equal-share point is a favourite of about three points
    assert pos > neg + 0.08
    assert shares({"vegas_spread": -2.0})[0] > shares({"vegas_spread": -2.0})[2]
    assert shares({"vegas_spread": -5.0})[2] > shares({"vegas_spread": -5.0})[0]


# --- semantics: positive = pass-heavy = trailing --------------------------------
def test_sign_convention_underdogs_trail_favourites_lead():
    dog_pos, _, dog_neg = shares({"vegas_spread": +10})
    fav_pos, _, fav_neg = shares({"vegas_spread": -10})
    assert dog_pos > dog_neg and dog_pos == pytest.approx(0.510, abs=0.01)  # measured-fit anchors
    assert fav_neg > fav_pos and fav_neg == pytest.approx(0.363, abs=0.01)
    assert dog_pos > gs.BASE_RATE > fav_pos
    assert fav_neg > gs.reference_shares()["negative_script_prob"] > dog_neg


def test_spread_sign_matches_nflverse_home_positive_convention():
    # nflverse spread_line is POSITIVE when the home team is favoured: the home team's sportsbook spread is -line
    assert gs.team_spread_from_schedule(3.5, is_home=True) == -3.5
    assert gs.team_spread_from_schedule(3.5, is_home=False) == 3.5
    assert gs.team_spread_from_schedule("-7", is_home=True) == 7.0
    assert gs.team_spread_from_schedule(None, is_home=True) is None
    assert gs.team_spread_from_schedule("n/a", is_home=False) is None
    home = shares({"vegas_spread": gs.team_spread_from_schedule(6.5, True)})   # home favourite
    away = shares({"vegas_spread": gs.team_spread_from_schedule(6.5, False)})  # the road dog
    assert home[2] > home[0] and away[0] > away[2]
    assert home[0] < away[0] and home[2] > away[2]


def test_expected_margin_is_the_negated_spread_and_the_spread_wins_when_both_given():
    assert shares({"expected_margin": 7.0}) == shares({"vegas_spread": -7.0})
    both = gs.compute_game_script({"vegas_spread": +3.0, "expected_margin": 20.0})
    assert both["margin_source"] == "vegas_spread"
    assert both["positive_script_prob"] == gs.compute_game_script({"vegas_spread": +3.0})["positive_script_prob"]
    via = gs.compute_game_script({"expected_margin": 7.0})
    assert via["margin_source"] == "expected_margin" and via["contributions"]["vegas_spread"]["via"] == "expected_margin"
    # an unusable spread falls through to a usable expected_margin
    assert shares({"vegas_spread": "n/a", "expected_margin": 7.0}) == shares({"vegas_spread": -7.0})


def test_aliases_trailing_and_leading_always_name_the_physical_states():
    for spread in (-9.0, 0.0, 6.0):
        out = gs.compute_game_script({"vegas_spread": spread})
        assert out["trailing_script_prob"] == out["positive_script_prob"]
        assert out["leading_script_prob"] == out["negative_script_prob"]


# --- monotonicity ----------------------------------------------------------------
def test_spread_moves_trailing_up_leading_down_everywhere():
    grid = [x * 0.5 for x in range(-44, 45)]
    out = [shares({"vegas_spread": s}) for s in grid]
    for (p0, _, n0), (p1, _, n1) in zip(out, out[1:]):
        assert p1 >= p0 and n1 <= n0
    assert out[-1][0] > out[0][0] + 0.5 and out[0][2] > out[-1][2] + 0.3  # and by a lot end to end


def test_a_higher_total_never_lowers_either_tail_at_any_spread():
    # the contract's monotone property; an unguarded volatility scale breaks it for 10+ point underdogs
    totals = [30.0, 36.0, 41.0, 45.28, 50.0, 55.0, 60.0, 70.0]
    for spread in [x * 0.5 for x in range(-44, 45)]:
        rows = [shares({"vegas_spread": spread, "vegas_total": t}) for t in totals]
        for (p0, n0, l0), (p1, n1, l1) in zip(rows, rows[1:]):
            assert p1 >= p0 - 1e-12, (spread, p0, p1)
            assert l1 >= l0 - 1e-12, (spread, l0, l1)
            assert n1 <= n0 + 1e-12, (spread, n0, n1)


def test_total_widens_both_tails_around_a_pickem():
    low = shares({"vegas_total": 36})
    high = shares({"vegas_total": 55})
    assert high[0] > low[0] and high[2] > low[2] and high[1] < low[1]
    # measured-fit anchors (module docstring): neutral share 0.551 at 36, 0.481 at 55
    assert low[1] == pytest.approx(0.551, abs=0.01) and high[1] == pytest.approx(0.481, abs=0.01)


def test_each_factor_moved_one_sd_in_its_direction_never_lowers_the_pass_heavy_share():
    base = gs.compute_game_script(neutral())["positive_script_prob"]
    for factor in gs.FACTORS:
        bumped = dict(neutral(), **{factor.key: factor.mean + factor.direction * factor.sd})
        moved = gs.compute_game_script(bumped)["positive_script_prob"]
        assert moved >= base, factor.key
        opposite = dict(neutral(), **{factor.key: factor.mean - factor.direction * factor.sd})
        assert gs.compute_game_script(opposite)["positive_script_prob"] <= base, factor.key


def test_the_spread_and_total_actually_do_something_per_factor():
    base = gs.compute_game_script(neutral())["positive_script_prob"]
    for key in ("vegas_spread", "vegas_total"):
        f = FACTOR_BY_KEY[key]
        assert gs.compute_game_script(dict(neutral(), **{key: f.mean + f.sd}))["positive_script_prob"] > base + 0.005


# --- bounds ----------------------------------------------------------------------
def test_probabilities_are_bounded_and_sum_to_one_over_a_wide_grid():
    for spread in [x * 1.5 for x in range(-20, 21)]:
        for total in (28.0, 38.0, 45.28, 52.0, 65.0):
            out = gs.compute_game_script({"vegas_spread": spread, "vegas_total": total})
            parts = (out["positive_script_prob"], out["negative_script_prob"], out["neutral_script_prob"])
            assert all(0.0 <= p <= 1.0 for p in parts)
            assert sum(parts) == pytest.approx(1.0, abs=1e-6)
            assert out["trailing_script_prob"] + out["leading_script_prob"] + out["neutral_script_prob"] == pytest.approx(1.0, abs=1e-6)


def test_gain_reasoning_plus_one_and_two_sd_stay_plausible_and_well_below_the_ceiling():
    f = FACTOR_BY_KEY
    plus1 = {k: f[k].mean + f[k].direction * f[k].sd for k in BRIEF_INPUTS}
    plus2 = {k: f[k].mean + f[k].direction * 2 * f[k].sd for k in BRIEF_INPUTS}
    plus3 = {k: f[k].mean + f[k].direction * 3 * f[k].sd for k in BRIEF_INPUTS}
    p1 = gs.compute_game_script(plus1)["positive_script_prob"]
    p2 = gs.compute_game_script(plus2)["positive_script_prob"]
    p3 = gs.compute_game_script(plus3)["positive_script_prob"]
    assert 0.38 < p1 < 0.50          # a 6.4-point underdog trails ~43% of snaps (module docstring)
    assert p1 < p2 < p3 < 0.80       # monotone and nowhere near a certainty
    assert p2 < 0.70 and p3 < 0.95


def test_huge_and_infinite_values_are_clipped_not_exploded():
    ceiling = gs.compute_game_script({"vegas_spread": gs.SPREAD_CLIP})["positive_script_prob"]
    assert gs.SPREAD_CLIP == pytest.approx(3 * FACTOR_BY_KEY["vegas_spread"].sd)
    assert gs.compute_game_script({"vegas_spread": 1e9})["positive_script_prob"] == ceiling
    assert gs.compute_game_script({"vegas_spread": 75.0})["positive_script_prob"] == ceiling
    assert ceiling < 0.80
    floor = gs.compute_game_script({"vegas_spread": -1e9})
    assert floor["negative_script_prob"] == gs.compute_game_script({"vegas_spread": -gs.SPREAD_CLIP})["negative_script_prob"]
    assert floor["positive_script_prob"] > 0.0  # even a 19-point favourite spends some snaps trailing
    assert gs.compute_game_script({"vegas_spread": "1e999"})["margin_source"] == "none"  # inf -> missing
    big_total = gs.compute_game_script({"vegas_spread": 3.0, "vegas_total": 1e6})
    capped_total = gs.compute_game_script({"vegas_spread": 3.0, "vegas_total": 45.28 + 4 * 4.51})  # past the 3 SD clip
    assert big_total["positive_script_prob"] == capped_total["positive_script_prob"]


# --- bad data --------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf"), "n/a", "", True, False, [], {}, object()])
def test_unusable_values_are_missing_not_errors(bad):
    inputs = {key: bad for key in BRIEF_INPUTS}
    inputs["expected_margin"] = bad
    out = gs.compute_game_script(inputs)
    assert out["positive_script_prob"] == gs.BASE_RATE
    assert out["inputs_used"] == [] and out["coverage"] == 0.0


def test_numeric_strings_and_numpy_scalars_are_read():
    assert shares({"vegas_spread": "-7.5"}) == shares({"vegas_spread": -7.5})
    assert shares({"vegas_spread": "+3"}) == shares({"vegas_spread": 3})
    np = pytest.importorskip("numpy")
    assert shares({"vegas_spread": np.float64(-7.5), "vegas_total": np.int64(48)}) == shares(
        {"vegas_spread": -7.5, "vegas_total": 48}
    )


def test_totals_and_pace_must_be_positive_to_count():
    assert gs.compute_game_script({"vegas_total": -10})["inputs_used"] == []
    assert gs.compute_game_script({"vegas_total": 0})["inputs_used"] == []
    assert gs.compute_game_script({"pace_seconds_per_play": -5})["inputs_used"] == []
    assert gs.compute_game_script({"pace_seconds_per_play": 28.0})["inputs_used"] == ["pace_seconds_per_play"]
    # a negative spread is meaningful (favourite), not bad data
    assert gs.compute_game_script({"vegas_spread": -3.0})["inputs_used"] == ["vegas_spread"]


def test_proe_and_rroe_read_fractions_and_percentage_points_alike():
    as_fraction = gs.compute_game_script({"proe": 0.05, "rroe": -0.05})
    as_percent_points = gs.compute_game_script({"proe": 5.0, "rroe": -5.0})
    as_text = gs.compute_game_script({"proe": "5%", "rroe": "-5%"})
    for out in (as_fraction, as_percent_points, as_text):
        assert out["inputs_used"] == ["proe", "rroe"]
        assert out["contributions"]["proe"]["value"] == pytest.approx(0.05)
        assert out["contributions"]["rroe"]["value"] == pytest.approx(-0.05)
    assert gs.compute_game_script({"proe": 500.0})["inputs_used"] == []  # nonsense, not a rate
    assert gs.compute_game_script({"proe": -0.03})["contributions"]["proe"]["value"] == pytest.approx(-0.03)


def test_every_brief_named_input_is_accepted_and_reported():
    sample = {
        "vegas_spread": -4.5, "vegas_total": 49.5, "team_epa_per_play": 0.08, "opp_epa_per_play": -0.04,
        "pace_seconds_per_play": 30.5, "proe": 0.03, "rroe": -0.02,
    }
    out = gs.compute_game_script(sample)
    assert out["inputs_used"] == [f.key for f in gs.FACTORS]
    assert out["inputs_defaulted"] == [] and out["coverage"] == 1.0
    assert set(out["contributions"]) == set(BRIEF_INPUTS)
    for key in BRIEF_INPUTS:
        assert out["contributions"][key]["value"] == pytest.approx(sample[key])
        assert out["contributions"][key]["via"] == key
    for key in BRIEF_INPUTS:
        single = gs.compute_game_script({key: sample[key]})
        assert single["inputs_used"] == [key]
    # documented aliases map onto the canonical factor
    aliased = gs.compute_game_script(
        {"total_line": 49.5, "team_epa": 0.08, "opp_epa": -0.04, "pace": 30.5,
         "pass_rate_over_expected": 0.03, "run_rate_over_expected": -0.02}
    )
    assert set(aliased["inputs_used"]) == set(BRIEF_INPUTS) - {"vegas_spread"}


# --- honesty about the adjustments ----------------------------------------------
def test_with_a_spread_in_hand_epa_pace_proe_rroe_carry_no_weight_and_say_so():
    assert gs.ADJUSTMENT_POINTS_PER_Z == 0.0  # no out-of-sample lift -> inert (module docstring)
    base = {"vegas_spread": 4.0, "vegas_total": 47.0}
    extreme = dict(base, team_epa_per_play=0.25, opp_epa_per_play=-0.25, pace_seconds_per_play=27.0, proe=0.10, rroe=-0.10)
    a = gs.compute_game_script(base)
    b = gs.compute_game_script(extreme)
    for key in gs.OUTPUT_KEYS:
        assert a[key] == b[key]
    assert set(b["inputs_inert"]) == {"team_epa_per_play", "opp_epa_per_play", "pace_seconds_per_play", "proe", "rroe"}
    for key in b["inputs_inert"]:
        assert b["contributions"][key]["role"] == "inert" and b["contributions"][key]["effect"] == 0.0
    assert any("No effect" in line for line in b["drivers"])


def test_adjustment_machinery_is_monotone_bounded_and_off_by_one_constant(monkeypatch):
    monkeypatch.setattr(gs, "ADJUSTMENT_POINTS_PER_Z", 1.5)
    base = {"vegas_spread": 0.0}
    ref = gs.compute_game_script(base)["positive_script_prob"]
    assert gs.compute_game_script(neutral())["positive_script_prob"] == ref  # all-average stays at the reference
    for factor in gs.ADJUSTMENT_FACTORS:
        up = gs.compute_game_script(dict(base, **{factor.key: factor.mean + factor.direction * factor.sd}))
        down = gs.compute_game_script(dict(base, **{factor.key: factor.mean - factor.direction * factor.sd}))
        assert up["positive_script_prob"] > ref > down["positive_script_prob"], factor.key
        assert up["contributions"][factor.key]["role"] == "adjustment"
        assert up["contributions"][factor.key]["effect"] > 0
    # bounded: every adjustment input at +3 SD still moves the effective spread by at most the cap
    extreme = {f.key: f.mean + f.direction * 3 * f.sd for f in gs.ADJUSTMENT_FACTORS}
    out = gs.compute_game_script(dict(base, **extreme))
    assert abs(out["expected_margin_points"]) <= gs.ADJUSTMENT_MAX_POINTS + 1e-9
    assert out["expected_margin_points"] == pytest.approx(-gs.ADJUSTMENT_MAX_POINTS, abs=1e-3)
    capped = gs.compute_game_script({"vegas_spread": gs.ADJUSTMENT_MAX_POINTS})["positive_script_prob"]
    assert out["positive_script_prob"] == pytest.approx(capped, abs=1e-4)


def test_missing_spread_epa_stand_in_and_missing_data_shrinks_toward_neutral():
    ref = gs.BASE_RATE
    team_only = gs.compute_game_script({"team_epa_per_play": 0.10})
    both = gs.compute_game_script({"team_epa_per_play": 0.10, "opp_epa_per_play": -0.10})
    assert team_only["margin_source"] == both["margin_source"] == "epa_proxy"
    # a stronger offense is expected to lead more and trail less than the league-average
    assert both["positive_script_prob"] < team_only["positive_script_prob"] < ref
    assert both["negative_script_prob"] > team_only["negative_script_prob"] > gs.reference_shares()["negative_script_prob"]
    assert both["coverage"] > team_only["coverage"] > 0.0
    assert both["expected_margin_points"] > team_only["expected_margin_points"] > 0.0
    assert any("No spread supplied" in line for line in both["drivers"])
    # proxy is slope * differential (39.8 pts per 1.0 EPA/play): 0.20 differential -> about +8 points
    assert both["expected_margin_points"] == pytest.approx(gs.MARGIN_PER_EPA_DIFF * 0.20, abs=0.2)
    # a real spread always wins over the proxy and the EPA inputs are then not double counted
    real = gs.compute_game_script({"vegas_spread": 7.0, "team_epa_per_play": 0.20, "opp_epa_per_play": -0.20})
    assert real["margin_source"] == "vegas_spread"
    assert real["positive_script_prob"] == gs.compute_game_script({"vegas_spread": 7.0})["positive_script_prob"]


def test_proxy_epa_is_weaker_than_a_real_line_for_the_same_expected_margin():
    # an 8-point favourite by proxy still reads as a favourite, but the result says where the number came from
    proxy = gs.compute_game_script({"team_epa_per_play": 0.10, "opp_epa_per_play": -0.10})
    line = gs.compute_game_script({"vegas_spread": -8.0})
    assert proxy["negative_script_prob"] > proxy["positive_script_prob"]
    assert proxy["margin_source"] == "epa_proxy" and line["margin_source"] == "vegas_spread"
    assert proxy["negative_script_prob"] == pytest.approx(line["negative_script_prob"], abs=0.02)


# --- determinism, purity, logging ----------------------------------------------
def test_deterministic_and_inputs_not_mutated():
    inputs = {"vegas_spread": -6.5, "vegas_total": 51.0, "proe": 0.02, "junk": "x"}
    snapshot = copy.deepcopy(inputs)
    first = gs.compute_game_script(inputs)
    second = gs.compute_game_script(inputs)
    assert first == second
    assert inputs == snapshot
    frozen = MappingProxyType({"vegas_spread": -6.5, "vegas_total": 51.0})  # any read-only Mapping works
    assert gs.compute_game_script(frozen)["positive_script_prob"] == gs.compute_game_script(
        {"vegas_spread": -6.5, "vegas_total": 51.0}
    )["positive_script_prob"]


def test_debug_log_emitted_on_the_pack_logger(caplog):
    with caplog.at_level(logging.DEBUG, logger="nfl_pack"):
        out = gs.compute_game_script({"vegas_spread": -7.0, "vegas_total": 47.0})
    messages = [rec.getMessage() for rec in caplog.records if rec.name == "nfl_pack.game_script"]
    assert len(messages) == 1
    assert f"positive={out['positive_script_prob']:.4f}" in messages[0]
    assert f"negative={out['negative_script_prob']:.4f}" in messages[0]
    assert f"neutral={out['neutral_script_prob']:.4f}" in messages[0]
    assert "margin_source=vegas_spread" in messages[0]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="nfl_pack"):
        gs.compute_game_script({"vegas_spread": -7.0})
    assert caplog.records == []


# --- the semantics switch ---------------------------------------------------------
def test_flipping_the_switch_swaps_positive_and_negative_but_never_the_aliases(monkeypatch):
    default = gs.compute_game_script({"vegas_spread": 8.0, "vegas_total": 50.0})
    monkeypatch.setattr(gs, "POSITIVE_SCRIPT_IS_PASS_HEAVY", False)
    flipped = gs.compute_game_script({"vegas_spread": 8.0, "vegas_total": 50.0})
    assert flipped["positive_script_prob"] == default["negative_script_prob"]
    assert flipped["negative_script_prob"] == default["positive_script_prob"]
    assert flipped["neutral_script_prob"] == default["neutral_script_prob"]
    assert flipped["trailing_script_prob"] == default["trailing_script_prob"]
    assert flipped["leading_script_prob"] == default["leading_script_prob"]
    assert flipped["references"]["positive_script_prob"] == default["references"]["negative_script_prob"]
    assert flipped["positive_script_is_pass_heavy"] is False and default["positive_script_is_pass_heavy"] is True
    assert flipped["z_score"] == -default["z_score"]
    assert "flipped" in flipped["basis"]
    # neutral-in still gives reference-out under the flip
    assert gs.compute_game_script({})["positive_script_prob"] == gs.reference_shares()["positive_script_prob"]
    assert gs.compute_game_script(neutral())["negative_script_prob"] == gs.reference_shares()["negative_script_prob"]


def test_flip_makes_the_briefs_formulas_ride_the_opposite_physical_state(monkeypatch):
    dog = {"vegas_spread": 10.0}
    qb_default, _ = gs.apply_game_script(PROJECTION, gs.compute_game_script(dog))
    monkeypatch.setattr(gs, "POSITIVE_SCRIPT_IS_PASS_HEAVY", False)
    qb_flipped, _ = gs.apply_game_script(PROJECTION, gs.compute_game_script(dog))
    # default: a 10-point dog is pass-heavy -> more QB attempts, fewer RB attempts; flipped: the reverse
    assert qb_default["qb_attempts"] > PROJECTION["qb_attempts"] > qb_flipped["qb_attempts"]
    assert qb_default["rb_attempts"] < PROJECTION["rb_attempts"] < qb_flipped["rb_attempts"]
    # a result carrying only the unambiguous aliases is mapped through the switch at apply time
    trailing_only = {"trailing_script_prob": 0.50, "leading_script_prob": 0.10}
    out, _ = gs.apply_game_script(PROJECTION, trailing_only, recenter=False)
    assert out["qb_attempts"] == pytest.approx(36.0 * brief_qb(0.10, 0.50))  # flipped: positive = leading = 0.10
    monkeypatch.setattr(gs, "POSITIVE_SCRIPT_IS_PASS_HEAVY", True)
    out, _ = gs.apply_game_script(PROJECTION, trailing_only, recenter=False)
    assert out["qb_attempts"] == pytest.approx(36.0 * brief_qb(0.50, 0.10))


# --- apply: the brief's formulas --------------------------------------------------
@pytest.mark.parametrize(
    "pos, neg",
    [(0.0, 0.0), (0.5, 0.1), (0.1, 0.5), (0.3, 0.3), (1.0, 0.0), (0.0, 1.0), (0.2968, 0.1872), (0.7, 0.05)],
)
def test_literal_formula_equals_the_brief(pos, neg):
    out, log = gs.apply_game_script(PROJECTION, fake_result(pos, neg), recenter=False)
    assert out["qb_attempts"] == pytest.approx(36.0 * (1 + pos * 0.20 - neg * 0.25))
    assert out["rb_attempts"] == pytest.approx(18.5 * (1 + neg * 0.30 - pos * 0.20))
    assert out["qb_attempts"] == pytest.approx(36.0 * brief_qb(pos, neg))
    assert {entry["field"] for entry in log} == {"qb_attempts", "rb_attempts"}
    assert all(entry["applied"] and entry["stage"] == "game_script" and entry["recentered"] is False for entry in log)
    by_field = {entry["field"]: entry for entry in log}
    assert by_field["qb_attempts"]["factor"] == pytest.approx(brief_qb(pos, neg), abs=1e-4)
    assert by_field["rb_attempts"]["before"] == 18.5


def test_recentered_formula_divides_by_the_same_formula_at_the_reference():
    refs = (0.30, 0.20)
    out, log = gs.apply_game_script(PROJECTION, fake_result(0.55, 0.08, refs=refs))
    assert out["qb_attempts"] == pytest.approx(36.0 * brief_qb(0.55, 0.08) / brief_qb(*refs))
    assert out["rb_attempts"] == pytest.approx(18.5 * brief_rb(0.55, 0.08) / brief_rb(*refs))
    assert all(entry["recentered"] is True for entry in log)
    explicit, _ = gs.apply_game_script(PROJECTION, fake_result(0.55, 0.08, refs=refs), recenter=True)
    assert explicit == out
    # without references on the result the module's measured reference shares are used
    ref = gs.reference_shares()
    out2, _ = gs.apply_game_script(PROJECTION, fake_result(0.55, 0.08))
    assert out2["qb_attempts"] == pytest.approx(
        36.0 * brief_qb(0.55, 0.08) / brief_qb(ref["positive_script_prob"], ref["negative_script_prob"])
    )


def test_recenter_is_an_exact_noop_at_the_reference_and_the_literal_formula_is_not():
    result = gs.compute_game_script({})
    out, log = gs.apply_game_script(PROJECTION, result)
    assert out == PROJECTION
    assert all(entry["factor"] == 1.0 for entry in log)
    average, _ = gs.apply_game_script(PROJECTION, gs.compute_game_script(neutral()))
    assert average == PROJECTION
    literal, _ = gs.apply_game_script(PROJECTION, result, recenter=False)
    pos, neg = result["positive_script_prob"], result["negative_script_prob"]
    assert literal["qb_attempts"] == pytest.approx(36.0 * brief_qb(pos, neg))
    assert literal["qb_attempts"] != pytest.approx(36.0, abs=0.01)  # the raw brief formula bites a neutral game


def test_apply_input_is_not_mutated_and_everything_else_is_copied_through():
    projection = {
        "qb_attempts": 36.0, "rb_attempts": 18.5, "qb_yards": 255.0, "qb_td_prob": 0.55, "wr_yards": 71.0,
        "player": "Somebody", "note": None,
    }
    snapshot = copy.deepcopy(projection)
    out, log = gs.apply_game_script(projection, gs.compute_game_script({"vegas_spread": 9.0}))
    assert projection == snapshot
    assert out is not projection
    for key in ("qb_yards", "qb_td_prob", "wr_yards", "player", "note"):
        assert out[key] == projection[key]  # yards and probabilities are not this model's to move
    assert out["qb_attempts"] != projection["qb_attempts"] and out["rb_attempts"] != projection["rb_attempts"]
    assert len(log) == 2


def test_missing_or_non_numeric_fields_are_recorded_not_raised():
    result = gs.compute_game_script({"vegas_spread": 9.0})
    out, log = gs.apply_game_script({}, result)
    assert out == {}
    assert [(e["field"], e["applied"]) for e in log] == [("qb_attempts", False), ("rb_attempts", False)]
    assert all("reason" in e for e in log)
    out, log = gs.apply_game_script({"qb_attempts": 30.0}, result)
    applied = {e["field"]: e["applied"] for e in log}
    assert applied == {"qb_attempts": True, "rb_attempts": False} and "rb_attempts" not in out
    out, log = gs.apply_game_script({"qb_attempts": "n/a", "rb_attempts": float("nan")}, result)
    assert out["qb_attempts"] == "n/a" and math.isnan(out["rb_attempts"])
    assert not any(e["applied"] for e in log)
    out, log = gs.apply_game_script(None, result)  # a missing projection is an empty one
    assert out == {} and not any(e["applied"] for e in log)
    out, _ = gs.apply_game_script({"qb_attempts": "36.0", "rb_attempts": 18}, result, recenter=False)
    assert out["qb_attempts"] == pytest.approx(36.0 * brief_qb(result["positive_script_prob"], result["negative_script_prob"]))


def test_result_without_usable_probabilities_moves_nothing():
    for bad in ({}, {"positive_script_prob": 0.4}, {"positive_script_prob": "n/a", "negative_script_prob": None},
                {"trailing_script_prob": 0.3}, None, "nope"):
        out, log = gs.apply_game_script(PROJECTION, bad)
        assert out == PROJECTION
        assert [(e["field"], e["applied"]) for e in log] == [("qb_attempts", False), ("rb_attempts", False)]


def test_apply_reads_the_aliases_when_the_brief_names_are_absent():
    via_alias, _ = gs.apply_game_script(
        PROJECTION, {"trailing_script_prob": 0.45, "leading_script_prob": 0.12}, recenter=False
    )
    via_brief, _ = gs.apply_game_script(PROJECTION, fake_result(0.45, 0.12), recenter=False)
    assert via_alias == via_brief


def test_apply_clamps_nonsense_probabilities_and_never_goes_negative():
    out, log = gs.apply_game_script(PROJECTION, fake_result(1.7, -0.4), recenter=False)  # -> (1.0, 0.0)
    assert out["qb_attempts"] == pytest.approx(36.0 * 1.20)
    assert out["rb_attempts"] == pytest.approx(18.5 * 0.80)
    both, _ = gs.apply_game_script(PROJECTION, fake_result(0.9, 0.9), recenter=False)  # scaled to sum to 1
    assert both["qb_attempts"] == pytest.approx(36.0 * brief_qb(0.5, 0.5))
    assert both["rb_attempts"] == pytest.approx(18.5 * brief_rb(0.5, 0.5))
    zero, _ = gs.apply_game_script({"qb_attempts": 0.0, "rb_attempts": 0.0}, fake_result(0.4, 0.1))
    assert zero == {"qb_attempts": 0.0, "rb_attempts": 0.0}
    for pos in (0.0, 0.25, 0.5, 0.75, 1.0):
        for neg in (0.0, 0.25, 0.5, 0.75, 1.0):
            if pos + neg > 1.0:
                continue
            res, _ = gs.apply_game_script(PROJECTION, fake_result(pos, neg))
            assert res["qb_attempts"] >= 0.0 and res["rb_attempts"] >= 0.0
    assert [e["applied"] for e in log] == [True, True]


def test_apply_only_ever_touches_the_two_attempt_fields():
    full = {key: 10.0 + i for i, key in enumerate(c.PROJECTION_FIELDS)}
    out, log = gs.apply_game_script(full, gs.compute_game_script({"vegas_spread": 12.0}))
    changed = {key for key in full if out[key] != full[key]}
    assert changed == {"qb_attempts", "rb_attempts"}
    assert {e["field"] for e in log} == {"qb_attempts", "rb_attempts"}


# --- the full path: compute -> apply, football sanity ----------------------------
def test_dog_throws_more_and_runs_less_favourite_the_reverse_neutral_neither():
    base = {"qb_attempts": 34.0, "rb_attempts": 20.0}
    dog, _ = gs.apply_game_script(base, gs.compute_game_script({"vegas_spread": +9.5, "vegas_total": 47}))
    fav, _ = gs.apply_game_script(base, gs.compute_game_script({"vegas_spread": -9.5, "vegas_total": 47}))
    even, _ = gs.apply_game_script(base, gs.compute_game_script({"vegas_total": 45.28}))
    assert dog["qb_attempts"] > even["qb_attempts"] == base["qb_attempts"] > fav["qb_attempts"]
    assert fav["rb_attempts"] > even["rb_attempts"] == base["rb_attempts"] > dog["rb_attempts"]
    # magnitude sanity, worked by hand from the brief's formulas at the fitted shares for a 9.5-point side
    # (trailing 0.50 / leading 0.09, recentred on 0.297 / 0.187): QB x1.064 and RB x0.931 for the dog,
    # QB x0.931 and RB x1.079 for the favourite (trailing 0.16 / leading 0.35)
    assert 1.04 < dog["qb_attempts"] / 34.0 < 1.09
    assert 0.91 < dog["rb_attempts"] / 20.0 < 0.95
    assert 0.91 < fav["qb_attempts"] / 34.0 < 0.95
    assert 1.05 < fav["rb_attempts"] / 20.0 < 1.11


def test_attempt_factors_are_monotone_in_the_spread():
    base = {"qb_attempts": 34.0, "rb_attempts": 20.0}
    qb, rb = [], []
    for spread in [x * 1.0 for x in range(-18, 19)]:
        out, _ = gs.apply_game_script(base, gs.compute_game_script({"vegas_spread": spread}))
        qb.append(out["qb_attempts"])
        rb.append(out["rb_attempts"])
    assert all(b >= a for a, b in zip(qb, qb[1:]))  # bigger underdog -> more QB attempts
    assert all(b <= a for a, b in zip(rb, rb[1:]))  # ... and fewer RB attempts


@pytest.mark.parametrize(
    "label, inputs, check",
    [
        ("big home favourite", {"vegas_spread": -10.5, "vegas_total": 47.0},
         lambda p, n, l: l > 0.33 and p < 0.17 and l > 2 * p),
        ("touchdown dog in a shootout", {"vegas_spread": 7.0, "vegas_total": 54.0},
         lambda p, n, l: p > 0.42 and n < 0.46 and p > 3 * l),
        ("pick'em in a low-total grind", {"vegas_spread": 0.0, "vegas_total": 38.0},
         lambda p, n, l: n > 0.53 and p > l),
        ("four-and-a-half point favourite", {"vegas_spread": -4.5, "vegas_total": 45.0},
         lambda p, n, l: l > p and n > 0.50),
    ],
)
def test_realistic_scenarios(label, inputs, check):
    p, n, l = shares(inputs)
    assert check(p, n, l), (label, p, n, l)


def test_pace_and_pass_lean_do_not_pull_a_favourite_into_a_shootout():
    # honest-limits regression: with the line in hand a fast, pass-happy favourite is still a favourite
    calm = gs.compute_game_script({"vegas_spread": -6.0, "vegas_total": 48.0})
    hot = gs.compute_game_script(
        {"vegas_spread": -6.0, "vegas_total": 48.0, "pace_seconds_per_play": 27.0, "proe": 0.08, "rroe": -0.08}
    )
    assert hot["negative_script_prob"] > hot["positive_script_prob"]
    assert hot["negative_script_prob"] == calm["negative_script_prob"]
