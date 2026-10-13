"""Tests for the parlay correlation engine of the NFL model upgrade pack (offline, pure).

Every expected number below is either the brief's coefficient written out by hand, a hand calculation
of the documented formula, or a value measured on nflverse 2023-2025 (quoted in the module docstring);
nothing is read back from the module under test except the public constants a test is probing.
"""
from __future__ import annotations

import ast
import copy
import math
from pathlib import Path

import pytest

from modules import parlay_correlation as pc

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------
REF = {
    "collapse_probability": 0.16,
    "pass_game_collapse": 0.12,
    "run_collapse": 0.176,
    "explosion_probability": 0.117,
    "coverage_mismatch_factor": 0.0,
    "explosive_play_prob_pass": 0.2727,
    "explosive_play_prob_run": 0.1263,
    "rz_pass_funnel": 0.0,
    "rz_run_funnel": 0.0,
    "positive_script_prob": 0.2968,
    "negative_script_prob": 0.1872,
    "neutral_script_prob": 0.516,
}
SCRIPT_VAR = 0.549 ** 2  # measured game-level sd of the net script share (module docstring)
SIGNED_VAR = 0.35 ** 2   # documented prior sd of the signed factors


def bundle(**over) -> dict:
    """A team_signals-shaped bundle at the league average with some probabilities overridden."""
    probs = dict(REF)
    probs.update(over)
    return {"probabilities": probs, "references": dict(REF), "has_signal": True, "inputs_defaulted": [],
            "vegas_spread": None}


ADVERSE = bundle(collapse_probability=0.40, pass_game_collapse=0.30, explosion_probability=0.05)
FAVOURABLE = bundle(collapse_probability=0.05, pass_game_collapse=0.05, explosion_probability=0.30)


def prop(market, side="over", team="KC", position=None, name="Player", line=None, prob=0.5,
         game="BUF-KC", home="BUF", away="KC", player_id=None, **extra) -> dict:
    opponent = home if team == away else away
    leg = {
        "leg_type": "prop", "market": market, "side": side, "team": team, "opponent": opponent,
        "home_team": home, "away_team": away, "game_id": game, "player_id": player_id or name,
        "player_name": name, "position": position, "line": line, "probability": prob,
        "description": f"{name} {side} {line if line is not None else ''} {market}".replace("  ", " "),
    }
    leg.update(extra)
    return leg


def qb_over(**kw):
    kw.setdefault("name", "Quarterback")
    kw.setdefault("line", 265.5)
    return prop("passing_yards", "over", position="QB", **kw)


def wr_over(**kw):
    kw.setdefault("name", "Receiver")
    kw.setdefault("line", 68.5)
    return prop("receiving_yards", "over", position="WR", **kw)


def team_leg(kind="moneyline", team="KC", prob=0.6, game="BUF-KC", home="BUF", away="KC", line=None) -> dict:
    opponent = home if team == away else away
    return {
        "leg_type": kind, "market": kind, "side": "home" if team == home else "away", "team": team,
        "opponent": opponent, "home_team": home, "away_team": away, "game_id": game, "line": line,
        "probability": prob, "description": f"{team} {kind}",
    }


def base2(rho=0.45):
    return [[1.0, rho], [rho, 1.0]]


def hand_qb_wr(p_qc, p_pgc, p_e, cov_f=0.0, *, wr_side=1, qb_side=1):
    """Pack rho of QB-yards-over x WR-yards-over written out from the documented formula."""
    s2_qc, s2_pgc, s2_e = p_qc * (1 - p_qc), p_pgc * (1 - p_pgc), p_e * (1 - p_e)
    cov = qb_side * wr_side * (0.45 * 0.40 * s2_e + 0.22 * 0.22 * SCRIPT_VAR)
    var_qb = 0.65 ** 2 * s2_qc + 0.45 ** 2 * s2_e + 0.22 ** 2 * SCRIPT_VAR
    var_wr = 0.55 ** 2 * s2_pgc + 0.40 ** 2 * s2_e + 0.25 ** 2 * SIGNED_VAR + 0.22 ** 2 * SCRIPT_VAR
    ref_qb = 0.65 ** 2 * 0.16 * 0.84 + 0.45 ** 2 * 0.117 * 0.883 + 0.22 ** 2 * SCRIPT_VAR
    ref_wr = 0.55 ** 2 * 0.12 * 0.88 + 0.40 ** 2 * 0.117 * 0.883 + 0.25 ** 2 * SIGNED_VAR + 0.22 ** 2 * SCRIPT_VAR
    tau_qb = max(0.29 ** 2 - ref_qb, 0.25 * 0.29 ** 2)
    tau_wr = max(0.62 ** 2 - ref_wr, 0.25 * 0.62 ** 2)
    return cov / math.sqrt((var_qb + tau_qb) * (var_wr + tau_wr))


# ---------------------------------------------------------------------------
# Loadings, constants, idiosyncratic table
# ---------------------------------------------------------------------------
def test_loadings_are_the_briefs_coefficients():
    L = pc.LOADINGS
    assert L["qb_passing_yards"] == {"qb_collapse": -0.65, "explosion": 0.45, "pass_script": 0.22}
    assert L["qb_passing_tds"] == {"qb_collapse": -0.35, "red_zone_pass": 0.35, "pass_script": 0.22}
    assert L["qb_interceptions"] == {"qb_collapse": 0.40}
    assert L["wr_receiving_yards"] == {"pass_game_collapse": -0.55, "explosion": 0.40, "coverage": 0.25,
                                       "pass_script": 0.22}
    assert L["wr_receiving_tds"] == {"pass_game_collapse": -0.40, "pass_script": 0.22}
    assert L["te_receiving_yards"] == {"pass_game_collapse": -0.45, "coverage": 0.20, "pass_script": 0.22}
    assert L["te_receiving_tds"] == {"pass_game_collapse": -0.35, "red_zone_pass": 0.25}
    assert L["rb_rushing_yards"] == {"run_collapse": -0.50, "explosion": 0.35, "pass_script": -0.25}
    assert L["rb_rushing_tds"] == {"run_collapse": -0.35, "red_zone_run": 0.30, "pass_script": -0.25}
    assert L["team_result"] == {"explosion": 0.30, "qb_collapse": -0.30}
    # every regime a loading uses is a declared regime
    for family, row in L.items():
        assert set(row) <= set(pc.REGIMES), family


def test_receptions_load_at_seven_tenths_of_the_same_players_yards():
    for pos in ("wr", "te"):
        yards = pc.LOADINGS[f"{pos}_receiving_yards"]
        catches = pc.LOADINGS[f"{pos}_receptions"]
        assert set(catches) == set(yards)
        for regime, value in yards.items():
            assert catches[regime] == pytest.approx(0.7 * value, abs=1e-4)


def test_anytime_td_takes_the_larger_td_loading_of_the_position():
    assert pc.LOADINGS["rb_anytime_td"] == pc.LOADINGS["rb_rushing_tds"]
    assert pc.LOADINGS["wr_anytime_td"] == pc.LOADINGS["wr_receiving_tds"]
    assert pc.LOADINGS["te_anytime_td"] == pc.LOADINGS["te_receiving_tds"]
    # element-wise larger magnitude, first wins a tie
    assert pc._larger({"a": -0.3, "b": 0.1}, {"a": 0.2, "b": -0.4}) == {"a": -0.3, "b": -0.4}


def test_idiosyncratic_table_is_measured_and_documented():
    qb = pc.IDIOSYNCRATIC["qb_passing_yards"]
    assert qb["cv"] == pytest.approx(0.29)
    assert qb["n_units"] == 98 and qb["n_games"] > 1000
    assert qb["tau2"] == pytest.approx(0.25 * 0.29 ** 2)  # the brief's loadings exceed the observed variance
    assert qb["binding"] == "floor"
    wr = pc.IDIOSYNCRATIC["wr_receiving_yards"]
    assert wr["binding"] == "residual"
    assert wr["tau2"] == pytest.approx(0.62 ** 2 - (0.55 ** 2 * 0.12 * 0.88 + 0.40 ** 2 * 0.117 * 0.883
                                                    + 0.25 ** 2 * SIGNED_VAR + 0.22 ** 2 * SCRIPT_VAR), abs=2e-4)
    for family, meta in pc.IDIOSYNCRATIC.items():
        assert meta["tau2"] >= pc.TAU_MIN_SHARE * meta["cv"] ** 2 - 1e-12, family
        assert meta["n_units"] > 0 and meta["source"]
    # TD / count markets are far noisier than a QB's yardage
    assert pc.IDIOSYNCRATIC["te_receiving_tds"]["cv"] > 3 * pc.IDIOSYNCRATIC["qb_passing_yards"]["cv"]


# ---------------------------------------------------------------------------
# Regime state
# ---------------------------------------------------------------------------
def test_regime_state_standardises_against_the_reference():
    state = pc.regime_state(bundle(collapse_probability=0.30))["qb_collapse"]
    sigma = math.sqrt(0.16 * 0.84)
    assert state["p"] == pytest.approx(0.30)
    assert state["reference"] == pytest.approx(0.16)
    assert state["sigma"] == pytest.approx(sigma, abs=1e-4)
    assert state["z"] == pytest.approx((0.30 - 0.16) / sigma, abs=1e-3)
    assert state["variance"] == pytest.approx(0.30 * 0.70, abs=1e-4)  # Bernoulli of the GAME's probability
    assert set(pc.regime_state(bundle())) == set(pc.REGIMES)


def test_regime_state_script_is_the_signed_net_and_signed_factors_use_their_own_scale():
    state = pc.regime_state(bundle(positive_script_prob=0.45, negative_script_prob=0.10, neutral_script_prob=0.45,
                                   coverage_mismatch_factor=0.5, rz_pass_funnel=-0.2))
    script = state["pass_script"]
    assert script["p"] == pytest.approx(0.35)
    assert script["reference"] == pytest.approx(0.2968 - 0.1872, abs=1e-4)
    assert script["z"] == pytest.approx((0.35 - 0.1096) / 0.549, abs=2e-3)
    assert script["variance"] == pytest.approx(SCRIPT_VAR, abs=1e-3)
    assert state["coverage"]["z"] == pytest.approx(0.5 / 0.35, abs=1e-3)
    assert state["coverage"]["sigma"] == pytest.approx(0.35)
    assert state["red_zone_pass"]["z"] == pytest.approx(-0.2 / 0.35, abs=1e-3)


@pytest.mark.parametrize("junk", [None, {}, [], "n/a", 7, float("nan"), {"probabilities": None},
                                  {"probabilities": {"collapse_probability": "oops"}}])
def test_regime_state_never_raises_and_marks_missing(junk):
    state = pc.regime_state(junk)
    assert set(state) == set(pc.REGIMES)
    assert state["qb_collapse"]["available"] is False
    assert state["qb_collapse"]["p"] is None and state["qb_collapse"]["z"] is None


def test_regime_state_accepts_percentages_and_a_flat_probability_mapping():
    flat = pc.regime_state({"collapse_probability": 30})  # 30 means 30%
    assert flat["qb_collapse"]["p"] == pytest.approx(0.30)
    assert flat["qb_collapse"]["reference"] == pytest.approx(0.16)  # documented neutral when no reference given


# ---------------------------------------------------------------------------
# Market family and loadings of a leg
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "leg, expected",
    [
        (prop("passing_yards", position="QB"), "qb_passing_yards"),
        (prop("passing_tds", position="QB"), "qb_passing_tds"),
        (prop("interceptions", position="QB"), "qb_interceptions"),
        (prop("receiving_yards", position="WR"), "wr_receiving_yards"),
        (prop("receiving_yards", position="TE"), "te_receiving_yards"),
        (prop("receptions", position="TE"), "te_receptions"),
        (prop("receiving_tds", position="WR"), "wr_receiving_tds"),
        (prop("rushing_yards", position="RB"), "rb_rushing_yards"),
        (prop("rushing_tds", position="HB"), "rb_rushing_tds"),
        (prop("anytime_td", position="RB"), "rb_anytime_td"),
        (prop("anytime_td", position="TE"), "te_anytime_td"),
        (prop("receiving_yards", position=None), "wr_receiving_yards"),   # position inferred
        (prop("passing_yards", position=None), "qb_passing_yards"),
        (prop("receiving_yards", position="RB"), None),                    # the brief has no RB receiving row
        (prop("anytime_td", position="QB"), None),
        (prop("anytime_td", position=None), None),
        (prop("passing_yards", position="WR"), None),
        (prop("sacks", position="DE"), None),
        (team_leg("moneyline"), "team_result"),
        (team_leg("spread", line=-3.5), "team_result"),
        (prop("total", "over"), "game_total"),
        (None, None),
        ("not a leg", None),
        ({}, None),
    ],
)
def test_market_family(leg, expected):
    assert pc.market_family(leg) == expected


def test_under_flips_every_loading_sign():
    over = pc.leg_regime_loadings(qb_over())
    under = pc.leg_regime_loadings(prop("passing_yards", "under", position="QB"))
    assert over == {"qb_collapse": -0.65, "explosion": 0.45, "pass_script": 0.22}
    assert under == {"qb_collapse": 0.65, "explosion": -0.45, "pass_script": -0.22}
    # interceptions are the one market a collapse HELPS the over of
    assert pc.leg_regime_loadings(prop("interceptions", "over", position="QB"))["qb_collapse"] == 0.40
    assert pc.leg_regime_loadings(prop("interceptions", "under", position="QB"))["qb_collapse"] == -0.40


def test_team_leg_loads_on_its_own_offence_and_the_opponent_with_opposite_sign():
    ml = team_leg("moneyline", team="KC", home="BUF", away="KC")
    assert pc.leg_regime_loadings(ml) == {"explosion": 0.30, "qb_collapse": -0.30}
    exposures = pc.leg_exposures(ml)
    assert exposures[("KC", "explosion")] == 0.30
    assert exposures[("KC", "qb_collapse")] == -0.30
    assert exposures[("BUF", "explosion")] == -0.30
    assert exposures[("BUF", "qb_collapse")] == 0.30


def test_game_total_over_loads_both_offences_with_the_same_sign_and_under_flips():
    over = pc.leg_exposures(prop("total", "over", team=None, game="BUF-KC"))
    assert over[("KC", "explosion")] == over[("BUF", "explosion")] == pytest.approx(0.15)
    assert over[("KC", "qb_collapse")] == over[("BUF", "qb_collapse")] == pytest.approx(-0.15)
    under = pc.leg_exposures(prop("total", "under", team=None, game="BUF-KC"))
    assert under[("KC", "explosion")] == pytest.approx(-0.15)


def test_unmodelled_leg_has_no_loadings():
    assert pc.leg_regime_loadings(prop("sacks", position="DE")) == {}
    assert pc.leg_regime_loadings(None) == {}
    assert pc.leg_regime_loadings(prop("passing_yards", side=None, position="QB")) == {}  # no side, no sign


# ---------------------------------------------------------------------------
# Pair correlation
# ---------------------------------------------------------------------------
def test_pair_rho_matches_hand_calculation_of_the_documented_formula():
    out = pc.pair_correlation(qb_over(), wr_over(), {"KC": bundle(collapse_probability=0.30,
                                                                  pass_game_collapse=0.20,
                                                                  explosion_probability=0.20)})
    assert out["rho"] == pytest.approx(hand_qb_wr(0.30, 0.20, 0.20), abs=2e-4)
    assert out["kind"] == "regime_aligned"
    assert set(out["regimes"]) == {"KC:explosion", "KC:pass_script"}  # the only regimes the two legs share
    assert out["regimes"]["KC:explosion"]["loading_a"] == 0.45
    assert out["regimes"]["KC:explosion"]["loading_b"] == 0.40
    assert "not fitted" in out["note"]


def test_pair_rho_flips_sign_for_an_under_leg_and_keeps_its_magnitude():
    signals = {"KC": ADVERSE}
    both_over = pc.pair_correlation(qb_over(), wr_over(), signals)["rho"]
    wr_under = pc.pair_correlation(qb_over(), prop("receiving_yards", "under", position="WR", name="Receiver",
                                                   line=68.5), signals)["rho"]
    assert wr_under == pytest.approx(-both_over, abs=1e-4)
    both_under = pc.pair_correlation(prop("passing_yards", "under", position="QB", name="Quarterback", line=265.5),
                                     prop("receiving_yards", "under", position="WR", name="Receiver", line=68.5),
                                     signals)["rho"]
    assert both_under == pytest.approx(both_over, abs=1e-4)


def test_opposing_team_legs_get_the_opposite_sign():
    signals = {"KC": bundle(explosion_probability=0.25), "BUF": bundle(explosion_probability=0.25)}
    ml_kc = team_leg("moneyline", team="KC")
    same = pc.pair_correlation(ml_kc, qb_over(team="KC"), signals)["rho"]
    other = pc.pair_correlation(ml_kc, qb_over(team="BUF", name="Opposing QB"), signals)["rho"]
    assert same > 0.02
    assert other < -0.02
    assert other == pytest.approx(-same, abs=1e-4)  # identical bundles -> identical magnitude, opposite sign


def test_pack_rho_is_monotone_in_the_shared_regime_probability():
    rhos = [
        pc.pair_correlation(qb_over(), wr_over(), {"KC": bundle(explosion_probability=p)})["rho"]
        for p in (0.05, 0.15, 0.30, 0.45)
    ]
    assert rhos == sorted(rhos)
    assert rhos[-1] > rhos[0] + 0.01  # Bernoulli variance p(1-p) grows towards 50/50


def test_pair_has_no_opinion_without_a_shared_regime_signal_or_known_market():
    # different games
    far = prop("receiving_yards", position="WR", team="NYJ", home="MIA", away="NYJ", game="MIA-NYJ")
    out = pc.pair_correlation(qb_over(), far, {"KC": ADVERSE, "NYJ": ADVERSE})
    assert out["rho"] is None and out["kind"] == "different_games"
    # opposing offences' props share no regime of one offence
    out = pc.pair_correlation(qb_over(team="KC"), qb_over(team="BUF", name="Other QB"),
                              {"KC": ADVERSE, "BUF": ADVERSE})
    assert out["rho"] is None and out["kind"] == "no_shared_regime"
    # a team without signals
    out = pc.pair_correlation(qb_over(), wr_over(), {"BUF": ADVERSE})
    assert out["rho"] is None and out["kind"] == "no_signal"
    # unknown market
    out = pc.pair_correlation(qb_over(), prop("sacks", position="DE"), {"KC": ADVERSE})
    assert out["rho"] is None and out["kind"] == "unmodelled_market"


# ---------------------------------------------------------------------------
# build_correlation_matrix
# ---------------------------------------------------------------------------
def test_no_signals_leaves_the_base_matrix_exactly_unchanged():
    legs = [qb_over(), wr_over(), prop("rushing_yards", position="RB", name="Back", line=60.5)]
    base = [[1.0, 0.4512345, -0.0731], [0.4512345, 1.0, -0.05], [-0.0731, -0.05, 1.0]]
    for signals in (None, {}, {"NYJ": ADVERSE}, {"KC": {"probabilities": {}, "has_signal": False}},
                    {"KC": dict(ADVERSE, has_signal=False)}):
        out = pc.build_correlation_matrix(legs, signals, base_matrix=base)
        assert out["matrix"] == base
        assert out["pack_pairs"] == 0
        assert out["has_signal"] is False
        assert all(pair["touched"] is False and pair["blended_rho"] == pair["base_rho"] for pair in out["pairs"])


def test_no_base_gives_identity_with_zeros_when_there_are_no_signals():
    out = pc.build_correlation_matrix([qb_over(), wr_over()], None)
    assert out["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert out["weight"] == pc.DEFAULT_PACK_WEIGHT


def test_blend_arithmetic_against_a_hand_calculation(monkeypatch):
    monkeypatch.setattr(pc, "WEIGHT_BY_EXPLAINED_SHARE", False)
    probs = dict(collapse_probability=0.30, pass_game_collapse=0.20, explosion_probability=0.20)
    out = pc.build_correlation_matrix([qb_over(), wr_over()], {"KC": bundle(**probs)}, base_matrix=base2(0.45))
    pack = hand_qb_wr(0.30, 0.20, 0.20)
    pack_ref = hand_qb_wr(0.16, 0.12, 0.117)            # the regime model's own rho in a league-average game
    w = pc.DEFAULT_PACK_WEIGHT
    expected = 0.45 + w * (pack - pack_ref)              # all twelve signals present -> coverage 1, evidence 1
    pair = out["pairs"][0]
    assert pair["pack_rho"] == pytest.approx(pack, abs=2e-4)
    assert pair["pack_rho_reference"] == pytest.approx(pack_ref, abs=2e-4)
    assert pair["weight_used"] == pytest.approx(w, abs=1e-4)
    assert pair["blended_rho"] == pytest.approx(expected, abs=3e-4)
    assert out["matrix"][0][1] == out["matrix"][1][0] == pytest.approx(expected, abs=3e-4)
    assert out["pack_pairs"] == 1 and out["has_signal"] is True
    # an explicit weight replaces the default
    out = pc.build_correlation_matrix([qb_over(), wr_over()], {"KC": bundle(**probs)},
                                      base_matrix=base2(0.45), weight=0.5)
    assert out["matrix"][0][1] == pytest.approx(0.45 + 0.5 * (pack - pack_ref), abs=3e-4)


def test_explained_share_shrinks_the_weight_for_noisy_pairs(monkeypatch):
    monkeypatch.setattr(pc, "WEIGHT_BY_EXPLAINED_SHARE", True)   # opt-in damping (off by default)
    probs = dict(collapse_probability=0.30, pass_game_collapse=0.20, explosion_probability=0.20)
    out = pc.build_correlation_matrix([qb_over(), wr_over()], {"KC": bundle(**probs)}, base_matrix=base2(0.45))
    weight = out["pairs"][0]["weight_used"]
    explained = out["pairs"][0]["explained"]
    assert 0.0 < explained < 1.0
    assert weight == pytest.approx(pc.DEFAULT_PACK_WEIGHT * explained, abs=2e-4)
    assert weight < pc.DEFAULT_PACK_WEIGHT
    # a TD pair (Poisson-noisy, cv ~1.7) barely moves
    td = pc.build_correlation_matrix(
        [prop("passing_tds", position="QB", name="Quarterback", line=1.5),
         prop("receiving_tds", position="WR", name="Receiver", line=0.5)],
        {"KC": bundle(**probs)}, base_matrix=base2(0.35))
    assert td["pairs"][0]["weight_used"] < weight
    assert abs(td["matrix"][0][1] - 0.35) < 0.02


def test_matrix_is_symmetric_unit_diagonal_and_in_range():
    legs = [qb_over(), wr_over(), prop("receiving_yards", position="TE", name="Tight End", line=45.5),
            prop("rushing_yards", position="RB", name="Back", line=60.5), team_leg("moneyline", team="KC"),
            prop("passing_yards", "under", position="QB", team="BUF", name="Other QB", line=250.5)]
    base = [[1.0 if i == j else 0.15 for j in range(6)] for i in range(6)]
    out = pc.build_correlation_matrix(legs, {"KC": ADVERSE, "BUF": FAVOURABLE}, base_matrix=base)
    m = out["matrix"]
    for i in range(6):
        assert m[i][i] == 1.0
        for j in range(6):
            assert m[i][j] == m[j][i]
            assert -1.0 <= m[i][j] <= 1.0
    assert out["pack_pairs"] > 0


def test_contradiction_sentinel_and_structural_pairs_are_preserved():
    legs = [qb_over(), wr_over(), prop("receptions", position="WR", name="Receiver", line=5.5)]
    base = [[1.0, -2.0, 0.45], [-2.0, 1.0, 1.0], [0.45, 1.0, 1.0]]
    out = pc.build_correlation_matrix(legs, {"KC": ADVERSE}, base_matrix=base)
    m = out["matrix"]
    assert m[0][1] == m[1][0] == -2.0          # mutually exclusive sentinel
    assert m[1][2] == m[2][1] == 1.0           # structural +1 (a ladder / duplicate)
    assert m[0][2] != 0.45                     # the one ordinary same-game pair was touched
    assert out["psd"] is None                  # a sentinel makes a PSD claim meaningless
    kinds = {(p["i"], p["j"]): p["touched"] for p in out["pairs"]}
    assert kinds[(0, 1)] is False and kinds[(1, 2)] is False and kinds[(0, 2)] is True


def test_legs_in_different_games_are_untouched():
    far = prop("receiving_yards", position="WR", team="NYJ", home="MIA", away="NYJ", game="MIA-NYJ", name="Far WR")
    base = base2(0.07)
    out = pc.build_correlation_matrix([qb_over(), far], {"KC": ADVERSE, "NYJ": ADVERSE}, base_matrix=base)
    assert out["matrix"] == base
    assert out["pairs"][0]["pack_rho"] is None and out["pairs"][0]["kind"] == "different_games"


def test_weight_scales_with_signal_coverage(monkeypatch):
    monkeypatch.setattr(pc, "WEIGHT_BY_EXPLAINED_SHARE", False)
    legs = [qb_over(), wr_over()]
    full = bundle(collapse_probability=0.30, pass_game_collapse=0.20, explosion_probability=0.20)
    partial = copy.deepcopy(full)
    for key in ("positive_script_prob", "negative_script_prob", "neutral_script_prob"):
        del partial["probabilities"][key]
    w_full = pc.build_correlation_matrix(legs, {"KC": full}, base_matrix=base2())["pairs"][0]
    w_part = pc.build_correlation_matrix(legs, {"KC": partial}, base_matrix=base2())["pairs"][0]
    mass_all = sum(abs(v) for v in pc.LOADINGS["qb_passing_yards"].values()) \
        + sum(abs(v) for v in pc.LOADINGS["wr_receiving_yards"].values())
    mass_ok = mass_all - 2 * 0.22  # the script loading is lost on both legs
    assert w_full["weight_used"] == pytest.approx(pc.DEFAULT_PACK_WEIGHT, abs=1e-4)
    assert w_part["weight_used"] == pytest.approx(pc.DEFAULT_PACK_WEIGHT * mass_ok / mass_all, abs=1e-3)
    assert w_part["weight_used"] < w_full["weight_used"]
    assert w_part["coverage"] == pytest.approx(mass_ok / mass_all, abs=1e-3)


def test_weight_zero_is_an_exact_identity_and_weights_are_clamped():
    base = base2(0.45)
    signals = {"KC": ADVERSE}
    assert pc.build_correlation_matrix([qb_over(), wr_over()], signals, base_matrix=base, weight=0.0)["matrix"] == base
    assert pc.build_correlation_matrix([qb_over(), wr_over()], signals, weight=7)["weight"] == 1.0
    assert pc.build_correlation_matrix([qb_over(), wr_over()], signals, weight=-1)["weight"] == 0.0
    assert pc.build_correlation_matrix([qb_over(), wr_over()], signals, weight="x")["weight"] == pc.DEFAULT_PACK_WEIGHT


def test_blended_entry_is_clipped_to_the_builder_cap():
    # a non-structural base of 0.998 is untouched with no signal but is capped once the pack blends it
    out = pc.build_correlation_matrix([qb_over(), wr_over()], {"KC": ADVERSE}, base_matrix=base2(0.998), weight=1.0)
    assert out["matrix"][0][1] <= 0.95 + 1e-12
    assert pc.build_correlation_matrix([qb_over(), wr_over()], None, base_matrix=base2(0.998))["matrix"][0][1] == 0.998


def test_a_league_average_bundle_adds_nothing_even_when_has_signal_is_true():
    # nfl_upgrade_pipeline.team_signals({}) reports has_signal=True because its derived inputs count as
    # coverage, but every probability sits on its reference: a missing feed must stay neutral.
    neutral = bundle()
    neutral["probabilities"]["run_collapse"] = 0.1735  # what the pipeline really returns for {}
    base = base2(0.45)
    out = pc.build_correlation_matrix([qb_over(), wr_over()], {"KC": neutral}, base_matrix=base)
    assert out["matrix"] == base
    assert out["has_signal"] is False
    strength = pc.parlay_strength([qb_over(), wr_over()], {"KC": neutral}, matrix=base)
    assert strength["score"] == 50.0 and strength["has_signal"] is False


def test_evidence_ramps_up_from_the_neutral_floor():
    legs = [qb_over(), wr_over()]
    weights = []
    for p in (0.16, 0.165, 0.18, 0.22):
        out = pc.build_correlation_matrix(legs, {"KC": bundle(collapse_probability=p)}, base_matrix=base2())
        weights.append(out["pairs"][0]["weight_used"])
    assert weights[0] == 0.0
    assert weights == sorted(weights)
    assert weights[-1] > 0.05


def test_pairs_report_every_field_the_page_needs():
    out = pc.build_correlation_matrix([qb_over(), wr_over()], {"KC": ADVERSE}, base_matrix=base2())
    pair = out["pairs"][0]
    for key in ("i", "j", "labels", "base_rho", "pack_rho", "pack_rho_reference", "blended_rho", "weight_used",
                "touched", "kind", "note", "coverage", "evidence", "explained", "same_game"):
        assert key in pair
    assert isinstance(out["psd"], bool)
    assert out["basis"] and "PRIOR" in out["basis"]


def test_psd_flag_reports_an_inconsistent_matrix_without_repairing_it():
    legs = [qb_over(), wr_over(), prop("receptions", position="WR", name="Other WR", line=5.5)]
    bad = [[1.0, -0.9, -0.9], [-0.9, 1.0, -0.9], [-0.9, -0.9, 1.0]]
    out = pc.build_correlation_matrix(legs, None, base_matrix=bad)
    assert out["psd"] is False and out["min_eigenvalue"] < 0
    assert out["matrix"] == bad
    good = pc.build_correlation_matrix(legs, None, base_matrix=[[1, 0.3, 0.2], [0.3, 1, 0.1], [0.2, 0.1, 1]])
    assert good["psd"] is True


# ---------------------------------------------------------------------------
# parlay_strength: neutral case, contradictions
# ---------------------------------------------------------------------------
def test_no_signals_is_neutral_and_never_invents_a_signal():
    out = pc.parlay_strength([qb_over(), wr_over()], {})
    assert out["score"] == 50.0
    assert out["lift"] is None or out["lift"] == 1.0
    assert out["has_signal"] is False
    assert out["alignment"] == 0.0 and out["fragility"] == 0.0
    assert out["flags"] == {"mega": False, "trap": False, "contradiction": False,
                            "boost_overs": False, "boost_unders": False}
    assert any("No team signals" in reason for reason in out["reasons"])
    assert "PRIOR" in out["basis"]
    assert pc.parlay_strength([qb_over(), wr_over()], None)["score"] == 50.0
    assert pc.parlay_strength([], {"KC": ADVERSE})["score"] == 50.0


def test_contradiction_is_structural_and_needs_no_signal():
    over = prop("passing_yards", "over", position="QB", name="Quarterback", line=250.5)
    under = prop("passing_yards", "under", position="QB", name="Quarterback", line=240.5)
    out = pc.parlay_strength([over, under], {})
    assert out["flags"]["contradiction"] is True and out["flags"]["trap"] is True
    assert out["score"] <= pc.CONTRADICTION_SCORE_CAP
    assert out["flags"]["mega"] is False
    # the builder's matrix sentinel is honoured too
    sentinel = pc.parlay_strength([qb_over(), wr_over()], {"KC": FAVOURABLE}, matrix=[[1, -2.0], [-2.0, 1]])
    assert sentinel["flags"]["contradiction"] is True and sentinel["score"] <= pc.CONTRADICTION_SCORE_CAP
    assert sentinel["lift"] == 0.0
    # a middle (over a low line, under a high line) is NOT a contradiction
    middle = pc.parlay_strength([prop("passing_yards", "over", position="QB", name="Quarterback", line=230.5),
                                 prop("passing_yards", "under", position="QB", name="Quarterback", line=260.5)], {})
    assert middle["flags"]["contradiction"] is False


def test_opposing_moneylines_are_a_contradiction():
    out = pc.parlay_strength([team_leg("moneyline", team="KC"), team_leg("moneyline", team="BUF")], {})
    assert out["flags"]["contradiction"] is True


# ---------------------------------------------------------------------------
# parlay_strength: trap, mega, boosts
# ---------------------------------------------------------------------------
def test_high_collapse_qb_wr_overs_are_a_trap_with_negative_alignment():
    legs = [qb_over(), wr_over()]
    m = pc.build_correlation_matrix(legs, {"KC": ADVERSE}, base_matrix=base2(0.45))["matrix"]
    out = pc.parlay_strength(legs, {"KC": ADVERSE}, matrix=m, base_matrix=base2(0.45))
    assert out["flags"]["trap"] is True
    assert out["flags"]["mega"] is False
    assert out["alignment"] <= pc.TRAP_ALIGNMENT
    assert out["alignment"] < -0.5
    assert out["fragility"] == pytest.approx(1.0)         # both legs hurt by a +24pt collapse excess
    assert out["score"] < 25.0
    text = " ".join(out["reasons"])
    assert "Trap" in text and "QB collapse risk 40%" in text and "16% league average" in text
    assert all(leg["tailwind"] < 0 for leg in out["legs"])


def test_favourable_stack_is_mega_and_boosts_the_overs():
    legs = [qb_over(), wr_over()]
    m = pc.build_correlation_matrix(legs, {"KC": FAVOURABLE}, base_matrix=base2(0.45))["matrix"]
    out = pc.parlay_strength(legs, {"KC": FAVOURABLE}, matrix=m)
    assert out["flags"]["mega"] is True and out["flags"]["trap"] is False
    assert out["flags"]["boost_overs"] is True and out["flags"]["boost_unders"] is False
    assert out["alignment"] >= pc.MEGA_ALIGNMENT
    assert out["lift"] >= pc.MEGA_LIFT
    assert out["fragility"] == 0.0
    assert out["score"] >= pc.MEGA_SCORE
    text = " ".join(out["reasons"])
    assert "Mega" in text and "Boost overs" in text and "explosion chance 30%" in text


def test_collapse_prone_game_boosts_correlated_unders_and_is_not_fragile():
    legs = [prop("passing_yards", "under", position="QB", name="Quarterback", line=265.5),
            prop("receiving_yards", "under", position="WR", name="Receiver", line=68.5)]
    out = pc.parlay_strength(legs, {"KC": ADVERSE}, matrix=base2(0.45))
    assert out["flags"]["boost_unders"] is True and out["flags"]["boost_overs"] is False
    assert out["flags"]["trap"] is False
    assert out["alignment"] > 0.5
    assert out["fragility"] == 0.0   # a collapse helps an under
    assert any("Boost unders" in reason for reason in out["reasons"])


def test_trap_also_fires_for_an_under_stack_riding_an_explosive_game():
    legs = [prop("passing_yards", "under", position="QB", name="Quarterback", line=265.5),
            prop("receiving_yards", "under", position="WR", name="Receiver", line=68.5)]
    out = pc.parlay_strength(legs, {"KC": FAVOURABLE}, matrix=base2(0.45))
    assert out["flags"]["trap"] is True and out["alignment"] <= pc.TRAP_ALIGNMENT
    assert out["flags"]["boost_unders"] is False


def test_mega_and_trap_are_never_both_true_across_a_sweep():
    legs = [qb_over(), wr_over()]
    seen_mega = seen_trap = False
    for qc in (0.04, 0.10, 0.16, 0.25, 0.40):
        for pgc in (0.05, 0.12, 0.30):
            for ex in (0.04, 0.117, 0.30):
                sig = {"KC": bundle(collapse_probability=qc, pass_game_collapse=pgc, explosion_probability=ex)}
                m = pc.build_correlation_matrix(legs, sig, base_matrix=base2(0.45))["matrix"]
                flags = pc.parlay_strength(legs, sig, matrix=m)["flags"]
                assert not (flags["mega"] and flags["trap"])
                seen_mega |= flags["mega"]
                seen_trap |= flags["trap"]
    assert seen_mega and seen_trap  # the sweep actually reaches both flags


def test_rule_table_boost_the_regimes_do_not_support_is_a_trap():
    # WR over + RB over share a negative script loading and the explosion regime is dormant: the regimes
    # imply a (slightly) negative rho while the rule table is told to credit +0.30.
    legs = [wr_over(), prop("rushing_yards", position="RB", name="Back", line=60.5)]
    sig = {"KC": bundle(explosion_probability=0.05, positive_script_prob=0.40, negative_script_prob=0.10,
                        neutral_script_prob=0.50)}
    assert pc.pair_correlation(legs[0], legs[1], sig)["rho"] <= pc.PACK_NEAR_ZERO_RHO
    out = pc.parlay_strength(legs, sig, matrix=base2(0.30))
    assert out["flags"]["trap"] is True
    assert any("rule table credits" in reason for reason in out["reasons"])
    # the same pair with a rule table that makes no claim is not trapped on that clause
    quiet = pc.parlay_strength(legs, sig, matrix=base2(-0.05))
    assert not any("rule table credits" in reason for reason in quiet["reasons"])


def test_boundary_trap_alignment_is_inclusive(monkeypatch):
    legs = [qb_over(), wr_over()]
    sig = {"KC": ADVERSE}
    alignment = pc.parlay_strength(legs, sig, matrix=base2(0.45))["alignment"]
    monkeypatch.setattr(pc, "TRAP_ALIGNMENT", alignment + 1e-3)
    assert pc.parlay_strength(legs, sig, matrix=base2(0.45))["flags"]["trap"] is True
    monkeypatch.setattr(pc, "TRAP_ALIGNMENT", alignment - 1e-3)
    assert pc.parlay_strength(legs, sig, matrix=base2(0.45))["flags"]["trap"] is False


def _exact_alignment(legs, signals):
    """The unrounded mean tilt of the informative legs (reaches into the module to test an equality edge)."""
    tstates = pc._team_states(signals)
    preps = [pc._prep(leg, i) for i, leg in enumerate(legs)]
    tilts = [pc._leg_state(p, tstates)["tilt"] for p in preps]
    return sum(tilts) / len(tilts)


def test_alignment_thresholds_are_inclusive_at_the_exact_value(monkeypatch):
    legs = [qb_over(), wr_over()]
    exact = _exact_alignment(legs, {"KC": ADVERSE})
    monkeypatch.setattr(pc, "TRAP_ALIGNMENT", exact)
    assert pc.parlay_strength(legs, {"KC": ADVERSE}, matrix=base2(0.45))["flags"]["trap"] is True   # <=, not <
    monkeypatch.undo()
    exact = _exact_alignment(legs, {"KC": FAVOURABLE})
    monkeypatch.setattr(pc, "MEGA_ALIGNMENT", exact)
    assert pc.parlay_strength(legs, {"KC": FAVOURABLE}, matrix=base2(0.45))["flags"]["mega"] is True  # >=, not >


def test_out_of_range_base_entries_are_held_to_the_unit_interval_but_the_sentinel_stays():
    legs = [qb_over(), wr_over(), prop("receptions", position="WR", name="Receiver", line=5.5)]
    base = [[1.0, 1.3, -1.2], [1.3, 1.0, -2.0], [-1.2, -2.0, 1.0]]
    out = pc.build_correlation_matrix(legs, {"KC": ADVERSE}, base_matrix=base)["matrix"]
    assert out[0][1] == out[1][0] == 1.0
    assert out[0][2] == out[2][0] == -1.0
    assert out[1][2] == out[2][1] == -2.0


def test_boundaries_of_every_mega_condition(monkeypatch):
    legs = [qb_over(), wr_over()]
    sig = {"KC": FAVOURABLE}
    base = pc.parlay_strength(legs, sig, matrix=base2(0.45))
    assert base["flags"]["mega"] is True
    # alignment
    monkeypatch.setattr(pc, "MEGA_ALIGNMENT", base["alignment"] + 1e-3)
    assert pc.parlay_strength(legs, sig, matrix=base2(0.45))["flags"]["mega"] is False
    monkeypatch.undo()
    # lift (exact lift supplied)
    ok = pc.parlay_strength(legs, sig, joint_probability=0.30, independent_probability=0.25, matrix=base2(0.45))
    assert ok["lift"] == pytest.approx(1.2) and ok["flags"]["mega"] is True
    low = pc.parlay_strength(legs, sig, joint_probability=0.2875, independent_probability=0.25, matrix=base2(0.45))
    assert low["lift"] == pytest.approx(1.15) and low["flags"]["mega"] is True        # exactly at MEGA_LIFT
    under = pc.parlay_strength(legs, sig, joint_probability=0.2870, independent_probability=0.25, matrix=base2(0.45))
    assert under["lift"] < pc.MEGA_LIFT and under["flags"]["mega"] is False
    # fragility ceiling
    monkeypatch.setattr(pc, "MEGA_FRAGILITY_MAX", -1e-3)
    assert pc.parlay_strength(legs, sig, matrix=base2(0.45))["flags"]["mega"] is False
    monkeypatch.undo()
    # a strongly adverse leg blocks mega: raise the bar above the weaker (WR) leg's tilt
    wr_tilt = base["legs"][1]["tailwind"]
    monkeypatch.setattr(pc, "STRONG_ADVERSE_TILT", wr_tilt + 1e-3)
    assert pc.parlay_strength(legs, sig, matrix=base2(0.45))["flags"]["mega"] is False
    monkeypatch.undo()
    # two legs are required
    single = pc.parlay_strength([qb_over()], sig, matrix=[[1.0]])
    assert single["flags"]["mega"] is False


def test_boost_requires_two_same_side_legs_sharing_a_favourable_driver():
    sig = {"KC": FAVOURABLE}
    assert pc.parlay_strength([qb_over()], sig)["flags"]["boost_overs"] is False  # one leg is not a stack
    # an over and an under never boost each other
    mixed = pc.parlay_strength([qb_over(), prop("receiving_yards", "under", position="WR", name="Receiver",
                                                line=68.5)], sig)
    assert mixed["flags"]["boost_overs"] is False and mixed["flags"]["boost_unders"] is False
    # a regime that is only mildly favourable (below Z_FAVOURABLE) does not boost
    mild = {"KC": bundle(collapse_probability=0.14, pass_game_collapse=0.11, explosion_probability=0.125)}
    assert pc.parlay_strength([qb_over(), wr_over()], mild)["flags"]["boost_overs"] is False


# ---------------------------------------------------------------------------
# parlay_strength: score, lift, fragility, per-leg numbers
# ---------------------------------------------------------------------------
def test_score_is_monotone_in_alignment():
    legs = [qb_over(), wr_over()]
    scores = []
    alignments = []
    for qc in (0.45, 0.35, 0.25, 0.16, 0.10, 0.05):
        sig = {"KC": bundle(collapse_probability=qc, pass_game_collapse=qc * 0.75)}
        out = pc.parlay_strength(legs, sig, matrix=base2(0.45))
        scores.append(out["score"])
        alignments.append(out["alignment"])
    assert alignments == sorted(alignments)
    assert scores == sorted(scores)
    assert scores[-1] - scores[0] > 40


def test_score_is_monotone_in_lift():
    legs = [qb_over(), wr_over()]
    sig = {"KC": bundle(collapse_probability=0.20)}
    scores = [pc.parlay_strength(legs, sig, joint_probability=0.25 * lift, independent_probability=0.25)["score"]
              for lift in (0.5, 0.8, 1.0, 1.2, 1.5, 2.5)]
    assert scores == sorted(scores)
    assert scores[0] < scores[2] < scores[-1]


def test_score_formula_components_add_up():
    legs = [qb_over(), wr_over()]
    out = pc.parlay_strength(legs, {"KC": ADVERSE}, joint_probability=0.30, independent_probability=0.25)
    c = out["components"]
    assert c["alignment_points"] == pytest.approx(30.0 * out["alignment"], abs=0.01)
    assert c["lift_points"] == pytest.approx(20.0 * math.log(1.2) / math.log(1.5), abs=0.01)
    assert c["fragility_points"] == pytest.approx(-15.0 * out["fragility"], abs=0.01)
    assert c["trap_points"] == -15.0
    expected = 50.0 + c["alignment_points"] + c["lift_points"] + c["fragility_points"] + c["trap_points"]
    assert out["score"] == pytest.approx(max(0.0, min(100.0, expected)), abs=0.02)
    assert 0.0 <= out["score"] <= 100.0


def test_fragility_is_shared_collapse_exposure_above_reference():
    legs = [qb_over(), wr_over()]
    sig = {"KC": bundle(collapse_probability=0.25, pass_game_collapse=0.15)}
    out = pc.parlay_strength(legs, sig)
    # both legs are hurt by the passing-game collapse group; severity = (0.25 - 0.16) / 0.15 = 0.6
    assert out["fragility"] == pytest.approx(0.6, abs=1e-3)
    # a rushing leg is not exposed to the passing collapse
    mixed = pc.parlay_strength([qb_over(), prop("rushing_yards", position="RB", name="Back", line=60.5)], sig)
    assert mixed["fragility"] == pytest.approx(0.5 * 0.6, abs=1e-3)
    # below reference: no exposure
    calm = pc.parlay_strength(legs, {"KC": bundle(collapse_probability=0.08, pass_game_collapse=0.05)})
    assert calm["fragility"] == 0.0


def test_leg_shift_is_the_briefs_formula_relative_to_a_league_average_game():
    sig = {"KC": bundle(collapse_probability=0.30, explosion_probability=0.20, positive_script_prob=0.40,
                        negative_script_prob=0.10, neutral_script_prob=0.50)}
    out = pc.parlay_strength([qb_over(), wr_over()], sig)
    qb = out["legs"][0]
    expected = -0.65 * (0.30 - 0.16) + 0.45 * (0.20 - 0.117) + 0.22 * ((0.40 - 0.10) - (0.2968 - 0.1872))
    assert qb["shift"] == pytest.approx(expected, abs=1e-3)
    var = (0.65 ** 2 * 0.30 * 0.70 + 0.45 ** 2 * 0.20 * 0.80 + 0.22 ** 2 * SCRIPT_VAR)
    tau2 = 0.25 * 0.29 ** 2
    z = expected / math.sqrt(var + tau2)
    assert qb["z"] == pytest.approx(z, abs=2e-3)
    assert qb["tailwind"] == pytest.approx(math.tanh(z / 0.30), abs=2e-3)
    assert qb["has_signal"] is True and qb["drivers"]


def test_legs_without_a_team_signal_are_scored_neutral_not_penalised():
    far = prop("receiving_yards", position="WR", team="NYJ", home="MIA", away="NYJ", game="MIA-NYJ", name="Far WR")
    with_far = pc.parlay_strength([qb_over(), wr_over(), far], {"KC": FAVOURABLE}, matrix=[[1, .45, 0], [.45, 1, 0], [0, 0, 1]])
    without = pc.parlay_strength([qb_over(), wr_over()], {"KC": FAVOURABLE}, matrix=base2(0.45))
    assert with_far["alignment"] == pytest.approx(without["alignment"])
    assert with_far["n_signal_legs"] == 2
    assert with_far["legs"][2]["has_signal"] is False and with_far["legs"][2]["tailwind"] == 0.0
    assert any("no usable signals" in reason for reason in with_far["reasons"])


def test_lift_matches_the_two_leg_closed_form():
    legs = [qb_over(prob=0.5), wr_over(prob=0.5)]
    out = pc.parlay_strength(legs, {}, matrix=base2(0.5))
    joint = 0.25 + math.asin(0.5) / (2 * math.pi)        # bivariate normal orthant probability
    assert out["lift"] == pytest.approx(joint / 0.25, abs=1e-3)
    exact = pc.parlay_strength(legs, {}, joint_probability=0.31, independent_probability=0.25)
    assert exact["lift"] == pytest.approx(1.24)
    # an uncorrelated matrix is lift 1; a Frechet-bound clip keeps it feasible
    assert pc.parlay_strength(legs, {}, matrix=base2(0.0))["lift"] == pytest.approx(1.0)
    tight = pc.parlay_strength([qb_over(prob=0.9), wr_over(prob=0.9)], {}, matrix=base2(1.0))
    assert tight["lift"] == pytest.approx(0.9 / 0.81, abs=1e-3)


# ---------------------------------------------------------------------------
# flag_parlay / rank_parlays
# ---------------------------------------------------------------------------
def test_flag_parlay_is_the_flags_and_reasons_of_parlay_strength():
    legs = [qb_over(), wr_over()]
    full = pc.parlay_strength(legs, {"KC": ADVERSE}, matrix=base2(0.45))
    thin = pc.flag_parlay(legs, {"KC": ADVERSE}, matrix=base2(0.45))
    assert thin["flags"] == full["flags"] and thin["reasons"] == full["reasons"]


def _candidates():
    good = [qb_over(), wr_over()]
    bad = [qb_over(team="KC", name="Bad QB"), wr_over(team="KC", name="Bad WR")]
    far = [prop("receiving_yards", position="WR", team="NYJ", home="MIA", away="NYJ", game="MIA-NYJ", name="Far WR"),
           prop("rushing_yards", position="RB", team="NYJ", home="MIA", away="NYJ", game="MIA-NYJ", name="Far RB",
                line=55.5)]
    return good, bad, far


def test_rank_orders_by_score_with_stable_ties():
    good, bad, far = _candidates()
    signals = {"KC": FAVOURABLE}
    ranked = pc.rank_parlays([far, bad, good], signals)
    # candidate 1 (bad) uses the same KC bundle as good, so both are favourable; far has no signal (50.0)
    assert [row["index"] for row in ranked] == [1, 2, 0]
    assert [row["rank"] for row in ranked] == [1, 2, 3]
    assert ranked[0]["score"] == ranked[1]["score"] > ranked[2]["score"] == 50.0
    # a clearly worse ticket sinks
    adverse_signals = {"KC": ADVERSE, "NYJ": FAVOURABLE}
    ranked = pc.rank_parlays([good, far], adverse_signals)
    assert ranked[0]["index"] == 1
    assert ranked[0]["flags"]["trap"] is False and ranked[1]["flags"]["trap"] is True


def test_rank_uses_the_evaluator_hook_for_lift_and_breaks_ties_on_ev():
    good, bad, _ = _candidates()
    calls = []

    def evaluator(legs):
        calls.append(len(legs))
        return {"adjusted_probability": 0.30, "naive_probability": 0.25,
                "ev_adjusted": 0.05 if legs[0]["player_name"] == "Bad QB" else 0.02}

    ranked = pc.rank_parlays([good, bad], {"KC": FAVOURABLE}, evaluator=evaluator)
    assert calls == [2, 2]
    assert ranked[0]["lift"] == pytest.approx(1.2) == ranked[1]["lift"]
    assert ranked[0]["score"] == ranked[1]["score"]
    assert [row["index"] for row in ranked] == [1, 0]           # tie on score -> higher EV first
    assert ranked[0]["ev_adjusted"] == 0.05 and ranked[0]["adjusted_probability"] == 0.30


def test_rank_survives_a_bad_evaluator_and_bad_candidates():
    good, _, _ = _candidates()

    def broken(legs):
        raise RuntimeError("boom")

    ranked = pc.rank_parlays([good, None, "x", []], {"KC": FAVOURABLE}, evaluator=broken)
    assert len(ranked) == 4
    assert ranked[0]["index"] == 0
    assert ranked[0]["evaluator_error"].startswith("RuntimeError")
    assert pc.rank_parlays(None, {}) == []


# ---------------------------------------------------------------------------
# Robustness, determinism, honesty, imports
# ---------------------------------------------------------------------------
BAD_LEGS = [
    None, {}, "leg", 7, float("nan"),
    {"market": float("nan"), "side": None, "team": float("nan"), "game_id": "nan"},
    {"market": "passing_yards", "side": "over", "team": None, "position": "QB"},
    {"market": "receiving_yards", "side": "sideways", "team": "KC", "position": "WR", "line": "n/a"},
    {"leg_type": "prop", "market": 12, "side": 3, "team": ["KC"], "probability": "high"},
]
BAD_SIGNALS = [
    None, {}, [], "x", {"KC": None}, {"KC": "bad"}, {"KC": {"probabilities": None}}, {None: ADVERSE},
    {"KC": {"probabilities": {"collapse_probability": float("nan"), "explosion_probability": "x",
                              "positive_script_prob": None}, "has_signal": True}},
    {"kc": ADVERSE},   # lower-case key is normalised
]


@pytest.mark.parametrize("signals", BAD_SIGNALS)
def test_bad_signals_never_raise(signals):
    legs = [qb_over(), wr_over()]
    out = pc.build_correlation_matrix(legs, signals, base_matrix=base2())
    assert len(out["matrix"]) == 2
    assert pc.parlay_strength(legs, signals)["score"] <= 100.0
    pc.pair_correlation(legs[0], legs[1], signals)
    pc.rank_parlays([legs], signals)


def test_bad_legs_never_raise_and_have_no_opinion():
    out = pc.build_correlation_matrix(BAD_LEGS, {"KC": ADVERSE})
    n = len(BAD_LEGS)
    assert len(out["matrix"]) == n and all(len(row) == n for row in out["matrix"])
    assert all(out["matrix"][i][i] == 1.0 for i in range(n))
    # junk legs never receive a pack correlation
    for pair in out["pairs"]:
        if pair["i"] <= 5 or pair["j"] <= 5:
            assert pair["pack_rho"] is None
    strength = pc.parlay_strength(BAD_LEGS, {"KC": ADVERSE})
    assert 0.0 <= strength["score"] <= 100.0
    for a in BAD_LEGS:
        for b in BAD_LEGS:
            assert pc.pair_correlation(a, b, {"KC": ADVERSE})["rho"] is None or isinstance(
                pc.pair_correlation(a, b, {"KC": ADVERSE})["rho"], float)
    assert pc.build_correlation_matrix(None, None)["matrix"] == []
    assert pc.build_correlation_matrix("abc", None)["matrix"] == []


def test_bad_base_matrix_is_ignored_with_a_warning_and_nan_entries_read_as_zero():
    legs = [qb_over(), wr_over()]
    wrong = pc.build_correlation_matrix(legs, None, base_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert wrong["matrix"] == [[1.0, 0.0], [0.0, 1.0]] and wrong["warnings"]
    nan = pc.build_correlation_matrix(legs, None, base_matrix=[[1.0, float("nan")], [float("nan"), 1.0]])
    assert nan["matrix"] == [[1.0, 0.0], [0.0, 1.0]] and nan["warnings"]
    words = pc.build_correlation_matrix(legs, None, base_matrix=[["a", "b"], ["c", "d"]])
    assert words["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    # the builder's non-symmetric garbage is read from the upper triangle
    asym = pc.build_correlation_matrix(legs, None, base_matrix=[[1.0, 0.3], [0.9, 1.0]])
    assert asym["matrix"][0][1] == asym["matrix"][1][0] == 0.3


def test_inputs_are_never_mutated():
    legs = [qb_over(), wr_over(), team_leg("moneyline")]
    signals = {"KC": copy.deepcopy(ADVERSE), "BUF": copy.deepcopy(FAVOURABLE)}
    base = [[1.0, 0.45, 0.2], [0.45, 1.0, 0.1], [0.2, 0.1, 1.0]]
    snap = copy.deepcopy((legs, signals, base))
    pc.build_correlation_matrix(legs, signals, base_matrix=base)
    pc.parlay_strength(legs, signals, matrix=base)
    pc.rank_parlays([legs], signals)
    pc.pair_correlation(legs[0], legs[1], signals)
    pc.regime_state(signals["KC"])
    assert (legs, signals, base) == snap


def test_everything_is_deterministic():
    legs = [qb_over(), wr_over(), prop("rushing_yards", position="RB", name="Back", line=60.5)]
    signals = {"KC": ADVERSE}
    base = [[1.0, 0.45, -0.08], [0.45, 1.0, -0.05], [-0.08, -0.05, 1.0]]
    first = (pc.build_correlation_matrix(legs, signals, base_matrix=base), pc.parlay_strength(legs, signals, matrix=base),
             pc.rank_parlays([legs, legs[:2]], signals))
    second = (pc.build_correlation_matrix(legs, signals, base_matrix=base), pc.parlay_strength(legs, signals, matrix=base),
              pc.rank_parlays([legs, legs[:2]], signals))
    assert first == second


def test_honesty_strings_and_disclosure():
    assert "PRIOR" in pc.MODEL_BASIS and "not" in pc.MODEL_BASIS.lower()
    doc = pc.__doc__ or ""
    assert "PRIORS, NOT FITS" in doc and "HONEST LIMITS" in doc
    info = pc.disclosure()
    assert set(info) >= {"basis", "regimes", "loadings", "idiosyncratic", "thresholds", "limits"}
    assert info["loadings"]["qb_passing_yards"]["qb_collapse"] == -0.65
    assert info["thresholds"]["default_pack_weight"] == pc.DEFAULT_PACK_WEIGHT
    pair = pc.pair_correlation(qb_over(), wr_over(), {"KC": ADVERSE})
    assert "Prior loadings, not fitted" in pair["note"]
    strength = pc.parlay_strength([qb_over(), wr_over()], {"KC": ADVERSE})
    assert "PRIOR" in strength["basis"] and "Lift" in strength["basis"]


def test_module_imports_only_pure_logic_dependencies():
    source = Path(pc.__file__).read_text(encoding="utf-8")
    modules: set[str] = set()      # what each import statement pulls from
    names: set[str] = set()        # every imported name, so `from modules import x` is seen too
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
            names |= {alias.name.split(".")[-1] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            names |= {alias.name for alias in node.names}
    assert modules <= {"__future__", "math", "statistics", "typing", "numpy", "modules.nfl_pack_common"}
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4", "nfl_parlay_builder"}
    assert not (banned & (modules | names))


# ---------------------------------------------------------------------------
# Integration with the sibling modules (offline)
# ---------------------------------------------------------------------------
def test_integration_with_pipeline_team_signals():
    from modules import nfl_upgrade_pipeline as pipe

    legs = [qb_over(), wr_over()]
    neutral = pipe.team_signals({})
    out = pc.build_correlation_matrix(legs, {"KC": neutral}, base_matrix=base2(0.45))
    assert out["matrix"] == base2(0.45)  # no matchup features -> the rule table is untouched
    assert pc.parlay_strength(legs, {"KC": neutral}, matrix=base2(0.45))["score"] == 50.0

    hot = pipe.team_signals({"opp_pressure_rate": 0.45, "opp_blitz_rate": 0.40, "ol_injuries": 3.0,
                             "opp_pass_rush_win_rate": 0.18, "vegas_spread": 9.0, "vegas_total": 40.0})
    calm = pipe.team_signals({"opp_pressure_rate": 0.18, "opp_blitz_rate": 0.15, "ol_injuries": 0.0,
                             "opp_pass_rush_win_rate": 0.06, "vegas_spread": -3.0, "vegas_total": 50.0})
    s_hot = pc.parlay_strength(legs, {"KC": hot}, matrix=base2(0.45))
    s_calm = pc.parlay_strength(legs, {"KC": calm}, matrix=base2(0.45))
    assert s_hot["has_signal"] and s_calm["has_signal"]
    assert s_hot["alignment"] < s_calm["alignment"]
    assert s_hot["score"] < s_calm["score"]
    assert s_hot["fragility"] > s_calm["fragility"]


def test_integration_with_builder_leg_shape():
    builder = pytest.importorskip("modules.nfl_parlay_builder")
    qb = builder.make_parlay_leg("prop", description="QB over 265.5 passing yards", probability=0.52, price=-110,
                                 game_id="2026_06_BUF_KC", team="KC", opponent="BUF", home_team="KC", away_team="BUF",
                                 side="over", line=265.5, market="passing_yards", player_id="qb1",
                                 player_name="Quarterback", position="QB")
    wr = builder.make_parlay_leg("prop", description="WR over 68.5 receiving yards", probability=0.5, price=-115,
                                 game_id="2026_06_BUF_KC", team="KC", opponent="BUF", home_team="KC", away_team="BUF",
                                 side="over", line=68.5, market="receiving_yards", player_id="wr1",
                                 player_name="Receiver", position="WR")
    ml = builder.make_parlay_leg("moneyline", description="KC moneyline", probability=0.6, price=-150,
                                 game_id="2026_06_BUF_KC", team="KC", home_team="KC", away_team="BUF")
    assert pc.market_family(qb) == "qb_passing_yards"
    assert pc.market_family(wr) == "wr_receiving_yards"
    assert pc.market_family(ml) == "team_result"
    legs = [qb, wr, ml]
    base = builder.correlation_matrix(legs)
    out = pc.build_correlation_matrix(legs, {"KC": ADVERSE, "BUF": FAVOURABLE}, base_matrix=base)
    assert out["pack_pairs"] >= 1
    assert all(out["matrix"][i][j] == out["matrix"][j][i] for i in range(3) for j in range(3))
    strength = pc.parlay_strength(legs, {"KC": ADVERSE, "BUF": FAVOURABLE}, matrix=out["matrix"], base_matrix=base)
    assert strength["has_signal"] is True and 0.0 <= strength["score"] <= 100.0


def test_a_league_average_game_leaves_the_rule_table_exactly_and_deviations_move_it_symmetrically():
    base = base2(0.45)
    average = pc.build_correlation_matrix([qb_over(), wr_over()], {"KC": bundle()}, base_matrix=base)
    assert average["matrix"] == base                      # no level bias: the pack adds only game-specific information
    worse = pc.build_correlation_matrix([qb_over(), wr_over()], {"KC": ADVERSE}, base_matrix=base)["matrix"][0][1]
    better = pc.build_correlation_matrix([qb_over(), wr_over()], {"KC": FAVOURABLE}, base_matrix=base)["matrix"][0][1]
    assert worse < 0.45 < better or better < 0.45 < worse   # opposite deviations move the link in opposite directions
    assert abs(worse - 0.45) < 0.2 and abs(better - 0.45) < 0.2   # evidence, not a replacement
