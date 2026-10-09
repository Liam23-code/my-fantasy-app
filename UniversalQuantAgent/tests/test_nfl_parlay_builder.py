"""Contract for modules.nfl_parlay_builder -- the leg / correlation / combined-probability
engine behind the NFL Parlays page.

Everything here is offline and built from hand-made legs. The numeric tests do not trust the
module's own copula: the bivariate probability is checked against an independent Simpson
integration (a different formulation) and against numpy sampling written in the test, the ML/spread
joint against a simulation of the very margin model those legs are derived from, and the
three-leg Monte-Carlo path against the exact orthant formula.
"""

import math
import random
import sys
import time
import unittest
from itertools import combinations
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from betting.odds_math import american_to_decimal, decimal_to_american
from betting.parlay_engine import (
    _parlay_ev,
    _risk_tier,
    correlation_adjusted_probability,
    detect_correlations,
    make_leg as engine_make_leg,
    naive_joint_probability,
    parlay_decimal_odds,
)

from modules import nfl_parlay_builder as parlay_module
from modules.nfl_parlay_builder import (
    CONTRADICTORY_RHO,
    CORRELATION_KINDS,
    LEG_TYPES,
    MAX_ABS_BOOK_PRICE,
    MAX_ABS_PRICE,
    MAX_ABS_RHO,
    MAX_PARLAY_LEGS,
    MAX_PROP_LINE,
    MAX_STAKE,
    bivariate_normal_cdf,
    can_add_leg,
    correlation_matrix,
    correlation_rule_table,
    evaluate_parlay_legs,
    joint_probability,
    joint_probability_detail,
    leg_correlation,
    legs_conflict,
    make_parlay_leg,
    nearest_correlation_matrix,
    pairwise_findings,
)

GAME = "2026_05_BUF_KC"
OTHER_GAME = "2026_05_DAL_NYG"
TIER_RANK = {"Low": 0, "Medium": 1, "High": 2}
STRENGTHS = {"None", "Weak", "Moderate", "Strong"}
DIRECTIONS = {"positive", "negative", "contradictory", "none"}


def ml(team="KC", *, p=0.6, price=-150, game=GAME, home="KC", away="BUF", **extra):
    return make_parlay_leg(
        "moneyline", description=f"{team} moneyline", probability=p, price=price, game_id=game,
        team=team, home_team=home, away_team=away, **extra,
    )


def spread(team="KC", line=-3.5, *, p=0.5, price=-110, game=GAME, home="KC", away="BUF", **extra):
    return make_parlay_leg(
        "spread", description=f"{team} {line:+g}", probability=p, price=price, game_id=game,
        team=team, home_team=home, away_team=away, line=line, **extra,
    )


def prop(player, market, side="over", line=60.5, *, team="KC", p=0.5, price=-110, game=GAME,
         player_id=None, **extra):
    return make_parlay_leg(
        "prop", description=f"{player} {side} {line:g} {market}", probability=p, price=price,
        game_id=game, team=team, market=market, side=side, line=line,
        player_id=player_id or player.lower().replace(" ", "_"), player_name=player, **extra,
    )


def independent_legs(probabilities, price=-110):
    """One leg per game, so the correlation layer treats them as independent."""
    return [
        make_parlay_leg(
            "moneyline", description=f"Team{i} moneyline", probability=p, price=price,
            game_id=f"G{i}", team=f"T{i:02d}",
        )
        for i, p in enumerate(probabilities)
    ]


def assert_all_finite(testcase, value, path="result", allowed_none=()):
    """Recursively assert no NaN/inf (and no None outside ``allowed_none`` keys)."""
    if isinstance(value, dict):
        for key, item in value.items():
            if item is None:
                testcase.assertIn(key, allowed_none, f"unexpected None at {path}.{key}")
                continue
            assert_all_finite(testcase, item, f"{path}.{key}", allowed_none)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_all_finite(testcase, item, f"{path}[{index}]", allowed_none)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        testcase.assertTrue(math.isfinite(value), f"non-finite number at {path}: {value!r}")


# --------------------------------------------------------------------------- #
# Independent reference implementations used only by the tests
# --------------------------------------------------------------------------- #


def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def brute_force_bivariate(h, k, rho, steps=4000):
    """P(Z1<=h, Z2<=k) = integral of phi(x) * Phi((k - rho x)/sqrt(1-rho^2)) dx, by Simpson's rule."""
    lo = -9.0
    hi = h
    if hi <= lo:
        return 0.0
    n = steps if steps % 2 == 0 else steps + 1
    width = (hi - lo) / n
    scale = math.sqrt(1.0 - rho * rho)

    def f(x):
        return _norm_pdf(x) * _phi((k - rho * x) / scale)

    total = f(lo) + f(hi)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(lo + i * width)
    return total * width / 3.0


class MakeParlayLegTests(unittest.TestCase):
    def test_moneyline_leg_derives_side_opponent_market_and_engine_keys(self):
        leg = ml("KC", p=0.62, price=-150)
        self.assertEqual(leg["leg_type"], "moneyline")
        self.assertEqual((leg["team"], leg["opponent"], leg["side"], leg["market"]), ("KC", "BUF", "home", "moneyline"))
        self.assertEqual((leg["home_team"], leg["away_team"]), ("KC", "BUF"))
        self.assertIsNone(leg["line"])
        self.assertEqual(leg["model_probability"], 0.62)
        self.assertEqual(leg["probability"], 0.62)
        self.assertAlmostEqual(leg["decimal_odds"], american_to_decimal(-150))
        self.assertAlmostEqual(leg["implied_probability"], 1 / american_to_decimal(-150))
        engine = engine_make_leg(
            description="KC moneyline", model_probability=0.62, price=-150, team="KC", game_id=GAME,
            market="moneyline", side="home", confidence=0.7,
        )
        for key, value in engine.items():
            self.assertEqual(leg[key], value, key)

    def test_moneyline_from_side_and_game_teams(self):
        leg = make_parlay_leg(
            "moneyline", description="BUF ML", probability=0.4, price=130, game_id=GAME,
            side="away", home_team="kc", away_team="buf",
        )
        self.assertEqual((leg["team"], leg["opponent"], leg["side"]), ("BUF", "KC", "away"))

    def test_team_only_infers_side_from_home_team(self):
        leg = make_parlay_leg("moneyline", description="BUF ML", probability=0.4, price=130, team="BUF", home_team="KC")
        self.assertEqual((leg["side"], leg["opponent"]), ("away", "KC"))

    def test_spread_leg_needs_line_and_keeps_it(self):
        leg = spread("BUF", 3.5, p=0.51)
        self.assertEqual((leg["market"], leg["line"], leg["side"], leg["leg_type"]), ("spread", 3.5, "away", "spread"))
        with self.assertRaisesRegex(ValueError, "line"):
            make_parlay_leg("spread", description="x", probability=0.5, price=-110, team="KC")
        with self.assertRaisesRegex(ValueError, "60"):
            spread("KC", -75)

    def test_prop_leg_identity_and_defaults(self):
        leg = prop("Patrick Mahomes", "passing_yards", "over", 274.5, p=0.52, price=-115, position="qb")
        self.assertEqual((leg["market"], leg["side"], leg["line"], leg["position"]), ("passing_yards", "over", 274.5, "QB"))
        self.assertEqual((leg["team"], leg["player_id"], leg["player_name"]), ("KC", "patrick_mahomes", "Patrick Mahomes"))
        self.assertEqual(leg["confidence"], 0.7)

    def test_prop_player_identity_falls_back_to_normalised_name(self):
        leg = make_parlay_leg(
            "prop", description="x", probability=0.5, price=-110, market="Receiving Yards", side="OVER",
            line=60.5, player_name="Travis Kelce", team="KC",
        )
        self.assertEqual(leg["player_id"], "name:travis_kelce")
        self.assertEqual((leg["market"], leg["side"]), ("receiving_yards", "over"))

    def test_leg_id_is_stable_readable_and_excludes_price(self):
        a = prop("Travis Kelce", "receiving_yards", "over", 64.5, price=-110)
        b = prop("Travis Kelce", "receiving_yards", "over", 64.5, price=-125, p=0.55)
        self.assertEqual(a["leg_id"], b["leg_id"])
        self.assertEqual(a["leg_id"], "prop:2026_05_BUF_KC:KC:travis_kelce:receiving_yards:over:64.5")
        self.assertNotEqual(a["leg_id"], prop("Travis Kelce", "receiving_yards", "over", 69.5)["leg_id"])
        self.assertNotEqual(a["leg_id"], prop("Travis Kelce", "receiving_yards", "under", 64.5)["leg_id"])
        self.assertNotEqual(ml("KC")["leg_id"], ml("BUF")["leg_id"])
        self.assertNotEqual(ml("KC")["leg_id"], spread("KC", -3.5)["leg_id"])
        self.assertRegex(a["leg_id"], r"^[A-Za-z0-9._:+-]+$")

    def test_validation_matrix(self):
        base = dict(description="x", probability=0.5, price=-110, team="KC")
        bad = [
            ({"leg_type": "totals"}, "Leg type"),
            ({"leg_type": None}, "Leg type"),
            ({"leg_type": "moneyline", "description": "  "}, "description"),
            ({"leg_type": "moneyline", "probability": 0.0}, "probability"),
            ({"leg_type": "moneyline", "probability": 1.0}, "probability"),
            ({"leg_type": "moneyline", "probability": 1.2}, "probability"),
            ({"leg_type": "moneyline", "probability": -0.1}, "probability"),
            ({"leg_type": "moneyline", "probability": float("nan")}, "finite"),
            ({"leg_type": "moneyline", "probability": "abc"}, "number"),
            ({"leg_type": "moneyline", "probability": True}, "number"),
            ({"leg_type": "moneyline", "price": 50}, "100"),
            ({"leg_type": "moneyline", "price": -99}, "100"),
            ({"leg_type": "moneyline", "price": 0}, "100"),
            ({"leg_type": "moneyline", "price": float("inf")}, "finite"),
            ({"leg_type": "moneyline", "price": None}, "number"),
            ({"leg_type": "moneyline", "confidence": 1.5}, "Confidence"),
            ({"leg_type": "moneyline", "confidence": float("nan")}, "finite"),
            ({"leg_type": "moneyline", "side": "over"}, "home"),
            ({"leg_type": "moneyline", "team": None}, "team"),
            ({"leg_type": "moneyline", "home_team": "KC", "away_team": "KC"}, "different"),
            ({"leg_type": "moneyline", "home_team": "DAL", "away_team": "NYG"}, "not playing"),
            ({"leg_type": "moneyline", "home_team": "KC", "away_team": "BUF", "side": "away"}, "does not match"),
            ({"leg_type": "moneyline", "home_team": "KC", "away_team": "BUF", "opponent": "DAL"}, "Opponent"),
            ({"leg_type": "spread", "line": float("nan")}, "finite"),
            ({"leg_type": "prop", "market": None, "side": "over", "player_id": "p"}, "market"),
            ({"leg_type": "prop", "market": "moneyline", "side": "over", "player_id": "p"}, "game markets"),
            ({"leg_type": "prop", "market": "passing_yards", "side": "home", "player_id": "p"}, "over"),
            ({"leg_type": "prop", "market": "passing_yards", "side": None, "player_id": "p"}, "over"),
            ({"leg_type": "prop", "market": "passing_yards", "side": "over"}, "player"),
            ({"leg_type": "prop", "market": "passing_yards", "side": "over", "player_id": "p", "line": -1}, "negative"),
        ]
        for overrides, message in bad:
            with self.subTest(overrides=overrides):
                kwargs = {**base, **overrides}
                leg_type = kwargs.pop("leg_type")
                with self.assertRaisesRegex(ValueError, message):
                    make_parlay_leg(leg_type, **kwargs)

    def test_each_leg_type_builds(self):
        self.assertEqual(LEG_TYPES, ("moneyline", "spread", "prop"))
        for leg in (ml(), spread(), prop("A Player", "receptions", "over", 5.5)):
            self.assertIn(leg["leg_type"], LEG_TYPES)
            self.assertTrue(leg["leg_id"])

    def test_moneyline_ignores_a_stray_line(self):
        leg = make_parlay_leg("moneyline", description="x", probability=0.5, price=-110, team="KC", line=-3.5)
        self.assertIsNone(leg["line"])


class LegCorrelationTests(unittest.TestCase):
    """Each documented rule fires with its exact kind string."""

    def assertRule(self, a, b, kind, direction, rho=None, strength=None):
        forward, backward = leg_correlation(a, b), leg_correlation(b, a)
        self.assertEqual(forward, backward, "leg_correlation must be symmetric")
        self.assertEqual(forward["kind"], kind, forward["note"])
        self.assertEqual(forward["direction"], direction, forward["note"])
        self.assertIn(forward["kind"], CORRELATION_KINDS)
        self.assertIn(forward["strength"], STRENGTHS)
        self.assertGreaterEqual(forward["rho"], -MAX_ABS_RHO)
        self.assertLessEqual(forward["rho"], MAX_ABS_RHO)
        self.assertTrue(forward["note"])
        if rho is not None:
            self.assertAlmostEqual(forward["rho"], rho, places=4, msg=forward["note"])
        if strength is not None:
            self.assertEqual(forward["strength"], strength)
        return forward

    def scenarios(self):
        """(kind, a, b) for every kind in the rule table."""
        mahomes = prop("Patrick Mahomes", "passing_yards", "over", 274.5, player_id="qb1")
        kelce_yards = prop("Travis Kelce", "receiving_yards", "over", 64.5, player_id="te1")
        kelce_catches = prop("Travis Kelce", "receptions", "over", 5.5, player_id="te1")
        kelce_td = prop("Travis Kelce", "receiving_tds", "over", 0.5, player_id="te1")
        return [
            ("different_games", ml("KC"), ml("DAL", game=OTHER_GAME, home="NYG", away="DAL")),
            ("duplicate_leg", ml("KC"), ml("KC", price=-140)),
            ("same_team_ml_and_spread_favorite", ml("KC"), spread("KC", -3.5)),
            ("same_team_ml_and_spread_underdog", ml("BUF", p=0.4), spread("BUF", 3.5)),
            ("same_team_spread_ladder", spread("KC", -3.5), spread("KC", -7.5, p=0.4)),
            ("opposing_moneylines", ml("KC"), ml("BUF", p=0.4)),
            ("opposing_ml_and_spread_exclusive", ml("KC"), spread("BUF", -1.5)),
            ("opposing_ml_and_spread_overlap", ml("KC"), spread("BUF", 3.5)),
            ("opposing_spreads_exclusive", spread("KC", -3.5), spread("BUF", 3.5)),
            ("opposing_spreads_middle", spread("KC", -3.5), spread("BUF", 7.5)),
            ("team_result_and_rush_script", ml("KC"), prop("Isiah Pacheco", "rushing_yards", "over", 64.5)),
            ("team_result_and_player_scoring", ml("KC"), prop("Patrick Mahomes", "passing_tds", "over", 1.5)),
            ("team_result_and_player_volume", ml("KC"), prop("Patrick Mahomes", "passing_yards", "over", 274.5)),
            ("team_result_and_opposing_player", ml("KC"), prop("Josh Allen", "passing_tds", "over", 1.5, team="BUF")),
            ("same_prop_over_and_under", prop("Travis Kelce", "receiving_yards", "over", 64.5),
             prop("Travis Kelce", "receiving_yards", "under", 64.5)),
            ("same_prop_middle", prop("Travis Kelce", "receiving_yards", "over", 54.5),
             prop("Travis Kelce", "receiving_yards", "under", 74.5)),
            ("same_prop_ladder", prop("Travis Kelce", "receiving_yards", "over", 54.5),
             prop("Travis Kelce", "receiving_yards", "over", 74.5, p=0.3)),
            ("same_player_catches_and_yards", kelce_yards, kelce_catches),
            ("same_player_receiving_volume_and_score", kelce_yards, kelce_td),
            ("same_player_passing_volume_and_score", mahomes, prop("Patrick Mahomes", "passing_tds", "over", 1.5, player_id="qb1")),
            ("same_player_td_overlap", prop("Isiah Pacheco", "anytime_td", "over", 0.5, player_id="rb1"),
             prop("Isiah Pacheco", "rushing_tds", "over", 0.5, player_id="rb1")),
            ("rb_volume_and_touchdown", prop("Isiah Pacheco", "rushing_yards", "over", 64.5, player_id="rb1"),
             prop("Isiah Pacheco", "rushing_tds", "over", 0.5, player_id="rb1")),
            ("same_player_multi_market", prop("Patrick Mahomes", "passing_yards", "over", 274.5, player_id="qb1"),
             prop("Patrick Mahomes", "rushing_yards", "over", 14.5, player_id="qb1")),
            ("qb_pass_catcher_stack", mahomes, kelce_yards),
            ("run_pass_tradeoff", mahomes, prop("Isiah Pacheco", "rushing_yards", "over", 64.5)),
            ("teammate_backs_split", prop("Isiah Pacheco", "rushing_yards", "over", 64.5),
             prop("Kareem Hunt", "rushing_yards", "over", 30.5)),
            ("teammate_pass_catchers", kelce_yards, prop("Rashee Rice", "receiving_yards", "over", 70.5)),
            ("same_team_offense", mahomes, prop("Patrick Mahomes", "passing_tds", "over", 1.5, player_id="qb9")),
            ("opposing_qbs_shootout", mahomes, prop("Josh Allen", "passing_yards", "over", 259.5, team="BUF")),
            ("opposing_passing_games", mahomes, prop("Stefon Diggs", "receiving_yards", "over", 70.5, team="BUF")),
            ("opposing_scoring_shootout", prop("Isiah Pacheco", "rushing_tds", "over", 0.5),
             prop("Josh Allen", "passing_tds", "over", 1.5, team="BUF")),
            ("same_game_unrelated", prop("Isiah Pacheco", "rushing_yards", "over", 64.5),
             prop("James Cook", "rushing_yards", "over", 60.5, team="BUF")),
        ]

    def test_every_documented_kind_is_produced_with_the_right_kind_string(self):
        seen = set()
        for kind, a, b in self.scenarios():
            with self.subTest(kind=kind):
                found = leg_correlation(a, b)
                self.assertEqual(found["kind"], kind, found["note"])
                self.assertEqual(found, leg_correlation(b, a))
                seen.add(found["kind"])
        self.assertEqual(seen, set(CORRELATION_KINDS), "the rule table and the rules must stay in sync")

    def test_result_shape_range_and_vocabulary_for_all_scenarios(self):
        for kind, a, b in self.scenarios():
            with self.subTest(kind=kind):
                found = leg_correlation(a, b)
                self.assertEqual(set(found), {"rho", "kind", "direction", "strength", "note", "copula_rho"})
                self.assertGreaterEqual(found["rho"], -MAX_ABS_RHO)
                self.assertLessEqual(found["rho"], MAX_ABS_RHO)
                self.assertIn(found["direction"], DIRECTIONS)
                self.assertIn(found["strength"], STRENGTHS)
                self.assertIsInstance(found["note"], str)
                self.assertGreater(len(found["note"]), 20)
                if found["direction"] == "positive":
                    self.assertGreater(found["rho"], 0)
                if found["direction"] == "negative":
                    self.assertLess(found["rho"], 0)

    def test_different_games_are_independent(self):
        found = self.assertRule(
            ml("KC"), prop("Dak Prescott", "passing_yards", "over", 260.5, team="DAL", game=OTHER_GAME),
            "different_games", "none", rho=0.0, strength="None",
        )
        self.assertEqual(found["rho"], 0.0)

    def test_team_overlap_identifies_same_game_without_game_ids(self):
        a = make_parlay_leg("moneyline", description="KC ML", probability=0.6, price=-150, team="KC", opponent="BUF")
        b = make_parlay_leg("prop", description="Mahomes o274.5", probability=0.5, price=-110, team="KC",
                            market="passing_yards", side="over", line=274.5, player_id="qb1")
        self.assertEqual(leg_correlation(a, b)["kind"], "team_result_and_player_volume")
        c = make_parlay_leg("prop", description="Dak", probability=0.5, price=-110, team="DAL",
                            market="passing_yards", side="over", line=260.5, player_id="qb7")
        self.assertEqual(leg_correlation(a, c)["kind"], "different_games")

    def test_differing_game_ids_win_over_team_overlap(self):
        a = ml("KC", game="week5")
        b = ml("KC", game="week6")
        self.assertEqual(leg_correlation(a, b)["kind"], "different_games")

    def test_moneyline_and_same_team_spread(self):
        fav = self.assertRule(ml("KC"), spread("KC", -3.5), "same_team_ml_and_spread_favorite", "positive", rho=0.95, strength="Strong")
        self.assertIn("-3.5", fav["note"])
        dog = self.assertRule(ml("BUF", p=0.4), spread("BUF", 3.5), "same_team_ml_and_spread_underdog", "positive", rho=0.95)
        self.assertIn("+3.5", dog["note"])
        # pick'em reads as the favourite case (covering 0 is winning)
        self.assertRule(ml("KC"), spread("KC", 0), "same_team_ml_and_spread_favorite", "positive")

    def test_opposing_moneyline_and_spread_handled_by_spread_sign(self):
        # the opponent laying points can never cover while losing: mutually exclusive
        self.assertRule(ml("KC"), spread("BUF", -1.5), "opposing_ml_and_spread_exclusive", "contradictory")
        self.assertRule(ml("KC"), spread("BUF", 0), "opposing_ml_and_spread_exclusive", "contradictory")
        # the opponent getting points can: a narrow band, strongly negative but not blocked
        found = self.assertRule(ml("KC"), spread("BUF", 3.5), "opposing_ml_and_spread_overlap", "negative", rho=-0.95)
        self.assertIn("3.5", found["note"])

    def test_two_spreads_one_game(self):
        self.assertRule(spread("KC", -3.5), spread("BUF", 3.5), "opposing_spreads_exclusive", "contradictory")
        self.assertRule(spread("KC", -7.5), spread("BUF", 3.5), "opposing_spreads_exclusive", "contradictory")
        found = self.assertRule(spread("KC", -3.5), spread("BUF", 7.5), "opposing_spreads_middle", "negative", rho=-0.95)
        self.assertIn("4-point window", found["note"])
        self.assertRule(spread("KC", -3.5), spread("KC", -7.5, p=0.4), "same_team_spread_ladder", "positive", rho=0.95)

    def test_opposing_moneylines_are_contradictory(self):
        found = self.assertRule(ml("KC"), ml("BUF", p=0.4), "opposing_moneylines", "contradictory", strength="Strong")
        self.assertEqual(found["rho"], -MAX_ABS_RHO)

    def test_qb_and_pass_catcher_stack_delegates_to_the_engine_pattern(self):
        qb = prop("Patrick Mahomes", "passing_yards", "over", 274.5, player_id="qb1")
        wr = prop("Travis Kelce", "receiving_yards", "over", 64.5, player_id="te1")
        found = self.assertRule(qb, wr, "qb_pass_catcher_stack", "positive", rho=0.45, strength="Moderate")
        # the existing engine independently sees the same pattern on the same legs
        engine_findings = detect_correlations([qb, wr])
        self.assertEqual([f["kind"] for f in engine_findings], ["qb_pass_catcher_stack"])
        self.assertEqual(found["direction"], engine_findings[0]["direction"])
        for qb_market, catcher_market, rho in (
            ("passing_yards", "receptions", 0.40), ("passing_yards", "receiving_tds", 0.25),
            ("passing_tds", "receiving_tds", 0.35), ("passing_tds", "receiving_yards", 0.30),
            ("passing_tds", "receptions", 0.20),
        ):
            with self.subTest(markets=(qb_market, catcher_market)):
                self.assertRule(
                    prop("Patrick Mahomes", qb_market, "over", 1.5, player_id="qb1"),
                    prop("Travis Kelce", catcher_market, "over", 0.5, player_id="te1"),
                    "qb_pass_catcher_stack", "positive", rho=rho,
                )

    def test_under_under_stack_is_positive_and_mixed_sides_are_negative(self):
        qb_u = prop("Patrick Mahomes", "passing_yards", "under", 274.5, player_id="qb1")
        wr_u = prop("Travis Kelce", "receiving_yards", "under", 64.5, player_id="te1")
        wr_o = prop("Travis Kelce", "receiving_yards", "over", 64.5, player_id="te1")
        self.assertEqual(detect_correlations([qb_u, wr_u]), [], "engine ignores unders; the builder must not")
        self.assertRule(qb_u, wr_u, "qb_pass_catcher_stack", "positive", rho=0.45)
        self.assertRule(qb_u, wr_o, "qb_pass_catcher_stack", "negative", rho=-0.45)

    def test_stack_needs_the_same_team(self):
        qb = prop("Patrick Mahomes", "passing_yards", "over", 274.5, player_id="qb1")
        rival_wr = prop("Stefon Diggs", "receiving_yards", "over", 70.5, team="BUF", player_id="wr9")
        found = leg_correlation(qb, rival_wr)
        self.assertEqual(found["kind"], "opposing_passing_games")
        self.assertLess(found["rho"], 0.2)

    def test_rb_rushing_and_team_result_game_script(self):
        rb = prop("Isiah Pacheco", "rushing_yards", "over", 64.5)
        fav_ml = self.assertRule(rb, ml("KC", p=0.62), "team_result_and_rush_script", "positive", rho=0.22)
        self.assertIn("favourite", fav_ml["note"])
        self.assertRule(rb, spread("KC", -6.5), "team_result_and_rush_script", "positive", rho=0.22)
        # an underdog or a +spread cover still helps a back, but less
        self.assertRule(rb, ml("KC", p=0.40), "team_result_and_rush_script", "positive", rho=0.12)
        self.assertRule(rb, spread("KC", 3.5), "team_result_and_rush_script", "positive", rho=0.12)
        # backing the other team is against the back; the damping depends on the back's own team being the favourite
        opp_vs_underdog = self.assertRule(rb, ml("BUF", p=0.4), "team_result_and_opposing_player", "negative")
        self.assertAlmostEqual(opp_vs_underdog["rho"], -0.22 * 0.6, places=4)
        opp_vs_favorite = self.assertRule(rb, ml("BUF", p=0.6), "team_result_and_opposing_player", "negative")
        self.assertAlmostEqual(opp_vs_favorite["rho"], -0.12 * 0.6, places=4)

    def test_team_result_with_player_scoring_and_volume(self):
        for market, rho in (("passing_tds", 0.28), ("rushing_tds", 0.24), ("receiving_tds", 0.24)):
            with self.subTest(market=market):
                self.assertRule(ml("KC"), prop("Some Player", market, "over", 0.5), "team_result_and_player_scoring", "positive", rho=rho)
        for market, rho in (("passing_yards", 0.08), ("receiving_yards", 0.12), ("receptions", 0.08)):
            with self.subTest(market=market):
                self.assertRule(ml("KC"), prop("Some Player", market, "over", 5.5), "team_result_and_player_volume", "positive", rho=rho)

    def test_under_flips_the_team_result_sign(self):
        over = leg_correlation(ml("KC"), prop("Some Player", "receiving_tds", "over", 0.5))
        under = leg_correlation(ml("KC"), prop("Some Player", "receiving_tds", "under", 0.5))
        self.assertAlmostEqual(under["rho"], -over["rho"], places=4)
        self.assertEqual(under["direction"], "negative")
        self.assertIn("under", under["note"].lower())

    def test_opposing_player_is_negative_and_damped_for_volume_markets(self):
        scoring = leg_correlation(ml("KC"), prop("Josh Allen", "passing_tds", "over", 1.5, team="BUF"))
        volume = leg_correlation(ml("KC"), prop("Josh Allen", "passing_yards", "over", 259.5, team="BUF"))
        self.assertAlmostEqual(scoring["rho"], -0.28 * 0.6, places=4)
        self.assertAlmostEqual(volume["rho"], -0.08 * 0.4, places=4)
        self.assertLess(abs(volume["rho"]), abs(scoring["rho"]))

    def test_same_prop_over_under_depends_on_the_lines(self):
        over = prop("Travis Kelce", "receiving_yards", "over", 64.5)
        under_same = prop("Travis Kelce", "receiving_yards", "under", 64.5)
        under_lower = prop("Travis Kelce", "receiving_yards", "under", 54.5)
        under_higher = prop("Travis Kelce", "receiving_yards", "under", 74.5)
        self.assertRule(over, under_same, "same_prop_over_and_under", "contradictory")
        self.assertRule(over, under_lower, "same_prop_over_and_under", "contradictory")
        self.assertRule(over, under_higher, "same_prop_middle", "negative", rho=-0.95)
        unknown = make_parlay_leg("prop", description="x", probability=0.5, price=-110, team="KC", game_id=GAME,
                                  market="receiving_yards", side="under", player_id="kelce")
        no_line = make_parlay_leg("prop", description="y", probability=0.5, price=-110, team="KC", game_id=GAME,
                                  market="receiving_yards", side="over", player_id="kelce")
        self.assertEqual(leg_correlation(unknown, no_line)["direction"], "contradictory")

    def test_same_prop_ladder_and_duplicate(self):
        low = prop("Travis Kelce", "receiving_yards", "over", 54.5)
        high = prop("Travis Kelce", "receiving_yards", "over", 74.5, p=0.3)
        self.assertRule(low, high, "same_prop_ladder", "positive", rho=0.95)
        self.assertRule(low, prop("Travis Kelce", "receiving_yards", "over", 54.5, price=-130), "duplicate_leg", "positive", rho=0.95)

    def test_same_player_markets(self):
        yards = prop("Travis Kelce", "receiving_yards", "over", 64.5, player_id="te1")
        catches = prop("Travis Kelce", "receptions", "over", 5.5, player_id="te1")
        self.assertRule(yards, catches, "same_player_catches_and_yards", "positive", rho=0.70, strength="Strong")
        td = prop("Travis Kelce", "receiving_tds", "over", 0.5, player_id="te1")
        self.assertRule(yards, td, "same_player_receiving_volume_and_score", "positive", rho=0.35)
        self.assertRule(catches, td, "same_player_receiving_volume_and_score", "positive", rho=0.25)
        # over + under across markets of one player: same magnitude, opposite sign
        self.assertRule(yards, prop("Travis Kelce", "receptions", "under", 5.5, player_id="te1"),
                        "same_player_catches_and_yards", "negative", rho=-0.70)

    def test_rb_volume_and_touchdown_uses_the_engine_pattern(self):
        volume = prop("Isiah Pacheco", "rushing_yards", "over", 64.5, player_id="rb1")
        score = prop("Isiah Pacheco", "rushing_tds", "over", 0.5, player_id="rb1")
        self.assertEqual([f["kind"] for f in detect_correlations([volume, score])], ["rb_volume_and_touchdown"])
        self.assertRule(volume, score, "rb_volume_and_touchdown", "positive", rho=0.40)

    def test_teammate_and_opposing_offence_priors(self):
        rb1 = prop("Isiah Pacheco", "rushing_yards", "over", 64.5)
        rb2 = prop("Kareem Hunt", "rushing_yards", "over", 30.5)
        self.assertRule(rb1, rb2, "teammate_backs_split", "negative", rho=-0.10)
        wr1 = prop("Travis Kelce", "receiving_yards", "over", 64.5)
        wr2 = prop("Rashee Rice", "receiving_yards", "over", 70.5)
        self.assertRule(wr1, wr2, "teammate_pass_catchers", "positive", rho=0.06)
        self.assertRule(rb1, wr1, "run_pass_tradeoff", "negative", rho=-0.05)
        qb_a = prop("Patrick Mahomes", "passing_yards", "over", 274.5)
        qb_b = prop("Josh Allen", "passing_yards", "over", 259.5, team="BUF")
        self.assertRule(qb_a, qb_b, "opposing_qbs_shootout", "positive", rho=0.15)
        self.assertRule(prop("Patrick Mahomes", "passing_yards", "under", 274.5), qb_b, "opposing_qbs_shootout", "negative", rho=-0.15)
        self.assertRule(prop("Patrick Mahomes", "passing_yards", "under", 274.5),
                        prop("Josh Allen", "passing_yards", "under", 259.5, team="BUF"),
                        "opposing_qbs_shootout", "positive", rho=0.15)

    def test_unrelated_same_game_players_get_only_a_small_prior(self):
        found = self.assertRule(
            prop("Isiah Pacheco", "rushing_yards", "over", 64.5), prop("James Cook", "rushing_yards", "over", 60.5, team="BUF"),
            "same_game_unrelated", "positive", rho=0.03, strength="Weak",
        )
        self.assertLess(found["rho"], 0.05)

    def test_copula_rho_is_exact_for_structural_pairs_and_equal_to_rho_otherwise(self):
        structural = {
            "duplicate_leg": 1.0, "same_team_ml_and_spread_favorite": 1.0, "same_team_ml_and_spread_underdog": 1.0,
            "same_team_spread_ladder": 1.0, "opposing_ml_and_spread_overlap": -1.0, "opposing_spreads_middle": -1.0,
            "same_prop_middle": -1.0, "same_prop_ladder": 1.0,
        }
        contradictory = {"opposing_moneylines", "opposing_ml_and_spread_exclusive", "opposing_spreads_exclusive",
                         "same_prop_over_and_under"}
        checked = set()
        for kind, a, b in self.scenarios():
            found = leg_correlation(a, b)
            with self.subTest(kind=kind):
                if kind in structural:
                    self.assertEqual(found["copula_rho"], structural[kind])
                    self.assertEqual(abs(found["rho"]), MAX_ABS_RHO)
                elif kind in contradictory:
                    self.assertEqual(found["copula_rho"], CONTRADICTORY_RHO)
                    self.assertEqual(found["direction"], "contradictory")
                else:
                    self.assertEqual(found["copula_rho"], found["rho"])
                    self.assertLessEqual(abs(found["copula_rho"]), MAX_ABS_RHO)
                checked.add(kind)
        self.assertEqual(checked, set(CORRELATION_KINDS))

    def test_malformed_leg_is_a_friendly_error(self):
        with self.assertRaises(ValueError):
            leg_correlation(ml("KC"), "not a leg")

    def test_legs_built_directly_with_the_engine_make_leg_still_correlate(self):
        a = engine_make_leg(description="qb", model_probability=0.5, price=-110, player_id="qb1", team="KC",
                            game_id=GAME, market="passing_yards", side="over")
        b = engine_make_leg(description="wr", model_probability=0.5, price=-110, player_id="te1", team="KC",
                            game_id=GAME, market="receiving_yards", side="over")
        self.assertEqual(leg_correlation(a, b)["kind"], "qb_pass_catcher_stack")

    def test_symmetry_over_every_pair_of_a_mixed_slate(self):
        legs = [a for _, a, _ in self.scenarios()] + [b for _, _, b in self.scenarios()]
        for a, b in combinations(legs, 2):
            self.assertEqual(leg_correlation(a, b), leg_correlation(b, a), (a["leg_id"], b["leg_id"]))


class MatrixAndFindingsTests(unittest.TestCase):
    def setUp(self):
        self.qb = prop("Patrick Mahomes", "passing_yards", "over", 274.5, player_id="qb1")
        self.wr = prop("Travis Kelce", "receiving_yards", "over", 64.5, player_id="te1")
        self.kc = ml("KC")
        self.dal = ml("DAL", game=OTHER_GAME, home="NYG", away="DAL")
        self.legs = [self.qb, self.wr, self.kc, self.dal]

    def test_matrix_is_symmetric_with_a_unit_diagonal(self):
        matrix = correlation_matrix(self.legs)
        self.assertEqual(len(matrix), 4)
        for i in range(4):
            self.assertEqual(len(matrix[i]), 4)
            self.assertEqual(matrix[i][i], 1.0)
            for j in range(4):
                self.assertEqual(matrix[i][j], matrix[j][i])
        self.assertAlmostEqual(matrix[0][1], 0.45)
        self.assertEqual(matrix[0][3], 0.0)
        self.assertEqual(matrix[2][3], 0.0)

    def test_matrix_feeds_joint_probability_exactly_as_the_evaluator_does(self):
        legs = [ml("KC", p=0.62), spread("BUF", 3.5, p=0.5), self.qb, self.wr]
        matrix = correlation_matrix(legs)
        self.assertEqual(matrix[0][1], -1.0, "an ML against the opponent's plus-points spread is exactly opposed")
        probabilities = [leg["model_probability"] for leg in legs]
        direct = joint_probability(probabilities, matrix)
        self.assertAlmostEqual(evaluate_parlay_legs(legs)["adjusted_probability"], direct, places=6)

    def test_contradictory_pair_uses_the_sentinel(self):
        matrix = correlation_matrix([self.kc, ml("BUF", p=0.4)])
        self.assertEqual(matrix[0][1], CONTRADICTORY_RHO)
        self.assertEqual(matrix[1][0], CONTRADICTORY_RHO)

    def test_empty_and_single_leg_matrices(self):
        self.assertEqual(correlation_matrix([]), [])
        self.assertEqual(correlation_matrix([self.kc]), [[1.0]])

    def test_pairwise_findings_shape_and_filtering(self):
        findings = pairwise_findings(self.legs)
        pairs = [f["legs"] for f in findings]
        self.assertIn((0, 1), pairs)
        self.assertNotIn((0, 3), pairs, "independent pairs are omitted by default")
        first = next(f for f in findings if f["legs"] == (0, 1))
        self.assertEqual(first["labels"], (self.qb["description"], self.wr["description"]))
        self.assertEqual(first["kind"], "qb_pass_catcher_stack")
        for key in ("rho", "kind", "direction", "strength", "note", "legs", "labels"):
            self.assertIn(key, first)
        everything = pairwise_findings(self.legs, include_none=True)
        self.assertEqual(len(everything), 6)
        self.assertTrue(all(f["legs"][0] < f["legs"][1] for f in everything))


class JointProbabilityTests(unittest.TestCase):
    def test_two_leg_closed_form_matches_brute_force_integration(self):
        cases = [
            (0.6, 0.3, 0.95), (0.3, 0.7, 0.95), (0.5, 0.5, -0.95), (0.6, 0.7, -0.95), (0.02, 0.03, 0.9),
            (0.97, 0.99, 0.5), (0.5, 0.5, 0.3), (0.9, 0.1, -0.5), (0.45, 0.55, 0.06), (0.62, 0.5, 0.2),
            (0.3, 0.3, 0.45), (0.8, 0.8, -0.2), (0.5, 0.5, 0.0),
        ]
        for p1, p2, rho in cases:
            with self.subTest(p=(p1, p2), rho=rho):
                from statistics import NormalDist
                h, k = NormalDist().inv_cdf(p1), NormalDist().inv_cdf(p2)
                reference = brute_force_bivariate(h, k, rho) if abs(rho) < 1 else None
                got = joint_probability([p1, p2], [[1, rho], [rho, 1]])
                self.assertAlmostEqual(got, reference, delta=1e-3)
                self.assertAlmostEqual(got, reference, delta=1e-6)  # the closed form is in fact far tighter
                self.assertAlmostEqual(bivariate_normal_cdf(h, k, rho), reference, delta=1e-6)

    def test_two_leg_matches_independent_numpy_sampling(self):
        from statistics import NormalDist
        rng = np.random.default_rng(11)
        for p1, p2, rho in ((0.55, 0.45, 0.4), (0.7, 0.3, -0.6), (0.2, 0.25, 0.8)):
            with self.subTest(p=(p1, p2), rho=rho):
                z = rng.standard_normal((400_000, 2))
                x = z[:, 0]
                y = rho * z[:, 0] + math.sqrt(1 - rho * rho) * z[:, 1]
                sampled = float(np.mean((x <= NormalDist().inv_cdf(p1)) & (y <= NormalDist().inv_cdf(p2))))
                self.assertAlmostEqual(joint_probability([p1, p2], [[1, rho], [rho, 1]]), sampled, delta=4e-3)

    def test_independence_reproduces_the_product(self):
        for probabilities in ([0.6, 0.4], [0.7, 0.55, 0.62], [0.5] * 6, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]):
            n = len(probabilities)
            identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
            with self.subTest(n=n):
                self.assertAlmostEqual(joint_probability(probabilities, identity), math.prod(probabilities), places=12)

    def test_independent_blocks_multiply_exactly(self):
        p = [0.6, 0.5, 0.55, 0.4]
        rho_ab, rho_cd = 0.4, -0.3
        matrix = [[1, rho_ab, 0, 0], [rho_ab, 1, 0, 0], [0, 0, 1, rho_cd], [0, 0, rho_cd, 1]]
        expected = joint_probability(p[:2], [[1, rho_ab], [rho_ab, 1]]) * joint_probability(p[2:], [[1, rho_cd], [rho_cd, 1]])
        self.assertAlmostEqual(joint_probability(p, matrix), expected, places=12)
        detail = joint_probability_detail(p, matrix)
        self.assertEqual(detail["method"], "closed_form")
        self.assertEqual(detail["std_error"], 0.0)

    def test_perfect_positive_correlation_approaches_the_smallest_probability(self):
        self.assertAlmostEqual(joint_probability([0.6, 0.3], [[1, 1], [1, 1]]), 0.3, places=9)
        self.assertAlmostEqual(joint_probability([0.6, 0.3], [[1, 0.9999], [0.9999, 1]]), 0.3, delta=0.01)
        self.assertAlmostEqual(joint_probability([0.6, 0.3], [[1, 0.95], [0.95, 1]]), 0.3, delta=0.005)
        # three legs, all nearly comonotone
        matrix = [[1, 0.99, 0.99], [0.99, 1, 0.99], [0.99, 0.99, 1]]
        self.assertAlmostEqual(joint_probability([0.6, 0.5, 0.4], matrix), 0.4, delta=0.01)

    def test_strong_negative_correlation_approaches_the_lower_frechet_bound(self):
        # rho = -1 is exactly countermonotone: the joint probability is max(0, p1 + p2 - 1)
        self.assertAlmostEqual(joint_probability([0.62, 0.5], [[1, -1], [-1, 1]]), 0.12, places=9)
        self.assertEqual(joint_probability([0.4, 0.3], [[1, -1], [-1, 1]]), 0.0)
        # a capped -0.95 is close but visibly looser, which is why structural pairs use the exact limit
        self.assertAlmostEqual(joint_probability([0.62, 0.5], [[1, -0.95], [-0.95, 1]]), 0.12, delta=0.02)
        self.assertGreater(joint_probability([0.62, 0.5], [[1, -0.95], [-0.95, 1]]), 0.12)

    def test_exactly_nested_legs_inside_a_larger_block_stay_nested(self):
        # legs 0 and 1 are the same event at two thresholds (rho = 1); leg 2 is independent of both
        matrix = [[1, 1, 0.3], [1, 1, 0.3], [0.3, 0.3, 1]]
        detail = joint_probability_detail([0.62, 0.5, 0.55], matrix)
        self.assertEqual(detail["method"], "monte_carlo")
        self.assertFalse(detail["psd_adjusted"], "a singular but valid matrix is not an adjustment")
        pair_only = joint_probability([0.5, 0.55], [[1, 0.3], [0.3, 1]])
        self.assertAlmostEqual(detail["probability"], pair_only, delta=0.004)

    def test_correlation_direction_moves_the_probability(self):
        base = joint_probability([0.55, 0.5, 0.6], np.eye(3).tolist())
        positive = joint_probability([0.55, 0.5, 0.6], [[1, 0.4, 0.4], [0.4, 1, 0.4], [0.4, 0.4, 1]])
        self.assertGreater(positive, base)
        negative = joint_probability([0.55, 0.5, 0.6], [[1, -0.3, -0.3], [-0.3, 1, -0.3], [-0.3, -0.3, 1]])
        self.assertLess(negative, base)

    def test_frechet_bounds_hold_on_a_randomised_grid(self):
        rng = random.Random(2026)
        for trial in range(120):
            n = rng.randint(2, 5)
            p = [rng.uniform(0.02, 0.98) for _ in range(n)]
            matrix = [[1.0] * n for _ in range(n)]
            for i, j in combinations(range(n), 2):
                matrix[i][j] = matrix[j][i] = rng.uniform(-0.99, 0.99)  # frequently not PSD on purpose
            value = joint_probability(p, matrix, draws=20_000)
            lower, upper = max(0.0, sum(p) - (n - 1)), min(p)
            with self.subTest(trial=trial, n=n):
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, lower - 1e-12)
                self.assertLessEqual(value, upper + 1e-12)
                self.assertGreaterEqual(value, 0.0)

    def test_frechet_clamp_absorbs_monte_carlo_noise_at_low_draw_counts(self):
        """With very few draws the raw estimate wanders outside the Frechet bounds; the result must not."""
        clamped_cases = 0
        for p in (0.05, 0.2, 0.25, 0.55, 0.6, 0.9, 0.94, 0.98):
            for rho in (1.0, -0.5):
                matrix = [[1, rho, rho], [rho, 1, rho], [rho, rho, 1]]
                detail = joint_probability_detail([p] * 3, matrix, draws=1000)
                with self.subTest(p=p, rho=rho):
                    self.assertGreaterEqual(detail["probability"], detail["lower_bound"] - 1e-12)
                    self.assertLessEqual(detail["probability"], detail["upper_bound"] + 1e-12)
                    if detail["clamped"]:
                        clamped_cases += 1
                        self.assertTrue(
                            math.isclose(detail["probability"], detail["lower_bound"], abs_tol=1e-12)
                            or math.isclose(detail["probability"], detail["upper_bound"], abs_tol=1e-12)
                        )
        self.assertGreater(clamped_cases, 0, "the grid no longer exercises the clamp; pick inputs that do")

    def test_deterministic_across_calls(self):
        p = [0.6, 0.5, 0.45, 0.7]
        matrix = [[1, 0.3, 0.2, 0.1], [0.3, 1, -0.2, 0.0], [0.2, -0.2, 1, 0.4], [0.1, 0.0, 0.4, 1]]
        self.assertEqual(joint_probability(p, matrix), joint_probability(p, matrix))
        self.assertEqual(joint_probability_detail(p, matrix), joint_probability_detail(p, matrix))
        self.assertEqual(joint_probability(p[:2], [[1, 0.3], [0.3, 1]]), joint_probability(p[:2], [[1, 0.3], [0.3, 1]]))

    def test_non_psd_matrix_is_projected_and_still_valid(self):
        bad = [[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]]
        self.assertLess(min(np.linalg.eigvalsh(np.array(bad))), 0)
        fixed = np.array(nearest_correlation_matrix(bad))
        self.assertTrue(np.allclose(np.diag(fixed), 1.0))
        self.assertTrue(np.allclose(fixed, fixed.T))
        self.assertGreater(min(np.linalg.eigvalsh(fixed)), 0)
        detail = joint_probability_detail([0.6, 0.5, 0.55], bad)
        self.assertTrue(detail["psd_adjusted"])
        self.assertGreaterEqual(detail["probability"], detail["lower_bound"])
        self.assertLessEqual(detail["probability"], detail["upper_bound"])
        self.assertTrue(math.isfinite(detail["probability"]))
        # an already valid matrix is not reported as adjusted
        self.assertFalse(joint_probability_detail([0.6, 0.5, 0.55], np.eye(3).tolist())["psd_adjusted"])

    def test_contradictory_pair_forces_exactly_zero(self):
        self.assertEqual(joint_probability([0.6, 0.4], [[1, CONTRADICTORY_RHO], [CONTRADICTORY_RHO, 1]]), 0.0)
        three = [[1, 0.2, 0], [0.2, 1, CONTRADICTORY_RHO], [0, CONTRADICTORY_RHO, 1]]
        self.assertEqual(joint_probability([0.6, 0.5, 0.5], three), 0.0)
        self.assertEqual(joint_probability_detail([0.6, 0.5, 0.5], three)["method"], "contradictory")
        # asymmetric input is still caught
        self.assertEqual(joint_probability([0.6, 0.4], [[1, 0], [CONTRADICTORY_RHO, 1]]), 0.0)

    def test_eight_legs_are_fast_and_accurate_enough(self):
        p = [0.62, 0.55, 0.5, 0.58, 0.6, 0.52, 0.65, 0.57]
        matrix = np.eye(8)
        for i, j in combinations(range(8), 2):
            matrix[i, j] = matrix[j, i] = 0.25
        started = time.perf_counter()
        detail = joint_probability_detail(p, matrix.tolist())
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 2.0)
        self.assertEqual(detail["method"], "monte_carlo")
        self.assertLess(detail["std_error"], 0.002)
        self.assertGreaterEqual(detail["draws"], 100_000)
        self.assertGreater(detail["probability"], math.prod(p))

    def test_monte_carlo_matches_the_exact_trivariate_orthant_formula(self):
        # P(Z1<=0, Z2<=0, Z3<=0) = 1/8 + (asin r12 + asin r13 + asin r23) / (4 pi)
        for rho in (0.3, 0.6, -0.2):
            with self.subTest(rho=rho):
                matrix = [[1, rho, rho], [rho, 1, rho], [rho, rho, 1]]
                exact = 1 / 8 + 3 * math.asin(rho) / (4 * math.pi)
                detail = joint_probability_detail([0.5, 0.5, 0.5], matrix)
                self.assertEqual(detail["method"], "monte_carlo")
                self.assertAlmostEqual(detail["probability"], exact, delta=0.003)
                self.assertLess(detail["std_error"], 0.002)

    def test_standard_error_is_small_for_typical_parlays(self):
        for p in ([0.5, 0.5, 0.5, 0.5], [0.9, 0.2, 0.7, 0.4, 0.55], [0.05, 0.1, 0.3]):
            n = len(p)
            matrix = np.full((n, n), 0.3)
            np.fill_diagonal(matrix, 1.0)
            with self.subTest(p=p):
                self.assertLess(joint_probability_detail(p, matrix.tolist())["std_error"], 0.002)

    def test_adding_a_leg_never_raises_the_joint_probability(self):
        base_p = [0.6, 0.55, 0.5]
        n = 4
        matrix = np.full((n, n), 0.25)
        np.fill_diagonal(matrix, 1.0)
        before = joint_probability(base_p, matrix[:3, :3].tolist())
        for extra in (0.9, 0.5, 0.1):
            with self.subTest(extra=extra):
                self.assertLessEqual(joint_probability(base_p + [extra], matrix.tolist()), before + 1e-9)

    def test_degenerate_probabilities(self):
        self.assertEqual(joint_probability([], []), 1.0)
        self.assertEqual(joint_probability([0.4], [[1.0]]), 0.4)
        self.assertEqual(joint_probability([0.0, 0.5], [[1, 0.3], [0.3, 1]]), 0.0)
        self.assertAlmostEqual(joint_probability([1.0, 0.5], [[1, 0.3], [0.3, 1]]), 0.5, places=12)
        self.assertAlmostEqual(joint_probability([1.0, 1.0], [[1, 0.3], [0.3, 1]]), 1.0, places=12)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            joint_probability([0.5, 1.2], [[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            joint_probability([0.5, 0.5], [[1, 0]])
        with self.assertRaises(ValueError):
            joint_probability([0.5, 0.5], [[1, float("nan")], [float("nan"), 1]])
        with self.assertRaises(ValueError):
            joint_probability([0.5, float("nan")], [[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            nearest_correlation_matrix([[1, 0, 0], [0, 1, 0]])

    def test_a_tiny_draw_request_is_raised_to_a_sane_floor(self):
        matrix = [[1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1]]
        detail = joint_probability_detail([0.5, 0.5, 0.5], matrix, draws=10)
        self.assertGreaterEqual(detail["draws"], 1_000)

    def test_ml_and_spread_joint_matches_a_simulation_of_the_margin_model(self):
        """ML/spread legs are threshold events on one normal margin; the copula must reproduce that."""
        from betting.moneyline_model import NFL_MARGIN_STDEV, win_probability_from_spread

        mu = 4.5  # KC expected to win by 4.5
        margin = np.random.default_rng(5).normal(mu, NFL_MARGIN_STDEV, 600_000)
        kc_ml = ml("KC", p=win_probability_from_spread(mu))
        kc_cover = spread("KC", -3.5, p=float(np.mean(margin > 3.5)))
        buf_plus = spread("BUF", 3.5, p=float(np.mean(margin < 3.5)))
        buf_plus_7 = spread("BUF", 7.5, p=float(np.mean(margin < 7.5)))
        kc_minus_3 = spread("KC", -3.5, p=float(np.mean(margin > 3.5)))
        cases = [
            ([kc_ml, kc_cover], np.mean((margin > 0) & (margin > 3.5))),            # nested (favourite cover)
            ([kc_ml, buf_plus], np.mean((margin > 0) & (margin < 3.5))),            # narrow band
            ([kc_minus_3, buf_plus_7], np.mean((margin > 3.5) & (margin < 7.5))),   # middle
        ]
        for legs, simulated in cases:
            with self.subTest(legs=[leg["description"] for leg in legs]):
                result = evaluate_parlay_legs(legs)
                self.assertAlmostEqual(result["adjusted_probability"], float(simulated), delta=0.006)


class MarginModelSlateTests(unittest.TestCase):
    """Random slates of moneyline/spread legs against a direct simulation of the margin they all derive from."""

    def test_random_team_leg_slates_match_the_margin_simulation(self):
        rng = random.Random(99)
        margin = np.random.default_rng(3).normal(3.0, 13.5, 400_000)

        def build(kind, team, line):
            if kind == "ml":
                hit = (margin > 0) if team == "KC" else (margin < 0)
                return ml(team, p=float(hit.mean())), hit
            hit = (margin > -line) if team == "KC" else (margin < line)
            return spread(team, line, p=float(hit.mean())), hit

        feasible = contradictory = 0
        for trial in range(90):
            legs, hits, used = [], [], set()
            for _ in range(rng.randint(2, 4)):
                leg, hit = build(rng.choice(["ml", "sp", "sp"]), rng.choice(["KC", "BUF"]),
                                 rng.choice([-10.5, -7, -3.5, -1.5, 0, 1.5, 3.5, 7, 10.5]))
                if leg["leg_id"] not in used:
                    used.add(leg["leg_id"])
                    legs.append(leg)
                    hits.append(hit)
            if len(legs) < 2:
                continue
            simulated = float(np.mean(np.logical_and.reduce(hits)))
            result = evaluate_parlay_legs(legs)
            names = [leg["description"] for leg in legs]
            with self.subTest(trial=trial, legs=names):
                if result["has_contradiction"]:
                    contradictory += 1
                    self.assertLess(simulated, 0.002, "flagged contradictory but the margin model can satisfy every leg")
                    self.assertEqual(result["adjusted_probability"], 0.0)
                else:
                    feasible += 1
                    self.assertAlmostEqual(result["adjusted_probability"], simulated, delta=0.006)
        self.assertGreater(feasible, 20)
        self.assertGreater(contradictory, 10)


class EvaluateParlayTests(unittest.TestCase):
    def two_leg_stack(self):
        return [
            prop("Patrick Mahomes", "passing_yards", "over", 274.5, player_id="qb1", p=0.52, price=-115),
            prop("Travis Kelce", "receiving_yards", "over", 64.5, player_id="te1", p=0.5, price=-110),
        ]

    def test_needs_at_least_two_legs(self):
        for legs in ([], [ml("KC")]):
            with self.subTest(n=len(legs)):
                with self.assertRaisesRegex(ValueError, "at least 2 legs"):
                    evaluate_parlay_legs(legs)

    def test_bad_stake_and_book_price_and_bad_leg_raise_friendly_errors(self):
        legs = self.two_leg_stack()
        for stake in (0, -5, float("nan")):
            with self.subTest(stake=stake):
                with self.assertRaises(ValueError):
                    evaluate_parlay_legs(legs, stake=stake)
        with self.assertRaisesRegex(ValueError, "book price"):
            evaluate_parlay_legs(legs, book_price=50)
        broken = {**legs[0], "model_probability": 1.4, "probability": 1.4}
        with self.assertRaisesRegex(ValueError, "Leg 1"):
            evaluate_parlay_legs([broken, legs[1]])
        with self.assertRaises(ValueError):
            evaluate_parlay_legs([{"description": "x"}, legs[1]])

    def test_required_keys_are_present(self):
        result = evaluate_parlay_legs(self.two_leg_stack())
        required = {
            "num_legs", "legs", "naive_probability", "adjusted_probability", "engine_adjusted_probability",
            "correlation_pairs", "matrix", "labels", "decimal_odds", "american_odds", "payout", "fair_decimal_odds",
            "fair_american_odds", "ev_naive", "ev_adjusted", "ev_pct", "edge", "confidence", "risk_tier", "risk_score",
            "risk_factors", "verdict", "verdict_tone", "warnings", "stake", "price_source", "has_contradiction",
            "probability_method", "probability_std_error", "probability_bounds", "correlation_lift", "summary",
            "risk_breakdown", "has_fair_price", "parlay_implied_probability", "psd_adjusted",
        }
        self.assertLessEqual(required, set(result))
        self.assertEqual(result["num_legs"], 2)
        self.assertEqual(len(result["legs"]), 2)
        self.assertEqual(result["labels"], [leg["description"] for leg in self.two_leg_stack()])
        self.assertEqual(result["legs"][0]["index"], 0)
        self.assertIn(result["risk_tier"], TIER_RANK)
        self.assertEqual(result["price_source"], "legs_product")

    def test_naive_probability_and_odds_reuse_the_engine(self):
        legs = [ml("KC", p=0.62, price=-150), prop("Dak Prescott", "passing_yards", "over", 260.5, team="DAL",
                                                  game=OTHER_GAME, p=0.5, price=-110),
                spread("NYG", 3.5, p=0.51, price=-105, game="G3", home="PHI", away="NYG")]
        result = evaluate_parlay_legs(legs)
        engine_legs = [engine_make_leg(description=l["description"], model_probability=l["model_probability"],
                                       price=l["price"], player_id=l["player_id"], team=l["team"], game_id=l["game_id"],
                                       market=l["market"], side=l["side"]) for l in legs]
        self.assertAlmostEqual(result["naive_probability"], naive_joint_probability(engine_legs), places=6)
        self.assertAlmostEqual(result["naive_probability"], 0.62 * 0.5 * 0.51, places=6)
        self.assertAlmostEqual(result["decimal_odds"], parlay_decimal_odds(engine_legs), places=5)
        self.assertEqual(result["american_odds"], int(decimal_to_american(parlay_decimal_odds(engine_legs))))
        expected_engine = correlation_adjusted_probability(engine_legs)["adjusted_probability"]
        self.assertAlmostEqual(result["engine_adjusted_probability"], expected_engine, places=6)

    def test_payout_and_ev_algebra(self):
        for stake in (100.0, 25.0, 7.5):
            legs = self.two_leg_stack() + [ml("KC", p=0.62, price=-150)]
            result = evaluate_parlay_legs(legs, stake=stake)
            decimal = result["decimal_odds"]
            self.assertAlmostEqual(result["payout"], stake * (decimal - 1), delta=0.01)
            for key, probability in (("ev_adjusted", result["adjusted_probability"]), ("ev_naive", result["naive_probability"])):
                with self.subTest(stake=stake, key=key):
                    expected = probability * result["payout"] - (1 - probability) * stake
                    self.assertAlmostEqual(result[key], expected, delta=0.02)
                    self.assertAlmostEqual(result[key], _parlay_ev(probability, decimal, stake), delta=0.02)
            self.assertAlmostEqual(result["ev_pct"], result["ev_adjusted"] / stake, delta=0.006 / stake + 1e-4)
            self.assertAlmostEqual(result["edge"], result["adjusted_probability"] - 1 / decimal, places=5)
            self.assertEqual(result["edge"] > 0, result["ev_adjusted"] > 0)
            self.assertAlmostEqual(result["parlay_implied_probability"], 1 / decimal, places=5)

    def test_fair_price_inverts_the_adjusted_probability(self):
        result = evaluate_parlay_legs(self.two_leg_stack())
        self.assertAlmostEqual(result["fair_decimal_odds"], 1 / result["adjusted_probability"], delta=2e-4 * result["fair_decimal_odds"])
        self.assertAlmostEqual(american_to_decimal(result["fair_american_odds"]), result["fair_decimal_odds"], delta=0.02)
        self.assertTrue(result["has_fair_price"])

    def test_positive_correlation_lifts_and_negative_lowers_the_probability(self):
        positive = evaluate_parlay_legs(self.two_leg_stack())
        self.assertGreater(positive["adjusted_probability"], positive["naive_probability"])
        self.assertGreater(positive["correlation_lift"], 0)
        self.assertGreater(positive["ev_adjusted"], positive["ev_naive"])
        negative = evaluate_parlay_legs([ml("KC", p=0.62), spread("BUF", 3.5, p=0.5)])
        self.assertLess(negative["adjusted_probability"], negative["naive_probability"])
        self.assertLess(negative["correlation_lift"], 0)
        # roughly the exact band: P(win) + P(cover) - 1 = 0.12 (never above it)
        self.assertLessEqual(negative["adjusted_probability"], 0.12 + 0.002)
        independent = evaluate_parlay_legs(independent_legs([0.6, 0.55]))
        self.assertAlmostEqual(independent["adjusted_probability"], independent["naive_probability"], places=6)
        self.assertEqual(independent["correlation_pairs"], [])
        self.assertEqual(independent["probability_method"], "independent")

    def test_engine_cross_check_differs_because_it_cannot_see_moneyline_and_spread(self):
        legs = [ml("KC", p=0.62), spread("KC", -3.5, p=0.5)]
        result = evaluate_parlay_legs(legs)
        self.assertAlmostEqual(result["engine_adjusted_probability"], 0.62 * 0.5, places=6)  # engine sees nothing
        self.assertAlmostEqual(result["adjusted_probability"], 0.5, delta=0.003)           # nested: the cover chance
        self.assertGreater(result["adjusted_probability"], result["engine_adjusted_probability"] + 0.15)

    def test_adjusted_probability_respects_frechet_bounds_in_the_result(self):
        result = evaluate_parlay_legs(self.two_leg_stack() + [ml("KC", p=0.62)])
        bounds = result["probability_bounds"]
        self.assertGreaterEqual(result["adjusted_probability"], bounds["lower"] - 1e-9)
        self.assertLessEqual(result["adjusted_probability"], bounds["upper"] + 1e-9)
        self.assertAlmostEqual(bounds["upper"], 0.5)

    def test_contradictory_legs_never_raise_and_price_as_a_certain_loss(self):
        cases = [
            [ml("KC"), ml("BUF", p=0.4)],
            [spread("KC", -3.5), spread("BUF", 3.5)],
            [prop("Travis Kelce", "receiving_yards", "over", 64.5), prop("Travis Kelce", "receiving_yards", "under", 64.5),
             ml("KC")],
        ]
        for legs in cases:
            for stake in (100.0, 40.0):
                with self.subTest(legs=[l["description"] for l in legs], stake=stake):
                    result = evaluate_parlay_legs(legs, stake=stake)
                    self.assertEqual(result["adjusted_probability"], 0.0)
                    self.assertAlmostEqual(result["ev_adjusted"], -stake, places=2)
                    self.assertEqual(result["risk_tier"], "High")
                    self.assertEqual(result["risk_score"], 100.0)
                    self.assertTrue(result["has_contradiction"])
                    self.assertTrue(any("cannot both win" in w for w in result["warnings"]))
                    self.assertEqual(result["verdict_tone"], "danger")
                    self.assertEqual(result["verdict"], "Contradictory legs")
                    self.assertIsNone(result["fair_decimal_odds"])
                    self.assertIsNone(result["fair_american_odds"])
                    self.assertFalse(result["has_fair_price"])
                    self.assertEqual(result["probability_method"], "contradictory")
                    assert_all_finite(self, result, allowed_none={"fair_decimal_odds", "fair_american_odds", "game_id", "team",
                                                                  "market", "side", "line", "player_name"})

    def test_opposed_legs_with_no_room_to_both_win_price_at_zero_without_being_contradictory(self):
        # KC 45% to win and BUF +3.5 only 50% to cover: the band between them has probability 0 by the model's own numbers
        result = evaluate_parlay_legs([ml("KC", p=0.45), spread("BUF", 3.5, p=0.5)])
        self.assertEqual(result["adjusted_probability"], 0.0)
        self.assertFalse(result["has_contradiction"])
        self.assertTrue(any("cannot all win together" in w for w in result["warnings"]))
        self.assertEqual(result["risk_tier"], "High")
        self.assertAlmostEqual(result["ev_adjusted"], -100.0, places=2)
        assert_all_finite(self, result, allowed_none={"fair_decimal_odds", "fair_american_odds", "game_id", "team",
                                                      "market", "side", "line", "player_name"})

    def test_no_nan_or_inf_anywhere_for_a_range_of_parlays(self):
        slates = [
            self.two_leg_stack(),
            self.two_leg_stack() + [ml("KC", p=0.62), spread("KC", -3.5), prop("Isiah Pacheco", "rushing_yards", "over", 64.5)],
            independent_legs([0.01, 0.02, 0.03]),
            independent_legs([0.99, 0.98, 0.97]),
            independent_legs([0.5] * 10, price=+100),
            independent_legs([0.9, 0.9], price=-5000),
            independent_legs([0.05, 0.05], price=+50000),
        ]
        for legs in slates:
            with self.subTest(n=len(legs)):
                result = evaluate_parlay_legs(legs)
                assert_all_finite(self, result, allowed_none={"game_id", "team", "market", "side", "line", "player_name"})
                self.assertTrue(0.0 <= result["adjusted_probability"] <= 1.0)

    def test_confidence_follows_the_engine_formula(self):
        legs = [ml("KC", confidence=0.8), prop("Travis Kelce", "receiving_yards", "over", 64.5, confidence=0.6),
                spread("DAL", -2.5, game=OTHER_GAME, home="DAL", away="NYG", confidence=0.7)]
        result = evaluate_parlay_legs(legs)
        self.assertAlmostEqual(result["confidence"], round(((0.8 + 0.6 + 0.7) / 3) * (1 - 0.04), 4), places=4)
        two = evaluate_parlay_legs(legs[:2])
        self.assertAlmostEqual(two["confidence"], 0.7, places=4)

    def test_book_price_overrides_the_leg_product_for_ev(self):
        legs = self.two_leg_stack()
        default = evaluate_parlay_legs(legs)
        quoted = evaluate_parlay_legs(legs, book_price=+240)
        self.assertEqual(quoted["price_source"], "book_quote")
        self.assertEqual(quoted["american_odds"], 240)
        self.assertAlmostEqual(quoted["decimal_odds"], 3.4)
        self.assertEqual(quoted["adjusted_probability"], default["adjusted_probability"])
        self.assertAlmostEqual(quoted["payout"], 240.0, places=2)
        self.assertAlmostEqual(
            quoted["ev_adjusted"], quoted["adjusted_probability"] * 240 - (1 - quoted["adjusted_probability"]) * 100, delta=0.01
        )
        self.assertFalse(any("reprice" in w for w in quoted["warnings"]))
        self.assertTrue(any("reprice" in w for w in default["warnings"]))

    def test_warnings(self):
        same_game = evaluate_parlay_legs(self.two_leg_stack())
        self.assertTrue(any("one game" in w for w in same_game["warnings"]))
        duplicate = evaluate_parlay_legs([ml("KC"), ml("KC", price=-140), ml("DAL", game=OTHER_GAME, home="NYG", away="DAL")])
        self.assertTrue(any("more than once" in w for w in duplicate["warnings"]))
        nested = evaluate_parlay_legs([ml("KC"), spread("KC", -3.5)])
        self.assertTrue(any("nearly the same bet" in w for w in nested["warnings"]))
        long = evaluate_parlay_legs(independent_legs([0.8] * 9))
        self.assertTrue(any("very long parlay" in w for w in long["warnings"]))
        self.assertFalse(any("very long parlay" in w for w in evaluate_parlay_legs(independent_legs([0.8] * 8))["warnings"]))
        tiny = evaluate_parlay_legs(independent_legs([0.1, 0.1, 0.1]))
        self.assertTrue(any("below 1%" in w for w in tiny["warnings"]))
        fine = evaluate_parlay_legs(independent_legs([0.6, 0.6]))
        self.assertEqual(fine["warnings"], [])

    def test_risk_tier_agrees_with_the_engine_thresholds_on_a_grid(self):
        probabilities = [0.99, 0.9, 0.7, 0.5, 0.36, 0.35, 0.349, 0.34, 0.2, 0.16, 0.15, 0.149, 0.1, 0.05, 0.02]
        checked = 0
        for n in range(2, 7):
            for target in probabilities:
                leg_p = target ** (1.0 / n)
                if leg_p >= 1.0:
                    continue
                result = evaluate_parlay_legs(independent_legs([leg_p] * n))
                joint = result["adjusted_probability"]
                engine_tier = _risk_tier(n, joint).capitalize()
                with self.subTest(n=n, p=target):
                    self.assertEqual(result["risk_tier"], engine_tier, (n, joint, result["risk_score"]))
                    self.assertGreaterEqual(TIER_RANK[result["risk_tier"]], TIER_RANK[engine_tier])
                    self.assertEqual(result["risk_breakdown"]["engine_tier"], engine_tier)
                checked += 1
        self.assertGreater(checked, 60)

    def test_documented_engine_boundaries(self):
        two_legs_at_floor = evaluate_parlay_legs(independent_legs([math.sqrt(0.36)] * 2))
        self.assertEqual(two_legs_at_floor["risk_tier"], "Low")
        self.assertEqual(evaluate_parlay_legs(independent_legs([math.sqrt(0.30)] * 2))["risk_tier"], "Medium")
        self.assertEqual(evaluate_parlay_legs(independent_legs([0.9] * 3))["risk_tier"], "Medium")
        self.assertEqual(evaluate_parlay_legs(independent_legs([0.97] * 5))["risk_tier"], "High")
        self.assertEqual(evaluate_parlay_legs(independent_legs([0.3, 0.3, 0.3]))["risk_tier"], "High")

    def test_long_odds_legs_and_correlation_can_raise_the_tier_above_the_engine_baseline(self):
        # engine: 2 legs at 49% is Low; two +1200 legs make it fragile, so the score escalates it
        long_shots = evaluate_parlay_legs(independent_legs([0.7, 0.7], price=+1200))
        self.assertEqual(_risk_tier(2, long_shots["adjusted_probability"]), "low")
        self.assertEqual(long_shots["risk_tier"], "Medium")
        self.assertTrue(any("Longest-odds leg" in f for f in long_shots["risk_factors"]))
        safe = evaluate_parlay_legs(independent_legs([0.7, 0.7], price=-110))
        self.assertEqual(safe["risk_tier"], "Low")
        self.assertGreater(long_shots["risk_score"], safe["risk_score"])
        # heavier correlation concentration scores higher than the same legs spread across games
        stacked = evaluate_parlay_legs([ml("KC", p=0.6), prop("Patrick Mahomes", "passing_tds", "over", 1.5, p=0.6),
                                        prop("Travis Kelce", "receiving_tds", "over", 0.5, p=0.6)])
        self.assertGreater(stacked["risk_breakdown"]["correlation"], 0)

    def test_risk_is_monotone_when_a_long_shot_leg_is_added(self):
        bases = [
            independent_legs([0.8, 0.8]), independent_legs([0.6, 0.6]), independent_legs([0.9, 0.9, 0.9]),
            independent_legs([0.5, 0.6, 0.7, 0.8]), self_stack(), [ml("KC"), spread("KC", -3.5)],
        ]
        longshots = [
            make_parlay_leg("moneyline", description="Long shot", probability=0.05, price=+1800, game_id="LS", team="LSA"),
            make_parlay_leg("moneyline", description="Mild shot", probability=0.4, price=+150, game_id="LS", team="LSA"),
            make_parlay_leg("moneyline", description="Heavy chalk", probability=0.95, price=-900, game_id="LS", team="LSA"),
        ]
        for base in bases:
            before = evaluate_parlay_legs(base)
            for extra in longshots:
                after = evaluate_parlay_legs(base + [extra])
                with self.subTest(base=len(base), extra=extra["description"]):
                    self.assertGreaterEqual(TIER_RANK[after["risk_tier"]], TIER_RANK[before["risk_tier"]])
                    self.assertGreaterEqual(after["risk_score"], before["risk_score"] - 1e-9)

    def test_risk_score_grows_with_leg_count_and_falling_probability(self):
        scores = [evaluate_parlay_legs(independent_legs([0.8] * n))["risk_score"] for n in range(2, 9)]
        self.assertEqual(scores, sorted(scores))
        self.assertLess(scores[0], scores[-1])
        by_prob = [evaluate_parlay_legs(independent_legs([p, p]))["risk_score"] for p in (0.9, 0.8, 0.6, 0.4, 0.2, 0.1)]
        self.assertEqual(by_prob, sorted(by_prob))
        for score in scores + by_prob:
            self.assertTrue(0.0 <= score <= 100.0)

    def test_risk_factors_cite_real_numbers(self):
        result = evaluate_parlay_legs([ml("KC", p=0.62, price=-150), spread("KC", -3.5), prop("Isiah Pacheco", "rushing_yards", "over", 64.5)])
        text = " ".join(result["risk_factors"])
        self.assertIn("3 legs", text)
        self.assertIn(f"{result['adjusted_probability'] * 100:.1f}%", text)
        self.assertIn("rho +0.95", text)
        self.assertTrue(all(isinstance(item, str) and item for item in result["risk_factors"]))

    def test_verdict_follows_ev_per_dollar_staked(self):
        value = evaluate_parlay_legs(independent_legs([0.75, 0.75], price=+100))
        self.assertGreater(value["ev_pct"], 0.06)
        self.assertEqual((value["verdict"], value["verdict_tone"]), ("Strong value", "success"))
        poor = evaluate_parlay_legs(independent_legs([0.45, 0.45], price=-110))
        self.assertLess(poor["ev_pct"], -0.02)
        self.assertEqual((poor["verdict"], poor["verdict_tone"]), ("Overpriced", "danger"))
        # a long shot with a modest probability edge but a big EV is judged by return on stake
        long_shot = evaluate_parlay_legs(independent_legs([0.2, 0.2, 0.2], price=+500))
        self.assertGreater(long_shot["ev_pct"], 0.06)
        self.assertEqual(long_shot["verdict"], "Strong value")
        self.assertLess(long_shot["edge"], 0.03, "a probability gap this small would read as 'fair' under a raw-gap rule")
        fair = evaluate_parlay_legs(independent_legs([0.2, 0.2, 0.2], price=+400))  # exactly fair: 0.008 * 125 = 1
        self.assertAlmostEqual(fair["ev_adjusted"], 0.0, places=2)
        self.assertEqual(fair["verdict"], "Fair price")

    def test_inputs_are_not_mutated(self):
        legs = self.two_leg_stack()
        snapshot = [dict(leg) for leg in legs]
        evaluate_parlay_legs(legs)
        self.assertEqual(legs, snapshot)

    def test_engine_built_legs_are_accepted(self):
        a = engine_make_leg(description="qb", model_probability=0.5, price=-110, player_id="qb1", team="KC",
                            game_id=GAME, market="passing_yards", side="over")
        b = engine_make_leg(description="wr", model_probability=0.5, price=-110, player_id="te1", team="KC",
                            game_id=GAME, market="receiving_yards", side="over")
        result = evaluate_parlay_legs([a, b])
        self.assertGreater(result["adjusted_probability"], result["naive_probability"])

    def test_result_is_deterministic(self):
        legs = self.two_leg_stack() + [ml("KC"), spread("KC", -3.5, p=0.5), prop("Isiah Pacheco", "rushing_yards", "over", 64.5)]
        self.assertEqual(evaluate_parlay_legs(legs), evaluate_parlay_legs(legs))

    def test_eight_leg_same_game_parlay_is_fast(self):
        legs = [
            ml("KC", p=0.62), spread("KC", -3.5, p=0.5),
            prop("Patrick Mahomes", "passing_yards", "over", 274.5, player_id="qb1"),
            prop("Travis Kelce", "receiving_yards", "over", 64.5, player_id="te1"),
            prop("Travis Kelce", "receptions", "over", 5.5, player_id="te1"),
            prop("Isiah Pacheco", "rushing_yards", "over", 64.5, player_id="rb1"),
            prop("Josh Allen", "passing_yards", "over", 259.5, team="BUF", player_id="qb2"),
            prop("Rashee Rice", "receiving_yards", "over", 70.5, player_id="wr1"),
        ]
        started = time.perf_counter()
        result = evaluate_parlay_legs(legs)
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertLess(result["probability_std_error"], 0.002)
        self.assertEqual(result["risk_tier"], "High")


def self_stack():
    return [
        prop("Patrick Mahomes", "passing_yards", "over", 274.5, player_id="qb1", p=0.55),
        prop("Travis Kelce", "receiving_yards", "over", 64.5, player_id="te1", p=0.55),
    ]


class ConflictTests(unittest.TestCase):
    def test_legs_conflict_reports_duplicates_and_contradictions(self):
        self.assertEqual(legs_conflict([ml("KC"), spread("KC", -3.5), ml("DAL", game=OTHER_GAME, home="NYG", away="DAL")]), [])
        duplicates = legs_conflict([ml("KC"), ml("KC", price=-140)])
        self.assertEqual(len(duplicates), 1)
        self.assertIn("Duplicate leg", duplicates[0])
        self.assertIn("KC moneyline", duplicates[0])
        contradictions = legs_conflict([ml("KC"), ml("BUF", p=0.4), spread("BUF", 3.5)])
        self.assertTrue(any("KC moneyline" in m and "BUF moneyline" in m and "can't both win" in m for m in contradictions))
        self.assertEqual(legs_conflict([]), [])

    def test_can_add_leg_blocks_duplicates_and_contradictions_with_a_reason(self):
        existing = [ml("KC"), prop("Travis Kelce", "receiving_yards", "over", 64.5)]
        allowed, message = can_add_leg(existing, ml("KC", price=-135))
        self.assertFalse(allowed)
        self.assertIn("already in your parlay", message)
        allowed, message = can_add_leg(existing, ml("BUF", p=0.4))
        self.assertFalse(allowed)
        self.assertIn("BUF moneyline", message)
        self.assertIn("KC moneyline", message)
        self.assertIn("cannot both win", message)
        allowed, message = can_add_leg(existing, prop("Travis Kelce", "receiving_yards", "under", 64.5))
        self.assertFalse(allowed)
        self.assertIn("cannot both hit", message)

    def test_can_add_leg_allows_compatible_legs_and_flags_near_duplicates(self):
        existing = [ml("KC"), prop("Travis Kelce", "receiving_yards", "over", 64.5)]
        self.assertEqual(can_add_leg(existing, ml("DAL", game=OTHER_GAME, home="NYG", away="DAL")), (True, ""))
        self.assertEqual(can_add_leg([], ml("KC")), (True, ""))
        allowed, message = can_add_leg(existing, spread("KC", -3.5))
        self.assertTrue(allowed)
        self.assertIn("nearly the same bet", message)
        allowed, message = can_add_leg(existing, prop("Patrick Mahomes", "passing_yards", "over", 274.5, player_id="qb1"))
        self.assertEqual((allowed, message), (True, ""))

    def test_a_middle_is_allowed_but_an_empty_window_is_not(self):
        self.assertTrue(can_add_leg([spread("KC", -3.5)], spread("BUF", 7.5))[0])
        self.assertFalse(can_add_leg([spread("KC", -3.5)], spread("BUF", 3.5))[0])


class RuleTableTests(unittest.TestCase):
    def test_rule_table_rows_are_complete(self):
        rows = correlation_rule_table()
        self.assertEqual({row["kind"] for row in rows}, set(CORRELATION_KINDS))
        for row in rows:
            self.assertEqual(set(row), {"kind", "prior", "summary"})
            self.assertTrue(row["prior"] and row["summary"])

    def test_every_nonexclusive_prior_stays_within_the_documented_ceiling(self):
        self.assertEqual(MAX_ABS_RHO, 0.95)
        self.assertEqual(CONTRADICTORY_RHO, -2.0)
        self.assertLess(CONTRADICTORY_RHO, -1.0, "must sit below every legitimate correlation, -1 included")


# --------------------------------------------------------------------------- #
# Regression tests from the independent audit. Each class pins one defect that the
# original suite missed; the docstring says what used to go wrong.
# --------------------------------------------------------------------------- #

NAN = float("nan")
_NONE_OK = {"fair_decimal_odds", "fair_american_odds", "game_id", "team", "market", "side", "line", "player_name"}


class LongShotPrecisionTests(unittest.TestCase):
    """A 10-leg parlay of 20% legs (1e-7) was rounded to 0.0 and reported as impossible, with no fair price."""

    def test_a_ten_leg_long_shot_is_priced_not_called_impossible(self):
        result = evaluate_parlay_legs(independent_legs([0.2] * 10))
        expected = 0.2 ** 10
        self.assertAlmostEqual(result["adjusted_probability"], expected, delta=expected * 1e-6)
        self.assertAlmostEqual(result["naive_probability"], expected, delta=expected * 1e-6)
        self.assertTrue(result["has_fair_price"])
        self.assertAlmostEqual(result["fair_decimal_odds"], 1 / expected, delta=1e-6 / expected)
        self.assertGreater(result["fair_american_odds"], 9e8)
        self.assertGreater(result["parlay_implied_probability"], 0)
        text = " ".join(result["warnings"] + result["risk_factors"])
        self.assertNotIn("cannot all win", text)
        self.assertNotIn("cannot win together", text)
        self.assertIn("less than 0.01%", text, "a real long shot must not read as '0.0%'")
        self.assertTrue(any("below 1%" in w for w in result["warnings"]))
        self.assertEqual(result["risk_tier"], "High")
        p, payout = result["adjusted_probability"], result["payout"]
        self.assertAlmostEqual(result["ev_adjusted"], p * payout - (1 - p) * 100, delta=0.02)
        assert_all_finite(self, result, allowed_none=_NONE_OK)

    def test_fair_price_of_a_one_in_a_hundred_billion_ticket_is_not_capped_at_a_million(self):
        result = evaluate_parlay_legs(independent_legs([0.1] * 11))
        self.assertAlmostEqual(result["fair_decimal_odds"], 1e11, delta=1e11 * 1e-6)
        self.assertAlmostEqual(american_to_decimal(result["fair_american_odds"]), 1e11, delta=1e11 * 1e-6)

    def test_extreme_long_shot_stays_finite_and_positive(self):
        result = evaluate_parlay_legs(independent_legs([0.05] * 30))
        self.assertGreater(result["adjusted_probability"], 0.0)
        self.assertTrue(result["has_fair_price"])
        assert_all_finite(self, result, allowed_none=_NONE_OK)

    def test_leg_summaries_keep_a_long_shot_leg_probability(self):
        result = evaluate_parlay_legs(independent_legs([1e-9, 0.5], price=+100000))
        self.assertEqual(result["legs"][0]["probability"], 1e-9)
        self.assertGreater(result["legs"][0]["implied_probability"], 0)

    def test_fair_american_odds_come_from_the_engine_conversion_for_ordinary_probabilities(self):
        from betting.odds_math import fair_price_from_probability

        for legs in (independent_legs([0.6, 0.55]), independent_legs([0.3, 0.2, 0.25]), independent_legs([0.9, 0.95])):
            result = evaluate_parlay_legs(legs)
            self.assertEqual(result["fair_american_odds"], fair_price_from_probability(result["adjusted_probability"]))

    def test_a_joint_chance_that_underflows_a_float_is_not_blamed_on_opposed_legs(self):
        result = evaluate_parlay_legs(independent_legs([1e-9] * 40))  # 1e-360 does not fit in a double
        self.assertEqual(result["adjusted_probability"], 0.0)
        self.assertEqual(result["naive_probability"], 0.0)
        self.assertFalse(any("cannot all win" in w for w in result["warnings"]), result["warnings"])
        self.assertFalse(any("cannot win together" in f for f in result["risk_factors"]), result["risk_factors"])
        self.assertTrue(any("too small to represent" in f for f in result["risk_factors"]))
        assert_all_finite(self, result, allowed_none=_NONE_OK)

    def test_a_genuinely_impossible_pair_is_still_reported_as_impossible(self):
        result = evaluate_parlay_legs([ml("KC", p=0.45), spread("BUF", 3.5, p=0.5)])
        self.assertEqual(result["adjusted_probability"], 0.0)
        self.assertFalse(result["has_fair_price"])
        self.assertTrue(any("cannot all win together" in w for w in result["warnings"]))
        self.assertTrue(any("0%" in f for f in result["risk_factors"]))

    def test_ordinary_probabilities_keep_one_decimal_text(self):
        result = evaluate_parlay_legs(independent_legs([0.6, 0.55]))
        self.assertIn("33.0%", " ".join(result["risk_factors"]))
        self.assertIn("33.0%", result["summary"])


class InputBoundsTests(unittest.TestCase):
    """Absurd numbers used to escape as OverflowError / a bare engine ValueError / inf, and the leg count was unbounded."""

    def test_a_leg_price_beyond_any_real_sportsbook_is_a_friendly_error(self):
        for price in (MAX_ABS_PRICE + 1, -MAX_ABS_PRICE - 1, 1e6, -1e6, 1e200, -1e300):
            with self.subTest(price=price):
                with self.assertRaisesRegex(ValueError, "no sportsbook posts"):
                    make_parlay_leg("moneyline", description="x", probability=0.5, price=price, team="KC")
        for price in (MAX_ABS_PRICE, -MAX_ABS_PRICE):
            make_parlay_leg("moneyline", description="x", probability=0.5, price=price, team="KC")

    def test_the_evaluator_rejects_hand_built_legs_with_absurd_prices(self):
        good = ml("KC")
        for price in (1e200, -1e20):
            with self.subTest(price=price):
                with self.assertRaisesRegex(ValueError, "Leg 2"):
                    evaluate_parlay_legs([good, {**ml("DAL", game=OTHER_GAME, home="NYG", away="DAL"), "price": price}])

    def test_book_price_and_stake_have_ceilings(self):
        legs = independent_legs([0.6, 0.6])
        with self.assertRaisesRegex(ValueError, "book price"):
            evaluate_parlay_legs(legs, book_price=MAX_ABS_BOOK_PRICE * 10)
        with self.assertRaisesRegex(ValueError, "stake"):
            evaluate_parlay_legs(legs, stake=MAX_STAKE * 10)
        ok = evaluate_parlay_legs(legs, book_price=MAX_ABS_BOOK_PRICE, stake=MAX_STAKE)
        assert_all_finite(self, ok, allowed_none=_NONE_OK)

    def test_prop_line_ceiling(self):
        with self.assertRaisesRegex(ValueError, "not a real stat line"):
            prop("Some Player", "passing_yards", "over", MAX_PROP_LINE + 1)

    def test_leg_count_is_capped_with_a_friendly_message(self):
        with self.assertRaisesRegex(ValueError, f"at most {MAX_PARLAY_LEGS} legs"):
            evaluate_parlay_legs(independent_legs([0.9] * (MAX_PARLAY_LEGS + 1)))

    def test_the_largest_allowed_parlays_price_quickly_and_finitely(self):
        started = time.perf_counter()
        for legs in (
            independent_legs([0.9] * MAX_PARLAY_LEGS, price=-MAX_ABS_PRICE),
            independent_legs([0.001] * MAX_PARLAY_LEGS, price=MAX_ABS_PRICE),
        ):
            result = evaluate_parlay_legs(legs)
            assert_all_finite(self, result, allowed_none=_NONE_OK)
        one_block = [
            make_parlay_leg("prop", description=f"WR{i} over", probability=0.6, price=-110, game_id=GAME, team="KC",
                            market="receiving_yards", side="over", line=10.5 + i, player_id=f"wr{i}")
            for i in range(MAX_PARLAY_LEGS)
        ]
        result = evaluate_parlay_legs(one_block)
        self.assertEqual(result["probability_method"], "monte_carlo")
        assert_all_finite(self, result, allowed_none=_NONE_OK)
        self.assertLess(time.perf_counter() - started, 10.0)

    def test_joint_probability_tolerates_a_missing_or_nonsense_draw_count(self):
        matrix = [[1, 0.3, 0.3], [0.3, 1, 0.3], [0.3, 0.3, 1]]
        default = joint_probability([0.5, 0.5, 0.5], matrix)
        for draws in (None, float("nan"), "lots"):
            with self.subTest(draws=draws):
                self.assertEqual(joint_probability([0.5, 0.5, 0.5], matrix, draws=draws), default)

    def test_monte_carlo_result_does_not_depend_on_the_memory_chunking(self):
        p = [0.6, 0.5, 0.55, 0.45, 0.7]
        matrix = np.full((5, 5), 0.3)
        np.fill_diagonal(matrix, 1.0)
        baseline = joint_probability_detail(p, matrix.tolist(), draws=120_000)
        original = parlay_module._MC_CHUNK_CELLS
        try:
            parlay_module._MC_CHUNK_CELLS = 5 * 7_000  # forces many small chunks
            chunked = joint_probability_detail(p, matrix.tolist(), draws=120_000)
        finally:
            parlay_module._MC_CHUNK_CELLS = original
        self.assertEqual(baseline["probability"], chunked["probability"])

    def test_a_very_wide_block_samples_in_bounded_chunks(self):
        """100 linked legs x 50,000 draws would be 40MB per chunk; the width-aware cap keeps it near 2M cells."""
        from unittest import mock

        sizes = []
        real_default_rng = np.random.default_rng

        class Recorder:
            def __init__(self, seed):
                self._rng = real_default_rng(seed)

            def standard_normal(self, size):
                sizes.append(size)
                return self._rng.standard_normal(size)

        n = 100
        matrix = np.full((n, n), 0.2)
        np.fill_diagonal(matrix, 1.0)
        with mock.patch.object(np.random, "default_rng", lambda seed: Recorder(seed)):
            detail = joint_probability_detail([0.9] * n, matrix.tolist(), draws=200_000)
        self.assertEqual(detail["method"], "monte_carlo")
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes) * n, parlay_module._MC_CHUNK_CELLS)
        self.assertGreater(max(sizes), 1_000, "the cap must not collapse the chunk to nothing")
        self.assertTrue(math.isfinite(detail["probability"]))

    def test_nearest_correlation_matrix_rejects_nan(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            nearest_correlation_matrix([[1, NAN], [NAN, 1]])

    def test_bivariate_cdf_handles_infinite_and_nan_arguments(self):
        inf = float("inf")
        self.assertAlmostEqual(bivariate_normal_cdf(inf, 0.3, 0.5), _phi(0.3), places=12)
        self.assertAlmostEqual(bivariate_normal_cdf(0.3, inf, -0.5), _phi(0.3), places=12)
        self.assertEqual(bivariate_normal_cdf(-inf, 0.3, 0.5), 0.0)
        self.assertEqual(bivariate_normal_cdf(inf, inf, 0.5), 1.0)
        with self.assertRaises(ValueError):
            bivariate_normal_cdf(NAN, 0.0, 0.5)


class PlayerIdentityTests(unittest.TestCase):
    """The name-based player id dropped accents and non-Latin letters, so distinct people could collapse into one."""

    def named(self, name, market="receiving_yards", **extra):
        return make_parlay_leg("prop", description=f"{name} over", probability=0.5, price=-110, game_id=GAME, team="KC",
                               market=market, side="over", line=60.5, player_name=name, **extra)

    def test_accented_and_plain_spellings_are_the_same_player(self):
        self.assertEqual(self.named("Audric Estimé")["player_id"], "name:audric_estime")
        self.assertEqual(self.named("Audric Estimé")["leg_id"], self.named("Audric Estime")["leg_id"])
        self.assertNotEqual(self.named("Zoë Smith")["player_id"], self.named("Zo Smith")["player_id"])

    def test_different_non_latin_names_are_different_players(self):
        a, b = self.named("李雷", "receiving_yards"), self.named("王芳", "receptions")
        self.assertNotEqual(a["player_id"], b["player_id"])
        self.assertNotEqual(a["leg_id"], self.named("王芳", "receiving_yards")["leg_id"])
        self.assertFalse(leg_correlation(a, b)["kind"].startswith("same_player"), leg_correlation(a, b)["kind"])
        self.assertEqual(legs_conflict([a, self.named("王芳", "receiving_yards")]), [])

    def test_names_without_letters_still_get_distinct_stable_ids(self):
        ids = {self.named(name)["player_id"] for name in ("???", "!!!", "...")}
        self.assertEqual(len(ids), 3)
        self.assertEqual(self.named("???")["player_id"], self.named("???")["player_id"])
        for player_id in ids:
            self.assertNotEqual(player_id, "name:")

    def test_ids_stay_widget_safe_for_unicode_names(self):
        for name in ("Audric Estimé", "李雷", "D'Andre Swift", "Amon-Ra St. Brown"):
            self.assertRegex(self.named(name)["leg_id"], r"^[A-Za-z0-9._:+-]+$")


class SignedZeroTests(unittest.TestCase):
    """A pick'em built as -spread_line carried -0.0: a different leg_id and a note reading 'covering -0'."""

    def test_negative_zero_line_is_normalised(self):
        neg = spread("KC", -0.0)
        pos = spread("KC", 0.0)
        self.assertEqual(neg["leg_id"], pos["leg_id"])
        self.assertEqual(math.copysign(1.0, neg["line"]), 1.0)
        note = leg_correlation(ml("KC"), neg)["note"]
        self.assertNotRegex(note, r"(?<![\d.])-0(?![\d.])")

    def test_hand_built_negative_zero_line_is_normalised_by_the_evaluator(self):
        legs = [ml("KC"), {**spread("KC", 0.0), "line": -0.0}]
        result = evaluate_parlay_legs(legs)
        self.assertEqual(math.copysign(1.0, result["legs"][1]["line"]), 1.0)


class StructuralTieTests(unittest.TestCase):
    """Legs tied by +/-1 are one variable; a prop must correlate equally with each of them.

    The pair rules damp an opposing-team prop, so the matrix was invalid, and the nearest-matrix
    projection that rescued it loosened the +/-1 pair itself: an ML + opposing-spread band with a
    prop priced 0.074 against ~0.056 from a coherent simulation (and a middle + prop 0.066 vs ~0.058).
    """

    MU, SD = 3.0, 13.5

    def simulate(self, seed, c_prop, p_prop, size=800_000):
        """Margin draws plus a prop whose latent is correlated ``c_prop`` with the margin (win-direction)."""
        rng = np.random.default_rng(seed)
        margin = rng.normal(self.MU, self.SD, size)
        z = (margin - self.MU) / self.SD
        latent = -c_prop * z + math.sqrt(1 - c_prop ** 2) * rng.standard_normal(size)
        from statistics import NormalDist
        return margin, latent <= NormalDist().inv_cdf(p_prop)

    def test_matrix_gives_tied_legs_equal_and_opposite_correlation_with_a_third_leg(self):
        legs = [ml("KC", p=0.62), spread("BUF", 3.5, p=0.5), prop("Patrick Mahomes", "passing_tds", "over", 1.5, p=0.55, player_id="qb1")]
        matrix = correlation_matrix(legs)
        self.assertEqual(matrix[0][1], -1.0)
        self.assertAlmostEqual(matrix[0][2], (0.28 + 0.168) / 2, places=9)
        self.assertAlmostEqual(matrix[1][2], -matrix[0][2], places=12)
        self.assertEqual(matrix[2][0], matrix[0][2])
        # the pair's own documented rule is not rewritten; only the copula input is
        self.assertAlmostEqual(leg_correlation(legs[1], legs[2])["rho"], -0.168, places=6)
        evaluated = evaluate_parlay_legs(legs)
        self.assertEqual(evaluated["matrix"], matrix)
        self.assertFalse(evaluated["psd_adjusted"], "a consistent matrix needs no projection")

    def test_tied_group_correlations_are_consistent_for_every_other_leg_in_a_mixed_slate(self):
        legs = [ml("KC", p=0.55), spread("KC", 1.5, p=0.58), prop("Isiah Pacheco", "rushing_yards", "over", 64.5, p=0.5),
                prop("Josh Allen", "passing_yards", "over", 259.5, team="BUF", p=0.5), spread("KC", -7.5, p=0.35)]
        matrix = np.array(correlation_matrix(legs))
        for i, j in combinations(range(len(legs)), 2):
            if abs(abs(matrix[i, j]) - 1.0) < 1e-12:
                for k in range(len(legs)):
                    if k not in (i, j):
                        self.assertAlmostEqual(matrix[i, k], matrix[i, j] * matrix[j, k], places=9, msg=(i, j, k))

    def test_unequal_pair_rules_for_a_nested_pair_are_averaged(self):
        # a 55% moneyline reads as a favourite (rush prior 0.22); a +1.5 spread reads as not laying points (0.12)
        legs = [ml("KC", p=0.55), spread("KC", 1.5, p=0.58), prop("Isiah Pacheco", "rushing_yards", "over", 64.5, p=0.5)]
        self.assertAlmostEqual(leg_correlation(legs[0], legs[2])["rho"], 0.22, places=6)
        self.assertAlmostEqual(leg_correlation(legs[1], legs[2])["rho"], 0.12, places=6)
        matrix = correlation_matrix(legs)
        self.assertAlmostEqual(matrix[0][2], 0.17, places=9)
        self.assertAlmostEqual(matrix[1][2], 0.17, places=9)

    def test_band_plus_prop_matches_a_coherent_simulation(self):
        c = (0.28 + 0.168) / 2
        margin, prop_hit = self.simulate(31, c, 0.55)
        win, cover = margin > 0, margin < 3.5
        legs = [ml("KC", p=float(win.mean())), spread("BUF", 3.5, p=float(cover.mean())),
                prop("Patrick Mahomes", "passing_tds", "over", 1.5, p=0.55, player_id="qb1")]
        truth = float(np.mean(win & cover & prop_hit))
        result = evaluate_parlay_legs(legs)
        self.assertAlmostEqual(result["adjusted_probability"], truth, delta=0.004,
                               msg="the projection used to loosen the exact pair: 0.074 instead of ~0.056")
        # and it is not simply the pair times the prop's marginal
        self.assertLess(result["adjusted_probability"], 0.12)

    def test_middle_plus_prop_matches_a_coherent_simulation(self):
        c = (0.08 + 0.08 * 0.4) / 2
        margin, prop_hit = self.simulate(32, c, 0.5)
        kc_cover, buf_cover = margin > 3.5, margin < 7.5
        legs = [spread("KC", -3.5, p=float(kc_cover.mean())), spread("BUF", 7.5, p=float(buf_cover.mean())),
                prop("Patrick Mahomes", "passing_yards", "over", 274.5, p=0.5, player_id="qb1")]
        truth = float(np.mean(kc_cover & buf_cover & prop_hit))
        result = evaluate_parlay_legs(legs)
        self.assertAlmostEqual(result["adjusted_probability"], truth, delta=0.003)

    def test_two_tied_groups_in_one_ticket_stay_exact_inside_each_group(self):
        legs = [ml("KC", p=0.62), spread("KC", -3.5, p=0.5),
                prop("Travis Kelce", "receiving_yards", "over", 54.5, p=0.6, player_id="te1"),
                prop("Travis Kelce", "receiving_yards", "over", 74.5, p=0.3, player_id="te1")]
        matrix = correlation_matrix(legs)
        self.assertEqual(matrix[0][1], 1.0)
        self.assertEqual(matrix[2][3], 1.0)
        cross = {round(matrix[i][j], 12) for i in (0, 1) for j in (2, 3)}
        self.assertEqual(len(cross), 1, "all four cross-group entries describe one shared correlation")
        result = evaluate_parlay_legs(legs)
        self.assertLessEqual(result["adjusted_probability"], 0.3 + 1e-9)

    def test_contradictory_or_unrelated_matrices_are_left_alone(self):
        contradictory = correlation_matrix([ml("KC"), ml("BUF", p=0.4), prop("Travis Kelce", "receiving_yards", "over", 64.5)])
        self.assertEqual(contradictory[0][1], CONTRADICTORY_RHO)
        plain = [[1.0, 0.2, 0.1], [0.2, 1.0, -0.3], [0.1, -0.3, 1.0]]
        self.assertEqual(parlay_module._tie_structural_groups(plain), plain)
        # a tie cycle that cannot close (A=B, B=C, but A=-C) is not rewritten either
        broken = [[1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]
        self.assertEqual(parlay_module._tie_structural_groups(broken), broken)

    def test_public_matrix_and_evaluator_matrix_always_agree(self):
        rng = random.Random(404)
        pool = [
            ml("KC", p=0.6), ml("BUF", p=0.4), spread("KC", -3.5, p=0.5), spread("BUF", 7.5, p=0.6), spread("KC", -7.5, p=0.35),
            prop("Patrick Mahomes", "passing_yards", "over", 274.5, p=0.5, player_id="qb1"),
            prop("Travis Kelce", "receiving_yards", "over", 64.5, p=0.5, player_id="te1"),
            prop("Travis Kelce", "receptions", "over", 5.5, p=0.55, player_id="te1"),
            prop("Josh Allen", "passing_tds", "over", 1.5, team="BUF", p=0.55, player_id="qb2"),
        ]
        for trial in range(40):
            legs = rng.sample(pool, rng.randint(2, 6))
            with self.subTest(trial=trial):
                self.assertEqual(evaluate_parlay_legs(legs)["matrix"], correlation_matrix(legs))


class MissingValueTests(unittest.TestCase):
    """A pandas NaN team/game became the real-looking code 'NAN' / 'nan', linking unrelated legs; NaN lines leaked into output."""

    def test_nan_team_and_game_are_missing_not_real_codes(self):
        for missing in (NAN, "nan", "None", "<NA>", "NaT", " ", None):
            with self.subTest(missing=missing):
                a = make_parlay_leg("prop", description="a", probability=0.5, price=-110, game_id=missing, team=missing,
                                    market="receiving_yards", side="over", line=60.5, player_id="p1")
                self.assertIsNone(a["team"])
                self.assertIsNone(a["game_id"])
        a = make_parlay_leg("prop", description="a", probability=0.5, price=-110, game_id=NAN, team=NAN,
                            market="passing_yards", side="over", line=250.5, player_id="p1")
        b = make_parlay_leg("prop", description="b", probability=0.5, price=-110, game_id=NAN, team=NAN,
                            market="receiving_yards", side="over", line=60.5, player_id="p2")
        self.assertEqual(leg_correlation(a, b)["kind"], "different_games",
                         "two legs with no team and no game must not be linked as teammates")

    def test_nan_team_on_a_team_leg_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "team"):
            make_parlay_leg("moneyline", description="x", probability=0.5, price=-110, team=NAN)

    def test_a_stray_nan_line_is_harmless_on_a_moneyline_and_on_a_prop_but_an_error_on_a_spread(self):
        moneyline = make_parlay_leg("moneyline", description="x", probability=0.5, price=-110, team="KC", line=NAN)
        self.assertIsNone(moneyline["line"])
        prop_leg = make_parlay_leg("prop", description="x", probability=0.5, price=-110, market="passing_yards",
                                   side="over", line=NAN, player_id="p")
        self.assertIsNone(prop_leg["line"])
        with self.assertRaisesRegex(ValueError, "finite"):
            make_parlay_leg("spread", description="x", probability=0.5, price=-110, team="KC", line=NAN)

    def test_hand_built_legs_with_nan_fields_never_leak_nan(self):
        a = prop("Travis Kelce", "receiving_yards", "over", 64.5, player_id="te1")
        b = ml("KC")
        dirty = {**a, "line": NAN, "player_name": NAN, "confidence": None, "team": "kc"}
        result = evaluate_parlay_legs([dirty, b])
        assert_all_finite(self, result, allowed_none=_NONE_OK)
        self.assertIsNone(result["legs"][0]["line"])
        self.assertEqual(result["legs"][0]["team"], "KC", "summaries show the normalised team")
        self.assertAlmostEqual(result["legs"][0]["confidence"], 0.7)

    def test_a_team_leg_with_no_team_makes_no_opposing_claim_about_a_prop(self):
        teamless = {"description": "ML ?", "model_probability": 0.6, "price": -150, "game_id": GAME,
                    "market": "moneyline", "side": "home", "team": None}
        found = leg_correlation(teamless, prop("Travis Kelce", "receiving_yards", "over", 64.5))
        self.assertEqual(found["kind"], "same_game_unrelated")
        self.assertEqual(found["rho"], 0.0)
        self.assertNotIn("None", found["note"])


class DuplicateDetectionTests(unittest.TestCase):
    """leg_id includes the team, so one prop entered with the team filled in and once without escaped every duplicate check."""

    def pair(self):
        with_team = prop("Travis Kelce", "receiving_yards", "over", 64.5, player_id="te1", price=-110)
        without_team = make_parlay_leg("prop", description="Kelce o64.5 (no team)", probability=0.5, price=-115, game_id=GAME,
                                       market="receiving_yards", side="over", line=64.5, player_id="te1")
        self.assertNotEqual(with_team["leg_id"], without_team["leg_id"])
        return with_team, without_team

    def test_legs_conflict_and_can_add_leg_catch_the_same_bet_under_two_ids(self):
        a, b = self.pair()
        messages = legs_conflict([a, b])
        self.assertEqual(len(messages), 1)
        self.assertIn("Duplicate leg", messages[0])
        allowed, message = can_add_leg([a], b)
        self.assertFalse(allowed)
        self.assertIn("already in your parlay", message)

    def test_evaluator_warns_about_the_same_bet_under_two_ids(self):
        a, b = self.pair()
        result = evaluate_parlay_legs([a, b])
        self.assertTrue(any("appears more than once" in w for w in result["warnings"]), result["warnings"])
        self.assertTrue(any(f.startswith("Duplicate leg") for f in result["risk_factors"]))
        self.assertAlmostEqual(result["adjusted_probability"], 0.5, places=9)

    def test_three_copies_are_one_warning_but_two_conflict_messages(self):
        legs = [ml("KC"), ml("KC", price=-140), ml("KC", price=-130)]
        self.assertEqual(sum("Duplicate leg" in m for m in legs_conflict(legs)), 2)
        result = evaluate_parlay_legs(legs)
        self.assertEqual(sum("appears more than once" in w for w in result["warnings"]), 1)

    def test_distinct_bets_are_not_duplicates(self):
        legs = [prop("Travis Kelce", "receiving_yards", "over", 64.5), prop("Travis Kelce", "receiving_yards", "over", 74.5, p=0.3),
                prop("Travis Kelce", "receptions", "over", 5.5)]
        self.assertEqual(legs_conflict(legs), [])
        self.assertFalse(any("appears more than once" in w for w in evaluate_parlay_legs(legs)["warnings"]))


class HonestOutputTests(unittest.TestCase):
    """Strings and warnings must state what the numbers actually are."""

    def test_summary_shows_cents_for_a_fractional_stake_and_whole_dollars_otherwise(self):
        legs = independent_legs([0.6, 0.6])
        self.assertIn("on $7.50 ", evaluate_parlay_legs(legs, stake=7.5)["summary"])
        self.assertIn("on $0.40 ", evaluate_parlay_legs(legs, stake=0.4)["summary"])
        self.assertIn("on $100 ", evaluate_parlay_legs(legs, stake=100)["summary"])
        self.assertIn("on $1,250 ", evaluate_parlay_legs(legs, stake=1250)["summary"])

    def test_summary_says_when_the_price_is_the_unrepriced_leg_product_of_correlated_legs(self):
        correlated = evaluate_parlay_legs([ml("KC", p=0.62), spread("KC", -3.5, p=0.5)])
        self.assertIn("product of the leg prices", correlated["summary"])
        self.assertNotIn("product of the leg prices", evaluate_parlay_legs([ml("KC", p=0.62), spread("KC", -3.5, p=0.5)], book_price=200)["summary"])
        self.assertNotIn("product of the leg prices", evaluate_parlay_legs(independent_legs([0.6, 0.6]))["summary"])

    def test_imprecise_simulation_is_flagged_but_ordinary_parlays_are_not(self):
        one_block = [
            make_parlay_leg("prop", description=f"WR{i} over", probability=0.6, price=-110, game_id=GAME, team="KC",
                            market="receiving_yards", side="over", line=10.5 + i, player_id=f"wr{i}")
            for i in range(30)
        ]
        result = evaluate_parlay_legs(one_block)
        self.assertGreater(result["probability_std_error"] / result["adjusted_probability"], 0.05)
        self.assertTrue(any("imprecise" in w for w in result["warnings"]), result["warnings"])
        stack = evaluate_parlay_legs([ml("KC", p=0.62), prop("Patrick Mahomes", "passing_tds", "over", 1.5, p=0.6, player_id="qb1"),
                                      prop("Travis Kelce", "receiving_tds", "over", 0.5, p=0.5, player_id="te1")])
        self.assertFalse(any("imprecise" in w for w in stack["warnings"]))

    def test_a_middle_is_not_described_as_the_same_bet(self):
        """can_add_leg said a middle (opposite sides of one game) was 'nearly the same bet'."""
        for existing, new in (
            (spread("KC", -3.5), spread("BUF", 7.5, p=0.6)),
            (ml("KC", p=0.6), spread("BUF", 3.5, p=0.5)),
            (prop("Travis Kelce", "receiving_yards", "over", 54.5), prop("Travis Kelce", "receiving_yards", "under", 74.5, p=0.6)),
        ):
            with self.subTest(new=new["description"]):
                allowed, message = can_add_leg([existing], new)
                self.assertTrue(allowed)
                self.assertNotIn("nearly the same bet", message)
                self.assertIn("works almost directly against", message)
                self.assertIn("rho -0.95", message)
        allowed, message = can_add_leg([ml("KC")], spread("KC", -3.5))
        self.assertIn("nearly the same bet", message, "genuinely redundant legs still get the original heads-up")

    def test_opposing_teams_wording_and_tied_leg_wording(self):
        qbs = leg_correlation(prop("Patrick Mahomes", "passing_yards", "over", 274.5),
                              prop("Josh Allen", "passing_yards", "over", 259.5, team="BUF"))
        self.assertIn("opposing teams", qbs["note"])
        inconsistent = leg_correlation(ml("KC", p=0.40), spread("KC", -3.5, p=0.55))
        self.assertIn("less likely of the two legs (40.0%", inconsistent["note"],
                      "the joint is min(p), which is the moneyline here, not 'the cover alone'")


class AnytimeTdCorrelationTests(unittest.TestCase):
    """The Player Props page offers "Anytime TD (Rush + Rec)"; it must not be priced as an unrelated leg."""

    def test_team_result_links_to_anytime_td_like_any_other_td_leg(self):
        for team_leg in (ml("KC"), spread("KC", -3.5)):
            with self.subTest(leg=team_leg["description"]):
                found = leg_correlation(team_leg, prop("Some Player", "anytime_td", "over", 0.5))
                self.assertEqual(found["kind"], "team_result_and_player_scoring", found["note"])
                self.assertAlmostEqual(found["rho"], 0.24, places=4)
                self.assertIn(found["kind"], CORRELATION_KINDS)

    def test_opposing_team_is_negative(self):
        found = leg_correlation(ml("BUF", p=0.4), prop("Some Player", "anytime_td", "over", 0.5, team="KC"))
        self.assertEqual(found["kind"], "team_result_and_opposing_player")
        self.assertLess(found["rho"], 0)

    def test_same_player_td_overlap_is_strong_and_symmetric(self):
        for other in ("rushing_tds", "receiving_tds"):
            with self.subTest(other=other):
                a = prop("Same Guy", "anytime_td", "over", 0.5, player_id="p1")
                b = prop("Same Guy", other, "over", 0.5, player_id="p1")
                found = leg_correlation(a, b)
                self.assertEqual(found["kind"], "same_player_td_overlap", found["note"])
                self.assertAlmostEqual(found["rho"], 0.80, places=4)
                self.assertEqual(found, leg_correlation(b, a))
                self.assertIn(found["kind"], CORRELATION_KINDS)

    def test_same_player_volume_and_anytime_td(self):
        for market, kind in (("receiving_yards", "same_player_receiving_volume_and_score"), ("receptions", "same_player_receiving_volume_and_score")):
            with self.subTest(market=market):
                found = leg_correlation(prop("WR One", "anytime_td", "over", 0.5, player_id="w1"), prop("WR One", market, "over", 5.5, player_id="w1"))
                self.assertEqual(found["kind"], kind, found["note"])
                self.assertGreater(found["rho"], 0.2)

    def test_two_teams_anytime_tds_move_together_in_a_shootout(self):
        found = leg_correlation(prop("Home Guy", "anytime_td", "over", 0.5, team="KC"), prop("Away Guy", "anytime_td", "over", 0.5, team="BUF"))
        self.assertEqual(found["kind"], "opposing_scoring_shootout", found["note"])
        self.assertGreater(found["rho"], 0)


if __name__ == "__main__":
    unittest.main()
