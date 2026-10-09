"""Contract for modules.nfl_prop_analysis -- the pure-logic engine behind the NFL Player Props page.

Every test runs offline against hand-built player dicts that carry the real row
shape of ``fantasy.projections.load_forward_projections``; only the two
``LiveDataTests`` touch the network and they skip themselves when the data
cannot be loaded.
"""

import json
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from betting import prop_model
from betting.odds_math import expected_value, hold, implied_probability, remove_vig_two_way
from betting.prop_model import STAT_FIELDS, evaluate_prop, stat_distribution
from fantasy import player_status

from modules import nfl_prop_analysis as npa
from modules.odds_input import evaluate_price

NATIVE_LABELS = [label for label, spec in npa.STAT_CATEGORIES.items() if spec["market"] in STAT_FIELDS]


def _history(base: float) -> list[dict]:
    """17 weeks of deterministic week-to-week wobble, in the real ``history`` row shape."""
    return [
        {
            "week": week,
            "points": round(base + ((week * 7) % 11 - 5) * 1.5, 1),
            "targets": 6.0,
            "carries": 12.0,
            "target_share": 0.2,
            "air_yards_share": 0.1,
            "opponent_defense_rank": (week * 5) % 32 + 1,
        }
        for week in range(1, 18)
    ]


def make_player(player_id, name, position, team, games=17, **stats) -> dict:
    """A player dict with the keys of a real forward-projection row."""
    row = {
        "player_id": player_id,
        "name": name,
        "position": position,
        "team": team,
        "season": 2026,
        "prior_season": 2025,
        "projection_season": 2026,
        "carries": 0.0,
        "pass_attempts": 0.0,
        "passing_tds": 0.0,
        "passing_yards": 0.0,
        "interceptions": 0.0,
        "receiving_tds": 0.0,
        "receiving_yards": 0.0,
        "receptions": 0.0,
        "rushing_tds": 0.0,
        "rushing_yards": 0.0,
        "targets": 0.0,
        "projection": 240.0,
        "expected_fantasy_points": 240.0,
        "games_played": games,
        "prior_games_played": games,
        "points_per_game": 15.0,
        "floor": 170.0,
        "median": 235.0,
        "ceiling": 330.0,
        "expected_games": 13.5,
        "projection_confidence": 0.9,
        "drivers": ["1,000 rushing yards", "60 receptions", "10 total TDs"],
        "projection_basis": (
            "2026 projection from 2025 actuals (17 games -> 13.5 expected); "
            "rate regressed to the position median, 35% market-reconciled"
        ),
        "scoring_mode": "ppr",
        "history": _history(15.0),
        "age": 27.0,
        "years_experience": 5,
        "adp": 40.0,
    }
    row.update(stats)
    return row


def qb_pocket(**over):
    base = dict(passing_yards=4250.0, passing_tds=30.0, pass_attempts=560.0, rushing_yards=180.0, rushing_tds=1.0, carries=45.0)
    base.update(over)
    return make_player("00-KC-QB1", "Pat Pocket", "QB", "KC", **base)


def qb_runner(**over):
    base = dict(passing_yards=3700.0, passing_tds=26.0, pass_attempts=480.0, rushing_yards=560.0, rushing_tds=12.0, carries=110.0)
    base.update(over)
    return make_player("00-BUF-QB1", "Rob Runner", "QB", "BUF", games=16, **base)


def rb_one(**over):
    base = dict(
        rushing_yards=1190.0, rushing_tds=9.0, carries=260.0, receptions=48.0, receiving_yards=410.0, receiving_tds=3.0, targets=60.0,
    )
    base.update(over)
    return make_player("00-KC-RB1", "Rex Back", "RB", "KC", **base)


def wr_one(**over):
    base = dict(receiving_yards=1300.0, receiving_tds=9.0, receptions=95.0, targets=140.0)
    base.update(over)
    return make_player("00-KC-WR1", "Wes Receiver", "WR", "KC", **base)


def _skill_pool() -> list[dict]:
    """A small league: a rich KC offense, a running-QB BUF offense, and players who must never be offered a stat."""
    return [
        qb_pocket(),
        make_player("00-KC-QB2", "Bo Backup", "QB", "KC", games=1, passing_yards=12.0, passing_tds=0.0, rushing_yards=2.0),
        rb_one(),
        make_player("00-KC-RB2", "Ray Second", "RB", "KC", rushing_yards=310.0, rushing_tds=2.0, carries=70.0, receptions=15.0, receiving_yards=110.0, receiving_tds=0.0),
        wr_one(),
        make_player("00-KC-WR2", "Will Two", "WR", "KC", receiving_yards=820.0, receiving_tds=5.0, receptions=60.0, targets=90.0),
        make_player("00-KC-WR3", "Walt Three", "WR", "KC", receiving_yards=540.0, receiving_tds=3.0, receptions=41.0, targets=65.0),
        make_player("00-KC-WR4", "Trick Thrower", "WR", "KC", receiving_yards=300.0, receiving_tds=1.0, receptions=24.0, passing_yards=800.0, passing_tds=6.0),
        make_player("00-KC-TE1", "Tex End", "TE", "KC", receiving_yards=760.0, receiving_tds=6.0, receptions=70.0, targets=95.0),
        make_player("00-KC-K1", "Kyle Kicker", "K", "KC", extra_points_made=40.0),
        qb_runner(),
        make_player("00-BUF-RB1", "Bill Back", "RB", "BUF", rushing_yards=900.0, rushing_tds=7.0, carries=210.0, receptions=30.0, receiving_yards=240.0, receiving_tds=1.0),
        make_player("00-BUF-WR1", "Buff Wideout", "WR", "BUF", receiving_yards=980.0, receiving_tds=6.0, receptions=70.0),
        # A team whose only roster entries can never be offered a prop.
        make_player("00-NYJ-K1", "Nyj Kicker", "K", "NYJ"),
        {"player_id": "DEF-NYJ", "name": "NYJ Defense", "position": "DST", "team": "NYJ", "games_played": 17, "passing_yards": None, "rushing_yards": None},
    ]


def assert_clean(testcase: unittest.TestCase, value, path: str = "result") -> None:
    """Recursively assert no NaN/inf floats and no ``None`` anywhere in a returned structure."""
    if value is None:
        testcase.fail(f"None leaked at {path}")
    if isinstance(value, float):
        testcase.assertTrue(math.isfinite(value), f"non-finite float at {path}: {value!r}")
    elif isinstance(value, dict):
        for key, inner in value.items():
            assert_clean(testcase, inner, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            assert_clean(testcase, inner, f"{path}[{index}]")


class IsolatedStatusCase(unittest.TestCase):
    """Point the player-status overlay at an empty temp file (also when run without pytest's conftest)."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self._status_path = Path(self._dir.name) / "player_status.json"
        self._status_path.write_text("{}", encoding="utf-8")
        patcher = mock.patch.object(player_status, "STATUS_PATH", self._status_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        player_status.clear_cache()
        self.addCleanup(player_status.clear_cache)

    def set_status(self, player_id: str, status: str) -> None:
        self._status_path.write_text(
            json.dumps({player_id: {"status": status, "last_updated": "2026-10-01T00:00:00Z", "source": "test"}}),
            encoding="utf-8",
        )
        player_status.clear_cache()


# ---------------------------------------------------------------------------
# Category table
# ---------------------------------------------------------------------------


class CategoryContractTests(unittest.TestCase):
    def test_required_labels_in_order(self):
        self.assertEqual(
            list(npa.STAT_CATEGORIES),
            [
                "Passing Yards", "Passing TDs", "Rushing Yards", "Rushing TDs",
                "Receiving Yards", "Receiving TDs", "Receptions", "Anytime TD (Rush + Rec)",
            ],
        )
        self.assertEqual(list(npa.CATEGORY_LABELS), list(npa.STAT_CATEGORIES))

    def test_every_entry_has_the_documented_keys_and_sane_values(self):
        for label, spec in npa.STAT_CATEGORIES.items():
            with self.subTest(label=label):
                for key in ("market", "positions", "unit", "family", "min_line", "max_line", "step", "blurb"):
                    self.assertIn(key, spec)
                self.assertTrue(spec["market"] in STAT_FIELDS or spec["market"] == "anytime_td")
                self.assertIsInstance(spec["positions"], tuple)
                self.assertIn(spec["unit"], ("yds", "TDs", "rec"))
                self.assertIn(spec["family"], ("gaussian", "poisson"))
                self.assertEqual(spec["step"], 0.5)
                self.assertGreater(spec["min_line"], 0)
                self.assertLessEqual(spec["min_line"], spec["max_line"])
                self.assertTrue(spec["blurb"])

    def test_family_matches_what_the_engine_uses_for_native_markets(self):
        player = rb_one()
        for label in NATIVE_LABELS:
            with self.subTest(label=label):
                engine_family = stat_distribution(player, npa.STAT_CATEGORIES[label]["market"])["family"]
                self.assertEqual(npa.STAT_CATEGORIES[label]["family"], engine_family)
        self.assertEqual(npa.STAT_CATEGORIES["Anytime TD (Rush + Rec)"]["family"], "poisson")

    def test_anytime_td_is_only_priced_at_half_a_touchdown(self):
        spec = npa.STAT_CATEGORIES["Anytime TD (Rush + Rec)"]
        self.assertEqual((spec["min_line"], spec["max_line"]), (0.5, 0.5))


# ---------------------------------------------------------------------------
# Pool, eligibility, ordering
# ---------------------------------------------------------------------------


class EligibilityAndPoolTests(IsolatedStatusCase):
    def setUp(self):
        super().setUp()
        self.pool = _skill_pool()

    def names(self, players):
        return [p["name"] for p in players]

    def test_wr_is_never_offered_passing_markets_even_with_passing_volume(self):
        thrower = next(p for p in self.pool if p["name"] == "Trick Thrower")
        self.assertGreater(thrower["passing_yards"] / thrower["games_played"], 20)
        self.assertFalse(npa.is_eligible(thrower, "Passing Yards"))
        self.assertFalse(npa.is_eligible(thrower, "Passing TDs"))
        for category in ("Passing Yards", "Passing TDs"):
            self.assertNotIn("Trick Thrower", self.names(npa.players_for_team(self.pool, "KC", category)))

    def test_qb_is_offered_rushing_only_when_he_has_rushing_volume(self):
        pocket, runner = qb_pocket(), qb_runner()
        # 180 yds / 17 games = 10.6 per game clears the 5/game bar; 1 TD / 17 games = 0.06 clears 0.05.
        self.assertTrue(npa.is_eligible(pocket, "Rushing Yards"))
        # a QB with no rushing TDs at all is not offered the TD markets
        no_td = qb_pocket(rushing_tds=0.0, rushing_yards=40.0)
        self.assertFalse(npa.is_eligible(no_td, "Rushing TDs"))
        self.assertFalse(npa.is_eligible(no_td, "Rushing Yards"))
        self.assertFalse(npa.is_eligible(no_td, "Anytime TD (Rush + Rec)"))
        self.assertTrue(npa.is_eligible(runner, "Rushing TDs"))
        self.assertTrue(npa.is_eligible(runner, "Anytime TD (Rush + Rec)"))
        # ... and a QB is never offered receiving markets
        self.assertFalse(npa.is_eligible(runner, "Receiving Yards"))

    def test_low_volume_and_non_skill_players_are_filtered_out(self):
        backup = next(p for p in self.pool if p["name"] == "Bo Backup")
        self.assertFalse(npa.is_eligible(backup, "Passing Yards"))  # 12 yds in 1 game is below the volume floor
        kicker = next(p for p in self.pool if p["name"] == "Kyle Kicker")
        defence = next(p for p in self.pool if p["position"] == "DST")
        for category in npa.STAT_CATEGORIES:
            self.assertFalse(npa.is_eligible(kicker, category))
            self.assertFalse(npa.is_eligible(defence, category))
        self.assertEqual(npa.eligible_categories(kicker), [])

    def test_players_with_no_games_are_not_eligible(self):
        self.assertFalse(npa.is_eligible(wr_one(games_played=0, prior_games_played=0), "Receiving Yards"))

    def test_unknown_category_is_not_eligible_and_players_for_team_raises(self):
        self.assertFalse(npa.is_eligible(rb_one(), "Punting Yards"))
        with self.assertRaises(ValueError):
            npa.players_for_team(self.pool, "KC", "Punting Yards")

    def test_teams_in_pool_only_lists_teams_with_an_offerable_player(self):
        self.assertEqual(npa.teams_in_pool(self.pool), ["BUF", "KC"])
        self.assertEqual(npa.teams_in_pool([]), [])
        self.assertEqual(npa.teams_in_pool(None), [])

    def test_players_for_team_sorts_by_most_volume_first(self):
        # Ray Second (110 yds / 17 games = 6.5 per game) is below the 8-per-game volume floor and is left out.
        self.assertEqual(
            self.names(npa.players_for_team(self.pool, "KC", "Receiving Yards")),
            ["Wes Receiver", "Will Two", "Tex End", "Walt Three", "Rex Back", "Trick Thrower"],
        )
        self.assertEqual(
            self.names(npa.players_for_team(self.pool, "KC", "Rushing Yards")),
            ["Rex Back", "Ray Second", "Pat Pocket"],
        )
        self.assertEqual(self.names(npa.players_for_team(self.pool, "KC", "Passing Yards")), ["Pat Pocket"])

    def test_players_for_team_without_category_lists_everyone_offerable_by_projection(self):
        pool = _skill_pool()
        pool[0]["projection"] = 300.0  # the KC QB
        pool[2]["projection"] = 280.0  # the KC RB1
        names = self.names(npa.players_for_team(pool, "KC"))
        self.assertEqual(names[:2], ["Pat Pocket", "Rex Back"])
        self.assertNotIn("Kyle Kicker", names)
        self.assertNotIn("Bo Backup", names)

    def test_team_codes_are_normalised(self):
        self.assertEqual(npa.normalize_team("LAR"), "LA")
        self.assertEqual(npa.normalize_team(" wsh "), "WAS")
        self.assertEqual(npa.normalize_team("kc"), "KC")
        self.assertEqual(npa.normalize_team(None), "")
        self.assertEqual(npa.players_for_team(self.pool, "kc", "Passing Yards")[0]["name"], "Pat Pocket")
        self.assertIn("Kansas City Chiefs", npa.team_label("KC"))

    def test_player_label_and_lookup(self):
        wr = wr_one()
        self.assertEqual(npa.player_label(wr), "Wes Receiver (WR)")
        self.assertEqual(npa.player_label({"name": "Mystery"}), "Mystery")
        self.assertIs(npa.find_player(self.pool, "00-KC-WR1"), next(p for p in self.pool if p["player_id"] == "00-KC-WR1"))
        self.assertIsNone(npa.find_player(self.pool, "missing"))
        self.assertIsNone(npa.find_player(None, "x"))


class LoadPlayerPoolTests(unittest.TestCase):
    def setUp(self):
        npa.clear_player_pool_cache()
        self.addCleanup(npa.clear_player_pool_cache)

    def test_failure_returns_empty_list_and_never_raises(self):
        with mock.patch.object(npa, "load_forward_projections", side_effect=RuntimeError("offline")):
            self.assertEqual(npa.load_player_pool(), [])
        with mock.patch.object(npa, "load_forward_projections", side_effect=OSError("disk")):
            self.assertEqual(npa.load_player_pool(2024), [])
        self.assertEqual(npa.load_player_pool("not a season"), [])

    def test_default_season_is_the_last_completed_one(self):
        with mock.patch.object(npa, "upcoming_season", return_value=2031), mock.patch.object(
            npa, "load_forward_projections", return_value=[{"player_id": "a"}]
        ) as loader:
            rows = npa.load_player_pool()
        loader.assert_called_once_with(season=2030)
        self.assertEqual(rows, [{"player_id": "a"}])

    def test_explicit_season_is_passed_through_and_non_dict_rows_dropped(self):
        with mock.patch.object(npa, "load_forward_projections", return_value=[{"player_id": "a"}, "junk", None]) as loader:
            rows = npa.load_player_pool(2024)
        loader.assert_called_once_with(season=2024)
        self.assertEqual(rows, [{"player_id": "a"}])

    def test_success_is_cached_but_an_outage_is_not(self):
        with mock.patch.object(npa, "load_forward_projections", return_value=[{"player_id": "a"}]) as loader:
            npa.load_player_pool(2024)
            npa.load_player_pool(2024)
        self.assertEqual(loader.call_count, 1)
        npa.clear_player_pool_cache()
        with mock.patch.object(npa, "load_forward_projections", return_value=[]) as empty_loader:
            self.assertEqual(npa.load_player_pool(2024), [])
            self.assertEqual(npa.load_player_pool(2024), [])
        self.assertEqual(empty_loader.call_count, 2)  # an empty result is retried, never cached


# ---------------------------------------------------------------------------
# project_stat / suggest_line
# ---------------------------------------------------------------------------


class ProjectionTests(IsolatedStatusCase):
    def test_native_projection_matches_the_engine_distribution(self):
        player = rb_one()
        for label in NATIVE_LABELS:
            with self.subTest(label=label):
                market = npa.STAT_CATEGORIES[label]["market"]
                dist = stat_distribution(player, market)
                projection = npa.project_stat(player, label)
                for key in ("mean", "stdev", "volatility", "games", "family"):
                    self.assertEqual(projection[key], dist[key], key)
                # a ~0-mean count stat has no defined cv in the engine (None); the page-facing value is capped, never None
                expected_cv = dist["cv"] if dist["cv"] is not None else npa.CV_UNDEFINED
                self.assertAlmostEqual(projection["cv"], expected_cv, places=4)
                self.assertAlmostEqual(projection["per_game_rate"], player[STAT_FIELDS[market]] / 17, places=3)
                self.assertEqual(projection["season_total"], player[STAT_FIELDS[market]])

    def test_projection_carries_the_documented_keys(self):
        projection = npa.project_stat(rb_one(), "Rushing Yards")
        for key in (
            "player_id", "name", "team", "position", "category", "market", "unit", "family", "mean", "stdev", "cv",
            "volatility", "games", "season_total", "per_game_rate", "data_season", "confidence", "confidence_label",
            "risk_tier", "volatility_label", "projected_fantasy_points", "projection", "fantasy_points_per_game",
            "expected_games", "projection_confidence", "projection_basis", "drivers", "status", "status_factor",
            "status_note", "eligible", "components",
        ):
            self.assertIn(key, projection)
        self.assertEqual(projection["category"], "Rushing Yards")
        self.assertEqual(projection["data_season"], 2025)
        self.assertEqual(projection["projected_fantasy_points"], 240.0)
        self.assertEqual(projection["drivers"][0], "1,000 rushing yards")
        self.assertIn("rate regressed", projection["projection_basis"])
        self.assertEqual(projection["status"], "HEALTHY")
        self.assertTrue(projection["eligible"])
        self.assertEqual(projection["unit"], "yds")

    def test_anytime_td_composes_the_two_poisson_rates(self):
        player = rb_one(rushing_tds=6.0, receiving_tds=3.0, games_played=12, prior_games_played=12)
        projection = npa.project_stat(player, "Anytime TD (Rush + Rec)")
        self.assertEqual(projection["market"], "anytime_td")
        self.assertEqual(projection["family"], "poisson")
        self.assertAlmostEqual(projection["mean"], (6.0 + 3.0) / 12, places=3)
        self.assertAlmostEqual(projection["stdev"], math.sqrt(0.75), places=3)
        self.assertAlmostEqual(projection["cv"], math.sqrt(0.75) / 0.75, places=3)
        self.assertEqual(projection["components"], {"rushing_td_per_game": 0.5, "receiving_td_per_game": 0.25})
        self.assertEqual(projection["season_total"], 9.0)
        # passing TDs never count towards the market
        qb = qb_runner(passing_tds=40.0, rushing_tds=4.0, games_played=16)
        self.assertAlmostEqual(npa.project_stat(qb, "Anytime TD (Rush + Rec)")["mean"], 4 / 16, places=3)

    def test_volatility_is_the_engines_and_sample_is_the_games_played(self):
        projection = npa.project_stat(wr_one(), "Receptions")
        self.assertEqual(projection["games"], 17.0)
        self.assertGreaterEqual(projection["volatility"], 0.0)
        self.assertLessEqual(projection["volatility"], 1.0)
        self.assertIn(projection["volatility_label"], ("Low", "Moderate", "High"))

    def test_games_fall_back_to_prior_games_played_like_the_engine(self):
        player = wr_one()
        player["games_played"] = None
        self.assertEqual(npa.project_stat(player, "Receiving Yards")["games"], 17.0)

    def test_zero_games_and_unsupported_category_raise_value_error(self):
        player = wr_one(games_played=0, prior_games_played=0)
        with self.assertRaises(ValueError) as caught:
            npa.project_stat(player, "Receiving Yards")
        self.assertIn("no games played", str(caught.exception))
        with self.assertRaises(ValueError):
            npa.evaluate_line(player, "Receiving Yards", 60.5)
        with self.assertRaises(ValueError) as caught:
            npa.project_stat(wr_one(), "Punting Yards")
        self.assertIn("Punting Yards", str(caught.exception))
        with self.assertRaises(ValueError):
            npa.evaluate_line(wr_one(), "Punting Yards", 60.5)

    def test_ineligible_request_does_not_raise_but_is_flagged(self):
        projection = npa.project_stat(wr_one(), "Passing Yards")
        self.assertFalse(projection["eligible"])
        self.assertEqual(projection["mean"], 0.0)
        evaluation = npa.evaluate_line(wr_one(), "Passing Yards", 200.5)
        self.assertTrue(any("not normally offered" in w for w in evaluation["warnings"]))
        self.assertLess(evaluation["probability_over"], 0.001)

    def test_questionable_status_trims_the_rate_and_out_zeroes_it(self):
        player = wr_one()
        healthy = npa.project_stat(player, "Receiving Yards")
        self.set_status(player["player_id"], "QUESTIONABLE")
        trimmed = npa.project_stat(player, "Receiving Yards")
        self.assertEqual(trimmed["status"], "QUESTIONABLE")
        self.assertAlmostEqual(trimmed["mean"], healthy["mean"] * 0.88, delta=0.02)
        self.assertAlmostEqual(trimmed["status_factor"], 0.88, places=3)
        self.assertEqual(trimmed["per_game_rate"], healthy["per_game_rate"])  # the raw rate is never altered
        self.assertIn("88%", trimmed["status_note"])
        self.set_status(player["player_id"], "OUT")
        out = npa.project_stat(player, "Receiving Yards")
        self.assertEqual(out["status"], "OUT")
        self.assertEqual(out["mean"], 0.0)
        self.assertIn("OUT", out["status_note"])
        self.assertEqual(npa.player_status(player), "OUT")

    def test_suggest_line_is_the_nearest_half_point_and_respects_the_floor(self):
        def line(mean, category="Passing Yards"):
            return npa.suggest_line({"mean": mean, "category": category})

        self.assertEqual(line(249.8), 249.5)  # 0.3 away vs 0.7 away from 250.5
        self.assertEqual(line(250.1), 250.5)
        self.assertEqual(line(249.5), 249.5)
        self.assertEqual(line(250.0), 250.5)  # an exact tie rounds up
        self.assertEqual(line(0.29, "Rushing TDs"), 0.5)
        self.assertEqual(line(5.4, "Receptions"), 5.5)
        # never below the category minimum
        self.assertEqual(line(1.2, "Rushing Yards"), npa.STAT_CATEGORIES["Rushing Yards"]["min_line"])
        self.assertEqual(line(0.0, "Receptions"), 0.5)
        self.assertEqual(line(-3.3, "Rushing Yards"), 4.5)
        # nor above the maximum (and Anytime TD is pinned to 0.5)
        self.assertEqual(line(9.0, "Passing TDs"), 5.5)
        self.assertEqual(line(0.93, "Anytime TD (Rush + Rec)"), 0.5)
        self.assertEqual(line(1.7, "Anytime TD (Rush + Rec)"), 0.5)

    def test_suggest_line_accepts_a_real_projection(self):
        projection = npa.project_stat(wr_one(), "Receiving Yards")
        suggested = npa.suggest_line(projection)
        self.assertEqual(suggested % 1, 0.5)
        self.assertLess(abs(suggested - projection["mean"]), 1.0)


# ---------------------------------------------------------------------------
# evaluate_line
# ---------------------------------------------------------------------------

#: The player each category is exercised with (all genuinely eligible).
CATEGORY_PLAYERS = {
    "Passing Yards": qb_pocket,
    "Passing TDs": qb_pocket,
    "Rushing Yards": rb_one,
    "Rushing TDs": rb_one,
    "Receiving Yards": wr_one,
    "Receiving TDs": wr_one,
    "Receptions": wr_one,
    "Anytime TD (Rush + Rec)": rb_one,
}

REQUIRED_KEYS = (
    "player_id", "name", "team", "position", "category", "market", "line", "over_price", "under_price",
    "probability_over", "probability_under", "push_probability", "market_fair_probability_over",
    "market_fair_probability_under", "edge_over", "edge_under", "ev_over", "ev_under", "implied_probability_over",
    "implied_probability_under", "value_gap_over", "value_gap_under", "recommended_side", "recommended_ev",
    "recommended_edge", "confidence", "confidence_label", "volatility", "volatility_label", "cv", "risk_tier",
    "projection", "z_score", "verdict", "verdict_tone", "p10", "p50", "p90", "reasoning", "warnings",
)


class EvaluateLineTests(IsolatedStatusCase):
    def test_all_eight_categories_round_trip(self):
        for label, factory in CATEGORY_PLAYERS.items():
            with self.subTest(label=label):
                player = factory()
                projection = npa.project_stat(player, label)
                line = npa.suggest_line(projection)
                result = npa.evaluate_line(player, label, line)
                for key in REQUIRED_KEYS:
                    self.assertIn(key, result)
                self.assertEqual(result["category"], label)
                self.assertEqual(result["line"], line)
                self.assertEqual(result["over_price"], -110.0)
                self.assertEqual(result["under_price"], -110.0)
                self.assertEqual(result["name"], player["name"])
                self.assertEqual(result["team"], player["team"])
                self.assertEqual(result["projection"]["mean"], projection["mean"])
                self.assertIn(result["recommended_side"], ("over", "under"))
                self.assertIn(result["confidence_label"], ("Low", "Moderate", "High"))
                self.assertIn(result["volatility_label"], ("Low", "Moderate", "High"))
                self.assertIn(result["risk_tier"], ("low", "medium", "high"))
                self.assertIn(result["verdict_tone"], ("success", "info", "neutral", "danger"))
                self.assertTrue(0.0 <= result["confidence"] <= 1.0)
                self.assertTrue(0.0 <= result["volatility"] <= 1.0)
                self.assertGreaterEqual(len(result["reasoning"]), 5)
                self.assertTrue(result["headline"])
                # A line at the mean of a symmetric-ish curve is close to a coin flip.
                self.assertTrue(0.2 < result["probability_over"] < 0.8)

    def test_native_markets_match_betting_prop_model_evaluate_prop_exactly(self):
        for label in NATIVE_LABELS:
            player = CATEGORY_PLAYERS[label]()
            market = npa.STAT_CATEGORIES[label]["market"]
            mean = npa.project_stat(player, label)["mean"]
            for offset in (-1.0, 0.0, 1.0):
                line = max(0.5, math.floor(mean) + 0.5 + offset)
                for over_price, under_price in ((-110.0, -110.0), (-125.0, 105.0), (140.0, -170.0)):
                    with self.subTest(label=label, line=line, prices=(over_price, under_price)):
                        engine = evaluate_prop(
                            player, {"market": market, "line": line, "over_price": over_price, "under_price": under_price}
                        )
                        result = npa.evaluate_line(player, label, line, over_price, under_price)
                        self.assertEqual(result["probability_over"], engine["model_probability_over"])
                        self.assertEqual(result["probability_under"], engine["model_probability_under"])
                        self.assertEqual(result["confidence"], engine["confidence"])
                        self.assertEqual(result["risk_tier"], engine["risk_tier"])
                        for key in (
                            "market_fair_probability_over", "market_fair_probability_under", "edge_over", "edge_under",
                            "ev_over", "ev_under", "recommended_side", "recommended_edge", "recommended_ev",
                        ):
                            self.assertEqual(result[key], engine[key], key)
                        for key in ("mean", "stdev", "volatility", "games", "family"):
                            self.assertEqual(result["distribution"][key], engine["distribution"][key], key)
                        self.assertEqual(result["push_probability"], 0.0)
                        self.assertEqual(result["status"], engine["status"])

    def test_the_home_rolled_probability_path_agrees_with_the_engine_on_half_point_lines(self):
        """Anytime TD uses _line_probabilities; it must reproduce the engine for every native market."""
        for label in NATIVE_LABELS:
            player = CATEGORY_PLAYERS[label]()
            market = npa.STAT_CATEGORIES[label]["market"]
            dist = stat_distribution(player, market)
            for line in (0.5, 1.5, 4.5, 24.5, 76.5, 249.5):
                with self.subTest(label=label, line=line):
                    engine = evaluate_prop(player, {"market": market, "line": line, "over_price": -110.0, "under_price": -110.0})
                    over, under, push = npa._line_probabilities(dist["mean"], dist["stdev"], dist["family"], line)
                    self.assertEqual((over, under, push), (engine["model_probability_over"], engine["model_probability_under"], 0.0))

    def test_over_plus_under_is_one_on_half_point_lines(self):
        for label, factory in CATEGORY_PLAYERS.items():
            player = factory()
            mean = npa.project_stat(player, label)["mean"]
            for line in (0.5, 1.5, math.floor(mean) + 0.5, math.floor(mean) + 8.5):
                with self.subTest(label=label, line=line):
                    result = npa.evaluate_line(player, label, line)
                    self.assertEqual(result["push_probability"], 0.0)
                    self.assertFalse(result["integer_line"])
                    self.assertAlmostEqual(result["probability_over"] + result["probability_under"], 1.0, delta=2e-4)
                    self.assertAlmostEqual(
                        result["market_fair_probability_over"] + result["market_fair_probability_under"], 1.0, delta=2e-4
                    )

    def test_half_point_lines_never_report_a_push_across_a_sweep_of_distributions(self):
        for family in ("poisson", "gaussian"):
            for step in range(1, 400):
                mean = step * 0.37
                stdev = math.sqrt(mean) if family == "poisson" else mean * 0.45
                for line in (0.5, 1.5, 2.5, math.floor(mean) + 0.5, math.floor(mean) + 1.5, math.floor(mean) + 9.5):
                    over, under, push = npa._line_probabilities(mean, stdev, family, line)
                    self.assertEqual(push, 0.0, (family, mean, line, over, under))
                    self.assertAlmostEqual(over + under, 1.0, delta=1.5e-4)

    def test_integer_lines_carry_a_push_probability_and_still_sum_to_one(self):
        for label, factory in CATEGORY_PLAYERS.items():
            player = factory()
            mean = npa.project_stat(player, label)["mean"]
            for line in (1.0, 2.0, float(math.floor(mean)), float(math.floor(mean)) + 1.0):
                if line <= 0:
                    continue
                with self.subTest(label=label, line=line):
                    result = npa.evaluate_line(player, label, line)
                    self.assertTrue(result["integer_line"])
                    self.assertGreaterEqual(result["push_probability"], 0.0)
                    self.assertAlmostEqual(
                        result["probability_over"] + result["probability_under"] + result["push_probability"], 1.0, delta=2e-4
                    )
                    self.assertTrue(any("Integer line" in w for w in result["warnings"]))
        # near the mean the push is a real, non-trivial slice
        receptions = npa.evaluate_line(wr_one(), "Receptions", 5.0)
        self.assertGreater(receptions["push_probability"], 0.10)
        yards = npa.evaluate_line(wr_one(), "Receiving Yards", 76.0)
        self.assertGreater(yards["push_probability"], 0.003)

    def test_integer_line_poisson_matches_closed_form(self):
        player = wr_one()  # 95 receptions / 17 games
        lam = npa.project_stat(player, "Receptions")["mean"]
        k = 5
        pmf = lambda i: math.exp(-lam) * lam**i / math.factorial(i)  # noqa: E731
        cdf_k = sum(pmf(i) for i in range(0, k + 1))
        cdf_below = sum(pmf(i) for i in range(0, k))
        result = npa.evaluate_line(player, "Receptions", 5.0)
        self.assertAlmostEqual(result["push_probability"], pmf(5), delta=1.5e-4)
        self.assertAlmostEqual(result["probability_over"], 1 - cdf_k, delta=1.5e-4)
        self.assertAlmostEqual(result["probability_under"], cdf_below, delta=1.5e-4)

    def test_integer_line_gaussian_agrees_with_the_half_point_lines_around_it(self):
        player = wr_one()
        integer = npa.evaluate_line(player, "Receiving Yards", 76.0)
        above = npa.evaluate_line(player, "Receiving Yards", 76.5)
        below = npa.evaluate_line(player, "Receiving Yards", 75.5)
        self.assertEqual(integer["probability_over"], above["probability_over"])
        self.assertEqual(integer["probability_under"], below["probability_under"])
        self.assertAlmostEqual(
            integer["push_probability"], round(1 - above["probability_over"] - below["probability_under"], 4), places=4
        )

    def test_integer_line_ev_refunds_the_stake_on_a_push(self):
        player = wr_one()
        result = npa.evaluate_line(player, "Receptions", 5.0, -120.0, 100.0, stake=50.0)
        p_over, p_under = result["probability_over"], result["probability_under"]
        self.assertAlmostEqual(result["ev_over"], round(p_over * (50.0 * (100 / 120)) - p_under * 50.0, 2), delta=0.011)
        self.assertAlmostEqual(result["ev_under"], round(p_under * 50.0 - p_over * 50.0, 2), delta=0.011)
        # edges compare the *decided* win probability to the de-vigged fair probability
        decided_over = p_over / (p_over + p_under)
        fair_over, _ = remove_vig_two_way(-120.0, 100.0)
        self.assertAlmostEqual(result["edge_over"], decided_over - fair_over, delta=2e-4)

    def test_anytime_td_matches_the_closed_form(self):
        player = rb_one(rushing_tds=6.0, receiving_tds=3.0, games_played=12, prior_games_played=12)
        lam = 9.0 / 12
        result = npa.evaluate_line(player, "Anytime TD (Rush + Rec)", 0.5)
        self.assertAlmostEqual(result["probability_over"], 1 - math.exp(-lam), delta=1e-4)
        self.assertAlmostEqual(result["probability_under"], math.exp(-lam), delta=1e-4)
        self.assertEqual(result["push_probability"], 0.0)
        self.assertEqual(result["family"], "poisson")
        self.assertAlmostEqual(result["mean"], lam, places=3)
        self.assertAlmostEqual(result["z_score"], (0.5 - lam) / math.sqrt(lam), places=2)
        # at a +150 / -180 price the EV is the textbook number
        priced = npa.evaluate_line(player, "Anytime TD (Rush + Rec)", 0.5, 150.0, -180.0)
        self.assertAlmostEqual(priced["ev_over"], round(priced["probability_over"] * 150 - priced["probability_under"] * 100, 2), delta=0.011)

    def test_anytime_td_with_no_receiving_tds_equals_the_rushing_td_market(self):
        player = rb_one(receiving_tds=0.0)
        anytime = npa.evaluate_line(player, "Anytime TD (Rush + Rec)", 0.5)
        rushing = npa.evaluate_line(player, "Rushing TDs", 0.5)
        for key in ("probability_over", "probability_under", "confidence", "risk_tier", "edge_over", "edge_under", "ev_over", "ev_under"):
            self.assertEqual(anytime[key], rushing[key], key)

    def test_probability_over_never_rises_as_the_line_rises_in_every_category(self):
        for label, factory in CATEGORY_PLAYERS.items():
            player = factory()
            mean = npa.project_stat(player, label)["mean"]
            if npa.STAT_CATEGORIES[label]["unit"] == "yds":
                half_ladder = [max(0.5, math.floor(mean * f) + 0.5) for f in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.9)]
            else:
                half_ladder = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
            half_ladder = sorted(set(half_ladder))
            with self.subTest(label=label, ladder="half-point"):
                probabilities = [npa.evaluate_line(player, label, line)["probability_over"] for line in half_ladder]
                for lower, higher in zip(probabilities, probabilities[1:]):
                    self.assertLessEqual(higher, lower)
                self.assertLess(probabilities[-1], probabilities[0])
            mixed = sorted({*half_ladder, *[float(math.floor(x)) for x in half_ladder if math.floor(x) > 0]})
            with self.subTest(label=label, ladder="mixed integer and half-point"):
                probabilities = [npa.evaluate_line(player, label, line)["probability_over"] for line in mixed]
                for lower, higher in zip(probabilities, probabilities[1:]):
                    self.assertLessEqual(higher, lower)

    def test_ev_equals_betting_odds_math_expected_value(self):
        player = wr_one()
        for stake in (100.0, 25.0, 250.0):
            for over_price, under_price in ((-110.0, -110.0), (-135.0, 115.0), (180.0, -220.0)):
                with self.subTest(stake=stake, prices=(over_price, under_price)):
                    result = npa.evaluate_line(player, "Receiving Yards", 70.5, over_price, under_price, stake=stake)
                    self.assertEqual(result["ev_over"], round(expected_value(result["probability_over"], over_price, stake), 2))
                    self.assertEqual(result["ev_under"], round(expected_value(result["probability_under"], under_price, stake), 2))
                    expected_best = "over" if result["edge_over"] >= result["edge_under"] else "under"
                    self.assertEqual(result["recommended_side"], expected_best)
                    self.assertEqual(result["recommended_ev"], result[f"ev_{expected_best}"])
                    self.assertEqual(result["recommended_edge"], result[f"edge_{expected_best}"])

    def test_market_probabilities_and_value_gaps_follow_odds_input(self):
        result = npa.evaluate_line(rb_one(), "Rushing Yards", 70.5, -125.0, 105.0)
        fair_over, fair_under = remove_vig_two_way(-125.0, 105.0)
        self.assertEqual(result["market_fair_probability_over"], round(fair_over, 4))
        self.assertEqual(result["market_fair_probability_under"], round(fair_under, 4))
        self.assertEqual(result["implied_probability_over"], round(implied_probability(-125.0), 4))
        self.assertEqual(result["implied_probability_under"], round(implied_probability(105.0), 4))
        self.assertEqual(result["hold"], round(hold(-125.0, 105.0), 4))
        for side, price in (("over", -125.0), ("under", 105.0)):
            priced = evaluate_price(result[f"probability_{side}"], price)
            self.assertEqual(result[f"value_gap_{side}"], priced["value_gap"])
            self.assertEqual(result[f"pricing_{side}"]["fair_price"], priced["fair_price"])
        best = result["recommended_side"]
        priced_best = evaluate_price(result[f"probability_{best}"], -125.0 if best == "over" else 105.0)
        self.assertEqual(result["verdict"], priced_best["verdict"])
        self.assertEqual(result["verdict_tone"], priced_best["verdict_tone"])
        self.assertEqual(result["recommended_value_gap"], priced_best["value_gap"])

    def test_z_score_percentiles_and_distribution_block(self):
        player = rb_one()
        result = npa.evaluate_line(player, "Rushing Yards", 90.5)
        mean, stdev = result["mean"], result["stdev"]
        self.assertAlmostEqual(result["z_score"], (90.5 - mean) / stdev, places=2)
        self.assertAlmostEqual(result["line_minus_mean"], 90.5 - mean, places=2)
        self.assertLessEqual(result["p10"], result["p50"])
        self.assertLessEqual(result["p50"], result["p90"])
        self.assertAlmostEqual(result["p50"], mean, delta=0.06)
        self.assertAlmostEqual(result["p90"], mean + 1.28155 * stdev, delta=0.1)
        self.assertEqual(result["percentiles"], {"p10": result["p10"], "p50": result["p50"], "p90": result["p90"]})
        self.assertEqual(
            sorted(result["distribution"]), ["cv", "family", "games", "mean", "stdev", "volatility"]
        )
        poisson = npa.evaluate_line(player, "Receptions", 2.5)
        lam = poisson["mean"]
        cdf = lambda k: sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k + 1))  # noqa: E731
        for key, q in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            k = int(poisson[key])
            self.assertGreaterEqual(cdf(k), q)
            self.assertTrue(k == 0 or cdf(k - 1) < q)

    def test_status_out_collapses_the_over_probability(self):
        for label, factory in CATEGORY_PLAYERS.items():
            player = factory()
            line = npa.suggest_line(npa.project_stat(player, label))
            healthy = npa.evaluate_line(player, label, line)
            self.set_status(player["player_id"], "OUT")
            with self.subTest(label=label):
                out = npa.evaluate_line(player, label, line)
                self.assertEqual(out["status"], "OUT")
                self.assertLess(out["probability_over"], 0.01)
                self.assertGreater(out["probability_under"], 0.99)
                self.assertLess(out["probability_over"], healthy["probability_over"])
                self.assertEqual(out["recommended_side"], "under")
                self.assertTrue(any("OUT" in w for w in out["warnings"]))
                self.assertTrue(any(item["title"] == "Availability" and item["tone"] == "con" for item in out["reasoning"]))
            self.set_status("nobody", "OUT")

    def test_doubtful_status_lowers_the_over_probability_and_is_reported(self):
        player = rb_one()
        healthy = npa.evaluate_line(player, "Rushing Yards", 70.5)
        self.set_status(player["player_id"], "DOUBTFUL")
        doubtful = npa.evaluate_line(player, "Rushing Yards", 70.5)
        self.assertEqual(doubtful["status"], "DOUBTFUL")
        self.assertLess(doubtful["probability_over"], healthy["probability_over"])
        self.assertIn("DOUBTFUL", doubtful["status_note"])
        self.assertTrue(any("DOUBTFUL" in w for w in doubtful["warnings"]))
        # and it still matches the engine, which applies the same overlay
        engine = evaluate_prop(player, {"market": "rushing_yards", "line": 70.5, "over_price": -110.0, "under_price": -110.0})
        self.assertEqual(doubtful["probability_over"], engine["model_probability_over"])
        self.assertEqual(doubtful["confidence"], engine["confidence"])

    def test_bad_inputs_raise_value_error(self):
        player = wr_one()
        for bad_line in (0, -3.5, float("nan"), float("inf"), "abc", None):
            with self.subTest(line=bad_line), self.assertRaises(ValueError):
                npa.evaluate_line(player, "Receiving Yards", bad_line)
        for bad_price in (0, 50, -99, float("nan"), "x"):
            with self.subTest(price=bad_price), self.assertRaises(ValueError):
                npa.evaluate_line(player, "Receiving Yards", 70.5, bad_price, -110.0)
            with self.subTest(under=bad_price), self.assertRaises(ValueError):
                npa.evaluate_line(player, "Receiving Yards", 70.5, -110.0, bad_price)
        for bad_stake in (0, -5, float("nan")):
            with self.subTest(stake=bad_stake), self.assertRaises(ValueError):
                npa.evaluate_line(player, "Receiving Yards", 70.5, stake=bad_stake)

    def test_no_nan_inf_or_none_anywhere_in_any_returned_dict(self):
        players = [qb_pocket(), qb_runner(), rb_one(), wr_one(), make_player("00-X", "Zero Volume", "WR", "KC")]
        for player in players:
            for label in npa.STAT_CATEGORIES:
                assert_clean(self, npa.project_stat(player, label), f"project_stat[{player['name']}/{label}]")
                for line in (0.5, 1.0, 4.5, 24.5, 76.0, 249.5, 400.5):
                    for prices in ((-110.0, -110.0), (-300.0, 240.0), (100.0, 100.0)):
                        result = npa.evaluate_line(player, label, line, *prices, pool=_skill_pool())
                        assert_clean(self, result, f"evaluate_line[{player['name']}/{label}/{line}/{prices}]")
        player = rb_one()
        self.set_status(player["player_id"], "OUT")
        for label in npa.STAT_CATEGORIES:
            assert_clean(self, npa.evaluate_line(player, label, 0.5), f"OUT[{label}]")

    def test_warnings_flag_thin_samples_extreme_lines_and_odd_prices(self):
        thin = npa.evaluate_line(wr_one(games_played=3, prior_games_played=3, receiving_yards=210.0), "Receiving Yards", 70.5)
        self.assertTrue(any("Thin sample" in w and "3 game" in w for w in thin["warnings"]))
        far = npa.evaluate_line(wr_one(), "Receiving Yards", 220.5)
        self.assertTrue(any("standard deviations from the projection" in w for w in far["warnings"]))
        outside = npa.evaluate_line(wr_one(), "Receiving Yards", 400.5)
        self.assertTrue(any("outside the usual" in w for w in outside["warnings"]))
        negative_hold = npa.evaluate_line(wr_one(), "Receiving Yards", 70.5, 150.0, 150.0)
        self.assertTrue(any("negative hold" in w for w in negative_hold["warnings"]))
        steep = npa.evaluate_line(wr_one(), "Receiving Yards", 70.5, -200.0, -200.0)
        self.assertTrue(any("hold" in w for w in steep["warnings"]))
        clean = npa.evaluate_line(wr_one(), "Receiving Yards", 76.5)
        self.assertEqual(clean["warnings"], [])

    def test_confidence_and_volatility_labels(self):
        self.assertEqual(npa.confidence_label(0.93), "High")
        self.assertEqual(npa.confidence_label(0.75), "High")
        self.assertEqual(npa.confidence_label(0.60), "Moderate")
        self.assertEqual(npa.confidence_label(0.50), "Moderate")
        self.assertEqual(npa.confidence_label(0.30), "Low")
        self.assertEqual(npa.volatility_label(0.85), "High")
        self.assertEqual(npa.volatility_label(0.70), "High")
        self.assertEqual(npa.volatility_label(0.50), "Moderate")
        self.assertEqual(npa.volatility_label(0.35), "Moderate")
        self.assertEqual(npa.volatility_label(0.10), "Low")
        # a full-season starter reads High confidence; a 2-game cameo reads Low
        full = npa.evaluate_line(wr_one(), "Receiving Yards", 76.5)
        cameo = npa.evaluate_line(wr_one(games_played=2, prior_games_played=2, receiving_yards=120.0, projection_confidence=0.3), "Receiving Yards", 60.5)
        self.assertEqual(full["confidence_label"], "High")
        self.assertEqual(cameo["confidence_label"], "Low")
        self.assertLess(cameo["confidence"], full["confidence"])


class ApiContractTests(IsolatedStatusCase):
    """Pins the exact returned keys: the page is written against these, so a rename must fail loudly."""

    def test_evaluate_line_keys(self):
        result = npa.evaluate_line(rb_one(), "Rushing Yards", 70.5, pool=_skill_pool())
        self.assertEqual(
            sorted(result),
            sorted([
                "category", "confidence", "confidence_label", "cv", "distribution", "edge_over", "edge_under", "ev_over",
                "ev_under", "family", "has_edge", "headline", "hold", "implied_probability_over",
                "implied_probability_under", "integer_line", "line", "line_minus_mean", "market",
                "market_fair_probability_over", "market_fair_probability_under", "mean", "name", "over_price", "p10",
                "p50", "p90", "percentiles", "player_id", "position", "pricing_over", "pricing_under",
                "probability_over", "probability_under", "projection", "push_probability", "reasoning",
                "recommended_edge", "recommended_ev", "recommended_price", "recommended_probability",
                "recommended_side", "recommended_value_gap", "risk_tier", "stake", "status", "status_note", "stdev",
                "team", "under_price", "unit", "value_gap_over", "value_gap_under", "verdict", "verdict_tone",
                "volatility", "volatility_label", "warnings", "z_score",
            ]),
        )
        self.assertEqual(
            sorted(result["pricing_over"]),
            sorted([
                "decimal_odds", "ev_pct", "expected_value", "fair_price", "implied_probability", "is_value",
                "model_probability", "price", "stake", "value_gap", "verdict", "verdict_tone",
            ]),
        )
        self.assertEqual(sorted(result["distribution"]), ["cv", "family", "games", "mean", "stdev", "volatility"])
        self.assertIsInstance(result["warnings"], list)
        self.assertIsInstance(result["reasoning"], list)

    def test_project_stat_keys(self):
        for label in ("Rushing Yards", "Anytime TD (Rush + Rec)"):
            projection = npa.project_stat(rb_one(), label)
            self.assertEqual(
                sorted(projection),
                sorted([
                    "category", "components", "confidence", "confidence_label", "cv", "data_season", "drivers",
                    "eligible", "expected_games", "family", "fantasy_points_per_game", "games", "market", "mean", "name",
                    "per_game_rate", "player_id", "position", "projected_fantasy_points", "projection",
                    "projection_basis", "projection_confidence", "risk_tier", "season_total", "status", "status_factor",
                    "status_note", "stdev", "team", "unit", "volatility", "volatility_label",
                ]),
            )

    def test_validate_inputs_and_peer_rank_keys(self):
        self.assertEqual(
            sorted(npa.validate_inputs("Passing Yards", "249.5")),
            sorted(["blank_line", "defaults_used", "errors", "field_errors", "line", "message", "notes", "ok", "over_price", "under_price"]),
        )
        self.assertEqual(
            sorted(npa.peer_rank(_skill_pool(), wr_one(), "Receiving Yards")),
            sorted(["median_rate", "n", "percentile", "position", "rank", "rate"]),
        )

    def test_category_entries_carry_the_required_and_extra_keys(self):
        for spec in npa.STAT_CATEGORIES.values():
            self.assertEqual(
                sorted(spec),
                sorted([
                    "blurb", "family", "fields", "market", "max_line", "min_line", "min_per_game", "positions",
                    "short_label", "step", "unit",
                ]),
            )

    def test_results_are_plain_json_serialisable_data(self):
        """No numpy scalars or custom objects: safe to cache, pickle or hand to a UI layer."""
        result = npa.evaluate_line(wr_one(), "Receptions", 5.5, pool=_skill_pool())
        self.assertEqual(json.loads(json.dumps(result))["line"], 5.5)


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


class ReasoningTests(IsolatedStatusCase):
    def titles(self, items):
        return [item["title"] for item in items]

    def test_every_category_has_five_or_more_numeric_items(self):
        pool = _skill_pool()
        for label, factory in CATEGORY_PLAYERS.items():
            player = factory()
            with self.subTest(label=label):
                result = npa.evaluate_line(player, label, npa.suggest_line(npa.project_stat(player, label)), pool=pool)
                items = result["reasoning"]
                self.assertGreaterEqual(len(items), 5)
                for item in items:
                    self.assertEqual(sorted(item), ["detail", "title", "tone"])
                    self.assertIn(item["tone"], ("pro", "con", "neutral"))
                    self.assertTrue(item["title"])
                    self.assertRegex(item["detail"], r"\d", item["title"])
                    self.assertNotIn("nan", item["detail"].lower())
                    self.assertNotIn("None", item["detail"])

    def test_required_topics_are_covered_with_real_numbers(self):
        player = wr_one()
        pool = _skill_pool()
        result = npa.evaluate_line(player, "Receiving Yards", 60.5, -115.0, -105.0, pool=pool)
        items = {item["title"]: item["detail"] for item in result["reasoning"]}
        rate = 1300 / 17
        self.assertIn(f"{rate:.1f} yds per game", items["Per-game rate and sample"])
        self.assertIn("17 games", items["Per-game rate and sample"])
        self.assertIn("1,300 yds", items["Per-game rate and sample"])
        self.assertIn("percentile", items["Rank among WRs"])
        self.assertIn("#1 of 5", items["Rank among WRs"])  # Wes tops the five fixture WRs with 6+ games who qualify
        self.assertIn("17 games in 2025", items["Availability"])
        self.assertIn("13.5 games", items["Availability"])
        spread = items["Volatility and spread of outcomes"]
        self.assertIn(f"{result['cv']:.2f}", spread)
        self.assertIn(f"{result['p10']:g}", spread)
        self.assertIn(f"{result['p90']:g}", spread)
        line_item = items["Your line vs the projection"]
        self.assertIn("below the projection", line_item)  # a 60.5 line vs a ~76.5 mean
        self.assertIn(f"{result['z_score']:+.2f} standard deviations", line_item)
        self.assertIn("Why a Gaussian curve", items)
        self.assertIn("rate regressed", items["Where the projection comes from"])
        self.assertIn("1,000 rushing yards", items["Where the projection comes from"])
        self.assertIn("healthy", items["Availability"].lower())
        market = items["Model vs market price"]
        self.assertIn(f"{result['probability_over']:.1%}", market)
        self.assertIn(f"{result['market_fair_probability_over']:.1%}", market)
        self.assertIn("-115", items["What the price needs"])
        self.assertIn("-105", items["What the price needs"])

    def test_poisson_markets_explain_why_counts_are_poisson(self):
        result = npa.evaluate_line(wr_one(), "Receiving TDs", 0.5)
        items = {item["title"]: item["detail"] for item in result["reasoning"]}
        self.assertIn("Why a Poisson count model", items)
        self.assertIn("variance equals the mean", items["Why a Poisson count model"])
        self.assertIn(f"{result['mean']:.2f}", items["Why a Poisson count model"])

    def test_anytime_td_reasoning_splits_the_two_rates(self):
        result = npa.evaluate_line(rb_one(rushing_tds=6.0, receiving_tds=3.0, games_played=12, prior_games_played=12), "Anytime TD (Rush + Rec)", 0.5)
        detail = {item["title"]: item["detail"] for item in result["reasoning"]}["Per-game rate and sample"]
        self.assertIn("0.50 rushing TDs", detail)
        self.assertIn("0.25 receiving TDs", detail)
        self.assertIn("0.75 TDs per game", detail)

    def test_rank_item_only_appears_with_a_pool_and_matches_peer_rank(self):
        player = wr_one()
        without_pool = npa.evaluate_line(player, "Receiving Yards", 60.5)
        self.assertFalse(any(title.startswith("Rank among") for title in self.titles(without_pool["reasoning"])))
        with_pool = npa.evaluate_line(player, "Receiving Yards", 60.5, pool=_skill_pool())
        self.assertTrue(any(title.startswith("Rank among") for title in self.titles(with_pool["reasoning"])))
        direct = npa.reasoning_for(player, "Receiving Yards", with_pool["projection"], with_pool, pool=_skill_pool())
        self.assertEqual(direct, with_pool["reasoning"])
        self.assertEqual(len(npa.reasoning_for(player, "Receiving Yards", with_pool["projection"], with_pool)), len(with_pool["reasoning"]) - 1)

    def test_availability_and_thin_sample_tones(self):
        player = wr_one()
        self.set_status(player["player_id"], "QUESTIONABLE")
        questionable = npa.evaluate_line(player, "Receiving Yards", 60.5)
        availability = next(item for item in questionable["reasoning"] if item["title"] == "Availability")
        self.assertEqual(availability["tone"], "con")
        self.assertIn("88%", availability["detail"])
        thin = npa.evaluate_line(wr_one(games_played=3, prior_games_played=3, receiving_yards=210.0), "Receiving Yards", 60.5)
        sample = next(item for item in thin["reasoning"] if item["title"] == "Per-game rate and sample")
        self.assertEqual(sample["tone"], "con")
        self.assertIn("thin sample", sample["detail"])

    def test_tones_are_relative_to_the_recommended_side(self):
        player = wr_one()
        high = npa.evaluate_line(player, "Receiving Yards", 110.5)  # far above the mean: under is the side
        self.assertEqual(high["recommended_side"], "under")
        self.assertEqual(next(i for i in high["reasoning"] if i["title"] == "Your line vs the projection")["tone"], "pro")
        low = npa.evaluate_line(player, "Receiving Yards", 40.5)  # far below the mean: over is the side
        self.assertEqual(low["recommended_side"], "over")
        self.assertEqual(next(i for i in low["reasoning"] if i["title"] == "Your line vs the projection")["tone"], "pro")
        self.assertEqual(next(i for i in low["reasoning"] if i["title"] == "Model vs market price")["tone"], "pro")
        flip = npa.evaluate_line(player, "Receiving Yards", 76.5, -300.0, 250.0)
        self.assertIn(next(i for i in flip["reasoning"] if i["title"] == "Model vs market price")["tone"], ("pro", "con"))


class PeerRankTests(unittest.TestCase):
    def test_rank_percentile_and_median(self):
        pool = [
            make_player(f"00-WR-{i}", f"Wide {i}", "WR", "KC", receiving_yards=yards)
            for i, yards in enumerate((1500.0, 1200.0, 900.0, 700.0, 500.0))
        ]
        ranking = npa.peer_rank(pool, pool[2], "Receiving Yards")
        self.assertEqual((ranking["rank"], ranking["n"]), (3, 5))
        self.assertEqual(ranking["percentile"], 50.0)
        self.assertAlmostEqual(ranking["median_rate"], 900 / 17, places=3)
        self.assertEqual(npa.peer_rank(pool, pool[0], "Receiving Yards")["percentile"], 90.0)
        self.assertEqual(npa.peer_rank(pool, pool[4], "Receiving Yards")["percentile"], 10.0)
        self.assertEqual(npa.peer_rank(pool, pool[0], "Receiving Yards")["rank"], 1)

    def test_player_outside_the_pool_is_still_ranked_and_cameos_do_not_pad_the_field(self):
        pool = [make_player(f"00-WR-{i}", f"Wide {i}", "WR", "KC", receiving_yards=y) for i, y in enumerate((1500.0, 1200.0, 900.0))]
        cameo = make_player("00-WR-CAMEO", "Cameo", "WR", "KC", games=2, receiving_yards=100.0)  # 50 / game but only 2 games
        newcomer = make_player("00-WR-NEW", "Newcomer", "WR", "KC", receiving_yards=1000.0)
        ranking = npa.peer_rank(pool + [cameo], newcomer, "Receiving Yards")
        self.assertEqual((ranking["rank"], ranking["n"]), (3, 4))  # the cameo is excluded from the field

    def test_too_few_peers_or_unknown_category_returns_none(self):
        pool = [make_player("00-WR-1", "Solo", "WR", "KC", receiving_yards=900.0)]
        self.assertIsNone(npa.peer_rank(pool, pool[0], "Receiving Yards"))
        self.assertIsNone(npa.peer_rank(pool, pool[0], "Punting Yards"))
        self.assertIsNone(npa.peer_rank(None, pool[0], "Receiving Yards"))


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------


class ValidateInputsTests(unittest.TestCase):
    def test_valid_line_and_default_prices(self):
        result = npa.validate_inputs("Passing Yards", "249.5")
        self.assertTrue(result["ok"])
        self.assertEqual((result["line"], result["over_price"], result["under_price"]), (249.5, -110.0, -110.0))
        self.assertEqual(result["errors"], [])
        self.assertEqual(sorted(result["defaults_used"]), ["over_price", "under_price"])
        self.assertEqual(result["message"], "")
        self.assertFalse(result["blank_line"])

    def test_blank_line_is_pending_not_an_error(self):
        for blank in ("", "   ", None):
            result = npa.validate_inputs("Passing Yards", blank)
            self.assertFalse(result["ok"])
            self.assertTrue(result["blank_line"])
            self.assertEqual(result["errors"], [])
            self.assertIsNone(result["line"])
            self.assertIn("Paste", result["message"])

    def test_invalid_lines_give_friendly_messages(self):
        for text, fragment in (
            ("abc", "number"),
            ("-5", "negative"),
            ("0", "at least"),
            ("5000", "at most"),
            ("20", "at least"),
            ("249.25", "half-point"),
        ):
            with self.subTest(text=text):
                result = npa.validate_inputs("Passing Yards", text)
                self.assertFalse(result["ok"])
                self.assertIsNone(result["line"])
                self.assertEqual(len(result["errors"]), 1)
                self.assertIn(fragment, result["errors"][0])
                self.assertEqual(result["field_errors"]["line"], result["errors"][0])
                self.assertEqual(result["message"], result["errors"][0])

    def test_integer_and_typographic_input_are_accepted(self):
        self.assertEqual(npa.validate_inputs("Receptions", "5")["line"], 5.0)
        self.assertEqual(npa.validate_inputs("Receptions", " 5.5 ")["line"], 5.5)
        self.assertEqual(npa.validate_inputs("Receptions", "−5.5")["field_errors"]["line"] is not None, True)

    def test_prices_parse_and_report_errors(self):
        result = npa.validate_inputs("Receptions", "5.5", "-125", "+105")
        self.assertTrue(result["ok"])
        self.assertEqual((result["over_price"], result["under_price"]), (-125.0, 105.0))
        self.assertEqual(result["defaults_used"], [])
        decimal = npa.validate_inputs("Receptions", "5.5", "1.91", "2.10")
        self.assertEqual(decimal["over_price"], -110.0)
        self.assertEqual(decimal["under_price"], 110.0)
        bad = npa.validate_inputs("Receptions", "5.5", "-50", "abc")
        self.assertFalse(bad["ok"])
        self.assertIsNone(bad["over_price"])
        self.assertIsNone(bad["under_price"])
        self.assertEqual(len(bad["errors"]), 2)
        self.assertTrue(bad["errors"][0].startswith("Over price"))
        self.assertTrue(bad["errors"][1].startswith("Under price"))
        self.assertIsNotNone(bad["field_errors"]["over_price"])

    def test_one_blank_price_defaults_with_a_note(self):
        result = npa.validate_inputs("Receptions", "5.5", "+120", "")
        self.assertTrue(result["ok"])
        self.assertEqual((result["over_price"], result["under_price"]), (120.0, -110.0))
        self.assertEqual(result["defaults_used"], ["under_price"])
        self.assertIn("Under price left blank", result["notes"][0])

    def test_anytime_td_is_only_accepted_at_half(self):
        self.assertTrue(npa.validate_inputs("Anytime TD (Rush + Rec)", "0.5")["ok"])
        self.assertFalse(npa.validate_inputs("Anytime TD (Rush + Rec)", "1.5")["ok"])
        self.assertFalse(npa.validate_inputs("Anytime TD (Rush + Rec)", "1")["ok"])

    def test_unknown_category_is_reported(self):
        result = npa.validate_inputs("Punting Yards", "40.5")
        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["field_errors"]["category"])
        self.assertTrue(result["errors"])

    def test_a_valid_result_feeds_evaluate_line(self):
        parsed = npa.validate_inputs("Receiving Yards", "70.5", "-120", "100")
        self.assertTrue(parsed["ok"])
        with mock.patch.object(player_status, "STATUS_PATH", Path(tempfile.gettempdir()) / "definitely-missing-status.json"):
            player_status.clear_cache()
            result = npa.evaluate_line(wr_one(), "Receiving Yards", parsed["line"], parsed["over_price"], parsed["under_price"])
        player_status.clear_cache()
        self.assertEqual((result["line"], result["over_price"], result["under_price"]), (70.5, -120.0, 100.0))


# ---------------------------------------------------------------------------
# Regressions found by the independent audit
# ---------------------------------------------------------------------------


def _titled(result) -> dict:
    return {item["title"]: item for item in result["reasoning"]}


class AuditRegressionTests(IsolatedStatusCase):
    """One test per defect the audit found; each fails against the pre-audit module."""

    def setUp(self):
        super().setUp()
        npa.clear_player_pool_cache()
        self.addCleanup(npa.clear_player_pool_cache)

    # -- integer-line explanations must reconcile with the edge they claim ------------------

    def _assert_text_reconciles(self, result):
        headline = result["headline"]
        model = float(re.search(r"model (?:probability of )?([\d.]+)%", headline).group(1)) / 100
        fair = float(re.search(r"vs ([\d.]+)% no-vig fair", headline).group(1)) / 100
        self.assertAlmostEqual(model - fair, result["recommended_edge"], delta=0.0012, msg=headline)
        side = result["recommended_side"]
        needs = _titled(result)["What the price needs"]["detail"]
        breakeven = float(re.search(rf"([\d.]+)% \({side}\)", needs).group(1)) / 100
        if result["push_probability"] > 0:
            stated = float(re.search(r"decided is ([\d.]+)%", needs).group(1)) / 100
        else:
            stated = float(re.search(rf"{side} probability is ([\d.]+)%", needs).group(1)) / 100
        self.assertAlmostEqual(stated - breakeven, result["recommended_value_gap"], delta=0.0012, msg=needs)

    def test_headline_and_price_text_reconcile_with_edge_and_value_gap_on_every_line_type(self):
        seen = set()
        for label, factory in CATEGORY_PLAYERS.items():
            player = factory()
            mean = npa.project_stat(player, label)["mean"]
            lines = {1.0, 2.0, float(math.floor(mean)), float(math.floor(mean)) + 1.0, math.floor(mean) + 0.5, 0.5}
            for line in sorted(x for x in lines if x > 0):
                for prices in ((-110.0, -110.0), (-130.0, 110.0), (140.0, -170.0), (-105.0, -115.0)):
                    with self.subTest(label=label, line=line, prices=prices):
                        result = npa.evaluate_line(player, label, line, *prices)
                        self._assert_text_reconciles(result)
                        seen.add((result["push_probability"] > 0, result["has_edge"]))
        # the sweep really exercised pushing lines in both headline forms (edge / no edge)
        self.assertIn((True, True), seen)
        self.assertIn((True, False), seen)

    def test_integer_line_text_states_the_decided_probability_and_half_point_text_does_not(self):
        result = npa.evaluate_line(wr_one(), "Receptions", 5.0, -130.0, 110.0, pool=_skill_pool())
        side = result["recommended_side"]
        decided = result[f"pricing_{side}"]["model_probability"]
        self.assertAlmostEqual(decided, result[f"probability_{side}"] / (1 - result["push_probability"]), delta=2e-4)
        self.assertEqual(result["recommended_probability"], result[f"probability_{side}"])  # the contract: outright P(side)
        self.assertIn(f"{decided:.1%} if decided", result["headline"])
        self.assertIn("push", result["headline"])
        market = _titled(result)["Model vs market price"]["detail"]
        self.assertIn("given the bet is decided", market)
        self.assertIn(f"{decided:.1%}", _titled(result)["What the price needs"]["detail"])
        half = npa.evaluate_line(wr_one(), "Receptions", 5.5, -130.0, 110.0)
        self.assertNotIn("decided", half["headline"])
        self.assertNotIn("decided", _titled(half)["Model vs market price"]["detail"])
        self.assertNotIn("decided", _titled(half)["What the price needs"]["detail"])

    # -- hostile data must degrade, not leak NaN --------------------------------------------

    def test_non_finite_inputs_are_treated_as_missing_and_flagged_not_propagated(self):
        nan, inf = float("nan"), float("inf")
        cases = (
            ("games_played", "Receiving Yards", 70.5),
            ("receiving_yards", "Receiving Yards", 70.5),
            ("receptions", "Receptions", 5.5),
            ("receiving_tds", "Receiving TDs", 0.5),
            ("receiving_tds", "Anytime TD (Rush + Rec)", 0.5),
            ("rushing_tds", "Anytime TD (Rush + Rec)", 0.5),
            ("rushing_yards", "Rushing Yards", 70.5),
        )
        for field, label, line in cases:
            for bad in (nan, inf, -inf, "abc"):
                with self.subTest(field=field, label=label, bad=bad):
                    player = rb_one()
                    player[field] = bad
                    result = npa.evaluate_line(player, label, line, pool=_skill_pool())
                    assert_clean(self, result)
                    assert_clean(self, npa.project_stat(player, label))
                    self.assertTrue(
                        any(field in w and "non-numeric or non-finite" in w for w in result["warnings"]), result["warnings"]
                    )
                    self.assertIs(player[field], bad)  # the caller's row is never rewritten

    def test_non_finite_games_played_falls_back_to_prior_games_and_both_bad_raises(self):
        player = rb_one()
        player["games_played"] = float("nan")
        self.assertEqual(npa.project_stat(player, "Rushing Yards")["games"], 17.0)
        both = rb_one(games_played=float("nan"), prior_games_played=float("inf"))
        with self.assertRaises(ValueError):
            npa.project_stat(both, "Rushing Yards")

    def test_a_clean_row_is_passed_to_the_engine_unchanged(self):
        player = wr_one()
        self.assertEqual(npa._nonfinite_fields(player), [])
        self.assertEqual(npa._sane_row(player), player)
        none_stat = wr_one(receptions=None)  # None is the engine's own "missing", not a data fault
        self.assertEqual(npa._nonfinite_fields(none_stat), [])

    # -- absurd and mistyped arguments -------------------------------------------------------

    def test_absurd_lines_raise_value_error_before_any_work(self):
        for bad in (npa.MAX_LINE + 0.5, 1e5, 1e9, 1e308):
            with self.subTest(line=bad):
                with self.assertRaises(ValueError) as caught:
                    npa.evaluate_line(wr_one(), "Receptions", bad)
                self.assertIn("at most", str(caught.exception))
        result = npa.evaluate_line(wr_one(), "Receptions", npa.MAX_LINE)  # the cap itself is still priced
        self.assertEqual(result["probability_under"], 1.0)
        assert_clean(self, result)

    def test_bool_arguments_are_rejected_not_read_as_one_or_zero(self):
        for bad in (True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    npa.evaluate_line(wr_one(), "Receiving Yards", bad)
                with self.assertRaises(ValueError):
                    npa.evaluate_line(wr_one(), "Receiving Yards", 70.5, bad, -110.0)
                with self.assertRaises(ValueError):
                    npa.evaluate_line(wr_one(), "Receiving Yards", 70.5, -110.0, bad)
                with self.assertRaises(ValueError):
                    npa.evaluate_line(wr_one(), "Receiving Yards", 70.5, stake=bad)

    def test_a_player_that_is_not_a_row_raises_value_error_not_attribute_error(self):
        for bad in (None, "Wes Receiver", 5, [("name", "x")]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    npa.project_stat(bad, "Receiving Yards")
                with self.assertRaises(ValueError):
                    npa.evaluate_line(bad, "Receiving Yards", 70.5)

    # -- lookups -----------------------------------------------------------------------------

    def test_blank_id_and_blank_team_never_match_rows_that_merely_lack_them(self):
        stray = {"name": "No Id", "position": "WR", "team": "", "games_played": 10, "receiving_yards": 800.0}
        pool = [stray, *_skill_pool()]
        for blank in (None, "", "   "):
            self.assertIsNone(npa.find_player(pool, blank))
        for team in ("", None, "  "):
            self.assertEqual(npa.players_for_team(pool, team), [])
            self.assertEqual(npa.players_for_team(pool, team, "Receiving Yards"), [])
        with self.assertRaises(ValueError):
            npa.players_for_team(pool, "", "Punting Yards")  # a bad category is still a bad category
        self.assertIs(npa.find_player(pool, "00-KC-WR1"), next(p for p in pool if p.get("player_id") == "00-KC-WR1"))

    def test_load_player_pool_survives_loaders_that_fail_while_being_read(self):
        class Exploding:
            def __iter__(self):
                raise RuntimeError("mid-iteration failure")

        for odd in (Exploding(), None):
            with self.subTest(odd=odd), mock.patch.object(npa, "load_forward_projections", return_value=odd):
                self.assertEqual(npa.load_player_pool(2024), [])

    # -- season labelling --------------------------------------------------------------------

    def test_data_season_is_the_actuals_season_never_the_projection_season(self):
        self.assertEqual(npa.project_stat(rb_one(prior_season=2024), "Rushing Yards")["data_season"], 2024)
        row = rb_one()
        row.pop("prior_season")  # projection_season 2026 means the actuals are 2025
        self.assertEqual(npa.project_stat(row, "Rushing Yards")["data_season"], 2025)
        row.pop("projection_season")  # only "season": 2026 -- the projection year -- remains: unknown, not 2026
        self.assertIsNone(npa.project_stat(row, "Rushing Yards")["data_season"])
        detail = _titled(npa.evaluate_line(row, "Rushing Yards", 70.5))["Per-game rate and sample"]["detail"]
        self.assertIn("in the source season", detail)
        self.assertNotIn("2026", detail)

    # -- reasoning honesty -------------------------------------------------------------------

    def test_count_market_volatility_tone_follows_cv_not_the_engines_unused_volatility(self):
        player = wr_one()
        engine = {"volatility": 0.95, "confidence": 0.9}
        with mock.patch.object(npa, "compute_final_projection", return_value=engine), mock.patch.object(
            prop_model, "compute_final_projection", return_value=engine
        ):
            receptions = npa.evaluate_line(player, "Receptions", 5.5)
            yards = npa.evaluate_line(player, "Receiving Yards", 70.5)
        self.assertLess(receptions["cv"], 0.5)
        item = _titled(receptions)["Volatility and spread of outcomes"]
        self.assertNotEqual(item["tone"], "con")  # cv 0.42: the text says "tighter band", the tone must agree
        self.assertIn("tighter band", item["detail"])
        self.assertIn("(0.95, High)", item["detail"])
        self.assertIn("informational", item["detail"])
        wide = _titled(yards)["Volatility and spread of outcomes"]  # the same volatility does drive a yardage spread
        self.assertEqual(wide["tone"], "con")
        self.assertIn("wide band", wide["detail"])
        self.assertNotIn("informational", wide["detail"])
        calm = {"volatility": 0.05, "confidence": 0.9}
        with mock.patch.object(npa, "compute_final_projection", return_value=calm), mock.patch.object(
            prop_model, "compute_final_projection", return_value=calm
        ):
            steady = npa.evaluate_line(player, "Receiving Yards", 70.5)
        self.assertEqual(_titled(steady)["Volatility and spread of outcomes"]["tone"], "pro")

    def test_qb_passing_yards_reasoning_compares_the_spread_with_observed_history(self):
        qb = qb_pocket()
        for volatility, verdict, tone in ((1.0, "wider than history", "con"), (0.0, "sits inside", "neutral")):
            engine = {"volatility": volatility, "confidence": 0.9}
            with self.subTest(volatility=volatility), mock.patch.object(
                npa, "compute_final_projection", return_value=engine
            ), mock.patch.object(prop_model, "compute_final_projection", return_value=engine):
                item = _titled(npa.evaluate_line(qb, "Passing Yards", 250.5))["Why a Gaussian curve"]
            self.assertIn(verdict, item["detail"])
            self.assertIn("0.33", item["detail"])
            self.assertIn("0.27 to 0.47", item["detail"])
            self.assertEqual(item["tone"], tone)
        other = _titled(npa.evaluate_line(wr_one(), "Receiving Yards", 70.5))["Why a Gaussian curve"]
        self.assertNotIn("passing yards varied", other["detail"])
        self.assertEqual(other["tone"], "neutral")

    def test_poisson_text_quantifies_the_over_dispersion_and_availability_avoids_stale_season_wording(self):
        result = npa.evaluate_line(wr_one(), "Receiving TDs", 0.5)
        self.assertIn("1.0 to 1.2 times the mean", _titled(result)["Why a Poisson count model"]["detail"])
        self.assertNotIn("scoring", _titled(npa.evaluate_line(wr_one(), "Receptions", 5.5))["Why a Poisson count model"]["detail"])
        availability = _titled(result)["Availability"]["detail"]
        self.assertNotIn("coming season", availability)
        self.assertIn("13.5 games over its projection season", availability)

    def test_hold_warning_quotes_the_standard_hold_the_module_itself_displays(self):
        steep = npa.evaluate_line(wr_one(), "Receiving Yards", 70.5, -200.0, -200.0)
        message = next(w for w in steep["warnings"] if "hold" in w)
        self.assertIn("33.3% hold", message)
        self.assertIn(f"{hold(-110.0, -110.0):.1%}", message)  # 4.8%, not a hand-typed 4.5%
        standard = npa.evaluate_line(wr_one(), "Receiving Yards", 76.5)
        self.assertIn(f"{standard['hold']:.1%}", _titled(standard)["What the price needs"]["detail"])

    def test_anytime_td_rounding_stays_within_the_documented_five_hundredths_of_a_point(self):
        for rush, rec, games in ((1.0, 0.0, 16), (6.0, 3.0, 12), (0.0, 1.0, 17), (3.0, 4.0, 16), (2.0, 1.0, 14)):
            with self.subTest(rush=rush, rec=rec, games=games):
                player = rb_one(rushing_tds=rush, receiving_tds=rec, games_played=games, prior_games_played=games)
                exact = 1 - math.exp(-(rush + rec) / games)
                result = npa.evaluate_line(player, "Anytime TD (Rush + Rec)", 0.5)
                self.assertAlmostEqual(result["probability_over"], exact, delta=6e-4)

    # -- input parsing -----------------------------------------------------------------------

    def test_validate_inputs_accepts_a_bare_leading_or_trailing_point(self):
        self.assertEqual(npa.validate_inputs("Anytime TD (Rush + Rec)", ".5")["line"], 0.5)
        self.assertTrue(npa.validate_inputs("Receptions", ".5")["ok"])
        self.assertEqual(npa.validate_inputs("Receptions", "+.5")["line"], 0.5)
        self.assertEqual(npa.validate_inputs("Receptions", "5.")["line"], 5.0)
        for still_bad in (".", "-.5", "1.2.3"):
            with self.subTest(text=still_bad):
                self.assertFalse(npa.validate_inputs("Receptions", still_bad)["ok"])

    # -- eligibility vs the pre-filled line --------------------------------------------------

    def test_a_player_at_the_eligibility_floor_gets_a_prefilled_line_near_his_own_projection(self):
        """The floor and ``min_line`` are two tables; they must not pre-fill a line far above an eligible player's mean."""
        make = {
            "Passing Yards": lambda rate: qb_pocket(passing_yards=rate * 17, passing_tds=20.0),
            "Rushing Yards": lambda rate: rb_one(rushing_yards=rate * 17),
            "Receiving Yards": lambda rate: wr_one(receiving_yards=rate * 17),
        }
        for label, factory in make.items():
            spec = npa.STAT_CATEGORIES[label]
            floor = spec["min_per_game"]
            player = factory(floor)
            with self.subTest(label=label, floor=floor):
                self.assertTrue(npa.is_eligible(player, label))
                self.assertFalse(npa.is_eligible(factory(floor - 1.0), label))
                projection = npa.project_stat(player, label)
                z = (npa.suggest_line(projection) - projection["mean"]) / projection["stdev"]
                self.assertLessEqual(abs(z), 1.0, f"pre-fill sits {z:+.2f} sd from the mean at the floor")
        # the concrete case the audit found: a 2-game, 42-yard backup is no longer offered Passing Yards
        backup = make_player("00-NO-QB9", "Cam Cameo", "QB", "NO", games=2, passing_yards=42.0)
        self.assertFalse(npa.is_eligible(backup, "Passing Yards"))
        self.assertNotIn("Cam Cameo", [p["name"] for p in npa.players_for_team([backup], "NO", "Passing Yards")])

    def test_rank_text_counts_the_player_honestly_even_when_he_is_not_in_the_peer_field(self):
        pool = [make_player(f"00-WR-{i}", f"Wide {i}", "WR", "KC", receiving_yards=y) for i, y in enumerate((1500.0, 1200.0, 900.0))]
        newcomer = make_player("00-WR-NEW", "Newcomer", "WR", "KC", games=3, receiving_yards=90.0)  # 30/g over 3 games: last, and not a "6+ games" peer
        result = npa.evaluate_line(newcomer, "Receiving Yards", 100.5, pool=pool)
        detail = _titled(result)["Rank among WRs"]["detail"]
        self.assertIn("#4 of 4 WRs", detail)
        self.assertIn("the player plus 3 peers with 6+ games who qualify", detail)

    def test_sample_size_thresholds_are_exported_for_the_page(self):
        for name in ("THIN_SAMPLE_GAMES", "FULL_SAMPLE_GAMES", "MAX_LINE", "OBSERVED_QB_PASSING_CV"):
            self.assertIn(name, npa.__all__)
            self.assertTrue(hasattr(npa, name))
        self.assertEqual([n for n in npa.__all__ if not hasattr(npa, n)], [])


# ---------------------------------------------------------------------------
# Live data (skipped offline)
# ---------------------------------------------------------------------------


class LiveDataTests(IsolatedStatusCase):
    def _pool(self):
        npa.clear_player_pool_cache()
        pool = npa.load_player_pool()
        if not pool:
            self.skipTest("live NFL player pool could not be loaded (offline?)")
        return pool

    def test_load_player_pool_returns_real_rows(self):
        pool = self._pool()
        self.assertGreater(len(pool), 300)
        for key in ("player_id", "name", "position", "team", "games_played", "passing_yards", "rushing_yards", "receiving_yards"):
            self.assertIn(key, pool[0])
        teams = npa.teams_in_pool(pool)
        self.assertGreaterEqual(len(teams), 30)
        qbs = npa.players_for_team(pool, teams[0], "Passing Yards")
        self.assertTrue(all(p["position"] == "QB" for p in qbs))

    def test_star_qb_passing_yards_at_his_own_mean_is_close_to_a_coin_flip(self):
        pool = self._pool()
        qbs = [p for p in pool if p.get("position") == "QB" and (p.get("games_played") or 0) >= 12]
        if not qbs:
            self.skipTest("no full-season QB in the live pool")
        star = max(qbs, key=lambda p: float(p.get("passing_yards") or 0))
        projection = npa.project_stat(star, "Passing Yards")
        result = npa.evaluate_line(star, "Passing Yards", npa.suggest_line(projection), pool=pool)
        self.assertAlmostEqual(result["probability_over"], 0.5, delta=0.05)
        self.assertGreaterEqual(len(result["reasoning"]), 6)
        self.assertTrue(any(item["title"].startswith("Rank among QBs") for item in result["reasoning"]))
        assert_clean(self, result)


class OneGameSampleWordingTests(unittest.TestCase):
    """A one-game sample must read "1 game", never "1 games" (seen on real one-game backup QBs)."""

    def test_helper_pluralises(self):
        for count, text in ((1, "1 game"), (1.0, "1 game"), (2, "2 games"), (17, "17 games"), (0, "0 games")):
            with self.subTest(count=count):
                self.assertEqual(npa._games_text(count), text)

    def test_one_game_player_reads_naturally_everywhere(self):
        player = qb_pocket()
        player.update(games_played=1, passing_yards=210.0, passing_tds=1.0, pass_attempts=30.0)
        result = npa.evaluate_line(player, "Passing Yards", 199.5)
        text = " ".join(f"{item['title']} {item['detail']}" for item in result["reasoning"]) + " " + " ".join(result["warnings"])
        self.assertIn("1 game", text)
        self.assertNotRegex(text, r"1 games")
        self.assertNotIn("game(s)", text)


if __name__ == "__main__":
    unittest.main()
