"""Contract + regression tests for modules.nfl_moneyline_analysis.

Everything runs offline from hand-built fixtures and monkeypatched loaders.
Two clearly-marked live tests hit nflverse and skip themselves when the data
cannot be loaded.
"""
from __future__ import annotations

import math
import re
import sys
import threading
import unittest
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from betting.moneyline_model import NFL_MARGIN_STDEV, win_probability_from_spread
from betting.odds_math import fair_price_from_probability, remove_vig_two_way
from modules import nfl_moneyline_analysis as m
from modules.nfl import NFL_TEAMS
from modules.odds_input import evaluate_price


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def assert_all_finite(testcase: unittest.TestCase, value, path: str = "result") -> None:
    """Recursively assert there is no NaN / inf anywhere in a returned structure."""
    if isinstance(value, dict):
        for key, inner in value.items():
            assert_all_finite(testcase, inner, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            assert_all_finite(testcase, inner, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)):
        testcase.assertTrue(math.isfinite(float(value)), f"{path} is {value!r}")


_RAW_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime", "away_team", "away_score",
    "home_team", "home_score", "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_qb_name", "away_qb_name", "home_qb_id", "away_qb_id", "roof", "temp", "wind", "div_game",
    "home_rest", "away_rest",
]


def raw_game(week, away, home, *, gameday="2026-10-11", gametime="13:00", away_score=None, home_score=None,
             spread=None, total=None, home_ml=None, away_ml=None, home_qb=None, away_qb=None,
             game_type="REG", roof="outdoors", wind=5.0, temp=60.0, div=0, home_rest=7, away_rest=7, season=2026):
    nan = float("nan")
    return {
        "game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "game_type": game_type, "week": week,
        "gameday": gameday, "weekday": "Sunday", "gametime": gametime, "away_team": away,
        "away_score": nan if away_score is None else away_score, "home_team": home,
        "home_score": nan if home_score is None else home_score,
        "spread_line": nan if spread is None else spread, "total_line": nan if total is None else total,
        "home_moneyline": nan if home_ml is None else home_ml, "away_moneyline": nan if away_ml is None else away_ml,
        "home_qb_name": home_qb, "away_qb_name": away_qb, "home_qb_id": None, "away_qb_id": None,
        "roof": roof, "temp": temp, "wind": wind, "div_game": div, "home_rest": home_rest, "away_rest": away_rest,
    }


def raw_schedule(rows):
    return pd.DataFrame(rows, columns=_RAW_COLUMNS)


def team_row(*, pf=22.5, pa=22.5, off=50.0, dfn=50.0, off_epa=0.0, def_epa=0.0, spp=30.4, games=12.0):
    return {"games": games, "n_points": games, "pf": pf, "pa": pa, "off_score": off, "def_score": dfn,
            "off_epa": off_epa, "def_epa": def_epa, "spp": spp}


def snapshot(teams: dict[str, dict]) -> dict:
    """A hand-built season snapshot (what ``_season_snapshot`` returns) with a league row of simple means."""
    def mean(key):
        return float(np.mean([t[key] for t in teams.values()]))
    return {
        "teams": teams, "scoring_source": "schedule",
        "league": {
            "off_score": mean("off_score"), "def_score": mean("def_score"), "off_epa": mean("off_epa"),
            "def_epa": mean("def_epa"), "spp": mean("spp"), "pf": mean("pf"), "total_plays_pg": 121.0,
            "games_avg": mean("games"), "points_games_avg": mean("n_points"),
        },
    }


def make_ratings(extra: dict[str, dict] | None = None, *, prior=None):
    """Ratings with a strong BUF, a weak NYJ, a fast KC and slow DEN, the rest of the league average-ish."""
    teams = {
        "BUF": team_row(pf=30.0, pa=17.0, off=85, dfn=75, off_epa=0.14, def_epa=-0.09),
        "NYJ": team_row(pf=16.0, pa=29.0, off=15, dfn=20, off_epa=-0.12, def_epa=0.10),
        "KC": team_row(pf=24.0, pa=21.0, off=65, dfn=55, off_epa=0.05, def_epa=-0.02, spp=27.0),
        "DEN": team_row(pf=21.0, pa=20.0, off=45, dfn=60, off_epa=0.0, def_epa=-0.03, spp=33.5),
        "DAL": team_row(pf=23.0, pa=23.0),
        "PHI": team_row(pf=22.0, pa=22.0),
        "MIA": team_row(pf=22.5, pa=22.5, spp=27.0),
        "NE": team_row(pf=22.5, pa=22.5, spp=27.2),
        "TEN": team_row(pf=22.5, pa=22.5, spp=33.8),
        "SEA": team_row(pf=22.5, pa=22.5, spp=33.6),
    }
    teams.update(extra or {})
    return m.blend_ratings(snapshot(teams), prior, season=2026, prior_season=2025)


def qb_frame(rows):
    """Weekly player-stat frame.  Each row: (player_id, name, team, week, att, cmp, yds, td, int, sacks, epa)."""
    return pd.DataFrame([
        {"player_id": r[0], "player_display_name": r[1], "position": "QB", "season_type": "REG", "team": r[2], "week": r[3],
         "attempts": r[4], "completions": r[5], "passing_yards": r[6], "passing_tds": r[7], "passing_interceptions": r[8],
         "sacks_suffered": r[9], "passing_epa": r[10]}
        for r in rows
    ])


def season_of_qb(team_qbs: dict[str, tuple[str, str, float]], weeks=range(1, 11), att=35):
    """``{team: (player_id, name, epa_per_dropback)}`` -> a QB frame, ``att`` attempts a week."""
    rows = []
    for team, (pid, name, epa_db) in team_qbs.items():
        for week in weeks:
            rows.append((pid, name, team, week, att, int(att * 0.64), att * 7, 2, 1, 2, epa_db * (att + 2)))
    return qb_frame(rows)


def make_qb_context_frame():
    prior = season_of_qb({
        "BUF": ("00-BUF", "Josh Allen", 0.22), "NYJ": ("00-NYJ", "Geno Smith", -0.08), "KC": ("00-KC", "Patrick Mahomes", 0.12),
        "DEN": ("00-DEN", "Bo Nix", 0.05), "DAL": ("00-DAL", "Dak Prescott", 0.15), "PHI": ("00-PHI", "Jalen Hurts", 0.04),
        "MIA": ("00-MIA", "Tua Tagovailoa", 0.03), "NE": ("00-NE", "Drake Maye", 0.10),
        "TEN": ("00-TEN", "Cam Ward", -0.12), "SEA": ("00-SEA", "Sam Darnold", 0.08),
    }, weeks=range(1, 18), att=34)
    backups = qb_frame([("00-BAK", "Case Keenum", "NYJ", w, 28, 15, 150, 0, 2, 3, -0.30 * 31) for w in (3, 4)])
    return pd.concat([prior, backups], ignore_index=True)


def make_qb_context():
    return m.build_qb_context(None, make_qb_context_frame(), season=2026, prior_season=2025)


def analysis(home="BUF", away="NYJ", *, ratings=None, qb_context=None, **game_kwargs):
    game = {"game_id": f"2026_05_{away}_{home}", "week": 5, "home_team": home, "away_team": away,
            "kickoff_label": "Sun Oct 11 - 1:00 PM ET", "home_qb": None, "away_qb": None}
    game.update(game_kwargs)
    return m.analyze_game(game, ratings if ratings is not None else make_ratings(), qb_context if qb_context is not None else make_qb_context())


class _CacheIsolated(unittest.TestCase):
    def setUp(self):
        m.clear_caches()
        self._saved = {name: getattr(m, name) for name in ("_download_schedule", "_download_pbp", "_download_player_stats")}

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(m, name, value)
        m.clear_caches()


# --------------------------------------------------------------------------
# Season / schedule
# --------------------------------------------------------------------------

class CurrentSeasonTests(unittest.TestCase):
    def test_march_through_december_is_that_calendar_year(self):
        for month, day in ((3, 1), (7, 4), (9, 10), (12, 31)):
            self.assertEqual(m.current_season(date(2026, month, day)), 2026, (month, day))

    def test_january_and_february_belong_to_the_prior_season(self):
        for month, day in ((1, 1), (1, 31), (2, 1), (2, 28)):
            self.assertEqual(m.current_season(date(2027, month, day)), 2026, (month, day))

    def test_accepts_datetime_and_defaults_to_today(self):
        self.assertEqual(m.current_season(datetime(2026, 10, 3, 12, 0)), 2026)
        self.assertIsInstance(m.current_season(), int)


class ScheduleNormalizationTests(_CacheIsolated):
    def test_played_requires_both_scores_not_nan(self):
        raw = raw_schedule([
            raw_game(1, "NE", "SEA", away_score=10, home_score=13),
            raw_game(1, "SF", "LA", away_score=27, home_score=float("nan")),   # NaN home score: not played
            raw_game(1, "DEN", "KC"),                                            # both NaN: not played
            raw_game(1, "GB", "MIN", away_score=0, home_score=0),                # 0-0 is a real (played) score
        ])
        frame = m.normalize_schedule(raw)
        played = dict(zip(frame["home_team"], frame["played"]))
        self.assertEqual(played, {"SEA": True, "LA": False, "KC": False, "MIN": True})
        self.assertTrue(frame["played"].dtype == bool)

    def test_none_scores_are_not_played(self):
        raw = raw_schedule([raw_game(1, "NE", "SEA")])
        raw["home_score"] = raw["home_score"].astype(object)
        raw.loc[0, "home_score"] = None
        raw.loc[0, "away_score"] = 5
        self.assertFalse(bool(m.normalize_schedule(raw)["played"].iloc[0]))

    def test_regular_season_only_and_team_codes_normalised(self):
        raw = raw_schedule([
            raw_game(1, "LAR", "WSH"),
            raw_game(19, "KC", "BUF", game_type="POST"),
            raw_game(0, "KC", "BUF", game_type="PRE"),
        ])
        frame = m.normalize_schedule(raw)
        self.assertEqual(len(frame), 1)
        self.assertEqual((frame.loc[0, "away_team"], frame.loc[0, "home_team"]), ("LA", "WAS"))

    def test_market_columns_keep_nflverse_sign_and_nan_when_missing(self):
        raw = raw_schedule([
            raw_game(5, "TB", "DAL", spread=9.5, total=47.5, home_ml=-535, away_ml=400),
            raw_game(6, "SF", "SEA"),
        ])
        frame = m.normalize_schedule(raw).set_index("home_team")
        self.assertEqual(frame.loc["DAL", "market_spread_home"], 9.5)  # positive = home favored
        self.assertEqual(frame.loc["DAL", "market_home_ml"], -535)
        self.assertTrue(math.isnan(frame.loc["SEA", "market_spread_home"]))

    def test_kickoff_label_format(self):
        raw = raw_schedule([
            raw_game(5, "PHI", "JAX", gameday="2026-10-11", gametime="09:30"),
            raw_game(5, "CIN", "MIA", gameday="2026-10-11", gametime="13:00"),
            raw_game(5, "TB", "DAL", gameday="2026-10-08", gametime="20:15"),
            raw_game(5, "SF", "SEA", gameday="2026-10-11", gametime="00:05"),
        ])
        labels = dict(zip(m.normalize_schedule(raw)["home_team"], m.normalize_schedule(raw)["kickoff_label"]))
        self.assertEqual(labels["MIA"], "Sun Oct 11 - 1:00 PM ET")
        self.assertEqual(labels["JAX"], "Sun Oct 11 - 9:30 AM ET")
        self.assertEqual(labels["DAL"], "Thu Oct 8 - 8:15 PM ET")
        self.assertEqual(labels["SEA"], "Sun Oct 11 - 12:05 AM ET")
        self.assertEqual(m._kickoff_label("", ""), "TBD")
        self.assertEqual(m._kickoff_label("2026-10-11", ""), "Sun Oct 11")

    def test_rows_are_sorted_by_kickoff(self):
        raw = raw_schedule([
            raw_game(5, "CIN", "MIA", gameday="2026-10-11", gametime="13:00"),
            raw_game(5, "TB", "DAL", gameday="2026-10-08", gametime="20:15"),
            raw_game(5, "BAL", "ATL", gameday="2026-10-11", gametime="20:20"),
        ])
        self.assertEqual(list(m.normalize_schedule(raw)["home_team"]), ["DAL", "MIA", "ATL"])

    def test_empty_and_degenerate_inputs(self):
        self.assertTrue(m.normalize_schedule(None).empty)
        self.assertTrue(m.normalize_schedule(pd.DataFrame()).empty)
        self.assertEqual(list(m.normalize_schedule(None).columns), m.SCHEDULE_COLUMNS)
        self.assertTrue(m.normalize_schedule(raw_schedule([raw_game(1, "A", "B", game_type="POST")])).empty)

    def test_normalize_team_aliases(self):
        for alias, code in (("LAR", "LA"), ("wsh", "WAS"), ("JAC", "JAX"), ("OAK", "LV"), ("ARZ", "ARI"), ("kc", "KC"), (" sea ", "SEA")):
            self.assertEqual(m.normalize_team(alias), code)
        self.assertEqual(m.team_name("SEA"), "Seattle Seahawks")


def _schedule_frame(played_weeks: dict[int, int], total_weeks=4, per_week=2):
    """Frame with ``per_week`` games per week; ``played_weeks[w]`` of them are played."""
    rows = []
    teams = ["KC", "BUF", "DAL", "PHI", "SEA", "SF", "DEN", "MIA"]
    for week in range(1, total_weeks + 1):
        for i in range(per_week):
            done = i < played_weeks.get(week, 0)
            rows.append(raw_game(week, teams[(2 * i + 1) % 8], teams[(2 * i) % 8], gameday=f"2026-09-{10 + week:02d}", gametime=f"{13 + i}:00",
                                 away_score=17 if done else None, home_score=24 if done else None))
    return m.normalize_schedule(raw_schedule(rows))


class WeekSelectionTests(unittest.TestCase):
    def test_weeks_in_schedule_sorted_ints(self):
        schedule = _schedule_frame({})
        self.assertEqual(m.weeks_in_schedule(schedule), [1, 2, 3, 4])
        self.assertTrue(all(isinstance(w, int) for w in m.weeks_in_schedule(schedule)))
        self.assertEqual(m.weeks_in_schedule(m.normalize_schedule(None)), [])

    def test_default_week_is_first_week_with_an_unplayed_game(self):
        self.assertEqual(m.default_week(_schedule_frame({1: 2, 2: 2})), 3)

    def test_partially_played_week_is_still_current(self):
        self.assertEqual(m.default_week(_schedule_frame({1: 2, 2: 1})), 2)

    def test_all_played_returns_last_week(self):
        self.assertEqual(m.default_week(_schedule_frame({1: 2, 2: 2, 3: 2, 4: 2})), 4)

    def test_none_played_returns_week_one(self):
        self.assertEqual(m.default_week(_schedule_frame({})), 1)

    def test_empty_schedule_defaults_to_week_one(self):
        self.assertEqual(m.default_week(m.normalize_schedule(None)), 1)

    def test_nan_scores_do_not_count_as_played(self):
        # Week 1 has real scores, week 2 has NaN scores (nflverse's unplayed encoding): default must be week 2.
        schedule = _schedule_frame({1: 2})
        self.assertTrue(schedule[schedule.week == 2]["home_score"].isna().all())
        self.assertEqual(m.default_week(schedule), 2)


class GamesForWeekTests(unittest.TestCase):
    def setUp(self):
        raw = raw_schedule([
            raw_game(4, "NE", "SEA", gameday="2026-10-04", gametime="13:00", away_score=10, home_score=13, home_qb="Sam Darnold", away_qb="Drake Maye"),
            raw_game(5, "TB", "DAL", gameday="2026-10-08", gametime="20:15", spread=9.5, total=47.5, home_ml=-535, away_ml=400,
                     home_qb="Dak Prescott", away_qb="Jalon Daniels"),
            raw_game(5, "CIN", "MIA", gameday="2026-10-11", gametime="13:00", spread=-8.5, home_qb="Malik Willis", away_qb="Joe Burrow"),
            raw_game(5, "SF", "SEA", gameday="2026-10-11", gametime="09:30"),
            raw_game(6, "NE", "SEA", gameday="2026-10-18", gametime="13:00"),
        ])
        self.schedule = m.normalize_schedule(raw)

    def test_ordered_by_kickoff_with_label_and_status(self):
        games = m.games_for_week(self.schedule, 5)
        self.assertEqual([g["label"] for g in games], ["TB @ DAL", "SF @ SEA", "CIN @ MIA"])
        self.assertTrue(all(g["status"] == "Upcoming" and g["played"] is False for g in games))
        self.assertTrue(all(g["week"] == 5 for g in games))

    def test_final_status_is_away_then_home(self):
        game = m.games_for_week(self.schedule, 4)[0]
        self.assertEqual(game["status"], "Final 10-13")
        self.assertEqual((game["away_score"], game["home_score"]), (10, 13))
        self.assertTrue(game["played"])

    def test_missing_values_are_none_not_nan(self):
        games = {g["label"]: g for g in m.games_for_week(self.schedule, 5)}
        sf = games["SF @ SEA"]
        for key in ("market_spread_home", "market_total", "market_home_ml", "market_away_ml", "away_qb", "away_score", "home_score"):
            self.assertIsNone(sf[key], key)
        tb = games["TB @ DAL"]
        self.assertEqual((tb["market_spread_home"], tb["market_total"], tb["market_home_ml"], tb["market_away_ml"]), (9.5, 47.5, -535.0, 400.0))
        assert_all_finite(self, games)

    def test_unlisted_starter_is_filled_from_most_recent_earlier_game(self):
        game = {g["label"]: g for g in m.games_for_week(self.schedule, 6)}["NE @ SEA"]
        self.assertEqual((game["home_qb"], game["home_qb_source"]), ("Sam Darnold", "last_game"))
        self.assertEqual((game["away_qb"], game["away_qb_source"]), ("Drake Maye", "last_game"))
        listed = {g["label"]: g for g in m.games_for_week(self.schedule, 5)}["TB @ DAL"]
        self.assertEqual((listed["home_qb_source"], listed["away_qb_source"]), ("listed", "listed"))
        none = {g["label"]: g for g in m.games_for_week(self.schedule, 5)}["SF @ SEA"]
        self.assertEqual(none["away_qb_source"], "none")  # SF has no earlier named starter in this fixture
        self.assertEqual(none["home_qb_source"], "last_game")  # SEA's week-4 starter

    def test_fallback_never_looks_at_later_games(self):
        raw = raw_schedule([
            raw_game(3, "NE", "SEA", gameday="2026-09-27", home_qb="Starter Before"),
            raw_game(4, "NE", "SEA", gameday="2026-10-04"),                      # no starter listed
            raw_game(5, "NE", "SEA", gameday="2026-10-11", home_qb="Starter After"),
        ])
        schedule = m.normalize_schedule(raw)
        game = m.games_for_week(schedule, 4)[0]
        self.assertEqual((game["home_qb"], game["home_qb_source"]), ("Starter Before", "last_game"))
        self.assertEqual(game["away_qb_source"], "none")

    def test_empty_inputs(self):
        self.assertEqual(m.games_for_week(self.schedule, 99), [])
        self.assertEqual(m.games_for_week(m.normalize_schedule(None), 1), [])


class LoadScheduleTests(_CacheIsolated):
    def test_load_schedule_returns_normalised_frame(self):
        m._download_schedule = lambda season: raw_schedule([raw_game(1, "NE", "SEA", season=season), raw_game(19, "KC", "BUF", game_type="POST", season=season)])
        frame = m.load_schedule(2026)
        self.assertEqual(list(frame.columns), m.SCHEDULE_COLUMNS)
        self.assertEqual(len(frame), 1)

    def test_returned_frame_is_a_copy(self):
        m._download_schedule = lambda season: raw_schedule([raw_game(1, "NE", "SEA", season=season)])
        first = m.load_schedule(2026)
        first.loc[0, "home_team"] = "XXX"
        self.assertEqual(m.load_schedule(2026).loc[0, "home_team"], "SEA")

    def test_download_failure_raises_schedule_unavailable_with_message(self):
        def boom(season):
            raise ConnectionError("no network")
        m._download_schedule = boom
        with self.assertRaises(m.ScheduleUnavailable) as caught:
            m.load_schedule(2026)
        self.assertIn("2026", str(caught.exception))
        self.assertIsInstance(caught.exception, RuntimeError)

    def test_empty_schedule_raises_schedule_unavailable(self):
        m._download_schedule = lambda season: raw_schedule([])
        with self.assertRaises(m.ScheduleUnavailable):
            m.load_schedule(2030)

    def test_failures_are_not_cached(self):
        calls = {"n": 0}

        def flaky(season):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("slow")
            return raw_schedule([raw_game(1, "NE", "SEA", season=season)])
        m._download_schedule = flaky
        with self.assertRaises(m.ScheduleUnavailable):
            m.load_schedule(2026)
        self.assertEqual(len(m.load_schedule(2026)), 1)

    def test_load_moneyline_inputs_propagates_schedule_failure(self):
        def boom(season):
            raise OSError("offline")
        m._download_schedule = boom
        with self.assertRaises(m.ScheduleUnavailable):
            m.load_moneyline_inputs(2026)


# --------------------------------------------------------------------------
# Ratings
# --------------------------------------------------------------------------

class BlendRatingsTests(unittest.TestCase):
    def test_every_team_present_and_nothing_is_nan_with_missing_teams(self):
        ratings = make_ratings()
        self.assertEqual(set(ratings["teams"]), set(NFL_TEAMS))
        assert_all_finite(self, ratings)
        self.assertEqual(ratings["teams"]["ATL"]["data_quality"], "none")
        self.assertTrue(any("ATL" in w for w in ratings["meta"]["warnings"]))

    def test_team_with_no_data_is_exactly_league_average(self):
        ratings = make_ratings()
        row, league = ratings["teams"]["ATL"], ratings["league"]
        self.assertEqual(row["points_for_pg"], round(league["points_pg"], 2))
        self.assertEqual(row["points_against_pg"], round(league["points_pg"], 2))
        self.assertEqual(row["off_epa"], round(league["off_epa"], 4))
        self.assertEqual(row["pace_spp"], round(league["pace_spp"], 2))
        self.assertEqual(row["neutral_margin"], 0.0)
        self.assertEqual(row["strength_rating"], 50.0)
        self.assertEqual(row["blend_weight_current"], 0.0)

    def test_zero_current_season_games_uses_regressed_prior_and_never_nan(self):
        prior = snapshot({"BUF": team_row(pf=30.0, pa=17.0, off=85, dfn=75, off_epa=0.14, def_epa=-0.09, games=17),
                          "NYJ": team_row(pf=16.0, pa=29.0, off=15, dfn=20, off_epa=-0.12, def_epa=0.10, games=17)})
        ratings = m.blend_ratings(None, prior, season=2026, prior_season=2025)
        assert_all_finite(self, ratings)
        buf = ratings["teams"]["BUF"]
        self.assertEqual(buf["games_current"], 0)
        self.assertEqual(buf["games_prior"], 17)
        self.assertEqual(buf["blend_weight_current"], 0.0)
        self.assertEqual(buf["data_quality"], "prior_only")
        # deviation from the league mean is carried at PRIOR_CARRYOVER
        league_pf = ratings["league"]["points_pg"]
        self.assertAlmostEqual(buf["points_for_pg"], league_pf + m.PRIOR_CARRYOVER * (30.0 - league_pf), places=1)
        self.assertTrue(any("No 2026" in w for w in ratings["meta"]["warnings"]))

    def test_blend_weight_is_n_over_n_plus_k(self):
        current = snapshot({"BUF": team_row(games=6.0), "NYJ": team_row(games=3.0), "KC": team_row(games=12.0)})
        prior = snapshot({"BUF": team_row(games=17.0), "NYJ": team_row(games=17.0), "KC": team_row(games=17.0)})
        ratings = m.blend_ratings(current, prior, season=2026, prior_season=2025)
        weights = ratings["meta"]["blend_weight_current"]
        self.assertAlmostEqual(weights["BUF"], 6.0 / (6.0 + m.BLEND_K_GAMES), places=3)
        self.assertAlmostEqual(weights["NYJ"], 3.0 / (3.0 + m.BLEND_K_GAMES), places=3)
        self.assertAlmostEqual(weights["KC"], 12.0 / (12.0 + m.BLEND_K_GAMES), places=3)
        self.assertEqual(ratings["meta"]["games_current"]["BUF"], 6)
        self.assertEqual(ratings["meta"]["k_games"], m.BLEND_K_GAMES)

    def test_current_season_dominates_as_games_accumulate(self):
        def rating_after(games):
            current = snapshot({"BUF": team_row(pf=34.0, pa=14.0, games=games), "NYJ": team_row(pf=11.0, pa=31.0, games=games)})
            prior = snapshot({"BUF": team_row(pf=20.0, pa=24.0, games=17), "NYJ": team_row(pf=25.0, pa=20.0, games=17)})
            return m.blend_ratings(current, prior, season=2026, prior_season=2025)["teams"]["BUF"]["points_for_pg"]
        self.assertLess(rating_after(2), rating_after(6))
        self.assertLess(rating_after(6), rating_after(16))

    def test_prior_only_deviation_is_regressed(self):
        prior = snapshot({"BUF": team_row(off_epa=0.10, games=17), "NYJ": team_row(off_epa=-0.10, games=17)})
        buf = m.blend_ratings(None, prior, season=2026, prior_season=2025)["teams"]["BUF"]
        self.assertAlmostEqual(buf["off_epa"], 0.10 * m.PRIOR_CARRYOVER, places=3)

    def test_current_only_is_shrunk_by_its_own_weight(self):
        current = snapshot({"BUF": team_row(off_epa=0.10, games=6.0), "NYJ": team_row(off_epa=-0.10, games=6.0)})
        buf = m.blend_ratings(current, None, season=2026, prior_season=2025)["teams"]["BUF"]
        self.assertAlmostEqual(buf["off_epa"], 0.10 * 0.5, places=3)  # n=6, K=6 -> w=0.5, the missing prior counts as average
        self.assertEqual(buf["data_quality"], "current_only")

    def test_both_seasons_missing_gives_neutral_everywhere(self):
        ratings = m.blend_ratings(None, None, season=2026, prior_season=2025)
        assert_all_finite(self, ratings)
        self.assertTrue(all(r["strength_rating"] == 50.0 and r["data_quality"] == "none" for r in ratings["teams"].values()))
        self.assertEqual(ratings["league"]["points_pg"], 22.5)

    def test_strength_rating_orders_teams_and_stays_in_range(self):
        ratings = make_ratings()
        buf, nyj, atl = (ratings["teams"][t]["strength_rating"] for t in ("BUF", "NYJ", "ATL"))
        self.assertGreater(buf, atl)
        self.assertGreater(atl, nyj)
        self.assertGreater(buf, 60)
        self.assertLess(nyj, 40)
        for row in ratings["teams"].values():
            self.assertTrue(0.0 <= row["strength_rating"] <= 100.0)
            self.assertTrue(0.0 <= row["offensive_efficiency"] <= 100.0)
            self.assertTrue(0.0 <= row["defensive_efficiency"] <= 100.0)

    def test_strength_rating_formula_matches_neutral_margin(self):
        row = make_ratings()["teams"]["BUF"]
        self.assertAlmostEqual(row["strength_rating"], round(50.0 + m.STRENGTH_POINTS_SCALE * row["neutral_margin"], 1), places=1)

    def test_net_efficiency_is_off_plus_def_minus_100(self):
        for row in make_ratings()["teams"].values():
            self.assertAlmostEqual(row["net_efficiency"], row["offensive_efficiency"] + row["defensive_efficiency"] - 100.0, places=1)

    def test_extreme_inputs_are_clamped(self):
        wild = {"BUF": team_row(pf=400.0, pa=-5.0, off=900, dfn=-50, off_epa=9.0, def_epa=-9.0, spp=0.5, games=17)}
        ratings = make_ratings(wild)
        buf = ratings["teams"]["BUF"]
        self.assertTrue(0 <= buf["offensive_efficiency"] <= 100 and 0 <= buf["defensive_efficiency"] <= 100)
        self.assertTrue(5 <= buf["points_for_pg"] <= 45 and 5 <= buf["points_against_pg"] <= 45)
        self.assertTrue(22 <= buf["pace_spp"] <= 40)
        assert_all_finite(self, ratings)

    def test_early_season_warning_mentions_weight(self):
        current = snapshot({"BUF": team_row(games=3.0), "NYJ": team_row(games=3.0)})
        ratings = m.blend_ratings(current, current, season=2026, prior_season=2025)
        self.assertTrue(any("games of 2026" in w for w in ratings["meta"]["warnings"]))


# --- synthetic play-by-play so the real build_team_season_stats path is exercised offline ---

def synthetic_pbp(games, *, season=2026, seed=0):
    """``games`` = [(week, home, away)].  120 run/pass plays per game, 6-play drives, 29 s between snaps."""
    rng = np.random.RandomState(seed)
    bias = {"KC": 0.20, "BUF": 0.05, "DAL": -0.05, "PHI": -0.15}
    rows = []
    for week, home, away in games:
        game_id = f"{season}_{week:02d}_{away}_{home}"
        score = {home: 0, away: 0}
        for i in range(120):
            offense, defense = (home, away) if (i // 6) % 2 == 0 else (away, home)
            if i % 6 == 5 and rng.rand() < 0.4:
                score[offense] += 7
            rows.append({
                "game_id": game_id, "season": season, "season_type": "REG", "week": week, "home_team": home, "away_team": away,
                "posteam": offense, "defteam": defense, "play_type": "pass" if rng.rand() < 0.55 else "run",
                "yards_gained": float(rng.randint(-2, 15)), "epa": float(rng.normal(bias.get(offense, 0) - bias.get(defense, 0) * 0.5, 0.6)),
                "success": float(rng.rand() < 0.45), "turnover": 0.0, "interception": 0.0, "fumble_lost": 0.0,
                "touchdown": float(i % 6 == 5 and score[offense] > 0), "down": float(rng.randint(1, 5)),
                "third_down_converted": 0.0, "third_down_failed": 0.0, "yardline_100": float(rng.randint(1, 99)),
                "drive": float(i // 6), "game_seconds_remaining": float(3600 - i * 29), "posteam_score_post": float(score[offense]),
            })
    return pd.DataFrame(rows)


SYN_GAMES = [(1, "BUF", "KC"), (1, "PHI", "DAL"), (2, "KC", "PHI"), (2, "DAL", "BUF"), (3, "BUF", "PHI"), (3, "DAL", "KC")]


def synthetic_schedule(season=2026, *, unplayed_week=None):
    rows = []
    for week, home, away in SYN_GAMES:
        done = unplayed_week is None or week != unplayed_week
        rows.append(raw_game(week, away, home, season=season, away_score=17 if done else None, home_score=24 if done else None))
    if unplayed_week is None:
        rows.append(raw_game(4, "PHI", "BUF", season=season))  # an unplayed game whose scores are NaN
    return raw_schedule(rows)


class SeasonSnapshotTests(unittest.TestCase):
    def test_snapshot_from_real_builder_is_finite_and_counts_games(self):
        pbp = synthetic_pbp(SYN_GAMES)
        schedule = m.normalize_schedule(synthetic_schedule())
        snap = m._season_snapshot(pbp, schedule)
        self.assertEqual(set(snap["teams"]), {"KC", "BUF", "DAL", "PHI"})
        for team in snap["teams"].values():
            self.assertEqual(team["games"], 3.0)
            self.assertEqual(team["n_points"], 3.0)  # the NaN-score week-4 game is excluded
            self.assertTrue(17.0 <= team["pf"] <= 24.0 and 17.0 <= team["pa"] <= 24.0)  # games are 24-17 / 17-24
            self.assertAlmostEqual(team["spp"], 29.0, places=3)
        assert_all_finite(self, snap)
        self.assertEqual(snap["league"]["total_plays_pg"], 120.0)

    def test_points_ignore_unplayed_games_with_nan_scores(self):
        schedule = m.normalize_schedule(synthetic_schedule())
        points = m._points_from_schedule(schedule)
        # Every team here played 3 games of 24-17 (home) / 17-24 (away); week-4 NaN game must not make anything NaN.
        for stats in points.values():
            self.assertTrue(math.isfinite(stats["pf"]) and math.isfinite(stats["pa"]))
        self.assertEqual(points["BUF"]["n"], 3)

    def test_play_by_play_points_fallback_when_schedule_missing(self):
        snap = m._season_snapshot(synthetic_pbp(SYN_GAMES), None)
        self.assertEqual(snap["scoring_source"], "play-by-play")
        self.assertTrue(all(math.isfinite(t["pf"]) and math.isfinite(t["pa"]) for t in snap["teams"].values()))

    def test_nothing_usable_returns_none(self):
        self.assertIsNone(m._season_snapshot(None, None))
        self.assertIsNone(m._season_snapshot(pd.DataFrame(), m.normalize_schedule(None)))

    def test_zero_pace_is_treated_as_missing(self):
        pbp = synthetic_pbp(SYN_GAMES).drop(columns=["game_seconds_remaining"])
        snap = m._season_snapshot(pbp, m.normalize_schedule(synthetic_schedule()))
        self.assertTrue(all("spp" not in t for t in snap["teams"].values()))
        ratings = m.blend_ratings(snap, None, season=2026, prior_season=2025)
        assert_all_finite(self, ratings)
        self.assertAlmostEqual(ratings["teams"]["KC"]["pace_spp"], ratings["league"]["pace_spp"], places=2)


class LoadTeamRatingsTests(_CacheIsolated):
    def _patch(self, *, pbp_fail=(), schedule_fail=()):
        def download_pbp(season):
            if season in pbp_fail:
                raise ConnectionError("pbp offline")
            return synthetic_pbp(SYN_GAMES, season=season, seed=season)

        def download_schedule(season):
            if season in schedule_fail:
                raise ConnectionError("schedule offline")
            return synthetic_schedule(season)
        m._download_pbp, m._download_schedule = download_pbp, download_schedule

    def test_end_to_end_blend_from_both_seasons(self):
        self._patch()
        ratings = m.load_team_ratings(2026)
        assert_all_finite(self, ratings)
        self.assertEqual(ratings["meta"]["season"], 2026)
        self.assertEqual(ratings["meta"]["prior_season"], 2025)
        kc = ratings["teams"]["KC"]
        self.assertEqual((kc["games_current"], kc["games_prior"]), (3, 3))
        self.assertAlmostEqual(kc["blend_weight_current"], 3.0 / (3.0 + m.BLEND_K_GAMES), places=3)
        self.assertEqual(kc["data_quality"], "full")
        self.assertEqual(len(ratings["teams"]), 32)
        self.assertIn("2026", ratings["meta"]["source"])

    def test_explicit_prior_season(self):
        self._patch()
        self.assertEqual(m.load_team_ratings(2026, 2024)["meta"]["prior_season"], 2024)

    def test_prior_season_failure_degrades_with_warning_and_never_raises(self):
        self._patch(pbp_fail=(2025,), schedule_fail=(2025,))
        ratings = m.load_team_ratings(2026)
        assert_all_finite(self, ratings)
        self.assertEqual(ratings["teams"]["KC"]["games_prior"], 0)
        self.assertTrue(any("2025" in w for w in ratings["meta"]["warnings"]))

    def test_total_failure_returns_neutral_ratings_with_warning(self):
        self._patch(pbp_fail=(2026, 2025), schedule_fail=(2026, 2025))
        ratings = m.load_team_ratings(2026)
        assert_all_finite(self, ratings)
        self.assertEqual(len(ratings["teams"]), 32)
        self.assertTrue(all(r["strength_rating"] == 50.0 for r in ratings["teams"].values()))
        self.assertTrue(ratings["meta"]["warnings"])
        self.assertTrue(any("every rating is the league average" in w for w in ratings["meta"]["warnings"]))
        self.assertLess(max(len(w) for w in ratings["meta"]["warnings"]), 400)  # does not list all 32 teams

    def test_current_season_play_by_play_failure_is_reported_even_when_scoring_loads(self):
        self._patch(pbp_fail=(2026,))
        ratings = m.load_team_ratings(2026)
        assert_all_finite(self, ratings)
        self.assertTrue(any("2026 play-by-play unavailable" in w for w in ratings["meta"]["warnings"]))
        self.assertGreater(ratings["teams"]["KC"]["games_current"], 0)  # scoring still counts

    def test_current_season_schedule_failure_is_reported(self):
        self._patch(schedule_fail=(2026,))
        ratings = m.load_team_ratings(2026)
        self.assertTrue(any("2026 schedule unavailable" in w for w in ratings["meta"]["warnings"]))

    def test_degraded_results_are_not_cached_for_long(self):
        calls = {"n": 0}

        def flaky(season):
            calls["n"] += 1
            if season == 2026 and calls["n"] <= 2:
                raise ConnectionError("offline")
            return synthetic_pbp(SYN_GAMES, season=season)
        m._download_pbp = flaky
        m._download_schedule = lambda season: synthetic_schedule(season)
        first = m.load_team_ratings(2026)
        self.assertTrue(any("play-by-play unavailable" in w for w in first["meta"]["warnings"]))
        key = ("ratings", 2026, 2025)
        expiry = m._MEMO[key][0]
        import time
        self.assertLessEqual(expiry - time.monotonic(), m._MEMO_DEGRADED_SECONDS + 1)  # short TTL, so a retry comes soon

    def test_schedule_failure_falls_back_to_play_by_play_points(self):
        self._patch(schedule_fail=(2026,))
        ratings = m.load_team_ratings(2026)
        assert_all_finite(self, ratings)
        self.assertGreater(ratings["teams"]["KC"]["games_current"], 0)

    def test_repeated_calls_are_cached_and_isolated(self):
        calls = {"n": 0}
        original = synthetic_pbp

        def counting(season):
            calls["n"] += 1
            return original(SYN_GAMES, season=season)
        m._download_pbp = counting
        m._download_schedule = lambda season: synthetic_schedule(season)
        first = m.load_team_ratings(2026)
        second = m.load_team_ratings(2026)
        self.assertEqual(calls["n"], 2)  # one download per season, then served from cache
        first["teams"]["KC"]["strength_rating"] = -1.0
        self.assertNotEqual(second["teams"]["KC"]["strength_rating"], -1.0)


# --------------------------------------------------------------------------
# Quarterbacks
# --------------------------------------------------------------------------

class PasserRatingTests(unittest.TestCase):
    def test_known_value(self):
        # 65/100, 800 yds, 6 TD, 2 INT -> (1.75 + 1.25 + 1.2 + 1.875) / 6 * 100
        self.assertAlmostEqual(m._passer_rating(65, 100, 800, 6, 2), 101.25, places=2)

    def test_perfect_rating_and_floor(self):
        self.assertAlmostEqual(m._passer_rating(78, 100, 1300, 13, 0), 158.33, places=1)
        self.assertEqual(m._passer_rating(0, 100, 0, 0, 20), 0.0)
        self.assertEqual(m._passer_rating(0, 0, 0, 0, 0), 0.0)


class QBContextTests(_CacheIsolated):
    def setUp(self):
        super().setUp()
        self.context = make_qb_context()

    def test_structure(self):
        for key in ("teams", "players", "names", "league", "meta"):
            self.assertIn(key, self.context)
        allen = self.context["players"]["00-BUF"]
        for key in ("name", "team", "attempts", "attempts_current", "attempts_prior", "eff_attempts", "eff_dropbacks",
                    "passer_rating", "epa_per_dropback", "role", "is_team_primary", "impact_points"):
            self.assertIn(key, allen)
        self.assertEqual(allen["role"], "starter")
        self.assertEqual(allen["team"], "BUF")
        assert_all_finite(self, self.context)

    def test_prior_season_is_recency_weighted(self):
        allen = self.context["players"]["00-BUF"]
        self.assertEqual(allen["attempts_prior"], 17 * 34)
        self.assertAlmostEqual(allen["eff_attempts"], 17 * 34 * m.QB_PRIOR_WEIGHT, delta=1.0)
        self.assertEqual(allen["attempts_current"], 0)

    def test_current_and_prior_are_pooled_by_attempts(self):
        current = season_of_qb({"BUF": ("00-BUF", "Josh Allen", -0.10)}, weeks=range(1, 5), att=40)
        prior = season_of_qb({"BUF": ("00-BUF", "Josh Allen", 0.30)}, weeks=range(1, 18), att=34)
        context = m.build_qb_context(current, prior, season=2026, prior_season=2025)
        allen = context["players"]["00-BUF"]
        # pooled EPA/dropback sits between the two seasons, much closer to the larger (prior) sample
        self.assertTrue(-0.10 < allen["epa_per_dropback"] < 0.30)
        self.assertGreater(allen["epa_per_dropback"], 0.15)
        self.assertEqual((allen["attempts_current"], allen["attempts_prior"]), (160, 578))

    def test_starter_and_backup_roles_and_replacement_baseline(self):
        keenum = self.context["players"]["00-BAK"]
        self.assertEqual(keenum["role"], "backup")
        baseline = self.context["league"]
        self.assertLess(baseline["replacement_epa"], baseline["epa_per_dropback"])
        self.assertGreaterEqual(baseline["starters_counted"], 8)
        for key in ("passer_rating", "epa_per_dropback", "replacement_epa", "replacement_passer_rating"):
            self.assertTrue(math.isfinite(baseline[key]))

    def test_impact_sign_and_ordering(self):
        players, baseline = self.context["players"], self.context["league"]
        allen = m.qb_impact_points(players["00-BUF"], baseline)
        smith = m.qb_impact_points(players["00-NYJ"], baseline)
        keenum = m.qb_impact_points(players["00-BAK"], baseline)
        self.assertGreater(allen, 0)
        self.assertLess(smith, 0)
        self.assertLess(keenum, smith)  # a backup is worse than a below-average starter
        self.assertGreater(allen, smith)

    def test_impact_is_clamped(self):
        baseline = {"epa_per_dropback": 0.0, "replacement_epa": -0.1, "passer_rating": 90.0, "replacement_passer_rating": 70.0}
        huge = {"role": "starter", "epa_per_dropback": 5.0, "eff_dropbacks": 5000.0}
        awful = {"role": "starter", "epa_per_dropback": -5.0, "eff_dropbacks": 5000.0}
        self.assertEqual(m.qb_impact_points(huge, baseline), m.QB_IMPACT_CLAMP)
        self.assertEqual(m.qb_impact_points(awful, baseline), -m.QB_IMPACT_CLAMP)

    def test_small_samples_are_shrunk_toward_the_prior(self):
        baseline = {"epa_per_dropback": 0.05, "replacement_epa": -0.10, "passer_rating": 92.0, "replacement_passer_rating": 74.0}
        small = {"role": "starter", "epa_per_dropback": 0.30, "eff_dropbacks": 40.0}
        large = {"role": "starter", "epa_per_dropback": 0.30, "eff_dropbacks": 1200.0}
        self.assertLess(m.qb_impact_points(small, baseline), m.qb_impact_points(large, baseline))
        self.assertGreater(m.qb_impact_points(small, baseline), 0)

    def test_backup_is_shrunk_to_replacement_not_to_average(self):
        baseline = {"epa_per_dropback": 0.05, "replacement_epa": -0.10, "passer_rating": 92.0, "replacement_passer_rating": 74.0}
        unproven_starter = {"role": "starter", "epa_per_dropback": 0.05, "eff_dropbacks": 20.0}
        backup = {"role": "backup", "epa_per_dropback": 0.05, "eff_dropbacks": 20.0}
        self.assertAlmostEqual(m.qb_impact_points(unproven_starter, baseline), 0.0, delta=0.05)
        self.assertLess(m.qb_impact_points(backup, baseline), -1.0)

    def test_unknown_or_missing_qb_is_priced_as_unproven_between_average_and_replacement(self):
        baseline = self.context["league"]
        unproven = m.qb_impact_points({"role": "unknown"}, baseline)
        backup = m.qb_impact_points({"role": "backup", "eff_dropbacks": 0.0}, baseline)
        self.assertLess(unproven, 0)
        self.assertLess(backup, unproven)  # a measured backup is worse than an unproven starter
        self.assertEqual(m.qb_impact_points(None, baseline), unproven)
        self.assertEqual(m.qb_impact_points({}, baseline), unproven)
        self.assertAlmostEqual(unproven, m.QB_POINTS_PER_EPA * -m.QB_UNPROVEN_DELTA_EPA, delta=0.01)
        self.assertGreaterEqual(unproven, -m.QB_IMPACT_CLAMP)

    def test_baseline_without_derived_keys_is_completed_relative_to_its_average(self):
        bare = {"epa_per_dropback": 0.10, "passer_rating": 95.0}
        self.assertAlmostEqual(m.qb_impact_points({"role": "unknown"}, bare), m.QB_POINTS_PER_EPA * -m.QB_UNPROVEN_DELTA_EPA, delta=0.01)
        self.assertAlmostEqual(m.qb_impact_points({"role": "backup", "eff_dropbacks": 0.0}, bare), m.QB_POINTS_PER_EPA * -m.QB_REPLACEMENT_DELTA_DEFAULT, delta=0.01)
        self.assertEqual(m.qb_impact_points({"role": "starter"}, bare), 0.0)

    def test_rating_fallback_when_no_epa(self):
        baseline = {"epa_per_dropback": 0.05, "replacement_epa": -0.10, "passer_rating": 92.0, "replacement_passer_rating": 74.0}
        good = {"role": "starter", "epa_per_dropback": None, "passer_rating": 110.0, "eff_attempts": 900.0}
        poor = {"role": "starter", "epa_per_dropback": None, "passer_rating": 70.0, "eff_attempts": 900.0}
        self.assertGreater(m.qb_impact_points(good, baseline), 0)
        self.assertLess(m.qb_impact_points(poor, baseline), 0)

    def test_team_primary_and_embedded_impact(self):
        buf = self.context["teams"]["BUF"]
        self.assertEqual(buf["name"], "Josh Allen")
        self.assertTrue(buf["is_team_primary"])
        embedded = buf["embedded_impact"]
        self.assertIsNone(embedded["current"])
        self.assertAlmostEqual(embedded["prior"], buf["impact_points"], places=2)
        nyj = self.context["teams"]["NYJ"]["embedded_impact"]["prior"]
        self.assertLess(nyj, self.context["players"]["00-NYJ"]["impact_points"] + 1e-9)  # Keenum's snaps drag the team's embedded value down

    def test_resolve_by_id_name_suffix_and_team_primary(self):
        ctx = self.context
        self.assertEqual(m.resolve_qb(ctx, "BUF", player_id="00-BUF")["source"], "schedule_id")
        by_name = m.resolve_qb(ctx, "BUF", name="josh  allen")
        self.assertEqual((by_name["source"], by_name["player_id"], by_name["found"]), ("schedule_name", "00-BUF", True))
        self.assertEqual(m.resolve_qb(ctx, "SEA", name="Sam Darnold Jr.")["name"], "Sam Darnold")
        primary = m.resolve_qb(ctx, "KC")
        self.assertEqual((primary["name"], primary["source"]), ("Patrick Mahomes", "team_primary"))
        self.assertNotIn("embedded_impact", primary)

    def test_resolve_unknown_name_and_unknown_team(self):
        unknown = m.resolve_qb(self.context, "BUF", name="Nobody Atall")
        self.assertEqual((unknown["role"], unknown["found"], unknown["source"], unknown["name"]), ("unknown", False, "schedule_unknown", "Nobody Atall"))
        self.assertIsNone(unknown["passer_rating"])
        none = m.resolve_qb(self.context, "ZZZ")
        self.assertEqual((none["role"], none["name"]), ("unknown", "Unknown QB"))
        self.assertEqual(m.resolve_qb(None, "BUF")["role"], "unknown")

    def test_neutral_context_when_no_data(self):
        context = m.build_qb_context(None, None, season=2026, prior_season=2025)
        self.assertEqual(context["players"], {})
        self.assertTrue(context["meta"]["warnings"])
        assert_all_finite(self, context)

    def test_load_qb_context_fail_soft(self):
        def boom(season):
            raise ConnectionError("offline")
        m._download_player_stats = boom
        context = m.load_qb_context(2026)
        self.assertEqual(context["players"], {})
        self.assertTrue(context["meta"]["warnings"])
        assert_all_finite(self, context)

    def test_load_qb_context_end_to_end_and_cached(self):
        calls = {"n": 0}

        def download(season):
            calls["n"] += 1
            return season_of_qb({"BUF": ("00-BUF", "Josh Allen", 0.2), "NYJ": ("00-NYJ", "Geno Smith", -0.05)}, weeks=range(1, 6 if season == 2026 else 18))
        m._download_player_stats = download
        first = m.load_qb_context(2026)
        second = m.load_qb_context(2026)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(first["teams"]["BUF"]["attempts_current"], 5 * 35)
        self.assertEqual(second["meta"]["season"], 2026)

    def test_malformed_frames_do_not_raise(self):
        no_identity = pd.DataFrame({"position": ["QB"], "attempts": [30], "team": ["BUF"], "week": [1]})
        context = m.build_qb_context(no_identity, None, season=2026, prior_season=2025)
        self.assertEqual(context["players"], {})
        no_epa = pd.DataFrame({"player_id": ["00-X"], "position": ["QB"], "attempts": [30], "completions": [20], "team": ["BUF"], "week": [1]})
        context = m.build_qb_context(no_epa, None, season=2026, prior_season=2025)
        self.assertIn("00-X", context["players"])
        self.assertEqual(context["players"]["00-X"]["name"], "00-X")
        assert_all_finite(self, context)
        self.assertTrue(math.isfinite(m.qb_impact_points(context["players"]["00-X"], context["league"])))


# --------------------------------------------------------------------------
# Game analysis
# --------------------------------------------------------------------------

class SignConventionTests(unittest.TestCase):
    def test_home_favorite_has_positive_spread_and_win_probability_above_half(self):
        result = analysis("BUF", "NYJ")
        self.assertGreater(result["scoring"]["model_spread_home"], 3.0)
        self.assertGreater(result["probabilities"]["home_win"], 0.6)
        self.assertEqual((result["favored"], result["favored_team"]), ("home", "BUF"))
        self.assertAlmostEqual(result["margin"], abs(result["scoring"]["model_spread_home"]))

    def test_road_favorite_has_negative_spread(self):
        result = analysis("NYJ", "BUF")
        self.assertLess(result["scoring"]["model_spread_home"], 0)
        self.assertLess(result["probabilities"]["home_win"], 0.5)
        self.assertEqual((result["favored"], result["favored_team"]), ("away", "BUF"))

    def test_swapping_venue_flips_the_sign_and_differs_by_twice_home_field(self):
        home_strong = analysis("BUF", "NYJ")["scoring"]["model_spread_home"]
        road_strong = analysis("NYJ", "BUF")["scoring"]["model_spread_home"]
        self.assertAlmostEqual(home_strong + road_strong, 2 * m.HOME_FIELD_POINTS, places=1)

    def test_equal_teams_are_separated_only_by_home_field(self):
        ratings = make_ratings({"DAL": team_row(), "PHI": team_row()})
        ratings["teams"]["PHI"] = dict(ratings["teams"]["DAL"], team="PHI")
        result = analysis("DAL", "PHI", ratings=ratings, qb_context=m._neutral_qb_context())
        self.assertAlmostEqual(result["scoring"]["model_spread_home"], m.HOME_FIELD_POINTS, places=2)
        self.assertAlmostEqual(result["probabilities"]["home_win"], win_probability_from_spread(m.HOME_FIELD_POINTS, stdev=m.WIN_PROB_STDEV), places=4)

    def test_probabilities_sum_to_one_and_covers_are_complements(self):
        for home, away in (("BUF", "NYJ"), ("NYJ", "BUF"), ("KC", "DEN"), ("DAL", "PHI")):
            probs = analysis(home, away, market_spread_home=3.5)["probabilities"]
            self.assertAlmostEqual(probs["home_win"] + probs["away_win"], 1.0, places=9)
            self.assertAlmostEqual(probs["home_cover"] + probs["away_cover"], 1.0, places=9)
            for key in ("home_win", "away_win", "home_cover", "away_cover"):
                self.assertTrue(0.0 < probs[key] < 1.0)

    def test_win_probability_and_fair_prices_come_from_the_engine(self):
        result = analysis("BUF", "NYJ")
        spread = result["scoring"]["model_spread_home"]
        self.assertEqual(result["probabilities"]["home_win"], win_probability_from_spread(spread, stdev=m.WIN_PROB_STDEV))
        self.assertEqual(result["moneyline"]["home_fair_price"], fair_price_from_probability(result["probabilities"]["home_win"]))
        self.assertEqual(result["moneyline"]["away_fair_price"], fair_price_from_probability(result["probabilities"]["away_win"]))
        self.assertLess(result["moneyline"]["home_fair_price"], 0)  # favorite
        self.assertGreater(result["moneyline"]["away_fair_price"], 0)  # underdog
        self.assertEqual(result["teams"]["home"]["win_probability"], result["probabilities"]["home_win"])
        self.assertEqual(result["teams"]["away"]["fair_moneyline"], result["moneyline"]["away_fair_price"])
        self.assertEqual(result["probabilities"]["margin_stdev"], NFL_MARGIN_STDEV)
        self.assertEqual(result["probabilities"]["win_prob_stdev"], m.WIN_PROB_STDEV)


class SpreadTierTests(unittest.TestCase):
    def test_precedence_user_then_market_then_model(self):
        market = analysis(market_spread_home=3.5)
        self.assertEqual((market["probabilities"]["spread_source"], market["probabilities"]["spread_used_home"]), ("market", 3.5))

        game = {"game_id": "x", "home_team": "BUF", "away_team": "NYJ", "market_spread_home": 3.5}
        both = m.analyze_game(game, make_ratings(), make_qb_context(), user_spread_home=-1.5)
        self.assertEqual((both["probabilities"]["spread_source"], both["probabilities"]["spread_used_home"]), ("user", -1.5))

        model_only = analysis()
        self.assertEqual(model_only["probabilities"]["spread_source"], "model")
        used = model_only["probabilities"]["spread_used_home"]
        self.assertEqual(used * 2, round(used * 2))  # a half-point line
        self.assertLessEqual(abs(used - model_only["scoring"]["model_spread_home"]), 0.25 + 1e-9)

    def test_zero_user_spread_beats_market(self):
        game = {"game_id": "x", "home_team": "BUF", "away_team": "NYJ", "market_spread_home": 3.5}
        result = m.analyze_game(game, make_ratings(), make_qb_context(), user_spread_home=0.0)
        self.assertEqual(result["probabilities"]["spread_source"], "user")

    def test_cover_probability_formula_and_monotonicity(self):
        result = analysis()
        model = result["scoring"]["model_spread_home"]
        expected = 1.0 - 0.5 * (1.0 + math.erf((5.5 - model) / NFL_MARGIN_STDEV / math.sqrt(2.0)))
        game = {"game_id": "x", "home_team": "BUF", "away_team": "NYJ"}
        at_5_5 = m.analyze_game(game, make_ratings(), make_qb_context(), user_spread_home=5.5)["probabilities"]["home_cover"]
        self.assertAlmostEqual(at_5_5, expected, places=3)
        covers = [m.analyze_game(game, make_ratings(), make_qb_context(), user_spread_home=s)["probabilities"]["home_cover"] for s in (-3.5, 0.5, 3.5, 7.5, 12.5)]
        self.assertEqual(covers, sorted(covers, reverse=True))  # a bigger number to cover is harder

    def test_covering_the_exact_model_spread_is_a_coin_flip(self):
        game = {"game_id": "x", "home_team": "BUF", "away_team": "NYJ"}
        base = m.analyze_game(game, make_ratings(), make_qb_context())
        spread = base["scoring"]["model_spread_home"]
        result = m.analyze_game(game, make_ratings(), make_qb_context(), user_spread_home=spread)
        self.assertAlmostEqual(result["probabilities"]["home_cover"], 0.5, places=3)

    def test_market_underdog_cover_for_a_road_favorite(self):
        # NYJ (weak) at home vs BUF; market says BUF -7.5 -> spread_home = -7.5; the model has BUF by less than... compare tiers
        result = analysis("NYJ", "BUF", market_spread_home=-7.5)
        self.assertEqual(result["probabilities"]["spread_used_home"], -7.5)
        self.assertAlmostEqual(result["probabilities"]["home_cover"] + result["probabilities"]["away_cover"], 1.0, places=9)


class PaceTests(unittest.TestCase):
    def test_label_thresholds(self):
        self.assertEqual(m._pace_label(m.PACE_LABEL_PLAYS), "Fast")
        self.assertEqual(m._pace_label(m.PACE_LABEL_PLAYS - 0.01), "Average")
        self.assertEqual(m._pace_label(0.0), "Average")
        self.assertEqual(m._pace_label(-m.PACE_LABEL_PLAYS + 0.01), "Average")
        self.assertEqual(m._pace_label(-m.PACE_LABEL_PLAYS), "Slow")
        self.assertEqual(m._pace_label(12.0), "Fast")
        self.assertEqual(m._pace_label(-12.0), "Slow")

    def test_two_fast_teams_project_a_fast_game_and_more_scoring_than_two_slow_teams(self):
        fast = analysis("MIA", "NE")["pace"]
        slow = analysis("TEN", "SEA")["pace"]
        self.assertEqual(fast["label"], "Fast")
        self.assertEqual(slow["label"], "Slow")
        self.assertGreater(fast["plays_projection"], slow["plays_projection"])
        self.assertGreater(fast["vs_league"], 0)
        self.assertLess(slow["vs_league"], 0)
        fast_total = analysis("MIA", "NE")["scoring"]["model_total"]
        slow_total = analysis("TEN", "SEA")["scoring"]["model_total"]
        self.assertGreater(fast_total, slow_total)

    def test_pace_fields(self):
        pace = analysis("KC", "DEN")["pace"]
        for key in ("plays_projection", "seconds_per_play", "label", "vs_league", "league_plays", "league_seconds_per_play", "home_spp", "away_spp"):
            self.assertIn(key, pace)
        self.assertAlmostEqual(pace["seconds_per_play"], (pace["home_spp"] + pace["away_spp"]) / 2, places=1)
        self.assertAlmostEqual(pace["vs_league"], pace["plays_projection"] - pace["league_plays"], places=1)
        self.assertIn(pace["label"], ("Fast", "Average", "Slow"))


class ScoringTests(unittest.TestCase):
    def test_total_and_expected_points_are_consistent(self):
        scoring = analysis()["scoring"]
        self.assertAlmostEqual(scoring["home_expected_points"] + scoring["away_expected_points"], scoring["model_total"], places=2)
        self.assertAlmostEqual(scoring["home_expected_points"] - scoring["away_expected_points"], scoring["model_spread_home"], places=2)
        self.assertGreater(scoring["home_expected_points"], 3.0)
        self.assertIsNone(scoring["market_total"])
        self.assertIsNone(scoring["total_vs_market"])

    def test_market_total_comparison(self):
        scoring = analysis(market_total=44.5)["scoring"]
        self.assertEqual(scoring["market_total"], 44.5)
        self.assertAlmostEqual(scoring["total_vs_market"], scoring["model_total"] - 44.5, places=2)

    def test_base_projection_comes_from_the_engine_and_adjustments_reconcile(self):
        result = analysis("BUF", "NYJ", qb_context=m._neutral_qb_context())
        adj = result["scoring"]["adjustments"]
        reconstructed = adj["base_spread"] + adj["efficiency_spread"] + adj["home_field"] + adj["qb_spread"] + adj["rest_spread"]
        self.assertAlmostEqual(reconstructed, result["scoring"]["model_spread_home"], delta=0.05)
        total = adj["base_total"] + adj["efficiency_total"] + adj["qb_total"] + adj["weather_total"]
        self.assertAlmostEqual(total, result["scoring"]["model_total"], delta=0.05)
        self.assertEqual(adj["home_field"], m.HOME_FIELD_POINTS)

    def test_wind_trims_outdoor_totals_only(self):
        calm = analysis("BUF", "NYJ", wind=5.0, roof="outdoors")["scoring"]["model_total"]
        windy = analysis("BUF", "NYJ", wind=25.0, roof="outdoors")["scoring"]
        dome = analysis("BUF", "NYJ", wind=25.0, roof="dome")["scoring"]["model_total"]
        self.assertAlmostEqual(calm - windy["model_total"], min(m.WIND_TOTAL_CAP, m.WIND_TOTAL_PER_MPH * (25.0 - m.WIND_THRESHOLD_MPH)), delta=0.05)
        self.assertEqual(dome, calm)
        self.assertLess(windy["adjustments"]["weather_total"], 0)
        self.assertAlmostEqual(analysis("BUF", "NYJ", wind=80.0, roof="outdoors")["scoring"]["adjustments"]["weather_total"], -m.WIND_TOTAL_CAP)

    def test_rest_edge_moves_the_spread_toward_the_rested_team(self):
        even = analysis("BUF", "NYJ", home_rest=7, away_rest=7)["scoring"]["model_spread_home"]
        home_rested = analysis("BUF", "NYJ", home_rest=11, away_rest=7)["scoring"]["model_spread_home"]
        away_rested = analysis("BUF", "NYJ", home_rest=7, away_rest=4)["scoring"]["model_spread_home"]
        self.assertAlmostEqual(home_rested - even, m.REST_POINTS_PER_DAY * 4, delta=0.05)
        self.assertGreater(away_rested - even, 0)  # visitor on a short week helps the home side
        capped = analysis("BUF", "NYJ", home_rest=30, away_rest=7)["scoring"]["adjustments"]["rest_spread"]
        self.assertAlmostEqual(capped, m.REST_POINTS_PER_DAY * m.REST_DAYS_CAP)


class QuarterbackEffectTests(unittest.TestCase):
    def test_backup_starter_costs_points_and_flags_a_warning(self):
        ctx = make_qb_context()
        base = analysis("NYJ", "BUF", qb_context=ctx, away_qb="Josh Allen")
        backup = analysis("NYJ", "BUF", qb_context=ctx, away_qb="Case Keenum")  # BUF starts the (NYJ-era) backup
        self.assertGreater(base["scoring"]["model_spread_home"] * -1, backup["scoring"]["model_spread_home"] * -1)
        self.assertTrue(any("Case Keenum" in w and "backup" in w for w in backup["warnings"]))
        self.assertLess(backup["teams"]["away"]["qb_impact_points"], base["teams"]["away"]["qb_impact_points"])
        self.assertEqual(backup["teams"]["away"]["qb_role"], "backup")

    def test_unknown_starter_is_priced_as_unproven_and_flagged(self):
        result = analysis("BUF", "NYJ", away_qb="Brand New Rookie")
        away = result["teams"]["away"]
        self.assertEqual((away["qb_role"], away["qb_is_estimate"]), ("unknown", True))
        self.assertLess(away["qb_impact_points"], 0)
        self.assertGreater(away["qb_impact_points"], analysis("BUF", "NYJ", away_qb="Case Keenum")["teams"]["away"]["qb_impact_points"])
        self.assertTrue(math.isfinite(away["qb_rating"]))
        self.assertTrue(any("Brand New Rookie" in w and "unproven" in w for w in result["warnings"]))

    def test_tiny_sample_rating_is_replaced_by_a_flagged_estimate(self):
        ctx = m.build_qb_context(
            None, pd.concat([make_qb_context_frame(), qb_frame([("00-TINY", "Three Attempts", "NYJ", 5, 3, 0, 0, 0, 1, 0, -2.0)])], ignore_index=True),
            season=2026, prior_season=2025)
        result = analysis("BUF", "NYJ", qb_context=ctx, away_qb="Three Attempts")
        away = result["teams"]["away"]
        self.assertTrue(away["qb_is_estimate"])
        self.assertEqual(away["qb_rating"], round(ctx["league"]["replacement_passer_rating"], 1))
        self.assertEqual(away["qb_epa_per_dropback"], ctx["league"]["replacement_epa"])
        self.assertLess(away["qb_impact_points"], 0)
        full = analysis("BUF", "NYJ", qb_context=ctx)["teams"]["home"]  # Josh Allen has a real, unflagged rating
        self.assertFalse(full["qb_is_estimate"])
        self.assertNotEqual(full["qb_rating"], round(ctx["league"]["replacement_passer_rating"], 1))

    def test_team_primary_is_used_when_no_starter_is_named(self):
        result = analysis("BUF", "NYJ")
        self.assertEqual(result["teams"]["home"]["qb_name"], "Josh Allen")
        self.assertEqual(result["teams"]["away"]["qb_name"], "Geno Smith")

    def test_adjustment_is_impact_minus_embedded_and_bounded(self):
        for side in ("home", "away"):
            team = analysis()["teams"][side]
            self.assertAlmostEqual(team["qb_adjustment_points"], team["qb_impact_points"] - team["qb_embedded_points"], delta=0.02)
            self.assertLessEqual(abs(team["qb_adjustment_points"]), m.QB_IMPACT_CLAMP)
            self.assertLessEqual(abs(team["qb_impact_points"]), m.QB_IMPACT_CLAMP)

    def test_last_game_fallback_warning(self):
        result = analysis("BUF", "NYJ", home_qb="Josh Allen", home_qb_source="last_game")
        self.assertTrue(any("most recent game" in w and "Josh Allen" in w for w in result["warnings"]))

    def test_missing_qb_context_applies_no_adjustment(self):
        result = analysis("BUF", "NYJ", qb_context={}, home_qb="Josh Allen", away_qb="Geno Smith")
        self.assertEqual(result["scoring"]["adjustments"]["qb_spread"], 0.0)
        self.assertEqual(result["scoring"]["adjustments"]["qb_total"], 0.0)
        self.assertTrue(any("Quarterback data unavailable" in w for w in result["warnings"]))
        for side in ("home", "away"):
            team = result["teams"][side]
            self.assertEqual((team["qb_impact_points"], team["qb_embedded_points"], team["qb_adjustment_points"]), (0.0, 0.0, 0.0))
        self.assertFalse(any("unproven" in w or "backup" in w for w in result["warnings"]))  # no invented QB claims
        self.assertEqual(result["confidence_factors"]["qb"], 0.88)
        assert_all_finite(self, result)


class MissingDataTests(unittest.TestCase):
    def test_team_missing_from_ratings_uses_neutral_fallback_with_warning(self):
        ratings = make_ratings()
        del ratings["teams"]["NYJ"]
        result = analysis("BUF", "NYJ", ratings=ratings)
        away = result["teams"]["away"]
        self.assertEqual(away["strength_rating"], 50.0)
        self.assertEqual(away["data_quality"], "none")
        self.assertTrue(any("NYJ" in w and "league-average" in w for w in result["warnings"]))
        assert_all_finite(self, result)

    def test_team_without_any_data_in_ratings_also_warns(self):
        result = analysis("BUF", "ATL")  # ATL has no data in the fixture ratings
        self.assertTrue(any("ATL" in w for w in result["warnings"]))

    def test_empty_ratings_and_context_still_produce_a_finite_analysis(self):
        for ratings in ({}, None):
            result = m.analyze_game({"home_team": "KC", "away_team": "DEN"}, ratings, None)
            assert_all_finite(self, result)
            self.assertAlmostEqual(result["scoring"]["model_spread_home"], m.HOME_FIELD_POINTS, places=2)
            self.assertGreaterEqual(len(result["reasoning"]), 5)

    def test_invalid_matchups_raise_value_error(self):
        with self.assertRaises(ValueError):
            m.analyze_game({"home_team": "KC", "away_team": "KC"}, make_ratings(), None)
        with self.assertRaises(ValueError):
            m.analyze_game({"home_team": "KC"}, make_ratings(), None)

    def test_team_aliases_are_normalised(self):
        result = m.analyze_game({"home_team": "LAR", "away_team": "WSH"}, make_ratings(), None)
        self.assertEqual((result["home_team"], result["away_team"]), ("LA", "WAS"))

    def test_nan_values_in_the_game_dict_are_treated_as_missing(self):
        nan = float("nan")
        game = {"home_team": "BUF", "away_team": "NYJ", "market_spread_home": nan, "market_total": nan, "market_home_ml": nan,
                "market_away_ml": nan, "wind": nan, "temp": nan, "home_rest": nan, "away_rest": nan, "home_qb": nan, "away_qb": nan,
                "home_qb_id": nan, "away_qb_id": nan, "roof": nan}
        result = m.analyze_game(game, make_ratings(), make_qb_context())
        assert_all_finite(self, result)
        self.assertEqual(result["probabilities"]["spread_source"], "model")
        self.assertIsNone(result["scoring"]["market_total"])
        self.assertIsNone(result["moneyline"]["market_home_ml"])
        self.assertEqual(result["teams"]["home"]["qb_name"], "Josh Allen")  # fell back to the team's primary QB, not a QB named "nan"
        self.assertEqual(result["scoring"]["adjustments"]["rest_spread"], 0.0)
        self.assertEqual(result["scoring"]["adjustments"]["weather_total"], 0.0)

    def test_user_spread_nan_is_ignored(self):
        result = m.analyze_game({"home_team": "BUF", "away_team": "NYJ", "market_spread_home": 6.5}, make_ratings(), None, user_spread_home=float("nan"))
        self.assertEqual((result["probabilities"]["spread_source"], result["probabilities"]["spread_used_home"]), ("market", 6.5))

    def test_played_game_warns_but_still_analyzes(self):
        result = analysis("BUF", "NYJ", played=True, status="Final 20-17")
        self.assertTrue(any("already been played" in w for w in result["warnings"]))
        self.assertEqual(result["status"], "Final 20-17")


class MarketBlockTests(unittest.TestCase):
    def test_no_market_gives_none_fields(self):
        moneyline = analysis()["moneyline"]
        for key in ("market_home_ml", "market_away_ml", "market_home_prob_novig", "market_away_prob_novig", "edge_home_vs_market", "market_hold"):
            self.assertIsNone(moneyline[key], key)
        self.assertIsNone(analysis()["probabilities"]["market_spread_home"])

    def test_no_vig_probabilities_and_edge(self):
        result = analysis(market_home_ml=-250.0, market_away_ml=210.0)
        moneyline = result["moneyline"]
        fair_home, fair_away = remove_vig_two_way(-250.0, 210.0)
        self.assertAlmostEqual(moneyline["market_home_prob_novig"], fair_home, places=4)
        self.assertAlmostEqual(moneyline["market_away_prob_novig"], fair_away, places=4)
        self.assertAlmostEqual(moneyline["market_home_prob_novig"] + moneyline["market_away_prob_novig"], 1.0, places=3)
        self.assertAlmostEqual(moneyline["edge_home_vs_market"], result["probabilities"]["home_win"] - moneyline["market_home_prob_novig"], places=4)
        self.assertAlmostEqual(moneyline["edge_home_vs_market"], -moneyline["edge_away_vs_market"], places=3)
        self.assertGreater(moneyline["market_hold"], 0)

    def test_incomplete_market_moneyline_is_ignored_with_warning(self):
        result = analysis(market_home_ml=-250.0)
        self.assertIsNone(result["moneyline"]["market_home_prob_novig"])
        self.assertTrue(any("incomplete" in w for w in result["warnings"]))

    def test_nonsensical_moneylines_are_ignored(self):
        result = analysis(market_home_ml=0.0, market_away_ml=0.0)
        self.assertIsNone(result["moneyline"]["market_home_ml"])
        assert_all_finite(self, result)


class ConfidenceTests(unittest.TestCase):
    def test_bounded_and_labelled(self):
        for home, away in (("BUF", "NYJ"), ("NYJ", "BUF"), ("DAL", "PHI"), ("KC", "DEN")):
            result = analysis(home, away)
            self.assertTrue(0.0 <= result["confidence"] <= 1.0)
            self.assertIn(result["confidence_label"], ("Low", "Moderate", "High"))
            self.assertEqual(set(result["confidence_factors"]), {"margin", "sample", "qb", "market"})

    def test_bigger_edge_means_more_confidence(self):
        self.assertGreater(analysis("BUF", "NYJ")["confidence"], analysis("DAL", "PHI")["confidence"])
        self.assertEqual(analysis("DAL", "PHI")["confidence_label"], "Low")

    def test_backup_quarterback_lowers_confidence(self):
        ctx = make_qb_context()
        sure = analysis("BUF", "NYJ", qb_context=ctx)["confidence"]
        shaky = analysis("BUF", "NYJ", qb_context=ctx, home_qb="Case Keenum")
        self.assertEqual(shaky["confidence_factors"]["qb"], 0.88)
        self.assertLess(shaky["confidence"], sure)

    def test_market_disagreement_lowers_confidence(self):
        model = analysis("BUF", "NYJ")["scoring"]["model_spread_home"]
        agree = analysis("BUF", "NYJ", market_spread_home=round(model * 2) / 2)["confidence"]
        far = analysis("BUF", "NYJ", market_spread_home=round(model) - 9.0)["confidence"]
        opposite = analysis("BUF", "NYJ", market_spread_home=-6.5)["confidence"]
        self.assertGreater(agree, far)
        self.assertGreater(far, opposite)

    def test_confidence_components_directly(self):
        value, label, factors = m._confidence(0.80, 8.0, 0, 0, 0.5, False)
        self.assertEqual(label, "High")
        self.assertAlmostEqual(factors["margin"], 0.6, places=3)
        self.assertEqual(factors["sample"], 1.0)
        low, low_label, _ = m._confidence(0.52, 0.0, 2, 2, None, False)
        self.assertEqual(low_label, "Low")
        self.assertLess(low, value)


class ReasoningTests(unittest.TestCase):
    def _check(self, result):
        reasoning = result["reasoning"]
        self.assertGreaterEqual(len(reasoning), 5)
        for item in reasoning:
            self.assertEqual(set(item), {"title", "detail", "tone", "team"})
            self.assertIn(item["tone"], ("pro", "con", "neutral"))
            self.assertTrue(item["title"].strip())
            self.assertGreaterEqual(len(re.findall(r"\d+(?:\.\d+)?", item["detail"])), 2, f"too few numbers in: {item['title']}")
            self.assertIn(item["team"], (None, result["home_team"], result["away_team"]))
            self.assertNotRegex(item["detail"], r"nan|inf\b|None")
        return reasoning

    def test_without_market_has_five_or_more_items_with_numbers(self):
        reasoning = self._check(analysis("BUF", "NYJ"))
        titles = " | ".join(r["title"] for r in reasoning)
        self.assertIn("Team strength", titles)
        self.assertIn("offense vs", titles)
        self.assertIn("Quarterback", titles)
        self.assertIn("pace", titles.lower())
        self.assertIn("Home field", titles)
        self.assertIn("No market line", titles)
        self.assertIn("Sample size", titles)

    def test_with_market_cites_the_gap_and_probabilities(self):
        result = analysis("BUF", "NYJ", market_spread_home=3.5, market_home_ml=-190.0, market_away_ml=160.0, market_total=42.5)
        reasoning = self._check(result)
        market = next(r for r in reasoning if r["title"] == "Market check")
        self.assertIn("BUF -3.5", market["detail"])
        self.assertIn("-point gap", market["detail"])
        self.assertIn("%", market["detail"])
        pace = next(r for r in reasoning if "pace" in r["title"])
        self.assertIn("42.5", pace["detail"])

    def test_disagreeing_market_is_a_con(self):
        result = analysis("BUF", "NYJ", market_spread_home=-6.5)
        market = next(r for r in result["reasoning"] if r["title"] == "Market check")
        self.assertEqual(market["tone"], "con")
        self.assertIn("market picks NYJ", market["detail"])

    def test_agreeing_market_is_a_pro(self):
        model = analysis("BUF", "NYJ")["scoring"]["model_spread_home"]
        result = analysis("BUF", "NYJ", market_spread_home=round(model * 2) / 2)
        market = next(r for r in result["reasoning"] if r["title"] == "Market check")
        self.assertEqual(market["tone"], "pro")
        self.assertEqual(market["team"], "BUF")

    def test_situation_item_cites_rest_division_and_weather(self):
        result = analysis("BUF", "NYJ", home_rest=7, away_rest=4, div_game=True, wind=22.0, roof="outdoors", temp=28.0)
        situation = next(r for r in result["reasoning"] if r["title"] == "Home field and situation")
        for fragment in ("2.0 points", "rest", "divisional", "22 mph", "28F"):
            self.assertIn(fragment, situation["detail"])
        domed = analysis("BUF", "NYJ", wind=22.0, roof="dome")
        text = next(r for r in domed["reasoning"] if r["title"] == "Home field and situation")["detail"]
        self.assertIn("roofed", text)

    def test_backup_qb_adds_a_risk_item(self):
        result = analysis("BUF", "NYJ", home_qb="Case Keenum")
        risk = [r for r in result["reasoning"] if r["title"] == "Quarterback risk"]
        self.assertEqual(len(risk), 1)
        self.assertEqual(risk[0]["tone"], "con")
        self.assertIn("Case Keenum", risk[0]["detail"])
        self._check(result)

    def test_strength_item_cites_both_ratings(self):
        result = analysis("BUF", "NYJ")
        detail = result["reasoning"][0]["detail"]
        for side in ("home", "away"):
            self.assertIn(f"{result['teams'][side]['strength_rating']:.0f}", detail)
        offense = next(r for r in result["reasoning"] if r["title"] == "BUF offense vs NYJ defense")["detail"]
        self.assertIn(f"{result['teams']['home']['off_epa']:+.3f}", offense)
        self.assertIn(f"{result['teams']['away']['def_epa']:+.3f}", offense)

    def test_tone_is_relative_to_the_favorite(self):
        result = analysis("BUF", "NYJ")
        strength = result["reasoning"][0]
        self.assertEqual(strength["team"], "BUF")
        self.assertEqual(strength["tone"], "pro")  # the stronger team is also the model favorite
        flipped = analysis("NYJ", "BUF", home_qb="Case Keenum")  # BUF still favored on the road
        self.assertEqual(flipped["favored_team"], "BUF")
        self.assertEqual(flipped["reasoning"][0]["tone"], "pro")

    def test_sample_size_item_is_con_early_in_the_season(self):
        early = make_ratings(prior=None)
        item = next(r for r in analysis(ratings=early)["reasoning"] if r["title"] == "Sample size and data quality")
        self.assertIn("regressed", item["detail"])
        self.assertRegex(item["detail"], r"\d+%")

    def test_even_matchup_title(self):
        ratings = make_ratings({"DAL": team_row(), "PHI": team_row()})
        ratings["teams"]["PHI"] = dict(ratings["teams"]["DAL"], team="PHI")
        result = analysis("DAL", "PHI", ratings=ratings, qb_context=m._neutral_qb_context())
        self.assertEqual(result["reasoning"][0]["title"], "Team strength is even")
        self.assertEqual(result["reasoning"][0]["tone"], "neutral")


class AnalysisShapeTests(unittest.TestCase):
    def test_required_keys_and_team_fields(self):
        result = analysis(market_spread_home=5.5, market_total=44.5, market_home_ml=-220.0, market_away_ml=185.0)
        for key in ("game_id", "week", "away_team", "home_team", "kickoff_label", "teams", "pace", "scoring", "probabilities", "moneyline",
                    "favored", "favored_team", "margin", "confidence", "confidence_label", "reasoning", "warnings"):
            self.assertIn(key, result)
        for side in ("home", "away"):
            team = result["teams"][side]
            for key in ("team", "strength_rating", "offensive_efficiency", "defensive_efficiency", "net_efficiency", "qb_name", "qb_rating",
                        "qb_impact_points", "pace_spp", "expected_points", "win_probability", "spread_cover_probability", "fair_moneyline"):
                self.assertIn(key, team)
        for key in ("home_expected_points", "away_expected_points", "model_spread_home", "model_total", "market_total"):
            self.assertIn(key, result["scoring"])
        for key in ("home_win", "away_win", "home_cover", "away_cover", "spread_used_home", "spread_source"):
            self.assertIn(key, result["probabilities"])
        for key in ("home_fair_price", "away_fair_price", "market_home_ml", "market_away_ml", "market_home_prob_novig",
                    "market_away_prob_novig", "edge_home_vs_market"):
            self.assertIn(key, result["moneyline"])
        for key in ("plays_projection", "seconds_per_play", "label", "vs_league"):
            self.assertIn(key, result["pace"])
        assert_all_finite(self, result)

    def test_analysis_is_deterministic_and_does_not_mutate_inputs(self):
        ratings, ctx = make_ratings(), make_qb_context()
        before = repr(ratings), repr(ctx)
        game = {"home_team": "BUF", "away_team": "NYJ", "market_spread_home": 7.0}
        first = m.analyze_game(game, ratings, ctx)
        second = m.analyze_game(game, ratings, ctx)
        self.assertEqual(first, second)
        self.assertEqual((repr(ratings), repr(ctx)), before)
        self.assertEqual(game, {"home_team": "BUF", "away_team": "NYJ", "market_spread_home": 7.0})

    def test_every_team_pair_in_a_league_is_finite(self):
        ratings, ctx = make_ratings(), make_qb_context()
        codes = list(ratings["teams"])[:12]
        for home in codes:
            for away in codes:
                if home != away:
                    assert_all_finite(self, m.analyze_game({"home_team": home, "away_team": away}, ratings, ctx))


class EvaluateMoneylinePriceTests(unittest.TestCase):
    def test_matches_odds_input_for_both_sides(self):
        result = analysis("BUF", "NYJ", market_home_ml=-250.0, market_away_ml=210.0)
        for side, price in (("home", -250.0), ("away", 210.0)):
            evaluated = m.evaluate_moneyline_price(result, side, price)
            probability = result["probabilities"]["home_win" if side == "home" else "away_win"]
            expected = evaluate_price(probability, price)
            for key, value in expected.items():
                self.assertEqual(evaluated[key], value, key)
            self.assertEqual(evaluated["side"], side)
            self.assertEqual(evaluated["team"], "BUF" if side == "home" else "NYJ")

    def test_includes_no_vig_market_probability_when_available(self):
        result = analysis("BUF", "NYJ", market_home_ml=-250.0, market_away_ml=210.0)
        evaluated = m.evaluate_moneyline_price(result, "home", -250.0)
        self.assertAlmostEqual(evaluated["market_probability_novig"], remove_vig_two_way(-250.0, 210.0)[0], places=4)
        self.assertAlmostEqual(evaluated["edge_vs_market_novig"], result["probabilities"]["home_win"] - evaluated["market_probability_novig"], places=4)
        bare = m.evaluate_moneyline_price(analysis("BUF", "NYJ"), "home", -250.0)
        self.assertIsNone(bare["market_probability_novig"])
        self.assertIsNone(bare["edge_vs_market_novig"])

    def test_value_direction_and_stake(self):
        result = analysis("BUF", "NYJ")
        cheap = m.evaluate_moneyline_price(result, "home", -110.0, stake=50.0)
        # a price 300 cents shorter than the model's own fair price can never be value
        dear = m.evaluate_moneyline_price(result, "home", result["moneyline"]["home_fair_price"] - 300.0)
        self.assertGreater(cheap["expected_value"], 0)
        self.assertLess(dear["expected_value"], 0)
        self.assertEqual(cheap["stake"], 50.0)
        self.assertTrue(cheap["is_value"])
        self.assertFalse(dear["is_value"])

    def test_rejects_bad_side_and_price(self):
        result = analysis("BUF", "NYJ")
        with self.assertRaises(ValueError):
            m.evaluate_moneyline_price(result, "draw", -110.0)
        with self.assertRaises(ValueError):
            m.evaluate_moneyline_price(result, "home", 0.0)


class PublicSurfaceTests(unittest.TestCase):
    def test_constants_and_exports(self):
        self.assertEqual(m.MARGIN_STDEV, NFL_MARGIN_STDEV)
        self.assertEqual(m.HOME_FIELD_POINTS, 2.0)
        for name in ("MARGIN_STDEV", "HOME_FIELD_POINTS", "ScheduleUnavailable", "current_season", "load_schedule", "weeks_in_schedule",
                     "default_week", "games_for_week", "load_team_ratings", "load_qb_context", "qb_impact_points", "analyze_game",
                     "evaluate_moneyline_price"):
            self.assertTrue(hasattr(m, name), name)
            self.assertIn(name, m.__all__)

    def test_no_streamlit_import(self):
        source = Path(m.__file__).read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\s*(import|from)\s+streamlit")


# --------------------------------------------------------------------------
# Independent-audit regression tests
#
# Every class below pins a defect found by auditing the module against real
# nflverse data and hostile inputs (see each docstring for the failure).
# --------------------------------------------------------------------------

class NeutralSiteTests(_CacheIsolated):
    """nflverse marks London / Madrid / Melbourne / Mexico City games ``location == "Neutral"``.

    The market spread at those venues carries no home-field edge.  The module used to add 2.0 points to the
    designated home team anyway: in the 2025 backtest that left the model 1.85 points too friendly to the
    nominal home side on exactly those games (mean market-minus-model gap -1.85, mean abs. gap 3.39 vs 1.97 fixed).
    """

    def _frame(self):
        raw = raw_schedule([
            raw_game(4, "IND", "WAS", gameday="2026-10-04", gametime="09:30"),
            raw_game(4, "NE", "BUF", gameday="2026-10-04", gametime="13:00"),
            raw_game(4, "DEN", "SF", gameday="2026-10-04", gametime="16:25"),
        ])
        raw["location"] = ["Neutral", "Home", None]
        return m.normalize_schedule(raw)

    def test_schedule_flags_neutral_venues(self):
        frame = self._frame().set_index("home_team")
        self.assertEqual([bool(frame.loc[t, "neutral_site"]) for t in ("WAS", "BUF", "SF")], [True, False, False])
        self.assertIn("neutral_site", m.SCHEDULE_COLUMNS)
        self.assertTrue(frame["neutral_site"].dtype == bool)

    def test_location_is_matched_case_insensitively_and_missing_column_means_not_neutral(self):
        raw = raw_schedule([raw_game(1, "NE", "SEA")])
        raw["location"] = ["  NEUTRAL "]
        self.assertTrue(bool(m.normalize_schedule(raw)["neutral_site"].iloc[0]))
        self.assertFalse(bool(m.normalize_schedule(raw_schedule([raw_game(1, "NE", "SEA")]))["neutral_site"].iloc[0]))

    def test_games_for_week_exposes_the_flag_as_a_real_bool(self):
        games = {g["home_team"]: g for g in m.games_for_week(self._frame(), 4)}
        self.assertIs(games["WAS"]["neutral_site"], True)
        self.assertIs(games["BUF"]["neutral_site"], False)
        self.assertIs(games["SF"]["neutral_site"], False)

    def test_no_home_field_points_at_a_neutral_site(self):
        home = analysis("BUF", "NYJ")
        neutral = analysis("BUF", "NYJ", neutral_site=True)
        self.assertEqual(neutral["scoring"]["adjustments"]["home_field"], 0.0)
        self.assertEqual(home["scoring"]["adjustments"]["home_field"], m.HOME_FIELD_POINTS)
        self.assertAlmostEqual(home["scoring"]["model_spread_home"] - neutral["scoring"]["model_spread_home"], m.HOME_FIELD_POINTS, delta=0.011)
        self.assertAlmostEqual(home["scoring"]["model_total"], neutral["scoring"]["model_total"], delta=0.011)  # venue does not change the total
        self.assertIs(neutral["context"]["neutral_site"], True)
        self.assertIs(home["context"]["neutral_site"], False)

    def test_who_is_designated_home_does_not_matter_at_a_neutral_site(self):
        a = analysis("KC", "DEN", neutral_site=True)
        b = analysis("DEN", "KC", neutral_site=True)
        self.assertAlmostEqual(a["scoring"]["model_spread_home"], -b["scoring"]["model_spread_home"], delta=0.02)
        self.assertEqual(a["favored_team"], b["favored_team"])

    def test_equal_teams_at_a_neutral_site_are_a_pick_em(self):
        ratings = make_ratings({"DAL": team_row(), "PHI": team_row()})
        ratings["teams"]["PHI"] = dict(ratings["teams"]["DAL"], team="PHI")
        result = analysis("DAL", "PHI", ratings=ratings, qb_context=m._neutral_qb_context(), neutral_site=True)
        self.assertAlmostEqual(result["scoring"]["model_spread_home"], 0.0, places=2)
        self.assertEqual(result["probabilities"]["home_win"], 0.5)
        self.assertEqual(result["probabilities"]["away_win"], 0.5)

    def test_reasoning_says_neutral_and_does_not_claim_home_field(self):
        result = analysis("BUF", "NYJ", neutral_site=True, home_rest=7, away_rest=7)
        item = next(r for r in result["reasoning"] if "situation" in r["title"])
        self.assertEqual(item["title"], "Neutral site and situation")
        self.assertIn("neutral-site", item["detail"].lower())
        self.assertIn("2.0 points", item["detail"])  # names what is being withheld
        self.assertNotIn("Home field is worth", item["detail"])
        self.assertEqual((item["tone"], item["team"]), ("neutral", None))  # nothing favours either side

    def test_rest_edge_can_favour_the_visitor_at_a_neutral_site(self):
        result = analysis("BUF", "NYJ", neutral_site=True, home_rest=4, away_rest=10)  # BUF is the favourite
        item = next(r for r in result["reasoning"] if "situation" in r["title"])
        self.assertEqual(item["team"], "NYJ")
        self.assertEqual(item["tone"], "con")
        self.assertLess(result["scoring"]["adjustments"]["rest_spread"], 0)

    def test_missing_or_text_flags_mean_not_neutral(self):
        for value in (None, float("nan"), "False", "no", 0):
            self.assertIs(analysis(neutral_site=value)["context"]["neutral_site"], False, repr(value))
        self.assertIs(analysis(neutral_site="true")["context"]["neutral_site"], True)

    def test_regular_home_games_keep_home_field_in_the_reasoning(self):
        result = analysis("BUF", "NYJ")
        item = next(r for r in result["reasoning"] if "situation" in r["title"])
        self.assertEqual(item["title"], "Home field and situation")
        self.assertEqual((item["tone"], item["team"]), ("pro", "BUF"))


class FlagParsingTests(unittest.TestCase):
    """``bool(float("nan"))`` is True: a missing ``div_game`` / ``played`` read from a DataFrame row used to become a claim."""

    def test_truthy_reads_flags_strictly(self):
        for value in (True, 1, 1.0, "true", "Yes", "1", np.bool_(True)):
            self.assertTrue(m._truthy(value), repr(value))
        for value in (None, False, 0, 0.0, "", "false", "no", "0", "nan", float("nan"), np.nan, pd.NA, np.bool_(False)):
            self.assertFalse(m._truthy(value), repr(value))

    def test_nan_div_game_is_not_divisional(self):
        result = analysis(div_game=float("nan"))
        self.assertIs(result["context"]["div_game"], False)
        situation = next(r for r in result["reasoning"] if "situation" in r["title"])["detail"]
        self.assertNotIn("divisional", situation)
        self.assertIs(analysis(div_game="False")["context"]["div_game"], False)
        self.assertIs(analysis(div_game=1)["context"]["div_game"], True)

    def test_nan_played_is_not_played(self):
        result = analysis(played=float("nan"))
        self.assertIs(result["played"], False)
        self.assertFalse(any("already been played" in w for w in result["warnings"]))

    def test_divisional_note_does_not_claim_an_unmodelled_effect(self):
        detail = next(r for r in analysis(div_game=True)["reasoning"] if "situation" in r["title"])["detail"]
        self.assertIn("divisional", detail)
        self.assertIn("not modelled", detail)

    def test_nan_roof_is_missing_not_the_string_nan(self):
        result = analysis(roof=float("nan"), wind=20.0)
        self.assertIsNone(result["context"]["roof"])
        self.assertLess(result["scoring"]["adjustments"]["weather_total"], 0)  # unknown roof is treated as outdoors

    def test_nan_or_missing_team_codes_never_become_a_phantom_team(self):
        for value in (None, float("nan"), pd.NA, "", "  ", "nan"):
            self.assertEqual(m.normalize_team(value), "", repr(value))
        raw = raw_schedule([raw_game(1, "NE", "SEA"), raw_game(1, "KC", "BUF")])
        raw.loc[1, "home_team"] = float("nan")
        frame = m.normalize_schedule(raw)
        self.assertEqual(list(frame["home_team"]), ["SEA"])
        with self.assertRaises(ValueError):
            m.analyze_game({"home_team": float("nan"), "away_team": "KC"}, make_ratings(), None)


class MalformedDataTests(_CacheIsolated):
    """The module promises that a data failure never raises into the UI.

    Download errors were covered, but a download that *succeeds* with a frame the builders cannot digest (an
    upstream schema change) used to escape as a raw TypeError / AttributeError from load_team_ratings and
    load_qb_context, and from load_schedule as something other than ScheduleUnavailable.
    """

    def test_ratings_survive_an_undigestible_pbp_frame_and_keep_the_schedule_scoring(self):
        m._download_schedule = lambda season: synthetic_schedule(season)
        for label, bad in (("text in a numeric column", synthetic_pbp(SYN_GAMES).assign(epa="oops")), ("a string", "garbage"), ("a list", [1, 2, 3])):
            m.clear_caches()
            m._download_pbp = lambda season, bad=bad: bad
            ratings = m.load_team_ratings(2026)  # must not raise
            assert_all_finite(self, ratings)
            self.assertEqual(len(ratings["teams"]), 32, label)
            self.assertTrue(any("could not be processed" in w and "2026" in w for w in ratings["meta"]["warnings"]), label)
            self.assertGreater(ratings["teams"]["KC"]["games_current"], 0, label)  # scoring from the schedule still counts
            self.assertTrue(math.isfinite(ratings["teams"]["KC"]["points_for_pg"]))

    def test_unprocessable_pbp_is_retried_soon_not_cached_for_the_full_ttl(self):
        import time
        m._download_schedule = lambda season: synthetic_schedule(season)
        m._download_pbp = lambda season: "garbage"
        m.load_team_ratings(2026)
        self.assertLessEqual(m._MEMO[("ratings", 2026, 2025)][0] - time.monotonic(), m._MEMO_DEGRADED_SECONDS + 1)

    def test_everything_unusable_degrades_to_league_average_with_a_warning(self):
        m._download_schedule = lambda season: [1, 2]
        m._download_pbp = lambda season: "garbage"
        ratings = m.load_team_ratings(2026)
        assert_all_finite(self, ratings)
        self.assertTrue(all(row["strength_rating"] == 50.0 for row in ratings["teams"].values()))
        self.assertTrue(ratings["meta"]["warnings"])

    def test_blend_failure_degrades_to_league_average_instead_of_raising(self):
        m._download_schedule = lambda season: synthetic_schedule(season)
        m._download_pbp = lambda season: synthetic_pbp(SYN_GAMES, season=season)
        original, calls = m.blend_ratings, {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyError("boom")
            return original(*args, **kwargs)
        m.blend_ratings = flaky
        try:
            ratings = m.load_team_ratings(2026)
        finally:
            m.blend_ratings = original
        assert_all_finite(self, ratings)
        self.assertTrue(any("could not be built" in w and "KeyError" in w for w in ratings["meta"]["warnings"]))
        self.assertTrue(all(row["strength_rating"] == 50.0 for row in ratings["teams"].values()))

    def test_qb_context_survives_unparseable_frames(self):
        for bad in ("garbage", [1, 2, 3], 42):
            m.clear_caches()
            m._download_player_stats = lambda season, bad=bad: bad
            context = m.load_qb_context(2026)  # must not raise
            self.assertEqual(context["players"], {})
            self.assertTrue(any("could not be processed" in w for w in context["meta"]["warnings"]), repr(bad))
            assert_all_finite(self, context)

    def test_qb_builder_failure_degrades_to_a_neutral_context(self):
        m._download_player_stats = lambda season: make_qb_context_frame()
        original = m.build_qb_context

        def boom(*args, **kwargs):
            raise ValueError("boom")
        m.build_qb_context = boom
        try:
            context = m.load_qb_context(2026)
        finally:
            m.build_qb_context = original
        self.assertEqual(context["players"], {})
        self.assertTrue(any("could not be processed (ValueError)" in w for w in context["meta"]["warnings"]))
        assert_all_finite(self, context)

    def test_unreadable_schedule_surfaces_as_the_documented_exception(self):
        for bad in ([1, 2], "nope", 42):
            m.clear_caches()
            m._download_schedule = lambda season, bad=bad: bad
            with self.assertRaises(m.ScheduleUnavailable) as caught:
                m.load_schedule(2026)
            self.assertIn("2026", str(caught.exception))

    def test_analysis_works_without_ratings_meta(self):
        ratings = make_ratings()
        del ratings["meta"]
        result = analysis("BUF", "NYJ", ratings=ratings)
        assert_all_finite(self, result)
        sample = next(r for r in result["reasoning"] if r["title"] == "Sample size and data quality")
        self.assertIn("current-season", sample["detail"])  # no season label available, and no KeyError

    def test_id_only_player_frames_never_collapse_onto_one_empty_name(self):
        """Without a name column the id is the name; ``"00-0012345"`` has no letters, so its name key was ``""`` for everyone."""
        frame = qb_frame([("00-0012345", "x", "BUF", w, 30, 20, 210, 1, 0, 1, 3.0) for w in (1, 2, 3)]
                         + [("00-0067890", "x", "NYJ", w, 30, 20, 210, 1, 0, 1, 3.0) for w in (1, 2, 3)]).drop(columns=["player_display_name"])
        context = m.build_qb_context(frame, None, season=2026, prior_season=2025)
        self.assertEqual(set(context["players"]), {"00-0012345", "00-0067890"})
        self.assertNotIn("", context["names"])
        for odd_name in ("!!!", "李明", "  "):  # symbols only; a non-Latin name; blank
            resolved = m.resolve_qb(context, "BUF", name=odd_name)
            self.assertFalse(resolved["found"], odd_name)
            self.assertEqual(resolved["role"], "unknown")

    def test_resolve_qb_ignores_an_empty_name_key_in_a_foreign_context(self):
        foreign = {"players": {"p1": {"player_id": "p1", "name": "Someone", "role": "starter"}}, "names": {"": "p1"}, "teams": {}}
        resolved = m.resolve_qb(foreign, "BUF", name="!!!")
        self.assertFalse(resolved["found"])
        self.assertEqual(resolved["source"], "schedule_unknown")


class QuarterbackSourceTests(unittest.TestCase):
    """When the schedule names no starter the model silently assumed the team's most-used QB and showed his name as fact."""

    def test_assumed_primary_is_flagged_in_source_and_warning(self):
        result = analysis("BUF", "NYJ")  # no home_qb / away_qb in the game dict
        for side, code, name in (("home", "BUF", "Josh Allen"), ("away", "NYJ", "Geno Smith")):
            self.assertEqual(result["teams"][side]["qb_source"], "team_primary")
            self.assertTrue(any(w.startswith(f"{code}: no starting quarterback is listed") and name in w for w in result["warnings"]), code)

    def test_named_starters_are_not_flagged_as_assumed(self):
        result = analysis("BUF", "NYJ", home_qb="Josh Allen", away_qb="Geno Smith")
        self.assertEqual(result["teams"]["home"]["qb_source"], "schedule_name")
        self.assertFalse(any("no starting quarterback" in w for w in result["warnings"]))

    def test_id_match_is_reported(self):
        result = analysis("BUF", "NYJ", home_qb="Josh Allen", home_qb_id="00-BUF")
        self.assertEqual(result["teams"]["home"]["qb_source"], "schedule_id")

    def test_listed_but_unseen_name_is_schedule_unknown(self):
        self.assertEqual(analysis("BUF", "NYJ", away_qb="Brand New Rookie")["teams"]["away"]["qb_source"], "schedule_unknown")

    def test_last_game_fill_keeps_its_own_warning_not_the_generic_one(self):
        result = analysis("BUF", "NYJ", home_qb="Josh Allen", home_qb_source="last_game")
        self.assertTrue(any("most recent game" in w for w in result["warnings"]))
        self.assertFalse(any(w.startswith("BUF: no starting quarterback") for w in result["warnings"]))

    def test_team_with_no_identifiable_quarterback_says_so_instead_of_inventing_a_name_history(self):
        result = analysis("BUF", "ATL")  # ATL has no passers in the fixture and the game names none
        away = result["teams"]["away"]
        self.assertEqual(away["qb_source"], "none")
        warning = next(w for w in result["warnings"] if w.startswith("ATL:"))
        self.assertIn("could be identified", warning)
        self.assertNotIn("Unknown QB", warning)  # not "no NFL passing history found for Unknown QB"

    def test_missing_qb_data_adds_no_per_team_assumption_warnings(self):
        result = analysis("BUF", "NYJ", qb_context={})
        self.assertFalse(any("no starting quarterback" in w for w in result["warnings"]))
        self.assertEqual(sum("Quarterback data unavailable" in w for w in result["warnings"]), 1)

    def test_risk_item_cites_the_engine_stdev_not_a_hard_coded_number(self):
        result = analysis("BUF", "NYJ", home_qb="Case Keenum")
        risk = next(r for r in result["reasoning"] if r["title"] == "Quarterback risk")["detail"]
        self.assertIn(f"{m.MARGIN_STDEV:g}-point", risk)
        self.assertNotIn("wider outcomes", risk)  # an unmeasured claim


class DegenerateCaseTests(unittest.TestCase):
    """Cases found by randomized fuzzing (3000 random games): odd but legal inputs that broke a stated guarantee."""

    @staticmethod
    def _numbers(detail):
        return re.findall(r"\d+(?:\.\d+)?", detail)

    def test_every_reasoning_item_cites_two_numbers_even_for_a_pickem_against_a_pickem_line(self):
        ratings = make_ratings({"DAL": team_row(), "PHI": team_row()})
        ratings["teams"]["PHI"] = dict(ratings["teams"]["DAL"], team="PHI")
        result = analysis("DAL", "PHI", ratings=ratings, qb_context=m._neutral_qb_context(), neutral_site=True, market_spread_home=0.0)
        self.assertEqual(result["scoring"]["model_spread_home"], 0.0)
        for item in result["reasoning"]:
            self.assertGreaterEqual(len(self._numbers(item["detail"])), 2, item["title"])
        market = next(r for r in result["reasoning"] if r["title"] == "Market check")
        self.assertIn("pick'em", market["detail"])
        self.assertIn("50.0%", market["detail"])  # the model's own probabilities are always cited

    def test_neutral_site_item_cites_numbers_without_any_other_context(self):
        result = analysis("BUF", "NYJ", neutral_site=True)  # no rest, wind, temperature or division info
        item = next(r for r in result["reasoning"] if r["title"] == "Neutral site and situation")
        self.assertGreaterEqual(len(self._numbers(item["detail"])), 2)
        self.assertIn("0.0 points", item["detail"])
        self.assertIn("2.0 points", item["detail"])

    def test_pass_through_metadata_is_sanitised(self):
        nan, inf = float("nan"), float("inf")
        result = analysis("BUF", "NYJ", game_id=nan, week=nan, season=inf, kickoff_label=nan, status=nan)
        self.assertEqual((result["week"], result["season"]), (None, None))
        self.assertEqual((result["game_id"], result["kickoff_label"], result["status"]), ("NYJ@BUF", "TBD", "Upcoming"))
        assert_all_finite(self, result)
        clean = analysis("BUF", "NYJ", week="5", season=2026.0, game_id="2026_05_NYJ_BUF", kickoff_label="Sun Oct 11 - 1:00 PM ET", status="Final 20-17")
        self.assertEqual((clean["week"], clean["season"], clean["game_id"], clean["status"]), (5, 2026, "2026_05_NYJ_BUF", "Final 20-17"))
        self.assertIsInstance(clean["week"], int)

    def test_non_finite_qb_names_count_as_unlisted(self):
        result = analysis("BUF", "NYJ", home_qb=float("inf"), away_qb=-float("inf"))
        self.assertEqual(result["teams"]["home"]["qb_name"], "Josh Allen")  # fell back to the team's primary, not a QB named "inf"
        self.assertEqual(result["teams"]["away"]["qb_name"], "Geno Smith")
        self.assertEqual(m._text_or_empty(float("inf")), "")


class EvaluatePriceInputTests(unittest.TestCase):
    """NaN / inf / +50 used to pass straight through the price math and come back as NaN or a fictional probability."""

    def setUp(self):
        self.result = analysis("BUF", "NYJ")

    def test_non_american_prices_are_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf"), 0.0, 50.0, -99.0, 99.99, None, True, "abc"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                m.evaluate_moneyline_price(self.result, "home", bad)

    def test_boundary_prices_are_accepted_and_finite(self):
        for good in (100.0, -100.0, 100, -10000.0, 10000.0, np.float64(-110.0)):  # text is the caller's job (odds_input.parse_odds)
            evaluated = m.evaluate_moneyline_price(self.result, "away", good)
            assert_all_finite(self, evaluated)
            self.assertTrue(0.0 < evaluated["implied_probability"] < 1.0)

    def test_bad_stakes_are_rejected(self):
        for bad in (float("nan"), float("inf"), -1.0, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                m.evaluate_moneyline_price(self.result, "home", -110.0, stake=bad)
        self.assertEqual(m.evaluate_moneyline_price(self.result, "home", -110.0, stake=0.0)["expected_value"], 0.0)

    def test_extreme_prices_price_correctly(self):
        home = self.result["probabilities"]["home_win"]
        long_shot = m.evaluate_moneyline_price(self.result, "home", 10000.0)  # +10000 pays 101x
        self.assertAlmostEqual(long_shot["expected_value"], round(100.0 * (home * 100.0 - (1.0 - home)), 2), places=2)
        chalk = m.evaluate_moneyline_price(self.result, "home", -10000.0)
        self.assertAlmostEqual(chalk["implied_probability"], 10000.0 / 10100.0, places=4)
        self.assertLess(chalk["expected_value"], 0)


# --------------------------------------------------------------------------
# Live data (skipped automatically when nflverse cannot be reached)
# --------------------------------------------------------------------------

class LiveDataTests(unittest.TestCase):
    """Two live checks.  nflreadpy has no request timeout, so everything is loaded once in a daemon
    thread with a hard deadline: a stalled or offline network skips these tests instead of hanging."""

    LOAD_TIMEOUT_SECONDS = 60.0

    @classmethod
    def setUpClass(cls):
        m.clear_caches()
        cls.season = m.current_season()
        loaded: dict = {}

        def work():
            try:
                loaded["schedule"] = m.load_schedule(cls.season)
                loaded["ratings"] = m.load_team_ratings(cls.season)
                loaded["qb"] = m.load_qb_context(cls.season)
            except Exception as error:  # ScheduleUnavailable or anything else the network throws
                loaded["error"] = f"{type(error).__name__}: {error}"

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join(cls.LOAD_TIMEOUT_SECONDS)
        if worker.is_alive():
            raise unittest.SkipTest(f"nflverse did not answer within {cls.LOAD_TIMEOUT_SECONDS:.0f}s")
        if "error" in loaded:
            raise unittest.SkipTest(f"nflverse data unavailable: {loaded['error']}")
        cls.schedule, cls.ratings, cls.qb_context = loaded["schedule"], loaded["ratings"], loaded["qb"]

    def test_live_schedule_and_ratings_are_finite_and_complete(self):
        schedule = self.schedule
        self.assertEqual(list(schedule.columns), m.SCHEDULE_COLUMNS)
        weeks = m.weeks_in_schedule(schedule)
        self.assertIn(m.default_week(schedule), weeks)
        self.assertTrue(set(schedule["away_team"]) <= set(NFL_TEAMS))
        unplayed = schedule[~schedule["played"]]
        if len(unplayed):  # an unplayed game never reports a score
            self.assertTrue(unplayed["home_score"].isna().all())
        if "no data" in self.ratings["meta"]["source"]:
            self.skipTest("play-by-play unavailable")
        self.assertEqual(set(self.ratings["teams"]), set(NFL_TEAMS))
        assert_all_finite(self, self.ratings)

    def test_live_analysis_of_a_real_game_is_finite_everywhere(self):
        if "no data" in self.ratings["meta"]["source"]:
            self.skipTest("play-by-play unavailable")
        week = m.default_week(self.schedule)
        games = m.games_for_week(self.schedule, week)
        self.assertTrue(games)
        for game in games:
            result = m.analyze_game(game, self.ratings, self.qb_context)
            assert_all_finite(self, result)
            self.assertGreaterEqual(len(result["reasoning"]), 5)
            probs = result["probabilities"]
            self.assertAlmostEqual(probs["home_win"] + probs["away_win"], 1.0, places=9)
            self.assertTrue(0 < probs["home_win"] < 1)
            if game["market_home_ml"] is not None and game["market_away_ml"] is not None:
                self.assertIsNotNone(result["moneyline"]["market_home_prob_novig"])


if __name__ == "__main__":
    unittest.main()
