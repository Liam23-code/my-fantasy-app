"""Tests for ``modules.nfl_team_features`` -- synthetic frames through the patched I/O boundary (no network)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from modules import nfl_team_features as tf

TEAMS10 = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN"]
SEASON = 2025
PASS_PLAYS, RUN_PLAYS = 18, 12


@pytest.fixture(autouse=True)
def _fresh_cache():
    tf.clear_cache()
    yield
    tf.clear_cache()


def _pass_epa(defteam_index: int) -> float:
    """Defense i allows 0.05 * i EPA per dropback: defense 0 is the best, defense 9 the worst."""
    return 0.05 * defteam_index


def make_pbp(weeks=(1, 2, 3), teams=TEAMS10, *, rename=None, extra_week=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(pbp, participation)``: every team plays once a week, 18 dropbacks + 12 runs per game.

    Team ``i`` is pressured on ``2 + i`` of its 18 dropbacks per game (participation ``was_pressure``).
    """
    plays, part = [], []
    all_weeks = list(weeks) + ([extra_week] if extra_week else [])
    for week in all_weeks:
        order = teams[week % len(teams):] + teams[: week % len(teams)]
        for g in range(len(order) // 2):
            a, b = order[2 * g], order[2 * g + 1]
            game_id = f"{SEASON}_{week:02d}_{a}_{b}"
            for offense, defense in ((a, b), (b, a)):
                oi, di = teams.index(offense), teams.index(defense)
                for n in range(PASS_PLAYS + RUN_PLAYS):
                    is_pass = n < PASS_PLAYS
                    pid = (0 if offense == a else 1000) + n + 1
                    name = None if rename is None else rename.get(offense, offense)
                    plays.append({
                        "game_id": game_id, "play_id": pid, "season_type": "REG", "week": week,
                        "posteam": name or offense, "defteam": defense, "play_type": "pass" if is_pass else "run",
                        "down": 1, "qb_dropback": 1 if is_pass else 0, "qb_scramble": 0,
                        "pass_attempt": 1 if is_pass else 0, "sack": 0, "qb_hit": 0,
                        "epa": _pass_epa(di) if is_pass else 0.0, "yards_gained": 6.0, "air_yards": 8.0,
                        "passer_player_id": f"QB-{offense}" if is_pass else None,
                        "passer_player_name": f"Q. {offense}" if is_pass else None,
                        "rusher_player_id": None if is_pass else f"RB-{offense}",
                        "yardline_100": 60.0, "fixed_drive": n // 6, "drive": n // 6, "fixed_drive_result": "Punt",
                        "game_seconds_remaining": 3000 - 60 * n, "half_seconds_remaining": 1500 - 30 * n,
                        "qtr": 1 + n // 10, "wp": 0.5, "pass_oe": 1.0,
                    })
                    if is_pass:
                        part.append({
                            "nflverse_game_id": game_id, "play_id": pid, "possession_team": name or offense,
                            "was_pressure": n < 2 + oi, "number_of_pass_rushers": 4,
                            "defense_man_zone_type": "ZONE_COVERAGE",
                        })
    return pd.DataFrame(plays), pd.DataFrame(part)


def install(monkeypatch, pbp, part=None, pfr_pass=None):
    empty = pd.DataFrame()

    def load_pbp(season):
        return pbp if int(season) == SEASON else empty

    def load_participation(season):
        return (part if part is not None else empty) if int(season) == SEASON else empty

    def load_pfr(season, stat_type):
        return pfr_pass if (int(season) == SEASON and stat_type == "pass" and pfr_pass is not None) else empty

    monkeypatch.setattr(tf, "_load_pbp", load_pbp)
    monkeypatch.setattr(tf, "_load_participation", load_participation)
    monkeypatch.setattr(tf, "_load_pfr", load_pfr)
    monkeypatch.setattr(tf, "_load_ngs", lambda season, stat_type: empty)
    monkeypatch.setattr(tf, "_load_snaps", lambda season: empty)


# --- pure helpers -------------------------------------------------------------------------------
def test_shrink_is_the_documented_games_over_games_plus_k_blend():
    value, weight = tf.shrink(0.40, 4.0, 0.30, 8.0)
    assert weight == pytest.approx(4.0 / 12.0)
    assert value == pytest.approx(0.4 * (4 / 12) + 0.3 * (8 / 12))


def test_shrink_edge_cases():
    assert tf.shrink(None, 0.0, 0.3, 8.0) == (0.3, 0.0)
    assert tf.shrink(0.4, 0.0, 0.3, 8.0) == (0.3, 0.0)      # no games -> the prior
    assert tf.shrink(0.4, 5.0, None, 8.0) == (0.4, 1.0)      # no prior -> the current value
    assert tf.shrink(None, 3.0, None, 8.0) == (None, 0.0)
    assert tf.shrink(float("nan"), 3.0, 0.3, 8.0) == (0.3, 0.0)
    value, weight = tf.shrink(0.5, 10.0, 0.3, 0.0)           # K = 0 trusts the sample fully
    assert (value, weight) == (0.5, 1.0)


def test_regress_prior_pulls_toward_the_league_by_its_own_reliability():
    value, lam = tf.regress_prior(0.60, 17.0, 0.50, 8.0)
    assert lam == pytest.approx(17 / 25)
    assert value == pytest.approx(0.5 + lam * 0.10)
    assert tf.regress_prior(0.6, 0.0, 0.5, 8.0)[0] == pytest.approx(0.5)  # no sample: all league level


@pytest.mark.parametrize("raw, expected", [("LAR", "LA"), ("la", "LA"), (" KC ", "KC"), ("WSH", "WAS"), ("JAC", "JAX"),
                                           ("XXX", None), (None, None), (float("nan"), None), ("", None)])
def test_normalize_team(raw, expected):
    assert tf.normalize_team(raw) == expected


def test_percentile_grades_direction_and_guard():
    values = {f"T{i}": float(i) for i in range(10)}
    lower_better = tf._percentile_grades(values, higher_is_better=False)
    assert lower_better["T0"] == pytest.approx(100.0) and lower_better["T9"] == pytest.approx(0.0)
    higher_better = tf._percentile_grades(values, higher_is_better=True)
    assert higher_better["T9"] == pytest.approx(100.0) and higher_better["T0"] == pytest.approx(0.0)
    assert np.mean(list(lower_better.values())) == pytest.approx(50.0)
    assert tf._percentile_grades({"A": 1.0, "B": 2.0}, higher_is_better=True) == {}      # too few teams to grade


def test_ranks_worst_last_and_ties():
    ranks = tf._ranks_worst_last({f"T{i}": float(i) for i in range(10)})
    assert ranks["T0"] == 1.0 and ranks["T9"] == 10.0
    assert tf._ranks_worst_last({"A": 1.0}) == {}


def test_every_metric_has_a_k_in_range_and_a_study_note():
    for metric in tf._METRICS:
        k = tf.SHRINKAGE_K[metric.mid]
        assert 2.0 <= k <= 40.0, metric.mid
        assert metric.mid in tf._K_STUDY and tf._K_STUDY[metric.mid]


def test_scale_maps_are_sane_and_documented():
    for name in ("pressure", "blitz"):
        a, b = tf._SCALE_MAPS[name]
        assert 0.0 <= a <= 0.25 and 0.5 <= b <= 1.5
        assert name in tf._SCALE_STUDY
    num, den = tf._mapped(pd.Series([10.0]), pd.Series([40.0]), "pressure")
    a, b = tf._SCALE_MAPS["pressure"]
    assert float(num.iloc[0] / den.iloc[0]) == pytest.approx(a + b * 0.25)


def test_no_free_feed_keys_are_declared_and_not_in_the_vocabulary():
    assert set(tf.NO_FREE_FEED) == {"opp_coverage_bust_rate", "opp_cb1_shadow_rate", "opp_bracket_coverage_rate"}
    assert not set(tf.NO_FREE_FEED) & set(tf.OFFENSE_KEYS + tf.DEFENSE_KEYS)
    assert set(tf.FEATURE_BASIS) == set(tf.OFFENSE_KEYS + tf.DEFENSE_KEYS)
    assert set(tf._KIND.values()) <= {"real", "proxy"}


# --- table assembly -----------------------------------------------------------------------------
def test_pass_block_proxy_is_one_minus_pressure_shrunk_to_the_league(monkeypatch):
    pbp, part = make_pbp()
    install(monkeypatch, pbp, part)
    table = tf.team_feature_table(SEASON)
    assert set(table) == set(TEAMS10)
    raw = {t: 1.0 - (2 + i) / PASS_PLAYS for i, t in enumerate(TEAMS10)}
    league = float(np.mean(list(raw.values())))
    w = 3.0 / (3.0 + tf.SHRINKAGE_K["ol_pressure_allowed"])
    for team in TEAMS10:
        expected = w * raw[team] + (1.0 - w) * league
        assert table[team]["offense"]["ol_pass_block_win_rate"] == pytest.approx(expected, abs=2e-4)
    # monotone: the line that allows fewer pressures grades better
    values = [table[t]["offense"]["ol_pass_block_win_rate"] for t in TEAMS10]
    assert values == sorted(values, reverse=True)


def test_opposing_pressure_uses_the_defenders_side(monkeypatch):
    pbp, part = make_pbp()
    install(monkeypatch, pbp, part)
    table = tf.team_feature_table(SEASON)
    for team in TEAMS10:
        assert 0.0 < table[team]["defense"]["opp_pressure_rate"] < 1.0
    # every defense faces three different offenses, so the table must carry real variation
    assert len({round(table[t]["defense"]["opp_pressure_rate"], 4) for t in TEAMS10}) > 1


def test_defense_grades_and_rank_follow_epa_allowed(monkeypatch):
    pbp, part = make_pbp()
    install(monkeypatch, pbp, part)
    table = tf.team_feature_table(SEASON)
    grades = [table[t]["defense"]["opp_coverage_grade"] for t in TEAMS10]
    assert grades == sorted(grades, reverse=True)               # defense 0 allows the least EPA -> best grade
    assert table[TEAMS10[0]]["defense"]["opp_coverage_grade"] > table[TEAMS10[-1]]["defense"]["opp_coverage_grade"]
    ranks = [table[t]["defense"]["opp_defense_rank"] for t in TEAMS10]
    assert ranks[0] < ranks[-1]                                 # 1 = best defense, higher = worse


def test_every_emitted_key_carries_kind_and_basis_and_no_unfed_key_leaks(monkeypatch):
    pbp, part = make_pbp()
    install(monkeypatch, pbp, part)
    table = tf.team_feature_table(SEASON)
    for team, row in table.items():
        keys = set(row["offense"]) | set(row["defense"])
        assert keys <= set(tf.OFFENSE_KEYS) | set(tf.DEFENSE_KEYS)
        assert not keys & set(tf.NO_FREE_FEED)
        assert set(row["kinds"]) == keys and set(row["basis"]) == keys
        assert set(row["kinds"].values()) <= {"real", "proxy"}
        assert all(isinstance(text, str) and text for text in row["basis"].values())
        assert row["games"] == 3 and row["season"] == SEASON and row["through_week"] is None
    assert table["ARI"]["kinds"]["ol_pass_block_win_rate"] == "proxy"
    assert table["ARI"]["kinds"]["team_epa_per_play"] == "real"


def test_through_week_cuts_off_later_games_without_leakage(monkeypatch):
    pbp, part = make_pbp(weeks=(1, 2), extra_week=3)
    install(monkeypatch, pbp, part)
    cut = tf.team_feature_table(SEASON, through_week=2)
    tf.clear_cache()
    only_two, part2 = make_pbp(weeks=(1, 2))
    install(monkeypatch, only_two, part2)
    expected = tf.team_feature_table(SEASON)
    assert set(cut) == set(expected)
    for team in expected:
        assert cut[team]["offense"] == expected[team]["offense"]
        assert cut[team]["defense"] == expected[team]["defense"]
        assert cut[team]["games"] == expected[team]["games"] == 2
        assert cut[team]["through_week"] == 2


def test_week_zero_means_priors_only_and_gives_nothing_for_a_season_with_no_prior(monkeypatch):
    pbp, part = make_pbp()
    install(monkeypatch, pbp, part)
    assert tf.team_feature_table(SEASON, through_week=0) == {}


def test_abbreviations_are_normalised(monkeypatch):
    pbp, part = make_pbp(rename={"ARI": "LAR"})              # a spelling the table must fold into "LA"
    install(monkeypatch, pbp, part)
    table = tf.team_feature_table(SEASON)
    assert "LAR" not in table
    assert table["LA"]["offense"]                          # its offense was recorded under the folded code
    assert not table["ARI"]["offense"]                     # ARI only appears as an opponent's defender in this frame
    assert table["ARI"]["defense"]


def test_pfr_fallback_is_mapped_onto_the_participation_scale(monkeypatch):
    pbp, _ = make_pbp()
    rows = []
    for week in (1, 2, 3):
        order = TEAMS10[week % 10:] + TEAMS10[: week % 10]
        for g in range(5):
            a, b = order[2 * g], order[2 * g + 1]
            for off, de in ((a, b), (b, a)):
                rows.append({"team": off, "opponent": de, "week": week, "game_type": "REG", "times_pressured": 4.0})
    install(monkeypatch, pbp, None, pfr_pass=pd.DataFrame(rows))
    table = tf.team_feature_table(SEASON)
    a, b = tf._SCALE_MAPS["pressure"]
    expected = a + b * (4.0 / PASS_PLAYS)
    for team in TEAMS10:
        assert table[team]["defense"]["opp_pressure_rate"] == pytest.approx(expected, abs=2e-4)
        assert "mapped" in table[team]["basis"]["opp_pressure_rate"] or "PFR" in table[team]["basis"]["opp_pressure_rate"]


# --- failure policy -----------------------------------------------------------------------------
def test_every_feed_empty_returns_an_empty_table_and_never_raises(monkeypatch):
    install(monkeypatch, pd.DataFrame())
    assert tf.team_feature_table(SEASON) == {}
    assert tf.league_reference(SEASON) == {}


def test_a_loader_that_raises_degrades_to_an_empty_table(monkeypatch):
    def boom(season):
        raise RuntimeError("provider outage")

    install(monkeypatch, pd.DataFrame())
    monkeypatch.setattr(tf, "_load_pbp", boom)
    assert tf.team_feature_table(SEASON) == {}


@pytest.mark.parametrize("bad", ["nope", object(), [], float("nan")])
def test_a_garbage_season_argument_never_raises(monkeypatch, bad):
    install(monkeypatch, pd.DataFrame())
    assert isinstance(tf.team_feature_table(bad), dict)          # falls back to the default season


def test_a_frame_missing_its_columns_is_treated_as_no_data(monkeypatch):
    install(monkeypatch, pd.DataFrame({"game_id": ["g"], "week": [1]}))
    assert tf.team_feature_table(SEASON) == {}


def test_results_are_deterministic_and_the_cache_is_not_aliased(monkeypatch):
    pbp, part = make_pbp()
    install(monkeypatch, pbp, part)
    first = tf.team_feature_table(SEASON)
    snapshot = copy.deepcopy(first)
    first["ARI"]["offense"]["ol_pass_block_win_rate"] = -99.0     # mutate what the caller got
    first.pop("ATL")
    second = tf.team_feature_table(SEASON)
    assert second == snapshot


def test_league_reference_summarises_the_table(monkeypatch):
    pbp, part = make_pbp()
    install(monkeypatch, pbp, part)
    ref = tf.league_reference(SEASON)
    table = tf.team_feature_table(SEASON)
    values = [table[t]["offense"]["ol_pass_block_win_rate"] for t in TEAMS10]
    assert ref["ol_pass_block_win_rate"]["n"] == len(TEAMS10)
    assert ref["ol_pass_block_win_rate"]["mean"] == pytest.approx(float(np.mean(values)))
    assert ref["ol_pass_block_win_rate"]["sd"] == pytest.approx(float(np.std(values, ddof=1)))
