"""Tests for ``modules.nfl_matchup_features`` -- every loader is monkeypatched at the module's I/O boundary (no network)."""
from __future__ import annotations

import ast
import copy
import datetime as dt
import json
import math
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from modules import coverage_mismatch_model, explosion_model, explosive_play_model, game_script_model
from modules import nfl_matchup_features as nmf
from modules import nfl_player_features as npf
from modules import nfl_team_features as ntf
from modules import nfl_upgrade_pipeline as pipe
from modules import pass_game_collapse, qb_collapse_model, red_zone_funnel_model, run_game_collapse

TODAY = dt.date(2026, 10, 12)      # season in progress = 2026 (September rule)
SEASON = 2026

# originals, captured before the autouse fixture replaces them
_REAL_STATUS_OVERLAY = nmf._status_overlay
_REAL_SNAP_FRAME = nmf._snap_frame
_REAL_LOAD_SCHEDULE = nmf._load_schedule
_REAL_OL_INJURY_COUNT = nmf._ol_injury_count


# ---------------------------------------------------------------------------------------------
# fixtures and builders
# ---------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _env(monkeypatch):
    nmf.clear_cache()
    monkeypatch.setattr(ntf, "_today", lambda: TODAY)
    monkeypatch.setattr(nmf, "_team_table", lambda season, through_week: {})
    monkeypatch.setattr(nmf, "_player_table", lambda season, through_week: {})
    monkeypatch.setattr(nmf, "_load_schedule", lambda season: pd.DataFrame())
    monkeypatch.setattr(nmf, "_status_overlay", lambda: {})
    monkeypatch.setattr(nmf, "_snap_frame", lambda season: pd.DataFrame())
    yield
    nmf.clear_cache()


PROXY_KEYS = {
    "ol_pass_block_win_rate", "ol_run_block_win_rate", "ol_continuity", "opp_pass_rush_win_rate", "opp_coverage_grade",
    "opp_run_defense_grade", "opp_explosive_play_prevention", "opp_rz_pass_defense_grade", "opp_rz_run_defense_grade",
}

KC_OFF = {
    "ol_pass_block_win_rate": 0.66, "ol_run_block_win_rate": 0.72, "ol_continuity": 0.80, "team_epa_per_play": 0.10,
    "qb_efficiency": 0.15, "qb_deep_attempt_rate": 0.12, "pace_seconds_per_play": 31.5, "wr1_separation": 3.1,
    "rb_yards_after_contact": 2.0, "proe": 0.03, "rroe": -0.03,
}
KC_DEF = {"opp_pressure_rate": 0.25, "opp_blitz_rate": 0.20, "opp_coverage_grade": 75.0, "opp_def_epa_per_play": -0.04}
BUF_OFF = {"team_epa_per_play": 0.21, "explosive_pass_rate": 0.09, "qb_efficiency": 0.30}
BUF_DEF = {
    "opp_pressure_rate": 0.36, "opp_blitz_rate": 0.30, "opp_coverage_grade": 40.0, "opp_def_epa_per_play": 0.05,
    "opp_defense_rank": 22.0, "opp_run_defense_grade": 30.0, "opp_stacked_box_rate": 0.25,
}


def team_row(offense=None, defense=None, *, games=5, lead="P.Mahomes", kinds=None, basis=None):
    offense, defense = dict(offense or {}), dict(defense or {})
    keys = [*offense, *defense]
    k = {key: ("proxy" if key in PROXY_KEYS else "real") for key in keys}
    k.update(kinds or {})
    b = {key: f"fake team basis for {key}" for key in keys}
    for key in ("qb_efficiency", "qb_deep_attempt_rate"):
        if key in b:
            b[key] = f"REAL: fake | lead={lead}; 2026 wk1-5: {games} g, n=190 dropbacks; weight on current 38%"
    b.update(basis or {})
    return {"offense": offense, "defense": defense, "kinds": k, "basis": b, "games": games, "season": SEASON,
            "through_week": 5}


def team_table(**overrides):
    table = {
        "KC": team_row(KC_OFF, KC_DEF, lead="P.Mahomes"),
        "BUF": team_row(BUF_OFF, BUF_DEF, lead="J.Allen"),
    }
    table.update(overrides)
    return table


def player_entry(name, team, position, features, *, games=5, dropbacks=None, kind="proxy"):
    basis = {key: f"{kind}: fake player feed; n={dropbacks if dropbacks is not None else 100} dropbacks; raw 0.1"
             for key in features}
    return {"name": name, "team": team, "position": position, "features": dict(features),
            "kinds": {key: kind for key in features}, "basis": basis, "games": games, "season": SEASON,
            "through_week": 5}


MAHOMES = "00-0033873"
BACKUP = "00-0039999"
ALLEN = "00-0034857"
RECEIVER = "00-0036001"
RUNNER = "00-0036002"


def player_table():
    return {
        MAHOMES: player_entry("Patrick Mahomes", "KC", "QB", {
            "qb_time_to_throw": 2.70, "qb_pressure_epa": -0.35, "qb_sack_avoidance": 0.94, "qb_scramble_rate": 0.06,
            "qb_efficiency": 0.11}, games=5, dropbacks=190),
        BACKUP: player_entry("Spare Quarterback", "KC", "QB", {"qb_time_to_throw": 3.2, "qb_efficiency": -0.2},
                             games=2, dropbacks=25),
        ALLEN: player_entry("Josh Allen", "BUF", "QB", {"qb_time_to_throw": 2.9, "qb_efficiency": 0.30}, games=5,
                            dropbacks=200),
        RECEIVER: player_entry("Alpha Receiver", "KC", "WR", {"wr_separation": 3.3, "wr_deep_route_rate": 0.21},
                               games=5),
        RUNNER: player_entry("Bravo Runner", "KC", "RB", {"rb_yards_after_contact": 2.6, "rb_breakaway_rate": 0.05},
                             games=5),
    }


GAME_KC_BUF = {"game_id": "2026_06_BUF_KC", "season": 2026, "game_type": "REG", "week": 6, "gameday": "2026-10-18",
               "home_team": "KC", "away_team": "BUF", "spread_line": 3.5, "total_line": 47.5}


def schedule(*rows, extra=None):
    frame = pd.DataFrame(list(rows))
    for column, value in (extra or {}).items():
        frame[column] = value
    return frame


def install(monkeypatch, *, teams=None, players=None, sched=None, overlay=None, snaps=None):
    """Patch the loaders and return the recorder of the arguments each was called with."""
    calls = {"team": [], "player": [], "schedule": []}

    def fake_team(season, through_week):
        calls["team"].append((season, through_week))
        return copy.deepcopy(teams if teams is not None else {})

    def fake_player(season, through_week):
        calls["player"].append((season, through_week))
        return copy.deepcopy(players if players is not None else {})

    def fake_schedule(season):
        calls["schedule"].append(season)
        return sched.copy() if sched is not None else pd.DataFrame()

    monkeypatch.setattr(nmf, "_team_table", fake_team)
    monkeypatch.setattr(nmf, "_player_table", fake_player)
    monkeypatch.setattr(nmf, "_load_schedule", fake_schedule)
    if overlay is not None:
        monkeypatch.setattr(nmf, "_status_overlay", lambda: copy.deepcopy(overlay))
    if snaps is not None:
        monkeypatch.setattr(nmf, "_snap_frame", lambda season: snaps.copy())
    return calls


def full_world(monkeypatch, **kw):
    kw.setdefault("teams", team_table())
    kw.setdefault("players", player_table())
    kw.setdefault("sched", schedule(GAME_KC_BUF))
    return install(monkeypatch, **kw)


OL_NAMES = [("Alpha Tackle", "T", 60), ("Bravo Guard", "G", 58), ("Charlie Center", "C", 57),
            ("Delta Guard", "G", 55), ("Echo Tackle", "T", 54), ("Foxtrot Reserve", "T", 12)]


def ol_snaps(team="KC", weeks=(1, 2, 3, 4, 5), extra_rows=()):
    rows = []
    for week in weeks:
        for name, pos, snaps in OL_NAMES:
            rows.append({"team": team, "week": week, "position": pos, "player": name,
                         "pfr_player_id": f"PFR-{name.split()[0]}", "offense_snaps": snaps})
        rows.append({"team": team, "week": week, "position": "TE", "player": "Golf Tight", "pfr_player_id": "PFR-Golf",
                     "offense_snaps": 50})
        rows.append({"team": "BUF", "week": week, "position": "T", "player": "Other Tackle", "pfr_player_id": "PFR-O",
                     "offense_snaps": 65})
    rows.extend(extra_rows)
    return pd.DataFrame(rows)


def status_key(name):
    return "nm:" + "".join(ch for ch in name.lower() if ch.isalpha())


def overlay_of(*pairs, stamp="2026-10-10T12:00:00Z"):
    return {status_key(name): {"status": status, "last_updated": stamp, "name": name, "source": "sleeper"}
            for name, status in pairs}


def flat(context):
    return pipe.model_inputs(context)


# ---------------------------------------------------------------------------------------------
# market lines: sign convention, supplied values, no look-ahead
# ---------------------------------------------------------------------------------------------
def test_home_team_gets_negative_spread_line(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    # nflverse: spread_line +3.5 = HOME (KC) favoured by 3.5 -> sportsbook sign -3.5
    assert ctx["game"]["vegas_spread"] == -3.5
    assert ctx["game"]["vegas_total"] == 47.5
    assert ctx["is_home"] is True
    assert ctx["sources"]["vegas_spread"]["kind"] == "real"
    assert "spread_line +3.5" in ctx["sources"]["vegas_spread"]["basis"]


def test_away_team_gets_positive_spread_line(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("BUF", "KC", season=SEASON, week=6)
    assert ctx["game"]["vegas_spread"] == 3.5          # away underdog by 3.5
    assert ctx["is_home"] is False


def test_away_favourite_has_negative_vegas_spread(monkeypatch):
    row = dict(GAME_KC_BUF, spread_line=-2.5)           # away (BUF) favoured by 2.5
    full_world(monkeypatch, sched=schedule(row))
    away = nmf.build_game_context("BUF", "KC", season=SEASON, week=6)
    home = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert away["game"]["vegas_spread"] == -2.5
    assert home["game"]["vegas_spread"] == 2.5


def test_sign_convention_matches_the_game_script_models_helper(monkeypatch):
    for line in (3.5, -2.5, 10.0, 0.0):
        full_world(monkeypatch, sched=schedule(dict(GAME_KC_BUF, spread_line=line)))
        home = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)["game"]["vegas_spread"]
        away = nmf.build_game_context("BUF", "KC", season=SEASON, week=6)["game"]["vegas_spread"]
        assert home == game_script_model.team_spread_from_schedule(line, True)
        assert away == game_script_model.team_spread_from_schedule(line, False)


def test_pickem_spread_is_positive_zero(monkeypatch):
    full_world(monkeypatch, sched=schedule(dict(GAME_KC_BUF, spread_line=0.0)))
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["game"]["vegas_spread"] == 0.0
    assert math.copysign(1.0, ctx["game"]["vegas_spread"]) == 1.0


def test_game_not_found_omits_lines_and_warns(monkeypatch):
    full_world(monkeypatch, sched=schedule(dict(GAME_KC_BUF, week=9)))
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert "vegas_spread" not in ctx["game"] and "vegas_total" not in ctx["game"]
    assert any("no schedule row" in w for w in ctx["warnings"])
    assert ctx["sources"]["vegas_spread"]["kind"] == "default"
    assert ctx["is_home"] is None


def test_schedule_without_published_lines_warns(monkeypatch):
    full_world(monkeypatch, sched=schedule(dict(GAME_KC_BUF, spread_line=float("nan"), total_line=None)))
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert "vegas_spread" not in ctx["game"] and "vegas_total" not in ctx["game"]
    assert any("no spread_line" in w for w in ctx["warnings"])
    assert any("no total_line" in w for w in ctx["warnings"])
    assert ctx["is_home"] is True        # the row still identifies the home team


def test_supplied_lines_beat_schedule_and_skip_the_lookup(monkeypatch):
    calls = full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, is_home=True, vegas_spread=-6.5, vegas_total=51)
    assert ctx["game"]["vegas_spread"] == -6.5 and ctx["game"]["vegas_total"] == 51.0
    assert ctx["sources"]["vegas_spread"]["kind"] == "override"
    assert calls["schedule"] == []       # nothing to look up when the caller supplied lines and is_home


def test_supplied_spread_out_of_range_is_dropped_and_schedule_used(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, vegas_spread=99)
    assert ctx["game"]["vegas_spread"] == -3.5
    assert any("vegas_spread=99" in w for w in ctx["warnings"])


def test_override_dict_beats_vegas_argument(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, vegas_spread=-6.5,
                                 overrides={"vegas_spread": -1.0})
    assert ctx["game"]["vegas_spread"] == -1.0
    assert ctx["sources"]["vegas_spread"]["kind"] == "override"
    assert "-1.0" in ctx["sources"]["vegas_spread"]["basis"]


def test_game_result_columns_are_never_used(monkeypatch):
    base = schedule(GAME_KC_BUF)
    poisoned = schedule(GAME_KC_BUF, extra={"result": 99, "home_score": 70, "away_score": 0, "overtime": 1})
    full_world(monkeypatch, sched=base)
    a = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    full_world(monkeypatch, sched=poisoned)
    b = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert a == b
    assert not ({"result", "home_score", "away_score"} & set(nmf._SCHEDULE_COLUMNS))


def test_schedule_loader_selects_only_pre_game_columns(monkeypatch):
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"game_id": ["g"], "season": [2026], "game_type": ["REG"], "week": [6], "gameday": ["2026-10-18"],
                          "home_team": ["KC"], "away_team": ["BUF"], "spread_line": [3.5], "total_line": [47.5],
                          "result": [7], "home_score": [30], "away_score": [23], "home_moneyline": [-180]})
    fake = types.SimpleNamespace(load_schedules=lambda seasons: frame)
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)
    out = _REAL_LOAD_SCHEDULE(2026)
    assert set(out.columns) == set(nmf._SCHEDULE_COLUMNS)
    assert "result" not in out.columns and "home_moneyline" not in out.columns


def test_is_home_conflict_prefers_the_schedule_and_warns(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, is_home=False)
    assert ctx["is_home"] is True
    assert ctx["game"]["vegas_spread"] == -3.5
    assert any("disagrees with the schedule" in w for w in ctx["warnings"])


def test_divisional_double_meeting(monkeypatch):
    rows = [dict(GAME_KC_BUF), dict(GAME_KC_BUF, week=14, gameday="2026-12-13", game_id="2026_14_KC_BUF",
                                    home_team="BUF", away_team="KC", spread_line=-1.0, total_line=50.0)]
    full_world(monkeypatch, sched=schedule(*rows))
    second = nmf.build_game_context("KC", "BUF", season=SEASON, week=14)
    # BUF home, spread_line -1.0 = the AWAY team (KC) favoured by 1 -> KC (away) gets +spread_line = -1.0
    assert second["game"]["vegas_spread"] == -1.0 and second["game"]["vegas_total"] == 50.0
    assert second["is_home"] is False
    first = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert first["game"]["vegas_spread"] == -3.5 and first["game"]["vegas_total"] == 47.5
    assert first["is_home"] is True


def test_double_meeting_without_week_takes_next_game_in_season_in_progress(monkeypatch):
    rows = [dict(GAME_KC_BUF, week=2, gameday="2026-09-20", game_id="a"),
            dict(GAME_KC_BUF, week=6, gameday="2026-10-18", game_id="b", spread_line=7.0),
            dict(GAME_KC_BUF, week=14, gameday="2026-12-13", game_id="c", spread_line=1.0)]
    full_world(monkeypatch, sched=schedule(*rows))
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON)
    assert ctx["game"]["vegas_spread"] == -7.0                # the first meeting on or after today


def test_double_meeting_without_week_in_a_past_season_is_ambiguous(monkeypatch):
    rows = [dict(GAME_KC_BUF, season=2025, week=2, gameday="2025-09-20"),
            dict(GAME_KC_BUF, season=2025, week=14, gameday="2025-12-13")]
    full_world(monkeypatch, sched=schedule(*rows))
    ctx = nmf.build_game_context("KC", "BUF", season=2025)
    assert "vegas_spread" not in ctx["game"]
    assert any("meet more than once" in w for w in ctx["warnings"])


def test_schedule_lines_switch_off(monkeypatch):
    calls = full_world(monkeypatch)
    monkeypatch.setattr(nmf, "USE_SCHEDULE_LINES", False)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, is_home=True)
    assert "vegas_spread" not in ctx["game"] and "vegas_total" not in ctx["game"]
    assert calls["schedule"] == []
    assert "USE_SCHEDULE_LINES" in ctx["sources"]["vegas_spread"]["basis"]
    # a supplied line still works with the switch off
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, is_home=True, vegas_spread=-3.0)
    assert ctx["game"]["vegas_spread"] == -3.0


# ---------------------------------------------------------------------------------------------
# sections, provenance, precedence
# ---------------------------------------------------------------------------------------------
def test_sections_are_routed_from_the_right_loader(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6,
                                 player={"player_id": RECEIVER, "name": "Alpha Receiver", "team": "KC", "position": "WR"})
    assert ctx["offense"]["team_epa_per_play"] == 0.10
    assert ctx["offense"]["ol_pass_block_win_rate"] == 0.66
    assert ctx["opponent_defense"]["opp_pressure_rate"] == 0.36
    assert ctx["qb"]["qb_time_to_throw"] == 2.70
    assert ctx["player"] == {"wr_separation": 3.3, "wr_deep_route_rate": 0.21}
    assert set(ctx["game"]) == {"vegas_spread", "vegas_total", "opp_epa_per_play"}
    assert ctx["has_features"] is True
    for key in ctx["offense"]:
        assert key in ntf.OFFENSE_KEYS
    for key in ctx["opponent_defense"]:
        assert key in ntf.DEFENSE_KEYS


def test_opponent_defense_comes_from_the_opponents_row(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["opponent_defense"]["opp_pressure_rate"] == 0.36        # BUF's defence, not KC's 0.25
    other = nmf.build_game_context("BUF", "KC", season=SEASON, week=6)
    assert other["opponent_defense"]["opp_pressure_rate"] == 0.25      # now KC's defence
    assert other["offense"]["team_epa_per_play"] == 0.21               # BUF's own offence


def test_opponent_offence_epa_is_wired_into_the_game_section(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["game"]["opp_epa_per_play"] == 0.21                      # BUF offence team_epa_per_play
    assert ctx["offense"]["team_epa_per_play"] == 0.10
    assert "BUF offence team_epa_per_play" in ctx["sources"]["opp_epa_per_play"]["basis"]
    flat_inputs = flat(ctx)
    assert flat_inputs["opp_epa_per_play"] == 0.21 and flat_inputs["team_epa_per_play"] == 0.10
    script = game_script_model.compute_game_script(flat_inputs)
    assert script["inputs_used"]
    assert "opp_epa_per_play" in script["inputs_used"]


def test_sources_carry_loader_kind_and_basis(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["sources"]["team_epa_per_play"] == {"kind": "real", "basis": "fake team basis for team_epa_per_play"}
    assert ctx["sources"]["ol_pass_block_win_rate"]["kind"] == "proxy"
    assert ctx["sources"]["opp_coverage_grade"]["kind"] == "proxy"
    assert ctx["sources"]["qb_time_to_throw"]["kind"] == "proxy"
    assert "fake player feed" in ctx["sources"]["qb_time_to_throw"]["basis"]
    # every present numeric key has a source entry; defaults name keys that are NOT in the sections
    present = {k for name in nmf.SECTIONS for k in ctx[name]}
    assert present <= set(ctx["sources"])
    for key, src in ctx["sources"].items():
        assert src["kind"] in {"real", "proxy", "override", "derived", "default"} and src["basis"]
        if src["kind"] == "default":
            assert key not in present


def test_unlabelled_loader_value_is_never_presented_as_measured(monkeypatch):
    teams = team_table(KC=team_row({"team_epa_per_play": 0.1}, kinds={"team_epa_per_play": "mystery"}),
                       BUF=team_row({}, {"opp_pressure_rate": 0.3}))
    full_world(monkeypatch, teams=teams, sched=schedule())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["sources"]["team_epa_per_play"]["kind"] == "proxy"
    assert "treated as proxy" in ctx["sources"]["team_epa_per_play"]["basis"]


def test_non_numeric_loader_values_are_skipped(monkeypatch):
    teams = team_table(KC=team_row({"team_epa_per_play": float("nan"), "proe": "n/a", "rroe": -0.02}, KC_DEF))
    full_world(monkeypatch, teams=teams)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["offense"] == {"rroe": -0.02}


def test_section_order_matches_the_pipeline_merge_order():
    assert nmf.SECTIONS == pipe.CONTEXT_SECTIONS


def test_later_section_wins_and_sources_describe_the_winner(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    # team table qb_efficiency 0.15 (offense) vs the QB's own 0.11 (qb): the qb section wins, offense copy is removed
    assert "qb_efficiency" not in ctx["offense"]
    assert ctx["qb"]["qb_efficiency"] == 0.11
    assert flat(ctx)["qb_efficiency"] == 0.11
    assert "fake player feed" in ctx["sources"]["qb_efficiency"]["basis"]


def test_a_key_in_two_sections_always_carries_the_same_number(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6,
                                 player={"player_id": RUNNER, "name": "Bravo Runner", "position": "RB"})
    seen = {}
    for name in nmf.SECTIONS:
        for key, value in ctx[name].items():
            if key in seen:
                assert seen[key] == value
            seen[key] = value
    assert ctx["player"]["rb_yards_after_contact"] == 2.6
    assert "rb_yards_after_contact" not in ctx["offense"]       # team lead RB value (2.0) lost to the player's own


# ---------------------------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------------------------
def test_override_wins_and_is_labelled(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6,
                                 overrides={"opp_pressure_rate": 0.45, "team_epa_per_play": 0.2})
    assert ctx["opponent_defense"]["opp_pressure_rate"] == 0.45
    assert ctx["offense"]["team_epa_per_play"] == 0.2
    for key in ("opp_pressure_rate", "team_epa_per_play"):
        assert ctx["sources"][key]["kind"] == "override"
        assert "replaces the" in ctx["sources"][key]["basis"]
    assert flat(ctx)["opp_pressure_rate"] == 0.45


@pytest.mark.parametrize("key, value, section", [
    ("ol_continuity", 0.9, "offense"),
    ("proe", 0.04, "offense"),
    ("opp_cb1_shadow_rate", 0.4, "opponent_defense"),
    ("opp_bracket_coverage_rate", 0.1, "opponent_defense"),
    ("opp_run_defense_grade", 80, "opponent_defense"),
    ("qb_time_to_throw", 3.0, "qb"),
    ("qb_rz_efficiency", 0.3, "qb"),
    ("vegas_total", 44.5, "game"),
    ("ol_injuries", 2, "game"),
    ("opp_epa_per_play", 0.1, "game"),
    ("wr_separation", 3.5, "player"),
    ("te_route_depth", 8.0, "player"),
    ("wr_vs_cb_matchup_grade", 70, "player"),
])
def test_overrides_are_routed_by_the_documented_table(key, value, section):
    ctx = nmf.build_game_context("KC", "BUF", overrides={key: value})
    assert ctx[section][key] == float(value)
    assert all(key not in ctx[other] for other in nmf.SECTIONS if other != section)
    assert nmf.OVERRIDE_SPECS[key][0] == section


def test_override_removes_the_shadowing_loader_values(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6,
                                 player={"player_id": RUNNER, "name": "Bravo Runner", "position": "RB"},
                                 overrides={"rb_yards_after_contact": 1.1, "qb_efficiency": -0.05})
    assert ctx["player"]["rb_yards_after_contact"] == 1.1
    assert "rb_yards_after_contact" not in ctx["offense"]
    assert ctx["qb"]["qb_efficiency"] == -0.05
    assert "qb_efficiency" not in ctx["offense"]
    flat_inputs = flat(ctx)
    assert flat_inputs["rb_yards_after_contact"] == 1.1 and flat_inputs["qb_efficiency"] == -0.05


def test_override_beats_a_later_section_copy_when_the_player_is_the_quarterback(monkeypatch):
    # a projected QB sits in both ``qb`` and ``player`` (``player`` is merged last): an override routed to
    # ``qb`` must clear the ``player`` copy or the loader value would shadow it
    full_world(monkeypatch)
    row = {"player_id": MAHOMES, "name": "Patrick Mahomes", "team": "KC", "position": "QB"}
    plain = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, player=row)
    assert plain["qb"]["qb_time_to_throw"] == plain["player"]["qb_time_to_throw"] == 2.70
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, player=row, overrides={"qb_time_to_throw": 3.4})
    assert ctx["qb"]["qb_time_to_throw"] == 3.4
    assert "qb_time_to_throw" not in ctx["player"]
    assert flat(ctx)["qb_time_to_throw"] == 3.4
    assert ctx["sources"]["qb_time_to_throw"]["kind"] == "override"


@pytest.mark.parametrize("key, value", [
    ("opp_pressure_rate", 150), ("opp_pressure_rate", -0.1), ("ol_injuries", 7), ("ol_injuries", -1),
    ("opp_defense_rank", 40), ("opp_coverage_grade", 250), ("qb_time_to_throw", "fast"),
    ("qb_time_to_throw", float("nan")), ("qb_time_to_throw", float("inf")), ("qb_time_to_throw", True),
    ("proe", 1.0), ("vegas_spread", 80), ("not_a_real_key", 1.0),
])
def test_bad_overrides_are_dropped_with_a_warning(key, value):
    ctx = nmf.build_game_context("KC", "BUF", overrides={key: value, "opp_blitz_rate": 0.3})
    assert all(key not in ctx[name] for name in nmf.SECTIONS)
    assert any(key in w for w in ctx["warnings"])
    assert ctx["opponent_defense"]["opp_blitz_rate"] == 0.3       # the good override alongside it survives


def test_override_percent_and_fraction_readings():
    ctx = nmf.build_game_context("KC", "BUF", overrides={
        "opp_pressure_rate": 31, "opp_blitz_rate": "28%", "opp_coverage_grade": "85%", "proe": 3.2, "rroe": "-3.2%",
        "opp_stacked_box_rate": 0.22})
    assert ctx["opponent_defense"]["opp_pressure_rate"] == pytest.approx(0.31)
    assert ctx["opponent_defense"]["opp_blitz_rate"] == pytest.approx(0.28)
    assert ctx["opponent_defense"]["opp_coverage_grade"] == 85.0
    assert ctx["offense"]["proe"] == pytest.approx(0.032)
    assert ctx["offense"]["rroe"] == pytest.approx(-0.032)
    assert ctx["opponent_defense"]["opp_stacked_box_rate"] == 0.22
    assert "percentage" in ctx["sources"]["opp_pressure_rate"]["basis"]


def test_override_alias_is_canonicalised_and_beats_the_overlay(monkeypatch):
    full_world(monkeypatch, overlay=overlay_of(("Alpha Tackle", "OUT")), snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, overrides={"ol_injury_count": 3})
    assert ctx["game"]["ol_injuries"] == 3.0
    assert "ol_injury_count" not in ctx["game"]
    assert ctx["sources"]["ol_injuries"]["kind"] == "override"
    assert "supplied as 'ol_injury_count'" in ctx["sources"]["ol_injuries"]["basis"]


def test_model_output_keys_cannot_be_overridden():
    ctx = nmf.build_game_context("KC", "BUF", overrides={"qb_collapse_probability": 0.9, "trailing_script_prob": 0.8,
                                                         "collapse_probability": 0.7})
    assert all(key not in ctx[name] for name in nmf.SECTIONS
               for key in ("qb_collapse_probability", "trailing_script_prob", "collapse_probability"))
    assert sum("model output" in w for w in ctx["warnings"]) == 3


def test_non_mapping_overrides_are_ignored_not_raised():
    ctx = nmf.build_game_context("KC", "BUF", overrides=[("opp_pressure_rate", 0.4)])
    assert ctx["opponent_defense"] == {}
    assert any("overrides ignored" in w for w in ctx["warnings"])


def test_none_override_values_never_set_or_erase(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, overrides={"opp_pressure_rate": None})
    assert ctx["opponent_defense"]["opp_pressure_rate"] == 0.36
    assert ctx["sources"]["opp_pressure_rate"]["kind"] == "real"


def test_override_table_covers_every_model_input():
    modules = (qb_collapse_model, pass_game_collapse, run_game_collapse, explosion_model, explosive_play_model,
               coverage_mismatch_model, red_zone_funnel_model, game_script_model)
    missing = []
    for module in modules:
        keys = [factor.key for factor in module.FACTORS]
        keys += list(getattr(module, "CANONICAL_INPUTS", ())) + list(getattr(module, "NO_FREE_FEED_INPUTS", ()))
        keys += list(getattr(module, "_SPLIT_ALIASES", {}))
        for key in keys:
            if key not in nmf.OVERRIDE_SPECS and key not in nmf.MODEL_OUTPUT_KEYS:
                missing.append(key)
    assert missing == []
    # the direct signed funnel inputs of the red-zone model are accepted too
    assert {red_zone_funnel_model.DIRECT_PASS_KEY, red_zone_funnel_model.DIRECT_RUN_KEY} <= set(nmf.OVERRIDE_SPECS)
    for key, (section, kind, low, high) in nmf.OVERRIDE_SPECS.items():
        assert section in nmf.SECTIONS and kind in {"rate", "grade", "value", "fraction"} and low < high


def test_every_documented_alias_resolves_to_a_canonical_key():
    aliases = nmf._alias_map()
    assert aliases["ol_injury_count"] == "ol_injuries"
    assert aliases["pass_block_win_rate"] == "ol_pass_block_win_rate"
    assert "wr1_separation" not in aliases                       # canonical in explosion_model
    assert aliases["yprr_vs_man"] == "wr_yprr_vs_man"            # pass_game_collapse's split aliases
    for alias, canonical in aliases.items():
        assert canonical in nmf.OVERRIDE_SPECS or canonical in nmf.MODEL_OUTPUT_KEYS, (alias, canonical)


# ---------------------------------------------------------------------------------------------
# player and quarterback selection
# ---------------------------------------------------------------------------------------------
def test_player_matched_by_id(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, player={"player_id": RUNNER, "position": "RB"})
    assert ctx["player"]["rb_breakaway_rate"] == 0.05
    assert ctx["player_match"]["player_id"] == RUNNER
    assert ctx["player_match"]["how"] == "matched by id"


def test_player_matched_by_name_with_position_and_team(monkeypatch):
    full_world(monkeypatch)
    row = {"name": "alpha receiver", "team": "KC", "position": "WR"}
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, player=row)
    assert ctx["player"]["wr_separation"] == 3.3
    assert ctx["player_match"]["how"] == "matched by name"


def test_ambiguous_player_name_gets_no_player_features(monkeypatch):
    table = player_table()
    table["00-0040001"] = player_entry("Sam Smith", "KC", "WR", {"wr_separation": 2.5})
    table["00-0040002"] = player_entry("Sam Smith", "KC", "WR", {"wr_separation": 3.9})
    full_world(monkeypatch, players=table)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, player={"name": "Sam Smith", "position": "WR"})
    assert ctx["player"] == {}
    assert ctx["player_match"] is None
    assert any("more than one player" in w for w in ctx["warnings"])
    # the quarterback is unaffected
    assert ctx["qb"]["qb_time_to_throw"] == 2.70


def test_player_listed_with_another_team_is_still_found_but_flagged(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("BUF", "KC", season=SEASON, week=6,
                                 player={"name": "Alpha Receiver", "team": "BUF", "position": "WR"})
    assert ctx["player"]["wr_separation"] == 3.3
    assert any("listed with KC" in w for w in ctx["warnings"])


def test_unusable_player_rows_do_not_raise(monkeypatch):
    full_world(monkeypatch)
    for bad in ({}, {"position": "WR"}, "", 12, object()):
        ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, player=bad)
        assert ctx["player"] == {}


def test_default_quarterback_is_the_one_with_most_dropbacks(monkeypatch):
    table = player_table()
    table[BACKUP]["games"] = 9                    # more games than the starter, far fewer dropbacks
    full_world(monkeypatch, players=table)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["qb_player"]["player_id"] == MAHOMES
    assert ctx["qb_player"]["how"] == "most dropbacks on the team"
    assert ctx["qb"]["qb_time_to_throw"] == 2.70


def test_default_quarterback_is_chosen_within_the_offence_team(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("BUF", "KC", season=SEASON, week=6)
    assert ctx["qb_player"]["player_id"] == ALLEN
    assert ctx["qb"]["qb_time_to_throw"] == 2.9


def test_quarterback_named_by_name_or_id(monkeypatch):
    full_world(monkeypatch)
    by_name = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, qb="Spare Quarterback")
    by_id = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, qb={"player_id": BACKUP})
    for ctx in (by_name, by_id):
        assert ctx["qb"]["qb_time_to_throw"] == 3.2
        assert ctx["qb_player"]["player_id"] == BACKUP
        assert ctx["qb_player"]["how"] == "named by the caller"


def test_named_quarterback_not_found_omits_qb_and_drops_team_level_qb_keys(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, qb="Nobody Atall")
    assert ctx["qb"] == {}
    assert ctx["qb_player"] is None
    assert "qb_efficiency" not in ctx["offense"] and "qb_deep_attempt_rate" not in ctx["offense"]
    assert any("Nobody Atall" in w for w in ctx["warnings"])
    assert ctx["offense"]["team_epa_per_play"] == 0.10          # non-QB team keys untouched


def test_team_level_qb_keys_are_dropped_when_the_quarterback_is_not_the_primary(monkeypatch):
    table = player_table()
    table[BACKUP]["features"] = {"qb_time_to_throw": 3.2}       # no player-level qb_efficiency to take over
    table[BACKUP]["kinds"] = {"qb_time_to_throw": "real"}
    full_world(monkeypatch, players=table)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, qb={"player_id": BACKUP})
    assert "qb_efficiency" not in ctx["offense"] and "qb_deep_attempt_rate" not in ctx["offense"]
    assert any("describe P.Mahomes" in w for w in ctx["warnings"])
    # but they are kept for the primary quarterback
    keep = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, qb={"player_id": MAHOMES})
    assert keep["offense"]["qb_deep_attempt_rate"] == 0.12


def test_projected_quarterback_is_the_quarterback(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6,
                                 player={"player_id": BACKUP, "name": "Spare Quarterback", "position": "QB"},
                                 qb="Patrick Mahomes")
    assert ctx["qb_player"]["player_id"] == BACKUP
    assert ctx["qb_player"]["how"] == "the projected player"
    assert ctx["qb"] == ctx["player"] and ctx["qb"]["qb_time_to_throw"] == 3.2
    assert any("qb argument ignored" in w for w in ctx["warnings"])


def test_player_table_missing_leaves_qb_and_player_neutral(monkeypatch):
    full_world(monkeypatch, players={})
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, player={"name": "Alpha Receiver"})
    assert ctx["qb"] == {} and ctx["player"] == {}
    assert ctx["offense"]["team_epa_per_play"] == 0.10
    assert any("player features unavailable" in w for w in ctx["warnings"])


# ---------------------------------------------------------------------------------------------
# weeks and seasons
# ---------------------------------------------------------------------------------------------
def test_through_week_defaults_to_week_minus_one(monkeypatch):
    calls = full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["through_week"] == 5 and ctx["week"] == 6
    assert calls["team"] == [(SEASON, 5)] and calls["player"] == [(SEASON, 5)]


def test_explicit_through_week_is_respected(monkeypatch):
    calls = full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, through_week=3)
    assert ctx["through_week"] == 3
    assert calls["team"] == [(SEASON, 3)] and calls["player"] == [(SEASON, 3)]
    assert not any("look-ahead" in w for w in ctx["warnings"])


def test_through_week_not_before_week_warns_but_is_honoured(monkeypatch):
    calls = full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, through_week=6)
    assert calls["team"] == [(SEASON, 6)]
    assert any("look-ahead" in w for w in ctx["warnings"])


def test_no_week_means_latest_data(monkeypatch):
    calls = full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON)
    assert ctx["through_week"] is None and calls["team"] == [(SEASON, None)]


def test_week_one_means_nothing_precedes_it(monkeypatch):
    calls = full_world(monkeypatch, players={})
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=1)
    assert ctx["through_week"] == 0 and calls["team"] == [(SEASON, 0)]
    assert any("nothing precedes week 1" in w for w in ctx["warnings"])


@pytest.mark.parametrize("today, expected", [
    (dt.date(2026, 10, 12), 2026), (dt.date(2026, 9, 1), 2026), (dt.date(2027, 2, 20), 2026), (dt.date(2026, 3, 1), 2025),
])
def test_season_defaults_follow_the_team_feature_rule(monkeypatch, today, expected):
    monkeypatch.setattr(ntf, "_today", lambda: today)
    calls = full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF")
    assert ctx["season"] == expected
    assert calls["team"][0][0] == expected and calls["player"][0][0] == expected
    assert ntf._default_season() == expected


def test_invalid_season_and_week_fall_back_with_warnings(monkeypatch):
    calls = full_world(monkeypatch)
    ctx = nmf.build_game_context("KC", "BUF", season="soon", week="never", through_week=-4)
    assert ctx["season"] == SEASON and ctx["week"] is None and ctx["through_week"] is None
    assert calls["team"] == [(SEASON, None)]
    assert len(ctx["warnings"]) >= 3


def test_early_season_warning(monkeypatch):
    full_world(monkeypatch, teams=team_table(KC=team_row(KC_OFF, KC_DEF, games=3)))
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=4)
    assert any("early season" in w and "KC 3 g" in w for w in ctx["warnings"])
    full_world(monkeypatch)
    later = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert not any("early season" in w for w in later["warnings"])


# ---------------------------------------------------------------------------------------------
# offensive-line injuries
# ---------------------------------------------------------------------------------------------
def test_ol_injuries_from_the_overlay(monkeypatch):
    overlay = overlay_of(("Alpha Tackle", "OUT"), ("Bravo Guard", "QUESTIONABLE"), ("Echo Tackle", "Banana"),
                         ("Golf Tight", "OUT"), ("Foxtrot Reserve", "OUT"))
    full_world(monkeypatch, overlay=overlay, snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    # OUT = 1.0, QUESTIONABLE = 0.5; an unknown status, a non-OL and the sixth lineman are ignored
    assert ctx["game"]["ol_injuries"] == 1.5
    assert ctx["game"]["ol_health"] == pytest.approx(1 - 1.5 / 5)
    assert ctx["sources"]["ol_injuries"]["kind"] == "real"
    assert "Alpha Tackle OUT" in ctx["sources"]["ol_injuries"]["basis"]
    assert "Bravo Guard QUESTIONABLE" in ctx["sources"]["ol_injuries"]["basis"]
    assert "Foxtrot" not in ctx["sources"]["ol_injuries"]["basis"].split("among")[0]
    assert ctx["sources"]["ol_health"]["kind"] == "derived"
    assert flat(ctx)["ol_injuries"] == 1.5


@pytest.mark.parametrize("status, weight", [("OUT", 1.0), ("DOUBTFUL", 1.0), ("HOLDOUT", 1.0), ("SUSPENDED", 1.0),
                                            ("QUESTIONABLE", 0.5), ("Out", 1.0), ("gtd", 0.5)])
def test_ol_status_weights(monkeypatch, status, weight):
    full_world(monkeypatch, overlay=overlay_of(("Charlie Center", status)), snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["game"]["ol_injuries"] == weight


def test_empty_overlay_leaves_ol_injuries_neutral(monkeypatch):
    full_world(monkeypatch, overlay={}, snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert "ol_injuries" not in ctx["game"] and "ol_health" not in ctx["game"]
    assert ctx["sources"]["ol_injuries"]["kind"] == "default"
    assert "no player-status file" in ctx["sources"]["ol_injuries"]["basis"]
    assert any("no player-status file" in w for w in ctx["warnings"])
    assert "ol_injuries" not in flat(ctx)


def test_overlay_without_any_row_for_the_teams_ol_is_neutral_not_zero(monkeypatch):
    overlay = overlay_of(("Some Receiver", "OUT"), ("Other Tackle", "OUT"), ("Golf Tight", "QUESTIONABLE"))
    full_world(monkeypatch, overlay=overlay, snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert "ol_injuries" not in ctx["game"]                    # a team with no flagged OL is not "0 out"
    assert "no player-status file row flags" in ctx["sources"]["ol_injuries"]["basis"]


def test_overlay_records_keyed_by_gsis_id_match_on_the_name_field(monkeypatch):
    overlay = {"00-0012345": {"status": "OUT", "last_updated": "2026-10-10T00:00:00Z", "name": "Delta Guard"}}
    full_world(monkeypatch, overlay=overlay, snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["game"]["ol_injuries"] == 1.0


def test_only_the_teams_own_top_five_ol_are_considered(monkeypatch):
    overlay = overlay_of(("Other Tackle", "OUT"))              # a BUF lineman
    full_world(monkeypatch, overlay=overlay, snaps=ol_snaps())
    kc = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert "ol_injuries" not in kc["game"]


def test_ol_injuries_not_applied_to_a_game_already_played(monkeypatch):
    full_world(monkeypatch, overlay=overlay_of(("Alpha Tackle", "OUT")), snaps=ol_snaps())
    # snap data runs to week 5: a week-3 projection (through week 2) is a past game, the overlay is "now"
    past = nmf.build_game_context("KC", "BUF", season=SEASON, week=3)
    assert "ol_injuries" not in past["game"]
    assert "not applied" in past["sources"]["ol_injuries"]["basis"]
    assert "week 2" in past["sources"]["ol_injuries"]["basis"]
    # the upcoming week 6 (through week 5) and "latest data" both apply it
    assert nmf.build_game_context("KC", "BUF", season=SEASON, week=6)["game"]["ol_injuries"] == 1.0
    assert nmf.build_game_context("KC", "BUF", season=SEASON)["game"]["ol_injuries"] == 1.0


def test_top_five_ol_is_ranked_by_snaps_not_by_name_or_listing_order(monkeypatch):
    # a sixth lineman with more total snaps than Echo (54 x 5 = 270) displaces him from the top five
    extra = [{"team": "KC", "week": w, "position": "G", "player": "Zulu Backup", "pfr_player_id": "PFR-Z",
              "offense_snaps": 56} for w in (1, 2, 3, 4, 5)]
    full_world(monkeypatch, overlay=overlay_of(("Zulu Backup", "OUT"), ("Echo Tackle", "OUT")),
               snaps=ol_snaps(extra_rows=extra))
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["game"]["ol_injuries"] == 1.0                  # Zulu counts, displaced Echo does not
    assert "Zulu Backup OUT" in ctx["sources"]["ol_injuries"]["basis"]
    assert "Echo Tackle OUT" not in ctx["sources"]["ol_injuries"]["basis"]


def test_ol_injuries_not_applied_to_a_past_season(monkeypatch):
    full_world(monkeypatch, overlay=overlay_of(("Alpha Tackle", "OUT")), snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=2025, week=6)
    assert "ol_injuries" not in ctx["game"]
    assert "not applied" in ctx["sources"]["ol_injuries"]["basis"]
    assert "2025" in ctx["sources"]["ol_injuries"]["basis"]


def test_stale_player_status_file_is_not_applied(monkeypatch):
    old = overlay_of(("Alpha Tackle", "OUT"), stamp="2026-09-06T17:35:42Z")      # 36 days before TODAY
    full_world(monkeypatch, overlay=old, snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert "ol_injuries" not in ctx["game"]
    assert "no player-status file fresh enough" in ctx["sources"]["ol_injuries"]["basis"]
    assert "36 days" in ctx["sources"]["ol_injuries"]["basis"]
    fresh = overlay_of(("Alpha Tackle", "OUT"), stamp="2026-10-06T00:00:00Z")    # 6 days: inside the limit
    full_world(monkeypatch, overlay=fresh, snaps=ol_snaps())
    assert nmf.build_game_context("KC", "BUF", season=SEASON, week=6)["game"]["ol_injuries"] == 1.0


def test_ol_injuries_override_wins_and_health_follows_it(monkeypatch):
    full_world(monkeypatch, overlay=overlay_of(("Alpha Tackle", "OUT")), snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, overrides={"ol_injuries": 3})
    assert ctx["game"]["ol_injuries"] == 3.0
    assert ctx["game"]["ol_health"] == pytest.approx(0.4)
    assert ctx["sources"]["ol_health"]["kind"] == "derived"
    assert not any("left neutral" in w and "ol_injuries" in w for w in ctx["warnings"])


def test_ol_injuries_override_with_no_overlay_still_derives_health(monkeypatch):
    full_world(monkeypatch, overlay={}, snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, overrides={"ol_injuries": 2.0})
    assert ctx["game"]["ol_injuries"] == 2.0 and ctx["game"]["ol_health"] == pytest.approx(0.6)
    assert ctx["sources"]["ol_injuries"]["kind"] == "override"


def test_ol_health_override_wins_over_the_derived_value(monkeypatch):
    full_world(monkeypatch, overlay=overlay_of(("Alpha Tackle", "OUT")), snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, overrides={"ol_health": "55%"})
    assert ctx["game"]["ol_health"] == pytest.approx(0.55)
    assert ctx["sources"]["ol_health"]["kind"] == "override"
    assert ctx["game"]["ol_injuries"] == 1.0                 # untouched


def test_ol_health_is_bounded(monkeypatch):
    all_out = overlay_of(*[(name, "OUT") for name, _, _ in OL_NAMES[:5]])
    full_world(monkeypatch, overlay=all_out, snaps=ol_snaps())
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["game"]["ol_injuries"] == 5.0 and ctx["game"]["ol_health"] == 0.0


def test_ol_injury_count_through_the_real_overlay_and_snap_loaders(monkeypatch, tmp_path):
    """No fake overlay or snap frame: the file on disk and ``nfl_team_features``' snap loader/prepare step."""
    from fantasy import player_status

    path = tmp_path / "player_status.json"
    path.write_text(json.dumps({
        "nm:alphatackle": {"status": "OUT", "last_updated": "2026-10-11T08:00:00Z", "name": "Alpha Tackle"},
        "00-0099": {"status": "QUESTIONABLE", "last_updated": "2026-10-11T08:00:00Z", "name": "Delta Guard"},
    }), encoding="utf-8")
    monkeypatch.setattr(player_status, "STATUS_PATH", path)
    player_status.clear_cache()
    raw = ol_snaps(team="LAR")                      # the loader spells the Rams LAR; the prepare step folds to LA
    raw["game_type"] = "REG"
    monkeypatch.setattr(ntf, "_load_snaps", lambda season: raw.copy())
    monkeypatch.setattr(nmf, "_status_overlay", _REAL_STATUS_OVERLAY)
    monkeypatch.setattr(nmf, "_snap_frame", _REAL_SNAP_FRAME)
    count, basis = _REAL_OL_INJURY_COUNT("LA", SEASON, 5)
    assert count == 1.5
    assert "2026-10-11" in basis and "Alpha Tackle OUT" in basis
    ctx = nmf.build_game_context("LAR", "BUF", season=SEASON, week=6)
    assert ctx["team"] == "LA" and ctx["game"]["ol_injuries"] == 1.5


# ---------------------------------------------------------------------------------------------
# fail-soft, identity, shape
# ---------------------------------------------------------------------------------------------
def test_every_loader_empty_is_neutral_with_explanations():
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, player={"name": "Alpha Receiver"})
    assert ctx["has_features"] is False
    for name in nmf.SECTIONS:
        assert ctx[name] == {}
    assert ctx["team"] == "KC" and ctx["opponent"] == "BUF" and ctx["season"] == SEASON
    joined = " | ".join(ctx["warnings"])
    assert "team features unavailable" in joined
    assert "player features unavailable" in joined
    assert "market lines not set" in joined
    assert "no player-status file" in joined
    assert pipe.team_signals(ctx)["has_signal"] is False


def test_loaders_that_raise_are_contained(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(nmf, "_team_table", boom)
    monkeypatch.setattr(nmf, "_player_table", boom)
    monkeypatch.setattr(nmf, "_load_schedule", boom)
    monkeypatch.setattr(nmf, "_ol_injury_count", boom)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, vegas_spread=-3, overrides={"opp_blitz_rate": 0.3})
    assert ctx["has_features"] is False                       # only caller numbers got in
    assert ctx["opponent_defense"] == {"opp_blitz_rate": 0.3}
    assert ctx["game"]["vegas_spread"] == -3.0
    joined = " | ".join(ctx["warnings"])
    assert "team feature loader failed" in joined and "provider down" in joined
    assert "player feature loader failed" in joined


def test_non_mapping_loader_results_are_neutral(monkeypatch):
    monkeypatch.setattr(nmf, "_team_table", lambda season, through_week: ["not", "a", "table"])
    monkeypatch.setattr(nmf, "_player_table", lambda season, through_week: None)
    monkeypatch.setattr(nmf, "_load_schedule", lambda season: {"not": "a frame"})
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert ctx["has_features"] is False


def test_unexpected_internal_error_returns_a_neutral_context(monkeypatch):
    monkeypatch.setattr(nmf, "_build", lambda *a, **k: (_ for _ in ()).throw(ValueError("bug")))
    ctx = nmf.build_game_context("KC", "BUF")
    assert ctx["has_features"] is False and any("bug" in w for w in ctx["warnings"])


def test_team_spellings_fold_to_nflverse_abbreviations(monkeypatch):
    teams = team_table(LA=team_row({"team_epa_per_play": 0.07}, {"opp_pressure_rate": 0.31}, lead="M.Stafford"))
    sched = schedule(dict(GAME_KC_BUF, home_team="LA", away_team="KC", spread_line=2.0, game_id="g"))
    calls = full_world(monkeypatch, teams=teams, sched=sched)
    ctx = nmf.build_game_context("LAR", " kc ", season=SEASON, week=6)
    assert ctx["team"] == "LA" and ctx["opponent"] == "KC"
    assert ctx["offense"]["team_epa_per_play"] == 0.07
    assert ctx["opponent_defense"]["opp_pressure_rate"] == 0.25
    assert ctx["game"]["vegas_spread"] == -2.0 and ctx["is_home"] is True
    assert calls["team"] == [(SEASON, 5)]
    other = nmf.build_game_context("STL", "KC", season=SEASON, week=6)
    assert other["team"] == "LA"


def test_unrecognised_or_identical_teams_are_neutral_with_a_warning(monkeypatch):
    calls = full_world(monkeypatch)
    bad = nmf.build_game_context("XYZ", "BUF", season=SEASON, week=6)
    assert bad["has_features"] is False and bad["offense"] == {} and bad["opponent_defense"] == {}
    assert any("unrecognised team 'XYZ'" in w for w in bad["warnings"])
    same = nmf.build_game_context("KC", "KC", season=SEASON, week=6)
    assert same["has_features"] is False and any("both KC" in w for w in same["warnings"])
    assert calls["team"] == []                                 # no loader call for an unusable pairing


def test_unusable_teams_still_accept_caller_numbers():
    ctx = nmf.build_game_context(None, "BUF", overrides={"opp_pressure_rate": 0.4}, vegas_total=44)
    assert ctx["opponent_defense"]["opp_pressure_rate"] == 0.4 and ctx["game"]["vegas_total"] == 44.0
    assert ctx["has_features"] is False


def test_neutral_context_shape():
    ctx = nmf.neutral_context()
    assert set(ctx) >= {"team", "opponent", "season", "week", "through_week", "is_home", "offense", "opponent_defense",
                        "qb", "player", "game", "sources", "warnings", "has_features"}
    assert ctx["has_features"] is False and ctx["warnings"] == [] and ctx["sources"] == {}
    for name in nmf.SECTIONS:
        assert ctx[name] == {}
    ctx["offense"]["x"] = 1
    assert nmf.neutral_context()["offense"] == {}               # a fresh object every call


def test_neutral_context_is_an_exact_noop_in_the_pipeline():
    line = {"passing_yards": 250.0, "completion_pct": 0.65, "passing_tds": 1.8, "interceptions": 0.7, "attempts": 35.0,
            "team_points": 24.0}
    applied = pipe.project_with_pack(line, position="QB", context=nmf.neutral_context())
    assert applied["line"] == line and applied["neutral"] is True


def test_has_features_needs_a_measured_feature():
    only_override = nmf.build_game_context("KC", "BUF", overrides={"opp_pressure_rate": 0.4}, vegas_spread=-3)
    assert only_override["has_features"] is False
    assert only_override["sources"]["opp_pressure_rate"]["kind"] == "override"


def test_build_is_deterministic(monkeypatch):
    full_world(monkeypatch, overlay=overlay_of(("Alpha Tackle", "OUT")), snaps=ol_snaps())
    kwargs = dict(season=SEASON, week=6, player={"name": "Alpha Receiver", "position": "WR"},
                  overrides={"opp_blitz_rate": 0.33, "ol_health": 0.8})
    first = nmf.build_game_context("KC", "BUF", **kwargs)
    second = nmf.build_game_context("KC", "BUF", **kwargs)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_inputs_are_not_mutated(monkeypatch):
    full_world(monkeypatch)
    overrides = {"opp_pressure_rate": 31, "ol_injury_count": 1, "bogus": "x"}
    player = {"player_id": RECEIVER, "name": "Alpha Receiver", "team": "KC", "position": "WR"}
    qb = {"player_id": MAHOMES, "name": "Patrick Mahomes"}
    before = copy.deepcopy((overrides, player, qb))
    nmf.build_game_context("KC", "BUF", season=SEASON, week=6, player=player, qb=qb, overrides=overrides)
    assert (overrides, player, qb) == before


def test_returned_context_does_not_alias_the_loader_tables(monkeypatch):
    tables = {"teams": team_table(), "players": player_table()}
    monkeypatch.setattr(nmf, "_team_table", lambda season, through_week: tables["teams"])
    monkeypatch.setattr(nmf, "_player_table", lambda season, through_week: tables["players"])
    snapshot = copy.deepcopy(tables)
    ctx = nmf.build_game_context("KC", "BUF", season=SEASON, week=6, player={"player_id": RUNNER})
    ctx["offense"]["team_epa_per_play"] = 99.0
    ctx["qb"]["qb_time_to_throw"] = 99.0
    assert tables == snapshot


def test_context_for_player_uses_the_pool_row(monkeypatch):
    calls = full_world(monkeypatch)
    row = {"player_id": RECEIVER, "name": "Alpha Receiver", "team": "KC", "position": "WR", "projection": 12.3}
    ctx = nmf.context_for_player(row, "BUF", season=SEASON, week=6)
    assert ctx["team"] == "KC" and ctx["opponent"] == "BUF"
    assert ctx["player"]["wr_separation"] == 3.3
    assert ctx["game"]["vegas_spread"] == -3.5
    assert row == {"player_id": RECEIVER, "name": "Alpha Receiver", "team": "KC", "position": "WR", "projection": 12.3}
    assert calls["team"] == [(SEASON, 5)]
    assert nmf.context_for_player(row, "BUF", team="BUF", season=SEASON, week=6)["team"] == "BUF"


def test_context_for_player_without_a_team_is_neutral(monkeypatch):
    full_world(monkeypatch)
    ctx = nmf.context_for_player({"player_id": RECEIVER, "name": "Alpha Receiver"}, "BUF", season=SEASON, week=6)
    assert ctx["has_features"] is False and any("unrecognised team" in w for w in ctx["warnings"])


# ---------------------------------------------------------------------------------------------
# cache and I/O boundary
# ---------------------------------------------------------------------------------------------
def test_schedule_is_ttl_cached_fails_soft_and_clear_cache_refreshes(monkeypatch):
    seen = {"calls": 0, "fail": False}
    frame = pd.DataFrame([GAME_KC_BUF])

    def load_schedules(seasons):
        seen["calls"] += 1
        if seen["fail"]:
            raise OSError("offline")
        return frame

    monkeypatch.setitem(sys.modules, "nflreadpy", types.SimpleNamespace(load_schedules=load_schedules))
    first = _REAL_LOAD_SCHEDULE(2026)
    _REAL_LOAD_SCHEDULE(2026)
    assert seen["calls"] == 1 and not first.empty
    nmf.clear_cache()
    _REAL_LOAD_SCHEDULE(2026)
    assert seen["calls"] == 2
    nmf.clear_cache()
    seen["fail"] = True
    failed = _REAL_LOAD_SCHEDULE(2026)
    assert isinstance(failed, pd.DataFrame) and failed.empty      # never raises


def test_clear_cache_also_clears_the_loader_caches(monkeypatch):
    hit = []
    monkeypatch.setattr(ntf, "clear_cache", lambda: hit.append("team"))
    monkeypatch.setattr(npf, "clear_cache", lambda: hit.append("player"))
    nmf.clear_cache()
    assert hit == ["team", "player"]


def test_module_imports_no_network_library():
    source = Path(nmf.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4", "urllib"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned)
    lowered = source.lower()
    for term in ("draftkings.com", "fanduel.com", "betmgm.com", "caesars.com", "espnbet.com"):
        assert term not in lowered


# ---------------------------------------------------------------------------------------------
# integration with the pipeline
# ---------------------------------------------------------------------------------------------
BAD_PROTECTION_TEAMS = {
    "KC": team_row({"ol_pass_block_win_rate": 0.60, "ol_continuity": 0.62, "ol_run_block_win_rate": 0.64,
                    "team_epa_per_play": -0.05}, {}, lead="P.Mahomes"),
    "BUF": team_row({"team_epa_per_play": 0.15},
                    {"opp_pressure_rate": 0.40, "opp_blitz_rate": 0.36, "opp_pass_rush_win_rate": 0.38,
                     "opp_simulated_pressure_success": 0.34, "opp_def_epa_per_play": 0.0}, lead="J.Allen"),
}
GOOD_PROTECTION_TEAMS = {
    "KC": team_row({"ol_pass_block_win_rate": 0.80, "ol_continuity": 0.95, "ol_run_block_win_rate": 0.78,
                    "team_epa_per_play": 0.15}, {}, lead="P.Mahomes"),
    "BUF": team_row({"team_epa_per_play": -0.05},
                    {"opp_pressure_rate": 0.24, "opp_blitz_rate": 0.18, "opp_pass_rush_win_rate": 0.24,
                     "opp_simulated_pressure_success": 0.20, "opp_def_epa_per_play": 0.0}, lead="J.Allen"),
}
BAD_QB = {MAHOMES: player_entry("Patrick Mahomes", "KC", "QB", {
    "qb_time_to_throw": 3.1, "qb_pressure_epa": -0.8, "qb_sack_avoidance": 0.90, "qb_release_time": 2.6,
    "qb_scramble_rate": 0.03}, games=5, dropbacks=190)}
GOOD_QB = {MAHOMES: player_entry("Patrick Mahomes", "KC", "QB", {
    "qb_time_to_throw": 2.5, "qb_pressure_epa": -0.2, "qb_sack_avoidance": 0.96, "qb_release_time": 2.1,
    "qb_scramble_rate": 0.08}, games=5, dropbacks=190)}


def test_context_flows_through_the_pipeline_with_non_neutral_probabilities(monkeypatch):
    full_world(monkeypatch, teams=copy.deepcopy(BAD_PROTECTION_TEAMS), players=copy.deepcopy(BAD_QB),
               overlay=overlay_of(("Alpha Tackle", "OUT"), ("Bravo Guard", "OUT")), snaps=ol_snaps())
    bad = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    assert bad["has_features"] is True and bad["game"]["ol_injuries"] == 2.0
    full_world(monkeypatch, teams=copy.deepcopy(GOOD_PROTECTION_TEAMS), players=copy.deepcopy(GOOD_QB),
               overlay=overlay_of(), snaps=ol_snaps())
    good = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)

    bad_signals = pipe.compute_game_signals(bad, position="QB")
    good_signals = pipe.compute_game_signals(good, position="QB")
    neutral_signals = pipe.compute_game_signals(nmf.neutral_context(), position="QB")

    reference = neutral_signals["probabilities"]["collapse_probability"]
    assert bad_signals["has_signal"] is True
    assert bad_signals["probabilities"]["collapse_probability"] > reference + 0.05
    assert good_signals["probabilities"]["collapse_probability"] < reference
    assert bad_signals["probabilities"]["collapse_probability"] > good_signals["probabilities"]["collapse_probability"]
    assert bad_signals["probabilities"]["pass_game_collapse"] > good_signals["probabilities"]["pass_game_collapse"]
    # the lines reached the script model
    assert bad_signals["inputs"]["vegas_spread"] == -3.5 and bad_signals["inputs"]["vegas_total"] == 47.5
    assert "vegas_spread" in bad_signals["results"]["game_script"]["inputs_used"]
    assert bad_signals["results"]["game_script"]["z_score"] != neutral_signals["results"]["game_script"]["z_score"]
    # team-level read (what the parlay layer correlates)
    team = pipe.team_signals(bad)
    assert team["has_signal"] is True and team["vegas_spread"] == -3.5


def test_bad_protection_context_lowers_a_quarterback_line(monkeypatch):
    full_world(monkeypatch, teams=copy.deepcopy(BAD_PROTECTION_TEAMS), players=copy.deepcopy(BAD_QB),
               overlay=overlay_of(("Alpha Tackle", "OUT"), ("Bravo Guard", "OUT")), snaps=ol_snaps())
    bad = nmf.build_game_context("KC", "BUF", season=SEASON, week=6)
    line = {"passing_yards": 270.0, "completion_pct": 0.66, "passing_tds": 1.9, "interceptions": 0.7, "attempts": 36.0,
            "team_points": 25.0}
    adjusted = pipe.project_with_pack(line, position="QB", context=bad)
    assert adjusted["neutral"] is False
    assert adjusted["line"]["passing_yards"] < line["passing_yards"]
    assert adjusted["line"]["interceptions"] > line["interceptions"] * 0.99 or adjusted["line"]["passing_tds"] < line["passing_tds"]
    assert line["passing_yards"] == 270.0                       # the input line is untouched


def test_player_row_flows_through_the_pipeline(monkeypatch):
    full_world(monkeypatch, teams=copy.deepcopy(BAD_PROTECTION_TEAMS),
               players={**copy.deepcopy(BAD_QB), RECEIVER: player_entry("Alpha Receiver", "KC", "WR",
                                                                         {"wr_separation": 2.3, "wr_deep_route_rate": 0.1})})
    ctx = nmf.context_for_player({"player_id": RECEIVER, "name": "Alpha Receiver", "team": "KC", "position": "WR"},
                                 "BUF", season=SEASON, week=6)
    signals = pipe.compute_game_signals(ctx, position="WR")
    assert signals["inputs"]["wr_separation"] == 2.3
    assert signals["has_signal"] is True
    line = {"receiving_yards": 70.0, "receiving_tds": 0.5, "receptions": 5.5, "targets": 8.0}
    assert pipe.project_with_pack(line, position="WR", context=ctx)["neutral"] is False
