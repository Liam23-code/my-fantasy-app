"""Tests for the player-level feature loader of the NFL model upgrade pack.

Everything is offline: the module's private ``_load_*`` boundary is replaced with
synthetic frames (a ten-team "world" generated from a fixed seed, plus small
hand-built frames whose answers are worked out by hand in the assertions).
"""
from __future__ import annotations

import ast
import copy
import logging
import math
import sys
import types
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules import nfl_player_features as pf

LOADERS = (
    "_load_pbp",
    "_load_ngs",
    "_load_pfr",
    "_load_participation",
    "_load_rosters",
    "_load_players",
    "_load_snaps",
)
#: the real boundary functions, captured before any test patches them
ORIGINAL_LOADERS = {name: getattr(pf, name) for name in LOADERS}

SEASON = 2025
N_TEAMS = 10
ALL_KEYS = tuple(pf.MIN_VOLUME)
FIRST = ["Aaron", "Brian", "Calvin", "Derek", "Edgar", "Felix", "Grant", "Hugo", "Isaac", "Jonah"]
SURNAME = {"QB": "Quarterman", "WR1": "Wideone", "WR2": "Widetwo", "TE": "Tightend", "RB": "Backson"}
ROLE_POSITION = {"QB": "QB", "WR1": "WR", "WR2": "WR", "TE": "TE", "RB": "RB"}
ROLE_NUM = {"QB": 1, "WR1": 2, "WR2": 3, "TE": 4, "RB": 5}


def pid(role: str, i: int) -> str:
    return f"00-{ROLE_NUM[role]}0{i:05d}"


def pfr(role: str, i: int) -> str:
    return f"Pfr{role}{i:02d}"


def full_name(role: str, i: int) -> str:
    return f"{FIRST[i]} {SURNAME[role]}"


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------
def pbp_row(game_id: str, play_id: int, week: int, team: str, play_type: str, season: int = SEASON, **kw) -> dict:
    row = dict(
        game_id=game_id, play_id=float(play_id), season=season, season_type="REG", week=week, posteam=team,
        play_type=play_type, qb_dropback=0, qb_scramble=0, sack=0, complete_pass=0, air_yards=np.nan,
        yards_gained=0.0, epa=0.0, yardline_100=50.0, pass_touchdown=0, rush_touchdown=0, two_point_attempt=0,
        passer_player_id=None, receiver_player_id=None, rusher_player_id=None, id=None,
    )
    row.update(kw)
    return row


class World:
    """One season of synthetic data in the shape of every nflverse feed the module reads."""

    def __init__(self, season, pbp, part, ngs_pass, ngs_rec, pfr_rush, pfr_rec, rosters, players, snaps):
        self.season = season
        self.pbp, self.part = pbp, part
        self.ngs_pass, self.ngs_rec = ngs_pass, ngs_rec
        self.pfr_rush, self.pfr_rec = pfr_rush, pfr_rec
        self.rosters, self.players, self.snaps = rosters, players, snaps


def make_world(weeks=range(1, 7), *, season=SEASON, seed=11, poison_from=None, unknown_te_pfr=3) -> World:
    """Ten teams, one QB / two WR / one TE / one RB each, ``30 + 2i`` dropbacks a week for team ``i``.

    Team ``i`` has a slower QB (``time_to_throw``), a bigger-separation WR1, more yards after contact for its
    RB, more breakaways and a higher sack rate the larger ``i`` is, so ordering assertions are meaningful.
    From ``poison_from`` on, every number is absurd (a leak would be unmistakable).
    """
    rng = np.random.default_rng(seed)
    pbp, part, ngs_p, ngs_r, rush, snaps = [], [], [], [], [], []
    rec_air: dict[str, list[float]] = {}
    fillers = ";".join(f"00-99{k:05d}" for k in range(6))
    for week in weeks:
        poison = poison_from is not None and week >= poison_from
        for i in range(N_TEAMS):
            team = f"T{i:02d}"
            gid = f"{season}_{week:02d}_{team}_ZZZ"
            qb, wr1, wr2, te, rb = (pid(r, i) for r in ("QB", "WR1", "WR2", "TE", "RB"))
            recvs, probs = [wr1, wr2, te, rb], [0.40, 0.28, 0.20, 0.12]
            n_db = 30 + 2 * i
            sack_p = 0.5 if poison else 0.04 + 0.004 * i
            play, attempts, ttt_sum = 0, 0, 0.0
            tgts: Counter = Counter()
            for _ in range(n_db):
                play += 1
                roll = rng.random()
                pressured = bool(rng.random() < (0.5 if poison else 0.28))
                cov = str(rng.choice(["MAN_COVERAGE", "ZONE_COVERAGE", ""], p=[0.3, 0.6, 0.1]))
                yl = float(rng.integers(1, 99))
                field = ";".join([qb, wr1, wr2, te, rb, fillers]) if rng.random() < 0.85 else ";".join([qb, wr1, te, rb, fillers])
                ttt = np.nan
                if roll < sack_p:
                    pressured = True
                    pbp.append(pbp_row(gid, play, week, team, "pass", season, qb_dropback=1, sack=1, passer_player_id=qb,
                                       yards_gained=-7.0, epa=-6.0 if poison else -2.0, yardline_100=yl))
                elif roll < sack_p + 0.04:
                    pbp.append(pbp_row(gid, play, week, team, "run", season, qb_dropback=1, qb_scramble=1, rusher_player_id=qb,
                                       yards_gained=6.0, epa=0.3, yardline_100=yl, rush_touchdown=int(yl <= 6)))
                else:
                    air = 40.0 if poison else float(rng.integers(-2, 36))
                    recv = None if rng.random() < 0.06 else str(rng.choice(recvs, p=probs))
                    complete = int(rng.random() < 0.64)
                    td = int(complete and yl <= 20 and rng.random() < 0.3)
                    epa = -6.0 if poison else float(rng.normal(0.02 * i - 0.1, 1.0))
                    pbp.append(pbp_row(gid, play, week, team, "pass", season, qb_dropback=1, complete_pass=complete,
                                       passer_player_id=qb, receiver_player_id=recv, air_yards=air,
                                       yards_gained=float(air + rng.integers(0, 8)) if complete else 0.0,
                                       epa=epa, yardline_100=yl, pass_touchdown=td))
                    ttt = 6.0 if poison else float(2.3 + 0.05 * i + rng.normal(0, 0.15))
                    attempts += 1
                    ttt_sum += ttt
                    if recv is not None:
                        tgts[recv] += 1
                        rec_air.setdefault(recv, []).append(air)
                part.append(
                    dict(nflverse_game_id=gid, play_id=float(play), time_to_throw=ttt, was_pressure=pressured,
                         defense_man_zone_type=cov, offense_players=field)
                )
            brk_p = 0.5 if poison else 0.04 + 0.004 * i
            carries = 0
            for _ in range(20):
                play += 1
                yl = float(rng.integers(1, 99))
                yards = float(15 + rng.integers(0, 20)) if rng.random() < brk_p else float(rng.integers(-1, 9))
                pbp.append(pbp_row(gid, play, week, team, "run", season, rusher_player_id=rb, yards_gained=yards,
                                   epa=0.0, yardline_100=yl, rush_touchdown=int(yl <= 20 and rng.random() < 0.15)))
                carries += 1
            if attempts:
                ngs_p.append(
                    dict(player_gsis_id=qb, player_display_name=full_name("QB", i), player_position="QB", season=season,
                         season_type="REG", week=week, avg_time_to_throw=ttt_sum / attempts, attempts=attempts)
                )
            for role, rid in (("WR1", wr1), ("WR2", wr2), ("TE", te)):
                if tgts[rid]:
                    base = {"WR1": 2.0, "WR2": 1.8, "TE": 1.5}[role] + 0.15 * i
                    sep = 9.0 if poison else float(base + rng.normal(0, 0.2))
                    ngs_r.append(
                        dict(player_gsis_id=rid, player_display_name=full_name(role, i), player_position=ROLE_POSITION[role],
                             season=season, season_type="REG", week=week, avg_separation=sep, targets=int(tgts[rid]))
                    )
            yac = 9.0 if poison else float(1.5 + 0.1 * i + rng.normal(0, 0.1))
            rush.append(
                dict(game_id=gid, season=season, week=week, game_type="REG", team=team, pfr_player_id=pfr("RB", i),
                     carries=carries, rushing_yards_after_contact=yac * carries)
            )
            for role in ("QB", "WR1", "WR2", "TE", "RB"):
                snaps.append(
                    dict(season=season, game_type="REG", week=week, pfr_player_id=pfr(role, i), team=team, offense_snaps=60)
                )
        rush.append(
            dict(game_id=f"{season}_{week:02d}_GHOST", season=season, week=week, game_type="REG", team="T00",
                 pfr_player_id="PfrGhost00", carries=40, rushing_yards_after_contact=80.0)
        )
    roster_rows, player_rows = [], []
    for i in range(N_TEAMS):
        for role in ("QB", "WR1", "WR2", "TE", "RB"):
            position = "HB" if (role == "RB" and i == 0) else ROLE_POSITION[role]
            player_rows.append(
                dict(gsis_id=pid(role, i), pfr_id=pfr(role, i), display_name=full_name(role, i), position=ROLE_POSITION[role],
                     latest_team=f"T{i:02d}")
            )
            for week in weeks:
                roster_rows.append(
                    dict(season=season, team=f"T{i:02d}", position=position, full_name=full_name(role, i), gsis_id=pid(role, i),
                         pfr_id=pfr(role, i), week=week, game_type="REG")
                )
    rec_rows = []
    for i in range(N_TEAMS):
        for role in ("WR1", "WR2", "TE"):
            air = rec_air.get(pid(role, i), [])
            if not air:
                continue
            key = "PfrUnknown" if (role == "TE" and i == unknown_te_pfr) else pfr(role, i)
            rec_rows.append(dict(season=season, player=full_name(role, i), pfr_id=key, tm=f"T{i:02d}", pos=ROLE_POSITION[role],
                                 tgt=len(air), adot=float(np.mean(air))))
    def with_season_rows(rows, vcol, wcol):
        """Append the week-0 row NGS publishes: the attempt / target weighted mean of the weekly rows."""
        frame = pd.DataFrame(rows)
        extra = []
        for _, grp in frame.groupby("player_gsis_id", sort=False):
            row = grp.iloc[0].to_dict()
            row.update({"week": 0, vcol: float((grp[vcol] * grp[wcol]).sum() / grp[wcol].sum()), wcol: int(grp[wcol].sum())})
            extra.append(row)
        return pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)

    ngs_pass = with_season_rows(ngs_p, "avg_time_to_throw", "attempts")
    ngs_rec = with_season_rows(ngs_r, "avg_separation", "targets")
    return World(
        season, pd.DataFrame(pbp), pd.DataFrame(part), ngs_pass, ngs_rec, pd.DataFrame(rush), pd.DataFrame(rec_rows),
        pd.DataFrame(roster_rows), pd.DataFrame(player_rows), pd.DataFrame(snaps),
    )


def install(monkeypatch, *worlds: World, calls: Counter | None = None, participation=None, skip=()) -> Counter:
    """Patch the module's I/O boundary with ``worlds`` (by season). ``skip`` names loaders that raise."""
    calls = calls if calls is not None else Counter()
    by_season = {w.season: w for w in worlds}
    part_seasons = set(by_season) if participation is None else set(participation)

    def frame(maker):
        def inner(*args):
            season = args[0] if args else None
            out = maker(by_season.get(season), season)
            return out.copy() if isinstance(out, pd.DataFrame) else out

        return inner

    makers = {
        "_load_pbp": lambda w, s: w.pbp if w is not None else pd.DataFrame(),
        "_load_participation": lambda w, s: w.part if (w is not None and s in part_seasons) else pd.DataFrame(),
        "_load_rosters": lambda w, s: w.rosters if w is not None else pd.DataFrame(),
        "_load_snaps": lambda w, s: w.snaps if w is not None else pd.DataFrame(),
    }

    def make(name, fn):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            if name in skip:
                raise RuntimeError(f"{name} unavailable (test)")
            out = fn(*args, **kwargs)
            return out.copy() if isinstance(out, pd.DataFrame) else out

        monkeypatch.setattr(pf, name, wrapper)

    for name, maker in makers.items():
        make(name, frame(maker))

    def ngs(season, stat_type):
        w = by_season.get(season)
        if w is None:
            return pd.DataFrame()
        return {"passing": w.ngs_pass, "receiving": w.ngs_rec}.get(stat_type, pd.DataFrame())

    def pfr_fn(season, stat_type, summary_level="week"):
        w = by_season.get(season)
        if w is None:
            return pd.DataFrame()
        if stat_type == "rush" and summary_level == "week":
            return w.pfr_rush
        if stat_type == "rec" and summary_level == "season":
            return w.pfr_rec
        return pd.DataFrame()

    first = next(iter(by_season.values()))
    make("_load_ngs", ngs)
    make("_load_pfr", pfr_fn)
    make("_load_players", lambda *a, **k: first.players)
    return calls


@pytest.fixture(autouse=True)
def _offline_and_clean(monkeypatch):
    """No test may reach a provider: every loader is an empty stub unless the test installs a world."""
    for name in LOADERS:
        monkeypatch.setattr(pf, name, lambda *a, **k: pd.DataFrame())
    pf.clear_cache()
    yield
    pf.clear_cache()


@pytest.fixture(scope="module")
def world() -> World:
    return make_world()


@pytest.fixture()
def small_k(monkeypatch):
    """Tiny gates / K so a six-week synthetic season clears every feature (real constants tested separately)."""
    for key in ALL_KEYS:
        monkeypatch.setitem(pf.SHRINKAGE_K, key, 20.0)
        monkeypatch.setitem(pf.EFFECTIVE_GATE, key, 5)


@pytest.fixture()
def table(monkeypatch, world, small_k):
    install(monkeypatch, world)
    return pf.player_feature_table(SEASON)


# ---------------------------------------------------------------------------
# Vocabulary / constants
# ---------------------------------------------------------------------------
def test_vocabulary_matches_the_contract():
    assert pf.QB_KEYS == (
        "qb_time_to_throw", "qb_release_time", "qb_pressure_epa", "qb_scramble_rate",
        "qb_sack_avoidance", "qb_efficiency", "qb_deep_attempt_rate", "qb_rz_efficiency",
    )
    assert set(pf.FEATURE_KEYS_BY_POSITION["QB"]) == set(pf.QB_KEYS)
    assert set(pf.FEATURE_KEYS_BY_POSITION["RB"]) == {"rb_yards_after_contact", "rb_breakaway_rate", "rb_rz_efficiency"}
    assert set(pf.FEATURE_KEYS_BY_POSITION["WR"]) == {
        "wr_separation", "wr_deep_route_rate", "wr_yprr_vs_man", "wr_yprr_vs_zone",
    }
    assert set(pf.FEATURE_KEYS_BY_POSITION["TE"]) == set(pf.FEATURE_KEYS_BY_POSITION["WR"]) | {
        "te_route_depth", "te_target_share_under_pressure", "te_rz_target_share",
    }
    union = set().union(*map(set, pf.FEATURE_KEYS_BY_POSITION.values()))
    assert union == set(ALL_KEYS) == set(pf.SHRINKAGE_K) == set(pf.FEATURE_BASIS) == set(pf._UNITS) == set(pf._OPPORTUNITY)


def test_every_feature_is_documented_in_the_module_docstring():
    for key in ALL_KEYS:
        assert key in pf.__doc__, key


def test_effective_gate_follows_the_documented_formula():
    for key in ALL_KEYS:
        k = pf.SHRINKAGE_K[key]
        want = max(pf.MIN_VOLUME[key], math.ceil(k * pf.MIN_WEIGHT / (1.0 - pf.MIN_WEIGHT)))
        assert pf.EFFECTIVE_GATE[key] == want, key
        # at the gate a feature carries at least MIN_WEIGHT of the player's own value
        assert pf.EFFECTIVE_GATE[key] / (pf.EFFECTIVE_GATE[key] + k) >= pf.MIN_WEIGHT - 1e-9
        assert 5.0 <= k <= 1000.0
    # the noisy ones are gated far above the base volume (no skill signal -> almost never emitted)
    assert pf.EFFECTIVE_GATE["qb_rz_efficiency"] == 177
    assert pf.EFFECTIVE_GATE["qb_sack_avoidance"] == 150
    assert pf.EFFECTIVE_GATE["wr_separation"] == pf.MIN_VOLUME["wr_separation"]


def test_real_keys_are_only_the_directly_published_ones():
    assert pf.REAL_KEYS == {"qb_time_to_throw", "wr_separation", "rb_yards_after_contact"}
    for key in ALL_KEYS:
        text = pf.FEATURE_BASIS[key]
        if key in pf.REAL_KEYS:
            assert text.startswith("real:"), key
        elif key == "te_route_depth":
            assert text.startswith("real when a full season"), key
        else:
            assert text.startswith("proxy:"), key
        assert str(pf.EFFECTIVE_GATE[key]) in text, key


def test_default_season_is_the_latest_complete_one():
    assert pf._default_season(date(2026, 10, 11)) == 2025
    assert pf._default_season(date(2027, 1, 15)) == 2025
    assert pf._default_season(date(2027, 2, 1)) == 2026


def test_module_imports_no_network_library_and_defers_nflreadpy():
    tree = ast.parse(Path(pf.__file__).read_text(encoding="utf-8"))
    banned = {"requests", "httpx", "urllib3", "aiohttp", "selenium", "playwright", "bs4"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned)
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    } | {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
    assert "nflreadpy" not in top_level  # imported lazily, inside the loaders only


# ---------------------------------------------------------------------------
# I/O boundary
# ---------------------------------------------------------------------------
def _fake_provider(**functions) -> types.ModuleType:
    fake = types.ModuleType("nflreadpy")
    for name, fn in functions.items():
        setattr(fake, name, fn)
    return fake


def test_loaders_swallow_provider_errors(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("season out of range")

    fake = _fake_provider(
        load_pbp=boom, load_nextgen_stats=boom, load_pfr_advstats=boom, load_participation=boom,
        load_rosters=boom, load_players=boom, load_snap_counts=boom,
    )
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)
    calls = {
        "_load_pbp": (2025,), "_load_ngs": (2025, "passing"), "_load_pfr": (2025, "rush", "week"),
        "_load_participation": (2025,), "_load_rosters": (2025,), "_load_players": (), "_load_snaps": (2025,),
    }
    for name, args in calls.items():
        out = ORIGINAL_LOADERS[name](*args)
        assert isinstance(out, pd.DataFrame) and out.empty, name


def test_loaders_return_pandas_and_project_pbp_columns(monkeypatch):
    pl = pytest.importorskip("polars")
    seen = {}

    def load_pbp(seasons):
        seen["pbp"] = seasons
        return pl.DataFrame({"game_id": ["g"], "week": [1], "epa": [0.1], "desc": ["a long text column"], "play_type": ["pass"]})

    def load_nextgen_stats(seasons, stat_type=None):
        seen["ngs"] = (seasons, stat_type)
        return pl.DataFrame({"player_gsis_id": ["00-1"], "week": [0]})

    def load_pfr_advstats(seasons, stat_type=None, summary_level=None):
        seen["pfr"] = (seasons, stat_type, summary_level)
        return pl.DataFrame({"pfr_player_id": ["p"], "carries": [3]})

    fake = _fake_provider(load_pbp=load_pbp, load_nextgen_stats=load_nextgen_stats, load_pfr_advstats=load_pfr_advstats)
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)
    out = ORIGINAL_LOADERS["_load_pbp"](2025)
    assert isinstance(out, pd.DataFrame) and set(out.columns) == {"game_id", "week", "epa", "play_type"}  # desc dropped
    assert seen["pbp"] == [2025]
    ngs = ORIGINAL_LOADERS["_load_ngs"](2025, "receiving")
    assert isinstance(ngs, pd.DataFrame) and seen["ngs"] == ([2025], "receiving")
    rush = ORIGINAL_LOADERS["_load_pfr"](2025, "rush", "week")
    assert isinstance(rush, pd.DataFrame) and seen["pfr"] == ([2025], "rush", "week")


def test_to_pandas_accepts_frames_dicts_none():
    assert pf._to_pandas(None).empty
    frame = pd.DataFrame({"a": [1]})
    assert pf._to_pandas(frame) is frame
    assert list(pf._to_pandas({"a": [1, 2]})["a"]) == [1, 2]


def test_safe_never_raises_and_never_returns_a_non_frame():
    assert pf._safe(lambda: 1 / 0).empty
    assert pf._safe(lambda: None).empty
    assert pf._safe(lambda: [1, 2]).empty
    frame = pd.DataFrame({"a": [1]})
    assert pf._safe(lambda: frame) is frame


# ---------------------------------------------------------------------------
# Play-by-play preparation and the raw (unshrunk) statistics, with hand-computed answers
# ---------------------------------------------------------------------------
def _prep(rows, through_week=None):
    return pf._prepare_pbp(pd.DataFrame(rows), pf._week_predicate(through_week))


def test_prepare_pbp_filters_and_flags():
    rows = [
        pbp_row("g", 1, 1, "AAA", "pass", qb_dropback=1, complete_pass=1, passer_player_id="Q1", receiver_player_id="R1",
                air_yards=25.0, yards_gained=30.0, epa=1.0, yardline_100=15.0, pass_touchdown=1),
        pbp_row("g", 2, 1, "AAA", "pass", qb_dropback=1, sack=1, passer_player_id="Q1", yards_gained=-7.0, epa=-2.0),
        pbp_row("g", 3, 1, "AAA", "run", qb_dropback=1, qb_scramble=1, rusher_player_id="Q1", yards_gained=8.0, epa=0.5),
        pbp_row("g", 4, 1, "AAA", "run", rusher_player_id="B1", yards_gained=17.0, epa=0.2),
        pbp_row("g", 5, 19, "AAA", "pass", qb_dropback=1, passer_player_id="Q1", season_type="POST"),
        pbp_row("g", 6, 1, "AAA", "pass", qb_dropback=1, passer_player_id="Q1", two_point_attempt=1),
        pbp_row("g", 7, 1, "AAA", "no_play", passer_player_id="Q1"),
        pbp_row("g", 8, 3, "AAA", "pass", qb_dropback=1, passer_player_id="Q1", receiver_player_id="R1", air_yards=5.0),
    ]
    out = _prep(rows)
    assert list(out["play_id"]) == [1.0, 2.0, 3.0, 4.0, 8.0]  # POST, two-point and no_play rows are gone
    attempt, sack, scramble, run = out.iloc[0], out.iloc[1], out.iloc[2], out.iloc[3]
    assert attempt.dropback and attempt.attempt and attempt.target and attempt.redzone and attempt.pass_td and attempt.complete
    assert attempt.qb_id == "Q1" and attempt.recv_id == "R1"
    assert sack.dropback and sack.sack and not sack.attempt and not sack.target and sack.qb_id == "Q1"
    assert scramble.dropback and scramble.scramble and not scramble.attempt and not scramble.carry
    assert scramble.qb_id == "Q1"  # the scrambler is the QB
    assert run.carry and run.rush_id == "B1" and not run.dropback and pd.isna(run.qb_id)
    # through_week removes week 3 and nothing else
    assert list(_prep(rows, through_week=2)["play_id"]) == [1.0, 2.0, 3.0, 4.0]
    assert _prep(rows, through_week=1)["week"].max() == 1


def test_prepare_pbp_empty_inputs_keep_the_schema():
    for frame in (pd.DataFrame(), None, pd.DataFrame({"week": [1], "play_type": ["kickoff"], "season_type": ["REG"]})):
        out = pf._prepare_pbp(frame, pf._week_predicate(None))
        assert out.empty and set(pf._PBP_OUT) <= set(out.columns)
        assert pf._qb_pbp_raw(out) == {} and pf._receiver_pbp_raw(out) == {} and pf._rb_pbp_raw(out) == {}


def test_qb_raw_statistics_exact():
    rows, play = [], 0

    def add(**kw):
        nonlocal play
        play += 1
        kw.setdefault("passer_player_id", "Q1")
        rows.append(pbp_row("g", play, 1, "AAA", kw.pop("play_type", "pass"), qb_dropback=1, **kw))

    for air, epa in ((25, 1.0), (10, 1.0), (30, 1.0), (5, 1.0), (22, 1.0), (0, 1.0)):  # six attempts, three deep
        add(air_yards=float(air), epa=epa, complete_pass=1)
    add(sack=1, epa=-2.0, yardline_100=10.0)  # a red-zone sack
    add(sack=1, epa=-2.0)
    add(play_type="run", qb_scramble=1, passer_player_id=None, rusher_player_id="Q1", epa=0.0, yardline_100=5.0,
        rush_touchdown=1, yards_gained=5.0)
    add(play_type="run", qb_scramble=1, passer_player_id=None, rusher_player_id="Q1", epa=0.0, yards_gained=3.0)
    rows[0].update(yardline_100=12.0, pass_touchdown=1)  # one red-zone passing TD
    rows[1].update(yardline_100=18.0)  # red-zone incompletion-ish attempt
    raw = pf._qb_pbp_raw(_prep(rows))
    value, n = raw["qb_efficiency"]["Q1"]
    assert n == 10 and value == pytest.approx((6 * 1.0 - 4.0) / 10)
    assert raw["qb_scramble_rate"]["Q1"] == (pytest.approx(0.2), 10.0)
    assert raw["qb_sack_avoidance"]["Q1"] == (pytest.approx(0.8), 10.0)
    assert raw["qb_deep_attempt_rate"]["Q1"] == (pytest.approx(3 / 6), 6.0)
    # red zone dropbacks: the TD pass, the 18-yd attempt, the 10-yd sack, the 5-yd scramble TD -> 2 TDs / 4
    assert raw["qb_rz_efficiency"]["Q1"] == (pytest.approx(0.5), 4.0)


def test_receiver_and_rusher_raw_statistics_exact():
    rows, play = [], 0
    for air in (25.0, 5.0, 21.0, -3.0):
        play += 1
        rows.append(pbp_row("g", play, 1, "AAA", "pass", qb_dropback=1, passer_player_id="Q1", receiver_player_id="R1",
                            air_yards=air, complete_pass=1))
    play += 1
    rows.append(pbp_row("g", play, 1, "AAA", "pass", qb_dropback=1, passer_player_id="Q1", air_yards=40.0))  # throwaway
    for yards, yl, td in ((3, 50, 0), (16, 15, 1), (15, 10, 0), (2, 40, 0), (1, 90, 0), (4, 30, 0)):
        play += 1
        rows.append(pbp_row("g", play, 1, "AAA", "run", rusher_player_id="B1", yards_gained=float(yards),
                            yardline_100=float(yl), rush_touchdown=td))
    play += 1
    rows.append(pbp_row("g", play, 1, "AAA", "run", qb_dropback=1, qb_scramble=1, rusher_player_id="Q1", yards_gained=30.0))
    prepared = _prep(rows)
    recv = pf._receiver_pbp_raw(prepared)
    assert recv["wr_deep_route_rate"]["R1"] == (pytest.approx(0.5), 4.0)
    assert recv["te_route_depth"]["R1"] == (pytest.approx(12.0), 4.0)
    assert "Q1" not in recv["wr_deep_route_rate"]
    rb = pf._rb_pbp_raw(prepared)
    assert rb["rb_breakaway_rate"]["B1"] == (pytest.approx(2 / 6), 6.0)  # 16 and 15 yards; the QB scramble is no carry
    assert rb["rb_rz_efficiency"]["B1"] == (pytest.approx(1 / 2), 2.0)  # two red-zone carries, one TD
    assert "Q1" not in rb["rb_breakaway_rate"]


def test_ngs_raw_season_row_versus_weekly_rows_and_no_leak():
    rows = [
        dict(player_gsis_id="A", season_type="REG", week=0, v=2.80, w=500),
        dict(player_gsis_id="A", season_type="REG", week=1, v=2.6, w=100),
        dict(player_gsis_id="A", season_type="REG", week=2, v=2.8, w=100),
        dict(player_gsis_id="A", season_type="REG", week=3, v=3.0, w=200),
        dict(player_gsis_id="A", season_type="POST", week=19, v=9.0, w=50),
        dict(player_gsis_id="B", season_type="REG", week=1, v=2.5, w=40),
        dict(player_gsis_id="B", season_type="REG", week=3, v=3.5, w=60),
        dict(player_gsis_id="C", season_type="REG", week=2, v=np.nan, w=60),
    ]
    frame = pd.DataFrame(rows)
    values, where = pf._ngs_raw(frame, "v", "w", None)
    assert values["A"] == (2.80, 500.0) and where["A"] == "season row"  # whole-season row when nothing can leak
    assert values["B"] == (pytest.approx((2.5 * 40 + 3.5 * 60) / 100), 100.0) and where["B"] == "weekly rows"
    assert "C" not in values
    values, where = pf._ngs_raw(frame, "v", "w", 2)  # week 3 exists -> the season row would leak it
    assert values["A"] == (pytest.approx(2.7), 200.0) and where["A"] == "weekly rows"
    assert values["B"] == (2.5, 40.0)
    values, where = pf._ngs_raw(frame, "v", "w", 3)  # through the last week the season row is fine again
    assert values["A"] == (2.80, 500.0) and where["A"] == "season row"
    assert pf._ngs_raw(pd.DataFrame(), "v", "w", None) == ({}, {})
    assert pf._ngs_raw(pd.DataFrame({"x": [1]}), "v", "w", None) == ({}, {})


def test_pfr_rush_and_adot_raw():
    rows = pd.DataFrame(
        [
            dict(pfr_player_id="R1", week=1, game_type="REG", carries=10, rushing_yards_after_contact=20.0),
            dict(pfr_player_id="R1", week=2, game_type="REG", carries=20, rushing_yards_after_contact=70.0),
            dict(pfr_player_id="R1", week=3, game_type="REG", carries=10, rushing_yards_after_contact=100.0),
            dict(pfr_player_id="R1", week=19, game_type="POST", carries=30, rushing_yards_after_contact=900.0),
            dict(pfr_player_id="R2", week=1, game_type="REG", carries=np.nan, rushing_yards_after_contact=5.0),
            dict(pfr_player_id=None, week=1, game_type="REG", carries=5, rushing_yards_after_contact=5.0),
        ]
    )
    full = pf._pfr_rush_raw(rows, pf._week_predicate(None))
    assert full == {"R1": (pytest.approx(190 / 40), 40.0)}  # published yards-after-contact / carries, REG only
    assert pf._pfr_rush_raw(rows, pf._week_predicate(2)) == {"R1": (pytest.approx(90 / 30), 30.0)}
    assert pf._pfr_rush_raw(pd.DataFrame(), pf._week_predicate(None)) == {}

    rec = pd.DataFrame(
        [
            dict(pfr_id="T1", tm="AAA", tgt=30, adot=8.0),
            dict(pfr_id="T1", tm="BBB", tgt=20, adot=6.0),
            dict(pfr_id="T1", tm="2TM", tgt=50, adot=7.2),  # the traded player's total row is the largest-n row
            dict(pfr_id="T2", tm="CCC", tgt=0, adot=5.0),
            dict(pfr_id="T3", tm="DDD", tgt=12, adot=np.nan),
        ]
    )
    assert pf._pfr_adot_raw(rec) == {"T1": (7.2, 50.0)}


def test_participation_raw_exact():
    gid = "2025_01_AAA_BBB"
    plays = [
        # (play_id, kind, recv, air, complete, yards, epa, pressured, ttt, cov, on field)
        (1, "att", "T1", 5.0, 1, 10.0, 0.4, True, 2.0, "MAN_COVERAGE", "Q1;W1;T1"),
        (2, "att", "W1", 15.0, 1, 20.0, 1.0, False, 2.5, "ZONE_COVERAGE", "Q1;W1;X9"),
        (3, "att", "T1", 8.0, 0, 0.0, -0.6, "True", 2.2, "ZONE_COVERAGE", "Q1;W1;T1"),
        (4, "sack", None, np.nan, 0, -7.0, -2.0, True, np.nan, "MAN_COVERAGE", "Q1;W1;T1"),
        (5, "att", "W1", 6.0, 1, 8.0, 0.2, False, 3.0, "MAN_COVERAGE", "Q1;W1;X9"),
        (6, "att", "W1", 12.0, 0, 0.0, -0.4, False, 2.9, "ZONE_COVERAGE", "Q1;W1;X9"),
        (7, "att", "W1", 9.0, 1, 5.0, 0.0, True, 2.1, "ZONE_COVERAGE", "Q1;W1;T1"),
        (8, "att", "W1", 9.0, 1, 5.0, 0.0, False, 9.5, "", "Q1;W1;T1"),  # ttt out of range, no coverage label
    ]
    pbp_rows, part_rows = [], []
    for play_id, kind, recv, air, complete, yards, epa, pressured, ttt, cov, field in plays:
        if kind == "sack":
            pbp_rows.append(pbp_row(gid, play_id, 1, "AAA", "pass", qb_dropback=1, sack=1, passer_player_id="Q1",
                                    yards_gained=yards, epa=epa))
        else:
            pbp_rows.append(pbp_row(gid, play_id, 1, "AAA", "pass", qb_dropback=1, passer_player_id="Q1",
                                    receiver_player_id=recv, air_yards=air, complete_pass=complete, yards_gained=yards, epa=epa))
        part_rows.append(dict(nflverse_game_id=gid, play_id=float(play_id), time_to_throw=ttt, was_pressure=pressured,
                              defense_man_zone_type=cov, offense_players=field))
    part_rows.append(dict(part_rows[0]))  # a duplicated participation row must not double count
    pos = {"Q1": "QB", "W1": "WR", "T1": "TE"}
    raw = pf._participation_raw(_prep(pbp_rows), pd.DataFrame(part_rows), pos)
    # unpressured throws with a sane time_to_throw: plays 2, 5, 6 -> median 2.9
    assert raw["qb_release_time"]["Q1"] == (pytest.approx(2.9), 3.0)
    # pressured dropbacks (incl. the sack and the string "True"): plays 1, 3, 4, 7
    assert raw["qb_pressure_epa"]["Q1"] == (pytest.approx((0.4 - 0.6 - 2.0 + 0.0) / 4), 4.0)
    # vs man: plays 1, 4, 5. W1 on the field for all three, T1 for 1 and 4; only W1 caught play 5 (8 yds), T1 play 1 (10)
    assert raw["wr_yprr_vs_man"]["W1"] == (pytest.approx(8.0 / 3), 3.0)
    assert raw["wr_yprr_vs_man"]["T1"] == (pytest.approx(10.0 / 2), 2.0)
    # vs zone: plays 2, 3, 6, 7. W1 caught 20 + 5; T1 was on the field for 3 and 7 and caught nothing
    assert raw["wr_yprr_vs_zone"]["W1"] == (pytest.approx(25.0 / 4), 4.0)
    assert raw["wr_yprr_vs_zone"]["T1"] == (0.0, 2.0)
    assert "Q1" not in raw["wr_yprr_vs_man"]  # QBs never get a route-proxy split
    # pressured targets: plays 1, 3 (T1) and 7 (W1); T1 was on the field for all three, targeted on two
    assert raw["te_target_share_under_pressure"]["T1"] == (pytest.approx(2 / 3), 3.0)
    assert "W1" not in raw["te_target_share_under_pressure"]


def test_participation_raw_is_empty_without_a_usable_feed():
    prepared = _prep([pbp_row("g", 1, 1, "AAA", "pass", qb_dropback=1, passer_player_id="Q1", air_yards=5.0)])
    for part in (pd.DataFrame(), pd.DataFrame({"x": [1]}), None):
        assert pf._participation_raw(prepared, part, {"Q1": "QB"}) == {}
    assert pf._participation_raw(pf._empty_pbp(), pd.DataFrame({"nflverse_game_id": ["g"], "play_id": [1.0]}), {}) == {}


def test_te_red_zone_share_uses_the_games_the_player_was_active():
    rows, play = [], 0
    for week, recv in ((1, "X"), (1, "W"), (1, "W"), (2, "X"), (2, "Y")):
        play += 1
        rows.append(pbp_row(f"g{week}", play, week, "AAA", "pass", qb_dropback=1, passer_player_id="Q1",
                            receiver_player_id=recv, air_yards=3.0, yardline_100=12.0))
    play += 1
    rows.append(pbp_row("g1", play, 1, "AAA", "pass", qb_dropback=1, passer_player_id="Q1", receiver_player_id="X",
                        air_yards=3.0, yardline_100=60.0))  # not a red-zone target
    games = pd.DataFrame(
        [dict(pid="X", week=1, team="AAA"), dict(pid="X", week=2, team="AAA"), dict(pid="Y", week=2, team="AAA"),
         dict(pid="W", week=1, team="AAA")]
    )
    out = pf._te_rz_share_raw(_prep(rows), games, {"X", "Y"})
    assert out["X"] == (pytest.approx(2 / 5), 5.0)
    assert out["Y"] == (pytest.approx(1 / 2), 2.0)  # Y was only active in week 2, whose team total is 2
    assert "W" not in out  # not a TE
    assert pf._te_rz_share_raw(_prep(rows), pd.DataFrame(), {"X"}) == {}


def test_id_bridge_players_first_then_rosters_and_unmatched_dropped():
    players = pd.DataFrame(
        [dict(gsis_id="G1", pfr_id="P1"), dict(gsis_id="G2", pfr_id=None), dict(gsis_id="", pfr_id="P3"),
         dict(gsis_id="G5", pfr_id="  ")]
    )
    rosters = pd.DataFrame([dict(gsis_id="GX", pfr_id="P1"), dict(gsis_id="G4", pfr_id="P4")])
    mapping = pf._pfr_to_gsis(rosters, players)
    assert mapping == {"P1": "G1", "P4": "G4"}  # players table wins; blanks and unknowns are never guessed
    assert pf._pfr_to_gsis(pd.DataFrame(), pd.DataFrame()) == {}


def test_identity_position_name_team_and_through_week():
    rosters = pd.DataFrame(
        [
            dict(gsis_id="A", week=1, team="AAA", position="WR", full_name="Al Early", game_type="REG"),
            dict(gsis_id="A", week=5, team="BBB", position="TE", full_name="Al Early", game_type="REG"),
            dict(gsis_id="B", week=1, team="CCC", position="HB", full_name="Bo Back", game_type="REG"),
            dict(gsis_id="C", week=1, team="DDD", position="FB", full_name="Cy Fullback", game_type="REG"),
            dict(gsis_id="K", week=1, team="EEE", position="K", full_name="Kick Er", game_type="REG"),
        ]
    )
    players = pd.DataFrame(
        [dict(gsis_id="C", display_name="Cyrus Fullback", position="RB"), dict(gsis_id="Z", display_name="Zed Late", position="WR")]
    )
    ngs = pd.DataFrame([dict(player_gsis_id="D", player_display_name="Dee Ngs", player_position="WR")])
    position, name, team = pf._identity(rosters, players, [ngs], None)
    assert position == {"A": "TE", "B": "RB", "C": "RB", "Z": "WR", "D": "WR"}  # latest roster week wins; K is not a skill spot
    assert name["A"] == "Al Early" and name["C"] == "Cy Fullback" and name["D"] == "Dee Ngs" and name["Z"] == "Zed Late"
    assert team["A"] == "BBB" and team["B"] == "CCC"
    position, _, team = pf._identity(rosters, players, [ngs], 3)  # a week-5 move cannot be known at week 3
    assert position["A"] == "WR" and team["A"] == "AAA"


# ---------------------------------------------------------------------------
# Shrinkage and gates
# ---------------------------------------------------------------------------
def _assemble(raw, position, **kw):
    return pf._shrink_and_assemble(
        raw, kw.get("sources", {}), kw.get("notes", {}), position=position, name=kw.get("name", {}), team=kw.get("team", {}),
        games=kw.get("games", {}), season=2025, through_week=kw.get("through_week"), kind_overrides=kw.get("kind_overrides", {}),
    )


def test_shrinkage_formula_league_mean_and_gate(monkeypatch):
    monkeypatch.setitem(pf.SHRINKAGE_K, "qb_scramble_rate", 100.0)
    monkeypatch.setitem(pf.EFFECTIVE_GATE, "qb_scramble_rate", 50)
    raw = {"qb_scramble_rate": {f"Q{j}": (0.02 * (j + 1), 100.0 + 40 * j) for j in range(9)}}
    raw["qb_scramble_rate"]["Qlow"] = (0.60, 49.0)  # one below the gate: omitted, and NOT in the league mean
    raw["qb_scramble_rate"]["Wr"] = (0.90, 500.0)  # a WR with a "QB" stat is not a QB
    position = {f"Q{j}": "QB" for j in range(9)} | {"Qlow": "QB", "Wr": "WR"}
    out = _assemble(raw, position, sources={"qb_scramble_rate": "nflverse pbp 2025"})
    assert "Qlow" not in out and "Wr" not in out and len(out) == 9
    pool = [(0.02 * (j + 1), 100.0 + 40 * j) for j in range(9)]
    league = sum(v * n for v, n in pool) / sum(n for _, n in pool)
    for j, (value, n) in enumerate(pool):
        w = n / (n + 100.0)
        assert out[f"Q{j}"]["features"]["qb_scramble_rate"] == pytest.approx(w * value + (1 - w) * league, abs=1e-4)
    # a smaller sample is pulled harder toward the league mean than a bigger one with the same raw value
    near = {"qb_scramble_rate": {"A": (0.20, 60.0), "B": (0.20, 600.0)} | {f"F{j}": (0.05, 300.0) for j in range(8)}}
    pos2 = {k: "QB" for k in near["qb_scramble_rate"]}
    got = _assemble(near, pos2)
    assert got["A"]["features"]["qb_scramble_rate"] < got["B"]["features"]["qb_scramble_rate"] < 0.20


def test_too_few_qualifying_players_means_no_reference_and_no_feature(monkeypatch):
    monkeypatch.setitem(pf.EFFECTIVE_GATE, "qb_efficiency", 10)
    raw = {"qb_efficiency": {f"Q{j}": (0.1 * j, 100.0) for j in range(pf.MIN_LEAGUE_PLAYERS - 1)}}
    assert _assemble(raw, {k: "QB" for k in raw["qb_efficiency"]}) == {}
    raw["qb_efficiency"]["Qx"] = (0.5, 100.0)
    out = _assemble(raw, {k: "QB" for k in raw["qb_efficiency"]})
    assert len(out) == pf.MIN_LEAGUE_PLAYERS


def test_te_gets_wr_keys_shrunk_toward_the_te_mean_not_the_wr_mean(monkeypatch):
    monkeypatch.setitem(pf.SHRINKAGE_K, "wr_deep_route_rate", 10.0)
    monkeypatch.setitem(pf.EFFECTIVE_GATE, "wr_deep_route_rate", 5)
    raw = {"wr_deep_route_rate": {f"W{j}": (0.30, 50.0) for j in range(9)} | {f"T{j}": (0.10, 50.0) for j in range(9)}}
    position = {f"W{j}": "WR" for j in range(9)} | {f"T{j}": "TE" for j in range(9)}
    out = _assemble(raw, position)
    assert out["W0"]["features"]["wr_deep_route_rate"] == pytest.approx(0.30)  # all-equal pool: mean == raw
    assert out["T0"]["features"]["wr_deep_route_rate"] == pytest.approx(0.10)
    assert "TE league mean" in out["T0"]["basis"]["wr_deep_route_rate"]
    assert "WR league mean" in out["W0"]["basis"]["wr_deep_route_rate"]
    assert out["T0"]["position"] == "TE" and out["W0"]["position"] == "WR"


def test_kind_note_and_override_flow_into_kinds_and_basis(monkeypatch):
    monkeypatch.setitem(pf.EFFECTIVE_GATE, "te_route_depth", 5)
    monkeypatch.setitem(pf.EFFECTIVE_GATE, "wr_separation", 5)
    raw = {"te_route_depth": {f"T{j}": (6.0 + j, 40.0) for j in range(9)},
           "wr_separation": {f"T{j}": (2.0 + 0.1 * j, 40.0) for j in range(9)}}
    position = {f"T{j}": "TE" for j in range(9)}
    out = _assemble(
        raw, position,
        sources={"te_route_depth": "nflverse pbp 2025 mean air_yards on targets", "wr_separation": "fallback feed"},
        notes={("T0", "wr_separation"): "NGS receiving avg_separation 2025 (season row)"},
        kind_overrides={("T1", "te_route_depth"): "real"},
    )
    assert out["T0"]["kinds"] == {"te_route_depth": "proxy", "wr_separation": "real"}
    assert out["T1"]["kinds"]["te_route_depth"] == "real" and out["T2"]["kinds"]["te_route_depth"] == "proxy"
    assert "NGS receiving avg_separation 2025 (season row)" in out["T0"]["basis"]["wr_separation"]
    assert "fallback feed" in out["T1"]["basis"]["wr_separation"]
    for entry in out.values():
        for key, text in entry["basis"].items():
            assert text.startswith(entry["kinds"][key] + ":") and "n=40" in text and "weight" in text


# ---------------------------------------------------------------------------
# End to end through the patched boundary
# ---------------------------------------------------------------------------
def test_table_shape_keys_kinds_and_basis_for_every_feature(table):
    assert len(table) == 5 * N_TEAMS
    seen_keys: set[str] = set()
    for gsis, entry in table.items():
        assert set(entry) == {"name", "team", "position", "features", "kinds", "basis", "games", "season", "through_week"}
        assert entry["season"] == SEASON and entry["through_week"] is None
        assert entry["name"] and entry["team"].startswith("T") and entry["games"] == 6
        assert set(entry["features"]) == set(pf.FEATURE_KEYS_BY_POSITION[entry["position"]]), entry["name"]
        assert set(entry["kinds"]) == set(entry["basis"]) == set(entry["features"])
        for key, value in entry["features"].items():
            assert isinstance(value, float) and math.isfinite(value)
            kind = entry["kinds"][key]
            assert kind == ("real" if key in pf.REAL_KEYS or key == "te_route_depth" and entry["basis"][key].startswith("real") else "proxy")
            text = entry["basis"][key]
            assert text.startswith(kind + ":") and "2025" in text and "n=" in text and "shrunk toward" in text
            assert "K=20" in text  # the (patched) constant that produced the weight is disclosed
        seen_keys |= set(entry["features"])
    assert seen_keys == set(ALL_KEYS)  # all 18 features came out of one synthetic season
    by_pos = Counter(e["position"] for e in table.values())
    assert by_pos == {"QB": N_TEAMS, "WR": 2 * N_TEAMS, "TE": N_TEAMS, "RB": N_TEAMS}


def test_directly_published_numbers_are_real_and_the_rest_are_proxies(table):
    qb = table[pid("QB", 4)]
    assert qb["kinds"]["qb_time_to_throw"] == "real" and "NGS passing avg_time_to_throw" in qb["basis"]["qb_time_to_throw"]
    assert all(kind == "proxy" for key, kind in qb["kinds"].items() if key != "qb_time_to_throw")
    wr = table[pid("WR1", 4)]
    assert wr["kinds"]["wr_separation"] == "real" and "NGS receiving avg_separation" in wr["basis"]["wr_separation"]
    assert wr["kinds"]["wr_yprr_vs_man"] == wr["kinds"]["wr_deep_route_rate"] == "proxy"
    for key in ("wr_yprr_vs_man", "wr_yprr_vs_zone"):  # the route-run approximation is disclosed, not hidden
        assert "on the field" in pf.FEATURE_BASIS[key] or "as wr_yprr_vs_man" in pf.FEATURE_BASIS[key]
    assert "over-counts blocking snaps" in pf.FEATURE_BASIS["wr_yprr_vs_man"]
    rb = table[pid("RB", 4)]
    assert rb["kinds"]["rb_yards_after_contact"] == "real" and "PFR" in rb["basis"]["rb_yards_after_contact"]
    assert rb["kinds"]["rb_breakaway_rate"] == rb["kinds"]["rb_rz_efficiency"] == "proxy"
    te = table[pid("TE", 4)]
    assert te["kinds"]["te_route_depth"] == "real" and "PFR receiving adot" in te["basis"]["te_route_depth"]
    # a TE whose PFR id does not map falls back to the pbp mean air_yards and says so
    unmapped = table[pid("TE", 3)]
    assert unmapped["kinds"]["te_route_depth"] == "proxy" and "mean air_yards" in unmapped["basis"]["te_route_depth"]


def test_feature_ordering_follows_the_underlying_signal(table):
    def series(role, key):
        return [table[pid(role, i)]["features"][key] for i in range(N_TEAMS)]

    def spearman(values):
        return float(pd.Series(values).corr(pd.Series(range(len(values))), method="spearman"))

    assert spearman(series("QB", "qb_time_to_throw")) > 0.9  # slower QB = higher time to throw
    assert spearman(series("QB", "qb_release_time")) > 0.8
    assert spearman(series("WR1", "wr_separation")) > 0.9
    assert spearman(series("RB", "rb_yards_after_contact")) > 0.9
    assert spearman(series("QB", "qb_sack_avoidance")) < -0.5  # more sacks as i grows
    assert spearman(series("RB", "rb_breakaway_rate")) > 0.5
    assert 2.2 < table[pid("QB", 0)]["features"]["qb_time_to_throw"] < 3.0


def test_end_to_end_shrinkage_matches_an_independent_calculation(table, world):
    p = world.pbp
    db = p[(p.qb_dropback == 1) & p.play_type.isin(["pass", "run"])]
    qb = db.passer_player_id.where(db.passer_player_id.notna(), db.rusher_player_id)
    grp = db.groupby(qb)
    raw, n = grp["qb_scramble"].mean(), grp.size()
    league = float((raw * n).sum() / n.sum())
    for gsis in raw.index:
        w = n[gsis] / (n[gsis] + 20.0)
        want = w * raw[gsis] + (1 - w) * league
        assert table[gsis]["features"]["qb_scramble_rate"] == pytest.approx(want, abs=1e-4)
        assert f"n={n[gsis]:.0f} dropbacks" in table[gsis]["basis"]["qb_scramble_rate"]
    # RBs: breakaway share of designed carries, same procedure
    runs = p[(p.play_type == "run") & (p.qb_dropback == 0)]
    g2 = runs.groupby("rusher_player_id")
    braw, bn = g2.yards_gained.apply(lambda s: float((s >= 15).mean())), g2.size()
    bl = float((braw * bn).sum() / bn.sum())
    for gsis in braw.index:
        w = bn[gsis] / (bn[gsis] + 20.0)
        assert table[gsis]["features"]["rb_breakaway_rate"] == pytest.approx(w * braw[gsis] + (1 - w) * bl, abs=1e-4)


def test_volume_gates_omit_rather_than_shrink_to_nothing(monkeypatch, world, small_k):
    install(monkeypatch, world)
    # dropbacks over six weeks are 6 * (30 + 2i) = 180, 192, 204 ... 288 for teams 0..9

    def qbs_with_scramble_rate(gate):
        monkeypatch.setitem(pf.EFFECTIVE_GATE, "qb_scramble_rate", gate)
        pf.clear_cache()
        out = pf.player_feature_table(SEASON)
        return out, {g for g, e in out.items() if "qb_scramble_rate" in e["features"]}

    out, have = qbs_with_scramble_rate(190)  # everyone but team 0 (180 dropbacks)
    assert have == {pid("QB", i) for i in range(1, N_TEAMS)}
    assert "qb_scramble_rate" not in out[pid("QB", 0)]["features"]
    assert "qb_efficiency" in out[pid("QB", 0)]["features"]  # other features have their own gate
    _, have = qbs_with_scramble_rate(200)  # exactly MIN_LEAGUE_PLAYERS (8) clear it: still a reference
    assert have == {pid("QB", i) for i in range(2, N_TEAMS)} and len(have) == pf.MIN_LEAGUE_PLAYERS
    # one fewer than MIN_LEAGUE_PLAYERS clear 210 -> no league reference -> omitted for everyone, not "shrunk to nothing"
    _, have = qbs_with_scramble_rate(210)
    assert have == set()


def test_real_constants_gate_the_noisy_features_hard(monkeypatch, world):
    install(monkeypatch, world)  # real K / gates; six synthetic weeks (120 carries, ~40 red-zone dropbacks each)
    out = pf.player_feature_table(SEASON)
    emitted = {key for e in out.values() for key in e["features"]}
    assert "qb_rz_efficiency" not in emitted  # needs 177 red-zone dropbacks
    assert "rb_rz_efficiency" not in emitted  # needs 52 red-zone carries
    assert {"qb_scramble_rate", "qb_efficiency", "wr_separation", "rb_yards_after_contact"} <= emitted
    # the same player's value is shrunk much harder under the real K than under the test K
    qb = out[pid("QB", 5)]
    assert "K=98" in qb["basis"]["qb_scramble_rate"] and "K=113" in qb["basis"]["qb_time_to_throw"]


def test_through_week_does_not_leak_later_weeks(monkeypatch, small_k):
    full = make_world(range(1, 9), poison_from=7)  # weeks 7-8 are absurd on every feed
    clean = make_world(range(1, 7))
    install(monkeypatch, full)
    cut = pf.player_feature_table(SEASON, through_week=6)
    uncut = pf.player_feature_table(SEASON)
    pf.clear_cache()
    install(monkeypatch, clean)
    reference = pf.player_feature_table(SEASON, through_week=6)
    assert cut and set(cut) == set(reference)
    for gsis, entry in cut.items():
        assert entry["through_week"] == 6 and entry["games"] == reference[gsis]["games"] == 6
        assert entry["features"] == reference[gsis]["features"], entry["name"]
        assert entry["kinds"] == reference[gsis]["kinds"]
        # NGS season rows would contain weeks 7-8, so weekly rows are used
        if "qb_time_to_throw" in entry["features"]:
            assert "weekly rows" in entry["basis"]["qb_time_to_throw"]
    # ... and the poisoned weeks really are visible when nobody cuts them off
    assert uncut[pid("QB", 0)]["features"]["qb_time_to_throw"] > cut[pid("QB", 0)]["features"]["qb_time_to_throw"] + 0.3
    assert uncut[pid("QB", 0)]["games"] == 8
    # a full-season request (no through_week) may use the published season rows and the PFR adot
    assert uncut[pid("TE", 0)]["kinds"]["te_route_depth"] == "real"
    assert cut[pid("TE", 0)]["kinds"]["te_route_depth"] == "proxy"


def test_through_week_bounds(monkeypatch, world, small_k):
    install(monkeypatch, world)
    assert pf.player_feature_table(SEASON, through_week=0) == {}
    assert pf.player_feature_table(SEASON, through_week=-3) == {}
    assert pf.player_feature_table(SEASON, through_week="x") == {}
    assert pf.player_feature_table("not a season") == {}
    three = pf.player_feature_table(SEASON, through_week=3)
    assert three and all(e["games"] == 3 and e["through_week"] == 3 for e in three.values())
    assert all("weeks 1-3" in text for e in three.values() for k, text in e["basis"].items() if k == "qb_efficiency")


def test_pfr_ids_are_bridged_to_gsis_and_the_match_rate_is_reported(monkeypatch, world, small_k):
    install(monkeypatch, world)
    out = pf.player_feature_table(SEASON)
    stats = pf._last_build_stats()
    assert stats["pfr_rush_rows"] == N_TEAMS + 1 and stats["pfr_rush_mapped"] == N_TEAMS  # the ghost id is dropped
    assert stats["games_source"] == "snap counts" and stats["participation_season"] == SEASON
    assert stats["participation_is_fallback"] is False
    assert pid("RB", 2) in out and "rb_yards_after_contact" in out[pid("RB", 2)]["features"]
    assert not any(key.startswith("Pfr") for key in out)  # nothing is keyed by a PFR id
    # snap counts (PFR ids) give each player's games through the same bridge
    assert out[pid("WR2", 5)]["games"] == 6


def test_participation_falls_back_to_the_latest_available_season(monkeypatch, small_k):
    now, last = make_world(season=2026, seed=3), make_world(season=2025, seed=4)
    install(monkeypatch, now, last, participation={2025})
    out = pf.player_feature_table(2026)
    stats = pf._last_build_stats()
    assert stats["participation_season"] == 2025 and stats["participation_is_fallback"] is True
    qb = out[pid("QB", 3)]
    for key in ("qb_release_time", "qb_pressure_epa"):
        assert key in qb["features"]
        text = qb["basis"][key]
        assert "2025" in text and "fallback" in text and "no 2026 participation" in text
        assert qb["kinds"][key] == "proxy"
    assert not any(key.startswith(("wr_", "te_", "rb_")) for key in qb["features"])  # a QB gets only QB keys
    assert "wr_yprr_vs_zone" in out[pid("WR1", 3)]["features"]
    assert "fallback" in out[pid("WR1", 3)]["basis"]["wr_yprr_vs_zone"]
    # non-participation features still come from the requested season
    assert "nflverse pbp 2026" in qb["basis"]["qb_efficiency"] and "fallback" not in qb["basis"]["qb_efficiency"]
    assert all(e["season"] == 2026 for e in out.values())


def test_fallback_participation_only_describes_players_who_are_current(monkeypatch, small_k):
    now, last = make_world(season=2026, seed=3), make_world(season=2025, seed=4)
    retired = "00-1099999"
    team9 = last.pbp["posteam"] == "T09"
    last.pbp.loc[team9, "passer_player_id"] = last.pbp.loc[team9, "passer_player_id"].replace({pid("QB", 9): retired})
    last.pbp.loc[team9, "rusher_player_id"] = last.pbp.loc[team9, "rusher_player_id"].replace({pid("QB", 9): retired})
    last.part["offense_players"] = last.part["offense_players"].str.replace(pid("QB", 9), retired)
    # the all-time players table still knows him as a QB, but he is on no 2026 roster and threw no 2026 pass
    now.players = pd.concat(
        [now.players, pd.DataFrame([dict(gsis_id=retired, pfr_id="PfrRetired", display_name="Rick Retired", position="QB",
                                         latest_team="T09")])],
        ignore_index=True,
    )
    install(monkeypatch, now, last, participation={2025})
    out = pf.player_feature_table(2026)
    assert retired not in out
    assert "qb_release_time" in out[pid("QB", 2)]["features"]  # a current QB still gets the fallback number


def test_the_table_does_not_depend_on_feed_row_order(monkeypatch, small_k):
    a, b = make_world(), make_world()
    rng = np.random.default_rng(5)
    for frame_name in ("pbp", "part", "ngs_pass", "ngs_rec", "pfr_rush", "pfr_rec", "rosters", "players", "snaps"):
        frame = getattr(b, frame_name)
        setattr(b, frame_name, frame.iloc[rng.permutation(len(frame))].reset_index(drop=True))
    install(monkeypatch, a)
    first = pf.player_feature_table(SEASON)
    pf.clear_cache()
    install(monkeypatch, b)
    second = pf.player_feature_table(SEASON)
    assert first and set(first) == set(second)
    for gsis, entry in first.items():
        assert second[gsis]["features"] == pytest.approx(entry["features"], abs=1e-9), entry["name"]
        assert second[gsis]["kinds"] == entry["kinds"] and second[gsis]["team"] == entry["team"]


def test_participation_outside_the_lookback_is_omitted(monkeypatch, small_k):
    now = make_world(season=2026, seed=3)
    old = make_world(season=2023, seed=5)
    install(monkeypatch, now, old, participation={2023})  # three seasons back: 2026, 2025, 2024 are tried, not 2023
    out = pf.player_feature_table(2026)
    emitted = {key for e in out.values() for key in e["features"]}
    assert pf._last_build_stats()["participation_season"] is None
    assert not emitted & {"qb_release_time", "qb_pressure_epa", "wr_yprr_vs_man", "wr_yprr_vs_zone", "te_target_share_under_pressure"}
    assert {"qb_efficiency", "qb_time_to_throw", "wr_separation", "te_rz_target_share", "rb_breakaway_rate"} <= emitted


def test_partial_outage_keeps_what_is_still_computable(monkeypatch, world, small_k):
    install(monkeypatch, world, skip={"_load_ngs", "_load_pfr", "_load_participation", "_load_snaps", "_load_players"})
    out = pf.player_feature_table(SEASON)
    assert out
    emitted = {key for e in out.values() for key in e["features"]}
    assert not emitted & {"qb_time_to_throw", "wr_separation", "rb_yards_after_contact", "qb_release_time",
                          "qb_pressure_epa", "wr_yprr_vs_man", "te_target_share_under_pressure"}
    assert {"qb_efficiency", "qb_scramble_rate", "rb_breakaway_rate", "te_rz_target_share"} <= emitted
    te = out[pid("TE", 1)]
    assert "pbp involvement (approximation)" in te["basis"]["te_rz_target_share"]  # the weaker games source is disclosed
    assert pf._last_build_stats()["games_source"].startswith("pbp involvement")


def test_every_provider_failure_mode_yields_an_empty_table_not_an_exception(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    for name in LOADERS:
        monkeypatch.setattr(pf, name, boom)
    assert pf.player_feature_table(2025) == {} and pf.player_feature_table(2025, through_week=4) == {}
    for junk in (None, [1, 2], {"a": 1}, "text", 5):
        for name in LOADERS:
            monkeypatch.setattr(pf, name, lambda *a, _j=junk, **k: _j)
        pf.clear_cache()
        assert pf.player_feature_table(2025) == {}
    for name in LOADERS:  # empty frames
        monkeypatch.setattr(pf, name, lambda *a, **k: pd.DataFrame())
    assert pf.player_feature_table(2025) == {}


def test_a_failure_inside_the_build_is_logged_not_raised(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise KeyError("unexpected column layout")

    monkeypatch.setattr(pf, "_compute", broken)
    with caplog.at_level(logging.WARNING, logger="nfl_pack.player_features"):
        assert pf.player_feature_table(2025) == {}
    assert any("failed" in record.getMessage() for record in caplog.records)


def test_garbage_frames_do_not_raise(monkeypatch):
    junk = pd.DataFrame({"week": ["a", None], "play_type": [1, 2], "season_type": [None, None], "gsis_id": [1, 2], "pfr_id": [3, 4]})
    for name in LOADERS:
        monkeypatch.setattr(pf, name, lambda *a, **k: junk.copy())
    assert pf.player_feature_table(2025) == {}


def test_cache_ttl_isolation_and_empty_results_are_not_cached(monkeypatch, world, small_k):
    # a failed build is not cached: the next call may succeed
    install(monkeypatch, world, skip=set(LOADERS))
    assert pf.player_feature_table(SEASON) == {}
    calls = install(monkeypatch, world)
    first = pf.player_feature_table(SEASON)
    assert first
    n = sum(calls.values())
    assert n > 0
    second = pf.player_feature_table(SEASON)
    assert sum(calls.values()) == n and second == first  # served from the cache
    second[pid("QB", 0)]["features"]["qb_efficiency"] = 99.0  # callers get their own copy
    assert pf.player_feature_table(SEASON)[pid("QB", 0)]["features"]["qb_efficiency"] != 99.0
    pf.player_feature_table(SEASON, through_week=4)  # a different key is a different build
    assert sum(calls.values()) > n
    pf.clear_cache()
    before = sum(calls.values())
    pf.player_feature_table(SEASON)
    assert sum(calls.values()) > before
    monkeypatch.setattr(pf, "CACHE_SECONDS", 0.0)  # an expired entry is rebuilt
    before = sum(calls.values())
    pf.player_feature_table(SEASON)
    assert sum(calls.values()) > before


def test_default_season_is_used_when_none_given(monkeypatch, world, small_k):
    monkeypatch.setattr(pf, "_default_season", lambda today=None: SEASON)
    calls = install(monkeypatch, world)
    out = pf.player_feature_table()
    assert out and all(e["season"] == SEASON for e in out.values()) and calls["_load_pbp"] >= 1


def test_the_build_is_deterministic_and_sorted(monkeypatch, world, small_k):
    install(monkeypatch, world)
    a = pf.player_feature_table(SEASON)
    pf.clear_cache()
    b = pf.player_feature_table(SEASON)
    assert a == b
    assert list(a) == sorted(a)
    for entry in a.values():
        assert list(entry["features"]) == sorted(entry["features"])
        assert list(entry["kinds"]) == list(entry["features"]) == list(entry["basis"])


def test_the_loader_never_mutates_the_frames_it_is_handed(monkeypatch, world, small_k):
    snapshot = {k: getattr(world, k).copy(deep=True) for k in ("pbp", "part", "ngs_pass", "ngs_rec", "pfr_rush", "pfr_rec", "rosters", "players", "snaps")}

    def handed(frame):
        return lambda *a, **k: frame  # the very same object every time, no copy

    monkeypatch.setattr(pf, "_load_pbp", handed(world.pbp))
    monkeypatch.setattr(pf, "_load_participation", handed(world.part))
    monkeypatch.setattr(pf, "_load_rosters", handed(world.rosters))
    monkeypatch.setattr(pf, "_load_players", handed(world.players))
    monkeypatch.setattr(pf, "_load_snaps", handed(world.snaps))
    monkeypatch.setattr(pf, "_load_ngs", lambda season, st: world.ngs_pass if st == "passing" else world.ngs_rec)
    monkeypatch.setattr(pf, "_load_pfr", lambda season, st, level="week": world.pfr_rush if st == "rush" else world.pfr_rec)
    assert pf.player_feature_table(SEASON)
    for key, before in snapshot.items():
        pd.testing.assert_frame_equal(getattr(world, key), before, check_like=False)


# ---------------------------------------------------------------------------
# find_player_features
# ---------------------------------------------------------------------------
def _entry(name, team, position, **features):
    return {"name": name, "team": team, "position": position, "features": dict(features) or {"qb_efficiency": 0.1},
            "kinds": {}, "basis": {}, "games": 17, "season": 2025, "through_week": None}


NAME_TABLE = {
    "00-0001": _entry("A.J. Brown", "PHI", "WR"),
    "00-0002": _entry("Marvin Mims Jr.", "DEN", "WR"),
    "00-0003": _entry("Mike Williams", "NYJ", "WR"),
    "00-0004": _entry("Mike Williams", "PIT", "WR"),
    "00-0005": _entry("Josh Allen", "BUF", "QB"),
    "00-0006": _entry("Josh Allen", "JAX", "TE"),
    "00-0007": _entry("Amon-Ra St. Brown", "DET", "WR"),
    "00-0008": _entry("Kenneth Walker III", "SEA", "RB"),
    "00-0009": _entry("Antonio Brown", "FA", "WR"),
    "00-0010": _entry("Jaylen Waddle", "MIA", "WR"),
    "00-0011": _entry("José Müller", "LA", "TE"),
}


def test_find_by_id_comes_first_and_returns_a_copy():
    got = pf.find_player_features(NAME_TABLE, player_id="00-0005", name="Somebody Else")
    assert got["name"] == "Josh Allen" and got["player_id"] == "00-0005" and got["team"] == "BUF"
    got["features"]["qb_efficiency"] = 9.9
    assert NAME_TABLE["00-0005"]["features"]["qb_efficiency"] == 0.1 and "player_id" not in NAME_TABLE["00-0005"]
    assert pf.find_player_features(NAME_TABLE, player_id="00-9999") is None  # unknown id, no name: nothing guessed


@pytest.mark.parametrize(
    "query, expected",
    [
        ("A.J. Brown", "00-0001"), ("AJ Brown", "00-0001"), ("a.j. brown", "00-0001"), ("A J Brown", "00-0001"),
        ("Marvin Mims", "00-0002"), ("Marvin Mims Jr", "00-0002"), ("MARVIN MIMS JR.", "00-0002"),
        ("Kenneth Walker", "00-0008"), ("Kenneth Walker III", "00-0008"),
        ("Amon-Ra St. Brown", "00-0007"), ("Amon Ra St Brown", "00-0007"), ("amon-ra st brown", "00-0007"),
        ("Jose Muller", "00-0011"), ("José Müller", "00-0011"),
        ("J.Waddle", "00-0010"), ("J. Waddle", "00-0010"),
    ],
)
def test_find_by_normalised_name(query, expected):
    got = pf.find_player_features(NAME_TABLE, name=query)
    assert got is not None and got["player_id"] == expected


def test_find_never_returns_a_wrong_player_on_an_ambiguous_name():
    assert pf.find_player_features(NAME_TABLE, name="Mike Williams") is None  # two of them
    assert pf.find_player_features(NAME_TABLE, name="Josh Allen") is None  # a QB and a TE in this contrived table
    assert pf.find_player_features(NAME_TABLE, name="A. Brown") is None  # A.J. Brown vs Antonio Brown
    assert pf.find_player_features(NAME_TABLE, name="Mike Williams", team="DAL") is None  # the hint rules both out
    assert pf.find_player_features(NAME_TABLE, name="Mike Williams", position="QB") is None
    assert pf.find_player_features(NAME_TABLE, name="Brown") is None  # a bare surname never matches
    assert pf.find_player_features(NAME_TABLE, name="Waddle") is None
    assert pf.find_player_features(NAME_TABLE, name="Nobody Here") is None


def test_find_uses_team_and_position_to_break_ties():
    assert pf.find_player_features(NAME_TABLE, name="Mike Williams", team="PIT")["player_id"] == "00-0004"
    assert pf.find_player_features(NAME_TABLE, name="mike williams", team="nyj")["player_id"] == "00-0003"
    assert pf.find_player_features(NAME_TABLE, name="Josh Allen", position="QB")["player_id"] == "00-0005"
    assert pf.find_player_features(NAME_TABLE, name="Josh Allen", position="TE")["player_id"] == "00-0006"
    assert pf.find_player_features(NAME_TABLE, name="Josh Allen", team="JAC")["player_id"] == "00-0006"  # legacy alias
    assert pf.find_player_features(NAME_TABLE, name="A. Brown", team="PHI")["player_id"] == "00-0001"
    assert pf.find_player_features(NAME_TABLE, name="Kenneth Walker", position="HB")["player_id"] == "00-0008"


def test_find_treats_a_contradicting_hint_as_a_miss_rather_than_a_guess():
    # the only "Kenneth Walker" in the table is on SEA; a caller that says DAL may mean somebody else
    assert pf.find_player_features(NAME_TABLE, name="Kenneth Walker", team="DAL") is None
    assert pf.find_player_features(NAME_TABLE, name="Kenneth Walker", position="WR") is None
    assert pf.find_player_features(NAME_TABLE, name="Kenneth Walker", position="K") is None
    # leave the hint off to accept a unique name wherever the player now plays
    assert pf.find_player_features(NAME_TABLE, name="Kenneth Walker")["player_id"] == "00-0008"


def test_find_handles_empty_and_missing_arguments():
    assert pf.find_player_features(None, name="A.J. Brown") is None
    assert pf.find_player_features({}, name="A.J. Brown") is None
    assert pf.find_player_features(NAME_TABLE) is None
    assert pf.find_player_features(NAME_TABLE, name="   ") is None
    assert pf.find_player_features(NAME_TABLE, player_id=None, name=None, team="PHI") is None


def test_find_on_a_real_shaped_table(table):
    target = table[pid("WR1", 6)]
    got = pf.find_player_features(table, name=target["name"], team=target["team"], position="WR")
    assert got["player_id"] == pid("WR1", 6) and got["features"] == target["features"]
    assert pf.find_player_features(table, name=target["name"].upper())["player_id"] == pid("WR1", 6)
    assert pf.find_player_features(table, player_id=pid("RB", 0))["position"] == "RB"
    assert table[pid("RB", 0)]["position"] == "RB"  # roster said "HB"
