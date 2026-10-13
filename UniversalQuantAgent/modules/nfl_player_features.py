"""Player-level feature table for the NFL model upgrade pack (QB, WR/TE, RB).

What it does
------------
:func:`player_feature_table` turns real nflverse data (read through
``nflreadpy``) into the *player* half of the canonical input vocabulary of
``docs/nfl_model_upgrade_pack.md`` section 4, keyed by nflverse GSIS player id::

    {"00-0033873": {"name": "Patrick Mahomes", "team": "KC", "position": "QB",
                    "features": {"qb_time_to_throw": 2.71, ...},
                    "kinds":    {"qb_time_to_throw": "real", ...},
                    "basis":    {"qb_time_to_throw": "NGS passing ...", ...},
                    "games": 17, "season": 2025, "through_week": None}}

Feature keys (all snake_case, as in the contract)
-------------------------------------------------
QB      qb_time_to_throw, qb_release_time, qb_pressure_epa, qb_scramble_rate,
        qb_sack_avoidance, qb_efficiency, qb_deep_attempt_rate, qb_rz_efficiency
WR / TE wr_separation, wr_deep_route_rate, wr_yprr_vs_man, wr_yprr_vs_zone
TE only te_route_depth, te_target_share_under_pressure, te_rz_target_share
RB      rb_yards_after_contact, rb_breakaway_rate, rb_rz_efficiency

(A TE also receives the ``wr_*`` keys, shrunk toward the *TE* league mean, because
the contract lists them as "WR/TE keys"; ``te_*`` keys are TE only.)

Honesty rules (``offline_data_contract.md`` rules 3-5)
-------------------------------------------------------
* **One fail-soft I/O boundary** of small private ``_load_*`` functions; every
  one swallows provider errors and returns an empty frame, and the public
  functions never raise (an outage yields ``{}``). Tests monkeypatch the
  ``_load_*`` functions; nothing here imports an HTTP library.
* **kind ``"real"``** is reserved for a number a public feed publishes directly:
  NGS ``avg_time_to_throw`` and ``avg_separation``, PFR rushing yards after contact
  per attempt (and PFR ``adot`` when a whole season is asked for). Everything else
  is a **computed ``"proxy"``** from play-by-play (+ participation) and says so.
* **Every number carries a ``basis`` string**: feed, season, sample size, the raw
  value, the league reference it was shrunk toward, and the shrinkage weight.
* **Small samples are shrunk, not trusted.** Each value is
  ``w * raw + (1 - w) * league_mean`` with ``w = n / (n + K)`` where ``n`` is the
  feature's own opportunity count and ``league_mean`` is the volume-weighted mean
  of that feature among *qualifying* players at the same position (>= the gate).
  ``K`` per feature is :data:`SHRINKAGE_K`; it is the odd/even-week split-half
  estimate ``K = n_half * (1 - r) / r`` measured on 2022-2025 regular-season
  nflverse data (the derivation and n are in the comment next to the constant).
  A "real" feature is shrunk the same way; its basis string shows the raw
  published number, so nothing is hidden.
* **Volume gates**: below the gate a feature is *omitted* rather than shrunk to
  nothing (a missing feature is neutral downstream; a shrunk-to-nothing one would
  look measured). The gate is ``max(MIN_VOLUME, ceil(K * w / (1 - w)))`` with
  ``w = MIN_WEIGHT = 0.15`` (:data:`EFFECTIVE_GATE`), so a feature whose
  split-half reliability is poor (``qb_rz_efficiency``, ``rb_rz_efficiency``,
  ``qb_sack_avoidance``, ``rb_yards_after_contact``) needs far more volume than
  the base gate -- and ``qb_rz_efficiency`` is, honestly, almost never emitted.
  A feature is also omitted when fewer than :data:`MIN_LEAGUE_PLAYERS` players
  clear the gate (no reference).
* **No leakage.** ``through_week=N`` uses regular-season weeks ``1..N`` only. NGS
  season rows (``week == 0``) are used only when they cannot contain later
  weeks; otherwise the weekly rows are aggregated (see "Known limits").
* **Participation** (``load_participation``, 2016 onward; the feed does not yet
  have the in-progress season) feeds ``qb_release_time``, ``qb_pressure_epa``,
  ``wr_yprr_vs_*`` and ``te_target_share_under_pressure``. If the requested
  season has no participation rows the whole block falls back to the latest
  earlier season that does (up to two back); every such basis string names that
  season. If none exists the block is omitted.

Known limits (also in each basis string)
----------------------------------------
* ``wr_yprr_vs_*`` is **yards per *route-proxy***: the denominator is the
  offense's dropbacks while the player was on the field (participation lists who
  was on the field, not who ran a route). It over-counts blocking snaps, most for
  TEs and for RBs; it is comparable between players of the same position only.
* NGS publishes weekly rows only for weeks with qualifying volume (roughly >= 5
  targets / ~15 attempts), so a ``through_week`` aggregate of NGS separation or
  time-to-throw covers only those weeks. The season row, used when no later week
  can leak in, covers every tracked play.
* ``te_rz_target_share`` divides by the team's red-zone targets in the games the
  player had offensive snaps (snap counts); if the snap feed is unavailable the
  games are approximated by weeks with a play-by-play touch, which biases the
  share upward for a low-usage player, and the basis string says so.
* The PFR -> GSIS id bridge is the nflverse players/rosters tables; unmatched PFR
  ids are dropped, not guessed.
"""
from __future__ import annotations

import copy
import logging
import math
import re
import threading
import time
import unicodedata
from datetime import date
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

LOG = logging.getLogger("nfl_pack.player_features")

# ---------------------------------------------------------------------------
# Vocabulary, gates, shrinkage constants
# ---------------------------------------------------------------------------
QB_KEYS: tuple[str, ...] = (
    "qb_time_to_throw",
    "qb_release_time",
    "qb_pressure_epa",
    "qb_scramble_rate",
    "qb_sack_avoidance",
    "qb_efficiency",
    "qb_deep_attempt_rate",
    "qb_rz_efficiency",
)
WR_KEYS: tuple[str, ...] = ("wr_separation", "wr_deep_route_rate", "wr_yprr_vs_man", "wr_yprr_vs_zone")
TE_KEYS: tuple[str, ...] = ("te_route_depth", "te_target_share_under_pressure", "te_rz_target_share")
RB_KEYS: tuple[str, ...] = ("rb_yards_after_contact", "rb_breakaway_rate", "rb_rz_efficiency")

#: Feature keys each position can receive.
FEATURE_KEYS_BY_POSITION: dict[str, tuple[str, ...]] = {
    "QB": QB_KEYS,
    "WR": WR_KEYS,
    "TE": WR_KEYS + TE_KEYS,
    "RB": RB_KEYS,
}
POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# feature -> positions whose league mean it is shrunk toward
_KEY_POSITIONS: dict[str, tuple[str, ...]] = {key: ("QB",) for key in QB_KEYS}
_KEY_POSITIONS.update({key: ("WR", "TE") for key in WR_KEYS})
_KEY_POSITIONS.update({key: ("TE",) for key in TE_KEYS})
_KEY_POSITIONS.update({key: ("RB",) for key in RB_KEYS})

#: Features published *directly* by a public feed (everything else is a proxy).
REAL_KEYS: frozenset[str] = frozenset({"qb_time_to_throw", "wr_separation", "rb_yards_after_contact"})

#: Display unit used in basis strings.
_UNITS: dict[str, str] = {
    "qb_time_to_throw": "s",
    "qb_release_time": "s",
    "qb_pressure_epa": "EPA/dropback",
    "qb_scramble_rate": "",
    "qb_sack_avoidance": "",
    "qb_efficiency": "EPA/dropback",
    "qb_deep_attempt_rate": "",
    "qb_rz_efficiency": "",
    "wr_separation": "yd",
    "wr_deep_route_rate": "",
    "wr_yprr_vs_man": "yd/route",
    "wr_yprr_vs_zone": "yd/route",
    "te_route_depth": "yd",
    "te_target_share_under_pressure": "",
    "te_rz_target_share": "",
    "rb_yards_after_contact": "yd/att",
    "rb_breakaway_rate": "",
    "rb_rz_efficiency": "",
}

#: Minimum opportunity count ``n`` below which a feature is omitted (never shrunk to nothing).
#: ``n`` is the feature's own denominator, named in :data:`_OPPORTUNITY`.
MIN_VOLUME: dict[str, int] = {
    "qb_time_to_throw": 50,  # NGS attempts
    "qb_release_time": 40,  # unpressured throws with a tracked time_to_throw
    "qb_pressure_epa": 20,  # pressured dropbacks
    "qb_scramble_rate": 50,  # dropbacks
    "qb_sack_avoidance": 50,  # dropbacks
    "qb_efficiency": 50,  # dropbacks
    "qb_deep_attempt_rate": 50,  # attempts with a charted air_yards
    "qb_rz_efficiency": 15,  # red-zone dropbacks
    "wr_separation": 20,  # NGS targets
    "wr_deep_route_rate": 20,  # targets with a charted air_yards
    "wr_yprr_vs_man": 40,  # route-proxy snaps vs man
    "wr_yprr_vs_zone": 60,  # route-proxy snaps vs zone
    "te_route_depth": 20,  # targets
    "te_target_share_under_pressure": 30,  # team pressured targets while the TE was on the field
    "te_rz_target_share": 12,  # team red-zone targets in the TE's games
    "rb_yards_after_contact": 30,  # carries
    "rb_breakaway_rate": 30,  # carries
    "rb_rz_efficiency": 10,  # red-zone carries
}

_OPPORTUNITY: dict[str, str] = {
    "qb_time_to_throw": "attempts",
    "qb_release_time": "unpressured throws",
    "qb_pressure_epa": "pressured dropbacks",
    "qb_scramble_rate": "dropbacks",
    "qb_sack_avoidance": "dropbacks",
    "qb_efficiency": "dropbacks",
    "qb_deep_attempt_rate": "attempts",
    "qb_rz_efficiency": "red-zone dropbacks",
    "wr_separation": "targets",
    "wr_deep_route_rate": "targets",
    "wr_yprr_vs_man": "route-proxy snaps vs man",
    "wr_yprr_vs_zone": "route-proxy snaps vs zone",
    "te_route_depth": "targets",
    "te_target_share_under_pressure": "team pressured targets while on the field",
    "te_rz_target_share": "team red-zone targets in his games",
    "rb_yards_after_contact": "carries",
    "rb_breakaway_rate": "carries",
    "rb_rz_efficiency": "red-zone carries",
}

#: Empirical-Bayes shrinkage constant K per feature: ``w = n / (n + K)``.
#: Derived as ``K = n_half * (1 - r) / r``: ``r`` is the Pearson correlation, across
#: qualifying player-seasons (both halves >= gate / 2 opportunities), between a
#: player's value in odd regular-season weeks and in even weeks, each season
#: centred on its own mean; pooled over nflverse seasons 2022-2025. ``n_half`` is
#: the mean per-half opportunity count. Measured (N player-seasons, r with a 90%
#: bootstrap interval, n_half, raw K), rounded; K is clipped to [5, 1000]:
#:
#:   qb_time_to_throw     N=196  r=0.60 [0.50, 0.71]  n_half=171   K=113   (NGS weekly rows)
#:   qb_release_time      N=198  r=0.44 [0.32, 0.54]  n_half=128   K=166
#:   qb_pressure_epa      N=193  r=0.13 [0.01, 0.25]  n_half=57    K=379
#:   qb_scramble_rate     N=206  r=0.66 [0.57, 0.73]  n_half=188   K=98
#:   qb_sack_avoidance    N=206  r=0.18 [0.06, 0.30]  n_half=188   K=845
#:   qb_efficiency        N=206  r=0.50 [0.39, 0.60]  n_half=188   K=188
#:   qb_deep_attempt_rate N=202  r=0.24 [0.07, 0.38]  n_half=169   K=536
#:   qb_rz_efficiency     N=161  r=-0.05 [-0.21, 0.09] n_half=30   K=inf -> 1000 (no measurable skill signal)
#:   wr_separation        N=460  r=0.46 [0.39, 0.52]  n_half=37    K=43    (NGS weekly rows)
#:   wr_deep_route_rate   N=708  r=0.58 [0.52, 0.63]  n_half=35    K=26
#:   wr_yprr_vs_man       N=784  r=0.45 [0.39, 0.50]  n_half=66    K=82
#:   wr_yprr_vs_zone      N=851  r=0.45 [0.40, 0.51]  n_half=103   K=126
#:   te_route_depth       N=214  r=0.59 [0.46, 0.70]  n_half=30    K=20
#:   te_target_share_under_pressure N=222 r=0.27 [0.13, 0.37] n_half=34 K=95
#:   te_rz_target_share   N=469  r=0.71 [0.67, 0.75]  n_half=28    K=11
#:   rb_yards_after_contact N=301 r=0.12 [0.00, 0.24] n_half=72    K=537   (PFR weekly rows)
#:   rb_breakaway_rate    N=301  r=0.22 [0.11, 0.32]  n_half=72    K=255
#:   rb_rz_efficiency     N=213  r=0.05 [-0.08, 0.18] n_half=16    K=294
#:
#: Where r is small the interval is wide and K is therefore poorly determined;
#: the point estimate is used (and the weight floor below does the rest).
SHRINKAGE_K: dict[str, float] = {
    "qb_time_to_throw": 113.0,
    "qb_release_time": 166.0,
    "qb_pressure_epa": 379.0,
    "qb_scramble_rate": 98.0,
    "qb_sack_avoidance": 845.0,
    "qb_efficiency": 188.0,
    "qb_deep_attempt_rate": 536.0,
    "qb_rz_efficiency": 1000.0,
    "wr_separation": 43.0,
    "wr_deep_route_rate": 26.0,
    "wr_yprr_vs_man": 82.0,
    "wr_yprr_vs_zone": 126.0,
    "te_route_depth": 20.0,
    "te_target_share_under_pressure": 95.0,
    "te_rz_target_share": 11.0,
    "rb_yards_after_contact": 537.0,
    "rb_breakaway_rate": 255.0,
    "rb_rz_efficiency": 294.0,
}

#: A value whose shrinkage weight ``n / (n + K)`` would be below this is "shrunk to
#: nothing" (>85% league mean): the feature is omitted instead. Together with
#: :data:`MIN_VOLUME` this gives :data:`EFFECTIVE_GATE`.
MIN_WEIGHT = 0.15


def _effective_gate(key: str) -> int:
    return max(int(MIN_VOLUME[key]), int(math.ceil(SHRINKAGE_K[key] * MIN_WEIGHT / (1.0 - MIN_WEIGHT))))


#: The opportunity count a player needs for a feature to be emitted:
#: ``max(MIN_VOLUME, ceil(K * MIN_WEIGHT / (1 - MIN_WEIGHT)))``.
EFFECTIVE_GATE: dict[str, int] = {key: _effective_gate(key) for key in MIN_VOLUME}

#: A feature needs at least this many qualifying players at a position to have a
#: league reference; below that it is omitted for everyone.
MIN_LEAGUE_PLAYERS = 8
#: Seconds a built table stays cached (an empty / failed build is never cached).
CACHE_SECONDS = 3600.0

_REG_MAX_WEEK = 18
_RZ_YARDLINE = 20.0
_DEEP_AIR_YARDS = 20.0
_BREAKAWAY_YARDS = 15.0
_PARTICIPATION_LOOKBACK = 3  # requested season, then up to two earlier ones
_COVERAGE_TYPES = {"MAN_COVERAGE": "man", "ZONE_COVERAGE": "zone"}

FEATURE_BASIS: dict[str, str] = {
    "qb_time_to_throw": (
        "real: NGS passing avg_time_to_throw (snap-to-release seconds), regular season, attempt-weighted; "
        f"shrunk toward the QB league mean; gate >= {EFFECTIVE_GATE['qb_time_to_throw']} attempts"
    ),
    "qb_release_time": (
        "proxy: median participation time_to_throw on unpressured throws (was_pressure False), joined to play-by-play; "
        f"gate >= {EFFECTIVE_GATE['qb_release_time']} throws; latest participation season if the requested one has none"
    ),
    "qb_pressure_epa": (
        "proxy: mean pbp EPA per dropback when participation was_pressure is True (sacks included); "
        f"gate >= {EFFECTIVE_GATE['qb_pressure_epa']} pressured dropbacks; latest participation season if needed"
    ),
    "qb_scramble_rate": (
        f"proxy: pbp qb_scramble per dropback; gate >= {EFFECTIVE_GATE['qb_scramble_rate']} dropbacks"
    ),
    "qb_sack_avoidance": (
        f"proxy: 1 - pbp sacks per dropback; gate >= {EFFECTIVE_GATE['qb_sack_avoidance']} dropbacks"
    ),
    "qb_efficiency": (
        f"proxy: pbp EPA per dropback (sacks and scrambles included); gate >= {EFFECTIVE_GATE['qb_efficiency']} dropbacks"
    ),
    "qb_deep_attempt_rate": (
        f"proxy: share of pbp pass attempts with air_yards >= {_DEEP_AIR_YARDS:.0f}; "
        f"gate >= {EFFECTIVE_GATE['qb_deep_attempt_rate']} attempts"
    ),
    "qb_rz_efficiency": (
        "proxy: touchdowns (pass, or QB scramble) per red-zone dropback (yardline_100 <= 20); "
        f"gate >= {EFFECTIVE_GATE['qb_rz_efficiency']} red-zone dropbacks"
    ),
    "wr_separation": (
        "real: NGS receiving avg_separation (yards at the catch point), regular season, target-weighted; "
        f"gate >= {EFFECTIVE_GATE['wr_separation']} targets"
    ),
    "wr_deep_route_rate": (
        f"proxy: share of the player's pbp targets with air_yards >= {_DEEP_AIR_YARDS:.0f} "
        f"(not a route-tree measure); gate >= {EFFECTIVE_GATE['wr_deep_route_rate']} targets"
    ),
    "wr_yprr_vs_man": (
        "proxy: receiving yards on pbp targets / dropbacks the player was on the field for (participation "
        "offense_players) vs man coverage; the denominator over-counts blocking snaps; "
        f"gate >= {EFFECTIVE_GATE['wr_yprr_vs_man']} snaps"
    ),
    "wr_yprr_vs_zone": (
        "proxy: as wr_yprr_vs_man but vs zone coverage; "
        f"gate >= {EFFECTIVE_GATE['wr_yprr_vs_zone']} snaps"
    ),
    "te_route_depth": (
        "real when a full season is asked for (PFR receiving adot), else proxy: mean pbp air_yards on the player's "
        f"targets; gate >= {EFFECTIVE_GATE['te_route_depth']} targets"
    ),
    "te_target_share_under_pressure": (
        "proxy: the TE's share of his team's pressured-dropback targets on plays he was on the field "
        f"(participation was_pressure); gate >= {EFFECTIVE_GATE['te_target_share_under_pressure']} such team targets"
    ),
    "te_rz_target_share": (
        "proxy: the TE's share of his team's red-zone targets (yardline_100 <= 20) in the games he had offensive "
        f"snaps; gate >= {EFFECTIVE_GATE['te_rz_target_share']} team red-zone targets"
    ),
    "rb_yards_after_contact": (
        "real: PFR rushing yards after contact / carries (published yac_att; weekly rows summed), regular season; "
        f"gate >= {EFFECTIVE_GATE['rb_yards_after_contact']} carries"
    ),
    "rb_breakaway_rate": (
        f"proxy: share of designed pbp carries gaining >= {_BREAKAWAY_YARDS:.0f} yards; "
        f"gate >= {EFFECTIVE_GATE['rb_breakaway_rate']} carries"
    ),
    "rb_rz_efficiency": (
        "proxy: rushing touchdowns per red-zone carry (yardline_100 <= 20); "
        f"gate >= {EFFECTIVE_GATE['rb_rz_efficiency']} red-zone carries"
    ),
}

_PBP_COLUMNS: tuple[str, ...] = (
    "game_id", "play_id", "season", "season_type", "week", "posteam", "play_type",
    "qb_dropback", "qb_scramble", "sack", "complete_pass", "air_yards", "yards_gained", "epa",
    "yardline_100", "pass_touchdown", "rush_touchdown", "two_point_attempt",
    "passer_player_id", "receiver_player_id", "rusher_player_id", "id",
)

# ---------------------------------------------------------------------------
# Fail-soft I/O boundary (tests monkeypatch these; nothing else touches a provider)
# ---------------------------------------------------------------------------
_EMPTY = pd.DataFrame()


def _to_pandas(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


def _load_pbp(season: int) -> pd.DataFrame:
    """Play-by-play for one season (projected to the columns this module reads); empty on any failure."""
    try:
        import nflreadpy as nfl

        data = nfl.load_pbp([int(season)])
        if hasattr(data, "select") and hasattr(data, "columns"):
            keep = [c for c in _PBP_COLUMNS if c in data.columns]
            data = data.select(keep)
        return _to_pandas(data)
    except Exception as error:  # provider down, season missing, dependency absent
        LOG.debug("pbp %s unavailable: %s", season, error)
        return pd.DataFrame()


def _load_ngs(season: int, stat_type: str) -> pd.DataFrame:
    """NGS ``passing`` / ``receiving`` / ``rushing`` rows (week 0 = season aggregate)."""
    try:
        import nflreadpy as nfl

        return _to_pandas(nfl.load_nextgen_stats([int(season)], stat_type=stat_type))
    except Exception as error:
        LOG.debug("ngs %s %s unavailable: %s", season, stat_type, error)
        return pd.DataFrame()


def _load_pfr(season: int, stat_type: str, summary_level: str = "week") -> pd.DataFrame:
    """PFR advanced stats (``pass`` / ``rush`` / ``rec`` / ``def``) at ``week`` or ``season`` level."""
    try:
        import nflreadpy as nfl

        return _to_pandas(nfl.load_pfr_advstats([int(season)], stat_type=stat_type, summary_level=summary_level))
    except Exception as error:
        LOG.debug("pfr %s %s %s unavailable: %s", season, stat_type, summary_level, error)
        return pd.DataFrame()


def _load_participation(season: int) -> pd.DataFrame:
    """Play-level participation (``nflverse_game_id``, ``play_id``, ``time_to_throw``, ...); empty if absent."""
    try:
        import nflreadpy as nfl

        return _to_pandas(nfl.load_participation([int(season)]))
    except Exception as error:  # e.g. ValueError: season out of range for the in-progress year
        LOG.debug("participation %s unavailable: %s", season, error)
        return pd.DataFrame()


def _load_rosters(season: int) -> pd.DataFrame:
    """Season rosters (``gsis_id``, ``pfr_id``, ``full_name``, ``position``, ``team``, ``week``)."""
    try:
        import nflreadpy as nfl

        return _to_pandas(nfl.load_rosters([int(season)]))
    except Exception as error:
        LOG.debug("rosters %s unavailable: %s", season, error)
        return pd.DataFrame()


def _load_players() -> pd.DataFrame:
    """All-time player table (``gsis_id``, ``pfr_id``, ``display_name``, ``position``, ``latest_team``)."""
    try:
        import nflreadpy as nfl

        return _to_pandas(nfl.load_players())
    except Exception as error:
        LOG.debug("players unavailable: %s", error)
        return pd.DataFrame()


def _load_snaps(season: int) -> pd.DataFrame:
    """Per player-game snap counts (``pfr_player_id``, ``week``, ``team``, ``offense_snaps``)."""
    try:
        import nflreadpy as nfl

        return _to_pandas(nfl.load_snap_counts([int(season)]))
    except Exception as error:
        LOG.debug("snap counts %s unavailable: %s", season, error)
        return pd.DataFrame()


def _safe(loader: Callable[..., Any], *args: Any) -> pd.DataFrame:
    """Call a (possibly monkeypatched) loader; any failure or non-frame result is an empty frame."""
    try:
        frame = loader(*args)
    except Exception as error:
        LOG.debug("loader %s failed: %s", getattr(loader, "__name__", loader), error)
        return pd.DataFrame()
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Small frame helpers
# ---------------------------------------------------------------------------
def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].astype("string").fillna("").str.strip()
    return pd.Series("", index=frame.index, dtype="string")


def _ids(frame: pd.DataFrame, column: str) -> pd.Series:
    """String ids as an object column with ``None`` for missing / blank."""
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype=object)
    text = frame[column].astype("string").str.strip()
    return text.astype(object).where(text.notna() & (text != ""), None)


def _week_predicate(through_week: int | None) -> Callable[[pd.Series], pd.Series]:
    if through_week is None:
        return lambda week: week >= 1
    return lambda week: (week >= 1) & (week <= through_week)


def _default_season(today: date | None = None) -> int:
    """Latest *complete* season (matches ``modules.nfl.default_analysis_season``)."""
    today = today or date.today()
    return today.year - 1 if today.month >= 2 else today.year - 2


# ---------------------------------------------------------------------------
# Play-by-play preparation
# ---------------------------------------------------------------------------
_PBP_OUT = (
    "game_id", "play_id", "week", "posteam", "dropback", "sack", "scramble", "attempt", "target", "carry",
    "redzone", "complete", "pass_td", "rush_td", "epa", "air_yards", "yards", "qb_id", "recv_id", "rush_id",
)


def _empty_pbp() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in _PBP_OUT})


def _prepare_pbp(pbp: pd.DataFrame, week_ok: Callable[[pd.Series], pd.Series]) -> pd.DataFrame:
    """Regular-season run/pass plays within ``week_ok`` with the flags every feature reads."""
    if pbp is None or len(pbp) == 0:
        return _empty_pbp()
    week = _num(pbp, "week")
    keep = week_ok(week).fillna(False).astype(bool)
    if "season_type" in pbp.columns:
        keep &= _text(pbp, "season_type").str.upper().eq("REG").to_numpy(dtype=bool)
    else:
        keep &= (week <= _REG_MAX_WEEK).fillna(False).to_numpy(dtype=bool)
    play_type = _text(pbp, "play_type").str.lower()
    keep &= play_type.isin(["pass", "run"]).to_numpy(dtype=bool)
    keep &= ~_num(pbp, "two_point_attempt").fillna(0).eq(1).to_numpy(dtype=bool)
    p = pbp.loc[keep]
    if p.empty:
        return _empty_pbp()
    pt = _text(p, "play_type").str.lower()
    dropback = _num(p, "qb_dropback").fillna(0).eq(1)
    sack = _num(p, "sack").fillna(0).eq(1)
    scramble = _num(p, "qb_scramble").fillna(0).eq(1)
    attempt = dropback & ~sack & ~scramble & pt.eq("pass")
    passer, rusher, recv, any_id = _ids(p, "passer_player_id"), _ids(p, "rusher_player_id"), _ids(p, "receiver_player_id"), _ids(p, "id")
    qb_id = passer.where(passer.notna(), rusher.where(scramble))
    qb_id = qb_id.where(qb_id.notna(), any_id).where(dropback)
    out = pd.DataFrame(
        {
            "game_id": _ids(p, "game_id"),
            "play_id": _num(p, "play_id"),
            "week": _num(p, "week"),
            "posteam": _ids(p, "posteam"),
            "dropback": dropback,
            "sack": sack,
            "scramble": scramble,
            "attempt": attempt,
            "target": attempt & recv.notna(),
            "carry": pt.eq("run") & ~dropback & rusher.notna(),
            "redzone": (_num(p, "yardline_100") <= _RZ_YARDLINE).fillna(False),
            "complete": _num(p, "complete_pass").fillna(0).eq(1),
            "pass_td": _num(p, "pass_touchdown").fillna(0).eq(1),
            "rush_td": _num(p, "rush_touchdown").fillna(0).eq(1),
            "epa": _num(p, "epa"),
            "air_yards": _num(p, "air_yards"),
            "yards": _num(p, "yards_gained"),
            "qb_id": qb_id,
            "recv_id": recv,
            "rush_id": rusher.where(pt.eq("run") & ~dropback),
        },
        index=p.index,
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Raw (unshrunk) per-player statistics:  feature -> {player_id: (value, n)}
# ---------------------------------------------------------------------------
Raw = dict[str, dict[str, tuple[float, float]]]


def _pack(value: pd.Series, n: pd.Series) -> dict[str, tuple[float, float]]:
    joined = pd.concat([value.rename("v"), n.rename("n")], axis=1).dropna()
    joined = joined[joined["n"] > 0]
    return {str(pid): (float(row.v), float(row.n)) for pid, row in joined.iterrows()}


def _qb_pbp_raw(df: pd.DataFrame) -> Raw:
    out: Raw = {}
    db = df[df["dropback"] & df["qb_id"].notna()]
    if db.empty:
        return out
    g = db.groupby("qb_id")
    n_db = g.size()
    epa = db[db["epa"].notna()].groupby("qb_id")["epa"].agg(["mean", "count"])
    out["qb_efficiency"] = _pack(epa["mean"], epa["count"])
    out["qb_scramble_rate"] = _pack(g["scramble"].mean(), n_db)
    out["qb_sack_avoidance"] = _pack(1.0 - g["sack"].mean(), n_db)
    att = db[db["attempt"] & db["air_yards"].notna()]
    if not att.empty:
        deep = (att["air_yards"] >= _DEEP_AIR_YARDS).groupby(att["qb_id"]).agg(["mean", "size"])
        out["qb_deep_attempt_rate"] = _pack(deep["mean"], deep["size"])
    rz = db[db["redzone"]]
    if not rz.empty:
        td = (rz["pass_td"] | (rz["rush_td"] & rz["scramble"])).groupby(rz["qb_id"]).agg(["mean", "size"])
        out["qb_rz_efficiency"] = _pack(td["mean"], td["size"])
    return out


def _receiver_pbp_raw(df: pd.DataFrame) -> Raw:
    out: Raw = {}
    tg = df[df["target"] & df["recv_id"].notna()]
    if tg.empty:
        return out
    charted = tg[tg["air_yards"].notna()]
    if not charted.empty:
        deep = (charted["air_yards"] >= _DEEP_AIR_YARDS).groupby(charted["recv_id"]).agg(["mean", "size"])
        out["wr_deep_route_rate"] = _pack(deep["mean"], deep["size"])
        adot = charted.groupby("recv_id")["air_yards"].agg(["mean", "count"])
        out["te_route_depth"] = _pack(adot["mean"], adot["count"])
    return out


def _rb_pbp_raw(df: pd.DataFrame) -> Raw:
    out: Raw = {}
    cr = df[df["carry"] & df["rush_id"].notna() & df["yards"].notna()]
    if cr.empty:
        return out
    brk = (cr["yards"] >= _BREAKAWAY_YARDS).groupby(cr["rush_id"]).agg(["mean", "size"])
    out["rb_breakaway_rate"] = _pack(brk["mean"], brk["size"])
    rz = cr[cr["redzone"]]
    if not rz.empty:
        td = rz["rush_td"].groupby(rz["rush_id"]).agg(["mean", "size"])
        out["rb_rz_efficiency"] = _pack(td["mean"], td["size"])
    return out


def _ngs_raw(
    frame: pd.DataFrame, value_col: str, weight_col: str, through_week: int | None
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    """``{gsis: (value, n)}`` and ``{gsis: "season row" | "weekly rows"}`` from NGS rows.

    The week-0 season row (every tracked play) is used only when it cannot contain
    a week after ``through_week``; otherwise weekly rows (qualifying weeks only)
    are aggregated, weighted by ``weight_col``.
    """
    if frame is None or frame.empty or "player_gsis_id" not in frame.columns:
        return {}, {}
    week = _num(frame, "week")
    reg = (
        _text(frame, "season_type").str.upper().eq("REG")
        if "season_type" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    value, weight, pid = _num(frame, value_col), _num(frame, weight_col), _ids(frame, "player_gsis_id")
    usable = reg & value.notna() & weight.gt(0) & pid.notna()
    weekly = usable & (week >= 1)
    max_week = week[weekly].max() if weekly.any() else np.nan
    in_window = weekly if through_week is None else weekly & (week <= through_week)
    values: dict[str, tuple[float, float]] = {}
    where: dict[str, str] = {}
    if in_window.any():
        d = pd.DataFrame({"pid": pid[in_window], "vw": (value * weight)[in_window], "w": weight[in_window]})
        agg = d.groupby("pid").agg(vw=("vw", "sum"), w=("w", "sum"))
        for key, row in agg.iterrows():
            values[str(key)] = (float(row.vw / row.w), float(row.w))
            where[str(key)] = "weekly rows"
    season_ok = through_week is None or (not np.isnan(max_week) and through_week >= max_week)
    if season_ok:
        season_rows = usable & week.eq(0)
        for key, v, w in zip(pid[season_rows], value[season_rows], weight[season_rows]):
            values[str(key)] = (float(v), float(w))
            where[str(key)] = "season row"
    return values, where


def _pfr_rush_raw(frame: pd.DataFrame, week_ok: Callable[[pd.Series], pd.Series]) -> dict[str, tuple[float, float]]:
    """``{pfr_id: (yards_after_contact_per_carry, carries)}`` summed over regular-season weekly rows."""
    if frame is None or frame.empty or "pfr_player_id" not in frame.columns:
        return {}
    week = _num(frame, "week")
    keep = week_ok(week).fillna(False).astype(bool)
    if "game_type" in frame.columns:
        keep &= _text(frame, "game_type").str.upper().eq("REG").to_numpy(dtype=bool)
    carries, yac = _num(frame, "carries"), _num(frame, "rushing_yards_after_contact")
    keep &= (carries.notna() & yac.notna()).to_numpy(dtype=bool)
    d = pd.DataFrame({"pid": _ids(frame, "pfr_player_id")[keep], "c": carries[keep], "y": yac[keep]})
    d = d[d["pid"].notna()]
    if d.empty:
        return {}
    agg = d.groupby("pid").agg(c=("c", "sum"), y=("y", "sum"))
    agg = agg[agg["c"] > 0]
    return {str(pid): (float(row.y / row.c), float(row.c)) for pid, row in agg.iterrows()}


def _pfr_adot_raw(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """``{pfr_id: (adot, targets)}`` from the PFR season-level receiving table."""
    if frame is None or frame.empty or "pfr_id" not in frame.columns:
        return {}
    adot, tgt, pid = _num(frame, "adot"), _num(frame, "tgt"), _ids(frame, "pfr_id")
    ok = adot.notna() & tgt.gt(0) & pid.notna()
    d = pd.DataFrame({"pid": pid[ok], "v": adot[ok], "n": tgt[ok]})
    if d.empty:
        return {}
    # A traded player has per-team rows plus a total ("2TM"); the largest-n row is the total.
    d = d.sort_values("n").drop_duplicates("pid", keep="last")
    return {str(r.pid): (float(r.v), float(r.n)) for r in d.itertuples()}


def _slim_participation(part: pd.DataFrame) -> pd.DataFrame:
    if part is None or part.empty or "nflverse_game_id" not in part.columns or "play_id" not in part.columns:
        return pd.DataFrame(columns=["game_id", "play_id", "ttt", "pressured", "cov", "offense_players"])
    pressured = pd.to_numeric(part["was_pressure"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}), errors="coerce") \
        if "was_pressure" in part.columns else pd.Series(np.nan, index=part.index)
    out = pd.DataFrame(
        {
            "game_id": _ids(part, "nflverse_game_id"),
            "play_id": _num(part, "play_id"),
            "ttt": _num(part, "time_to_throw"),
            "pressured": pressured,
            "cov": _text(part, "defense_man_zone_type").str.upper().map(_COVERAGE_TYPES),
            "offense_players": _text(part, "offense_players"),
        }
    )
    return out.drop_duplicates(["game_id", "play_id"]).reset_index(drop=True)


def _participation_raw(pbp: pd.DataFrame, part: pd.DataFrame, pos_of: Mapping[str, str]) -> Raw:
    """Features that need the play-level participation join (all computed within one season)."""
    out: Raw = {}
    slim = _slim_participation(part)
    if pbp.empty or slim.empty:
        return out
    m = pbp[pbp["dropback"]].merge(slim, on=["game_id", "play_id"], how="inner")
    if m.empty:
        return out
    # --- QB: release time on unpressured throws, EPA when pressured -------------------------------
    throws = m[m["attempt"] & m["pressured"].eq(0) & m["ttt"].between(0.1, 8.0) & m["qb_id"].notna()]
    if not throws.empty:
        grp = throws.groupby("qb_id")["ttt"].agg(["median", "count"])
        out["qb_release_time"] = _pack(grp["median"], grp["count"])
    pressed = m[m["pressured"].eq(1) & m["qb_id"].notna() & m["epa"].notna()]
    if not pressed.empty:
        grp = pressed.groupby("qb_id")["epa"].agg(["mean", "count"])
        out["qb_pressure_epa"] = _pack(grp["mean"], grp["count"])
    # --- WR/TE: yards per route-proxy snap by coverage shell ----------------------------------------
    receivers = {pid for pid, pos in pos_of.items() if pos in ("WR", "TE")}
    covered = m[m["cov"].notna()]
    if receivers and not covered.empty:
        field = covered[["game_id", "play_id", "cov", "offense_players"]].copy()
        field["pid"] = field["offense_players"].str.split(";")
        field = field.explode("pid")
        field = field[field["pid"].isin(receivers)]
        snaps = field.groupby(["pid", "cov"]).size()
        hit = covered[covered["target"] & covered["recv_id"].notna()].copy()
        hit["recv_yards"] = np.where(hit["complete"], hit["yards"].fillna(0.0), 0.0)
        yards = hit.groupby(["recv_id", "cov"])["recv_yards"].sum()
        for cov, key in (("man", "wr_yprr_vs_man"), ("zone", "wr_yprr_vs_zone")):
            values: dict[str, tuple[float, float]] = {}
            for (pid, shell), n in snaps.items():
                if shell != cov or n <= 0:
                    continue
                values[str(pid)] = (float(yards.get((pid, cov), 0.0)) / float(n), float(n))
            if values:
                out[key] = values
    # --- TE: share of the team's pressured-dropback targets while on the field -----------------------
    tes = {pid for pid, pos in pos_of.items() if pos == "TE"}
    press_targets = m[m["target"] & m["pressured"].eq(1) & m["recv_id"].notna()]
    if tes and not press_targets.empty:
        field = press_targets[["recv_id", "offense_players"]].copy()
        field["pid"] = field["offense_players"].str.split(";")
        field = field.explode("pid")
        field = field[field["pid"].isin(tes)]
        on_field = field.groupby("pid").size()
        own = field[field["pid"] == field["recv_id"]].groupby("pid").size()
        values = {}
        for pid, n in on_field.items():
            values[str(pid)] = (float(own.get(pid, 0)) / float(n), float(n))
        if values:
            out["te_target_share_under_pressure"] = values
    return out


# ---------------------------------------------------------------------------
# Identity (id bridge, position, name, team, games)
# ---------------------------------------------------------------------------
def _pfr_to_gsis(rosters: pd.DataFrame, players: pd.DataFrame) -> dict[str, str]:
    """``{pfr_id: gsis_id}`` from the nflverse players table, then rosters (first writer wins)."""
    mapping: dict[str, str] = {}
    for frame in (players, rosters):
        if frame is None or frame.empty or "pfr_id" not in frame.columns or "gsis_id" not in frame.columns:
            continue
        pfr, gsis = _ids(frame, "pfr_id"), _ids(frame, "gsis_id")
        ok = pfr.notna() & gsis.notna()
        for key, value in zip(pfr[ok], gsis[ok]):
            mapping.setdefault(str(key), str(value))
    return mapping


def _clean_position(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text in ("HB", "RB"):
        return "RB"
    return text if text in POSITIONS else None


def _identity(
    rosters: pd.DataFrame, players: pd.DataFrame, ngs_frames: Iterable[pd.DataFrame], through_week: int | None
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """``(position, name, roster_team)`` per GSIS id. Roster rows after ``through_week`` are ignored."""
    position: dict[str, str] = {}
    name: dict[str, str] = {}
    team: dict[str, str] = {}
    if rosters is not None and not rosters.empty and "gsis_id" in rosters.columns:
        r = rosters.copy()
        r["gsis_key"] = _ids(r, "gsis_id")
        r = r[r["gsis_key"].notna()]
        week = _num(r, "week")
        if through_week is not None and week.notna().any():
            windowed = r[(week <= through_week) | week.isna()]
            r = windowed if not windowed.empty else r
        if "game_type" in r.columns:
            reg = r[_text(r, "game_type").str.upper().isin(["REG", ""])]
            r = reg if not reg.empty else r
        r = r.assign(week_key=_num(r, "week").fillna(0)).sort_values("week_key", kind="stable")
        for row in r.itertuples():
            pid = str(row.gsis_key)
            pos = _clean_position(getattr(row, "position", None))
            if pos:
                position[pid] = pos
            full = getattr(row, "full_name", None)
            if isinstance(full, str) and full.strip():
                name[pid] = full.strip()
            tm = getattr(row, "team", None)
            if isinstance(tm, str) and tm.strip():
                team[pid] = tm.strip().upper()
    if players is not None and not players.empty and "gsis_id" in players.columns:
        pid_col, pos_col = _ids(players, "gsis_id"), players["position"] if "position" in players.columns else None
        names = players["display_name"] if "display_name" in players.columns else None
        for i, pid in enumerate(pid_col):
            if pid is None:
                continue
            pid = str(pid)
            if pid not in position and pos_col is not None:
                pos = _clean_position(pos_col.iloc[i])
                if pos:
                    position[pid] = pos
            if pid not in name and names is not None and isinstance(names.iloc[i], str) and names.iloc[i].strip():
                name[pid] = names.iloc[i].strip()
    for frame in ngs_frames:
        if frame is None or frame.empty or "player_gsis_id" not in frame.columns:
            continue
        ids = _ids(frame, "player_gsis_id")
        pos_col = frame["player_position"] if "player_position" in frame.columns else None
        names = frame["player_display_name"] if "player_display_name" in frame.columns else None
        for i, pid in enumerate(ids):
            if pid is None:
                continue
            pid = str(pid)
            if pid not in position and pos_col is not None:
                pos = _clean_position(pos_col.iloc[i])
                if pos:
                    position[pid] = pos
            if pid not in name and names is not None and isinstance(names.iloc[i], str) and names.iloc[i].strip():
                name[pid] = names.iloc[i].strip()
    return position, name, team


def _pbp_involvement(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, week, team) in which the player was the QB, target or rusher."""
    if df.empty:
        return pd.DataFrame(columns=["pid", "week", "team"])
    parts = []
    for column in ("qb_id", "recv_id", "rush_id"):
        sub = df[df[column].notna()][[column, "week", "posteam"]]
        parts.append(sub.rename(columns={column: "pid", "posteam": "team"}))
    out = pd.concat(parts, ignore_index=True)
    out = out[out["pid"].notna() & out["team"].notna() & out["week"].notna()]
    return out.drop_duplicates().reset_index(drop=True)


def _player_games(
    snaps: pd.DataFrame,
    pfr_map: Mapping[str, str],
    week_ok: Callable[[pd.Series], pd.Series],
    involvement: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """``(DataFrame[pid, week, team], source)`` -- the games each player was on the field for."""
    if snaps is not None and not snaps.empty and "pfr_player_id" in snaps.columns:
        week = _num(snaps, "week")
        keep = week_ok(week).fillna(False).astype(bool) & _num(snaps, "offense_snaps").gt(0).to_numpy(dtype=bool)
        if "game_type" in snaps.columns:
            keep &= _text(snaps, "game_type").str.upper().eq("REG").to_numpy(dtype=bool)
        d = pd.DataFrame(
            {
                "pid": _ids(snaps, "pfr_player_id")[keep].map(lambda key: pfr_map.get(str(key)) if key is not None else None),
                "week": week[keep],
                "team": _ids(snaps, "team")[keep],
            }
        )
        d = d[d["pid"].notna() & d["team"].notna()].drop_duplicates().reset_index(drop=True)
        if not d.empty:
            return d, "snap counts"
    return involvement, "pbp involvement (approximation)"


def _te_rz_share_raw(df: pd.DataFrame, games: pd.DataFrame, tes: set[str]) -> dict[str, tuple[float, float]]:
    rz = df[df["target"] & df["redzone"] & df["recv_id"].notna() & df["posteam"].notna()]
    if rz.empty or games.empty or not tes:
        return {}
    team_rz = rz.groupby(["posteam", "week"]).size().rename("team_n").reset_index()
    own_rz = rz.groupby(["recv_id", "posteam", "week"]).size().rename("own_n").reset_index()
    pg = games[games["pid"].isin(tes)]
    if pg.empty:
        return {}
    joined = pg.merge(team_rz, left_on=["team", "week"], right_on=["posteam", "week"], how="inner")
    joined = joined.merge(
        own_rz, left_on=["pid", "team", "week"], right_on=["recv_id", "posteam", "week"], how="left", suffixes=("", "_own")
    )
    joined["own_n"] = joined["own_n"].fillna(0)
    agg = joined.groupby("pid").agg(own=("own_n", "sum"), team=("team_n", "sum"))
    agg = agg[agg["team"] > 0]
    return {str(pid): (float(r.own / r.team), float(r.team)) for pid, r in agg.iterrows()}


def _latest_team(df: pd.DataFrame) -> dict[str, str]:
    inv = _pbp_involvement(df)
    if inv.empty:
        return {}
    last = inv.sort_values("week", kind="stable").drop_duplicates("pid", keep="last")
    return {str(r.pid): str(r.team) for r in last.itertuples()}


# ---------------------------------------------------------------------------
# Shrinkage + assembly
# ---------------------------------------------------------------------------
def _format(value: float, unit: str) -> str:
    return f"{value:.3f} {unit}".strip() if unit else f"{value:.3f}"


def _shrink_and_assemble(
    raw: Raw,
    sources: Mapping[str, str],
    notes: Mapping[tuple[str, str], str],
    *,
    position: Mapping[str, str],
    name: Mapping[str, str],
    team: Mapping[str, str],
    games: Mapping[str, int],
    season: int,
    through_week: int | None,
    kind_overrides: Mapping[tuple[str, str], str],
) -> dict[str, dict[str, Any]]:
    """Apply the gates and ``n / (n + K)`` shrinkage and build the public table."""
    table: dict[str, dict[str, Any]] = {}
    for key in sorted(raw):
        if key not in _KEY_POSITIONS:
            continue
        gate, k = EFFECTIVE_GATE[key], SHRINKAGE_K[key]
        for pos in _KEY_POSITIONS[key]:
            qualifying = [
                (pid, v, n) for pid, (v, n) in raw[key].items() if position.get(pid) == pos and n >= gate
            ]
            if len(qualifying) < MIN_LEAGUE_PLAYERS:
                continue
            total_n = sum(n for _, _, n in qualifying)
            league_mean = sum(v * n for _, v, n in qualifying) / total_n
            for pid, value, n in qualifying:
                weight = n / (n + k)
                shrunk = weight * value + (1.0 - weight) * league_mean
                if not np.isfinite(shrunk):
                    continue
                entry = table.get(pid)
                if entry is None:
                    entry = {
                        "name": name.get(pid, ""),
                        "team": team.get(pid, ""),
                        "position": pos,
                        "features": {},
                        "kinds": {},
                        "basis": {},
                        "games": int(games.get(pid, 0)),
                        "season": int(season),
                        "through_week": through_week,
                    }
                    table[pid] = entry
                unit = _UNITS.get(key, "")
                kind = kind_overrides.get((pid, key), "real" if key in REAL_KEYS else "proxy")
                feed = notes.get((pid, key)) or sources.get(key, "")
                entry["features"][key] = round(float(shrunk), 4)
                entry["kinds"][key] = kind
                entry["basis"][key] = (
                    f"{kind}: {feed}; n={n:.0f} {_OPPORTUNITY[key]}; raw {_format(value, unit)}; "
                    f"shrunk toward {pos} league mean {_format(league_mean, unit)} "
                    f"({len(qualifying)} qualifying {pos}s) with weight {weight:.2f} (K={k:g})"
                )
    for pid in table:
        table[pid]["features"] = dict(sorted(table[pid]["features"].items()))
        table[pid]["kinds"] = dict(sorted(table[pid]["kinds"].items()))
        table[pid]["basis"] = dict(sorted(table[pid]["basis"].items()))
    return dict(sorted(table.items()))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
_LAST_STATS: dict[str, Any] = {}


def _compute(
    season: int, week_ok: Callable[[pd.Series], pd.Series], through_week: int | None
) -> tuple[Raw, dict[str, Any]]:
    """Gather every feed once and return ``(raw statistics, context)``; the context feeds assembly."""
    pbp = _prepare_pbp(_safe(_load_pbp, season), week_ok)
    ngs_pass = _safe(_load_ngs, season, "passing")
    ngs_rec = _safe(_load_ngs, season, "receiving")
    pfr_rush = _safe(_load_pfr, season, "rush", "week")
    rosters = _safe(_load_rosters, season)
    players = _safe(_load_players)
    snaps = _safe(_load_snaps, season)
    position, name, roster_team = _identity(rosters, players, (ngs_pass, ngs_rec), through_week)
    pfr_map = _pfr_to_gsis(rosters, players)

    raw: Raw = {}
    sources: dict[str, str] = {}
    notes: dict[tuple[str, str], str] = {}
    kind_overrides: dict[tuple[str, str], str] = {}

    raw.update(_qb_pbp_raw(pbp))
    raw.update(_receiver_pbp_raw(pbp))
    raw.update(_rb_pbp_raw(pbp))

    label = f"{season}" + (f" weeks 1-{through_week}" if through_week is not None else "")
    for key, text in (
        ("qb_efficiency", f"nflverse pbp {label}"),
        ("qb_scramble_rate", f"nflverse pbp {label}"),
        ("qb_sack_avoidance", f"nflverse pbp {label}"),
        ("qb_deep_attempt_rate", f"nflverse pbp {label}"),
        ("qb_rz_efficiency", f"nflverse pbp {label}"),
        ("wr_deep_route_rate", f"nflverse pbp {label}"),
        ("te_route_depth", f"nflverse pbp {label} mean air_yards on targets"),
        ("rb_breakaway_rate", f"nflverse pbp {label}"),
        ("rb_rz_efficiency", f"nflverse pbp {label}"),
    ):
        sources[key] = text

    # NGS (real): time to throw and separation
    ttt, ttt_where = _ngs_raw(ngs_pass, "avg_time_to_throw", "attempts", through_week)
    if ttt:
        raw["qb_time_to_throw"] = ttt
        for pid, where in ttt_where.items():
            notes[(pid, "qb_time_to_throw")] = f"NGS passing avg_time_to_throw {season} ({where})"
    sep, sep_where = _ngs_raw(ngs_rec, "avg_separation", "targets", through_week)
    if sep:
        raw["wr_separation"] = sep
        for pid, where in sep_where.items():
            notes[(pid, "wr_separation")] = f"NGS receiving avg_separation {season} ({where})"

    # PFR (real): yards after contact per carry, mapped PFR id -> GSIS
    stats = {"pfr_rush_rows": 0, "pfr_rush_mapped": 0}
    yac_pfr = _pfr_rush_raw(pfr_rush, week_ok)
    yac: dict[str, tuple[float, float]] = {}
    for pfr_id, value in yac_pfr.items():
        stats["pfr_rush_rows"] += 1
        gsis = pfr_map.get(pfr_id)
        if gsis is None:
            continue
        stats["pfr_rush_mapped"] += 1
        yac[gsis] = value
    if yac:
        raw["rb_yards_after_contact"] = yac
        sources["rb_yards_after_contact"] = f"PFR rushing yards after contact per carry {label}"

    # te_route_depth: the published PFR adot when no week can have been cut off
    if through_week is None:
        adot_pfr = _pfr_adot_raw(_safe(_load_pfr, season, "rec", "season"))
        mapped = {pfr_map[p]: v for p, v in adot_pfr.items() if p in pfr_map}
        if mapped:
            merged = dict(raw.get("te_route_depth", {}))
            for gsis, value in mapped.items():
                merged[gsis] = value
                kind_overrides[(gsis, "te_route_depth")] = "real"
                notes[(gsis, "te_route_depth")] = f"PFR receiving adot {season} (season row)"
            raw["te_route_depth"] = merged

    # Games / red-zone denominators
    involvement = _pbp_involvement(pbp)
    games_df, games_source = _player_games(snaps, pfr_map, week_ok, involvement)
    games = games_df.groupby("pid")["week"].nunique().to_dict() if not games_df.empty else {}
    tes = {pid for pid, pos in position.items() if pos == "TE"}
    rz_share = _te_rz_share_raw(pbp, games_df, tes)
    if rz_share:
        raw["te_rz_target_share"] = rz_share
        sources["te_rz_target_share"] = f"nflverse pbp {label} red-zone targets; games from {games_source}"

    # Participation block (possibly a fallback season)
    part_season: int | None = None
    part_frame = pd.DataFrame()
    for candidate in range(season, season - _PARTICIPATION_LOOKBACK, -1):
        frame = _safe(_load_participation, candidate)
        if not frame.empty:
            part_frame, part_season = frame, candidate
            break
    if part_season is not None:
        if part_season == season:
            pbp_part = pbp
        else:
            pbp_part = _prepare_pbp(_safe(_load_pbp, part_season), lambda week: week >= 1)
        part_raw = _participation_raw(pbp_part, part_frame, position)
        label_p = f"{part_season}" + (f" weeks 1-{through_week}" if part_season == season and through_week is not None else "")
        suffix = "" if part_season == season else f" [fallback: no {season} participation feed, latest available is {part_season}]"
        for key, values in part_raw.items():
            if part_season != season:
                # a fallback-season number only describes players who are on a current roster / current data
                current = set(roster_team) | set(_latest_team(pbp))
                values = {pid: v for pid, v in values.items() if pid in current}
            if not values:
                continue
            raw[key] = values
            sources[key] = f"nflverse participation + pbp {label_p}{suffix}"
    stats.update(
        {
            "participation_season": part_season,
            "participation_is_fallback": part_season is not None and part_season != season,
            "games_source": games_source,
            "pfr_id_map_size": len(pfr_map),
        }
    )

    team = dict(roster_team)
    team.update(_latest_team(pbp))
    context = {
        "position": position,
        "name": name,
        "team": team,
        "games": games,
        "sources": sources,
        "notes": notes,
        "kind_overrides": kind_overrides,
        "stats": stats,
    }
    return raw, context


def _build_table(season: int, through_week: int | None) -> dict[str, dict[str, Any]]:
    raw, ctx = _compute(season, _week_predicate(through_week), through_week)
    _LAST_STATS.clear()
    _LAST_STATS.update(ctx["stats"])
    if not raw:
        return {}
    games_by_pid = {pid: int(n) for pid, n in ctx["games"].items()}
    return _shrink_and_assemble(
        raw,
        ctx["sources"],
        ctx["notes"],
        position=ctx["position"],
        name=ctx["name"],
        team=ctx["team"],
        games=games_by_pid,
        season=season,
        through_week=through_week,
        kind_overrides=ctx["kind_overrides"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_CACHE: dict[tuple[int, int | None], tuple[float, dict[str, dict[str, Any]]]] = {}
_CACHE_LOCK = threading.Lock()


def clear_cache() -> None:
    """Drop every cached table (tests, and a manual refresh after new data lands)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def player_feature_table(season: int | None = None, *, through_week: int | None = None) -> dict[str, dict[str, Any]]:
    """Player feature table keyed by GSIS id (see the module docstring for the shape).

    ``season`` defaults to the latest *complete* season. ``through_week=N`` keeps
    regular-season weeks ``1..N`` only (``0`` -> ``{}``: nothing precedes week 1).
    Never raises: any provider failure yields ``{}`` (a missing table is a normal
    state -- downstream models stay neutral). Results are cached for
    :data:`CACHE_SECONDS`; an empty result is not cached, and callers always get
    their own copy.
    """
    try:
        season_i = _default_season() if season is None else int(season)
        week_i = None if through_week is None else int(through_week)
    except (TypeError, ValueError):
        return {}
    if week_i is not None and week_i < 1:
        return {}
    key = (season_i, week_i)
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None and now - hit[0] < CACHE_SECONDS:
            return copy.deepcopy(hit[1])
    try:
        table = _build_table(season_i, week_i)
    except Exception as error:  # the contract: a provider/data problem is never an exception
        LOG.warning("player feature table %s/%s failed: %s", season_i, week_i, error, exc_info=LOG.isEnabledFor(logging.DEBUG))
        return {}
    if table:
        with _CACHE_LOCK:
            _CACHE[key] = (now, table)
    return copy.deepcopy(table)


# --- name matching ---------------------------------------------------------
_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_SHORT_NAME_RE = re.compile(r"^\s*([A-Za-z])\.\s*([A-Za-z][A-Za-z'’\- ]*)$")
_TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB", "HST": "HOU", "JAC": "JAX", "KAN": "KC",
    "LAR": "LA", "LVR": "LV", "OAK": "LV", "NWE": "NE", "NOR": "NO", "SDG": "LAC", "SD": "LAC",
    "SFO": "SF", "STL": "LA", "TAM": "TB", "WSH": "WAS",
}


def _norm_name(value: Any) -> str:
    """Lowercase ascii, no punctuation, suffix (Jr/Sr/II/III/IV/V) stripped, single spaces.

    ``"A.J. Brown"`` -> ``"aj brown"``, ``"Amon-Ra St. Brown"`` -> ``"amon ra st brown"``.
    """
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[.'`]", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = _SUFFIX_RE.sub(" ", text)
    return " ".join(text.split())


def _norm_team(value: Any) -> str:
    text = str(value or "").strip().upper()
    return _TEAM_ALIASES.get(text, text)


def _name_candidates(table: Mapping[str, Mapping[str, Any]], name: str) -> list[str]:
    """Player ids whose name matches ``name`` (exact normalized form first, then ``F.Last``)."""
    squashed = _norm_name(name).replace(" ", "")
    if not squashed:
        return []
    exact = [pid for pid, entry in table.items() if _norm_name(entry.get("name")).replace(" ", "") == squashed]
    if exact:
        return exact
    short = _SHORT_NAME_RE.match(str(name))
    if not short:
        return []
    initial = short.group(1).lower()
    surname = _norm_name(short.group(2)).split()
    if not surname:
        return []
    matches = []
    for pid, entry in table.items():
        tokens = _norm_name(entry.get("name")).split()
        if len(tokens) >= 2 and tokens[0].startswith(initial) and tokens[-len(surname):] == surname and len(tokens) - 1 >= len(surname):
            matches.append(pid)
    return matches


def find_player_features(
    table: Mapping[str, Mapping[str, Any]] | None,
    *,
    player_id: str | None = None,
    name: str | None = None,
    team: str | None = None,
    position: str | None = None,
) -> dict[str, Any] | None:
    """One player's entry from a :func:`player_feature_table` result, or ``None``.

    ``player_id`` (GSIS) is tried first. Otherwise ``name`` is matched after
    normalization (case, accents, punctuation, ``A.J.`` vs ``AJ``, Jr/Sr/II/III;
    a pbp-style ``"J.Waddle"`` is matched as initial + surname). ``team`` and
    ``position``, when supplied, are *filters*: a candidate that contradicts them
    is not returned, and an ambiguous name that they do not resolve returns
    ``None`` -- a wrong player is never guessed. (A team hint that disagrees with
    the table's team therefore yields ``None`` rather than a possibly different
    person of the same name.) The result is a copy of the table entry with
    ``player_id`` added.
    """
    if not table:
        return None
    pid_in = str(player_id).strip() if player_id is not None else ""
    if pid_in and pid_in in table:
        found = pid_in
    elif name and str(name).strip():
        candidates = _name_candidates(table, str(name))
        want_pos = _clean_position(position) if position else None
        if position and want_pos is None:
            return None
        if want_pos:
            candidates = [pid for pid in candidates if str(table[pid].get("position", "")).upper() == want_pos]
        if team and str(team).strip():
            want_team = _norm_team(team)
            candidates = [pid for pid in candidates if _norm_team(table[pid].get("team")) == want_team]
        if len(candidates) != 1:
            return None
        found = candidates[0]
    else:
        return None
    result = copy.deepcopy(dict(table[found]))
    result["player_id"] = found
    return result


def _last_build_stats() -> dict[str, Any]:
    """Diagnostics of the most recent build (id-match counts, participation season); for tests / reports."""
    return dict(_LAST_STATS)


__all__ = [
    "CACHE_SECONDS",
    "EFFECTIVE_GATE",
    "FEATURE_BASIS",
    "FEATURE_KEYS_BY_POSITION",
    "MIN_LEAGUE_PLAYERS",
    "MIN_VOLUME",
    "MIN_WEIGHT",
    "SHRINKAGE_K",
    "clear_cache",
    "find_player_features",
    "player_feature_table",
]
