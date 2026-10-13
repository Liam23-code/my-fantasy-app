"""Team-level canonical features of the NFL model upgrade pack, built from real nflverse data.

``team_feature_table(season, through_week=W)`` turns play-by-play, PFR advanced stats,
NGS tracking stats and snap counts into the *team* inputs of ``docs/nfl_model_upgrade_pack.md``
section 4 for all 32 teams: offense keys describe the team with the ball, ``opp_*`` defense
keys describe the defense that team's opponent brings. It is the team half of the future
``nfl_matchup_features`` loader; player- and game-level keys are not produced here.

I/O and failure policy
----------------------
* One boundary: ``_load_pbp``, ``_load_pfr``, ``_load_ngs``, ``_load_snaps``,
  ``_load_participation``. Each wraps ``nflreadpy``, returns a pandas frame and fails soft to an
  EMPTY frame on any error (offline_data_contract.md rule 3). Tests monkeypatch exactly these.
  A small private TTL cache keeps the heavy frames; an empty (failed) result is cached only for
  a few minutes so a transient outage heals. No network library is imported here.
* ``team_feature_table`` never raises: any failure degrades to ``{}`` or to fewer keys.
* No leakage: ``through_week=W`` keeps regular-season rows with ``week <= W`` of ``season``.
  Postseason rows are always dropped (the pack describes regular-season form).

What is REAL and what is a PROXY
--------------------------------
Every emitted key carries ``kinds[key]`` (``"real"``/``"proxy"``) and a ``basis[key]`` string
naming the feed, the sample and the shrinkage. See :data:`FEATURE_BASIS` for the static
description. In short: EPA, explosive rates, pace, red-zone TD rate, PROE, QB deep rate,
NGS separation, NGS stacked-box rate, PFR pressure/blitz/missed-tackle rates, participation
man-coverage rate are measured quantities. The block-win rates, continuity, pass-rush win rate,
the coverage/run-defense/explosive-prevention/red-zone grades are PROXIES -- proprietary
PFF/ESPN grades are not free, and the proxy is stated in each basis string.

In-season stability (shrinkage)
-------------------------------
A team with ``g`` games of a feature gets ``w = g / (g + K)`` weight on its current value and
``1 - w`` on its prior: the same team's value for the previous regular season (for
player-anchored keys -- ``qb_efficiency``, ``qb_deep_attempt_rate``, ``wr1_separation``,
``rb_yards_after_contact`` -- the *same player's* previous-season value, else the league mean),
or the league mean of the current season when no prior exists. ``K`` is the number of games at
which a team's split-half reliability of that metric is ~0.5 (see :data:`SHRINKAGE_K`; the
measured reliabilities and seasons are in the comment above it). Grades, the defense rank and
the run-block composite are derived AFTER shrinking the underlying metric.

Feed lag: ``load_participation`` has no 2026 rows (it lags the season) and FTN is not used. A
participation-based key falls back to the most recent season that has rows (at most 3 back) and
its basis names that season; the key is omitted if none has them.

Honest limits
-------------
* ``K`` and the in-season weights are measured on 2018-2025 team seasons; they are priors for
  how far to trust a short sample, not a fitted model of the next game.
* Team-level aggregates mix schedule strength, game script and injuries; none is adjusted.
* ``opp_coverage_bust_rate``, ``opp_cb1_shadow_rate`` and ``opp_bracket_coverage_rate`` have no
  free feed: they are NEVER emitted (see :data:`NO_FREE_FEED`); consumers keep them neutral.
* ``rroe`` is exactly ``-proe`` on the same plays (pass-rate-over-expected is a dropback share);
  it is provided because the models name both keys.
"""
from __future__ import annotations

import copy
import datetime as _dt
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Iterable, Mapping

import numpy as np
import pandas as pd

LOG = logging.getLogger("nfl_pack.team_features")

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
#: nflverse team abbreviations (the Rams are ``LA``, not ``LAR``).
TEAMS: tuple[str, ...] = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
)
_TEAM_SET = frozenset(TEAMS)
_TEAM_ALIASES: dict[str, str] = {
    "LAR": "LA", "STL": "LA", "SL": "LA", "LVR": "LV", "OAK": "LV", "SD": "LAC", "SDG": "LAC", "WSH": "WAS",
    "ARZ": "ARI", "JAC": "JAX", "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB",
    "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
}

OFFENSE_KEYS: tuple[str, ...] = (
    "ol_pass_block_win_rate", "ol_run_block_win_rate", "ol_continuity", "team_epa_per_play", "qb_efficiency",
    "explosive_pass_rate", "explosive_run_rate", "pace_seconds_per_play", "red_zone_efficiency", "proe", "rroe",
    "team_rz_pass_rate", "wr1_separation", "rb_yards_after_contact", "qb_deep_attempt_rate",
)
DEFENSE_KEYS: tuple[str, ...] = (
    "opp_pressure_rate", "opp_blitz_rate", "opp_simulated_pressure_success", "opp_pass_rush_win_rate",
    "opp_coverage_grade", "opp_run_defense_grade", "opp_stacked_box_rate", "opp_explosive_play_prevention",
    "opp_missed_tackle_rate", "opp_man_coverage_rate", "opp_def_epa_per_play", "opp_defense_rank",
    "opp_rz_pass_defense_grade", "opp_rz_run_defense_grade",
)
#: Contract section 4 keys this module deliberately never emits (no free nflverse feed).
NO_FREE_FEED: tuple[str, ...] = ("opp_coverage_bust_rate", "opp_cb1_shadow_rate", "opp_bracket_coverage_rate")

#: Static description of each key's feed and proxy (also what the docs quote).
FEATURE_BASIS: dict[str, str] = {
    "ol_pass_block_win_rate": "PROXY: 1 - share of the team's dropbacks that were pressured or sacked "
                              "(participation was_pressure | sack; PFR times_pressured mapped onto that scale when "
                              "the season has no participation rows). Not a PFF/ESPN pass-block win rate.",
    "ol_run_block_win_rate": "PROXY: mean of z(PFR rush yards-before-contact per carry) and z(rush EPA per designed "
                             "run), re-standardised across the teams and placed on the consuming models' prior scale "
                             "(mean 0.70, sd 0.04). A relative standing (league mean 0.70 by construction), not a "
                             "measured block win rate.",
    "ol_continuity": "PROXY: share of offensive-line snaps (positions T/G/C/OL, load_snap_counts) taken by the "
                     "team's five most-used linemen. Upper bound on true same-five-together snaps.",
    "team_epa_per_play": "REAL: mean nflverse EPA per run/pass scrimmage play, regular season.",
    "qb_efficiency": "REAL: EPA per dropback (incl. sacks, scrambles) of the team's primary QB (most dropbacks).",
    "explosive_pass_rate": "REAL: share of pass plays (incl. sacks) gaining 20+ yards.",
    "explosive_run_rate": "REAL: share of designed runs (no scrambles/kneels) gaining 10+ yards.",
    "pace_seconds_per_play": "REAL: mean game-clock seconds between consecutive offensive snaps in the same drive, "
                             "neutral situations (win prob 20-80%, before the last 2 minutes of a half). "
                             "Game-clock time, so clock stops after incompletions are not counted.",
    "red_zone_efficiency": "REAL: share of drives with a snap inside the opponent 20 that ended in a touchdown.",
    "proe": "REAL: mean nflfastR pass_oe / 100 (dropback rate over expected) on run/pass plays.",
    "rroe": "REAL: -proe on the same plays (run rate over expected is the mirror of dropback rate over expected).",
    "team_rz_pass_rate": "REAL: pass plays / (run + pass plays) with yardline_100 <= 20.",
    "wr1_separation": "REAL: NGS receiving avg_separation (yards), target-weighted over weeks, of the team's "
                      "target-leading WR.",
    "rb_yards_after_contact": "REAL: PFR rushing yards after contact per carry of the team's lead RB (most carries).",
    "qb_deep_attempt_rate": "REAL: share of the primary QB's pass attempts with air_yards >= 20.",
    "opp_pressure_rate": "REAL: share of opposing dropbacks that were pressured or sacked (participation "
                         "was_pressure | sack; PFR times_pressured mapped onto that scale when the season has no "
                         "participation rows).",
    "opp_blitz_rate": "REAL: share of opposing dropbacks with 5+ pass rushers (participation "
                      "number_of_pass_rushers; PFR times_blitzed mapped onto that scale when absent).",
    "opp_simulated_pressure_success": "REAL by definition: share of dropbacks with <= 4 pass rushers that were "
                                      "pressured (participation was_pressure, number_of_pass_rushers). Feed lags the "
                                      "season; falls back to the latest season with rows.",
    "opp_pass_rush_win_rate": "PROXY: PFR pressures (hurries + hits + sacks) per opposing dropback, as charted by "
                              "PFR (load_pfr_advstats pass). A pass-rush win rate in spirit, but PFR counts pressures, "
                              "not one-on-one wins; when participation exists it is a second, independently charted "
                              "reading of the same thing opp_pressure_rate measures.",
    "opp_coverage_grade": "PROXY: percentile 0-100 of EPA per dropback allowed (lower EPA = higher grade); "
                          "confounded by pass rush.",
    "opp_run_defense_grade": "PROXY: percentile 0-100 of EPA per designed run allowed (lower EPA = higher grade).",
    "opp_stacked_box_rate": "REAL: share of opposing rush attempts that faced 8+ defenders in the box (NGS rushing "
                            "percent_attempts_gte_eight_defenders, attempts-weighted, attributed to the defense).",
    "opp_explosive_play_prevention": "PROXY: percentile 0-100 of 20+ yard plays allowed per play, inverted "
                                     "(fewer allowed = higher grade).",
    "opp_missed_tackle_rate": "REAL: PFR defensive missed tackles / (tackles + missed tackles).",
    "opp_man_coverage_rate": "REAL: share of dropbacks defended in man coverage (participation "
                             "defense_man_zone_type). Feed lags the season; falls back to the latest season with rows.",
    "opp_def_epa_per_play": "REAL: mean EPA per run/pass play allowed (higher = worse defense).",
    "opp_defense_rank": "REAL (derived): rank 1-32 of EPA per play allowed, 1 = best defense, 32 = worst.",
    "opp_rz_pass_defense_grade": "PROXY: percentile 0-100 of red-zone EPA per dropback allowed (inverted).",
    "opp_rz_run_defense_grade": "PROXY: percentile 0-100 of red-zone EPA per designed run allowed (inverted).",
}

_KIND: dict[str, str] = {
    key: ("proxy" if text.startswith("PROXY") else "real") for key, text in FEATURE_BASIS.items()
}

# ---------------------------------------------------------------------------
# Shrinkage constants
# ---------------------------------------------------------------------------
#: K per underlying metric = games at which that metric's split-half reliability across the 32 teams is ~0.5,
#: so the in-season weight on the current value, ``g / (g + K)``, equals the reliability of a ``g``-game sample.
#: MEASURED where the metric is a plain team ratio (``_K_STUDY`` says so per metric): for each regular season
#: 2018-2025 a team's games are split into odd/even weeks, the 32 team values correlated, the half-season
#: correlations Fisher-z averaged across seasons, and Spearman-Brown'd down to one game
#: (rho1 = r / (n_h - (n_h - 1) r), n_h = 8.5 games per half; K = (1 - rho1) / rho1, clipped to [2, 40]).
#: The spread is wide on purpose: EPA/play and the stacked-box rate settle within a handful of games (K 4-5)
#: while red-zone touchdown rate and red-zone EPA are close to noise over a whole half (K 40, the clip). A
#: single player's value (QB, WR1, lead RB) and the top-five OL share were NOT in the study and keep the
#: documented prior ``K = 8``.
SHRINKAGE_K: dict[str, float] = {
    "ol_pressure_allowed": 6.1,
    "ol_run_ybc_per_carry": 12.8,
    "off_rush_epa": 12.6,
    "ol_continuity": 8.0,
    "team_epa_per_play": 5.0,
    "qb_efficiency": 8.0,
    "explosive_pass_rate": 18.9,
    "explosive_run_rate": 22.5,
    "pace_seconds_per_play": 7.9,
    "red_zone_efficiency": 40.0,
    "proe": 2.8,
    "team_rz_pass_rate": 10.7,
    "wr1_separation": 8.0,
    "rb_yards_after_contact": 8.0,
    "qb_deep_attempt_rate": 8.0,
    "opp_pressure_rate": 12.6,
    "opp_blitz_rate": 2.8,
    "opp_simulated_pressure_success": 21.8,
    "opp_pass_rush_win_rate": 19.2,
    "opp_pass_epa_allowed": 33.5,
    "opp_rush_epa_allowed": 30.6,
    "opp_stacked_box_rate": 3.7,
    "opp_explosive_rate_allowed": 40.0,
    "opp_missed_tackle_rate": 38.2,
    "opp_man_coverage_rate": 2.0,
    "opp_def_epa_per_play": 21.2,
    "opp_rz_pass_epa_allowed": 40.0,
    "opp_rz_run_epa_allowed": 40.0,
}
_K_STUDY: dict[str, str] = {
    "ol_pressure_allowed": "measured: odd/even-week split-half r=+0.58 over 8 seasons (2018-2025) -> one-game reliability 0.141 -> K=6.1",
    "ol_run_ybc_per_carry": "measured: odd/even-week split-half r=+0.40 over 8 seasons (2018-2025) -> one-game reliability 0.072 -> K=12.8",
    "off_rush_epa": "measured: odd/even-week split-half r=+0.40 over 8 seasons (2018-2025) -> one-game reliability 0.073 -> K=12.6",
    "ol_continuity": "PRIOR (not measured): single-player / top-five metrics were not part of the split-half study; K=8 is a documented prior",
    "team_epa_per_play": "measured: odd/even-week split-half r=+0.63 over 8 seasons (2018-2025) -> one-game reliability 0.167 -> K=5.0",
    "qb_efficiency": "PRIOR (not measured): single-player / top-five metrics were not part of the split-half study; K=8 is a documented prior",
    "explosive_pass_rate": "measured: odd/even-week split-half r=+0.31 over 8 seasons (2018-2025) -> one-game reliability 0.050 -> K=18.9",
    "explosive_run_rate": "measured: odd/even-week split-half r=+0.27 over 8 seasons (2018-2025) -> one-game reliability 0.043 -> K=22.5",
    "pace_seconds_per_play": "measured: odd/even-week split-half r=+0.52 over 8 seasons (2018-2025) -> one-game reliability 0.112 -> K=7.9",
    "red_zone_efficiency": "measured: odd/even-week split-half r=+0.06 over 8 seasons (2018-2025) -> one-game reliability 0.020 -> K=40.0",
    "proe": "measured: odd/even-week split-half r=+0.75 over 8 seasons (2018-2025) -> one-game reliability 0.260 -> K=2.8",
    "team_rz_pass_rate": "measured: odd/even-week split-half r=+0.44 over 8 seasons (2018-2025) -> one-game reliability 0.086 -> K=10.7",
    "wr1_separation": "PRIOR (not measured): single-player / top-five metrics were not part of the split-half study; K=8 is a documented prior",
    "rb_yards_after_contact": "PRIOR (not measured): single-player / top-five metrics were not part of the split-half study; K=8 is a documented prior",
    "qb_deep_attempt_rate": "PRIOR (not measured): single-player / top-five metrics were not part of the split-half study; K=8 is a documented prior",
    "opp_pressure_rate": "measured: odd/even-week split-half r=+0.40 over 8 seasons (2018-2025) -> one-game reliability 0.074 -> K=12.6",
    "opp_blitz_rate": "measured: odd/even-week split-half r=+0.76 over 8 seasons (2018-2025) -> one-game reliability 0.266 -> K=2.8",
    "opp_simulated_pressure_success": "measured: odd/even-week split-half r=+0.28 over 8 seasons (2018-2025) -> one-game reliability 0.044 -> K=21.8",
    "opp_pass_rush_win_rate": "measured: odd/even-week split-half r=+0.31 over 8 seasons (2018-2025) -> one-game reliability 0.050 -> K=19.2",
    "opp_pass_epa_allowed": "measured: odd/even-week split-half r=+0.20 over 8 seasons (2018-2025) -> one-game reliability 0.029 -> K=33.5",
    "opp_rush_epa_allowed": "measured: odd/even-week split-half r=+0.22 over 8 seasons (2018-2025) -> one-game reliability 0.032 -> K=30.6",
    "opp_stacked_box_rate": "measured: odd/even-week split-half r=+0.70 over 8 seasons (2018-2025) -> one-game reliability 0.214 -> K=3.7",
    "opp_explosive_rate_allowed": "measured: odd/even-week split-half r=+0.17 over 8 seasons (2018-2025) -> one-game reliability 0.024 -> K=40.0",
    "opp_missed_tackle_rate": "measured: odd/even-week split-half r=+0.18 over 8 seasons (2018-2025) -> one-game reliability 0.025 -> K=38.2",
    "opp_man_coverage_rate": "measured: odd/even-week split-half r=+0.82 over 8 seasons (2018-2025) -> one-game reliability 0.344 -> K=2.0",
    "opp_def_epa_per_play": "measured: odd/even-week split-half r=+0.29 over 8 seasons (2018-2025) -> one-game reliability 0.045 -> K=21.2",
    "opp_rz_pass_epa_allowed": "measured: odd/even-week split-half r=+0.16 over 8 seasons (2018-2025) -> one-game reliability 0.022 -> K=40.0",
    "opp_rz_run_epa_allowed": "measured: odd/even-week split-half r=-0.08 over 8 seasons (2018-2025) -> one-game reliability 0.020 -> K=40.0",
}

#: Linear maps ``consumer_scale = a + b * source`` for the two series that exist on two scales. ``opp_pressure_rate``
#: and ``opp_blitz_rate`` are defined (and the models calibrated) on the participation ``was_pressure`` /
#: ``number_of_pass_rushers`` scale, but participation lags the season, so a season in progress is read from the PFR
#: charting counts, which sit on a LOWER scale (PFR pressure mean 0.211 vs participation 0.300).
#: Feeding that through unmapped would make every defense look like a weak pass rush against the models'
#: reference. Each map is an OLS fit of the team-season participation value on the PFR value over the regular
#: seasons 2023-2025 (sacks and scrambles are coded as pressure from 2023 on), see ``_SCALE_STUDY``.
#: ``missed_tackle`` has one source and stays the identity.
_SCALE_MAPS: dict[str, tuple[float, float]] = {
    "pressure": (0.1231, 0.8392),
    "blitz": (0.0445, 0.8932),
    "missed_tackle": (0.0, 1.0),
}
_SCALE_STUDY: dict[str, str] = {
    "pressure": "OLS participation(pressure|sack per dropback) on PFR times_pressured per dropback, team-seasons "
                "2023-2025, n=96, r=0.741, residual sd 0.0243",
    "blitz": "OLS participation(rushers >= 5 per dropback) on PFR times_blitzed per dropback, team-seasons "
             "2023-2025, n=96, r=0.877, residual sd 0.0318",
}

#: ``ol_run_block_win_rate`` has no free measured scale.  The consuming models read it as an ESPN-style block win
#: rate (prior mean 0.70, sd 0.04 across teams), so the proxy's standardised composite z is placed on that scale.
RUN_BLOCK_SCALE: tuple[float, float] = (0.70, 0.04)
#: Full-season team-level sds of the two components and of their mean-of-z composite (set by the study).
_RUN_BLOCK_REF: dict[str, float] = {"ybc": 0.25, "epa": 0.08, "composite_both": 0.9}

#: Minimum prior-season sample (den units of the metric) for a player's own prior to be used.
_PLAYER_PRIOR_MIN_DEN: dict[str, float] = {
    "qb_efficiency": 150.0,        # dropbacks (~4 games of a starter)
    "qb_deep_attempt_rate": 150.0, # pass attempts
    "wr1_separation": 40.0,        # targets
    "rb_yards_after_contact": 60.0,  # carries
}

_MIN_SEASON = 1999
_MAX_FEED_FALLBACK = 3          # seasons back a lagging feed may fall back
_MIN_TEAMS_FOR_GRADE = 8        # a percentile among fewer teams is meaningless: omit instead
_NEUTRAL_WP = (0.20, 0.80)
_NEUTRAL_MIN_HALF_SECONDS = 120.0
_NEUTRAL_QUARTERS = (1, 2, 3)
_PACE_MIN_GAP = 1.0
_PACE_MAX_GAP = 45.0
_TIMING_PLAY_TYPES = frozenset({"pass", "run", "no_play", "qb_kneel", "qb_spike"})
_OL_POSITIONS = frozenset({"T", "G", "C", "OL", "OT", "OG", "LT", "RT", "LG", "RG"})
_RB_POSITIONS = frozenset({"RB", "HB", "FB"})
_MAX_WEEK = 22

# ---------------------------------------------------------------------------
# Private TTL cache (heavy frames only; failures are cached briefly)
# ---------------------------------------------------------------------------
_TTL_OK = 6 * 3600.0
_TTL_EMPTY = 300.0
_TTL_TABLE = 1800.0
_CACHE: dict[Any, tuple[float, Any, float]] = {}
_LOCK = threading.RLock()


def _cached(key: Any, loader: Callable[[], Any], *, ttl_ok: float = _TTL_OK, ttl_empty: float = _TTL_EMPTY) -> Any:
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and now - hit[0] < hit[2]:
            return hit[1]
    value = loader()
    empty = (isinstance(value, pd.DataFrame) and value.empty) or (isinstance(value, dict) and not value)
    with _LOCK:
        _CACHE[key] = (now, value, ttl_empty if empty else ttl_ok)
    return value


def clear_cache() -> None:
    """Drop every cached frame and table (tests and a manual refresh use this)."""
    with _LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------
# The I/O boundary: nflreadpy -> pandas, fail soft to an empty frame
# ---------------------------------------------------------------------------
_PBP_COLUMNS: tuple[str, ...] = (
    "game_id", "play_id", "season_type", "week", "posteam", "defteam", "play_type", "down", "qb_dropback",
    "qb_scramble", "pass_attempt", "sack", "qb_hit", "epa", "yards_gained", "air_yards", "passer_player_id",
    "passer_player_name", "rusher_player_id", "yardline_100", "fixed_drive", "drive", "fixed_drive_result",
    "game_seconds_remaining", "half_seconds_remaining", "qtr", "wp", "pass_oe",
)
_PARTICIPATION_COLUMNS: tuple[str, ...] = (
    "nflverse_game_id", "play_id", "possession_team", "was_pressure", "number_of_pass_rushers",
    "defense_man_zone_type",
)


def _to_pandas(data: Any, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Polars/pandas/None -> pandas, optionally projecting to the columns that exist."""
    if data is None:
        return pd.DataFrame()
    if columns is not None and hasattr(data, "select") and hasattr(data, "columns"):
        present = [c for c in columns if c in data.columns]
        if present:
            data = data.select(present)
    if hasattr(data, "to_pandas"):
        data = data.to_pandas()
    return data if isinstance(data, pd.DataFrame) else pd.DataFrame()


def _load_pbp(season: int) -> pd.DataFrame:
    """Play-by-play for one season (projected columns). Empty frame on any failure."""
    def go() -> pd.DataFrame:
        try:
            import nflreadpy as nfl

            return _to_pandas(nfl.load_pbp([int(season)]), _PBP_COLUMNS)
        except Exception as error:  # noqa: BLE001 - fail soft by contract
            LOG.debug("pbp %s unavailable: %s", season, error)
            return pd.DataFrame()

    return _cached(("pbp", int(season)), go)


def _load_pfr(season: int, stat_type: str) -> pd.DataFrame:
    """PFR advanced stats, week level (``stat_type`` = pass | rush | rec | def)."""
    def go() -> pd.DataFrame:
        try:
            import nflreadpy as nfl

            return _to_pandas(nfl.load_pfr_advstats([int(season)], stat_type=stat_type, summary_level="week"))
        except Exception as error:  # noqa: BLE001
            LOG.debug("pfr %s/%s unavailable: %s", season, stat_type, error)
            return pd.DataFrame()

    return _cached(("pfr", int(season), stat_type), go)


def _load_ngs(season: int, stat_type: str) -> pd.DataFrame:
    """NGS stats (``stat_type`` = passing | rushing | receiving); ``week == 0`` rows are season aggregates."""
    def go() -> pd.DataFrame:
        try:
            import nflreadpy as nfl

            return _to_pandas(nfl.load_nextgen_stats([int(season)], stat_type=stat_type))
        except Exception as error:  # noqa: BLE001
            LOG.debug("ngs %s/%s unavailable: %s", season, stat_type, error)
            return pd.DataFrame()

    return _cached(("ngs", int(season), stat_type), go)


def _load_snaps(season: int) -> pd.DataFrame:
    """Per player-game snap counts."""
    def go() -> pd.DataFrame:
        try:
            import nflreadpy as nfl

            return _to_pandas(nfl.load_snap_counts([int(season)]))
        except Exception as error:  # noqa: BLE001
            LOG.debug("snaps %s unavailable: %s", season, error)
            return pd.DataFrame()

    return _cached(("snaps", int(season)), go)


def _load_participation(season: int) -> pd.DataFrame:
    """Play-level participation (was_pressure, rushers, man/zone). Lags the season; empty when absent."""
    def go() -> pd.DataFrame:
        try:
            import nflreadpy as nfl

            return _to_pandas(nfl.load_participation([int(season)]), _PARTICIPATION_COLUMNS)
        except Exception as error:  # noqa: BLE001
            LOG.debug("participation %s unavailable: %s", season, error)
            return pd.DataFrame()

    return _cached(("participation", int(season)), go)


# ---------------------------------------------------------------------------
# Small frame helpers
# ---------------------------------------------------------------------------
def normalize_team(value: Any) -> str | None:
    """nflverse abbreviation for any common spelling (``LAR`` -> ``LA``), or ``None``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().upper()
    text = _TEAM_ALIASES.get(text, text)
    return text if text in _TEAM_SET else None


def _norm_teams(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=object)
    text = series.astype("object").where(series.notna(), None)
    text = text.map(lambda v: normalize_team(v) if v is not None else None)
    return text.astype("object")


def _num(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _flag(frame: pd.DataFrame, name: str) -> pd.Series:
    return _num(frame, name).fillna(0.0) > 0


def _has(frame: pd.DataFrame, *names: str) -> bool:
    return (not frame.empty) and all(n in frame.columns for n in names)


def _rows(team: pd.Series, week: pd.Series, num: pd.Series, den: pd.Series,
          player: pd.Series | None = None, name: pd.Series | None = None) -> pd.DataFrame:
    """Per team-game numerator/denominator rows: value = sum(num) / sum(den)."""
    out = pd.DataFrame({"team": team.values, "week": week.values, "num": num.values.astype(float),
                        "den": den.values.astype(float)})
    if player is not None:
        out["player"] = player.values
        out["name"] = name.values if name is not None else None
    out = out[out["team"].notna() & out["week"].notna() & out["den"].notna() & out["num"].notna()]
    if player is not None:
        out = out[out["player"].notna()]
    keys = ["team", "week"] + (["player"] if player is not None else [])
    agg: dict[str, Any] = {"num": "sum", "den": "sum"}
    if player is not None:
        agg["name"] = "first"
    return out.groupby(keys, as_index=False, sort=True).agg(agg)


def _empty_rows(player: bool = False) -> pd.DataFrame:
    cols = ["team", "week", "num", "den"] + (["player", "name"] if player else [])
    return pd.DataFrame({c: pd.Series(dtype=float if c in ("week", "num", "den") else object) for c in cols})


# ---------------------------------------------------------------------------
# One season's prepared frames and per-metric component rows
# ---------------------------------------------------------------------------
class _Season:
    """Lazily prepared frames for one season. Cheap to build; loaders are cached."""

    def __init__(self, season: int) -> None:
        self.season = int(season)
        self._memo: dict[str, Any] = {}
        #: metric id -> which feed actually produced its rows ("participation" | "pfr_mapped" | ...)
        self.sources: dict[str, str] = {}

    def _get(self, name: str, fn: Callable[[], Any]) -> Any:
        if name not in self._memo:
            self._memo[name] = fn()
        return self._memo[name]

    # -- play-by-play ------------------------------------------------------
    @property
    def plays(self) -> pd.DataFrame:
        return self._get("plays", lambda: _prepare_plays(_load_pbp(self.season)))

    @property
    def timing(self) -> pd.DataFrame:
        return self._get("timing", lambda: _prepare_timing(_load_pbp(self.season)))

    @property
    def team_games(self) -> pd.DataFrame:
        """Long table ``team, opp, week, game_id`` of regular-season games."""
        def build() -> pd.DataFrame:
            p = self.plays
            if p.empty:
                return pd.DataFrame(columns=["team", "opp", "week", "game_id"])
            pairs = p[["game_id", "week", "posteam", "defteam"]].drop_duplicates()
            a = pairs.rename(columns={"posteam": "team", "defteam": "opp"})
            b = pairs.rename(columns={"defteam": "team", "posteam": "opp"})
            both = pd.concat([a, b], ignore_index=True).drop_duplicates(["game_id", "team"])
            return both[["team", "opp", "week", "game_id"]].sort_values(["week", "team"]).reset_index(drop=True)
        return self._get("team_games", build)

    def games_through(self, weeks: Collection[int] | None) -> dict[str, int]:
        tg = self.team_games
        if tg.empty:
            return {}
        if weeks is not None:
            tg = tg[tg["week"].isin(list(weeks))]
        return {str(k): int(v) for k, v in tg.groupby("team")["game_id"].nunique().items()}

    def max_week(self, weeks: Collection[int] | None) -> int:
        tg = self.team_games
        if tg.empty:
            return 0
        if weeks is not None:
            tg = tg[tg["week"].isin(list(weeks))]
        return int(tg["week"].max()) if not tg.empty else 0

    # -- feed frames -------------------------------------------------------
    def pfr(self, stat_type: str) -> pd.DataFrame:
        return self._get(f"pfr_{stat_type}", lambda: _prepare_pfr(_load_pfr(self.season, stat_type)))

    def ngs(self, stat_type: str) -> pd.DataFrame:
        return self._get(f"ngs_{stat_type}", lambda: _prepare_ngs(_load_ngs(self.season, stat_type)))

    @property
    def snaps(self) -> pd.DataFrame:
        return self._get("snaps", lambda: _prepare_snaps(_load_snaps(self.season)))

    @property
    def participation(self) -> pd.DataFrame:
        """Participation rows joined to regular-season dropbacks: defteam, week, was_pressure, rushers, man."""
        return self._get("participation", lambda: _prepare_participation(_load_participation(self.season), self.plays))

    def rows(self, metric: "_Metric") -> pd.DataFrame:
        return self._get(f"rows_{metric.mid}", lambda: _safe_builder(metric, self))


def _safe_builder(metric: "_Metric", season: _Season) -> pd.DataFrame:
    try:
        out = metric.builder(season)
    except Exception as error:  # noqa: BLE001 - a bad frame must not take the whole table down
        LOG.debug("metric %s failed for %s: %s", metric.mid, season.season, error)
        out = None
    return out if isinstance(out, pd.DataFrame) else _empty_rows(metric.reducer == "player")


def _prepare_plays(raw: pd.DataFrame) -> pd.DataFrame:
    need = {"game_id", "week", "posteam", "defteam", "play_type"}
    if raw is None or raw.empty or not need.issubset(raw.columns):
        return pd.DataFrame()
    d = raw
    if "season_type" in d.columns:
        d = d[d["season_type"].astype(str).str.upper() == "REG"]
    d = d[d["play_type"].isin(["run", "pass"])]
    if "down" in d.columns:
        d = d[d["down"].notna()]          # drops 2-point tries
    if d.empty:
        return pd.DataFrame()
    is_pass = d["play_type"] == "pass"
    scramble = _flag(d, "qb_scramble")
    sack = _flag(d, "sack")
    hit = _flag(d, "qb_hit")
    if "qb_dropback" in d.columns:
        db = _flag(d, "qb_dropback")
    else:
        db = is_pass | scramble
    pass_att = _flag(d, "pass_attempt") if "pass_attempt" in d.columns else (is_pass & ~sack)
    passer = d["passer_player_id"] if "passer_player_id" in d.columns else pd.Series(None, index=d.index, dtype=object)
    rusher = d["rusher_player_id"] if "rusher_player_id" in d.columns else pd.Series(None, index=d.index, dtype=object)
    pname = d["passer_player_name"] if "passer_player_name" in d.columns else pd.Series(None, index=d.index, dtype=object)
    drive = d["fixed_drive"] if "fixed_drive" in d.columns else (d["drive"] if "drive" in d.columns else None)
    drive_result = d["fixed_drive_result"] if "fixed_drive_result" in d.columns else None
    out = pd.DataFrame({
        "game_id": d["game_id"].values,
        "play_id": _num(d, "play_id").values,
        "week": _num(d, "week").values,
        "posteam": _norm_teams(d["posteam"]).values,
        "defteam": _norm_teams(d["defteam"]).values,
        "is_pass": is_pass.values,
        "is_run": (~is_pass & ~scramble).values,      # designed run: play_type run and not a scramble
        "is_run_all": (~is_pass).values,              # every play_type == run play (QB scrambles included)
        "scramble": scramble.values,
        "db": db.values,
        "pass_att": pass_att.values,
        "sack": sack.values,
        "sack_or_hit": (sack | hit).values,
        "epa": _num(d, "epa").values,
        "yards": _num(d, "yards_gained").values,
        "air": _num(d, "air_yards").values,
        "passer": passer.where(is_pass).values,
        "qb_id": passer.where(is_pass, rusher.where(scramble)).values,
        "passer_name": pname.values,
        "yl": _num(d, "yardline_100").values,
        "drive": (drive.values if drive is not None else np.full(len(d), np.nan)),
        "drive_result": (drive_result.values if drive_result is not None else np.full(len(d), None, dtype=object)),
        "gsr": _num(d, "game_seconds_remaining").values,
        "hsr": _num(d, "half_seconds_remaining").values,
        "qtr": _num(d, "qtr").values,
        "wp": _num(d, "wp").values,
        "pass_oe": _num(d, "pass_oe").values,
    })
    out = out[out["posteam"].notna() & out["defteam"].notna() & out["week"].notna()].copy()
    out["week"] = out["week"].astype(int)
    out["rz"] = out["yl"] <= 20
    return out.reset_index(drop=True)


def _prepare_timing(raw: pd.DataFrame) -> pd.DataFrame:
    """Snap-timing frame: run/pass plus no-plays, kneels and spikes (so a nullified snap breaks an interval)."""
    need = {"game_id", "play_id", "posteam", "play_type", "game_seconds_remaining"}
    if raw is None or raw.empty or not need.issubset(raw.columns):
        return pd.DataFrame()
    d = raw
    if "season_type" in d.columns:
        d = d[d["season_type"].astype(str).str.upper() == "REG"]
    d = d[d["play_type"].isin(_TIMING_PLAY_TYPES) & d["posteam"].notna() & d["game_seconds_remaining"].notna()]
    if d.empty:
        return pd.DataFrame()
    drive = d["fixed_drive"] if "fixed_drive" in d.columns else (d["drive"] if "drive" in d.columns else None)
    if drive is None:
        return pd.DataFrame()
    out = pd.DataFrame({
        "game_id": d["game_id"].values,
        "play_id": _num(d, "play_id").values,
        "week": _num(d, "week").values,
        "posteam": _norm_teams(d["posteam"]).values,
        "play_type": d["play_type"].values,
        "drive": drive.values,
        "qtr": _num(d, "qtr").values,
        "gsr": _num(d, "game_seconds_remaining").values,
        "hsr": _num(d, "half_seconds_remaining").values,
        "wp": _num(d, "wp").values,
    })
    out = out[out["posteam"].notna() & out["week"].notna() & out["drive"].notna()].copy()
    out["week"] = out["week"].astype(int)
    return out.sort_values(["game_id", "play_id"], kind="mergesort").reset_index(drop=True)


def _prepare_pfr(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty or "team" not in raw.columns or "week" not in raw.columns:
        return pd.DataFrame()
    d = raw
    if "game_type" in d.columns:
        d = d[d["game_type"].astype(str).str.upper() == "REG"]
    d = d.copy()
    d["team"] = _norm_teams(d["team"]).values
    d["opponent"] = _norm_teams(d["opponent"]).values if "opponent" in d.columns else None
    d["week"] = _num(d, "week")
    d = d[d["team"].notna() & d["week"].notna()]
    d["week"] = d["week"].astype(int)
    return d.reset_index(drop=True)


def _prepare_ngs(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty or "week" not in raw.columns:
        return pd.DataFrame()
    d = raw
    if "season_type" in d.columns:
        d = d[d["season_type"].astype(str).str.upper() == "REG"]
    d = d.copy()
    d["week"] = _num(d, "week")
    d = d[d["week"].notna() & (d["week"] > 0)]            # week 0 = season aggregate
    team_col = "team_abbr" if "team_abbr" in d.columns else ("team" if "team" in d.columns else None)
    if team_col is None or d.empty:
        return pd.DataFrame()
    d["team"] = _norm_teams(d[team_col]).values
    d = d[d["team"].notna()]
    d["week"] = d["week"].astype(int)
    return d.reset_index(drop=True)


def _prepare_snaps(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty or not {"team", "week", "position", "player"}.issubset(raw.columns):
        return pd.DataFrame()
    d = raw
    if "game_type" in d.columns:
        d = d[d["game_type"].astype(str).str.upper() == "REG"]
    d = d.copy()
    d["team"] = _norm_teams(d["team"]).values
    d["week"] = _num(d, "week")
    d = d[d["team"].notna() & d["week"].notna()]
    d["week"] = d["week"].astype(int)
    return d.reset_index(drop=True)


def _prepare_participation(raw: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty or plays.empty or not {"nflverse_game_id", "play_id"}.issubset(raw.columns):
        return pd.DataFrame()
    pa = pd.DataFrame({
        "game_id": raw["nflverse_game_id"].values,
        "play_id": _num(raw, "play_id").values,
        "pressure": raw["was_pressure"].map(_truthy).values if "was_pressure" in raw.columns else np.nan,
        "rushers": _num(raw, "number_of_pass_rushers").values,
        "man_zone": raw["defense_man_zone_type"].astype(str).str.upper().values
        if "defense_man_zone_type" in raw.columns else None,
    })
    base = plays.loc[plays["db"], ["game_id", "play_id", "week", "defteam", "posteam", "sack"]]
    joined = pa.merge(base, on=["game_id", "play_id"], how="inner")
    return joined.reset_index(drop=True)


def _truthy(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "t", "yes"):
            return 1.0
        if text in ("false", "0", "f", "no"):
            return 0.0
        return float("nan")
    try:
        return 1.0 if bool(value) else 0.0
    except Exception:  # noqa: BLE001
        return float("nan")


# -- metric builders: each returns rows(team, week, num, den[, player, name]) ----------------------
def _by(plays: pd.DataFrame, mask: pd.Series | None, team_col: str, num: pd.Series, den: pd.Series | None = None) -> pd.DataFrame:
    if plays.empty:
        return _empty_rows()
    p = plays if mask is None else plays[mask]
    n = num if mask is None else num[mask]
    dd = pd.Series(1.0, index=p.index) if den is None else (den if mask is None else den[mask])
    return _rows(p[team_col], p["week"], n, dd)


def _b_off_epa(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["epa"].notna(), "posteam", p["epa"].fillna(0.0))


def _b_off_rush_epa(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["is_run"] & p["epa"].notna(), "posteam", p["epa"].fillna(0.0))


def _b_explosive_pass(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["is_pass"] & p["yards"].notna(), "posteam", (p["yards"] >= 20).astype(float))


def _b_explosive_run(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["is_run_all"] & p["yards"].notna(), "posteam", (p["yards"] >= 10).astype(float))


def _b_pace(s: _Season) -> pd.DataFrame:
    """Game-clock seconds between consecutive snaps of one drive in neutral situations.

    Neutral = win probability 20-80%, quarters 1-3, outside the last two minutes of a half; both snaps are
    run/pass plays of the same team, drive and quarter; interval 1-45 s (clock stoppages understate cadence).
    """
    q = s.timing
    if q.empty:
        return _empty_rows()
    grp = q.groupby(["game_id", "posteam", "drive"], sort=False)
    prev_gsr = grp["gsr"].shift(1)
    prev_type = grp["play_type"].shift(1)
    prev_qtr = grp["qtr"].shift(1)
    delta = prev_gsr - q["gsr"]
    ok = (
        delta.between(_PACE_MIN_GAP, _PACE_MAX_GAP)
        & (q["qtr"] == prev_qtr)
        & q["qtr"].isin(_NEUTRAL_QUARTERS)
        & (q["hsr"] > _NEUTRAL_MIN_HALF_SECONDS)
        & q["wp"].between(*_NEUTRAL_WP)
        & prev_type.isin(["pass", "run"])
        & q["play_type"].isin(["pass", "run"])
    )
    if not ok.any():
        return _empty_rows()
    return _rows(q.loc[ok, "posteam"], q.loc[ok, "week"], delta[ok], pd.Series(1.0, index=delta[ok].index))


def _b_red_zone_efficiency(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty or p["drive"].isna().all() or p["drive_result"].isna().all():
        return _empty_rows()
    rz = p[p["rz"] & p["drive"].notna()]
    if rz.empty:
        return _empty_rows()
    trips = rz.sort_values(["game_id", "play_id"], kind="mergesort").groupby(
        ["posteam", "week", "game_id", "drive"], as_index=False, sort=False)["drive_result"].last()
    td = (trips["drive_result"].astype(str).str.lower() == "touchdown").astype(float)
    return _rows(trips["posteam"], trips["week"], td, pd.Series(1.0, index=trips.index))


def _b_proe(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["pass_oe"].notna(), "posteam", p["pass_oe"].fillna(0.0) / 100.0)


def _b_rz_pass_rate(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["rz"], "posteam", p["db"].astype(float))


def _b_qb_efficiency(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows(True)
    m = p["db"] & p["qb_id"].notna() & p["epa"].notna()
    q = p[m]
    if q.empty:
        return _empty_rows(True)
    return _rows(q["posteam"], q["week"], q["epa"], pd.Series(1.0, index=q.index), q["qb_id"], q["passer_name"])


def _b_qb_deep(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows(True)
    q = p[p["pass_att"] & p["passer"].notna() & p["air"].notna()]
    if q.empty:
        return _empty_rows(True)
    return _rows(q["posteam"], q["week"], (q["air"] >= 20).astype(float), pd.Series(1.0, index=q.index),
                 q["passer"], q["passer_name"])


def _b_wr1_separation(s: _Season) -> pd.DataFrame:
    d = s.ngs("receiving")
    if d.empty or not {"avg_separation", "targets"}.issubset(d.columns):
        return _empty_rows(True)
    if "player_position" in d.columns:
        d = d[d["player_position"].astype(str).str.upper() == "WR"]
    pid = "player_gsis_id" if "player_gsis_id" in d.columns else "player_display_name"
    if pid not in d.columns:
        return _empty_rows(True)
    targets = _num(d, "targets")
    sep = _num(d, "avg_separation")
    ok = targets.notna() & (targets > 0) & sep.notna()
    d = d[ok]
    if d.empty:
        return _empty_rows(True)
    name = d["player_display_name"] if "player_display_name" in d.columns else d[pid]
    return _rows(d["team"], d["week"], sep[ok] * targets[ok], targets[ok], d[pid], name)


def _rb_ids(s: _Season) -> set[Any] | None:
    """PFR ids of running backs (from snap-count positions); ``None`` if positions are unavailable."""
    sn = s.snaps
    if sn.empty or "pfr_player_id" not in sn.columns:
        return None
    pos = sn["position"].astype(str).str.upper()
    ids = sn.loc[pos.isin(_RB_POSITIONS), "pfr_player_id"].dropna()
    return set(ids) if len(ids) else None


def _b_rb_yac(s: _Season) -> pd.DataFrame:
    d = s.pfr("rush")
    if d.empty or not {"carries", "rushing_yards_after_contact"}.issubset(d.columns):
        return _empty_rows(True)
    pid = "pfr_player_id" if "pfr_player_id" in d.columns else "pfr_player_name"
    if pid not in d.columns:
        return _empty_rows(True)
    ids = _rb_ids(s)
    if ids is not None:
        d = d[d[pid].isin(ids)]
    else:  # no position source: drop anyone who threw a pass this season (QBs)
        qbs = s.pfr("pass")
        if not qbs.empty and pid in qbs.columns:
            d = d[~d[pid].isin(set(qbs[pid].dropna()))]
    carries = _num(d, "carries")
    yac = _num(d, "rushing_yards_after_contact")
    ok = carries.notna() & (carries > 0) & yac.notna()
    d = d[ok]
    if d.empty:
        return _empty_rows(True)
    name = d["pfr_player_name"] if "pfr_player_name" in d.columns else d[pid]
    return _rows(d["team"], d["week"], yac[ok], carries[ok], d[pid], name)


def _dropbacks(s: _Season, team_col: str) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return pd.DataFrame(columns=["team", "week", "db"])
    g = p[p["db"]].groupby([team_col, "week"]).size().rename("db").reset_index()
    return g.rename(columns={team_col: "team"})


def _pfr_pass_by(s: _Season, group_col: str, column: str) -> pd.DataFrame:
    d = s.pfr("pass")
    if d.empty or column not in d.columns or group_col not in d.columns:
        return pd.DataFrame(columns=["team", "week", "x"])
    x = d.assign(x=_num(d, column))
    x = x[x["x"].notna() & x[group_col].notna()]
    g = x.groupby([group_col, "week"], as_index=False)["x"].sum()
    return g.rename(columns={group_col: "team"})


def _mapped(num: pd.Series, den: pd.Series, name: str) -> tuple[pd.Series, pd.Series]:
    """Rows of ``a + b * (num / den)`` as (num', den): num' = a * den + b * num (a ratio again)."""
    a, b = _SCALE_MAPS[name]
    return a * den + b * num, den


def _b_pressure(side: str, mid: str) -> Callable[[_Season], pd.DataFrame]:
    """Share of dropbacks that were pressured or sacked, by ``defteam`` (generated) or ``posteam`` (allowed).

    Primary feed: participation ``was_pressure`` (the scale the QB/pass-game collapse models are calibrated on).
    When the season has no participation rows (the feed lags the season) the PFR ``times_pressured`` series is used,
    mapped onto that scale (``_SCALE_MAPS['pressure']``) and the season's source is recorded as ``pfr_mapped``.
    """
    def build(s: _Season) -> pd.DataFrame:
        part = s.participation
        if not part.empty:
            keep = part["pressure"].notna() | part["sack"].astype(bool)
            q = part[keep]
            if not q.empty:
                hit = ((q["pressure"].fillna(0.0) > 0) | q["sack"].astype(bool)).astype(float)
                s.sources[mid] = "participation"
                return _rows(q[side], q["week"], hit, pd.Series(1.0, index=q.index))
        group = "opponent" if side == "defteam" else "team"
        pr = _pfr_pass_by(s, group, "times_pressured")
        j = pr.merge(_dropbacks(s, side), on=["team", "week"], how="inner")
        j = j[j["db"] > 0]
        if j.empty:
            return _empty_rows()
        num, den = _mapped(j["x"], j["db"], "pressure")
        s.sources[mid] = "pfr_mapped"
        return _rows(j["team"], j["week"], num, den)
    return build


def _b_blitz(s: _Season) -> pd.DataFrame:
    """Share of dropbacks with 5+ pass rushers (participation); PFR ``times_blitzed`` mapped when it has no rows."""
    part = s.participation
    if not part.empty:
        q = part[part["rushers"].notna()]
        if not q.empty:
            s.sources["opp_blitz_rate"] = "participation"
            return _rows(q["defteam"], q["week"], (q["rushers"] >= 5).astype(float), pd.Series(1.0, index=q.index))
    pr = _pfr_pass_by(s, "opponent", "times_blitzed")
    j = pr.merge(_dropbacks(s, "defteam"), on=["team", "week"], how="inner")
    j = j[j["db"] > 0]
    if j.empty:
        return _empty_rows()
    num, den = _mapped(j["x"], j["db"], "blitz")
    s.sources["opp_blitz_rate"] = "pfr_mapped"
    return _rows(j["team"], j["week"], num, den)


def _b_pfr_pressure_mapped(s: _Season) -> pd.DataFrame:
    """PFR pressures (hurries + hits + sacks) per opposing dropback, mapped to the was_pressure scale."""
    pr = _pfr_pass_by(s, "opponent", "times_pressured")
    j = pr.merge(_dropbacks(s, "defteam"), on=["team", "week"], how="inner")
    j = j[j["db"] > 0]
    if j.empty:
        return _empty_rows()
    num, den = _mapped(j["x"], j["db"], "pressure")
    s.sources["opp_pass_rush_win_rate"] = "pfr_mapped"
    return _rows(j["team"], j["week"], num, den)


def _b_ol_run_ybc(s: _Season) -> pd.DataFrame:
    d = s.pfr("rush")
    if d.empty or not {"carries", "rushing_yards_before_contact"}.issubset(d.columns):
        return _empty_rows()
    carries = _num(d, "carries")
    ybc = _num(d, "rushing_yards_before_contact")
    ok = carries.notna() & (carries > 0) & ybc.notna()
    return _rows(d.loc[ok, "team"], d.loc[ok, "week"], ybc[ok], carries[ok])


def _b_ol_continuity(s: _Season) -> pd.DataFrame:
    d = s.snaps
    if d.empty or "offense_snaps" not in d.columns:
        return _empty_rows(True)
    d = d[d["position"].astype(str).str.upper().isin(_OL_POSITIONS)]
    snaps = _num(d, "offense_snaps")
    ok = snaps.notna() & (snaps > 0)
    d = d[ok]
    if d.empty:
        return _empty_rows(True)
    return _rows(d["team"], d["week"], snaps[ok], snaps[ok], d["player"], d["player"])


def _b_opp_def_epa(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["epa"].notna(), "defteam", p["epa"].fillna(0.0))


def _b_opp_pass_epa(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["db"] & p["epa"].notna(), "defteam", p["epa"].fillna(0.0))


def _b_opp_rush_epa(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["is_run"] & p["epa"].notna(), "defteam", p["epa"].fillna(0.0))


def _b_opp_rz_pass_epa(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["rz"] & p["db"] & p["epa"].notna(), "defteam", p["epa"].fillna(0.0))


def _b_opp_rz_run_epa(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["rz"] & p["is_run"] & p["epa"].notna(), "defteam", p["epa"].fillna(0.0))


def _b_opp_explosive(s: _Season) -> pd.DataFrame:
    p = s.plays
    if p.empty:
        return _empty_rows()
    return _by(p, p["yards"].notna(), "defteam", (p["yards"] >= 20).astype(float))


def _b_opp_stacked_box(s: _Season) -> pd.DataFrame:
    d = s.ngs("rushing")
    tg = s.team_games
    if d.empty or tg.empty or not {"rush_attempts", "percent_attempts_gte_eight_defenders"}.issubset(d.columns):
        return _empty_rows()
    att = _num(d, "rush_attempts")
    pct = _num(d, "percent_attempts_gte_eight_defenders")
    x = pd.DataFrame({"team": d["team"].values, "week": d["week"].values, "att": att.values, "pct": pct.values})
    x = x[x["att"].notna() & (x["att"] > 0) & x["pct"].notna()]
    x = x.merge(tg[["team", "week", "opp"]], on=["team", "week"], how="inner")
    if x.empty:
        return _empty_rows()
    return _rows(x["opp"], x["week"], x["pct"] / 100.0 * x["att"], x["att"])


def _b_opp_missed_tackle(s: _Season) -> pd.DataFrame:
    d = s.pfr("def")
    if d.empty or not {"def_missed_tackles", "def_tackles_combined"}.issubset(d.columns):
        return _empty_rows()
    miss = _num(d, "def_missed_tackles").fillna(0.0)
    tk = _num(d, "def_tackles_combined").fillna(0.0)
    den = miss + tk
    ok = den > 0
    num, den = _mapped(miss[ok], den[ok], "missed_tackle")
    s.sources["opp_missed_tackle_rate"] = "pfr_mapped"
    return _rows(d.loc[ok, "team"], d.loc[ok, "week"], num, den)


def _b_opp_sim_pressure(s: _Season) -> pd.DataFrame:
    d = s.participation
    if d.empty:
        return _empty_rows()
    ok = d["pressure"].notna() & d["rushers"].notna() & (d["rushers"] <= 4)
    return _rows(d.loc[ok, "defteam"], d.loc[ok, "week"], d.loc[ok, "pressure"], pd.Series(1.0, index=d.index[ok]))


def _b_opp_man(s: _Season) -> pd.DataFrame:
    d = s.participation
    if d.empty or "man_zone" not in d.columns:
        return _empty_rows()
    mz = d["man_zone"].astype(str)
    ok = mz.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])
    return _rows(d.loc[ok, "defteam"], d.loc[ok, "week"], (mz[ok] == "MAN_COVERAGE").astype(float),
                 pd.Series(1.0, index=d.index[ok]))


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Metric:
    mid: str
    side: str                     # "offense" | "defense"
    builder: Callable[[_Season], pd.DataFrame]
    unit: str                     # what ``den`` counts, for basis strings
    reducer: str = "ratio"        # ratio | player | top5
    lag: bool = False             # feed lags the season: may fall back to an earlier season
    window_matched_prior: bool = False   # prior is taken over the same weeks (window-length-biased metrics)


_METRICS: tuple[_Metric, ...] = (
    _Metric("ol_pressure_allowed", "offense", _b_pressure("posteam", "ol_pressure_allowed"), "dropbacks"),
    _Metric("ol_run_ybc_per_carry", "offense", _b_ol_run_ybc, "carries"),
    _Metric("off_rush_epa", "offense", _b_off_rush_epa, "designed runs"),
    _Metric("ol_continuity", "offense", _b_ol_continuity, "OL snaps", reducer="top5", window_matched_prior=True),
    _Metric("team_epa_per_play", "offense", _b_off_epa, "plays"),
    _Metric("qb_efficiency", "offense", _b_qb_efficiency, "dropbacks", reducer="player"),
    _Metric("explosive_pass_rate", "offense", _b_explosive_pass, "pass plays"),
    _Metric("explosive_run_rate", "offense", _b_explosive_run, "designed runs"),
    _Metric("pace_seconds_per_play", "offense", _b_pace, "timed snaps"),
    _Metric("red_zone_efficiency", "offense", _b_red_zone_efficiency, "red-zone trips"),
    _Metric("proe", "offense", _b_proe, "plays"),
    _Metric("team_rz_pass_rate", "offense", _b_rz_pass_rate, "red-zone plays"),
    _Metric("wr1_separation", "offense", _b_wr1_separation, "targets", reducer="player"),
    _Metric("rb_yards_after_contact", "offense", _b_rb_yac, "carries", reducer="player"),
    _Metric("qb_deep_attempt_rate", "offense", _b_qb_deep, "pass attempts", reducer="player"),
    _Metric("opp_pressure_rate", "defense", _b_pressure("defteam", "opp_pressure_rate"), "dropbacks"),
    _Metric("opp_blitz_rate", "defense", _b_blitz, "dropbacks"),
    _Metric("opp_simulated_pressure_success", "defense", _b_opp_sim_pressure, "<=4-rusher dropbacks", lag=True),
    _Metric("opp_pass_rush_win_rate", "defense", _b_pfr_pressure_mapped, "dropbacks"),
    _Metric("opp_pass_epa_allowed", "defense", _b_opp_pass_epa, "dropbacks"),
    _Metric("opp_rush_epa_allowed", "defense", _b_opp_rush_epa, "designed runs"),
    _Metric("opp_stacked_box_rate", "defense", _b_opp_stacked_box, "rush attempts"),
    _Metric("opp_explosive_rate_allowed", "defense", _b_opp_explosive, "plays"),
    _Metric("opp_missed_tackle_rate", "defense", _b_opp_missed_tackle, "tackle attempts"),
    _Metric("opp_man_coverage_rate", "defense", _b_opp_man, "charted dropbacks", lag=True),
    _Metric("opp_def_epa_per_play", "defense", _b_opp_def_epa, "plays"),
    _Metric("opp_rz_pass_epa_allowed", "defense", _b_opp_rz_pass_epa, "red-zone dropbacks"),
    _Metric("opp_rz_run_epa_allowed", "defense", _b_opp_rz_run_epa, "red-zone designed runs"),
)
_METRIC_BY_ID: dict[str, _Metric] = {m.mid: m for m in _METRICS}


# ---------------------------------------------------------------------------
# Reduction: component rows -> per-team value over a set of weeks
# ---------------------------------------------------------------------------
@dataclass
class _Reduced:
    frame: pd.DataFrame                       # index team; value, games, den, [player, name]
    players: pd.DataFrame | None = None       # index player; value, den  (player metrics only)
    last_week: dict[str, int] = field(default_factory=dict)


def _empty_reduced() -> _Reduced:
    return _Reduced(pd.DataFrame({"value": pd.Series(dtype=float), "games": pd.Series(dtype=float),
                                  "den": pd.Series(dtype=float)}, index=pd.Index([], name="team")))


def _reduce(metric: _Metric, rows: pd.DataFrame, weeks: Collection[int] | None) -> _Reduced:
    if rows is None or rows.empty:
        return _empty_reduced()
    r = rows if weeks is None else rows[rows["week"].isin(list(weeks))]
    r = r[r["den"] > 0]
    if r.empty:
        return _empty_reduced()
    last = {str(k): int(v) for k, v in r.groupby("team")["week"].max().items()}
    if metric.reducer == "ratio":
        g = r.groupby("team").agg(num=("num", "sum"), den=("den", "sum"), games=("week", "nunique"))
        g["value"] = g["num"] / g["den"]
        return _Reduced(g[["value", "games", "den"]], None, last)
    if metric.reducer == "player":
        tp = r.groupby(["team", "player"], as_index=False).agg(
            num=("num", "sum"), den=("den", "sum"), games=("week", "nunique"), name=("name", "first"))
        tp["value"] = tp["num"] / tp["den"]
        lead = tp.sort_values(["team", "den", "player"], ascending=[True, False, True], kind="mergesort")
        lead = lead.drop_duplicates("team").set_index("team")
        pl = r.groupby("player").agg(num=("num", "sum"), den=("den", "sum"), games=("week", "nunique"))
        pl["value"] = pl["num"] / pl["den"]
        return _Reduced(lead[["value", "games", "den", "player", "name"]], pl[["value", "den", "games"]], last)
    if metric.reducer == "top5":
        rows_out = []
        for team, grp in r.groupby("team"):
            per = grp.groupby("player")["num"].sum().sort_values(ascending=False)
            if len(per) < 5 or per.sum() <= 0:
                continue
            rows_out.append((team, float(per.head(5).sum() / per.sum()), float(grp["week"].nunique()), float(per.sum())))
        if not rows_out:
            return _empty_reduced()
        f = pd.DataFrame(rows_out, columns=["team", "value", "games", "den"]).set_index("team")
        return _Reduced(f, None, last)
    return _empty_reduced()


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------
def shrink(current: float | None, games: float, prior: float | None, k: float) -> tuple[float | None, float]:
    """``(value, weight_on_current)`` with ``w = games / (games + k)`` toward ``prior``.

    No current value (or no games) returns the prior at weight 0; no prior with a current value
    returns the current value at weight 1 (the caller passes the league mean as the prior).
    """
    cur_ok = current is not None and math.isfinite(current) and games > 0
    if not cur_ok:
        if prior is None or not math.isfinite(prior):
            return None, 0.0
        return float(prior), 0.0
    if prior is None or not math.isfinite(prior):
        return float(current), 1.0
    kk = max(float(k), 0.0)
    w = games / (games + kk) if (games + kk) > 0 else 1.0
    return float(w * current + (1.0 - w) * prior), float(w)


def regress_prior(prior_value: float, prior_games: float, league_value: float, k: float) -> tuple[float, float]:
    """``(regressed_prior, lambda)``: the prior season pulled toward the league level by its own reliability.

    ``lambda = prior_games / (prior_games + k)`` is the reliability of a ``prior_games``-game sample (the same
    K as the in-season weight), so a noisy last season (red-zone TD rate) is mostly the league level while a
    stable one (EPA/play) is mostly itself.  ``league_value`` is the league level the result is expressed on.
    """
    g = max(float(prior_games), 0.0)
    kk = max(float(k), 0.0)
    lam = g / (g + kk) if (g + kk) > 0 else 1.0
    return float(league_value + lam * (prior_value - league_value)), float(lam)


def _level_shift(cur: pd.DataFrame, prior: pd.DataFrame) -> float:
    """How far the league level moved from the prior window to the current one, noise-discounted.

    Raw shift = mean over common teams of ``cur - prior``.  Its standard error comes from the spread of the
    per-team differences; a positive-part James-Stein factor ``max(0, 1 - (se/shift)^2)`` keeps a large, real
    level change (a feed that re-defined man coverage) and drops one that is inside the noise (one week of EPA).
    """
    if cur.empty or prior.empty:
        return 0.0
    common = cur.index.intersection(prior.index)
    if len(common) < _MIN_TEAMS_FOR_GRADE:
        return 0.0
    diff = cur.loc[common, "value"].astype(float) - prior.loc[common, "value"].astype(float)
    shift = float(diff.mean())
    if not math.isfinite(shift) or shift == 0.0:
        return 0.0
    se = float(diff.std(ddof=1)) / math.sqrt(len(diff))
    if not math.isfinite(se):
        return 0.0
    return shift * max(0.0, 1.0 - (se / shift) ** 2)


@dataclass
class _Resolved:
    metric: _Metric
    values: dict[str, float]
    weight: dict[str, float]
    games: dict[str, float]
    den: dict[str, float]
    #: team -> (regressed prior used, label, raw prior value, lambda) or None
    prior: dict[str, tuple[float, str, float | None, float] | None]
    lead: dict[str, str]                           # team -> player name (player metrics)
    cur_season: int
    note: str
    last_week: dict[str, int]
    weeks_label: str
    source: str = ""                               # feed that produced the current rows ("" = the default feed)
    shift: float = 0.0                             # league-level move applied to the prior


class _Ctx:
    """Per-call memo of seasons and reduced metrics."""

    def __init__(self) -> None:
        self._seasons: dict[int, _Season] = {}
        self._reduced: dict[tuple[str, int, Any], _Reduced] = {}

    def season(self, season: int) -> _Season:
        if season not in self._seasons:
            self._seasons[season] = _Season(season)
        return self._seasons[season]

    def reduced(self, metric: _Metric, season: int, weeks: Collection[int] | None) -> _Reduced:
        key = (metric.mid, season, None if weeks is None else tuple(sorted(weeks)))
        if key not in self._reduced:
            self._reduced[key] = _reduce(metric, self.season(season).rows(metric), weeks)
        return self._reduced[key]


def _resolve(ctx: _Ctx, metric: _Metric, season: int, weeks: Collection[int] | None) -> _Resolved:
    k = SHRINKAGE_K[metric.mid]
    cur_season, cur_weeks, note = season, weeks, ""
    cur = ctx.reduced(metric, season, weeks)
    if cur.frame.empty and metric.lag:
        for back in range(1, _MAX_FEED_FALLBACK + 1):
            s2 = season - back
            if s2 < _MIN_SEASON:
                break
            c2 = ctx.reduced(metric, s2, None)
            if not c2.frame.empty:
                cur, cur_season, cur_weeks = c2, s2, None
                note = f"feed has no {season} rows; using {s2}"
                break
    prior_season = cur_season - 1
    prior_weeks: Collection[int] | None = None
    if metric.window_matched_prior and cur_season == season and cur.last_week:
        # a metric that mechanically decays with window length (top-five OL snap share) is compared with
        # last season over the SAME weeks, not over a full season
        prior_weeks = frozenset(range(1, max(cur.last_week.values()) + 1))
    prior = (ctx.reduced(metric, prior_season, prior_weeks)
             if prior_season >= _MIN_SEASON else _empty_reduced())
    cur_frame, prior_frame = cur.frame, prior.frame
    league_cur = float(cur_frame["value"].mean()) if not cur_frame.empty else None
    league_prior = float(prior_frame["value"].mean()) if not prior_frame.empty else None
    shift = _level_shift(cur_frame, prior_frame)
    # the league level the prior is expressed on: last season's mean moved by the (noise-discounted) shift
    anchor = (league_prior + shift) if league_prior is not None else None

    values: dict[str, float] = {}
    weight: dict[str, float] = {}
    games: dict[str, float] = {}
    den: dict[str, float] = {}
    priors: dict[str, tuple[float, str, float | None, float] | None] = {}
    lead: dict[str, str] = {}
    min_den = _PLAYER_PRIOR_MIN_DEN.get(metric.mid, 0.0)
    teams = sorted(set(cur_frame.index) | set(prior_frame.index))
    for team in teams:
        has_cur = team in cur_frame.index
        cur_val = float(cur_frame.at[team, "value"]) if has_cur else None
        g = float(cur_frame.at[team, "games"]) if has_cur else 0.0
        n = float(cur_frame.at[team, "den"]) if has_cur else 0.0
        raw: tuple[float, float, str] | None = None        # (prior value, prior games, label)
        if metric.reducer == "player":
            if has_cur:
                pid = cur_frame.at[team, "player"]
                if (prior.players is not None and pid in prior.players.index
                        and float(prior.players.at[pid, "den"]) >= min_den):
                    raw = (float(prior.players.at[pid, "value"]), float(prior.players.at[pid, "games"]),
                           f"{prior_season} same player")
                lead[team] = str(cur_frame.at[team, "name"] or pid)
            elif team in prior_frame.index:
                raw = (float(prior_frame.at[team, "value"]), float(prior_frame.at[team, "games"]),
                       f"{prior_season} team lead")
                lead[team] = str(prior_frame.at[team, "name"] or prior_frame.at[team, "player"])
        elif team in prior_frame.index:
            raw = (float(prior_frame.at[team, "value"]), float(prior_frame.at[team, "games"]),
                   f"{prior_season} team" + (f" wk1-{max(prior_weeks)}" if prior_weeks else ""))
        target: tuple[float, str, float | None, float] | None = None
        if raw is not None and anchor is not None:
            eff, lam = regress_prior(raw[0], raw[1], anchor, k)
            target = (eff, raw[2], raw[0], lam)
        elif league_cur is not None and has_cur:
            target = (league_cur, "league mean", None, 0.0)
        value, w = shrink(cur_val, g, target[0] if target else None, k)
        if value is None:
            continue
        values[team], weight[team], games[team], den[team], priors[team] = value, w, g, n, target
    weeks_label = (f"{cur_season} full season" if cur_weeks is None
                   else f"{cur_season} wk<={max(cur_weeks) if len(cur_weeks) else 0}")
    source = ctx.season(cur_season).sources.get(metric.mid, "")
    return _Resolved(metric, values, weight, games, den, priors, lead, cur_season, note, cur.last_week,
                     weeks_label, source, shift)


# ---------------------------------------------------------------------------
# Grades and ranks
# ---------------------------------------------------------------------------
def _percentile_grades(values: Mapping[str, float], *, higher_is_better: bool) -> dict[str, float]:
    """0-100 percentile rank across teams (best = 100, league average = 50); ties share a rank."""
    items = {t: float(v) for t, v in values.items() if v is not None and math.isfinite(v)}
    n = len(items)
    if n < _MIN_TEAMS_FOR_GRADE:
        return {}
    s = pd.Series(items)
    rank = s.rank(method="average", ascending=True)       # 1..n, higher value = higher rank
    pct = (rank - 1.0) / (n - 1.0) * 100.0
    if not higher_is_better:
        pct = 100.0 - pct
    return {str(t): float(v) for t, v in pct.items()}


def _ranks_worst_last(values: Mapping[str, float]) -> dict[str, float]:
    """1 = lowest value ... n = highest (worst when higher is worse); ties broken by team code."""
    items = [(float(v), str(t)) for t, v in values.items() if v is not None and math.isfinite(v)]
    if len(items) < _MIN_TEAMS_FOR_GRADE:
        return {}
    items.sort()
    return {t: float(i + 1) for i, (_, t) in enumerate(items)}


def _zscores(values: Mapping[str, float]) -> dict[str, float]:
    s = pd.Series({t: float(v) for t, v in values.items() if v is not None and math.isfinite(v)})
    if len(s) < _MIN_TEAMS_FOR_GRADE:
        return {}
    sd = float(s.std(ddof=0))
    if sd <= 0:
        return {str(t): 0.0 for t in s.index}
    return {str(t): float((v - s.mean()) / sd) for t, v in s.items()}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _what(key: str) -> str:
    """First sentence of the static description (kept short for per-team basis strings)."""
    return FEATURE_BASIS[key].split(". ")[0].rstrip(".")


def _basis_text(resolved: _Resolved, team: str, what: str) -> str:
    g = resolved.games.get(team, 0.0)
    n = resolved.den.get(team, 0.0)
    w = resolved.weight.get(team, 0.0)
    k = SHRINKAGE_K[resolved.metric.mid]
    prior = resolved.prior.get(team)
    prior_text = f"{prior[1]} {prior[0]:.3f}" if prior else "no prior"
    if g > 0:
        last = resolved.last_week.get(team)
        span = f"{resolved.cur_season} wk1-{last}" if (last and resolved.cur_season) else f"{resolved.cur_season}"
        sample = f"{span}: {int(g)} g, n={int(n)} {resolved.metric.unit}"
    else:
        sample = f"no {resolved.cur_season} games"
    lead = resolved.lead.get(team)
    who = f" lead={lead};" if lead else ""
    note = f" [{resolved.note}]" if resolved.note else ""
    return f"{what} |{who} {sample}; weight on current {w:.0%} (K={k:g} games), prior {prior_text}{note}"


def _build_table(season: int, through_week: int | None) -> dict[str, dict]:
    ctx = _Ctx()
    weeks: frozenset[int] | None = None if through_week is None else frozenset(range(1, int(through_week) + 1))
    resolved: dict[str, _Resolved] = {m.mid: _resolve(ctx, m, season, weeks) for m in _METRICS}

    offense: dict[str, dict[str, float]] = {t: {} for t in TEAMS}
    defense: dict[str, dict[str, float]] = {t: {} for t in TEAMS}
    kinds: dict[str, dict[str, str]] = {t: {} for t in TEAMS}
    basis: dict[str, dict[str, str]] = {t: {} for t in TEAMS}

    def put(side: dict[str, dict[str, float]], key: str, team: str, value: float, text: str) -> None:
        if value is None or not math.isfinite(value):
            return
        side[team][key] = round(float(value), 5)
        kinds[team][key] = _KIND[key]
        basis[team][key] = text

    def direct(side, mid: str, key: str, *, transform: Callable[[float], float] | None = None) -> None:
        res = resolved[mid]
        for team, value in res.values.items():
            if team in side:
                put(side, key, team, transform(value) if transform else value,
                    _basis_text(res, team, _what(key)))

    # --- direct offense keys
    # PROXY: the share of dropbacks on which the line did NOT allow a pressure/sack (shrunk before inverting).
    direct(offense, "ol_pressure_allowed", "ol_pass_block_win_rate", transform=lambda v: 1.0 - v)
    direct(offense, "ol_continuity", "ol_continuity")
    direct(offense, "team_epa_per_play", "team_epa_per_play")
    direct(offense, "qb_efficiency", "qb_efficiency")
    direct(offense, "explosive_pass_rate", "explosive_pass_rate")
    direct(offense, "explosive_run_rate", "explosive_run_rate")
    direct(offense, "pace_seconds_per_play", "pace_seconds_per_play")
    direct(offense, "red_zone_efficiency", "red_zone_efficiency")
    direct(offense, "proe", "proe")
    direct(offense, "proe", "rroe", transform=lambda v: -v)
    direct(offense, "team_rz_pass_rate", "team_rz_pass_rate")
    direct(offense, "wr1_separation", "wr1_separation")
    direct(offense, "rb_yards_after_contact", "rb_yards_after_contact")
    direct(offense, "qb_deep_attempt_rate", "qb_deep_attempt_rate")
    # --- run-block composite (derived after shrinking both components)
    ybc, rep = resolved["ol_run_ybc_per_carry"], resolved["off_rush_epa"]
    z_ybc, z_rep = _zscores(ybc.values), _zscores(rep.values)
    comp: dict[str, float] = {}
    for team in TEAMS:
        parts = [z[team] for z in (z_ybc, z_rep) if team in z]
        if parts:
            comp[team] = float(np.mean(parts))
    # Re-standardise the composite across teams and place it on the scale the consuming models read an
    # ``ol_run_block_win_rate`` on (RUN_BLOCK_SCALE: mean 0.70, sd 0.04). The proxy is a RELATIVE standing: there is
    # no free measured run-block win rate, so its league mean is fixed by construction, and a user-supplied
    # PFF/ESPN-style number stays comparable with it.
    if len(comp) >= _MIN_TEAMS_FOR_GRADE:
        c_mean = float(np.mean(list(comp.values())))
        c_sd = float(np.std(list(comp.values())))
        for team, value in comp.items():
            z_std = (value - c_mean) / c_sd if c_sd > 0 else 0.0
            scaled = RUN_BLOCK_SCALE[0] + RUN_BLOCK_SCALE[1] * max(-3.0, min(3.0, z_std))
            used = [name for name, z in (("PFR yards-before-contact/carry", z_ybc), ("rush EPA/designed run", z_rep)) if team in z]
            src = ybc if team in z_ybc else rep
            put(offense, "ol_run_block_win_rate", team, scaled,
                "proxy: z(" + " + ".join(used) + f") re-standardised across teams and placed on the models' scale "
                f"(mean {RUN_BLOCK_SCALE[0]:.2f}, sd {RUN_BLOCK_SCALE[1]:.2f}) | "
                + _basis_text(src, team, "components shrunk"))

    # --- direct defense keys
    direct(defense, "opp_pressure_rate", "opp_pressure_rate")
    direct(defense, "opp_blitz_rate", "opp_blitz_rate")
    direct(defense, "opp_simulated_pressure_success", "opp_simulated_pressure_success")
    direct(defense, "opp_pass_rush_win_rate", "opp_pass_rush_win_rate")
    direct(defense, "opp_stacked_box_rate", "opp_stacked_box_rate")
    direct(defense, "opp_missed_tackle_rate", "opp_missed_tackle_rate")
    direct(defense, "opp_man_coverage_rate", "opp_man_coverage_rate")
    direct(defense, "opp_def_epa_per_play", "opp_def_epa_per_play")

    # --- grades (percentile of the shrunk metric, lower allowed = better)
    def grades(mid: str, key: str) -> None:
        res = resolved[mid]
        for team, pct in _percentile_grades(res.values, higher_is_better=False).items():
            put(defense, key, team, pct, _basis_text(res, team, _what(key)))

    grades("opp_pass_epa_allowed", "opp_coverage_grade")
    grades("opp_rush_epa_allowed", "opp_run_defense_grade")
    grades("opp_explosive_rate_allowed", "opp_explosive_play_prevention")
    grades("opp_rz_pass_epa_allowed", "opp_rz_pass_defense_grade")
    grades("opp_rz_run_epa_allowed", "opp_rz_run_defense_grade")
    # --- defense rank by EPA/play allowed (1 best ... 32 worst)
    res = resolved["opp_def_epa_per_play"]
    for team, rank in _ranks_worst_last(res.values).items():
        put(defense, "opp_defense_rank", team, rank, _basis_text(res, team, _what("opp_defense_rank")))

    games = ctx.season(season).games_through(weeks)
    if not games and weeks is not None:
        games = {}
    table: dict[str, dict] = {}
    for team in TEAMS:
        if not offense[team] and not defense[team]:
            continue
        table[team] = {
            "offense": {k: offense[team][k] for k in OFFENSE_KEYS if k in offense[team]},
            "defense": {k: defense[team][k] for k in DEFENSE_KEYS if k in defense[team]},
            "kinds": dict(kinds[team]),
            "basis": dict(basis[team]),
            "games": int(games.get(team, 0)),
            "season": int(season),
            "through_week": through_week,
        }
    return table


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _today() -> _dt.date:
    return _dt.date.today()


def _default_season() -> int:
    """The season in progress from September on, else the most recent completed one."""
    today = _today()
    return today.year if today.month >= 9 else today.year - 1


def _clean_week(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        week = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0, min(week, _MAX_WEEK))


def team_feature_table(season: int | None = None, *, through_week: int | None = None) -> dict[str, dict]:
    """Team features for all teams with data: ``{team: {offense, defense, kinds, basis, games, season, through_week}}``.

    ``season=None`` picks the season in progress from September on, else the last completed one.
    ``through_week=W`` uses only regular-season games with ``week <= W`` (``0`` = preseason: priors only).
    Never raises; ``{}`` when nothing could be computed.
    """
    try:
        try:
            year = _default_season() if season is None else int(season)
        except (TypeError, ValueError, OverflowError):
            year = _default_season()
        week = _clean_week(through_week)

        def build() -> dict[str, dict]:
            return _build_table(year, week)

        table = _cached(("table", year, week), build, ttl_ok=_TTL_TABLE)
        return copy.deepcopy(table)
    except Exception as error:  # noqa: BLE001 - fail soft by contract
        LOG.debug("team_feature_table failed: %s", error)
        return {}


def league_reference(season: int | None = None, *, through_week: int | None = None) -> dict[str, dict[str, float]]:
    """``{key: {"mean", "sd", "n"}}`` across the teams of :func:`team_feature_table` (sample sd, ddof=1)."""
    try:
        table = team_feature_table(season, through_week=through_week)
        out: dict[str, dict[str, float]] = {}
        for key in (*OFFENSE_KEYS, *DEFENSE_KEYS):
            side = "offense" if key in OFFENSE_KEYS else "defense"
            vals = [row[side][key] for row in table.values() if key in row[side]]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out[key] = {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                        "n": float(len(arr))}
        return out
    except Exception as error:  # noqa: BLE001
        LOG.debug("league_reference failed: %s", error)
        return {}


__all__ = [
    "DEFENSE_KEYS",
    "FEATURE_BASIS",
    "NO_FREE_FEED",
    "OFFENSE_KEYS",
    "SHRINKAGE_K",
    "TEAMS",
    "clear_cache",
    "league_reference",
    "normalize_team",
    "shrink",
    "team_feature_table",
]
