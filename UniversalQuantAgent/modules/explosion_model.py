"""Explosion model: an ELITE OFFENSE against a WEAK DEFENSE (NFL model upgrade pack).

Output ``explosion_probability`` in ``[0, 1]`` -- the probability that the offence
being projected has an *explosive* game.

EVENT DEFINITION (what a probability means)
-------------------------------------------
An *explosion* is a regular-season team-game in which the offence scores **35 or
more points** (final score, overtime included).  Measured on nflverse
schedules, regular seasons 2019-2025: **437 of 3,742 team-games = 11.68 %**
(by season 9.2 % .. 15.6 %).  ``BASE_RATE = 0.117`` is that league frequency, so
``p = 0.20`` means "a one-in-five chance that this team posts 35+ points".
(The brief suggested 30+ points; measured, that is 25.8 % of team-games -- not
the 10-15 % "explosive" tail -- so the threshold was moved to 35.)

Shape of the model (a conjunction, not a sum)
---------------------------------------------
* ``offense_strength`` -- the prior-weighted mean of nine league-standardised
  offence factors (EPA/play, QB EPA/dropback, explosive pass / run rate,
  red-zone efficiency, WR1 separation, RB yards after contact, OL health, pace),
  divided by the league SD of that composite and multiplied by the offence
  rating's reliability (see below).  ``+1`` is therefore an offence whose
  *observed* composite is 1 / 0.75 = 1.3 league-SDs above average (what is
  left after shrinking an observed rating towards the league mean).
* ``defense_weakness`` -- the same for five defence factors (EPA/play allowed,
  explosive-play prevention, coverage grade, pressure rate, defence rank);
  ``+1`` is a defence whose observed composite is 1 / 0.49 = 2.0 league-SDs
  worse than average.
* They are combined as ``z = off + dfn + conjunction`` with
  ``conjunction = INTERACTION * max(0, off) * max(0, dfn) * gate_multiplier``
  where the gate multiplier runs from ``GATE_FLOOR`` (defence not near the
  bottom five) to ``1`` (``opp_defense_rank`` 28-32 of 32 = the BOTTOM FIVE; it
  ramps from rank 24).  ``z`` is soft-capped by ``Z_CAP * tanh(z / Z_CAP)`` so
  the probability can approach, but never reach, ``CEILING``::

      p = sigmoid(logit(BASE_RATE) + GAIN * z_capped)

  Consequences, all asserted in the tests: average offence vs average defence is
  *exactly* ``BASE_RATE``; elite-vs-elite and weak-vs-weak cancel to (nearly)
  the base rate; one-sided advantages move it moderately; the product term only
  fires when both sides point towards an explosion, and fires fully only against
  a bottom-five defence; the result is monotone in every single input.  The
  cancellation is incomplete and tilts towards the offence (the more reliable
  rating): +/-1 SD on every input gives 0.15 for elite-vs-elite and 0.09 for
  weak-vs-weak (base 0.117), a realistic elite offence against an elite defence
  0.18 and a weak offence against a bottom-five defence 0.07, while elite vs
  bottom-five is 0.46.  In-sample data show the same tilt (elite offence vs
  elite defence 12.8 %, weak vs weak 7.5 %, elite vs weak defence 35.8 %).

Why each side is rescaled
-------------------------
The nine offence factors are only loosely correlated (r ~0.2-0.9) while the five
defence factors are near-duplicates of one another (EPA allowed, coverage grade
and rank are r ~0.9), so the raw weighted means have different spreads across
team-seasons (SD 0.646 offence, 0.825 defence; 2019-2025, n=224).  Each side is
divided by its composite SD so +1 means the same thing on both.  Then each side
is multiplied by the reliability of the underlying rating -- the split-half
(odd vs even weeks, Spearman-Brown) full-season reliability of EPA/play: **0.75
for offence, 0.49 for defence** (n=224) -- the usual shrinkage of an observed
rating towards the league mean.  A defence's season rating simply carries less
signal than an offence's, and this reliability is a property of the *features*,
not something fitted to the 35+ outcome.

Numbers (properties of the constants, not fits)
-----------------------------------------------
``BASE_RATE`` 0.117, ``GAIN`` 0.55, ``INTERACTION`` 0.6, ``GATE_FLOOR`` 0.5,
``CEILING`` 0.80.  ``GAIN`` is the log-odds change of "35+ points" for a shift of
+3 points in expected scoring under the league points distribution (mean 22.86,
SD 9.96, n=3,742 team-games; normal approximation 12.1 % -> 19.3 %, a log-odds
change of 0.55).  +3 points is a bit under one SD of true team-offence talent
(team-season points/game has SD 4.29, of which about 2.4 is single-game noise).
Every input one SD in the explosive direction (OL health is capped at 1.0 =
perfect health = +0.69 SD; the defence-rank gate is 0.44 at an average rank of
25.8) gives ``p ~= 0.28``; every input +2 SD (gate fully open) gives
``p ~= 0.54``; every input pinned at the +3 SD clip gives ``p ~= 0.69`` -- all
below ``CEILING`` (0.80 < 0.9), far below 0.95.  The mirror image, every input
one SD in the *un*-explosive direction, gives ``p ~= 0.05``.

Input vocabulary (canonical keys; any subset may be supplied)
--------------------------------------------------------------
OFFENCE (``+`` = more explosive): ``team_epa_per_play``, ``qb_efficiency``
(EPA/dropback), ``explosive_pass_rate``, ``explosive_run_rate``,
``wr1_separation`` (alias ``wr_separation``), ``rb_yards_after_contact``,
``red_zone_efficiency``, ``ol_health`` (= 1 - ``ol_injuries`` / 5; derived from
``ol_injuries`` when only that is given), ``pace_seconds_per_play`` (``-``).
DEFENCE (weak defence = more explosive): ``opp_def_epa_per_play`` (``+``),
``opp_defense_rank`` (``+``; 1 best .. 32 worst), ``opp_pressure_rate`` (``-``),
``opp_coverage_grade`` (``-``), ``opp_explosive_play_prevention`` (``-``).
Rates may arrive as 0.31 or 31; grades are percentile ranks 0-100 (clamped to
that scale), ``opp_defense_rank`` is clamped to 1-32, ``ol_injuries`` to 0-5.
``None`` / NaN / text are *missing*; a missing input is neutral and counts
against ``coverage``, so partial data shrinks toward the base rate.  When
``opp_defense_rank`` is missing but ``opp_def_epa_per_play`` is supplied, the
bottom-five gate uses a rank estimated from the EPA (normal-CDF mapping, flagged
in ``derived_inputs``); the estimate never enters the factor score, so the EPA
is not counted twice.

Model-private extras (documented additions, per the contract): the result also
carries ``offense_strength``, ``defense_weakness``, ``bottom_five_gate`` (0-1),
``gate_rank`` / ``gate_rank_source``, ``conjunction`` (the product term),
``z_raw``, ``ceiling``, ``derived_inputs``, ``offense_coverage`` /
``defense_coverage`` and ``event``.

HONEST LIMITS
-------------
* Factor weights are **priors, not fitted coefficients**; nothing here was
  regressed on game outcomes.  ``GAIN``, ``INTERACTION``, ``GATE_FLOOR`` and
  ``CEILING`` are structural anchors, not estimates.  The association check
  below was run *after* the constants were fixed and was not used to tune them.
* Association with the event, nflverse regular seasons 2019-2025, 3,710
  team-games (full-season team features of the team and of its opponent, the
  two EPA/play-based offence and defence composites; OL health left neutral):
  in-sample Spearman(p, 35+) = 0.28, Spearman(p, points scored) = 0.44, AUC
  0.755, Brier 0.0948 vs 0.1033 for a constant 11.7 % (8 % better).  This is
  optimistic -- the season features contain the game being predicted.  Using
  the PRIOR season's features instead (n=3,230) it is Spearman 0.12, AUC 0.605
  and Brier 0.1019 vs 0.1026: close to no skill beyond the base rate, which
  is what a one-game 12 % tail event and weak year-over-year persistence of
  the ratings (EPA/play r = 0.43 offence, 0.23 defence, 2019-2025) should
  give.  Treat the output as a matchup *lean*, not a forecast.  Dropping the
  product term (``INTERACTION = 0``) moves the in-sample Brier from 0.0948 to
  0.0955: the conjunction is a structural statement of the brief, only
  marginally supported by the data.
* Calibration is loose: the mean prediction over those team-games is 13.1 %
  against 11.7 % observed (``BASE_RATE`` is the league *average* frequency, but
  the matchup of two average units sits on the convex part of the logistic and
  the empirical rate of a middle-quartile offence against a middle-quartile
  defence is about 9 %).  In-sample the lowest deciles are over-predicted
  (0.056 vs 0.013 observed) and the top decile under-predicted (0.27 vs 0.34).
  The apply-functions re-centre on ``BASE_RATE``, so this offset does not bias
  a league-average projection.
* The league references (``mean`` / ``sd``) are across *full-season team-season*
  values.  Early-season (small-sample) inputs are noisier than those SDs
  suggest; the model does not widen for that.
* Overlapping inputs (team EPA vs QB EPA vs explosive rates; EPA allowed vs
  coverage grade vs rank) are kept at modest weights and rescaled per side, but
  the double counting is limited, not removed.
* ``opp_coverage_grade`` and ``opp_explosive_play_prevention`` have no free PFF-
  style feed; the loader supplies nflverse *proxies* (percentile of pass EPA /
  20+ yard play rate allowed).  ``ol_health`` has no free *live* feed (injury
  data is never fetched live); until the caller supplies ``ol_injuries`` or
  ``ol_health`` it is neutral.
* OL health saturates.  Starters-out is rare and skewed (mean 0.37 per game,
  SD 0.53 -> health SD 0.106), so the pack-wide +/-3 SD input clip is reached
  at about two starters out: three, four or five out score the same as two.
* Empirically the bottom-five bucket is not much worse than ranks 23-27 (19.9 %
  vs 16.3 % of games allowing 35+, middle of the league 9.8 %), so the gate is a
  structural statement of the brief ("elite vs bottom-5"), not a measured
  cliff; it only scales the product term.
* The 35+ point event is a *team* outcome.  The yardage multipliers the brief
  attaches (``qb_yards`` +45 %, ``wr_yards`` +40 %, ``rb_yards`` +35 %,
  ``team_points`` +30 % at ``p = 1``) are the brief's, applied verbatim; they are
  not estimated from data.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Mapping

from modules.nfl_pack_common import (
    Factor,
    clamp,
    describe_drivers,
    logistic_probability,
    logit,
    r,
    read_input,
    scale_field,
    score_factors,
    to_float,
)

LOG = logging.getLogger("nfl_pack.explosion")

MODEL = "explosion"
OUTPUT_KEY = "explosion_probability"
STAGE = "explosion"

# ---------------------------------------------------------------------------
# The brief's coefficients, verbatim (contract section 3).
# ---------------------------------------------------------------------------
QB_YARDS_COEFFICIENT = 0.45
WR_YARDS_COEFFICIENT = 0.40
RB_YARDS_COEFFICIENT = 0.35
TEAM_POINTS_COEFFICIENT = 0.30

#: ``(projection field, coefficient)`` in the order the brief lists them.
ADJUSTED_FIELDS: tuple[tuple[str, float], ...] = (
    ("qb_yards", QB_YARDS_COEFFICIENT),
    ("wr_yards", WR_YARDS_COEFFICIENT),
    ("rb_yards", RB_YARDS_COEFFICIENT),
    ("team_points", TEAM_POINTS_COEFFICIENT),
)

# ---------------------------------------------------------------------------
# Event definition and base rate.
# ---------------------------------------------------------------------------
EVENT_THRESHOLD_POINTS = 35
EVENT_DEFINITION = "team scores 35+ points in a regular-season game (final score, overtime included)"
#: nflverse schedules, REG 2019-2025: 437 of 3,742 team-games (11.68%; by season 9.2%-15.6%).
BASE_RATE = 0.117

# ---------------------------------------------------------------------------
# Factors.  Weights are PRIORS ordered by causal importance for a 35+ point game:
#   offence:  team EPA/play (the best single summary of how good the whole
#             offence is) > QB EPA/dropback (the engine of a scoring offence;
#             overlaps team EPA, hence a bit lower) > explosive pass rate (35
#             points needs chunk plays; passing chunks are the bigger channel)
#             > OL health (a known game-day state that gates everything else)
#             > red-zone efficiency (points per trip, but noisy and weakly
#             persistent) > explosive run rate = WR1 separation (real but
#             secondary channels) > RB yards after contact (a run-game quality
#             detail) > pace (more snaps = more chances, a small effect).
#   defence:  EPA/play allowed (the best single summary) > explosive-play
#             prevention (35 points is built on chunk plays) > coverage grade
#             (a pass-EPA proxy, overlaps the first) > pressure rate (suppresses
#             the pass game but is only one lever) > defence rank (the same
#             information as EPA allowed, kept light to limit double counting;
#             its main job is the bottom-five gate below).
# League references (mean/sd) are across team-seasons (or QB/RB-seasons) of
# nflverse regular seasons 2019-2025 via nflreadpy: pbp, PFR advanced stats, NGS.
# ---------------------------------------------------------------------------
OFFENSE_FACTORS: tuple[Factor, ...] = (
    # pbp 2019-2025 REG, team-seasons n=224, EPA/play over run+pass plays: mean 0.0004, sd 0.0915
    Factor("team_epa_per_play", mean=0.000, sd=0.0915, weight=3.0, direction=+1, label="Team EPA/play",
           unit="EPA/play", aliases=("off_epa_per_play", "offensive_epa_per_play"),
           source="nflverse pbp 2019-2025 REG, team-seasons n=224"),
    # pbp 2019-2025 REG, QB-seasons >= 200 dropbacks (scrambles & sacks included) n=257: mean 0.0415, sd 0.1296
    Factor("qb_efficiency", mean=0.0415, sd=0.1296, weight=2.5, direction=+1, label="QB EPA/dropback",
           unit="EPA/db", aliases=("qb_epa_per_dropback",),
           source="nflverse pbp 2019-2025 REG, QB-seasons >=200 dropbacks n=257"),
    # pbp 2019-2025 REG, share of pass plays (sacks included) gaining 20+ yd, team-seasons n=224: mean 0.0840, sd 0.0172
    Factor("explosive_pass_rate", mean=0.0840, sd=0.0172, weight=1.5, direction=+1, label="Explosive pass rate",
           kind="rate", unit="%", source="nflverse pbp 2019-2025 REG, 20+ yd pass share, team-seasons n=224"),
    # PRIOR on the weight; reference from nflverse injury reports x snap-count top-5 OL, 2019-2025 weeks 3-18,
    # team-games n=3,294: starters out (out/doubtful = 1, questionable = 0.5) mean 0.367, sd 0.529 ->
    # health = 1 - out/5: mean 0.9265, sd 0.106.  Report-based, so long-term IR absences are under-counted
    # (a snap-count proxy that counts every missing expected starter averages 1.01 out = 0.797 health).
    Factor("ol_health", mean=0.9265, sd=0.106, weight=1.2, direction=+1, label="OL health", kind="rate", unit="%",
           source="nflverse injuries x snap_counts 2019-2025 wk3+, team-games n=3294 (report-based; no live feed)"),
    # repo definition (modules.nfl._red_zone_efficiency): TD per drive that reached the opponent 20, REG
    # 2019-2025 team-seasons n=224 (~55 trips each): mean 0.5685, sd 0.0795
    Factor("red_zone_efficiency", mean=0.5685, sd=0.0795, weight=1.0, direction=+1, label="Red-zone TD rate",
           kind="rate", unit="%", source="nflverse pbp 2019-2025 REG, TD per RZ drive, team-seasons n=224"),
    # pbp 2019-2025 REG, share of run plays (QB runs included) gaining 10+ yd, team-seasons n=224:
    # mean 0.1158, sd 0.0209
    Factor("explosive_run_rate", mean=0.1158, sd=0.0209, weight=0.8, direction=+1, label="Explosive run rate",
           kind="rate", unit="%", source="nflverse pbp 2019-2025 REG, 10+ yd run share, team-seasons n=224"),
    # NGS receiving 2019-2025 season rows (week 0), team WR1 = most targets, n=216 team-seasons:
    # avg_separation mean 2.84 yd, sd 0.446 (8 of 224 team-seasons have no NGS WR row)
    Factor("wr1_separation", mean=2.84, sd=0.446, weight=0.8, direction=+1, label="WR1 separation", unit="yd",
           aliases=("wr_separation",), source="nflverse NGS receiving 2019-2025, team WR1 n=216"),
    # PFR rushing 2019-2025 season rows, RBs with >= 100 att, n=328: yac_att mean 2.046, sd 0.396
    Factor("rb_yards_after_contact", mean=2.046, sd=0.396, weight=0.6, direction=+1, label="RB yards after contact",
           unit="yd/att", source="nflverse PFR rush 2019-2025, RB >=100 att n=328"),
    # repo definition (modules.nfl._seconds_per_play): game-clock seconds between consecutive offensive snaps of
    # one team in a game, kept when 0-60 s, REG 2019-2025 team-seasons n=224: mean 29.77, sd 1.35.  NOT filtered
    # to neutral situations (the neutral-only, same-drive variant measures 33.25 / 1.43 on the same data) -- the
    # supplying loader must use the same definition.  Faster pace (lower seconds) -> more explosive.
    # Re-pointed to the feature loader's definition (neutral-situation game-clock seconds between snaps) so the
    # reference matches what nfl_team_features actually emits: REG 2023-2025 team-seasons n=96, mean 32.68, sd 0.93.
    Factor("pace_seconds_per_play", mean=32.68, sd=0.93, weight=0.5, direction=-1, label="Seconds per play",
           unit="s", source="nflverse pbp via nfl_team_features loader (neutral situations) REG 2023-2025 team-seasons n=96"),
)

DEFENSE_FACTORS: tuple[Factor, ...] = (
    # pbp 2019-2025 REG, EPA/play allowed (run+pass), team-seasons n=224: mean 0.0001, sd 0.0738
    Factor("opp_def_epa_per_play", mean=0.000, sd=0.0738, weight=3.0, direction=+1, label="Opp EPA/play allowed",
           unit="EPA/play", aliases=("def_epa_per_play_allowed", "defensive_epa_per_play"),
           source="nflverse pbp 2019-2025 REG, team-seasons n=224"),
    # grade = percentile of 20+ yd play rate allowed, inverted (high = good D): uniform by construction.  Season
    # percentile of 32 teams, n=224 team-seasons: mean 50.0, sd 28.9.  Proxy for a PFF-style grade.
    Factor("opp_explosive_play_prevention", mean=50.0, sd=28.9, weight=2.0, direction=-1,
           label="Opp explosive-play prevention", unit="pctile", aliases=("opp_explosive_prevention",),
           source="nflverse pbp 2019-2025 proxy: inverted percentile of 20+ yd play rate allowed (n=224)"),
    # grade = inverted percentile of pass EPA/dropback allowed (proxy, uniform by construction): mean 50.0, sd 28.9
    Factor("opp_coverage_grade", mean=50.0, sd=28.9, weight=1.5, direction=-1, label="Opp coverage grade",
           unit="pctile", source="nflverse pbp 2019-2025 proxy: inverted percentile of pass EPA allowed (n=224)"),
    # PFR pass 2019-2025 weekly rows: pressures by opponent / pbp dropbacks faced, team-seasons n=223:
    # mean 0.2173, sd 0.0323.  More pressure = a better defence = less explosion (direction -1).
    # Re-pointed to the participation was_pressure | sack scale the loader and qb_collapse_model use (2023-2025,
    # defenses n=96: mean 0.300, sd 0.036). The earlier 0.217 was the lower PFR charting scale.
    Factor("opp_pressure_rate", mean=0.300, sd=0.036, weight=1.2, direction=-1, label="Opp pressure rate",
           kind="rate", unit="%", source="nflverse participation was_pressure|sack per dropback, 2023-2025 defenses n=96"),
    # derived: season rank of EPA/play allowed, 1 best .. 32 worst: uniform 1..32 -> mean 16.5, sd 9.25
    Factor("opp_defense_rank", mean=16.5, sd=9.25, weight=1.0, direction=+1, label="Opp defence rank (32 = worst)",
           unit="rank", aliases=("opp_def_rank",), source="derived from EPA/play allowed, uniform 1-32"),
)

FACTORS: tuple[Factor, ...] = OFFENSE_FACTORS + DEFENSE_FACTORS
OFFENSE_KEYS: tuple[str, ...] = tuple(f.key for f in OFFENSE_FACTORS)
DEFENSE_KEYS: tuple[str, ...] = tuple(f.key for f in DEFENSE_FACTORS)

# ---------------------------------------------------------------------------
# Side scaling (measured properties of the features, not fits to the event).
# ---------------------------------------------------------------------------
#: SD across 2019-2025 team-seasons (n=224) of the raw weighted-mean offence z (OL health unsupplied).
OFFENSE_COMPOSITE_SD = 0.646
#: SD across the same team-seasons of the raw weighted-mean defence z.
DEFENSE_COMPOSITE_SD = 0.825
#: Spearman-Brown corrected odd/even-week split-half reliability of team-season EPA/play (n=224): offence.
OFFENSE_RELIABILITY = 0.75
#: Same for EPA/play allowed (defence): a defence's rating carries about two thirds the signal.
DEFENSE_RELIABILITY = 0.49

# ---------------------------------------------------------------------------
# Combination constants (structural anchors, not fits).
# ---------------------------------------------------------------------------
#: Slope of the logistic per unit of combined score: the log-odds change of P(35+ points) for +3 points
#: of expected scoring under N(22.9, 9.96) team-game points, 2019-2025 (n=3,742): 0.55.
GAIN = 0.55
#: Full-strength weight of the elite-offence x weak-defence product term (the conjunction).
INTERACTION = 0.6
#: Share of ``INTERACTION`` that survives against a defence that is NOT near the bottom five.
GATE_FLOOR = 0.5
#: Defence rank (1 best .. 32 worst) at which the bottom-five gate is fully open.
BOTTOM_FIVE_RANK = 28
#: Rank at which the gate starts to open (it ramps linearly from here).
BOTTOM_FIVE_RAMP_START = 24
#: The probability asymptote: p can approach, never exceed, this.
CEILING = 0.80
#: Soft-cap scale: the combined score at which the logistic would just hit CEILING.
Z_CAP = (logit(CEILING) - logit(BASE_RATE)) / GAIN

#: Defence ranks in the league (the rank scale is 1..N_TEAMS).
N_TEAMS = 32
OL_STARTERS = 5

_RANK_FACTOR = next(f for f in DEFENSE_FACTORS if f.key == "opp_defense_rank")
_EPA_ALLOWED_FACTOR = next(f for f in DEFENSE_FACTORS if f.key == "opp_def_epa_per_play")
_HEALTH_FACTOR = next(f for f in OFFENSE_FACTORS if f.key == "ol_health")
#: Inputs that live on a bounded scale are clamped to it (monotone-preserving).
_SCALE_LIMITS: dict[str, tuple[float, float]] = {
    "opp_explosive_play_prevention": (0.0, 100.0),
    "opp_coverage_grade": (0.0, 100.0),
    "opp_defense_rank": (1.0, float(N_TEAMS)),
}

BASIS = (
    "Explosion probability = logistic(logit(BASE_RATE) + GAIN x z): z combines a prior-weighted offence-strength "
    "score and a prior-weighted defence-weakness score (each rescaled by its composite SD and the reliability of "
    "season EPA/play) with an elite x weak interaction gated by a bottom-five defence, soft-capped so p can "
    "approach but never reach the ceiling. Factor weights, GAIN, INTERACTION and CEILING are priors, not fitted "
    f"coefficients. League references are nflverse regular seasons 2019-2025 team-seasons (n=224); event = "
    f"{EVENT_DEFINITION}, base rate 11.68% (437 of 3,742 team-games). Grades and OL health are proxies (no free "
    "PFF-style or live injury feed); missing inputs are neutral."
)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def bottom_five_gate(rank: float | None) -> float:
    """``0`` for a defence ranked 24th or better, ramping to ``1`` at rank 28 (bottom five) and beyond."""
    if rank is None:
        return 0.0
    span = float(BOTTOM_FIVE_RANK - BOTTOM_FIVE_RAMP_START)
    return clamp((rank - BOTTOM_FIVE_RAMP_START) / span, 0.0, 1.0)


def combine_scores(
    offense_strength: float, defense_weakness: float, gate: float = 0.0
) -> tuple[float, float, float]:
    """``(z_capped, z_raw, conjunction)`` for an offence-strength / defence-weakness pair.

    ``z_raw = off + dfn + conjunction`` with ``conjunction = INTERACTION *
    max(0, off) * max(0, dfn) * (GATE_FLOOR + (1 - GATE_FLOOR) * gate)``: the
    additive parts cancel for elite-vs-elite and weak-vs-weak, the product term
    only fires when *both* sides point towards an explosion and fires fully only
    against a bottom-five defence (``gate = 1``).  ``z_capped`` is
    ``Z_CAP * tanh(z_raw / Z_CAP)`` (odd, monotone, slope 1 at 0), which bounds
    the probability below ``CEILING``.
    """
    multiplier = GATE_FLOOR + (1.0 - GATE_FLOOR) * clamp(gate, 0.0, 1.0)
    conjunction = INTERACTION * max(0.0, offense_strength) * max(0.0, defense_weakness) * multiplier
    z_raw = offense_strength + defense_weakness + conjunction
    return Z_CAP * math.tanh(z_raw / Z_CAP), z_raw, conjunction


def explosion_probability_from_scores(offense_strength: float, defense_weakness: float, gate: float = 0.0) -> float:
    """The probability for a given score pair (no input handling) -- handy for docs and tests."""
    z_capped, _, _ = combine_scores(offense_strength, defense_weakness, gate)
    return clamp(logistic_probability(BASE_RATE, z_capped, GAIN), 0.0, 1.0)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _as_dict(inputs: Any) -> dict[str, Any]:
    """A plain dict copy of a mapping (or of anything with ``.items()``, e.g. a pandas row); ``{}`` otherwise."""
    if isinstance(inputs, Mapping):
        return dict(inputs)
    items = getattr(inputs, "items", None)
    if callable(items):
        try:
            return dict(items())
        except Exception:  # a malformed object is "no data", never an error
            return {}
    return {}


def _prepare_inputs(inputs: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """A cleaned copy of ``inputs`` plus a ``derived_inputs`` audit.

    * bounded-scale inputs (grades 0-100, rank 1-32) are clamped to their scale;
    * ``ol_health`` is derived from ``ol_injuries`` (``1 - n/5``, ``n`` clamped to
      0-5) when it was not usable itself.
    """
    data = _as_dict(inputs)
    derived: dict[str, dict[str, Any]] = {}
    for factor in FACTORS:
        limits = _SCALE_LIMITS.get(factor.key)
        if limits is None:
            continue
        value, _ = read_input(data, factor)
        if value is not None:
            data[factor.key] = clamp(value, limits[0], limits[1])
    health, _ = read_input(data, _HEALTH_FACTOR)
    if health is None:
        injuries = to_float(data.get("ol_injuries"))
        if injuries is not None:
            injuries = clamp(injuries, 0.0, float(OL_STARTERS))
            data["ol_health"] = 1.0 - injuries / OL_STARTERS
            derived["ol_health"] = {"from": "ol_injuries", "ol_injuries": r(injuries, 3), "value": r(data["ol_health"], 4)}
    return data, derived


def _gate_rank(data: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """``(rank, source)`` for the bottom-five gate: the supplied rank, else one estimated from EPA allowed."""
    rank, _ = read_input(data, _RANK_FACTOR)
    if rank is not None:
        return rank, "supplied"
    epa, _ = read_input(data, _EPA_ALLOWED_FACTOR)
    if epa is None:
        return None, None
    z = clamp((epa - _EPA_ALLOWED_FACTOR.mean) / _EPA_ALLOWED_FACTOR.sd, -3.0, 3.0)
    estimate = clamp(0.5 + N_TEAMS * _normal_cdf(z), 1.0, float(N_TEAMS))
    return estimate, "estimated_from_opp_def_epa_per_play"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_explosion(inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Explosion probability for one offence-vs-defence matchup.

    ``inputs`` is a flat mapping of canonical keys (see the module docstring);
    ``None`` / ``{}`` returns the neutral result ``explosion_probability ==
    BASE_RATE``.  Never raises on bad data.  The result carries every contract
    key (``model``, ``explosion_probability``, ``probability``,
    ``reference_probability``, ``references``, ``z_score``, ``coverage``,
    ``inputs_used``, ``inputs_defaulted``, ``contributions``, ``drivers``,
    ``basis``) plus ``offense_strength`` / ``defense_weakness`` sub-scores.
    """
    data, derived = _prepare_inputs(inputs)
    offense = score_factors(data, OFFENSE_FACTORS)
    defense = score_factors(data, DEFENSE_FACTORS)

    offense_strength = OFFENSE_RELIABILITY * float(offense["z"]) / OFFENSE_COMPOSITE_SD
    defense_weakness = DEFENSE_RELIABILITY * float(defense["z"]) / DEFENSE_COMPOSITE_SD
    gate_rank, gate_source = _gate_rank(data)
    gate = bottom_five_gate(gate_rank)
    if gate_source and gate_source != "supplied":
        derived["opp_defense_rank"] = {"from": "opp_def_epa_per_play", "value": r(gate_rank or 0.0, 2),
                                       "note": "estimate used for the bottom-five gate only"}
    z_capped, z_raw, conjunction = combine_scores(offense_strength, defense_weakness, gate)
    probability = clamp(logistic_probability(BASE_RATE, z_capped, GAIN), 0.0, 1.0)

    contributions: dict[str, dict[str, Any]] = {}
    for side, scored in (("offense", offense), ("defense", defense)):
        for key, item in scored["contributions"].items():
            entry = dict(item)
            entry["side"] = side
            if key in derived:
                entry["via"] = derived[key]["from"]
            contributions[key] = entry

    total_weight = sum(f.weight for f in FACTORS)
    offense_weight = sum(f.weight for f in OFFENSE_FACTORS)
    coverage = (offense["coverage"] * offense_weight + defense["coverage"] * (total_weight - offense_weight)) / total_weight
    used = list(offense["used"]) + list(defense["used"])
    defaulted = list(offense["defaulted"]) + list(defense["defaulted"])

    result: dict[str, Any] = {
        "model": MODEL,
        OUTPUT_KEY: r(probability),
        "probability": r(probability),
        "reference_probability": r(BASE_RATE),
        "references": {OUTPUT_KEY: r(BASE_RATE)},
        "z_score": r(z_capped, 4),
        "z_raw": r(z_raw, 4),
        "coverage": r(coverage, 3),
        "inputs_used": used,
        "inputs_defaulted": defaulted,
        "contributions": contributions,
        "drivers": describe_drivers(contributions),
        "basis": BASIS,
        "offense_strength": r(offense_strength, 4),
        "defense_weakness": r(defense_weakness, 4),
        "bottom_five_gate": r(gate, 3),
        "gate_rank": None if gate_rank is None else r(gate_rank, 2),
        "gate_rank_source": gate_source,
        "conjunction": r(conjunction, 4),
        "ceiling": CEILING,
        "offense_coverage": r(offense["coverage"], 3),
        "defense_coverage": r(defense["coverage"], 3),
        "derived_inputs": derived,
        "event": EVENT_DEFINITION,
    }
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "explosion: %s=%.4f (reference %.4f, offense=%+.3f, defense=%+.3f, gate=%.2f, conjunction=%+.3f, "
            "z=%+.3f, coverage=%.0f%%, used=%d, defaulted=%d) drivers=%s",
            OUTPUT_KEY, probability, BASE_RATE, offense_strength, defense_weakness, gate, conjunction, z_capped,
            100.0 * coverage, len(used), len(defaulted), "; ".join(result["drivers"]),
        )
    return result


def _result_number(result: Any, *path: str) -> float | None:
    """Walk ``result`` along ``path`` (mappings only) and return a finite float or ``None``."""
    node: Any = result
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return to_float(node)


def apply_explosion(
    projection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    recenter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the brief's explosion boosts to a projection; ``(new_projection, adjustments)``.

    ::

        qb_yards    *= (1 + p * 0.45)      wr_yards    *= (1 + p * 0.40)
        rb_yards    *= (1 + p * 0.35)      team_points *= (1 + p * 0.30)

    ``recenter=True`` (default) divides each multiplier by the same formula at
    ``BASE_RATE`` so a league-average situation is an exact no-op;
    ``recenter=False`` is the literal brief formula.  The input mapping is never
    mutated; a field that is absent or non-numeric is recorded as
    ``applied=False``; unrelated keys pass through untouched.
    """
    adjusted: dict[str, Any] = dict(projection) if isinstance(projection, Mapping) else {}
    adjustments: list[dict[str, Any]] = []

    probability = _result_number(result, OUTPUT_KEY)
    if probability is None:
        probability = _result_number(result, "probability")
    if probability is None:
        for field, _ in ADJUSTED_FIELDS:
            adjustments.append(
                {"stage": STAGE, "field": field, "applied": False, "reason": "result carries no explosion probability"}
            )
        return adjusted, adjustments
    probability = clamp(probability, 0.0, 1.0)

    reference = _result_number(result, "references", OUTPUT_KEY)
    if reference is None:
        reference = _result_number(result, "reference_probability")
    if reference is None:
        reference = BASE_RATE
    reference = clamp(reference, 0.0, 1.0)

    for field, coefficient in ADJUSTED_FIELDS:
        scale_field(
            adjusted,
            field,
            stage=STAGE,
            probability=probability,
            coefficient=coefficient,
            direction=+1,
            reference=reference,
            recenter=recenter,
            adjustments=adjustments,
        )
    return adjusted, adjustments


__all__ = [
    "ADJUSTED_FIELDS",
    "BASE_RATE",
    "BASIS",
    "BOTTOM_FIVE_RAMP_START",
    "BOTTOM_FIVE_RANK",
    "CEILING",
    "DEFENSE_COMPOSITE_SD",
    "DEFENSE_FACTORS",
    "DEFENSE_KEYS",
    "DEFENSE_RELIABILITY",
    "EVENT_DEFINITION",
    "EVENT_THRESHOLD_POINTS",
    "FACTORS",
    "GAIN",
    "GATE_FLOOR",
    "INTERACTION",
    "OFFENSE_COMPOSITE_SD",
    "OFFENSE_FACTORS",
    "OFFENSE_KEYS",
    "OFFENSE_RELIABILITY",
    "QB_YARDS_COEFFICIENT",
    "RB_YARDS_COEFFICIENT",
    "TEAM_POINTS_COEFFICIENT",
    "WR_YARDS_COEFFICIENT",
    "Z_CAP",
    "apply_explosion",
    "bottom_five_gate",
    "combine_scores",
    "compute_explosion",
    "explosion_probability_from_scores",
]
