"""Parlay correlation engine for the NFL model upgrade pack.

The brief: *inputs* -- every collapse probability, the explosion probability, the coverage
mismatch factor, the explosive-play probabilities and the game-script probabilities; *outputs* --
a ``correlation_matrix`` and a ``parlay_strength_score``; *uses* -- boost correlated overs, boost
correlated unders, flag "mega parlays", flag "trap parlays", rank parlay opportunities.

This module is pure logic (stdlib + numpy + :mod:`modules.nfl_pack_common`).  It never imports the
parlay builder or a page: the builder imports *this* module lazily and passes its own rule-table
matrix in as ``base_matrix``.

What it is, in one paragraph
----------------------------
A **latent regime-factor model**.  A game has a handful of regimes (the QB's passing game
collapses, the pass game collapses, the run game collapses, the offence explodes, the script goes
pass-heavy, a coverage mismatch exists, the red-zone funnel favours the pass / the run).  Each
market's *relative outcome* loads on those regimes with the **brief's own coefficients** as the
loading magnitudes, the regime's variance is Bernoulli ``p (1 - p)`` of this game's probability,
and everything else about the leg is idiosyncratic noise whose size comes from the observed weekly
coefficient of variation.  Two legs that load on the same regime of the same offence are
correlated through it.  The same numbers also say whether a stack is riding a regime that is
*favourable* this game (a boost / a "mega") or *adverse* (a "trap").

PRIORS, NOT FITS
----------------
The project has no offline joint-outcome data, so **nothing here is fitted**.  The loadings are the
brief's coefficients read as "relative change in the stat when the regime is on" (a prior; the
brief applies them as ``stat *= 1 - p * c``).  The team-result loadings (explosion +0.30,
QB collapse -0.30) are a PRIOR of this module: the brief gives no team-result coefficient, only
``team_points *= 1 + p * 0.30`` for explosion.  Idiosyncratic variances are measured (below); the
two signed-factor variances are a measured one (script) and a prior (coverage / red-zone).

The model
---------
For leg ``a`` (WIN event, i.e. an under flips the sign of every loading) with loadings ``L_ak`` on
regimes ``k`` (a regime is a *(team, regime)* pair: KC's explosion is not BUF's)::

    rho_ab = sum_k L_ak L_bk s2_k / sqrt( (sum_k L_ak^2 s2_k + tau_a^2) (sum_k L_bk^2 s2_k + tau_b^2) )

clipped to +/-0.9, where ``s2_k = p_k (1 - p_k)`` of the game's current regime probability
(``PACK_RHO_CAP``).  Signed factors (coverage mismatch, red-zone funnels, the net script) use a
documented fixed variance: ``SCRIPT_NET_VARIANCE = 0.302`` (measured: sd 0.549 of the game-level
share of pass-heavy minus run-heavy snaps, nflverse pbp 2023-2025 REG, n = 1,632 team-games) and
``SIGNED_FACTOR_SD = 0.35`` (a prior).  ``tau_a^2`` is the residual of the observed weekly
variance after the regimes at league-average probabilities, with a floor of ``TAU_MIN_SHARE`` (25%)
of it::

    tau^2 = max(cv^2 - sum_k L_k^2 s2_ref_k, TAU_MIN_SHARE * cv^2)

Where the floor binds (QB passing yards: the brief's -0.65 on a 16% Bernoulli regime alone is
0.057 of a 0.084 weekly variance) the regime model claims more variance than was observed; that is a
statement about the size of the brief's coefficients, reported in :data:`IDIOSYNCRATIC`.

Loadings (signed for the over / the backed team):

    qb passing yards   qb_collapse -0.65   explosion +0.45   pass_script +0.22
    qb passing TDs     qb_collapse -0.35   red_zone_pass +0.35   pass_script +0.22
    qb interceptions   qb_collapse +0.40
    wr rec. yards      pass_game_collapse -0.55   explosion +0.40   coverage +0.25   pass_script +0.22
    wr rec. TDs        pass_game_collapse -0.40   pass_script +0.22
    te rec. yards      pass_game_collapse -0.45   coverage +0.20   pass_script +0.22
    te rec. TDs        pass_game_collapse -0.35   red_zone_pass +0.25
    receptions         0.7 x the same player's yards
    rb rushing yards   run_collapse -0.50   explosion +0.35   pass_script -0.25
    rb rushing TDs     run_collapse -0.35   red_zone_run +0.30   pass_script -0.25
    anytime TD         the larger |loading| of the position's rush / receiving TD rows
    team result        own explosion +0.30, own qb_collapse -0.30; the OPPONENT's regimes with the
                       opposite sign (PRIOR, see above)
    game total (over)  both offences, same sign, at half the team-result size (a total is the sum
                       of two offences' points)

``pass_script`` is the signed net ``positive_script_prob - negative_script_prob`` (positive =
pass-heavy, the pack's convention; the brief's +0.20 / -0.25 attempt coefficients are folded into
one 0.22 loading on the net).

What is deliberately NOT a regime
---------------------------------
``explosive_play_prob_pass`` / ``explosive_play_prob_run`` and ``neutral_script_prob`` are read
(they are part of a team's signal bundle) but carry no loading: the brief uses the explosive-play
probabilities as an *additive mean shift* (+18 / +12 yards), not as a variance driver, and the
explosion regime already carries the big-play variance.  The QB-collapse and pass-game-collapse
regimes are treated as separate, independent factors although the pass-game model itself consumes
the QB's collapse probability; that UNDERSTATES the QB-to-receiver link the factor model sees (the
rule table, which this module never replaces, carries the structural part).

How the pack's correlation is blended (additive evidence, never a replacement)
------------------------------------------------------------------------------
:func:`build_correlation_matrix` only touches a pair when it is in the same game, is not
structural (``|base| < 0.999``, not the builder's ``-2.0`` mutually-exclusive sentinel) and both
legs have signal.  Then::

    blended = clip(base + w * (pack_rho - pack_rho_reference), -0.95, 0.95)
    w       = (weight or DEFAULT_PACK_WEIGHT) * coverage * evidence

This is the pack's re-centring principle applied to correlation. The rule table already carries
the league-average link between two markets (the QB-to-receiver structure the regime model cannot
see, plus the average game environment); ``pack_rho_reference`` is what the regime model itself
says for a league-average game. So the pack contributes only how THIS game's regimes move the
link away from average, and a league-average game returns the rule table exactly. (An earlier
convex blend, ``(1 - w) * base + w * pack_rho``, pulled every stack toward the regime model's much
smaller number even in an average game -- a level bias, not information.)

* ``coverage`` -- share of the pair's loading mass that sits on regimes whose state was supplied;
* ``evidence`` -- 0 for a bundle that is indistinguishable from the league average (every regime
  within ``Z_EVIDENCE_FLOOR`` SD of its reference), rising linearly to 1 at ``Z_EVIDENCE_FULL`` SD.
  This is what keeps a missing feed neutral: ``nfl_upgrade_pipeline.team_signals({})`` reports
  ``has_signal=True`` (its derived QB-collapse / script inputs count as coverage), but every
  probability sits on its reference, so the pack adds nothing;
* ``explained`` -- the geometric mean of the two legs' regime-explained variance share. Reported
  per pair; it only scales ``w`` when ``WEIGHT_BY_EXPLAINED_SHARE`` is switched on (off by default
  now that the blend is a deviation from the reference rather than a convex mix).

Every pair report carries ``base_rho``, ``pack_rho``, ``pack_rho_reference`` (the pack's rho at
league-average regime probabilities, so ``pack_rho - pack_rho_reference`` is the game-specific
part), ``weight_used`` and ``blended_rho``.  With no signals the output matrix equals
``base_matrix`` exactly (or the identity-with-zeros matrix when there is no base).  The matrix is
not forced positive semi-definite here (the builder's ``joint_probability`` projects each block
itself, and an eigenvalue clip would loosen the structural +/-1 pairs); ``psd`` reports it.

The strength score
------------------
Per leg (informative = the leg's regimes were supplied and the game is not exactly average)::

    shift   = sum_k L_k (p_k - ref_k)           expected relative change in the stat vs an average game
    z       = shift / sqrt(sum_k L_k^2 s2_k + tau^2)
    tilt    = tanh(z / TILT_SCALE)              in (-1, 1), positive = this game helps the leg's WIN

Per parlay::

    alignment = mean tilt over informative legs                          in [-1, 1]
    lift      = joint / independent when both are supplied, else the Kirkwood product of the
                matrix's Gaussian-copula pair lifts clipped to the Frechet bounds
    fragility = max over (team, passing|rushing collapse group) of
                  (share of informative legs hurt by that collapse) x
                  clip((max p - ref) / COLLAPSE_SEVERITY_FULL, 0, 1)   in [0, 1]
    score     = clip(50 + 30 alignment + 20 clip(ln lift / ln 1.5, -1, 1)
                        - 15 fragility - 15 [trap], 0, 100);  contradiction -> min(score, 5)

Monotone non-decreasing in alignment and lift, non-increasing in fragility.  With no informative
legs the score is exactly 50.0 and every flag but ``contradiction`` is False.

Flags (thresholds are module constants; boundaries are tested):

* ``trap`` -- a stack (>= 2 informative legs, same game, same side; or same team + same direction
  with a rule-table rho >= ``STACK_BASE_RHO``) whose mean tilt is <= ``TRAP_ALIGNMENT``; OR a
  contradiction; OR a pair the rule table credits (base >= ``STACK_BASE_RHO``) whose regime-implied
  rho is <= ``PACK_NEAR_ZERO_RHO``.
* ``mega`` -- >= 2 informative legs, lift >= ``MEGA_LIFT``, alignment >= ``MEGA_ALIGNMENT``,
  fragility <= ``MEGA_FRAGILITY_MAX``, no leg with tilt <= ``STRONG_ADVERSE_TILT``, score >=
  ``MEGA_SCORE``, no contradiction.  Trap wins over mega.
* ``boost_overs`` / ``boost_unders`` -- >= 2 informative same-game legs on the same side whose
  loadings (|L| >= ``BOOST_MIN_LOADING``) agree in sign on one (team, driver) that is currently
  favourable (sign x z >= ``Z_FAVOURABLE`` for every such leg).  For this rule, fragility and the
  trap's shared-driver wording, the QB-collapse and pass-game-collapse regimes are ONE passing-collapse
  driver (the pass-game model consumes the QB's collapse probability); the pair-correlation formula
  keeps them as separate independent factors, as the brief's loading table specifies.

Calibration (measured in scratch, nflverse via nflreadpy, regular season 2023-2025)
-----------------------------------------------------------------------------------
Weekly cv per family and sample sizes are in :data:`FAMILY_CV`.  As a sanity check on the rule
table the module does NOT replace: same-team weekly correlations (n = 1,250 team-games) QB passing
yards ~ WR1 receiving yards +0.51, ~ WR2 +0.42, ~ TE1 +0.38; QB passing TDs ~ WR1 receiving TDs
+0.45; QB yards ~ QB TDs +0.51; WR1 ~ WR2 yards +0.04; QB yards ~ RB1 rushing yards -0.09; WR1
receptions ~ yards +0.76; RB1 rushing yards ~ TDs +0.43.  The regime-only pack rho for a QB-yards /
WR-yards stack is about +0.2: the regimes carry the *game-environment* part of the link, not the
structural one -- which is why the weight above is shrunk by the explained share.

HONEST LIMITS
-------------
* All loadings are priors; the team-result loadings are this module's own prior.
* Regimes of different offences are independent here (no shared pace / weather factor); the rule
  table keeps the opposing-offence links.
* The Gaussian copula and the Kirkwood lift are approximations; pass ``joint_probability`` from the
  builder for an exact number.  Sportsbooks reprice same-game parlays, so a high lift is not free
  value.
* ``mega`` and ``trap`` are screening labels, not predictions.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from modules.nfl_pack_common import clamp, r, to_float, to_rate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Weight on the game-specific part of the pack's rho (``pack_rho - pack_rho_reference``) at full coverage /
#: evidence. 1.0 would move a link by exactly what the regime model says changed; 0.75 keeps a quarter of that
#: back as shrinkage for the regime model being a prior, not a fit.
DEFAULT_PACK_WEIGHT = 0.75
#: The regime model's rho is clipped to +/- this.
PACK_RHO_CAP = 0.9
#: A blended (touched) entry is clipped to +/- this, like the builder's heuristic cap.
BLEND_CLIP = 0.95
#: ``|base|`` at or above this is a structural (nested / opposed) pair and is never touched.
STRUCTURAL_ABS_RHO = 0.999
#: The builder's "these two legs cannot both win" matrix sentinel.
CONTRADICTORY_RHO = -2.0
#: Any off-diagonal entry below this is read as the mutually-exclusive sentinel.
CONTRADICTION_BELOW = -1.5

#: Game-level variance of the net script share (pass-heavy minus run-heavy snaps).  MEASURED: sd 0.549,
#: nflverse pbp 2023-2025 REG, 1,632 team-games (|score diff| >= 7 at the snap).
SCRIPT_NET_SD = 0.549
SCRIPT_NET_VARIANCE = SCRIPT_NET_SD ** 2
#: Sd of the signed factors (coverage mismatch, red-zone funnels).  PRIOR, not measured.
SIGNED_FACTOR_SD = 0.35
#: Floor on a sigma used to standardise a regime (keeps a regime at p ~ 0 from exploding z).
SIGMA_FLOOR = 0.02
#: Idiosyncratic variance is never allowed below this share of the observed weekly variance.
TAU_MIN_SHARE = 0.25
#: Also scale the blend weight by the legs' regime-explained variance share (see module docstring). Off: the
#: deviation form already leaves the rule table's level alone, so no extra damping is needed.
WEIGHT_BY_EXPLAINED_SHARE = False
#: A bundle whose every regime is within this many SD of its reference carries no game-specific evidence.
Z_EVIDENCE_FLOOR = 0.02
#: ... and full evidence once some regime is this many SD off its reference (linear ramp between).
Z_EVIDENCE_FULL = 0.20

#: A regime is favourable / adverse to a leg once sign(loading) x z passes this many SD.
Z_FAVOURABLE = 0.5
#: Loadings smaller than this are not "drivers" for the boost rule.
BOOST_MIN_LOADING = 0.15
#: tilt = tanh(z / TILT_SCALE), z in units of the leg's weekly sd.
TILT_SCALE = 0.30
#: A stack whose mean tilt is at or below this is a trap.
TRAP_ALIGNMENT = -0.25
#: A mega parlay needs at least this alignment ...
MEGA_ALIGNMENT = 0.30
#: ... at least this lift ...
MEGA_LIFT = 1.15
#: ... at most this fragility ...
MEGA_FRAGILITY_MAX = 0.25
#: ... no leg at or below this tilt ...
STRONG_ADVERSE_TILT = -0.35
#: ... and at least this score.
MEGA_SCORE = 60.0
#: A rule-table rho at or above this makes a same-team, same-direction pair "look like a stack".
STACK_BASE_RHO = 0.20
#: A regime-implied rho at or below this counts as "negative / near zero" against a rule-table boost.
PACK_NEAR_ZERO_RHO = 0.01
#: Probability points above reference at which a collapse regime is a full-severity exposure.
COLLAPSE_SEVERITY_FULL = 0.15

#: Neutral score (no informative legs) and the score formula's points.
NEUTRAL_SCORE = 50.0
SCORE_ALIGNMENT_POINTS = 30.0
SCORE_LIFT_POINTS = 20.0
SCORE_FRAGILITY_POINTS = 15.0
SCORE_TRAP_POINTS = 15.0
#: A contradictory ticket cannot win; its score is capped here.
CONTRADICTION_SCORE_CAP = 5.0
#: Lift at which the lift term saturates (ln lift / ln LIFT_FULL).
LIFT_FULL = 1.5

#: Ratio of a receptions loading to the same player's yards loading (brief-derived prior).
RECEPTIONS_TO_YARDS = 0.7

MODEL_BASIS = (
    "Latent regime-factor model with PRIOR loadings (the brief's own coefficients, not fitted to joint "
    "outcomes); regime variance is Bernoulli p(1-p) of this game's probability, signed factors use a "
    "documented variance; idiosyncratic variance from nflverse 2023-2025 weekly player stats. Nothing was "
    "fitted to joint outcomes."
)

# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------
REGIMES: tuple[str, ...] = (
    "qb_collapse",
    "pass_game_collapse",
    "run_collapse",
    "explosion",
    "pass_script",
    "coverage",
    "red_zone_pass",
    "red_zone_run",
)

REGIME_LABELS: dict[str, str] = {
    "qb_collapse": "QB collapse risk",
    "pass_game_collapse": "pass-game collapse risk",
    "run_collapse": "run-game collapse risk",
    "explosion": "explosion chance",
    "pass_script": "net pass-heavy script",
    "coverage": "coverage mismatch",
    "red_zone_pass": "red-zone pass funnel",
    "red_zone_run": "red-zone run funnel",
}

#: regime -> (kind, pack signal key, league-average default used only if the bundle carries no reference).
#: Defaults are the sibling models' documented neutral values (qb 0.16, pass game 0.12, run 0.176,
#: explosion 0.117, net script 0.2968 - 0.1872).
_REGIME_SOURCE: dict[str, tuple[str, str, float]] = {
    "qb_collapse": ("prob", "collapse_probability", 0.16),
    "pass_game_collapse": ("prob", "pass_game_collapse", 0.12),
    "run_collapse": ("prob", "run_collapse", 0.176),
    "explosion": ("prob", "explosion_probability", 0.117),
    "pass_script": ("script", "positive_script_prob", 0.2968 - 0.1872),
    "coverage": ("signed", "coverage_mismatch_factor", 0.0),
    "red_zone_pass": ("signed", "rz_pass_funnel", 0.0),
    "red_zone_run": ("signed", "rz_run_funnel", 0.0),
}
_SCRIPT_REF_POS = 0.2968
_SCRIPT_REF_NEG = 0.1872

#: Collapse regimes grouped for the fragility measure: the pass game and its QB are one exposure.
COLLAPSE_GROUPS: dict[str, tuple[str, ...]] = {
    "passing": ("qb_collapse", "pass_game_collapse"),
    "rushing": ("run_collapse",),
}


def _exposure_key(team: str, regime: str) -> tuple[str, str]:
    """The (team, driver) key the boost rule groups on: a collapse group's regimes are one driver."""
    for group, regimes in COLLAPSE_GROUPS.items():
        if regime in regimes:
            return team, f"{group}_collapse"
    return team, regime


def _reference_variance(regime: str) -> float:
    kind, _, default = _REGIME_SOURCE[regime]
    if kind == "prob":
        return default * (1.0 - default)
    if kind == "script":
        return SCRIPT_NET_VARIANCE
    return SIGNED_FACTOR_SD ** 2


#: Variance of each regime at its league-average state (what ``tau`` is derived against).
REFERENCE_VARIANCE: dict[str, float] = {regime: _reference_variance(regime) for regime in REGIMES}

# ---------------------------------------------------------------------------
# Loadings (market family -> {regime: signed loading for the OVER / backed team})
# ---------------------------------------------------------------------------
LOADINGS: dict[str, dict[str, float]] = {
    "qb_passing_yards": {"qb_collapse": -0.65, "explosion": 0.45, "pass_script": 0.22},
    "qb_passing_tds": {"qb_collapse": -0.35, "red_zone_pass": 0.35, "pass_script": 0.22},
    "qb_interceptions": {"qb_collapse": 0.40},
    "wr_receiving_yards": {"pass_game_collapse": -0.55, "explosion": 0.40, "coverage": 0.25, "pass_script": 0.22},
    "wr_receiving_tds": {"pass_game_collapse": -0.40, "pass_script": 0.22},
    "te_receiving_yards": {"pass_game_collapse": -0.45, "coverage": 0.20, "pass_script": 0.22},
    "te_receiving_tds": {"pass_game_collapse": -0.35, "red_zone_pass": 0.25},
    "rb_rushing_yards": {"run_collapse": -0.50, "explosion": 0.35, "pass_script": -0.25},
    "rb_rushing_tds": {"run_collapse": -0.35, "red_zone_run": 0.30, "pass_script": -0.25},
    # PRIOR (the brief gives no team-result coefficient; explosion 0.30 is its team_points coefficient).
    "team_result": {"explosion": 0.30, "qb_collapse": -0.30},
    # A game total is the sum of two offences: each offence loads at half the team-result size.
    "game_total": {"explosion": 0.15, "qb_collapse": -0.15},
}
#: The brief has no receiving coefficient for a running back (pass-game collapse covers WR / TE only).
RB_RECEIVING_TD_LOADINGS: dict[str, float] = {}


def _larger(*tables: Mapping[str, float]) -> dict[str, float]:
    """Element-wise the loading of larger magnitude across ``tables`` (the first wins a tie)."""
    out: dict[str, float] = {}
    for table in tables:
        for regime, loading in table.items():
            if regime not in out or abs(loading) > abs(out[regime]):
                out[regime] = loading
    return out


def _derive_loadings() -> None:
    for pos in ("wr", "te"):
        LOADINGS[f"{pos}_receptions"] = {
            regime: round(loading * RECEPTIONS_TO_YARDS, 4) for regime, loading in LOADINGS[f"{pos}_receiving_yards"].items()
        }
    LOADINGS["rb_anytime_td"] = _larger(LOADINGS["rb_rushing_tds"], RB_RECEIVING_TD_LOADINGS)
    LOADINGS["wr_anytime_td"] = _larger(LOADINGS["wr_receiving_tds"])
    LOADINGS["te_anytime_td"] = _larger(LOADINGS["te_receiving_tds"])


_derive_loadings()

# ---------------------------------------------------------------------------
# Idiosyncratic variance (observed weekly coefficient of variation)
# ---------------------------------------------------------------------------
_CV_SOURCE = "nflverse weekly player stats, regular season 2023-2025 (nflreadpy), player-seasons with >= 8 games"
#: family -> measured weekly cv (median over qualifying player-seasons), the number of player-seasons /
#: team-games behind it and the games they cover.  Re-derived in scratch for this module.
FAMILY_CV: dict[str, dict[str, Any]] = {
    "qb_passing_yards": {"cv": 0.29, "n_units": 98, "n_games": 1334, "unit": "QB-seasons (>= 15 att/game)"},
    "qb_passing_tds": {"cv": 0.77, "n_units": 97, "n_games": 1325, "unit": "QB-seasons (>= 15 att/game)"},
    "qb_interceptions": {"cv": 1.18, "n_units": 96, "n_games": 1300, "unit": "QB-seasons (>= 15 att/game)"},
    "wr_receiving_yards": {"cv": 0.62, "n_units": 202, "n_games": 2979, "unit": "WR-seasons (mean >= 35 yds)"},
    "wr_receptions": {"cv": 0.50, "n_units": 178, "n_games": 2619, "unit": "WR-seasons (mean >= 3 rec)"},
    "wr_receiving_tds": {"cv": 1.70, "n_units": 215, "n_games": 3082, "unit": "WR-seasons (mean >= 0.2 TD)"},
    "wr_anytime_td": {"cv": 1.56, "n_units": 213, "n_games": 3053, "unit": "WR-seasons (TD rate >= 0.20)"},
    "te_receiving_yards": {"cv": 0.67, "n_units": 102, "n_games": 1480, "unit": "TE-seasons (mean >= 25 yds)"},
    "te_receptions": {"cv": 0.54, "n_units": 100, "n_games": 1446, "unit": "TE-seasons (mean >= 2.5 rec)"},
    "te_receiving_tds": {"cv": 1.90, "n_units": 103, "n_games": 1451, "unit": "TE-seasons (mean >= 0.15 TD)"},
    "te_anytime_td": {"cv": 1.60, "n_units": 77, "n_games": 1064, "unit": "TE-seasons (TD rate >= 0.20)"},
    "rb_rushing_yards": {"cv": 0.56, "n_units": 133, "n_games": 1999, "unit": "RB-seasons (mean >= 30 yds)"},
    "rb_rushing_tds": {"cv": 1.53, "n_units": 131, "n_games": 1951, "unit": "RB-seasons (mean >= 0.2 TD)"},
    "rb_anytime_td": {"cv": 1.32, "n_units": 147, "n_games": 2192, "unit": "RB-seasons (TD rate >= 0.20)"},
    # points scored by a team (a proxy for "the team's result"), schedules 2023-2025 REG
    "team_result": {"cv": 0.44, "n_units": 1632, "n_games": 816, "unit": "team-games (points scored)"},
    "game_total": {"cv": 0.30, "n_units": 816, "n_games": 816, "unit": "games (total points)"},
}
#: Number of offences whose regimes enter a family's relative outcome (a team result / total sees two).
_OFFENCE_COUNT = {"team_result": 2, "game_total": 2}


def _build_idiosyncratic() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for family, meta in FAMILY_CV.items():
        cv2 = float(meta["cv"]) ** 2
        count = _OFFENCE_COUNT.get(family, 1)
        regime_var = count * sum(load * load * REFERENCE_VARIANCE[regime] for regime, load in LOADINGS[family].items())
        residual = cv2 - regime_var
        floor = TAU_MIN_SHARE * cv2
        out[family] = {
            "cv": float(meta["cv"]),
            "cv2": round(cv2, 4),
            "regime_variance_ref": round(regime_var, 4),
            "tau2": max(residual, floor),
            "binding": "residual" if residual >= floor else "floor",
            "n_units": meta["n_units"],
            "n_games": meta["n_games"],
            "unit": meta["unit"],
            "source": _CV_SOURCE,
        }
    return out


#: family -> {cv, cv2, regime_variance_ref, tau2, binding, n_units, n_games, unit, source}
IDIOSYNCRATIC: dict[str, dict[str, Any]] = _build_idiosyncratic()

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
_MISSING_TEXT = frozenset({"", "nan", "none", "null", "<na>", "nat"})
_POSITIONS = ("QB", "RB", "WR", "TE")
_POSITION_ALIASES = {"HB": "RB", "FB": "RB", "TB": "RB"}
_TOTAL_MARKETS = frozenset({"total", "game_total", "total_points", "totals", "game_total_points", "over_under"})
_MARKET_ALIASES = {
    "passing_yds": "passing_yards",
    "pass_yards": "passing_yards",
    "pass_tds": "passing_tds",
    "passing_touchdowns": "passing_tds",
    "passing_interceptions": "interceptions",
    "ints": "interceptions",
    "rushing_yds": "rushing_yards",
    "rush_yards": "rushing_yards",
    "rush_tds": "rushing_tds",
    "rushing_touchdowns": "rushing_tds",
    "receiving_yds": "receiving_yards",
    "rec_yards": "receiving_yards",
    "rec_tds": "receiving_tds",
    "receiving_touchdowns": "receiving_tds",
    "anytime_touchdown": "anytime_td",
    "anytime_td_scorer": "anytime_td",
}
#: market -> (the one position it is modelled for, family)
_SINGLE_POSITION_MARKETS = {
    "passing_yards": ("QB", "qb_passing_yards"),
    "passing_tds": ("QB", "qb_passing_tds"),
    "interceptions": ("QB", "qb_interceptions"),
    "rushing_yards": ("RB", "rb_rushing_yards"),
    "rushing_tds": ("RB", "rb_rushing_tds"),
}
_RECEIVING_MARKETS = frozenset({"receiving_yards", "receiving_tds", "receptions"})
_NORMAL = NormalDist()
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(48)


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        text = str(value).strip()
    except Exception:  # pragma: no cover - exotic objects with a broken __str__
        return None
    return None if text.lower() in _MISSING_TEXT else text


def _code(value: Any) -> str | None:
    text = _text(value)
    return text.upper() if text else None


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.casefold()).strip("_")


def _view(leg: Any) -> dict[str, Any]:
    """Normalised read-only view of a leg dict; tolerates ``None`` / junk (everything becomes ``None``)."""
    src: Mapping[str, Any] = leg if isinstance(leg, Mapping) else {}
    market = (_text(src.get("market")) or "").lower().replace(" ", "_").replace("-", "_") or None
    market = _MARKET_ALIASES.get(market or "", market)
    leg_type = (_text(src.get("leg_type")) or "").lower()
    if leg_type not in ("moneyline", "spread", "prop"):
        leg_type = market if market in ("moneyline", "spread") else ("prop" if market else "")
    position = (_text(src.get("position")) or "").upper()
    position = _POSITION_ALIASES.get(position, position)
    position = position if position in _POSITIONS else None
    name = _text(src.get("player_name"))
    player = _text(src.get("player_id")) or (f"name:{_slug(name)}" if name else None)
    prob = to_float(src.get("model_probability"))
    if prob is None:
        prob = to_float(src.get("probability"))
    return {
        "type": leg_type,
        "market": market,
        "side": (_text(src.get("side")) or "").lower() or None,
        "team": _code(src.get("team")),
        "opponent": _code(src.get("opponent")),
        "home": _code(src.get("home_team")),
        "away": _code(src.get("away_team")),
        "game_id": _text(src.get("game_id")),
        "player": player,
        "player_name": name,
        "position": position,
        "line": to_float(src.get("line")),
        "prob": prob,
        "description": _text(src.get("description")),
    }


def _label(view: Mapping[str, Any]) -> str:
    if view.get("description"):
        return str(view["description"])
    market = str(view.get("market") or "leg").replace("_", " ")
    who = view.get("player_name") or view.get("player") or view.get("team") or "leg"
    return f"{who} {view.get('side') or ''} {market}".replace("  ", " ").strip()


def _teams(view: Mapping[str, Any]) -> set[str]:
    return {t for t in (view.get("team"), view.get("opponent"), view.get("home"), view.get("away")) if t}


def _same_game(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a.get("game_id") and b.get("game_id"):
        return a["game_id"] == b["game_id"]
    return bool(_teams(a) & _teams(b))


def _game_key(view: Mapping[str, Any]) -> str | None:
    if view.get("game_id"):
        return "id:" + str(view["game_id"])
    for pair in ((view.get("home"), view.get("away")), (view.get("team"), view.get("opponent"))):
        if pair[0] and pair[1]:
            return "teams:" + "|".join(sorted(pair))
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _phi_inv(p: float) -> float:
    return _NORMAL.inv_cdf(min(max(float(p), 1e-12), 1.0 - 1e-12))


def _bvn_cdf(h: float, k: float, rho: float) -> float:
    """``P(Z1 <= h, Z2 <= k)`` at correlation ``rho`` (Plackett identity, Gauss-Legendre in ``asin rho``)."""
    if rho >= 1.0 - 1e-12:
        return _phi(min(h, k))
    if rho <= -1.0 + 1e-12:
        return max(0.0, _phi(h) + _phi(k) - 1.0)
    base = _phi(h) * _phi(k)
    if abs(rho) < 1e-15:
        return base
    theta_end = math.asin(rho)
    theta = 0.5 * theta_end * (_GL_NODES + 1.0)
    exponent = -(h * h - 2.0 * h * k * np.sin(theta) + k * k) / (2.0 * np.cos(theta) ** 2)
    integral = 0.5 * theta_end * float(np.sum(_GL_WEIGHTS * np.exp(exponent)))
    return min(1.0, max(0.0, base + integral / (2.0 * math.pi)))


# ---------------------------------------------------------------------------
# Regime state of one team's signal bundle
# ---------------------------------------------------------------------------
def _signal_maps(signal: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(signal, Mapping):
        return {}, {}
    probs = signal.get("probabilities")
    refs = signal.get("references")
    if not isinstance(probs, Mapping):
        probs = signal  # a flat mapping of probabilities is accepted too
    if not isinstance(refs, Mapping):
        refs = {}
    return probs, refs


def _state_raw(regime: str, probs: Mapping[str, Any], refs: Mapping[str, Any]) -> dict[str, Any]:
    kind, key, default_ref = _REGIME_SOURCE[regime]
    state: dict[str, Any] = {"p": None, "reference": default_ref, "sigma": None, "z": None,
                             "variance": None, "available": False, "kind": kind, "source": key}
    if kind == "prob":
        ref = to_rate(refs.get(key))
        ref = default_ref if ref is None else ref
        state["reference"] = ref
        state["sigma"] = max(math.sqrt(ref * (1.0 - ref)), SIGMA_FLOOR)
        p = to_rate(probs.get(key))
        if p is None:
            return state
        state.update(p=p, z=(p - ref) / state["sigma"], variance=p * (1.0 - p), available=True)
        return state
    if kind == "script":
        ref_pos = to_rate(refs.get("positive_script_prob"))
        ref_neg = to_rate(refs.get("negative_script_prob"))
        ref_pos = _SCRIPT_REF_POS if ref_pos is None else ref_pos
        ref_neg = _SCRIPT_REF_NEG if ref_neg is None else ref_neg
        ref = ref_pos - ref_neg
        state["reference"] = ref
        state["sigma"] = SCRIPT_NET_SD
        pos = to_rate(probs.get("positive_script_prob"))
        neg = to_rate(probs.get("negative_script_prob"))
        neutral = to_rate(probs.get("neutral_script_prob"))
        if neg is None and pos is not None and neutral is not None:
            neg = max(0.0, 1.0 - pos - neutral)
        if pos is None or neg is None:
            return state
        net = pos - neg
        state.update(p=net, z=(net - ref) / SCRIPT_NET_SD, variance=SCRIPT_NET_VARIANCE, available=True)
        return state
    ref = to_float(refs.get(key))
    ref = default_ref if ref is None else clamp(ref, -1.0, 1.0)
    state["reference"] = ref
    state["sigma"] = SIGNED_FACTOR_SD
    value = to_float(probs.get(key))
    if value is None:
        return state
    value = clamp(value, -1.0, 1.0)
    state.update(p=value, z=(value - ref) / SIGNED_FACTOR_SD, variance=SIGNED_FACTOR_SD ** 2, available=True)
    return state


def _regime_state_raw(signal: Any) -> dict[str, dict[str, Any]]:
    probs, refs = _signal_maps(signal)
    return {regime: _state_raw(regime, probs, refs) for regime in REGIMES}


def regime_state(team_signal: Any) -> dict[str, dict[str, Any]]:
    """``{regime: {"p", "reference", "sigma", "z", "variance", "available"}}`` for one team's signals.

    ``team_signal`` is ``nfl_upgrade_pipeline.team_signals(context)`` output (``{"probabilities",
    "references", ...}``) or a flat mapping of probabilities.  ``p`` is the regime's probability (the
    net ``positive - negative`` script share for ``pass_script``; the signed factor for ``coverage`` /
    ``red_zone_*``); ``reference`` the league-average value (the bundle's own, else the sibling model's
    documented neutral); ``sigma`` is ``sqrt(ref (1 - ref))`` for a probability (the documented sd for a
    signed factor) and ``z = (p - reference) / sigma``.  ``variance`` is the *game's* regime variance
    (``p (1 - p)`` for a probability, the documented constant for a signed factor).  A regime the bundle
    does not supply has ``p = None`` and ``available = False``.  Never raises.
    """
    out: dict[str, dict[str, Any]] = {}
    for regime, raw in _regime_state_raw(team_signal).items():
        out[regime] = {
            "p": None if raw["p"] is None else r(raw["p"], 4),
            "reference": r(raw["reference"], 4),
            "sigma": None if raw["sigma"] is None else r(raw["sigma"], 4),
            "z": None if raw["z"] is None else r(raw["z"], 4),
            "variance": None if raw["variance"] is None else r(raw["variance"], 4),
            "available": bool(raw["available"]),
        }
    return out


def _usable_signal(signal: Any) -> bool:
    if not isinstance(signal, Mapping):
        return False
    if "has_signal" in signal and not signal["has_signal"]:
        return False
    return True


def _team_states(signals_by_team: Any) -> dict[str, dict[str, Any]]:
    """``{TEAM: {"states", "evidence"}}`` for every team with a usable bundle (never mutates the input)."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(signals_by_team, Mapping):
        return out
    for key, signal in signals_by_team.items():
        code = _code(key)
        if code is None or not _usable_signal(signal):
            continue
        states = _regime_state_raw(signal)
        zs = [abs(s["z"]) for s in states.values() if s["available"]]
        if not zs:
            continue
        evidence = clamp((max(zs) - Z_EVIDENCE_FLOOR) / (Z_EVIDENCE_FULL - Z_EVIDENCE_FLOOR), 0.0, 1.0)
        out[code] = {"states": states, "evidence": evidence}
    return out


def _state_of(tstates: Mapping[str, Any], team: str, regime: str) -> dict[str, Any] | None:
    entry = tstates.get(team)
    if entry is None:
        return None
    state = entry["states"].get(regime)
    return state if state is not None and state["available"] else None


# ---------------------------------------------------------------------------
# Legs -> market family -> exposures
# ---------------------------------------------------------------------------
def _family(view: Mapping[str, Any]) -> tuple[str | None, bool]:
    """``(family, position_inferred)`` of a leg view; ``(None, False)`` when the pack has no loading row."""
    market, leg_type, pos = view.get("market"), view.get("type"), view.get("position")
    if leg_type in ("moneyline", "spread") or market in ("moneyline", "spread"):
        return "team_result", False
    if market in _TOTAL_MARKETS:
        return "game_total", False
    if market is None:
        return None, False
    if market in _SINGLE_POSITION_MARKETS:
        wanted, family = _SINGLE_POSITION_MARKETS[market]
        return (family, pos is None) if pos in (None, wanted) else (None, False)
    if market in _RECEIVING_MARKETS:
        if pos in ("WR", "TE"):
            return f"{pos.lower()}_{market}", False
        if pos is None:
            return f"wr_{market}", True
        return None, False
    if market == "anytime_td":
        if pos in ("RB", "WR", "TE"):
            return f"{pos.lower()}_anytime_td", False
        return None, False
    return None, False


def market_family(leg: Any) -> str | None:
    """The :data:`LOADINGS` row for a leg, or ``None`` when the pack has no coefficient for it.

    ``team_result`` for a moneyline / spread, ``game_total`` for a game total, otherwise
    ``<qb|wr|te|rb>_<stat>``.  A passing prop is a QB's, a rushing prop an RB's; a receiving prop with
    no position is read as a WR's; an anytime TD needs an RB / WR / TE position (a QB's has no
    coefficient); a receiving market on an RB, or any unknown market, is ``None`` (no pack opinion).
    """
    return _family(_view(leg))[0]


def _side_sign(view: Mapping[str, Any]) -> int | None:
    side = view.get("side")
    return 1 if side == "over" else -1 if side == "under" else None


def _own_team(view: Mapping[str, Any], family: str | None) -> str | None:
    if family == "team_result":
        if view.get("team"):
            return view["team"]
        if view.get("side") == "home":
            return view.get("home")
        if view.get("side") == "away":
            return view.get("away")
        return None
    if view.get("team"):
        return view["team"]
    opponent, home, away = view.get("opponent"), view.get("home"), view.get("away")
    if opponent and home and away:
        return away if opponent == home else home if opponent == away else None
    return None


def _exposures(view: Mapping[str, Any], family: str | None) -> dict[tuple[str, str], float]:
    """``{(team, regime): loading}`` for the leg's WIN event."""
    if family is None or family not in LOADINGS:
        return {}
    base = LOADINGS[family]
    out: dict[tuple[str, str], float] = {}
    if family == "team_result":
        own = _own_team(view, family)
        opp = view.get("opponent")
        if opp is None and own is not None:
            home, away = view.get("home"), view.get("away")
            opp = away if own == home else home if own == away else None
        if own:
            for regime, load in base.items():
                out[(own, regime)] = load
        if opp and opp != own:
            for regime, load in base.items():
                out[(opp, regime)] = -load
        return out
    sign = _side_sign(view)
    if sign is None:
        return {}
    if family == "game_total":
        teams = [t for t in (view.get("home"), view.get("away")) if t]
        if not teams:
            teams = [t for t in (view.get("team"), view.get("opponent")) if t]
        for team in dict.fromkeys(teams):
            for regime, load in base.items():
                out[(team, regime)] = sign * load
        return out
    team = _own_team(view, family)
    if team is None:
        return {}
    return {(team, regime): sign * load for regime, load in base.items()}


def leg_regime_loadings(leg: Any) -> dict[str, float]:
    """Signed loadings of the leg's WIN event on its OWN offence's regimes (``{}`` = no pack opinion).

    An under flips every sign.  A team leg returns the backed team's loadings; the opponent's regimes
    (the same numbers with the opposite sign) are in :func:`leg_exposures`, because a flat
    ``{regime: loading}`` cannot hold both offences.  A game total returns the per-offence loading.
    """
    view = _view(leg)
    family, _ = _family(view)
    exposures = _exposures(view, family)
    own = _own_team(view, family)
    if family == "game_total":
        teams = sorted({team for team, _ in exposures})
        own = teams[0] if teams else None
    return {regime: load for (team, regime), load in exposures.items() if team == own}


def leg_exposures(leg: Any) -> dict[tuple[str, str], float]:
    """``{(team, regime): loading}`` for the leg's WIN event across every offence it depends on."""
    view = _view(leg)
    return _exposures(view, _family(view)[0])


def _prep(leg: Any, index: int) -> dict[str, Any]:
    view = _view(leg)
    family, inferred = _family(view)
    exposures = _exposures(view, family)
    meta = IDIOSYNCRATIC.get(family) if family else None
    return {
        "index": index,
        "leg": leg,
        "view": view,
        "family": family,
        "position_inferred": inferred,
        "exposures": exposures,
        "tau2": None if meta is None else float(meta["tau2"]),
        "label": _label(view),
        "game": _game_key(view),
        "own_team": _own_team(view, family),
    }


# ---------------------------------------------------------------------------
# Pair correlation
# ---------------------------------------------------------------------------
def _clean_rho(cov: float, va: float, vb: float, ta: float, tb: float) -> float:
    denom = math.sqrt(max((va + ta) * (vb + tb), 0.0))
    if denom <= 0.0:
        return 0.0
    return clamp(cov / denom, -PACK_RHO_CAP, PACK_RHO_CAP)


def _describe_regime(team: str, regime: str, state: Mapping[str, Any]) -> str:
    label = REGIME_LABELS[regime]
    p, ref, z = state["p"], state["reference"], state["z"]
    if state["kind"] == "prob":
        return f"{team} {label} {p * 100:.0f}% vs {ref * 100:.0f}% league average ({z:+.1f} SD)"
    return f"{team} {label} {p:+.2f} vs {ref:+.2f} league average ({z:+.1f} SD)"


def _pair_calc(pa: Mapping[str, Any], pb: Mapping[str, Any], tstates: Mapping[str, Any]) -> dict[str, Any]:
    """The pack's opinion of one pair of prepared legs (never raises)."""
    same = _same_game(pa["view"], pb["view"])
    out: dict[str, Any] = {
        "rho": None, "kind": "", "note": "", "regimes": {}, "coverage": 0.0, "evidence": 0.0,
        "explained": 0.0, "weight_scale": 0.0, "pack_rho_reference": None, "same_game": same,
    }
    ea, eb = pa["exposures"], pb["exposures"]
    if not ea or not eb or pa["tau2"] is None or pb["tau2"] is None:
        out["kind"] = "unmodelled_market"
        missing = [p["label"] for p in (pa, pb) if not p["exposures"] or p["tau2"] is None]
        out["note"] = ("No regime loading for " + " / ".join(missing)
                       + " (unknown market, position or side), so the pack has no opinion.")
        return out
    if not same:
        out["kind"] = "different_games"
        out["note"] = "Legs in different games: the pack has no opinion (they are treated as independent)."
        return out
    avail_a = {key: load for key, load in ea.items() if _state_of(tstates, *key) is not None}
    avail_b = {key: load for key, load in eb.items() if _state_of(tstates, *key) is not None}
    if not avail_a or not avail_b:
        out["kind"] = "no_signal"
        out["note"] = "A leg's team has no signals, so the pack adds nothing and the rule table stands."
        return out
    shared = sorted(key for key in avail_a if key in avail_b)
    if not shared:
        out["kind"] = "no_shared_regime"
        out["note"] = "The legs load on no common regime of one offence, so the pack has no opinion."
        return out

    def variance(avail: Mapping[tuple[str, str], float], reference: bool) -> float:
        total = 0.0
        for (team, regime), load in avail.items():
            var = REFERENCE_VARIANCE[regime] if reference else _state_of(tstates, team, regime)["variance"]
            total += load * load * var
        return total

    result = {}
    for reference in (False, True):
        cov = 0.0
        for team, regime in shared:
            var = REFERENCE_VARIANCE[regime] if reference else _state_of(tstates, team, regime)["variance"]
            cov += avail_a[(team, regime)] * avail_b[(team, regime)] * var
        va, vb = variance(avail_a, reference), variance(avail_b, reference)
        result[reference] = (cov, va, vb, _clean_rho(cov, va, vb, pa["tau2"], pb["tau2"]))
    cov, va, vb, rho = result[False]
    rho_ref = result[True][3]

    mass_total = sum(abs(v) for v in ea.values()) + sum(abs(v) for v in eb.values())
    mass_avail = sum(abs(v) for v in avail_a.values()) + sum(abs(v) for v in avail_b.values())
    coverage = mass_avail / mass_total if mass_total > 0 else 0.0
    evidence = min(tstates[team]["evidence"] for team, _ in shared)
    explained = math.sqrt(max(va / (va + pa["tau2"]), 0.0) * max(vb / (vb + pb["tau2"]), 0.0))
    scale = coverage * evidence * (explained if WEIGHT_BY_EXPLAINED_SHARE else 1.0)

    regimes: dict[str, dict[str, Any]] = {}
    for team, regime in shared:
        state = _state_of(tstates, team, regime)
        regimes[f"{team}:{regime}"] = {
            "team": team, "regime": regime, "label": REGIME_LABELS[regime],
            "loading_a": avail_a[(team, regime)], "loading_b": avail_b[(team, regime)],
            "p": r(state["p"], 4), "reference": r(state["reference"], 4), "z": r(state["z"], 3),
            "variance": r(state["variance"], 4),
            "covariance": r(avail_a[(team, regime)] * avail_b[(team, regime)] * state["variance"], 5),
        }
    top = sorted(shared, key=lambda key: -abs(avail_a[key] * avail_b[key]))[:2]
    drivers = "; ".join(_describe_regime(t, k, _state_of(tstates, t, k)) for t, k in top)
    kind = "regime_aligned" if rho >= 0.02 else "regime_opposed" if rho <= -0.02 else "regime_neutral"
    out.update(
        rho=rho, kind=kind, regimes=regimes, coverage=coverage, evidence=evidence, explained=explained,
        weight_scale=scale, pack_rho_reference=rho_ref,
        note=(f"{pa['label']} and {pb['label']} share {len(shared)} regime(s); regime-implied rho {rho:+.2f} "
              f"(league-average game {rho_ref:+.2f}). Main driver(s): {drivers}. Prior loadings, not fitted."),
    )
    return out


def _public_pair(calc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rho": None if calc["rho"] is None else r(calc["rho"], 4),
        "kind": calc["kind"],
        "note": calc["note"],
        "regimes": calc["regimes"],
        "coverage": r(calc["coverage"], 4),
        "evidence": r(calc["evidence"], 4),
        "explained": r(calc["explained"], 4),
        "weight_scale": r(calc["weight_scale"], 4),
        "pack_rho_reference": None if calc["pack_rho_reference"] is None else r(calc["pack_rho_reference"], 4),
        "same_game": bool(calc["same_game"]),
    }


def pair_correlation(leg_a: Any, leg_b: Any, signals_by_team: Any = None) -> dict[str, Any]:
    """The pack's regime-implied correlation of two legs' WIN events.

    Returns ``{"rho": float | None, "kind", "note", "regimes", ...}``.  ``rho`` is ``None`` (no pack
    opinion) when a leg's market / position / side is not modelled, the legs are in different games,
    a leg's team has no signals, or the legs share no regime of one offence.  ``kind`` is one of
    ``unmodelled_market``, ``different_games``, ``no_signal``, ``no_shared_regime``,
    ``regime_aligned`` (rho >= +0.02), ``regime_opposed`` (rho <= -0.02) or ``regime_neutral``.
    ``regimes`` maps ``"TEAM:regime"`` to the loadings, state and covariance contribution.  Never
    raises; inputs are not modified.
    """
    tstates = _team_states(signals_by_team)
    return _public_pair(_pair_calc(_prep(leg_a, 0), _prep(leg_b, 1), tstates))


def _read_matrix(matrix: Any, n: int) -> list[list[float]] | None:
    """A copy of ``matrix`` as an ``n x n`` list of floats, or ``None`` if it is not one."""
    if matrix is None:
        return None
    try:
        arr = np.array(matrix, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape != (n, n):
        return None
    return arr.tolist()


def build_correlation_matrix(
    legs: Any,
    signals_by_team: Any,
    *,
    base_matrix: Any = None,
    weight: float | None = None,
) -> dict[str, Any]:
    """Blend the pack's regime correlation into a rule-table matrix (additive evidence, never a replacement).

    Returns ``{"matrix", "pairs", "weight", "pack_pairs", "has_signal", "psd", "min_eigenvalue",
    "warnings", "basis"}``.  ``matrix`` is symmetric with a unit diagonal and entries in ``[-1, 1]``;
    a ``-2.0`` contradiction sentinel and any structural ``+/-1`` in ``base_matrix`` are preserved.
    Only same-game, non-structural pairs whose legs both have signal are touched::

        blended = clip(base + w * (pack_rho - pack_rho_reference), -0.95, 0.95)
        w       = clip(weight or DEFAULT_PACK_WEIGHT, 0, 1) * coverage * evidence

    (the pack's contribution is only how this game's regimes differ from a league-average game; see
    the module docstring).  With ``signals_by_team`` empty / ``None``, or
    no signal for the teams, the matrix equals ``base_matrix`` exactly (the identity with zeros when
    there is no base).  Each ``pairs`` entry has ``i``, ``j``, ``labels``, ``base_rho``, ``pack_rho``
    (``None`` = no opinion), ``pack_rho_reference``, ``blended_rho``, ``weight_used``, ``touched``,
    ``kind`` and ``note``.  Never raises on bad input; ``legs`` / ``signals_by_team`` are not modified.
    """
    leg_list = _as_list(legs)
    n = len(leg_list)
    preps = [_prep(leg, i) for i, leg in enumerate(leg_list)]
    tstates = _team_states(signals_by_team)
    nominal = to_float(weight)
    nominal = DEFAULT_PACK_WEIGHT if nominal is None else clamp(nominal, 0.0, 1.0)
    warnings: list[str] = []
    base = _read_matrix(base_matrix, n)
    if base_matrix is not None and base is None:
        warnings.append("base_matrix is not an n x n numeric matrix for these legs; it was ignored.")

    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    pairs: list[dict[str, Any]] = []
    touched_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            base_ij = 0.0
            if base is not None:
                upper, lower = base[i][j], base[j][i]
                value = upper if math.isfinite(upper) else lower
                if math.isfinite(value):
                    base_ij = value
                else:
                    warnings.append(f"base_matrix[{i}][{j}] is not finite; read as 0.")
            calc = _pair_calc(preps[i], preps[j], tstates)
            sentinel = base_ij <= CONTRADICTION_BELOW
            structural = abs(base_ij) >= STRUCTURAL_ABS_RHO
            w = nominal * calc["weight_scale"]
            touched = bool(calc["rho"] is not None and calc["same_game"] and not sentinel
                           and not structural and w > 0.0)
            if touched:
                ref = calc["pack_rho_reference"]
                ref = calc["rho"] if ref is None else ref
                entry = clamp(base_ij + w * (calc["rho"] - ref), -BLEND_CLIP, BLEND_CLIP)
                touched_count += 1
            else:
                # untouched: the base value, with the mutually-exclusive sentinel kept as is and any other
                # out-of-range number held to the documented [-1, 1]
                entry = base_ij if sentinel else clamp(base_ij, -1.0, 1.0)
            matrix[i][j] = matrix[j][i] = entry
            pairs.append({
                "i": i, "j": j,
                "labels": [preps[i]["label"], preps[j]["label"]],
                "base_rho": r(base_ij, 4),
                "pack_rho": None if calc["rho"] is None else r(calc["rho"], 4),
                "pack_rho_reference": None if calc["pack_rho_reference"] is None else r(calc["pack_rho_reference"], 4),
                "blended_rho": r(entry, 4),
                "weight_used": r(w if touched else 0.0, 4),
                "coverage": r(calc["coverage"], 4),
                "evidence": r(calc["evidence"], 4),
                "explained": r(calc["explained"], 4),
                "touched": touched,
                "same_game": bool(calc["same_game"]),
                "kind": calc["kind"],
                "note": calc["note"],
            })
    psd: bool | None = None
    min_eig: float | None = None
    if n >= 2 and all(matrix[i][j] > CONTRADICTION_BELOW for i in range(n) for j in range(n)):
        try:
            values = np.linalg.eigvalsh(np.array(matrix, dtype=float))
            min_eig = float(values.min())
            psd = bool(min_eig >= -1e-9)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            psd = None
    return {
        "matrix": matrix,
        "pairs": pairs,
        "weight": nominal,
        "pack_pairs": touched_count,
        # only teams these legs actually depend on count (a bundle for an unrelated team is not signal)
        "has_signal": any(
            tstates[team]["evidence"] > 0.0
            for team in {team for prep in preps for team, _ in prep["exposures"]}
            if team in tstates
        ),
        "psd": psd,
        "min_eigenvalue": None if min_eig is None else r(min_eig, 6),
        "warnings": warnings,
        "basis": MODEL_BASIS,
    }


# ---------------------------------------------------------------------------
# Parlay strength
# ---------------------------------------------------------------------------
def _leg_state(prep: Mapping[str, Any], tstates: Mapping[str, Any]) -> dict[str, Any]:
    drivers: list[dict[str, Any]] = []
    shift = 0.0
    var = 0.0
    for (team, regime), load in sorted(prep["exposures"].items()):
        state = _state_of(tstates, team, regime)
        if state is None:
            continue
        contribution = load * (state["p"] - state["reference"])
        shift += contribution
        var += load * load * state["variance"]
        drivers.append({
            "team": team, "regime": regime, "loading": load, "p": state["p"], "reference": state["reference"],
            "z": state["z"], "contribution": contribution, "state": state,
        })
    drivers.sort(key=lambda d: (-abs(d["contribution"]), d["team"], d["regime"]))
    informative = bool(drivers) and any(tstates[d["team"]]["evidence"] > 0.0 for d in drivers)
    total = var + (prep["tau2"] or 0.0)
    sd = math.sqrt(total) if total > 0.0 else None
    z = shift / sd if sd else 0.0
    tilt = math.tanh(z / TILT_SCALE) if informative else 0.0
    return {"drivers": drivers, "shift": shift, "sd": sd, "z": z, "tilt": tilt, "informative": informative}


def _estimate_lift(probs: Sequence[float | None], matrix: Sequence[Sequence[float]]) -> float | None:
    """Kirkwood product of the Gaussian-copula pair lifts, clipped to the Frechet bounds."""
    n = len(probs)
    if n < 2:
        return 1.0
    if any(p is None or not 0.0 < p < 1.0 for p in probs):
        return None
    ps = [float(p) for p in probs]
    prod = math.prod(ps)
    if prod <= 0.0:
        return None
    lift = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            rho = min(matrix[i][j], matrix[j][i])
            if rho <= CONTRADICTION_BELOW:
                return 0.0
            if abs(rho) < 1e-12:
                continue
            joint = _bvn_cdf(_phi_inv(ps[i]), _phi_inv(ps[j]), clamp(rho, -1.0, 1.0))
            lift *= joint / (ps[i] * ps[j])
    joint_all = lift * prod
    lower = max(0.0, sum(ps) - (n - 1))
    upper = min(ps)
    return min(max(joint_all, lower), upper) / prod


def _structural_contradiction(preps: Sequence[Mapping[str, Any]]) -> bool:
    """Contradictions detectable without the builder's matrix (kept deliberately small)."""
    for a_index in range(len(preps)):
        for b_index in range(a_index + 1, len(preps)):
            a, b = preps[a_index]["view"], preps[b_index]["view"]
            if not _same_game(a, b):
                continue
            if a["type"] == "moneyline" and b["type"] == "moneyline" and a["team"] and b["team"] and a["team"] != b["team"]:
                return True
            same_player = a["player"] and a["player"] == b["player"]
            if same_player and a["market"] and a["market"] == b["market"] and {a["side"], b["side"]} == {"over", "under"}:
                over, under = (a, b) if a["side"] == "over" else (b, a)
                if over["line"] is None or under["line"] is None or over["line"] >= under["line"]:
                    return True
    return False


def _team_direction(prep: Mapping[str, Any]) -> tuple[str, int] | None:
    family = prep["family"]
    if family in (None, "game_total") or prep["own_team"] is None:
        return None
    if family == "team_result":
        return prep["own_team"], 1
    sign = _side_sign(prep["view"])
    if sign is None:
        return None
    if family == "qb_interceptions":
        sign = -sign
    return prep["own_team"], sign


def _stack_groups(preps: Sequence[Mapping[str, Any]], rule_matrix: Sequence[Sequence[float]] | None) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()

    def add(kind: str, members: list[int], **extra: Any) -> None:
        key = tuple(sorted(members))
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            groups.append({"kind": kind, "legs": list(key), **extra})

    by_side: dict[tuple[str, str], list[int]] = {}
    for prep in preps:
        if prep["family"] in (None, "team_result") or prep["game"] is None:
            continue
        side = prep["view"]["side"]
        if side in ("over", "under"):
            by_side.setdefault((prep["game"], side), []).append(prep["index"])
    for (game, side), members in sorted(by_side.items()):
        add("same_game_same_side", members, game=game, side=side)

    if rule_matrix is not None:
        by_dir: dict[tuple[str, str, int], list[int]] = {}
        for prep in preps:
            direction = _team_direction(prep)
            if direction is None or prep["game"] is None:
                continue
            by_dir.setdefault((prep["game"], direction[0], direction[1]), []).append(prep["index"])
        for (game, team, sign), members in sorted(by_dir.items()):
            linked = any(rule_matrix[i][j] >= STACK_BASE_RHO for i in members for j in members if i < j)
            if linked:
                add("same_team_stack", members, game=game, team=team, direction=sign)
    return groups


def _neutral_strength(n: int, contradiction: bool, lift: float | None, reasons: list[str],
                      legs_out: list[dict[str, Any]]) -> dict[str, Any]:
    score = NEUTRAL_SCORE
    if contradiction:
        score = min(score, CONTRADICTION_SCORE_CAP)
        reasons.insert(0, "Contradiction: two legs cannot both win, so the ticket cannot hit.")
    return {
        "score": r(score, 2), "lift": None if lift is None else r(lift, 4), "alignment": 0.0, "fragility": 0.0,
        "flags": {"mega": False, "trap": bool(contradiction), "contradiction": bool(contradiction),
                  "boost_overs": False, "boost_unders": False},
        "reasons": reasons, "legs": legs_out, "stacks": [], "basis": MODEL_BASIS, "has_signal": False,
        "n_legs": n, "n_signal_legs": 0,
        "components": {"alignment_points": 0.0, "lift_points": 0.0, "fragility_points": 0.0, "trap_points": 0.0},
    }



def _signed_zero_safe(value: float, digits: int) -> str:
    """``+3`` / ``-3`` / ``0`` -- never ``-0`` or ``+0`` (a value that rounds to zero is just zero)."""
    rounded = round(float(value), digits)
    if rounded == 0:
        return f"{0:.{digits}f}"
    return f"{rounded:+.{digits}f}"

def parlay_strength(
    legs: Any,
    signals_by_team: Any,
    *,
    matrix: Any = None,
    joint_probability: float | None = None,
    independent_probability: float | None = None,
    base_matrix: Any = None,
) -> dict[str, Any]:
    """Score a parlay 0-100 on how well its legs ride the regimes this game is in (see module docstring).

    Returns ``{"score", "lift", "alignment", "fragility", "flags", "reasons", "legs", "basis",
    "has_signal", "stacks", "n_legs", "n_signal_legs", "components"}``.  ``flags`` has ``mega``,
    ``trap``, ``contradiction``, ``boost_overs`` and ``boost_unders``.

    ``matrix`` is the correlation matrix the page prices with (the builder's, or
    :func:`build_correlation_matrix`'s ``matrix``); without it the lift is estimated from the pack's own
    regime correlations only.  ``joint_probability`` / ``independent_probability`` (the builder's
    adjusted and naive probabilities) give the exact ``lift = joint / independent``.  ``base_matrix`` is
    the pure rule table when ``matrix`` is already blended (used for the "rule table credits a boost the
    regimes do not" trap clause).  ``legs`` with no usable signals score exactly 50.0 with no flag but
    ``contradiction``, whatever ``matrix`` says; nothing is invented.  Never raises; inputs unchanged.
    """
    leg_list = _as_list(legs)
    n = len(leg_list)
    preps = [_prep(leg, i) for i, leg in enumerate(leg_list)]
    tstates = _team_states(signals_by_team)
    reasons: list[str] = []

    given = _read_matrix(matrix, n)
    rule = _read_matrix(base_matrix, n) or given
    contradiction = _structural_contradiction(preps)
    if given is not None and any(
        given[i][j] <= CONTRADICTION_BELOW or given[j][i] <= CONTRADICTION_BELOW for i in range(n) for j in range(i + 1, n)
    ):
        contradiction = True

    infos = [_leg_state(p, tstates) for p in preps]
    informative = [i for i, info in enumerate(infos) if info["informative"]]

    legs_out: list[dict[str, Any]] = []
    for prep, info in zip(preps, infos):
        top = info["drivers"][0] if info["drivers"] else None
        legs_out.append({
            "index": prep["index"], "label": prep["label"], "team": prep["own_team"], "family": prep["family"],
            "side": prep["view"]["side"], "has_signal": bool(info["informative"]),
            "tailwind": r(info["tilt"], 4), "shift": r(info["shift"], 4), "z": r(info["z"], 4),
            "drivers": [] if top is None else [_describe_regime(d["team"], d["regime"], d["state"]) for d in info["drivers"][:2]],
        })

    # lift: exact when both probabilities are given, else estimated from the matrix (pack-only without one)
    lift: float | None = None
    lift_basis = "no matrix or probabilities available"
    joint, indep = to_float(joint_probability), to_float(independent_probability)
    if joint is not None and indep is not None and indep > 0.0:
        lift = max(joint, 0.0) / indep
        lift_basis = "joint probability / independent product, as supplied"
    elif n >= 2:
        est = given
        source = "the supplied correlation matrix"
        if est is None:
            est = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    calc = _pair_calc(preps[i], preps[j], tstates)
                    if calc["rho"] is not None and calc["evidence"] > 0.0:
                        est[i][j] = est[j][i] = calc["rho"]
            source = "the pack's own regime correlations (no matrix supplied)"
        if contradiction:
            lift, lift_basis = 0.0, "a contradiction makes the joint probability zero"
        else:
            lift = _estimate_lift([p["view"]["prob"] for p in preps], est)
            lift_basis = f"Kirkwood product of Gaussian-copula pair lifts from {source}, clipped to Frechet bounds"
    elif n == 1:
        lift, lift_basis = 1.0, "a single leg has no correlation lift"

    if len(informative) == 0:
        if n == 0:
            reasons.append("No legs were supplied.")
        else:
            reasons.append("No team signals were available for these legs, so the pack added no regime evidence "
                           "and the score is neutral.")
        out = _neutral_strength(n, contradiction, lift, reasons, legs_out)
        out["basis"] = MODEL_BASIS + " Lift: " + lift_basis + "."
        return out

    # alignment, fragility ---------------------------------------------------------------------
    tilts = [infos[i]["tilt"] for i in informative]
    alignment = sum(tilts) / len(tilts)

    fragility = 0.0
    frag_note = None
    for team in sorted({d["team"] for i in informative for d in infos[i]["drivers"]}):
        for group, regimes in COLLAPSE_GROUPS.items():
            hurt = 0
            for i in informative:
                if any(d["team"] == team and d["regime"] in regimes and d["loading"] < 0 for d in infos[i]["drivers"]):
                    hurt += 1
            if hurt == 0:
                continue
            excess = 0.0
            worst = None
            for regime in regimes:
                state = _state_of(tstates, team, regime)
                if state is not None and state["p"] - state["reference"] > excess:
                    excess = state["p"] - state["reference"]
                    worst = (regime, state)
            severity = clamp(excess / COLLAPSE_SEVERITY_FULL, 0.0, 1.0)
            value = (hurt / len(informative)) * severity
            if value > fragility and worst is not None:
                fragility = value
                frag_note = (hurt, len(informative), team, group, _describe_regime(team, worst[0], worst[1]))

    # stacks, trap, boosts -----------------------------------------------------------------------
    trap = False
    stacks_out: list[dict[str, Any]] = []
    trap_reasons: list[str] = []
    for group in _stack_groups(preps, rule):
        members = [i for i in group["legs"] if infos[i]["informative"]]
        if len(members) < 2:
            continue
        mean_tilt = sum(infos[i]["tilt"] for i in members) / len(members)
        totals: dict[tuple[str, str], float] = {}
        for i in members:
            for d in infos[i]["drivers"]:
                totals[(d["team"], d["regime"])] = totals.get((d["team"], d["regime"]), 0.0) + d["contribution"]
        worst_key = min(totals, key=lambda k: (totals[k], k)) if totals else None
        adverse = bool(mean_tilt <= TRAP_ALIGNMENT)
        stacks_out.append({"kind": group["kind"], "legs": members, "alignment": r(mean_tilt, 4), "adverse": adverse})
        if adverse:
            trap = True
            names = ", ".join(preps[i]["label"] for i in members)
            driver = ""
            if worst_key is not None:
                st = _state_of(tstates, *worst_key)
                driver = f" The shared driver is adverse: {_describe_regime(worst_key[0], worst_key[1], st)}."
            trap_reasons.append(
                f"Trap: {names} look like a stack, but this game's regimes work against them "
                f"(alignment {mean_tilt:+.2f}, trap threshold {TRAP_ALIGNMENT:+.2f}).{driver}"
            )

    if rule is not None:  # the rule table credits a boost the regimes do not
        for a_i in informative:
            for b_i in informative:
                if b_i <= a_i or not _same_game(preps[a_i]["view"], preps[b_i]["view"]):
                    continue
                if rule[a_i][b_i] < STACK_BASE_RHO:
                    continue
                calc = _pair_calc(preps[a_i], preps[b_i], tstates)
                if calc["rho"] is not None and calc["evidence"] > 0.0 and calc["rho"] <= PACK_NEAR_ZERO_RHO:
                    trap = True
                    trap_reasons.append(
                        f"Trap: the rule table credits {preps[a_i]['label']} + {preps[b_i]['label']} with "
                        f"rho {rule[a_i][b_i]:+.2f}, but the regimes behind them imply only {calc['rho']:+.2f}, "
                        "so the claimed boost is not supported this game."
                    )
    if contradiction:
        trap = True

    boost = {"over": False, "under": False}
    boost_reasons: list[str] = []
    by_side: dict[tuple[str, str], list[int]] = {}
    for i in informative:
        prep = preps[i]
        if prep["family"] in (None, "team_result") or prep["game"] is None:
            continue
        if prep["view"]["side"] in ("over", "under"):
            by_side.setdefault((prep["game"], prep["view"]["side"]), []).append(i)
    for (game, side), members in sorted(by_side.items()):
        if len(members) < 2 or boost[side]:
            continue
        # one entry per leg per exposure key; the QB and pass-game collapse regimes are one passing-collapse
        # exposure here (the pass-game model consumes the QB's collapse probability), as in fragility
        candidates: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
        for i in members:
            for d in infos[i]["drivers"]:
                if abs(d["loading"]) < BOOST_MIN_LOADING:
                    continue
                key = _exposure_key(d["team"], d["regime"])
                held = candidates.setdefault(key, {}).get(i)
                if held is None or abs(d["loading"]) > abs(held["loading"]):
                    candidates[key][i] = d
        best = None
        for key, by_leg in sorted(candidates.items()):
            rows = [by_leg[i] for i in sorted(by_leg)]
            if len(rows) < 2:
                continue
            signs = {1 if d["loading"] > 0 else -1 for d in rows}
            if len(signs) != 1:
                continue
            sign = signs.pop()
            if all(sign * d["z"] >= Z_FAVOURABLE for d in rows):
                strength = sum(abs(d["loading"]) * abs(d["z"]) for d in rows)
                if best is None or strength > best[0]:
                    best = (strength, key, rows)
        if best is not None:
            boost[side] = True
            described = "; ".join(_describe_regime(d["team"], d["regime"], d["state"]) for d in best[2])
            boost_reasons.append(
                f"Boost {side}s: {len(best[2])} {side} legs in this game load on the same driver and it is "
                f"favourable to {side}s ({described})."
            )

    # score and the remaining flags --------------------------------------------------------------
    lift_term = 0.0
    if lift is not None:
        lift_term = clamp(math.log(max(lift, 1e-9)) / math.log(LIFT_FULL), -1.0, 1.0)
    alignment_points = SCORE_ALIGNMENT_POINTS * alignment
    lift_points = SCORE_LIFT_POINTS * lift_term
    fragility_points = -SCORE_FRAGILITY_POINTS * fragility
    trap_points = -SCORE_TRAP_POINTS if trap else 0.0
    score = NEUTRAL_SCORE + alignment_points + lift_points + fragility_points + trap_points
    if contradiction:
        score = min(score, CONTRADICTION_SCORE_CAP)
    score = clamp(score, 0.0, 100.0)

    mega = bool(
        not trap
        and not contradiction
        and len(informative) >= 2
        and lift is not None and lift >= MEGA_LIFT
        and alignment >= MEGA_ALIGNMENT
        and fragility <= MEGA_FRAGILITY_MAX
        and all(infos[i]["tilt"] > STRONG_ADVERSE_TILT for i in informative)
        and score >= MEGA_SCORE
    )

    # reasons -----------------------------------------------------------------------------------
    if contradiction:
        reasons.append("Contradiction: two legs cannot both win, so the ticket cannot hit.")
    reasons.extend(trap_reasons)
    if mega:
        reasons.append(
            f"Mega: {len(informative)} legs ride regimes that favour them (alignment {alignment:+.2f}), the "
            f"combined lift is {lift:.2f}x the independent product, and collapse exposure is low "
            f"(fragility {fragility:.2f})."
        )
    reasons.extend(boost_reasons)
    for i in informative:
        info, prep = infos[i], preps[i]
        if info["drivers"]:
            d = info["drivers"][0]
            reasons.append(
                f"{prep['label']}: {_describe_regime(d['team'], d['regime'], d['state'])}; the leg's expected "
                f"stat moves about {_signed_zero_safe(info['shift'] * 100, 0)}% against a league-average game (tailwind {_signed_zero_safe(info['tilt'], 2)})."
            )
    if frag_note is not None and fragility > 0.0:
        hurt, total_legs, team, group, text = frag_note
        reasons.append(
            f"Fragility {fragility:.2f}: {hurt} of {total_legs} legs lose if {team}'s {group} game collapses "
            f"({text})."
        )
    if lift is not None:
        reasons.append(f"Lift {lift:.2f}x the independent product ({lift_basis}).")
    skipped = n - len(informative)
    if skipped:
        reasons.append(f"{skipped} leg(s) had no usable signals and are scored neutral.")

    flags = {
        "mega": mega,
        "trap": bool(trap),
        "contradiction": bool(contradiction),
        "boost_overs": bool(boost["over"]),
        "boost_unders": bool(boost["under"]),
    }
    return {
        "score": r(score, 2),
        "lift": None if lift is None else r(lift, 4),
        "alignment": r(alignment, 4),
        "fragility": r(fragility, 4),
        "flags": flags,
        "reasons": reasons,
        "legs": legs_out,
        "stacks": stacks_out,
        "basis": MODEL_BASIS + " Lift: " + lift_basis + ".",
        "has_signal": True,
        "n_legs": n,
        "n_signal_legs": len(informative),
        "components": {
            "alignment_points": r(alignment_points, 2),
            "lift_points": r(lift_points, 2),
            "fragility_points": r(fragility_points, 2),
            "trap_points": r(trap_points, 2),
        },
    }


def flag_parlay(legs: Any, signals_by_team: Any, **kwargs: Any) -> dict[str, Any]:
    """Just the flags and reasons of :func:`parlay_strength` (same keyword arguments)."""
    strength = parlay_strength(legs, signals_by_team, **kwargs)
    return {"flags": strength["flags"], "reasons": strength["reasons"], "has_signal": strength["has_signal"],
            "score": strength["score"]}


def rank_parlays(
    candidates: Any,
    signals_by_team: Any,
    *,
    evaluator: Any = None,
) -> list[dict[str, Any]]:
    """Rank candidate parlays (each a list of legs) best first.

    ``evaluator(legs)`` may return ``{"adjusted_probability", "naive_probability", "ev_adjusted"}``; the
    two probabilities give the exact lift and ``ev_adjusted`` breaks score ties.  Order: score
    descending, then ``ev_adjusted`` descending (missing last), then adjusted probability descending,
    then the original order (stable).  An evaluator that raises is ignored for that candidate.
    """
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(_as_list(candidates)):
        legs = _as_list(candidate)
        evaluated: dict[str, Any] = {}
        error = None
        if callable(evaluator):
            try:
                result = evaluator(legs)
                evaluated = dict(result) if isinstance(result, Mapping) else {}
            except Exception as exc:  # an evaluator bug must not break the ranking
                error = f"{type(exc).__name__}: {exc}"
        adjusted = to_float(evaluated.get("adjusted_probability"))
        naive = to_float(evaluated.get("naive_probability"))
        ev = to_float(evaluated.get("ev_adjusted"))
        strength = parlay_strength(legs, signals_by_team, joint_probability=adjusted, independent_probability=naive)
        rows.append({
            "index": index, "legs": legs, "score": strength["score"], "flags": strength["flags"],
            "reasons": strength["reasons"], "lift": strength["lift"], "alignment": strength["alignment"],
            "fragility": strength["fragility"], "has_signal": strength["has_signal"],
            "adjusted_probability": adjusted, "naive_probability": naive, "ev_adjusted": ev,
            "evaluator_error": error, "strength": strength,
        })

    def key(row: Mapping[str, Any]) -> tuple[float, float, float, int]:
        ev = row["ev_adjusted"]
        prob = row["adjusted_probability"]
        return (-row["score"], -(ev if ev is not None else -math.inf),
                -(prob if prob is not None else -math.inf), row["index"])

    rows.sort(key=key)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def disclosure() -> dict[str, Any]:
    """Everything a page needs to disclose how the engine works (priors, variances, thresholds)."""
    return {
        "basis": MODEL_BASIS,
        "regimes": {regime: REGIME_LABELS[regime] for regime in REGIMES},
        "loadings": {family: dict(rows) for family, rows in LOADINGS.items()},
        "idiosyncratic": {family: dict(meta) for family, meta in IDIOSYNCRATIC.items()},
        "thresholds": {
            "default_pack_weight": DEFAULT_PACK_WEIGHT, "z_favourable": Z_FAVOURABLE, "trap_alignment": TRAP_ALIGNMENT,
            "mega_alignment": MEGA_ALIGNMENT, "mega_lift": MEGA_LIFT, "mega_fragility_max": MEGA_FRAGILITY_MAX,
            "mega_score": MEGA_SCORE, "strong_adverse_tilt": STRONG_ADVERSE_TILT, "stack_base_rho": STACK_BASE_RHO,
            "pack_near_zero_rho": PACK_NEAR_ZERO_RHO,
        },
        "limits": [
            "Loadings are the brief's coefficients used as priors; nothing is fitted to joint outcomes.",
            "Team-result loadings are this module's own prior (the brief gives none).",
            "Regimes of different offences are independent; the rule table keeps opposing-offence links.",
            "The Gaussian copula and the Kirkwood lift are approximations; sportsbooks reprice same-game parlays.",
        ],
    }


__all__ = [
    "BLEND_CLIP",
    "BOOST_MIN_LOADING",
    "COLLAPSE_GROUPS",
    "COLLAPSE_SEVERITY_FULL",
    "CONTRADICTION_BELOW",
    "CONTRADICTORY_RHO",
    "DEFAULT_PACK_WEIGHT",
    "FAMILY_CV",
    "IDIOSYNCRATIC",
    "LOADINGS",
    "MEGA_ALIGNMENT",
    "MEGA_FRAGILITY_MAX",
    "MEGA_LIFT",
    "MEGA_SCORE",
    "MODEL_BASIS",
    "NEUTRAL_SCORE",
    "PACK_NEAR_ZERO_RHO",
    "PACK_RHO_CAP",
    "REGIMES",
    "REGIME_LABELS",
    "SIGNED_FACTOR_SD",
    "STACK_BASE_RHO",
    "STRONG_ADVERSE_TILT",
    "STRUCTURAL_ABS_RHO",
    "TAU_MIN_SHARE",
    "TILT_SCALE",
    "TRAP_ALIGNMENT",
    "WEIGHT_BY_EXPLAINED_SHARE",
    "Z_EVIDENCE_FLOOR",
    "Z_EVIDENCE_FULL",
    "Z_FAVOURABLE",
    "build_correlation_matrix",
    "disclosure",
    "flag_parlay",
    "leg_exposures",
    "leg_regime_loadings",
    "market_family",
    "pair_correlation",
    "parlay_strength",
    "rank_parlays",
    "regime_state",
]
