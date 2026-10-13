"""Coverage mismatch factor for the NFL model upgrade pack.

``compute_coverage_mismatch`` answers one question for one receiver: *does the
coverage this defense is likely to play hand him a soft matchup or a hard one?*
It returns ``coverage_mismatch_factor`` in ``[-1, +1]`` where **positive is
favourable to the offensive player** and ``0`` is neutral.
``apply_coverage_mismatch`` then applies the brief's formulas::

    wr_yards *= (1 + coverage_mismatch_factor * 0.25)
    te_yards *= (1 + coverage_mismatch_factor * 0.20)

What the number means (there is no "event")
-------------------------------------------
Unlike the collapse / explosion models this is a *signed factor*, not a
probability, so there is no event definition and no base rate to calibrate:
``BASE_RATE`` is the neutral reference ``0.0``. ``+1`` / ``-1`` mean "the
extreme of the evidence supplied", never a certainty: the tanh squash only
approaches the bound (every factor at the 3-SD clip gives ``+-0.947``). The
coefficients cap the yardage effect at ``+-25%`` (WR) and ``+-20%`` (TE) no
matter what the factor says, and in practice a two-SD-better-than-average
matchup on every dimension (``+0.83``) is worth ``+20.8%`` / ``+16.7%`` (a
bounded 0-100 grade cannot reach +2 SD by itself; see "GAIN" for what valid
inputs actually reach).
``recenter`` on the apply function is accepted for interface uniformity and has
**no effect**: the factor is signed with neutral ``0``, so the re-centring
divisor ``1 + 0 * c`` is exactly ``1``.

Inputs (canonical keys; any subset, ``None`` / ``{}`` is neutral)
------------------------------------------------------------------
============================= ================ =====================================================
key                           unit             free public feed?
============================= ================ =====================================================
``wr_vs_cb_matchup_grade``    grade 0-100, >50 **NO free feed.** Neutral until supplied.
                              favours the WR
``opp_cb1_shadow_rate``       rate             **NO free feed** (nflverse has no CB-to-WR assignment).
``cb_man_coverage_grade``     grade 0-100,     **NO free feed.** PFR ``def`` has per-defender targets,
``cb_zone_coverage_grade``    higher = better  completions, ``rat`` and ``dadot`` (an *overall*
                              CB               coverage quality) but no man/zone split.
``wr_yprr_vs_man``            yards / route    Approximate: nflverse participation + pbp give pass
``wr_yprr_vs_zone``                            plays on the field, which this model's references use
                                               as routes. True routes run are PFF-only.
``opp_man_coverage_rate``     rate             Real: participation ``defense_man_zone_type`` share
                                               of pass plays (2018-2024; see the regime note below).
``te_vs_lb_coverage_grade``   grade 0-100, >50 **NO free feed.** Neutral until supplied.
                              favours the TE
``slot_wr_vs_nickel_grade``   grade 0-100, >50 **NO free feed** (nflverse carries no receiver
                              favours slot WR  alignment).
============================= ================ =====================================================

No key was added to the contract's vocabulary. Grades are percentile ranks on
0-100 and are *not* rescaled: a value outside ``[0, 100]`` is a data error and
is treated as missing (``"65%"`` is read as 65, ``0.65`` stays 0.65). Rates
accept a fraction or a percent (``0.35`` / ``35``). ``wr_yprr_*`` outside
``[0, 10]`` is treated as missing. A missing ``opp_man_coverage_rate`` defaults
to :data:`MAN_RATE_LEAGUE` (0.448) -- *not* the helper's built-in 0.30, which is
the pre-2023 label regime: the free feed's man/zone label changed in 2023 (share
of pass plays labelled man: 35% / 37% / 34% / 30% / 29% in 2018-2022, then 41% /
48% in 2023 / 2024), so a team rate computed from a 2022 or earlier season is
not on the same scale as a 2023+ one. As a *blend weight* the model is
forgiving of that, but a loader should compute the rate from the latest season.

Roles and the design
--------------------
``role`` is one of ``"wr"``, ``"slot"``, ``"te"`` (anything else raises
``ValueError``: it is a programming error, never a data error). It decides
which matchup grade is the primary evidence. Five factors feed one weighted,
direction-signed z-score (``nfl_pack_common.score_factors``); missing factors
count as 0 *and* stay in the denominator, so partial data shrinks toward 0.
The score then goes through ``signed_unit`` (tanh), so the output is smooth,
bounded and exactly 0 at neutral::

    coverage_mismatch_factor = tanh(GAIN * z)

========================================== ======= ======= ======= ==========================
factor (what is scored)                    wr      slot    te      derived from
========================================== ======= ======= ======= ==========================
``wr_vs_cb_matchup_grade`` (shadow-wtd)    **3.0** 0.5     0.5     grade, shadow rate
``slot_wr_vs_nickel_grade``                0.5     **3.0** 0.5     grade
``te_vs_lb_coverage_grade``                0.5     0.5     **3.0** grade
``cb_coverage_grade_at_opp_mix`` (dir -1)  1.5     1.5     --      CB man/zone x man rate
``wr_yprr_at_opp_mix``                     1.0     1.0     --      WR man/zone yprr x man rate
========================================== ======= ======= ======= ==========================

* **Shadow weighting.** ``opp_cb1_shadow_rate`` *amplifies* how much the
  WR1-vs-CB1 grade matters: ``effective grade = 50 + (grade - 50) * (0.5 +
  shadow_rate)``. Shadow 0 halves the deviation, 0.5 takes it at face value,
  1.0 multiplies it by 1.5. A missing shadow rate means the 0.5 pivot
  (:data:`DEFAULT_SHADOW_RATE`), i.e. the supplied grade stands unamplified.
* **CB at the defense's coverage mix.** ``blend_man_zone(cb_man, cb_zone,
  opp_man_rate)`` -- the CB's grade weighted by how often this defense plays
  man. Direction ``-1``: a better CB is worse for the receiver.
* **WR yprr at the defense's coverage mix.** Each side is standardised against
  league WRs (``(yprr_man - 1.668) / 0.829`` and ``(yprr_zone - 1.609) /
  0.660``) and the two z-scores are blended with ``blend_man_zone(z_man,
  z_zone, opp_man_rate)``. A league-average WR is therefore exactly neutral
  against *any* man rate, while "a WR who beats man, facing a man-heavy
  defense" scores high because the man side carries the weight; together with
  the CB factor this is "...with a weak man CB". Both yprr inputs only ever
  raise the factor. A lone split is used as given (the helper's behaviour).
  This is a *level*, not an incremental split effect: see the honest limits.
* **Why the weights are ordered this way (priors, not fits).** The direct,
  individual matchup grade is the closest thing to the causal question (3.0).
  The opposing CB's level at the coverage mix is next (1.5): new information
  about the opponent, but it partly overlaps the primary grade when both are
  supplied. The WR's man/zone level is third (1.0): it is the noisiest input (a
  split rests on ~150 routes and one 70-yard score moves it by 0.5 yprr) and a
  receiver's overall level is already inside the engine's baseline. Off-role
  grades get 0.5: a defense's soft spots are correlated by scheme (a weak-
  evidence assumption, see the check below), so a "wrong-role" grade is muted,
  not ignored. For the TE the man/zone terms are omitted: the brief supplies WR
  and CB splits only, and standardising a TE against WR references would
  mis-score him. A key supplied but unused in the role is listed in
  ``inputs_ignored``.
* **Coverage differs by role.** The WR and slot roles score five factors
  (total weight 6.5), the TE role three (total 4.0). A lone primary grade
  therefore carries 46% of the weight for a WR / slot (``coverage`` 0.46) but
  75% for a TE (``coverage`` 0.75), and a TE-vs-LB grade of 80 alone gives
  ``+0.44`` where a WR-vs-CB grade of 80 alone gives ``+0.28``. That is a
  consequence of the brief giving the TE exactly one dedicated input, not a
  tuned choice; the smaller 0.20 coefficient is the only other damping.

League references (derived from real data, see the constants)
-------------------------------------------------------------
* ``MAN_RATE_LEAGUE = 0.448``: nflverse participation + pbp, regular season
  2023-2024, pass plays carrying a man/zone label, **n = 38,781 plays**
  (team-season sd 0.094, n = 64 team-seasons, median ~609 plays each).
* ``YPRR_MAN_MEAN/SD = 1.668 / 0.829`` and ``YPRR_ZONE_MEAN/SD = 1.609 / 0.660``:
  the same feeds, 2023-2024 regular season, WR-seasons with >= 100 man and
  >= 100 zone routes (**n = 204**); correlation of the two sides 0.57. WR yprr
  is receiving yards / pass plays with the WR on the field (``routes`` is
  approximated), so these are starter-level WRs, not all WRs.
* ``WR_MIX_Z_SD = 0.89``: sd of the blended per-side z-score on those 204
  WR-seasons at the league mix (0.89 at man rates 0.30, 0.448 and 0.60, so it
  does not depend on the mix).
* Grades: ``prior`` -- a uniform team percentile (mean 50, sd 28.9). No feed.

GAIN
----
``GAIN = 0.6`` is the slope at 0 of ``tanh(GAIN * z)`` where ``z`` is the
weighted mean z-score. Every factor +1 SD favourable gives ``tanh(0.6) =
+0.537`` (``wr_yards`` +13.4%, ``te_yards`` +10.7%); every factor +2 SD gives
``+0.834`` (+20.8% / +16.7%); the 3-SD clip gives ``+0.947``. The extreme stays
well below 1 on purpose: one matchup grade does not make a receiver's day
certain. It is a prior chosen so a very good matchup is worth a low-teens
percent of yards (shadow-coverage and defense-versus-position studies put the
extremes in the 10-20% range), not a fitted value.

A 0-100 percentile grade tops out at ``(100 - 50) / 28.9 = +1.73`` SD, so the
+2 SD row is reachable only through the shadow multiplier (a grade of 100 with
a 0.66 shadow rate is an effective +2.0 SD) and the yprr terms; a literal grade
above 100 is rejected as a data error, not clipped. The most favourable
*valid* input set (every grade at its best, shadow rate 1.0, yprr far above
league) gives ``+0.88`` for a WR, ``+0.83`` slot, ``+0.80`` TE and its mirror
``-0.87`` / ``-0.81`` / ``-0.80``; every grade at its best with no shadow rate
and no yprr gives ``+0.71`` (WR / slot) and ``+0.78`` (TE).

Honest limits
-------------
* Weights and the 0.6 gain are **priors**. There is no free offline dataset of
  WR-vs-CB assignments to fit them against.
* Six of the nine inputs (the five grades and the shadow rate) have no free
  feed, so with free data alone most calls return a small number driven by the
  WR's own man/zone yprr. That
  is the honest output for "almost no matchup evidence". Team-level coverage
  *quality* (``opp_coverage_grade``) is deliberately not scored here: the
  collapse stages already use it, and scoring it again would double count the
  same defense.
* The WR man/zone term is a **level**, and a receiver's level is already inside
  the projection engine's per-game baseline. A signed factor cannot be
  re-centred away, so the term leans a good receiver up and a poor one down by
  a couple of percent (a WR at 2.2 yprr on both sides is +2.0% wr_yards, one at
  1.2 yprr is -1.5%); that is why it carries a low weight.
* The man/zone piece was sanity-checked on real data. As a level it is real: a
  WR's blended man/zone yprr in season t-1 against his overall yprr in season t
  has Spearman 0.53 (n = 403 WR pairs, >= 60 routes each side, one label regime
  per pair). The *incremental* mismatch -- how far a WR's man-minus-zone gap
  times the opponent's man-rate deviation moves him relative to his own
  average -- is weak. Year-over-year (same WR, consecutive seasons) the gap has
  Spearman 0.14 (n = 388 pairs, >= 60 routes each side) to 0.19 (n = 240,
  >= 100), against 0.41-0.46 for man-only or zone-only yprr: most of the gap is
  sampling noise. At game level (n = 5,238 WR-games with >= 15 routes, 2019-22
  and 2024) the *ex-ante* edge (prior-season gap x opponent prior-season
  man-rate deviation) has Spearman +0.005 against the game's yards-per-route
  residual -- no detectable association; the *contemporaneous* version (prior
  gap x the game's actual man-rate deviation) is +0.026 (p ~ 0.06) and its top
  tercile beat its bottom tercile by +0.06 yds/route (~4%): right sign, tiny.
  So read the WR term as "who is the better receiver against this mix", not as
  a proven scheme-specific edge. The weight stays at its prior; it was not
  tuned to either number. Team man rate itself is persistent (pooled Spearman
  0.59, n = 160 team-season pairs, within-regime).
* The off-role weight rests on a team-level fact only: season-demeaned EPA per
  target allowed to WRs vs to TEs has Spearman +0.22 across 224 defense-seasons
  (p ~ 0.001), but ranges from -0.25 to +0.54 in individual seasons (n = 32
  each). It supports "weakly correlated", not the grades themselves.
* ``+-1`` is a bounded score, not a calibrated probability. ``probability`` in
  the result is only an alias of the factor (a signed number in ``[-1, 1]``)
  kept so generic audit code finds a primary number; do not treat it as one.
* The result does not know sample size. Callers should supply man/zone splits
  only when they rest on a sensible number of routes (the references above use
  >= 100 on each side).
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Mapping

from modules import nfl_pack_common as common
from modules.nfl_pack_common import Factor

MODEL_NAME = "coverage_mismatch"
OUTPUT_KEY = "coverage_mismatch_factor"
STAGE = "coverage_mismatch"
LOG = logging.getLogger("nfl_pack.coverage_mismatch")

ROLES: tuple[str, ...] = ("wr", "slot", "te")

#: Neutral reference of a signed factor (NOT a probability). Kept as a public
#: constant so the apply function and the result's ``references`` agree.
BASE_RATE = 0.0
#: Slope at zero of ``tanh(GAIN * z)`` -- see the module docstring ("GAIN").
GAIN = 0.6

#: The brief's coefficients, named after the field they scale.
WR_YARDS_COEFFICIENT = 0.25
TE_YARDS_COEFFICIENT = 0.20

#: A team percentile grade: 50 is average, sd of a uniform 0-100 rank is 28.9.
NEUTRAL_GRADE = 50.0
GRADE_SD = 28.9
#: Shadow rate at which the supplied WR-vs-CB grade is taken at face value
#: ((0.5 + shadow) == 1.0); used when ``opp_cb1_shadow_rate`` is missing.
DEFAULT_SHADOW_RATE = 0.5
#: Regular-season pass plays with a man/zone label, nflverse participation + pbp,
#: 2023-2024, n = 38,781 plays; team-season sd 0.094 (n = 64 team-seasons).
#: NOT comparable with 2018-2022 (0.29-0.37): the label regime changed in 2023.
MAN_RATE_LEAGUE = 0.448
MAN_RATE_TEAM_SD = 0.094
#: WR yards per route (receiving yards / pass plays on the field), nflverse
#: participation + pbp, 2023-2024 REG, WR-seasons with >= 100 man AND >= 100
#: zone routes, n = 204: man mean 1.668 sd 0.829, zone mean 1.609 sd 0.660.
YPRR_MAN_MEAN = 1.668
YPRR_MAN_SD = 0.829
YPRR_ZONE_MEAN = 1.609
YPRR_ZONE_SD = 0.660
#: sd of the blended per-side z-score at the league man rate (same n = 204; 0.888
#: at 0.448, 0.905 at 0.30, 0.891 at 0.60).
WR_MIX_Z_SD = 0.89
#: A yards-per-route above this is a data error (a 1-route 80-yd TD), not evidence.
YPRR_MAX = 10.0

#: The nine inputs named by the brief / contract section 4 (report order).
CANONICAL_INPUTS: tuple[str, ...] = (
    "wr_vs_cb_matchup_grade",
    "opp_cb1_shadow_rate",
    "cb_man_coverage_grade",
    "cb_zone_coverage_grade",
    "wr_yprr_vs_man",
    "wr_yprr_vs_zone",
    "opp_man_coverage_rate",
    "te_vs_lb_coverage_grade",
    "slot_wr_vs_nickel_grade",
)

#: Free-feed status per input, for the feature loader and for the docs.
INPUT_FEEDS: dict[str, str] = {
    "wr_vs_cb_matchup_grade": "no free feed",
    "opp_cb1_shadow_rate": "no free feed",
    "cb_man_coverage_grade": "no free feed (PFR def has overall coverage only, no man/zone split)",
    "cb_zone_coverage_grade": "no free feed (PFR def has overall coverage only, no man/zone split)",
    "wr_yprr_vs_man": "proxy: nflverse participation + pbp, pass plays on field as routes",
    "wr_yprr_vs_zone": "proxy: nflverse participation + pbp, pass plays on field as routes",
    "opp_man_coverage_rate": "real: participation defense_man_zone_type share of pass plays",
    "te_vs_lb_coverage_grade": "no free feed",
    "slot_wr_vs_nickel_grade": "no free feed (no receiver alignment in nflverse)",
}

_GRADE_SOURCE = "prior: uniform team percentile rank 0-100 (mean 50, sd 28.9); no free feed"

# Weights are PRIORS, ordered by causal importance (module docstring, "Why the
# weights are ordered this way"): direct matchup grade > opposing CB at the
# coverage mix > WR man/zone level (noisiest, already in the baseline) > off-role grades.
FACTORS: tuple[Factor, ...] = (
    Factor(
        "wr_vs_cb_matchup_grade", mean=NEUTRAL_GRADE, sd=GRADE_SD, weight=3.0, direction=+1,
        label="WR vs CB matchup grade (shadow-weighted)", source=_GRADE_SOURCE,
    ),
    Factor(
        "cb_coverage_grade_at_opp_mix", mean=NEUTRAL_GRADE, sd=GRADE_SD, weight=1.5, direction=-1,
        label="CB coverage grade at the defense's man/zone mix", source=_GRADE_SOURCE,
    ),
    Factor(
        "wr_yprr_at_opp_mix", mean=0.0, sd=WR_MIX_Z_SD, weight=1.0, direction=+1,
        label="WR yprr at the defense's man/zone mix (vs league WRs)", unit="z",
        source=(
            "nflverse participation+pbp 2023-2024 REG: per-side z vs league WR-seasons >= 100 routes each side "
            "(n=204), blended at the opponent's man rate; sd of the blend 0.89"
        ),
    ),
    Factor(
        "slot_wr_vs_nickel_grade", mean=NEUTRAL_GRADE, sd=GRADE_SD, weight=0.5, direction=+1,
        label="Slot WR vs nickel CB grade", source=_GRADE_SOURCE,
    ),
    Factor(
        "te_vs_lb_coverage_grade", mean=NEUTRAL_GRADE, sd=GRADE_SD, weight=0.5, direction=+1,
        label="TE vs LB coverage grade", source=_GRADE_SOURCE,
    ),
)

# Per-role weights (the primary matchup grade carries 3.0). A factor absent from a
# role is not scored for it at all.
_ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "wr": {
        "wr_vs_cb_matchup_grade": 3.0, "cb_coverage_grade_at_opp_mix": 1.5, "wr_yprr_at_opp_mix": 1.0,
        "slot_wr_vs_nickel_grade": 0.5, "te_vs_lb_coverage_grade": 0.5,
    },
    "slot": {
        "slot_wr_vs_nickel_grade": 3.0, "cb_coverage_grade_at_opp_mix": 1.5, "wr_yprr_at_opp_mix": 1.0,
        "wr_vs_cb_matchup_grade": 0.5, "te_vs_lb_coverage_grade": 0.5,
    },
    "te": {
        "te_vs_lb_coverage_grade": 3.0, "wr_vs_cb_matchup_grade": 0.5, "slot_wr_vs_nickel_grade": 0.5,
    },
}


def _build_role_factors() -> dict[str, tuple[Factor, ...]]:
    built: dict[str, tuple[Factor, ...]] = {}
    for role, weights in _ROLE_WEIGHTS.items():
        chosen = [replace(f, weight=weights[f.key]) for f in FACTORS if f.key in weights]
        built[role] = tuple(sorted(chosen, key=lambda f: -f.weight))
    built["wr"] = FACTORS  # the module-level FACTORS *is* the WR-role factor set
    return built


#: ``role -> factors`` actually scored for that role (``FACTORS`` is the ``"wr"`` entry).
ROLE_FACTORS: dict[str, tuple[Factor, ...]] = _build_role_factors()

#: Canonical input keys that feed each factor.
_FACTOR_SOURCES: dict[str, tuple[str, ...]] = {
    "wr_vs_cb_matchup_grade": ("wr_vs_cb_matchup_grade", "opp_cb1_shadow_rate"),
    "slot_wr_vs_nickel_grade": ("slot_wr_vs_nickel_grade",),
    "te_vs_lb_coverage_grade": ("te_vs_lb_coverage_grade",),
    "cb_coverage_grade_at_opp_mix": ("cb_man_coverage_grade", "cb_zone_coverage_grade", "opp_man_coverage_rate"),
    "wr_yprr_at_opp_mix": ("wr_yprr_vs_man", "wr_yprr_vs_zone", "opp_man_coverage_rate"),
}

_BASIS = (
    "Coverage-mismatch factor tanh(GAIN*z) over up to five matchup factors; weights and gain are PRIORS, not fitted. "
    "References: man rate (n=38,781 pass plays) and WR man/zone yprr (n=204 WR-seasons) measured on nflverse "
    "participation+pbp 2023-2024; grades are a uniform-percentile prior. Six of nine inputs (five grades and the CB shadow rate) have no "
    "free feed and are neutral until supplied; WR yprr uses pass plays on field as routes (approximation) and is a "
    "level already partly in the baseline. Signed factor, not a probability: +-1 is the extreme of the evidence, "
    "capped by the 0.25 / 0.20 coefficients."
)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------
def _grade(value: Any) -> float | None:
    """A 0-100 percentile grade, or ``None``. Not rescaled: ``"65%"`` is 65, ``0.65`` is 0.65."""
    if isinstance(value, str) and value.strip().endswith("%"):
        number = common.to_float(value.strip()[:-1])
    else:
        number = common.to_float(value)
    if number is None or number < 0.0 or number > 100.0:
        return None
    return number


def _yprr(value: Any) -> float | None:
    """Yards per route run in ``[0, YPRR_MAX]``, or ``None``."""
    number = common.to_float(value)
    if number is None or number < 0.0 or number > YPRR_MAX:
        return None
    return number


_PARSERS = {
    "wr_vs_cb_matchup_grade": _grade,
    "opp_cb1_shadow_rate": common.to_rate,
    "cb_man_coverage_grade": _grade,
    "cb_zone_coverage_grade": _grade,
    "wr_yprr_vs_man": _yprr,
    "wr_yprr_vs_zone": _yprr,
    "opp_man_coverage_rate": common.to_rate,
    "te_vs_lb_coverage_grade": _grade,
    "slot_wr_vs_nickel_grade": _grade,
}


def _read_inputs(inputs: Mapping[str, Any]) -> dict[str, float | None]:
    return {key: parser(inputs.get(key)) for key, parser in _PARSERS.items()}


def _normalise_role(role: Any) -> str:
    key = role.strip().lower() if isinstance(role, str) else role
    if key not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    return key


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute_coverage_mismatch(inputs: Mapping[str, Any] | None = None, *, role: str = "wr") -> dict[str, Any]:
    """Signed coverage-mismatch factor in ``[-1, +1]`` (positive favours the offensive player).

    ``role`` (``"wr"`` / ``"slot"`` / ``"te"``) picks the primary matchup grade;
    an unknown role raises ``ValueError``. Bad data never raises: ``None``,
    ``NaN``, strings, out-of-range grades and non-mappings are treated as
    missing. ``{}`` / ``None`` returns the neutral ``0.0``. See the module
    docstring for inputs, weights and the honest limits.
    """
    role_key = _normalise_role(role)
    data: Mapping[str, Any] = inputs if isinstance(inputs, Mapping) else {}
    raw = _read_inputs(data)
    factors = ROLE_FACTORS[role_key]
    role_factor_keys = {f.key for f in factors}

    opp_rate = raw["opp_man_coverage_rate"]
    mix_rate = MAN_RATE_LEAGUE if opp_rate is None else opp_rate

    scoring: dict[str, float] = {}
    feeds: dict[str, list[str]] = {}

    # 1) WR1-vs-CB1 grade, amplified (or damped) by the shadow rate.
    grade = raw["wr_vs_cb_matchup_grade"]
    shadow = raw["opp_cb1_shadow_rate"]
    shadow_multiplier = 0.5 + (DEFAULT_SHADOW_RATE if shadow is None else shadow)
    if grade is not None:
        scoring["wr_vs_cb_matchup_grade"] = NEUTRAL_GRADE + (grade - NEUTRAL_GRADE) * shadow_multiplier
        feeds["wr_vs_cb_matchup_grade"] = ["wr_vs_cb_matchup_grade"] + (["opp_cb1_shadow_rate"] if shadow is not None else [])

    # 2) The two single-grade matchups.
    for key in ("slot_wr_vs_nickel_grade", "te_vs_lb_coverage_grade"):
        if raw[key] is not None:
            scoring[key] = raw[key]
            feeds[key] = [key]

    # 3) The opposing CB's grade at this defense's man/zone mix (higher = better CB = worse for the WR).
    cb_man, cb_zone = raw["cb_man_coverage_grade"], raw["cb_zone_coverage_grade"]
    cb_at_mix = common.blend_man_zone(cb_man, cb_zone, mix_rate)
    if cb_at_mix is not None:
        scoring["cb_coverage_grade_at_opp_mix"] = cb_at_mix
        cb_feeds = [k for k, v in (("cb_man_coverage_grade", cb_man), ("cb_zone_coverage_grade", cb_zone)) if v is not None]
        if cb_man is not None and cb_zone is not None and opp_rate is not None:  # a blend weight needs both sides
            cb_feeds.append("opp_man_coverage_rate")
        feeds["cb_coverage_grade_at_opp_mix"] = cb_feeds

    # 4) The WR's man/zone yprr: each side standardised against league WRs, blended at the defense's mix.
    wr_man, wr_zone = raw["wr_yprr_vs_man"], raw["wr_yprr_vs_zone"]
    z_man = None if wr_man is None else (wr_man - YPRR_MAN_MEAN) / YPRR_MAN_SD
    z_zone = None if wr_zone is None else (wr_zone - YPRR_ZONE_MEAN) / YPRR_ZONE_SD
    wr_at_mix = common.blend_man_zone(z_man, z_zone, mix_rate)
    if wr_at_mix is not None:
        scoring["wr_yprr_at_opp_mix"] = wr_at_mix
        wr_feeds = [k for k, v in (("wr_yprr_vs_man", wr_man), ("wr_yprr_vs_zone", wr_zone)) if v is not None]
        if wr_man is not None and wr_zone is not None and opp_rate is not None:
            wr_feeds.append("opp_man_coverage_rate")
        feeds["wr_yprr_at_opp_mix"] = wr_feeds

    # Score only what this role scores.
    role_scoring = {k: v for k, v in scoring.items() if k in role_factor_keys}
    scored = common.score_factors(role_scoring, factors)
    z = float(scored["z"])
    factor = common.clamp(common.r(common.signed_unit(GAIN * z, 1.0)), -1.0, 1.0)

    # --- audit: which canonical inputs were used / ignored / defaulted -------------------------
    used_keys: set[str] = set()
    for fkey in scored["used"]:
        used_keys.update(feeds.get(fkey, ()))
    supplied = {k for k in CANONICAL_INPUTS if raw[k] is not None}
    relevant = {src for f in factors for src in _FACTOR_SOURCES[f.key]}

    inputs_used = [k for k in CANONICAL_INPUTS if k in used_keys]
    inputs_ignored = [k for k in CANONICAL_INPUTS if k in supplied and k not in used_keys]
    inputs_defaulted = [k for k in CANONICAL_INPUTS if k in relevant and k not in supplied]
    notes: list[str] = []
    for key in inputs_ignored:
        if key not in relevant:
            notes.append(f"{key} is not scored for role {role_key!r}")
        elif key == "opp_cb1_shadow_rate":
            notes.append("opp_cb1_shadow_rate only scales wr_vs_cb_matchup_grade, which was not supplied")
        elif key == "opp_man_coverage_rate":
            notes.append("opp_man_coverage_rate is only a man/zone blend weight and no complete man/zone pair was supplied")

    contributions = scored["contributions"]
    for fkey, contribution in contributions.items():
        contribution["inputs"] = list(feeds.get(fkey, ()))

    result: dict[str, Any] = {
        "model": MODEL_NAME,
        "role": role_key,
        OUTPUT_KEY: factor,
        "probability": factor,  # alias only: a signed score in [-1, 1], NOT a probability
        "reference_probability": BASE_RATE,
        "references": {OUTPUT_KEY: BASE_RATE},
        "output_range": [-1.0, 1.0],
        "z_score": common.r(z, 4),
        "coverage": common.r(scored["coverage"], 3),
        "inputs_used": inputs_used,
        "inputs_defaulted": inputs_defaulted,
        "inputs_ignored": inputs_ignored,
        "contributions": contributions,
        "drivers": common.describe_drivers(contributions),
        "shadow_multiplier": common.r(shadow_multiplier, 4),
        "man_rate_used": common.r(mix_rate, 4),
        "implied_multipliers": {
            "wr_yards": common.r(1.0 + factor * WR_YARDS_COEFFICIENT),
            "te_yards": common.r(1.0 + factor * TE_YARDS_COEFFICIENT),
        },
        "notes": notes,
        "basis": f"{_BASIS} [role={role_key}, {len(inputs_used)} of {len(relevant)} relevant inputs used]",
    }
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "%s[%s]: %s=%+.4f (reference %.1f, z=%+.3f, coverage=%.0f%%, used=%d, defaulted=%d, ignored=%d) drivers=%s",
            MODEL_NAME, role_key, OUTPUT_KEY, factor, BASE_RATE, z, 100.0 * float(scored["coverage"]),
            len(inputs_used), len(inputs_defaulted), len(inputs_ignored), "; ".join(result["drivers"]),
        )
    return result


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def _factor_from(result: Any) -> float | None:
    """The result's signed factor clamped to ``[-1, 1]``, or ``None`` when unusable."""
    if not isinstance(result, Mapping):
        return None
    value = common.to_float(result.get(OUTPUT_KEY))
    if value is None:
        return None
    return common.clamp(value, -1.0, 1.0)


def apply_coverage_mismatch(
    projection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    recenter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the brief's formulas to a flat projection; ``(new_projection, adjustments)``.

    ``wr_yards *= 1 + f * 0.25`` and ``te_yards *= 1 + f * 0.20`` where ``f`` is
    ``result["coverage_mismatch_factor"]`` (clamped to ``[-1, 1]``). The input
    mapping is never mutated, other keys are copied through, and a field that is
    absent or non-numeric is recorded as ``applied=False``.

    ``recenter`` is accepted for interface uniformity with the probability
    models and has **no effect**: the factor is signed with neutral ``0``, so
    dividing by the same formula at the reference gives ``1 + 0 * c == 1``
    exactly (the reference is pinned to ``0`` here, whatever ``result`` carries).
    """
    new_projection: dict[str, Any] = dict(projection) if isinstance(projection, Mapping) else {}
    adjustments: list[dict[str, Any]] = []
    factor = _factor_from(result)
    for field, coefficient in (("wr_yards", WR_YARDS_COEFFICIENT), ("te_yards", TE_YARDS_COEFFICIENT)):
        if factor is None:
            adjustments.append(
                {"stage": STAGE, "field": field, "applied": False,
                 "reason": f"result has no usable {OUTPUT_KEY}"}
            )
            continue
        common.scale_field(
            new_projection, field,
            stage=STAGE, probability=factor, coefficient=coefficient, direction=+1,
            reference=BASE_RATE, recenter=recenter, adjustments=adjustments,
        )
        adjustments[-1]["signal"] = OUTPUT_KEY
    return new_projection, adjustments


__all__ = [
    "BASE_RATE",
    "CANONICAL_INPUTS",
    "DEFAULT_SHADOW_RATE",
    "FACTORS",
    "GAIN",
    "GRADE_SD",
    "INPUT_FEEDS",
    "LOG",
    "MAN_RATE_LEAGUE",
    "MAN_RATE_TEAM_SD",
    "MODEL_NAME",
    "OUTPUT_KEY",
    "ROLES",
    "ROLE_FACTORS",
    "STAGE",
    "TE_YARDS_COEFFICIENT",
    "WR_MIX_Z_SD",
    "WR_YARDS_COEFFICIENT",
    "YPRR_MAN_MEAN",
    "YPRR_MAN_SD",
    "YPRR_MAX",
    "YPRR_ZONE_MEAN",
    "YPRR_ZONE_SD",
    "apply_coverage_mismatch",
    "compute_coverage_mismatch",
]
