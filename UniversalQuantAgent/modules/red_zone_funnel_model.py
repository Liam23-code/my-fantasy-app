"""Red-zone funnel model for the NFL upgrade pack: where does this matchup push red-zone touchdowns?

Output (both SIGNED, bounded to ``[-1, +1]``, neutral ``0``)
------------------------------------------------------------
``rz_pass_funnel`` and ``rz_run_funnel``.  Positive means *the matchup funnels
red-zone touchdowns toward that phase and the offense is built to take
advantage of it*; negative means the matchup funnels them away from it.  The
brief applies them as::

    qb_td_prob *= (1 + rz_pass_funnel * 0.35)
    rb_td_prob *= (1 + rz_run_funnel  * 0.30)
    te_td_prob *= (1 + rz_pass_funnel * 0.25)

Event definition and base rate
------------------------------
A signed factor has no probability, but the model is anchored to one measurable
quantity so that "neutral" means something.  ``BASE_RATE`` is the league share
of red-zone touchdowns that are scored through the air:

    *Of all offensive touchdowns scored on a snap that began at or inside the
    opponent's 20-yard line (``yardline_100 <= 20``), regular season, excluding
    two-point tries, kneels, spikes, special-teams and defensive scores, what
    fraction were passing touchdowns?*

nflverse play-by-play 2019-2025, n = 7,008 red-zone touchdowns (3,974 pass,
3,034 rush): **0.567** (season by season 0.550-0.577).  ``rz_pass_funnel = 0``
and ``rz_run_funnel = 0`` mean "the matchup leaves that split at the league
value".  ``result["probability"]`` is the *implied* pass share of this
matchup's red-zone touchdowns if the brief's three coefficients are applied to
the league split (see :func:`implied_pass_td_share`); it is a read-out of what
the funnels mean, not a forecast fitted to outcomes, and
``reference_probability`` is ``BASE_RATE``.

How the number is built (binding design)
----------------------------------------
1. **Direction comes only from the contrast between the opponent's red-zone
   pass defense and run defense.**  Both are team percentile grades 0-100
   (higher = better defense).  A defense that is stout against the run and soft
   against the pass funnels offenses to the pass (``rz_pass_funnel > 0``,
   ``rz_run_funnel < 0``); the mirror image flips the signs; a uniformly strong
   or uniformly weak defense has contrast ~0 and so funnels nothing.  The two
   grades carry equal weight, so the contrast is exactly antisymmetric::

       z_contrast = ((run_grade - 50) - (pass_grade - 50)) / (2 * 28.9)

   If one grade is missing it is treated as average and the contrast is
   halved (missing data shrinks toward neutral, as everywhere in the pack).
   ``opp_rz_pass_funnel`` / ``opp_rz_run_funnel`` (signed, ``[-1, 1]``) may be
   supplied directly; they take precedence over the grades (a funnel supplied
   for one phase and not the other is mirrored onto the other phase).
2. **The offense amplifies or damps the funnel for the specific phase.**  An
   amplifier ``a = 1 + AMP_RANGE * tanh(z_offense / AMP_SCALE)`` (``AMP_RANGE =
   0.6`` so ``a`` lies in 0.4 .. 1.6) is built from how much of this offense's
   red-zone scoring runs through the phase: ``team_rz_pass_rate`` (inverted for
   the run phase), ``qb_rz_efficiency`` / ``te_rz_target_share`` for the pass
   phase, ``rb_rz_efficiency`` for the run phase.  A team that cannot run in
   the red zone gains little from a soft run defense (``a < 1``), and is
   likewise little affected by a stout one.  It scales the *signed* funnel, so
   it moves ``|funnel|`` and never flips the sign.  With no contrast the
   amplifier has nothing to scale: **the offense's own strength is already in
   the player's baseline TD probability, so adding it here would double count
   it.**  This is a pure matchup factor.
3. **Squash.** ``funnel = tanh(GAIN * z_contrast * a)``: smooth, bounded,
   odd in the contrast.

Inputs (canonical keys; ``None``/NaN/text are "missing"; rates accept 0.31 or 31)
--------------------------------------------------------------------------------
``opp_rz_pass_defense_grade`` / ``opp_rz_run_defense_grade`` (0-100 team
percentile, higher = better red-zone defense; values outside 0-100 are
rejected as data errors, not clamped), ``team_rz_pass_rate`` (share of red-zone
plays that are dropbacks), ``qb_rz_efficiency`` (TD per red-zone dropback),
``rb_rz_efficiency`` (TD per red-zone carry), ``te_rz_target_share`` (the TE's
share of his team's red-zone targets), plus the optional precomputed
``opp_rz_pass_funnel`` / ``opp_rz_run_funnel`` (signed ``[-1, 1]``; outside the
range = rejected).  No input in this model lacks a free feed: all are
derivable from nflverse play-by-play (the loader builds the grades from
red-zone EPA/play allowed or similar); see "Honest limits" for how little
signal the defensive grades carry.

GAIN
----
``GAIN = 0.30``.  With every input one standard deviation in the favourable
direction (run-defense grade +1 SD, pass-defense grade -1 SD, pass rate, QB,
RB and TE inputs +1 SD) ``rz_pass_funnel = +0.38`` and ``rz_run_funnel =
-0.27``; with the grades at the ends of the scale (100th / 0th percentile,
about 1.7 SD -- a +2 SD grade does not exist on a 0-100 scale) and the offense
inputs at +2 SD, ``+0.66`` / ``-0.41`` -- well short of the +-1 ceiling.  The
most extreme contrast the grade scale allows, at a neutral offense, is
``+-0.48``; a supplied direct funnel can reach +-1.  The scale is anchored to
the data rather than to the association: one observed SD of grade contrast
(0.71 in z units, measured on 224 defense-seasons) maps to ``rz_pass_funnel =
+0.21``, which with the brief's coefficients implies a +0.033 move in the pass
share of red-zone touchdowns -- about one *true-talent* SD of team variation in
that share (0.032: observed sd 0.096 less binomial noise).  The funnel is
sized so a typical matchup never claims more than the league's real team
spread.

League references (``Factor.mean`` / ``Factor.sd``)
---------------------------------------------------
All from nflverse play-by-play (``nflreadpy.load_pbp``), 2019-2025 regular
season, plays snapped at or inside the 20 (two-point tries excluded), dropbacks
= ``qb_dropback`` (sacks and scrambles included, spikes not), runs =
``rush`` and not a dropback and not a kneel; position from ``load_rosters``:

* ``team_rz_pass_rate``: team-seasons, n = 224 (7 seasons x 32 teams, ~158 red-zone
  plays each): mean 0.5385, sd 0.0728.
* ``qb_rz_efficiency``: passing TD per red-zone dropback, QB-seasons with >= 40
  red-zone dropbacks, n = 199: mean 0.2261, sd 0.0525.
* ``rb_rz_efficiency``: rushing TD per red-zone designed carry, RB-seasons
  with >= 25 red-zone carries, n = 213: mean 0.1672, sd 0.0660.
* ``te_rz_target_share``: a team's top TE by season targets, his share of the
  team's red-zone targets, n = 224: mean 0.1577, sd 0.0699.
* defense grades: mean 50 and sd 28.9 are properties of a 0-100 percentile
  scale (uniform), not of football; they are priors.  The measured spread of
  32-team rank percentiles with ``(rank - 1) / 31`` is 29.8; the difference is
  immaterial.

These are observed (noisy) distributions, so a z-score is "how unusual is this
observed value", not "how good is the underlying talent".

Weights are PRIORS, not fits
----------------------------
Causal order: the defensive contrast (6 of 14) is the only source of
direction, so it carries the most weight; the offense's red-zone tendency (3)
sets how much of the touchdown volume passes through each phase at all; the
conversion efficiencies (2 each) say how well it cashes in; the TE share (1)
only redistributes the pass phase among receivers.  Nothing was fitted.

Honest limits
-------------
* **The defensive contrast is close to noise on free data.**  Red-zone
  samples are tiny (about 85 dropbacks and 73 runs per defense-season).  On
  2019-2025 team-seasons (EPA/play allowed in the red zone, percentile
  ranked), the odd/even-week split-half correlation of the pass-minus-run
  contrast is 0.04, year over year -0.02; first-half contrast against the
  second-half pass share of red-zone TDs allowed has Spearman 0.00 (n = 224),
  and a game-level version with grades from prior games only is also 0.00
  (n = 2,343 defense-games).  Observed team-season variation in the pass
  share allowed (sd 0.096) is almost entirely binomial noise (sd 0.090).  The
  funnel direction is therefore a *prior*, not a measured edge: treat it as
  a small lean, and it should be the first thing dropped if a better
  red-zone defense signal is not available.
* The same holds end to end: ``rz_pass_funnel`` computed from prior-games
  grades plus team-level stand-ins for the offense inputs, against the pass
  share of that offense's red-zone TDs in the game, is Spearman 0.01
  (n = 2,236 offense-games, weeks 5-18, 2019-2025).  No attempt was made to
  tune anything to move it.
* The offense side is better supported but modest: first-half red-zone pass
  rate against second-half pass share of red-zone TDs scored is Spearman 0.15
  (p = 0.03, n = 224; 0.09 at game level, n = 2,236); the pass rate's
  first-half vs second-half r is 0.32 and the TE's odd/even-week target
  share r is 0.53.  QB and RB red-zone TD rates have essentially no
  persistence on this sample (odd/even r 0.02 and 0.05), so those two
  amplifier weights are the least empirically supported inputs.  By design the offense can only scale a defensive contrast, so the
  one signal with a measured association never moves the funnel on its own
  (its level is already in the player's baseline TD probability).
* ``result["probability"]`` is implied by three brief coefficients applied to
  a league split; it is not calibrated against game outcomes.
* **Monotonicity of the offense inputs is in ``|funnel|``, not in the signed
  value.**  The amplifier scales the funnel, so raising the team's pass rate
  (or the QB / TE inputs) raises ``rz_pass_funnel`` while that funnel is
  positive and makes it *more negative* while it is negative (a pass-heavy
  offence is more exposed to a pass-stout red-zone defense); the run phase is
  the mirror image.  This is the literal "amplify or damp" reading of the
  design and the one place the pack-wide "never lowers the output" rule holds
  for the magnitude rather than the signed number.  The alternative reading
  (a strong offense *resists* an adverse matchup) is equally plausible and no
  free data separates them.  From a neutral matchup nothing moves, by design.
* Grades are percentile ranks on 0-100 only: a ``0-1`` fraction is not
  rescaled (indistinguishable from the 1st percentile) and ``"85%"`` is read as
  85, not 0.85.

Pure and offline: standard library and :mod:`modules.nfl_pack_common` only.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Mapping

from modules import nfl_pack_common as common
from modules.nfl_pack_common import Factor

LOG = logging.getLogger("nfl_pack.red_zone_funnel")

MODEL_NAME = "red_zone_funnel"
STAGE = "red_zone_funnel"

# --- the brief's coefficients, verbatim -------------------------------------
QB_TD_PROB_COEFFICIENT = 0.35  # qb_td_prob *= (1 + rz_pass_funnel * 0.35)
RB_TD_PROB_COEFFICIENT = 0.30  # rb_td_prob *= (1 + rz_run_funnel  * 0.30)
TE_TD_PROB_COEFFICIENT = 0.25  # te_td_prob *= (1 + rz_pass_funnel * 0.25)

# --- anchor, gain and amplifier shape ---------------------------------------
# Share of offensive red-zone TDs scored by pass. nflverse pbp 2019-2025 REG, yardline_100 <= 20,
# offensive TDs only, no 2-pt tries: 3,974 pass + 3,034 rush = 7,008 TDs -> 0.5671.
BASE_RATE = 0.567
# tanh gain on the contrast z (see "GAIN" in the module docstring); a prior anchored to the
# measured true-talent spread of the red-zone TD split, NOT fitted to the association check.
GAIN = 0.30
# Offense amplifier a = 1 + AMP_RANGE * tanh(z / AMP_SCALE)  ->  0.4 .. 1.6; 1.0 for a neutral offense.
AMP_RANGE = 0.60
AMP_SCALE = 1.5

#: Signed ``[-1, 1]`` precomputed funnels a caller may supply; they win over the grade contrast.
DIRECT_PASS_KEY = "opp_rz_pass_funnel"
DIRECT_RUN_KEY = "opp_rz_run_funnel"
DIRECT_KEYS: tuple[str, ...] = (DIRECT_PASS_KEY, DIRECT_RUN_KEY)

# A direct funnel of exactly +-1 would need atanh(+-1) = inf; keep it a hair inside.
_DIRECT_CLIP = 0.9999
_DIRECT_TOLERANCE = 1e-9

# --- factors (weights are priors; order = causal importance, see module docstring) -------------
# Grades: the contrast is the only source of direction. Equal weights make it exactly antisymmetric.
OPP_RZ_RUN_DEFENSE_GRADE = Factor(
    key="opp_rz_run_defense_grade",
    mean=50.0,
    sd=28.9,
    weight=3.0,
    direction=+1,  # a STOUTER run defense funnels offenses to the pass
    label="Opponent red-zone run defense",
    unit="pctile",
    source="prior: uniform 0-100 percentile scale (sd 28.9); measured 29.8 for 32-team ranks, pbp 2019-2025 n=224",
)
OPP_RZ_PASS_DEFENSE_GRADE = Factor(
    key="opp_rz_pass_defense_grade",
    mean=50.0,
    sd=28.9,
    weight=3.0,
    direction=-1,  # a SOFTER pass defense funnels offenses to the pass
    label="Opponent red-zone pass defense",
    unit="pctile",
    source="prior: uniform 0-100 percentile scale (sd 28.9); measured 29.8 for 32-team ranks, pbp 2019-2025 n=224",
)
# Offense tendency: how much of the red-zone volume goes through the pass at all.
# nflverse pbp 2019-2025 REG, team-seasons n=224 (~158 red-zone plays each): mean 0.5385, sd 0.0728.
TEAM_RZ_PASS_RATE = Factor(
    key="team_rz_pass_rate",
    mean=0.5385,
    sd=0.0728,
    weight=3.0,
    direction=+1,  # as stated for the pass phase; the run phase uses the inverse (see _RUN_AMP_FACTORS)
    label="Team red-zone pass rate",
    kind="rate",
    unit="%",
    source="nflverse pbp 2019-2025 REG, red-zone dropbacks / (dropbacks + designed runs), team-seasons n=224",
)
# Efficiency: how well the offense cashes in once it uses the phase.
# QB-seasons >= 40 RZ dropbacks, n=199: mean 0.2261, sd 0.0525 (passing TD per RZ dropback).
QB_RZ_EFFICIENCY = Factor(
    key="qb_rz_efficiency",
    mean=0.2261,
    sd=0.0525,
    weight=2.0,
    direction=+1,
    label="QB red-zone TD rate",
    kind="rate",
    unit="%",
    source="nflverse pbp 2019-2025 REG, passing TD per RZ dropback, QB-seasons >= 40 RZ dropbacks n=199",
)
# RB-seasons >= 25 RZ designed carries, n=213: mean 0.1672, sd 0.0660 (rushing TD per RZ carry).
RB_RZ_EFFICIENCY = Factor(
    key="rb_rz_efficiency",
    mean=0.1672,
    sd=0.0660,
    weight=2.0,
    direction=+1,  # run phase only
    label="RB red-zone TD rate",
    kind="rate",
    unit="%",
    source="nflverse pbp 2019-2025 REG, rushing TD per RZ carry, RB-seasons >= 25 RZ carries n=213",
)
# TE1 (most season targets) share of team red-zone targets, n=224: mean 0.1577, sd 0.0699.
TE_RZ_TARGET_SHARE = Factor(
    key="te_rz_target_share",
    mean=0.1577,
    sd=0.0699,
    weight=1.0,
    direction=+1,  # pass phase only
    label="TE red-zone target share",
    kind="rate",
    unit="%",
    source="nflverse pbp 2019-2025 REG, top TE by season targets, share of team RZ targets, team-seasons n=224",
)

#: Every canonical input, in causal order. ``direction`` is stated for ``rz_pass_funnel``'s grade
#: contrast and the pass-phase amplifier; ``rb_rz_efficiency`` is run-phase only and
#: ``team_rz_pass_rate`` is inverted for the run phase.
FACTORS: tuple[Factor, ...] = (
    OPP_RZ_RUN_DEFENSE_GRADE,
    OPP_RZ_PASS_DEFENSE_GRADE,
    TEAM_RZ_PASS_RATE,
    QB_RZ_EFFICIENCY,
    RB_RZ_EFFICIENCY,
    TE_RZ_TARGET_SHARE,
)

_GRADE_FACTORS: tuple[Factor, ...] = (OPP_RZ_RUN_DEFENSE_GRADE, OPP_RZ_PASS_DEFENSE_GRADE)
_PASS_AMP_FACTORS: tuple[Factor, ...] = (TEAM_RZ_PASS_RATE, QB_RZ_EFFICIENCY, TE_RZ_TARGET_SHARE)
_RUN_AMP_FACTORS: tuple[Factor, ...] = (dataclasses.replace(TEAM_RZ_PASS_RATE, direction=-1), RB_RZ_EFFICIENCY)
_AMP_KEYS: tuple[str, ...] = (
    TEAM_RZ_PASS_RATE.key,
    QB_RZ_EFFICIENCY.key,
    RB_RZ_EFFICIENCY.key,
    TE_RZ_TARGET_SHARE.key,
)
_PHASE = {
    OPP_RZ_RUN_DEFENSE_GRADE.key: "contrast",
    OPP_RZ_PASS_DEFENSE_GRADE.key: "contrast",
    TEAM_RZ_PASS_RATE.key: "both",
    QB_RZ_EFFICIENCY.key: "pass",
    TE_RZ_TARGET_SHARE.key: "pass",
    RB_RZ_EFFICIENCY.key: "run",
}

BASIS = (
    "Red-zone funnel: tanh(GAIN x opponent pass-vs-run red-zone defense contrast x offense phase amplifier). "
    "Weights are priors, not fitted. League references are observed nflverse play-by-play 2019-2025 regular-season "
    "distributions (team/QB/RB/TE-seasons, n=199-224); BASE_RATE 0.567 = pass share of 7,008 red-zone TDs. "
    "On that sample the defensive red-zone contrast has near-zero persistence (split-half r 0.04) and no "
    "out-of-sample association with the TD split (Spearman 0.00), so the funnel is a prior lean, not a measured edge."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def implied_pass_td_share(rz_pass_funnel: float, rz_run_funnel: float, *, base_rate: float = BASE_RATE) -> float:
    """Pass share of red-zone TDs implied by applying the brief's coefficients to the league split.

    ``p = base * (1 + f_pass * 0.35)``, ``u = (1 - base) * (1 + f_run * 0.30)``, share = ``p / (p + u)``.
    Exactly ``base_rate`` at ``(0, 0)``.  A read-out of what the funnels mean, not a fitted forecast.
    """
    pass_part = base_rate * max(0.0, 1.0 + rz_pass_funnel * QB_TD_PROB_COEFFICIENT)
    run_part = (1.0 - base_rate) * max(0.0, 1.0 + rz_run_funnel * RB_TD_PROB_COEFFICIENT)
    total = pass_part + run_part
    return base_rate if total <= 0 else pass_part / total


def _as_mapping(inputs: Any) -> Mapping[str, Any]:
    return inputs if isinstance(inputs, Mapping) else {}


def _grade_value(raw: Any) -> float | None:
    """A 0-100 percentile grade as a float, or ``None`` if unusable or out of range.

    ``"85%"`` is the 85th percentile: ``common.to_float`` reads a trailing ``%`` as a fraction
    (``0.85``), which on a 0-100 scale would silently become the 1st percentile, so it is scaled back.
    A bare ``0.85`` is *not* rescaled (it cannot be told apart from the 1st percentile); grades are
    percentile ranks on 0-100 only.
    """
    value = common.to_float(raw)
    if value is None:
        return None
    if isinstance(raw, str) and raw.strip().endswith("%"):
        value *= 100.0
    if value < 0.0 or value > 100.0:
        return None
    return value


def _clean_grades(source: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``source`` with grades normalised to floats and out-of-range ones blanked.

    A 0-100 grade of -5 or 4000 is a data error, rejected rather than clamped into plausibility.
    """
    clean = dict(source)
    for factor in _GRADE_FACTORS:
        for key in (factor.key, *factor.aliases):
            if key in clean:
                clean[key] = _grade_value(clean[key])
    return clean


def _read_direct(source: Mapping[str, Any], key: str) -> float | None:
    """A signed funnel in ``[-1, 1]`` or ``None`` (out-of-range is rejected, not clamped)."""
    if key not in source:
        return None
    value = common.to_float(source[key])
    if value is None or abs(value) > 1.0 + _DIRECT_TOLERANCE:
        return None
    return common.clamp(value, -1.0, 1.0)


def _funnel_to_contrast(value: float) -> float:
    """Inverse of the squash at a neutral amplifier: the contrast z that yields ``value``."""
    return math.atanh(common.clamp(value, -_DIRECT_CLIP, _DIRECT_CLIP)) / GAIN


def _amplifier(z_offense: float) -> float:
    return 1.0 + AMP_RANGE * common.signed_unit(z_offense, AMP_SCALE)


def _squash(contrast_z: float, amplifier: float) -> float:
    return common.clamp(common.signed_unit(contrast_z * amplifier, 1.0 / GAIN), -1.0, 1.0)


def _driver_lines(
    *,
    source: str,
    contributions: Mapping[str, Mapping[str, Any]],
    direct_pass: float | None,
    direct_run: float | None,
    f_pass: float,
    f_run: float,
    pass_amp: float,
    run_amp: float,
) -> list[str]:
    lines: list[str] = []
    if source == "direct":
        parts = []
        if direct_pass is not None:
            parts.append(f"pass {direct_pass:+.2f}")
        if direct_run is not None:
            parts.append(f"run {direct_run:+.2f}")
        lines.append(f"Supplied opponent red-zone funnel ({', '.join(parts)}) sets the direction")
    elif source == "grades":
        run_c = contributions.get(OPP_RZ_RUN_DEFENSE_GRADE.key)
        pass_c = contributions.get(OPP_RZ_PASS_DEFENSE_GRADE.key)
        parts = []
        if run_c:
            parts.append(f"run defense {run_c['value']:.0f}th pctile")
        if pass_c:
            parts.append(f"pass defense {pass_c['value']:.0f}th pctile")
        where = "toward the pass" if f_pass > 0 else "toward the run"
        if abs(f_pass) < 0.02:
            lines.append(f"Opponent red-zone {' / '.join(parts)}: balanced, no funnel")
        else:
            lines.append(
                f"Opponent red-zone {' / '.join(parts)}: funnels {where} "
                f"(pass funnel {f_pass:+.2f}, run funnel {f_run:+.2f})"
            )
    if source == "none":
        lines.append("No opponent red-zone defense contrast supplied: funnels stay neutral (offense inputs only scale a contrast)")
    shaped = []
    for key in _AMP_KEYS:
        item = contributions.get(key)
        if not item:
            continue
        # ``z`` is signed toward the pass amplifier for pass / "both" inputs and toward the run
        # amplifier for the run-only input (the run phase reads team_rz_pass_rate inverted).
        effect_z = float(item.get("z", 0.0))
        if abs(effect_z) < 0.5:
            continue
        shaped.append((abs(effect_z) * float(item.get("weight", 1.0)), key, item, effect_z))
    for _, key, item, effect_z in sorted(shaped, key=lambda t: t[0], reverse=True)[:2]:
        unit = str(item.get("unit") or "")
        value = float(item.get("value", 0.0))
        shown = f"{value * 100:.1f}%" if unit == "%" else f"{value:.2f}"
        phase = item.get("phase", "pass")
        if key == TEAM_RZ_PASS_RATE.key:
            verb = (
                f"amplifies the pass funnel (x{pass_amp:.2f}) and damps the run funnel (x{run_amp:.2f})"
                if effect_z > 0
                else f"damps the pass funnel (x{pass_amp:.2f}) and amplifies the run funnel (x{run_amp:.2f})"
            )
        else:
            side = "run" if phase == "run" else "pass"
            amp = run_amp if side == "run" else pass_amp
            verb = f"{'amplifies' if effect_z > 0 else 'damps'} the {side} funnel (x{amp:.2f})"
        lines.append(f"{item.get('label', key)} {shown} ({effect_z:+.1f} SD) {verb}")
    return lines


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute_red_zone_funnel(inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Signed red-zone funnels ``rz_pass_funnel`` / ``rz_run_funnel`` in ``[-1, 1]`` (neutral 0).

    See the module docstring for the construction.  Never raises on bad data; ``None`` / ``{}``
    returns the neutral result.  Contract keys: ``model``, ``rz_pass_funnel``, ``rz_run_funnel``,
    ``probability`` (implied pass share of red-zone TDs), ``reference_probability``
    (``BASE_RATE``), ``references`` (0.0 for both funnels), ``z_score`` (grade contrast),
    ``coverage``, ``inputs_used``, ``inputs_defaulted``, ``contributions``, ``drivers``, ``basis``;
    extras: ``contrast_source``, ``pass_amplifier``, ``run_amplifier``, ``inputs_overridden``.
    """
    source = _as_mapping(inputs)
    clean = _clean_grades(source)
    grades = common.score_factors(clean, _GRADE_FACTORS)
    pass_amp_scored = common.score_factors(clean, _PASS_AMP_FACTORS)
    run_amp_scored = common.score_factors(clean, _RUN_AMP_FACTORS)

    direct_pass = _read_direct(source, DIRECT_PASS_KEY)
    direct_run = _read_direct(source, DIRECT_RUN_KEY)
    if direct_pass is not None or direct_run is not None:
        contrast_source = "direct"
        pass_contrast = (
            _funnel_to_contrast(direct_pass) if direct_pass is not None else -_funnel_to_contrast(direct_run)  # type: ignore[arg-type]
        )
        run_contrast = (
            _funnel_to_contrast(direct_run) if direct_run is not None else -_funnel_to_contrast(direct_pass)  # type: ignore[arg-type]
        )
    else:
        contrast_source = "grades" if grades["used"] else "none"
        pass_contrast = float(grades["z"])
        run_contrast = -float(grades["z"])

    pass_amp = _amplifier(float(pass_amp_scored["z"]))
    run_amp = _amplifier(float(run_amp_scored["z"]))
    f_pass = _squash(pass_contrast, pass_amp)
    f_run = _squash(run_contrast, run_amp)

    # --- audit: contributions, coverage, used / defaulted ------------------------------------
    contributions: dict[str, dict[str, Any]] = {}
    if contrast_source != "direct":
        for key, item in grades["contributions"].items():
            contributions[key] = dict(item, phase="contrast")
    for key, item in pass_amp_scored["contributions"].items():
        contributions[key] = dict(item, phase=_PHASE[key])
    for key, item in run_amp_scored["contributions"].items():
        if key in contributions:
            contributions[key]["run_z"] = item["z"]
        else:
            contributions[key] = dict(item, phase=_PHASE[key])
    for key, value in ((DIRECT_PASS_KEY, direct_pass), (DIRECT_RUN_KEY, direct_run)):
        if value is None:
            continue
        z = _funnel_to_contrast(value)
        contributions[key] = {
            "value": common.r(value, 4),
            "z": common.r(z, 3),
            "weight": sum(f.weight for f in _GRADE_FACTORS),
            "effect": common.r(z, 4),
            "label": "Supplied opponent red-zone " + ("pass" if key == DIRECT_PASS_KEY else "run") + " funnel",
            "unit": "",
            "via": key,
            "phase": "contrast",
        }

    total_weight = sum(f.weight for f in FACTORS)
    grade_slot = sum(f.weight for f in _GRADE_FACTORS)
    used_amp = set(pass_amp_scored["used"]) | set(run_amp_scored["used"])
    used_grades = [] if contrast_source == "direct" else list(grades["used"])
    overridden = list(grades["used"]) if contrast_source == "direct" else []
    observed = (
        grade_slot
        if contrast_source == "direct"
        else sum(f.weight for f in _GRADE_FACTORS if f.key in used_grades)
    ) + sum(f.weight for f in FACTORS if f.key in used_amp)
    inputs_used = [f.key for f in FACTORS if f.key in used_grades or f.key in used_amp]
    inputs_used += [key for key, value in ((DIRECT_PASS_KEY, direct_pass), (DIRECT_RUN_KEY, direct_run)) if value is not None]
    inputs_defaulted = [f.key for f in FACTORS if f.key not in inputs_used and f.key not in overridden]

    references = {"rz_pass_funnel": 0.0, "rz_run_funnel": 0.0}
    probability = implied_pass_td_share(f_pass, f_run)
    result: dict[str, Any] = {
        "model": MODEL_NAME,
        "rz_pass_funnel": common.r(f_pass),
        "rz_run_funnel": common.r(f_run),
        "probability": common.r(probability),
        "reference_probability": common.r(BASE_RATE),
        "references": references,
        "z_score": common.r(pass_contrast, 4),
        "coverage": common.r(observed / total_weight if total_weight > 0 else 0.0, 3),
        "inputs_used": inputs_used,
        "inputs_defaulted": inputs_defaulted,
        "inputs_overridden": overridden,
        "contributions": contributions,
        "contrast_source": contrast_source,
        "pass_amplifier": common.r(pass_amp),
        "run_amplifier": common.r(run_amp),
        "drivers": _driver_lines(
            source=contrast_source,
            contributions=contributions,
            direct_pass=direct_pass,
            direct_run=direct_run,
            f_pass=f_pass,
            f_run=f_run,
            pass_amp=pass_amp,
            run_amp=run_amp,
        ),
        "basis": BASIS,
    }
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "%s: rz_pass_funnel=%+.4f rz_run_funnel=%+.4f (contrast=%s z=%+.3f, pass_amp=x%.2f, run_amp=x%.2f, "
            "implied_pass_share=%.4f vs %.4f, coverage=%.0f%%, used=%d, defaulted=%d) drivers=%s",
            MODEL_NAME,
            result["rz_pass_funnel"],
            result["rz_run_funnel"],
            contrast_source,
            pass_contrast,
            pass_amp,
            run_amp,
            probability,
            BASE_RATE,
            100.0 * result["coverage"],
            len(inputs_used),
            len(inputs_defaulted),
            "; ".join(result["drivers"]),
        )
    return result


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def _funnel_value(result: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(result, Mapping):
        return None
    value = common.to_float(result.get(key))
    return None if value is None else common.clamp(value, -1.0, 1.0)


def _reference(result: Mapping[str, Any] | None, key: str) -> float:
    if not isinstance(result, Mapping):
        return 0.0
    refs = result.get("references")
    if isinstance(refs, Mapping):
        value = common.to_float(refs.get(key))
        if value is not None:
            return value
    return 0.0


def _probability_cap(value: Any) -> float | None:
    """Ceiling for a TD field given its *current* value, or ``None`` for no ceiling.

    The pipeline hands this stage the engine's per-game touchdown RATES, which are expected counts
    (a QB's ``passing_tds`` of 1.3 is not a probability), as well as true probabilities. Funnel is a
    BOOST stage, so a ceiling that is too eager silently cuts a legitimate value:

    * ``value <= 1.0``: fraction scale, ceiling 1.0.
    * ``1.0 < value <= 1.5``: cannot be a fraction-scale probability; as an expected count (1.3 TDs)
      or a percent-scale rate (1.2%) it is nowhere near its own ceiling -> no ceiling. (A flat 1.0
      here would CUT a 1.3 expected-TD rate by 23% while the funnel is meant to raise it.)
    * ``value > 1.5``: percent scale, ceiling 100.0 (never below the value).

    The same rule ``qb_collapse_model`` uses, so the two stages agree on one field.
    """
    number = common.to_float(value)
    if number is None:
        return None
    if number <= 1.0:
        return 1.0
    if number <= 1.5:
        return None
    return max(100.0, number)


def apply_red_zone_funnel(
    projection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    recenter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the brief's red-zone funnel multipliers; returns ``(new_projection, adjustments)``.

    ``qb_td_prob *= 1 + rz_pass_funnel * 0.35``; ``rb_td_prob *= 1 + rz_run_funnel * 0.30``;
    ``te_td_prob *= 1 + rz_pass_funnel * 0.25``.  The funnels are signed with neutral 0
    (``result["references"]`` holds 0.0), so ``recenter=True`` divides by ``1 + 0 * c = 1`` and gives
    the same numbers as ``recenter=False`` -- the flag is accepted for contract uniformity and
    has no effect here.  A field that is a probability (<= 1.0) is capped at 1.0, one on a percent
    scale (> 1.5) at 100.0, and an expected count between 1.0 and 1.5 is left uncapped (see
    :func:`_probability_cap`).  A missing / non-numeric field is recorded as ``applied=False``;
    the input mapping is never mutated.
    """
    adjusted: dict[str, Any] = dict(projection) if isinstance(projection, Mapping) else {}
    adjustments: list[dict[str, Any]] = []
    plan = (
        ("qb_td_prob", "rz_pass_funnel", QB_TD_PROB_COEFFICIENT),
        ("rb_td_prob", "rz_run_funnel", RB_TD_PROB_COEFFICIENT),
        ("te_td_prob", "rz_pass_funnel", TE_TD_PROB_COEFFICIENT),
    )
    for field, funnel_key, coefficient in plan:
        funnel = _funnel_value(result, funnel_key)
        if funnel is None:
            adjustments.append(
                {"stage": STAGE, "field": field, "applied": False, "reason": f"result has no usable {funnel_key}"}
            )
            continue
        common.scale_field(
            adjusted,
            field,
            stage=STAGE,
            probability=funnel,
            coefficient=coefficient,
            direction=+1,
            reference=_reference(result, funnel_key),
            recenter=recenter,
            adjustments=adjustments,
            cap=_probability_cap(adjusted.get(field)),
        )
        adjustments[-1]["funnel"] = funnel_key
    return adjusted, adjustments


__all__ = [
    "AMP_RANGE",
    "AMP_SCALE",
    "BASE_RATE",
    "BASIS",
    "DIRECT_KEYS",
    "DIRECT_PASS_KEY",
    "DIRECT_RUN_KEY",
    "FACTORS",
    "GAIN",
    "LOG",
    "MODEL_NAME",
    "QB_TD_PROB_COEFFICIENT",
    "RB_TD_PROB_COEFFICIENT",
    "STAGE",
    "TE_TD_PROB_COEFFICIENT",
    "apply_red_zone_funnel",
    "compute_red_zone_funnel",
    "implied_pass_td_share",
]
