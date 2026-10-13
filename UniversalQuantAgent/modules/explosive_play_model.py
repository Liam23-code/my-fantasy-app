"""Explosive-play model: per-game chance that a WR / RB produces a 30+ yard play.

Part of the NFL model upgrade pack (see ``docs/nfl_model_upgrade_pack.md``).  Two
independent factor models share one result:

* ``explosive_play_prob_pass`` -- a wide receiver breaks a long catch.
* ``explosive_play_prob_run``  -- a running back breaks a long run.

and the brief's additive adjustments are applied by :func:`apply_explosive_play`::

    wr_yards += explosive_play_prob_pass * 18
    rb_yards += explosive_play_prob_run  * 12

The module is pure and offline: standard library plus ``modules.nfl_pack_common``.

What the probabilities mean (EVENT DEFINITIONS)
-----------------------------------------------
Both are **per-player-game** probabilities, measured on nflverse play-by-play,
regular season, 2019-2025 (``nflreadpy.load_pbp``; plays nullified by penalty are
``no_play`` there and excluded; 2-point tries, kneels, spikes and sacks excluded;
positions from ``load_players``):

* ``explosive_play_prob_pass`` = P(a WR with **>= 4 targets** in the game has **at
  least one reception of 30+ yards**).  Measured base rate **0.2727**, n = 8,383
  WR-games (per season 0.266-0.288).
* ``explosive_play_prob_run``  = P(an RB with **>= 8 carries** in the game has **at
  least one carry of 30+ yards**).  Measured base rate **0.1263**, n = 4,617
  RB-games (per season 0.110-0.148).

So ``p = 0.27`` means "about one in four qualifying wide-receiver games contains a
30-yard catch", and ``0.126`` means "about one in eight qualifying running-back
games contains a 30-yard run".

Why 30 yards, and why WR only.  The suggested 20+ / 15+ thresholds are coin flips
at the player-game level (WR >= 4 targets, 20+: 0.542; RB >= 8 carries, 15+: 0.464,
same sample), which makes the brief's literal ``+ p*18`` worth ~10 yards of WR
production at an average player-game.  30+ keeps the event recognisable ("big play")
and common while keeping the literal formula at a plausible size (0.27*18 = 4.9 yd,
0.126*12 = 1.5 yd).  Honest note: the pass base rate (0.27) sits above the usual
"common but not routine" 8-20% band (the run rate, 0.126, is inside it).  A 40+ yard
threshold would give 0.134 for WR (n = 8,383) but halves the event count and with it
the power of every check below, so 30+ is a compromise, not the only defensible
choice.  The event is defined over WR (pooled WR+TE would be 0.2355,
n = 11,323) because the brief only touches ``wr_yards`` and every pass input is
WR-specific.  Input thresholds (``air_yards >= 20``, carries ``>= 15`` yd) are the
pack vocabulary's and are *traits*, not the event, so they need not equal it.

The brief's "+18 / +12 yards" are the brief's.  For scale: a WR reception of 30+ yd
averages 42.0 yd against 12.7 for all WR receptions; an RB carry of 30+ averages
45.0 against 4.3.  18 and 12 are therefore well under the realised excess of one
such play (~29 and ~41 yd); the coefficients are conservative, and a WR game has
0.32 30-yard catches on average, not 1 -- ``p`` is a probability of *at least one*,
which the additive formula treats as an expected count.  That is the brief's
simplification; ``recenter=True`` keeps it from biasing an average game.

How ``p`` is built
------------------
Each output is ``sigmoid(logit(BASE_RATE) + GAIN * z)`` where ``z`` is the
weight-normalised mean of the direction-signed z-scores of the inputs the caller
supplied (``nfl_pack_common.score_factors``).  An input that is missing contributes
0 but keeps its weight in the denominator, so **missing data shrinks toward the
league-average case**; ``coverage`` says how much weight was observed.  With no
inputs, or exactly league-average inputs, ``p == BASE_RATE`` (the reference).

Inputs (canonical keys; ``+1`` raises the probability, ``-1`` lowers it)
-----------------------------------------------------------------------
PASS (weights are PRIORS ordered by causal importance, high to low)

==============================  ====  ======  ===========================================
key                             dir   weight  league reference (feed, seasons, n)
==============================  ====  ======  ===========================================
wr_deep_route_rate (rate)       +1    1.5     0.168 sd 0.073: share of a WR's targets with
                                              air_yards >= 20; pbp 2019-2025, WR-seasons
                                              >= 50 targets, n = 589 (per *target*: routes
                                              are not in a free feed)
qb_deep_attempt_rate (rate)     +1    1.2     0.116 sd 0.027: share of attempts with
                                              air_yards >= 20; pbp 2019-2025, QB-seasons
                                              >= 200 att, n = 251
opp_explosive_play_prevention   -1    1.2     50 sd 28.9: team percentile grade (100 = best
 (grade 0-100)                                at preventing 20+ yd plays); uniform-percentile
                                              reference by construction (measured sd of the
                                              32-team grade 28.9)
opp_coverage_bust_rate (rate)   +1    0.9     0.04 sd 0.015: PRIOR, no free feed
wr_separation (yards; alias     +1    0.8     2.90 sd 0.50: NGS receiving avg_separation,
 wr1_separation)                              2019-2025 REG, WRs >= 50 targets, n = 590
opp_missed_tackle_rate (rate)   +1    0.7     0.0852 sd 0.0144: PFR def, team sum
                                              m_tkl / (comb + m_tkl); 2023-2025 team-seasons,
                                              n = 96 (PFR's count shifted in 2022, 0.062
                                              vs ~0.09 in other seasons, so older seasons
                                              are not pooled)
==============================  ====  ======  ===========================================

Order of the pass weights: exposure first (how often the receiver / quarterback go
deep decides whether a 30-yard catch is on the menu at all), then the defence's
tendency to give up chunk plays, then a bust (the single most direct cause of a
free deep ball, but rare and unmeasured, hence below the exposure inputs),
then separation (measured over *all* targets, most of them short, so it speaks
more to completion than to length) and finally tackling (YAC conversion only).

RUN

==============================  ====  ======  ===========================================
rb_breakaway_rate (rate)        +1    1.4     0.0445 sd 0.0195: share of carries >= 15 yd;
                                              pbp 2019-2025, RB-seasons >= 100 carries,
                                              n = 328 (the sd includes sampling noise)
ol_run_block_win_rate (rate)    +1    1.2     0.72 sd 0.04: PRIOR (an ESPN-style run block
                                              win rate; no free feed, see limits)
opp_missed_tackle_rate (rate)   +1    0.9     as above
opp_explosive_play_prevention   -1    0.8     as above (all-play proxy, pass-dominated, so
                                              it is the least specific run input)
==============================  ====  ======  ===========================================

Order of the run weights: the runner's own breakaway trait, then the blocking that
creates the crease, then tackling (a missed tackle turns a 5-yard run into a 35),
then a general defensive-prevention grade.

GAIN (logit points per unit of ``z``)
-------------------------------------
``GAIN`` is a calibration constant, not a weight.  It was chosen so that all inputs
one SD in the "explosive" direction give a clearly better but believable game and
two SD stay far from certainty, then compared with the out-of-sample slope below.
The pass slope is the lower of the two (+0.29, 95% CI +0.14..+0.44) and
``GAIN_PASS`` = 0.45 sits just above that interval, a deliberate compromise with the
0.60 a pure plausibility argument would pick; the run slope (+0.46, CI +0.29..+0.63)
contains ``GAIN_RUN`` = 0.50.  If anything the pass probabilities are a touch
over-dispersed against the data; the adjustment they drive is small either way.
``z`` is the weight-normalised mean z, so "all inputs +k SD in the explosive
direction" means ``z = k`` (each input is clipped at +/-3 SD, so ``z`` cannot pass 3)::

    weighted z        -1       0 (reference)    +1       +2       +3 (the clip)
    pass, GAIN 0.45   0.193    0.273            0.370    0.480    0.591
    run,  GAIN 0.50   0.081    0.126            0.192    0.282    0.393

Both are nowhere near 0.95, and +1 SD on everything moves ``wr_yards`` by +1.8 yd and
``rb_yards`` by +0.8 yd under the re-centred formula.  The z = 3 column is a clip,
not a reachable state: the opponent grade lives on 0-100, so it can only be 1.73 SD
from its mean.  "+2 SD on everything" therefore realises ``z`` = 1.95 (pass 0.474,
run 0.277), and the model's absolute ceiling (every other input at the +3 SD clip,
the grade at 0) is ``z`` = 2.76 (pass 0.565, run 0.365).

Honest limits
-------------
* **Weights are priors, not fitted.**  No joint outcome data exist offline to fit
  them; the ordering above is causal reasoning written before the checks below were
  run, and it was *not* changed in response to them.  Only ``GAIN`` and the
  references are touched by data.
* **Player-game check of the unfit composite** (nflverse 2020-2025 qualifying
  player-games; every input measured in the *previous* season so nothing leaks; the
  model run exactly as shipped; opponent missed-tackle rate re-centred per season;
  the OL input is a z-matched yards-before-contact proxy because no free run-block
  win rate exists):

  - pass, n = 7,217: AUC 0.527; the observed event rate rises 0.267 -> 0.305 from
    the lowest to the highest quintile of ``z`` (the shipped model predicts 0.231 ->
    0.316).  Calibration slope of event logit on ``z`` is +0.29 (95% CI
    +0.14..+0.44) against ``GAIN_PASS`` 0.45.  Standardised coefficients (complete
    cases, n = 3,787): ``wr_deep_route_rate`` +0.12 (se 0.04); QB deep rate -0.01,
    prevention -0.05, separation -0.01, missed tackles -0.05 -- all
    indistinguishable from 0.
  - run, n = 3,995: AUC 0.570; observed 0.080 -> 0.167 across quintiles (shipped
    model 0.094 -> 0.178).  Slope +0.46 (CI +0.29..+0.63) against ``GAIN_RUN`` 0.50.
    Standardised coefficients (n = 2,906): ``rb_breakaway_rate`` +0.23 (se 0.06), OL
    proxy +0.10 (se 0.06), missed tackles 0.00, prevention -0.04.

  Read plainly: the model separates running-back games modestly and wide-receiver
  games barely; on the pass side only the receiver's own deep-target share shows a
  signal, and an AUC of 0.53 is close to a coin flip.  A single-game 30-yard play is
  mostly luck.  Use ``p`` as a small nudge, which is what the re-centred
  ``(p - ref) * amount`` makes it.  The pass slope's CI sits below the 0.60 a pure
  plausibility argument would pick and just under ``GAIN_PASS`` itself; 0.45 is a
  compromise, not a fit.  (The check's inputs are prior-season measurements of noisy
  small-sample rates, so classical measurement error pulls its slope toward zero;
  how much is not measured here.)  On 2024-2025 alone the pass slope is +0.25 (CI
  -0.02..+0.51, AUC 0.525, n = 2,331), indistinguishable from 0, and the run slope
  is +0.42 (CI +0.12..+0.71, AUC 0.559, n = 1,337).
* **Team-season check** (Spearman, 2019-2025, prior-season feature vs this-season
  outcome; a team-season rests on ~37 WR-games / ~20 RB-games so binomial SE is
  0.07-0.10 and a modest real effect looks small): WR deep-target share +0.14,
  QB deep-attempt rate +0.14, WR separation -0.04 (+0.11 against 30+ yards per
  target), missed-tackle +0.00, prevention -0.08.  Run: breakaway +0.25,
  yards-before-contact +0.23, missed-tackle +0.08, prevention -0.04.  Defences'
  explosive-play rate allowed barely persists (rho +0.14), so prevention and
  tackling are the least trustworthy inputs on both sides.
* **Redundancy:** ``wr_deep_route_rate`` and ``qb_deep_attempt_rate`` are near
  duplicates at team level (r = 0.87, n = 224 team-seasons), so supplying both
  counts "this offence throws deep" twice.  Both are kept because the brief names
  both.  (Pearson 0.87, Spearman 0.86.)
* **No free feed:** ``opp_coverage_bust_rate`` (charting) and
  ``ol_run_block_win_rate`` (tracking) are accepted and *neutral until supplied*;
  their references are priors only for interpreting a caller's number.  A proxy
  (yards before contact, deep completions over expected) must be converted to that
  scale by the caller; feeding it a percentile or a raw yardage will mislead.
  ``opp_missed_tackle_rate`` should be supplied relative to the same season's league
  mean (PFR's missed-tackle count is not stable across seasons).
* Observed rates (breakaway rate especially) are noisy small-sample numbers; the
  loader should shrink them before they get here.
* ``p`` is for a *qualifying* player-game (>= 4 targets / >= 8 carries).  It says
  nothing about a WR3 with two targets.  TE and receiving RBs are not adjusted
  (the brief names ``wr_yards`` and ``rb_yards`` only).

Applying it
-----------
``apply_explosive_play`` uses ``nfl_pack_common.add_to_field``.  With
``recenter=True`` (default) the addition is ``(p - reference) * amount``, so an
average player-game is an exact no-op and only the deviation moves the line;
``recenter=False`` is the literal ``+ p * amount``.  Fields never go below 0.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from modules.nfl_pack_common import (
    PACK_VERSION,
    PROJECTION_FIELDS,
    Factor,
    add_to_field,
    clamp,
    describe_drivers,
    logistic_probability,
    r,
    score_factors,
    to_float,
    to_rate,
)

LOG = logging.getLogger("nfl_pack.explosive_play")

MODEL = "explosive_play"
STAGE = "explosive_play"

#: Output keys.
PASS_KEY = "explosive_play_prob_pass"
RUN_KEY = "explosive_play_prob_run"

#: The brief's coefficients, verbatim: ``wr_yards += p_pass * 18`` / ``rb_yards += p_run * 12``.
WR_YARDS_COEFFICIENT = 18.0
RB_YARDS_COEFFICIENT = 12.0

# --- events and base rates -------------------------------------------------------
EVENT_PASS = "WR with >= 4 targets has >= 1 reception of 30+ yards in the game"
EVENT_RUN = "RB with >= 8 carries has >= 1 carry of 30+ yards in the game"
#: nflverse pbp, REG season 2019-2025: 2,286 events in 8,383 WR-games; 583 in 4,617 RB-games.
BASE_RATE_PASS = 0.2727
BASE_RATE_RUN = 0.1263
BASE_RATE_N = {PASS_KEY: 8383, RUN_KEY: 4617}
#: Logit points per unit of weighted z (see the module docstring for the resulting numbers).
GAIN_PASS = 0.45
GAIN_RUN = 0.50

#: Public per-output constants (a dict because the model has two outputs).
BASE_RATE: dict[str, float] = {PASS_KEY: BASE_RATE_PASS, RUN_KEY: BASE_RATE_RUN}
GAIN: dict[str, float] = {PASS_KEY: GAIN_PASS, RUN_KEY: GAIN_RUN}

# --- factors ---------------------------------------------------------------------
_PREVENTION = dict(
    mean=50.0, sd=28.9, direction=-1, label="Opponent explosive-play prevention", kind="value", unit="pctile",
    source="team percentile grade (100 = best); uniform-percentile reference, measured sd of the 32-team grade 28.9",
)
_MISSED_TACKLE = dict(
    mean=0.0931, sd=0.0144, direction=+1, label="Opponent missed-tackle rate", kind="rate", unit="%",
    source="nfl_team_features loader (PFR def missed tackles) REG 2023-2025 team-seasons n=96 mean; sd = raw team-season sd",
)

#: Pass factors, ordered by prior causal importance (weights are priors, NOT fitted).
PASS_FACTORS: tuple[Factor, ...] = (
    Factor(
        "wr_deep_route_rate", mean=0.168, sd=0.073, weight=1.5, direction=+1, label="WR deep-target share",
        kind="rate", unit="%",
        source="nflverse pbp 2019-2025 REG, WR-seasons >=50 targets, n=589; share of targets with air_yards>=20",
    ),
    Factor(
        "qb_deep_attempt_rate", mean=0.116, sd=0.027, weight=1.2, direction=+1, label="QB deep-attempt rate",
        kind="rate", unit="%",
        source="nflverse pbp 2019-2025 REG, QB-seasons >=200 att, n=251; share of attempts with air_yards>=20",
    ),
    Factor("opp_explosive_play_prevention", weight=1.2, **_PREVENTION),
    Factor(
        "opp_coverage_bust_rate", mean=0.04, sd=0.015, weight=0.9, direction=+1, label="Opponent coverage-bust rate",
        kind="rate", unit="%", source="prior (no free feed: charting data)",
    ),
    Factor(
        "wr_separation", mean=3.05, sd=0.40, weight=0.8, direction=+1, label="WR separation", unit="yd",
        aliases=("wr1_separation",),
        source="NGS receiving avg_separation 2019-2025 REG, WRs >=50 targets, n=590",
    ),
    Factor("opp_missed_tackle_rate", weight=0.7, **_MISSED_TACKLE),
)

#: Run factors, ordered by prior causal importance (weights are priors, NOT fitted).
RUN_FACTORS: tuple[Factor, ...] = (
    Factor(
        "rb_breakaway_rate", mean=0.0445, sd=0.0195, weight=1.4, direction=+1, label="RB breakaway rate",
        kind="rate", unit="%",
        source="nflverse pbp 2019-2025 REG, RB-seasons >=100 carries, n=328; share of carries >=15 yd",
    ),
    Factor(
        "ol_run_block_win_rate", mean=0.70, sd=0.04, weight=1.2, direction=+1, label="OL run-block win rate",
        kind="rate", unit="%", source="prior (no free feed: tracking data; ESPN-style run block win rate scale)",
    ),
    Factor("opp_missed_tackle_rate", weight=0.9, **_MISSED_TACKLE),
    Factor("opp_explosive_play_prevention", weight=0.8, **_PREVENTION),
)


def _union(*groups: tuple[Factor, ...]) -> tuple[Factor, ...]:
    seen: dict[str, Factor] = {}
    for group in groups:
        for factor in group:
            seen.setdefault(factor.key, factor)
    return tuple(seen.values())


#: Every canonical input of the model, once each (the pass factor stands in for the
#: two shared keys; use ``PASS_FACTORS`` / ``RUN_FACTORS`` for the weights actually applied).
FACTORS: tuple[Factor, ...] = _union(PASS_FACTORS, RUN_FACTORS)

#: Plausible domain of a *value* input; anything outside is a data error, treated as missing
#: (the same stance ``to_rate`` takes for a rate outside 0-100).
_VALID_VALUE = {
    "opp_explosive_play_prevention": lambda v: 0.0 <= v <= 100.0,
    "wr_separation": lambda v: 0.0 < v <= 20.0,
}


def _clean_inputs(inputs: Any) -> dict[str, float]:
    """``{key_used: number}`` for every usable input; bad, non-finite or out-of-domain values dropped."""
    clean: dict[str, float] = {}
    if not isinstance(inputs, Mapping) or not inputs:
        return clean
    for factor in FACTORS:
        valid = _VALID_VALUE.get(factor.key)
        for key in (factor.key, *factor.aliases):
            if key not in inputs:
                continue
            value = to_rate(inputs[key]) if factor.kind == "rate" else to_float(inputs[key])
            if value is None or (valid is not None and not valid(value)):
                continue
            clean[key] = value
            break
    return clean


def _score_output(clean: Mapping[str, float], factors: tuple[Factor, ...], key: str) -> dict[str, Any]:
    scored = score_factors(clean, factors)
    base = BASE_RATE[key]
    probability = clamp(logistic_probability(base, scored["z"], GAIN[key]), 0.0, 1.0)
    weights = {f.key: f.weight for f in factors}
    return {
        "probability": r(probability),
        "reference": r(base),
        "z_score": r(scored["z"], 4),
        "coverage": r(scored["coverage"], 3),
        "inputs_used": list(scored["used"]),
        "inputs_defaulted": list(scored["defaulted"]),
        "contributions": scored["contributions"],
        "drivers": describe_drivers(scored["contributions"]),
        "weights": weights,
        "gain": GAIN[key],
        "event": EVENT_PASS if key == PASS_KEY else EVENT_RUN,
        "base_rate_n": BASE_RATE_N[key],
    }


_BASIS = (
    "Two logistic factor models (pass, run) over nflverse-derived league references; weights are PRIORS ordered by "
    "causal reasoning, not fitted. Base rates measured on nflverse pbp 2019-2025 REG: P(WR with >=4 targets has a "
    "30+ yd reception)=0.2727 (n=8,383 WR-games); P(RB with >=8 carries has a 30+ yd carry)=0.1263 (n=4,617 RB-games). "
    "p is a per-player-game probability; the brief's '+p*18 / +p*12 yards' is applied re-centered so an average "
    "player-game is unchanged. opp_coverage_bust_rate and ol_run_block_win_rate have no free feed (neutral until "
    "supplied); separation, deep rates, prevention and missed-tackle references are empirical. "
    "Associations with real outcomes are weak (see module docstring)."
)


def compute_explosive_play(inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Probability that a qualifying WR / RB has a 30+ yard play, from the brief's eight inputs.

    ``inputs`` is a flat mapping of canonical keys (see the module docstring); any
    subset is fine, ``None`` / ``{}`` returns the references.  Bad values (``None``,
    ``NaN``, text, booleans, out-of-range grades or separation) are treated as
    missing; a rate may be a fraction or a percentage.  Never raises.
    """
    clean = _clean_inputs(inputs)
    outputs = {
        PASS_KEY: _score_output(clean, PASS_FACTORS, PASS_KEY),
        RUN_KEY: _score_output(clean, RUN_FACTORS, RUN_KEY),
    }
    p_out, r_out = outputs[PASS_KEY], outputs[RUN_KEY]

    used = [f.key for f in FACTORS if f.key in p_out["inputs_used"] or f.key in r_out["inputs_used"]]
    defaulted = [f.key for f in FACTORS if f.key not in used]
    contributions: dict[str, dict[str, Any]] = {}
    for key in used:
        entry: dict[str, Any] | None = None
        by_output: dict[str, dict[str, float]] = {}
        for out_key, out in outputs.items():
            item = out["contributions"].get(key)
            if item is None:
                continue
            entry = entry if entry is not None else dict(item)
            by_output[out_key] = {"weight": item["weight"], "effect": item["effect"]}
        if entry is not None:
            entry["outputs"] = by_output
            contributions[key] = entry
    total_weight = sum(f.weight for f in PASS_FACTORS) + sum(f.weight for f in RUN_FACTORS)
    used_weight = sum(
        f.weight for f in PASS_FACTORS if f.key in p_out["inputs_used"]
    ) + sum(f.weight for f in RUN_FACTORS if f.key in r_out["inputs_used"])
    drivers = [f"pass: {line}" for line in p_out["drivers"]] + [f"run: {line}" for line in r_out["drivers"]]

    result: dict[str, Any] = {
        "model": MODEL,
        "pack_version": PACK_VERSION,
        PASS_KEY: p_out["probability"],
        RUN_KEY: r_out["probability"],
        "probability": p_out["probability"],
        "reference_probability": p_out["reference"],
        "references": {PASS_KEY: p_out["reference"], RUN_KEY: r_out["reference"]},
        "z_score": p_out["z_score"],
        "z_scores": {PASS_KEY: p_out["z_score"], RUN_KEY: r_out["z_score"]},
        "coverage": r(used_weight / total_weight, 3),
        "coverages": {PASS_KEY: p_out["coverage"], RUN_KEY: r_out["coverage"]},
        "inputs_used": used,
        "inputs_defaulted": defaulted,
        "contributions": contributions,
        "drivers": drivers,
        "outputs": outputs,
        "basis": _BASIS,
    }
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "%s: %s=%.4f (ref %.4f, z=%+.3f, cov=%.0f%%) %s=%.4f (ref %.4f, z=%+.3f, cov=%.0f%%) used=%d defaulted=%d drivers=%s",
            MODEL,
            PASS_KEY, p_out["probability"], p_out["reference"], p_out["z_score"], 100.0 * p_out["coverage"],
            RUN_KEY, r_out["probability"], r_out["reference"], r_out["z_score"], 100.0 * r_out["coverage"],
            len(used), len(defaulted), "; ".join(drivers),
        )
    return result


# ---------------------------------------------------------------------------
# Applying the result to a projection
# ---------------------------------------------------------------------------
def _probability(result: Mapping[str, Any], key: str) -> float | None:
    value = to_float(result.get(key))
    if value is None and key == PASS_KEY:
        value = to_float(result.get("probability"))
    return None if value is None else clamp(value, 0.0, 1.0)


def _reference(result: Mapping[str, Any], key: str) -> float:
    references = result.get("references")
    value = to_float(references.get(key)) if isinstance(references, Mapping) else None
    if value is None and key == PASS_KEY:
        value = to_float(result.get("reference_probability"))
    if value is None:
        value = BASE_RATE[key]
    return clamp(value, 0.0, 1.0)


def apply_explosive_play(
    projection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    recenter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Add the brief's explosive-play yards to ``wr_yards`` and ``rb_yards``.

    ``wr_yards += explosive_play_prob_pass * 18`` and
    ``rb_yards += explosive_play_prob_run * 12``.  With ``recenter=True`` (default) each
    addition is ``(p - reference) * amount`` so an average player-game is an exact
    no-op; ``recenter=False`` is the literal brief.  Returns ``(new_projection,
    adjustments)``; ``projection`` is not mutated and a field that is missing or not
    numeric is recorded as ``applied=False``.  Yards never go below 0.
    """
    adjusted: dict[str, Any] = dict(projection) if isinstance(projection, Mapping) else {}
    adjustments: list[dict[str, Any]] = []
    source: Mapping[str, Any] = result if isinstance(result, Mapping) else {}
    for key, field, amount in (
        (PASS_KEY, "wr_yards", WR_YARDS_COEFFICIENT),
        (RUN_KEY, "rb_yards", RB_YARDS_COEFFICIENT),
    ):
        if field not in PROJECTION_FIELDS:  # defensive: the vocabulary is shared with the pipeline
            continue
        probability = _probability(source, key)
        if probability is None:
            adjustments.append(
                {"stage": STAGE, "field": field, "applied": False, "reason": f"result has no usable {key}"}
            )
            continue
        add_to_field(
            adjusted,
            field,
            stage=STAGE,
            probability=probability,
            amount=amount,
            reference=_reference(source, key),
            recenter=recenter,
            adjustments=adjustments,
        )
    return adjusted, adjustments


__all__ = [
    "BASE_RATE",
    "BASE_RATE_N",
    "BASE_RATE_PASS",
    "BASE_RATE_RUN",
    "EVENT_PASS",
    "EVENT_RUN",
    "FACTORS",
    "GAIN",
    "GAIN_PASS",
    "GAIN_RUN",
    "LOG",
    "PASS_FACTORS",
    "PASS_KEY",
    "RB_YARDS_COEFFICIENT",
    "RUN_FACTORS",
    "RUN_KEY",
    "WR_YARDS_COEFFICIENT",
    "apply_explosive_play",
    "compute_explosive_play",
]
