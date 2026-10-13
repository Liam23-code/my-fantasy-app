"""Game-script model: how much of an offense's game is played pass-heavy, run-heavy or neutral.

The brief
---------
Inputs: Vegas spread; Vegas total; team EPA; opponent EPA; pace; pass rate over
expectation (PROE); run rate over expectation (RROE).  Outputs:
``positive_script_prob``, ``negative_script_prob``, ``neutral_script_prob``.  Applied as::

    qb_attempts *= (1 + positive_script_prob * 0.20 - negative_script_prob * 0.25)
    rb_attempts *= (1 + negative_script_prob * 0.30 - positive_script_prob * 0.20)

Semantics (contract section 5 -- decided, not relitigated)
----------------------------------------------------------
The two formulas are only football-correct if **positive script = pass-heavy**
(the offense is *trailing* by 7+ at the snap: QB attempts up, RB attempts down) and
**negative script = run-heavy** (the offense is *leading* by 7+: QB attempts down,
RB attempts up).  Neutral is ``|score differential| < 7``.  To make that impossible
to misread the result carries both vocabularies: ``positive_script_prob`` /
``negative_script_prob`` (the brief's names) and the unambiguous aliases
``trailing_script_prob`` (= pass-heavy) / ``leading_script_prob`` (= run-heavy).
``POSITIVE_SCRIPT_IS_PASS_HEAVY`` (default ``True``) flips which of the two physical
states feeds "positive" and "negative" for a user who meant the opposite; the
aliases never move.

EVENT DEFINITION (what ``0.2968`` means)
----------------------------------------
An *offensive snap* is a scrimmage pass or run (nflverse ``play_type`` in
``{pass, run}``: sacks and scrambles included; kneels, spikes, punts, kicks and
plays nullified by penalty excluded), regular season.  Its *state* is set by
``score_differential`` (offense minus defense) at the snap: ``<= -7`` pass-heavy
(trailing), ``>= +7`` run-heavy (leading), otherwise neutral.  The three
probabilities are **play-share probabilities**: the probability that a given
offensive snap of this team in this game is taken in that state, i.e. the share
of the team's snaps that will be played there.  They sum to 1 (to within 1e-6, in
fact exactly after the 4-decimal rounding).  ``BASE_RATE = 0.2968`` is the
trailing share for a pick'em at an average total.  Measured on nflverse pbp
2018-2025 regular season: 4,254 team-games (2,127 games), 264,889 offensive
snaps; league-wide mean shares trailing 0.3086 / neutral 0.4883 / leading 0.2031
(the mean exceeds the pick'em value because shares are convex in the spread).

How the numbers are produced (this model is empirical, not a prior)
-------------------------------------------------------------------
For every team-game the share of that team's snaps in each state was counted and a
3-class **ordered logit** was fitted to the offense's pre-game spread and total
(weighted by snaps, i.e. a play-level multinomial likelihood)::

    x      = GAIN * spread / 10                    # spread: sportsbook sign, + = underdog
    s      = exp(TOTAL_VOLATILITY * (total - 45.28) / 10)
    P(trailing) = sigmoid((THRESHOLD_TRAILING + x) / s_t)
    P(leading)  = 1 - sigmoid((THRESHOLD_LEADING  + x) / s_l)
    P(neutral)  = 1 - P(trailing) - P(leading)
    s_t = s if THRESHOLD_TRAILING + x < 0 else 1       # monotone-in-total guard
    s_l = s if THRESHOLD_LEADING  + x > 0 else 1

Shipped parameters (fit on all of 2018-2025, n = 4,254 team-games / 264,889 snaps):
``GAIN = 0.904`` (logit units per 10 points of expected margin),
``THRESHOLD_TRAILING = -0.8627``, ``THRESHOLD_LEADING = +1.4681``,
``TOTAL_VOLATILITY = 0.0911`` (a higher total scales the latent noise up: both
tails fatten and neutral shrinks).

The guard (``s_t`` / ``s_l``) exists because the contract (section 2) requires each
input to be monotone in its stated direction.  A bare scale parameter would make a
higher total *lower* a heavy underdog's (9.5+ points) trailing share and a heavy
favourite's (16+ points) leading share -- a quirk of the functional form, worth at
most 1.2 share points and well inside the noise.  Holding the scale at 1 on a tail's
majority side removes it: all-season log-loss 0.99201 -> 0.99202 per snap, 2025
holdout 0.97098 -> 0.97101, 304 of 4,254 team-games change.  Refitting under the
guard moves every parameter by less than 0.001, so the unconstrained maximum-
likelihood values above are shipped unchanged.

Two facts the brief's sanity list did not anticipate, reported rather than hidden:

* **A 0-point spread is NOT symmetric**: trailing 0.297, neutral 0.516, leading
  0.187 (realised in the (-1.5, +1.5] spread bucket: 0.302 / 0.516 / 0.182).  The
  asymmetry is real, not a modelling artefact: in decided game time (|diff| >= 7)
  the trailing team takes 60.3% of the snaps (and holds the ball 56.2% of the
  clock), snapping every 29.2 game-clock seconds against 32.1 for the leader.  The
  fitted thresholds are therefore not mirror images.  The equal-share point is a
  3-point favourite.  Direction is as sanity-checked: a -10 favourite is
  leading 0.363 > trailing 0.146; a +10 dog is trailing 0.510 > leading 0.085.
* The brief's coefficients are bigger than the *pass-rate* effects in the data
  (pass rate vs neutral: trailing +17%, leading -14%; run rate: trailing -22%,
  leading +18%) -- attempts also move with snap volume, and the coefficients are the
  user's, applied verbatim.  Nothing here re-tunes them.

Out-of-sample result (fit 2018-2024: 3,710 team-games / 231,948 snaps; scored on the
held-out 2025 season: 544 team-games / 32,941 snaps), log-loss per snap and mean
absolute error of the team-game share (trailing / neutral / leading):

    constant league shares             1.0312   0.298 / 0.248 / 0.236
    spread-only ordered logit          0.9725   0.269 / 0.245 / 0.212
    spread + total (shipped form)      0.9710   0.268 / 0.244 / 0.212

(The per-team-game MAE is large because a game is only ~62 snaps; the calibration
by spread bucket is tight: shipped model vs realised, snap-weighted, in nine spread
buckets over 2018-2025 is within 2.4 share points everywhere -- worst case the leading
share of 3.5-6.5 point underdogs, 0.125 modelled vs 0.101 realised -- and the
league-wide shares come out 0.308 / 0.489 / 0.204 against 0.309 / 0.488 / 0.203.)
Leave-one-season-out over all eight seasons (guarded form) agrees: the total improves
log-loss in 7 of 8 held-out seasons (mean -0.0006; 2018 is the exception, +0.0008).

EPA differential, pace, PROE and RROE -- accepted, audited, and (honestly) inert
--------------------------------------------------------------------------------
The orchestrator's rule was: add them as small bounded adjustments *only if they
help out-of-sample*.  They do not, once the spread is known (the market spread
already prices the matchup).  Protocol: spread+total parameters fixed, each
coefficient fitted on 2018-2024, scored on 2025 (and again leave-one-season-out):

    candidate (per snap log-loss change)     2025 holdout   LOSO mean (folds improved)
    EPA differential, latent shift             +0.0007          -0.0003  (5/8)
    pace, latent shift                         +0.0001          +0.0001  (4/8)
    pace, volatility scale                     -0.0002          -0.0001  (7/8)
    PROE, latent shift                         -0.0001          +0.0000  (5/8)

Letting EPA differential into the *same* fit lowers the in-sample loss (0.9951 ->
0.9933) but makes 2025 worse (0.9710 -> 0.9760): it trades the spread's weight
away.  Effect sizes after the spread (play-weighted, game-clustered): EPA
differential +1 SD = -1.2 / +0.0 / +1.2 share points (trailing / neutral /
leading, t about 2); pace +1 SD <= 0.5 points (t about 1); PROE +1 SD <= 0.4 points
(t < 1).  So with a spread in hand these four inputs carry **no weight**
(``ADJUSTMENT_POINTS_PER_Z = 0.0``); the machinery to give them a bounded effect is
in place, with prior (unfitted) weights ordered by causal importance, and is a
one-constant change if a later refit shows lift.  They are still *accepted*: they
appear in ``inputs_used``, in ``contributions`` (``role="inert"``, ``effect=0``)
and in ``inputs_inert``.  RROE is the mirror of PROE on the same snaps
(``rroe == -proe``); they are two prior factors with opposite directions so that
supplying both does not double count by construction of the weights.

One input *does* matter when the spread is missing: **team EPA and opponent EPA
stand in for it**.  The pre-game EPA/play differential correlates 0.72 with the
market's expected margin (n = 4,254), slope 39.8 points of margin per 1.0 EPA/play
of differential (``MARGIN_PER_EPA_DIFF``).  Used as the margin it beats league
shares out of sample (2025: log-loss 1.0098 vs 1.0312 constant vs 0.9710 with the
real spread; Spearman with realised trailing share 0.22 vs 0.38) but is clearly
weaker than a real line, and ``margin_source`` says so.  A real spread always wins.

GAIN
----
``GAIN = 0.904`` is the fitted slope, not a free choice.  Consequences: an
underdog by +1 SD of the league spread (6.4 pts) is trailing for 0.43 of its snaps;
+2 SD (12.9 pts) 0.575; the largest input the model accepts (the spread is clipped
at +/-3 SD = 19.3 pts) 0.707 -- comfortably below 0.95, because the data say that
even a heavy dog spends a large share of its snaps within one score or ahead (all
10+ point underdogs, 2018-2025: trailing 0.577 of snaps, i.e. 42% not trailing).

Honest limits
-------------
* Weights for the inert adjustments are priors; the spread/total mapping is fitted
  to *closing* lines (nflverse ``spread_line`` / ``total_line``) -- an opening or
  stale line will be slightly less informative than the fit assumes.
* It is a *share* model.  A game's realised mix is very noisy (about 62 snaps; the
  per-game Spearman between spread and trailing share is only +0.34) and the model
  says nothing about *when* in the game the script happens, or about in-game
  injuries and weather.
* It is conditional on the spread.  It does not know a team's own pace or
  style beyond what the line already contains; PROE/pace/EPA had no demonstrable
  lift (see above) -- a statement about 2018-2025 NFL data, not a law.
* Pace is measured here as game-clock seconds between consecutive snaps in neutral
  situations (win probability 20-80%, first three quarters, outside the last two
  minutes of a half); clock stoppages make it understate true cadence.  Mean 32.2 s,
  SD 1.34 s (team-seasons).  Callers must supply the same definition or treat it as
  neutral.  PROE is nflfastR's ``pass_oe`` (percentage points / 100).
* No input here lacks a free feed: spread/total from nflverse schedules (or the
  user's own odds file -- never a live sportsbook, ``offline_data_contract.md``
  rule 1) and EPA/pace/PROE from nflverse pbp.  Missing inputs are neutral.
* The apply step moves ``qb_attempts`` and ``rb_attempts`` only; yards and touchdown
  probabilities are untouched (the brief gives no coefficient for them).

Inputs (canonical keys, contract section 4)
-------------------------------------------
``vegas_spread`` (points, sportsbook sign: -7 = this team favoured by 7; also
``expected_margin`` = -vegas_spread), ``vegas_total`` (points), ``team_epa_per_play``,
``opp_epa_per_play`` (offensive EPA/play of each side), ``pace_seconds_per_play``
(lower = faster), ``proe`` and ``rroe`` (fractions, +0.05 = +5 points; values with
``|x| > 1`` are read as percentage points).  nflverse's ``spread_line`` is positive
when the HOME team is favoured (verified: corr(spread_line, home - away) = +0.45,
slope 1.04), so a HOME team's ``vegas_spread`` is ``-spread_line`` and an away
team's is ``+spread_line`` -- see :func:`team_spread_from_schedule`.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Mapping

from modules.nfl_pack_common import (
    Factor,
    Z_CLIP,
    clamp,
    r,
    score_factors,
    sigmoid,
    to_float,
)

LOG = logging.getLogger("nfl_pack.game_script")

MODEL = "game_script"

#: The three outputs, in the brief's order (they always sum to 1).
OUTPUT_KEYS: tuple[str, ...] = ("positive_script_prob", "negative_script_prob", "neutral_script_prob")

#: What BASE_RATE (and every probability this model returns) means.
EVENT_DEFINITION = (
    "Share of a team's offensive snaps (nflverse play_type pass or run, regular season; sacks and scrambles "
    "included, kneels/spikes/penalty-nullified plays excluded) taken while the team trails by 7+ points "
    "(pass-heavy state), leads by 7+ (run-heavy state) or is within 6 (neutral). BASE_RATE = 0.2968 is the "
    "trailing share for a pick'em at an average total; league-wide mean shares on 2018-2025 are trailing 0.3086 / "
    "neutral 0.4883 / leading 0.2031 (4,254 team-games, 264,889 snaps)."
)

# ---------------------------------------------------------------------------
# Semantics switch
# ---------------------------------------------------------------------------
#: True (default): positive script = pass-heavy = offense trailing by 7+;
#: negative script = run-heavy = offense leading by 7+.  False swaps the two so the
#: brief's formulas ride the opposite physical state.  The ``trailing_*`` /
#: ``leading_*`` aliases never move.
POSITIVE_SCRIPT_IS_PASS_HEAVY = True

#: A script state begins one possession (7 points) from level.
SCRIPT_THRESHOLD_POINTS = 7

# ---------------------------------------------------------------------------
# The brief's coefficients, verbatim (named after the field they scale)
# ---------------------------------------------------------------------------
QB_ATTEMPTS_POSITIVE_COEFFICIENT = 0.20   # qb_attempts *= 1 + pos*0.20 - neg*0.25
QB_ATTEMPTS_NEGATIVE_COEFFICIENT = 0.25
RB_ATTEMPTS_NEGATIVE_COEFFICIENT = 0.30   # rb_attempts *= 1 + neg*0.30 - pos*0.20
RB_ATTEMPTS_POSITIVE_COEFFICIENT = 0.20

# ---------------------------------------------------------------------------
# Calibrated mapping: nflverse pbp 2018-2025 regular season, 4,254 team-games,
# 264,889 offensive snaps (see module docstring for the protocol and holdout).
# ---------------------------------------------------------------------------
#: Ordered-logit slope: logit units of "trailing lean" per 10 points of spread.
GAIN = 0.904
#: Latent cut-point between pass-heavy (trailing >= 7) and neutral, at spread 0.
THRESHOLD_TRAILING = -0.8627
#: Latent cut-point between neutral and run-heavy (leading >= 7), at spread 0.
THRESHOLD_LEADING = 1.4681
#: Latent-noise scale per 10 points of total above the league mean (exp scale).
TOTAL_VOLATILITY = 0.0911
#: Points of expected margin per 1.0 EPA/play of (team - opponent) offensive-EPA
#: differential: through-origin OLS on the market margin, 2018-2025, n = 4,254,
#: corr 0.72.  Used ONLY when no spread is supplied.
MARGIN_PER_EPA_DIFF = 39.8

# ---------------------------------------------------------------------------
# Optional bounded adjustments (inert at the shipped setting -- see docstring)
# ---------------------------------------------------------------------------
#: Spread-points of pass-heavy lean per unit of the adjustment factors' weighted z.
#: 0.0 because none of EPA differential / pace / PROE / RROE showed out-of-sample
#: lift beyond the spread + total (2025 holdout and leave-one-season-out).
ADJUSTMENT_POINTS_PER_Z = 0.0
#: Whatever the gain, the adjustments can never move the effective spread by more.
ADJUSTMENT_MAX_POINTS = 2.0

# ---------------------------------------------------------------------------
# FACTORS: league references from nflverse; weights are PRIORS, not fits.
#
# `direction` is stated relative to the PASS-HEAVY (trailing) share, independent of
# POSITIVE_SCRIPT_IS_PASS_HEAVY: +1 = a higher value raises trailing_script_prob.
# Ordering by causal importance for who trails and who leads:
#   1. vegas_spread  -- the market's expected margin: it IS the expected state.
#   2. vegas_total   -- scoring volatility: widens both tails (a volatility factor;
#                       a bigger total raises trailing AND leading and lowers neutral,
#                       and never lowers either tail -- see the guard in _shares()).
#   3./4. team_epa / opp_epa -- the two offenses' quality; already inside the
#                       spread, they only matter as its stand-in.
#   5./6. proe / rroe -- style tendencies (and exact mirrors of one another).
#   7. pace          -- cadence changes how many snaps fit in the clock, barely the
#                       state mix; weakest.
# ---------------------------------------------------------------------------
FACTORS: tuple[Factor, ...] = (
    # nflverse schedules 2018-2025 REG, n=4,254 team-games (each game once as favourite,
    # once as underdog, so mean is exactly 0): sd 6.44 points.
    Factor("vegas_spread", mean=0.0, sd=6.44, weight=6.0, direction=+1,
           label="Vegas spread (+ = underdog)", unit="pts",
           source="nflverse schedules 2018-2025 REG, n=4254 team-games"),
    # nflverse schedules 2018-2025 REG, n=2,127 games: mean 45.28, sd 4.51 points.
    Factor("vegas_total", mean=45.28, sd=4.51, weight=1.5, direction=+1,
           label="Vegas total", unit="pts",
           aliases=("game_total", "total_line", "over_under"),
           source="nflverse schedules 2018-2025 REG, n=2127 games"),
    # nflverse pbp 2018-2025 REG, pass+run snaps, team-seasons n=256: mean 0.0013, sd 0.0926.
    Factor("team_epa_per_play", mean=0.0013, sd=0.0926, weight=1.0, direction=-1,
           label="Team offensive EPA/play", unit="EPA/play",
           aliases=("team_epa", "off_epa_per_play"),
           source="nflverse pbp 2018-2025 REG, team-seasons n=256"),
    Factor("opp_epa_per_play", mean=0.0013, sd=0.0926, weight=1.0, direction=+1,
           label="Opponent offensive EPA/play", unit="EPA/play",
           aliases=("opp_epa",),
           source="nflverse pbp 2018-2025 REG, team-seasons n=256"),
    # nflverse pbp pass_oe/100, team-seasons n=256: mean -0.0148, sd 0.0436.  The negative
    # mean is nflfastR's xpass over-predicting passes on this play selection.
    Factor("proe", mean=-0.0148, sd=0.0436, weight=0.5, direction=+1,
           label="Pass rate over expected", unit="fraction",
           aliases=("pass_rate_over_expected", "pass_rate_over_expectation"),
           source="nflverse pbp pass_oe 2018-2025 REG, team-seasons n=256"),
    # Exact mirror of PROE on the same snaps: mean +0.0148, same sd, opposite direction.
    Factor("rroe", mean=0.0148, sd=0.0436, weight=0.5, direction=-1,
           label="Run rate over expected", unit="fraction",
           aliases=("run_rate_over_expected", "run_rate_over_expectation"),
           source="nflverse pbp pass_oe 2018-2025 REG, mirror of PROE, team-seasons n=256"),
    # Game-clock seconds between consecutive snaps of a drive, neutral situations; derived
    # from nflverse pbp 2018-2025 REG, 90,622 snap intervals, team-seasons n=256: mean 32.16,
    # sd 1.34.  Higher seconds = slower = a more conservative (run-leaning) style.
    Factor("pace_seconds_per_play", mean=32.16, sd=1.34, weight=0.4, direction=-1,
           label="Neutral-situation seconds per play", unit="s",
           aliases=("pace", "seconds_per_play"),
           source="nflverse pbp 2018-2025 REG, neutral-situation game-clock deltas, n=90622"),
)

#: The probability-bearing event the contract's BASE_RATE refers to: the share of
#: snaps taken trailing by 7+ at a pick'em spread and an average total.
BASE_RATE = r(sigmoid(THRESHOLD_TRAILING))

_FACTOR_BY_KEY = {f.key: f for f in FACTORS}
_SPREAD = _FACTOR_BY_KEY["vegas_spread"]
_TOTAL = _FACTOR_BY_KEY["vegas_total"]
_TEAM_EPA = _FACTOR_BY_KEY["team_epa_per_play"]
_OPP_EPA = _FACTOR_BY_KEY["opp_epa_per_play"]
_EPA_KEYS = (_TEAM_EPA.key, _OPP_EPA.key)
#: Factors that may nudge the effective spread when ADJUSTMENT_POINTS_PER_Z != 0.
ADJUSTMENT_FACTORS: tuple[Factor, ...] = tuple(
    _FACTOR_BY_KEY[k] for k in ("team_epa_per_play", "opp_epa_per_play", "proe", "rroe", "pace_seconds_per_play")
)
_PASS_LEAN_KEYS = ("proe", "rroe")
_POSITIVE_ONLY_KEYS = ("vegas_total", "pace_seconds_per_play")
#: The largest |spread| the mapping uses (3 SD; the data's max is 22 points).
SPREAD_CLIP = Z_CLIP * _SPREAD.sd

_TOTAL_WEIGHT = sum(f.weight for f in FACTORS)


# ---------------------------------------------------------------------------
# The calibrated mapping
# ---------------------------------------------------------------------------
def _shares(underdog_points: float, volatility_scale: float = 1.0) -> tuple[float, float, float]:
    """``(trailing, neutral, leading)`` snap shares, unrounded, summing to 1.

    ``underdog_points`` is the effective spread (positive = underdog / expected to
    trail); ``volatility_scale`` is ``exp(TOTAL_VOLATILITY * (total - mean) / 10)``.

    The volatility scale divides each latent cut-point distance.  Left unguarded it
    would *lower* a tail share once that tail is already the majority side (an
    underdog of 9.5+ points trailing more than half its snaps, a favourite of 16+
    points leading more than half), so a higher total would reduce a script
    probability there.  The contract requires every input to be monotone in its
    stated direction, so a tail whose cut-point is on its majority side ignores the
    total (scale 1).  Cost on the 2018-2025 fit: log-loss 0.99201 -> 0.99202 per
    snap; 304 of 4,254 team-games change, by at most 1.2 share points.
    """
    x = GAIN * underdog_points / 10.0
    a_trailing = THRESHOLD_TRAILING + x
    a_not_leading = THRESHOLD_LEADING + x
    # trailing = sigmoid(a/s) rises with s only while a < 0; leading = 1 - sigmoid(a/s) only while a > 0
    scale_trailing = volatility_scale if a_trailing < 0.0 else 1.0
    scale_leading = volatility_scale if a_not_leading > 0.0 else 1.0
    p_trailing = sigmoid(a_trailing / scale_trailing)
    p_not_leading = sigmoid(a_not_leading / scale_leading)
    p_leading = 1.0 - p_not_leading
    p_neutral = max(0.0, p_not_leading - p_trailing)
    return p_trailing, p_neutral, p_leading


def _rounded(trailing: float, leading: float) -> tuple[float, float, float]:
    """Round trailing/leading to 4 dp and make neutral the exact remainder (sum == 1)."""
    t = r(trailing)
    l = r(leading)
    n = r(max(0.0, 1.0 - t - l))
    return t, n, l


_REF_TRAILING, _REF_NEUTRAL, _REF_LEADING = _rounded(*(lambda s: (s[0], s[2]))(_shares(0.0, 1.0)))


def reference_shares() -> dict[str, float]:
    """The neutral (pick'em, average total) shares under the current semantics switch."""
    pos, neg = _assign(_REF_TRAILING, _REF_LEADING)
    return {"positive_script_prob": pos, "negative_script_prob": neg, "neutral_script_prob": _REF_NEUTRAL}


def _assign(trailing: float, leading: float) -> tuple[float, float]:
    """``(positive, negative)`` for the current ``POSITIVE_SCRIPT_IS_PASS_HEAVY``."""
    if POSITIVE_SCRIPT_IS_PASS_HEAVY:
        return trailing, leading
    return leading, trailing


# ---------------------------------------------------------------------------
# Input reading (never raises)
# ---------------------------------------------------------------------------
def _pass_lean_fraction(value: float) -> float | None:
    """PROE/RROE as a fraction: ``|x| > 1`` is read as percentage points; ``|x| > 100`` is bad data."""
    if abs(value) > 100.0:
        return None
    return value / 100.0 if abs(value) > 1.0 else value


def _read_values(source: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    """``({canonical key: validated value}, {canonical key: input key actually used})``."""
    values: dict[str, float] = {}
    via: dict[str, str] = {}
    for factor in FACTORS:
        candidates: list[tuple[str, float]] = []
        for key in (factor.key, *factor.aliases):
            if key in source:
                candidates.append((key, 1.0))
        if factor.key == _SPREAD.key and "expected_margin" in source:
            candidates.append(("expected_margin", -1.0))
        for key, sign in candidates:
            number = to_float(source.get(key))
            if number is None:
                continue
            number *= sign
            if factor.key in _POSITIVE_ONLY_KEYS and number <= 0.0:
                continue
            if factor.key in _PASS_LEAN_KEYS:
                number = _pass_lean_fraction(number)
                if number is None:
                    continue
            values[factor.key] = number
            via[factor.key] = key
            break
    return values, via


def team_spread_from_schedule(spread_line: Any, is_home: bool) -> float | None:
    """A team's ``vegas_spread`` (sportsbook sign) from nflverse's ``spread_line``.

    nflverse ``spread_line`` is positive when the HOME team is favoured (verified on
    2017-2025: corr with home - away margin +0.45, slope 1.04), so the home team's
    spread is ``-spread_line`` and the away team's is ``+spread_line``.
    """
    line = to_float(spread_line)
    if line is None:
        return None
    return -line if is_home else line


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def compute_game_script(inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Script probabilities for one offense in one game.

    See the module docstring for the event definition, calibration and limits.
    Returns ``model``, ``positive_script_prob`` / ``negative_script_prob`` /
    ``neutral_script_prob`` (sum to 1), the aliases ``trailing_script_prob``
    (pass-heavy) / ``leading_script_prob`` (run-heavy), ``probability`` (alias of
    ``positive_script_prob``), ``reference_probability``, ``references`` (all three
    neutral values), ``z_score`` (lean toward the *positive* script in SDs of the
    league spread), ``coverage``, ``inputs_used``, ``inputs_defaulted``,
    ``inputs_inert``, ``contributions``, ``drivers``, ``basis`` and the audit extras
    ``margin_source``, ``expected_margin_points``, ``volatility_scale``.
    """
    source: Mapping[str, Any] = inputs if isinstance(inputs, Mapping) else {}
    values, via = _read_values(source)
    lean_sign = 1.0 if POSITIVE_SCRIPT_IS_PASS_HEAVY else -1.0

    # -- 1. the effective spread (positive = underdog) --------------------------------
    team_dev = opp_dev = 0.0
    proxy_active = False
    if _SPREAD.key in values:
        margin_source = via[_SPREAD.key]
        base_points = clamp(values[_SPREAD.key], -SPREAD_CLIP, SPREAD_CLIP)
    elif any(k in values for k in _EPA_KEYS):
        margin_source = "epa_proxy"
        proxy_active = True
        if _TEAM_EPA.key in values:
            team_dev = clamp((values[_TEAM_EPA.key] - _TEAM_EPA.mean) / _TEAM_EPA.sd, -Z_CLIP, Z_CLIP) * _TEAM_EPA.sd
        if _OPP_EPA.key in values:
            opp_dev = clamp((values[_OPP_EPA.key] - _OPP_EPA.mean) / _OPP_EPA.sd, -Z_CLIP, Z_CLIP) * _OPP_EPA.sd
        base_points = clamp(-MARGIN_PER_EPA_DIFF * (team_dev - opp_dev), -SPREAD_CLIP, SPREAD_CLIP)
    else:
        margin_source = "none"
        base_points = 0.0

    # -- 2. optional bounded adjustments (inert at the shipped gain) --------------------
    adj_factors = tuple(f for f in ADJUSTMENT_FACTORS if not (proxy_active and f.key in _EPA_KEYS))
    scored = score_factors({k: v for k, v in values.items() if k in {f.key for f in adj_factors}}, adj_factors)
    raw_adjust = ADJUSTMENT_POINTS_PER_Z * scored["z"]
    adjust_points = clamp(raw_adjust, -ADJUSTMENT_MAX_POINTS, ADJUSTMENT_MAX_POINTS)
    adjust_scale = (adjust_points / raw_adjust) if raw_adjust else 1.0
    underdog_points = base_points + adjust_points

    # -- 3. volatility from the total ---------------------------------------------------
    total_points = 0.0
    if _TOTAL.key in values:
        z_total = clamp((values[_TOTAL.key] - _TOTAL.mean) / _TOTAL.sd, -Z_CLIP, Z_CLIP)
        total_points = z_total * _TOTAL.sd
    volatility_scale = math.exp(TOTAL_VOLATILITY * total_points / 10.0)

    # -- 4. shares ----------------------------------------------------------------------
    trailing_raw, _, leading_raw = _shares(underdog_points, volatility_scale)
    trailing, neutral, leading = _rounded(trailing_raw, leading_raw)
    positive, negative = _assign(trailing, leading)
    references = reference_shares()

    # -- 5. audit -----------------------------------------------------------------------
    used = [f.key for f in FACTORS if f.key in values]
    defaulted = [f.key for f in FACTORS if f.key not in values]
    coverage = sum(f.weight for f in FACTORS if f.key in values) / _TOTAL_WEIGHT
    contributions: dict[str, dict[str, Any]] = {}
    inert: list[str] = []
    for factor in FACTORS:
        if factor.key not in values:
            continue
        value = values[factor.key]
        z_raw = clamp((value - factor.mean) / factor.sd, -Z_CLIP, Z_CLIP)
        z_signed = z_raw * factor.direction * lean_sign  # + pushes the POSITIVE script up
        if factor.key == _SPREAD.key:
            role, effect = "spread", lean_sign * base_points / _SPREAD.sd
        elif factor.key == _TOTAL.key:
            role, effect = "volatility", 0.0
        elif proxy_active and factor.key in _EPA_KEYS:
            part = -MARGIN_PER_EPA_DIFF * team_dev if factor.key == _TEAM_EPA.key else MARGIN_PER_EPA_DIFF * opp_dev
            role, effect = "proxy", lean_sign * part / _SPREAD.sd
        elif ADJUSTMENT_POINTS_PER_Z != 0.0:
            share = scored["contributions"].get(factor.key, {}).get("effect", 0.0)
            role, effect = "adjustment", lean_sign * ADJUSTMENT_POINTS_PER_Z * share * adjust_scale / _SPREAD.sd
        else:
            role, effect = "inert", 0.0
            inert.append(factor.key)
        contributions[factor.key] = {
            "value": r(value, 4),
            "z": r(z_signed, 3),
            "weight": factor.weight,
            "effect": r(effect, 4),
            "label": factor.label,
            "unit": factor.unit,
            "via": via[factor.key],
            "role": role,
        }

    drivers: list[str] = []
    if margin_source in (_SPREAD.key, "expected_margin"):
        side = "underdog" if base_points > 0 else ("favourite" if base_points < 0 else "pick'em")
        drivers.append(
            f"Spread {values[_SPREAD.key]:+.1f} pts ({side}): trailing {trailing:.0%} vs {_REF_TRAILING:.0%} "
            f"reference, leading {leading:.0%} vs {_REF_LEADING:.0%}"
        )
    elif proxy_active:
        diff = team_dev - opp_dev
        drivers.append(
            f"No spread supplied: EPA differential {diff:+.3f}/play stands in for a {-base_points:+.1f} pt "
            f"expected margin (weaker than a real line)"
        )
    if _TOTAL.key in values and abs(volatility_scale - 1.0) >= 0.005:
        drivers.append(
            f"Total {values[_TOTAL.key]:.1f} {'widens' if volatility_scale > 1 else 'narrows'} both tails "
            f"(neutral share {neutral:.0%})"
        )
    if inert:
        drivers.append("No effect on the shares (no out-of-sample lift beyond the spread): " + ", ".join(inert))

    basis = (
        "Ordered-logit mapping from the offense's pre-game spread (and total as a volatility scale) to play-share "
        "probabilities of pass-heavy (trailing by 7+), neutral and run-heavy (leading by 7+) states, fitted on nflverse "
        "pbp 2018-2025 regular season (4,254 team-games, 264,889 snaps; 2025 held out). "
        f"Margin source: {margin_source}. EPA/pace/PROE/RROE carry no weight with a spread in hand (no out-of-sample "
        "lift); adjustment weights are priors, not fitted; "
        + ("positive script = pass-heavy (trailing)." if POSITIVE_SCRIPT_IS_PASS_HEAVY
           else "positive script = run-heavy (leading) [POSITIVE_SCRIPT_IS_PASS_HEAVY flipped].")
    )

    result: dict[str, Any] = {
        "model": MODEL,
        "positive_script_prob": positive,
        "negative_script_prob": negative,
        "neutral_script_prob": neutral,
        "trailing_script_prob": trailing,
        "leading_script_prob": leading,
        "probability": positive,
        "reference_probability": references["positive_script_prob"],
        "references": references,
        "z_score": r(lean_sign * underdog_points / _SPREAD.sd, 4),
        "coverage": r(coverage, 3),
        "inputs_used": used,
        "inputs_defaulted": defaulted,
        "inputs_inert": inert,
        "contributions": contributions,
        "drivers": drivers,
        "basis": basis,
        "margin_source": margin_source,
        "expected_margin_points": r(-underdog_points, 3),
        "volatility_scale": r(volatility_scale, 4),
        "positive_script_is_pass_heavy": bool(POSITIVE_SCRIPT_IS_PASS_HEAVY),
    }
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "%s: positive=%.4f negative=%.4f neutral=%.4f (trailing=%.4f leading=%.4f; references %.4f/%.4f/%.4f) "
            "margin_source=%s expected_margin=%+.1f scale=%.3f coverage=%.0f%% used=%d defaulted=%d",
            MODEL, positive, negative, neutral, trailing, leading,
            references["positive_script_prob"], references["negative_script_prob"], references["neutral_script_prob"],
            margin_source, -underdog_points, volatility_scale, 100.0 * coverage, len(used), len(defaulted),
        )
    return result


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def _script_pair(result: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    """``(positive, negative, positive_reference, negative_reference)`` or ``None``.

    Reads the brief's names first, then the trailing/leading aliases (mapped through
    the current semantics switch).  Probabilities are clamped to ``[0, 1]`` and
    scaled down if they sum past 1; a missing reference falls back to the module's.
    """
    if not isinstance(result, Mapping):
        return None
    positive = to_float(result.get("positive_script_prob"))
    negative = to_float(result.get("negative_script_prob"))
    if positive is None or negative is None:
        trailing = to_float(result.get("trailing_script_prob"))
        leading = to_float(result.get("leading_script_prob"))
        if trailing is None or leading is None:
            return None
        positive, negative = _assign(trailing, leading)
    positive = clamp(positive, 0.0, 1.0)
    negative = clamp(negative, 0.0, 1.0)
    if positive + negative > 1.0:
        total = positive + negative
        positive, negative = positive / total, negative / total
    defaults = reference_shares()
    refs = result.get("references")
    refs = refs if isinstance(refs, Mapping) else {}
    ref_pos = to_float(refs.get("positive_script_prob"))
    ref_neg = to_float(refs.get("negative_script_prob"))
    ref_pos = defaults["positive_script_prob"] if ref_pos is None else clamp(ref_pos, 0.0, 1.0)
    ref_neg = defaults["negative_script_prob"] if ref_neg is None else clamp(ref_neg, 0.0, 1.0)
    return positive, negative, ref_pos, ref_neg


def _scale_attempts(
    projection: dict[str, Any],
    key: str,
    *,
    coef_positive: float,
    coef_negative: float,
    positive: float,
    negative: float,
    ref_positive: float,
    ref_negative: float,
    recenter: bool,
    adjustments: list[dict[str, Any]],
) -> None:
    """In-place ``projection[key] *= 1 + coef_positive*pos + coef_negative*neg`` with an audit entry.

    ``coef_*`` are signed (a QB's negative-script coefficient is ``-0.25``).  The
    brief's two-term form cannot be split into two ``scale_field`` calls without
    changing it (``(1+a)(1-b) != 1+a-b``), so it is applied here in one step with the
    same audit shape as ``scale_field``.
    """
    before = to_float(projection.get(key))
    if before is None:
        adjustments.append({"stage": MODEL, "field": key, "applied": False, "reason": "field missing or not numeric"})
        return
    factor = 1.0 + coef_positive * positive + coef_negative * negative
    if recenter:
        denominator = 1.0 + coef_positive * ref_positive + coef_negative * ref_negative
        if denominator > 1e-9:
            factor /= denominator
    factor = max(0.0, factor)
    after = before * factor
    projection[key] = after
    adjustments.append(
        {
            "stage": MODEL,
            "field": key,
            "applied": True,
            "before": r(before, 4),
            "after": r(after, 4),
            "factor": r(factor, 4),
            "positive_script_prob": r(positive),
            "negative_script_prob": r(negative),
            "reference_positive": r(ref_positive),
            "reference_negative": r(ref_negative),
            "coefficient_positive": abs(coef_positive),
            "coefficient_negative": abs(coef_negative),
            "recentered": bool(recenter),
        }
    )


def apply_game_script(
    projection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    recenter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the brief's two script formulas to ``qb_attempts`` and ``rb_attempts``.

    ::

        qb_attempts *= (1 + positive*0.20 - negative*0.25)
        rb_attempts *= (1 + negative*0.30 - positive*0.20)

    ``recenter=True`` (default) divides each factor by the same formula evaluated at
    the model's reference shares, so an average situation is an exact no-op;
    ``recenter=False`` is the literal formula.  Returns ``(new_projection,
    adjustments)``; the input mapping is never mutated, every other key is copied
    through, and a missing/non-numeric field is recorded as ``applied=False``.
    Attempts are counts, not probabilities, so no cap applies (the factor is only
    floored at 0).  If ``result`` has no usable script probabilities nothing moves
    and both fields are recorded as not applied.
    """
    new_projection: dict[str, Any] = dict(projection) if isinstance(projection, Mapping) else {}
    adjustments: list[dict[str, Any]] = []
    pair = _script_pair(result)
    if pair is None:
        for key in ("qb_attempts", "rb_attempts"):
            adjustments.append(
                {"stage": MODEL, "field": key, "applied": False, "reason": "script probabilities missing or not numeric"}
            )
        return new_projection, adjustments
    positive, negative, ref_positive, ref_negative = pair
    common = {
        "positive": positive,
        "negative": negative,
        "ref_positive": ref_positive,
        "ref_negative": ref_negative,
        "recenter": recenter,
        "adjustments": adjustments,
    }
    _scale_attempts(
        new_projection, "qb_attempts",
        coef_positive=+QB_ATTEMPTS_POSITIVE_COEFFICIENT, coef_negative=-QB_ATTEMPTS_NEGATIVE_COEFFICIENT, **common,
    )
    _scale_attempts(
        new_projection, "rb_attempts",
        coef_positive=-RB_ATTEMPTS_POSITIVE_COEFFICIENT, coef_negative=+RB_ATTEMPTS_NEGATIVE_COEFFICIENT, **common,
    )
    return new_projection, adjustments


__all__ = [
    "ADJUSTMENT_FACTORS",
    "ADJUSTMENT_MAX_POINTS",
    "ADJUSTMENT_POINTS_PER_Z",
    "BASE_RATE",
    "EVENT_DEFINITION",
    "FACTORS",
    "GAIN",
    "MARGIN_PER_EPA_DIFF",
    "MODEL",
    "OUTPUT_KEYS",
    "POSITIVE_SCRIPT_IS_PASS_HEAVY",
    "QB_ATTEMPTS_NEGATIVE_COEFFICIENT",
    "QB_ATTEMPTS_POSITIVE_COEFFICIENT",
    "RB_ATTEMPTS_NEGATIVE_COEFFICIENT",
    "RB_ATTEMPTS_POSITIVE_COEFFICIENT",
    "SCRIPT_THRESHOLD_POINTS",
    "SPREAD_CLIP",
    "THRESHOLD_LEADING",
    "THRESHOLD_TRAILING",
    "TOTAL_VOLATILITY",
    "apply_game_script",
    "compute_game_script",
    "reference_shares",
    "team_spread_from_schedule",
]
