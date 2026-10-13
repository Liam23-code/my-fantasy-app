"""Run-game collapse model: how likely is a team's rushing game to fall apart?

Output
------
``run_collapse`` in ``[0, 1]``. It is applied to an RB projection exactly as the
brief states (``recenter=False``)::

    rb_yards    *= (1 - run_collapse * 0.50)
    rb_td_prob  *= (1 - run_collapse * 0.35)

The default (``recenter=True``) divides each multiplier by the same expression
evaluated at the model's reference probability (see ``nfl_pack_common``), so a
league-average situation leaves a projection untouched.

Event definition (what ``p`` means)
-----------------------------------
A **run-game collapse** is a team-game in which the team's *designed RB
rushing* (RB/FB carries; QB scrambles, QB runs by non-RBs and two-point tries
excluded) produced

* total rushing yards **<= 60 % of that team's own regular-season per-game
  average** (the average over the team's season, this game included), **and**
* **negative total rushing EPA** on those carries.

``BASE_RATE = 0.176``: **17.6 % of team-games** meet it (nflverse play-by-play,
regular seasons 2019-2025, n = 3,742 team-games, 224 team-seasons; per season
0.147-0.197). Sensitivity: a leave-one-out team average gives 0.190; a 50 %
threshold gives 0.108; a 70 % threshold gives 0.252. Collapse games are
short, ineffective games (mean 15 RB carries and 40 yards, against 23 carries
and 105 yards otherwise). So ``run_collapse = 0.30`` reads "about a 30 % chance
that this team's backs get <= 60 % of their usual yards on negative EPA".

Inputs (canonical keys; any subset may be supplied)
---------------------------------------------------
============================  ==========  ===  ======  ============================================
key                           unit        dir  weight  league reference (mean / sd) and where it comes from
============================  ==========  ===  ======  ============================================
``ol_run_block_win_rate``     rate        -1   1.30    0.70 / 0.04, PRIOR (tracking-based; no free feed)
``trailing_script_prob``      rate        +1   1.20    0.309 / 0.118, nflverse pbp 2019-2025, n=3,742
``opp_run_defense_grade``     pctl 0-100  +1   1.00    50 / 28.9, by construction (uniform percentile)
``opp_stacked_box_rate``      rate        +1   0.80    0.216 / 0.082, participation 2019-2023, n=160
``ol_injuries``               count 0-5   +1   0.70    0.58 / 0.75, snap_counts 2019-2025, n=3,294
``rb_yards_after_contact``    yds/att     -1   0.45    2.01 / 0.41, PFR rush 2019-2025, n=328
``rb_breakaway_rate``         rate        -1   0.25    0.0445 / 0.0195, pbp 2019-2025, n=328
============================  ==========  ===  ======  ============================================

Aliases (same meaning): ``negative_game_script_prob`` -> ``trailing_script_prob``
(the brief's RB sense: the offense is TRAILING and abandons the run);
``offensive_line_run_block_win_rate`` / ``run_block_win_rate``;
``offensive_line_injuries`` / ``ol_injury_count``; ``opponent_run_defense_grade``;
``stacked_box_rate``. When ``ol_injuries`` is absent, ``ol_health`` (the contract's
derived ``1 - ol_injuries / 5``) is converted back to a count, as the QB and
pass-game models do; an explicit ``ol_injuries`` always wins.
**``negative_script_prob`` is deliberately NOT an alias.** In ``game_script_model``
it means *run-heavy / leading*, the opposite of the brief's phrase; accepting it
here would silently invert the factor.

Why this ordering of weights (all PRIORS, not fitted; ordered by how directly
the cause produces a bad rushing game, and fixed *before* the association check
below was run, then left unchanged by it):

1. ``ol_run_block_win_rate`` - blockers losing their assignments is the
   proximate cause of a run play going nowhere; the most persistent driver of
   yards before contact.
2. ``trailing_script_prob`` - the volume lever. A trailing offense abandons the
   run; collapse games average 15 carries vs 23. Ranked just below 1 because the
   *pre-game* probability forecasts a state that is mostly decided in-game.
3. ``opp_run_defense_grade`` - front quality; persistent, but its free proxy
   (rush EPA allowed) is noisy.
4. ``opp_stacked_box_rate`` - the mechanism through which a front stops the run
   (8+ in the box); partly a scheme *response* to the offense, so lower.
5. ``ol_injuries`` - lineup change the season-long win rate has not absorbed
   yet; overlaps with 1.
6. ``rb_yards_after_contact`` - what the back creates himself; mitigates, but
   a team-level collapse is mostly decided before contact.
7. ``rb_breakaway_rate`` - highest-variance, smallest per-game signal.

Inputs are not de-correlated (the weights are priors). Measured on the free
proxies, the overlap is mostly among the own-offense factors: breakaway rate vs
the OL proxy rho +0.51, breakaway vs yards after contact +0.46; every other pair
is |rho| <= 0.20 (script vs OL proxy +0.20, script vs run-D grade +0.17). The
lineup-injury factor (5) conceptually overlaps with the win rate (1).

Scale of the inputs a loader must respect
-----------------------------------------
* ``ol_run_block_win_rate`` is the ESPN-style *tracking* win rate (about 0.70 for
  an average line). There is **no free feed** for it. The free proxy named in
  the contract (PFR ``ybc_att``, RB yards before contact per attempt; team-
  season mean 2.31, sd 0.38, n=224) is on a different scale and **must be
  rescaled to this key's scale before it is passed** (e.g. ``0.70 + 0.04 * z``);
  passing 2.3 or a different rate unscaled makes the factor meaningless.
* ``opp_stacked_box_rate``: nflverse *participation* ``defenders_in_box`` has a
  definition break in 2024-25 (share of designed runs facing 8+ falls from ~0.21
  in 2019-2023 to 0.10 in 2024 and 0.09 in 2025). NGS rushing
  ``percent_attempts_gte_eight_defenders`` does not break (0.20-0.25 every year;
  offense-side, 2019-2025, mean 0.218, sd 0.075, n=220). For 2024+ use NGS or a
  within-season z-score; an unscaled 2024 participation rate will read as an
  extreme "light box".
* ``opp_run_defense_grade`` is a literal 0-100 percentile (higher = better
  defense). A value of ``0.8`` is the 0.8th percentile, not 80 %.
* ``ol_injuries`` counts OL starters out (questionable = 0.5). The reference is
  an *absence* measure: starters (top five OL by snaps in the previous three
  games) who took zero snaps; it cannot see injury designations.
* A value that cannot be real is treated as missing, not clamped into
  plausibility, and is listed under ``inputs_rejected``: grade outside 0-100,
  an ``ol_injuries`` count outside 0-5 (a line has five starters; the contract
  range is 0-5, and the QB and pass-game models reject the same values),
  yards after contact outside 0-10, a rate outside 0-100 %. Text that is not a
  number at all (``None``, ``NaN``, ``"n/a"``) is simply absent, not rejected.

GAIN
----
``GAIN = 0.8`` is a prior. ``p = sigmoid(logit(0.176) + 0.8 * z)`` where ``z`` is
the weight-normalised mean of the per-input z-scores. All inputs one SD in the
risk direction gives ``z = 1`` and ``p = 0.322``; two SD gives ``0.514`` in
formula terms. In practice the grade cannot exceed 100 (+1.73 SD), so "everything
two SD bad" with the grade held at its maximum is ``0.505`` (z = 1.95). Inputs are
clipped at 3 SD, so ``0.702`` is the hard ceiling of the formula and the practical
one is about ``0.66`` (grade at 100, everything else at the clip). All one SD in
the *benign* direction gives ``0.088`` (about ``0.09`` once the injuries count is
floored at zero, which is only -0.77 SD). Anchors from the same data (strictly
pre-game proxies, n = 2,830): top-quartile run defense AND underdog by 3+ realised
0.247 (n=348, RB yards 0.88 x own average); bottom-quartile defense AND favourite
by 3+ realised 0.092 (n=347, 1.13 x). The brief's multiplier at p=0.322 is x0.92
for ``rb_yards`` recentered, x0.84 literal.

HONEST LIMITS
-------------
* Weights and GAIN are priors; nothing here is fitted. The one empirical
  sanity check (scratch script, Spearman, risk-signed feature vs the event;
  season-to-date proxies from prior games only, so no look-ahead) was weak:

  =====================================  =================  ====================
  feature (free proxy)                    vs event, full-    vs event, pre-game
                                          season baseline    (to-date) baseline
  =====================================  =================  ====================
  OL run block (-RB yds before contact)   -0.03 [-.07,+.01]  -0.06 [-.10,-.03]
  script (Vegas spread, + = underdog)     +0.07 [+.04,+.10]  +0.05 [+.01,+.09]
  opp run defense (RB-run EPA allowed)    +0.09 [+.06,+.13]  +0.10 [+.06,+.13]
  opp stacked box (>=8, within-week z)    -0.00 [-.04,+.03]  +0.00 [-.03,+.04]
  OL starters absent                      +0.01 [-.03,+.04]  +0.01 [-.02,+.05]
  RB yards after contact (-yac/att)       -0.02 [-.06,+.02]  -0.06 [-.10,-.02]
  RB breakaway (-15+ yd rate)             -0.05 [-.09,-.01]  -0.08 [-.12,-.04]
  composite ``run_collapse``              +0.04, AUC 0.533   +0.02, AUC 0.512
  =====================================  =================  ====================

  (n = 2,830-3,742 team-games, 2019-2025; the composite row uses the 2,830 with
  four or more prior games; 95 % CIs by Fisher z. For the composite only, the OL
  proxy and the within-week stacked-box z were z-matched onto the factors' own
  scales (13 rows where that put the stacked-box rate below zero were scored
  with the factor missing); the other inputs went in on their native scales.)

  Only the opponent run defense and the game script show the expected sign with
  a CI away from zero. The three own-offense efficiency proxies (OL, yards after
  contact, breakaways) show the OPPOSITE sign: the event is "bad relative to the
  team's own average", the season-level efficiency of the offense is already in
  that average, and a hot or cold stretch regresses. That does not falsify the
  prior about a real tracking win rate (the proxy is not it), and the sign and
  weights were left as specified. The composite does not rank team-games better
  than chance in the forecast-faithful frame, its Brier score is no better than
  always predicting the base rate (0.146-0.152 vs 0.144-0.149), and its top
  decile (mean predicted 0.30) realised 0.17-0.18 (bottom decile: predicted 0.10,
  realised 0.15-0.18). **Do not read ``run_collapse`` as a calibrated
  probability, and do not treat a small move in it as an edge.**
* The realised in-game trailing share is far more tied to the event (rho +0.25)
  than any pre-game forecast of it (+0.05 to +0.07): script is mostly decided
  in-game.
* ``stacked box``, ``ol_injuries`` and the OL win rate have no free pre-game
  feed that carries signal here; they stay in the model because the brief
  names them, and are neutral until supplied.
* The multipliers apply to ``rb_yards`` (rushing yards only) and ``rb_td_prob``
  only; receiving production and ``rb_attempts`` are not touched (game-script
  volume belongs to ``game_script_model``).
* ``rb_td_prob`` and ``rb_yards`` are scaled together with the *same* ``p``; the
  brief gives no separate TD collapse driver, so they are perfectly correlated.

The module imports only the standard library and ``modules.nfl_pack_common``.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from modules import nfl_pack_common as common
from modules.nfl_pack_common import Factor

#: Logger for this model. ``nfl_pack_common`` also logs the standard line on ``nfl_pack``.
LOG = logging.getLogger("nfl_pack.run_game_collapse")

MODEL_NAME = "run_game_collapse"
#: Key of the model's primary output in the result dict.
OUTPUT_KEY = "run_collapse"
#: ``stage`` label written into every audit entry.
STAGE = "run_game_collapse"

#: Share of 2019-2025 regular-season team-games (n=3,742) whose designed-RB rushing yards were
#: <= 60% of the team's own season per-game average AND whose RB rushing EPA was negative.
#: nflverse pbp, play_type run, rush_attempt==1, qb_scramble==0, rusher position RB/FB.
BASE_RATE = 0.176

#: Logistic slope applied to the weight-normalised z-score (a prior). In z terms: all inputs +1 SD
#: in the risk direction -> 0.322; +2 SD -> 0.514; +3 SD (the clip) -> 0.702; all -1 SD -> 0.088.
#: (A 0-100 grade tops out at +1.73 SD, so the practical ceiling is ~0.66; see the docstring.)
GAIN = 0.8

#: The brief's coefficients, named after the field they scale.
RB_YARDS_COEFFICIENT = 0.50
RB_TD_PROB_COEFFICIENT = 0.35

EVENT_DEFINITION = (
    "team designed-RB rushing yards <= 60% of the team's own season per-game average AND negative "
    "RB rushing EPA; 17.6% of 2019-2025 regular-season team-games (n=3,742)"
)

BASIS = (
    "Logistic of a weighted z-score of seven inputs. Weights and GAIN are priors, not fitted. League "
    "references come from nflverse 2019-2025 where a free feed exists (pbp, participation, snap_counts, "
    "PFR rush) and are priors otherwise (OL run-block win rate: tracking-based, no free feed; run-defense "
    "grade: percentile by construction). BASE_RATE 0.176 = share of 2019-2025 REG team-games (n=3,742) "
    "with designed-RB rushing yards <= 60% of the team's season average and negative RB rushing EPA. "
    "A scratch sanity check with season-to-date proxies was weak (composite Spearman +0.02 to +0.04, "
    "AUC 0.51-0.53); treat as a prior-driven adjustment, not a calibrated forecast."
)

# Reference sources (state feed + seasons + n):
#   ol_run_block_win_rate : PRIOR. Tracking-based (ESPN/NGS-style); no free feed. Secondary reports put
#                           leading teams in the mid-70s %; no league average was found, so 0.70 / 0.04 is assumed.
#   trailing_script_prob  : nflverse pbp 2019-2025 REG run+pass plays (2pt excluded), share of a team's
#                           plays snapped trailing by >= 7: team-game mean 0.309 (n=3,742; league play share
#                           0.307). sd 0.118 = sd of the Vegas-spread-implied expectation (OLS slope 0.0184
#                           per point of spread, R2 0.12): the spread of a PRE-GAME forecast, not the
#                           realised in-game share (sd 0.34).
#   opp_run_defense_grade : team percentile rank 0-100, mean 50, sd 28.9 (uniform percentile); by construction.
#   opp_stacked_box_rate  : participation defenders_in_box >= 8 on designed runs, DEFENSE side, 2019-2023 REG,
#                           n=160 team-seasons (avg 403 runs each): mean 0.216, sd 0.082. Cross-check NGS
#                           rushing (offense side, 2019-2025, n=220): 0.218 / 0.075. 2024-25 participation
#                           breaks (0.09), so those seasons are excluded from the reference.
#   ol_injuries           : nflverse snap_counts 2019-2025 REG, team-games from week 3 (n=3,294); starters =
#                           top-5 OL by snaps in the prior 3 games; "out" = zero snaps. mean 0.58, sd 0.75.
#   rb_yards_after_contact: PFR advstats rush 2019-2025 REG, RB/FB with >= 100 carries, n=328 RB-seasons:
#                           mean 2.01 yds/att, sd 0.41 (pooled 2.03).
#   rb_breakaway_rate     : nflverse pbp 2019-2025 REG, designed RB runs gaining >= 15 yds, RB-seasons with
#                           >= 100 carries, n=328: mean 0.0445, sd 0.0195 (includes sampling noise on purpose).
FACTORS: tuple[Factor, ...] = (
    Factor(
        "ol_run_block_win_rate", mean=0.70, sd=0.04, weight=1.30, direction=-1,
        label="OL run-block win rate", kind="rate", unit="%",
        aliases=("offensive_line_run_block_win_rate", "run_block_win_rate"), source="prior",
    ),
    Factor(
        "trailing_script_prob", mean=0.309, sd=0.118, weight=1.20, direction=+1,
        label="Trailing game-script probability", kind="rate", unit="%",
        aliases=("negative_game_script_prob",), source="nflverse pbp 2019-2025 REG, n=3742 team-games",
    ),
    Factor(
        "opp_run_defense_grade", mean=50.0, sd=28.9, weight=1.00, direction=+1,
        label="Opponent run-defense grade", unit="pctl",
        aliases=("opponent_run_defense_grade",), source="prior (uniform percentile by construction)",
    ),
    Factor(
        "opp_stacked_box_rate", mean=0.216, sd=0.082, weight=0.80, direction=+1,
        label="Opponent stacked-box rate", kind="rate", unit="%",
        aliases=("stacked_box_rate",), source="nflverse participation 2019-2023 REG, n=160 team-seasons",
    ),
    Factor(
        "ol_injuries", mean=0.58, sd=0.75, weight=0.70, direction=+1,
        label="OL starters out", unit="starters",
        aliases=("offensive_line_injuries", "ol_injury_count"),
        source="nflverse snap_counts 2019-2025 REG, n=3294 team-games",
    ),
    Factor(
        "rb_yards_after_contact", mean=2.01, sd=0.41, weight=0.45, direction=-1,
        label="RB yards after contact", unit="yds/att", source="PFR rush 2019-2025 REG, n=328 RB-seasons",
    ),
    Factor(
        "rb_breakaway_rate", mean=0.0445, sd=0.0195, weight=0.25, direction=-1,
        label="RB breakaway rate (15+ yd runs)", kind="rate", unit="%",
        source="nflverse pbp 2019-2025 REG, n=328 RB-seasons",
    ),
)

#: Physical windows for the non-rate inputs, ``(low, high)`` inclusive. A number outside is a data
#: error: it is rejected (treated as missing and reported), never clamped into plausibility.
#: Rates are gated by ``to_rate`` (0-100 %).
_DOMAIN: dict[str, tuple[float, float]] = {
    "ol_injuries": (0.0, 5.0),
    "opp_run_defense_grade": (0.0, 100.0),
    "rb_yards_after_contact": (0.0, 10.0),
}


def _clean_inputs(inputs: Any) -> tuple[dict[str, float], list[dict[str, str]], dict[str, str]]:
    """``(clean, rejected, derived)``: canonical-key floats the scorer may use (never raises).

    Anything that is not a finite number is simply absent. A number outside its physical domain
    is *rejected* and reported. For each factor the canonical key wins over its aliases.
    """
    data: Mapping[str, Any] = inputs if isinstance(inputs, Mapping) else {}
    clean: dict[str, float] = {}
    rejected: list[dict[str, str]] = []
    derived: dict[str, str] = {}

    def _read(factor: Factor, key: str) -> float | None:
        if key not in data:
            return None
        raw = data[key]
        number = common.to_float(raw)
        if number is None:
            return None  # not a number: absent, not an error
        reason: str | None = None
        value: float | None
        if factor.kind == "rate":
            value = common.to_rate(raw)
            if value is None:
                reason = "rate outside 0-100%"
        else:
            value = number
            if factor.key == "opp_run_defense_grade" and isinstance(raw, str) and raw.strip().endswith("%"):
                value = number * 100.0  # "78%" typed as a percentile
            window = _DOMAIN.get(factor.key)
            if window is not None and not (window[0] <= value <= window[1]):
                reason = f"outside the possible range {window[0]:g} to {window[1]:g}"
        if reason is not None:
            rejected.append({"key": key, "value": repr(raw)[:40], "reason": reason})
            return None
        return value

    for factor in FACTORS:
        for key in (factor.key, *factor.aliases):
            value = _read(factor, key)
            if value is not None:
                clean[factor.key] = value
                break
    if "ol_injuries" not in clean and "ol_health" in data:
        health = common.to_rate(data["ol_health"])
        if health is not None:
            clean["ol_injuries"] = round(5.0 * (1.0 - health), 4)
            derived["ol_injuries"] = "5 * (1 - ol_health)"
        elif common.to_float(data["ol_health"]) is not None:
            rejected.append({"key": "ol_health", "value": repr(data["ol_health"])[:40], "reason": "rate outside 0-100%"})
    return clean, rejected, derived


def _multipliers(probability: float, reference: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for field, coefficient in (("rb_yards", RB_YARDS_COEFFICIENT), ("rb_td_prob", RB_TD_PROB_COEFFICIENT)):
        out[field] = {
            "literal": common.r(common.adjustment_factor(probability, coefficient, direction=-1, recenter=False)),
            "recentered": common.r(
                common.adjustment_factor(probability, coefficient, direction=-1, reference=reference, recenter=True)
            ),
        }
    return out


def compute_run_game_collapse(inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run-game collapse probability for one team-game.

    ``inputs`` is a flat mapping of the canonical keys in the module docstring (any subset;
    ``None`` / ``{}`` returns the reference). Bad values are treated as missing; this never raises.
    Returns the standard model dict (``run_collapse``, ``probability``, ``reference_probability``,
    ``references``, ``z_score``, ``coverage``, ``inputs_used``, ``inputs_defaulted``,
    ``contributions``, ``drivers``, ``basis``) plus ``event`` (what the probability means),
    ``multipliers`` (the implied ``rb_yards`` / ``rb_td_prob`` factors, literal and recentered),
    ``inputs_rejected`` (numbers outside their physical range, ignored) and ``inputs_derived``
    (``ol_injuries`` computed from ``ol_health``).
    """
    clean, rejected, derived = _clean_inputs(inputs)
    result = common.run_factor_model(
        model=MODEL_NAME,
        primary_key=OUTPUT_KEY,
        inputs=clean,
        factors=FACTORS,
        base_rate=BASE_RATE,
        gain=GAIN,
        basis=BASIS,
    )
    probability = float(result[OUTPUT_KEY])
    reference = float(result["reference_probability"])
    result["event"] = EVENT_DEFINITION
    result["multipliers"] = _multipliers(probability, reference)
    result["inputs_rejected"] = rejected
    result["inputs_derived"] = derived
    result["basis"] = (
        f"{BASIS} Inputs supplied: {len(result['inputs_used'])} of {len(FACTORS)} "
        f"(coverage {result['coverage']:.0%})."
    )
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "run_game_collapse: run_collapse=%.4f reference=%.4f -> rb_yards x%.3f, rb_td_prob x%.3f "
            "(recentered; literal x%.3f / x%.3f); coverage=%.0f%%; defaulted=%s; rejected=%d",
            probability,
            reference,
            result["multipliers"]["rb_yards"]["recentered"],
            result["multipliers"]["rb_td_prob"]["recentered"],
            result["multipliers"]["rb_yards"]["literal"],
            result["multipliers"]["rb_td_prob"]["literal"],
            100.0 * float(result["coverage"]),
            ",".join(result["inputs_defaulted"]) or "-",
            len(rejected),
        )
    return result


def _read_probability(result: Any) -> float | None:
    """The run-collapse probability in ``result`` as a fraction (percent input accepted), or ``None``."""
    if not isinstance(result, Mapping):
        return None
    for key in (OUTPUT_KEY, "probability"):
        value = common.to_rate(result.get(key))
        if value is not None:
            return value
    return None


def _read_reference(result: Mapping[str, Any]) -> float:
    """The neutral probability to re-centre on (``references`` -> ``reference_probability`` -> BASE_RATE)."""
    references = result.get("references")
    candidates = [references.get(OUTPUT_KEY) if isinstance(references, Mapping) else None,
                  result.get("reference_probability")]
    for candidate in candidates:
        value = common.to_rate(candidate)
        if value is not None:
            return value
    return BASE_RATE


def _probability_cap(value: Any) -> float | None:
    """1.0 for a fraction-scale probability, 100.0 for a percent-scale one (``<= 1.5`` is a fraction).

    Never below the value itself: a cap is a ceiling for the adjustment, so it must not drag a
    value that already exceeds it (a malformed 1.2) *down* when the adjustment is a boost.
    """
    number = common.to_float(value)
    if number is None:
        return None
    return max(1.0 if number <= 1.5 else 100.0, number)


def apply_run_game_collapse(
    projection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    recenter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the run-game collapse to an RB projection.

    ``rb_yards *= (1 - p * 0.50)`` and ``rb_td_prob *= (1 - p * 0.35)`` with ``p = result['run_collapse']``.
    ``recenter=True`` (default) divides each factor by the same formula at the result's reference
    probability so an average situation is a no-op; ``recenter=False`` is the literal brief formula.
    ``rb_td_prob`` is capped at 1.0 (or 100.0 if it is on a percent scale). Returns
    ``(new_projection, adjustments)``; ``projection`` is never mutated, every other key is copied
    through, and a missing / non-numeric field is recorded as ``applied=False``.
    """
    new_projection: dict[str, Any] = dict(projection) if isinstance(projection, Mapping) else {}
    adjustments: list[dict[str, Any]] = []
    probability = _read_probability(result)
    if probability is None:
        for field in ("rb_yards", "rb_td_prob"):
            adjustments.append(
                {"stage": STAGE, "field": field, "applied": False, "reason": "result has no usable run_collapse"}
            )
        return new_projection, adjustments
    reference = _read_reference(result)
    common.scale_field(
        new_projection, "rb_yards", stage=STAGE, probability=probability, coefficient=RB_YARDS_COEFFICIENT,
        direction=-1, reference=reference, recenter=recenter, adjustments=adjustments,
    )
    common.scale_field(
        new_projection, "rb_td_prob", stage=STAGE, probability=probability, coefficient=RB_TD_PROB_COEFFICIENT,
        direction=-1, reference=reference, recenter=recenter, adjustments=adjustments,
        cap=_probability_cap(new_projection.get("rb_td_prob")),
    )
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "run_game_collapse apply: run_collapse=%.4f reference=%.4f recenter=%s %s",
            probability,
            reference,
            recenter,
            "; ".join(
                f"{a['field']} {a['before']}->{a['after']}" if a.get("applied") else f"{a['field']} skipped"
                for a in adjustments
            ),
        )
    return new_projection, adjustments


__all__ = [
    "BASE_RATE",
    "BASIS",
    "EVENT_DEFINITION",
    "FACTORS",
    "GAIN",
    "LOG",
    "MODEL_NAME",
    "OUTPUT_KEY",
    "RB_TD_PROB_COEFFICIENT",
    "RB_YARDS_COEFFICIENT",
    "STAGE",
    "apply_run_game_collapse",
    "compute_run_game_collapse",
]
