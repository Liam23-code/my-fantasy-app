"""QB collapse probability for the NFL model upgrade pack.

``compute_qb_collapse`` turns twelve protection / pass-rush / quarterback inputs
into ``collapse_probability`` in ``[0, 1]``; ``apply_qb_collapse`` applies the
brief's haircuts to a quarterback projection.  Pure, offline and deterministic:
the module imports only the standard library and :mod:`modules.nfl_pack_common`.

EVENT DEFINITION (what ``collapse_probability = 0.16`` means)
-------------------------------------------------------------
A **collapse** is a regular-season game in which the quarterback who took the
most dropbacks for his team (the *starter*, at least 20 dropbacks) finishes with
**EPA per dropback <= -0.25**.  A dropback is every ``qb_dropback`` play in
nflverse play-by-play (passes, sacks and scrambles, penalties included).

* Measured base rate: **0.1586 (580 of 3,658 starter games, nflverse pbp
  2019-2025)**; by season 0.125 (2020) to 0.181 (2023).  ``BASE_RATE = 0.16``.
* So ``p = 0.16`` is "an average starter-game: roughly one in six".  The league
  mean starter-game is +0.05 EPA/dropback (sd 0.31); -0.25 is the 16th percentile.
* The threshold was chosen so the event is common enough to be useful (about one
  game in six) and sits on a round number; it was not tuned to any model output.

THE MODEL
---------
A transparent weighted z-score through a logistic (``nfl_pack_common``)::

    z = sum(weight_i * direction_i * clip(z_i)) / sum(weight over ALL twelve)
    collapse_probability = sigmoid(logit(BASE_RATE) + GAIN * z)

An input the caller did not supply is neutral (``z_i = 0``) but still counts in
the denominator, so missing data shrinks the answer toward ``BASE_RATE``.

Inputs (direction = effect on collapse risk; weights are PRIORS)
----------------------------------------------------------------
============================== ==== ====== ================ ==========================
input                          dir  weight league mean/sd   free feed
============================== ==== ====== ================ ==========================
ol_pass_block_win_rate         -1   2.0    0.700 / 0.039    proxy: 1 - pressure allowed
ol_injuries (starters out)     +1   1.2    0.37  / 0.54     NO live feed (see below)
ol_continuity                  -1   0.8    0.822 / 0.073    nflverse snap counts
opp_pressure_rate              +1   1.1    0.300 / 0.036    pbp + participation
opp_pass_rush_win_rate         +1   0.7    0.300 / 0.036    proxy: pressure + sack rate
opp_simulated_pressure_success +1   0.5    0.269 / 0.036    participation (<= 4 rushers)
opp_blitz_rate                 +1   0.5    0.261 / 0.067    participation (>= 5 rushers)
qb_pressure_epa                -1   1.0    -0.447 / 0.194   pbp + participation
qb_sack_avoidance              -1   0.7    0.933 / 0.023    pbp
qb_time_to_throw               +1   0.6    2.82 s / 0.157   NGS passing
qb_release_time                +1   0.3    2.34 s / 0.141   participation (unpressured)
qb_scramble_rate               -1   0.2    0.052 / 0.029    pbp
============================== ==== ====== ================ ==========================

Why this order (a prior judgement, not a fit)
    1. *Protection and opponent pressure are the cause*, so they carry the most
       weight: protection 4.0 of 9.6 (42%), opponent pressure 2.8 (29%).
       Pass-block win rate is first because it is the direct summary of the
       protection the passer will actually get; the acute injury count outranks
       season-long continuity because it is the news that moves a line.
    2. The three opponent pressure inputs (``opp_pressure_rate``,
       ``opp_pass_rush_win_rate``, ``opp_simulated_pressure_success``) are one
       measurement seen three ways (correlation 0.94 between pressure rate and
       <=4-man pressure rate; the first two are the *same* series on free data).
       Their weights sum to 2.3 (24%), *below* the protection cluster's 4.0, so
       a strong pass rush cannot out-vote a clean pocket on its own.  Blitz rate
       is a different mechanism (correlation 0.18 with pressure) and gets a
       modest weight: blitzes raise pressure (39% vs 27%) but also create hot
       reads and big plays, so it is a weaker driver of a *collapse* than raw
       pressure.
    3. Quarterback skill *modulates* (2.8, 29%).  ``qb_pressure_epa`` is the most
       direct measure of how he copes; sack avoidance next (sacks are the single
       most expensive dropback outcome); time to throw and release time are
       exposure measures and are collinear with each other (0.64), so release
       time gets half the weight.  ``qb_scramble_rate`` has the smallest weight
       and direction -1: a quarterback who escapes is *more* resilient to
       pressure, but scrambling is also a symptom of a collapsing pocket, so the
       sign is the one the brief wants and the weight is deliberately small.

Cluster shares of total weight (``CLUSTER_WEIGHT_SHARE``): protection 0.417,
opponent pressure 0.292 (of which the collinear trio 0.240), quarterback 0.292.

GAIN = 1.0
    All twelve inputs +1 SD in the risk direction (``z = 1``) gives
    **p = 0.34**; all +2 SD gives **p = 0.58**; the +3 SD clip is the ceiling
    at **p = 0.79**; all -1 SD gives p = 0.065 (0.068 in practice: a fully
    healthy line is only -0.7 SD on ``ol_injuries``, which cannot go below zero
    starters out).  Twelve inputs all one SD bad at once is a far-out scenario
    (the inputs are largely independent), so a one-in-three collapse chance
    there is deliberately restrained, and nothing the model can say reaches
    0.95.  GAIN was chosen for those plausibility anchors, not fitted to data.
    (On real 2023-2025 games the model's output ranged 0.06 to 0.46 with a
    median of 0.16.)

Applying it (the brief's formulas, coefficients named after the field)
----------------------------------------------------------------------
::

    qb_yards          *= (1 - p * 0.65)    QB_YARDS_COEFFICIENT
    qb_completion_pct *= (1 - p * 0.20)    QB_COMPLETION_PCT_COEFFICIENT
    qb_td_prob        *= (1 - p * 0.35)    QB_TD_PROB_COEFFICIENT
    qb_int_prob       *= (1 + p * 0.40)    QB_INT_PROB_COEFFICIENT

With ``recenter=True`` (default) each multiplier is divided by the same formula
at ``BASE_RATE`` so a league-average situation is an exact no-op (the historical
averages the engine starts from already contain an average amount of collapse).
``recenter=False`` is the literal formula above.  The collapse probability is
logged at DEBUG on ``nfl_pack.qb_collapse`` by both functions.

Caps: ``qb_completion_pct``, ``qb_td_prob`` and ``qb_int_prob`` are held to a
ceiling of 1.0 when the current value is <= 1.0 (fraction scale) and 100.0 when
it is > 1.5 (percent scale).  A value between 1.0 and 1.5 cannot be a
fraction-scale probability and is left uncapped (the contract's literal "<= 1.5
-> cap 1.0" would cut it or silently cancel a boost).  The upgrade pipeline
maps the engine's expected counts (``passing_tds``) onto ``qb_td_prob``, so
this band is not hypothetical.

Additions to the canonical vocabulary
-------------------------------------
* ``ol_health`` (1 - ol_injuries / 5) is accepted when ``ol_injuries`` is absent
  and is converted back to a starters-out count.
* Aliases: ``pass_block_win_rate``, ``ol_injury_count``, ``time_to_throw``,
  ``release_time``, ``scramble_rate``, ``sack_avoidance``, ``pass_rush_win_rate``.
* Out-of-domain numbers are *rejected, not clamped* and listed in the result's
  ``inputs_rejected``: a rate below 0 or above 100, a negative or >5 injury
  count, a non-positive or >6 s throw time, a sack-avoidance below 0.5 (that is
  almost certainly a sack *rate*).
* The result also carries ``cluster_effects`` (each cluster's signed share of the
  z-score) and ``event`` (the definition above).

HONEST LIMITS
-------------
* **Weights are priors, not fitted coefficients.**  There is no outcome-level
  data in this project to fit them against; the order above is a causal
  judgement.  The league means / SDs, in contrast, are measured (see each
  ``Factor.source``).
* **Modest signal.**  The sanity check below finds a real but modest
  association (AUC 0.63, Spearman -0.21 against game EPA), and five of the
  twelve inputs show no detectable game-level association at all.  Game-level
  quarterback EPA is mostly noise, so read the probability as a tilt around
  0.16, not as a forecast of a specific disaster.
* **Proxies are proxies.**  ``ol_pass_block_win_rate`` free data is
  ``1 - pressure allowed``, which is partly an *outcome* (and partly the
  quarterback's own doing), not a pure block-win measurement; PFF / ESPN
  pass-block and pass-rush win rates are on different scales (league means near
  0.6 and 0.4) and must be converted before being supplied, or they will read as
  extreme.  ``opp_pass_rush_win_rate`` free data is the same series as
  ``opp_pressure_rate`` (pressure includes sacks from 2023), so supplying both
  counts one fact twice by design (that is why the trio is down-weighted).
* **Season-level inputs, game-level event.**  Features describe a team's season,
  not tonight's matchup, scheme or weather; early-season values are noisy and
  ``ol_continuity`` runs high early because few injuries have happened yet.
* **No free live feed for ``ol_injuries``.**  The reference was measured from
  historical nflverse injury reports; the live number must come from the
  project's own injury file / player-status overlay or a user override
  (offline_data_contract rule 1).  Until supplied it is neutral.
* **Measurement regime.**  nflverse codes sacks and scrambles as pressure only
  from 2023, and participation-based throw times shifted that year, so every
  pressure/throw-time reference uses 2023-2025 only.
* A collapse is partly luck (a tipped ball, a fumble return).  No input set
  removes that, and the 0.16 base already contains it.

SANITY CHECK (honest, not tuned; scratch script on nflverse 2023-2025)
----------------------------------------------------------------------
Sample: 1,596 regular-season starter games, 269 collapses (16.9%).  For each
game the team and QB features were computed from the *other* games of that
season (leave-one-game-out, so a game never feeds its own features),
``ol_injuries`` came from the historical injury reports (season top-5 OL by
snaps; Out/Doubtful = 1, Questionable = 0.5), and this module scored it (mean
coverage 98%).  Weights, GAIN and BASE_RATE were fixed *before* the check and
were not touched after seeing it; only the league mean/SD standardisation was
measured on the same seasons.  It is not a forward backtest (the "other games"
include later ones): it asks only whether the ranking points the right way.

* Model p vs the game's EPA/dropback: Spearman -0.21 (95% bootstrap CI -0.25
  to -0.16); vs the collapse event +0.17.  AUC for the event 0.63 (CI 0.60 to
  0.67; 0.66 / 0.64 / 0.60 in 2023 / 2024 / 2025).  Brier 0.136 vs 0.140 for
  always saying 0.16: a small gain.
* By quintile of p (n = 319-320 each): predicted 0.10 / 0.13 / 0.16 / 0.19 /
  0.27, observed 0.09 / 0.12 / 0.16 / 0.21 / 0.27; mean predicted 0.168 vs
  observed 0.169; top / bottom quintile ratio 3.0.  Games with p >= 0.30 (n=59)
  collapsed 36% of the time, games with p <= 0.08 (n=48) 6%.  That agreement is
  better than untuned priors have a right to; treat it as one three-season
  sample, not proof that the weights are right.
* By input (Spearman against the event, signed so + means "riskier"): OL
  pass-block win rate +0.12, OL continuity +0.11, QB pressure EPA +0.13, QB
  sack avoidance +0.13, opponent pressure rate / pass-rush win rate +0.06,
  simulated pressure +0.06.  **No detectable association (n.s. at n ~ 1,500)**:
  OL injuries +0.04, opponent blitz rate +0.01, QB time to throw 0.00, QB
  release time -0.04 (wrong sign, n.s.), QB scramble rate +0.02.  Those
  inputs keep their prior weights (a larger sample, or a pre-game injury feed,
  might justify revisiting them); this module does not tune to the check.
* By cluster, alone: protection AUC 0.60, quarterback 0.60, opponent pressure
  0.55 (the pass rush is the weakest cluster at game level, which fits it
  carrying the smaller weight).
* Secondary, mechanically overlapping (the season includes the game):
  team-season pressure allowed vs collapse frequency, Spearman +0.39 (n=96).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from modules import nfl_pack_common as common
from modules.nfl_pack_common import Factor

LOG = logging.getLogger("nfl_pack.qb_collapse")

MODEL = "qb_collapse"
OUTPUT_KEY = "collapse_probability"
STAGE = "qb_collapse"

# --- the event ---------------------------------------------------------------
#: Starter game with EPA per dropback at or below this is a "collapse".
EVENT_EPA_THRESHOLD = -0.25
#: A starter game needs at least this many dropbacks to count.
EVENT_MIN_DROPBACKS = 20
#: Measured on nflverse pbp 2019-2025 regular season: 580 / 3,658 = 0.1586.
BASE_RATE = 0.16
#: Multiplies the weighted mean z-score (all inputs +1 SD -> p ~ 0.34).
GAIN = 1.0

EVENT_DEFINITION = (
    "Starting QB (most dropbacks for his team, at least 20) finishes a regular-season game with "
    "EPA per dropback <= -0.25; base rate 0.1586 (580 of 3,658 starter games, nflverse pbp 2019-2025)."
)

# --- the brief's coefficients, named after the field -------------------------
QB_YARDS_COEFFICIENT = 0.65
QB_COMPLETION_PCT_COEFFICIENT = 0.20
QB_TD_PROB_COEFFICIENT = 0.35
QB_INT_PROB_COEFFICIENT = 0.40

#: A probability-like field at or below this is read as a fraction (cap 1.0), above as a percent (cap 100).
_FRACTION_SCALE_LIMIT = 1.5

# --- factors (weights are priors; mean / sd are measured league references) --
_SRC_PBP_PART = "nflverse pbp + participation, REG 2023-2025"
FACTORS: tuple[Factor, ...] = (
    # Protection cluster (4.0).  The cause side of the matchup.
    Factor(
        "ol_pass_block_win_rate", mean=0.700, sd=0.039, weight=2.0, direction=-1,
        label="OL pass-block win rate", kind="rate", unit="%", aliases=("pass_block_win_rate",),
        source=f"{_SRC_PBP_PART}, team-seasons n=96: 1 - (pressure or sack per dropback allowed), mean 0.6997 sd 0.0386",
    ),
    Factor(
        "ol_injuries", mean=0.37, sd=0.54, weight=1.2, direction=+1,
        label="OL starters out", kind="value", unit="starters", aliases=("ol_injury_count",),
        source="nflverse injuries + snap counts, REG 2023-2025, team-games n=1632: season top-5 OL by snaps, "
               "Out/Doubtful=1 Questionable=0.5, mean 0.369 sd 0.541 (reference only; no free live feed)",
    ),
    Factor(
        "ol_continuity", mean=0.822, sd=0.073, weight=0.8, direction=-1,
        label="OL continuity", kind="rate", unit="%",
        source="nflverse snap counts, REG 2019-2025, team-seasons n=224: top-5 OL share of OL snaps, mean 0.8223 sd 0.0730",
    ),
    # Opponent pressure cluster (2.8).  Trio below is one measurement seen three ways.
    Factor(
        "opp_pressure_rate", mean=0.300, sd=0.036, weight=1.1, direction=+1,
        label="Opponent pressure rate", kind="rate", unit="%",
        source=f"{_SRC_PBP_PART}, defenses n=96: pressure or sack per dropback faced, mean 0.3004 sd 0.0360",
    ),
    Factor(
        "opp_pass_rush_win_rate", mean=0.300, sd=0.036, weight=0.7, direction=+1,
        label="Opponent pass-rush win rate", kind="rate", unit="%", aliases=("pass_rush_win_rate",),
        source=f"{_SRC_PBP_PART}, defenses n=96, PROXY (pressures + sacks per dropback; same series as pressure rate): "
               "mean 0.3004 sd 0.0360",
    ),
    Factor(
        "opp_simulated_pressure_success", mean=0.269, sd=0.036, weight=0.5, direction=+1,
        label="Opponent simulated-pressure success", kind="rate", unit="%",
        source=f"{_SRC_PBP_PART}, defenses n=96: pressure rate on dropbacks with <= 4 rushers, mean 0.2690 sd 0.0363",
    ),
    Factor(
        "opp_blitz_rate", mean=0.261, sd=0.067, weight=0.5, direction=+1,
        label="Opponent blitz rate", kind="rate", unit="%",
        source=f"{_SRC_PBP_PART}, defenses n=96: share of dropbacks with >= 5 rushers, mean 0.2614 sd 0.0667 "
               "(FTN n_blitzers>=1 reads 0.284)",
    ),
    # Quarterback cluster (2.8).  Modulates; the scramble rate is deliberately smallest.
    Factor(
        "qb_pressure_epa", mean=-0.447, sd=0.194, weight=1.0, direction=-1,
        label="QB EPA/dropback under pressure", kind="value", unit="EPA/db",
        source=f"{_SRC_PBP_PART}, QB-seasons >= 200 dropbacks and >= 40 pressured, n=117: mean -0.4474 sd 0.1937",
    ),
    Factor(
        "qb_sack_avoidance", mean=0.933, sd=0.023, weight=0.7, direction=-1,
        label="QB sack avoidance", kind="rate", unit="%", aliases=("sack_avoidance",),
        source="nflverse pbp, REG 2023-2025, QB-seasons >= 200 dropbacks, n=117: 1 - sacks/dropback, mean 0.9334 sd 0.0233",
    ),
    Factor(
        "qb_time_to_throw", mean=2.82, sd=0.157, weight=0.6, direction=+1,
        label="QB time to throw", kind="value", unit="s", aliases=("time_to_throw",),
        source="nflverse NGS passing avg_time_to_throw, REG 2023-2025, week 0, attempts >= 200, n=110: mean 2.824 sd 0.157",
    ),
    Factor(
        "qb_release_time", mean=2.34, sd=0.141, weight=0.3, direction=+1,
        label="QB release time", kind="value", unit="s", aliases=("release_time",),
        source=f"{_SRC_PBP_PART}, QB-seasons >= 200 dropbacks, n=117: median time_to_throw on unpressured dropbacks, "
               "mean 2.335 sd 0.141",
    ),
    Factor(
        "qb_scramble_rate", mean=0.052, sd=0.029, weight=0.2, direction=-1,
        label="QB scramble rate", kind="rate", unit="%", aliases=("scramble_rate",),
        source="nflverse pbp, REG 2023-2025, QB-seasons >= 200 dropbacks, n=117: scrambles/dropback, mean 0.0518 sd 0.0294",
    ),
)

#: Factor keys by causal cluster (used for the cluster shares and ``cluster_effects``).
CLUSTERS: dict[str, tuple[str, ...]] = {
    "protection": ("ol_pass_block_win_rate", "ol_injuries", "ol_continuity"),
    "opponent_pressure": (
        "opp_pressure_rate", "opp_pass_rush_win_rate", "opp_simulated_pressure_success", "opp_blitz_rate",
    ),
    "qb": ("qb_pressure_epa", "qb_sack_avoidance", "qb_time_to_throw", "qb_release_time", "qb_scramble_rate"),
}
#: The three strongly collinear opponent-pressure inputs.
PRESSURE_TRIO: tuple[str, ...] = ("opp_pressure_rate", "opp_pass_rush_win_rate", "opp_simulated_pressure_success")

_TOTAL_WEIGHT = sum(f.weight for f in FACTORS)
_WEIGHT_BY_KEY = {f.key: f.weight for f in FACTORS}
#: Each cluster's share of the total weight (sums to 1).
CLUSTER_WEIGHT_SHARE: dict[str, float] = {
    name: round(sum(_WEIGHT_BY_KEY[k] for k in keys) / _TOTAL_WEIGHT, 3) for name, keys in CLUSTERS.items()
}
#: The collinear trio's share of the total weight.
PRESSURE_TRIO_WEIGHT_SHARE: float = round(sum(_WEIGHT_BY_KEY[k] for k in PRESSURE_TRIO) / _TOTAL_WEIGHT, 3)

BASIS = (
    "Weighted z-score through a logistic: P(starting QB posts EPA/dropback <= -0.25 over >= 20 dropbacks), "
    "base rate 0.1586 (580 of 3,658 starter games, nflverse pbp 2019-2025, rounded to 0.16). "
    "Weights are PRIORS (causal ordering: protection, then opponent pressure, then QB skill), not fitted. "
    "League means/SDs are measured from nflverse 2023-2025 (OL continuity 2019-2025; see each factor's source); "
    "missing inputs are neutral and shrink the result toward the base rate."
)

# Physically impossible values are rejected (treated as missing), not clamped into plausibility.
# Rates are validated by ``to_rate`` (0-100%); this table adds the non-rate windows and one rate window.
_DOMAIN: dict[str, tuple[float, float]] = {
    "ol_injuries": (0.0, 5.0),
    "qb_time_to_throw": (1.0, 6.0),
    "qb_release_time": (1.0, 6.0),
    "qb_pressure_epa": (-8.0, 8.0),
    "qb_sack_avoidance": (0.5, 1.0),  # below 0.5 is a sack RATE passed by mistake
}


def _clean_inputs(inputs: Any) -> tuple[dict[str, float], list[dict[str, str]], dict[str, str]]:
    """``(clean, rejected, derived)``: canonical-key floats the scorer may use.

    Anything that is not a finite number is simply absent.  A number outside its
    physical domain is *rejected* (and reported), never clamped to a plausible value.
    """
    try:
        # ``is not None`` rather than truthiness: a pandas row (Series) has no boolean value
        data = dict(inputs) if inputs is not None else {}
    except (TypeError, ValueError):
        data = {}
    clean: dict[str, float] = {}
    rejected: list[dict[str, str]] = []
    derived: dict[str, str] = {}

    def _read(factor: Factor, key: str) -> float | None:
        if key not in data:
            return None
        raw = data[key]
        number = common.to_float(raw)
        if number is None:
            return None  # missing / not a number: not an error, just absent
        value = common.to_rate(raw) if factor.kind == "rate" else number
        window = _DOMAIN.get(factor.key)
        reason = None
        if value is None:
            reason = "rate outside 0-100%"
        elif window is not None and not (window[0] <= value <= window[1]):
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


def compute_qb_collapse(inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Probability that the quarterback has a collapse game (see the module docstring).

    ``inputs`` is a flat mapping of the twelve canonical keys (any subset; ``None``
    or ``{}`` returns the neutral reference).  Never raises on bad data.  Returns
    the standard pack result (``model``, ``collapse_probability``, ``probability``,
    ``reference_probability``, ``references``, ``z_score``, ``coverage``,
    ``inputs_used``, ``inputs_defaulted``, ``contributions``, ``drivers``,
    ``basis``) plus ``inputs_rejected``, ``inputs_derived``, ``cluster_effects``
    and ``event``.
    """
    clean, rejected, derived = _clean_inputs(inputs)
    result = common.run_factor_model(
        model=MODEL,
        primary_key=OUTPUT_KEY,
        inputs=clean,
        factors=FACTORS,
        base_rate=BASE_RATE,
        gain=GAIN,
        basis=BASIS,
    )
    contributions = result["contributions"]
    result["cluster_effects"] = {
        name: common.r(sum(contributions[k]["effect"] for k in keys if k in contributions), 4)
        for name, keys in CLUSTERS.items()
    }
    result["inputs_rejected"] = rejected
    result["inputs_derived"] = derived
    result["event"] = EVENT_DEFINITION
    result["basis"] = (
        f"{BASIS} Inputs supplied: {len(result['inputs_used'])} of {len(FACTORS)} "
        f"(coverage {result['coverage']:.0%})."
    )
    LOG.debug(
        "qb_collapse: collapse_probability=%.4f (reference %.4f, z=%+.3f, coverage=%.0f%%, rejected=%d) clusters=%s",
        result[OUTPUT_KEY],
        result["reference_probability"],
        result["z_score"],
        100.0 * result["coverage"],
        len(rejected),
        result["cluster_effects"],
    )
    return result


def _result_probability(result: Any) -> float | None:
    if not isinstance(result, Mapping):
        return None
    value = common.to_float(result.get(OUTPUT_KEY))
    if value is None:
        value = common.to_float(result.get("probability"))
    return None if value is None else common.clamp(value, 0.0, 1.0)


def _result_reference(result: Any) -> float:
    if isinstance(result, Mapping):
        references = result.get("references")
        value = common.to_float(references.get(OUTPUT_KEY)) if isinstance(references, Mapping) else None
        if value is None:
            value = common.to_float(result.get("reference_probability"))
        if value is not None:
            return common.clamp(value, 0.0, 1.0)
    return BASE_RATE


def _probability_cap(value: float) -> float | None:
    """Ceiling for a probability-like field given its *current* value, or ``None`` for no ceiling.

    * ``value <= 1.0``: fraction scale, ceiling 1.0.
    * ``1.0 < value <= 1.5``: ambiguous.  It cannot be a fraction-scale probability, and as a
      percent-scale rate (1.2%) or an expected count (1.3 TDs) it is nowhere near its own
      ceiling, so no ceiling is enforced.  Capping at 1.0 here would cut the value, and
      capping at the value itself would silently cancel a boost.
    * ``value > 1.5``: percent scale, ceiling 100.0 (never below the value itself).
    """
    if value <= 1.0:
        return 1.0
    if value <= _FRACTION_SCALE_LIMIT:
        return None
    return max(100.0, value)


#: (field, coefficient, direction, capped at the probability ceiling)
_APPLY_SPEC: tuple[tuple[str, float, int, bool], ...] = (
    ("qb_yards", QB_YARDS_COEFFICIENT, -1, False),
    ("qb_completion_pct", QB_COMPLETION_PCT_COEFFICIENT, -1, True),
    ("qb_td_prob", QB_TD_PROB_COEFFICIENT, -1, True),
    ("qb_int_prob", QB_INT_PROB_COEFFICIENT, +1, True),
)


def apply_qb_collapse(
    projection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    recenter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the collapse probability in ``result`` to a quarterback projection.

    ``qb_yards *= 1 - p*0.65``, ``qb_completion_pct *= 1 - p*0.20``,
    ``qb_td_prob *= 1 - p*0.35``, ``qb_int_prob *= 1 + p*0.40``.  Returns
    ``(new_projection, adjustments)``; ``projection`` is never mutated.  With
    ``recenter=True`` each multiplier is divided by the same formula at the
    model's reference probability (a league-average game is a no-op);
    ``recenter=False`` is the literal brief formula.  A field that is missing or
    not numeric is recorded as ``applied=False``.  Completion percentage and the
    two probabilities are capped at 1.0 (100.0 when supplied on a percent scale);
    a value between 1.0 and 1.5 (not a fraction-scale probability) is left
    uncapped so a boost is never silently cancelled.
    """
    new_projection: dict[str, Any] = dict(projection) if isinstance(projection, Mapping) else {}
    adjustments: list[dict[str, Any]] = []
    probability = _result_probability(result)
    reference = _result_reference(result)
    for field, coefficient, direction, capped in _APPLY_SPEC:
        if probability is None:
            adjustments.append(
                {"stage": STAGE, "field": field, "applied": False, "reason": "no valid collapse_probability in result"}
            )
            continue
        before = common.to_float(new_projection.get(field))
        cap = _probability_cap(before) if (capped and before is not None) else None
        common.scale_field(
            new_projection,
            field,
            stage=STAGE,
            probability=probability,
            coefficient=coefficient,
            direction=direction,
            reference=reference,
            recenter=recenter,
            adjustments=adjustments,
            cap=cap,
        )
    LOG.debug(
        "qb_collapse apply: collapse_probability=%s (reference %.4f, recenter=%s) %s",
        "n/a" if probability is None else f"{probability:.4f}",
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
    "CLUSTERS",
    "CLUSTER_WEIGHT_SHARE",
    "EVENT_DEFINITION",
    "EVENT_EPA_THRESHOLD",
    "EVENT_MIN_DROPBACKS",
    "FACTORS",
    "GAIN",
    "MODEL",
    "OUTPUT_KEY",
    "PRESSURE_TRIO",
    "PRESSURE_TRIO_WEIGHT_SHARE",
    "QB_COMPLETION_PCT_COEFFICIENT",
    "QB_INT_PROB_COEFFICIENT",
    "QB_TD_PROB_COEFFICIENT",
    "QB_YARDS_COEFFICIENT",
    "apply_qb_collapse",
    "compute_qb_collapse",
]
