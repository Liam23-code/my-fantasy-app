"""Pass-game collapse probability for the NFL model upgrade pack.

``compute_pass_game_collapse`` turns eleven quarterback / protection / pass-rush /
coverage / receiver inputs into ``pass_game_collapse`` in ``[0, 1]``;
``apply_pass_game_collapse`` applies the brief's haircuts to a WR or TE projection.
Pure, offline and deterministic: the module imports only the standard library and
:mod:`modules.nfl_pack_common`.

EVENT DEFINITION (what ``pass_game_collapse = 0.12`` means)
-----------------------------------------------------------
A **pass-game collapse** is a regular-season team-game in which the team's gross
passing yards are **<= 0.70 x its own average over its other regular-season games
that season** AND its **team passing EPA is below zero** (nflverse team stats
``passing_yards`` / ``passing_epa``; sacks are inside the EPA).  In plain words: the
receivers collectively produced 30% less than usual *and* it was an actually bad
passing day, not a game the team won by running out the clock.

* Measured base rate: **0.1184 (625 of 5,278 team-games, nflverse team stats
  2016-2025)**; 0.1201 (511 of 4,254) over 2018-2025; by season 0.103 (2022) to
  0.140 (2023).  ``BASE_RATE = 0.12``.
* The user's suggested 0.65 x threshold gives 0.093 (n = 5,278), a little under
  the 10-18% band the brief asked for, so 0.70 was used.  The threshold was chosen
  for a recognisable, common event ("about one team-game in eight"); it was not
  tuned to any model output.
* ``p = 0.12`` is therefore "an average team-game"; ``p = 0.25`` is "about one
  game in four, roughly twice the norm".

THE MODEL
---------
A transparent weighted z-score through a logistic (``nfl_pack_common``)::

    z = sum(weight_i * direction_i * clip(z_i)) / sum(weight over ALL eleven)
    pass_game_collapse = sigmoid(logit(BASE_RATE) + GAIN * z)

An input the caller did not supply is neutral (``z_i = 0``) but still counts in the
denominator, so missing data shrinks the answer toward ``BASE_RATE``.  The function
is called once per player: a WR call legitimately carries no ``te_*`` inputs and a
TE call no ``wr_*`` inputs; those are neutral *by construction* (``role_coverage``
in the result says how much of each group was supplied).

Inputs (direction = effect on collapse risk; weights are PRIORS)
----------------------------------------------------------------
============================== ==== ====== ================= ===========================
input                          dir  weight league mean / sd  free feed
============================== ==== ====== ================= ===========================
qb_collapse_probability        +1   3.0    0.16  / 0.06      output of qb_collapse_model
opp_coverage_grade             +1   1.8    50    / 28.9      proxy: EPA/dropback allowed
ol_pass_block_win_rate         -1   1.5    0.700 / 0.039     proxy: 1 - pressure allowed
opp_pressure_rate              +1   1.5    0.300 / 0.036     pbp + participation
wr_expected_yprr_vs_coverage   -1   1.2    1.45  / 0.61      participation + pbp (proxy)
wr_separation                  -1   1.0    3.05  / 0.40      NGS receiving
ol_injuries (starters out)     +1   0.8    0.37  / 0.54      NO live feed (see below)
opp_cb1_shadow_rate            +1   0.8    0.15  / 0.15      NO free feed (prior)
te_route_depth                 +1   0.6    7.0   / 1.9       PFR advanced receiving
opp_bracket_coverage_rate      +1   0.6    0.08  / 0.05      NO free feed (prior)
te_target_share_under_pressure -1   0.5    0.19  / 0.055     pbp + participation
============================== ==== ====== ================= ===========================

``wr_expected_yprr_vs_coverage`` is the receiver's man / zone split resolved against
*this* defense: ``nfl_pack_common.blend_man_zone(wr_yprr_vs_man, wr_yprr_vs_zone,
opp_man_coverage_rate)`` (the helper assumes a 0.30 man rate when the opponent's is
missing).  A caller that has already done that supplies the number directly as
``wr_expected_yprr_vs_coverage`` and it wins over the split.

Why this order (a prior judgement, not a fit)
    1. *The quarterback is the passing game's engine*, so his collapse probability
       is the largest single weight (3.0 of 13.3, 23%).  It already contains the
       protection / pressure story that produced it.
    2. *Opponent coverage grade* (1.8) is the broad "how good is the pass defense
       the receivers actually face" summary and the one opponent-quality term the
       QB model's inputs do not contain, so it outranks any single protection input.
    3. *The two direct pressure inputs* (pass-block win rate 1.5, opponent pressure
       rate 1.5) are the same mechanism seen from both sides of the line.  They are
       kept below coverage and below the QB term because pressure is already inside
       ``qb_collapse_probability``: the brief asks for both, and the partial double
       count is deliberate, so each gets a moderate weight rather than a large one.
    4. *Receiver resilience* modulates: expected YPRR against this coverage mix (1.2)
       over raw separation (1.0), because the first already folds in the man / zone
       exposure and the two are collinear.
    5. *Acute news and mostly-redistributive coverage tweaks* come next: OL starters
       out (0.8; partly captured by the win-rate proxy and by the QB model), CB1
       shadowing (0.8; it relocates the WR1's targets more than it removes them and
       touches one receiver), then TE route depth (0.6; deeper routes are more
       exposed to a collapsing pocket), bracket coverage (0.6; it takes away the top
       target, not the offense) and the TE's share of pressured targets (0.5; the
       outlet mechanism, and the data below shows the league-wide effect is nil).

GAIN = 0.6
    With every one of the eleven inputs +1 SD in the risk direction (``z = 1``) the
    probability is **0.199** (about 1.7x the 0.12 base); all +2 SD gives **0.307**
    (the grade input saturates at 100, which is +1.73 SD, so it is held there); all
    -1 SD gives **0.072**.  The highest value any input set can reach (every input at
    its +3 SD clip or its physical limit) is **0.418**, so nothing the model can say
    approaches 0.95.  A single role call reaches slightly less: a WR call cannot
    carry the two ``te_*`` inputs (``z`` tops out at 0.917 of the full-coverage
    value; all +1 SD gives 0.191, ceiling 0.383) and a TE call cannot carry the two
    ``wr_*`` inputs (0.835; 0.184, ceiling 0.356).  GAIN was chosen for those
    plausibility anchors and then *cross-checked, not fitted*: in the 213 team-games
    of 2018-2025 where the three free proxies average at least +1 SD bad the event
    happened 20.2% of the time (model: 0.199), and in the 347 games where they
    average at least 1 SD good it happened 8.6% (model: 0.072).  The logistic slope
    of the event on that composite is 0.41 (95% CI 0.26-0.55) over 2018-2025 and 0.48
    (0.25-0.72) over 2023-2025, so 0.6 sits on the generous side of the measured
    proxy slope; the margin is for the inputs that cannot be checked offline (QB
    collapse, CB1 shadow, bracket, the WR / TE inputs).

Applying it (the brief's formulas, coefficients named after the field)
----------------------------------------------------------------------
::

    wr_yards   *= (1 - p * 0.55)   WR_YARDS_COEFFICIENT
    wr_td_prob *= (1 - p * 0.40)   WR_TD_PROB_COEFFICIENT
    te_yards   *= (1 - p * 0.45)   TE_YARDS_COEFFICIENT
    te_td_prob *= (1 - p * 0.35)   TE_TD_PROB_COEFFICIENT

With ``recenter=True`` (default) each multiplier is divided by the same formula at
``BASE_RATE`` so a league-average game is an exact no-op (the historical averages the
engine starts from already contain an average amount of collapse).
``recenter=False`` is the literal formula above.  Whichever of the ``wr_*`` /
``te_*`` fields are present in the projection are written (both if both); an absent
field is recorded ``applied=False``.  The two probabilities are capped at 1.0 (100.0
when supplied on a percent scale).

Additions to the canonical vocabulary (model-private, optional)
---------------------------------------------------------------
* ``wr_expected_yprr_vs_coverage`` (see above), ``wr_yprr_vs_man`` /
  ``wr_yprr_vs_zone`` / ``opp_man_coverage_rate`` as the split that produces it.
* ``qb_collapse_reference_probability``: the QB model's own ``reference_probability``.
  ``qb_collapse_probability`` is judged against 0.16 (``QB_COLLAPSE_REFERENCE``, the
  QB model's base rate at the time of writing); an orchestrator that passes the QB
  result's reference keeps a *neutral QB exactly neutral here* even if that base
  rate is retuned.  Accepted range 0.01-0.60.
* ``ol_health`` (1 - ol_injuries / 5) is accepted when ``ol_injuries`` is absent.
* Aliases: ``collapse_probability``, ``qb_collapse_prob``, ``pass_block_win_rate``,
  ``ol_injury_count``, ``opponent_pressure_rate``, ``opponent_coverage_grade``,
  ``cb1_shadow_rate``, ``bracket_coverage_rate``, ``wr1_separation``,
  ``te_adot``, ``te_pressure_target_share``.
* Out-of-domain numbers are *rejected, not clamped*, and listed in
  ``inputs_rejected``: a rate below 0 or above 100, a negative or >5 injury count, a
  grade outside 0-100, a negative or >10 yards-per-route, a separation outside 0-15
  yards.
* The result also carries ``event`` (the definition above), ``inputs_derived``,
  ``role_coverage`` and ``qb_collapse_reference``.

HONEST LIMITS
-------------
* **Weights are priors, not fitted coefficients.**  There is no outcome-level data
  in this project to fit them against; the order above is a causal judgement.  The
  league means / SDs, in contrast, are measured where a free feed exists (each
  ``Factor.source`` names the feed, seasons and n) and are priors otherwise.
* **The signal is weak.**  See the SANITY CHECK below: the free-data inputs rank
  team-games better than chance (AUC about 0.58) and barely improve the Brier score
  (0.1050 against 0.1057 for the constant 0.12).  A team passing collapse is mostly
  a one-game accident -- a team's collapse frequency does not even persist from
  one season to the next (year-to-year Spearman +0.04) -- so read the probability
  as a mild tilt around 0.12, not a forecast of a specific disaster.
* **Proxies are proxies.**  ``ol_pass_block_win_rate`` free data is
  ``1 - pressure allowed`` per dropback, partly an *outcome* and partly the
  quarterback's doing, not a charted block win; a PFF / ESPN pass-block win rate sits
  near 0.6 and must be converted before being supplied or it reads as a collapsing
  line.  ``opp_coverage_grade`` free data is the percentile of EPA/dropback allowed,
  which is pass rush and coverage together; it is not a pure coverage grade, and a
  percentile given as a 0-1 fraction would read as the 1st percentile.
  ``wr_*_yprr`` free data is yards per *dropback the receiver was on the field for*
  (a stand-in for routes run), and it ran about 0.25 lower in 2023-2025 than in
  2018-2022, so it is referenced to 2023-2025 only.  The opponents' man-coverage
  rate averaged 0.40 over 2023-2025 (team-season sd 0.11), well above the 0.30 the
  blend helper assumes when ``opp_man_coverage_rate`` is not supplied.
* **No free feed for two inputs.**  ``opp_cb1_shadow_rate`` and
  ``opp_bracket_coverage_rate`` need charting; they are accepted and neutral until a
  caller supplies them, and their references are labelled priors.  ``ol_injuries``
  has no *live* feed either (offline_data_contract rule 1): its reference was
  measured from historical reports, the live count must come from the project's own
  injury file / player-status overlay or a user override.
* **The QB term lives on the QB model's scale, and its weight is the least
  supported number here.**  ``qb_collapse_probability`` is judged against a 0.16
  reference (that model's neutral output; its starter-game median was 0.159) and
  an sd of 0.06 (measured on that model's 2023-2025 output, see the factor's
  ``source``); see ``qb_collapse_reference_probability``.  The brief makes it the
  largest weight, and the weight of 3.0 is a prior, not something the data picks:
  on 1,596 team-games the QB model's output alone ranks collapses at AUC 0.56
  (Spearman +0.07, p = 0.008), *less* well than opponent coverage grade alone
  (0.585), and adding it to the free-data composite does not raise the AUC (0.584
  -> 0.574).  It is largely built from the same protection / pressure inputs the
  other terms carry, so counting both is a partial double count by design.
* **A team event applied to one player.**  The event is a *team* passing collapse;
  the brief applies it to a single WR or TE.  The brief's coefficients, not this
  model, decide how much of a team collapse lands on one receiver.
* **Season-level inputs, game-level event.**  Features describe a team's season,
  not tonight's matchup, scheme or weather; early-season values are noisy.

SANITY CHECK (honest, not tuned; scratch script, nflverse regular seasons)
--------------------------------------------------------------------------
Unit: team-game, 2018-2025, n = 4,254, base 0.1201 (2023-2025: n = 1,632, 0.1305).
Features are leave-one-game-out season values (the game being scored is excluded
from its own inputs): the team's pressure allowed, the opponent's pressure
generated and EPA/dropback allowed (as a percentile), and the season's top-five OL
starters out from injury reports.  Spearman is feature-in-risk-direction vs the
event; AUC is the probability a collapse game outranks a non-collapse game.

====================================  =========  =====  =========  =====
feature (risk direction)              Spearman   AUC    2023-25 S  AUC
====================================  =========  =====  =========  =====
opp_coverage_grade (higher D)         +0.075     0.566  +0.104     0.589
team pressure allowed                 +0.051     0.545  +0.041     0.535
opponent pressure generated           +0.042     0.537  +0.064     0.555
OL starters out (n=4,222 / 1,632)     +0.029     0.522  +0.012     0.509
this module, 4 inputs, 5.6 of 13.3    +0.090     0.580  +0.106     0.591
====================================  =========  =====  =========  =====

* The module's score by quintile: mean predicted 0.099 / 0.110 / 0.119 / 0.128 /
  0.142 against observed 0.085 / 0.102 / 0.106 / 0.140 / 0.168 (2018-2025).  The
  ordering is right and the observed spread is wider than predicted; with only 42%
  of the weight supplied the score is compressed by design.
* **``ol_injuries`` shows no measurable association** (Spearman +0.03, p = 0.06
  pooled; +0.01 in 2023-2025).  Its weight is a prior kept because a missing
  starter is news a season-average proxy cannot see, not because this data backs it.
* **The QB term, checked on the games where the QB model has an output** (2023-2025
  starter-games, n = 1,596, base 0.125, 199 collapses; ``qb_collapse_model`` run
  on leave-one-game-out features, then joined to the team-game event):

  ===================================  =========  =====
  feature (risk direction)             Spearman   AUC
  ===================================  =========  =====
  qb_collapse_probability              +0.066     0.558
  opp_coverage_grade                   +0.098     0.585
  4-input composite (no QB term)       +0.097     0.584
  same + qb_collapse_probability       +0.085     0.574
  ===================================  =========  =====

  The QB model's number does carry signal about a *team* passing collapse (the
  quintile event rates run 0.103 / 0.094 / 0.122 / 0.154 / 0.150), but it is no
  better than the opponent grade and overlaps the protection / pressure inputs, so
  the composite does not improve when it is added.  Brier 0.1083 against 0.1091
  for the constant 0.125 (0.1092 for 0.12): negligible.  Do not read the 3.0
  weight as validated.
* **Receivers, team-season level** (the event is a team event and the inputs are
  individual, so this is only a sign check; team-season collapse frequency is
  itself noisy, about 17 games and two events each): targets-weighted team WR
  separation (NGS receiving, WR with >= 30 targets, 2018-2025, n = 247 team-seasons)
  against collapse frequency has Spearman +0.10 in the risk direction (lower
  separation, more collapses; p = 0.11, not significant); the top TE's aDOT (PFR,
  >= 30 targets, n = 246) has Spearman -0.03 (p = 0.64), i.e. nothing.  The TE
  route-depth weight is therefore a pure prior.
* Nothing involving ``opp_cb1_shadow_rate``, ``opp_bracket_coverage_rate``, the
  man / zone split or the TE pressured-target share could be checked offline.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Mapping

from modules import nfl_pack_common as common
from modules.nfl_pack_common import Factor

LOG = logging.getLogger("nfl_pack.pass_game_collapse")

MODEL = "pass_game_collapse"
OUTPUT_KEY = "pass_game_collapse"
STAGE = "pass_game_collapse"

# --- the event ---------------------------------------------------------------
#: A team-game is a collapse at or below this fraction of the team's other-game passing-yard average ...
EVENT_YARDS_FRACTION = 0.70
#: ... when its team passing EPA is also below this.
EVENT_EPA_CEILING = 0.0
#: Measured on nflverse team stats, regular season 2016-2025: 625 / 5,278 = 0.1184 (2018-2025: 511 / 4,254 = 0.1201).
BASE_RATE = 0.12
#: Multiplies the weighted mean z-score.  All eleven inputs +1 SD bad -> p = 0.199; +2 SD -> 0.307 (the grade input
#: saturates at 100 = +1.73 SD); the practical ceiling with every input at its +3 SD clip or physical limit is 0.418.
GAIN = 0.6

EVENT_DEFINITION = (
    "Team pass-game collapse: a regular-season team-game with gross passing yards <= 0.70 x the team's average over "
    "its other regular-season games that season AND team passing EPA < 0; base rate 0.1184 (625 of 5,278 team-games, "
    "nflverse team stats 2016-2025; 0.1201 over 2018-2025), rounded to 0.12."
)

# --- the brief's coefficients, named after the field -------------------------
WR_YARDS_COEFFICIENT = 0.55
WR_TD_PROB_COEFFICIENT = 0.40
TE_YARDS_COEFFICIENT = 0.45
TE_TD_PROB_COEFFICIENT = 0.35

#: Field -> brief coefficient (the same numbers as the constants above).
COEFFICIENTS: dict[str, float] = {
    "wr_yards": WR_YARDS_COEFFICIENT,
    "wr_td_prob": WR_TD_PROB_COEFFICIENT,
    "te_yards": TE_YARDS_COEFFICIENT,
    "te_td_prob": TE_TD_PROB_COEFFICIENT,
}

#: A probability-like field at or below this is read as a fraction (cap 1.0), above as a percent (cap 100).
_FRACTION_SCALE_LIMIT = 1.5

#: The collapse probability the QB model returns for a league-average quarterback (its BASE_RATE).
QB_COLLAPSE_REFERENCE = 0.16

# --- factors (weights are priors; mean / sd are measured league references where a feed exists) ---
_SRC_PBP_PART = "nflverse pbp + participation, REG 2023-2025"
FACTORS: tuple[Factor, ...] = (
    Factor(
        "qb_collapse_probability", mean=QB_COLLAPSE_REFERENCE, sd=0.06, weight=3.0, direction=+1,
        label="QB collapse probability", kind="rate", unit="%", aliases=("collapse_probability", "qb_collapse_prob"),
        source="qb_collapse_model's own output, nflverse REG 2023-2025 starter-games n=1596 (leave-one-game-out "
               "features, 98% input coverage): median 0.159, mean 0.168, sd 0.063 game-level (0.055 across "
               "QB-season means, n=194); sd rounded to 0.06; mean is that model's BASE_RATE 0.16 (its neutral "
               "output) so a neutral QB is neutral here",
    ),
    Factor(
        "opp_coverage_grade", mean=50.0, sd=28.9, weight=1.8, direction=+1,
        label="Opponent coverage grade", kind="value", unit="pctile", aliases=("opponent_coverage_grade",),
        source="definitional: uniform team percentile rank 0-100 (mean 50, sd 28.9); free-data proxy upstream is "
               "the percentile of EPA/dropback allowed",
    ),
    Factor(
        "ol_pass_block_win_rate", mean=0.700, sd=0.039, weight=1.5, direction=-1,
        label="OL pass-block win rate", kind="rate", unit="%", aliases=("pass_block_win_rate",),
        source=f"{_SRC_PBP_PART}, team-seasons n=96: 1 - was_pressure rate allowed per dropback, mean 0.6997 "
               "sd 0.0386 (PROXY for a charted block win rate, which sits near 0.6 and must be converted)",
    ),
    Factor(
        "opp_pressure_rate", mean=0.300, sd=0.036, weight=1.5, direction=+1,
        label="Opponent pressure rate", kind="rate", unit="%", aliases=("opponent_pressure_rate",),
        source=f"{_SRC_PBP_PART}, defenses n=96: was_pressure rate per dropback faced, mean 0.3004 sd 0.0361",
    ),
    Factor(
        "wr_expected_yprr_vs_coverage", mean=1.45, sd=0.61, weight=1.2, direction=-1,
        label="WR expected YPRR vs this coverage", kind="value", unit="yds/route",
        aliases=("expected_yprr_vs_coverage",),
        source=f"{_SRC_PBP_PART} + rosters, WR-seasons with >= 40 man and >= 100 zone dropbacks on the field, n=380: "
               "receiving yards per dropback-on-field, 0.30 man / 0.70 zone blend, mean 1.447 sd 0.613 "
               "(PROXY: dropbacks on the field, not charted routes run)",
    ),
    Factor(
        "wr_separation", mean=3.05, sd=0.40, weight=1.0, direction=-1,
        label="WR separation", kind="value", unit="yds", aliases=("wr1_separation",),
        source="nflverse NGS receiving avg_separation as emitted by nfl_player_features, REG 2023-2025, receivers "
               "with >= 20 targets, n=396: mean 3.05 sd 0.40 (re-pointed to the loader's population; the original "
               "REG 2016-2025 WR >= 50 targets reading was mean 2.850 sd 0.483)",
    ),
    Factor(
        "ol_injuries", mean=0.37, sd=0.54, weight=0.8, direction=+1,
        label="OL starters out", kind="value", unit="starters", aliases=("ol_injury_count",),
        source="nflverse injuries + snap counts, REG 2023-2025, team-games n=1632: season top-5 OL by snaps, "
               "Out/Doubtful=1 Questionable=0.5, mean 0.369 sd 0.541 (2018-2025 n=4254: 0.344 / 0.512); "
               "reference only, no free live feed",
    ),
    Factor(
        "opp_cb1_shadow_rate", mean=0.15, sd=0.15, weight=0.8, direction=+1,
        label="Opponent CB1 shadow rate", kind="rate", unit="%", aliases=("cb1_shadow_rate",),
        source="prior: no free feed (needs charting); a skewed rate, most defenses shadow rarely and a few often",
    ),
    Factor(
        "te_route_depth", mean=7.0, sd=1.9, weight=0.6, direction=+1,
        label="TE route depth (aDOT)", kind="value", unit="yds", aliases=("te_adot",),
        source="nflverse PFR advanced receiving adot, REG 2018-2025, TE with >= 30 targets, n=352: "
               "mean 7.007 sd 1.886",
    ),
    Factor(
        "opp_bracket_coverage_rate", mean=0.08, sd=0.05, weight=0.6, direction=+1,
        label="Opponent bracket-coverage rate", kind="rate", unit="%", aliases=("bracket_coverage_rate",),
        source="prior: no free feed (needs charting)",
    ),
    Factor(
        "te_target_share_under_pressure", mean=0.19, sd=0.055, weight=0.5, direction=-1,
        label="TE share of pressured targets", kind="rate", unit="%", aliases=("te_pressure_target_share",),
        source=f"{_SRC_PBP_PART} + rosters via nfl_player_features, REG 2023-2025, TEs with >= 30 team pressured "
               "targets while on the field, n=184: share of the team's pressured pass targets WHILE ON THE FIELD, mean "
               "0.193 sd 0.026 after shrinkage (the original all-snaps TE1 reading was mean 0.1420 sd 0.0549; the loader's "
               "on-field denominator is smaller, so its share is higher -- the reference follows what the loader emits)",
    ),
)

#: Inputs the pack cannot derive from a free public feed: accepted, neutral until a caller supplies them.
NO_FREE_FEED_INPUTS: tuple[str, ...] = ("opp_cb1_shadow_rate", "opp_bracket_coverage_rate")

#: Factor keys by who they describe.
ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "team": (
        "qb_collapse_probability", "opp_coverage_grade", "ol_pass_block_win_rate", "opp_pressure_rate",
        "ol_injuries", "opp_cb1_shadow_rate", "opp_bracket_coverage_rate",
    ),
    "wr": ("wr_expected_yprr_vs_coverage", "wr_separation"),
    "te": ("te_route_depth", "te_target_share_under_pressure"),
}

_TOTAL_WEIGHT = sum(f.weight for f in FACTORS)
_WEIGHT_BY_KEY = {f.key: f.weight for f in FACTORS}

BASIS = (
    "Weighted z-score through a logistic: P(team passing yards <= 0.70 x its season average over other games AND "
    "team passing EPA < 0), base rate 0.1184 (625 of 5,278 team-games, nflverse team stats 2016-2025, rounded to "
    "0.12). Weights are PRIORS (causal ordering: QB collapse, opponent coverage, pressure, receiver resilience), "
    "not fitted. League means/SDs are measured from nflverse where a free feed exists (see each factor's source) "
    "and priors otherwise (CB1 shadow rate and bracket-coverage rate have no free feed and are neutral until "
    "supplied); missing inputs are neutral and shrink the result toward the base rate."
)

# Physically impossible values are rejected (treated as missing), not clamped into plausibility.
# Rates are validated by ``to_rate`` (0-100%); this table adds the non-rate windows.
_DOMAIN: dict[str, tuple[float, float]] = {
    "ol_injuries": (0.0, 5.0),
    "opp_coverage_grade": (0.0, 100.0),
    "wr_separation": (0.0, 15.0),
    "te_route_depth": (-5.0, 40.0),
    "wr_expected_yprr_vs_coverage": (0.0, 10.0),
    "wr_yprr_vs_man": (0.0, 10.0),
    "wr_yprr_vs_zone": (0.0, 10.0),
}
_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "wr_yprr_vs_man": ("yprr_vs_man",),
    "wr_yprr_vs_zone": ("yprr_vs_zone",),
}
_QB_REFERENCE_WINDOW = (0.01, 0.60)


def _read_checked(
    data: Mapping[str, Any],
    key: str,
    kind: str,
    window: tuple[float, float] | None,
    rejected: list[dict[str, str]],
) -> float | None:
    """One value from ``data[key]``: ``None`` if absent / not a number, rejected (and reported) if out of domain."""
    if key not in data:
        return None
    raw = data[key]
    number = common.to_float(raw)
    if number is None:
        return None  # missing / not a number is a normal state, not an error
    value = common.to_rate(raw) if kind == "rate" else number
    reason = None
    if value is None:
        reason = "rate outside 0-100%"
    elif window is not None and not (window[0] <= value <= window[1]):
        reason = f"outside the possible range {window[0]:g} to {window[1]:g}"
    if reason is not None:
        rejected.append({"key": key, "value": repr(raw)[:40], "reason": reason})
        return None
    return value


def _clean_inputs(inputs: Any) -> tuple[dict[str, float], list[dict[str, str]], dict[str, str], float | None]:
    """``(clean, rejected, derived, qb_reference)``: canonical-key floats the scorer may use.

    Anything that is not a finite number is simply absent.  A number outside its
    physical domain is *rejected* (and reported), never clamped to a plausible value.
    ``qb_reference`` is the caller's reference for the QB probability, or ``None``.
    """
    if isinstance(inputs, Mapping):
        data: Mapping[str, Any] = inputs
    else:
        try:  # a pandas Series, a list of pairs ...; anything else is "no inputs"
            data = dict(inputs) if inputs is not None else {}
        except (TypeError, ValueError):
            data = {}
    clean: dict[str, float] = {}
    rejected: list[dict[str, str]] = []
    derived: dict[str, str] = {}

    for factor in FACTORS:
        for key in (factor.key, *factor.aliases):
            value = _read_checked(data, key, factor.kind, _DOMAIN.get(factor.key), rejected)
            if value is not None:
                clean[factor.key] = value
                if key != factor.key:
                    derived[factor.key] = f"alias {key}"
                break

    if "ol_injuries" not in clean and "ol_health" in data:
        health = common.to_rate(data["ol_health"])
        if health is not None:
            clean["ol_injuries"] = round(5.0 * (1.0 - health), 4)
            derived["ol_injuries"] = "5 * (1 - ol_health)"
        elif common.to_float(data["ol_health"]) is not None:
            rejected.append({"key": "ol_health", "value": repr(data["ol_health"])[:40], "reason": "rate outside 0-100%"})

    # The WR's man / zone split resolved against this defense's coverage mix.
    if "wr_expected_yprr_vs_coverage" not in clean:
        sides: dict[str, float | None] = {}
        for split_key, split_aliases in _SPLIT_ALIASES.items():
            sides[split_key] = None
            for key in (split_key, *split_aliases):
                value = _read_checked(data, key, "value", _DOMAIN[split_key], rejected)
                if value is not None:
                    sides[split_key] = value
                    break
        man, zone = sides["wr_yprr_vs_man"], sides["wr_yprr_vs_zone"]
        blended = common.blend_man_zone(man, zone, data.get("opp_man_coverage_rate"))
        if blended is not None:
            clean["wr_expected_yprr_vs_coverage"] = blended
            man_rate = common.to_rate(data.get("opp_man_coverage_rate"))
            if man is None:
                derived["wr_expected_yprr_vs_coverage"] = "wr_yprr_vs_zone only (no man split supplied)"
            elif zone is None:
                derived["wr_expected_yprr_vs_coverage"] = "wr_yprr_vs_man only (no zone split supplied)"
            elif man_rate is None:
                derived["wr_expected_yprr_vs_coverage"] = (
                    "blend_man_zone(wr_yprr_vs_man, wr_yprr_vs_zone) at the assumed 0.30 man rate "
                    "(opp_man_coverage_rate not supplied)"
                )
            else:
                derived["wr_expected_yprr_vs_coverage"] = (
                    f"blend_man_zone(wr_yprr_vs_man, wr_yprr_vs_zone) at opp_man_coverage_rate {man_rate:.3f}"
                )

    qb_reference = _read_checked(data, "qb_collapse_reference_probability", "rate", _QB_REFERENCE_WINDOW, rejected)
    return clean, rejected, derived, qb_reference


def _factors_for(qb_reference: float | None) -> tuple[Factor, ...]:
    """``FACTORS``, with the QB factor's mean replaced by the caller's reference when one was supplied."""
    if qb_reference is None:
        return FACTORS
    return tuple(
        dataclasses.replace(f, mean=qb_reference, source=f"{f.source}; mean overridden by the caller's QB reference")
        if f.key == "qb_collapse_probability"
        else f
        for f in FACTORS
    )


def compute_pass_game_collapse(inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Probability that the team's passing game collapses (see the module docstring).

    ``inputs`` is a flat mapping of canonical keys (any subset; ``None`` or ``{}``
    returns the neutral reference).  Never raises on bad data.  Returns the standard
    pack result (``model``, ``pass_game_collapse``, ``probability``,
    ``reference_probability``, ``references``, ``z_score``, ``coverage``,
    ``inputs_used``, ``inputs_defaulted``, ``contributions``, ``drivers``, ``basis``)
    plus ``inputs_rejected``, ``inputs_derived``, ``role_coverage``,
    ``no_free_feed_supplied``, ``qb_collapse_reference`` and ``event``.
    """
    clean, rejected, derived, qb_reference = _clean_inputs(inputs)
    factors = _factors_for(qb_reference)
    result = common.run_factor_model(
        model=MODEL,
        primary_key=OUTPUT_KEY,
        inputs=clean,
        factors=factors,
        base_rate=BASE_RATE,
        gain=GAIN,
        basis=BASIS,
    )
    used = set(result["inputs_used"])
    result["role_coverage"] = {
        group: common.r(sum(_WEIGHT_BY_KEY[k] for k in keys if k in used) / sum(_WEIGHT_BY_KEY[k] for k in keys), 3)
        for group, keys in ROLE_GROUPS.items()
    }
    result["no_free_feed_supplied"] = [k for k in NO_FREE_FEED_INPUTS if k in used]
    result["qb_collapse_reference"] = QB_COLLAPSE_REFERENCE if qb_reference is None else qb_reference
    result["inputs_rejected"] = rejected
    result["inputs_derived"] = derived
    result["event"] = EVENT_DEFINITION
    result["basis"] = (
        f"{BASIS} Inputs supplied: {len(result['inputs_used'])} of {len(FACTORS)} "
        f"(coverage {result['coverage']:.0%}; a WR call carries no te_* inputs and a TE call no wr_* inputs, "
        "which are neutral by construction)."
    )
    LOG.debug(
        "pass_game_collapse: p=%.4f (reference %.4f, z=%+.3f, coverage=%.0f%% role_coverage=%s, rejected=%d, "
        "derived=%s, qb_reference=%.3f)",
        result[OUTPUT_KEY],
        result["reference_probability"],
        result["z_score"],
        100.0 * result["coverage"],
        result["role_coverage"],
        len(rejected),
        derived,
        result["qb_collapse_reference"],
    )
    return result


def _result_probability(result: Any) -> float | None:
    if not isinstance(result, Mapping):
        return None
    value = common.to_rate(result.get(OUTPUT_KEY))
    if value is None:
        value = common.to_rate(result.get("probability"))
    return value


def _result_reference(result: Any) -> float:
    if isinstance(result, Mapping):
        references = result.get("references")
        value = common.to_rate(references.get(OUTPUT_KEY)) if isinstance(references, Mapping) else None
        if value is None:
            value = common.to_rate(result.get("reference_probability"))
        if value is not None:
            return value
    return BASE_RATE


def _probability_cap(value: float) -> float:
    """1.0 for a fraction-scale value, 100.0 for a percent-scale one; never below the value itself."""
    cap = 1.0 if value <= _FRACTION_SCALE_LIMIT else 100.0
    return max(cap, value)


#: (field, coefficient, capped at the probability ceiling).  Every pass-game haircut is the ``1 - p*c`` form.
_APPLY_SPEC: tuple[tuple[str, float, bool], ...] = (
    ("wr_yards", WR_YARDS_COEFFICIENT, False),
    ("wr_td_prob", WR_TD_PROB_COEFFICIENT, True),
    ("te_yards", TE_YARDS_COEFFICIENT, False),
    ("te_td_prob", TE_TD_PROB_COEFFICIENT, True),
)


def apply_pass_game_collapse(
    projection: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    recenter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the pass-game collapse probability in ``result`` to a WR / TE projection.

    ``wr_yards *= 1 - p*0.55``, ``wr_td_prob *= 1 - p*0.40``, ``te_yards *= 1 - p*0.45``,
    ``te_td_prob *= 1 - p*0.35``.  Returns ``(new_projection, adjustments)``;
    ``projection`` is never mutated.  With ``recenter=True`` each multiplier is divided
    by the same formula at the model's reference probability (a league-average game is
    a no-op); ``recenter=False`` is the literal brief formula.  Whichever ``wr_*`` /
    ``te_*`` fields the projection carries are scaled; a field that is missing or not
    numeric is recorded as ``applied=False``.  The two probabilities are capped at 1.0
    (100.0 when supplied on a percent scale); a cap never pushes a value against the
    direction of the adjustment.
    """
    new_projection: dict[str, Any] = dict(projection) if isinstance(projection, Mapping) else {}
    adjustments: list[dict[str, Any]] = []
    probability = _result_probability(result)
    reference = _result_reference(result)
    for field, coefficient, capped in _APPLY_SPEC:
        if probability is None:
            adjustments.append(
                {"stage": STAGE, "field": field, "applied": False, "reason": "no valid pass_game_collapse in result"}
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
            direction=-1,
            reference=reference,
            recenter=recenter,
            adjustments=adjustments,
            cap=cap,
        )
    LOG.debug(
        "pass_game_collapse apply: p=%s (reference %.4f, recenter=%s) %s",
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
    "COEFFICIENTS",
    "EVENT_DEFINITION",
    "EVENT_EPA_CEILING",
    "EVENT_YARDS_FRACTION",
    "FACTORS",
    "GAIN",
    "MODEL",
    "NO_FREE_FEED_INPUTS",
    "OUTPUT_KEY",
    "QB_COLLAPSE_REFERENCE",
    "ROLE_GROUPS",
    "STAGE",
    "TE_TD_PROB_COEFFICIENT",
    "TE_YARDS_COEFFICIENT",
    "WR_TD_PROB_COEFFICIENT",
    "WR_YARDS_COEFFICIENT",
    "apply_pass_game_collapse",
    "compute_pass_game_collapse",
]
