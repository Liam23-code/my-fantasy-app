"""NFL parlay builder: correlation-aware combined probability, EV, and risk for
moneyline, spread, and player-prop legs.

The Parlays page lets a user stack legs (a team moneyline, a point spread, a
player prop) and asks four questions: *how correlated are these legs*, *what is
the real chance they all hit*, *what is the expected value*, and *how risky is
the ticket*. This module answers them as pure logic -- no Streamlit, no network
-- from plain leg specs. The page (or another model) supplies each leg's own
model probability and sportsbook price; this module only combines legs, it
never invents one.

How the combined probability is computed
----------------------------------------
* **Pair correlation** (:func:`leg_correlation`): a transparent, documented
  rule table. Prop-vs-prop patterns the engine already knows
  (``betting.parlay_engine``: QB + pass-catcher stacks, RB volume + TD) are
  *delegated* to it; this module adds the interactions the engine does not
  recognise at all -- moneyline/spread legs, team-result vs player props, same
  player across markets, ladders and middles, opposing-offence shootouts.
* **Joint probability** (:func:`joint_probability`): a Gaussian copula. Each
  leg is a Bernoulli event whose probability is the leg's own; the copula only
  decides how the legs move together. Two legs use an exact closed form
  (bivariate normal CDF via the Plackett identity, accurate to ~1e-12); three or
  more use a fixed-seed Monte Carlo with antithetic variates and per-block
  factorisation (legs with no link between them multiply exactly), with a
  reported standard error well below 0.002.
* The result is clamped to the Frechet bounds
  ``[max(0, sum(p) - (n - 1)), min(p)]`` -- no dependence structure can leave
  them -- and is exactly ``0.0`` if any pair is mutually exclusive.

Why moneyline and spread legs get such strong correlations
----------------------------------------------------------
"Team wins" and "team covers -3.5" are both threshold events on the *same*
final-margin variable (the engine models that margin as normal with
``NFL_MARGIN_STDEV``), so for ML/spread pairs the dependence is not fuzzy: they
are comonotone (same team) or countermonotone (opposite teams), and the joint
probability is exactly ``min(p)`` or ``max(0, p1 + p2 - 1)``. The same holds
for a ladder or a middle on one player stat and for duplicate legs. For these
"structural" pairs :func:`leg_correlation` still reports a ``rho`` capped at
+/-0.95 (a readable strength), but also a ``copula_rho`` of exactly +/-1.0, and
:func:`correlation_matrix` / :func:`joint_probability` use the exact limit: a
cap of 0.95 overstates a narrow ML/spread band by up to ~0.015 of probability
(checked against a margin-model simulation), which is too much for a
parlay price.

Legs tied this way are one variable, not several: a third leg (a player prop, say)
must correlate with each of them to the same degree, up to sign. The pair rules do
not guarantee that on their own, and an inconsistent matrix has to be "repaired" by
the nearest-valid-matrix projection, which also loosens the exact pairs (a narrow
ML + opposing-spread band plus a prop priced ~30% too high). :func:`correlation_matrix`
therefore sets each tied group's correlation with every other leg to the signed mean
of the pair rules, so the projection is almost never needed.

HONEST LIMITS -- read before trusting a number
----------------------------------------------
* Every prop-related correlation is a **heuristic prior**, in the same spirit
  as ``betting.parlay_engine.CORRELATION_ADJUSTMENT``: it encodes well-known
  football structure (game script, shared passing volume) but is NOT a
  coefficient fitted to joint-outcome data, which this project does not have
  offline. Magnitudes are deliberately modest and the whole table is exposed
  via :data:`CORRELATION_KINDS` so a page can disclose it.
* A Gaussian copula has no tail dependence and treats every leg as a
  yes/no event with the probability supplied; it will not capture, for
  example, a QB injury wiping out every leg of a stack at once.
* Legs in different games are treated as independent (no weather, division, or
  schedule-spot effects).
* Leg probabilities are assumed calibrated and push-free; a whole-number
  line that lands exactly on the number is a push in real life and is not
  modelled here. Margins are treated as continuous too, so a "middle" or an
  ML-vs-opposing-spread band narrower than one point (for example a moneyline
  against the opponent's +0.5) is priced from the model's continuous margin
  although no whole-number final score can land inside it.
* Three or more linked legs are priced by simulation. The absolute error is
  small (see ``probability_std_error``), but for a hit chance of a few hundredths
  of a percent the relative error is not: :func:`evaluate_parlay_legs` warns when
  it exceeds about 5% of the number shown.
* Sportsbooks usually **reprice same-game parlays** for correlation, so the
  price actually offered can be shorter than the product of the leg prices
  used here. Pass the book's own combined price as ``book_price`` to
  :func:`evaluate_parlay_legs` for an honest EV.
* The tier/score is a rule of thumb for how fragile the ticket is, not a
  prediction.

Optional upgrade-pack layer (opt-in, default off)
--------------------------------------------------
:func:`correlation_matrix`, :func:`evaluate_parlay_legs` and
:func:`rank_parlay_opportunities` take ``pack_signals`` -- ``{TEAM:
nfl_upgrade_pipeline.team_signals(context)}``, one bundle per offence. The caller (the
page) builds those through ``modules.nfl_matchup_features``; this module stays pure and
offline, and imports :mod:`modules.parlay_correlation` lazily so it still works without
the pack.

* **Additive evidence, never a replacement.** The pack's regime model only *moves* the
  rule table's number by how this game's regimes differ from a league-average game
  (``blended = base + w * (pack_rho - pack_rho_at_average)``). A pair is touched only when
  both legs are in one game, the pair is not structural (|base| < 0.999, not the ``-2.0``
  contradiction sentinel) and both teams carry signal. The structural ``+/-1`` ties and the
  sentinel survive untouched, and the matrix is re-tied (:func:`_tie_structural_groups`)
  after blending so a third leg still sees a tied group as ONE variable.
* **Nothing changes unless asked.** With ``pack_signals=None`` every returned dict is
  byte-identical to the pre-pack one -- no new key, no new number. With signals that carry
  no information (a league-average bundle, a missing team) the numbers are also unchanged
  and the ``pack`` report says ``applied=False``.
* **The ``pack`` report.** ``{"applied", "weight", "pack_pairs", "pairs", "strength",
  "flags", "reasons", ...}``; ``strength`` is :func:`modules.parlay_correlation.parlay_strength`
  evaluated on the matrix the ticket was PRICED with. ``flags`` has ``mega``, ``trap``,
  ``contradiction``, ``boost_overs`` and ``boost_unders``. A flag adds a plain-English line
  (with numbers) to ``warnings`` and ``risk_factors``; no flag, no line.
* **Risk stays monotone.** A ``trap`` flag adds :data:`PACK_TRAP_RISK_POINTS` to the
  correlation-concentration component on top of its cap; no flag ever lowers the score.
* **Honest limits.** The pack's weights are priors, several of its inputs are proxies or
  neutral without a free feed (coverage mismatch is inert without a player feed), and a
  sportsbook reprices same-game parlays, so a "mega" is a screening label, not an edge.

Sign conventions
----------------
* ``probability`` is always the chance the leg *wins* as built (the over, the
  cover, the team to win) -- never the chance of the opposite side.
* A spread leg's ``line`` is the number the **backed team** receives: ``-3.5``
  means the team lays 3.5 points, ``+7`` means it gets 7. (nflverse's
  ``spread_line`` is positive when the HOME team is favoured, so a home leg
  uses ``-spread_line`` and an away leg uses ``+spread_line``.)
* ``rho`` is the correlation of the latent normals behind the two legs'
  *winning* events: positive means the legs tend to win (or lose) together.
* In a correlation matrix (the copula's input) every entry lies in ``[-1, 1]``;
  ``+/-1`` is the exact nested / opposed limit of a structural pair. An
  off-diagonal :data:`CONTRADICTORY_RHO` (``-2.0``, below any real correlation)
  means the two legs are mutually exclusive.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from itertools import combinations
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from betting.odds_math import (
    american_to_decimal,
    decimal_to_american,
    expected_value,
    fair_price_from_probability,
    implied_probability,
)
from betting.parlay_engine import (
    _risk_tier as _engine_risk_tier,
    correlation_adjusted_probability,
    detect_correlations,
    make_leg,
    naive_joint_probability,
    parlay_decimal_odds,
)
from modules.odds_input import format_american, value_verdict

LEG_TYPES = ("moneyline", "spread", "prop")

#: Strongest ``rho`` :func:`leg_correlation` reports for a non-exclusive pair.
MAX_ABS_RHO = 0.95
#: Matrix sentinel meaning "these two legs cannot both win". Below -1 on purpose,
#: so that -1.0 stays available for the exact "opposed" (countermonotone) limit.
CONTRADICTORY_RHO = -2.0
#: Total Monte-Carlo points (half are antithetic mirrors) for 3+ leg blocks.
DEFAULT_DRAWS = 400_000
#: Parlay length above which :func:`evaluate_parlay_legs` warns.
LONG_PARLAY_LEGS = 8
#: Hard ceiling on legs in one parlay (sportsbooks stop well short of this). Beyond it the
#: combined price overflows a float and the pairwise work grows quadratically.
MAX_PARLAY_LEGS = 40
#: Largest American price a single leg may carry; real books never post beyond this, so a bigger
#: number is a typo and would overflow the combined price of a long parlay.
MAX_ABS_PRICE = 100_000.0
#: Same ceiling for a book-quoted COMBINED parlay price, which legitimately runs far longer.
MAX_ABS_BOOK_PRICE = 100_000_000.0
#: Largest stake accepted (dollars); keeps payouts finite.
MAX_STAKE = 1_000_000_000.0
#: Largest prop line accepted (no NFL player stat is anywhere near this).
MAX_PROP_LINE = 10_000.0
#: Risk points an upgrade-pack TRAP flag adds to the correlation-concentration component, ON TOP of that
#: component's own cap (:data:`_RISK_CORR_MAX`). It only ever raises the score.
PACK_TRAP_RISK_POINTS = 5.0
#: Every warning / risk-factor line the pack layer writes starts with this, so a page can style them.
PACK_NOTE_PREFIX = "Upgrade pack: "
#: A matrix entry that moved by less than this (after tying) is not reported as a pack change.
_PACK_CHANGE_EPS = 5e-5

_MC_SEED = 20260930  # fixed: the same parlay must always price identically
_MC_CHUNK = 50_000
_MC_CHUNK_CELLS = 2_000_000  # legs x draws held in memory per Monte-Carlo chunk
#: Relative Monte-Carlo error above which a result is flagged as imprecise.
_IMPRECISE_RELATIVE_ERROR = 0.05
#: Strings a DataFrame cell turns into when a missing value is cast with astype(str).
_MISSING_TEXT = frozenset({"nan", "none", "null", "<na>", "nat"})
_EIG_FLOOR = 1e-8
_PSD_TOLERANCE = 1e-9
_RHO_EPS = 1e-12
_NONE_BELOW = 0.02  # |rho| under this reads as "no relationship"
_WEAK_BELOW = 0.20
_MODERATE_BELOW = 0.50
_STD_NORMAL = NormalDist()
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)

# --------------------------------------------------------------------------- #
# Rule table (heuristic priors) -- the single source of truth for the page's
# "how correlation is estimated" disclosure and for the tests.
# --------------------------------------------------------------------------- #

#: Every ``kind`` string :func:`leg_correlation` can return -> plain-English rule.
CORRELATION_KINDS: dict[str, dict[str, str]] = {
    "different_games": {"prior": "0.00", "summary": "Legs in different games are treated as independent."},
    "duplicate_leg": {"prior": "+0.95 (exact limit +1)", "summary": "The same bet entered twice: it can only win or lose once."},
    "same_team_ml_and_spread_favorite": {
        "prior": "+0.95 (exact limit +1)",
        "summary": "A team's moneyline with that team laying points: covering implies winning, so the joint chance is the cover chance.",
    },
    "same_team_ml_and_spread_underdog": {
        "prior": "+0.95 (exact limit +1)",
        "summary": "A team's moneyline with that team getting points: winning outright always covers, so the joint chance is the win chance.",
    },
    "same_team_spread_ladder": {
        "prior": "+0.95 (exact limit +1)",
        "summary": "Two spreads on one team: the bigger number's cover implies the smaller's.",
    },
    "opposing_moneylines": {"prior": "mutually exclusive", "summary": "Both teams cannot win the same game."},
    "opposing_ml_and_spread_exclusive": {
        "prior": "mutually exclusive",
        "summary": "A team to win plus its opponent laying (or pick'em on) the spread: the opponent cannot cover while losing.",
    },
    "opposing_ml_and_spread_overlap": {
        "prior": "-0.95 (exact limit -1)",
        "summary": "A team to win plus its opponent getting points: both hit only if the win is narrower than the points.",
    },
    "opposing_spreads_exclusive": {
        "prior": "mutually exclusive",
        "summary": "Both sides of one game's spread with no overlap between the numbers.",
    },
    "opposing_spreads_middle": {
        "prior": "-0.95 (exact limit -1)",
        "summary": "Both sides of one game's spread where the numbers overlap (a 'middle'): both hit only inside the window.",
    },
    "team_result_and_rush_script": {
        "prior": "+0.12 to +0.22",
        "summary": "A team winning/covering and its player's rushing-yards over: leading teams run more (strongest for favourites).",
    },
    "team_result_and_player_scoring": {
        "prior": "+0.24 to +0.28",
        "summary": "A team winning/covering and its player's touchdown over: touchdowns are points.",
    },
    "team_result_and_player_volume": {
        "prior": "+0.08 to +0.12",
        "summary": "A team winning/covering and its player's yardage or reception over: only a mild link (trailing teams throw more).",
    },
    "team_result_and_opposing_player": {
        "prior": "-0.03 to -0.17",
        "summary": "A team winning/covering and the OPPOSING team's player over: when a team wins, the other offence usually scored less.",
    },
    "same_prop_over_and_under": {
        "prior": "mutually exclusive",
        "summary": "Over and under on the same player stat where the over line is at or above the under line.",
    },
    "same_prop_middle": {
        "prior": "-0.95 (exact limit -1)",
        "summary": "Over a low line plus under a high line on one stat (a 'middle'): both hit only between the lines.",
    },
    "same_prop_ladder": {
        "prior": "+0.95 (exact limit +1)",
        "summary": "The same side of one stat at two lines: the harder line implies the easier one.",
    },
    "same_player_catches_and_yards": {"prior": "+0.70", "summary": "One player's receptions and receiving yards share the same targets."},
    "same_player_receiving_volume_and_score": {
        "prior": "+0.25 to +0.35",
        "summary": "One player's receiving volume (catches/yards) and receiving touchdowns.",
    },
    "same_player_passing_volume_and_score": {"prior": "+0.50", "summary": "A QB's passing yards and passing touchdowns."},
    "same_player_td_overlap": {
        "prior": "+0.80",
        "summary": "One player's Anytime TD and his rushing- or receiving-TD over: the first almost always contains the second.",
    },
    "rb_volume_and_touchdown": {"prior": "+0.40", "summary": "One player's rushing yards and rushing touchdowns (engine pattern)."},
    "same_player_multi_market": {
        "prior": "+0.15",
        "summary": "One player in different stat families: shared snaps, health, and game script.",
    },
    "qb_pass_catcher_stack": {
        "prior": "+0.20 to +0.45",
        "summary": "A QB's passing stat and a teammate's receiving stat (engine pattern; magnitude depends on the market pair).",
    },
    "run_pass_tradeoff": {"prior": "-0.05 to -0.08", "summary": "Same-team rushing vs passing/receiving production trade off with game script."},
    "teammate_backs_split": {"prior": "-0.10", "summary": "Two teammates' rushing stats compete for the same carries."},
    "teammate_pass_catchers": {"prior": "+0.06", "summary": "Two teammates' receiving stats rise together with the team's passing volume."},
    "same_team_offense": {"prior": "+0.05", "summary": "Same-team players share one offence's pace and scoring."},
    "opposing_qbs_shootout": {"prior": "+0.15", "summary": "Both QBs over passing yards: a shootout lifts both."},
    "opposing_passing_games": {"prior": "+0.10", "summary": "Opposing passing-game stats move together in a high-pace game."},
    "opposing_scoring_shootout": {"prior": "+0.08", "summary": "Opposing touchdown legs both benefit from a high-scoring game."},
    "same_game_unrelated": {
        "prior": "+/-0.03",
        "summary": "Same game, no specific mechanism: only a small shared-environment effect (pace, weather).",
    },
}

# market -> (kind, same-team rho vs team result, opposing-team factor).  The
# rushing market has two same-team values (favourite / underdog), see below.
_TEAM_RESULT_RHO: dict[str, tuple[str, float, float]] = {
    "passing_yards": ("team_result_and_player_volume", 0.08, -0.40),
    "receiving_yards": ("team_result_and_player_volume", 0.12, -0.40),
    "receptions": ("team_result_and_player_volume", 0.08, -0.40),
    "passing_tds": ("team_result_and_player_scoring", 0.28, -0.60),
    "rushing_tds": ("team_result_and_player_scoring", 0.24, -0.60),
    "receiving_tds": ("team_result_and_player_scoring", 0.24, -0.60),
    "rushing_yards": ("team_result_and_rush_script", 0.22, -0.60),
    # Anytime TD (rushing + receiving, nfl_prop_analysis.ANYTIME_TD_MARKET) scores like any other TD leg.
    "anytime_td": ("team_result_and_player_scoring", 0.24, -0.60),
}
_RUSH_SCRIPT_UNDERDOG_RHO = 0.12

#: (QB market, pass-catcher market) -> rho for the engine's QB + pass-catcher stack.
_QB_STACK_RHO: dict[tuple[str, str], float] = {
    ("passing_yards", "receiving_yards"): 0.45,
    ("passing_yards", "receptions"): 0.40,
    ("passing_yards", "receiving_tds"): 0.25,
    ("passing_tds", "receiving_tds"): 0.35,
    ("passing_tds", "receiving_yards"): 0.30,
    ("passing_tds", "receptions"): 0.20,
}
_RB_VOLUME_TD_RHO = 0.40
_ENGINE_UNKNOWN_RHO = 0.20  # a pattern the engine learns later and this table lacks

#: unordered same-player market pair -> (kind, rho)
_SAME_PLAYER_RHO: dict[frozenset[str], tuple[str, float]] = {
    frozenset({"receptions", "receiving_yards"}): ("same_player_catches_and_yards", 0.70),
    frozenset({"receiving_yards", "receiving_tds"}): ("same_player_receiving_volume_and_score", 0.35),
    frozenset({"receptions", "receiving_tds"}): ("same_player_receiving_volume_and_score", 0.25),
    frozenset({"passing_yards", "passing_tds"}): ("same_player_passing_volume_and_score", 0.50),
    frozenset({"anytime_td", "rushing_tds"}): ("same_player_td_overlap", 0.80),
    frozenset({"anytime_td", "receiving_tds"}): ("same_player_td_overlap", 0.80),
    frozenset({"anytime_td", "rushing_yards"}): ("rb_volume_and_touchdown", 0.35),
    frozenset({"anytime_td", "receiving_yards"}): ("same_player_receiving_volume_and_score", 0.30),
    frozenset({"anytime_td", "receptions"}): ("same_player_receiving_volume_and_score", 0.25),
}
_SAME_PLAYER_OTHER_RHO = 0.15

_PASS_MARKETS = frozenset({"passing_yards", "passing_tds"})
_RUSH_MARKETS = frozenset({"rushing_yards", "rushing_tds"})
_CATCH_MARKETS = frozenset({"receiving_yards", "receiving_tds", "receptions"})
_TD_MARKETS = frozenset({"passing_tds", "rushing_tds", "receiving_tds", "anytime_td"})
_SAME_GAME_UNRELATED_RHO = 0.03


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _finite(value: Any, label: str) -> float:
    """``value`` as a finite float, else a friendly ``ValueError`` (booleans are not numbers here)."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number (got {value!r}).") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _checked_price(value: Any, label: str, ceiling: float) -> float:
    """An American price: finite, at least 100 in size, and no larger than ``ceiling``."""
    price = _finite(value, label)
    if abs(price) < 100.0:
        raise ValueError(
            f"{label} must be American odds of at least 100 in size (for example -110, +105, -150)."
        )
    if abs(price) > ceiling:
        raise ValueError(f"{label} is beyond +/-{ceiling:,.0f}, which no sportsbook posts - check for a typo.")
    return price


def _missing(value: Any) -> bool:
    """``None`` or a float ``NaN`` (what a DataFrame cell holds for a missing value)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _text(value: Any) -> str | None:
    """Stripped text, or ``None`` for blank/missing.

    A pandas ``NaN``/``NA``/``NaT`` cell (what a DataFrame row hands over for a missing team
    or game) counts as missing: ``str(nan)`` is ``"nan"`` and would otherwise become a
    real-looking team ``"NAN"`` that links unrelated legs together.
    """
    if _missing(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in _MISSING_TEXT:
        return None
    return text


def _code(value: Any) -> str | None:
    """Upper-cased team code, or ``None`` for blank/missing."""
    text = _text(value)
    return text.upper() if text else None


def _slug(text: str) -> str:
    """Lower-case ASCII identifier of a name that stays distinct across people.

    Accents fold to their base letter (an acute e becomes ``e``, matching how a user types
    the name without the accent); letters with no ASCII form are spelled out by code point
    rather than dropped, so two different non-Latin names never collapse into the same
    (empty) slug; a name with no letters or digits at all gets a short hash.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch)).casefold()
    pieces = []
    for ch in folded:
        if ch.isascii():
            pieces.append(ch if ch.isalnum() else "_")
        elif ch.isalnum():
            pieces.append(f"_u{ord(ch):x}_")
        else:
            pieces.append("_")
    slug = re.sub(r"_+", "_", "".join(pieces)).strip("_")
    return slug or "x" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _r(value: float, digits: int) -> float:
    """Round, and never emit ``-0.0``."""
    return round(float(value), digits) + 0.0


def _rp(probability: float) -> float:
    """A probability to 10 significant digits, exactly ``0.0`` only when it really is zero.

    Plain ``round(p, 6)`` turns a genuine long-shot (a 10-leg parlay of 20% legs is 1e-7)
    into ``0.0``, which then reads as an impossible parlay with no fair price.
    """
    value = float(probability)
    return 0.0 if value <= 0.0 else float(f"{value:.10g}")


def _phi(x: float) -> float:
    """Standard normal CDF (erfc form keeps the lower tail accurate)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _phi_inv(p: float) -> float:
    return _STD_NORMAL.inv_cdf(min(max(float(p), 1e-12), 1.0 - 1e-12))


def _leg_id(
    leg_type: str,
    game_id: str | None,
    team: str | None,
    player: str | None,
    market: str | None,
    side: str | None,
    line: float | None,
) -> str:
    """Deterministic, readable, widget-key-safe identity of a bet (price excluded on purpose)."""
    parts = [
        leg_type,
        game_id or "-",
        team or "-",
        player or "-",
        market or "-",
        side or "-",
        f"{line:g}" if line is not None else "-",
    ]
    return re.sub(r"[^A-Za-z0-9._:+-]", "_", ":".join(parts))


def _strength(rho: float, contradictory: bool) -> str:
    if contradictory:
        return "Strong"
    size = abs(rho)
    if size < _NONE_BELOW:
        return "None"
    if size < _WEAK_BELOW:
        return "Weak"
    if size < _MODERATE_BELOW:
        return "Moderate"
    return "Strong"


# --------------------------------------------------------------------------- #
# Leg construction
# --------------------------------------------------------------------------- #


def make_parlay_leg(
    leg_type: str,
    *,
    description: str,
    probability: float,
    price: float,
    game_id: str | None = None,
    team: str | None = None,
    opponent: str | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    side: str | None = None,
    line: float | None = None,
    market: str | None = None,
    player_id: str | None = None,
    player_name: str | None = None,
    position: str | None = None,
    confidence: float = 0.7,
) -> dict[str, Any]:
    """Validate and build one parlay leg (a ``betting.parlay_engine.make_leg`` dict plus extras).

    ``probability`` is the model's chance THIS leg wins as built; ``price`` is
    its American sportsbook price (``|price| >= 100``). Per ``leg_type``:

    * ``"moneyline"`` -- needs the backed ``team`` (or ``side`` + ``home_team``/
      ``away_team`` to derive it). ``side`` is ``"home"``/``"away"`` (derived
      from ``team`` when the game's teams are given). ``market`` is forced to
      ``"moneyline"``; ``line`` is ignored.
    * ``"spread"`` -- as moneyline, plus ``line``: the number the backed team
      receives (``-3.5`` lays 3.5, ``+7`` gets 7). ``market`` is ``"spread"``.
    * ``"prop"`` -- needs ``market`` (e.g. ``"passing_yards"``), ``side`` of
      ``"over"``/``"under"``, and a player (``player_id`` or ``player_name``).
      ``line`` is strongly recommended: without it a ladder cannot be told
      from a duplicate and over/under on one stat is assumed contradictory.
      ``team`` (the player's team) enables game-script and stack correlations.

    A ``team``, ``game_id``, ``player_name`` etc. that is ``None``, blank, a float
    ``NaN`` or the text ``"nan"``/``"None"`` counts as not given (a DataFrame row
    with a missing cell must not become the team ``"NAN"``). Accents in a name-only
    player id are folded (an acute e becomes ``e``) and other letters are kept by
    code point, so different players never share an id.

    Raises ``ValueError`` with a user-presentable message on bad input, including
    a ``price`` beyond :data:`MAX_ABS_PRICE` (+/-100,000) or a prop ``line``
    beyond :data:`MAX_PROP_LINE`. The
    returned dict carries the engine keys (``description``, ``model_probability``,
    ``price``, ``player_id``, ``team``, ``game_id``, ``market``, ``side``,
    ``confidence``) plus ``leg_type``, ``leg_id`` (stable across reruns, price
    excluded), ``probability`` (alias of ``model_probability``), ``line``,
    ``opponent``, ``home_team``, ``away_team``, ``player_name``, ``position``,
    ``decimal_odds`` and ``implied_probability``.
    """
    kind = str(leg_type).strip().lower() if leg_type is not None else ""
    if kind not in LEG_TYPES:
        raise ValueError(f"Leg type must be one of {', '.join(LEG_TYPES)} (got {leg_type!r}).")
    label = _text(description)
    if not label:
        raise ValueError("Give the leg a short description (for example 'KC moneyline').")

    win_probability = _finite(probability, "The leg's probability")
    if not 0.0 < win_probability < 1.0:
        raise ValueError(
            f"The leg's probability must be strictly between 0% and 100% (got {win_probability * 100:.1f}%)."
        )
    american = _checked_price(price, "The leg's price", MAX_ABS_PRICE)
    confidence_value = _finite(0.7 if confidence is None else confidence, "Confidence")
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    team_code, opponent_code = _code(team), _code(opponent)
    home, away = _code(home_team), _code(away_team)
    if home and away and home == away:
        raise ValueError("The home and away teams must be different.")
    side_text = (_text(side) or "").lower() or None
    market_text = (_text(market) or "").lower().replace(" ", "_") or None
    # A moneyline has no line, so a stray one (e.g. a NaN from a DataFrame row) is ignored. A prop's line is
    # optional, so NaN reads as "not given"; a spread's line is required, so NaN is an error there. "+ 0.0"
    # folds -0.0 into 0.0 so a pick'em built as -spread_line has the same id and reads "0", not "-0".
    if kind == "moneyline" or line is None or (kind == "prop" and _missing(line)):
        line_value = None
    else:
        line_value = _finite(line, "The line") + 0.0
    game = _text(game_id)

    player_key: str | None = None
    name = _text(player_name)
    if kind in ("moneyline", "spread"):
        if side_text not in (None, "home", "away"):
            raise ValueError(f"A {kind} leg's side must be 'home' or 'away' (got {side!r}).")
        if team_code is None and side_text is not None:
            team_code = home if side_text == "home" else away
        if team_code is None:
            raise ValueError(f"A {kind} leg needs the team being backed.")
        if home and away and team_code not in (home, away):
            raise ValueError(f"{team_code} is not playing in {away} at {home}.")
        derived_side: str | None
        if home and team_code == home:
            derived_side = "home"
        elif away and team_code == away:
            derived_side = "away"
        elif home:
            derived_side = "away"  # not the home team -> must be the visitor
        elif away:
            derived_side = "home"
        else:
            derived_side = None
        if side_text and derived_side and side_text != derived_side:
            raise ValueError(f"Side '{side_text}' does not match team {team_code} in this game.")
        side_text = side_text or derived_side
        if home and team_code != home:
            derived_opponent: str | None = home  # backed team is not the host -> the host is the opponent
        elif away and team_code != away:
            derived_opponent = away
        else:
            derived_opponent = None
        if opponent_code and derived_opponent and opponent_code != derived_opponent:
            raise ValueError(f"Opponent {opponent_code} does not match this game ({away} at {home}).")
        opponent_code = derived_opponent or opponent_code
        if opponent_code and opponent_code == team_code:
            raise ValueError("A team cannot be its own opponent.")
        if side_text == "home":
            home, away = team_code, opponent_code or away
        elif side_text == "away":
            away, home = team_code, opponent_code or home
        if kind == "moneyline":
            market_text, line_value = "moneyline", None
        else:
            if line_value is None:
                raise ValueError("A spread leg needs its line (the points the backed team gets, e.g. -3.5 or +7).")
            if abs(line_value) > 60:
                raise ValueError("A spread line beyond 60 points is not a real NFL number.")
            market_text = "spread"
    else:
        if market_text is None:
            raise ValueError("A prop leg needs its market (for example 'passing_yards').")
        if market_text in ("moneyline", "spread"):
            raise ValueError("Use leg type 'moneyline' or 'spread' for game markets.")
        if side_text not in ("over", "under"):
            raise ValueError("A prop leg's side must be 'over' or 'under'.")
        player_id_text = _text(player_id)
        if player_id_text is None and name is None:
            raise ValueError("A prop leg needs a player (player_id or player_name).")
        player_key = player_id_text or f"name:{_slug(name or '')}"
        if line_value is not None and line_value < 0:
            raise ValueError("A prop line cannot be negative.")
        if line_value is not None and line_value > MAX_PROP_LINE:
            raise ValueError(f"A prop line above {MAX_PROP_LINE:,.0f} is not a real stat line.")
        if team_code and home and away and team_code not in (home, away):
            raise ValueError(f"{team_code} is not playing in {away} at {home}.")
        if opponent_code is None and team_code and (home or away):
            if team_code == home:
                opponent_code = away
            elif team_code == away:
                opponent_code = home

    leg = make_leg(
        description=label,
        model_probability=win_probability,
        price=american,
        player_id=player_key,
        team=team_code,
        game_id=game,
        market=market_text,
        side=side_text,
        confidence=confidence_value,
    )
    leg.update(
        {
            "leg_type": kind,
            "leg_id": _leg_id(kind, game, team_code, player_key, market_text, side_text, line_value),
            "probability": win_probability,
            "line": line_value,
            "opponent": opponent_code,
            "home_team": home,
            "away_team": away,
            "player_name": name,
            "position": (_text(position) or "").upper() or None,
            "decimal_odds": american_to_decimal(american),
            "implied_probability": implied_probability(american),
        }
    )
    return leg


def _view(leg: Mapping[str, Any]) -> dict[str, Any]:
    """Normalised, read-only view of a leg dict (tolerates engine-built legs missing the extras)."""
    if not isinstance(leg, Mapping):
        raise ValueError("A leg must be a dict built by make_parlay_leg.")
    market = (_text(leg.get("market")) or "").lower() or None
    leg_type = (_text(leg.get("leg_type")) or "").lower()
    if leg_type not in LEG_TYPES:
        leg_type = market if market in ("moneyline", "spread") else "prop"
    raw_line = leg.get("line")
    try:
        line = float(raw_line) + 0.0 if raw_line is not None and math.isfinite(float(raw_line)) else None
    except (TypeError, ValueError):
        line = None
    raw_prob = leg.get("model_probability", leg.get("probability"))
    try:
        prob = float(raw_prob) if raw_prob is not None and math.isfinite(float(raw_prob)) else None
    except (TypeError, ValueError):
        prob = None
    name = _text(leg.get("player_name"))
    player = _text(leg.get("player_id")) or (f"name:{_slug(name)}" if name else None)
    side = (_text(leg.get("side")) or "").lower() or None
    view = {
        "type": leg_type,
        "team": _code(leg.get("team")),
        "opponent": _code(leg.get("opponent")),
        "game_id": _text(leg.get("game_id")),
        "market": market,
        "side": side,
        "line": line,
        "player": player,
        "player_name": name,
        "prob": prob,
        "description": _text(leg.get("description")),
    }
    view["id"] = _text(leg.get("leg_id")) or _leg_id(
        leg_type, view["game_id"], view["team"], player, market, side, line
    )
    return view


def _tag(view: Mapping[str, Any]) -> str:
    """Short structured name of a leg for plain-English notes."""
    if view["type"] == "moneyline":
        return f"{view['team'] or 'team'} moneyline"
    if view["type"] == "spread":
        line = view["line"]
        return f"{view['team'] or 'team'} {line:+g}" if line is not None else f"{view['team'] or 'team'} spread"
    who = view["player_name"] or view["player"] or "player"
    line = f" {view['line']:g}" if view["line"] is not None else ""
    stat = (view["market"] or "stat").replace("_", " ")
    return f"{who} {view['side'] or ''}{line} {stat}".replace("  ", " ")


def _label(leg: Mapping[str, Any]) -> str:
    return _text(leg.get("description")) or _tag(_view(leg))


# --------------------------------------------------------------------------- #
# Pair correlation
# --------------------------------------------------------------------------- #


def _same_game(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a["game_id"] and b["game_id"]:
        return a["game_id"] == b["game_id"]
    teams_a = {t for t in (a["team"], a["opponent"]) if t}
    teams_b = {t for t in (b["team"], b["opponent"]) if t}
    return bool(teams_a & teams_b)


def _finding(
    kind: str, rho: float, note: str, *, contradictory: bool = False, exact: bool = False
) -> dict[str, Any]:
    """``exact`` marks a structural pair (nested / opposed events): its copula input is exactly +/-1."""
    return {"kind": kind, "rho": rho, "note": note, "contradictory": contradictory, "exact": exact}


def _pack(found: Mapping[str, Any]) -> dict[str, Any]:
    contradictory = bool(found["contradictory"])
    rho = -MAX_ABS_RHO if contradictory else max(-MAX_ABS_RHO, min(MAX_ABS_RHO, float(found["rho"])))
    rho = _r(rho, 4)
    strength = _strength(rho, contradictory)
    if contradictory:
        direction = "contradictory"
        copula_rho = CONTRADICTORY_RHO
    else:
        direction = "none" if strength == "None" else "positive" if rho > 0 else "negative"
        copula_rho = math.copysign(1.0, rho) if found.get("exact") else rho
    return {
        "rho": rho,
        "kind": found["kind"],
        "direction": direction,
        "strength": strength,
        "note": found["note"],
        "copula_rho": float(copula_rho),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value * 100:.1f}%"


def _pct_text(probability: float) -> str:
    """A hit chance as text without rounding a real long shot down to ``0.0%``."""
    pct = probability * 100.0
    if pct >= 1.0:
        return f"{pct:.1f}%"
    if pct >= 0.01:
        return f"{pct:.2f}%"
    return "less than 0.01%"


def _one_in(probability: float) -> str:
    """``1 / probability`` as 'N' for 'about 1 in N' (scientific once it passes a billion)."""
    odds = 1.0 / probability
    return f"{odds:,.0f}" if odds < 1e9 else f"{odds:.1e}"


def _team_team(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Moneyline/spread legs in one game: threshold events on the same margin, so exactly nested or opposed."""
    team_a, team_b = a["team"], b["team"]
    if not team_a or not team_b:
        return _finding("same_game_unrelated", 0.0, "The team behind one of these legs is unknown, so no relationship is assumed.")
    pa, pb = a["prob"], b["prob"]
    smaller = min(pa, pb) if pa is not None and pb is not None else None
    band = max(0.0, pa + pb - 1.0) if pa is not None and pb is not None else None
    moneylines = [v for v in (a, b) if v["type"] == "moneyline"]
    spreads = [v for v in (a, b) if v["type"] == "spread"]

    independent = pa * pb if pa is not None and pb is not None else None

    if team_a == team_b:
        if a["id"] == b["id"]:
            return _finding(
                "duplicate_leg",
                MAX_ABS_RHO,
                f"{_tag(a)} is entered twice; it is one bet, so the second leg adds price risk but no new chance to win.",
                exact=True,
            )
        if len(moneylines) == 1 and len(spreads) == 1:
            spread = spreads[0]
            line = spread["line"]
            if line is None or line <= 0:
                shown = "the spread" if line is None else f"{line:+g}"
                return _finding(
                    "same_team_ml_and_spread_favorite",
                    MAX_ABS_RHO,
                    f"{team_a} covering {shown} already requires {team_a} to win, so the pair is effectively one bet: "
                    f"both hit exactly as often as the less likely of the two legs ({_pct(smaller)}, versus "
                    f"{_pct(independent)} if they were independent).",
                    exact=True,
                )
            return _finding(
                "same_team_ml_and_spread_underdog",
                MAX_ABS_RHO,
                f"{team_a} winning outright always covers {line:+g}, so the spread leg adds almost nothing to the "
                f"moneyline: both hit exactly as often as the less likely of the two legs ({_pct(smaller)}, versus "
                f"{_pct(independent)} if they were independent).",
                exact=True,
            )
        if len(spreads) == 2:
            return _finding(
                "same_team_spread_ladder",
                MAX_ABS_RHO,
                f"Two {team_a} spreads ({_tag(a)} and {_tag(b)}) are one ladder: covering the tougher number "
                f"covers the easier one, so the pair hits exactly as often as the tougher leg ({_pct(smaller)}).",
                exact=True,
            )
        return _finding("duplicate_leg", MAX_ABS_RHO, f"{team_a} appears twice in the same market.", exact=True)

    # Opposite teams of one game.
    if len(moneylines) == 2:
        return _finding(
            "opposing_moneylines",
            -MAX_ABS_RHO,
            f"{team_a} and {team_b} cannot both win the same game.",
            contradictory=True,
        )
    if len(moneylines) == 1:
        moneyline, spread = moneylines[0], spreads[0]
        line = spread["line"]
        if line is not None and line <= 0:
            return _finding(
                "opposing_ml_and_spread_exclusive",
                -MAX_ABS_RHO,
                f"{moneyline['team']} to win and {spread['team']} {line:+g} cannot both hit: covering a pick'em or "
                f"negative number means {spread['team']} wins.",
                contradictory=True,
            )
        shown = "the points" if line is None else f"{line:+g}"
        narrower_than = "the spread" if line is None else f"{line:g} points"
        return _finding(
            "opposing_ml_and_spread_overlap",
            -MAX_ABS_RHO,
            f"{moneyline['team']} must win while {spread['team']} still covers {shown}: both only hit if "
            f"{moneyline['team']} wins by less than {narrower_than}, so the joint chance is just {_pct(band)} "
            f"(the two probabilities added, minus 100%) versus {_pct(independent)} if they were independent.",
            exact=True,
        )
    line_a, line_b = a["line"], b["line"]
    if line_a is not None and line_b is not None and line_a + line_b <= 0:
        return _finding(
            "opposing_spreads_exclusive",
            -MAX_ABS_RHO,
            f"{_tag(a)} and {_tag(b)} cannot both cover: the numbers leave no margin that satisfies both.",
            contradictory=True,
        )
    window = "" if line_a is None or line_b is None else f" (a {line_a + line_b:g}-point window)"
    return _finding(
        "opposing_spreads_middle",
        -MAX_ABS_RHO,
        f"{_tag(a)} and {_tag(b)} form a middle{window}: both hit only if the final margin lands between the numbers, "
        f"so the joint chance is just {_pct(band)} (the two probabilities added, minus 100%) versus "
        f"{_pct(independent)} if they were independent.",
        exact=True,
    )


def _prop_relation(team_leg: Mapping[str, Any], prop: Mapping[str, Any]) -> str | None:
    """Is the prop's player on the team the other leg backs (``"same"``), the opponent (``"opposing"``), or unknown?"""
    if not team_leg["team"]:
        return None  # without the backed team, "same" and "opposing" are both guesses
    if prop["team"]:
        return "same" if prop["team"] == team_leg["team"] else "opposing"
    if prop["opponent"]:
        return "opposing" if prop["opponent"] == team_leg["team"] else "same"
    return None


def _is_favored(team_leg: Mapping[str, Any]) -> bool:
    """Is the backed team the favourite? Spread legs read the sign of the line; moneylines the leg probability."""
    if team_leg["type"] == "spread" and team_leg["line"] is not None and team_leg["line"] != 0:
        return team_leg["line"] < 0
    return team_leg["prob"] is not None and team_leg["prob"] >= 0.5


def _team_prop(team_leg: dict[str, Any], prop: dict[str, Any]) -> dict[str, Any]:
    """A moneyline/spread leg against a player prop in the same game (game-script correlation)."""
    relation = _prop_relation(team_leg, prop)
    table = _TEAM_RESULT_RHO.get(prop["market"] or "")
    if relation is None or table is None:
        return _finding(
            "same_game_unrelated",
            0.0,
            f"No specific link between {_tag(team_leg)} and {_tag(prop)} is modelled (unknown team or market).",
        )
    kind, base, opposing_factor = table
    # Game-script strength depends on whether the PLAYER's team is the favourite: the backed team itself
    # for a same-team prop, its opponent (so the opposite of the backed team's status) otherwise.
    favored = _is_favored(team_leg) if relation == "same" else not _is_favored(team_leg)
    if prop["market"] == "rushing_yards" and not favored:
        base = _RUSH_SCRIPT_UNDERDOG_RHO
    sign = -1.0 if prop["side"] == "under" else 1.0
    result_word = "covering" if team_leg["type"] == "spread" else "winning"
    under_note = " The under flips the sign: the stat tends to fall short when the team does well." if sign < 0 else ""
    if relation == "same":
        rho = sign * base
        reasons = {
            "team_result_and_rush_script": (
                f"a {'favourite' if favored else 'team'} that is {result_word} leans on the run late, so rushing yards rise "
                "with the margin"
            ),
            "team_result_and_player_scoring": "touchdowns are points, so a player's TD chance rises with his team's chance to win",
            "team_result_and_player_volume": (
                "yards and catches are only loosely tied to the result (trailing teams throw more), so the link is small"
            ),
        }
        note = (
            f"{_tag(team_leg)} and {_tag(prop)}: {reasons[kind]} (prior rho {rho:+.2f}).{under_note}"
        )
        return _finding(kind, rho, note)
    rho = sign * base * opposing_factor
    note = (
        f"{_tag(prop)} is on the opposing offence of {_tag(team_leg)}: when {team_leg['team']} is {result_word}, "
        f"{prop['team'] or 'the other team'} has usually scored and gained less (prior rho {rho:+.2f}).{under_note}"
    )
    return _finding("team_result_and_opposing_player", rho, note)


def _engine_pattern(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any] | None:
    """Ask ``betting.parlay_engine`` whether it knows this prop pair.

    Both sides are passed as ``"over"``: the engine's patterns are about which
    markets belong together, while the over/under sign is applied here, so an
    under/under stack (which the engine ignores) and an over/under split
    (negative) still use the engine's knowledge.
    """

    def probe(view: Mapping[str, Any]) -> dict[str, Any]:
        return make_leg(
            description=_tag(dict(view)),
            model_probability=0.5,
            price=-110.0,
            player_id=view["player"],
            team=view["team"],
            game_id=view["game_id"],
            market=view["market"],
            side="over",
        )

    findings = detect_correlations([probe(a), probe(b)])
    return findings[0] if findings else None


def _category(market: str | None) -> str | None:
    if market in _PASS_MARKETS:
        return "pass"
    if market in _RUSH_MARKETS:
        return "rush"
    if market in _CATCH_MARKETS:
        return "catch"
    return None


def _prop_prop(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    same_player = bool(a["player"]) and a["player"] == b["player"]
    sign = 1.0 if a["side"] == b["side"] else -1.0
    sign_note = "" if sign > 0 else " They lean opposite ways (over vs under), so the link is negative."
    market_a, market_b = a["market"], b["market"]

    if same_player:
        if market_a == market_b:
            if a["side"] != b["side"]:
                over, under = (a, b) if a["side"] == "over" else (b, a)
                if over["line"] is not None and under["line"] is not None and over["line"] < under["line"]:
                    return _finding(
                        "same_prop_middle",
                        -MAX_ABS_RHO,
                        f"Over {over['line']:g} and under {under['line']:g} on the same stat is a middle: both hit only "
                        f"if the player lands between {over['line']:g} and {under['line']:g}, so the joint chance is the "
                        "two probabilities added, minus 100%.",
                        exact=True,
                    )
                return _finding(
                    "same_prop_over_and_under",
                    -MAX_ABS_RHO,
                    f"{_tag(over)} and {_tag(under)} cannot both hit: the over needs more than the under allows.",
                    contradictory=True,
                )
            if a["line"] == b["line"]:
                return _finding(
                    "duplicate_leg",
                    MAX_ABS_RHO,
                    f"{_tag(a)} is entered twice; it is one bet, so the second leg adds price risk but no new chance to win.",
                    exact=True,
                )
            return _finding(
                "same_prop_ladder",
                MAX_ABS_RHO,
                f"{_tag(a)} and {_tag(b)} are one ladder on a single stat: clearing the harder line clears the easier "
                "one, so the pair hits exactly as often as the harder leg.",
                exact=True,
            )
        pair = _engine_pattern(a, b)
        if pair is not None and pair["kind"] == "rb_volume_and_touchdown":
            rho = sign * _RB_VOLUME_TD_RHO
            return _finding(
                "rb_volume_and_touchdown",
                rho,
                f"{a['player_name'] or 'The player'}'s rushing volume drives his touchdown chances (existing engine "
                f"pattern, prior rho {rho:+.2f}).{sign_note}",
            )
        kind, magnitude = _SAME_PLAYER_RHO.get(frozenset({market_a or "", market_b or ""}), ("same_player_multi_market", _SAME_PLAYER_OTHER_RHO))
        rho = sign * magnitude
        return _finding(
            kind,
            rho,
            f"{_tag(a)} and {_tag(b)} come from the same player, so they share snaps, health and game script "
            f"(prior rho {rho:+.2f}).{sign_note}",
        )

    team_a, team_b = a["team"], b["team"]
    if team_a and team_b and team_a == team_b:
        engine = _engine_pattern(a, b)
        if engine is not None and engine["kind"] == "qb_pass_catcher_stack":
            qb_market, catcher_market = (market_a, market_b) if market_a in _PASS_MARKETS else (market_b, market_a)
            magnitude = _QB_STACK_RHO.get((qb_market or "", catcher_market or ""), _ENGINE_UNKNOWN_RHO)
            rho = sign * magnitude
            return _finding(
                "qb_pass_catcher_stack",
                rho,
                f"{_tag(a)} and {_tag(b)} run through the same passing game: when the QB throws for more, his "
                f"receivers catch the extra yards (existing engine pattern, prior rho {rho:+.2f}).{sign_note}",
            )
        if engine is not None:
            rho = sign * (_ENGINE_UNKNOWN_RHO if engine["direction"] == "positive" else -_ENGINE_UNKNOWN_RHO)
            return _finding(
                str(engine["kind"]),
                rho,
                f"Existing engine pattern '{engine['kind']}': {engine.get('note', '')} (prior rho {rho:+.2f}).",
            )
        cat_a, cat_b = _category(market_a), _category(market_b)
        cats = {cat_a, cat_b}
        if cat_a == "rush" and cat_b == "rush":
            magnitude, kind, why = -0.10, "teammate_backs_split", "two backs split one team's carries"
        elif cats == {"pass", "rush"}:
            magnitude, kind, why = -0.08, "run_pass_tradeoff", "a team that runs more passes less"
        elif cats == {"rush", "catch"}:
            magnitude, kind, why = -0.05, "run_pass_tradeoff", "a run-heavy script trims pass-catcher volume"
        elif cat_a == "catch" and cat_b == "catch":
            magnitude, kind, why = 0.06, "teammate_pass_catchers", "a high-volume passing day lifts every receiver a little"
        else:
            magnitude, kind, why = 0.05, "same_team_offense", "one offence's pace and scoring touch every teammate"
        rho = sign * magnitude
        return _finding(kind, rho, f"{_tag(a)} and {_tag(b)}: {why} (prior rho {rho:+.2f}).{sign_note}")

    if team_a and team_b and team_a != team_b:
        cat_a, cat_b = _category(market_a), _category(market_b)
        in_air = {"pass", "catch"}
        if market_a == "passing_yards" and market_b == "passing_yards":
            magnitude, kind, why = 0.15, "opposing_qbs_shootout", "a shootout pushes both QBs' yardage up (or a slow game holds both down)"
        elif cat_a in in_air and cat_b in in_air:
            magnitude, kind, why = 0.10, "opposing_passing_games", "both passing games swing with the game's pace"
        elif market_a in _TD_MARKETS and market_b in _TD_MARKETS:
            magnitude, kind, why = 0.08, "opposing_scoring_shootout", "a high-scoring game lifts both teams' touchdown chances"
        else:
            magnitude, kind, why = _SAME_GAME_UNRELATED_RHO, "same_game_unrelated", "only a small shared-environment effect applies"
        rho = sign * magnitude
        return _finding(kind, rho, f"{_tag(a)} and {_tag(b)} are on opposing teams: {why} (prior rho {rho:+.2f}).{sign_note}")

    rho = sign * _SAME_GAME_UNRELATED_RHO
    return _finding(
        "same_game_unrelated",
        rho,
        f"{_tag(a)} and {_tag(b)} are in the same game with no modelled mechanism, so only a small shared-environment "
        f"effect applies (prior rho {rho:+.2f}).{sign_note}",
    )


def leg_correlation(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Estimated correlation between two parlay legs.

    Returns ``{"rho", "kind", "direction", "strength", "note", "copula_rho"}``:

    * ``rho`` -- float in ``[-0.95, 0.95]``, rounded to 4 places; positive means
      the legs tend to win (or lose) together. Symmetric: ``(a, b)`` and
      ``(b, a)`` agree.
    * ``kind`` -- the rule that fired; always a key of :data:`CORRELATION_KINDS`.
    * ``direction`` -- ``"positive"``, ``"negative"``, ``"contradictory"``
      (mutually exclusive: the two can never both win) or ``"none"``.
    * ``strength`` -- ``"None"`` (|rho| < 0.02), ``"Weak"`` (< 0.20),
      ``"Moderate"`` (< 0.50) or ``"Strong"``; contradictory pairs are Strong.
    * ``note`` -- one plain-English sentence with the lines and numbers.
    * ``copula_rho`` -- the value :func:`correlation_matrix` feeds the copula.
      Equal to ``rho`` except for structural pairs whose events are exactly
      nested or opposed (ML + same-team spread, spread ladders, an ML against
      the opponent's plus-points spread, a middle, a ladder or middle on one
      stat, duplicates), where it is exactly ``+1.0`` / ``-1.0``; and
      :data:`CONTRADICTORY_RHO` for a mutually exclusive pair.

    These are HEURISTIC PRIORS, not fitted coefficients (see the module
    docstring). Prop-vs-prop stacks the engine already knows are delegated to
    ``betting.parlay_engine.detect_correlations``; moneyline/spread pairs are
    derived from the shared final-margin structure, so those are exact in kind
    (nested or opposed) rather than guesses. Legs in different games return
    ``rho == 0.0`` / ``kind == "different_games"``.
    """
    # Canonical order (by leg id) so (a, b) and (b, a) give the identical result, note included.
    view_a, view_b = sorted((_view(a), _view(b)), key=lambda view: view["id"])
    if not _same_game(view_a, view_b):
        return _pack(_finding("different_games", 0.0, "These legs are in different games, so they are treated as independent."))
    team_types = [v for v in (view_a, view_b) if v["type"] in ("moneyline", "spread")]
    if len(team_types) == 2:
        found = _team_team(view_a, view_b)
    elif len(team_types) == 1:
        team_leg = view_a if view_a["type"] in ("moneyline", "spread") else view_b
        prop = view_b if team_leg is view_a else view_a
        found = _team_prop(team_leg, prop)
    else:
        found = _prop_prop(view_a, view_b)
    return _pack(found)


def _pair_table(legs: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(i, j): leg_correlation(legs[i], legs[j]) for i, j in combinations(range(len(legs)), 2)}


def _tie_structural_groups(matrix: list[list[float]]) -> list[list[float]]:
    """Make every leg see a group of exactly-tied legs as ONE variable.

    Legs joined by ``+/-1`` entries (an ML and its same-team spread, a spread ladder, a
    middle, ...) are the same latent variable up to sign, so any third leg must have the
    same correlation to each member, up to that sign. The rule table prices every pair on
    its own and does not guarantee that (it damps an opposing-team prop, and an ML and a
    spread can fall on different sides of the favourite test), which makes the matrix
    invalid; the nearest-matrix projection that rescues it then also loosens the
    ``+/-1`` pairs themselves (a narrow ML + opposing-spread band priced 0.074 instead of
    ~0.056 once a prop joined the ticket). So each cross-group correlation is replaced by
    the sign-adjusted mean of the pair entries it covers. Matrices with a mutually
    exclusive pair, or a tie cycle that does not close, are returned unchanged.
    """
    n = len(matrix)
    if n < 3:
        return matrix
    if any(matrix[i][j] < -1.0 - 1e-9 for i in range(n) for j in range(n) if i != j):
        return matrix  # a contradictory ticket prices at 0 whatever the rest says

    def tied(i: int, j: int) -> bool:
        return i != j and abs(abs(matrix[i][j]) - 1.0) < 1e-12

    group = [-1] * n
    orient = [1.0] * n  # +1 / -1: which way each member points relative to its group's first leg
    groups: list[list[int]] = []
    for start in range(n):
        if group[start] != -1:
            continue
        gid = len(groups)
        group[start] = gid
        members, stack = [start], [start]
        while stack:
            i = stack.pop()
            for j in range(n):
                if not tied(i, j):
                    continue
                want = orient[i] * math.copysign(1.0, matrix[i][j])
                if group[j] == -1:
                    group[j], orient[j] = gid, want
                    members.append(j)
                    stack.append(j)
                elif orient[j] != want:
                    return matrix
        groups.append(members)
    if all(len(members) == 1 for members in groups):
        return matrix

    out = [row[:] for row in matrix]
    for members in groups:  # transitivity: two members of one group are tied even if no rule said so
        for i, j in combinations(members, 2):
            out[i][j] = out[j][i] = orient[i] * orient[j]
    for first, second in combinations(range(len(groups)), 2):
        cells = [(i, k) for i in groups[first] for k in groups[second]]
        shared = sum(orient[i] * orient[k] * matrix[i][k] for i, k in cells) / len(cells)
        for i, k in cells:
            out[i][k] = out[k][i] = orient[i] * orient[k] * shared
    return out


def _copula_matrix(table: Mapping[tuple[int, int], Mapping[str, Any]], n: int) -> list[list[float]]:
    """The matrix :func:`joint_probability` is fed, from a :func:`_pair_table`."""
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for (i, j), found in table.items():
        matrix[i][j] = matrix[j][i] = float(found["copula_rho"])
    return _tie_structural_groups(matrix)


def _pack_blend(
    legs: Sequence[Mapping[str, Any]],
    base_matrix: list[list[float]],
    pack_signals: Any,
) -> tuple[list[list[float]], dict[str, Any] | None, str | None]:
    """Blend the upgrade pack's regime correlation into ``base_matrix``.

    Returns ``(matrix, built, error)``. ``matrix`` IS ``base_matrix`` (the same object) unless the pack touched
    at least one pair; ``built`` is :func:`modules.parlay_correlation.build_correlation_matrix`'s report (``None``
    when the pack could not run) and ``error`` is a short text in that case. The pack is additive evidence, so
    any failure inside it must leave pricing exactly as the rule table had it.
    """
    try:
        from modules import parlay_correlation  # lazy: the pack is optional; this module works without it

        built = parlay_correlation.build_correlation_matrix(legs, pack_signals, base_matrix=base_matrix)
    except Exception as problem:  # noqa: BLE001 - never let optional evidence break pricing
        return base_matrix, None, f"{type(problem).__name__}: {problem}"
    if not built.get("pack_pairs"):
        return base_matrix, built, None
    # The pack blends pair by pair, so a tied structural group (an ML and its spread) can end up with two
    # different correlations to one third leg; re-tie so the group is still ONE variable to that leg.
    return _tie_structural_groups([list(row) for row in built["matrix"]]), built, None


def correlation_matrix(
    legs: Sequence[Mapping[str, Any]],
    *,
    pack_signals: Mapping[str, Any] | None = None,
) -> list[list[float]]:
    """Symmetric ``n x n`` matrix of pairwise copula correlations with a unit diagonal.

    This is the input :func:`joint_probability` expects, so
    ``joint_probability(probabilities, correlation_matrix(legs))`` prices the
    legs exactly as :func:`evaluate_parlay_legs` does. Entries are each pair's
    ``copula_rho`` (see :func:`leg_correlation`): within ``[-0.95, 0.95]`` for
    heuristic pairs, ``+/-1.0`` for structural pairs whose events are exactly
    nested or opposed, and :data:`CONTRADICTORY_RHO` (``-2.0``) for a mutually
    exclusive pair, which :func:`joint_probability` reads as "cannot both win".

    One adjustment: legs tied by ``+/-1`` form a group that behaves as a single
    variable, so a third leg's correlation to each member is set to the same
    magnitude (the mean of the pair rules, signed by which way the member points).
    The entry can therefore differ slightly from that pair's own ``rho`` in
    :func:`leg_correlation` whenever a structurally tied group shares a ticket with
    another leg; the matrix is what the price is built from.

    ``pack_signals`` (optional, ``{TEAM: nfl_upgrade_pipeline.team_signals(context)}``) blends the NFL upgrade
    pack's regime correlation into the rule table above -- see the module docstring. ``None`` (the default), an
    empty mapping or signals with no information return the rule-table matrix unchanged; the diagonal stays 1,
    the matrix stays symmetric, and the structural ``+/-1`` entries and the ``-2.0`` sentinel are never touched.
    """
    n = len(legs)
    base = _copula_matrix(_pair_table(legs), n)
    if not pack_signals:
        return base
    return _pack_blend(legs, base, pack_signals)[0]


def pairwise_findings(legs: Sequence[Mapping[str, Any]], *, include_none: bool = False) -> list[dict[str, Any]]:
    """One finding per leg pair: ``{"legs": (i, j), "labels": (descA, descB), **leg_correlation}``.

    Pairs with no relationship (``direction == "none"``, e.g. different games)
    are omitted unless ``include_none`` is true, so the list is what a page
    wants to show. Ordered by pair index.
    """
    findings = []
    for (i, j), found in _pair_table(legs).items():
        if found["direction"] == "none" and not include_none:
            continue
        findings.append({"legs": (i, j), "labels": (_label(legs[i]), _label(legs[j])), **found})
    return findings


def correlation_rule_table() -> list[dict[str, str]]:
    """The rule table as rows ``{"kind", "prior", "summary"}`` for a page's "how this works" disclosure."""
    return [{"kind": kind, **info} for kind, info in CORRELATION_KINDS.items()]


# --------------------------------------------------------------------------- #
# Conflicts
# --------------------------------------------------------------------------- #


def _duplicate_pairs(
    legs: Sequence[Mapping[str, Any]], table: Mapping[tuple[int, int], Mapping[str, Any]]
) -> list[tuple[int, int]]:
    """Index pairs ``(i, j)``, ``i < j``, that are the same bet.

    Two legs are the same bet when they share a ``leg_id`` OR the correlation rules see
    them as one (``duplicate_leg``): the id includes the team, so the same prop entered
    once with the team filled in and once without gets two ids but is still one bet.
    """
    ids = [_view(leg)["id"] for leg in legs]
    return [
        (i, j)
        for (i, j), found in table.items()
        if ids[i] == ids[j] or found["kind"] == "duplicate_leg"
    ]


def legs_conflict(legs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Hard conflicts a page should block: duplicate legs and mutually exclusive pairs.

    Messages are user-presentable. An empty list means the legs can all win at
    once (near-redundant or strongly correlated legs are warnings, not
    conflicts -- see :func:`evaluate_parlay_legs`).
    """
    messages: list[str] = []
    table = _pair_table(legs)
    for index in sorted({j for _, j in _duplicate_pairs(legs, table)}):
        messages.append(f"Duplicate leg: '{_label(legs[index])}' is already in the parlay.")
    for (i, j), found in table.items():
        if found["direction"] == "contradictory":
            messages.append(f"'{_label(legs[i])}' and '{_label(legs[j])}' can't both win: {found['note']}")
    return messages


def can_add_leg(existing: Sequence[Mapping[str, Any]], new: Mapping[str, Any]) -> tuple[bool, str]:
    """Can ``new`` join ``existing``? Returns ``(allowed, message)``.

    ``(False, reason)`` when ``new`` duplicates a leg (including the same bet
    entered under a different ``leg_id``) or is mutually exclusive with one.
    ``(True, "")`` otherwise; ``(True, heads_up)`` when it is nearly the same
    bet as an existing leg (rho >= 0.9) or works almost directly against one
    (rho <= -0.9, a middle).
    """
    new_view = _view(new)
    new_label = _label(new)
    heads_up = ""
    for leg in existing:
        if _view(leg)["id"] == new_view["id"]:
            return False, f"'{new_label}' is already in your parlay."
        found = leg_correlation(leg, new)
        if found["kind"] == "duplicate_leg":  # same bet under a different id (e.g. team filled in on one only)
            return False, f"'{new_label}' is already in your parlay."
        if found["direction"] == "contradictory":
            return False, f"'{new_label}' can't be combined with '{_label(leg)}': {found['note']}"
        if not heads_up and abs(found["rho"]) >= 0.9:
            if found["rho"] > 0:
                heads_up = (
                    f"Heads up: '{new_label}' is nearly the same bet as '{_label(leg)}' "
                    f"(rho {found['rho']:+.2f}). {found['note']}"
                )
            else:  # a middle / opposed band is the reverse of "the same bet": the two nearly cancel
                heads_up = (
                    f"Heads up: '{new_label}' works almost directly against '{_label(leg)}' "
                    f"(rho {found['rho']:+.2f}), so both rarely win together. {found['note']}"
                )
    return True, heads_up


# --------------------------------------------------------------------------- #
# Joint probability (Gaussian copula)
# --------------------------------------------------------------------------- #


def bivariate_normal_cdf(h: float, k: float, rho: float) -> float:
    """``P(Z1 <= h, Z2 <= k)`` for standard normals with correlation ``rho``.

    Uses the Plackett identity ``dPhi2/drho = phi2(h, k; rho)`` integrated with
    a 64-point Gauss-Legendre rule after the substitution ``rho = sin(theta)``
    (which removes the endpoint singularity), so the integrand is smooth for
    any ``|rho| < 1``. Agrees with scipy's multivariate normal CDF to ~1e-12.
    ``rho >= 1`` returns the comonotone limit, ``rho <= -1`` the countermonotone.
    """
    h, k, rho = float(h), float(k), float(rho)
    if math.isnan(h) or math.isnan(k) or math.isnan(rho):
        raise ValueError("h, k and rho must be numbers.")
    if h == -math.inf or k == -math.inf:
        return 0.0
    if h == math.inf:  # an unbounded variable drops out: only the other one constrains
        return 1.0 if k == math.inf else _phi(k)
    if k == math.inf:
        return _phi(h)
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


def _project_to_correlation(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """Nearest valid correlation matrix: symmetrise, clip eigenvalues, renormalise the diagonal.

    Returns ``(matrix, was_adjusted)``. A matrix that is already valid (no
    eigenvalue below ``-1e-9``) is returned unchanged apart from symmetrisation
    and a unit diagonal. That includes singular matrices, which are what exactly
    nested legs (rho = +/-1) produce and which the sampler handles.
    """
    sym = (matrix + matrix.T) / 2.0
    np.fill_diagonal(sym, 1.0)
    values, vectors = np.linalg.eigh(sym)
    if float(values.min()) >= -_PSD_TOLERANCE:
        return sym, False
    fixed = (vectors * np.clip(values, _EIG_FLOOR, None)) @ vectors.T
    scale = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(scale, scale)
    fixed = (fixed + fixed.T) / 2.0
    np.fill_diagonal(fixed, 1.0)
    return fixed, True


def nearest_correlation_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Public wrapper of the eigenvalue-clipping projection used before sampling.

    Heuristic rule tables can be mutually inconsistent (A tracks B, B tracks C,
    but A is set against C), which no real dependence structure allows; this
    returns the closest positive-definite matrix with a unit diagonal.
    """
    array = np.array(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("The correlation matrix must be square.")
    if not np.all(np.isfinite(array)):
        raise ValueError("The correlation matrix must be finite.")
    return _project_to_correlation(array)[0].tolist()


def _cholesky(corr: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:  # numerically singular despite the projection
        values, vectors = np.linalg.eigh(corr)
        return vectors * np.sqrt(np.clip(values, _EIG_FLOOR, None))


def _monte_carlo_block(
    thresholds: np.ndarray,
    corr: np.ndarray,
    stream_ids: Sequence[int],
    draws: int,
) -> tuple[float, float, int]:
    """``P(all latent normals <= threshold)`` by antithetic Monte Carlo.

    Returns ``(estimate, standard_error, points_used)`` where ``points_used``
    counts both halves of every antithetic pair.

    One RNG stream per leg (seeded from the leg's index) is mixed through the
    Cholesky factor, so results are identical run to run, and appending a leg
    leaves the earlier legs' draws untouched (common random numbers).
    """
    chol = _cholesky(corr)
    pairs = max(1_000, (int(draws) + 1) // 2)
    streams = [np.random.default_rng([_MC_SEED, int(i)]) for i in stream_ids]
    column = thresholds[:, None]
    total = total_sq = 0.0
    done = 0
    # Bound the chunk by legs x draws so a very wide block cannot exhaust memory. Each leg's stream is
    # sequential, so the chunk size never changes which numbers are drawn.
    chunk_size = max(1_000, min(_MC_CHUNK, _MC_CHUNK_CELLS // max(1, len(streams))))
    while done < pairs:
        chunk = min(chunk_size, pairs - done)
        latent = np.stack([stream.standard_normal(chunk) for stream in streams])
        mixed = chol @ latent
        hit_forward = np.all(mixed <= column, axis=0)
        hit_mirror = np.all(mixed >= -column, axis=0)  # antithetic: the same draw negated
        pair = 0.5 * (hit_forward.astype(float) + hit_mirror.astype(float))
        total += float(pair.sum())
        total_sq += float(np.dot(pair, pair))
        done += chunk
    mean = total / pairs
    variance = max(total_sq / pairs - mean * mean, 0.0)
    return mean, math.sqrt(variance / pairs), 2 * pairs


def _components(rho: np.ndarray) -> list[list[int]]:
    """Connected groups of legs linked by a non-zero correlation."""
    n = rho.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if abs(rho[i, j]) > _RHO_EPS:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def joint_probability_detail(
    probabilities: Sequence[float],
    rho_matrix: Sequence[Sequence[float]],
    *,
    draws: int = DEFAULT_DRAWS,
) -> dict[str, Any]:
    """:func:`joint_probability` plus how it was computed.

    Returns ``{"probability", "std_error", "method", "draws", "psd_adjusted",
    "lower_bound", "upper_bound", "clamped"}``. ``method`` is ``"single_leg"``,
    ``"independent"`` (no links: the exact product), ``"closed_form"`` (no
    block larger than two legs), ``"monte_carlo"`` (some block of 3+ legs),
    ``"contradictory"`` or ``"impossible_leg"``. ``std_error`` is the Monte
    Carlo standard error of ``probability`` (``0.0`` for the exact methods);
    ``draws`` is the sample count actually used per Monte-Carlo block (``0``
    when none); ``lower_bound``/``upper_bound`` are the Frechet bounds applied.
    """
    probs = [_finite(p, "A probability") for p in probabilities]
    n = len(probs)
    if any(not 0.0 <= p <= 1.0 for p in probs):
        raise ValueError("Every probability must be within [0, 1].")
    matrix = np.array(rho_matrix, dtype=float)
    if n == 0:
        return _detail(1.0, 0.0, "independent", 0, False, 1.0, 1.0, False)
    if matrix.shape != (n, n):
        raise ValueError(f"The correlation matrix must be {n} x {n} to match the probabilities.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("The correlation matrix must be finite.")
    try:  # None / NaN / junk fall back to the default rather than raising a bare TypeError
        requested = float(draws)
        requested = requested if math.isfinite(requested) else float(DEFAULT_DRAWS)
    except (TypeError, ValueError):
        requested = float(DEFAULT_DRAWS)
    draw_budget = int(min(5_000_000, max(1_000, requested)))

    lower = max(0.0, sum(probs) - (n - 1))
    upper = min(probs)
    if n == 1:
        return _detail(probs[0], 0.0, "single_leg", 0, False, lower, upper, False)
    contradictory = any(
        min(matrix[i, j], matrix[j, i]) < -1.0 - 1e-9 for i, j in combinations(range(n), 2)
    )
    if contradictory:
        return _detail(0.0, 0.0, "contradictory", 0, False, lower, upper, False)
    if upper <= 0.0:
        return _detail(0.0, 0.0, "impossible_leg", 0, False, lower, upper, False)

    # A leg that always wins cannot change the joint probability; drop it.
    keep = [i for i, p in enumerate(probs) if p < 1.0 - 1e-12]
    if not keep:
        return _detail(1.0, 0.0, "independent", 0, False, lower, upper, False)
    sub = matrix[np.ix_(keep, keep)]
    sub = (sub + sub.T) / 2.0
    np.fill_diagonal(sub, 1.0)
    sub = np.clip(sub, -1.0, 1.0)

    estimate, variance_ratio = 1.0, 0.0
    methods: set[str] = set()
    adjusted_any = False
    mc_draws = 0
    for group in _components(sub):
        members = [keep[g] for g in group]
        group_probs = [probs[i] for i in members]
        if len(group) == 1:
            estimate *= group_probs[0]
            continue
        block = sub[np.ix_(group, group)]
        if len(group) == 2:
            value = bivariate_normal_cdf(_phi_inv(group_probs[0]), _phi_inv(group_probs[1]), float(block[0, 1]))
            estimate *= value
            methods.add("closed_form")
            continue
        block, adjusted = _project_to_correlation(block)
        adjusted_any = adjusted_any or adjusted
        thresholds = np.array([_phi_inv(p) for p in group_probs])
        value, error, points = _monte_carlo_block(thresholds, block, members, draw_budget)
        estimate *= value
        if value > 0:
            variance_ratio += (error / value) ** 2
        mc_draws = max(mc_draws, points)
        methods.add("monte_carlo")
    method = "monte_carlo" if "monte_carlo" in methods else "closed_form" if methods else "independent"
    clamped_value = min(max(estimate, lower), upper)
    return _detail(
        clamped_value,
        estimate * math.sqrt(variance_ratio),
        method,
        mc_draws,
        adjusted_any,
        lower,
        upper,
        abs(clamped_value - estimate) > 1e-12,
    )


def _detail(
    probability: float,
    std_error: float,
    method: str,
    draws: int,
    psd_adjusted: bool,
    lower: float,
    upper: float,
    clamped: bool,
) -> dict[str, Any]:
    return {
        "probability": float(probability),
        "std_error": float(std_error),
        "method": method,
        "draws": int(draws),
        "psd_adjusted": bool(psd_adjusted),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "clamped": bool(clamped),
    }


def joint_probability(
    probabilities: Sequence[float],
    rho_matrix: Sequence[Sequence[float]],
    *,
    draws: int = DEFAULT_DRAWS,
) -> float:
    """Correlation-aware probability that every leg wins (Gaussian copula).

    ``probabilities`` are the legs' own win probabilities; ``rho_matrix`` is an
    ``n x n`` correlation matrix (see :func:`correlation_matrix`) with entries
    in ``[-1, 1]``; ``+1`` / ``-1`` are the exact nested / opposed limits (two
    legs that are the same event, or the two sides of one threshold). An
    off-diagonal entry below ``-1`` (:data:`CONTRADICTORY_RHO`) means the two
    legs are mutually exclusive and the result is exactly ``0.0``.

    * Legs with no link multiply exactly; ``rho == 0`` everywhere is the plain
      product, with no sampling noise.
    * Each block of two linked legs uses the exact bivariate-normal CDF.
    * A block of three or more is sampled (fixed seed, antithetic, ``draws``
      total points; ``DEFAULT_DRAWS`` gives a standard error under 0.001) after
      the block's matrix is projected to the nearest valid correlation matrix
      if it is not already one.
    * The result is clamped to the Frechet bounds
      ``[max(0, sum(p) - (n - 1)), min(p)]``.

    Deterministic: identical inputs always return the identical float. For the
    method, standard error and bounds behind a number use
    :func:`joint_probability_detail`.
    """
    return joint_probability_detail(probabilities, rho_matrix, draws=draws)["probability"]


# --------------------------------------------------------------------------- #
# Upgrade-pack report and risk bump
# --------------------------------------------------------------------------- #


def _pack_report(
    legs: Sequence[Mapping[str, Any]],
    pack_signals: Any,
    matrix: list[list[float]],
    base_matrix: list[list[float]],
    built: Mapping[str, Any] | None,
    error: str | None,
    adjusted_probability: float,
    naive_probability: float,
) -> dict[str, Any]:
    """The ``pack`` block of :func:`evaluate_parlay_legs` (never raises; a failure is reported inside it)."""
    report: dict[str, Any] = {
        "applied": False,
        "has_signal": False,
        "weight": None,
        "pack_pairs": 0,
        "pairs": [],
        "strength": None,
        "flags": {"mega": False, "trap": False, "contradiction": False, "boost_overs": False, "boost_unders": False},
        "reasons": [],
        "score": None,
        "error": error,
    }
    if built is not None:
        report["has_signal"] = bool(built.get("has_signal"))
        report["weight"] = built.get("weight")
        report["pack_pairs"] = int(built.get("pack_pairs") or 0)
        report["applied"] = report["pack_pairs"] > 0
        changed = []
        n = len(legs)
        for pair in built.get("pairs") or []:
            i, j = pair.get("i"), pair.get("j")
            if not isinstance(i, int) or not isinstance(j, int) or not (0 <= i < n and 0 <= j < n):
                continue
            after, before = matrix[i][j], base_matrix[i][j]
            if abs(after - before) >= _PACK_CHANGE_EPS:
                changed.append({
                    "legs": (i, j), "labels": tuple(pair.get("labels") or ()), "base_rho": round(before, 4),
                    "blended_rho": round(after, 4), "pack_rho": pair.get("pack_rho"), "kind": pair.get("kind"),
                    "note": pair.get("note"),
                })
        report["pairs"] = changed
    try:
        from modules import parlay_correlation

        strength = parlay_correlation.parlay_strength(
            legs, pack_signals, matrix=matrix, base_matrix=base_matrix,
            joint_probability=adjusted_probability, independent_probability=naive_probability,
        )
        report["strength"] = strength
        report["flags"] = dict(strength.get("flags") or report["flags"])
        report["reasons"] = list(strength.get("reasons") or [])
        report["score"] = strength.get("score")
        report["has_signal"] = bool(report["has_signal"] or strength.get("has_signal"))
    except Exception as problem:  # noqa: BLE001 - optional evidence must never break pricing
        report["error"] = report["error"] or f"{type(problem).__name__}: {problem}"
    return report


def _bump_risk(risk: dict[str, Any], points: float) -> dict[str, Any]:
    """``risk`` with ``points`` more correlation-concentration risk (only ever raises the score; blocked stays 100)."""
    if risk["score"] >= 100.0 or points <= 0:
        return risk
    score = round(min(100.0, risk["score"] + points), 1)
    tier = "High" if score >= _RISK_HIGH_AT else "Medium" if score >= _RISK_MEDIUM_AT else "Low"
    breakdown = dict(risk["breakdown"])
    breakdown["correlation"] = round(breakdown["correlation"] + points, 1)
    breakdown["pack_trap"] = round(points, 1)
    return {**risk, "score": score, "tier": tier, "breakdown": breakdown}


# --------------------------------------------------------------------------- #
# Parlay evaluation
# --------------------------------------------------------------------------- #

# Risk score (0-100): additive, monotone in every ingredient, and floored by
# the engine's own tier so the engine's thresholds stay the baseline.
_RISK_PROB_MAX = 55.0
_RISK_PROB_FLOOR = 0.01  # a 1% parlay scores the full probability component
_RISK_LEG_STEP = 2.5
_RISK_LEG_MAX = 20.0
_RISK_ODDS_MAX = 15.0
_RISK_ODDS_FROM, _RISK_ODDS_TO = 1.5, 11.0  # decimal odds where the long-shot points start / saturate
_RISK_CORR_MAX = 10.0
_RISK_CORR_PER_UNIT = 6.0
_RISK_MEDIUM_AT = 25.0
_RISK_HIGH_AT = 50.0
_ENGINE_TIER_FLOOR = {"low": 0.0, "medium": _RISK_MEDIUM_AT, "high": _RISK_HIGH_AT}


def _checked_leg(leg: Mapping[str, Any], position: int) -> dict[str, Any]:
    """A validated copy of ``leg`` carrying the engine keys, ``leg_id`` and a description."""
    view = _view(leg)
    where = f"Leg {position}"
    probability = _finite(leg.get("model_probability", leg.get("probability")), f"{where}'s probability")
    if not 0.0 < probability < 1.0:
        raise ValueError(f"{where}'s probability must be strictly between 0% and 100%.")
    price = _checked_price(leg.get("price"), f"{where}'s price", MAX_ABS_PRICE)
    raw_confidence = leg.get("confidence")
    confidence = _finite(0.7 if _missing(raw_confidence) else raw_confidence, f"{where}'s confidence")
    # Identity fields are replaced by their normalised form (upper-case team, NaN -> None, -0.0 -> 0.0), so the
    # engine cross-check, the summaries and the correlation rules all see the same team/game/player.
    return {
        **leg,
        "model_probability": probability,
        "price": price,
        "confidence": max(0.0, min(1.0, confidence)),
        "leg_id": view["id"],
        "leg_type": view["type"],
        "description": _text(leg.get("description")) or _tag(view),
        "team": view["team"],
        "game_id": view["game_id"],
        "market": view["market"],
        "side": view["side"],
        "line": view["line"],
        "player_id": view["player"],
        "player_name": view["player_name"],
    }


def _engine_leg(leg: Mapping[str, Any]) -> dict[str, Any]:
    return make_leg(
        description=leg["description"],
        model_probability=leg["model_probability"],
        price=leg["price"],
        player_id=leg.get("player_id"),
        team=leg.get("team"),
        game_id=leg.get("game_id"),
        market=leg.get("market"),
        side=leg.get("side"),
        confidence=leg["confidence"],
    )


def _exact_american(decimal_odds: float) -> float:
    """American price for ``decimal_odds`` WITHOUT the integer rounding of ``decimal_to_american``.

    Lets EV be computed through ``betting.odds_math.expected_value`` and still
    equal ``p * payout - (1 - p) * stake`` exactly.
    """
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def _risk(
    num_legs: int,
    probability: float,
    longest_decimal: float,
    correlation_mass: float,
    engine_tier: str,
    *,
    blocked: bool,
) -> dict[str, Any]:
    """Numeric risk score and tier; see :func:`evaluate_parlay_legs` for the rules."""
    if probability <= 0.0:
        probability_points = _RISK_PROB_MAX
    else:
        span = math.log(1.0 / _RISK_PROB_FLOOR)
        probability_points = _RISK_PROB_MAX * min(1.0, max(0.0, math.log(1.0 / min(probability, 1.0)) / span))
    leg_points = min(_RISK_LEG_MAX, _RISK_LEG_STEP * (num_legs - 1))
    odds_points = _RISK_ODDS_MAX * min(
        1.0, max(0.0, (longest_decimal - _RISK_ODDS_FROM) / (_RISK_ODDS_TO - _RISK_ODDS_FROM))
    )
    correlation_points = min(_RISK_CORR_MAX, _RISK_CORR_PER_UNIT * correlation_mass)
    raw = probability_points + leg_points + odds_points + correlation_points
    floor = _ENGINE_TIER_FLOOR[engine_tier]
    score = 100.0 if blocked else min(100.0, max(raw, floor))
    score = round(score, 1)
    tier = "High" if score >= _RISK_HIGH_AT else "Medium" if score >= _RISK_MEDIUM_AT else "Low"
    return {
        "score": score,
        "tier": tier,
        "floor_applied": (not blocked) and raw < floor,
        "breakdown": {
            "hit_probability": _r(probability_points, 1),
            "leg_count": _r(leg_points, 1),
            "longest_odds": _r(odds_points, 1),
            "correlation": _r(correlation_points, 1),
            "engine_floor": _r(floor, 1),
            "engine_tier": engine_tier.capitalize(),
        },
    }


def evaluate_parlay_legs(
    legs: Sequence[Mapping[str, Any]],
    *,
    stake: float = 100.0,
    book_price: float | None = None,
    draws: int = DEFAULT_DRAWS,
    pack_signals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Price a parlay of 2+ legs: correlation, combined probability, EV, and risk tier.

    ``pack_signals`` (optional, ``{TEAM: nfl_upgrade_pipeline.team_signals(context)}``) opts in to the NFL
    upgrade pack's regime correlation and adds a ``pack`` key (see the module docstring); ``None`` leaves
    every returned value and key exactly as it was before the pack existed.

    ``legs`` come from :func:`make_parlay_leg`. ``stake`` is in dollars.
    ``book_price`` (optional, American) is the combined price the sportsbook
    actually quotes -- books reprice same-game parlays, so supplying it makes EV
    honest; when omitted the price is the product of the legs' decimal odds.
    Fewer than two legs, more than :data:`MAX_PARLAY_LEGS`, a stake that is not in
    ``(0, MAX_STAKE]``, a ``book_price`` or leg price beyond what any sportsbook
    posts (:data:`MAX_ABS_BOOK_PRICE` / :data:`MAX_ABS_PRICE`), or an invalid leg
    raises ``ValueError`` with a friendly message; contradictory legs never raise
    (see below). A hand-built leg's identity fields are normalised (upper-case
    team, a NaN/``"nan"`` team or game becomes missing, ``-0.0`` becomes ``0.0``).

    Probabilities are reported to 10 significant digits, never rounded to zero
    unless they really are zero: a 10-leg parlay of 20% legs is 1e-7 and stays
    1e-7 (rounding to 6 decimals used to turn it into an "impossible" parlay).

    Returned keys
    -------------
    ``num_legs``; ``legs`` (per-leg summary dicts: ``index, leg_id, leg_type,
    description, probability, price, decimal_odds, implied_probability, edge,
    confidence, game_id, team, market, side, line, player_name``);
    ``labels``; ``stake``; ``price_source`` (``"legs_product"`` or
    ``"book_quote"``).

    Probabilities -- ``naive_probability`` (plain product via
    ``betting.parlay_engine.naive_joint_probability``);
    ``adjusted_probability`` (the Gaussian-copula joint probability, the
    headline number); ``engine_adjusted_probability`` (what
    ``betting.parlay_engine.correlation_adjusted_probability`` gives for the same
    legs, shown as a cross-check -- it applies a flat +/-12% per detected
    prop/total pattern and cannot see moneyline/spread legs at all, whereas the
    copula grades every pair, so the two differ whenever ML/spread legs or
    strongly/weakly correlated pairs are present); ``correlation_lift``
    (adjusted minus naive); ``probability_method``, ``probability_std_error``,
    ``probability_bounds`` (``{"lower", "upper"}`` Frechet), ``psd_adjusted``.

    Correlation -- ``correlation_pairs`` (:func:`pairwise_findings`: each pair's own
    documented ``rho``); ``matrix`` (:func:`correlation_matrix`: what the price was
    built from, which differs from a pair's ``rho`` only where a structurally tied
    group of legs shares the ticket with another leg); ``has_contradiction`` (bool).
    A duplicate is any pair the rules call ``duplicate_leg`` as well as a repeated
    ``leg_id``, so the same bet entered with and without a team is caught.

    Price -- ``decimal_odds``; ``american_odds`` (rounded); ``parlay_implied_probability``
    (``1 / decimal_odds``); ``payout`` (profit on ``stake`` if every leg hits);
    ``fair_decimal_odds`` / ``fair_american_odds`` (the price the adjusted
    probability would be fair at, unclamped even for a one-in-a-billion ticket;
    ``None`` only when the probability is 0 and ``has_fair_price`` is ``False`` --
    the only place ``None`` appears).

    Value -- ``ev_naive``, ``ev_adjusted`` (dollars on ``stake`` via
    ``betting.odds_math.expected_value``, equal to ``p * payout - (1 - p) *
    stake``), ``ev_pct`` (adjusted EV / stake), ``edge`` (adjusted probability
    minus the parlay price's implied probability; positive exactly when EV is),
    ``verdict`` / ``verdict_tone`` (``modules.odds_input.value_verdict`` applied
    to ``ev_pct / 2`` -- the probability-point gap an even-money bet needs for
    the same EV, so a long-shot parlay is judged by return on stake rather than
    by a probability gap that a 5% ticket can barely show; ``"Contradictory
    legs"`` / ``"danger"`` when legs are mutually exclusive), ``summary`` (one
    sentence; it adds that the price is the unrepriced leg product when correlated
    or same-game legs are priced without a ``book_price``).

    Risk -- ``confidence`` (average leg confidence less 4% per leg beyond two,
    as in the engine); ``risk_tier`` (``"Low"|"Medium"|"High"``);
    ``risk_score`` (0-100); ``risk_breakdown`` (points per ingredient);
    ``risk_factors`` and ``warnings`` (plain-English strings with numbers).

    Risk rules
    ----------
    The score adds four monotone ingredients -- hit probability (up to 55 points,
    log-scaled so a 1% ticket maxes it), leg count (2.5 per leg after the first,
    up to 20), the longest-odds leg (up to 15, from +50 to +1000) and
    correlation concentration (6 per unit of summed |rho|, up to 10). The score
    is then floored by the engine's own tier
    (``betting.parlay_engine._risk_tier``: <=2 legs and p>=35% is Low, <=4 legs
    and p>=15% is Medium, else High) at 25 / 50 points, and the tier is read off
    the score (<25 Low, <50 Medium, else High). So the tier is never below the
    engine's, equals it for ordinary legs, and only goes higher when a long-shot
    leg or heavily concentrated correlation adds real fragility. A contradictory
    parlay scores 100 / High with ``adjusted_probability == 0.0`` and
    ``ev_adjusted == -stake``.
    """
    items = list(legs)
    if len(items) < 2:
        raise ValueError("A parlay needs at least 2 legs - add another leg to price it.")
    if len(items) > MAX_PARLAY_LEGS:
        raise ValueError(f"A parlay can have at most {MAX_PARLAY_LEGS} legs (got {len(items)}) - remove a few.")
    stake_value = _finite(stake, "The stake")
    if stake_value <= 0:
        raise ValueError("The stake must be greater than zero.")
    if stake_value > MAX_STAKE:
        raise ValueError(f"The stake must be at most ${MAX_STAKE:,.0f}.")
    checked = [_checked_leg(leg, index + 1) for index, leg in enumerate(items)]
    book = None if book_price is None else _checked_price(book_price, "The book price", MAX_ABS_BOOK_PRICE)

    n = len(checked)
    engine_legs = [_engine_leg(leg) for leg in checked]
    probabilities = [leg["model_probability"] for leg in checked]
    labels = [leg["description"] for leg in checked]

    table = _pair_table(checked)
    matrix = _copula_matrix(table, n)
    base_matrix = matrix
    pack_built: dict[str, Any] | None = None
    pack_error: str | None = None
    if pack_signals is not None:
        matrix, pack_built, pack_error = _pack_blend(checked, base_matrix, pack_signals)
    findings = [
        {"legs": (i, j), "labels": (labels[i], labels[j]), **found}
        for (i, j), found in table.items()
        if found["direction"] != "none"
    ]
    contradictions = [f for f in findings if f["direction"] == "contradictory"]
    duplicate_pairs = _duplicate_pairs(checked, table)
    later_copies = {j for _, j in duplicate_pairs}
    # One entry per duplicated bet (its first occurrence), however many copies there are.
    duplicate_firsts = sorted({i for i, _ in duplicate_pairs if i not in later_copies})

    naive = naive_joint_probability(engine_legs)
    detail = joint_probability_detail(probabilities, matrix, draws=draws)
    adjusted = detail["probability"]
    engine_adjusted = correlation_adjusted_probability(engine_legs)["adjusted_probability"]

    if book is None:
        decimal_odds = parlay_decimal_odds(engine_legs)
        american_odds = int(decimal_to_american(decimal_odds))
        price_source = "legs_product"
    else:
        decimal_odds = american_to_decimal(book)
        american_odds = int(round(book))
        price_source = "book_quote"
    exact_price = _exact_american(decimal_odds)
    naive_r, adjusted_r = _rp(naive), _rp(adjusted)
    payout = stake_value * (decimal_odds - 1.0)
    ev_naive = expected_value(naive_r, exact_price, stake_value)
    ev_adjusted = expected_value(adjusted_r, exact_price, stake_value)
    parlay_implied = 1.0 / decimal_odds
    edge = adjusted_r - parlay_implied
    if adjusted_r > 0:
        fair_decimal: float | None = float(f"{1.0 / min(adjusted_r, 1.0 - 1e-9):.10g}")
        # The engine's own conversion where it is valid; it clamps at 1e-6, which would cap a 1e-8 long shot's
        # fair price at a hundredth of its real size, so below that decimal_to_american is applied directly.
        fair_american: float | None = (
            fair_price_from_probability(adjusted_r) if 1e-6 <= adjusted_r <= 1.0 - 1e-6 else decimal_to_american(fair_decimal)
        )
    else:
        fair_decimal = fair_american = None

    average_confidence = sum(leg["confidence"] for leg in checked) / n
    confidence = _r(max(0.0, min(1.0, average_confidence * (1.0 - 0.04 * (n - 2)))), 4)

    # value_verdict's cut-offs (1 / 3 probability points) are calibrated for single bets near even money.
    # A parlay's price can be anything, so a raw probability gap would call a +31%-EV long shot "slight".
    # Judge instead by the gap an even-money bet (decimal 2.0) would need for the same EV: ev_pct / 2.
    verdict, tone = value_verdict(ev_adjusted / stake_value / 2.0)
    if contradictions:
        verdict, tone = "Contradictory legs", "danger"

    longest = max(checked, key=lambda leg: american_to_decimal(leg["price"]))
    longest_decimal = american_to_decimal(longest["price"])
    correlation_mass = sum(abs(f["rho"]) for f in findings)
    engine_tier = _engine_risk_tier(n, adjusted_r)
    risk = _risk(n, adjusted_r, longest_decimal, correlation_mass, engine_tier, blocked=bool(contradictions))
    pack: dict[str, Any] | None = None
    if pack_signals is not None:
        pack = _pack_report(checked, pack_signals, matrix, base_matrix, pack_built, pack_error, adjusted_r, naive_r)
        if pack["flags"].get("trap") and not contradictions:
            risk = _bump_risk(risk, PACK_TRAP_RISK_POINTS)

    games = {leg.get("game_id") for leg in checked}
    one_game = all(
        _same_game(_view(checked[i]), _view(checked[j])) for i, j in combinations(range(n), 2)
    )
    linked = [f for f in findings if f["direction"] in ("positive", "negative") and abs(f["rho"]) >= _WEAK_BELOW]
    strongest = max(linked, key=lambda f: abs(f["rho"]), default=None)

    # Legs that are each possible but jointly impossible (an opposed band with no room), as opposed to a joint
    # chance that merely underflows: naive_r > 0 tells them apart.
    no_room = adjusted_r <= 0.0 and naive_r > 0.0 and not contradictions
    reprice_caveat = (
        (one_game or any(f["direction"] == "positive" and f["rho"] >= _WEAK_BELOW for f in findings))
        and book is None
        and not contradictions
    )

    warnings: list[str] = []
    for f in contradictions:
        warnings.append(f"'{f['labels'][0]}' and '{f['labels'][1]}' cannot both win ({f['note']}) - this parlay can never hit.")
    if no_room:
        warnings.append(
            "Given each leg's own probability, these legs cannot all win together (the opposed legs' probabilities "
            "add to 100% or less), so the model gives this parlay a 0% chance."
        )
    for first in duplicate_firsts:
        warnings.append(
            f"'{checked[first]['description']}' appears more than once; the duplicate adds price risk but no extra chance to win."
        )
    for f in findings:
        if f["direction"] == "positive" and f["rho"] >= 0.9 and f["kind"] != "duplicate_leg":
            warnings.append(
                f"'{f['labels'][0]}' and '{f['labels'][1]}' are nearly the same bet (rho {f['rho']:+.2f}); "
                "stacking them multiplies the price but barely adds risk-adjusted value."
            )
    if n > 1 and one_game:
        warnings.append(f"All {n} legs come from one game, so a single game script decides the whole parlay.")
    if n > LONG_PARLAY_LEGS:
        warnings.append(f"{n} legs is a very long parlay (over {LONG_PARLAY_LEGS}); small errors in each leg's probability compound.")
    if 0.0 < adjusted_r < 0.01:
        warnings.append(
            f"The model's hit chance for this parlay is {_pct_text(adjusted_r)} "
            f"(about 1 in {_one_in(adjusted_r)}) - below 1%."
        )
    relative_error = detail["std_error"] / adjusted_r if adjusted_r > 0 else 0.0
    if detail["method"] == "monte_carlo" and relative_error > _IMPRECISE_RELATIVE_ERROR:
        warnings.append(
            f"This hit chance comes from a simulation and is imprecise at this size (about +/-{relative_error * 100:.0f}% "
            "of the number shown), so treat the fair odds and EV as rough."
        )
    if reprice_caveat:
        warnings.append(
            "Sportsbooks usually reprice correlated same-game legs, so the price they offer can be lower than the "
            "product of the leg prices used here; enter the book's combined price for an honest EV."
        )

    factors: list[str] = [
        f"{n} legs: every one has to win, and each added leg multiplies the miss chance.",
    ]
    if adjusted_r > 0:
        factors.append(
            f"Combined hit probability is {_pct_text(adjusted_r)} (about 1 in {_one_in(adjusted_r)}) versus "
            f"{_pct_text(parlay_implied)} implied by the {format_american(american_odds)} price."
        )
    elif naive_r > 0:
        factors.append("Combined hit probability is 0%: at least two legs cannot win together.")
    else:
        factors.append("Combined hit probability is effectively 0%: the legs multiply to a number too small to represent.")
    if longest_decimal >= 3.0:
        factors.append(
            f"Longest-odds leg: '{longest['description']}' at {format_american(longest['price'])} "
            f"({implied_probability(longest['price']) * 100:.0f}% implied) - one long shot drags the whole ticket."
        )
    if linked and strongest is not None:
        factors.append(
            f"{len(linked)} correlated pair{'s' if len(linked) != 1 else ''}; strongest is '{strongest['labels'][0]}' + "
            f"'{strongest['labels'][1]}' (rho {strongest['rho']:+.2f}) - those legs tend to {'win or lose together' if strongest['rho'] > 0 else 'work against each other'}."
        )
    elif not findings and len(games) == n and None not in games:
        factors.append(f"The {n} legs are in {n} different games with no modelled link, so they are close to independent.")
    for f in contradictions:
        factors.append(f"Contradictory pair: '{f['labels'][0]}' and '{f['labels'][1]}'.")
    for first in duplicate_firsts:
        factors.append(f"Duplicate leg: {checked[first]['description']}.")
    if risk["floor_applied"]:
        factors.append(
            f"Baseline rule: {n} legs at a {_pct_text(adjusted_r)} hit chance falls in the {engine_tier.capitalize()} band "
            "(<=2 legs and >=35% is Low, <=4 legs and >=15% is Medium, otherwise High)."
        )

    if pack is not None and not contradictions:
        pack_flags = pack["flags"]
        reasons = pack.get("reasons") or []
        lead = reasons[0] if reasons else ""
        score_text = f" (strength {pack['score']:.0f}/100)" if isinstance(pack.get("score"), (int, float)) else ""
        if pack_flags.get("trap"):
            warnings.append(
                f"{PACK_NOTE_PREFIX}this looks like a correlated stack but this game's signals work against it{score_text}. {lead}".strip()
            )
            factors.append(f"{PACK_NOTE_PREFIX}TRAP flag{score_text}: the shared driver of these legs is adverse in this game.")
        elif pack_flags.get("mega"):
            factors.append(
                f"{PACK_NOTE_PREFIX}MEGA flag{score_text}: the legs ride regimes this game's signals favor. {lead}".strip()
            )
        for key, text in (("boost_overs", "overs"), ("boost_unders", "unders")):
            if pack_flags.get(key) and not pack_flags.get("trap"):
                factors.append(f"{PACK_NOTE_PREFIX}the same-direction {text} share a regime this game favors, so their link is firmer than the rule table alone says.")

    leg_summaries = [
        {
            "index": index,
            "leg_id": leg["leg_id"],
            "leg_type": leg["leg_type"],
            "description": leg["description"],
            "probability": _rp(leg["model_probability"]),
            "price": leg["price"],
            "decimal_odds": _r(american_to_decimal(leg["price"]), 6),
            "implied_probability": _rp(implied_probability(leg["price"])),
            "edge": _r(leg["model_probability"] - implied_probability(leg["price"]), 6),
            "confidence": _r(leg["confidence"], 4),
            "game_id": leg.get("game_id"),
            "team": leg.get("team"),
            "market": leg.get("market"),
            "side": leg.get("side"),
            "line": leg.get("line"),
            "player_name": leg.get("player_name"),
        }
        for index, leg in enumerate(checked)
    ]
    stake_text = f"${stake_value:,.0f}" if float(stake_value).is_integer() else f"${stake_value:,.2f}"
    summary = (
        f"{n}-leg parlay at {format_american(american_odds)}: model hit chance {_pct_text(adjusted_r)} "
        f"vs {_pct_text(parlay_implied)} implied, EV {'+' if ev_adjusted >= 0 else '-'}${abs(ev_adjusted):,.2f} on {stake_text} "
        f"({verdict.lower()}), {risk['tier']} risk."
    )
    if reprice_caveat:
        summary += " The price is the product of the leg prices; a sportsbook would likely quote less for these correlated legs."

    result = {
        "num_legs": n,
        "legs": leg_summaries,
        "labels": labels,
        "stake": _r(stake_value, 2),
        "price_source": price_source,
        "naive_probability": naive_r,
        "adjusted_probability": adjusted_r,
        "engine_adjusted_probability": _r(engine_adjusted, 6),
        "correlation_lift": _r(adjusted_r - naive_r, 12),
        "probability_method": detail["method"],
        "probability_std_error": _rp(detail["std_error"]),
        "probability_bounds": {"lower": _rp(detail["lower_bound"]), "upper": _rp(detail["upper_bound"])},
        "psd_adjusted": detail["psd_adjusted"],
        "correlation_pairs": findings,
        "matrix": matrix,
        "has_contradiction": bool(contradictions),
        "decimal_odds": _r(decimal_odds, 6),
        "american_odds": american_odds,
        "parlay_implied_probability": _rp(parlay_implied),
        "payout": _r(payout, 2),
        "fair_decimal_odds": fair_decimal,
        "fair_american_odds": fair_american,
        "has_fair_price": fair_decimal is not None,
        "ev_naive": _r(ev_naive, 2),
        "ev_adjusted": _r(ev_adjusted, 2),
        "ev_pct": _r(ev_adjusted / stake_value, 4),
        "edge": _r(edge, 12),
        "verdict": verdict,
        "verdict_tone": tone,
        "summary": summary,
        "confidence": confidence,
        "risk_tier": risk["tier"],
        "risk_score": risk["score"],
        "risk_breakdown": risk["breakdown"],
        "risk_factors": factors,
        "warnings": warnings,
    }
    if pack is not None:
        result["pack"] = pack
    return result


def rank_parlay_opportunities(
    candidate_leg_sets: Sequence[Sequence[Mapping[str, Any]]],
    *,
    pack_signals: Mapping[str, Any] | None,
    stake: float = 100.0,
) -> list[dict[str, Any]]:
    """Evaluate each candidate parlay with the upgrade pack and rank them (the brief's "rank parlay opportunities").

    Each candidate is a list of legs (see :func:`make_parlay_leg`). A candidate that cannot be priced (fewer than
    two legs, a contradiction the validator rejects) is dropped quietly. Returns
    :func:`modules.parlay_correlation.rank_parlays`' rows (rank, score, flags, reasons, lift, alignment,
    fragility, adjusted probability, EV, ...) each with the full ``evaluation`` of that parlay and
    ``candidate`` (the index into ``candidate_leg_sets``), best first. Ranking is by the pack's strength score,
    then expected value, then hit probability, and is stable; a flag is a screening label, not an edge.
    """
    from modules import parlay_correlation

    evaluations: dict[int, dict[str, Any]] = {}
    kept: list[tuple[int, list[Mapping[str, Any]]]] = []
    for index, legs in enumerate(candidate_leg_sets):
        try:
            evaluations[index] = evaluate_parlay_legs(list(legs), stake=stake, pack_signals=pack_signals or {})
        except ValueError:
            continue
        kept.append((index, list(legs)))
    if not kept:
        return []
    by_legs = {id(legs): evaluations[index] for index, legs in kept}

    def evaluator(legs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        found = by_legs[id(legs)] if id(legs) in by_legs else evaluate_parlay_legs(list(legs), stake=stake, pack_signals=pack_signals or {})
        return {
            "adjusted_probability": found["adjusted_probability"],
            "naive_probability": found["naive_probability"],
            "ev_adjusted": found["ev_adjusted"],
        }

    ranked = parlay_correlation.rank_parlays([legs for _, legs in kept], pack_signals or {}, evaluator=evaluator)
    out: list[dict[str, Any]] = []
    for row in ranked:
        position = row.get("index")
        if not isinstance(position, int) or not (0 <= position < len(kept)):
            continue
        original_index, _ = kept[position]
        merged = dict(row)
        merged["candidate"] = original_index
        merged["evaluation"] = evaluations[original_index]
        out.append(merged)
    return out


__all__ = [
    "CONTRADICTORY_RHO",
    "CORRELATION_KINDS",
    "DEFAULT_DRAWS",
    "LEG_TYPES",
    "LONG_PARLAY_LEGS",
    "MAX_ABS_BOOK_PRICE",
    "MAX_ABS_PRICE",
    "MAX_ABS_RHO",
    "MAX_PARLAY_LEGS",
    "MAX_PROP_LINE",
    "MAX_STAKE",
    "PACK_TRAP_RISK_POINTS",
    "bivariate_normal_cdf",
    "can_add_leg",
    "correlation_matrix",
    "correlation_rule_table",
    "evaluate_parlay_legs",
    "joint_probability",
    "joint_probability_detail",
    "leg_correlation",
    "legs_conflict",
    "make_parlay_leg",
    "nearest_correlation_matrix",
    "pairwise_findings",
    "rank_parlay_opportunities",
]
