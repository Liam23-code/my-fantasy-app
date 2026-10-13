"""NFL Player Props analysis engine: pick a player and a stat, paste the book's line, get a priced read.

This is the pure-logic layer behind the Player Props page. The page lets the
user choose a team, then a player, then a stat category, then paste a
sportsbook line (and optionally over/under prices, default -110/-110). This
module supplies every number and sentence the page shows:

* the model's own projection for that stat for ONE game (:func:`project_stat`),
* a sensible line to pre-fill (:func:`suggest_line`),
* probability over / under / push, de-vigged market probability, edge, expected
  value, confidence, volatility and risk tier (:func:`evaluate_line`),
* a reasoning section whose every line carries real numbers
  (:func:`reasoning_for`),
* friendly validation of the pasted text (:func:`validate_inputs`).

Nothing here imports Streamlit, touches the network on its own (only
:func:`load_player_pool` reads data, and it fails soft), or places a bet.

How the numbers are produced (and why they match the existing NFL Betting page)
-------------------------------------------------------------------------------
The seven native markets (passing/rushing/receiving yards, the three TD
markets, receptions) are delegated to the existing engine:
:func:`betting.prop_model.stat_distribution` builds the per-game distribution,
:func:`betting.prop_model.evaluate_prop` supplies the half-point probabilities,
confidence and risk tier, and :mod:`betting.odds_math` supplies the de-vig, edge
and EV maths. Probabilities, confidence and risk tier are therefore identical to
what the existing page shows for the same player and line. The player-status
overlay is applied exactly as ``evaluate_prop`` applies it (OUT / HOLDOUT /
SUSPENDED zero the stat inputs, DOUBTFUL / QUESTIONABLE trim them).

The one market the engine does not have, **Anytime TD (Rush + Rec)**, is
composed here: a Poisson count with ``lambda`` = rushing-TD rate + receiving-TD
rate per game, so ``P(at least one TD) = 1 - exp(-lambda)``. Passing TDs are not
part of it (it is the standard "to score a TD" market). Its confidence and risk
tier use the same engine helpers as the native markets.

Optional matchup upgrade pack
-----------------------------
:func:`project_stat` and :func:`evaluate_line` take ``pack_context`` (a canonical-feature
context from :mod:`modules.nfl_matchup_features`; this module never does that I/O). When given,
the per-game MEAN of the market is passed through the NFL upgrade pack's stages for the
player's position (collapse, explosion, coverage mismatch, explosive plays, red-zone funnel,
game script, volume pass-through; see ``docs/nfl_model_upgrade_pack.md``) AFTER the status
overlay, and the spread follows: a Gaussian market keeps its coefficient of variation, a count
market uses ``sqrt(mean)``. The probabilities are then priced from the adjusted
mean/stdev/family. ``pack_context=None`` (the default) returns exactly what this module has
always returned -- no extra key, no changed number. A market the pack does not touch for that
position (a QB's rushing yards, an RB's receiving line apart from volume pass-through) comes
back numerically unchanged; the QB completion-percentage stage has no market here.
Honest limits: the pack's weights are priors, several inputs are proxies or neutral without a
free feed, and the baseline is still the player's PRIOR-season per-game rate.

Sign and convention notes
-------------------------
* ``edge_*`` is model probability minus the market's **de-vigged** fair
  probability (the same convention as ``evaluate_prop``); ``value_gap_*`` is
  model probability minus the **raw** implied probability of the price (the
  break-even win rate, vig included -- :func:`modules.odds_input.evaluate_price`).
  Positive means the model rates that side likelier than the market requires. It
  is not a prediction that the bet wins.
* ``ev_*`` is expected profit on ``stake`` dollars if the model probability were
  the true probability.
* **Integer lines** (e.g. 249.0) can push. The engine's own over/under maths
  ignores pushes, so for an integer line this module prices the stat as an
  integer-valued outcome: over = P(X > line), under = P(X < line), push =
  P(X = line), with ``probability_over + probability_under + push_probability``
  = 1 (within 1e-4 rounding). A push returns the stake, so ``ev_*`` is
  ``P(win) * payout - P(lose) * stake`` and ``edge_*`` / ``value_gap_*`` compare
  the market to the model's win probability *given the bet is decided*
  (``P(side) / (1 - push)``), which is exactly the break-even comparison for a
  bet that refunds on a push. Half-point lines never push.

Honest limits of the model
--------------------------
* The per-game mean is the player's raw prior-season rate (season total divided
  by games played). It is **not** adjusted for tonight's opponent, game script,
  weather, a role change, or in-season 2026 games. The fantasy-points projection
  on the same row is regressed and market-reconciled; the stat mean is not.
* The spread of yardage markets is a modelling choice, not observed weekly
  variance: a Gaussian with coefficient of variation scaled between
  ``CV_LOW`` and ``CV_HIGH`` by the Quant Engine's 0-1 volatility read. That read
  is derived from the fantasy-points floor/ceiling band, and it is exactly 0 for
  players with almost no history (a thin-sample player can read "Low" volatility
  for lack of data -- :func:`evaluate_line` warns when the sample is small). A
  Gaussian also puts a little probability below zero yards, which a real stat
  cannot reach.
* **That spread is too wide for quarterback passing yards.** The engine's read
  saturates at 1.0 for most of the pool (cv 0.65), but in the 2025 nflverse weekly
  stats the median starting QB's weekly passing-yards cv was about 0.33 (middle half
  0.27 to 0.47), so Passing Yards probabilities for lines away from the mean sit
  closer to 50% than history supports and apparent edges are understated.
  :func:`reasoning_for` says so with those numbers. Skill-position yardage (WR / TE
  / RB cv about 0.6 to 1.0 in the same data) is at or above the top of the engine's
  range, so there the tails are likelier than shown. The module delegates to the
  engine on purpose (it must agree with the existing page), so this is reported,
  not corrected.
* Count markets are Poisson (variance = mean). Real weekly counts run a little
  over-dispersed relative to Poisson (variance about 1.0 to 1.2 times the mean for
  2025 starters), so extreme games are a bit likelier than shown. The engine's
  0-1 volatility read does not influence a count market's spread at all; it is
  reported for information only.
* Anytime TD prices off the per-game rate rounded to three decimals (the engine's
  own rounding for every market), so it can differ from the exact closed form
  ``1 - exp(-lambda)`` by up to about 0.05 probability points.
* ``data_season`` is the season of the actuals behind the rates (``prior_season`` on
  a forward-projection row). It is ``None`` when the row does not say; it is never
  the projection season.
* Bad data is tolerated, not trusted: NaN / inf / unparseable ``games_played`` or
  stat totals are treated as missing and flagged in ``warnings``, never allowed to
  turn probabilities into NaN. Lines above :data:`MAX_LINE` raise ``ValueError``.
* Confidence is "how much evidence is behind the projection" (sample size and
  the ensemble's own confidence), not a probability that the pick wins.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from betting import player_status_utils as status_utils
from betting.odds_math import edge_vs_fair, expected_value, hold, remove_vig_two_way
from betting.prop_model import (
    CV_HIGH,
    CV_LOW,
    STAT_FIELDS,
    _confidence,
    _normal_cdf,
    _poisson_cdf,
    _risk_tier,
    evaluate_prop,
    stat_distribution,
)
from fantasy.projections import load_forward_projections, upcoming_season
from projections.projection_engine import compute_final_projection

from modules import nfl_upgrade_pipeline
from modules.odds_input import (
    evaluate_price,
    format_american,
    format_money,
    format_pct,
    format_signed_pct,
    parse_number,
    parse_odds,
)

logger = logging.getLogger(__name__)

#: Default two-way price when the user leaves a price box blank.
DEFAULT_PRICE = -110.0

#: Market key for the composed rush + receiving TD market.
ANYTIME_TD_MARKET = "anytime_td"

#: Coefficient of variation reported when a count stat has a ~0 mean (the true
#: value is unbounded). Capped so no ``None``/``inf`` ever reaches a page.
CV_UNDEFINED = 5.0

#: Confidence / volatility label cut-offs. Chosen from the real pool: stars with
#: a full season sit at confidence 0.90-0.96 (High); one-to-five-game players sit
#: at 0.2-0.45 (Low). Volatility of 0.42-1.0 is typical for starters, so the
#: "High" band starts at 0.70.
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MODERATE = 0.50
VOLATILITY_HIGH = 0.70
VOLATILITY_MODERATE = 0.35

#: Games below which the per-game rate is flagged as a thin sample.
THIN_SAMPLE_GAMES = 6.0
#: Games at which the rate counts as a full-season sample.
FULL_SAMPLE_GAMES = 12.0

#: Largest line :func:`evaluate_line` will price. The Poisson tail sum is O(line),
#: so an absurd pasted number (1e9) would otherwise freeze the page for minutes
#: and 1e308 would overflow; no real NFL prop comes anywhere near this.
MAX_LINE = 10_000.0

#: Observed weekly passing-yards coefficient of variation for NFL starters:
#: median and middle-half range across the 31 quarterbacks with 10+ regular-season
#: games and 20+ passing yards per game in the 2025 nflverse weekly player stats.
#: Used only to tell the user how the engine's Gaussian spread compares with history
#: (see :func:`reasoning_for`); it never changes a probability.
OBSERVED_QB_PASSING_CV = (0.27, 0.33, 0.47)  # (p25, median, p75)

#: z for the 10th / 90th percentile of a normal distribution.
_Z90 = 1.2815515655446004

_POOL_TTL_SECONDS = 60 * 60
_POOL_CACHE: dict[int, tuple[float, list[dict[str, Any]]]] = {}
_POOL_LOCK = threading.Lock()

_POSITION_ALIASES = {"HB": "RB", "FB": "RB"}

# =========================================================================
# Stat categories
# =========================================================================


def _category_entry(
    market: str,
    positions: tuple[str, ...],
    unit: str,
    family: str,
    min_line: float,
    max_line: float,
    blurb: str,
    *,
    short_label: str,
    fields: tuple[str, ...],
    min_per_game: float,
) -> dict[str, Any]:
    return {
        "market": market,
        "positions": positions,
        "unit": unit,
        "family": family,
        "min_line": min_line,
        "max_line": max_line,
        "step": 0.5,
        "blurb": blurb,
        # Extras beyond the required contract, handy for the page:
        "short_label": short_label,
        "fields": fields,
        "min_per_game": min_per_game,
    }


#: Ordered label -> category spec. ``positions`` is who may be *offered* the
#: category; ``min_per_game`` is the smallest raw per-game rate of the relevant
#: stat that counts as real volume (so a third-string QB with one garbage-time
#: pass, or a kicker, is never offered a market nobody would post). A QB is
#: offered Rushing Yards / Rushing TDs / Anytime TD only when his own rate clears
#: that bar (a running QB does, a pocket passer does not); a non-QB is never
#: offered Passing Yards / Passing TDs whatever his totals say.
STAT_CATEGORIES: dict[str, dict[str, Any]] = {
    "Passing Yards": _category_entry(
        "passing_yards", ("QB",), "yds", "gaussian", 50.5, 450.5,
        "Total passing yards by the quarterback in one game.",
        # The floor sits just under min_line (50.5): a QB averaging less than that would have his
        # pre-filled line clamped *above* his own projection (e.g. 21 yds/g offered at 50.5), so
        # thin-sample backups are not offered a market nobody posts for them.
        short_label="Pass Yds", fields=("passing_yards",), min_per_game=45.0,
    ),
    "Passing TDs": _category_entry(
        "passing_tds", ("QB",), "TDs", "poisson", 0.5, 5.5,
        "Touchdown passes thrown by the quarterback in one game.",
        short_label="Pass TDs", fields=("passing_tds",), min_per_game=0.10,
    ),
    "Rushing Yards": _category_entry(
        "rushing_yards", ("RB", "QB", "WR"), "yds", "gaussian", 4.5, 225.5,
        "Total rushing yards in one game (running backs, plus quarterbacks and receivers with real carry volume).",
        short_label="Rush Yds", fields=("rushing_yards",), min_per_game=5.0,
    ),
    "Rushing TDs": _category_entry(
        "rushing_tds", ("RB", "QB", "WR"), "TDs", "poisson", 0.5, 2.5,
        "Rushing touchdowns scored in one game.",
        short_label="Rush TDs", fields=("rushing_tds",), min_per_game=0.05,
    ),
    "Receiving Yards": _category_entry(
        "receiving_yards", ("WR", "TE", "RB"), "yds", "gaussian", 4.5, 225.5,
        "Total receiving yards in one game.",
        short_label="Rec Yds", fields=("receiving_yards",), min_per_game=8.0,
    ),
    "Receiving TDs": _category_entry(
        "receiving_tds", ("WR", "TE", "RB"), "TDs", "poisson", 0.5, 2.5,
        "Receiving touchdowns scored in one game.",
        short_label="Rec TDs", fields=("receiving_tds",), min_per_game=0.05,
    ),
    "Receptions": _category_entry(
        "receptions", ("WR", "TE", "RB"), "rec", "poisson", 0.5, 15.5,
        "Catches in one game.",
        short_label="Rec", fields=("receptions",), min_per_game=0.5,
    ),
    "Anytime TD (Rush + Rec)": _category_entry(
        ANYTIME_TD_MARKET, ("RB", "WR", "TE", "QB"), "TDs", "poisson", 0.5, 0.5,
        "The standard 'to score a touchdown' market: at least one rushing or receiving TD "
        "(passing TDs do not count). Always priced at 0.5.",
        short_label="Anytime TD", fields=("rushing_tds", "receiving_tds"), min_per_game=0.05,
    ),
}

#: Category labels in display order.
CATEGORY_LABELS: tuple[str, ...] = tuple(STAT_CATEGORIES)

_STAT_INPUT_FIELDS = tuple(STAT_FIELDS.values())


def _category(label: str) -> dict[str, Any]:
    spec = STAT_CATEGORIES.get(label)
    if spec is None:
        raise ValueError(f"unsupported stat category: {label!r} (expected one of {list(STAT_CATEGORIES)})")
    return spec


# =========================================================================
# Small safe helpers
# =========================================================================


def _num(value: Any, default: float = 0.0) -> float:
    """``float(value)`` that never raises and never returns NaN/inf."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _position(player: Mapping[str, Any]) -> str:
    raw = str(player.get("position") or "").strip().upper()
    return _POSITION_ALIASES.get(raw, raw)


def _games(player: Mapping[str, Any]) -> float:
    """Real games behind the per-game rate -- same fallback as ``stat_distribution``."""
    return _num(player.get("games_played")) or _num(player.get("prior_games_played"))


def _name(player: Mapping[str, Any]) -> str:
    return str(player.get("name") or player.get("player_name") or player.get("player_id") or "This player")


def _team_of(player: Mapping[str, Any]) -> str:
    return normalize_team(player.get("team"))


def _season_total(player: Mapping[str, Any], spec: Mapping[str, Any]) -> float:
    return sum(_num(player.get(field)) for field in spec["fields"])


def _raw_rate(player: Mapping[str, Any], spec: Mapping[str, Any]) -> float:
    """Raw (status-unadjusted) per-game rate of the category's stat."""
    games = _games(player)
    return _season_total(player, spec) / games if games > 0 else 0.0


#: Row fields the engine reads *directly* (``float(player.get(...))``) with no NaN guard.
_ENGINE_NUMERIC_FIELDS = (*_STAT_INPUT_FIELDS, "games_played", "prior_games_played")


def _nonfinite_fields(player: Mapping[str, Any]) -> list[str]:
    """Engine-read fields of ``player`` holding NaN / inf / unparseable text (``None`` is fine: the engine treats it as 0)."""
    bad: list[str] = []
    for field in _ENGINE_NUMERIC_FIELDS:
        value = player.get(field)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            bad.append(field)
            continue
        if not math.isfinite(number):
            bad.append(field)
    return bad


def _sane_row(player: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``player`` that is safe to hand to :mod:`betting.prop_model`.

    The engine does ``float(player.get(field) or 0.0)`` with no NaN/inf guard, so a
    single NaN in ``games_played`` or a stat total turns every probability into NaN.
    Non-finite values become ``None`` -- which the engine already reads as 0 (stats)
    or as "fall back to ``prior_games_played``" (games) -- so a clean row is passed
    through byte-for-byte and a dirty one degrades to a neutral, warned result.
    """
    row = dict(player)
    for field in _nonfinite_fields(player):
        row[field] = None
    return row


def _data_season(player: Mapping[str, Any]) -> int | None:
    """The season of the *actuals* behind the rates, or ``None`` when the row does not say.

    On a forward-projection row ``season`` / ``projection_season`` is the season being
    *projected* (2026) and ``prior_season`` is the actuals season (2025), so ``season``
    is deliberately not used as a fallback: it would mislabel 2026 rates as 2026 actuals.
    """
    for key, offset in (("prior_season", 0), ("projection_season", -1)):
        value = player.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)) and value > 0:
            return int(value) + offset
    return None


def _ordinal(number: float) -> str:
    n = int(round(number))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fmt_stat(value: float, unit: str) -> str:
    """``232.4 yds`` / ``0.41 TDs`` / ``5.4 rec`` -- yards to one decimal, counts to two."""
    digits = 1 if unit == "yds" else 2
    return f"{value:,.{digits}f} {unit}"


# =========================================================================
# Teams and the player pool
# =========================================================================

_TEAM_ALIAS_CACHE: dict[str, str] | None = None
_TEAM_NAMES: dict[str, str] = {}


def _team_aliases() -> dict[str, str]:
    """Upper-cased alias / name -> nflverse team code, built once from ``modules.nfl``."""
    global _TEAM_ALIAS_CACHE
    if _TEAM_ALIAS_CACHE is not None:
        return _TEAM_ALIAS_CACHE
    aliases: dict[str, str] = {}
    try:
        from modules.nfl import NFL_TEAMS

        for code, (full_name, nickname, extra) in NFL_TEAMS.items():
            aliases[code.upper()] = code
            aliases[full_name.upper()] = code
            aliases[nickname.upper()] = code
            for alias in extra:
                aliases[str(alias).upper()] = code
            _TEAM_NAMES[code] = full_name
    except Exception:  # pragma: no cover - modules.nfl is a hard dependency of the app; defensive only
        aliases = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "ARZ": "ARI"}
    _TEAM_ALIAS_CACHE = aliases
    return aliases


def normalize_team(team: Any) -> str:
    """Map any common team spelling (``LAR``, ``Rams``, ``WSH``) to the nflverse code the pool uses.

    Unknown strings are returned upper-cased and stripped, so a code the alias
    table does not know still matches pool rows that carry it verbatim.
    """
    text = str(team or "").strip().upper()
    if not text:
        return ""
    return _team_aliases().get(text, text)


def team_label(team: str) -> str:
    """``"KC - Kansas City Chiefs"`` for a known code, the bare code otherwise."""
    code = normalize_team(team)
    _team_aliases()
    full_name = _TEAM_NAMES.get(code)
    return f"{code} - {full_name}" if full_name else code


def load_player_pool(season: int | None = None) -> list[dict[str, Any]]:
    """The NFL player pool the page selects from -- the existing NFL Betting page's data.

    ``season`` is the *source* season of actuals; it defaults to
    ``upcoming_season() - 1`` (the last completed season), exactly the data
    season :mod:`app.betting_shared` uses, so per-game rates agree between pages.
    Returns the forward-projection rows (:func:`fantasy.projections.load_forward_projections`)
    including kickers/defences; :func:`teams_in_pool` / :func:`players_for_team`
    do the skill-position filtering.

    Fail-soft: any data failure (offline, provider outage, bad season) returns
    ``[]`` and never raises. A non-empty result is cached in-process for an hour
    (the load takes several seconds); an empty result is never cached so a
    transient outage heals on the next call. Treat rows as read-only.
    """
    try:
        data_season = int(season) if season is not None else int(upcoming_season()) - 1
    except Exception:  # noqa: BLE001 - a bad season argument must not raise into the UI either
        return []
    now = time.monotonic()
    with _POOL_LOCK:
        cached = _POOL_CACHE.get(data_season)
        if cached is not None and now - cached[0] < _POOL_TTL_SECONDS:
            return list(cached[1])
    try:
        rows = load_forward_projections(season=data_season)
        # Filtering sits inside the guard too: a loader that returns something odd (a DataFrame,
        # a generator that fails part-way) must degrade to "no pool", not raise into the page.
        pool = [row for row in (rows if rows is not None else []) if isinstance(row, dict)]
    except Exception as error:  # noqa: BLE001 - the contract is "never raise into the UI"
        logger.warning("NFL player pool unavailable for season %s: %s", data_season, error)
        return []
    if pool:
        with _POOL_LOCK:
            _POOL_CACHE[data_season] = (time.monotonic(), pool)
    return list(pool)


def clear_player_pool_cache() -> None:
    """Drop the in-process pool cache (tests; the page's Refresh button)."""
    with _POOL_LOCK:
        _POOL_CACHE.clear()


def player_label(player: Mapping[str, Any]) -> str:
    """``"Josh Allen (QB)"`` -- the text a dropdown shows."""
    position = _position(player)
    name = _name(player)
    return f"{name} ({position})" if position else name


def find_player(pool: Iterable[Mapping[str, Any]] | None, player_id: Any) -> dict[str, Any] | None:
    """The pool row whose ``player_id`` equals ``player_id`` (string compare), else ``None``.

    A ``None`` / blank id (nothing selected yet) never matches: ``str(None)`` is ``"None"``,
    which would otherwise match any row that is merely missing its id.
    """
    if player_id is None or not str(player_id).strip():
        return None
    wanted = str(player_id)
    for player in pool or []:
        if isinstance(player, Mapping) and str(player.get("player_id")) == wanted:
            return player  # type: ignore[return-value]
    return None


def player_status(player: Mapping[str, Any]) -> str:
    """The live-overlay availability (``HEALTHY`` / ``QUESTIONABLE`` / ``OUT`` ...) for a player."""
    return status_utils.live_status(player)


def is_eligible(player: Mapping[str, Any], category_label: str) -> bool:
    """Whether ``player`` should be *offered* ``category_label`` (position and real volume).

    Requires an eligible position for the category, at least one game, and a raw
    per-game rate of the relevant stat at or above the category's
    ``min_per_game`` floor. Unknown categories are simply not eligible.
    """
    spec = STAT_CATEGORIES.get(category_label)
    if spec is None or not isinstance(player, Mapping):
        return False
    if _position(player) not in spec["positions"]:
        return False
    return _games(player) > 0 and _raw_rate(player, spec) >= spec["min_per_game"]


def eligible_categories(player: Mapping[str, Any]) -> list[str]:
    """Every category label ``player`` is eligible for, in display order."""
    return [label for label in STAT_CATEGORIES if is_eligible(player, label)]


def teams_in_pool(pool: Iterable[Mapping[str, Any]] | None) -> list[str]:
    """Sorted team codes with at least one skill player eligible for some category."""
    teams = {
        _team_of(player)
        for player in pool or []
        if isinstance(player, Mapping) and eligible_categories(player)
    }
    teams.discard("")
    return sorted(teams)


def players_for_team(
    pool: Iterable[Mapping[str, Any]] | None,
    team: str,
    category_label: str | None = None,
) -> list[dict[str, Any]]:
    """Players on ``team`` the page should offer for ``category_label``.

    With a category: only position-eligible players with real volume
    (:func:`is_eligible`), most season volume of that stat first (so the
    featured yardage / TD / reception producers lead the dropdown), ties broken
    by per-game rate then name. With ``category_label=None``: every player
    eligible for *any* category, best fantasy projection first.

    Players the live feed lists as OUT are **kept** (the page may badge them via
    :func:`player_status`); evaluating one collapses P(over) and warns. An
    unknown ``category_label`` raises ``ValueError``. A blank ``team`` (nothing selected
    yet) returns ``[]`` rather than every row that is merely missing a team.
    """
    wanted = normalize_team(team)
    if category_label is not None:
        _category(category_label)  # validate even when there is nobody to list
    if not wanted:
        return []
    members = [p for p in pool or [] if isinstance(p, Mapping) and _team_of(p) == wanted]
    if category_label is None:
        chosen = [p for p in members if eligible_categories(p)]
        chosen.sort(key=lambda p: (-_num(p.get("projection")), _name(p)))
        return chosen  # type: ignore[return-value]
    spec = _category(category_label)
    chosen = [p for p in members if is_eligible(p, category_label)]
    chosen.sort(key=lambda p: (-_season_total(p, spec), -_raw_rate(p, spec), _name(p)))
    return chosen  # type: ignore[return-value]


# =========================================================================
# Projection
# =========================================================================


def _games_text(count: float) -> str:
    """"1 game" / "12 games" -- a one-game sample must not read "1 games"."""
    return f"{count:g} game" if float(count) == 1.0 else f"{count:g} games"


def confidence_label(confidence: float) -> str:
    """``"High"`` (>= 0.75), ``"Moderate"`` (>= 0.50) or ``"Low"`` for a 0-1 confidence."""
    if confidence >= CONFIDENCE_HIGH:
        return "High"
    if confidence >= CONFIDENCE_MODERATE:
        return "Moderate"
    return "Low"


def volatility_label(volatility: float) -> str:
    """``"High"`` (>= 0.70), ``"Moderate"`` (>= 0.35) or ``"Low"`` for a 0-1 volatility."""
    if volatility >= VOLATILITY_HIGH:
        return "High"
    if volatility >= VOLATILITY_MODERATE:
        return "Moderate"
    return "Low"


def _anytime_td_distribution(row: Mapping[str, Any], ensemble: Mapping[str, Any], games: float) -> dict[str, Any]:
    """Poisson distribution for 'at least one rushing or receiving TD', same shape as ``stat_distribution``."""
    mean = max(0.0, sum(_num(row.get(field)) for field in ("rushing_tds", "receiving_tds")) / games)
    volatility = max(0.0, min(1.0, _num(ensemble.get("volatility"))))
    stdev = math.sqrt(max(mean, 1e-6))
    return {
        "mean": round(mean, 3),
        "stdev": round(stdev, 3),
        "cv": round(stdev / mean, 4) if mean > 0 else None,
        "volatility": round(volatility, 4),
        "games": games,
        "family": "poisson",
    }


def _status_note(name: str, status: str, factor: float) -> str:
    if status == status_utils.HEALTHY:
        if status_utils.has_status_data():
            return f"The live availability feed shows no injury flag for {name}; the full per-game rate is used."
        return f"No live availability data is loaded, so {name} is treated as healthy and the full per-game rate is used."
    if factor <= 0.0:
        return (
            f"{name} is {status} per the live availability feed: every stat input is zeroed, "
            "so the model gives the over essentially no chance."
        )
    return (
        f"{name} is {status} per the live availability feed: stat inputs are scaled to {factor:.0%} "
        f"of the healthy rate (x{factor:.2f})."
    )


def _status_adjusted(player: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """``(status, row)`` where ``row`` is scaled exactly as ``betting.prop_model.evaluate_prop`` scales it."""
    status = status_utils.live_status(player)
    if status != status_utils.HEALTHY:
        return status, status_utils.status_adjusted_row(player, _STAT_INPUT_FIELDS, status=status)
    return status, dict(player)


def _pack_adjust(
    market: str,
    position: str,
    distribution: dict[str, Any],
    row: Mapping[str, Any],
    games: float,
    pack_context: Mapping[str, Any],
    wr_role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pass a market's per-game mean through the upgrade pack; ``(distribution, report)``.

    Never raises (a defect leaves the distribution untouched and is reported). The spread follows the
    adjusted mean: a Gaussian market keeps its cv, a count (Poisson) market uses ``sqrt(mean)``.
    """
    mean_before = float(distribution["mean"])
    report: dict[str, Any] = {
        "applied": False, "neutral": True, "has_signal": False, "mean_before": mean_before, "mean_after": mean_before,
        "adjustments": [], "summary": {}, "explanations": [], "warnings": [], "guard": [], "stage_order": [], "signals": {},
    }
    if position not in nfl_upgrade_pipeline.POSITIONS:
        report["warnings"].append(f"The upgrade pack has no stage for position {position!r}; the projection is unchanged.")
        return distribution, report
    if market == ANYTIME_TD_MARKET:
        line = {"rushing_tds": _num(row.get("rushing_tds")) / games, "receiving_tds": _num(row.get("receiving_tds")) / games}
    else:
        line = {market: mean_before}
    try:
        outcome = nfl_upgrade_pipeline.project_with_pack(line, position=position, context=pack_context, wr_role=wr_role)
    except Exception as error:  # noqa: BLE001 - optional layer: never take a priced line down
        report["warnings"].append(f"The upgrade pack could not be applied ({type(error).__name__}); the projection is unchanged.")
        return distribution, report
    adjusted = outcome.get("line") or {}
    if market == ANYTIME_TD_MARKET:
        mean_after = max(0.0, sum(float(adjusted.get(key, line[key])) for key in line))
    else:
        mean_after = max(0.0, float(adjusted.get(market, mean_before)))
    signals = outcome.get("signals") or {}
    report.update(
        neutral=bool(outcome.get("neutral", True)),
        has_signal=bool(signals.get("has_signal")),
        adjustments=[dict(a) for a in outcome.get("adjustments") or []],
        summary=dict(outcome.get("summary") or {}),
        explanations=list(outcome.get("explanations") or []),
        warnings=list(outcome.get("warnings") or []),
        guard=list(outcome.get("guard") or []),
        stage_order=list(outcome.get("stage_order") or []),
        signals=dict(signals.get("probabilities") or {}),
        mean_after=mean_after,
    )
    if abs(mean_after - mean_before) < 5e-4:
        return distribution, report
    updated = dict(distribution)
    mean_new = round(mean_after, 3)
    if distribution["family"] == "poisson":
        stdev_new = math.sqrt(max(mean_new, 1e-6))
        cv_new: float | None = round(stdev_new / mean_new, 4) if mean_new > 0 else None
    else:
        cv_old = distribution.get("cv")
        cv_new = cv_old
        stdev_new = max(mean_new * float(cv_old), 0.01) if cv_old is not None else max(float(distribution["stdev"]), 0.01)
    updated.update(mean=mean_new, stdev=round(stdev_new, 3), cv=cv_new)
    report["applied"] = True
    return updated, report


def project_stat(
    player: Mapping[str, Any],
    category_label: str,
    *,
    pack_context: Mapping[str, Any] | None = None,
    wr_role: str = "wr",
) -> dict[str, Any]:
    """The model's projection of ``category_label`` for ONE game for ``player``.

    Built from :func:`betting.prop_model.stat_distribution` (so the seven native
    markets match the existing NFL Betting page exactly) after the live
    player-status overlay is applied the way ``evaluate_prop`` applies it.
    Anytime TD composes the rushing- and receiving-TD rates itself.

    Returned keys: ``player_id, name, team, position, category, market, unit,
    family`` (``"gaussian"``/``"poisson"``), ``mean, stdev, cv, volatility``
    (0-1), ``games`` (the sample behind the rate), ``season_total`` and
    ``per_game_rate`` (the raw, status-unadjusted prior-season figures),
    ``data_season``, ``confidence`` (0-1), ``confidence_label``, ``risk_tier``,
    ``volatility_label``, ``projected_fantasy_points``, ``projection``
    (same number), ``fantasy_points_per_game``, ``expected_games``,
    ``projection_confidence``, ``projection_basis``, ``drivers`` (list of str),
    ``status``, ``status_factor``, ``status_note``, ``eligible`` (whether the page
    would offer this player this category), ``components`` (for Anytime TD:
    ``rushing_td_per_game`` / ``receiving_td_per_game``; empty otherwise).

    ``data_season`` is the season of the actuals (``prior_season`` on a forward-projection
    row), or ``None`` when the row does not say -- never the *projection* season.

    ``pack_context`` (default ``None``) opts in to the NFL upgrade pack (see the module docstring): the
    adjusted ``mean`` / ``stdev`` / ``cv`` replace the status-adjusted ones, and the result gains
    ``pack_applied`` (the mean actually moved), ``mean_before_pack`` and ``pack`` (audit: ``adjustments``,
    ``summary``, ``explanations``, ``warnings``, ``guard``, ``stage_order``, ``signals``, ``mean_before``,
    ``mean_after``). ``wr_role`` (``"wr"`` | ``"slot"``) picks the coverage-mismatch role for a receiver.
    With ``pack_context=None`` no key is added and no number changes.

    Raises ``ValueError`` for an unsupported category, a player that is not a row
    (``None``, a string), or a player with no games played (no per-game rate exists).
    It does **not** raise for a position that would not normally be offered the
    stat -- check ``eligible``. Non-finite inputs (NaN / inf / unparseable text in
    ``games_played`` or a stat total) are treated as missing rather than allowed to
    turn every probability into NaN.
    """
    spec = _category(category_label)
    if not isinstance(player, Mapping):
        raise ValueError(f"player must be a player row (a mapping), got {type(player).__name__}")
    name = _name(player)
    games = _games(player)
    if games <= 0:
        raise ValueError(f"{name} has no games played in the source season, so there is no per-game rate to project from.")

    status, row = _status_adjusted(_sane_row(player))
    factor = status_utils.adjust_projection_for_status(1.0, status)
    ensemble = compute_final_projection(dict(row))
    market = spec["market"]
    if market == ANYTIME_TD_MARKET:
        distribution = _anytime_td_distribution(row, ensemble, games)
    else:
        distribution = stat_distribution(row, market, ensemble=ensemble)
    mean_before_pack = distribution["mean"]
    pack_report: dict[str, Any] | None = None
    if pack_context is not None:
        distribution, pack_report = _pack_adjust(
            market, _position(player), distribution, row, games, pack_context, wr_role
        )
    cv = distribution["cv"] if distribution["cv"] is not None else CV_UNDEFINED
    confidence = _confidence(row, ensemble, distribution["games"])

    total = _season_total(player, spec)
    components: dict[str, float] = {}
    if market == ANYTIME_TD_MARKET:
        components = {
            "rushing_td_per_game": round(_num(player.get("rushing_tds")) / games, 3),
            "receiving_td_per_game": round(_num(player.get("receiving_tds")) / games, 3),
        }
    fantasy_points = round(_num(player.get("projection") or player.get("expected_fantasy_points")), 2)
    drivers = [str(item) for item in (player.get("drivers") or [])]
    data_season = _data_season(player)
    result = {
        "player_id": player.get("player_id"),
        "name": name,
        "team": _team_of(player),
        "position": _position(player),
        "category": category_label,
        "market": market,
        "unit": spec["unit"],
        "family": distribution["family"],
        "mean": distribution["mean"],
        "stdev": distribution["stdev"],
        "cv": round(float(cv), 4),
        "volatility": distribution["volatility"],
        "games": float(distribution["games"]),
        "season_total": round(total, 2),
        "per_game_rate": round(total / games, 3),
        "data_season": data_season,
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "risk_tier": _risk_tier(distribution["cv"]),
        "volatility_label": volatility_label(distribution["volatility"]),
        "projected_fantasy_points": fantasy_points,
        "projection": fantasy_points,
        "fantasy_points_per_game": round(_num(player.get("points_per_game")), 2),
        "expected_games": round(_num(player.get("expected_games")), 1),
        "projection_confidence": round(_num(player.get("projection_confidence")), 3),
        "projection_basis": str(player.get("projection_basis") or ""),
        "drivers": drivers,
        "status": status,
        "status_factor": round(float(factor), 4),
        "status_note": _status_note(name, status, float(factor)),
        "eligible": is_eligible(player, category_label),
        "components": components,
    }
    if pack_report is not None:
        result["pack_applied"] = bool(pack_report["applied"])
        result["mean_before_pack"] = mean_before_pack
        result["pack"] = pack_report
    return result


def suggest_line(projection: Mapping[str, Any]) -> float:
    """The sportsbook-style line to pre-fill: the half-point (``x.5``) nearest the mean.

    Books post ``x.5`` lines so a bet can never push, so this returns
    ``floor(mean) + 0.5`` -- the ``x.5`` closest to the projection (a mean of
    249.8 gives 249.5, 250.1 gives 250.5, and an exact 4.0 gives 4.5). The result
    is clamped into the category's ``[min_line, max_line]`` (never below
    ``min_line``), which also pins Anytime TD to 0.5. ``projection`` is a
    :func:`project_stat` dict (or any mapping with ``mean`` and ``category``).
    """
    mean = _num(projection.get("mean"))
    line = math.floor(mean) + 0.5
    spec = STAT_CATEGORIES.get(str(projection.get("category")))
    if spec is not None:
        line = min(max(line, spec["min_line"]), spec["max_line"])
    else:
        line = max(line, 0.5)
    return float(line)


# =========================================================================
# Probabilities
# =========================================================================


def _poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def _poisson_quantile(lam: float, q: float) -> float:
    """Smallest count ``k`` with ``P(X <= k) >= q``."""
    if lam <= 0.0:
        return 0.0
    for k in range(0, 2000):
        if _poisson_cdf(k, lam) >= q:
            return float(k)
    return float(math.ceil(lam))  # pragma: no cover - unreachable at real NFL magnitudes


def _percentiles(mean: float, stdev: float, family: str) -> dict[str, float]:
    """p10 / p50 / p90 of one game's outcome. Gaussian values are floored at zero (stats cannot go negative)."""
    if family == "poisson":
        return {
            "p10": _poisson_quantile(mean, 0.10),
            "p50": _poisson_quantile(mean, 0.50),
            "p90": _poisson_quantile(mean, 0.90),
        }
    return {
        "p10": round(max(0.0, mean - _Z90 * stdev), 1),
        "p50": round(max(0.0, mean), 1),
        "p90": round(max(0.0, mean + _Z90 * stdev), 1),
    }


def _line_probabilities(mean: float, stdev: float, family: str, line: float) -> tuple[float, float, float]:
    """``(P(over), P(under), P(push))``, each rounded to 4 dp.

    A non-integer line cannot push and matches ``betting.prop_model.over_under_probability``
    exactly (Gaussian: continuous normal at the line; Poisson: ``floor(line)`` is the
    highest count that is still "under"). An integer line prices the stat as an
    integer outcome (see the module docstring): over = P(X > k), under = P(X < k),
    push = P(X = k); for the Gaussian this uses the usual half-unit continuity
    correction, which makes it agree with the half-point lines on either side.
    """
    line = float(line)
    integer_line = line.is_integer()
    if family == "poisson":
        if integer_line:
            k = int(line)
            over = 1.0 - _poisson_cdf(k, mean)
            under = _poisson_cdf(k - 1, mean) if k >= 1 else 0.0
        else:
            under = _poisson_cdf(math.floor(line), mean)
            over = 1.0 - under
    else:
        if integer_line:
            over = 1.0 - _normal_cdf((line + 0.5 - mean) / stdev)
            under = _normal_cdf((line - 0.5 - mean) / stdev)
        else:
            under = _normal_cdf((line - mean) / stdev)
            over = 1.0 - under
    over = round(min(1.0, max(0.0, over)), 4)
    under = round(min(1.0, max(0.0, under)), 4)
    push = round(max(0.0, 1.0 - over - under), 4) if integer_line else 0.0
    return over, under, push


def _price_two_way(
    p_over: float, p_under: float, push: float, over_price: float, under_price: float, stake: float
) -> dict[str, Any]:
    """Edge, EV and price checks for both sides, push-aware (see the module docstring)."""
    decided = 1.0 - push
    if decided > 1e-9:
        decided_over = min(1.0, p_over / decided)
        decided_under = min(1.0, p_under / decided)
    else:  # the bet would push essentially always -- no information either way
        decided_over = decided_under = 0.5
    fair_over, fair_under = remove_vig_two_way(over_price, under_price)
    edge_over = edge_vs_fair(decided_over, over_price, under_price, side="a")
    edge_under = edge_vs_fair(decided_under, over_price, under_price, side="b")
    # EV with a refund on a push: (1 - push) * EV(decided probability). With push == 0 this is
    # exactly betting.odds_math.expected_value(p, price, stake), the same call evaluate_prop makes.
    ev_over = decided * expected_value(decided_over, over_price, stake)
    ev_under = decided * expected_value(decided_under, under_price, stake)
    return {
        "decided_over": decided_over,
        "decided_under": decided_under,
        "fair_over": fair_over,
        "fair_under": fair_under,
        "edge_over": edge_over,
        "edge_under": edge_under,
        "ev_over": ev_over,
        "ev_under": ev_under,
        "pricing_over": evaluate_price(decided_over, over_price, stake=stake),
        "pricing_under": evaluate_price(decided_under, under_price, stake=stake),
    }


def _side_pricing(pricing: Mapping[str, Any], ev: float, stake: float) -> dict[str, Any]:
    """``evaluate_price`` output with its EV replaced by the push-aware EV for the bet."""
    merged = dict(pricing)
    merged["expected_value"] = round(ev, 2)
    merged["ev_pct"] = round(ev / stake, 4) if stake else 0.0
    return merged


def _clean_line(line: Any) -> float:
    # ``True`` would silently become a 1.0 line; a bool here is always a caller bug.
    value = float("nan") if isinstance(line, bool) else _num(line, float("nan"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"line must be a positive number, got {line!r}")
    if value > MAX_LINE:
        raise ValueError(f"line must be at most {MAX_LINE:,.0f} (no NFL prop is posted that high), got {line!r}")
    return value


def _clean_price(price: Any, label: str) -> float:
    value = float("nan") if isinstance(price, bool) else _num(price, float("nan"))
    if not math.isfinite(value) or abs(value) < 100:
        raise ValueError(f"{label} must be American odds of at least 100 in size (e.g. -110, +105), got {price!r}")
    return value


def evaluate_line(
    player: Mapping[str, Any],
    category_label: str,
    line: float,
    over_price: float = DEFAULT_PRICE,
    under_price: float = DEFAULT_PRICE,
    *,
    stake: float = 100.0,
    pool: Sequence[Mapping[str, Any]] | None = None,
    pack_context: Mapping[str, Any] | None = None,
    wr_role: str = "wr",
) -> dict[str, Any]:
    """Price one pasted sportsbook line against the model.

    ``line`` is the book's number (``249.5`` or an integer such as ``249``);
    ``over_price`` / ``under_price`` are American odds (default -110/-110);
    ``stake`` sets the dollar basis of ``ev_*``; ``pool`` (optional) lets the
    reasoning rank the player against his positional peers.

    The seven native markets delegate to :func:`betting.prop_model.evaluate_prop`
    (so probabilities, confidence and risk tier match the existing page byte for
    byte on half-point lines); Anytime TD and integer-line push handling are
    computed here. See the module docstring for the sign conventions.

    Returned keys:

    * identity: ``player_id, name, team, position, category, market, unit,
      family, line, over_price, under_price, stake, integer_line``
    * model: ``probability_over, probability_under, push_probability`` (0.0 on
      half-point lines), ``mean, stdev, cv, volatility, volatility_label,
      z_score`` ((line - mean) / stdev), ``line_minus_mean``, ``p10, p50, p90``,
      ``percentiles`` (same three), ``distribution`` (``mean, stdev, cv,
      volatility, games, family``)
    * market: ``market_fair_probability_over/under`` (de-vigged),
      ``implied_probability_over/under`` (raw, vig included), ``hold``
    * pricing: ``edge_over/under``, ``ev_over/under`` (dollars on ``stake``),
      ``value_gap_over/under``, ``pricing_over/pricing_under``
      (:func:`modules.odds_input.evaluate_price` dicts with a push-aware EV)
    * decision: ``recommended_side`` (``"over"``/``"under"``, the larger edge),
      ``recommended_edge, recommended_ev, recommended_price,
      recommended_probability, recommended_value_gap, has_edge`` (EV > 0),
      ``verdict, verdict_tone, headline``
    * quality: ``confidence`` (0-1), ``confidence_label``, ``risk_tier``
      (``"low"|"medium"|"high"``)
    * context: ``projection`` (the :func:`project_stat` dict), ``status``,
      ``status_note``, ``reasoning`` (see :func:`reasoning_for`), ``warnings``

    ``recommended_probability`` is the outright ``P(side)``. On an integer line the edge,
    value gap, verdict and headline compare the market with the win probability *given the
    bet is decided*, ``P(side) / (1 - push)``, which is ``pricing_<side>["model_probability"]``.

    Raises ``ValueError`` for an unsupported category, a player that is not a row, a
    non-positive or absurd (> :data:`MAX_LINE`) line, bad prices, a non-positive stake,
    or a player with no games played.
    """
    spec = _category(category_label)
    line = _clean_line(line)
    over_price = _clean_price(over_price, "over_price")
    under_price = _clean_price(under_price, "under_price")
    stake = float("nan") if isinstance(stake, bool) else _num(stake, float("nan"))
    if not math.isfinite(stake) or stake <= 0:
        raise ValueError("stake must be a positive number")

    if pack_context is None:
        projection = project_stat(player, category_label)
    else:
        projection = project_stat(player, category_label, pack_context=pack_context, wr_role=wr_role)
    market = spec["market"]
    mean, stdev, family = projection["mean"], projection["stdev"], projection["family"]
    integer_line = line.is_integer()

    if market == ANYTIME_TD_MARKET:
        confidence = projection["confidence"]
        risk_tier = projection["risk_tier"]
        p_over, p_under, push = _line_probabilities(mean, stdev, family, line)
        distribution = {
            key: projection[key] for key in ("mean", "stdev", "cv", "volatility", "games", "family")
        }
    else:
        engine = evaluate_prop(
            _sane_row(player), {"market": market, "line": line, "over_price": over_price, "under_price": under_price}
        )
        confidence = engine["confidence"]
        risk_tier = engine["risk_tier"]
        distribution = dict(engine["distribution"])
        distribution["cv"] = projection["cv"]
        if projection.get("pack_applied"):
            # The engine priced its own (unadjusted) distribution; price the pack-adjusted one instead.
            distribution["mean"], distribution["stdev"] = mean, stdev
            p_over, p_under, push = _line_probabilities(mean, stdev, family, line)
        elif integer_line:
            p_over, p_under, push = _line_probabilities(mean, stdev, family, line)
        else:
            p_over, p_under, push = engine["model_probability_over"], engine["model_probability_under"], 0.0

    priced = _price_two_way(p_over, p_under, push, over_price, under_price, stake)
    edge_over, edge_under = priced["edge_over"], priced["edge_under"]
    ev_over, ev_under = priced["ev_over"], priced["ev_under"]
    recommended = "over" if edge_over >= edge_under else "under"
    over_side = recommended == "over"
    recommended_pricing = priced["pricing_over"] if over_side else priced["pricing_under"]
    recommended_ev = ev_over if over_side else ev_under
    recommended_edge = edge_over if over_side else edge_under
    recommended_price = over_price if over_side else under_price
    recommended_prob = p_over if over_side else p_under
    book_hold = hold(over_price, under_price)

    z_score = (line - mean) / stdev if stdev > 0 else 0.0
    percentiles = _percentiles(mean, stdev, family)

    warnings = _build_warnings(
        player, spec, projection, line, integer_line, push, book_hold, z_score, over_price, under_price
    )

    evaluation: dict[str, Any] = {
        "player_id": projection["player_id"],
        "name": projection["name"],
        "team": projection["team"],
        "position": projection["position"],
        "category": category_label,
        "market": market,
        "unit": spec["unit"],
        "family": family,
        "line": line,
        "over_price": over_price,
        "under_price": under_price,
        "stake": stake,
        "integer_line": integer_line,
        "probability_over": p_over,
        "probability_under": p_under,
        "push_probability": push,
        "mean": mean,
        "stdev": stdev,
        "cv": projection["cv"],
        "volatility": projection["volatility"],
        "volatility_label": projection["volatility_label"],
        "z_score": round(z_score, 3),
        "line_minus_mean": round(line - mean, 3),
        "p10": percentiles["p10"],
        "p50": percentiles["p50"],
        "p90": percentiles["p90"],
        "percentiles": percentiles,
        "distribution": distribution,
        "market_fair_probability_over": round(priced["fair_over"], 4),
        "market_fair_probability_under": round(priced["fair_under"], 4),
        "implied_probability_over": priced["pricing_over"]["implied_probability"],
        "implied_probability_under": priced["pricing_under"]["implied_probability"],
        "hold": round(book_hold, 4),
        "edge_over": round(edge_over, 4),
        "edge_under": round(edge_under, 4),
        "ev_over": round(ev_over, 2),
        "ev_under": round(ev_under, 2),
        "value_gap_over": priced["pricing_over"]["value_gap"],
        "value_gap_under": priced["pricing_under"]["value_gap"],
        "pricing_over": _side_pricing(priced["pricing_over"], ev_over, stake),
        "pricing_under": _side_pricing(priced["pricing_under"], ev_under, stake),
        "recommended_side": recommended,
        "recommended_edge": round(recommended_edge, 4),
        "recommended_ev": round(recommended_ev, 2),
        "recommended_price": recommended_price,
        "recommended_probability": recommended_prob,
        "recommended_value_gap": recommended_pricing["value_gap"],
        "has_edge": recommended_ev > 0,
        "verdict": recommended_pricing["verdict"],
        "verdict_tone": recommended_pricing["verdict_tone"],
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "risk_tier": risk_tier,
        "projection": projection,
        "status": projection["status"],
        "status_note": projection["status_note"],
    }
    evaluation["headline"] = _headline(evaluation, spec)
    evaluation["reasoning"] = reasoning_for(player, category_label, projection, evaluation, pool=pool)
    evaluation["warnings"] = warnings
    return evaluation


def _build_warnings(
    player: Mapping[str, Any],
    spec: Mapping[str, Any],
    projection: Mapping[str, Any],
    line: float,
    integer_line: bool,
    push: float,
    book_hold: float,
    z_score: float,
    over_price: float,
    under_price: float,
) -> list[str]:
    name = projection["name"]
    unit = spec["unit"]
    warnings: list[str] = []
    bad_fields = _nonfinite_fields(player)
    if bad_fields:
        warnings.append(
            f"The data for {name} has a non-numeric or non-finite value in {', '.join(bad_fields)}; it was treated "
            "as missing (zero, or the prior-season games for the sample), so this projection is unreliable."
        )
    if projection["status"] != status_utils.HEALTHY:
        warnings.append(projection["status_note"])
    if not projection["eligible"]:
        warnings.append(
            f"{name} ({projection['position'] or 'no position'}) is not normally offered {projection['category']}: "
            f"his rate is {_fmt_stat(projection['per_game_rate'], unit)} per game, so the model has little to work with."
        )
    if projection["games"] < THIN_SAMPLE_GAMES:
        warnings.append(
            f"Thin sample: the per-game rate rests on only {_games_text(projection['games'])}, and the engine's "
            "volatility read is unreliable with so little history."
        )
    if projection["mean"] <= 0.0:
        warnings.append(
            "The model projects no production for this stat, so the over probability is essentially zero "
            "and the numbers are degenerate."
        )
    if line < spec["min_line"] or line > spec["max_line"]:
        warnings.append(
            f"A {line:g} line is outside the usual {spec['min_line']:g}-{spec['max_line']:g} range for {projection['category']}; "
            "double-check it was entered for the right stat."
        )
    if (line * 2) != int(line * 2):
        warnings.append(f"{line:g} is not a half-point step; sportsbook lines are x.0 or x.5.")
    if abs(z_score) > 3.0 and projection["mean"] > 0.0:
        warnings.append(
            f"The line is {abs(z_score):.1f} standard deviations from the projection of "
            f"{_fmt_stat(projection['mean'], unit)}; check the line belongs to this player and stat."
        )
    if integer_line:
        warnings.append(
            f"Integer line: it can push ({push:.1%} per the model) and a push refunds the stake. "
            "Probabilities, EV and edge account for that; edge and value gap compare against the win "
            "probability given the bet is decided."
        )
    if book_hold < 0:
        warnings.append(
            f"The prices {format_american(over_price)} / {format_american(under_price)} imply a negative hold "
            f"({book_hold:.1%}); check both sides were entered correctly."
        )
    elif book_hold > 0.10:
        warnings.append(
            f"The prices {format_american(over_price)} / {format_american(under_price)} carry a {book_hold:.1%} hold, "
            f"well above the {hold(DEFAULT_PRICE, DEFAULT_PRICE):.1%} of a standard -110/-110 market; "
            "check both sides were entered correctly."
        )
    return warnings


def _headline(evaluation: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    side = evaluation["recommended_side"]
    fair = evaluation[f"market_fair_probability_{side}"]
    outright = evaluation["recommended_probability"]
    push = evaluation["push_probability"]
    stake = evaluation["stake"]
    ev = evaluation["recommended_ev"]
    line_text = f"{evaluation['line']:g} {evaluation['projection']['unit']}"
    if push > 0:
        # The edge compares the market with P(side) / (1 - push), not the outright P(side), so
        # quote that number (and the split behind it) or the headline would not reconcile with it.
        decided = evaluation[f"pricing_{side}"]["model_probability"]
        model_text = f"{format_pct(decided)} if decided ({format_pct(outright)} win, {format_pct(push)} push)"
    else:
        model_text = format_pct(outright)
    if evaluation["has_edge"]:
        return (
            f"Lean {side.upper()} {line_text} ({evaluation['name']}, {evaluation['category']}): model "
            f"{model_text} vs {format_pct(fair)} no-vig fair, EV {format_money(ev)} per ${stake:,.0f}."
        )
    return (
        f"No edge at these prices: the better side, {side.upper()} {line_text}, has a model probability of "
        f"{model_text} vs {format_pct(fair)} no-vig fair, EV {format_money(ev)} per ${stake:,.0f}."
    )


# =========================================================================
# Peer ranking and reasoning
# =========================================================================


def peer_rank(
    pool: Iterable[Mapping[str, Any]] | None, player: Mapping[str, Any], category_label: str
) -> dict[str, Any] | None:
    """Where ``player``'s raw per-game rate sits among same-position players eligible for the category.

    Peers must be eligible for the category (position and real volume) and have
    played at least :data:`THIN_SAMPLE_GAMES` games, so one-game cameos do not
    pad the field and make a starter look better than he is. Returns
    ``{"rank", "n", "percentile", "median_rate", "rate", "position"}`` --
    ``rank`` is 1 for the best of ``n`` players (the player is counted in ``n``
    whether or not he is in ``pool``); ``percentile`` is the mid-rank percentile
    0-100 (ties count half). ``None`` when there are fewer than three comparable
    players or the category is unknown.
    """
    spec = STAT_CATEGORIES.get(category_label)
    if spec is None:
        return None
    position = _position(player)
    rate = _raw_rate(player, spec)
    own_id = str(player.get("player_id"))
    peers: list[float] = []
    for other in pool or []:
        if not isinstance(other, Mapping) or _position(other) != position:
            continue
        if str(other.get("player_id")) == own_id or not is_eligible(other, category_label):
            continue
        if _games(other) < THIN_SAMPLE_GAMES:
            continue
        peers.append(_raw_rate(other, spec))
    n = len(peers) + 1
    if n < 3:
        return None
    higher = sum(1 for value in peers if value > rate)
    lower = sum(1 for value in peers if value < rate)
    tied = len(peers) - higher - lower + 1  # includes the player
    ordered = sorted(peers + [rate])
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return {
        "rank": higher + 1,
        "n": n,
        "percentile": round(100.0 * (lower + 0.5 * tied) / n, 1),
        "median_rate": round(median, 3),
        "rate": round(rate, 3),
        "position": position,
    }


def reasoning_for(
    player: Mapping[str, Any],
    category_label: str,
    projection: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    pool: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """The 'why' behind a priced line, as ``{"title", "detail", "tone"}`` items (at least five, every one numeric).

    ``tone`` is relative to the model's recommended side: ``"pro"`` supports the
    read (or the trust one can place in it), ``"con"`` argues against it,
    ``"neutral"`` is context. Items: the per-game rate and its sample size; the
    player's rank among same-position peers (only when ``pool`` is given and has
    enough comparable players); volatility and what it means for the spread of
    outcomes; where the line sits versus the projection (z-score, plain
    English); why a Gaussian or Poisson curve is used; where the projection
    comes from (including the regression note on the row); availability;
    model vs market; and what the price needs to break even.
    """
    spec = _category(category_label)
    unit = spec["unit"]
    items: list[dict[str, str]] = []

    def add(title: str, detail: str, tone: str) -> None:
        items.append({"title": title, "detail": detail, "tone": tone})

    name = str(projection.get("name") or _name(player))
    games = _num(projection.get("games"))
    rate = _num(projection.get("per_game_rate"))
    total = _num(projection.get("season_total"))
    mean = _num(projection.get("mean"))
    stdev = _num(projection.get("stdev"))
    cv = _num(projection.get("cv"))
    volatility = _num(projection.get("volatility"))
    family = str(projection.get("family") or spec["family"])
    season = projection.get("data_season")
    season_text = f"{season}" if season else "the source season"
    line = _num(evaluation.get("line"))
    side = str(evaluation.get("recommended_side") or "over")

    # 1. Rate and sample size -------------------------------------------------
    if spec["market"] == ANYTIME_TD_MARKET:
        parts = projection.get("components") or {}
        rate_text = (
            f"{name} scored {_num(parts.get('rushing_td_per_game')):.2f} rushing TDs + "
            f"{_num(parts.get('receiving_td_per_game')):.2f} receiving TDs per game in {season_text} "
            f"= {rate:.2f} TDs per game ({total:,.0f} total over {_games_text(games)})."
        )
    else:
        rate_text = (
            f"{name} averaged {_fmt_stat(rate, unit)} per game in {season_text} "
            f"({total:,.0f} {unit} over {_games_text(games)})."
        )
    if games >= FULL_SAMPLE_GAMES:
        add("Per-game rate and sample", f"{rate_text} {games:g} games is a full-season sample, so the rate is well supported.", "pro")
    elif games >= THIN_SAMPLE_GAMES:
        add("Per-game rate and sample", f"{rate_text} {games:g} games is a moderate sample; expect the true rate to sit somewhat off this number.", "neutral")
    else:
        add("Per-game rate and sample", f"{rate_text} Only {_games_text(games)} is a thin sample: the rate could be mostly noise.", "con")

    # 2. Rank among peers -------------------------------------------------------
    if pool is not None:
        ranking = peer_rank(pool, player, category_label)
        if ranking is not None:
            pct = ranking["percentile"]
            if pct >= 75:
                supports = "over"
            elif pct <= 25:
                supports = "under"
            else:
                supports = None
            tone = "neutral" if supports is None else ("pro" if supports == side else "con")
            add(
                f"Rank among {ranking['position']}s",
                f"At {_fmt_stat(ranking['rate'], unit)} per game, {name} ranks #{ranking['rank']} of {ranking['n']} "
                f"{ranking['position']}s (the player plus {ranking['n'] - 1} peers with {THIN_SAMPLE_GAMES:g}+ games "
                f"who qualify for {category_label}; {_ordinal(pct)} percentile); the middle "
                f"{ranking['position']} sits at {_fmt_stat(ranking['median_rate'], unit)}.",
                tone,
            )

    # 3. Volatility and spread ----------------------------------------------------
    p10, p50, p90 = _num(evaluation.get("p10")), _num(evaluation.get("p50")), _num(evaluation.get("p90"))
    label = volatility_label(volatility)
    if family == "poisson":
        spread_text = (
            f"Counts vary by their own size: the standard deviation is sqrt(mean) = {stdev:.2f} {unit} on a mean of "
            f"{mean:.2f} (cv {cv:.2f}). The engine's volatility read for {name} ({volatility:.2f}, {label}) is "
            "informational here: it does not widen or narrow a count market, whose spread is fixed by the mean."
        )
    else:
        spread_text = (
            f"Volatility {volatility:.2f} ({label}) maps to a coefficient of variation of {cv:.2f}, so one standard "
            f"deviation is {_fmt_stat(stdev, unit)} around a mean of {_fmt_stat(mean, unit)}."
        )
    add(
        "Volatility and spread of outcomes",
        f"{spread_text} The middle 80% of games land between {p10:g} and {p90:g} {unit} (median {p50:g}); "
        f"{'a wide band means single games routinely land far from the average, so a small edge is easily swamped by variance' if cv >= 0.5 else 'a tighter band means outcomes cluster near the mean'}.",
        # The tone must agree with the sentence above it, which keys on cv. For a count market the
        # engine's volatility label is not part of the spread, so it must not colour the tone.
        (
            "con" if cv >= 0.5 else "pro" if (label == "Low" and family == "gaussian") else "neutral"
        ),
    )

    # 4. Line versus the projection ------------------------------------------------
    diff = line - mean
    z = _num(evaluation.get("z_score"))
    if abs(z) < 0.15:
        placement = f"The {line:g} line sits right on the projection of {_fmt_stat(mean, unit)} ({z:+.2f} standard deviations): a near coin flip before the price."
        line_tone = "neutral"
    elif diff > 0:
        placement = (
            f"The {line:g} line is {_fmt_stat(abs(diff), unit)} above the projection of {_fmt_stat(mean, unit)} "
            f"({z:+.2f} standard deviations), so the over needs a better-than-projected game."
        )
        line_tone = "pro" if side == "under" else "con"
    else:
        placement = (
            f"The {line:g} line is {_fmt_stat(abs(diff), unit)} below the projection of {_fmt_stat(mean, unit)} "
            f"({z:+.2f} standard deviations), so the under needs a worse-than-projected game."
        )
        line_tone = "pro" if side == "over" else "con"
    add("Your line vs the projection", placement, line_tone)

    # 5. Distribution family ---------------------------------------------------------
    if family == "poisson":
        add(
            "Why a Poisson count model",
            f"{category_label} is a count of discrete events, so it is modeled as Poisson with mean {mean:.2f} "
            f"(variance equals the mean). That gives P(0) = {_poisson_pmf(0, mean):.1%}. Weekly counts run a little "
            "over-dispersed relative to Poisson (variance about 1.0 to 1.2 times the mean for 2025 starters), so "
            "extreme games are slightly likelier than shown.",
            "neutral",
        )
    else:
        gaussian_text = (
            f"Yardage is modeled as a normal curve around {_fmt_stat(mean, unit)} with a standard deviation of "
            f"{_fmt_stat(stdev, unit)}: the coefficient of variation is scaled from {CV_LOW:.2f} to {CV_HIGH:.2f} "
            f"by the engine's volatility read ({volatility:.2f}). It is a modeling choice, not observed weekly variance."
        )
        gaussian_tone = "neutral"
        if spec["market"] == "passing_yards":
            low, median, high = OBSERVED_QB_PASSING_CV
            if cv > high:
                gaussian_text += (
                    f" For scale, the median starting QB's weekly passing yards varied with a cv of about {median:.2f} "
                    f"(middle half {low:.2f} to {high:.2f}) in 2025, so this {cv:.2f} curve is wider than history: it pulls "
                    "over/under probabilities toward 50% and shrinks apparent edges on lines away from the mean."
                )
                gaussian_tone = "con"
            elif cv < low:
                gaussian_text += (
                    f" For scale, the median starting QB's weekly passing yards varied with a cv of about {median:.2f} "
                    f"(middle half {low:.2f} to {high:.2f}) in 2025, so this {cv:.2f} curve is narrower than history and "
                    "tails are likelier than shown."
                )
                gaussian_tone = "con"
            else:
                gaussian_text += (
                    f" That {cv:.2f} sits inside the {low:.2f} to {high:.2f} middle half of starting QBs' weekly "
                    f"passing-yards cv in 2025 (median {median:.2f})."
                )
        add("Why a Gaussian curve", gaussian_text, gaussian_tone)

    # 6. Where the projection comes from ----------------------------------------------
    basis = str(projection.get("projection_basis") or "").strip()
    fantasy = _num(projection.get("projected_fantasy_points"))
    expected_games = _num(projection.get("expected_games"))
    drivers = [str(d) for d in (projection.get("drivers") or [])][:4]
    basis_bits = []
    if basis:
        basis_bits.append(f"Fantasy projection basis: {basis}.")
    if fantasy:
        basis_bits.append(
            f"That is {fantasy:.1f} projected points ({_num(projection.get('fantasy_points_per_game')):.1f} per game over "
            f"{expected_games:g} expected games)."
        )
    if drivers:
        basis_bits.append("Drivers: " + "; ".join(drivers) + ".")
    basis_bits.append(
        f"The {_fmt_stat(mean, unit)} stat mean above is the raw {season_text} per-game rate, "
        "not the regressed fantasy number, and it is not adjusted for opponent, game script or weather."
    )
    add("Where the projection comes from", " ".join(basis_bits), "neutral")

    # 7. Availability --------------------------------------------------------------------
    status = str(projection.get("status") or status_utils.HEALTHY)
    history_note = f"{name} played {_games_text(games)} in {season_text}"
    history_note += (
        f", and the fantasy projection expects {expected_games:g} games over its projection season; "
        "the per-game rate assumes the player is available."
        if expected_games > 0
        else "."
    )
    status_note = str(projection.get("status_note") or "No availability concern.")
    if status == status_utils.HEALTHY:
        add("Availability", f"{status_note} {history_note}", "pro" if status_utils.has_status_data() else "neutral")
    else:
        add("Availability", f"{status_note} {history_note}", "con")

    # 8 + 9. Model vs market and price math ----------------------------------------------
    if "probability_over" in evaluation and "market_fair_probability_over" in evaluation:
        p_over = _num(evaluation.get("probability_over"))
        p_under = _num(evaluation.get("probability_under"))
        push = _num(evaluation.get("push_probability"))
        fair_over = _num(evaluation.get("market_fair_probability_over"))
        fair_under = _num(evaluation.get("market_fair_probability_under"))
        push_text = f", push {push:.1%}" if push > 0 else ""
        edge = _num(evaluation.get("recommended_edge"))
        ev = _num(evaluation.get("recommended_ev"))
        stake = _num(evaluation.get("stake"), 100.0)
        pricing_over = evaluation.get("pricing_over") or {}
        pricing_under = evaluation.get("pricing_under") or {}
        # On an integer line the edge compares the market with P(side) / (1 - push), so show those
        # numbers next to the fair probabilities they are measured against; on a half-point line
        # (push == 0) the decided probability is just P(side) and nothing extra is needed.
        decided_text = ""
        if push > 0:
            decided_text = (
                f" A push refunds the stake, so the comparison uses the win probability given the bet is decided: "
                f"over {_num(pricing_over.get('model_probability'), p_over):.1%} / "
                f"under {_num(pricing_under.get('model_probability'), p_under):.1%}."
            )
        add(
            "Model vs market price",
            f"Model: over {p_over:.1%} / under {p_under:.1%}{push_text}.{decided_text} Market no-vig fair: "
            f"over {fair_over:.1%} / under {fair_under:.1%}. Best side {side.upper()}: edge "
            f"{format_signed_pct(edge)} versus fair, EV {format_money(ev)} per ${stake:,.0f}.",
            "pro" if ev > 0 else "con",
        )
        over_price = _num(evaluation.get("over_price"), DEFAULT_PRICE)
        under_price = _num(evaluation.get("under_price"), DEFAULT_PRICE)
        implied_over = _num(evaluation.get("implied_probability_over"))
        implied_under = _num(evaluation.get("implied_probability_under"))
        book_hold = _num(evaluation.get("hold"))
        side_probability = _num(evaluation.get("recommended_probability"))
        if push > 0:
            decided = _num((pricing_over if side == "over" else pricing_under).get("model_probability"), side_probability)
            probability_text = (
                f"The model's {side} win probability given the bet is decided is {format_pct(decided)} "
                f"({format_pct(side_probability)} outright, {push:.1%} push)"
            )
        else:
            probability_text = f"The model's {side} probability is {format_pct(side_probability)}"
        if projection.get("pack_applied"):
            pack = projection.get("pack") or {}
            before, after = _num(projection.get("mean_before_pack")), _num(projection.get("mean"))
            change = (after / before - 1.0) if before > 0 else 0.0
            sentences = " ".join(str(t) for t in (pack.get("explanations") or [])[:2])
            raises = after > before
            add(
                "Matchup upgrade pack",
                f"For this opponent the pack moves the {unit} projection from {_fmt_stat(before, unit)} to "
                f"{_fmt_stat(after, unit)} ({change:+.1%}); the line above is priced on the adjusted number. {sentences} "
                "The weights are priors, several inputs are proxies, and the baseline is the player's prior-season rate, "
                "so treat the change as a lean.",
                "pro" if (raises == (side == "over")) else "con",
            )
        add(
            "What the price needs",
            f"At {format_american(over_price)} / {format_american(under_price)} the break-even win rates are "
            f"{implied_over:.1%} (over) and {implied_under:.1%} (under); the book's hold is {book_hold:.1%}. "
            f"{probability_text}, a value gap of "
            f"{format_signed_pct(_num(evaluation.get('recommended_value_gap')))} against the raw break-even.",
            "pro" if _num(evaluation.get("recommended_value_gap")) > 0 else "neutral",
        )
    return items


# =========================================================================
# Input validation for the page
# =========================================================================


def validate_inputs(
    category_label: str,
    line_text: Any,
    over_price_text: Any = "",
    under_price_text: Any = "",
) -> dict[str, Any]:
    """Parse and sanity-check the pasted line and prices so the page can show friendly errors.

    Blank is not an error: a blank line means "nothing entered yet"
    (``blank_line=True``, ``ok=False``, no error), and a blank price defaults to
    -110. Anything unparseable, out of the category's ``[min_line, max_line]``
    range, or not a half-point step is reported in ``errors`` / ``field_errors``.

    Returned keys: ``ok`` (the line and both prices are usable), ``line``,
    ``over_price``, ``under_price`` (floats, or ``None`` when unusable),
    ``errors`` (list of messages), ``field_errors`` (``category`` / ``line`` /
    ``over_price`` / ``under_price`` -> message or ``None``), ``blank_line``,
    ``defaults_used`` (price fields that fell back to -110), ``notes`` (gentle
    hints, e.g. one price left blank) and ``message`` (the single most relevant
    line to show, ``""`` when everything is fine).
    """
    field_errors: dict[str, str | None] = {"category": None, "line": None, "over_price": None, "under_price": None}
    errors: list[str] = []
    notes: list[str] = []
    defaults_used: list[str] = []

    spec = STAT_CATEGORIES.get(category_label)
    if spec is None:
        field_errors["category"] = f"Choose a stat category ({', '.join(STAT_CATEGORIES)})."
        errors.append(field_errors["category"])  # type: ignore[arg-type]

    line: float | None = None
    blank_line = False
    parsed_line = parse_number(
        line_text,
        allow_negative=False,
        minimum=spec["min_line"] if spec else None,
        maximum=spec["max_line"] if spec else None,
        label=f"{category_label} line" if spec else "line",
    )
    if parsed_line.error:
        field_errors["line"] = parsed_line.error
        errors.append(parsed_line.error)
    elif parsed_line.value is None:
        blank_line = True
    else:
        line = parsed_line.value
        if line <= 0:
            field_errors["line"] = "The line must be greater than zero."
            errors.append(field_errors["line"])  # type: ignore[arg-type]
            line = None
        elif (line * 2) != int(line * 2):
            field_errors["line"] = "Prop lines move in half-point steps, e.g. 249.5 or 250."
            errors.append(field_errors["line"])  # type: ignore[arg-type]
            line = None

    prices: dict[str, float | None] = {}
    entered_blank: list[str] = []
    for field, text in (("over_price", over_price_text), ("under_price", under_price_text)):
        parsed = parse_odds(text)
        if parsed.error:
            field_errors[field] = parsed.error
            errors.append(f"{'Over' if field == 'over_price' else 'Under'} price: {parsed.error}")
            prices[field] = None
        elif parsed.price is None:
            prices[field] = DEFAULT_PRICE
            defaults_used.append(field)
            entered_blank.append(field)
        else:
            prices[field] = parsed.price
    if entered_blank and len(entered_blank) == 1:
        blank_name = "Over" if entered_blank[0] == "over_price" else "Under"
        notes.append(f"{blank_name} price left blank - assuming {format_american(DEFAULT_PRICE)}.")

    ok = spec is not None and line is not None and not errors
    if errors:
        message = errors[0]
    elif blank_line:
        message = "Paste the sportsbook line to price it against the model."
    else:
        message = ""
    return {
        "ok": bool(ok),
        "line": line,
        "over_price": prices.get("over_price"),
        "under_price": prices.get("under_price"),
        "errors": errors,
        "field_errors": field_errors,
        "blank_line": blank_line,
        "defaults_used": defaults_used,
        "notes": notes,
        "message": message,
    }


__all__ = [
    "ANYTIME_TD_MARKET",
    "CATEGORY_LABELS",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MODERATE",
    "CV_UNDEFINED",
    "DEFAULT_PRICE",
    "FULL_SAMPLE_GAMES",
    "MAX_LINE",
    "OBSERVED_QB_PASSING_CV",
    "STAT_CATEGORIES",
    "THIN_SAMPLE_GAMES",
    "VOLATILITY_HIGH",
    "VOLATILITY_MODERATE",
    "clear_player_pool_cache",
    "confidence_label",
    "eligible_categories",
    "evaluate_line",
    "find_player",
    "is_eligible",
    "load_player_pool",
    "normalize_team",
    "peer_rank",
    "player_label",
    "player_status",
    "players_for_team",
    "project_stat",
    "reasoning_for",
    "suggest_line",
    "team_label",
    "teams_in_pool",
    "validate_inputs",
    "volatility_label",
]
