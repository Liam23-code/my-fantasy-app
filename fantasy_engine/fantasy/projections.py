"""Forward projections: turn last season's actuals into next season's numbers.

Usage::

    from fantasy.projections import load_forward_projections, upcoming_season

    target = upcoming_season()                      # e.g. 2026
    players = load_forward_projections(scoring_mode="ppr")
    players[0]["projection_season"]                 # 2026
    players[0]["projection_basis"]                  # "2026 projection from 2025 actuals; ..."

Why this module exists
----------------------
:func:`fantasy.data_loader.load_real_projections` is explicit that its
``projection`` field is *last season's actual production*, not a forecast. That
is a legitimate baseline, but it is a **backward-looking** one, and every
surface that called it "a projection" was overstating what the number knew.
Two concrete failures follow from using it raw:

* It buys last season's production at this season's price. The holdout
  documented on :data:`fantasy.assistant.MARKET_BLEND_WEIGHT` measured exactly
  this -- a pure prior-season-points ranker finished 11th of 12.
* It projects an injured player's short season forward as a short season. A
  back who played 6 games and scored 90 points is not a 90-point player; he is
  a 15-points-per-game player who missed most of a year.

This module is the correction, and it is the reason the UI can honestly print
"2026 Projections" instead of a 2025 total with a 2026 label on it.

The three stages
----------------
1. **Availability.** Games played is regressed toward the position's own
   observed availability rate (:data:`AVAILABILITY_PRIOR_GAMES`), producing an
   ``expected_games`` for the target season rather than replaying last
   season's game count.
2. **Rate regression.** Points per game is shrunk toward the position's own
   median rate, weighted by sample size (:data:`RATE_PRIOR_GAMES`). A 17-game
   sample barely moves; a 3-game sample moves a lot. This is standard
   small-sample regression, computed from the pool rather than assumed.
3. **Market reconciliation.** The market's within-position ordering (ADP) is
   mapped onto the pool's own rate-derived value curve and blended in at
   :data:`MARKET_RECONCILIATION_WEIGHT`. ADP is forward-looking information our
   backward-looking stats cannot see -- team changes, holdouts, a rookie
   taking over a backfield -- and the holdout evidence above says it is the
   better predictor of next season. It is blended, not substituted, so a
   player the market has mispriced still moves the number.

What this still does not know
-----------------------------
Deliberately stated rather than papered over, because a projection that
overclaims is worse than one with a documented edge:

* **Rookies do not appear at all.** They have no prior-season production to
  project from, and this module does not invent one.
* **No age curve.** The nflverse feed behind the pool carries no birth date,
  and guessing one per player would be fabrication.
* **Team and role changes are only seen through ADP.** A back who changed
  teams in March is corrected only to the extent the market has priced it.

``projection_confidence`` on every returned player quantifies the first two:
sample size and whether the market had an opinion at all.
"""

from __future__ import annotations

import datetime as _dt
import statistics
from collections.abc import Mapping
from typing import Any

from fantasy.data_loader import latest_completed_season, load_real_projections
from fantasy.models import LeagueSettings
from fantasy.utils import clamp, safe_float

#: Games in an NFL regular season -- the ceiling on ``expected_games``.
REGULAR_SEASON_GAMES = 17

#: Strength of the availability prior, in games. At 17 an availability
#: estimate is a 50/50 blend of "what this player actually did" and "what this
#: position typically does", which is the right skepticism for a single
#: season of games-played evidence.
AVAILABILITY_PRIOR_GAMES = 17.0

#: Strength of the per-game rate prior, in games. Much weaker than the
#: availability prior on purpose: a player's scoring *rate* over a full season
#: is a far more reliable signal than their games-played count, so a 17-game
#: sample should mostly keep its own rate.
RATE_PRIOR_GAMES = 6.0

#: Games required before a player's rate contributes to their position's
#: median anchor. Without it, a one-game cameo drags the anchor around.
MIN_RATE_SAMPLE_GAMES = 6

#: How much of the final projection comes from the market's own ordering.
#:
#: Kept well below :data:`fantasy.assistant.MARKET_BLEND_WEIGHT` (1.0) on
#: purpose. That constant governs *ranking*, where the holdout showed the
#: market should decide the order outright; this one governs the *points
#: number itself*, which the ranking layer then re-reads. Pushing this to 1.0
#: too would double-count ADP -- the market would set the order and the value
#: scale the order is measured on. At 0.35 the projection stays predominantly
#: a real rate-times-games estimate, with the market correcting the situations
#: last season's box scores cannot see.
MARKET_RECONCILIATION_WEIGHT = 0.35


def upcoming_season(today: _dt.date | None = None) -> int:
    """The season currently being drafted for -- one past the last completed one.

    Between March and the following February this is the season about to be
    played, which is exactly the season a draft board should be projecting.
    """
    return latest_completed_season(today) + 1


def projection_season_label(season: int | None = None) -> str:
    """The UI label for a projection season, e.g. ``"2026 Projections"``."""
    return f"{int(season) if season is not None else upcoming_season()} Projections"


def projected_points(
    player: Any,
    league_settings: dict[str, Any] | LeagueSettings | None = None,
) -> float | None:
    """A player's precomputed projection, when it is usable under this league.

    Returns ``None`` -- meaning "score the raw stat line instead" -- when the
    player carries no precomputed projection, or when it carries one that was
    computed under a *different* scoring mode than this league uses. That last
    check is what makes it safe for scoring/ranking code to prefer a
    precomputed number: a PPR projection must never be silently reported as a
    standard-league total just because it was already sitting on the record.
    """
    if isinstance(player, Mapping):
        row: Mapping[str, Any] = player
    elif hasattr(player, "model_dump") and callable(player.model_dump):
        row = player.model_dump()
    elif hasattr(player, "__dict__"):
        row = vars(player)
    else:
        return None

    settings = league_settings if isinstance(league_settings, LeagueSettings) else LeagueSettings(**(league_settings or {}))
    source_mode = str(row.get("scoring_mode") or "").strip().lower()
    if source_mode and source_mode != str(settings.scoring_mode).strip().lower():
        return None
    if source_mode and settings.custom_rules:
        # A custom rule set can rescore anything; a projection baked under the
        # named mode is not it.
        return None

    for key in ("projection", "expected_fantasy_points"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _position_of(player: Mapping[str, Any]) -> str:
    return str(player.get("position", "")).strip().upper()


def _games_played(player: Mapping[str, Any], regular_season_games: int) -> float:
    games = safe_float(player.get("games_played"), 0.0)
    if games > 0:
        return min(float(regular_season_games), games)
    # No games column (an uploaded pool, say) -- treat the projection as a
    # full-season number rather than inventing a short season for it.
    return float(regular_season_games)


def _points_per_game(player: Mapping[str, Any], games: float) -> float:
    rate = safe_float(player.get("points_per_game"), 0.0)
    if rate > 0:
        return rate
    total = safe_float(player.get("projection"), 0.0) or safe_float(player.get("expected_fantasy_points"), 0.0)
    return total / games if games > 0 else 0.0


def _position_anchors(
    players: list[dict[str, Any]],
    regular_season_games: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-position median scoring rate and mean availability, from the pool itself."""
    rates: dict[str, list[float]] = {}
    availability: dict[str, list[float]] = {}
    for player in players:
        position = _position_of(player)
        if not position:
            continue
        games = _games_played(player, regular_season_games)
        availability.setdefault(position, []).append(clamp(games / regular_season_games, 0.0, 1.0))
        if games >= MIN_RATE_SAMPLE_GAMES:
            rates.setdefault(position, []).append(_points_per_game(player, games))

    anchors: dict[str, float] = {}
    for position, values in rates.items():
        anchors[position] = statistics.median(values) if values else 0.0
    # Positions where nobody cleared the sample floor still need an anchor;
    # use every player at the position rather than falling back to zero, which
    # would regress them all toward "worthless".
    for player in players:
        position = _position_of(player)
        if position and position not in anchors:
            same = [_points_per_game(p, _games_played(p, regular_season_games)) for p in players if _position_of(p) == position]
            anchors[position] = statistics.median(same) if same else 0.0

    mean_availability = {position: (sum(values) / len(values) if values else 1.0) for position, values in availability.items()}
    return anchors, mean_availability


def _market_curve(
    players: list[dict[str, Any]],
    rate_projections: dict[int, float],
) -> dict[int, float]:
    """Market-implied projection per player, from within-position ADP order.

    The market's Nth-best RB is valued at the Nth-best *rate* projection among
    RBs. That maps the market's ordering onto a real, pool-derived value curve
    instead of an assumed points-per-rank slope. Players the market has no
    opinion on (no ADP) get no entry and keep their rate projection.
    """
    by_position: dict[str, list[dict[str, Any]]] = {}
    for player in players:
        by_position.setdefault(_position_of(player), []).append(player)

    implied: dict[int, float] = {}
    for group in by_position.values():
        curve = sorted((rate_projections[id(p)] for p in group), reverse=True)
        if not curve:
            continue
        ranked = sorted(
            (p for p in group if p.get("adp") is not None),
            key=lambda p: safe_float(p["adp"]),
        )
        for rank, player in enumerate(ranked):
            implied[id(player)] = curve[min(rank, len(curve) - 1)]
    return implied


def _confidence(games: float, has_adp: bool, regular_season_games: int) -> float:
    """0-1 confidence: how much real evidence sits behind this projection."""
    sample = clamp(games / regular_season_games, 0.0, 1.0)
    return round(clamp(0.35 + 0.45 * sample + (0.20 if has_adp else 0.0), 0.0, 1.0), 3)


def project_forward(
    players: list[dict[str, Any]],
    target_season: int | None = None,
    market_weight: float | None = None,
    regular_season_games: int = REGULAR_SEASON_GAMES,
) -> list[dict[str, Any]]:
    """Project a pool of prior-season actuals forward to ``target_season``.

    Takes the output of :func:`fantasy.data_loader.load_real_projections` (or
    any pool with ``projection``/``points_per_game``/``games_played``) and
    returns **new** dicts -- the input is never mutated -- whose ``projection``
    is a forward estimate for ``target_season`` rather than a prior-season
    total. Each returned player additionally carries:

    ``projection_season``
        The season being projected (``target_season``).
    ``prior_season`` / ``prior_season_points``
        Where the estimate came from, so the change stays auditable.
    ``expected_games``
        Games the player is projected to play, after availability regression.
    ``projection_method`` / ``projection_basis``
        Human-readable statement of how the number was produced.
    ``projection_confidence``
        0-1, from sample size and whether the market had an ADP opinion.

    ``floor``/``median``/``ceiling`` are rescaled by the same ratio the
    projection moved, preserving the *shape* of the player's real observed
    week-to-week variance rather than inventing a new band.

    Returns ``[]`` for an empty pool, and leaves a player untouched (beyond
    the provenance fields) when there is no usable prior number to project.
    """
    pool = [dict(player) for player in (players or []) if isinstance(player, dict)]
    if not pool:
        return []

    target = int(target_season) if target_season is not None else upcoming_season()
    weight = clamp(MARKET_RECONCILIATION_WEIGHT if market_weight is None else float(market_weight), 0.0, 1.0)
    games_cap = max(1, int(regular_season_games))

    anchors, mean_availability = _position_anchors(pool, games_cap)

    rate_projections: dict[int, float] = {}
    expected_games: dict[int, float] = {}
    for player in pool:
        position = _position_of(player)
        games = _games_played(player, games_cap)
        rate = _points_per_game(player, games)

        anchor = anchors.get(position, 0.0)
        regressed_rate = (games * rate + RATE_PRIOR_GAMES * anchor) / (games + RATE_PRIOR_GAMES)

        position_availability = mean_availability.get(position, 1.0)
        availability = (games + AVAILABILITY_PRIOR_GAMES * position_availability) / (games_cap + AVAILABILITY_PRIOR_GAMES)
        projected_games = round(games_cap * clamp(availability, 0.0, 1.0), 1)

        expected_games[id(player)] = projected_games
        rate_projections[id(player)] = regressed_rate * projected_games

    implied = _market_curve(pool, rate_projections)

    projected: list[dict[str, Any]] = []
    for player in pool:
        key = id(player)
        rate_projection = rate_projections[key]
        market_projection = implied.get(key)
        if market_projection is None:
            projection = rate_projection
        else:
            projection = (1.0 - weight) * rate_projection + weight * market_projection

        prior_points = safe_float(player.get("projection"), 0.0) or safe_float(player.get("expected_fantasy_points"), 0.0)
        ratio = (projection / prior_points) if prior_points > 0 else 1.0
        games = _games_played(player, games_cap)
        projected_games = expected_games[key] or 1.0

        updated = dict(player)
        updated["prior_season"] = player.get("season")
        updated["prior_season_points"] = round(prior_points, 2)
        updated["prior_games_played"] = int(round(games))
        updated["projection"] = round(projection, 2)
        updated["expected_fantasy_points"] = round(projection, 2)
        updated["points_per_game"] = round(projection / projected_games, 2)
        updated["expected_games"] = projected_games
        updated["season"] = target
        updated["projection_season"] = target
        updated["projection_method"] = "regressed rate x expected games, market-reconciled"
        updated["projection_confidence"] = _confidence(games, player.get("adp") is not None, games_cap)
        updated["projection_basis"] = (
            f"{target} projection from {player.get('season', 'prior')} actuals "
            f"({int(round(games))} games -> {projected_games:g} expected); rate regressed to the "
            f"{_position_of(player) or 'position'} median, {weight:.0%} market-reconciled"
        )
        for band in ("floor", "median", "ceiling"):
            value = player.get(band)
            if value is not None:
                updated[band] = round(safe_float(value) * ratio, 2)
        projected.append(updated)

    projected.sort(key=lambda p: p["projection"], reverse=True)
    return projected


def load_forward_projections(
    season: int | None = None,
    target_season: int | None = None,
    scoring_mode: str = "ppr",
    custom_rules: dict[str, Any] | None = None,
    include_adp: bool = True,
    min_games: int = 1,
    market_weight: float | None = None,
) -> list[dict[str, Any]]:
    """Load real prior-season data and project it forward in one call.

    ``season`` is the *source* season of actuals (defaults to the most recent
    completed one); ``target_season`` is what gets projected (defaults to
    :func:`upcoming_season`). Every other argument is passed straight through
    to :func:`fantasy.data_loader.load_real_projections`, including its
    :class:`fantasy.data_loader.RealDataUnavailable` failure mode -- callers
    should surface an empty state rather than substituting fabricated players.
    """
    source = load_real_projections(
        season=season,
        scoring_mode=scoring_mode,
        custom_rules=custom_rules,
        include_adp=include_adp,
        min_games=min_games,
    )
    return project_forward(source, target_season=target_season, market_weight=market_weight)
