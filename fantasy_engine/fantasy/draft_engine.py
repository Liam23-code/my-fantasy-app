"""draft_engine: a round-aware, roster-reactive "who do I take now?" board.

This is the recommender the live Draft Room calls. It exists alongside -- not
instead of -- :func:`fantasy.assistant.suggest_draft_picks` (the frozen,
holdout-validated model behind ``user_brain`` / ``simulate_draft``) and
:func:`fantasy.assistant.get_best_pick_for_round` (kept for its own callers and
tests). Nothing here changes those.

Why a new module
----------------
The old Draft Room board had four structural faults, all confirmed against the
live code:

1. **Drafted players kept appearing.** The mock sim's seeded bots diverge from
   a real draft room, so a player taken in your real league stays on the sim
   board and gets recommended. There was no way to say "these are gone".
2. **A worse player outranked a clearly better one.** The primary sort key was
   ``abs(current_pick - adp)`` -- an *unsigned* distance on a quantity whose
   *sign* is the whole signal. A stud who fell 15 picks past ADP (a steal) and
   a player you would reach 15 picks early for scored identically, and VORP,
   the tiebreaker, was often never consulted.
3. **Recommendations sat 1-2 rounds off.** Same unsigned key: after round 3 the
   nearest ADP to your slot is structurally a round or two away, so the top
   pick was off by construction and the ``abs()`` could never prefer the
   faller.
4. **It collapsed after round 5.** Roster need entered only as a *multiplier*
   on VORP, and mid-draft almost every player's VORP is negative -- so a
   *higher* need made the score *more* negative and the engine demoted exactly
   the position you needed.

The fix is a single, legible **composite score**: signed ADP value, positional
scarcity, roster need (as an *additive* term, so it can never invert), VORP as
a tiebreaker, and a small breakout nudge. Weights put ADP value and roster need
above raw projection, exactly as a serious draft assistant does.

Public API
----------
``get_recommendations(current_pick, my_roster, drafted_players, board, ...)``
    The board. Every recommendation is drawn from ``board`` minus
    ``drafted_players`` (ids), minus live-OUT players, minus positions your
    roster has already maxed.
``mark_player_drafted(drafted_players, player_id)``
    Return a new set with ``player_id`` added -- the "Mark Player Drafted"
    primitive.
``available_players(board, drafted_players)``
    ``board`` with the drafted ids removed, nothing else.
``compute_breakout_prob(player_row)``
    Re-exported from :mod:`fantasy.breakout_model` for a stable import surface.

Every returned row carries the full decision set the UI needs: projected
points and points per game, positional rank ("WR9"), ADP and the signed value
/ reach vs your pick, VORP, expected games and an availability read, breakout
probability and tier, a 0-100 risk score, team-role context, a positional
scarcity label, a plain-English roster-fit sentence, the composite
``rank_score`` and its per-factor ``score_breakdown``, and a one-line
``rationale``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from fantasy import player_status as _status
from fantasy.breakout_model import compute_breakout_prob
from fantasy.models import LeagueSettings, roster_cap_reached
from fantasy.utils import clamp, safe_float

__all__ = [
    "COMPONENT_WEIGHTS",
    "available_players",
    "compute_breakout_prob",
    "get_recommendations",
    "mark_player_drafted",
]

#: How much each factor moves the composite ``rank_score``. ADP value and
#: roster need sit above raw projection on purpose -- a draft assistant that
#: ranks by projection alone is just a cheat sheet. VORP is the tiebreaker;
#: breakout is a nudge, never a driver.
COMPONENT_WEIGHTS: dict[str, float] = {
    "adp_value": 1.00,
    "roster_need": 0.85,
    "scarcity": 0.60,
    "value": 0.35,
    "breakout": 0.12,
}

#: Per-position scarcity bias -- same values :mod:`fantasy.assistant` uses, so
#: the two recommenders agree about which pools run thin. TE and RB dry up
#: fast; WR and (1-QB) QB pools stay deep.
POSITION_SCARCITY_BIAS: dict[str, float] = {
    "TE": 1.20,
    "RB": 1.10,
    "WR": 0.90,
    "QB": 0.85,
    "K": 0.80,
    "DST": 0.80,
}

#: One starter, no FLEX outlet -- a second one cannot be started, so it is
#: damped until the draft is nearly over (see ``late_round_ok_positions``).
_STREAMER_POSITIONS: frozenset[str] = frozenset({"K", "DST"})

_ENRICH_FIELDS = (
    "age",
    "years_experience",
    "experience",
    "target_share",
    "air_yards_share",
    "prior_season_points",
    "prior_games_played",
    "prior_injury_status",
    "points_per_game",
    "expected_games",
)


# ---------------------------------------------------------------------------
# Drafted-player bookkeeping
# ---------------------------------------------------------------------------
def _id_of(player: Mapping[str, Any]) -> str:
    return str((player or {}).get("player_id") or (player or {}).get("id") or "")


def _id_set(drafted_players: Iterable[Any] | None) -> set[str]:
    """Normalise whatever the caller holds ('gone' ids, or pick dicts) to a str id set."""
    ids: set[str] = set()
    for entry in drafted_players or []:
        if isinstance(entry, Mapping):
            value = entry.get("player_id") or entry.get("id")
        else:
            value = entry
        if value is not None and str(value):
            ids.add(str(value))
    return ids


def mark_player_drafted(drafted_players: Iterable[Any] | None, player_id: Any) -> set[str]:
    """Return a **new** id set with ``player_id`` added.

    The "Mark Player Drafted" primitive: a UI holds a set in session state,
    calls this on every "someone took X" event, and re-runs
    :func:`get_recommendations`. Idempotent; ignores empty ids.
    """
    ids = _id_set(drafted_players)
    if player_id is not None and str(player_id):
        ids.add(str(player_id))
    return ids


def available_players(board: list[dict[str, Any]] | None, drafted_players: Iterable[Any] | None) -> list[dict[str, Any]]:
    """``board`` with every drafted id removed -- and nothing else changed."""
    gone = _id_set(drafted_players)
    return [player for player in (board or []) if _id_of(player) not in gone]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _coerce_settings(league_settings: dict[str, Any] | LeagueSettings | None) -> LeagueSettings:
    if isinstance(league_settings, LeagueSettings):
        return league_settings
    return LeagueSettings(**(league_settings or {}))


def _num(row: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] is not None:
            return safe_float(row[key], default)
    return default


def _position(player: Mapping[str, Any]) -> str:
    position = str(player.get("position") or player.get("player_position") or "").strip().upper()
    return "DST" if position in {"DEF", "D/ST"} else position


def _round_of(current_pick: int | None, n_teams: int) -> int:
    if not current_pick or n_teams < 1:
        return 0
    return max(1, math.ceil(int(current_pick) / n_teams))


# ---------------------------------------------------------------------------
# Composite score components
# ---------------------------------------------------------------------------
def _adp_value_component(
    adp: float | None,
    current_pick: int | None,
    picks_until_next: int | None,
    vorp: float,
) -> tuple[float, float | None]:
    """Signed ADP fit. Returns ``(component, adp_value)`` where ``adp_value`` is
    ``current_pick - adp`` (positive = the player fell past ADP = value).

    A reach is graded against your *next* turn: a player whose ADP still lands
    within ``picks_until_next`` will be gone if you wait, so it is only a gentle
    "you passed better-ranked players" nudge; an ADP *past* that horizon is a
    real overpay and takes the full penalty on the overage. A player with no
    market ADP is an unknown -- never *better* than an on-slot ranked pick.
    """
    if adp is None:
        return (-0.25 if vorp > 0 else -0.55), None
    if not current_pick:
        # No pick number to compare against -- fall back to best-ADP-first.
        return -math.tanh((float(adp) - 40.0) / 60.0), None
    delta = float(current_pick) - float(adp)
    if delta >= 0:
        component = math.tanh(delta / 12.0)  # steal: 0 .. ~+1 (a full round late ≈ +0.75)
    else:
        picks_early = -delta
        horizon = max(1.0, float(picks_until_next or 0))
        if picks_early <= horizon:
            # within reach of your next turn: at most a third of a full reach
            component = -0.35 * math.tanh(picks_early / horizon)
        else:
            # past the horizon he would have survived -- full penalty on the overage
            component = -0.35 - math.tanh((picks_early - horizon) / 8.0) * 0.95
    return component, round(delta, 1)


def _scarcity_component(
    position: str,
    scarcity_raw: dict[str, float],
    replacement_level: float,
    dropoff_by_pos: Mapping[str, Any],
    run_pressure: Mapping[str, Any],
) -> tuple[float, float]:
    """Positional scarcity, biased per position, with a tier-cliff bonus and a
    live-run bump. Returns ``(component, raw_scarcity_points)``."""
    raw = safe_float(scarcity_raw.get(position, 0.0))
    repl = max(1.0, replacement_level)
    base = math.tanh(raw / max(15.0, 0.25 * repl))
    dropoff = safe_float(dropoff_by_pos.get(position, 0.0))
    cliff = 0.5 * math.tanh(dropoff / max(20.0, 0.20 * repl))
    component = (base + cliff) * POSITION_SCARCITY_BIAS.get(position, 1.0)
    # QB and TE only genuinely bite near an actual tier cliff -- otherwise a
    # streamable position reads as scarce all draft long.
    if position in {"QB", "TE"} and dropoff < 0.12 * repl:
        component *= 0.40
    component += 0.30 * safe_float(run_pressure.get(position, 0.0))
    return clamp(component, 0.0, 1.6), round(raw, 1)


def _roster_need_component(
    position: str,
    roster_counts: Mapping[str, int],
    needs: Mapping[str, Any],
    starting_slots: Mapping[str, float],
    flex_eligible: set[str],
) -> float:
    """Roster need as an **additive** term in ``[-1, 1]``.

    This is the fix for the "collapses after round 5" fault: the old engine
    multiplied VORP by a need factor, and mid-draft VORP is negative, so more
    need meant a worse score. As a summed term, filling a starting slot is
    always ``+`` and luxury depth is always ``-`` -- on both sides of zero.
    """
    have = int(roster_counts.get(position, 0))
    unfilled = int((needs.get(position) or {}).get("unfilled", 0) or 0)
    flex_unfilled = int((needs.get("FLEX") or {}).get("unfilled", 0) or 0)
    starting = safe_float(starting_slots.get(position, 0.0))

    if unfilled > 0:
        need = 1.0
    elif flex_unfilled > 0 and position in flex_eligible:
        need = 0.55
    elif have < starting + 1:
        need = 0.30
    else:
        need = -0.20

    # Graduated skill-position saturation -- the old starter_saturation_penalty
    # was hard-wired to 1.0 for RB/WR/TE, so a 3rd/4th RB got no de-emphasis
    # until the hard cap. Ramp it down instead.
    if position in {"RB", "WR"} and have >= 3:
        need -= 0.18 * (have - 2)
    elif position == "TE" and have >= 2:
        need -= 0.35
    elif position == "QB" and have >= 1 and starting <= 1.0:
        need -= 0.45
    return clamp(need, -1.0, 1.0)


def _value_component(vorp: float, replacement_level: float) -> float:
    scale = max(20.0, 0.15 * max(1.0, replacement_level))
    return math.tanh(vorp / scale)


def _risk_score(
    position: str,
    age: float,
    experience: float,
    live_flag: str,
    prior_injury: str,
    prior_games: float,
    volatility: float,
) -> tuple[int, str]:
    """A 0-100 composite (higher = riskier) from age, injury history, and boom/bust."""
    if age >= 20.0:
        curve = {"RB": (24.0, 30.0), "WR": (25.0, 32.0), "TE": (26.0, 33.0), "QB": (30.0, 38.0)}.get(position, (25.0, 32.0))
        low, high = curve
        age_risk = clamp((age - low) / max(1.0, high - low), 0.0, 1.0) * 40.0
    elif experience > 0.0:
        age_risk = clamp((experience - 4.0) / 6.0, 0.0, 1.0) * 30.0
    else:
        age_risk = 12.0

    injury_risk = {"DOUBTFUL": 25.0, "QUESTIONABLE": 12.0}.get(live_flag, 0.0)
    if prior_injury in {"IR", "PUP", "NFI"}:
        injury_risk += 18.0
    if 0.0 < prior_games < 12.0:
        injury_risk += 10.0
    injury_risk = min(injury_risk, 35.0)

    vol_risk = clamp(volatility, 0.0, 1.2) / 1.2 * 25.0

    total = int(round(clamp(age_risk + injury_risk + vol_risk, 0.0, 100.0)))
    band = "low" if total < 25 else "moderate" if total < 45 else "elevated" if total < 65 else "high"
    return total, band


def _availability_verdict(expected_games: float) -> str:
    if expected_games >= 16.0:
        return "full slate"
    if expected_games >= 14.0:
        return "minor risk"
    if expected_games >= 12.0:
        return "moderate risk"
    return "high risk"


def _scarcity_label(position: str, component: float) -> str:
    if component >= 0.85:
        level = "high"
    elif component >= 0.45:
        level = "moderate"
    else:
        level = "low"
    return f"{position} scarcity: {level}"


def _roster_fit_sentence(
    position: str,
    have: int,
    need_component: float,
    unfilled: int,
    flex_open: bool,
) -> str:
    plural = "s" if have != 1 else ""
    if unfilled > 0:
        return f"You have no {position} starter yet — this fills a required slot."
    if flex_open:
        return f"Your FLEX is open; a {have + 1}{_ordinal_suffix(have + 1)} {position} slots straight in."
    if have == 0:
        return f"Adds your first {position}."
    if need_component < 0:
        return f"You already roster {have} {position}{plural}; this is luxury depth."
    return f"Solid depth behind your {have} {position}{plural}."


def _ordinal_suffix(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _rationale(entry: dict[str, Any]) -> str:
    parts: list[str] = []
    adp = entry.get("adp")
    adp_value = entry.get("adp_value")
    if adp is not None and adp_value is not None:
        verdict = entry.get("adp_verdict", "on time")
        parts.append(f"ADP {adp:.0f} vs your #{entry['current_pick']} — {adp_value:+.0f} ({verdict})")
    elif adp is not None:
        parts.append(f"ADP {adp:.0f}")
    else:
        parts.append("no market ADP — value read only")
    parts.append(f"{entry['vorp']:+.0f} pts over {entry['position']} replacement ({entry['pos_rank_label']})")
    if entry["scarcity_component"] >= 0.6:
        parts.append(entry["scarcity_label"])
    if entry["roster_need_component"] >= 0.9:
        parts.append("fills a starting need")
    elif entry["roster_need_component"] < 0:
        parts.append("roster spot already covered")
    if entry["breakout_probability"] >= 0.30:
        parts.append(f"{entry['breakout_probability']:.0%} breakout ({entry['breakout_tier']})")
    if entry["risk_band"] in {"elevated", "high"}:
        parts.append(f"{entry['risk_band']} risk ({entry['risk_score']}/100)")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------
def get_recommendations(
    current_pick: int | None,
    my_roster: list[dict[str, Any]] | None,
    drafted_players: Iterable[Any] | None,
    board: list[dict[str, Any]] | None,
    league_settings: dict[str, Any] | LeagueSettings | None = None,
    *,
    picks_until_next: int | None = None,
    num_rounds: int | None = None,
    n_teams: int | None = None,
    pool: list[dict[str, Any]] | None = None,
    picks: list[dict[str, Any]] | None = None,
    limit: int = 10,
    risk_tolerance: str = "balanced",
    weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank the best real players to take at ``current_pick``.

    Parameters
    ----------
    current_pick
        Your overall pick number (1-indexed). Drives the signed ADP-value term
        and the round-band logic. ``None`` falls back to best-ADP-first.
    my_roster
        The players already on *your* roster (dicts with at least ``position``).
        Drives roster need and the graduated saturation ramp.
    drafted_players
        Ids (or pick dicts) for every player gone in your *real* draft room --
        the "Mark Player Drafted" set. Recommendations are drawn from
        ``board`` minus these, so a sim board that has drifted from your real
        room is corrected without touching the sim state.
    board
        Players still on the (sim) board -- pass
        :func:`fantasy.live_draft.user_turn_context`'s ``board`` straight through.
    league_settings
        A :class:`fantasy.models.LeagueSettings` or a plain dict.
    picks_until_next
        Picks between this one and your next -- sharpens scarcity and the
        signed ADP-value read.
    num_rounds, n_teams
        League geometry, for the internal :func:`fantasy.draft_state.build_draft_state`
        snapshot (rounds-remaining, late-round streamer flags).
    pool
        The full forward-projection pool (``load_forward_projections`` output).
        Optional but recommended: board rows are stripped of ``age`` /
        ``target_share`` / injury history by the draft-board normaliser, and
        this is where the breakout model and the risk score recover them.
        Joined by ``player_id``.
    picks
        The draft's pick log (``state["picks"]``) for live positional-run
        detection.
    limit
        How many rows to return.
    risk_tolerance
        Reserved for parity with the other recommenders; currently unused here
        (risk is surfaced as an explicit ``risk_score`` rather than folded into
        the ranking).
    weights
        Overrides :data:`COMPONENT_WEIGHTS` key by key.

    Returns
    -------
    list[dict]
        Best-first, each a plain dict with the ranking inputs, every Section-3
        display field, ``rank_score``, a ``score_breakdown`` list, and a
        ``rationale``. ``[]`` when nothing is left to recommend -- a normal
        empty state, not an error.
    """
    from fantasy.assistant import (
        _projection_of,
        _same_team_cap_reached,
        _scarcity_by_position,
        _starting_slots_per_position,
        replacement_levels,
    )
    from fantasy.data_loader import drop_synthetic, validate_players
    from fantasy.draft_state import build_draft_state

    settings = _coerce_settings(league_settings)
    active_weights = {**COMPONENT_WEIGHTS, **{str(k): float(v) for k, v in (weights or {}).items()}}
    teams = int(n_teams or settings.n_teams or 12)

    # 1. Drop the players your real room has taken, then validate/de-synthesize.
    live_board = available_players(board, drafted_players)
    pool_rows, _rejected = validate_players(drop_synthetic(list(live_board)), require_projection=False)
    if not pool_rows:
        return []

    # 2. Live availability overlay -- identical policy to get_best_pick_for_round:
    #    OUT drops off the board, HOLDOUT/SUSPENDED get a zeroed projection so
    #    VORP sinks them, DOUBTFUL/QUESTIONABLE flow through as a risk signal.
    if _status.has_status_data():
        pool_rows = [row for row in pool_rows if not _status.is_out(row)]
        if not pool_rows:
            return []
        pool_rows = _status.overlay_pool_status(pool_rows)

    # 3. Enrichment map: recover age / usage / injury history from the forward
    #    pool for any board row that lost them.
    enrich_by_id: dict[str, dict[str, Any]] = {}
    for source in pool or []:
        key = _id_of(source)
        if key:
            enrich_by_id[key] = {field: source.get(field) for field in _ENRICH_FIELDS if source.get(field) is not None}

    # 4. One draft-state snapshot for needs / drop-off / run pressure / status.
    state = build_draft_state(
        pool_rows,
        my_roster,
        settings,
        picks=picks,
        current_pick_overall=current_pick,
        picks_until_next=picks_until_next,
        num_rounds=num_rounds,
        n_teams=teams,
    )
    needs = state.get("needs") or {}
    dropoff_by_pos = state.get("dropoff_by_pos") or {}
    run_pressure = state.get("run_pressure") or {}
    status_by_id = state.get("status_by_id") or {}
    roster_counts = state.get("roster_counts") or {}
    late_ok_positions = set(state.get("late_round_ok_positions") or ())
    rounds_remaining = state.get("rounds_remaining")

    levels = replacement_levels(pool_rows, settings)
    scarcity_raw = _scarcity_by_position(pool_rows, settings, picks_until_next)
    starting_slots = _starting_slots_per_position(settings)
    flex_eligible = {str(p).strip().upper() for p in settings.flex_eligible}
    current_round = _round_of(current_pick, teams)

    entries: list[dict[str, Any]] = []
    position_counter: dict[str, int] = {}
    best_proj_at_pos: dict[str, float] = {}
    for player in sorted(pool_rows, key=lambda p: _projection_of(p, settings), reverse=True):
        position = _position(player)
        position_counter[position] = position_counter.get(position, 0) + 1
        pos_rank = position_counter[position]
        # The board is walked best-first, so the first row at a position is its
        # top remaining player -- the yardstick the tier factor is measured against.
        best_proj_at_pos.setdefault(position, _projection_of(player, settings))

        if roster_cap_reached(roster_counts, position):
            continue
        if _same_team_cap_reached(player, my_roster, settings):
            continue

        player_id = _id_of(player)
        enriched = {**player, **enrich_by_id.get(player_id, {})}

        projection = _projection_of(player, settings)
        replacement_level = safe_float(levels.get(position, 0.0))
        vorp = projection - replacement_level
        raw_adp = player.get("adp")
        adp = round(safe_float(raw_adp), 1) if raw_adp is not None else None

        live_flag = str(status_by_id.get(player_id) or _status.live_status(player) or "").strip().upper()
        if live_flag == _status.HEALTHY:
            live_flag = ""

        age = _num(enriched, "age")
        experience = _num(enriched, "years_experience", "experience")
        prior_injury = str(enriched.get("prior_injury_status") or "").strip().upper()
        prior_games = _num(enriched, "prior_games_played", "games_played")
        expected_games = _num(enriched, "expected_games", default=17.0)
        points_per_game = _num(enriched, "points_per_game") or (round(projection / expected_games, 2) if expected_games else 0.0)
        floor = _num(player, "floor")
        ceiling = _num(player, "ceiling")
        volatility = (
            clamp((ceiling - floor) / max(projection, 1.0), 0.0, 1.2)
            if ceiling > 0 and floor >= 0 and projection > 0
            else clamp(_num(player, "volatility"), 0.0, 1.2)
        )

        # --- components ---------------------------------------------------
        adp_component, adp_value = _adp_value_component(adp, current_pick, picks_until_next, vorp)
        # A streamer's ADP (K/DST sit at 150-190) is noise, and once it is
        # genuinely streamer time -- ``late_round_ok_positions`` has cleared the
        # position -- the roster-need term should drive. Without this, every
        # late-round kicker reads as a 15-25 pick "reach" past its ADP (the full
        # ~-1.2 penalty), so a required, still-empty K/DST slot ends up buried
        # beneath a luxury 6th WR who happens to sit right on the slot -- exactly
        # the "weaker player over the one I need" fault this module exists to fix.
        if position in _STREAMER_POSITIONS and position in late_ok_positions:
            adp_component = max(adp_component, -0.15)
        scarcity_component, scarcity_points = _scarcity_component(position, scarcity_raw, replacement_level, dropoff_by_pos, run_pressure)
        need_component = _roster_need_component(position, roster_counts, needs, starting_slots, flex_eligible)
        value_component = _value_component(vorp, replacement_level)

        # Positional scarcity describes a tier *cliff*, not a player: the loss
        # from waiting only applies if you would otherwise take a top-of-tier
        # player there. A TE3/TE4 -- who will still be on the board next round
        # -- should not inherit the full "elite TE is thinning" bonus. Scale it
        # by how close this player is to the best one left at his position.
        repl_scale = max(1.0, replacement_level)
        tier_span = max(1.0, best_proj_at_pos.get(position, projection) - replacement_level)
        tier_factor = clamp((projection - replacement_level) / tier_span, 0.0, 1.0)
        # ...and still zero it out for a body well below replacement level.
        value_gate = clamp(0.5 + 0.5 * math.tanh((vorp + 0.15 * repl_scale) / (0.35 * repl_scale)), 0.1, 1.0)
        scarcity_component *= tier_factor * value_gate
        if need_component > 0:
            need_component *= clamp(0.4 + 0.6 * value_gate, 0.4, 1.0)

        breakout = compute_breakout_prob(enriched, position=position, live_status=live_flag or None)
        breakout_prob = breakout["breakout_probability"]
        breakout_component = clamp(2.4 * (breakout_prob - 0.12), -0.15, 1.0)

        # Round-band penalty (additive, so it stays sign-safe): a K or DST is
        # near-dead weight until the draft is almost over -- the roster-need
        # term alone would otherwise float the top kicker into round 1. An
        # early second QB in a one-QB league gets a milder version.
        have_now = int(roster_counts.get(position, 0))
        round_band_penalty = 0.0
        if position in _STREAMER_POSITIONS and position not in late_ok_positions:
            round_band_penalty = -1.6 if (rounds_remaining is None or rounds_remaining > 3) else -0.4
        elif position == "QB" and have_now >= 1 and safe_float(starting_slots.get("QB", 1.0)) <= 1.0 and (current_round or 0) and current_round <= 8:
            round_band_penalty = -0.55

        rank_score = (
            active_weights["adp_value"] * adp_component
            + active_weights["roster_need"] * need_component
            + active_weights["scarcity"] * scarcity_component
            + active_weights["value"] * value_component
            + active_weights["breakout"] * breakout_component
            + round_band_penalty
        )

        risk_score, risk_band = _risk_score(position, age, experience, live_flag, prior_injury, prior_games, volatility)

        have = int(roster_counts.get(position, 0))
        need_info = needs.get(position) or {}
        unfilled = int(need_info.get("unfilled", 0) or 0)
        flex_open = int((needs.get("FLEX") or {}).get("unfilled", 0) or 0) > 0 and position in flex_eligible

        adp_verdict = "on time" if adp_value is None else "value" if adp_value >= 8 else "reach" if adp_value <= -8 else "on time"
        target_share = _num(enriched, "target_share", "air_yards_share")

        entry: dict[str, Any] = {
            "player_id": player_id,
            "name": player.get("name"),
            "position": position,
            "team": player.get("team", ""),
            # --- projection ---
            "projection": round(projection, 1),
            "points_per_game": round(points_per_game, 1),
            "position_rank": pos_rank,
            "pos_rank_label": f"{position}{pos_rank}",
            # --- ADP / round value ---
            "adp": adp,
            "current_pick": current_pick,
            "adp_value": adp_value,
            "adp_verdict": adp_verdict,
            # --- VORP ---
            "vorp": round(vorp, 1),
            "replacement_points": round(replacement_level, 1),
            # --- availability ---
            "expected_games": round(expected_games, 1),
            "availability_verdict": _availability_verdict(expected_games),
            # --- breakout ---
            "breakout_probability": breakout_prob,
            "breakout_tier": breakout["breakout_tier"],
            "breakout_drivers": breakout["drivers"],
            # --- risk ---
            "risk_score": risk_score,
            "risk_band": risk_band,
            # --- team situation ---
            "team_context": {
                "target_share": round(target_share, 3) if target_share else None,
                "age": round(age, 1) if age else None,
                "expected_games": round(expected_games, 1),
            },
            # --- scarcity ---
            "scarcity_points": scarcity_points,
            "scarcity_component": round(scarcity_component, 3),
            "scarcity_label": _scarcity_label(position, scarcity_component),
            # biased points figure, on the same scale the shared scarcity_pill bands on
            "scarcity": round(scarcity_points * POSITION_SCARCITY_BIAS.get(position, 1.0), 1),
            # --- roster fit ---
            "roster_need_component": round(need_component, 3),
            "roster_fit": _roster_fit_sentence(position, have, need_component, unfilled, flex_open),
            "need_label": ("Fills need" if need_component >= 0.9 else "Depth" if need_component >= 0.0 else "Bench"),
            # --- composite ---
            "rank_score": round(rank_score, 3),
            "score_breakdown": [
                {
                    "factor": "ADP value",
                    "weight": active_weights["adp_value"],
                    "raw": round(adp_component, 3),
                    "weighted": round(active_weights["adp_value"] * adp_component, 3),
                },
                {
                    "factor": "Roster need",
                    "weight": active_weights["roster_need"],
                    "raw": round(need_component, 3),
                    "weighted": round(active_weights["roster_need"] * need_component, 3),
                },
                {
                    "factor": "Scarcity",
                    "weight": active_weights["scarcity"],
                    "raw": round(scarcity_component, 3),
                    "weighted": round(active_weights["scarcity"] * scarcity_component, 3),
                },
                {
                    "factor": "Value (VORP)",
                    "weight": active_weights["value"],
                    "raw": round(value_component, 3),
                    "weighted": round(active_weights["value"] * value_component, 3),
                },
                {
                    "factor": "Breakout",
                    "weight": active_weights["breakout"],
                    "raw": round(breakout_component, 3),
                    "weighted": round(active_weights["breakout"] * breakout_component, 3),
                },
                {
                    "factor": "Round band",
                    "weight": 1.0,
                    "raw": round(round_band_penalty, 3),
                    "weighted": round(round_band_penalty, 3),
                },
            ],
            "round_band_penalty": round(round_band_penalty, 3),
            "status": live_flag or _status.HEALTHY,
            "round": current_round or None,
        }
        entry["rationale"] = _rationale(entry)
        entries.append(entry)

    entries.sort(key=lambda e: e["rank_score"], reverse=True)
    top = entries[: max(1, int(limit))]
    for index, entry in enumerate(top, start=1):
        entry["rank"] = index
    return top
