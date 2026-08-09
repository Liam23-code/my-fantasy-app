"""Draft assistant: who to take at your next pick, and why.

Usage::

    from fantasy.assistant import suggest_draft_picks

    suggestions = suggest_draft_picks(
        available_players,
        league_settings,
        my_roster=my_players,
        next_pick_overall=17,
        picks_until_next=8,
    )

This is the "Hybrid Draft Intelligence Model": five real signals, all computed
from the pool you pass in -- no fabricated players and no hard-coded rankings.

Value (projection / VORP)
    Projected points minus the *replacement level* at that position: the
    points of the best player who will still be undrafted once every team has
    filled its starting slots there. A 300-point QB in a 1-QB league is worth
    far less than a 300-point RB, and VORP is what makes that visible.

Timing (ADP)
    ADP decides *when* a player should go; projection/VORP decides how much
    they're *worth*. Relative to your current pick:

    * ``steal_bonus`` (> 0 when a player has fallen past their ADP) rewards
      players the market expected to be long gone -- a Justin Jefferson still
      on the board in round 5 is exactly this signal.
    * ``reach_penalty`` (> 0 when a player's ADP is well after your current
      pick) discourages jumping the market for someone who'd very likely still
      be there next round.

    Both are derived from the same ``current_pick - adp`` delta and are
    never simultaneously positive.

Scarcity (positional depth)
    The value you lose by waiting, computed from real VORP and pool depth,
    then biased by :data:`POSITION_SCARCITY_BIAS` -- TE and RB run thin fast in
    most leagues and are boosted; WR and (1-QB) QB pools stay deep and are
    discounted. The bias adjusts a real, data-driven number; it does not
    invent one.

Need (roster requirements + roster construction limits)
    Unfilled starting slots or unmet :data:`fantasy.models.ROSTER_POSITION_LIMITS`
    minimums are weighted up; positions already at their roster-construction
    *maximum* (e.g. a 3rd QB in a 1-2 QB league) are filtered out entirely --
    this assistant will never suggest an extra K, DST, QB, or TE past the cap.

Availability ("won't be there later")
    ``must_draft_now`` is set when a player's ADP falls before your next
    turn, with a safety margin subtracted (``AVAILABILITY_BUFFER_PICKS``) so
    the flag needs real confidence, not a razor-thin margin, before it fires
    -- a too-eager threshold is what causes reach-y "must draft now" calls
    late in a draft when a player would obviously have survived.

Risk, byes, and stacking (the user's edge over :mod:`fantasy.room_brain`)
    Three more signals moderate the score, none of which the simpler room
    brain considers: injury status and boom/bust volatility discount a pick
    (tunable via ``risk_tolerance``); a candidate sharing a bye week with 2+
    already-rostered players at the same position is mildly discounted;
    a QB/pass-catcher on the same NFL team as one you've already drafted gets
    a small stacking bonus. ``league_settings.max_players_per_nfl_team``, if
    set, is enforced as a hard filter alongside the position caps above.
"""

from __future__ import annotations

from typing import Any

from fantasy.data_loader import drop_synthetic, validate_players
from fantasy.models import LeagueSettings, roster_cap_reached, roster_min_required
from fantasy.scoring import calculate_fantasy_points
from fantasy.utils import safe_float

#: How much each signal moves the final score. VORP dominates on purpose:
#: an ablation against real 2025 data (simulate a user-brain-drafted team
#: against 11 room-brain bots across 25 seeds) showed steal/reach weighted
#: much above this made the automated pick *worse* on average, not better --
#: chasing a big ADP fall usually meant chasing the real ADP-vs-actuals
#: mismatch documented on fantasy.data_loader.load_real_projections (this
#: season's actuals are a backward-looking baseline; ADP reflects real,
#: forward-looking market information our own numbers can't see), not a
#: genuine value gap. steal/reach and must_draft_now are still fully
#: computed and surfaced for a human's judgment (that mismatch itself is
#: useful *information* even when it shouldn't dominate an automated pick);
#: their score weight is now a modest tie-breaker, not a primary driver.
SIGNAL_WEIGHTS: dict[str, float] = {
    "vorp": 1.0,
    "scarcity": 0.45,
    "steal": 0.15,
    "reach": 0.10,
    "need": 0.40,
}

#: Multiplier applied to a position's data-driven scarcity score before it is
#: weighted into the total. TE and RB pools thin out fast (few startable
#: players beyond the top tier); WR pools -- and QB pools in single-QB
#: leagues -- stay deep well into the draft, so the same raw scarcity number
#: should matter less there.
POSITION_SCARCITY_BIAS: dict[str, float] = {
    "TE": 1.20,
    "RB": 1.10,
    "WR": 0.90,
    "QB": 0.85,
    "K": 0.80,
    "DST": 0.80,
}

#: How much of a player's value estimate comes from the *market* (fused ADP)
#: rather than our own backward-looking VORP.
#:
#: This is the single most consequential constant in the file, and it is set
#: from measured out-of-sample evidence rather than taste. With it at 0.0
#: (pure VORP) the assistant looks dominant in-sample -- beating the bot
#: average by ~800 points when drafted and scored on the *same* season -- and
#: then finishes 10th of 12 on a holdout season it never saw (see
#: :mod:`fantasy.holdout`). The reason is visible in the picks: pure VORP buys
#: last season's production (Alvin Kamara 271 -> 101, James Conner 256 -> 33),
#: while the market had already priced those declines in and was right, and it
#: refuses players like Christian McCaffrey (54 points in an injured 2024, ADP
#: 7.8, 429 points the next season).
#:
#: Blending market-implied value in fixes the underlying error: ADP is
#: forward-looking and demonstrably the better predictor of next-season points
#: than last-season points are. Measured on 50 holdout drafts per setting
#: (draft on 2024, score on 2025, averaged over draft slots 1/4/6/9/12 to
#: remove slot luck):
#:
#: ===========  ==========  =========  ===========
#: blend        margin      win rate   mean rank
#: ===========  ==========  =========  ===========
#: 0.00         -406.2      0%         11.14 / 12
#: 0.80          -56.9      40%         7.52 / 12
#: 1.00          +78.4      64%         5.36 / 12
#: ===========  ==========  =========  ===========
#:
#: At 1.0 the *ordering* of players comes entirely from the market, but VORP
#: is not discarded -- it still supplies the value *scale* (what a player at
#: a given market rank is worth, i.e. the drop-off between ranks) and still
#: drives scarcity, replacement levels, and the value-trap gate below. VORP
#: moves from "player ranker", a role the holdout shows it is bad at, to
#: "value-curve estimator", which it is good at. The remaining edge over the
#: room comes from applying real positional scarcity and availability
#: modelling to the market's own rankings, while the bots distort theirs with
#: fixed positional round biases.
MARKET_BLEND_WEIGHT = 1.0

#: Score multiplier for a player flagged ``must_draft_now``. Kept modest for
#: the same reason ``steal``/``reach`` are (see :data:`SIGNAL_WEIGHTS`): the
#: real-data ablation showed a stronger boost chasing urgency over VORP hurt
#: the resulting roster more often than it helped.
MUST_DRAFT_BOOST = 1.1

#: Safety margin subtracted from the must-draft-now horizon: a player needs to
#: project as gone with this many picks of room to spare, not just barely
#: before your next turn, before the flag fires.
AVAILABILITY_BUFFER_PICKS = 2.0

#: Score multiplier by injury status -- moderates, never disqualifies (an OUT
#: designation from a stale feed shouldn't hard-block an otherwise-elite pick).
INJURY_RISK_MULTIPLIER: dict[str, float] = {
    "OUT": 0.50,
    "IR": 0.40,
    "DOUBTFUL": 0.70,
    "QUESTIONABLE": 0.90,
}

#: How much boom/bust volatility discounts a pick, by risk tolerance.
VOLATILITY_DISCOUNT_BY_RISK_TOLERANCE: dict[str, float] = {
    "safe": 0.35,
    "balanced": 0.15,
    "boom_bust": 0.0,
}

#: Score multiplier for a QB/pass-catcher pairing with an already-rostered
#: player from the same NFL team (correlated upside).
STACKING_BONUS = 1.08

#: Score multiplier when a candidate shares a bye week with 2+ already-
#: rostered players at the same position (bye-week roster risk).
BYE_WEEK_CLUSTER_PENALTY = 0.95


def _coerce_settings(league_settings: dict[str, Any] | LeagueSettings | None) -> LeagueSettings:
    if isinstance(league_settings, LeagueSettings):
        return league_settings
    return LeagueSettings(**(league_settings or {}))


def _projection_of(player: dict[str, Any], settings: LeagueSettings) -> float:
    """Prefer a real precomputed projection; score raw stats only as a fallback."""
    for key in ("projection", "expected_fantasy_points"):
        value = player.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    scored = calculate_fantasy_points(player, mode=settings.scoring_mode, custom_rules=settings.custom_rules)
    return float(scored["total_points"])


def _starting_slots_per_position(settings: LeagueSettings) -> dict[str, float]:
    """Starting slots each team must fill, with FLEX split across eligible positions."""
    slots = settings.roster_requirements.starting_slots()
    flex = float(slots.pop("FLEX", 0))
    per_position = {position: float(count) for position, count in slots.items()}
    eligible = [p for p in settings.flex_eligible if p in per_position] or list(settings.flex_eligible)
    if flex and eligible:
        share = flex / len(eligible)
        for position in eligible:
            per_position[position] = per_position.get(position, 0.0) + share
    return per_position


def replacement_levels(
    players: list[dict[str, Any]],
    league_settings: dict[str, Any] | LeagueSettings | None = None,
) -> dict[str, float]:
    """Projected points of the last startable player at each position.

    Positions with fewer players than league-wide starting slots have no true
    replacement level -- everyone available is a starter -- so the worst
    player at that position is used, which correctly makes VORP small rather
    than undefined.
    """
    settings = _coerce_settings(league_settings)
    by_position: dict[str, list[float]] = {}
    for player in players:
        position = str(player.get("position", "")).strip().upper()
        by_position.setdefault(position, []).append(_projection_of(player, settings))

    slots = _starting_slots_per_position(settings)
    levels: dict[str, float] = {}
    for position, projections in by_position.items():
        projections.sort(reverse=True)
        starters = int(round(slots.get(position, 0.0) * settings.n_teams))
        if starters <= 0:
            levels[position] = projections[-1]
        elif starters < len(projections):
            levels[position] = projections[starters]
        else:
            levels[position] = projections[-1]
    return levels


def _scarcity_by_position(
    pool: list[dict[str, Any]],
    settings: LeagueSettings,
    picks_until_next: int | None,
) -> dict[str, float]:
    """Points lost per position by waiting until your next pick.

    Without a ``picks_until_next`` horizon there is no "wait" to price, so the
    gap to the position's replacement level is used instead -- still a real
    scarcity signal, just a season-long one rather than a pick-relative one.
    This is the raw, data-driven number; :data:`POSITION_SCARCITY_BIAS` is
    applied on top of it by the caller, not baked in here.
    """
    by_position: dict[str, list[float]] = {}
    for player in pool:
        position = str(player.get("position", "")).strip().upper()
        by_position.setdefault(position, []).append(_projection_of(player, settings))

    levels = replacement_levels(pool, settings)
    scarcity: dict[str, float] = {}
    for position, projections in by_position.items():
        projections.sort(reverse=True)
        best = projections[0]
        if picks_until_next and picks_until_next > 0:
            # Roughly how many at this position go before you pick again.
            expected_gone = max(1, int(round(picks_until_next / max(len(by_position), 1))))
            index = min(expected_gone, len(projections) - 1)
            scarcity[position] = round(best - projections[index], 2)
        else:
            scarcity[position] = round(best - levels.get(position, 0.0), 2)
    return scarcity


def _roster_counts(my_roster: list[dict[str, Any]] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in my_roster or []:
        position = str((player or {}).get("position", "")).strip().upper()
        if position:
            counts[position] = counts.get(position, 0) + 1
    return counts


def capped_positions(
    my_roster: list[dict[str, Any]] | None,
    league_settings: dict[str, Any] | LeagueSettings | None = None,
) -> list[str]:
    """Positions where ``my_roster`` has already hit its roster-construction max.

    ``league_settings`` is accepted for a stable call signature but unused --
    the cap comes from :data:`fantasy.models.ROSTER_POSITION_LIMITS`, which is
    independent of league starting-lineup settings.
    """
    counts = _roster_counts(my_roster)
    return sorted(position for position in counts if roster_cap_reached(counts, position))


def _need_multiplier(roster_counts: dict[str, int], settings: LeagueSettings, position: str) -> float:
    if roster_cap_reached(roster_counts, position):
        return 0.0  # belt-and-suspenders: suggest_draft_picks already filters these out
    have = roster_counts.get(position, 0)
    threshold = max(roster_min_required(position), _starting_slots_per_position(settings).get(position, 0.0))
    if threshold <= 0:
        return 0.7  # position the league does not start and has no roster minimum
    if have < threshold:
        return 1.3
    if have < threshold + 1:
        return 1.0
    return 0.75


def _need_label(roster_counts: dict[str, int], settings: LeagueSettings, position: str) -> str:
    if roster_cap_reached(roster_counts, position):
        return "Roster full"
    have = roster_counts.get(position, 0)
    threshold = max(roster_min_required(position), _starting_slots_per_position(settings).get(position, 0.0))
    return "Fills need" if have < threshold else "Depth"


def _volatility_of(player: dict[str, Any], projection: float) -> float:
    """Ceiling-floor spread relative to projection -- 0 when either is missing."""
    floor, ceiling = player.get("floor"), player.get("ceiling")
    if floor is None or ceiling is None or projection <= 0:
        return 0.0
    return max(0.0, (safe_float(ceiling) - safe_float(floor)) / projection)


def _risk_multiplier(player: dict[str, Any], projection: float, risk_tolerance: str) -> float:
    """Injury status and boom/bust volatility moderate -- never disqualify -- a pick."""
    injury = str(player.get("injury_status") or "").strip().upper()
    multiplier = INJURY_RISK_MULTIPLIER.get(injury, 1.0)
    discount = VOLATILITY_DISCOUNT_BY_RISK_TOLERANCE.get(risk_tolerance, 0.15)
    if discount:
        multiplier *= max(0.5, 1.0 - discount * _volatility_of(player, projection))
    return multiplier


def _bye_week_multiplier(candidate: dict[str, Any], my_roster: list[dict[str, Any]] | None) -> float:
    """Discount a candidate whose bye week would stack 3+ deep at one position."""
    bye = candidate.get("bye_week")
    position = str(candidate.get("position", "")).strip().upper()
    if bye is None or not my_roster:
        return 1.0
    clashing = sum(
        1
        for player in my_roster
        if str((player or {}).get("position", "")).strip().upper() == position and (player or {}).get("bye_week") == bye
    )
    return BYE_WEEK_CLUSTER_PENALTY if clashing >= 2 else 1.0


def _stacking_multiplier(candidate: dict[str, Any], my_roster: list[dict[str, Any]] | None) -> float:
    """Bonus for a QB/pass-catcher pairing with an already-rostered same-team player."""
    position = str(candidate.get("position", "")).strip().upper()
    team = str(candidate.get("team", "")).strip().upper()
    if not team or position not in {"QB", "WR", "TE"} or not my_roster:
        return 1.0
    partner_positions = {"WR", "TE"} if position == "QB" else {"QB"}
    stacks = any(
        str((player or {}).get("team", "")).strip().upper() == team
        and str((player or {}).get("position", "")).strip().upper() in partner_positions
        for player in my_roster
    )
    return STACKING_BONUS if stacks else 1.0


def _same_team_cap_reached(
    candidate: dict[str, Any],
    my_roster: list[dict[str, Any]] | None,
    settings: LeagueSettings,
) -> bool:
    """True once ``my_roster`` already has ``max_players_per_nfl_team`` from this NFL team."""
    limit = settings.max_players_per_nfl_team
    if limit is None or not my_roster:
        return False
    team = str(candidate.get("team", "")).strip().upper()
    if not team:
        return False
    count = sum(1 for player in my_roster if str((player or {}).get("team", "")).strip().upper() == team)
    return count >= limit


def _rationale(entry: dict[str, Any], scarcity: float, picks_until_next: int | None) -> str:
    parts = [f"{entry['vorp']:+.1f} pts over {entry['position']} replacement"]
    if entry.get("position_rank"):
        parts.append(f"{entry['position']}{entry['position_rank']} available")
    if scarcity > 0:
        horizon = f"by your next pick (+{picks_until_next} picks)" if picks_until_next else "vs replacement"
        parts.append(f"~{scarcity:.0f} pts of {entry['position']} value gone {horizon}")
    if entry.get("must_draft_now"):
        parts.append("won't be there at your next turn -- must draft now")
    elif entry.get("steal_bonus"):
        parts.append(f"fell {entry['steal_bonus']:.0f} picks past ADP {entry['adp']:.0f} (steal)")
    elif entry.get("reach_penalty"):
        parts.append(f"a {entry['reach_penalty']:.0f}-pick reach vs ADP {entry['adp']:.0f}")
    elif entry.get("adp") is not None:
        parts.append(f"on par with ADP {entry['adp']:.0f}")
    label = entry.get("need_label")
    if label == "Fills need":
        parts.append("fills an empty roster need")
    elif label == "Depth":
        parts.append("bench depth, need already covered")
    if entry.get("games_played") and entry["games_played"] < 10:
        parts.append(f"small sample ({entry['games_played']} games)")
    return "; ".join(parts)


def suggest_draft_picks(
    available_players: list[dict[str, Any]],
    league_settings: dict[str, Any] | LeagueSettings | None = None,
    my_roster: list[dict[str, Any]] | None = None,
    next_pick_overall: int | None = None,
    picks_until_next: int | None = None,
    limit: int = 10,
    positions: list[str] | None = None,
    risk_tolerance: str = "balanced",
    availability_buffer: float | None = None,
) -> list[dict[str, Any]]:
    """Rank the best real players to take at your next pick.

    Synthetic and invalid records are dropped before anything is computed, so
    a fabricated player can never be recommended. Positions where
    ``my_roster`` has already hit its :data:`fantasy.models.ROSTER_POSITION_LIMITS`
    maximum, or ``league_settings.max_players_per_nfl_team`` for that NFL
    team, are dropped too -- this never suggests an extra K, DST, QB, or TE
    past a league's realistic roster construction, nor an extra player from a
    team you're already capped out on. ``risk_tolerance`` (``"safe"``/
    ``"balanced"``/``"boom_bust"``) controls how much injury status and
    boom/bust volatility discount a pick. Returns a list of plain dicts
    sorted best-first::

        [{"rank": 1, "player_id": ..., "name": ..., "position": ..., "team": ...,
          "projection": 389.8, "vorp": 121.4, "scarcity": 42.0, "scarcity_raw": 38.2,
          "adp": 3.4, "adp_value": 1.4, "steal_bonus": 1.4, "reach_penalty": 0.0,
          "must_draft_now": False, "need_multiplier": 1.3, "need_label": "Fills need",
          "risk_multiplier": 1.0, "score": 152.7, "position_rank": 1,
          "rationale": "..."}, ...]

    Returns ``[]`` when no real players are available -- callers should render
    an empty state rather than treating that as an error.
    """
    settings = _coerce_settings(league_settings)
    buffer = AVAILABILITY_BUFFER_PICKS if availability_buffer is None else float(availability_buffer)

    pool, _rejected = validate_players(drop_synthetic(list(available_players or [])), require_projection=False)
    if not pool:
        return []

    roster_counts = _roster_counts(my_roster)
    capped = {position for position in roster_counts if roster_cap_reached(roster_counts, position)}

    # Replacement level, scarcity, and positional rank all describe the *whole*
    # board, so they are computed before any position filter narrows the view --
    # otherwise asking only for QBs would make QBs look artificially scarce.
    levels = replacement_levels(pool, settings)
    scarcity = _scarcity_by_position(pool, settings, picks_until_next)

    wanted = {str(p).strip().upper() for p in positions} if positions else None

    # Market-implied value: where the market ranks a player, mapped onto what
    # a player at that rank is actually worth in this pool. If the market says
    # someone is the 8th-best player available, the 8th-best VORP in the pool
    # is the market's implied valuation of them -- a real, pool-derived
    # estimate rather than an assumed points curve. See MARKET_BLEND_WEIGHT.
    def _player_key(candidate: dict[str, Any]) -> str:
        return str(candidate.get("player_id") or candidate.get("id") or id(candidate))

    vorp_by_key = {
        _player_key(candidate): _projection_of(candidate, settings)
        - levels.get(str(candidate.get("position", "")).strip().upper(), 0.0)
        for candidate in pool
    }
    descending_vorps = sorted(vorp_by_key.values(), reverse=True)
    market_ranked = sorted(
        (candidate for candidate in pool if candidate.get("adp") is not None),
        key=lambda candidate: safe_float(candidate["adp"]),
    )
    market_rank_by_key = {_player_key(candidate): rank for rank, candidate in enumerate(market_ranked)}

    entries: list[dict[str, Any]] = []
    position_counter: dict[str, int] = {}
    for player in sorted(pool, key=lambda p: _projection_of(p, settings), reverse=True):
        position = str(player.get("position", "")).strip().upper()
        position_counter[position] = position_counter.get(position, 0) + 1
        if wanted is not None and position not in wanted:
            continue
        if position in capped:
            continue
        if _same_team_cap_reached(player, my_roster, settings):
            continue
        projection = _projection_of(player, settings)
        vorp = projection - levels.get(position, 0.0)

        adp = player.get("adp")
        adp_value: float | None = None
        steal_bonus = 0.0
        reach_penalty = 0.0
        must_draft_now = False
        if adp is not None and next_pick_overall:
            # current_pick - adp: positive means the player has fallen past
            # their ADP (a steal); negative means drafting them now would be
            # jumping the market ahead of ADP (a reach).
            timing_delta = float(next_pick_overall) - safe_float(adp)
            adp_value = round(timing_delta, 1)
            steal_bonus = round(max(0.0, timing_delta), 1)
            reach_penalty = round(max(0.0, -timing_delta), 1)
            if picks_until_next is not None:
                horizon = float(next_pick_overall) + float(picks_until_next) - buffer
                must_draft_now = safe_float(adp) < horizon

        need_multiplier = _need_multiplier(roster_counts, settings, position)
        # A 2nd QB (or K/DST) in a league that starts one is close to dead
        # weight -- it literally cannot be started. Pure best-market-value
        # ranking does not know that, and the leak-free holdout caught this
        # team taking a round-2 QB and then a round-5 QB. Local import keeps
        # fantasy.positional_pressure -> fantasy.models the only dependency
        # direction.
        from fantasy.positional_pressure import starter_saturation_penalty

        need_multiplier *= starter_saturation_penalty(position, roster_counts, settings)
        need_label = _need_label(roster_counts, settings, position)
        raw_scarcity = scarcity.get(position, 0.0)
        biased_scarcity = round(raw_scarcity * POSITION_SCARCITY_BIAS.get(position, 1.0), 2)
        risk_multiplier = _risk_multiplier(player, projection, risk_tolerance)
        risk_multiplier *= _bye_week_multiplier(player, my_roster)
        risk_multiplier *= _stacking_multiplier(player, my_roster)

        # "Take falling studs aggressively" -- but only real studs. ADP can
        # reflect information our VORP can't see (this season's actuals are a
        # backward-looking baseline; see fantasy.data_loader's projection_basis),
        # so a player can carry a strong ADP while our own numbers say they're
        # below replacement level. A steal_bonus that ignores VORP entirely
        # would chase that mismatch -- rewarding a big fall regardless of
        # whether the player is actually good -- so it's heavily discounted
        # whenever VORP itself isn't positive.
        effective_steal_bonus = steal_bonus if vorp > 0 else steal_bonus * 0.15

        # Blend our own VORP with the market's implied valuation of this
        # player. Players the market has no opinion on (no ADP) fall back to
        # pure VORP -- there is no market view to blend in.
        player_key = _player_key(player)
        market_rank = market_rank_by_key.get(player_key)
        if market_rank is not None and descending_vorps:
            market_vorp = descending_vorps[min(market_rank, len(descending_vorps) - 1)]
            effective_vorp = (1.0 - MARKET_BLEND_WEIGHT) * vorp + MARKET_BLEND_WEIGHT * market_vorp
        else:
            market_vorp = None
            effective_vorp = vorp

        score = (
            SIGNAL_WEIGHTS["vorp"] * effective_vorp
            + SIGNAL_WEIGHTS["scarcity"] * biased_scarcity
            + SIGNAL_WEIGHTS["steal"] * effective_steal_bonus
            - SIGNAL_WEIGHTS["reach"] * reach_penalty
        )
        score *= 1.0 + SIGNAL_WEIGHTS["need"] * (need_multiplier - 1.0)
        score *= risk_multiplier
        if must_draft_now:
            score *= MUST_DRAFT_BOOST

        entry = {
            "player_id": player.get("player_id") or player.get("id"),
            "name": player.get("name"),
            "position": position,
            "team": player.get("team", ""),
            "projection": round(projection, 2),
            "points_per_game": player.get("points_per_game"),
            "games_played": player.get("games_played"),
            "vorp": round(vorp, 2),
            "market_vorp": round(market_vorp, 2) if market_vorp is not None else None,
            "effective_vorp": round(effective_vorp, 2),
            "replacement_points": round(levels.get(position, 0.0), 2),
            "scarcity": biased_scarcity,
            "scarcity_raw": round(raw_scarcity, 2),
            "adp": round(safe_float(adp), 1) if adp is not None else None,
            "adp_value": adp_value,
            "steal_bonus": steal_bonus,
            "reach_penalty": reach_penalty,
            "must_draft_now": must_draft_now,
            "need_multiplier": need_multiplier,
            "need_label": need_label,
            "risk_multiplier": round(risk_multiplier, 3),
            "position_rank": position_counter[position],
            "score": round(score, 2),
        }
        entry["rationale"] = _rationale(entry, raw_scarcity, picks_until_next)
        entries.append(entry)

    entries.sort(key=lambda e: e["score"], reverse=True)
    top = entries[: max(1, limit)]
    for index, entry in enumerate(top, start=1):
        entry["rank"] = index
    return top


def next_pick_number(
    num_teams: int,
    draft_slot: int,
    round_number: int,
    snake: bool = True,
) -> int:
    """Overall pick number for ``draft_slot`` in ``round_number`` (1-indexed)."""
    if num_teams < 1:
        raise ValueError("num_teams must be >= 1")
    if not 1 <= draft_slot <= num_teams:
        raise ValueError(f"draft_slot must be between 1 and {num_teams}")
    if round_number < 1:
        raise ValueError("round_number must be >= 1")
    offset = draft_slot if (not snake or round_number % 2 == 1) else num_teams - draft_slot + 1
    return (round_number - 1) * num_teams + offset


def picks_between(num_teams: int, draft_slot: int, round_number: int, snake: bool = True) -> int:
    """How many picks happen between this pick and your next one."""
    current = next_pick_number(num_teams, draft_slot, round_number, snake)
    following = next_pick_number(num_teams, draft_slot, round_number + 1, snake)
    return following - current - 1
