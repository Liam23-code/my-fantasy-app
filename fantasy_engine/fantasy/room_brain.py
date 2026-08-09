"""room_brain: draft logic for every simulated team except the user's.

Usage::

    from fantasy.room_brain import room_brain_weight, is_round_one_chalk, round_one_pick

    if is_round_one_chalk(round_number):
        player = round_one_pick(candidates_sorted_by_adp, rng)
    else:
        weight = room_brain_weight(candidate, overall_pick, round_number)

This models a typical live/default draft room, not any one human's judgment:
ADP anchors *when* a player goes, VOR only breaks ties between similarly-timed
players, and a handful of well-known tendencies shape the rest -- RB runs
early, WR waves through the middle rounds, TE/QB patience until value forces
the issue, and risk aversion (avoid volatile players) in the first few rounds.
Round 1 is close to pure market order, same as most real drafts: with the
consensus top players still on the board, there's nothing yet for
scarcity/need/risk reasoning to bite on.

None of this is literally scraped from ESPN's draft room -- no such feed
exists (see :mod:`fantasy.mock_data_ingestion` for what real data actually
backs the "ESPN-style" label: FantasyPros consensus ADP and its dispersion).
It intentionally does NOT use :mod:`fantasy.user_brain`'s fuller signal set
(scarcity bias, risk/bye/stacking) -- that asymmetry is the point: the user's
simulated team drafts with a materially stronger model than the room.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fantasy.utils import clamp, safe_float

#: Picks of "reach" (drafting well before ADP) the room tolerates before
#: heavily suppressing a candidate, and how fast that suppression ramps.
REACH_TOLERANCE_PICKS = 4.0
REACH_DECAY_PICKS = 10.0

#: Early rounds where volatile players get down-weighted (risk aversion).
RISK_AVERSION_ROUNDS = 4
RISK_AVERSION_STRENGTH = 0.5

#: Rounds where injury-risk players get down-weighted, and by how much per
#: status. A stale "OUT" tag shouldn't hard-block an elite pick, so this
#: moderates rather than excludes.
INJURY_AVOIDANCE_ROUNDS = 5
INJURY_MULTIPLIER: dict[str, float] = {
    "OUT": 0.55,
    "IR": 0.45,
    "DOUBTFUL": 0.70,
    "QUESTIONABLE": 0.90,
}

#: Bounds on how much our own VORP may modify a market-driven weight.
#: VORP is a *tie-breaker* for the room, never the base: this project's
#: ``projection`` (and therefore ``vor``) is explicitly backward-looking --
#: last season's actuals, see ``fantasy.data_loader.load_real_projections`` --
#: while ADP is the market's forward-looking view. A player coming off an
#: injured or low-volume season can carry a deeply negative VOR *and* a strong
#: ADP (Garrett Wilson: ADP 30, VOR -71). Letting VOR scale the weight
#: unbounded made such players effectively undraftable (measured: a 46x weight
#: penalty, falling 135 picks past ADP); clamping it keeps the real discount
#: without the cliff.
VOR_MODIFIER_MIN = 0.60
VOR_MODIFIER_MAX = 1.50
#: Points of VOR that move the modifier by 0.5 from neutral.
VOR_MODIFIER_SCALE = 100.0

#: How much a player's own ``run_pressure`` field (see
#: :func:`fantasy.draft_fusion.fuse_draft_results`) boosts their weight, on
#: top of the live, pick-by-pick cluster detection the caller already applies
#: (:data:`fantasy.draft.RUN_BOOST`). These are complementary, not redundant:
#: ``run_pressure`` is a structural, precomputed "is this player's ADP inside
#: a real same-position cluster at all" signal; the caller's own detection is
#: "is that cluster active right now, at this exact pick."
RUN_PRESSURE_WEIGHT = 0.5

#: How strongly a player's real ``round_curve`` (see
#: :func:`fantasy.draft_fusion.fuse_draft_results`) pulls them toward the
#: rounds actual drafts took them in. Kept modest: the curve is derived from
#: an observed [high, low] range, which is a coarser signal than ADP itself.
ROUND_CURVE_STRENGTH = 0.35

#: A "steal" (ADP says gone, still here) only gets its full timing bonus when
#: VOR backs it up. Without this, a big ADP fall on a player who is actually
#: below replacement level (a real gap between real, forward-looking ADP and
#: this project's backward-looking-actuals VORP -- see
#: fantasy.data_loader.load_real_projections) reads as a great value when it
#: isn't one. Matches the identical "stud gate" already applied in
#: fantasy.assistant for the exact same reason.
VALUE_TRAP_DAMPENING = 0.15

#: Chance the room takes the 2nd-best-fused-ADP player over the best, in round
#: 1. Set to 0.0: round 1 is a fused-ADP chalk *lock*, not just a strong lean --
#: real snake drafts are close to perfectly chalky with the consensus top
#: players still on the board.
ROUND_ONE_UPSET_CHANCE = 0.0


def _position_round_multiplier(position: str, round_number: int) -> float:
    """Positional draft-round tendencies, as literal round thresholds (not buckets).

    RB premium rounds 1-3, WR waves rounds 3-7, TE patience until round 5+, QB
    patience until round 7+, K/DST essentially untouched before round 9.
    """
    if position == "RB":
        if round_number <= 3:
            return 1.35
        if round_number <= 7:
            return 1.00
        return 0.85
    if position == "WR":
        if round_number <= 2:
            return 1.00
        if round_number <= 7:
            return 1.25
        return 1.05
    if position == "TE":
        return 0.55 if round_number < 5 else 1.15
    if position == "QB":
        return 0.55 if round_number < 7 else 1.15
    if position in ("K", "DST"):
        return 0.05 if round_number <= 8 else 1.00
    return 1.0


def _timing_multiplier(adp: Any, overall_pick: int, tolerance_scale: float = 1.0, vor: float | None = None) -> float:
    """How well a candidate's ADP fits the current pick -- the dominant signal.

    A player already past their ADP (a "steal") gets a bonus that grows with
    how far they've fallen -- *unless* ``vor`` is at or below zero, in which
    case the bonus is heavily dampened (:data:`VALUE_TRAP_DAMPENING`): a big
    ADP fall on a player our own numbers say is below replacement level is a
    value trap, not a steal, and shouldn't be chased. A player well before
    their ADP (a "reach") is heavily suppressed once beyond
    ``REACH_TOLERANCE_PICKS * tolerance_scale`` -- see
    :func:`fantasy.mock_data_ingestion.reach_tolerance_scale` for where a
    position-specific scale (real ADP disagreement) comes from; the default
    of 1.0 applies the flat tolerance. Missing ADP is neutral.
    """
    if adp is None:
        return 1.0
    delta = overall_pick - safe_float(adp)  # > 0 = fell past ADP (steal); < 0 = reach
    if delta >= 0:
        steal_multiplier = 1.0 + delta / 10.0
        if vor is not None and vor <= 0:
            steal_multiplier = 1.0 + (steal_multiplier - 1.0) * VALUE_TRAP_DAMPENING
        return steal_multiplier
    reach = -delta
    tolerance = REACH_TOLERANCE_PICKS * tolerance_scale
    if reach <= tolerance:
        return 1.0
    return max(0.02, 1.0 - (reach - tolerance) / REACH_DECAY_PICKS)


def _round_curve_multiplier(round_curve: Any, round_number: int) -> float:
    """Boost a candidate in the rounds real drafts actually took them.

    ``round_curve`` (see :func:`fantasy.draft_fusion.fuse_draft_results`) is
    the fraction of a player's real observed pick range that landed in each
    round. A player whose range says "always round 3" should be much more
    likely to go in round 3 than round 8, which raw ADP proximity alone only
    partly captures. Missing or empty curves are neutral (1.0), so this can
    never penalize a player for having no such data.
    """
    if not isinstance(round_curve, dict) or not round_curve:
        return 1.0
    share = safe_float(round_curve.get(round_number))
    if share <= 0:
        # Real data says this round is outside the player's observed range.
        return 1.0 - ROUND_CURVE_STRENGTH
    return 1.0 + ROUND_CURVE_STRENGTH * share


def _vor_modifier(vor: float) -> float:
    """Bounded VORP nudge on a market-driven weight -- never a veto.

    Clamped to [:data:`VOR_MODIFIER_MIN`, :data:`VOR_MODIFIER_MAX`] so our own
    backward-looking VORP can meaningfully discount or promote a candidate
    without ever zeroing out what the market says.
    """
    return clamp(1.0 + (vor / VOR_MODIFIER_SCALE) * 0.5, VOR_MODIFIER_MIN, VOR_MODIFIER_MAX)


def room_candidate_pool(eligible: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """The players a real draft room is actually choosing between at this pick.

    Ordered by consensus market rank (fused ADP ascending), **not** by our own
    VORP. This is the other half of the falling-players fix: the caller's
    ``eligible`` list arrives sorted by VORP (from
    :func:`fantasy.draft.rank_players_for_draft`), so taking ``eligible[:size]``
    meant a high-ADP / negative-VOR player was never even *considered* until
    virtually everyone with better VORP was gone -- no amount of weighting
    inside the pool could fix a player who never entered it.

    Players with no ADP at all sort last and keep their incoming VORP order
    among themselves (Python's sort is stable), which is what should happen:
    once the market's board is exhausted, our own rankings are the only signal
    left.
    """
    ranked_by_market = sorted(
        eligible,
        key=lambda candidate: safe_float(candidate["adp"]) if candidate.get("adp") is not None else float("inf"),
    )
    return ranked_by_market[: max(1, size)]


def room_brain_weight(
    candidate: dict[str, Any],
    overall_pick: int,
    round_number: int,
    reach_tolerance_scale: float = 1.0,
) -> float:
    """Sampling weight the room assigns one candidate at one (non-round-1) pick.

    ``timing`` scales the VOR-based base weight *multiplicatively*, not
    additively -- deliberately, so ADP dominance holds regardless of how big
    VOR gets. An additive combination (``vor_weight + timing_weight``) lets an
    extreme VOR outlier swamp even a heavily-suppressed reach penalty (a
    300-point-VOR reach could still out-weigh a modest, well-timed pick);
    multiplying means a confirmed big reach (timing ~0.02) always shrinks
    *any* VOR by ~50x, and a confirmed big fall (timing several-fold) always
    grows it proportionally -- this is exactly the fix for a player like
    Saquon Barkley (real ADP ~25, elite VOR) falling to round 3-4: previously
    pure-VOR weighting had no ADP term to catch this at all.

    Positional round tendencies (:func:`_position_round_multiplier`) and early-
    round risk/injury aversion apply on top. Combine with roster-cap filtering
    and need/run bias exactly as the caller already does -- this replaces only
    the old ``max(vor, 0.01)`` base weight, not the surrounding cap/need/run
    machinery. ``reach_tolerance_scale`` lets a caller widen or narrow reach
    tolerance per position using real ADP disagreement data (see
    :mod:`fantasy.mock_data_ingestion`); the default applies the flat,
    position-agnostic tolerance. ``candidate["run_pressure"]``, when present
    (see :func:`fantasy.draft_fusion.fuse_draft_results`), adds a further
    boost -- see :data:`RUN_PRESSURE_WEIGHT`.
    """
    position = str(candidate.get("position", "")).strip().upper()
    vor = safe_float(candidate.get("vor"))

    # ADP proximity is the BASE, VOR only a bounded modifier -- see
    # VOR_MODIFIER_MIN/MAX. Multiplying an unbounded VOR by timing (the
    # previous design) had the right idea for reaches but made high-ADP /
    # negative-VOR players undraftable; a bounded modifier keeps the reach
    # protection (a big reach still drives timing toward 0.02, and at most
    # 1.5x cannot rescue it) while letting the market anchor the pick.
    timing = _timing_multiplier(candidate.get("adp"), overall_pick, reach_tolerance_scale, vor)
    weight = timing * _vor_modifier(vor)

    weight *= _position_round_multiplier(position, round_number)

    if round_number <= RISK_AVERSION_ROUNDS:
        volatility = safe_float(candidate.get("volatility"))
        weight *= max(0.4, 1.0 - RISK_AVERSION_STRENGTH * volatility)

    if round_number <= INJURY_AVOIDANCE_ROUNDS:
        injury = str(candidate.get("injury_status") or "").strip().upper()
        weight *= INJURY_MULTIPLIER.get(injury, 1.0)

    run_pressure = safe_float(candidate.get("run_pressure"))
    weight *= 1.0 + RUN_PRESSURE_WEIGHT * run_pressure

    weight *= _round_curve_multiplier(candidate.get("round_curve"), round_number)

    return max(weight, 0.01)


def is_round_one_chalk(round_number: int) -> bool:
    """Round 1 is drafted by the market, not by team-specific reasoning.

    With the consensus best players still available, every team -- room and
    user alike -- is playing best-player-available, and ADP already *is* that
    consensus. Fixes "first-round falls" caused by VOR/scarcity/need reasoning
    overriding the market this early, when there's nothing yet to reason about.
    """
    return round_number == 1


def round_one_pick(candidates_by_adp: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    """Pick for round 1 from ``candidates_by_adp`` (already sorted, best ADP first).

    Mostly the single best-ADP player; a small chance of the next-best keeps
    round 1 from being perfectly deterministic across seeds, matching how
    real snake drafts occasionally see a 1-pick swap even at the very top.
    """
    if len(candidates_by_adp) == 1:
        return candidates_by_adp[0]
    weights = np.array([1.0 - ROUND_ONE_UPSET_CHANCE, ROUND_ONE_UPSET_CHANCE])[: len(candidates_by_adp)]
    weights = weights / weights.sum()
    index = int(rng.choice(len(candidates_by_adp), p=weights))
    return candidates_by_adp[index]
