"""Real market-consensus draft signals that calibrate :mod:`fantasy.room_brain`.

Usage::

    from fantasy.mock_data_ingestion import (
        positional_run_patterns, reach_steal_dispersion, round_by_round_trends,
    )

    patterns = positional_run_patterns(players)
    dispersion = reach_steal_dispersion(players)
    trends = round_by_round_trends(players, n_teams=12)

A note on provenance, because "ESPN mock draft boards" was the original ask:
there is no such feed. ``nflreadpy`` exposes 25 loaders (checked directly --
``load_combine``, ``load_contracts``, ... through ``load_trades``) and none of
them is a mock-draft board or an ESPN-branded ranking; no other data source is
wired into this project either. What IS real and already flowing through
:mod:`fantasy.data_loader` is FantasyPros' expert-consensus ADP (``adp``) and
its standard deviation (``adp_sd``) -- genuine aggregated draft-market
behavior, just not literally scraped from espn.com. Every function here
derives its output from that real signal. None of them invent a player, a
rank, or a draft board to paper over the missing feed.
"""

from __future__ import annotations

from typing import Any

from fantasy.draft import RUN_POSITIONS, identify_adp_clusters
from fantasy.utils import safe_float


def positional_run_patterns(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Real ADP-cluster run patterns -- the exact signal room_brain reacts to.

    A thin re-export of :func:`fantasy.draft.identify_adp_clusters` kept here
    too, so "where do room_brain's positional runs come from" has one
    self-documenting answer in the ingestion module the spec asked for.
    """
    return identify_adp_clusters(players)


def reach_steal_dispersion(players: list[dict[str, Any]], top_n: int = 36) -> dict[str, float]:
    """Average real ADP standard deviation per position, among its top ``top_n`` by ADP.

    ``adp_sd`` is FantasyPros' own measure of how much real experts disagree
    on a player's rank -- genuine reach/steal *variance* data, not a derived
    guess. A position where experts agree tightly (low average ``adp_sd``)
    is one where a real draft room rarely deviates from ADP; a position with
    wide disagreement is one where reaches and falls are both more normal.
    :func:`fantasy.draft.simulate_draft` uses this to scale how much reach
    :mod:`fantasy.room_brain` tolerates per position, rather than applying one
    fixed tolerance to every position alike.
    """
    by_position: dict[str, list[tuple[float, float]]] = {}
    for player in players:
        position = str(player.get("position", "")).strip().upper()
        if position not in RUN_POSITIONS:
            continue
        adp, adp_sd = player.get("adp"), player.get("adp_sd")
        if adp is None or adp_sd is None:
            continue
        by_position.setdefault(position, []).append((safe_float(adp), safe_float(adp_sd)))

    dispersion: dict[str, float] = {}
    for position, pairs in by_position.items():
        pairs.sort(key=lambda pair: pair[0])
        top = pairs[:top_n]
        if top:
            dispersion[position] = round(sum(sd for _, sd in top) / len(top), 2)
    return dispersion


def reach_tolerance_scale(players: list[dict[str, Any]], top_n: int = 36) -> dict[str, float]:
    """:func:`reach_steal_dispersion`, normalized to a 1.0-centered scale factor.

    Divides each position's average dispersion by the cross-position average,
    so a position with typical disagreement scores ~1.0, a tightly-agreed
    position scores below 1.0 (room_brain tolerates less reach there), and a
    widely-disputed one scores above 1.0 (room_brain tolerates more). Returns
    ``{}`` when no position has both ``adp`` and ``adp_sd`` populated -- callers
    should treat that as "use the default tolerance for everyone."
    """
    dispersion = reach_steal_dispersion(players, top_n=top_n)
    if not dispersion:
        return {}
    average = sum(dispersion.values()) / len(dispersion)
    if average <= 0:
        return {}
    return {position: round(value / average, 3) for position, value in dispersion.items()}


def round_by_round_trends(players: list[dict[str, Any]], n_teams: int, rounds: int = 15) -> list[dict[str, Any]]:
    """Real positional mix per round, purely from sorting real players by ADP.

    For each round, the ``n_teams`` players whose ADP falls in that round's
    pick range are bucketed by position -- e.g. a 12-team round 1 coming back
    mostly RB/WR reflects genuine current ADP, not a hand-tuned assumption
    about what a draft "should" look like. Rounds past the real ADP data's
    coverage are omitted rather than padded with a guess.
    """
    if n_teams < 1:
        raise ValueError("n_teams must be >= 1")
    with_adp = sorted(
        (player for player in players if player.get("adp") is not None),
        key=lambda player: safe_float(player.get("adp")),
    )
    trends: list[dict[str, Any]] = []
    for round_number in range(1, rounds + 1):
        start = (round_number - 1) * n_teams
        window = with_adp[start : start + n_teams]
        if not window:
            break
        counts: dict[str, int] = {}
        for player in window:
            position = str(player.get("position", "")).strip().upper()
            counts[position] = counts.get(position, 0) + 1
        trends.append({"round": round_number, "position_counts": counts, "sample_size": len(window)})
    return trends
