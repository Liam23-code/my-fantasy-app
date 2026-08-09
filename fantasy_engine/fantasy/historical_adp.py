"""Real historical ADP snapshots -- the fix for holdout ADP leakage.

Usage::

    from fantasy.historical_adp import fetch_historical_adp, apply_historical_adp

    players = apply_historical_adp(players_2024_basis, season=2025)
    # every player's `adp` is now the market's *August 2025* view

Why this is ``historical_adp`` and not ``historical_adp_sim``
------------------------------------------------------------
This was specified as a *simulation* module, on the reasonable assumption
that real historical ADP was unavailable -- that is what previous turns
concluded, and it is why :mod:`fantasy.holdout` carried a documented leakage
caveat. That assumption turned out to be wrong. Fantasy Football Calculator's
public API accepts a ``year`` parameter and returns genuine pre-season
snapshots, verified live from this project's own runtime:

===========  ==============  =========================  ==========
season       real drafts     measurement window         players
===========  ==============  =========================  ==========
2023         3,146           2023-08-30 .. 2023-09-01   202
2024         1,371           2024-08-31 .. 2024-09-01   205
2025         8,470           2025-08-25 .. 2025-09-01   249
===========  ==============  =========================  ==========

Each window closes *before* that season's Week 1. So a 2025 draft run on
2024-actuals projections plus this 2025 ADP knows nothing about how 2025
played out -- which is precisely the leak-free holdout the simulation was
meant to approximate. Real measured data beats a reconstruction of it, so
this module fetches rather than simulates, and is named for what it actually
does: naming real data ``_sim`` would misrepresent its provenance, which
matters a great deal in a codebase that carefully separates real from
synthetic elsewhere (see :mod:`fantasy.draft_ingestion`).

The other sources named for simulation are still unavailable and still
contribute nothing: Sleeper exposes no ADP endpoint, Underdog has no
confirmed public API, and the FantasyPros feed reachable here carries a
single current scrape date. None of them is needed now.
"""

from __future__ import annotations

from typing import Any

from fantasy.draft_ingestion.ffcalculator import fetch_ffcalculator_adp
from fantasy.utils import normalize_player_name, safe_float

#: Seasons verified live to return a real pre-season snapshot.
VERIFIED_SEASONS = (2023, 2024, 2025)

_MEMO: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = {}


def fetch_historical_adp(
    season: int,
    scoring: str = "ppr",
    teams: int = 12,
) -> dict[str, dict[str, Any]]:
    """Real pre-season ADP for ``season``, keyed by normalized player name.

    Keyed by name because Fantasy Football Calculator uses its own player-id
    numbering, unrelated to this project's nflverse ids -- the same join
    :mod:`fantasy.draft_fusion` already performs. Returns ``{}`` when the
    season has no data or the fetch fails; callers should treat that as "no
    historical ADP available" rather than an error.
    """
    memo_key = (season, scoring, teams)
    if memo_key in _MEMO:
        return _MEMO[memo_key]

    records = fetch_ffcalculator_adp(scoring=scoring, teams=teams, season=season)
    table: dict[str, dict[str, Any]] = {}
    for record in records:
        key = normalize_player_name(record.get("name"))
        adp = record.get("adp")
        if not key or adp is None:
            continue
        table[key] = {
            "adp": round(safe_float(adp), 2),
            "stdev": round(safe_float(record.get("stdev")), 2),
            "high": safe_float(record.get("high")),
            "low": safe_float(record.get("low")),
            "position": str(record.get("position", "")).strip().upper(),
        }
    _MEMO[memo_key] = table
    return table


def apply_historical_adp(
    players: list[dict[str, Any]],
    season: int,
    scoring: str = "ppr",
    teams: int = 12,
) -> list[dict[str, Any]]:
    """Overlay ``season``'s real pre-season ADP onto ``players``.

    Replaces whatever ``adp`` the pool arrived with (today: a present-day
    fused value) so a historical draft is timed by the market as it actually
    stood that August. Each player's prior value is preserved under
    ``adp_present_day`` so nothing is lost.

    Players with no historical ADP get ``adp = None`` -- they genuinely were
    not on the market's board that year (rookies not yet drafted, players not
    yet relevant), and inventing a number for them would be exactly the
    fabrication this codebase avoids. Downstream code already treats a
    missing ADP as "no market opinion" rather than penalizing it.
    """
    table = fetch_historical_adp(season, scoring=scoring, teams=teams)
    if not table:
        return [dict(player) for player in players]

    updated: list[dict[str, Any]] = []
    for player in players:
        player = dict(player)
        match = table.get(normalize_player_name(player.get("name")))
        player["adp_present_day"] = player.get("adp")
        if match is not None:
            player["adp"] = match["adp"]
            player["adp_sd"] = match["stdev"]
            player["adp_season"] = season
        else:
            player["adp"] = None
            player["adp_season"] = season
        updated.append(player)
    return updated


def historical_adp_coverage(players: list[dict[str, Any]], season: int) -> dict[str, Any]:
    """How much of ``players`` the real ``season`` snapshot actually covers."""
    table = fetch_historical_adp(season)
    matched = sum(1 for player in players if normalize_player_name(player.get("name")) in table)
    return {
        "season": season,
        "pool_size": len(players),
        "snapshot_size": len(table),
        "matched": matched,
        "coverage": round(matched / len(players), 3) if players else 0.0,
    }
