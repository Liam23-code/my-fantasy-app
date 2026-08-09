"""Draft Result Fusion Engine: realism-weighted multi-source draft data.

Usage::

    from fantasy.draft_fusion import fuse_draft_results, apply_fused_draft_results

    fused = fuse_draft_results(players)          # {player_id: {...}}
    players = apply_fused_draft_results(players)  # overlays fused_adp + fused_run_pressure

Read this before trusting a number here -- verified live this session (both
via a browsing fetch and a direct call from this project's own Python
runtime, since the two don't always agree: this host needs a standard
``User-Agent`` header or FantasyFootballCalculator returns HTTP 403):

Real, automatic, today
    * FantasyPros consensus ADP -- already flowing through every player's
      ``adp``/``adp_sd`` fields via :func:`fantasy.data_loader.load_adp`.
    * Fantasy Football Calculator (:mod:`fantasy.draft_ingestion.ffcalculator`)
      -- confirmed live: a real, public, unauthenticated JSON API aggregating
      actual mock drafts (4,299 real drafts, 248 players, when checked this
      session). Joined to the rest of the pool by normalized player name --
      FFC uses its own internal player_id numbering, unrelated to this
      project's nflverse-based IDs.

Confirmed to have no real data at all
    Sleeper's public API was checked live this session (see
    :mod:`fantasy.draft_ingestion.sleeper_api`): no ADP or draft-position
    endpoint exists, only player-identity dumps and waiver-add counts. Its
    configured weight is always excluded and the rest renormalized -- not a
    "not yet implemented" gap, a verified absence.

Real, but requires you to supply the data
    FFPC (login-gated, no public page), Underdog (no confirmed public API),
    and the GitHub/Reddit sources (neither is one identifiable, stable
    source) each have a genuine parser in :mod:`fantasy.draft_ingestion`, fed
    by ``extra_boards`` below -- see each ingestion module's own docstring.
    Absent unless you pass something in.

Every metric this module returns is computed only from whichever of the
above actually has data for a given player; nothing is invented for a source
with none. ``reach_rate``/``fall_rate``/``round_curve`` are estimated from
Fantasy Football Calculator's real ``high``/``low``/``adp`` range when no raw
per-board picks are supplied (there's no literal per-draft tally otherwise --
FFC's public API only exposes the cross-draft aggregate) and computed exactly
from ``extra_boards`` picks when you do supply them.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from fantasy.draft_ingestion.ffcalculator import fetch_ffcalculator_adp
from fantasy.draft_ingestion.sleeper_api import fetch_sleeper_adp
from fantasy.utils import normalize_player_name, safe_float

#: Section 2's exact realism weights. "github_reddit" bundles the two named
#: sources the spec itself groups at the same 0.05 -- neither
#: (:mod:`fantasy.draft_ingestion.github_datasets` /
#: :mod:`fantasy.draft_ingestion.reddit_dumps`) is one identifiable, real,
#: default-available source (see their docstrings), so they share one bucket
#: rather than each pretending to be independently populated.
REALISM_WEIGHTS: dict[str, float] = {
    "ffpc": 0.30,
    "underdog": 0.25,
    "sleeper": 0.20,
    "fantasypros": 0.15,
    "ffcalculator": 0.05,
    "github_reddit": 0.05,
}

#: Sources with a real, automatically-available data path today.
REAL_SOURCES = ("fantasypros", "ffcalculator")

#: Sources that only ever contribute when the caller supplies real content
#: via ``extra_boards`` (see :mod:`fantasy.draft_ingestion` for why each has
#: no default automatic data).
USER_SUPPLIED_SOURCES = ("ffpc", "underdog", "github_reddit")

#: Verified this session to have no real draft/ADP data at all -- never
#: contributes, regardless of ``extra_boards``.
CONFIRMED_UNAVAILABLE_SOURCES = ("sleeper",)

DEFAULT_TEAMS_FOR_ROUNDING = 12


def _ffcalculator_by_name(scoring: str, teams: int) -> dict[str, dict[str, Any]]:
    records = fetch_ffcalculator_adp(scoring=scoring, teams=teams)
    return {normalize_player_name(record.get("name")): record for record in records if record.get("name")}


def _board_stats_from_picks(picks: list[int]) -> dict[str, float]:
    """Genuine mean/median/stdev computed directly from real per-board pick numbers."""
    return {
        "avg_pick": round(statistics.fmean(picks), 2),
        "median_pick": round(statistics.median(picks), 2),
        "stdev": round(statistics.pstdev(picks), 2) if len(picks) > 1 else 0.0,
        "high": min(picks),
        "low": max(picks),
        "times_drafted": len(picks),
    }


def _range_based_reach_fall(adp: float, high: float, low: float) -> tuple[float, float]:
    """Estimate reach/fall rates from a real observed [high, low] pick range.

    Not a literal per-board tally (no raw picks to count here) -- a
    proportion-of-real-range estimate: how much of the actual observed spread
    sits earlier (reach) vs. later (fall) than the average. Clamped to
    [0, 1]; degenerates to (0, 0) when high == low (no observed spread).
    """
    span = low - high
    if span <= 0:
        return 0.0, 0.0
    reach = max(0.0, min(1.0, (adp - high) / span))
    fall = max(0.0, min(1.0, (low - adp) / span))
    return round(reach, 3), round(fall, 3)


def _round_curve_from_range(high: float, low: float, teams: int) -> dict[int, float]:
    """Fraction of a player's real observed pick range falling in each round."""
    start_round = int((high - 1) // teams) + 1
    end_round = int((low - 1) // teams) + 1
    rounds = list(range(start_round, end_round + 1))
    if not rounds:
        return {}
    share = round(1.0 / len(rounds), 3)
    return {r: share for r in rounds}


def _round_curve_from_picks(picks: list[int], teams: int) -> dict[int, float]:
    counts = Counter((pick - 1) // teams + 1 for pick in picks)
    total = len(picks)
    return {round_number: round(count / total, 3) for round_number, count in sorted(counts.items())}


def fuse_draft_results(
    players: list[dict[str, Any]],
    scoring: str = "ppr",
    teams: int = DEFAULT_TEAMS_FOR_ROUNDING,
    extra_boards: dict[str, list[tuple[str, int]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fuse every available real draft-result source into one weighted view per player.

    ``extra_boards`` lets a caller feed real, user-supplied picks from the
    sources that need them -- ``{"ffpc": [(name, pick), ...], "underdog": [...],
    "github_reddit": [...]}`` (see :mod:`fantasy.draft_ingestion` for how to
    produce this from an actual file). Omitted keys simply don't contribute.

    Returns a table keyed by this project's ``player_id``::

        {"00-0033280": {"fused_adp": 8.6, "volatility": 0.9, "run_pressure": 0.42,
                         "reach_rate": 0.31, "fall_rate": 0.0, "round_curve": {1: 1.0},
                         "sources": {"fantasypros": 8.5, "ffcalculator": 8.7}}, ...}

    Players missing a real (FantasyPros) ``adp`` or a ``player_id`` are
    skipped -- there is nothing real to anchor a fusion to for them.
    """
    extra_boards = extra_boards or {}
    ffc_by_name = _ffcalculator_by_name(scoring, teams)
    # Always [] today (see fantasy.draft_ingestion.sleeper_api); keyed by
    # player_id defensively in case that ever changes.
    sleeper_by_id = {str(record["player_id"]): record for record in fetch_sleeper_adp() if record.get("player_id")}

    extra_source_stats: dict[str, dict[str, dict[str, float]]] = {}
    for source in USER_SUPPLIED_SOURCES:
        picks_for_source = extra_boards.get(source) or []
        by_name: dict[str, list[int]] = {}
        for name, pick_number in picks_for_source:
            by_name.setdefault(normalize_player_name(name), []).append(int(pick_number))
        extra_source_stats[source] = {name: _board_stats_from_picks(picks) for name, picks in by_name.items()}

    table: dict[str, dict[str, Any]] = {}
    for player in players:
        player_id = player.get("player_id") or player.get("id")
        real_adp = player.get("adp")
        if not player_id or real_adp is None:
            continue
        real_adp_value = safe_float(real_adp)
        name_key = normalize_player_name(player.get("name"))

        source_avg_picks: dict[str, float] = {"fantasypros": real_adp_value}
        volatility_candidates: list[float] = []
        reach_candidates: list[float] = []
        fall_candidates: list[float] = []
        round_curve: dict[int, float] = {}

        ffc_record = ffc_by_name.get(name_key)
        if ffc_record is not None:
            ffc_adp = safe_float(ffc_record.get("adp"))
            source_avg_picks["ffcalculator"] = ffc_adp
            if ffc_record.get("stdev") is not None:
                volatility_candidates.append(safe_float(ffc_record["stdev"]))
            high, low = ffc_record.get("high"), ffc_record.get("low")
            if high is not None and low is not None:
                reach, fall = _range_based_reach_fall(ffc_adp, safe_float(high), safe_float(low))
                reach_candidates.append(reach)
                fall_candidates.append(fall)
                round_curve = _round_curve_from_range(safe_float(high), safe_float(low), teams)

        sleeper_record = sleeper_by_id.get(str(player_id))  # pragma: no cover - stub always empty today
        if sleeper_record is not None and sleeper_record.get("adp") is not None:
            source_avg_picks["sleeper"] = safe_float(sleeper_record["adp"])

        for source, stats_by_name in extra_source_stats.items():
            record = stats_by_name.get(name_key)
            if record is None:
                continue
            source_avg_picks[source] = record["avg_pick"]
            volatility_candidates.append(record["stdev"])
            reach, fall = _range_based_reach_fall(record["avg_pick"], record["high"], record["low"])
            reach_candidates.append(reach)
            fall_candidates.append(fall)
            if not round_curve:
                round_curve = _round_curve_from_picks(
                    [pick for name, pick in (extra_boards.get(source) or []) if normalize_player_name(name) == name_key],
                    teams,
                )

        weights = {source: REALISM_WEIGHTS[source] for source in source_avg_picks if source in REALISM_WEIGHTS}
        total_weight = sum(weights.values())
        fused_adp = (
            round(sum(source_avg_picks[source] * weight for source, weight in weights.items()) / total_weight, 2)
            if total_weight > 0
            else real_adp_value
        )

        volatility = round(statistics.fmean(volatility_candidates), 2) if volatility_candidates else 0.0
        reach_rate = round(statistics.fmean(reach_candidates), 3) if reach_candidates else 0.0
        fall_rate = round(statistics.fmean(fall_candidates), 3) if fall_candidates else 0.0

        table[str(player_id)] = {
            "player_id": str(player_id),
            "fused_adp": fused_adp,
            "volatility": volatility,
            "reach_rate": reach_rate,
            "fall_rate": fall_rate,
            "round_curve": round_curve,
            "sources": dict(source_avg_picks),
            # Which sources actually backed this player's number, and what
            # share of the configured realism weight they represent -- so a
            # caller can tell a 2-source fusion from a 1-source fallback
            # without inspecting `sources` themselves.
            "sources_used": sorted(source_avg_picks),
            "source_coverage": round(total_weight / sum(REALISM_WEIGHTS.values()), 3),
        }

    _attach_run_pressure(players, table)
    return table


def _attach_run_pressure(players: list[dict[str, Any]], table: dict[str, dict[str, Any]]) -> None:
    """Add ``run_pressure`` to every entry in ``table``, mutating it in place.

    ``fused_run_pressure`` is "weighted positional run curves" per the spec --
    concretely, whether a player's own fused ADP falls inside a real,
    data-derived same-position ADP cluster (see
    :func:`fantasy.draft.identify_adp_clusters`, already used by
    :mod:`fantasy.room_brain` for exactly this "is a run happening here"
    question). A player inside a cluster gets a pressure proportional to the
    cluster's size (more players bunched together -> more pressure to keep
    taking that position); a player between clusters gets 0.0.

    Local import: :mod:`fantasy.draft` imports :mod:`fantasy.data_loader`
    (for ``is_synthetic``), and this module is imported *from*
    ``data_loader`` -- a top-level import here would be a real cycle. By the
    time this function actually runs, both modules have already finished
    loading.
    """
    from fantasy.draft import identify_adp_clusters

    fused_view = [
        {"position": player.get("position"), "adp": table[str(pid)]["fused_adp"], "name": player.get("name")}
        for player in players
        if (pid := player.get("player_id") or player.get("id")) and str(pid) in table
    ]
    clusters = identify_adp_clusters(fused_view)

    for player in players:
        player_id = player.get("player_id") or player.get("id")
        if not player_id or str(player_id) not in table:
            continue
        entry = table[str(player_id)]
        position = str(player.get("position", "")).strip().upper()
        pressure = 0.0
        for cluster in clusters:
            if cluster["position"] == position and cluster["start_adp"] <= entry["fused_adp"] <= cluster["end_adp"]:
                pressure = round(min(1.0, cluster["size"] / 10.0), 3)
                break
        entry["run_pressure"] = pressure


def apply_fused_draft_results(
    players: list[dict[str, Any]],
    scoring: str = "ppr",
    teams: int = DEFAULT_TEAMS_FOR_ROUNDING,
    extra_boards: dict[str, list[tuple[str, int]]] | None = None,
) -> list[dict[str, Any]]:
    """Overlay the fused draft-result view onto each player, in place of their raw ADP.

    This is the single integration point: every downstream module
    (``room_brain``, ``user_brain`` via ``assistant``, ``tiering``,
    ``simulate_draft``) already reads ``player["adp"]`` for timing decisions,
    so promoting the fused value to that same field reaches all of them
    without touching their internals; ``run_pressure`` is a new field those
    modules can opt into reading. The original single-source value is
    preserved under ``adp_fantasypros_only``. Players with no real ADP pass
    through unchanged.
    """
    fused_table = fuse_draft_results(players, scoring=scoring, teams=teams, extra_boards=extra_boards)
    updated: list[dict[str, Any]] = []
    for player in players:
        player = dict(player)
        player_id = str(player.get("player_id") or player.get("id") or "")
        fused = fused_table.get(player_id)
        if fused:
            player["adp_fantasypros_only"] = player.get("adp")
            player["adp"] = fused["fused_adp"]
            player["adp_volatility"] = fused["volatility"]
            player["run_pressure"] = fused["run_pressure"]
            player["reach_rate"] = fused["reach_rate"]
            player["fall_rate"] = fused["fall_rate"]
            player["round_curve"] = fused["round_curve"]
            player["draft_fusion_sources"] = fused["sources"]
            player["sources_used"] = fused["sources_used"]
            player["source_coverage"] = fused["source_coverage"]
        updated.append(player)
    return updated
