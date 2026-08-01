"""Weekly update pipeline: pull projections, recompute everything, persist a snapshot.

Usage::

    from fantasy.pipeline import update_weekly

    result = update_weekly(
        projection_source=lambda: fetch_this_weeks_projections(),  # mockable
        league_settings=league_settings,
        week=1,
        my_roster=my_roster_players,
        available_players_source=lambda: fetch_free_agents(),
    )

``projection_source`` and ``available_players_source`` are zero-argument
callables rather than data, specifically so tests (and the CLI) can inject a
fake data source without touching a real provider.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fantasy.draft import rank_players_for_draft
from fantasy.models import LeagueSettings
from fantasy.optimizer import optimize_lineup, start_sit_advice
from fantasy.waiver import waiver_recommendations

DEFAULT_SNAPSHOT_DIR = "data/snapshots"


def _coerce_league_settings(league_settings: dict[str, Any] | LeagueSettings) -> LeagueSettings:
    if isinstance(league_settings, LeagueSettings):
        return league_settings
    return LeagueSettings(**(league_settings or {}))


def snapshot_path(week: int, snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR) -> Path:
    return Path(snapshot_dir) / f"week_{week:02d}.json"


def persist_snapshot(week: int, players: list[dict[str, Any]], snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR) -> Path:
    """Write a minimal {player_id: {name, position, points}} snapshot for one week."""
    path = snapshot_path(week, snapshot_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        p["player_id"]: {"name": p.get("name", ""), "position": p.get("position", ""), "points": p.get("points", 0.0)} for p in players if p.get("player_id")
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_snapshot(week: int, snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR) -> dict[str, Any] | None:
    path = snapshot_path(week, snapshot_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def diff_snapshots(current: dict[str, Any], previous: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Per-player point deltas between two snapshots, sorted by |delta| descending."""
    if not previous:
        return []
    movers = []
    for player_id, entry in current.items():
        previous_entry = previous.get(player_id)
        if previous_entry is None:
            continue
        delta = round(entry["points"] - previous_entry["points"], 2)
        if delta != 0:
            movers.append({"player_id": player_id, "name": entry["name"], "position": entry["position"], "delta": delta})
    movers.sort(key=lambda m: abs(m["delta"]), reverse=True)
    return movers


def update_weekly(
    projection_source: Callable[[], list[Any]],
    league_settings: dict[str, Any] | LeagueSettings,
    week: int,
    my_roster: list[dict[str, Any]] | None = None,
    available_players_source: Callable[[], list[Any]] | None = None,
    snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
) -> dict[str, Any]:
    """Run one week's full refresh: score, rank, optimize, recommend, and snapshot.

    Returns ``{"week", "ranked_players", "lineup", "start_sit_advice",
    "waiver_recommendations", "movers", "snapshot_path"}``. ``lineup``/
    ``start_sit_advice`` are omitted when ``my_roster`` isn't given;
    ``waiver_recommendations`` is omitted when ``available_players_source``
    isn't given.
    """
    settings = _coerce_league_settings(league_settings)
    raw_projections = projection_source()
    ranked_players = rank_players_for_draft(raw_projections, settings)

    result: dict[str, Any] = {"week": week, "ranked_players": ranked_players}

    if my_roster is not None:
        result["lineup"] = optimize_lineup(my_roster, raw_projections, settings)
        result["start_sit_advice"] = start_sit_advice(my_roster, raw_projections, settings)

    if available_players_source is not None:
        league_state = {"league_settings": settings, "my_roster": my_roster or [], "current_week": week}
        result["waiver_recommendations"] = waiver_recommendations(league_state, available_players_source(), settings.scoring_mode)

    path = persist_snapshot(week, ranked_players, snapshot_dir)
    previous = load_snapshot(week - 1, snapshot_dir) if week > 1 else None
    current = load_snapshot(week, snapshot_dir)
    result["movers"] = diff_snapshots(current or {}, previous)
    result["snapshot_path"] = str(path)
    return result
