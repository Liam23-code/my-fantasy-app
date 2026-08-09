"""Public NFL player lookup and profile contracts."""
from __future__ import annotations

from typing import Any

from modules.nfl_stats import canonical_player_stats, detect_player_position, normalized_profile, resolve_player


def get_nfl_player(player_name: str, season: int | None = None) -> dict[str, Any]:
    """Return one clean player dictionary with canonical NFL metrics."""
    row, table, warnings = resolve_player(player_name, season)
    stats = canonical_player_stats(row)
    position = detect_player_position(stats)
    if stats.get("projection_low_confidence"):
        warnings = [*warnings, "Some NFL projection fields were derived or unavailable; confidence was reduced."]
    return {
        "player": str(row.get("player", player_name)),
        "player_id": str(row.get("player_id", "")),
        "team": str(row.get("team", "")),
        "position": position,
        "player_position": position,
        "season": int(row.get("season", season or 0)),
        "stats": stats,
        "source": table.attrs.get("source", "unknown"),
        "warnings": list(dict.fromkeys(warnings)),
    }


def get_nfl_player_profile(
    player_name: str,
    season: int | None = None,
    comparison_mode: str = "League",
) -> dict[str, Any]:
    """Return league/position percentiles using the canonical 60/40 blend."""
    return normalized_profile(player_name, season, comparison_mode)
