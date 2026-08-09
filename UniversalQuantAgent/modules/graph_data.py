"""Shared, outage-safe data adapters for the basketball Graph Lab."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import players, teams

from modules.data_quality import coerce_numeric, fuzzy_name_match, safe_number
from modules.nba_advanced import fetch_player_tables, latest_season
from modules.nba_cache import NBA_API_TIMEOUT, collect_warnings, fetch_nba_frames
from modules.projections import fetch_player_game_log
from modules.sports import fetch_nba_player_stats


TEAM_COLORS = {
    "ATL":"#E03A3E","BOS":"#007A33","BKN":"#111111","CHA":"#1D1160",
    "CHI":"#CE1141","CLE":"#860038","DAL":"#00538C","DEN":"#0E2240",
    "DET":"#C8102E","GSW":"#1D428A","HOU":"#CE1141","IND":"#002D62",
    "LAC":"#C8102E","LAL":"#552583","MEM":"#5D76A9","MIA":"#98002E",
    "MIL":"#00471B","MIN":"#0C2340","NOP":"#0C2340","NYK":"#006BB6",
    "OKC":"#007AC1","ORL":"#0077C0","PHI":"#006BB6","PHX":"#1D1160",
    "POR":"#E03A3E","SAC":"#5A2D81","SAS":"#2F2F2F","TOR":"#CE1141",
    "UTA":"#002B5C","WAS":"#002B5C",
}
TEAM_SECONDARY = {
    "ATL":"#FDB927","BOS":"#BA9653","BKN":"#A7A9AC","CHA":"#00788C",
    "CHI":"#111111","CLE":"#FDBB30","DAL":"#B8C4CA","DEN":"#FEC524",
    "DET":"#1D42BA","GSW":"#FFC72C","HOU":"#111111","IND":"#FDBB30",
    "LAC":"#1D428A","LAL":"#FDB927","MEM":"#12173F","MIA":"#F9A01B",
    "MIL":"#EEE1C6","MIN":"#78BE20","NOP":"#C8102E","NYK":"#F58426",
    "OKC":"#EF3B24","ORL":"#C4CED4","PHI":"#ED174C","PHX":"#E56020",
    "POR":"#111111","SAC":"#63727A","SAS":"#C4CED4","TOR":"#000000",
    "UTA":"#F9A01B","WAS":"#E31837",
}


def resolve_player(player_id_or_name: int | str) -> dict[str, Any]:
    """Resolve an NBA player id or fuzzy name without a network call."""
    roster = players.get_players()
    if isinstance(player_id_or_name, (int, np.integer)) or str(player_id_or_name).isdigit():
        player_id = int(player_id_or_name)
        match = next((row for row in roster if int(row.get("id", -1)) == player_id), None)
    else:
        match = fuzzy_name_match(
            str(player_id_or_name), roster,
            key=lambda row: str(row.get("full_name", "")), cutoff=.72,
        )
    if not match:
        raise ValueError(f"Could not identify NBA player {player_id_or_name!r}.")
    return dict(match)


def team_palette(abbreviation: str) -> tuple[str, str]:
    abbreviation = str(abbreviation).upper()
    return TEAM_COLORS.get(abbreviation, "#334155"), TEAM_SECONDARY.get(abbreviation, "#94A3B8")


def percentile(series: pd.Series, value: Any, higher_is_better: bool = True) -> float:
    """Return a finite 0-100 percentile for one value."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    target = safe_number(value, float("nan"))
    if values.empty or pd.isna(target):
        return 50.0
    score = (values <= target).mean() if higher_is_better else (values >= target).mean()
    return round(float(score) * 100.0, 1)


def load_player_table(season: str | None = None, last_n_games: int = 0) -> pd.DataFrame:
    """Return the merged public player table with safe numeric columns."""
    table = fetch_player_tables(season or latest_season(), last_n_games=last_n_games)
    return table.copy()


def player_row(
    player_id_or_name: int | str,
    season: str | None = None,
    last_n_games: int = 0,
) -> tuple[dict[str, Any], pd.Series, pd.DataFrame, list[str]]:
    """Return player identity, selected row, league table, and data warnings."""
    season = season or latest_season()
    player = resolve_player(player_id_or_name)
    table = load_player_table(season, last_n_games)
    warnings = list(table.attrs.get("warnings", [])) if isinstance(table, pd.DataFrame) else []
    row = pd.Series(dtype=object)
    if not table.empty:
        if "PLAYER_ID" in table:
            matches = table[pd.to_numeric(table["PLAYER_ID"], errors="coerce") == int(player["id"])]
        elif "PLAYER_NAME" in table:
            matches = table[table["PLAYER_NAME"].astype(str).str.lower() == str(player["full_name"]).lower()]
        else:
            matches = pd.DataFrame()
        if not matches.empty:
            row = matches.iloc[0]
    if row.empty:
        snapshot = fetch_nba_player_stats(player["full_name"], season)
        season_avg = snapshot.get("season_avg", {})
        advanced = snapshot.get("advanced", {})
        row = pd.Series({
            "PLAYER_ID": player["id"], "PLAYER_NAME": snapshot.get("player", player["full_name"]),
            "TEAM_ABBREVIATION": snapshot.get("team", ""),
            "MIN": season_avg.get("mpg"), "PTS": season_avg.get("ppg"),
            "REB": season_avg.get("rpg"), "AST": season_avg.get("apg"),
            "USG_PCT": safe_number(advanced.get("usage_pct")) / 100.0,
            "TS_PCT": safe_number(advanced.get("ts_pct")) / 100.0,
            "PER_ESTIMATE": advanced.get("per_estimate"), "PACE": 100.0,
        })
        table = pd.DataFrame([row])
        warnings.append("League table unavailable; the selected player's cached snapshot is shown.")
    return player, row, table, list(dict.fromkeys(str(item) for item in warnings if item))


def player_game_data(
    player_id_or_name: int | str,
    season: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    """Return normalized chronological game data for momentum visuals."""
    season = season or latest_season()
    player = resolve_player(player_id_or_name)
    raw = fetch_player_game_log(int(player["id"]), season)
    warnings = list(raw.attrs.get("warnings", [])) if isinstance(raw, pd.DataFrame) else []
    data = coerce_numeric(raw)
    if not data.empty:
        if "game_date" in data:
            data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
            data = data.sort_values("game_date")
        else:
            data["game_date"] = pd.date_range(end=pd.Timestamp.now(), periods=len(data), freq="3D")
        data = data.rename(columns={"pts":"points", "reb":"rebounds", "ast":"assists", "min":"minutes"})
    if data.empty or "points" not in data:
        snapshot = fetch_nba_player_stats(player["full_name"], season)
        season_avg = snapshot.get("season_avg", {})
        advanced = snapshot.get("advanced", {})
        data = pd.DataFrame([{
            "game_date": date, "points": season_avg.get("ppg", 0),
            "rebounds": season_avg.get("rpg", 0), "assists": season_avg.get("apg", 0),
            "minutes": season_avg.get("mpg", 0),
            "usage_pct": advanced.get("usage_pct", 0), "ts_pct": advanced.get("ts_pct", 0),
            "per_estimate": advanced.get("per_estimate", 0),
        } for date in pd.date_range(end=pd.Timestamp.now(), periods=10, freq="3D")])
        warnings.append("Game log unavailable; season-average fallback values are shown.")
    def numeric_series(column: str, default: float = 0.0) -> pd.Series:
        source = (
            data[column]
            if column in data
            else pd.Series(default, index=data.index, dtype=float)
        )
        return pd.to_numeric(source, errors="coerce").fillna(default)

    usage_events = (
        numeric_series("fga") + .44 * numeric_series("fta")
        + numeric_series("tov")
    ) if any(column in data for column in ("fga", "fta", "tov")) else None
    if "usage_pct" not in data:
        minutes = numeric_series("minutes").replace(0, np.nan)
        data["usage_pct"] = usage_events.div(minutes).mul(100).fillna(0) if usage_events is not None else 0.0
    if "ts_pct" not in data:
        attempts = numeric_series("fga") + .44 * numeric_series("fta")
        data["ts_pct"] = numeric_series("points").div(2 * attempts.replace(0, np.nan)).mul(100).fillna(0)
    if "per_estimate" not in data:
        minutes = numeric_series("minutes").replace(0, np.nan)
        positives = sum(
            (numeric_series(column) for column in (
                "points", "rebounds", "assists", "stl", "blk"
            )),
            pd.Series(0.0, index=data.index),
        )
        data["per_estimate"] = positives.div(minutes).mul(15).fillna(0)
    data = data.reset_index(drop=True)
    data["game_number"] = np.arange(1, len(data) + 1)
    data.attrs["warnings"] = list(dict.fromkeys(str(item) for item in warnings if item))
    return player, data, data.attrs["warnings"]


def load_team_table(season: str | None = None) -> pd.DataFrame:
    """Return merged base/advanced NBA team data for matchup curves."""
    season = season or latest_season()
    neutral = pd.DataFrame([{
        "TEAM_ID": team["id"], "TEAM_NAME": team["full_name"],
        "TEAM_ABBREVIATION": team["abbreviation"], "PTS": 110.0,
        "REB": 44.0, "AST": 25.0, "STL": 7.5, "BLK": 5.0,
        "PACE": 100.0, "OFF_RATING": 110.0, "DEF_RATING": 110.0,
    } for team in teams.get_teams()])
    common = dict(season=season, season_type_all_star="Regular Season", timeout=NBA_API_TIMEOUT)
    base_frames = fetch_nba_frames(
        f"graph_team_base_{season}",
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Base", **common
        ).get_data_frames(), fallback_factory=lambda: neutral,
    )
    advanced_frames = fetch_nba_frames(
        f"graph_team_advanced_{season}",
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced", **common
        ).get_data_frames(), fallback_factory=lambda: neutral,
    )
    base = base_frames[0] if base_frames else neutral.copy()
    advanced = advanced_frames[0] if advanced_frames else neutral.copy()
    if "TEAM_ID" in base and "TEAM_ID" in advanced:
        extra = [column for column in advanced if column == "TEAM_ID" or column not in base]
        result = base.merge(advanced[extra], on="TEAM_ID", how="left")
    else:
        result = neutral.copy()
    result.attrs["warnings"] = collect_warnings(base, advanced)
    return result
