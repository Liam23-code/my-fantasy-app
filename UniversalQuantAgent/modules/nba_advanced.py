"""Explainable advanced NBA analytics with replaceable data-provider helpers.

Calculations accept pandas objects so they remain testable and can later consume
a database, cache, injury feed, or machine-learning pipeline instead of NBA.com.
All scores are educational estimates rather than betting probabilities.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats, leaguedashteamclutch, leaguedashteamstats,
    leaguegamelog, teamdashlineups, teamplayeronoffsummary,
)
from nba_api.stats.static import teams

from modules.nba_cache import NBA_API_TIMEOUT, fetch_nba_frames
from modules.utils import clamp, safe_number


PLAYER_FEATURES = [
    "PTS", "REB", "AST", "STL", "BLK", "TS_PCT", "USG_PCT",
    "OFF_RATING", "DEF_RATING", "PACE",
]


def latest_season() -> str:
    """Return the NBA season containing today's date."""
    today = date.today()
    start = today.year if today.month >= 10 else today.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def find_team(team_name: str) -> dict[str, Any]:
    """Find a team by city, nickname, full name, or abbreviation."""
    query = team_name.strip().lower()
    for team in teams.get_teams():
        names = {str(team.get(key, "")).lower() for key in
                 ("city", "nickname", "full_name", "abbreviation")}
        if query in names:
            return team
    raise ValueError(f"Unknown NBA team: {team_name!r}.")


def _frames(endpoint: Any) -> list[pd.DataFrame]:
    """Return non-empty DataFrames from an nba_api endpoint response."""
    return [frame for frame in endpoint.get_data_frames() if not frame.empty]


def _neutral_team_table() -> pd.DataFrame:
    """Return league-average rows so downstream models always have team data."""
    return pd.DataFrame([
        {
            "TEAM_ID": team["id"],
            "TEAM_NAME": team["full_name"],
            "TEAM_ABBREVIATION": team["abbreviation"],
            "GP": 0, "W": 0, "L": 0,
            "PACE": 100.0,
            "OFF_RATING": 110.0,
            "DEF_RATING": 110.0,
            "NET_RATING": 0.0,
            "PTS": 110.0,
            "REB": 44.0,
            "AST": 25.0,
            "FG3A": 34.0,
            "TOV": 14.0,
            "DREB": 34.0,
            "STL": 7.5,
            "TS_PCT": .56,
        }
        for team in teams.get_teams()
    ])


def fetch_league_team_stats(season: str, last_n_games: int = 0) -> pd.DataFrame:
    """Fetch merged team statistics with retry, cache, and neutral fallback."""
    common = {
        "season": season,
        "season_type_all_star": "Regular Season",
        "last_n_games": last_n_games,
        "timeout": NBA_API_TIMEOUT,
    }
    base_frames = fetch_nba_frames(
        f"advanced_team_base_{season}_{last_n_games}",
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Base", **common
        ).get_data_frames(),
        fallback_factory=_neutral_team_table,
    )
    advanced_frames = fetch_nba_frames(
        f"advanced_team_ratings_{season}_{last_n_games}",
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced", **common
        ).get_data_frames(),
        fallback_factory=_neutral_team_table,
    )
    base = base_frames[0] if base_frames else _neutral_team_table()
    advanced = advanced_frames[0] if advanced_frames else _neutral_team_table()
    extra = [
        column for column in advanced.columns
        if column == "TEAM_ID" or column not in base.columns
    ]
    result = base.merge(advanced[extra], on="TEAM_ID", how="left")
    warnings = list(base.attrs.get("warnings", [])) + list(
        advanced.attrs.get("warnings", [])
    )
    result.attrs["warnings"] = list(dict.fromkeys(warnings))
    result.attrs["data_source"] = (
        "live"
        if base.attrs.get("data_source") == advanced.attrs.get("data_source") == "live"
        else "fallback"
    )
    return result


def fetch_league_game_logs(season: str) -> pd.DataFrame:
    """Fetch every team game for opponent and schedule calculations."""
    return leaguegamelog.LeagueGameLog(
        season=season, season_type_all_star="Regular Season",
        player_or_team_abbreviation="T", sorter="DATE", direction="DESC",
        timeout=NBA_API_TIMEOUT).get_data_frames()[0]


def fetch_clutch_stats(season: str) -> pd.DataFrame:
    """Fetch games within five points in the final five minutes."""
    return leaguedashteamclutch.LeagueDashTeamClutch(
        season=season, season_type_all_star="Regular Season",
        clutch_time="Last 5 Minutes", point_diff="5", per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced", timeout=NBA_API_TIMEOUT).get_data_frames()[0]


def fetch_on_off_splits(team_id: int, season: str) -> list[pd.DataFrame]:
    """Fetch player on-court and off-court summary tables for a team."""
    endpoint = teamplayeronoffsummary.TeamPlayerOnOffSummary(
        team_id=team_id, season=season, season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame",
        timeout=NBA_API_TIMEOUT)
    return _frames(endpoint)


def fetch_lineups(team_id: int, season: str) -> pd.DataFrame:
    """Fetch five-player lineup results for one team."""
    frames = _frames(teamdashlineups.TeamDashLineups(
        team_id=team_id, season=season, season_type_all_star="Regular Season",
        group_quantity="5", measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame", timeout=NBA_API_TIMEOUT))
    return frames[0] if frames else pd.DataFrame()


def opponent_ids_from_logs(team_id: int, game_logs: pd.DataFrame) -> list[int]:
    """Pair same-game rows to recover a team's opponents, including repeats."""
    opponents: list[int] = []
    if not {"GAME_ID", "TEAM_ID"}.issubset(game_logs.columns):
        return opponents
    selected = game_logs[game_logs["TEAM_ID"] == team_id]
    for game_id in selected["GAME_ID"].dropna().unique():
        ids = game_logs.loc[
            (game_logs["GAME_ID"] == game_id) & (game_logs["TEAM_ID"] != team_id),
            "TEAM_ID",
        ].tolist()
        if ids:
            opponents.append(int(ids[0]))
    return opponents


def _opponent_rows(
    team_id: int, league_stats: pd.DataFrame, game_logs: pd.DataFrame,
) -> pd.DataFrame:
    """Return opponent rows with repeat matchups weighted correctly."""
    ids = opponent_ids_from_logs(team_id, game_logs)
    indexed = league_stats.set_index("TEAM_ID")
    valid = [opponent_id for opponent_id in ids if opponent_id in indexed.index]
    return indexed.loc[valid].reset_index() if valid else pd.DataFrame()


def calculate_opponent_adjusted_metrics(
    team_id: int, league_stats: pd.DataFrame, game_logs: pd.DataFrame,
) -> dict[str, float]:
    """Adjust ratings for average opponent offense and defense faced."""
    rows = league_stats[league_stats["TEAM_ID"] == team_id]
    if rows.empty:
        raise ValueError(f"No league stats found for team id {team_id}.")
    team = rows.iloc[0]
    opponents = _opponent_rows(team_id, league_stats, game_logs)
    league_offense = safe_number(league_stats["OFF_RATING"].mean())
    league_defense = safe_number(league_stats["DEF_RATING"].mean())
    opponent_offense = safe_number(opponents.get("OFF_RATING", pd.Series()).mean(), league_offense)
    opponent_defense = safe_number(opponents.get("DEF_RATING", pd.Series()).mean(), league_defense)
    adjusted_offense = safe_number(team.get("OFF_RATING")) + opponent_defense - league_defense
    adjusted_defense = safe_number(team.get("DEF_RATING")) - opponent_offense + league_offense
    return {
        "opponent_adjusted_offensive_rating": round(adjusted_offense, 2),
        "opponent_adjusted_defensive_rating": round(adjusted_defense, 2),
        "opponent_adjusted_net_rating": round(adjusted_offense - adjusted_defense, 2),
    }


def calculate_strength_of_schedule(
    team_id: int, league_stats: pd.DataFrame, game_logs: pd.DataFrame,
) -> dict[str, Any]:
    """Estimate SOS as opponents' average net rating relative to the league."""
    opponents = _opponent_rows(team_id, league_stats, game_logs)
    league_average = safe_number(league_stats["NET_RATING"].mean())
    opponent_average = safe_number(opponents.get("NET_RATING", pd.Series()).mean(), league_average)
    sos = opponent_average - league_average
    label = "difficult" if sos > 1 else "easy" if sos < -1 else "average"
    return {"sos_rating": round(sos, 2), "schedule_label": label,
            "opponents_counted": len(opponents)}


def summarize_clutch(team_id: int, clutch_table: pd.DataFrame) -> dict[str, Any]:
    """Summarize last-five-minute performance from a clutch table."""
    rows = clutch_table[clutch_table["TEAM_ID"] == team_id]
    if rows.empty:
        return {"games": 0, "plus_minus": 0.0, "net_rating": 0.0,
                "note": "No qualifying clutch possessions were available."}
    row = rows.iloc[0]
    wins, losses = safe_number(row.get("W")), safe_number(row.get("L"))
    return {"games": int(wins + losses), "wins": int(wins), "losses": int(losses),
            "plus_minus": round(safe_number(row.get("PLUS_MINUS")), 2),
            "net_rating": round(safe_number(row.get("NET_RATING")), 2)}


def summarize_on_off(on_off_tables: list[pd.DataFrame], limit: int = 5) -> list[dict[str, Any]]:
    """Rank the top five players by absolute on/off net-rating swing."""
    if not on_off_tables:
        return []
    on = on_off_tables[0].copy()
    off = on_off_tables[1].copy() if len(on_off_tables) > 1 else pd.DataFrame()
    id_column = "VS_PLAYER_ID" if "VS_PLAYER_ID" in on else "PLAYER_ID"
    name_column = "VS_PLAYER_NAME" if "VS_PLAYER_NAME" in on else "PLAYER_NAME"
    results: list[dict[str, Any]] = []
    for _, row in on.iterrows():
        matches = off[off[id_column] == row.get(id_column)] if id_column in off else pd.DataFrame()
        off_rating = safe_number(matches.iloc[0].get("NET_RATING")) if not matches.empty else 0.0
        on_rating = safe_number(row.get("NET_RATING"))
        results.append({"player": str(row.get(name_column, "Unknown")),
                        "on_net_rating": round(on_rating, 2),
                        "off_net_rating": round(off_rating, 2),
                        "on_off_swing": round(on_rating - off_rating, 2)})
    return sorted(results, key=lambda item: abs(item["on_off_swing"]), reverse=True)[:limit]


def summarize_lineups(lineups: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    """Return the ten most-used lineups and their efficiency."""
    if lineups.empty:
        return []
    sort_column = "MIN" if "MIN" in lineups else "GP"
    table = lineups.sort_values(sort_column, ascending=False).head(limit)
    return [{"lineup": str(row.get("GROUP_NAME", row.get("GROUP_ID", "Unknown"))),
             "games": int(safe_number(row.get("GP"))),
             "minutes": round(safe_number(row.get("MIN")), 1),
             "offensive_rating": round(safe_number(row.get("OFF_RATING")), 2),
             "defensive_rating": round(safe_number(row.get("DEF_RATING")), 2),
             "net_rating": round(safe_number(row.get("NET_RATING")), 2)}
            for _, row in table.iterrows()]


def classify_playstyle(team_row: pd.Series, league_stats: pd.DataFrame) -> dict[str, Any]:
    """Assign transparent playstyle clusters using league medians."""
    def label(column: str, high: str, low: str) -> str:
        median = safe_number(league_stats[column].median())
        return high if safe_number(team_row.get(column)) >= median else low
    pace = label("PACE", "fast", "slow")
    shot = label("FG3A", "perimeter-heavy", "paint/midrange-heavy")
    turnover = label("TOV", "high-turnover", "ball-secure")
    return {"cluster": f"{pace}, {shot}, {turnover}", "pace_profile": pace,
            "shot_profile": shot, "turnover_profile": turnover,
            "inputs": {"pace": round(safe_number(team_row.get("PACE")), 2),
                       "three_point_attempts": round(safe_number(team_row.get("FG3A")), 2),
                       "turnovers": round(safe_number(team_row.get("TOV")), 2)}}


def team_games(team_id: int, game_logs: pd.DataFrame) -> pd.DataFrame:
    """Return one team's games in chronological order with parsed dates."""
    games = game_logs[game_logs["TEAM_ID"] == team_id].copy()
    date_column = "GAME_DATE" if "GAME_DATE" in games else "GAME_DATE_EST"
    games[date_column] = pd.to_datetime(games[date_column])
    return games.sort_values(date_column).reset_index(drop=True)


def detect_hot_cold_streaks(games: pd.DataFrame) -> dict[str, Any]:
    """Label last-five and last-ten win rates as hot, cold, or neutral."""
    def summarize(size: int) -> dict[str, Any]:
        recent = games.tail(size)
        wins = int((recent["WL"] == "W").sum()) if "WL" in recent else 0
        played = len(recent)
        rate = wins / played if played else 0.0
        status = ("hot" if played >= 3 and rate >= 0.7 else
                  "cold" if played >= 3 and rate <= 0.3 else "neutral")
        return {"games": played, "wins": wins,
                "win_pct": round(rate * 100, 1), "label": status}
    return {"last_5": summarize(5), "last_10": summarize(10)}


def detect_efficiency_spikes(season_row: pd.Series, recent_row: pd.Series) -> dict[str, Any]:
    """Compare recent TS%, offensive rating, and net rating with season levels."""
    results: dict[str, Any] = {}
    for output, column, scale in (("true_shooting", "TS_PCT", 100),
                                  ("offensive_rating", "OFF_RATING", 1),
                                  ("net_rating", "NET_RATING", 1)):
        baseline = safe_number(season_row.get(column)) * scale
        recent = safe_number(recent_row.get(column)) * scale
        change = recent - baseline
        signal = "spike" if change >= 2 else "drop" if change <= -2 else "stable"
        results[output] = {"season": round(baseline, 2), "recent": round(recent, 2),
                           "change": round(change, 2), "signal": signal}
    return results


def detect_role_changes(season_players: pd.DataFrame, recent_players: pd.DataFrame,
                        team_id: int | None = None,
                        minimum_minutes: float = 10) -> list[dict[str, Any]]:
    """Detect usage-rate changes between season and recent windows."""
    season, recent = season_players.copy(), recent_players.copy()
    if team_id is not None and "TEAM_ID" in season:
        season, recent = season[season["TEAM_ID"] == team_id], recent[recent["TEAM_ID"] == team_id]
    columns = ["PLAYER_ID", "PLAYER_NAME", "USG_PCT", "MIN"]
    merged = season[columns].merge(recent[columns], on="PLAYER_ID",
                                   suffixes=("_season", "_recent"))
    changes = []
    for _, row in merged.iterrows():
        if safe_number(row.get("MIN_recent")) < minimum_minutes:
            continue
        change = (safe_number(row.get("USG_PCT_recent")) -
                  safe_number(row.get("USG_PCT_season"))) * 100
        if abs(change) >= 2:
            changes.append({"player": row.get("PLAYER_NAME_season", "Unknown"),
                            "usage_change_pct_points": round(change, 2),
                            "role_signal": "expanded" if change > 0 else "reduced"})
    return sorted(changes, key=lambda item: abs(item["usage_change_pct_points"]), reverse=True)


def detect_injury_impact(games: pd.DataFrame, injury_date: str | date) -> dict[str, Any]:
    """Compare ten games before and after an externally supplied injury date."""
    data = games.copy()
    date_column = "GAME_DATE" if "GAME_DATE" in data else "GAME_DATE_EST"
    data[date_column] = pd.to_datetime(data[date_column])
    split = pd.Timestamp(injury_date)
    before, after = data[data[date_column] < split].tail(10), data[data[date_column] >= split].head(10)
    def summary(window: pd.DataFrame) -> dict[str, Any]:
        wins = (window["WL"] == "W").mean() * 100 if len(window) else 0.0
        margin = safe_number(window["PLUS_MINUS"].mean()) if "PLUS_MINUS" in window else 0.0
        return {"games": len(window), "win_pct": round(wins, 1),
                "average_plus_minus": round(margin, 2)}
    pre, post = summary(before), summary(after)
    return {"before": pre, "after": post,
            "plus_minus_change": round(post["average_plus_minus"] - pre["average_plus_minus"], 2)}


def detect_back_to_back_performance(games: pd.DataFrame) -> dict[str, Any]:
    """Compare zero-rest games with games after at least one rest day."""
    data = games.copy()
    date_column = "GAME_DATE" if "GAME_DATE" in data else "GAME_DATE_EST"
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column)
    data["REST_DAYS"] = data[date_column].diff().dt.days - 1
    def summary(window: pd.DataFrame) -> dict[str, Any]:
        wins = (window["WL"] == "W").mean() * 100 if len(window) else 0.0
        margin = safe_number(window["PLUS_MINUS"].mean()) if "PLUS_MINUS" in window else 0.0
        return {"games": len(window), "win_pct": round(wins, 1),
                "average_plus_minus": round(margin, 2)}
    short = summary(data[data["REST_DAYS"] <= 0])
    rested = summary(data[data["REST_DAYS"] > 0])
    return {"back_to_back": short, "rested": rested,
            "win_pct_change": round(short["win_pct"] - rested["win_pct"], 1)}


def fetch_player_tables(season: str, last_n_games: int = 0) -> pd.DataFrame:
    """Fetch merged player tables through the shared resilient data boundary."""
    common = {
        "season": season,
        "season_type_all_star": "Regular Season",
        "last_n_games": last_n_games,
        "timeout": NBA_API_TIMEOUT,
    }
    base_frames = fetch_nba_frames(
        f"advanced_player_base_{season}_{last_n_games}",
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            measure_type_detailed_defense="Base", **common
        ).get_data_frames(),
        fallback_factory=pd.DataFrame,
    )
    advanced_frames = fetch_nba_frames(
        f"advanced_player_ratings_{season}_{last_n_games}",
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            measure_type_detailed_defense="Advanced", **common
        ).get_data_frames(),
        fallback_factory=pd.DataFrame,
    )
    base = base_frames[0] if base_frames else pd.DataFrame()
    advanced = advanced_frames[0] if advanced_frames else pd.DataFrame()
    if base.empty or "PLAYER_ID" not in base:
        base.attrs["warnings"] = list(dict.fromkeys(
            list(base.attrs.get("warnings", []))
            + list(advanced.attrs.get("warnings", []))
        ))
        return base
    if advanced.empty or "PLAYER_ID" not in advanced:
        for column, default in (
            ("USG_PCT", .20), ("TS_PCT", .55), ("PACE", 100.0),
            ("OFF_RATING", 110.0), ("DEF_RATING", 110.0),
            ("NET_RATING", 0.0),
        ):
            if column not in base:
                base[column] = default
        return base
    extra = [
        column for column in advanced.columns
        if column == "PLAYER_ID" or column not in base
    ]
    result = base.merge(advanced[extra], on="PLAYER_ID", how="left")
    result.attrs["warnings"] = list(dict.fromkeys(
        list(base.attrs.get("warnings", []))
        + list(advanced.attrs.get("warnings", []))
    ))
    return result


def _player_row(name: str, table: pd.DataFrame) -> pd.Series:
    """Find an exact player name, then one unambiguous partial match."""
    query = name.strip().lower()
    exact = table[table["PLAYER_NAME"].str.lower() == query]
    partial = table[table["PLAYER_NAME"].str.lower().str.contains(query, regex=False)]
    if len(exact) == 1:
        return exact.iloc[0]
    if len(partial) == 1:
        return partial.iloc[0]
    raise ValueError(f"Could not uniquely identify NBA player {name!r}.")


def _percentile_vector(row: pd.Series, table: pd.DataFrame,
                       features: Iterable[str]) -> np.ndarray:
    """Represent a player as percentiles so every feature shares a scale."""
    values = []
    for feature in features:
        series = pd.to_numeric(table[feature], errors="coerce").fillna(0)
        values.append(float((series <= safe_number(row.get(feature))).mean()))
    return np.asarray(values, dtype=float)


def compare_players(player_one: str, player_two: str,
                    season: str | None = None) -> dict[str, Any]:
    """Compare advanced profiles using cosine similarity and percentiles."""
    season = season or latest_season()
    table = fetch_player_tables(season)
    first, second = _player_row(player_one, table), _player_row(player_two, table)
    features = [feature for feature in PLAYER_FEATURES if feature in table]
    one, two = _percentile_vector(first, table, features), _percentile_vector(second, table, features)
    denominator = np.linalg.norm(one) * np.linalg.norm(two)
    similarity = float(np.dot(one, two) / denominator) if denominator else 0.0
    def profile(row: pd.Series, vector: np.ndarray) -> dict[str, Any]:
        ranked = sorted(zip(features, vector), key=lambda item: item[1], reverse=True)
        return {"player": row["PLAYER_NAME"], "team": row.get("TEAM_ABBREVIATION", ""),
                "strengths": [{"metric": key, "percentile": round(value * 100, 1)}
                              for key, value in ranked[:3]],
                "weaknesses": [{"metric": key, "percentile": round(value * 100, 1)}
                               for key, value in ranked[-3:]],
                "metrics": {key.lower(): round(safe_number(row.get(key)), 3) for key in features}}
    difficulty = clamp((1 - similarity) * 50 + safe_number(two.mean()) * 50)
    return {"domain": "sports_player_comparison", "season": season,
            "similarity_score": round(similarity * 100, 1),
            "matchup_difficulty": round(difficulty, 1),
            "players": [profile(first, one), profile(second, two)],
            "explanation": "Similarity uses league-percentile profiles; difficulty is a heuristic, not a prediction."}

def _team_snapshot(team: dict[str, Any], league: pd.DataFrame,
                   recent: pd.DataFrame, logs: pd.DataFrame,
                   clutch: pd.DataFrame) -> dict[str, Any]:
    """Build one advanced team snapshot from shared league tables."""
    rows, recent_rows = league[league["TEAM_ID"] == team["id"]], recent[recent["TEAM_ID"] == team["id"]]
    if rows.empty:
        raise ValueError(f"No advanced stats found for {team['full_name']}.")
    season_row = rows.iloc[0]
    recent_row = recent_rows.iloc[0] if not recent_rows.empty else season_row
    games = team_games(team["id"], logs)
    return {
        "team": team["full_name"], "team_id": team["id"],
        "adjusted_ratings": calculate_opponent_adjusted_metrics(team["id"], league, logs),
        "strength_of_schedule": calculate_strength_of_schedule(team["id"], league, logs),
        "clutch": summarize_clutch(team["id"], clutch),
        "playstyle": classify_playstyle(season_row, league),
        "trends": {
            "streaks": detect_hot_cold_streaks(games),
            "efficiency": detect_efficiency_spikes(season_row, recent_row),
            "back_to_back": detect_back_to_back_performance(games),
            "injury_impact": {"status": "requires an injury date or external injury feed"},
        },
    }


def analyze_matchup(team_one: str, team_two: str,
                    season: str | None = None) -> dict[str, Any]:
    """Run schedule, trend, clutch, lineup, on/off, and playstyle analysis."""
    season = season or latest_season()
    first, second = find_team(team_one), find_team(team_two)
    if first["id"] == second["id"]:
        raise ValueError("Please choose two different NBA teams.")
    league = fetch_league_team_stats(season)
    recent = fetch_league_team_stats(season, last_n_games=10)
    logs, clutch = fetch_league_game_logs(season), fetch_clutch_stats(season)
    season_players = fetch_player_tables(season)
    recent_players = fetch_player_tables(season, last_n_games=10)
    snapshots = [_team_snapshot(first, league, recent, logs, clutch),
                 _team_snapshot(second, league, recent, logs, clutch)]
    for snapshot, team in zip(snapshots, (first, second)):
        snapshot["trends"]["role_changes"] = detect_role_changes(
            season_players, recent_players, team["id"])
        snapshot["top_player_on_off_splits"] = summarize_on_off(
            fetch_on_off_splits(team["id"], season))
        snapshot["top_lineups"] = summarize_lineups(fetch_lineups(team["id"], season))
    edge = (snapshots[0]["adjusted_ratings"]["opponent_adjusted_net_rating"] -
            snapshots[1]["adjusted_ratings"]["opponent_adjusted_net_rating"])
    favored = snapshots[0 if edge >= 0 else 1]["team"]
    styles = [snapshot["playstyle"] for snapshot in snapshots]
    conflicts = sum(styles[0][key] != styles[1][key] for key in
                    ("pace_profile", "shot_profile", "turnover_profile"))
    difficulty = clamp(100 - min(abs(edge), 20) * 3 + conflicts * 8)
    return {
        "domain": "sports_advanced", "subject": f"{first['abbreviation']} vs {second['abbreviation']}",
        "season": season, "teams": snapshots,
        "matchup": {"favored_team": favored,
                    "opponent_adjusted_net_rating_edge": round(abs(edge), 2),
                    "style_conflicts": conflicts,
                    "difficulty_score": round(difficulty, 1),
                    "explanation": "Difficulty rises when adjusted quality is close and playstyles conflict."},
        "future_ml_features": ["opponent_adjusted_ratings", "strength_of_schedule",
                               "clutch_net_rating", "recent_efficiency_changes",
                               "back_to_back_results", "playstyle_profiles",
                               "lineup_net_ratings", "player_on_off_swings"],
    }