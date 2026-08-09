"""NBA data collection, efficiency metrics, and optional advanced analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats, playergamelog
from nba_api.stats.static import players, teams

from modules.data_quality import coerce_numeric, fuzzy_name_match, safe_number
from modules.nba_cache import NBA_API_TIMEOUT, fetch_nba_frames


def _team_lookup(team_name: str) -> dict[str, Any]:
    """Find a team from its full name, city, nickname, or abbreviation."""
    query = team_name.strip().lower()
    if not query:
        raise ValueError("An NBA team name or abbreviation is required.")
    for team in teams.get_teams():
        searchable = {str(team.get(key, "")).lower() for key in
                      ("full_name", "city", "nickname", "abbreviation")}
        if query in searchable:
            return team
    raise ValueError(f"Unknown NBA team: {team_name!r}. Try an abbreviation such as DEN or BOS.")


def _latest_season() -> str:
    """Return the current NBA season label based on today's month."""
    from datetime import date

    today = date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _fetch_team_tables(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download team tables with retry, cache, and neutral fallbacks."""
    neutral = pd.DataFrame([
        {
            "TEAM_ID": team["id"], "TEAM_NAME": team["full_name"],
            "TEAM_ABBREVIATION": team["abbreviation"],
            "GP": 0, "W": 0, "L": 0, "PTS": 110.0, "REB": 44.0, "AST": 25.0,
            "PACE": 100.0, "OFF_RATING": 110.0, "DEF_RATING": 110.0,
            "NET_RATING": 0.0,
        }
        for team in teams.get_teams()
    ])
    common = {
        "season": season,
        "season_type_all_star": "Regular Season",
        "timeout": NBA_API_TIMEOUT,
    }
    base_frames = fetch_nba_frames(
        f"sports_team_base_{season}",
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Base", **common
        ).get_data_frames(),
        fallback_factory=lambda: neutral,
    )
    advanced_frames = fetch_nba_frames(
        f"sports_team_advanced_{season}",
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced", **common
        ).get_data_frames(),
        fallback_factory=lambda: neutral,
    )
    return base_frames[0], advanced_frames[0]


def calculate_efficiency_metrics(row: pd.Series) -> dict[str, float]:
    """Calculate transparent V1 versions of PER, TS%, and usage rate.

    NBA's official PER requires league pace adjustments. V1 uses a common
    box-score efficiency-per-minute estimate and names it ``per_estimate`` so it
    is never confused with the proprietary official calculation.
    """
    points = safe_number(row.get("PTS"))
    fga, fgm = safe_number(row.get("FGA")), safe_number(row.get("FGM"))
    fta, ftm = safe_number(row.get("FTA")), safe_number(row.get("FTM"))
    minutes = safe_number(row.get("MIN"))
    positives = (points + safe_number(row.get("REB")) + safe_number(row.get("AST"))
                 + safe_number(row.get("STL")) + safe_number(row.get("BLK")))
    negatives = (fga - fgm) + (fta - ftm) + safe_number(row.get("TOV"))
    per_estimate = ((positives - negatives) / minutes * 15) if minutes else 0.0

    true_shooting = points / (2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) else 0.0

    # NBA advanced tables provide USG_PCT. The fallback estimates possession
    # involvement from available shooting and turnover box-score events.
    usage = safe_number(row.get("USG_PCT"))
    if not usage:
        usage_events = fga + 0.44 * fta + safe_number(row.get("TOV"))
        usage = min(usage_events / max(minutes, 1), 1.0)
    return {
        "per_estimate": round(per_estimate, 2),
        "true_shooting_pct": round(true_shooting * 100, 2),
        "usage_rate_pct": round(usage * 100, 2),
    }


def _league_rank(table: pd.DataFrame, column: str, value: Any,
                 higher_is_better: bool = True) -> dict[str, int]:
    """Return an NBA rank record with ties sharing the better rank."""
    if column not in table:
        return {"rank":0,"out_of":0}
    values = pd.to_numeric(table[column],errors="coerce").dropna()
    target = safe_number(value,float("nan"))
    if values.empty or pd.isna(target):
        return {"rank":0,"out_of":int(len(values))}
    ahead = (values > target).sum() if higher_is_better else (values < target).sum()
    return {"rank":int(ahead)+1,"out_of":int(len(values))}

def fetch_nba_player_stats(player_name: str, season: str | None = None) -> dict[str, Any]:
    """Return season, recent, total, and advanced statistics for one player.

    The public schema is intentionally presentation-neutral so Streamlit, a
    future API, and projection models can consume exactly the same result.
    """
    season_label = season or _latest_season()
    static_match = fuzzy_name_match(
        player_name, players.get_players(), key=lambda item: item["full_name"], cutoff=.72
    )
    fallback_id = int(static_match["id"]) if static_match else 0
    fallback_name = str(static_match["full_name"]) if static_match else player_name.strip()
    fallback_base = pd.DataFrame([{
        "PLAYER_ID": fallback_id, "PLAYER_NAME": fallback_name,
        "TEAM_ABBREVIATION": "", "GP": 0, "MIN": 30.0,
        "PTS": 15.0, "REB": 5.0, "AST": 4.0, "FGA": 12.0,
        "FGM": 5.5, "FTA": 3.0, "FTM": 2.4, "TOV": 2.0,
        "STL": 0.8, "BLK": 0.5,
    }])
    fallback_advanced = pd.DataFrame([{
        "PLAYER_ID": fallback_id, "USG_PCT": .20, "TS_PCT": .55,
    }])
    common = {
        "season": season_label,
        "season_type_all_star": "Regular Season",
        "timeout": NBA_API_TIMEOUT,
    }
    base_frames = fetch_nba_frames(
        f"sports_player_base_{season_label}",
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            measure_type_detailed_defense="Base", **common
        ).get_data_frames(),
        fallback_factory=lambda: fallback_base,
    )
    advanced_frames = fetch_nba_frames(
        f"sports_player_advanced_{season_label}",
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            measure_type_detailed_defense="Advanced", **common
        ).get_data_frames(),
        fallback_factory=lambda: fallback_advanced,
    )
    base = base_frames[0] if base_frames else fallback_base
    advanced = advanced_frames[0] if advanced_frames else fallback_advanced
    # A disk cache may have been created from a previous player's fallback
    # during a full provider outage. Always add the requested local-roster
    # fallback when that shared league table does not contain the player.
    if fallback_id and (
        "PLAYER_ID" not in base
        or not (
            pd.to_numeric(base["PLAYER_ID"], errors="coerce") == fallback_id
        ).any()
    ):
        base = pd.concat([base, fallback_base], ignore_index=True)
    if fallback_id and (
        "PLAYER_ID" not in advanced
        or not (
            pd.to_numeric(advanced["PLAYER_ID"], errors="coerce") == fallback_id
        ).any()
    ):
        advanced = pd.concat([advanced, fallback_advanced], ignore_index=True)    # Prefer the stable player id resolved from nba_api's local roster. Names
    # may contain accents or provider-specific punctuation that fuzzy text
    # normalization cannot always preserve during an outage fallback.
    matches = pd.DataFrame()
    if fallback_id and "PLAYER_ID" in base:
        matches = base[
            pd.to_numeric(base["PLAYER_ID"], errors="coerce") == fallback_id
        ]
    if matches.empty:
        names = base["PLAYER_NAME"].dropna().astype(str).tolist()
        matched_name = fuzzy_name_match(player_name, names, cutoff=.68)
        if matched_name:
            matches = base[
                base["PLAYER_NAME"].astype(str).str.casefold()
                == str(matched_name).casefold()
            ]
    if matches.empty:
        raise ValueError(f"No player stats found for {player_name!r} in {season_label}.")
    row = matches.iloc[0]
    player_id = int(safe_number(row.get("PLAYER_ID")))
    advanced_row = pd.Series(dtype=object)
    if not advanced.empty and "PLAYER_ID" in advanced:
        advanced_matches = advanced[advanced["PLAYER_ID"] == player_id]
        if not advanced_matches.empty:
            advanced_row = advanced_matches.iloc[0]

    # When game logs are unavailable, synthesize a short, projection-safe
    # history from the player's season averages. This keeps all recent-form
    # views renderable and clearly behaves like a season-average fallback.
    fallback_games = pd.DataFrame([{
        "GAME_DATE": game_date,
        "MIN": row.get("MIN", 30.0), "PTS": row.get("PTS", 15.0),
        "REB": row.get("REB", 5.0), "AST": row.get("AST", 4.0),
        "FGA": row.get("FGA", 12.0), "FGM": row.get("FGM", 5.5),
        "FTA": row.get("FTA", 3.0), "FTM": row.get("FTM", 2.4),
        "TOV": row.get("TOV", 2.0), "STL": row.get("STL", .8),
        "BLK": row.get("BLK", .5),
    } for game_date in pd.date_range(end=pd.Timestamp.now(), periods=10, freq="3D")])
    frames = fetch_nba_frames(
        f"sports_player_log_{player_id}_{season_label}",
        lambda: playergamelog.PlayerGameLog(
            player_id=player_id, season=season_label,
            season_type_all_star="Regular Season",
            timeout=NBA_API_TIMEOUT,
        ).get_data_frames(),
        fallback_factory=lambda: fallback_games,
    )
    games = coerce_numeric(frames[0] if frames else fallback_games)
    if "game_date" in games:
        games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
        games = games.sort_values("game_date", ascending=False)

    def averages(window: int | None = None) -> dict[str, float]:
        sample = games if window is None else games.head(min(window, len(games)))
        fallback = {"pts":row.get("PTS"), "reb":row.get("REB"),
                    "ast":row.get("AST"), "min":row.get("MIN")}
        return {"ppg":round(safe_number(sample.get("pts", pd.Series(dtype=float)).mean(), fallback["pts"]),1),
                "rpg":round(safe_number(sample.get("reb", pd.Series(dtype=float)).mean(), fallback["reb"]),1),
                "apg":round(safe_number(sample.get("ast", pd.Series(dtype=float)).mean(), fallback["ast"]),1),
                "mpg":round(safe_number(sample.get("min", pd.Series(dtype=float)).mean(), fallback["min"]),1)}

    games_played = int(safe_number(row.get("GP"),len(games)))
    def season_total(column: str, average: str) -> float:
        if not games.empty and column in games:
            return round(safe_number(games[column].sum()),1)
        return round(safe_number(row.get(average))*games_played,1)
    totals = {"games_played":games_played,
              "points":season_total("pts","PTS"),
              "rebounds":season_total("reb","REB"),
              "assists":season_total("ast","AST"),
              "minutes":season_total("min","MIN")}
    usage = safe_number(advanced_row.get("USG_PCT"))
    if 0 < usage <= 1: usage *= 100
    attempts = safe_number(row.get("FGA")) + .44*safe_number(row.get("FTA"))
    true_shooting = safe_number(advanced_row.get("TS_PCT"))
    if not true_shooting and attempts:
        true_shooting = safe_number(row.get("PTS"))/(2*attempts)
    if 0 < true_shooting <= 1: true_shooting *= 100
    metrics_row = row.copy()
    metrics_row["USG_PCT"] = usage/100 if usage > 1 else usage
    metrics = calculate_efficiency_metrics(metrics_row)
    usage = usage or metrics["usage_rate_pct"]
    true_shooting = true_shooting or metrics["true_shooting_pct"]

    trend_series = []
    ordered_games = (
        games.sort_values("game_date").tail(20)
        if "game_date" in games else games.tail(20)
    )
    for game_number, (_, game) in enumerate(ordered_games.iterrows(), start=1):
        metric_input = pd.Series({
            "PTS":game.get("pts"), "REB":game.get("reb"), "AST":game.get("ast"),
            "MIN":game.get("min"), "FGA":game.get("fga"), "FGM":game.get("fgm"),
            "FTA":game.get("fta"), "FTM":game.get("ftm"), "TOV":game.get("tov"),
            "STL":game.get("stl"), "BLK":game.get("blk"),
        })
        game_metrics = calculate_efficiency_metrics(metric_input)
        game_date = game.get("game_date")
        trend_series.append({
            "game":game_number,
            "date":game_date.date().isoformat() if pd.notna(game_date) else "",
            "minutes":round(safe_number(game.get("min")),1),
            "usage_pct":game_metrics["usage_rate_pct"],
            "ts_pct":game_metrics["true_shooting_pct"],
            "per_estimate":game_metrics["per_estimate"],
        })

    per_values = base.apply(lambda player_row: calculate_efficiency_metrics(player_row)["per_estimate"],axis=1)
    per_table = pd.DataFrame({"PER_ESTIMATE":per_values})
    league_ranks = {
        "ppg":_league_rank(base,"PTS",row.get("PTS")),
        "rpg":_league_rank(base,"REB",row.get("REB")),
        "apg":_league_rank(base,"AST",row.get("AST")),
        "mpg":_league_rank(base,"MIN",row.get("MIN")),
        "usage_pct":_league_rank(advanced,"USG_PCT",advanced_row.get("USG_PCT")),
        "ts_pct":_league_rank(advanced,"TS_PCT",advanced_row.get("TS_PCT")),
        "per_estimate":_league_rank(per_table,"PER_ESTIMATE",metrics["per_estimate"]),
    }
    return {"player":str(row["PLAYER_NAME"]),
            "team":str(row.get("TEAM_ABBREVIATION","")),
            "season":int(str(season_label).split("-")[0]),
            "season_avg":{**averages(), "games_played":games_played},
            "last5_avg":averages(5), "last10_avg":averages(10),
            "season_totals":totals,
            "advanced":{"usage_pct":round(usage,2), "ts_pct":round(true_shooting,2),
                        "per_estimate":metrics["per_estimate"]},
            "league_ranks":league_ranks,
            "trend_series":trend_series}

def compare_teams(
    team_one: str,
    team_two: str,
    season: str | None = None,
    include_advanced: bool = False,
) -> dict[str, Any]:
    """Compare teams and optionally include the deeper analytics pipeline.

    Advanced mode is opt-in because it needs several additional NBA.com calls.
    """
    season = season or _latest_season()
    first, second = _team_lookup(team_one), _team_lookup(team_two)
    if first["id"] == second["id"]:
        raise ValueError("Please choose two different NBA teams.")

    base, advanced = _fetch_team_tables(season)
    advanced_fields = [column for column in
                       ["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "TS_PCT"]
                       if column in advanced.columns]
    table = base.merge(advanced[advanced_fields], on="TEAM_ID", how="left")

    def team_snapshot(team: dict[str, Any]) -> dict[str, Any]:
        rows = table[table["TEAM_ID"] == team["id"]]
        if rows.empty:
            raise ValueError(f"No {season} statistics found for {team['full_name']}.")
        row = rows.iloc[0]
        return {
            "name": team["full_name"],
            "abbreviation": team["abbreviation"],
            "wins": int(safe_number(row.get("W"))),
            "losses": int(safe_number(row.get("L"))),
            "points_per_game": round(safe_number(row.get("PTS")), 1),
            "rebounds_per_game": round(safe_number(row.get("REB")), 1),
            "assists_per_game": round(safe_number(row.get("AST")), 1),
            "offensive_rating": round(safe_number(row.get("OFF_RATING")), 1),
            "defensive_rating": round(safe_number(row.get("DEF_RATING")), 1),
            "net_rating": round(safe_number(row.get("NET_RATING")), 1),
            "pace": round(safe_number(row.get("PACE")), 1),
            "efficiency": calculate_efficiency_metrics(row),
            "league_ranks": {
                "points_per_game":_league_rank(table,"PTS",row.get("PTS")),
                "rebounds_per_game":_league_rank(table,"REB",row.get("REB")),
                "assists_per_game":_league_rank(table,"AST",row.get("AST")),
                "offensive_rating":_league_rank(table,"OFF_RATING",row.get("OFF_RATING")),
                "defensive_rating":_league_rank(table,"DEF_RATING",row.get("DEF_RATING"),False),
                "net_rating":_league_rank(table,"NET_RATING",row.get("NET_RATING")),
                "pace":_league_rank(table,"PACE",row.get("PACE")),
            },
        }

    first_stats, second_stats = team_snapshot(first), team_snapshot(second)
    edge = first_stats["net_rating"] - second_stats["net_rating"]
    favored = first_stats if edge >= 0 else second_stats
    result = {
        "domain": "sports",
        "subject": f"{first['abbreviation']} vs {second['abbreviation']}",
        "season": season,
        "teams": [first_stats, second_stats],
        "matchup": {
            "favored_team": favored["name"],
            "net_rating_edge": round(abs(edge), 2),
            "description": "The V1 favorite is the team with the higher season net rating.",
        },
    }
    if include_advanced:
        from modules.nba_advanced import analyze_matchup
        result["advanced"] = analyze_matchup(team_one, team_two, season)
    return result