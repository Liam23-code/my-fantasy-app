"""Beginner-friendly NFL team analytics powered by nflverse play-by-play data.

The public functions return plain dictionaries, matching the finance and NBA
modules. Data loading is isolated so a database or prediction model can replace
the provider later without changing the calculations or analyzer.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from modules.utils import clamp, safe_number


# nflverse uses these modern abbreviations. Common historical/alternate forms
# are included as aliases so users do not need to memorize provider conventions.
NFL_TEAMS = {
    "ARI": ("Arizona Cardinals", "Cardinals", ["ARZ"]),
    "ATL": ("Atlanta Falcons", "Falcons", []),
    "BAL": ("Baltimore Ravens", "Ravens", []),
    "BUF": ("Buffalo Bills", "Bills", []),
    "CAR": ("Carolina Panthers", "Panthers", []),
    "CHI": ("Chicago Bears", "Bears", []),
    "CIN": ("Cincinnati Bengals", "Bengals", []),
    "CLE": ("Cleveland Browns", "Browns", []),
    "DAL": ("Dallas Cowboys", "Cowboys", []),
    "DEN": ("Denver Broncos", "Broncos", []),
    "DET": ("Detroit Lions", "Lions", []),
    "GB": ("Green Bay Packers", "Packers", ["GNB"]),
    "HOU": ("Houston Texans", "Texans", []),
    "IND": ("Indianapolis Colts", "Colts", []),
    "JAX": ("Jacksonville Jaguars", "Jaguars", ["JAC"]),
    "KC": ("Kansas City Chiefs", "Chiefs", ["KAN"]),
    "LA": ("Los Angeles Rams", "Rams", ["LAR"]),
    "LAC": ("Los Angeles Chargers", "Chargers", []),
    "LV": ("Las Vegas Raiders", "Raiders", ["OAK"]),
    "MIA": ("Miami Dolphins", "Dolphins", []),
    "MIN": ("Minnesota Vikings", "Vikings", []),
    "NE": ("New England Patriots", "Patriots", ["NWE"]),
    "NO": ("New Orleans Saints", "Saints", ["NOR"]),
    "NYG": ("New York Giants", "Giants", []),
    "NYJ": ("New York Jets", "Jets", []),
    "PHI": ("Philadelphia Eagles", "Eagles", []),
    "PIT": ("Pittsburgh Steelers", "Steelers", []),
    "SEA": ("Seattle Seahawks", "Seahawks", []),
    "SF": ("San Francisco 49ers", "49ers", ["SFO"]),
    "TB": ("Tampa Bay Buccaneers", "Buccaneers", ["TAM", "Bucs"]),
    "TEN": ("Tennessee Titans", "Titans", []),
    "WAS": ("Washington Commanders", "Commanders", ["WSH", "Washington"]),
}


PBP_COLUMNS = [
    "game_id", "season", "season_type", "week", "home_team", "away_team",
    "posteam", "defteam", "play_type", "yards_gained", "epa", "success",
    "turnover", "interception", "fumble_lost", "touchdown", "down",
    "third_down_converted", "third_down_failed", "yardline_100", "drive",
    "game_seconds_remaining", "posteam_score_post",
]


def latest_completed_nfl_season() -> int:
    """Choose the most recent season likely to contain usable game data."""
    today = date.today()
    return today.year if today.month >= 9 else today.year - 1


def lookup_team(team_name: str) -> dict[str, Any]:
    """Resolve an abbreviation, full name, nickname, or supported alias."""
    query = team_name.strip().lower()
    if not query:
        raise ValueError("An NFL team name or abbreviation is required.")
    for abbreviation, (full_name, nickname, aliases) in NFL_TEAMS.items():
        accepted = {abbreviation.lower(), full_name.lower(), nickname.lower(),
                    *(alias.lower() for alias in aliases)}
        if query in accepted:
            return {"abbreviation": abbreviation, "name": full_name,
                    "nickname": nickname}
    raise ValueError(f"Unknown NFL team: {team_name!r}. Try KC, BUF, Cowboys, or Eagles.")


def fetch_play_by_play(season: int) -> pd.DataFrame:
    """Load one season from nflreadpy and return a pandas DataFrame."""
    try:
        import nflreadpy as nfl
    except ImportError as error:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from error

    # nflreadpy returns Polars. Converting here keeps all downstream modules on
    # the project's existing pandas interface.
    try:
        data = nfl.load_pbp([season], columns=PBP_COLUMNS)
    except TypeError:
        # Some nflreadpy releases do not expose column projection in load_pbp.
        data = nfl.load_pbp([season])
    frame = data.to_pandas() if hasattr(data, "to_pandas") else pd.DataFrame(data)
    if frame.empty:
        raise ValueError(f"No NFL play-by-play data was returned for {season}.")
    available = [column for column in PBP_COLUMNS if column in frame.columns]
    return frame[available].copy()


def _scrimmage_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-season run/pass plays with identified teams."""
    plays = pbp.copy()
    if "season_type" in plays:
        plays = plays[plays["season_type"] == "REG"]
    return plays[
        plays["play_type"].isin(["run", "pass"])
        & plays["posteam"].notna()
        & plays["defteam"].notna()
    ].copy()


def _percentile(series: pd.Series, value: float, higher_is_better: bool = True) -> float:
    """Convert a metric to an easy-to-combine 0-100 league percentile."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 50.0
    rank = safe_number((numeric <= value).mean()) * 100
    return rank if higher_is_better else 100 - rank


def _seconds_per_play(team_plays: pd.DataFrame) -> float:
    """Estimate pace from elapsed game-clock time between offensive snaps."""
    if "game_seconds_remaining" not in team_plays:
        return 0.0
    ordered = team_plays.sort_values(["game_id", "game_seconds_remaining"],
                                    ascending=[True, False]).copy()
    elapsed = -ordered.groupby("game_id")["game_seconds_remaining"].diff()
    # Long gaps are quarter breaks, halftime, timeouts, or data artifacts.
    valid = elapsed[(elapsed >= 0) & (elapsed <= 60)]
    return safe_number(valid.mean())


def _red_zone_efficiency(team_plays: pd.DataFrame) -> float:
    """Calculate touchdowns per drive that reached the opponent's 20."""
    if not {"yardline_100", "drive", "touchdown"}.issubset(team_plays.columns):
        return 0.0
    red_zone = team_plays[team_plays["yardline_100"] <= 20]
    if red_zone.empty:
        return 0.0
    drives = red_zone.groupby(["game_id", "drive"])["touchdown"].max()
    return safe_number(drives.mean()) * 100


def _third_down_rate(team_plays: pd.DataFrame) -> float:
    """Calculate successful third downs divided by third-down attempts."""
    third = team_plays[team_plays["down"] == 3] if "down" in team_plays else pd.DataFrame()
    if third.empty:
        return 0.0
    converted = pd.to_numeric(third.get("third_down_converted", 0), errors="coerce").fillna(0)
    failed = pd.to_numeric(third.get("third_down_failed", 0), errors="coerce").fillna(0)
    attempts = (converted + failed) > 0
    return safe_number(converted[attempts].mean()) * 100 if attempts.any() else 0.0


def build_team_season_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-by-play into one clean row per NFL team."""
    plays = _scrimmage_plays(pbp)
    teams_in_data = sorted(set(plays["posteam"].dropna()) | set(plays["defteam"].dropna()))
    rows: list[dict[str, Any]] = []
    for team in teams_in_data:
        offense, defense = plays[plays["posteam"] == team], plays[plays["defteam"] == team]
        game_ids = set(offense["game_id"].dropna()) | set(defense["game_id"].dropna())
        games = max(len(game_ids), 1)

        # The final posteam score observed in each game includes offensive,
        # defensive, and special-teams scoring credited to that team.
        points = (offense.groupby("game_id")["posteam_score_post"].max().sum()
                  if "posteam_score_post" in offense else 0)
        giveaways = safe_number(offense.get("turnover", pd.Series(dtype=float)).sum())
        takeaways = safe_number(defense.get("turnover", pd.Series(dtype=float)).sum())
        rows.append({
            "team": team,
            "games": games,
            "points_per_game": safe_number(points) / games,
            "offensive_yards_per_game": safe_number(offense["yards_gained"].sum()) / games,
            "defensive_yards_per_game": safe_number(defense["yards_gained"].sum()) / games,
            "offensive_epa_per_play": safe_number(offense["epa"].mean()),
            "defensive_epa_per_play": safe_number(defense["epa"].mean()),
            "offensive_success_rate": safe_number(offense["success"].mean()) * 100,
            "defensive_success_rate": safe_number(defense["success"].mean()) * 100,
            "turnover_margin": takeaways - giveaways,
            "pace_seconds_per_play": _seconds_per_play(offense),
            "red_zone_efficiency": _red_zone_efficiency(offense),
            "third_down_conversion_rate": _third_down_rate(offense),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("The dataset contained no regular-season run or pass plays.")

    # Percentile blends make the 0-100 efficiency scores transparent. Defensive
    # EPA, success, and yards are inverted because lower values are better.
    for index, row in table.iterrows():
        offense_score = (
            _percentile(table["offensive_epa_per_play"], row["offensive_epa_per_play"]) * 0.40
            + _percentile(table["offensive_success_rate"], row["offensive_success_rate"]) * 0.25
            + _percentile(table["offensive_yards_per_game"], row["offensive_yards_per_game"]) * 0.20
            + _percentile(table["points_per_game"], row["points_per_game"]) * 0.15
        )
        defense_score = (
            _percentile(table["defensive_epa_per_play"], row["defensive_epa_per_play"], False) * 0.45
            + _percentile(table["defensive_success_rate"], row["defensive_success_rate"], False) * 0.30
            + _percentile(table["defensive_yards_per_game"], row["defensive_yards_per_game"], False) * 0.25
        )
        table.loc[index, "offensive_efficiency_score"] = offense_score
        table.loc[index, "defensive_efficiency_score"] = defense_score
        table.loc[index, "net_efficiency"] = offense_score + defense_score - 100

    # SOS is the average net efficiency of opponents actually played.
    schedule = pbp[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")
    net_by_team = table.set_index("team")["net_efficiency"].to_dict()
    for index, row in table.iterrows():
        team = row["team"]
        opponents = schedule.loc[schedule["home_team"] == team, "away_team"].tolist()
        opponents += schedule.loc[schedule["away_team"] == team, "home_team"].tolist()
        values = [net_by_team[opponent] for opponent in opponents if opponent in net_by_team]
        table.loc[index, "strength_of_schedule"] = safe_number(pd.Series(values).mean())
    return table.round(3)


def get_team_stats(team_name: str, season: int | None = None,
                   team_table: pd.DataFrame | None = None) -> dict[str, Any]:
    """Return a clean team dictionary, optionally using an existing table."""
    team = lookup_team(team_name)
    season = season or latest_completed_nfl_season()
    table = team_table if team_table is not None else build_team_season_stats(fetch_play_by_play(season))
    rows = table[table["team"] == team["abbreviation"]]
    if rows.empty:
        raise ValueError(f"No {season} data found for {team['name']}.")
    metrics = {key: round(safe_number(value), 2)
               for key, value in rows.iloc[0].items() if key != "team"}
    return {**team, "season": season, "metrics": metrics}


def compare_nfl_teams(team_one: str, team_two: str,
                      season: int | None = None) -> dict[str, Any]:
    """Compare two NFL teams with explainable matchup rules."""
    season = season or latest_completed_nfl_season()
    first_lookup, second_lookup = lookup_team(team_one), lookup_team(team_two)
    if first_lookup["abbreviation"] == second_lookup["abbreviation"]:
        raise ValueError("Please choose two different NFL teams.")
    table = build_team_season_stats(fetch_play_by_play(season))
    first = get_team_stats(team_one, season, table)
    second = get_team_stats(team_two, season, table)
    one, two = first["metrics"], second["metrics"]

    comparison_fields = [
        ("EPA per play", "offensive_epa_per_play", True),
        ("success rate", "offensive_success_rate", True),
        ("net efficiency", "net_efficiency", True),
        ("turnover margin", "turnover_margin", True),
        ("strength of schedule", "strength_of_schedule", True),
    ]
    mismatches = []
    advantages = {first["abbreviation"]: 0.0, second["abbreviation"]: 0.0}
    for label, key, higher_is_better in comparison_fields:
        difference = safe_number(one.get(key)) - safe_number(two.get(key))
        if not higher_is_better:
            difference *= -1
        winner = first if difference >= 0 else second
        advantages[winner["abbreviation"]] += abs(difference)
        if abs(difference) >= (0.03 if "EPA" in label else 2.0):
            mismatches.append({"metric": label, "advantage": winner["name"],
                               "difference": round(abs(difference), 2)})

    net_edge = one["net_efficiency"] - two["net_efficiency"]
    favored = first if net_edge >= 0 else second
    # Close efficiency profiles are harder to separate. Contradictory metric
    # advantages add uncertainty without pretending this is a win probability.
    split_advantages = all(value > 0 for value in advantages.values())
    difficulty = clamp(100 - min(abs(net_edge), 40) * 1.8 + (10 if split_advantages else 0))
    return {
        "domain": "nfl",
        "subject": f"{first['abbreviation']} vs {second['abbreviation']}",
        "season": season,
        "teams": [first, second],
        "matchup": {
            "favored_team": favored["name"],
            "net_efficiency_edge": round(abs(net_edge), 2),
            "difficulty_score": round(difficulty, 1),
            "key_mismatches": sorted(mismatches, key=lambda item: item["difference"], reverse=True),
            "explanation": "The V1 NFL lean uses team efficiency; difficulty rises when profiles are close or metrics disagree.",
        },
        "future_model_features": [
            "epa_per_play", "success_rate", "turnover_margin", "schedule_strength",
            "pace", "red_zone_efficiency", "third_down_conversion_rate",
        ],
    }
