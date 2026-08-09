"""Robust, explainable NBA projections built from public nba_api data."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
    playergamelog,
)
from nba_api.stats.static import players, teams
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from modules.nba_advanced import latest_season
from modules.nba_cache import (
    FALLBACK_WARNING,
    ENHANCED_FALLBACK_WARNING,
    NBA_API_TIMEOUT,
    collect_warnings,
    fetch_nba_frames,
    used_fallback,
    select_fallback_minutes,
    weighted_available_average,
)
from modules.data_quality import (
    coerce_numeric, data_completeness, defense_fallback, fuzzy_name_match,
    normalize_columns as quality_normalize_columns, normalize_text,
    pace_fallback, parse_minutes, rolling_average, safe_number,
)

TARGETS = {"points": "pts", "rebounds": "reb", "assists": "ast", "pra": "pra"}
FEATURE_COLUMNS = [
    "minutes", "usage_rate", "true_shooting_pct", "pace",
    "opponent_defensive_rating", "team_offensive_rating", "home",
    "back_to_back", "availability", "points_last_5", "points_last_10",
    "rebounds_last_5", "rebounds_last_10", "assists_last_5", "assists_last_10",
]
VARIANTS = {
    "playerid": "player_id", "gameid": "game_id", "game_date_est": "game_date",
    "date": "game_date", "minutes": "min", "points": "pts",
    "rebounds": "reb", "assists": "ast", "teamid": "team_id",
    "team": "team_abbreviation", "team_abbr": "team_abbreviation",
    "offensive_rating": "off_rating", "offrtg": "off_rating", "ortg": "off_rating",
    "defensive_rating": "def_rating", "defrtg": "def_rating", "drtg": "def_rating",
    "usage_pct": "usg_pct", "usage_rate": "usg_pct",
    "true_shooting_pct": "ts_pct", "true_shooting_percentage": "ts_pct",
    "opponent": "opponent_abbreviation", "opp": "opponent_abbreviation",
    "opponent_team": "opponent_abbreviation", "opp_team": "opponent_abbreviation",
    "opponent_team_abbreviation": "opponent_abbreviation",
    "is_available": "available", "active": "available",
}
NUMERIC = ["team_id", "min", "pts", "reb", "ast", "fga", "fgm", "fta",
           "tov", "usg_pct", "ts_pct", "pace", "off_rating", "def_rating"]


def _text(value: Any) -> str:
    """Compatibility wrapper around the unified text normalizer."""
    return normalize_text(value)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize provider fields through the shared data-quality layer."""
    return quality_normalize_columns(frame, VARIANTS)


def _minutes(value: Any) -> float:
    """Compatibility wrapper for the centralized minutes parser."""
    return parse_minutes(value)


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and coerce projection inputs through one shared policy."""
    return coerce_numeric(normalize_columns(frame), NUMERIC)

def _team_aliases(team: dict[str, Any]) -> set[str]:
    return {_text(team.get(key, "")) for key in
            ("full_name", "nickname", "city", "abbreviation") if team.get(key)}


def _team_maps() -> tuple[dict[int, str], dict[str, str]]:
    by_id, aliases = {}, {}
    for team in teams.get_teams():
        abbreviation = team["abbreviation"].upper()
        by_id[int(team["id"])] = abbreviation
        aliases.update({alias: abbreviation for alias in _team_aliases(team)})
    aliases.update({"la lakers": "LAL", "la clippers": "LAC", "brk": "BKN",
                    "pho": "PHX", "no": "NOP", "cho": "CHA"})
    return by_id, aliases


def _abbr(value: Any) -> str:
    """Normalize team names and provider abbreviations."""
    _, aliases = _team_maps()
    return aliases.get(_text(value), str(value).strip().upper())


def _find_team(name: str) -> dict[str, Any]:
    query = _text(name)
    for team in teams.get_teams():
        if query in _team_aliases(team) or _abbr(query) == team["abbreviation"]:
            return team
    raise ValueError(f"Unknown NBA opponent: {name!r}.")


def _find_player(name: str) -> dict[str, Any]:
    """Resolve exact, partial, and fuzzy names through the quality layer."""
    match = fuzzy_name_match(name, players.get_players(),
                             key=lambda player: player["full_name"], cutoff=.72)
    if match:
        return match
    raise ValueError(f"Could not identify NBA player {name!r}.")

def fetch_player_game_log(player_id: int, season: str) -> pd.DataFrame:
    """Fetch one game log through the shared retry and cache policy."""
    frames = fetch_nba_frames(
        f"player_game_log_{player_id}_{season}",
        lambda: playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
            timeout=NBA_API_TIMEOUT,
        ).get_data_frames(),
    )
    return frames[0] if frames else pd.DataFrame()


def _default_player_row(player: dict[str, Any]) -> dict[str, Any]:
    """Return conservative league-average inputs when no season row exists."""
    return {
        "PLAYER_ID": int(player["id"]),
        "PLAYER_NAME": player["full_name"],
        "TEAM_ID": 0,
        "TEAM_ABBREVIATION": "",
        "GP": 1,
        "MIN": 28.0,
        "PTS": 15.0,
        "REB": 5.0,
        "AST": 4.0,
        "FGA": 12.0,
        "FGM": 5.5,
        "FTA": 3.5,
        "FTM": 2.8,
        "TOV": 2.0,
        "STL": 0.8,
        "BLK": 0.5,
        "FG3M": 1.2,
        "FG3A": 3.5,
    }


def season_average_game_log(
    player: dict[str, Any], season: str, games: int = 12
) -> pd.DataFrame:
    """Build a projection-safe history from cached or live season averages."""
    frames = fetch_nba_frames(
        f"player_season_average_{player['id']}_{season}",
        lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
            timeout=NBA_API_TIMEOUT,
        ).get_data_frames(),
        fallback_factory=lambda: pd.DataFrame([_default_player_row(player)]),
    )
    table = frames[0] if frames else pd.DataFrame([_default_player_row(player)])
    rows = (
        table[table["PLAYER_ID"] == int(player["id"])]
        if "PLAYER_ID" in table else pd.DataFrame()
    )
    source = rows.iloc[0].to_dict() if not rows.empty else _default_player_row(player)
    count = max(5, int(games))
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=count, freq="3D")
    records = []
    for index, game_date in enumerate(dates):
        record = {
            "GAME_DATE": game_date,
            "MATCHUP": (
                f"{source.get('TEAM_ABBREVIATION', '')} vs. LAL"
                if index % 2 == 0
                else f"{source.get('TEAM_ABBREVIATION', '')} @ BOS"
            ),
            "TEAM_ID": source.get("TEAM_ID", 0),
            "TEAM_ABBREVIATION": source.get("TEAM_ABBREVIATION", ""),
        }
        for column, default in _default_player_row(player).items():
            if column in {"PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP"}:
                continue
            record[column] = safe_number(source.get(column), safe_number(default))
        records.append(record)
    fallback = pd.DataFrame(records)
    fallback.attrs["data_source"] = table.attrs.get(
        "data_source", "season_average_fallback"
    )
    fallback.attrs["warnings"] = list(
        dict.fromkeys(collect_warnings(table) + [FALLBACK_WARNING])
    )
    return fallback


def _game_log_with_fuzzy_fallback(
    name: str,
    player: dict[str, Any],
    season: str,
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    """Try live/cached logs, then guarantee a season-average fallback."""
    pool = sorted(
        players.get_players(),
        key=lambda candidate: SequenceMatcher(
            None, _text(name), _text(candidate["full_name"])
        ).ratio(),
        reverse=True,
    )[:3]
    warnings: list[str] = []
    tried: set[int] = set()
    for candidate in [player, *pool]:
        if int(candidate["id"]) in tried:
            continue
        tried.add(int(candidate["id"]))
        frame = fetch_player_game_log(int(candidate["id"]), season)
        if not frame.empty:
            warnings.extend(collect_warnings(frame))
            if candidate["id"] != player["id"]:
                warnings.append(
                    f"Used fuzzy player match: {candidate['full_name']}."
                )
            return candidate, frame, list(dict.fromkeys(warnings))
        # A fallback-marked empty result means the provider is unavailable.
        # Do not multiply a known outage across several fuzzy candidates.
        if used_fallback(frame):
            warnings.extend(collect_warnings(frame))
            break

    fallback = season_average_game_log(player, season)
    warnings.extend(collect_warnings(fallback))
    warnings.append(FALLBACK_WARNING)
    return player, fallback, list(dict.fromkeys(warnings))


def fetch_team_context(season: str) -> pd.DataFrame:
    """Fetch ratings with retries, cache, and neutral league fallback rows."""
    neutral = pd.DataFrame(
        [
            {
                "TEAM_ID": team["id"],
                "TEAM_ABBREVIATION": team["abbreviation"],
                "PACE": 100.0,
                "OFF_RATING": 110.0,
                "DEF_RATING": 110.0,
            }
            for team in teams.get_teams()
        ]
    )
    frames = fetch_nba_frames(
        f"team_context_{season}",
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=NBA_API_TIMEOUT,
        ).get_data_frames(),
        fallback_factory=lambda: neutral,
    )
    frame = frames[0] if frames else neutral
    result = _numeric(frame)
    result.attrs["data_source"] = frame.attrs.get("data_source", "live")
    result.attrs["warnings"] = collect_warnings(frame)
    return result


def _opponents(data: pd.DataFrame) -> pd.Series:
    """Prefer opponent columns, otherwise parse the end of MATCHUP."""
    if "opponent_abbreviation" in data:
        result = data["opponent_abbreviation"].astype(str)
    elif "matchup" in data:
        result = data["matchup"].astype(str).str.extract(
            r"(?:vs\.?|@)\s*([A-Za-z]{2,4})\s*$", expand=False)
        result = result.fillna(data["matchup"].astype(str).str.split().str[-1])
    else:
        result = pd.Series("", index=data.index)
    return result.map(_abbr)


def _player_team(data: pd.DataFrame) -> str:
    """Resolve team abbreviation, falling back to TEAM_ID then MATCHUP."""
    if "team_abbreviation" in data and data["team_abbreviation"].notna().any():
        return _abbr(data["team_abbreviation"].dropna().iloc[-1])
    by_id, _ = _team_maps()
    if "team_id" in data and data["team_id"].notna().any():
        resolved = by_id.get(int(data["team_id"].dropna().iloc[-1]))
        if resolved:
            return resolved
    if "matchup" in data and data["matchup"].notna().any():
        return _abbr(str(data["matchup"].dropna().iloc[-1]).split()[0])
    return ""


def _previous(series: pd.Series, window: int, fallback: float) -> pd.Series:
    """Use centralized rolling-window behavior with no future leakage."""
    return rolling_average(series, window, fallback, prior_only=True)


def _sports_defense(team_id: int, season: str) -> float:
    """Try sports.py before ultimately using the league average."""
    try:
        from modules.sports import _fetch_team_tables
        _, advanced = _fetch_team_tables(season)
        advanced = _numeric(advanced)
        rows = advanced[advanced["team_id"] == team_id]
        return safe_number(rows.iloc[0].get("def_rating")) if not rows.empty else 0.
    except Exception:
        return 0.


def build_projection_features(game_log: pd.DataFrame, team_context: pd.DataFrame,
                              season: str | None = None) -> pd.DataFrame:
    """Normalize schemas and create safe, leakage-resistant pregame features."""
    if game_log is None or game_log.empty:
        raise ValueError("Cannot build features from an empty game log.")
    warnings = []
    game_source = str(game_log.attrs.get("data_source", "live"))
    data, context = _numeric(game_log), _numeric(team_context)
    if "game_date" not in data:
        data["game_date"] = pd.date_range("2000-01-01", periods=len(data))
        warnings.append("Game dates missing; row order used for rest flags.")
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    fallback_dates = pd.Series(pd.date_range("2000-01-01", periods=len(data)), index=data.index)
    data["game_date"] = data["game_date"].fillna(fallback_dates)
    data = data.sort_values("game_date").reset_index(drop=True)
    for column in ("pts", "reb", "ast"):
        data[column] = pd.to_numeric(data.get(column, pd.Series(0, index=data.index)),
                                     errors="coerce").fillna(0)
    data["pra"] = data["pts"] + data["reb"] + data["ast"]

    if "min" not in data:
        data["min"] = np.nan
        warnings.append("Minutes missing; cached window averages required.")
    valid_min = data["min"].where(data["min"] > 0)
    season_minutes = safe_number(valid_min.mean(), np.nan)
    last_10_minutes = safe_number(valid_min.tail(10).mean(), np.nan)
    last_5_minutes = safe_number(valid_min.tail(5).mean(), np.nan)
    min_fallback, minutes_source = select_fallback_minutes(
        season_minutes, last_10_minutes, last_5_minutes
    )
    rolling_minutes = valid_min.shift().rolling(5, min_periods=1).mean()
    data["min"] = valid_min.fillna(rolling_minutes).fillna(min_fallback)
    data["minutes"] = _previous(data["min"], 5, min_fallback)

    if "usg_pct" in data and data["usg_pct"].notna().any():
        usage_source = "provider"
        usage = data["usg_pct"].where(data["usg_pct"] > 1, data["usg_pct"] * 100)
    elif {"fga", "fta", "tov"}.issubset(data.columns):
        usage_source = "box_score_estimate"
        usage = (data["fga"].fillna(0) + .44 * data["fta"].fillna(0) +
                 data["tov"].fillna(0)) / data["min"].replace(0, np.nan) * 40
    else:
        usage_source = "scoring_rate_estimate"
        usage = data["pts"] / data["min"] * 40
        warnings.append("Usage inputs missing; scoring-rate estimate used.")
    usage = usage.replace([np.inf, -np.inf], np.nan)
    usage_fallback = safe_number(usage.mean(), 20.0)
    usage = usage.fillna(
        usage.shift().rolling(5, min_periods=1).mean()
    ).fillna(usage_fallback)
    data["usage_rate"] = _previous(usage, 5, usage_fallback)

    if "ts_pct" in data and data["ts_pct"].notna().any():
        efficiency_source = "provider"
        shooting = data["ts_pct"].where(data["ts_pct"] <= 1, data["ts_pct"] / 100)
    elif {"fga", "fta", "pts"}.issubset(data.columns):
        efficiency_source = "box_score_estimate"
        denominator = 2 * (data["fga"].fillna(0) + .44 * data["fta"].fillna(0))
        shooting = data["pts"] / denominator.replace(0, np.nan)
    else:
        efficiency_source = "recent_scoring_estimate"
        shooting = (data["pts"] / data["min"]).replace([np.inf, -np.inf], np.nan)
        shooting = shooting / safe_number(shooting.mean(), 1.0) * .55
        warnings.append("Shooting inputs missing; recent scoring estimate used.")
    shooting = shooting.replace([np.inf, -np.inf], np.nan)
    shooting_fallback = safe_number(shooting.mean(), .55)
    shooting = shooting.fillna(
        shooting.shift().rolling(10, min_periods=1).mean()
    ).fillna(shooting_fallback)
    data["true_shooting_pct"] = _previous(
        shooting, 10, shooting_fallback
    ) * 100

    for label, source in (("points", "pts"), ("rebounds", "reb"), ("assists", "ast")):
        data[f"{label}_last_5"] = _previous(data[source], 5, safe_number(data[source].mean()))
        data[f"{label}_last_10"] = _previous(data[source], 10, safe_number(data[source].mean()))
    matchup = data.get("matchup", pd.Series("", index=data.index)).astype(str)
    data["home"] = matchup.str.contains(r"\bvs\.?\b", case=False, regex=True).astype(int)
    data["back_to_back"] = (data["game_date"].diff().dt.days == 1).astype(int)
    availability = next((c for c in ("available", "availability", "active_status") if c in data), None)
    if availability:
        data["availability"] = pd.to_numeric(data[availability], errors="coerce").fillna(1).clip(0, 1)
    else:
        data["availability"] = 1.
        warnings.append("Live availability missing; played games treated as available.")
    data["opponent_abbreviation"] = _opponents(data)

    for column, default in (("pace", 100.), ("off_rating", 110.), ("def_rating", 110.)):
        if column not in context:
            context[column] = default
        context[column] = pd.to_numeric(context[column], errors="coerce")
        context[column] = context[column].fillna(safe_number(context[column].median(), default))
    if "team_abbreviation" not in context:
        by_id, _ = _team_maps()
        context["team_abbreviation"] = context.get("team_id", pd.Series(index=context.index)).map(by_id)
    context["team_abbreviation"] = context["team_abbreviation"].map(_abbr)
    indexed = context.drop_duplicates("team_abbreviation").set_index("team_abbreviation")
    league_pace = safe_number(context["pace"].mean(), 100.)
    league_offense = safe_number(context["off_rating"].mean(), 110.)
    league_defense = safe_number(context["def_rating"].mean(), 110.)
    team = _player_team(data)
    team_row = indexed.loc[team] if team in indexed.index else pd.Series()
    data["pace"] = pace_fallback(team_row.get("pace"), league_pace)
    data["team_offensive_rating"] = safe_number(team_row.get("off_rating"), league_offense)
    data["opponent_defensive_rating"] = data["opponent_abbreviation"].map(
        indexed["def_rating"].to_dict()).map(lambda value: defense_fallback(value, league_defense))

    for column in FEATURE_COLUMNS + list(TARGETS.values()):
        data[column] = pd.to_numeric(data.get(column, 0), errors="coerce")
        data[column] = data[column].fillna(safe_number(data[column].median()))
    data.attrs["warnings"] = [*team_context.attrs.get("warnings", []), *warnings]
    data.attrs["team_abbreviation"] = team
    data.attrs["data_source"] = game_source
    data.attrs["minutes_source"] = minutes_source
    data.attrs["usage_source"] = usage_source
    data.attrs["efficiency_source"] = efficiency_source
    result = data.reset_index(drop=True)
    result.attrs.update(data.attrs)
    return result


def chronological_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(data) < 20:
        raise ValueError("Use the low-data fallback for fewer than 20 games.")
    train_end = max(int(len(data) * .70), 1)
    validation_end = min(max(int(len(data) * .85), train_end + 1), len(data) - 1)
    return data.iloc[:train_end], data.iloc[train_end:validation_end], data.iloc[validation_end:]


def _opponent_context(opponent: dict[str, Any], context: pd.DataFrame,
                      season: str) -> tuple[pd.Series, list[str]]:
    normalized = _numeric(context)
    rows = pd.DataFrame()
    if "team_id" in normalized:
        rows = normalized[normalized["team_id"] == opponent["id"]]
    if rows.empty and "team_abbreviation" in normalized:
        rows = normalized[normalized["team_abbreviation"].map(_abbr) == opponent["abbreviation"]]
    if not rows.empty:
        return rows.iloc[0], []
    defense = _sports_defense(opponent["id"], season)
    return pd.Series({
        "def_rating": defense or safe_number(normalized.get("def_rating", pd.Series([110])).mean(), 110),
        "pace": safe_number(normalized.get("pace", pd.Series([100])).mean(), 100),
    }), ["Opponent context incomplete; sports.py or league averages used."]


def _upcoming(data: pd.DataFrame, opponent: dict[str, Any], context: pd.DataFrame,
              season: str) -> tuple[pd.DataFrame, list[str]]:
    latest = data.iloc[-1].copy()
    opponent_row, warnings = _opponent_context(opponent, context, season)
    normalized = _numeric(context)
    defense = safe_number(opponent_row.get("def_rating"), safe_number(
        normalized.get("def_rating", pd.Series([110])).mean(), 110))
    pace = safe_number(opponent_row.get("pace"), safe_number(
        normalized.get("pace", pd.Series([100])).mean(), 100))
    latest["opponent_defensive_rating"] = defense
    latest["pace"] = (safe_number(latest["pace"], 100) + pace) / 2
    latest["home"], latest["back_to_back"], latest["availability"] = .5, 0, 1
    frame = pd.DataFrame([latest[FEATURE_COLUMNS]])
    return frame.apply(pd.to_numeric, errors="coerce").fillna(0), warnings


def _usable_features(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    usable, dropped = [], []
    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        (usable if values.nunique(dropna=True) > 1 else dropped).append(column)
    return usable, dropped


def _recent(data: pd.DataFrame, window: int = 10) -> dict[str, float]:
    return {name: round(safe_number(data[target].tail(min(window, len(data))).mean()), 1)
            for name, target in TARGETS.items()}


def _window_profile(data: pd.DataFrame, window: int | None = None) -> dict[str, float | None]:
    """Summarize production and rate inputs for one historical window."""
    sample = data if window is None else data.tail(min(window, len(data)))

    def average(column: str) -> float | None:
        if column not in sample:
            return None
        value = pd.to_numeric(sample[column], errors="coerce").mean()
        return round(float(value), 3) if pd.notna(value) else None

    return {
        "points": average("pts"),
        "rebounds": average("reb"),
        "assists": average("ast"),
        "pra": average("pra"),
        "minutes": average("min"),
        "usage": average("usage_rate"),
        "true_shooting_pct": average("true_shooting_pct"),
    }


def _feature_summary(data: pd.DataFrame, warnings: list[str]) -> dict[str, Any]:
    """Expose the exact season/recent inputs consumed by fallback fusion."""
    season = _window_profile(data)
    last_10 = _window_profile(data, 10)
    last_5 = _window_profile(data, 5)
    usage_blend, usage_coverage = weighted_available_average([
        (season.get("usage"), .50),
        (last_10.get("usage"), .30),
        (last_5.get("usage"), .20),
    ])
    efficiency_blend, efficiency_coverage = weighted_available_average([
        (season.get("true_shooting_pct"), .50),
        (last_10.get("true_shooting_pct"), .30),
        (last_5.get("true_shooting_pct"), .20),
    ])
    fallback_minutes, minutes_source = select_fallback_minutes(
        season.get("minutes"), last_10.get("minutes"), last_5.get("minutes")
    )
    data_source = str(data.attrs.get("data_source", "live"))
    fallback_active = FALLBACK_WARNING in warnings or data_source != "live"
    usage_source = str(data.attrs.get("usage_source", "unknown"))
    efficiency_source = str(data.attrs.get("efficiency_source", "unknown"))
    usage_source_quality = {
        "provider": 100.0,
        "box_score_estimate": 90.0,
        "scoring_rate_estimate": 55.0,
    }.get(usage_source, 50.0)
    efficiency_source_quality = {
        "provider": 100.0,
        "box_score_estimate": 92.0,
        "recent_scoring_estimate": 55.0,
    }.get(efficiency_source, 50.0)
    minutes_quality = {
        "season_average": 92.0,
        "last_10": 76.0,
        "last_5": 62.0,
    }[minutes_source]
    return {
        "season_average": season,
        "last_10": last_10,
        "last_5": last_5,
        "minutes": {
            "selected": round(fallback_minutes, 3),
            "source": minutes_source,
            "quality": minutes_quality if fallback_active else 100.0,
        },
        "usage": {
            "season": season.get("usage"),
            "last_10": last_10.get("usage"),
            "last_5": last_5.get("usage"),
            "blended": round(usage_blend, 3),
            "source": usage_source,
            "quality": round(usage_coverage * usage_source_quality, 1),
        },
        "efficiency": {
            "season": season.get("true_shooting_pct"),
            "last_10": last_10.get("true_shooting_pct"),
            "last_5": last_5.get("true_shooting_pct"),
            "blended": round(efficiency_blend, 3),
            "source": efficiency_source,
            "quality": round(
                efficiency_coverage * efficiency_source_quality, 1
            ),
        },
        "fallback_status": {
            "active": fallback_active,
            "source": data_source,
            "warning": ENHANCED_FALLBACK_WARNING if fallback_active else "",
        },
    }

def _rolling_model(data: pd.DataFrame) -> tuple[dict, dict, dict]:
    projections, ranges, quality = {}, {}, {}
    for name, target in TARGETS.items():
        five = safe_number(data[target].tail(min(5, len(data))).mean())
        ten = safe_number(data[target].tail(min(10, len(data))).mean(), five)
        prediction = .6 * five + .4 * ten
        observed_spread = data[target].tail(10).std()
        uncertainty = (
            safe_number(observed_spread)
            if pd.notna(observed_spread)
            else abs(prediction) * .15
        )
        projections[name] = round(prediction, 1)
        ranges[name] = {"low": round(prediction - uncertainty, 1),
                        "high": round(prediction + uncertainty, 1)}
        quality[name] = {"validation_mae": 0., "test_mae": 0.}
    return projections, ranges, quality


def _drivers(row: pd.Series, history: pd.DataFrame, context: pd.DataFrame) -> list[str]:
    context = _numeric(context)
    result = []
    if row["usage_rate"] > safe_number(history["usage_rate"].median()):
        result.append("High recent usage increases scoring opportunity.")
    if row["opponent_defensive_rating"] > safe_number(context.get("def_rating", pd.Series([110])).median(), 110):
        result.append("Opponent defense rates weaker than the league midpoint.")
    if row["pace"] > safe_number(context.get("pace", pd.Series([100])).median(), 100):
        result.append("Fast expected pace creates more possessions.")
    if row["points_last_5"] > row["points_last_10"]:
        result.append("Last-five scoring form is above the ten-game baseline.")
    return result or ["Recent role and matchup context are close to normal levels."]


def project_player_statline(player_name: str, opponent_team: str,
                            season: str | None = None) -> dict[str, Any]:
    """Return the stable projection dictionary, regardless of provider gaps."""
    season = season or latest_season()
    primary, opponent = _find_player(player_name), _find_team(opponent_team)
    player, games, warnings = _game_log_with_fuzzy_fallback(player_name, primary, season)
    context = fetch_team_context(season)
    features = build_projection_features(games, context, season)
    warnings.extend(features.attrs.get("warnings", []))
    feature_summary = _feature_summary(features, warnings)
    fallback_status = feature_summary["fallback_status"]
    if fallback_status["active"]:
        warnings.append(ENHANCED_FALLBACK_WARNING)
    upcoming, extra = _upcoming(features, opponent, context, season)
    warnings.extend(extra)
    usable, dropped = _usable_features(features)
    if dropped:
        warnings.append("Zero-variance features omitted: " + ", ".join(dropped) + ".")

    fallback = len(features) < 20 or len(usable) < 2
    if fallback:
        projection, ranges, quality = _rolling_model(features)
        counts, model = (len(features), 0, 0), "RollingAverageFallback"
        warnings.append("Low or constant data: rolling averages used instead of ML.")
    else:
        train, validation, test = chronological_split(features)
        projection, ranges, quality = {}, {}, {}
        for name, target in TARGETS.items():
            evaluation = LinearRegression().fit(train[usable], train[target])
            val_prediction = evaluation.predict(validation[usable])
            test_prediction = evaluation.predict(test[usable])
            val_mae = mean_absolute_error(validation[target], val_prediction)
            test_mae = mean_absolute_error(test[target], test_prediction)
            quality[name] = {"validation_mae": round(val_mae, 2), "test_mae": round(test_mae, 2)}
            final = LinearRegression().fit(features[usable], features[target])
            value = safe_number(final.predict(upcoming[usable])[0])
            uncertainty = max(safe_number(np.std(test[target].to_numpy() - test_prediction)) * 1.645,
                              test_mae)
            projection[name] = round(value, 1)
            ranges[name] = {"low": round(value - uncertainty, 1),
                            "high": round(value + uncertainty, 1)}
        counts, model = (len(train), len(validation), len(test)), "LinearRegression"

    return {
        "domain": "nba_player_projection", "player": player["full_name"],
        "opponent": opponent["full_name"], "season": season,
        "projected_statline": projection, "confidence_ranges": ranges,
        "season_averages": {
            key: value
            for key, value in feature_summary["season_average"].items()
            if key in TARGETS
        },
        "recent_5_game_averages": _recent(features, 5),
        "recent_10_game_averages": _recent(features, 10),
        "feature_summary": feature_summary,
        "fallback_status": fallback_status,
        "key_drivers": _drivers(upcoming.iloc[0], features, context),
        "model_quality": quality,
        "data_splits": {"train_games": counts[0], "validation_games": counts[1],
                        "test_games": counts[2]},
        "availability": {"projected_available": True,
                         "note": "Live injury status unavailable; verify availability separately."},
        "warnings": list(dict.fromkeys(warnings)), "features_used": usable,
        "model": model,
        "data_quality": {"completeness_score": max(0.0, data_completeness(features, FEATURE_COLUMNS) - min(40.0, len(set(warnings)) * 5.0)),
                         "games_available": len(features),
                         "warning_count": len(set(warnings))},
    }