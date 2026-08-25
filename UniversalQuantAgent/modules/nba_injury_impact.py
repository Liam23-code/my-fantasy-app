"""Real NBA injury-context signals: opponent absences and teammate-absence stat splits.

Part of the matchup-aware betting engine (see matchup_engine.md). Two
genuinely different real signals live here, both built from data this
codebase already loads elsewhere -- no new fetch beyond what
``modules.injury_parser``/``modules.projections`` already do:

* :func:`opponent_injury_context` -- does the *opponent's* real, loaded
  injury report include a real, meaningful rim defender being out? Cross-
  references the real absentee list (``modules.injury_parser``, offline
  file-or-upload only -- see offline_data_contract.md) against real
  per-player rim-defense volume (``modules.nba_defense_model``'s
  underlying player-tracking data) rather than guessing from position.
* :func:`teammate_absence_split` -- a real, direct measurement: this
  player's own real per-game stats in games their real game log shows a
  named teammate also played, vs. games that teammate's own log shows no
  matching date for (an inferred absence, disclosed as such -- this is
  game-level, not minute-level, and does not distinguish "did not play"
  from "not on the roster that game" for any other reason). Small real
  sample sizes are disclosed via ``games_sampled``, never hidden behind a
  confident-looking single number.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from modules.data_quality import coerce_numeric, safe_dict, safe_get, safe_list, safe_number
from modules.injury_parser import load_injury_data_from_file
from modules.nba_advanced import latest_season
from modules.nba_defense_model import _RIM_CATEGORY, _fetch_pt_defend
from modules.projections import _find_player, _game_log_with_fuzzy_fallback

#: A real defender needs at least this many real rim-shot contests this
#: season before "rim defender out" is asserted -- guards against a
#: single garbage-time appearance looking like a real rim protector.
_MIN_RIM_CONTESTS = 100

_SEVERITY_BY_STATUS = {"OUT": 1.0, "QUESTIONABLE": 0.5}

_STATS = ("pts", "reb", "ast", "min")


def opponent_injury_context(opponent_team: str, season: str | None = None) -> dict[str, Any]:
    """Real absentee list for ``opponent_team`` from our own loaded injury report, plus a real rim-defender flag.

    Returns ``{"absences", "severity", "rim_defender_out", "warnings"}``.
    ``absences`` is every real ``OUT``/``QUESTIONABLE`` row for this team
    from :func:`modules.injury_parser.load_injury_data_from_file`
    (offline file-or-upload only). ``rim_defender_out`` is ``None`` unless
    a real absent player also shows up in this season's real
    player-tracking rim-defense data with at least ``_MIN_RIM_CONTESTS``
    real contests -- then it names them and their real defended FG%.
    """
    season = season or latest_season()
    warnings: list[str] = []
    try:
        report = safe_list(load_injury_data_from_file())
    except Exception as exc:
        report = []
        warnings.append(f"Injury report unavailable: {exc}")

    absences = [
        safe_dict(item)
        for item in report
        if safe_get(safe_dict(item), "team") == opponent_team
        and safe_get(safe_dict(item), "status") in _SEVERITY_BY_STATUS
    ]
    severity = round(sum(_SEVERITY_BY_STATUS.get(safe_get(row, "status"), 0.0) for row in absences), 2)

    rim_defender_out = None
    if absences:
        try:
            rim_frame = _fetch_pt_defend(season, _RIM_CATEGORY)
        except Exception as exc:
            rim_frame = pd.DataFrame()
            warnings.append(f"Rim-defense data unavailable: {exc}")
        if not rim_frame.empty and {"PLAYER_NAME", "FGA_LT_06", "LT_06_PCT"}.issubset(rim_frame.columns):
            absent_names = {str(safe_get(row, "player", "")).lower() for row in absences}
            candidates = rim_frame[rim_frame["PLAYER_NAME"].astype(str).str.lower().isin(absent_names)]
            candidates = candidates[pd.to_numeric(candidates["FGA_LT_06"], errors="coerce") >= _MIN_RIM_CONTESTS]
            if not candidates.empty:
                row = candidates.sort_values("FGA_LT_06", ascending=False).iloc[0]
                rim_defender_out = {
                    "player": str(row["PLAYER_NAME"]),
                    "real_rim_contests": int(safe_number(row["FGA_LT_06"])),
                    "real_defended_fg_pct": round(safe_number(row["LT_06_PCT"]), 4),
                }

    return {
        "opponent": opponent_team,
        "season": season,
        "absences": [{"player": safe_get(row, "player"), "status": safe_get(row, "status")} for row in absences],
        "severity": severity,
        "rim_defender_out": rim_defender_out,
        "warnings": warnings,
    }


def teammate_absence_split(player_name: str, teammate_name: str, season: str | None = None, *, min_games: int = 3) -> dict[str, Any]:
    """Real per-game stat split for ``player_name``: games with vs. without ``teammate_name`` (inferred from real logs)."""
    season = season or latest_season()
    warnings: list[str] = []

    player = _find_player(player_name)
    resolved_player, player_games_raw, player_warnings = _game_log_with_fuzzy_fallback(player_name, player, season)
    warnings.extend(player_warnings)
    player_games = coerce_numeric(player_games_raw)

    try:
        teammate = _find_player(teammate_name)
        resolved_teammate, teammate_games_raw, teammate_warnings = _game_log_with_fuzzy_fallback(teammate_name, teammate, season)
        warnings.extend(teammate_warnings)
        teammate_dates = set(pd.to_datetime(coerce_numeric(teammate_games_raw).get("game_date"), errors="coerce").dropna())
    except ValueError as exc:
        return {
            "player": player_name,
            "teammate": teammate_name,
            "season": season,
            "insufficient_sample": True,
            "warnings": warnings + [f"Could not resolve teammate: {exc}"],
        }

    if "game_date" not in player_games or player_games.empty:
        return {
            "player": resolved_player.get("full_name", player_name),
            "teammate": resolved_teammate.get("full_name", teammate_name),
            "season": season,
            "insufficient_sample": True,
            "warnings": warnings + ["No game log available for the primary player."],
        }

    player_dates = pd.to_datetime(player_games["game_date"], errors="coerce")
    with_teammate = player_games[player_dates.isin(teammate_dates)]
    without_teammate = player_games[~player_dates.isin(teammate_dates)]

    def _averages(frame: pd.DataFrame) -> dict[str, float]:
        return {stat: round(safe_number(pd.to_numeric(frame.get(stat), errors="coerce").mean()), 2) for stat in _STATS if stat in frame}

    with_averages = _averages(with_teammate)
    without_averages = _averages(without_teammate)
    insufficient = len(with_teammate) < min_games or len(without_teammate) < min_games

    delta = {
        stat: round(without_averages[stat] - with_averages[stat], 2)
        for stat in without_averages
        if stat in with_averages
    }

    return {
        "player": resolved_player.get("full_name", player_name),
        "teammate": resolved_teammate.get("full_name", teammate_name),
        "season": season,
        "games_with_teammate": len(with_teammate),
        "games_without_teammate": len(without_teammate),
        "averages_with_teammate": with_averages,
        "averages_without_teammate": without_averages,
        # Positive = real per-game rate goes up when the teammate is absent.
        "delta_without_minus_with": delta,
        "insufficient_sample": insufficient,
        "warnings": warnings,
    }
