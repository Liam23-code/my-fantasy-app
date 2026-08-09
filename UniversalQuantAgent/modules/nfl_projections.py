"""Position-aware NFL statistical projections with touchdown modeling.

Every projection is built from a player's own historical per-game rates
(``modules.nfl_stats.canonical_player_stats``), then scaled by the matchup,
weather, and pace context already produced by ``analyze_nfl_player``. Nothing
here defaults a real statistical category to zero unless the player's own
history is truly zero for that category (e.g. a quarterback with zero career
rush attempts, or a wide receiver with zero career carries).
"""
from __future__ import annotations

from typing import Any

from modules.data_quality import safe_number
from modules.nfl_analysis import analyze_nfl_player
from modules.nfl_stats import get_player_stats

QB_RUSHING_YARDS_FLOOR = 10.0
RB_RECEPTIONS_FLOOR = 1.0
RB_RECEIVING_YARDS_FLOOR = 5.0
WR_TARGETS_FLOOR = 3.0


def _per_game(total: Any, games: float) -> float:
    return safe_number(total) / games if games > 0 else 0.0


def _qb_projection(stats: dict[str, Any], games: float, context_factor: float = 1.0) -> dict[str, Any]:
    """Project passing and rushing production only; receiving is always zero."""
    games = max(safe_number(games), 1.0)
    historical_rush_attempts = safe_number(stats.get("rush_attempts"))
    historical_rush_yards = safe_number(stats.get("rush_yards"))

    attempts = _per_game(stats.get("pass_attempts"), games)
    passing_yards = _per_game(stats.get("pass_yards"), games) * context_factor
    passing_tds = _per_game(stats.get("pass_tds"), games) * context_factor
    interceptions = _per_game(stats.get("ints"), games)
    carries = _per_game(stats.get("rush_attempts"), games)
    rushing_yards = _per_game(stats.get("rush_yards"), games) * context_factor
    rushing_tds = _per_game(stats.get("rush_tds"), games) * context_factor

    # A quarterback with a real rushing track record still projects some
    # rushing volume even in a week where recent efficiency looks thin.
    if historical_rush_attempts > 0 and historical_rush_yards > 0:
        rushing_yards = max(rushing_yards, QB_RUSHING_YARDS_FLOOR)

    return {
        "position": "QB",
        "attempts": round(attempts, 1),
        "passing_yards": round(passing_yards, 1),
        "passing_tds": round(passing_tds, 2),
        "interceptions": round(interceptions, 2),
        "carries": round(carries, 1),
        "rushing_yards": round(rushing_yards, 1),
        "rushing_tds": round(rushing_tds, 2),
        "targets": 0.0,
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_tds": 0.0,
    }


def _rb_projection(stats: dict[str, Any], games: float, context_factor: float = 1.0) -> dict[str, Any]:
    """Project rushing and receiving production for a running back."""
    games = max(safe_number(games), 1.0)
    historical_targets = safe_number(stats.get("targets"))

    carries = _per_game(stats.get("rush_attempts"), games)
    rushing_yards = _per_game(stats.get("rush_yards"), games) * context_factor
    rushing_tds = _per_game(stats.get("rush_tds"), games) * context_factor
    targets = _per_game(stats.get("targets"), games)
    receptions = _per_game(stats.get("receptions"), games) * context_factor
    receiving_yards = _per_game(stats.get("rec_yards"), games) * context_factor
    receiving_tds = _per_game(stats.get("rec_tds"), games) * context_factor

    # A back with a real receiving role keeps a receiving floor even when a
    # given week's context factor would otherwise push the line toward zero.
    if historical_targets > 0:
        receptions = max(receptions, RB_RECEPTIONS_FLOOR)
        receiving_yards = max(receiving_yards, RB_RECEIVING_YARDS_FLOOR)

    return {
        "position": "RB",
        "carries": round(carries, 1),
        "rushing_yards": round(rushing_yards, 1),
        "rushing_tds": round(rushing_tds, 2),
        "targets": round(targets, 1),
        "receptions": round(receptions, 1),
        "receiving_yards": round(receiving_yards, 1),
        "receiving_tds": round(receiving_tds, 2),
    }


def _receiver_projection(position: str, stats: dict[str, Any], games: float, context_factor: float = 1.0) -> dict[str, Any]:
    """Project receiving production for a WR/TE; rushing only for gadget players."""
    games = max(safe_number(games), 1.0)
    historical_rush_attempts = safe_number(stats.get("rush_attempts"))

    targets = _per_game(stats.get("targets"), games)
    receptions = _per_game(stats.get("receptions"), games) * context_factor
    receiving_yards = _per_game(stats.get("rec_yards"), games) * context_factor
    receiving_tds = _per_game(stats.get("rec_tds"), games) * context_factor

    # Only gadget players with an actual carry history get any rushing line.
    if historical_rush_attempts > 0:
        carries = _per_game(stats.get("rush_attempts"), games)
        rushing_yards = _per_game(stats.get("rush_yards"), games) * context_factor
        rushing_tds = _per_game(stats.get("rush_tds"), games) * context_factor
    else:
        carries = 0.0
        rushing_yards = 0.0
        rushing_tds = 0.0

    if targets > 0:
        targets = max(targets, WR_TARGETS_FLOOR)

    return {
        "position": position,
        "targets": round(targets, 1),
        "receptions": round(receptions, 1),
        "receiving_yards": round(receiving_yards, 1),
        "receiving_tds": round(receiving_tds, 2),
        "carries": round(carries, 1),
        "rushing_yards": round(rushing_yards, 1),
        "rushing_tds": round(rushing_tds, 2),
    }


def _defense_projection(stats: dict[str, Any], games: float, context_factor: float = 1.0) -> dict[str, Any]:
    return {
        "position": "DEF",
        "pressure_rate": round(safe_number(stats.get("pressure_rate")) * context_factor, 1),
        "sack_probability": round(safe_number(stats.get("sack_probability")) * context_factor, 1),
        "epa_allowed": round(safe_number(stats.get("epa_allowed")), 3),
    }


def _fantasy_points(position: str, line: dict[str, Any]) -> float:
    passing = safe_number(line.get("passing_yards")) * 0.04 + safe_number(line.get("passing_tds")) * 4 - safe_number(line.get("interceptions")) * 2
    rushing = safe_number(line.get("rushing_yards")) * 0.1 + safe_number(line.get("rushing_tds")) * 6
    receiving = safe_number(line.get("receiving_yards")) * 0.1 + safe_number(line.get("receiving_tds")) * 6 + safe_number(line.get("receptions")) * 1
    if position == "QB":
        return passing + rushing
    if position == "DEF":
        return 0.0
    return rushing + receiving


def project_qb(player: Any) -> dict[str, Any]:
    """Standalone QB projection (no matchup/weather context) for direct use."""
    stats = get_player_stats(player)
    line = _qb_projection(stats, safe_number(stats.get("games"), 1.0))
    line["fantasy_points"] = round(_fantasy_points("QB", line), 2)
    return line


def project_rb(player: Any) -> dict[str, Any]:
    """Standalone RB projection (no matchup/weather context) for direct use."""
    stats = get_player_stats(player)
    line = _rb_projection(stats, safe_number(stats.get("games"), 1.0))
    line["fantasy_points"] = round(_fantasy_points("RB", line), 2)
    return line


def project_wr(player: Any) -> dict[str, Any]:
    """Standalone WR projection (no matchup/weather context) for direct use."""
    stats = get_player_stats(player)
    line = _receiver_projection("WR", stats, safe_number(stats.get("games"), 1.0))
    line["fantasy_points"] = round(_fantasy_points("WR", line), 2)
    return line


def project_te(player: Any) -> dict[str, Any]:
    """Standalone TE projection (no matchup/weather context) for direct use."""
    stats = get_player_stats(player)
    line = _receiver_projection("TE", stats, safe_number(stats.get("games"), 1.0))
    line["fantasy_points"] = round(_fantasy_points("TE", line), 2)
    return line


def _red_zone_td_factor(raw: dict[str, Any]) -> float:
    """Blend red-zone usage into a small, always-positive touchdown multiplier."""
    red_zone_signal = safe_number(raw.get("red_zone_share", raw.get("red_zone_target_rate")), 50.0)
    return max(0.7, min(1.3, 1.0 + (red_zone_signal - 50.0) / 250.0))


def _build_drivers(position: str, raw: dict[str, Any], analysis: dict[str, Any], td_factor: float) -> list[str]:
    games = max(safe_number(raw.get("games")), 1.0)
    matchup = analysis.get("matchup", {})
    weather = analysis.get("weather", {})
    drivers: list[str] = []
    if position == "QB":
        drivers.append(f"Averaging {safe_number(raw.get('pass_yards')) / games:.1f} pass yards and {safe_number(raw.get('pass_tds')) / games:.2f} pass TDs per game.")
        if safe_number(raw.get("rush_attempts")) > 0:
            drivers.append(f"Rushing floor built from {safe_number(raw.get('rush_yards')) / games:.1f} career rush yards/game.")
    elif position == "RB":
        drivers.append(f"Averaging {safe_number(raw.get('rush_yards')) / games:.1f} rush yards and {safe_number(raw.get('receptions')) / games:.1f} receptions per game.")
    elif position in {"WR", "TE"}:
        drivers.append(f"Averaging {safe_number(raw.get('targets')) / games:.1f} targets and {safe_number(raw.get('rec_yards')) / games:.1f} receiving yards per game.")
    else:
        drivers.append(f"Averaging {safe_number(raw.get('pressure_rate')):.1f}% pressure rate over {games:.0f} games.")
    drivers.append(f"Matchup difficulty {safe_number(matchup.get('difficulty'), 50):.0f}/100 vs {matchup.get('team') or 'opponent'}.")
    drivers.append(f"Weather adjustment factor {safe_number(weather.get('factor'), 1):.2f}.")
    drivers.append(f"Red-zone touchdown multiplier {td_factor:.2f}.")
    return drivers


def project_nfl_player(
    player_name: str,
    opponent_team: str | None = None,
    season: int | None = None,
    mode: str = "Adjusted",
    comparison_mode: str = "League",
) -> dict[str, Any]:
    """Build one premium, position-correct NFL projection with a confidence band."""
    analysis = analyze_nfl_player(player_name, opponent_team, season, mode, comparison_mode)
    position = analysis["position"]
    raw = analysis["raw_stats"]
    games = max(safe_number(raw.get("games")), 1.0)
    context_factor = safe_number(analysis.get("context_factor"), 1.0)
    td_factor = _red_zone_td_factor(raw)

    if position == "QB":
        line = _qb_projection(raw, games, context_factor)
    elif position == "RB":
        line = _rb_projection(raw, games, context_factor)
    elif position in {"WR", "TE"}:
        line = _receiver_projection(position, raw, games, context_factor)
    else:
        line = _defense_projection(raw, games, context_factor)

    line["passing_tds"] = round(safe_number(line.get("passing_tds")) * td_factor, 2)
    line["rushing_tds"] = round(safe_number(line.get("rushing_tds")) * td_factor, 2)
    line["receiving_tds"] = round(safe_number(line.get("receiving_tds")) * td_factor, 2)
    expected_fantasy_points = round(_fantasy_points(position, line), 2)
    line["fantasy_points"] = expected_fantasy_points
    line["expected_fantasy_points"] = expected_fantasy_points

    projection: dict[str, Any] = {
        **line,
        "fantasy_points_projection": expected_fantasy_points,
        "passing_yards_projection": safe_number(line.get("passing_yards")),
        "passing_tds_projection": safe_number(line.get("passing_tds")),
        "interceptions_projection": safe_number(line.get("interceptions")),
        "rushing_yards_projection": safe_number(line.get("rushing_yards")),
        "rushing_tds_projection": safe_number(line.get("rushing_tds")),
        "targets_projection": safe_number(line.get("targets")),
        "receptions_projection": safe_number(line.get("receptions")),
        "receiving_yards_projection": safe_number(line.get("receiving_yards")),
        "receiving_tds_projection": safe_number(line.get("receiving_tds")),
        "receiving_targets": safe_number(line.get("targets")),
        "receiving_receptions": safe_number(line.get("receptions")),
        "carries_projection": safe_number(line.get("carries")),
    }

    volatility = safe_number(analysis.get("volatility_score"), 50.0)
    confidence_score = round(max(5.0, min(97.0, 100.0 - volatility)), 1)
    spread = max(1.0, round(expected_fantasy_points * (volatility / 100.0) * 0.6, 1))
    low = round(expected_fantasy_points - spread, 1)
    high = round(expected_fantasy_points + spread, 1)

    return {
        "player": analysis["player"],
        "player_id": analysis["player_id"],
        "team": analysis["team"],
        "opponent": opponent_team or "",
        "position": position,
        "season": analysis["season"],
        "mode": mode,
        "comparison_mode": comparison_mode,
        "projection": projection,
        "drivers": _build_drivers(position, raw, analysis, td_factor),
        "confidence": {
            "score": confidence_score,
            "low": low,
            "high": high,
            "label": f"Projected: {expected_fantasy_points:.1f} FP ({low:.1f}-{high:.1f})",
        },
        "volatility_score": analysis["volatility_score"],
        "analysis": analysis,
        "source": analysis["source"],
        "warnings": analysis["warnings"],
    }
