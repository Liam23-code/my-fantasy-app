"""Persistent, week-aware management for a user's drafted fantasy team.

The manager is deliberately a thin orchestration layer over the engine's two
authoritative projection sources:

* :mod:`fantasy.projections` supplies season value; and
* :mod:`fantasy.weekly_projections` supplies matchup-adjusted weekly points
  and confidence.

The saved team lives at ``fantasy_engine/data/user_team.json``.  A team may be
a plain list of players or a document containing ``players``/``roster`` plus
optional ``league_settings``.  Both forms round-trip without being rewritten
into a different public shape.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fantasy.assistant import weekly_start_sit_advice
from fantasy.models import LeagueSettings
from fantasy.projections import projected_points
from fantasy.utils import clamp, normalize_player_name, safe_float
from fantasy.weekly_projections import WEEKS, build_weekly_projection, weekly_matchups

USER_TEAM_PATH = Path(__file__).resolve().parents[1] / "data" / "user_team.json"

INACTIVE_STATUSES = frozenset({"OUT", "IR", "DOUBTFUL", "SUSPENDED", "PUP", "NFI"})
BENCH_SLOTS = frozenset({"", "BENCH", "BN", "IR", "TAXI", "RESERVE"})
FLEX_POSITIONS = frozenset({"RB", "WR", "TE"})


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    raise TypeError(f"Cannot serialize {type(value).__name__} in a fantasy team")


def load_user_team() -> Any:
    """Load the saved team, returning ``[]`` when no team has been saved yet.

    Invalid JSON is reported as :class:`ValueError` instead of being silently
    treated as an empty roster; silently erasing a user's apparent team would
    be much worse than showing a recoverable data error in the UI.
    """
    if not USER_TEAM_PATH.exists():
        return []
    try:
        return json.loads(USER_TEAM_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Saved team is not valid JSON: {USER_TEAM_PATH}") from error


def save_user_team(team: Any) -> Any:
    """Atomically persist ``team`` and return its JSON-safe representation."""
    payload = _jsonable(team)
    if not isinstance(payload, (list, dict)):
        raise TypeError("team must be a player list or a mapping containing a roster")

    USER_TEAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{USER_TEAM_PATH.stem}.",
            suffix=".tmp",
            dir=USER_TEAM_PATH.parent,
            delete=False,
        ) as temporary:
            temp_name = temporary.name
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temp_name, USER_TEAM_PATH)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()
    return payload


def _team_parts(team: Any) -> tuple[list[dict[str, Any]], LeagueSettings]:
    settings_value: Any = None
    players_value: Any = team

    if hasattr(team, "model_dump") and callable(team.model_dump):
        team = team.model_dump()
        players_value = team
    if isinstance(team, Mapping):
        settings_value = team.get("league_settings") or team.get("settings")
        for key in ("players", "roster", "team"):
            candidate = team.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                players_value = candidate
                break

    if players_value is None:
        players_value = []
    if not isinstance(players_value, Sequence) or isinstance(players_value, (str, bytes, bytearray)):
        raise TypeError("team must be a player list or contain a players/roster list")

    players: list[dict[str, Any]] = []
    for index, player in enumerate(players_value):
        if isinstance(player, Mapping):
            row = dict(player)
        elif hasattr(player, "model_dump") and callable(player.model_dump):
            row = dict(player.model_dump())
        elif hasattr(player, "__dict__"):
            row = dict(vars(player))
        else:
            continue
        name = str(row.get("name") or row.get("player_name") or "").strip()
        player_id = str(row.get("player_id") or row.get("id") or f"team-player-{index}").strip()
        row["player_id"] = player_id
        row["name"] = name or player_id
        row["position"] = str(row.get("position") or "").strip().upper()
        team_code = str(row.get("team") or row.get("nfl_team") or "").strip().upper()
        row["team"] = team_code
        row["nfl_team"] = team_code
        row["slot"] = str(row.get("slot") or "BENCH").strip().upper()
        injury = row.get("injury_status", row.get("status"))
        row["injury_status"] = str(injury).strip().upper() if injury else None
        players.append(row)

    if isinstance(settings_value, LeagueSettings):
        settings = settings_value
    else:
        settings = LeagueSettings(**(dict(settings_value) if isinstance(settings_value, Mapping) else {}))
    return players, settings


def _validated_week(week: Any) -> int:
    if isinstance(week, bool):
        raise ValueError("week must be an integer from 1 through 18")
    try:
        number = int(week)
    except (TypeError, ValueError) as error:
        raise ValueError("week must be an integer from 1 through 18") from error
    if number not in WEEKS:
        raise ValueError("week must be an integer from 1 through 18")
    return number


def _player_key(player: Mapping[str, Any]) -> str:
    player_id = str(player.get("player_id") or player.get("id") or "").strip()
    return player_id or normalize_player_name(player.get("name") or player.get("player_name"))


def _weekly_value(player: Mapping[str, Any], week: int, scoring_mode: str) -> dict[str, float]:
    curve = player.get("weekly_projection")
    value = None
    if isinstance(curve, Mapping):
        value = curve.get(week, curve.get(str(week)))
    if not isinstance(value, Mapping) or not isinstance(value.get("points"), (int, float)):
        value = build_weekly_projection(player, scoring_mode)[week]
    return {
        "points": round(max(0.0, safe_float(value.get("points"))), 2),
        "confidence": round(clamp(safe_float(value.get("confidence"), 0.0), 0.0, 1.0), 3),
    }


def _season_value(player: Mapping[str, Any], settings: LeagueSettings) -> float:
    value = projected_points(player, settings)
    if value is not None:
        return max(0.0, float(value))
    for key in ("projection", "expected_fantasy_points", "points"):
        raw = player.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return max(0.0, float(raw))
    return 0.0


def _lineup_advice(team: Any, week: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], LeagueSettings]:
    players, settings = _team_parts(team)
    if not players:
        return [], players, settings
    advice = weekly_start_sit_advice(players, players, week, settings)
    return advice, players, settings


def weekly_team_projection(team: Any, week: int) -> dict[str, Any]:
    """Project the legal, optimized starting lineup for one week.

    Returns the total, weighted lineup confidence, and explicit starter/bench
    rows so UIs can show the number and explain how it was assembled.
    """
    week_number = _validated_week(week)
    advice, players, settings = _lineup_advice(team, week_number)
    by_name = {normalize_player_name(player["name"]): player for player in players}

    starters: list[dict[str, Any]] = []
    bench: list[dict[str, Any]] = []
    for recommendation in advice:
        player = by_name.get(normalize_player_name(recommendation.get("player")), {})
        value = _weekly_value(player, week_number, settings.scoring_mode) if player else {
            "points": safe_float(recommendation.get("projected_points")),
            "confidence": safe_float(recommendation.get("confidence")),
        }
        row = {
            "player_id": player.get("player_id"),
            "name": recommendation.get("player"),
            "position": recommendation.get("position"),
            "points": round(value["points"], 2),
            "confidence": round(value["confidence"], 3),
            "opponent": recommendation.get("opponent", "TBD"),
            "reason": recommendation.get("reason", ""),
        }
        if recommendation.get("start_or_bench") == "start":
            starters.append(row)
        else:
            bench.append(row)

    total = round(sum(player["points"] for player in starters), 2)
    confidence_weight = sum(max(player["points"], 0.1) for player in starters)
    if starters and confidence_weight:
        confidence = sum(player["confidence"] * max(player["points"], 0.1) for player in starters) / confidence_weight
    else:
        confidence = 0.0
    return {
        "week": week_number,
        "total_points": total,
        "confidence": round(clamp(confidence, 0.0, 1.0), 3),
        "starters": sorted(starters, key=lambda player: player["points"], reverse=True),
        "bench": sorted(bench, key=lambda player: player["points"], reverse=True),
    }


def find_weak_positions(team: Any, week: int) -> list[dict[str, Any]]:
    """Identify missing, unavailable, low-matchup, or low-confidence positions."""
    week_number = _validated_week(week)
    projection = weekly_team_projection(team, week_number)
    players, settings = _team_parts(team)
    starters = projection["starters"]
    starter_counts: dict[str, int] = {}
    for player in starters:
        position = str(player.get("position") or "").upper()
        starter_counts[position] = starter_counts.get(position, 0) + 1

    weaknesses: dict[str, dict[str, Any]] = {}

    def _add(position: str, severity: float, reason: str, player_name: str | None = None) -> None:
        if not position:
            return
        entry = weaknesses.setdefault(
            position,
            {"position": position, "severity": 0.0, "reasons": [], "players": []},
        )
        entry["severity"] += severity
        if reason not in entry["reasons"]:
            entry["reasons"].append(reason)
        if player_name and player_name not in entry["players"]:
            entry["players"].append(player_name)

    required_slots = settings.roster_requirements.starting_slots()
    flex_required = required_slots.pop("FLEX", 0)
    for position, required in required_slots.items():
        missing = max(0, required - starter_counts.get(position, 0))
        if missing:
            _add(position, 3.0 * missing, f"{missing} starting slot(s) unfilled")

    dedicated_flex_starts = sum(min(starter_counts.get(pos, 0), required_slots.get(pos, 0)) for pos in settings.flex_eligible)
    total_flex_eligible_starts = sum(starter_counts.get(pos, 0) for pos in settings.flex_eligible)
    missing_flex = max(0, flex_required - max(0, total_flex_eligible_starts - dedicated_flex_starts))
    if missing_flex:
        _add("FLEX", 2.5 * missing_flex, f"{missing_flex} FLEX slot(s) unfilled")

    for player in players:
        status = str(player.get("injury_status") or "").upper()
        weekly = _weekly_value(player, week_number, settings.scoring_mode)
        opponent = weekly_matchups(player)[week_number]["opponent"]
        position = player.get("position", "")
        if status in INACTIVE_STATUSES:
            _add(position, 2.5, f"{status} player reduces available depth", player["name"])
        if opponent == "BYE":
            _add(position, 2.5, "player is on bye", player["name"])
        season_per_game = _season_value(player, settings) / 17.0
        if player["name"] in {starter["name"] for starter in starters}:
            if season_per_game > 0 and weekly["points"] < season_per_game * 0.80:
                _add(position, 1.25, "starter projects at least 20% below season pace", player["name"])
            if weekly["confidence"] < 0.45:
                _add(position, 1.0, "starter projection has low confidence", player["name"])

    results = list(weaknesses.values())
    for entry in results:
        entry["severity"] = round(entry["severity"], 2)
        entry["reason"] = "; ".join(entry["reasons"])
    results.sort(key=lambda entry: (-entry["severity"], entry["position"]))
    return results


def _pool_players(player_pool: Any) -> list[dict[str, Any]]:
    if isinstance(player_pool, Mapping):
        for key in ("players", "available", "player_pool", "trade_pool", "projections"):
            if key in player_pool:
                player_pool = player_pool[key]
                break
    if not isinstance(player_pool, Sequence) or isinstance(player_pool, (str, bytes, bytearray)):
        return []
    rows: list[dict[str, Any]] = []
    for player in player_pool:
        if isinstance(player, Mapping):
            rows.append(dict(player))
        elif hasattr(player, "model_dump") and callable(player.model_dump):
            rows.append(dict(player.model_dump()))
    return rows


def recommend_add_drop(team: Any, week: int, player_pool: Any) -> list[dict[str, Any]]:
    """Recommend waiver additions and the same-position player to drop."""
    week_number = _validated_week(week)
    players, settings = _team_parts(team)
    if not players:
        return []
    roster_keys = {_player_key(player) for player in players}
    roster_names = {normalize_player_name(player.get("name")) for player in players}
    projection = weekly_team_projection(team, week_number)
    starting_names = {player["name"] for player in projection["starters"]}
    weak_positions = {entry["position"] for entry in find_weak_positions(team, week_number)}

    by_position: dict[str, list[dict[str, Any]]] = {}
    for player in players:
        by_position.setdefault(player.get("position", ""), []).append(player)

    recommendations: list[dict[str, Any]] = []
    for candidate in _pool_players(player_pool):
        if _player_key(candidate) in roster_keys or normalize_player_name(candidate.get("name")) in roster_names:
            continue
        position = str(candidate.get("position") or "").strip().upper()
        if not position:
            continue
        candidate_week = _weekly_value(candidate, week_number, settings.scoring_mode)
        candidate_season = _season_value(candidate, settings)
        if candidate_week["points"] <= 0:
            continue

        drop_options = by_position.get(position, [])
        drop_player = min(
            drop_options,
            key=lambda player: (
                player.get("name") in starting_names,
                _weekly_value(player, week_number, settings.scoring_mode)["points"],
                _season_value(player, settings),
            ),
            default=None,
        )
        if drop_player:
            drop_week = _weekly_value(drop_player, week_number, settings.scoring_mode)
            drop_season = _season_value(drop_player, settings)
            weekly_gain = candidate_week["points"] - drop_week["points"]
            season_gain = candidate_season - drop_season
        else:
            weekly_gain = candidate_week["points"]
            season_gain = candidate_season

        if drop_player and weekly_gain < 0.5 and season_gain < 8.5:
            continue
        weakness_bonus = 1.5 if position in weak_positions or (position in FLEX_POSITIONS and "FLEX" in weak_positions) else 0.0
        score = weekly_gain + season_gain / 68.0 + weakness_bonus
        add_name = str(candidate.get("name") or candidate.get("player_name") or _player_key(candidate))
        drop_name = drop_player.get("name") if drop_player else None
        recommendations.append(
            {
                "add": add_name,
                "add_player_id": candidate.get("player_id") or candidate.get("id"),
                "drop": drop_name,
                "drop_player_id": drop_player.get("player_id") if drop_player else None,
                "position": position,
                "projected_points": candidate_week["points"],
                "confidence": candidate_week["confidence"],
                "weekly_gain": round(weekly_gain, 2),
                "season_value_gain": round(season_gain, 2),
                "priority_score": round(score, 2),
                "reason": (
                    f"Add {add_name} for {candidate_week['points']:.1f} projected Week {week_number} points; "
                    + (f"drop {drop_name} for a {weekly_gain:+.1f}-point weekly change." if drop_name else "fills an empty roster position.")
                ),
            }
        )
    recommendations.sort(key=lambda item: item["priority_score"], reverse=True)
    return recommendations[:8]


def _remaining_curve_value(player: Mapping[str, Any], week: int, scoring_mode: str) -> float:
    curve = build_weekly_projection(player, scoring_mode)
    return sum(curve[current]["points"] for current in WEEKS if current >= week)


def recommend_trades(team: Any, week: int, player_pool: Any) -> list[dict[str, Any]]:
    """Recommend direct same-position upgrades from a supplied trade pool."""
    week_number = _validated_week(week)
    players, settings = _team_parts(team)
    if not players:
        return []
    roster_keys = {_player_key(player) for player in players}
    roster_names = {normalize_player_name(player.get("name")) for player in players}
    by_position: dict[str, list[dict[str, Any]]] = {}
    for player in players:
        by_position.setdefault(player.get("position", ""), []).append(player)

    remaining_value = {
        _player_key(player): _remaining_curve_value(player, week_number, settings.scoring_mode) for player in players
    }
    recommendations: list[dict[str, Any]] = []
    for target in _pool_players(player_pool):
        if _player_key(target) in roster_keys or normalize_player_name(target.get("name")) in roster_names:
            continue
        position = str(target.get("position") or "").strip().upper()
        offers = by_position.get(position, [])
        if not offers:
            continue
        target_remaining = _remaining_curve_value(target, week_number, settings.scoring_mode)
        offer = min(offers, key=lambda player: remaining_value[_player_key(player)])
        offer_remaining = remaining_value[_player_key(offer)]
        rest_gain = target_remaining - offer_remaining
        target_week = _weekly_value(target, week_number, settings.scoring_mode)
        offer_week = _weekly_value(offer, week_number, settings.scoring_mode)
        weekly_gain = target_week["points"] - offer_week["points"]
        season_gain = _season_value(target, settings) - _season_value(offer, settings)
        if rest_gain < 3.0 and weekly_gain < 1.0 and season_gain < 8.5:
            continue
        target_name = str(target.get("name") or target.get("player_name") or _player_key(target))
        score = rest_gain + max(0.0, weekly_gain) * 2.0 + max(0.0, season_gain) / 17.0
        recommendations.append(
            {
                "trade_for": target_name,
                "target_player_id": target.get("player_id") or target.get("id"),
                "offer": offer.get("name"),
                "offer_player_id": offer.get("player_id"),
                "position": position,
                "weekly_gain": round(weekly_gain, 2),
                "rest_of_season_gain": round(rest_gain, 2),
                "season_value_gain": round(season_gain, 2),
                "confidence": target_week["confidence"],
                "trade_score": round(score, 2),
                "reason": (
                    f"Target {target_name} as a {position} upgrade over {offer.get('name')}: "
                    f"{weekly_gain:+.1f} points in Week {week_number} and {rest_gain:+.1f} projected points "
                    "over the remaining schedule."
                ),
            }
        )
    recommendations.sort(key=lambda item: item["trade_score"], reverse=True)
    return recommendations[:8]


def _currently_starting(player: Mapping[str, Any]) -> bool:
    return str(player.get("slot") or "").strip().upper() not in BENCH_SLOTS


def _swap_eligible(incoming: Mapping[str, Any], outgoing: Mapping[str, Any]) -> bool:
    incoming_position = str(incoming.get("position") or "").upper()
    outgoing_position = str(outgoing.get("position") or "").upper()
    outgoing_slot = str(outgoing.get("slot") or "").upper()
    return incoming_position == outgoing_position or (
        incoming_position in FLEX_POSITIONS and (outgoing_position in FLEX_POSITIONS or outgoing_slot == "FLEX")
    )


def recommend_lineup_swaps(team: Any, week: int) -> list[dict[str, Any]]:
    """Compare saved slots with the optimized weekly lineup and suggest swaps."""
    week_number = _validated_week(week)
    projection = weekly_team_projection(team, week_number)
    players, settings = _team_parts(team)
    by_name = {normalize_player_name(player["name"]): player for player in players}
    recommended_starts = {normalize_player_name(player["name"]): player for player in projection["starters"]}
    current_starters = {normalize_player_name(player["name"]): player for player in players if _currently_starting(player)}

    incoming_keys = [key for key in recommended_starts if key not in current_starters]
    outgoing_keys = [key for key in current_starters if key not in recommended_starts]
    used_outgoing: set[str] = set()
    swaps: list[dict[str, Any]] = []

    for incoming_key in incoming_keys:
        incoming = by_name[incoming_key]
        incoming_value = _weekly_value(incoming, week_number, settings.scoring_mode)
        eligible_outgoing = [
            key
            for key in outgoing_keys
            if key not in used_outgoing and _swap_eligible(incoming, by_name[key])
        ]
        outgoing_key = min(
            eligible_outgoing,
            key=lambda key: _weekly_value(by_name[key], week_number, settings.scoring_mode)["points"],
            default=None,
        )
        outgoing = by_name[outgoing_key] if outgoing_key else None
        outgoing_points = (
            _weekly_value(outgoing, week_number, settings.scoring_mode)["points"] if outgoing else 0.0
        )
        gain = incoming_value["points"] - outgoing_points
        if outgoing_key:
            used_outgoing.add(outgoing_key)
        swaps.append(
            {
                "start": incoming.get("name"),
                "bench": outgoing.get("name") if outgoing else None,
                "position": incoming.get("position"),
                "projected_gain": round(gain, 2),
                "confidence": incoming_value["confidence"],
                "reason": (
                    f"Start {incoming.get('name')} over {outgoing.get('name')} for {gain:+.1f} projected points."
                    if outgoing
                    else f"Place {incoming.get('name')} in the Week {week_number} starting lineup."
                ),
            }
        )

    swaps.sort(key=lambda item: item["projected_gain"], reverse=True)
    return swaps


def bench_vs_start_decision(player: Any, week: int) -> dict[str, Any]:
    """Return an individual start/flex/sit signal for one player and week."""
    week_number = _validated_week(week)
    if isinstance(player, Mapping):
        row = dict(player)
    elif hasattr(player, "model_dump") and callable(player.model_dump):
        row = dict(player.model_dump())
    elif hasattr(player, "__dict__"):
        row = dict(vars(player))
    else:
        raise TypeError("player must be a mapping or object with fields")

    mode = str(row.get("scoring_mode") or "ppr")
    value = _weekly_value(row, week_number, mode)
    matchup = weekly_matchups(row)[week_number]
    status = str(row.get("injury_status") or row.get("status") or "").strip().upper()
    settings = LeagueSettings(scoring_mode=mode)
    season_per_game = _season_value(row, settings) / 17.0

    if matchup["opponent"] == "BYE":
        decision, reason = "sit", f"Week {week_number} bye; projected for 0.0 points."
    elif status in INACTIVE_STATUSES:
        decision, reason = "sit", f"Listed {status}; do not place in the active lineup."
    elif season_per_game <= 0 or value["points"] >= season_per_game * 0.90:
        decision = "start"
        reason = f"Projects for {value['points']:.1f} points, at or near season scoring pace."
    elif value["points"] >= season_per_game * 0.75:
        decision = "flex"
        reason = f"Projects for {value['points']:.1f} points; usable as a FLEX or matchup-dependent start."
    else:
        decision = "sit"
        reason = f"Projects {value['points']:.1f} points, well below the {season_per_game:.1f}-point season pace."

    return {
        "player": row.get("name") or row.get("player_name"),
        "position": str(row.get("position") or "").upper(),
        "week": week_number,
        "decision": decision,
        "points": value["points"],
        "confidence": value["confidence"],
        "opponent": matchup["opponent"],
        "reason": reason,
    }


def team_health_status(team: Any) -> dict[str, Any]:
    """Summarize roster availability and injury risk on a 0-100 scale."""
    players, _settings = _team_parts(team)
    issues: list[dict[str, Any]] = []
    deductions = 0.0
    weights = {
        "IR": 35.0,
        "OUT": 30.0,
        "SUSPENDED": 30.0,
        "PUP": 30.0,
        "NFI": 30.0,
        "DOUBTFUL": 20.0,
        "QUESTIONABLE": 10.0,
    }
    for player in players:
        status = str(player.get("injury_status") or "").upper()
        if not status:
            continue
        deduction = weights.get(status, 5.0)
        deductions += deduction
        issues.append(
            {
                "player": player.get("name"),
                "position": player.get("position"),
                "status": status,
                "impact": deduction,
            }
        )
    divisor = max(1.0, len(players) / 8.0)
    score = round(clamp(100.0 - deductions / divisor, 0.0, 100.0), 1)
    status_label = "healthy" if score >= 85 else ("watch" if score >= 60 else "critical")
    return {
        "status": status_label,
        "health_score": score,
        "total_players": len(players),
        "available_players": sum(
            1 for player in players if str(player.get("injury_status") or "").upper() not in INACTIVE_STATUSES
        ),
        "issues": sorted(issues, key=lambda issue: issue["impact"], reverse=True),
    }


def team_confidence_curve(team: Any) -> dict[int, dict[str, float]]:
    """Return the optimized team's points and weighted confidence for all 18 weeks."""
    curve: dict[int, dict[str, float]] = {}
    for week in WEEKS:
        projection = weekly_team_projection(team, week)
        curve[week] = {
            "points": projection["total_points"],
            "confidence": projection["confidence"],
        }
    return curve


__all__ = [
    "USER_TEAM_PATH",
    "bench_vs_start_decision",
    "find_weak_positions",
    "load_user_team",
    "recommend_add_drop",
    "recommend_lineup_swaps",
    "recommend_trades",
    "save_user_team",
    "team_confidence_curve",
    "team_health_status",
    "weekly_team_projection",
]
