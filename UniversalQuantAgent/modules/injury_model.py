"""Normalize free public NBA injury information into one safe interface."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup

from modules.data_quality import (
    fuzzy_name_match,
    safe_dict,
    safe_get,
    safe_list,
    safe_scalar_to_dict,
)
from modules.sportsbook import normalize_player_name, normalize_team_name

INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def normalize_status(value: Any) -> str:
    """Collapse provider-specific injury labels into four stable values."""
    if isinstance(value, dict):
        value = safe_get(value, "description", safe_get(value, "value", ""))
    text = str(value or "").upper()
    if any(word in text for word in ("OUT", "SUSPENDED", "DOUBTFUL")):
        return "OUT"
    if any(word in text for word in ("QUESTIONABLE", "GAME-TIME", "DAY-TO-DAY")):
        return "QUESTIONABLE"
    if "PROBABLE" in text:
        return "PROBABLE"
    return "ACTIVE"


def _text(value: Any, default: str = "") -> str:
    """Convert a provider value to text without exposing container reprs."""
    if isinstance(value, dict):
        value = (safe_get(value, "displayName") or safe_get(value, "description")
                 or safe_get(value, "name") or safe_get(value, "value"))
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return default
    try:
        return str(value).strip()
    except Exception:
        return default


def _details(node: Any) -> dict[str, Any]:
    """Build the stable details object used by every injury record."""
    payload = safe_dict(safe_get(node, "details"))
    description = (
        _text(safe_get(payload, "description"))
        or _text(safe_get(node, "shortComment"))
        or _text(safe_get(node, "longComment"))
        or _text(safe_get(node, "details"))
    )
    return {
        "description": description,
        "source": "ESPN",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _walk(node: Any, team: str, output: list[dict[str, Any]]) -> None:
    """Recursively inspect mixed ESPN payloads without assuming any node shape."""
    if isinstance(node, list):
        for item in safe_list(node):
            _walk(item, team, output)
        return
    if not isinstance(node, dict):
        return

    current_team = team
    if isinstance(safe_get(node, "injuries"), list):
        current_team = _text(safe_get(node, "abbreviation")) or _text(
            safe_get(node, "displayName"), team
        )

    athlete = safe_dict(safe_get(node, "athlete"))
    name = _text(safe_get(athlete, "displayName")) or _text(
        safe_get(node, "playerName")
    )

    # ESPN has returned type as a mapping, string, integer, and null across
    # different feeds. safe_get turns every malformed nested access into None.
    raw_status = safe_get(node, "status") or safe_get(
        safe_get(node, "type"), "description"
    )
    if raw_status is None:
        raw_status = safe_get(safe_scalar_to_dict(safe_get(node, "type")), "value")

    if name:
        output.append(
            {
                "team": normalize_team_name(current_team or ""),
                "player": normalize_player_name(name),
                "status": normalize_status(raw_status),
                "details": _details(node),
            }
        )

    for child in node.values():
        if isinstance(child, (dict, list)):
            _walk(child, current_team, output)


def _clean_record(record: Any) -> dict[str, Any] | None:
    """Validate one result before it crosses the ingestion boundary."""
    row = safe_dict(record)
    player = normalize_player_name(_text(safe_get(row, "player")))
    if not player:
        return None
    details = safe_dict(safe_get(row, "details"))
    return {
        "team": normalize_team_name(_text(safe_get(row, "team"))),
        "player": player,
        "status": normalize_status(safe_get(row, "status")),
        "details": {
            "description": _text(safe_get(details, "description")),
            "source": _text(safe_get(details, "source"), "ESPN"),
            "timestamp": _text(
                safe_get(details, "timestamp"),
                datetime.now(timezone.utc).isoformat(),
            ),
        },
    }


def fetch_injury_report(timeout: int = 12) -> list[dict[str, Any]]:
    """Return a deduplicated injury list with a stable four-field schema."""
    results: list[dict[str, Any]] = []
    try:
        response = requests.get(
            INJURY_URL,
            timeout=timeout,
            headers={"User-Agent": "UniversalQuantAgent/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        payload = safe_list(payload) if isinstance(payload, list) else safe_dict(payload)
        _walk(payload, "", results)
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        # ESPN's readable page is a secondary path if JSON is unavailable.
        try:
            html = requests.get(
                "https://www.espn.com/nba/injuries",
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            ).text
            for row in BeautifulSoup(html, "html.parser").select("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.select("td")]
                if len(cells) >= 3:
                    results.append(
                        {
                            "team": "",
                            "player": normalize_player_name(cells[0]),
                            "status": normalize_status(cells[1]),
                            "details": {
                                "description": cells[2],
                                "source": "ESPN",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        }
                    )
        except (requests.RequestException, TypeError, AttributeError):
            pass

    clean: dict[str, dict[str, Any]] = {}
    for result in results:
        record = _clean_record(result)
        if record:
            clean[record["player"].lower()] = record
    return list(clean.values())


def get_player_availability(player_name: str) -> dict[str, Any]:
    """Return availability plus an impact score where 100 means cannot play."""
    target = normalize_player_name(player_name)
    report = safe_list(fetch_injury_report())
    rows = [safe_dict(row) for row in report if isinstance(row, dict)]
    names = {
        _text(safe_get(row, "player")).lower(): row
        for row in rows
        if _text(safe_get(row, "player"))
    }
    matched_name = fuzzy_name_match(target, list(names), cutoff=.82)
    if matched_name:
        row = names[matched_name]
        status = normalize_status(safe_get(row, "status"))
        impact = {
            "OUT": 100.0,
            "QUESTIONABLE": 55.0,
            "PROBABLE": 20.0,
            "ACTIVE": 0.0,
        }.get(status, 45.0)
        details = safe_dict(safe_get(row, "details"))
        return {
            "player": target,
            "availability": status,
            "impact_score": impact,
            "detail": _text(safe_get(details, "description")),
            "source": _text(safe_get(details, "source"), "ESPN"),
            "warning": "",
        }
    return {
        "player": target,
        "availability": "ACTIVE",
        "impact_score": 0.0,
        "detail": "",
        "source": "fallback",
        "warning": (
            "Player was not listed; ACTIVE is an availability fallback, "
            "not confirmation."
        ),
    }