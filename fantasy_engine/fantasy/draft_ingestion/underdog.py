"""Underdog Fantasy draft board parser -- real parser, user-supplied JSON.

Usage::

    from fantasy.draft_ingestion.underdog import parse_underdog_draft_json

    picks = parse_underdog_draft_json(json_payload)
    # [("Jahmyr Gibbs", 3), ...]

No confirmed public, unauthenticated API for Underdog's contest draft boards
was found -- their draft results live inside the logged-in app. This module
parses a JSON payload **you** supply (e.g. copied from the app's own network
requests while logged in yourself), not something fetched automatically.

The expected shape is intentionally loose and documented rather than assumed
byte-exact, since the real export format was never confirmed live: a dict
with a ``"picks"`` list (or a bare list at the top level), each entry a dict
with a player-name-ish key (``"player_name"``/``"name"``/``"player"``) and a
pick-number-ish key (``"pick_number"``/``"pick"``/``"overall_pick"``).
"""

from __future__ import annotations

from typing import Any

_NAME_KEYS = ("player_name", "name", "player")
_PICK_KEYS = ("pick_number", "pick", "overall_pick")


def _first_present(entry: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in entry and entry[key] not in (None, ""):
            return entry[key]
    return None


def parse_underdog_draft_json(payload: Any) -> list[tuple[str, int]]:
    """Parse a user-supplied Underdog draft payload into ``(player_name, pick_number)`` pairs.

    Accepts either a bare list of pick entries or a dict with a ``"picks"``
    list. Returns ``[]`` for anything else, or when no entry yields both a
    name and an integer-parseable pick number.
    """
    if isinstance(payload, dict):
        entries = payload.get("picks")
    else:
        entries = payload
    if not isinstance(entries, list):
        return []

    picks: list[tuple[str, int]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = _first_present(entry, _NAME_KEYS)
        pick_number = _first_present(entry, _PICK_KEYS)
        if not name or pick_number is None:
            continue
        try:
            picks.append((str(name).strip(), int(pick_number)))
        except (TypeError, ValueError):
            continue
    return picks


def to_draft_boards(payload: Any) -> list[list[tuple[str, int]]]:
    """Unified :mod:`fantasy.draft_ingestion` entry point for this source.

    One supplied payload is one real draft board. Returns ``[]`` when the
    payload yields no usable picks.
    """
    picks = parse_underdog_draft_json(payload)
    return [picks] if picks else []
