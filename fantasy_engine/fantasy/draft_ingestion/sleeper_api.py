"""Sleeper public API -- checked live this session; has no draft/ADP data.

Usage::

    from fantasy.draft_ingestion.sleeper_api import fetch_sleeper_draft_boards

    fetch_sleeper_draft_boards()  # -> [] always, today -- see below

Sleeper's public API (``api.sleeper.app``) is real and reachable, but it is a
league/roster-management API, not a rankings or mock-draft service. Checked
live this session, from this project's own runtime:

* ``GET /v1/players/nfl`` -- a full player identity/roster dump (name,
  team, position, ...), tens of MB, no ADP or draft-position field anywhere.
* ``GET /v1/players/nfl/trending/add`` -- ``{"player_id", "count"}`` pairs;
  ``count`` is waiver-wire add volume, not a pick number or draft rank.

No endpoint exposing average draft position, mock-draft results, or anything
resembling ``[[player_id, pick_number], ...]`` was found. This module
therefore returns ``[]`` unconditionally rather than fabricate a plausible-
looking result -- it is not a "not yet implemented" placeholder, it is the
honest answer for this source today. If Sleeper ever ships a real
rankings/ADP endpoint, wire it in here.
"""

from __future__ import annotations

from typing import Any


def fetch_sleeper_draft_boards() -> list[list[tuple[str, int]]]:
    """Always returns ``[]`` -- see module docstring for why."""
    return []


def fetch_sleeper_adp() -> list[dict[str, Any]]:
    """Always returns ``[]`` -- see module docstring for why."""
    return []
