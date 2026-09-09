"""player_status_fetcher_nba: refresh the ``nba`` slice of ``multi_sport_status.json``.

The NBA sibling of :mod:`fantasy.online.player_status_fetcher` (NFL). Same
contract exactly: stdlib :mod:`urllib` only, ``None`` on any failure, an atomic
write, and a ``fetch=`` seam the tests use so the network is never hit. Nothing
imports this at load time and no engine calls it -- it runs only when a user
clicks **Refresh Player Status** on the NBA betting page.

Source: ESPN's public, keyless NBA injuries feed
(``https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries``). This
is a live overlay only -- ``modules.nba_injury_impact`` and the rest of the NBA
pipeline keep their own local injury data unchanged; this just lets a
same-day OUT / doubtful / questionable ruling reach the prop board.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fantasy.multi_sport_status import STATUS_PATH
from fantasy.online._multi_sport_common import (
    SOURCE,
    espn_injuries_url,
    normalize_espn_injuries,
    refresh_sport_status,
)

SPORT = "nba"
ESPN_INJURIES_URL = espn_injuries_url(SPORT)

__all__ = ["ESPN_INJURIES_URL", "SOURCE", "SPORT", "normalize_feed", "refresh_player_status"]


def normalize_feed(payload: Any, *, now: str | None = None) -> dict[str, dict[str, Any]]:
    """ESPN NBA injuries JSON -> the unified status map. ``{}`` on junk input."""
    return normalize_espn_injuries(payload, now=now)


def refresh_player_status(
    path: Path = STATUS_PATH,
    url: str = ESPN_INJURIES_URL,
    *,
    fetch: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Fetch the NBA feed and rewrite only the ``nba`` slice of ``path``.
    On any failure the existing file is left untouched."""
    return refresh_sport_status(SPORT, path, url, fetch=fetch)
