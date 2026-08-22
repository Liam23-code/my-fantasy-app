# WARNING: Scraping disabled. This module is kept only for reference. Do not re-enable network calls.
"""Injury live-fetch function -- permanently disabled.

This module used to fetch live injury data from ESPN's public JSON API,
falling back to scraping ESPN's readable injuries page (via
``requests`` + ``BeautifulSoup``) if the JSON call failed. That network
access is a legal/compliance risk this app does not accept, so the
function below is now a hard-fail stub: it raises immediately rather than
silently returning empty data, so a caller cannot mistake "scraping is
disabled" for "no injuries reported right now."

Injury data now comes only from ``data/injuries.json`` (see
:func:`modules.injury_parser.load_injury_data_from_file`) or a
user-uploaded file (see
:func:`modules.injury_parser.load_injury_data_from_user_upload`). The safe
parsing/normalization/impact-modeling logic that used to live here has
been extracted to :mod:`modules.injury_parser`, which has no network
dependency and remains fully usable.

Do not add ``requests``, ``httpx``, ``aiohttp``, ``urllib``, or any browser
automation back into this file.
"""
from __future__ import annotations

from typing import Any

_DISABLED_MESSAGE = (
    "Online injury scraping is disabled in this codebase. "
    "Use data/injuries.json or a user-uploaded injury file instead "
    "(see modules.injury_parser.load_injury_data_from_file / load_injury_data_from_user_upload)."
)


def fetch_injury_report(timeout: int = 12) -> list[dict[str, Any]]:
    raise RuntimeError(_DISABLED_MESSAGE)


def fetch_injury_data_online(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    raise RuntimeError(_DISABLED_MESSAGE)
