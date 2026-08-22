# WARNING: Scraping disabled. This module is kept only for reference. Do not re-enable network calls.
"""Sportsbook live-fetch functions -- permanently disabled.

This module used to make live HTTP requests to DraftKings, FanDuel,
BetMGM, Caesars, and ESPN BET (directly, or via their public HTML pages as
a fallback), plus ESPN's scoreboard endpoint for the daily slate. That
network access is a legal/compliance risk this app does not accept, so
every function below is now a hard-fail stub: it raises immediately rather
than silently returning empty data, so a caller (human or AI agent) cannot
mistake "scraping is disabled" for "no props were found right now."

Odds now come only from ``odds.json`` (our own curated file) or a
user-uploaded file -- see the offline betting engine
(``fantasy_engine/betting/odds_loader.py``) and, for the NBA-side pages
that used to call the functions below, from an equivalent offline source
once one is built. The safe parsing/normalization logic that used to live
here (name/team/category normalization, JSON-payload market parsing) has
been extracted to :mod:`modules.sportsbook_parser`, which has no network
dependency and remains fully usable.

Do not add ``requests``, ``httpx``, ``aiohttp``, ``urllib``, or any browser
automation (``selenium``, ``playwright``) back into this file. If a future
task genuinely needs a new data source, it should be a new, explicitly
reviewed offline loader (a file, or a user upload), not a revival of this
module.
"""
from __future__ import annotations

from datetime import date

_DISABLED_MESSAGE = (
    "Direct sportsbook/odds-site scraping is disabled in this codebase. "
    "Use odds.json or user-uploaded odds instead (see fantasy_engine/betting/odds_loader.py)."
)


def fetch_draftkings_props() -> list[dict]:
    raise RuntimeError(_DISABLED_MESSAGE)


def fetch_fanduel_props() -> list[dict]:
    raise RuntimeError(_DISABLED_MESSAGE)


def fetch_betmgm_props() -> list[dict]:
    raise RuntimeError(_DISABLED_MESSAGE)


def fetch_caesars_props() -> list[dict]:
    raise RuntimeError(_DISABLED_MESSAGE)


def fetch_espnbet_props() -> list[dict]:
    raise RuntimeError(_DISABLED_MESSAGE)


def fetch_all_sportsbook_props() -> list[dict]:
    raise RuntimeError(_DISABLED_MESSAGE)


def fetch_daily_games(game_date: date | None = None) -> list[dict]:
    raise RuntimeError(_DISABLED_MESSAGE)
