"""Shared, sport-agnostic normalization for the college sports (CFB, CBB) loaders.

Re-exports :func:`modules.sportsbook_parser.normalize_player_name` (it has
no NBA-specific content -- accent-stripping, suffix-stripping, title-casing
apply to any player name) but defines its own team-name normalization:
NBA's fixed 30-team alias table (``sportsbook_parser.TEAM_ALIASES``)
doesn't scale to CFB's 130+ FBS programs or CBB's 350+ Division I
programs, so a college team name is normalized by case/punctuation only,
not through a hardcoded alias map.
"""
from __future__ import annotations

import re

from modules.sportsbook_parser import normalize_player_name

__all__ = ["normalize_player_name", "normalize_college_team_name"]


def normalize_college_team_name(value: str) -> str:
    """Uppercase, punctuation-stripped, whitespace-collapsed team name -- no fixed alias table."""
    text = re.sub(r"[^A-Za-z0-9 ]", "", str(value or "")).strip().upper()
    return " ".join(text.split())
