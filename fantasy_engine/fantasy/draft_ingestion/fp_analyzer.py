"""FantasyPros Draft Analyzer CSV parser -- real parser, user-supplied file.

Usage::

    from fantasy.draft_ingestion.fp_analyzer import parse_fp_analyzer_csv

    records = parse_fp_analyzer_csv(csv_text)
    # [{"name": "Jahmyr Gibbs", "position": "RB", "team": "DET", "adp": 3.3}, ...]

FantasyPros' Draft Analyzer is a logged-in feature tied to a user's own mock
or real draft session; there is no public, unauthenticated endpoint for it
(unlike the general consensus rankings already used via
:func:`fantasy.data_loader.load_adp`, which come from a different, genuinely
public FantasyPros feed through ``nflreadpy``). This module cannot fetch it
automatically -- it parses a CSV **you** export from your own logged-in
Draft Analyzer session and supply here.

The column-matching below is deliberately permissive (several likely header
spellings, case-insensitive) rather than one hard-coded schema, because the
exact export format was not verified live this session (it requires a login
this project doesn't have) -- treat it as best-effort based on the general,
publicly-documented shape of FantasyPros' exports, not a confirmed byte-exact
contract.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from typing import Any

from fantasy.utils import safe_float

_NAME_COLUMNS = ("player", "name", "player name")
_POSITION_COLUMNS = ("position", "pos")
_TEAM_COLUMNS = ("team", "tm")
_ADP_COLUMNS = ("adp", "avg pick", "average pick", "avg. pick")
_STDEV_COLUMNS = ("std dev", "stdev", "std. dev")


def _find_column(header: Sequence[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.strip().lower(): column for column in header}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def parse_fp_analyzer_csv(csv_text: str) -> list[dict[str, Any]]:
    """Parse a FantasyPros Draft Analyzer CSV export into player ADP records.

    Returns ``[]`` for empty input, a header-only file, or a file missing
    both a recognizable name column and an ADP column -- there's nothing
    usable to extract in that case. Rows missing a name are skipped
    individually rather than failing the whole file.
    """
    if not csv_text or not csv_text.strip():
        return []

    reader = csv.DictReader(io.StringIO(csv_text))
    header = reader.fieldnames
    if not header:
        return []

    name_column = _find_column(header, _NAME_COLUMNS)
    adp_column = _find_column(header, _ADP_COLUMNS)
    if name_column is None or adp_column is None:
        return []

    position_column = _find_column(header, _POSITION_COLUMNS)
    team_column = _find_column(header, _TEAM_COLUMNS)
    stdev_column = _find_column(header, _STDEV_COLUMNS)

    records: list[dict[str, Any]] = []
    for row in reader:
        name = (row.get(name_column) or "").strip()
        if not name:
            continue
        record: dict[str, Any] = {
            "name": name,
            "position": (row.get(position_column) or "").strip().upper() if position_column else "",
            "team": (row.get(team_column) or "").strip().upper() if team_column else "",
            "adp": safe_float(row.get(adp_column)),
        }
        if stdev_column:
            record["stdev"] = safe_float(row.get(stdev_column))
        records.append(record)
    return records


def to_draft_boards(csv_text: str) -> list[list[tuple[str, int]]]:
    """Unified :mod:`fantasy.draft_ingestion` entry point for this source.

    A Draft Analyzer export is a per-player ADP table, not a pick sequence, so
    the single board returned here is that table ordered by ADP -- a
    consensus board rather than one observed draft. Returns ``[]`` when the
    CSV yields nothing usable.
    """
    records = parse_fp_analyzer_csv(csv_text)
    if not records:
        return []
    ordered = sorted(records, key=lambda record: safe_float(record.get("adp"), float("inf")))
    return [[(str(record["name"]), pick) for pick, record in enumerate(ordered, start=1)]]
