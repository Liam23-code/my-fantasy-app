"""Community GitHub draft-dataset loader -- real parser, user-supplied content.

Usage::

    from fantasy.draft_ingestion.github_datasets import load_github_dataset

    picks = load_github_dataset(csv_or_json_text, fmt="csv")

"A GitHub community draft dataset" is not one identifiable, stable source --
there are potentially many such repos, of widely varying quality, license,
and schema, and none was named as a specific URL. Rather than guess at (or
fabricate) a particular repo's contents, this module is a generic, defensive
CSV/JSON loader for whatever community dataset content **you** supply (e.g.
downloaded from a specific repo you've identified and vetted for license and
quality yourself).

Expected shape, same as :mod:`fantasy.draft_ingestion.underdog`: a name-ish
field and a pick-number-ish field per record, whether the source is CSV rows
or a JSON list of dicts.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

_NAME_KEYS = ("player_name", "name", "player")
_PICK_KEYS = ("pick_number", "pick", "overall_pick", "pick_no")


def _first_present(entry: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in entry.items()}
    for key in keys:
        value = lowered.get(key)
        if value not in (None, ""):
            return value
    return None


def _extract_picks(records: list[dict[str, Any]]) -> list[tuple[str, int]]:
    picks: list[tuple[str, int]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        name = _first_present(record, _NAME_KEYS)
        pick_number = _first_present(record, _PICK_KEYS)
        if not name or pick_number is None:
            continue
        try:
            picks.append((str(name).strip(), int(pick_number)))
        except (TypeError, ValueError):
            continue
    return picks


def load_github_dataset(content: str, fmt: str = "csv") -> list[tuple[str, int]]:
    """Parse user-supplied CSV or JSON draft-dataset content.

    ``fmt`` is ``"csv"`` or ``"json"``. Returns ``[]`` for empty input,
    unparseable content, an unsupported ``fmt``, or a dataset with no
    recognizable name/pick-number fields.
    """
    if not content or not content.strip():
        return []
    if fmt == "csv":
        try:
            records = list(csv.DictReader(io.StringIO(content)))
        except csv.Error:
            return []
        return _extract_picks(records)
    if fmt == "json":
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return []
        records = payload if isinstance(payload, list) else payload.get("picks", []) if isinstance(payload, dict) else []
        return _extract_picks(records)
    return []


def to_draft_boards(content: str, fmt: str = "csv") -> list[list[tuple[str, int]]]:
    """Unified :mod:`fantasy.draft_ingestion` entry point for this source."""
    picks = load_github_dataset(content, fmt=fmt)
    return [picks] if picks else []
