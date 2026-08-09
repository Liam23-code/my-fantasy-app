"""FFPC draft board HTML parser -- real parser, user-supplied page content.

Usage::

    from fantasy.draft_ingestion.ffpc import parse_ffpc_draft_board_html

    picks = parse_ffpc_draft_board_html(html_text)
    # [("Jahmyr Gibbs", 3), ("Ja'Marr Chase", 1), ...]  (name, pick_number)

Checked live this session: FFPC's (myffpc.com) draft results are behind
account login -- there is no public page to fetch automatically. This module
parses an HTML table **you** supply (e.g. saved from your own logged-in
session) rather than fetching anything itself.

Uses only :mod:`html.parser` from the standard library -- no new dependency
(``beautifulsoup4`` is not in this project's ``pyproject.toml``) for a parser
whose only job is reading a table out of HTML the caller already has. Expects
one ``<table>`` with a header row and, for each data row, a player-name cell
and a numeric pick-number cell -- column order and header spelling aren't
assumed beyond "contains a recognizable header token", since the exact page
markup was not verified live (the page requires login).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_NAME_HEADER_TOKENS = ("player", "name")
_PICK_HEADER_TOKENS = ("pick", "overall")


class _DraftTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._current_row is not None and self._current_cell is not None:
            self._current_row.append("".join(self._current_cell).strip())
            self._in_cell = False
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._current_cell is not None:
            self._current_cell.append(data)


def parse_ffpc_draft_board_html(html_text: str) -> list[tuple[str, int]]:
    """Parse an FFPC draft-board HTML table into ``(player_name, pick_number)`` pairs.

    Returns ``[]`` for empty input or a table whose header doesn't contain a
    recognizable name column and pick-number column -- there's nothing
    reliably extractable in that case. Rows that don't yield both a name and
    a parseable integer pick number are skipped individually.
    """
    if not html_text or not html_text.strip():
        return []

    parser = _DraftTableParser()
    parser.feed(html_text)
    rows = [row for row in parser.rows if row]
    if len(rows) < 2:
        return []

    header = [cell.strip().lower() for cell in rows[0]]
    name_index = next((i for i, cell in enumerate(header) if any(token in cell for token in _NAME_HEADER_TOKENS)), None)
    pick_index = next((i for i, cell in enumerate(header) if any(token in cell for token in _PICK_HEADER_TOKENS)), None)
    if name_index is None or pick_index is None:
        return []

    picks: list[tuple[str, int]] = []
    for row in rows[1:]:
        if len(row) <= max(name_index, pick_index):
            continue
        name = row[name_index].strip()
        pick_match = re.search(r"\d+", row[pick_index])
        if not name or not pick_match:
            continue
        picks.append((name, int(pick_match.group())))
    return picks


def to_draft_boards(html_text: str) -> list[list[tuple[str, int]]]:
    """Unified :mod:`fantasy.draft_ingestion` entry point for this source.

    One supplied HTML page is one real draft board. Returns ``[]`` when the
    page yields no usable rows.
    """
    picks = parse_ffpc_draft_board_html(html_text)
    return [picks] if picks else []
