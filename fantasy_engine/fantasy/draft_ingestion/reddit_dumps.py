"""Reddit draft-dump loader -- real parser, user-supplied content.

Usage::

    from fantasy.draft_ingestion.reddit_dumps import load_reddit_dump

    picks = load_reddit_dump(csv_or_json_text, fmt="json")

"A Reddit draft dump" is not one identifiable source, and Reddit's API has
required authentication (and, since the 2023 pricing changes, paid access for
meaningful volume) for programmatic use -- this project has neither an app
registration nor credentials for it. This module does not query Reddit at
all; it parses whatever CSV/JSON content **you** copy out of a specific post
you've already found (a comment table, a linked spreadsheet export, ...) and
supply directly.

Delegates to :func:`fantasy.draft_ingestion.github_datasets.load_github_dataset`
-- the expected shape (name-ish + pick-number-ish fields, CSV or JSON) is
identical; a distinct module exists only because the spec names Reddit as its
own source, not because the parsing logic differs.
"""

from __future__ import annotations

from fantasy.draft_ingestion.github_datasets import load_github_dataset


def load_reddit_dump(content: str, fmt: str = "json") -> list[tuple[str, int]]:
    """Parse user-supplied CSV/JSON content copied from a Reddit post.

    See :func:`fantasy.draft_ingestion.github_datasets.load_github_dataset`
    for the exact accepted shape and failure behavior (returns ``[]`` rather
    than raising on anything unparseable).
    """
    return load_github_dataset(content, fmt=fmt)


def to_draft_boards(content: str, fmt: str = "json") -> list[list[tuple[str, int]]]:
    """Unified :mod:`fantasy.draft_ingestion` entry point for this source."""
    picks = load_reddit_dump(content, fmt=fmt)
    return [picks] if picks else []
