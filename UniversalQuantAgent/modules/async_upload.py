"""Non-blocking-feeling upload ingestion for the Streamlit UI.

Honest scope note: Streamlit re-runs a page's script synchronously on each
interaction, so within a single render there is no *other* Python code
running that benefits from a parse call happening on a worker thread --
blocking on a future's ``.result()`` immediately is the same wall-clock
wait as calling the parser directly. Parsing a user's small (KB-sized)
CSV/JSON upload takes single-digit milliseconds either way; threading it
alone would not make the page feel faster.

Two things here are genuinely worth doing, so that's what this module
does:

* :func:`parse_uploads_concurrently` -- when the page has *multiple*
  independent uploads to parse (the Betting Engine page's NBA branch
  parses both a props file and a game-odds file), parsing them
  concurrently instead of one-after-another is a real, if modest, latency
  win, and it's the natural place a much larger upload would actually
  benefit from real concurrency.
* Showing a spinner while parsing runs, so the page visibly acknowledges
  the upload instead of looking frozen -- the actual perceived-responsiveness
  lever at these file sizes. That part lives in the UI page itself
  (``st.spinner``), not here, since it's a rendering concern.

See performance_notes.md for the measured reality at today's upload sizes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def parse_uploads_concurrently(tasks: dict[str, Callable[[], Any]]) -> dict[str, Any]:
    """Run each named parse callable concurrently; return ``{name: result}``.

    A task that raises propagates its exception to the caller under its
    own name's result lookup being skipped -- callers in this codebase
    already wrap their own parse calls in appropriate error handling (see
    the Betting Engine page), so this does not swallow exceptions the way
    :func:`betting.parallel_utils.parallel_map` does for batch work; a
    single named upload failing should surface, not vanish silently.
    """
    if not tasks:
        return {}
    if len(tasks) == 1:
        name, fn = next(iter(tasks.items()))
        return {name: fn()}

    with ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="upload-parse") as pool:
        futures = {name: pool.submit(fn) for name, fn in tasks.items()}
        return {name: future.result() for name, future in futures.items()}
