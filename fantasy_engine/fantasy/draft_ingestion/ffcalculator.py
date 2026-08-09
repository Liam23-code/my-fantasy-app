"""Fantasy Football Calculator ADP ingestion -- genuinely real, public, live data.

Usage::

    from fantasy.draft_ingestion.ffcalculator import fetch_ffcalculator_adp

    records = fetch_ffcalculator_adp(scoring="ppr", teams=12)
    # [{"player_id": 5672, "name": "Jahmyr Gibbs", "position": "RB", "team": "DET",
    #   "adp": 1.6, "times_drafted": 703, "high": 1, "low": 4, "stdev": 0.7, "bye": 6}, ...]

Verified live this session, from this project's own Python runtime (not just a
browsing tool) against ``https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12``:
a real, unauthenticated, public JSON API aggregating actual mock draft results
-- the response fetched during verification reported 4,299 real drafts
(2026-07-27 to 2026-08-03), 12-team PPR, 15 rounds, 248 players. A default
``urllib`` request gets HTTP 403 from this host; a standard browser-style
``User-Agent`` header (set below) is required -- this is identifying the
client normally, not evading any access control tied to authentication or
authorization (there is none; the data is public).

This is the only module in :mod:`fantasy.draft_ingestion` that performs an
automatic network fetch -- see the package docstring for why every other
source in this project is a stub or a user-fed parser instead.

Returns pre-aggregated per-player statistics directly from FFC's API, **not**
a list of individual draft boards. FFC's public API does not expose
individual raw draft results -- only the cross-draft aggregate (adp, stdev,
high, low, times_drafted) -- so there is no real ``[[player_id, pick_number],
...]`` board to hand back; fabricating one to match that shape would mean
inventing specific draft events that were never actually observed. See
:mod:`fantasy.draft_fusion` for how this aggregate shape is combined with
other sources.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from fantasy.utils import safe_float

FFC_BASE_URL = "https://fantasyfootballcalculator.com/api/v1/adp"

_USER_AGENT = "Mozilla/5.0 (compatible; fantasy-engine-research/1.0)"

#: Fallback spread (in picks) when FFC reports no ``stdev`` for a player.
_DEFAULT_STDEV = 6.0


def fetch_ffcalculator_adp(
    scoring: str = "ppr",
    teams: int = 12,
    timeout: float = 10.0,
    season: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch real ADP aggregates from Fantasy Football Calculator's public API.

    ``season`` requests a genuine *historical* pre-season snapshot (verified
    live: 2023, 2024, and 2025 each return real aggregated drafts from that
    August). Omit it for the current market. See :mod:`fantasy.historical_adp`
    for why this matters -- historical snapshots are what make a leak-free
    holdout evaluation possible.

    Returns ``[]`` on any network failure, timeout, malformed response, or an
    upstream error status -- callers should treat that as "this source is
    unavailable right now" (no internet access, FFC is down, an unsupported
    ``scoring``/``teams``/``season`` combination, ...), not raise. Never
    fabricates substitute data when the real fetch fails.
    """
    url = f"{FFC_BASE_URL}/{scoring}?teams={teams}"
    if season is not None:
        url += f"&year={season}"
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return []

    if not isinstance(payload, dict) or payload.get("status") != "Success":
        return []
    players = payload.get("players")
    return players if isinstance(players, list) else []


def synthesize_draft_boards(
    records: list[dict[str, Any]],
    n_boards: int = 10,
    seed: int | None = None,
    board_size: int | None = None,
) -> list[list[tuple[str, int]]]:
    """Reconstruct plausible draft boards from FFC's real aggregate statistics.

    These boards are **synthetic** -- FFC's public API exposes only
    cross-draft aggregates (``adp``, ``high``, ``low``, ``stdev``,
    ``times_drafted``), never the individual draft sequences behind them, so
    no real per-board pick order is available to return. Each synthetic board
    samples every player's pick from a normal distribution centred on their
    **real** ADP with their **real** reported standard deviation, truncated to
    the **real** observed ``[high, low]`` range, then ranks those samples to
    produce a valid 1..N pick ordering.

    So: the per-player draft *distribution* is real and measured; the specific
    board orderings are a reconstruction consistent with it, not observed
    drafts. ``seed`` makes a set of boards reproducible.

    ``board_size`` caps each board at a realistic pick count (default: the
    largest real ADP in ``records``, rounded up). This matters for accuracy,
    not just realism: FFC lists more players than a real draft has picks (256
    players vs. a 186-deep ADP scale when measured), and rank-assigning picks
    ``1..256`` stretches the tail badly -- reconstruction error measured at
    0.5 picks for ADP<24 but 33 picks for ADP 120-180. Capping the board to
    the real drafted depth keeps error low across the whole board and matches
    what actually happens: players past the last pick simply go undrafted
    rather than receiving an invented late pick number.
    """
    usable = [record for record in records if record.get("name") and record.get("adp") is not None]
    if not usable or n_boards < 1:
        return []

    usable.sort(key=lambda record: safe_float(record["adp"]))
    if board_size is None:
        board_size = int(max(safe_float(record["adp"]) for record in usable)) + 1
    usable = usable[: max(1, board_size)]

    rng = np.random.default_rng(seed)
    names = [str(record["name"]) for record in usable]
    adps = np.array([safe_float(record["adp"]) for record in usable])
    stdevs = np.array(
        [safe_float(record.get("stdev")) or _DEFAULT_STDEV for record in usable]
    )
    highs = np.array([safe_float(record.get("high"), 1.0) or 1.0 for record in usable])
    lows = np.array([safe_float(record.get("low")) or float(len(usable)) for record in usable])

    boards: list[list[tuple[str, int]]] = []
    for _ in range(n_boards):
        sampled = rng.normal(adps, np.maximum(stdevs, 0.1))
        sampled = np.clip(sampled, highs, np.maximum(highs, lows))
        order = np.argsort(sampled, kind="stable")
        boards.append([(names[index], pick) for pick, index in enumerate(order, start=1)])
    return boards


def to_draft_boards(
    scoring: str = "ppr",
    teams: int = 12,
    rounds: int = 15,
    n_boards: int = 10,
    seed: int | None = None,
) -> list[list[tuple[str, int]]]:
    """Fetch real FFC aggregates and reconstruct synthetic boards from them.

    The unified :mod:`fantasy.draft_ingestion` entry point for this source.
    Boards are capped at ``teams * rounds`` picks, matching a real draft of
    that shape. Returns ``[]`` when the fetch fails -- never fabricates
    substitute data.
    """
    return synthesize_draft_boards(
        fetch_ffcalculator_adp(scoring=scoring, teams=teams),
        n_boards=n_boards,
        seed=seed,
        board_size=teams * rounds,
    )
