"""Shared live player-status UI: the Refresh strip and the availability badge.

Phase A of the player-status overlay is a **cosmetic** pass. Page 25 (Mock
Draft) and page 26 (Saved Teams) each grew their own copy of the "Refresh
Player Status" strip and their own badge vocabulary; rolling that onto eight
more pages by copy/paste would have meant ten copies of the same forty lines.
This module is that one component, so the strip and the badges look and read
identically everywhere.

What this module deliberately does **not** do
---------------------------------------------
Nothing here filters a table, drops a row, or rescales a number. The engines
already apply the overlay where it belongs -- ``fantasy.assistant`` and
``fantasy.grader`` drop live-OUT players, ``betting.prop_model`` drops them
from prop tables, ``fantasy.optimizer`` zeroes them in DFS lineups -- and the
pages that use this module display stored projections or historical data,
where hiding a row would lose information rather than add it. So a flagged
player still appears, with a badge next to their name saying why.

Badge source
------------
:func:`player_status_badge` reads :func:`fantasy.player_status.effective_status`
-- the live overlay when the feed knows the player, otherwise the projection
pool's own stored ``injury_status``. That is the "everything we know" read, and
it is safe *here* precisely because a badge is not a decision: nothing is
filtered or rescaled off it. Every hard rule in the engines keeps keying on
``live_status`` instead, so a stale offseason ``injury_status`` can never be
escalated into dropping a player. :func:`live_badge` is the live-only variant
for callers that want to mirror an engine's own view.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import streamlit as st
from fantasy import multi_sport_status, player_status
from fantasy.online.player_status_fetcher import refresh_player_status
from fantasy.player_status import (
    HEALTHY,
    effective_status,
    flagged_count,
    has_status_data,
    live_status,
    load_player_status,
    status_last_updated,
)

#: Canonical status -> coloured badge. The exact vocabulary pages 25/26 already
#: use, so a player reads the same on every screen.
STATUS_BADGES: dict[str, str] = {
    "OUT": "\U0001f534 OUT",
    "HOLDOUT": "\U0001f7e0 Holdout",
    "SUSPENDED": "\U0001f7e0 Suspended",
    "DOUBTFUL": "\U0001f7e1 Doubtful",
    "QUESTIONABLE": "\U0001f7e1 Questionable",
    "HEALTHY": "\U0001f7e2 Healthy",
}

#: The five non-HEALTHY statuses, in the order a summary line lists them.
RISK_STATUSES: tuple[str, ...] = ("OUT", "HOLDOUT", "SUSPENDED", "DOUBTFUL", "QUESTIONABLE")

_HEALTHY_BADGE = STATUS_BADGES[HEALTHY]


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------
def status_badge(status: Any) -> str:
    """The coloured badge for one canonical status. Unknown reads healthy."""
    return STATUS_BADGES.get(str(status or "").strip().upper(), _HEALTHY_BADGE)


def _as_player(player: Any) -> Mapping[str, Any]:
    """Accept either a player row or a bare player id."""
    if isinstance(player, Mapping):
        return player
    return {"player_id": str(player or "")}


def player_status_badge(player: Any) -> str:
    """Badge for one player from the live overlay, falling back to the pool's
    own ``injury_status`` -- see the module docstring on why that is safe here."""
    return status_badge(effective_status(_as_player(player)))


def live_badge(player: Any) -> str:
    """Badge from the live overlay only -- what the engines themselves see."""
    return status_badge(live_status(_as_player(player)))


def player_label(player: Mapping[str, Any]) -> str:
    """``name (POS)`` for a badge line, degrading to whatever the row carries."""
    name = (
        player.get("name")
        or player.get("player_name")
        or player.get("full_name")
        or player.get("player_id")
        or "Unknown player"
    )
    position = str(player.get("position") or "").strip()
    return f"{name} ({position})" if position else str(name)


# ---------------------------------------------------------------------------
# The Refresh Player Status strip
# ---------------------------------------------------------------------------
def _stamp_label(stamp: str) -> str:
    return stamp.replace("T", " ").replace("Z", " UTC")


def render_status_strip(key: str, *, hint: str = "") -> None:
    """The "Refresh Player Status" strip: last-updated stamp, flagged count,
    and the one button on the page that touches the network.

    Degrades gracefully in both directions. A failed fetch keeps the existing
    JSON untouched (:func:`refresh_player_status` already guarantees that) and
    reports a warning rather than an error; an unwritable status file raises
    ``OSError``, which is caught here so a read-only data directory cannot take
    the whole page down.

    The path is resolved from ``player_status.STATUS_PATH`` at call time rather
    than relying on the fetcher's import-time default, so a test that points
    the module at a temp file gets the refresh written to that temp file.
    """
    info_column, button_column = st.columns([3, 1])
    updated = status_last_updated()
    with info_column:
        if updated:
            st.caption(
                f"Player status overlay · {flagged_count()} player(s) flagged · "
                f"updated {_stamp_label(updated)}"
            )
        else:
            st.caption(
                "Player status overlay not loaded — "
                + (hint or "this page shows its stored projections unchanged. ")
                + "Refresh to flag OUT / doubtful / holdout / suspended players (NFL)."
            )
    with button_column:
        st.markdown('<div style="height:.2rem"></div>', unsafe_allow_html=True)
        if st.button("Refresh Player Status", key=key, width="stretch"):
            try:
                result = refresh_player_status(player_status.STATUS_PATH)
            except OSError as error:
                st.warning(f"Could not write the player status file — kept the existing data. {error}")
                return
            if result.get("ok"):
                st.success(f"Updated {result['count']} player status flag(s) from {result['source']}.")
                st.rerun()
            else:
                st.warning(
                    result.get("error") or "Could not refresh player status — kept the existing data."
                )


# ---------------------------------------------------------------------------
# Badge surfaces
# ---------------------------------------------------------------------------
def status_counts(players: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """How many of ``players`` sit at each canonical status."""
    counts: dict[str, int] = {}
    for player in players:
        if not isinstance(player, Mapping):
            continue
        status = effective_status(player)
        counts[status] = counts.get(status, 0) + 1
    return counts


def availability_summary(counts: Mapping[str, int], total: int) -> str:
    """One caption summarising a badge list: the flag breakdown, or all-clear."""
    if not has_status_data():
        return (
            "Refresh player status above to check live availability — "
            "badges read from stored data until then."
        )
    flagged = sum(counts.get(status, 0) for status in RISK_STATUSES)
    if not flagged:
        return f"{_HEALTHY_BADGE} — no availability flags on any of these {total} player(s)."
    breakdown = ", ".join(
        f"{counts[status]} {status.lower()}" for status in RISK_STATUSES if counts.get(status)
    )
    return f"⚠️ Availability: {flagged} of {total} flagged — {breakdown}."


def render_player_badges(
    players: Iterable[Mapping[str, Any]],
    *,
    limit: int | None = None,
    empty_note: str = "No players to check.",
    extra: Any = None,
) -> None:
    """A markdown list of ``badge **Name** (POS)`` lines plus a summary caption.

    Purely additive: every player handed in is listed, flagged or not. ``extra``
    is an optional ``player -> str`` callback appending a trailing detail (a
    stored projection, an edge) to each line.
    """
    rows = [player for player in players if isinstance(player, Mapping)]
    if not rows:
        st.caption(empty_note)
        return
    shown = rows if limit is None else rows[:limit]
    lines = []
    for player in shown:
        suffix = f" · {extra(player)}" if extra is not None else ""
        lines.append(f"- {player_status_badge(player)} **{player_label(player)}**{suffix}")
    st.markdown("\n".join(lines))
    if limit is not None and len(rows) > len(shown):
        st.caption(f"Showing the first {len(shown)} of {len(rows)} players.")
    st.caption(availability_summary(status_counts(rows), len(rows)))


def render_flagged_watchlist(empty_note: str = "", *, limit: int | None = 50) -> None:
    """Every flagged player in the current overlay, badged.

    For pages with no player rows of their own (a team-level slate, a hub).
    The fetcher only writes non-HEALTHY records, so an empty overlay is the
    all-clear rather than a gap -- and says so with the healthy badge.

    ``limit`` caps the printed list. A real preseason pull runs to several
    hundred names, which as one flat list buries whatever the page is actually
    about; the count line below always reports the full total.
    """
    records = load_player_status()
    by_name: dict[str, str] = {}
    for key, record in records.items():
        if not isinstance(record, Mapping):
            continue
        status = str(record.get("status") or "").strip().upper()
        if status in ("", HEALTHY):
            continue
        name = str(record.get("name") or key).strip()
        by_name.setdefault(name, status)
    if not by_name:
        st.markdown(f"- {_HEALTHY_BADGE} — no players are flagged in the current status overlay.")
        st.caption(
            empty_note
            or "Refresh player status above to pull live OUT / doubtful / holdout / suspended flags (NFL)."
        )
        return
    ordered = sorted(
        by_name.items(),
        key=lambda item: (RISK_STATUSES.index(item[1]) if item[1] in RISK_STATUSES else 9, item[0]),
    )
    shown = ordered if limit is None else ordered[:limit]
    st.markdown("\n".join(f"- {status_badge(status)} **{name}**" for name, status in shown))
    if len(shown) < len(ordered):
        st.caption(f"Showing the first {len(shown)} of {len(ordered)} flagged players.")
    counts: dict[str, int] = {}
    for _name, status in ordered:
        counts[status] = counts.get(status, 0) + 1
    breakdown = ", ".join(
        f"{counts[status]} {status.lower()}" for status in RISK_STATUSES if counts.get(status)
    )
    st.caption(f"⚠️ Availability: {len(ordered)} player(s) flagged league-wide — {breakdown}.")


# ===========================================================================
# Multi-sport (MLB / NHL / NBA / CFB / CBB) -- the same strip and badges over
# ``fantasy.multi_sport_status`` instead of the NFL-only ``fantasy.player_status``.
# Everything above is unchanged; nothing here touches the NFL path.
# ===========================================================================

#: Sport code (any case) -> the ``player_status_fetcher_<sport>`` module name.
_MULTI_SPORT_CODES: tuple[str, ...] = ("mlb", "nhl", "nba", "cfb", "cbb")


def _multi_sport_refresher(sport: str) -> Callable[..., dict[str, Any]]:
    """Lazily import the one fetcher for ``sport``. Kept lazy so importing this
    module (every page does) never pulls in five feed modules it may not use."""
    code = str(sport or "").strip().lower()
    if code not in _MULTI_SPORT_CODES:
        raise ValueError(f"unknown multi-sport code {sport!r} (expected one of {_MULTI_SPORT_CODES})")
    module = __import__(f"fantasy.online.player_status_fetcher_{code}", fromlist=["refresh_player_status"])
    return module.refresh_player_status  # type: ignore[no-any-return]


def multi_sport_status_badge(sport: str, player: Any) -> str:
    """Badge for one player from the multi-sport live overlay, falling back to the
    row's own ``status`` / ``injury_status`` -- the "everything we know" read
    (safe here because a badge is not a decision; see the module docstring)."""
    return status_badge(multi_sport_status.effective_status(sport, _as_player(player)))


def multi_sport_live_badge(sport: str, player: Any) -> str:
    """Badge from the multi-sport live overlay only -- what the engines see."""
    return status_badge(multi_sport_status.live_status(sport, _as_player(player)))


def render_multi_sport_status_strip(sport: str, key: str, *, hint: str = "") -> None:
    """The "Refresh Player Status" strip for one non-NFL sport: last-updated
    stamp, flagged count, and the one button on the page that touches the
    network. Byte-for-byte the same UX as :func:`render_status_strip`, reading
    ``fantasy.multi_sport_status`` and calling that sport's ESPN fetcher.

    Degrades gracefully in both directions: a failed fetch keeps the existing
    JSON untouched and warns rather than errors; an unwritable file raises
    ``OSError``, caught here. The path is resolved from
    ``multi_sport_status.STATUS_PATH`` at call time so a test pointing the module
    at a temp file gets the refresh written there.
    """
    code = str(sport or "").strip().lower()
    label = code.upper()
    info_column, button_column = st.columns([3, 1])
    updated = multi_sport_status.status_last_updated(code)
    with info_column:
        if updated:
            st.caption(
                f"{label} player status overlay · {multi_sport_status.flagged_count(code)} player(s) flagged · "
                f"updated {_stamp_label(updated)}"
            )
        else:
            st.caption(
                f"{label} player status overlay not loaded — "
                + (hint or "this page prices the odds file exactly as loaded. ")
                + "Refresh to flag OUT / doubtful / holdout / suspended players."
            )
    with button_column:
        st.markdown('<div style="height:.2rem"></div>', unsafe_allow_html=True)
        if st.button("Refresh Player Status", key=key, width="stretch"):
            try:
                result = _multi_sport_refresher(code)(multi_sport_status.STATUS_PATH)
            except OSError as error:
                st.warning(f"Could not write the player status file — kept the existing data. {error}")
                return
            if result.get("ok"):
                st.success(f"Updated {result['count']} {label} player status flag(s) from {result['source']}.")
                st.rerun()
            else:
                st.warning(
                    result.get("error") or "Could not refresh player status — kept the existing data."
                )


def multi_sport_status_counts(sport: str, players: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """How many of ``players`` sit at each canonical status, for ``sport``."""
    counts: dict[str, int] = {}
    for player in players:
        if not isinstance(player, Mapping):
            continue
        status = multi_sport_status.effective_status(sport, player)
        counts[status] = counts.get(status, 0) + 1
    return counts


def multi_sport_availability_summary(sport: str, counts: Mapping[str, int], total: int) -> str:
    """One caption summarising a multi-sport badge list: the flag breakdown, or
    all-clear -- the sport-namespaced twin of :func:`availability_summary`."""
    if not multi_sport_status.has_status_data(sport):
        return (
            f"Refresh {str(sport).upper()} player status above to check live availability — "
            "badges read from the loaded data until then."
        )
    flagged = sum(counts.get(status, 0) for status in RISK_STATUSES)
    if not flagged:
        return f"{_HEALTHY_BADGE} — no availability flags on any of these {total} player(s)."
    breakdown = ", ".join(
        f"{counts[status]} {status.lower()}" for status in RISK_STATUSES if counts.get(status)
    )
    return f"⚠️ Availability: {flagged} of {total} flagged — {breakdown}."


def render_multi_sport_player_badges(
    sport: str,
    players: Iterable[Mapping[str, Any]],
    *,
    limit: int | None = None,
    empty_note: str = "No players to check.",
    extra: Any = None,
) -> None:
    """A markdown list of ``badge **Name** (POS)`` lines plus a summary caption,
    for one non-NFL sport. Purely additive: every player handed in is listed,
    flagged or not -- the sport-namespaced twin of :func:`render_player_badges`."""
    rows = [player for player in players if isinstance(player, Mapping)]
    if not rows:
        st.caption(empty_note)
        return
    shown = rows if limit is None else rows[:limit]
    lines = []
    for player in shown:
        suffix = f" · {extra(player)}" if extra is not None else ""
        lines.append(f"- {multi_sport_status_badge(sport, player)} **{player_label(player)}**{suffix}")
    st.markdown("\n".join(lines))
    if limit is not None and len(rows) > len(shown):
        st.caption(f"Showing the first {len(shown)} of {len(rows)} players.")
    st.caption(multi_sport_availability_summary(sport, multi_sport_status_counts(sport, rows), len(rows)))
