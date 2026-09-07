"""player_lookup: resolve a typed NFL name to a real player, from the full roster.

Why this exists
---------------
``modules.nfl_stats`` builds its player universe from play-by-play aggregation
and, on any provider hiccup, from a 17-row bundled snapshot of superstars. Both
paths made ordinary players unfindable: the pbp path keys rows on the nflverse
``"F.Last"`` abbreviation (users type ``"First Last"``) and mis-classifies
positions when the roster feed is down; the snapshot simply does not contain
them.

This module adds a proper identity index built from the nflverse roster feed
(``nflreadpy.load_rosters`` / ``load_players`` -- already an app dependency, no
new package). It resolves a query in four widening passes -- exact normalized
name, surname + first initial (which is what ``"J.Waddle"`` collapses to),
token overlap, then stdlib :mod:`difflib` -- and returns the canonical
``gsis_id`` plus position, team, age and experience.

It is defensive by construction: every provider call is wrapped, the index is
memoised per season (and a failed load is *not* memoised, so a transient outage
is retried rather than pinned), and :func:`lookup_player` returns ``None`` when
it genuinely cannot identify the player. ``rapidfuzz`` is deliberately not used
-- the repo ships no fuzzy-matching dependency and :mod:`difflib` is enough
once a real name index sits in front of it.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import date
from typing import Any

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")

#: season -> list[dict] identity rows. A season whose load failed is absent
#: (not stored as an empty list), so the next call retries.
_INDEX_CACHE: dict[int, list[dict[str, Any]]] = {}


def _norm(name: Any) -> str:
    """ascii, lowercase, no punctuation, suffixes stripped, single spaces."""
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = _SUFFIXES.sub("", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_initial_surname(normalized: str) -> str:
    """``"jaylen waddle"`` -> ``"j waddle"`` -- the shape an nflverse pbp key folds to."""
    parts = normalized.split()
    if len(parts) < 2:
        return normalized
    return f"{parts[0][0]} {parts[-1]}"


def _age_from_birth_date(value: Any, season: int) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            born = _parse(text, fmt)
        except ValueError:
            continue
        # nflverse ages are quoted as of Sept 1 of the season.
        anchor = date(int(season), 9, 1)
        years = anchor.year - born.year - ((anchor.month, anchor.day) < (born.month, born.day))
        return float(years) if 18 <= years <= 50 else None
    return None


def _parse(text: str, fmt: str) -> date:
    from datetime import datetime

    return datetime.strptime(text[: len(fmt) + 4], fmt).date()


def _first(row: Any, *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            try:
                import pandas as pd

                if pd.isna(row[name]):
                    continue
            except (ImportError, TypeError, ValueError):
                pass
            return row[name]
    return None


def _rows_from_frame(frame: Any, season: int) -> list[dict[str, Any]]:
    records = frame.to_dict("records") if hasattr(frame, "to_dict") else list(frame)
    out: list[dict[str, Any]] = []
    for row in records:
        position = str(_first(row, "position", "player_position", "depth_chart_position") or "").strip().upper()
        gsis = _first(row, "gsis_id", "player_id", "gsis_it_id", "nfl_id")
        name = _first(row, "full_name", "player_name", "display_name", "football_name", "player")
        if not name or not position:
            continue
        team = str(_first(row, "team", "latest_team", "recent_team", "club_code") or "").strip().upper()
        status = str(_first(row, "status", "status_description_abbr") or "").strip().upper()
        age = _first(row, "age")
        if age is None:
            age = _age_from_birth_date(_first(row, "birth_date", "birthdate"), season)
        experience = _first(row, "years_exp", "years_of_experience", "experience")
        out.append(
            {
                "player_id": str(gsis) if gsis else "",
                "name": str(name),
                "normalized": _norm(name),
                "position": position,
                "team": team,
                "status": status,
                "age": float(age) if isinstance(age, (int, float)) and 18 <= float(age) <= 50 else None,
                "years_experience": int(experience) if isinstance(experience, (int, float)) else None,
            }
        )
    return out


def _load_index(season: int) -> list[dict[str, Any]]:
    """Build (or return a memoised) identity index for ``season``.

    Tries the season roster feed first (smallest, position-correct), then the
    all-players feed. A failed load raises so it is not memoised.
    """
    if season in _INDEX_CACHE:
        return _INDEX_CACHE[season]

    import nflreadpy as nfl  # provider import kept local, like nfl_stats

    rows: list[dict[str, Any]] = []
    try:
        raw = nfl.load_rosters([season])
        frame = raw.to_pandas() if hasattr(raw, "to_pandas") else raw
        rows = _rows_from_frame(frame, season)
    except Exception:
        rows = []
    if len(rows) < 200:
        try:
            raw = nfl.load_players()
            frame = raw.to_pandas() if hasattr(raw, "to_pandas") else raw
            players_rows = _rows_from_frame(frame, season)
            # Prefer currently-teamed / active players when using the all-time feed.
            active = [r for r in players_rows if r["team"] or r["status"] in {"ACT", "A01", "RES", "CUT"}]
            rows = active or players_rows
        except Exception:
            pass
    if not rows:
        raise RuntimeError(f"no roster data available for {season}")

    # De-dupe on gsis id, keeping the first (roster feed) hit.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = row["player_id"] or row["normalized"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    _INDEX_CACHE[season] = deduped
    return deduped


def _score(query_norm: str, row: dict[str, Any]) -> float:
    """0-1 similarity of ``query_norm`` to one index row, best of a few views.

    Pure string similarity is only trusted at full strength when the two names
    also share a token (usually the surname) -- otherwise ``"zzyzx notaplayer"``
    creeps over threshold against ``"filler player17"`` on substring overlap.
    """
    candidate = row["normalized"]
    if candidate == query_norm:
        return 1.0
    q_parts, c_parts = query_norm.split(), candidate.split()
    shared_surname = bool(q_parts and c_parts and q_parts[-1] == c_parts[-1])

    # surname + first initial (the "J.Waddle" -> "j waddle" case)
    surname_initial = 0.92 if shared_surname and q_parts[0][:1] == c_parts[0][:1] else 0.0
    token_overlap = (
        len(set(q_parts) & set(c_parts)) / max(len(set(q_parts) | set(c_parts)), 1) if q_parts else 0.0
    )
    ratio = difflib.SequenceMatcher(None, query_norm, candidate).ratio()
    initials_ratio = difflib.SequenceMatcher(
        None, _first_initial_surname(query_norm), _first_initial_surname(candidate)
    ).ratio()

    # A shared FIRST name ("Michael Thomas" vs "Michael Pittman") must NOT
    # license a full-strength match -- only a shared surname does. Otherwise an
    # absent name resolves to an unrelated player instead of "unresolved".
    ratio = ratio if shared_surname else ratio * 0.55
    initials_ratio = initials_ratio * (0.92 if shared_surname else 0.5)
    return max(surname_initial, token_overlap * (1.0 if shared_surname else 0.7), ratio, initials_ratio)


def lookup_player(
    query: str | int,
    *,
    season: int | None = None,
    position_hint: str | None = None,
    min_confidence: float = 0.6,
) -> dict[str, Any] | None:
    """Resolve ``query`` (a name, or a gsis id) to a real player.

    Returns ``{"player_id", "name", "position", "team", "age",
    "years_experience", "confidence", "source", "candidates"}`` -- or ``None``
    when the roster feed is unavailable or nothing clears ``min_confidence``.
    ``candidates`` lists up to three near-misses (name + score) for a UI to
    offer. ``season`` defaults to the current calendar year's roster.
    """
    text = str(query or "").strip()
    if not text:
        return None
    season = int(season or date.today().year)

    try:
        index = _load_index(season)
    except Exception:
        # One retry against the previous season's roster -- rosters carry over.
        try:
            index = _load_index(season - 1)
        except Exception:
            return None
    if not index:
        return None

    # A raw gsis id short-circuits everything.
    if text.lower().startswith("00-") or text.isdigit():
        for row in index:
            if row["player_id"] == text:
                return {**_public(row), "confidence": 1.0, "source": "id", "candidates": []}

    query_norm = _norm(text)
    hint = str(position_hint or "").strip().upper()
    scored = sorted(
        (
            (
                _score(query_norm, row) * (1.03 if hint and row["position"] == hint else 1.0),
                row,
            )
            for row in index
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if not scored:
        return None
    best_score, best_row = scored[0]
    candidates = [{"name": row["name"], "position": row["position"], "score": round(score, 3)} for score, row in scored[1:4]]
    if best_score < min_confidence:
        return {
            "player_id": "",
            "name": text,
            "position": "",
            "team": "",
            "age": None,
            "years_experience": None,
            "confidence": round(best_score, 3),
            "source": "unresolved",
            "candidates": [{"name": best_row["name"], "position": best_row["position"], "score": round(best_score, 3)}, *candidates],
        }
    return {**_public(best_row), "confidence": round(min(best_score, 1.0), 3), "source": "roster", "candidates": candidates}


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_id": row["player_id"],
        "name": row["name"],
        "position": row["position"],
        "team": row["team"],
        "age": row["age"],
        "years_experience": row["years_experience"],
    }


def clear_cache() -> None:
    """Drop the memoised identity index (used by tests)."""
    _INDEX_CACHE.clear()
