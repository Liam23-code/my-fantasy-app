"""Projection adapter: turn anything into the canonical projection dict.

This is the one seam where the fantasy engine talks to an outside projection
source (e.g. the NFL analytics repo's ``project_nfl_player``). Everything
downstream of :func:`normalize_projection` -- scoring, draft ranking,
optimization, waivers, trades -- only ever sees the canonical shape, so none
of it needs to know or care where a projection came from.

Recognized input shapes:

1. **``project_nfl_player`` output** -- a dict with a nested ``"projection"``
   dict plus top-level identity (``player``, ``position``, ``team`` ...) and
   a ``"confidence"`` dict with ``low``/``high``. See
   ``UniversalQuantAgent/modules/nfl_projections.py``.
2. **Flat canonical dict** -- already matches (a superset of) the schema in
   :class:`fantasy.models.CanonicalProjection`.
3. **Minimal/legacy stat dict** -- uses short field names such as
   ``pass_yards``, ``rush_tds``, ``rec_yards``, ``ints`` (the naming used by
   ``modules/nfl_stats.py`` in the sibling repo).
4. **A player object** -- anything with attributes instead of dict keys
   (a dataclass, a pydantic model, a plain object). Converted via
   ``vars()``/``model_dump()`` before the same alias resolution runs.
5. **A bare player id/name (str)** -- requires a ``loader`` callable that
   turns the id/name into one of the above shapes (typically
   ``project_nfl_player`` itself, or ``functools.partial(project_nfl_player,
   opponent_team=..., season=...)``).

Nothing here executes code from the input; every field is looked up by a
fixed alias list and coerced with :func:`fantasy.utils.safe_float`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, overload

from fantasy.models import CanonicalProjection
from fantasy.utils import safe_float

IDENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "player_id": ("player_id", "id", "playerId"),
    "name": ("name", "player", "player_name"),
    "position": ("position", "player_position"),
    "team": ("team", "nfl_team"),
    "opponent": ("opponent", "opp", "opponent_team"),
    "season": ("season",),
}

STAT_ALIASES: dict[str, tuple[str, ...]] = {
    "passing_yards": ("passing_yards", "passing_yards_projection", "pass_yards"),
    "passing_tds": ("passing_tds", "passing_tds_projection", "pass_tds"),
    "interceptions": ("interceptions", "interceptions_projection", "ints"),
    "rushing_yards": ("rushing_yards", "rushing_yards_projection", "rush_yards"),
    "rushing_tds": ("rushing_tds", "rushing_tds_projection", "rush_tds"),
    "receptions": ("receptions", "receptions_projection", "receiving_receptions"),
    "receiving_yards": ("receiving_yards", "receiving_yards_projection", "rec_yards"),
    "receiving_tds": ("receiving_tds", "receiving_tds_projection", "rec_tds"),
    "fumbles_lost": ("fumbles_lost", "fumbles", "fumbles_lost_projection"),
}

DISTRIBUTION_ALIASES: dict[str, tuple[str, ...]] = {
    "expected_fantasy_points": ("expected_fantasy_points", "fantasy_points_projection", "fantasy_points"),
    "floor": ("floor",),
    "median": ("median",),
    "ceiling": ("ceiling",),
}

# Not part of the strict CanonicalProjection schema (section 4 of the spec),
# but draft/waiver rationale needs them when a source happens to provide
# them, so they are carried through on the plain-dict path. Silently dropped
# by CanonicalProjection's extra="ignore" when as_model=True.
EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "injury_status": ("injury_status", "status", "injury"),
    "bye_week": ("bye_week",),
    "ownership_pct": ("ownership_pct", "percent_owned", "owned_pct"),
    "adp": ("adp", "average_draft_position"),
    "schedule_difficulty": ("schedule_difficulty", "strength_of_schedule", "sos"),
}

ALL_ALIASES: dict[str, tuple[str, ...]] = {**IDENTITY_ALIASES, **STAT_ALIASES, **DISTRIBUTION_ALIASES, **EXTRA_ALIASES}


def _to_mapping(source: Any) -> dict[str, Any]:
    """Best-effort conversion of any object-like input into a plain dict."""
    if isinstance(source, Mapping):
        return dict(source)
    if hasattr(source, "model_dump") and callable(source.model_dump):
        return dict(source.model_dump())
    if hasattr(source, "_asdict") and callable(source._asdict):  # namedtuple
        return dict(source._asdict())
    if hasattr(source, "__dict__"):
        return dict(vars(source))
    raise TypeError(
        f"Cannot normalize projection of type {type(source).__name__}; expected a dict, a project_nfl_player()-shaped mapping, or an object with attributes."
    )


def _first_present(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in row and row[alias] is not None:
            return row[alias]
    return None


def _looks_like_project_nfl_player_output(row: dict[str, Any]) -> bool:
    projection = row.get("projection")
    return isinstance(projection, Mapping) and ("position" in row or "player" in row)


def _resolve_identity(row: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for field, aliases in IDENTITY_ALIASES.items():
        value = _first_present(row, aliases)
        identity[field] = value
    identity["season"] = int(safe_float(identity.get("season"), 0.0))
    identity["player_id"] = str(identity.get("player_id") or "")
    identity["name"] = str(identity.get("name") or "")
    identity["position"] = str(identity.get("position") or "").strip().upper()
    identity["team"] = str(identity.get("team") or "").strip().upper()
    identity["opponent"] = str(identity.get("opponent") or "").strip().upper()
    return identity


def _resolve_stats(row: dict[str, Any]) -> dict[str, float]:
    return {field: safe_float(_first_present(row, aliases), 0.0) for field, aliases in STAT_ALIASES.items()}


def _resolve_distribution(row: dict[str, Any]) -> dict[str, float | None]:
    resolved: dict[str, float | None] = {}
    for field, aliases in DISTRIBUTION_ALIASES.items():
        raw = _first_present(row, aliases)
        resolved[field] = None if raw is None else safe_float(raw)
    return resolved


def _resolve_extras(row: dict[str, Any]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for field, aliases in EXTRA_ALIASES.items():
        raw = _first_present(row, aliases)
        if field in {"bye_week"}:
            extras[field] = int(safe_float(raw)) if raw is not None else None
        elif field in {"ownership_pct", "adp", "schedule_difficulty"}:
            extras[field] = safe_float(raw) if raw is not None else None
        else:
            extras[field] = str(raw).strip().upper() if raw is not None else None
    return extras


@overload
def normalize_projection(source: Any, loader: Callable[[Any], Any] | None = None, as_model: Literal[False] = False) -> dict[str, Any]: ...
@overload
def normalize_projection(source: Any, loader: Callable[[Any], Any] | None = None, *, as_model: Literal[True]) -> CanonicalProjection: ...
def normalize_projection(
    source: Any,
    loader: Callable[[Any], Any] | None = None,
    as_model: bool = False,
) -> dict[str, Any] | CanonicalProjection:
    """Normalize any supported input into the canonical projection shape.

    Parameters
    ----------
    source:
        A dict, ``project_nfl_player()``-shaped dict, player object, or a
        bare id/name (string) to be resolved via ``loader``.
    loader:
        Called as ``loader(source)`` when ``source`` is a ``str``/``int`` and
        expected to return one of the richer shapes. Typically
        ``project_nfl_player`` from the NFL analytics repo.
    as_model:
        Return a validated :class:`fantasy.models.CanonicalProjection`
        instead of a plain dict.

    Examples
    --------
    >>> normalize_projection({"name": "Puka Nacua", "position": "wr", "rec_yards": 92})["receiving_yards"]
    92.0

    >>> from functools import partial
    >>> # normalize_projection("Lamar Jackson", loader=partial(project_nfl_player, opponent_team="CLE"))
    """
    if isinstance(source, (str, int)) and not isinstance(source, bool):
        if loader is None:
            raise ValueError(
                f"normalize_projection() received a bare id/name ({source!r}) with no loader; pass loader=project_nfl_player (or similar) to resolve it."
            )
        source = loader(source)

    row = _to_mapping(source)

    if _looks_like_project_nfl_player_output(row):
        stats_row = dict(row["projection"])
        identity_row = row
        confidence = row.get("confidence") or {}
        drivers = row.get("drivers") or []
    else:
        stats_row = row
        identity_row = row
        confidence = {}
        drivers = row.get("drivers") or []

    identity = _resolve_identity(identity_row)
    stats = _resolve_stats(stats_row)
    distribution = _resolve_distribution(stats_row)
    extras = _resolve_extras(identity_row)

    if distribution["floor"] is None and "low" in confidence:
        distribution["floor"] = safe_float(confidence.get("low"))
    if distribution["ceiling"] is None and "high" in confidence:
        distribution["ceiling"] = safe_float(confidence.get("high"))
    if distribution["median"] is None and distribution["expected_fantasy_points"] is not None:
        distribution["median"] = distribution["expected_fantasy_points"]

    canonical = {
        **identity,
        **stats,
        **distribution,
        **extras,
        "drivers": [str(item) for item in drivers] if isinstance(drivers, (list, tuple)) else [],
    }

    if as_model:
        return CanonicalProjection(**canonical)
    return canonical


def normalize_projections(
    sources: list[Any],
    loader: Callable[[Any], Any] | None = None,
) -> list[dict[str, Any]]:
    """Batch-normalize a list of projection sources (mixed shapes are fine)."""
    return [normalize_projection(source, loader=loader) for source in sources]
