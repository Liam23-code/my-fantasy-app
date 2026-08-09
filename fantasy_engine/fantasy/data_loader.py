"""Real NFL player data loading for the fantasy engine.

Usage::

    from fantasy.data_loader import load_real_projections, drop_synthetic, validate_players

    players = load_real_projections(scoring_mode="ppr")
    clean, rejected = validate_players(players)

Every player this module emits is a real NFL player. Two nflverse feeds back
it, both reached through ``nflreadpy``:

* :func:`nflreadpy.load_player_stats` -- actual weekly production for a
  completed season, aggregated per player and scored through
  :mod:`fantasy.scoring` to produce the ``projection`` field.
* :func:`nflreadpy.load_ff_rankings` -- FantasyPros expert consensus ranking
  (``ecr``), used as the ADP signal that :mod:`fantasy.assistant` consumes.

A deliberate honesty note about ``projection``: it is a *baseline derived from
the most recent completed season's actual production*, not a forward-looking
projection for the upcoming season. It does not know about team changes, holdouts,
rookies, or aging curves. Every player carries ``projection_basis`` saying so, and
callers that have real forward-looking projections should pass them in instead of
calling this module. Rookies have no prior-season production and therefore do not
appear at all.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from fantasy.scoring import calculate_fantasy_points
from fantasy.utils import normalize_player_name, safe_float, safe_int

SYNTHETIC_ID_PREFIX = "synthetic"

#: Fields every draftable player must carry before the engine will use it.
REQUIRED_FIELDS: tuple[str, ...] = ("player_id", "name", "position", "team", "projection")

DRAFTABLE_POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE", "K", "DST"})

# nflverse weekly-stats column -> the canonical stat name fantasy.scoring wants.
_STAT_COLUMNS: dict[str, str] = {
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "passing_interceptions": "interceptions",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "fumbles_lost_total": "fumbles_lost",
}

_IDENTITY_COLUMNS = ("player_id", "player_display_name", "position", "team", "week")

# (season, scoring_mode, custom-rules fingerprint) -> loaded players.
_PROJECTION_MEMO: dict[tuple[Any, ...], list[dict[str, Any]]] = {}


class RealDataUnavailable(RuntimeError):
    """Raised when no real NFL data could be loaded.

    Callers should catch this and surface an empty state rather than falling
    back to fabricated players -- silently substituting synthetic data is
    exactly the failure mode this module exists to prevent.
    """


def is_synthetic(player: Any) -> bool:
    """True when a player record is engine-fabricated rather than a real NFL player.

    Checks the id first (``synthetic:qb:0``), then falls back to the display
    name (``Synthetic QB 0``) so a record that lost its id on a round-trip
    through JSON or a rename still gets caught.
    """
    if isinstance(player, dict):
        player_id = player.get("player_id") or player.get("id") or ""
        name = player.get("name") or player.get("player_name") or ""
    else:
        player_id = getattr(player, "player_id", "") or getattr(player, "id", "") or ""
        name = getattr(player, "name", "") or ""
    return str(player_id).strip().lower().startswith(SYNTHETIC_ID_PREFIX) or str(
        name
    ).strip().lower().startswith(SYNTHETIC_ID_PREFIX)


def drop_synthetic(players: list[Any]) -> list[Any]:
    """Return only the real players from ``players``."""
    return [player for player in (players or []) if not is_synthetic(player)]


def validate_players(
    players: list[dict[str, Any]],
    require_projection: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``players`` into (valid, rejected) on the required-field contract.

    A player is valid when it is not synthetic and carries a non-empty
    ``player_id``, ``name``, ``position`` (a known draftable one), ``team``,
    and -- when ``require_projection`` -- a numeric ``projection``. Each
    rejected entry is returned as ``{"player": ..., "reasons": [...]}`` so a
    caller can show the user *why* a row was dropped instead of silently
    shrinking the pool.
    """
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for player in players or []:
        if not isinstance(player, dict):
            rejected.append({"player": player, "reasons": ["not a mapping"]})
            continue

        reasons: list[str] = []
        if is_synthetic(player):
            reasons.append("synthetic player")
        if not str(player.get("player_id") or player.get("id") or "").strip():
            reasons.append("missing id")
        if not str(player.get("name") or "").strip():
            reasons.append("missing name")

        position = str(player.get("position") or "").strip().upper()
        if not position:
            reasons.append("missing position")
        elif position not in DRAFTABLE_POSITIONS:
            reasons.append(f"undraftable position {position!r}")

        if not str(player.get("team") or "").strip():
            reasons.append("missing team")

        if require_projection:
            raw = player.get("projection")
            if raw is None:
                reasons.append("missing projection")
            elif not isinstance(raw, (int, float)) or isinstance(raw, bool):
                reasons.append("non-numeric projection")

        if reasons:
            rejected.append({"player": player, "reasons": reasons})
        else:
            valid.append(player)

    return valid, rejected


def _normalize_name(name: Any) -> str:
    """Fold a display name to a join key: ascii, lowercase, no punctuation/suffix.

    Thin re-export of :func:`fantasy.utils.normalize_player_name` -- kept here
    under its original name so existing imports of this module's own
    ``_normalize_name`` keep working; the actual logic lives in one place now
    that :mod:`fantasy.draft_fusion` needs the identical join for a second,
    unrelated source.
    """
    return normalize_player_name(name)


def _require_nflreadpy() -> Any:
    try:
        import nflreadpy  # local import: optional, network-backed dependency
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RealDataUnavailable(
            "nflreadpy is not installed, so real NFL projections cannot be loaded. "
            "Install it with `pip install nflreadpy`."
        ) from error
    return nflreadpy


def latest_completed_season(today: _dt.date | None = None) -> int:
    """Best guess at the most recent NFL season with a full set of results.

    A season labelled ``Y`` starts in September of ``Y`` and finishes in
    February of ``Y+1``, so before roughly March of ``Y+1`` the season ``Y``
    is not yet complete.
    """
    today = today or _dt.date.today()
    return today.year - 1 if today.month >= 3 else today.year - 2


def load_adp(limit_position_pool: bool = True) -> dict[str, dict[str, Any]]:
    """Load FantasyPros expert-consensus draft rankings keyed by normalized name.

    Returns ``{normalized_name: {"adp": float, "adp_sd": float, "bye": int,
    "position": str, "team": str}}``. ``adp`` is the ECR (expert consensus
    rank) for a redraft league -- the lower the number, the earlier the player
    is expected to go. Returns ``{}`` rather than raising when the feed is
    unavailable, since ADP is an optional enrichment.
    """
    try:
        nflreadpy = _require_nflreadpy()
        rankings = nflreadpy.load_ff_rankings().to_pandas()
    except Exception:
        return {}

    if rankings.empty or "player" not in rankings.columns:
        return {}

    if "page_type" in rankings.columns:
        redraft = rankings[rankings["page_type"] == "redraft-overall"]
        if not redraft.empty:
            rankings = redraft

    # Keep only the newest scrape so a player appears once, not once per week.
    if "scrape_date" in rankings.columns and not rankings["scrape_date"].isna().all():
        rankings = rankings[rankings["scrape_date"] == rankings["scrape_date"].max()]

    adp: dict[str, dict[str, Any]] = {}
    for row in rankings.to_dict("records"):
        position = str(row.get("pos") or "").strip().upper()
        if limit_position_pool and position and position not in DRAFTABLE_POSITIONS:
            continue
        key = _normalize_name(row.get("player"))
        ecr = safe_float(row.get("ecr"), 0.0)
        if not key or ecr <= 0:
            continue
        # Rankings are pre-sorted best-first; keep the strongest entry per player.
        if key in adp and adp[key]["adp"] <= ecr:
            continue
        adp[key] = {
            "adp": round(ecr, 1),
            "adp_sd": round(safe_float(row.get("sd"), 0.0), 2),
            "bye_week": safe_int(row.get("bye"), 0) or None,
            "position": position,
            "team": str(row.get("team") or "").strip().upper(),
        }
    return adp


def _weekly_stats(season: int) -> Any:
    """Regular-season weekly player stats for ``season`` as a pandas DataFrame."""
    nflreadpy = _require_nflreadpy()
    try:
        frame = nflreadpy.load_player_stats(seasons=[season])
    except Exception as error:
        raise RealDataUnavailable(f"Could not load {season} NFL player stats: {error}") from error

    try:
        import polars as pl

        if isinstance(frame, pl.DataFrame):
            if "season_type" in frame.columns:
                frame = frame.filter(pl.col("season_type") == "REG")
            frame = frame.to_pandas()
    except ImportError:  # pragma: no cover - nflreadpy depends on polars
        pass

    if hasattr(frame, "columns") and "season_type" in getattr(frame, "columns", []):
        frame = frame[frame["season_type"] == "REG"]
    return frame


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _build_drivers(stats: dict[str, float]) -> list[str]:
    """Describe what actually produced a player's points, biggest first."""
    candidates = [
        (stats.get("passing_yards", 0.0), lambda v: f"{v:,.0f} passing yards"),
        (stats.get("rushing_yards", 0.0), lambda v: f"{v:,.0f} rushing yards"),
        (stats.get("receiving_yards", 0.0), lambda v: f"{v:,.0f} receiving yards"),
        (stats.get("receptions", 0.0) * 10, lambda v: f"{v / 10:,.0f} receptions"),
    ]
    drivers = [render(value) for value, render in sorted(candidates, key=lambda c: -c[0]) if value > 0]
    total_tds = (
        stats.get("passing_tds", 0.0) + stats.get("rushing_tds", 0.0) + stats.get("receiving_tds", 0.0)
    )
    if total_tds > 0:
        drivers.append(f"{total_tds:,.0f} total TDs")
    return drivers[:4]


def load_real_projections(
    season: int | None = None,
    scoring_mode: str = "ppr",
    custom_rules: dict[str, Any] | None = None,
    include_adp: bool = True,
    min_games: int = 1,
) -> list[dict[str, Any]]:
    """Load every real NFL skill player with a scored projection baseline.

    Each returned player is a canonical projection dict (the shape
    :mod:`fantasy.adapter` and :mod:`fantasy.scoring` expect) plus:

    ``projection``
        Season-total fantasy points under ``scoring_mode``, from actual
        production in ``season``.
    ``floor`` / ``median`` / ``ceiling``
        25th / 50th / 85th percentile of the player's *weekly* scores, scaled
        back to a season-length total -- real observed variance, not a guess.
    ``games_played``, ``points_per_game``
        Sample size and rate, so callers can down-weight small samples.
    ``adp``, ``adp_sd``, ``bye_week``
        Present only when the FantasyPros feed matched this player.
    ``projection_basis``
        Human-readable statement of where the number came from.

    Raises :class:`RealDataUnavailable` when no real data could be loaded.
    """
    season = season or latest_completed_season()
    memo_key = (season, scoring_mode, repr(sorted((custom_rules or {}).items())), include_adp, min_games)
    if memo_key in _PROJECTION_MEMO:
        return [dict(player) for player in _PROJECTION_MEMO[memo_key]]

    frame = _weekly_stats(season)
    if frame is None or len(frame) == 0:
        raise RealDataUnavailable(f"nflverse returned no {season} regular-season player stats.")

    available_stats = {source: canonical for source, canonical in _STAT_COLUMNS.items() if source in frame.columns}
    keep = [column for column in _IDENTITY_COLUMNS if column in frame.columns] + list(available_stats)
    frame = frame[keep]

    if "position" in frame.columns:
        frame = frame[frame["position"].isin(sorted(DRAFTABLE_POSITIONS))]

    # One row per player-week, renamed to the canonical stat vocabulary.
    weekly = frame.rename(columns=available_stats)
    canonical_stats = sorted(set(available_stats.values()))
    for column in canonical_stats:
        weekly[column] = weekly[column].fillna(0.0)

    players: list[dict[str, Any]] = []
    adp_index = load_adp() if include_adp else {}

    for player_id, rows in weekly.groupby("player_id", sort=False):
        name = str(rows["player_display_name"].iloc[-1]) if "player_display_name" in rows else ""
        position = str(rows["position"].iloc[-1]).strip().upper() if "position" in rows else ""
        # A traded player has rows for both teams; the last week is their current one.
        if "week" in rows.columns:
            rows = rows.sort_values("week")
        team = str(rows["team"].iloc[-1]).strip().upper() if "team" in rows else ""

        games_played = int(len(rows))
        if games_played < max(1, min_games):
            continue

        totals = {stat: float(rows[stat].sum()) for stat in canonical_stats}
        weekly_points = [
            calculate_fantasy_points(
                {stat: float(row[stat]) for stat in canonical_stats},
                mode=scoring_mode,
                custom_rules=custom_rules,
            )["total_points"]
            for _, row in rows.iterrows()
        ]

        season_total = calculate_fantasy_points(totals, mode=scoring_mode, custom_rules=custom_rules)
        # Season totals trip the per-game yardage bonuses (a 1,700-yard receiver
        # is not a single 100-yard game), so sum the per-week scores instead.
        projection = round(sum(weekly_points), 2)

        ordered = sorted(weekly_points)

        player: dict[str, Any] = {
            "player_id": str(player_id),
            "name": name,
            "position": position,
            "team": team,
            "season": int(season),
            **{stat: round(value, 2) for stat, value in totals.items()},
            "projection": projection,
            "expected_fantasy_points": projection,
            "games_played": games_played,
            "points_per_game": round(projection / games_played, 2) if games_played else 0.0,
            "floor": round(_percentile(ordered, 0.25) * games_played, 2),
            "median": round(_percentile(ordered, 0.50) * games_played, 2),
            "ceiling": round(_percentile(ordered, 0.85) * games_played, 2),
            "drivers": _build_drivers(totals),
            "projection_basis": f"{season} regular-season actuals ({games_played} games, {scoring_mode} scoring)",
            "scoring_mode": scoring_mode,
        }
        player["breakdown"] = season_total["breakdown"]

        match = adp_index.get(_normalize_name(name))
        if match and (not match["position"] or match["position"] == position):
            player["adp"] = match["adp"]
            player["adp_sd"] = match["adp_sd"]
            if match.get("bye_week"):
                player["bye_week"] = match["bye_week"]

        players.append(player)

    valid, _rejected = validate_players(players)
    if not valid:
        raise RealDataUnavailable(
            f"Loaded {len(players)} {season} player rows but none passed validation."
        )

    # Overlays fused_adp (+ run_pressure) onto each player's `adp` field -- the
    # one integration point that reaches room_brain, user_brain (via
    # assistant), tiering, and simulate_draft, all of which already read
    # `player["adp"]` for timing. Local import: fantasy.draft_fusion imports
    # fantasy.draft (for identify_adp_clusters), which imports this module
    # (for is_synthetic) -- a top-level import here would be a real cycle.
    # This supersedes fantasy.adp_fusion's single-source-plus-personas fusion
    # (still independently valid and tested, just no longer the live path).
    from fantasy.draft_fusion import apply_fused_draft_results

    valid = apply_fused_draft_results(valid)

    valid.sort(key=lambda p: p["projection"], reverse=True)
    _PROJECTION_MEMO[memo_key] = [dict(player) for player in valid]
    return valid


def clear_cache() -> None:
    """Drop the in-process projection memo (next load re-fetches)."""
    _PROJECTION_MEMO.clear()
