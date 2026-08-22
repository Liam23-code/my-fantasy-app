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
import statistics
from typing import Any

from fantasy.scoring import calculate_fantasy_points, points_allowed_score
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
    # Standard kicker scoring -- nflverse already buckets field-goal makes by
    # distance, matching fantasy.scoring.BASE_MULTIPLIERS' tiers exactly.
    "fg_made_0_19": "field_goals_0_19",
    "fg_made_20_29": "field_goals_20_29",
    "fg_made_30_39": "field_goals_30_39",
    "fg_made_40_49": "field_goals_40_49",
    "fg_made_50_59": "field_goals_50_59",
    "fg_made_60_": "field_goals_60_plus",
    "pat_made": "extra_points_made",
    # Not scored by fantasy.scoring (SCORABLE_STATS doesn't include them) --
    # carried through as season totals purely as opportunity/usage signal for
    # quant.compute_usage_rate and similarity_engine's Volume feature.
    "carries": "carries",
    "targets": "targets",
    "attempts": "pass_attempts",
}

# Per-week *rate* columns (already a 0-1 share) -- averaged across the season
# rather than summed like the counting stats in _STAT_COLUMNS above.
_SHARE_COLUMNS: dict[str, str] = {
    "target_share": "target_share",
    "air_yards_share": "air_yards_share",
}

_IDENTITY_COLUMNS = ("player_id", "player_display_name", "position", "team", "week", "opponent_team")

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


def _season_rosters(season: int) -> dict[str, dict[str, Any]]:
    """Age and experience per player, from that exact season's roster snapshot.

    Returns ``{gsis_id: {"age": float, "years_experience": int}}``. Age is
    computed as of September 1 of ``season`` (kickoff convention), from
    ``birth_date`` -- fully deterministic given ``season``, unlike a
    real-clock ``datetime.now()`` age would be. Failing soft to ``{}`` on any
    error, the same way :func:`load_adp` does: this is an optional projection
    enrichment (feeds age-curve and shrinkage logic), not a required field,
    so a missing roster feed shouldn't break player-stats loading.
    """
    try:
        nflreadpy = _require_nflreadpy()
        rosters = nflreadpy.load_rosters(seasons=[season]).to_pandas()
    except Exception:
        return {}

    if rosters.empty or "gsis_id" not in rosters.columns or "birth_date" not in rosters.columns:
        return {}

    kickoff = _dt.date(season, 9, 1)
    reference: dict[str, dict[str, Any]] = {}
    for row in rosters.to_dict("records"):
        gsis_id = str(row.get("gsis_id") or "").strip()
        birth_date = row.get("birth_date")
        if not gsis_id or birth_date is None or (hasattr(birth_date, "__class__") and str(birth_date) == "NaT"):
            continue
        try:
            born = birth_date.date() if hasattr(birth_date, "date") else birth_date
            age = (kickoff - born).days / 365.25
        except (TypeError, ValueError, AttributeError):
            continue
        if not 18.0 <= age <= 50.0:
            continue
        entry = {"age": round(age, 1)}
        years_exp = row.get("years_exp")
        if years_exp is not None:
            entry["years_experience"] = safe_int(years_exp, 0)
        reference[gsis_id] = entry
    return reference


def _season_injury_reports(season: int) -> dict[str, str]:
    """Most recent injury-report status per player for ``season``.

    Returns ``{gsis_id: "OUT"|"QUESTIONABLE"|"DOUBTFUL"}`` -- already the
    exact vocabulary :data:`projections.projection_engine.INJURY_MULTIPLIERS`
    expects. A completed season has no "current" status the way a live week
    does; the last (highest-week) report of the season is used as a health
    trajectory signal for the forward projection, which is what the ensemble
    already treats a season-total record as. Players with no report at all
    are left unset, keeping the engine's own "ACTIVE" default. Fails soft to
    ``{}`` -- an optional enrichment, like age/roster data.
    """
    try:
        nflreadpy = _require_nflreadpy()
        reports = nflreadpy.load_injuries(seasons=[season]).to_pandas()
    except Exception:
        return {}
    if reports.empty or "gsis_id" not in reports.columns or "report_status" not in reports.columns:
        return {}

    latest_week: dict[str, int] = {}
    latest_status: dict[str, str] = {}
    for row in reports.to_dict("records"):
        gsis_id = str(row.get("gsis_id") or "").strip()
        status = str(row.get("report_status") or "").strip().upper()
        week = row.get("week")
        if not gsis_id or not status or week is None:
            continue
        week = safe_int(week)
        if week >= latest_week.get(gsis_id, -1):
            latest_week[gsis_id] = week
            latest_status[gsis_id] = status
    return latest_status


def _team_names(season: int) -> dict[str, str]:
    """Team abbreviation -> full display name, e.g. ``"ARI"`` -> ``"Arizona Cardinals"``.

    Used to name a DST "player" the way FantasyPros' ADP feed labels
    defenses (full team name, not the abbreviation), so :func:`load_adp`
    matching works unchanged for DST the same way it does for real players.
    Fails soft to ``{}`` -- an optional enrichment, not a required field.
    """
    try:
        nflreadpy = _require_nflreadpy()
        teams = nflreadpy.load_teams().to_pandas()
    except Exception:
        return {}
    if teams.empty or "team_abbr" not in teams.columns or "team_name" not in teams.columns:
        return {}
    return {str(row["team_abbr"]).strip().upper(): str(row["team_name"]) for _, row in teams.iterrows()}


_DST_COUNT_FIELDS: tuple[str, ...] = (
    "def_sacks",
    "def_interceptions",
    "def_fumble_recoveries",
    "def_touchdowns",
    "def_safeties",
    "def_blocked_kicks",
)


def _season_team_defense(season: int) -> dict[str, list[dict[str, Any]]]:
    """Per-team-per-week defensive/special-teams stats and points allowed.

    Returns ``{team_abbr: [{"week", "points_allowed", "def_sacks", ...}, ...]}``,
    each team's weeks in order. Points allowed comes from the schedule (a
    team's defense allows whatever its opponent scored that game), joined
    against nflverse's per-team weekly defensive stat totals. Fails soft to
    ``{}`` -- like :func:`_season_rosters` -- so a missing feed degrades to
    no DST pool rather than breaking real-player loading.
    """
    try:
        nflreadpy = _require_nflreadpy()
        team_stats = nflreadpy.load_team_stats(seasons=[season]).to_pandas()
        schedules = nflreadpy.load_schedules(seasons=[season]).to_pandas()
    except Exception:
        return {}

    if team_stats.empty or schedules.empty:
        return {}
    if "season_type" in team_stats.columns:
        team_stats = team_stats[team_stats["season_type"] == "REG"]
    if "game_type" in schedules.columns:
        schedules = schedules[schedules["game_type"] == "REG"]

    points_allowed_by_team_week: dict[tuple[str, int], float] = {}
    for row in schedules.to_dict("records"):
        week, home, away = row.get("week"), row.get("home_team"), row.get("away_team")
        home_score, away_score = row.get("home_score"), row.get("away_score")
        if week is None or home_score is None or away_score is None:
            continue
        week = safe_int(week)
        points_allowed_by_team_week[(str(home).strip().upper(), week)] = safe_float(away_score)
        points_allowed_by_team_week[(str(away).strip().upper(), week)] = safe_float(home_score)

    defense: dict[str, list[dict[str, Any]]] = {}
    for row in team_stats.to_dict("records"):
        team = str(row.get("team") or "").strip().upper()
        week = row.get("week")
        if not team or week is None:
            continue
        week = safe_int(week)
        defense.setdefault(team, []).append(
            {
                "week": week,
                "points_allowed": points_allowed_by_team_week.get((team, week), 0.0),
                "def_sacks": safe_float(row.get("def_sacks")),
                "def_interceptions": safe_float(row.get("def_interceptions")),
                "def_fumble_recoveries": safe_float(row.get("fumble_recovery_opp")),
                "def_touchdowns": safe_float(row.get("def_tds")) + safe_float(row.get("special_teams_tds")),
                "def_safeties": safe_float(row.get("def_safeties")),
                "def_blocked_kicks": (
                    safe_float(row.get("def_pat_blocks"))
                    + safe_float(row.get("def_fg_blocks"))
                    + safe_float(row.get("def_punt_blocks"))
                ),
            }
        )
    for weeks in defense.values():
        weeks.sort(key=lambda entry: entry["week"])
    return defense


def _defense_strength_ranks(season: int) -> dict[str, int]:
    """Rank each real NFL defense 1 (stingiest) to 32 (most generous) by average points allowed.

    Reuses the same per-team-per-week points-allowed data
    :func:`_season_team_defense` already computes for DST scoring, as a
    season-long strength-of-schedule proxy for the *opposing* offense's
    matchup context -- the exact shape :mod:`projections.projection_engine`'s
    ``_matchup_multiplier`` reads (``opponent_defense_rank``). Rank 1 is the
    toughest matchup (discounts the multiplier); rank 32 the easiest.
    """
    defense = _season_team_defense(season)
    if not defense:
        return {}
    averages = [
        (team, sum(week["points_allowed"] for week in weeks) / len(weeks))
        for team, weeks in defense.items()
        if weeks
    ]
    averages.sort(key=lambda item: item[1])
    return {team: rank for rank, (team, _average) in enumerate(averages, start=1)}


def _team_schedule(schedule_season: int) -> tuple[dict[str, int], dict[str, dict[int, str]]]:
    """One season's real bye weeks and per-team weekly opponents.

    Returns ``({team: bye_week}, {team: {week: opponent_team}})``. The
    schedule is authoritative -- every real team has exactly one bye, and
    every played week has exactly one real opponent. Fails soft to
    ``({}, {})``.
    """
    try:
        nflreadpy = _require_nflreadpy()
        schedules = nflreadpy.load_schedules(seasons=[schedule_season]).to_pandas()
    except Exception:
        return {}, {}
    if schedules.empty or "game_type" not in schedules.columns:
        return {}, {}
    reg = schedules[schedules["game_type"] == "REG"]
    if reg.empty or "week" not in reg.columns:
        return {}, {}

    opponents: dict[str, dict[int, str]] = {}
    all_weeks: set[int] = set()
    for row in reg.to_dict("records"):
        week, home, away = row.get("week"), row.get("home_team"), row.get("away_team")
        if week is None or not home or not away:
            continue
        week = safe_int(week)
        all_weeks.add(week)
        home = str(home).strip().upper()
        away = str(away).strip().upper()
        opponents.setdefault(home, {})[week] = away
        opponents.setdefault(away, {})[week] = home

    byes: dict[str, int] = {}
    for team, weeks in opponents.items():
        missing = sorted(all_weeks - set(weeks))
        if len(missing) == 1:
            byes[team] = missing[0]
    return byes, opponents


def _resolve_team_schedule(source_season: int) -> tuple[dict[str, int], dict[str, dict[int, str]]]:
    """Bye weeks and opponents for the season a pool will be forward-projected into.

    A pool loaded from ``source_season`` actuals exists to be projected to
    ``source_season + 1`` (see :func:`fantasy.projections.project_forward`)
    -- bye weeks and opponents change every year, so the *next* season's
    real schedule, not the completed one the stats came from, is what a
    live "who do I face this week" view needs. Falls back to
    ``source_season`` itself only if the next season's schedule isn't
    published yet.
    """
    byes, opponents = _team_schedule(source_season + 1)
    if byes or opponents:
        return byes, opponents
    return _team_schedule(source_season)


def _build_dst_drivers(totals: dict[str, float]) -> list[str]:
    """Describe what actually produced a defense's points, biggest first."""
    contributions = [
        (totals.get("def_sacks", 0.0) * 1.0, lambda v: f"{v:,.0f} sacks"),
        (totals.get("def_interceptions", 0.0) * 2.0, lambda v: f"{v / 2:,.0f} interceptions"),
        (totals.get("def_fumble_recoveries", 0.0) * 2.0, lambda v: f"{v / 2:,.0f} fumble recoveries"),
        (totals.get("def_touchdowns", 0.0) * 6.0, lambda v: f"{v / 6:,.0f} defensive/return TDs"),
        (totals.get("points_allowed_score", 0.0), lambda v: f"{v:,.0f} pts from points-allowed defense"),
    ]
    return [render(value) for value, render in sorted(contributions, key=lambda c: -c[0]) if value > 0][:4]


def _build_dst_players(
    season: int,
    scoring_mode: str,
    custom_rules: dict[str, Any] | None,
    min_games: int,
    adp_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one canonical player-shaped record per real NFL team defense."""
    defense = _season_team_defense(season)
    if not defense:
        return []
    team_names = _team_names(season)
    bye_weeks, _team_opponents = _resolve_team_schedule(season)

    players: list[dict[str, Any]] = []
    for team, weeks in defense.items():
        games_played = len(weeks)
        if games_played < max(1, min_games):
            continue

        weekly_points: list[float] = []
        history: list[dict[str, Any]] = []
        stat_rows: list[dict[str, Any]] = []
        for week_row in weeks:
            stat_row = {field: week_row[field] for field in _DST_COUNT_FIELDS}
            stat_row["points_allowed_score"] = points_allowed_score(week_row["points_allowed"])
            stat_rows.append(stat_row)
            points = calculate_fantasy_points(stat_row, mode=scoring_mode, custom_rules=custom_rules)["total_points"]
            weekly_points.append(points)
            history.append(
                {
                    "week": week_row["week"],
                    "points": round(points, 2),
                    "points_allowed": round(week_row["points_allowed"], 1),
                }
            )

        totals = {
            field: sum(row[field] for row in stat_rows) for field in (*_DST_COUNT_FIELDS, "points_allowed_score")
        }
        projection = round(sum(weekly_points), 2)
        ordered = sorted(weekly_points)
        name = team_names.get(team, team)

        player: dict[str, Any] = {
            "player_id": f"dst:{team}",
            "name": name,
            "position": "DST",
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
            "drivers": _build_dst_drivers(totals),
            "projection_basis": f"{season} regular-season actuals ({games_played} games, {scoring_mode} scoring)",
            "scoring_mode": scoring_mode,
            "history": history,
        }

        team_bye = bye_weeks.get(team)
        if team_bye is not None:
            player["bye_week"] = team_bye

        match = adp_index.get(_normalize_name(name))
        if match and (not match["position"] or match["position"] == "DST"):
            player["adp"] = match["adp"]
            player["adp_sd"] = match["adp_sd"]
            if match.get("bye_week") and team_bye is None:
                player["bye_week"] = match["bye_week"]

        players.append(player)
    return players


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _build_drivers(stats: dict[str, float]) -> list[str]:
    """Describe what actually produced a player's points, biggest first."""
    field_goals_made = sum(
        stats.get(field, 0.0)
        for field in (
            "field_goals_0_19",
            "field_goals_20_29",
            "field_goals_30_39",
            "field_goals_40_49",
            "field_goals_50_59",
            "field_goals_60_plus",
        )
    )
    candidates = [
        (stats.get("passing_yards", 0.0), lambda v: f"{v:,.0f} passing yards"),
        (stats.get("rushing_yards", 0.0), lambda v: f"{v:,.0f} rushing yards"),
        (stats.get("receiving_yards", 0.0), lambda v: f"{v:,.0f} receiving yards"),
        (stats.get("receptions", 0.0) * 10, lambda v: f"{v / 10:,.0f} receptions"),
        (field_goals_made * 10, lambda v: f"{v / 10:,.0f} field goals made"),
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
    ``carries``, ``targets``
        Season-total opportunity counts. Not fantasy-scored (see
        ``fantasy.scoring.SCORABLE_STATS``); feed the Quant Engine's usage
        and similarity models instead.
    ``target_share``, ``air_yards_share``
        Season-average (not summed) weekly share of the team's targets/air
        yards. Present only when nflverse's weekly stats carry the column.
    ``age``, ``years_experience``
        From that season's roster snapshot. Age is computed as of September
        1 of ``season`` (kickoff), not the real clock, so it stays
        deterministic. Present only when the roster feed matched this player.
    ``history``
        ``[{"week": int, "points": float, ...}, ...]`` in chronological
        order -- the same per-week scores that built
        ``floor``/``median``/``ceiling``, carried forward instead of
        discarded, plus real per-week usage signal for skill positions and,
        when the opponent that week faced a real defense, its
        ``opponent_defense_rank``. Feeds the Quant Engine's historical-model,
        momentum/trend, waiver-trend, and matchup signals.
    ``opponent_defense_rank``
        Season average of the per-week ranks above (1 = toughest defenses
        faced, 32 = easiest) -- a strength-of-schedule proxy for
        :func:`projections.projection_engine`'s matchup model, since a
        season total has no single "next opponent". Present only when at
        least one played week matched a ranked real defense.
    ``injury_status``
        The season's most recent real injury-report designation
        (``"OUT"``/``"QUESTIONABLE"``/``"DOUBTFUL"``), a health-trajectory
        signal for the forward projection. Present only when the player had
        at least one real report; otherwise the engine's own "ACTIVE"
        default applies.
    ``schedule``
        ``[{"week", "opponent", "defense_rank"}, ...]`` -- the player's real
        team's *upcoming* season schedule (``season + 1``), not the
        completed season the stats came from, since byes and opponents
        change every year. Feeds :mod:`fantasy.weekly_projections`' live
        per-week matchup display. ``defense_rank`` still reflects the source
        season's real defensive performance (the best available proxy,
        since the target season's defense hasn't happened yet). Falls back
        to the source season's own schedule only if next season's isn't
        published yet, and is omitted entirely if neither is available.
    ``bye_week``
        Derived the same way -- the *upcoming* season's bye, not the
        completed one.

    Team defenses (``position="DST"``) are included alongside individual
    players -- one record per real NFL team, built from that team's weekly
    defensive/special-teams stats and points allowed (the schedule's
    opponent score), scored with :data:`fantasy.scoring.BASE_MULTIPLIERS`'
    standard DST rules. Named by full team name (e.g. ``"Arizona
    Cardinals"``) to match FantasyPros' ADP feed, so ``adp`` is populated
    the same way it is for real players.

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
    available_shares = [column for column in _SHARE_COLUMNS if column in frame.columns]
    keep = [column for column in _IDENTITY_COLUMNS if column in frame.columns] + list(available_stats) + available_shares
    frame = frame[keep]

    if "position" in frame.columns:
        frame = frame[frame["position"].isin(sorted(DRAFTABLE_POSITIONS))]

    # One row per player-week, renamed to the canonical stat vocabulary.
    weekly = frame.rename(columns=available_stats)
    canonical_stats = sorted(set(available_stats.values()))
    for column in canonical_stats:
        weekly[column] = weekly[column].fillna(0.0)
    for column in available_shares:
        weekly[column] = weekly[column].fillna(0.0)

    players: list[dict[str, Any]] = []
    adp_index = load_adp() if include_adp else {}
    roster_index = _season_rosters(season)
    defense_ranks = _defense_strength_ranks(season)
    injury_index = _season_injury_reports(season)
    bye_weeks, team_opponents = _resolve_team_schedule(season)

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
        week_labels = list(rows["week"]) if "week" in rows.columns else list(range(1, games_played + 1))
        # Real per-week opportunity signal -- targets/carries are already in
        # canonical_stats (summed into totals above); a raw count or share
        # here, not the season aggregate, is what quant.waiver_engine's
        # usage_trend needs to detect a rising or falling role week to week.
        usage_fields = [field for field in ("targets", "carries") if field in canonical_stats]
        weekly_points: list[float] = []
        history: list[dict[str, Any]] = []
        opponent_ranks_faced: list[int] = []
        for week, (_, row) in zip(week_labels, rows.iterrows(), strict=True):
            points = calculate_fantasy_points(
                {stat: float(row[stat]) for stat in canonical_stats},
                mode=scoring_mode,
                custom_rules=custom_rules,
            )["total_points"]
            weekly_points.append(points)
            entry: dict[str, Any] = {
                "week": int(week) if isinstance(week, (int, float)) else week,
                "points": round(points, 2),
            }
            for field in usage_fields:
                entry[field] = round(float(row[field]), 2)
            for field in available_shares:
                entry[field] = round(float(row[field]), 4)
            opponent = str(row.get("opponent_team") or "").strip().upper()
            opponent_rank = defense_ranks.get(opponent)
            if opponent_rank is not None:
                entry["opponent_defense_rank"] = opponent_rank
                opponent_ranks_faced.append(opponent_rank)
            history.append(entry)

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
            # Real per-week history, chronological, dict-shaped records:
            # {"week", "points", "targets"/"carries" (if available),
            # "target_share"/"air_yards_share" (if available)}.
            # quant_engine/trend_engine/projection_engine's historical model
            # already read "points"/"week" from history records; deliberately
            # including real per-week targets/carries/target_share here too
            # (rather than season aggregates) is what lets
            # quant.waiver_engine's usage_trend/breakout_probability detect a
            # rising or falling role week to week instead of falling back to
            # its neutral default. A points-only record would also have been
            # safe there (no usage-like key -> neutral fallback, not a
            # misread), but a real per-week signal is strictly better.
            "history": history,
        }
        player["breakdown"] = season_total["breakdown"]
        for column in available_shares:
            # A rate, not a count -- averaged across the played weeks rather
            # than summed alongside the counting stats above.
            player[column] = round(float(rows[column].mean()), 4)

        roster = roster_index.get(str(player_id))
        if roster:
            player["age"] = roster["age"]
            if "years_experience" in roster:
                player["years_experience"] = roster["years_experience"]

        if opponent_ranks_faced:
            # Season-average strength of the defenses actually faced --
            # what projections.projection_engine's ``_matchup_multiplier``
            # reads as ``opponent_defense_rank`` (1 = toughest, 32 = easiest).
            # A single season-total record has no one "next opponent", so
            # this is a strength-of-schedule proxy, not a specific week's
            # matchup -- consistent with how the rest of this ensemble
            # already treats a season total as a forward-looking baseline.
            player["opponent_defense_rank"] = round(statistics.fmean(opponent_ranks_faced), 1)

        injury_status = injury_index.get(str(player_id))
        if injury_status:
            player["injury_status"] = injury_status

        team_bye = bye_weeks.get(team)
        if team_bye is not None:
            player["bye_week"] = team_bye

        upcoming = team_opponents.get(team)
        if upcoming:
            # fantasy.weekly_projections' own container
            # ("opponent"/"defense_rank" keys) -- the team's real *upcoming*
            # schedule, not the completed season the stats came from (see
            # _resolve_team_schedule). defense_rank still comes from the
            # source season's real defensive performance -- the target
            # season's defense hasn't happened yet, so last season's is the
            # best available proxy, same convention as opponent_defense_rank
            # above.
            #
            # Deliberately dense (includes the bye week as its own
            # "opponent": "BYE" entry) rather than skipping it: that
            # consumer's own week lookup falls back to indexing the list by
            # week number when no entry's "week" matches exactly, so a
            # missing bye-week entry shifts every later week's index-based
            # lookup by one and reports a phantom opponent instead of "BYE".
            schedule_weeks = dict(upcoming)
            if team_bye is not None:
                schedule_weeks.setdefault(team_bye, "BYE")
            player["schedule"] = [
                {
                    "week": week,
                    "opponent": opponent,
                    **(
                        {
                            "defense_rank": defense_ranks[opponent],
                            # quant_engine.compute_weekly_matchup_score reads
                            # this exact key on a 0-1 scale (1.0 = toughest),
                            # a different shape than defense_rank above --
                            # same underlying data, linearly rescaled so both
                            # of this pool's matchup consumers get real signal.
                            "opponent_strength": round(
                                1.0 - (defense_ranks[opponent] - 1) / (len(defense_ranks) - 1), 4
                            ),
                        }
                        if opponent in defense_ranks and len(defense_ranks) > 1
                        else {}
                    ),
                }
                for week, opponent in sorted(schedule_weeks.items())
            ]

        match = adp_index.get(_normalize_name(name))
        if match and (not match["position"] or match["position"] == position):
            player["adp"] = match["adp"]
            player["adp_sd"] = match["adp_sd"]
            if match.get("bye_week") and team_bye is None:
                player["bye_week"] = match["bye_week"]

        players.append(player)

    players.extend(_build_dst_players(season, scoring_mode, custom_rules, min_games, adp_index))

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
