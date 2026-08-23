"""Reliability-aware daily NBA prop recommendations."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Iterable

from modules.data_quality import safe_number
from modules.props import (VALID_CATEGORIES, _categories, _opponent_map,
                           _player_team_name, compare_props, prop_confidence)
from modules.projections import latest_season
from modules.reliability import get_reliability_score
from modules.sportsbook_parser import normalize_category, normalize_team_name
from modules.nba_props_loader import unified_props
from modules.nba_schedule import fetch_todays_games
from modules.parallel_utils import parallel_map

#: Each player evaluated here makes several real, independent nba_api
#: calls (compare_props -> fuse_projection, get_reliability_score) --
#: genuinely I/O-bound, so a bounded thread pool gives a real wall-clock
#: speedup (see modules/parallel_utils.py and performance_notes.md).
#: Bounded well under nba_cache.py's own retry/backoff assumptions so a
#: burst of concurrent calls doesn't look like a hammering client.
_MAX_CONCURRENT_PLAYER_FETCHES = 6


def _evaluate_one_player(
    args: tuple[tuple[str, str], set[str]], *, season: str | None, min_edge: float, min_confidence: float, min_reliability: float
) -> list[dict[str, Any]]:
    (player, opponent), found = args
    try:
        comparisons = compare_props(player, opponent, found, season)
        reliability = get_reliability_score(player, opponent, season)
    except Exception:
        return []
    best_rows = {}
    for row in comparisons:
        key = (row["player"], row["category"])
        if key not in best_rows or abs(row["edge"]) > abs(best_rows[key]["edge"]):
            best_rows[key] = row
    rows = []
    for row in best_rows.values():
        confidence = prop_confidence(row)
        reliable = safe_number(reliability["score"])
        if abs(row["edge"]) >= min_edge and confidence >= min_confidence and reliable >= min_reliability:
            score = min(100, abs(row["edge"]) * 8 + confidence * .42 + reliable * .28)
            rows.append({**row, "confidence_score": confidence,
                        "reliability_score": reliable,
                        "recommendation_score": round(score, 1),
                        "explanation": f"{row['lean'].title()} lean with {abs(row['edge']):.1f} raw edge and {reliable:.0f}/100 reliability."})
    return rows


def recommend_props(categories: Iterable[str] = VALID_CATEGORIES, min_edge: float = 1.5,
                    min_confidence: float = 55.0, season: str | None = None,
                    max_players: int = 25, min_reliability: float = 0.0
                    ) -> list[dict[str, Any]]:
    """Rank verified lines using edge, interval confidence, and reliability."""
    wanted = _categories(categories)
    lines = unified_props()
    opponents = _opponent_map(fetch_todays_games())
    groups: dict[tuple[str,str], set[str]] = defaultdict(set)
    inferred, active_season = {}, season or latest_season()
    for line in lines:
        player = line["player_name"]
        team = normalize_team_name(line.get("team",""))
        if team not in opponents:
            inferred.setdefault(player, _player_team_name(player, active_season))
            team = inferred[player]
        category = normalize_category(line.get("category",""))
        if team in opponents and category in wanted:
            groups[(player,opponents[team])].add(category)

    def _worker(args: tuple[tuple[str, str], set[str]]) -> list[dict[str, Any]]:
        return _evaluate_one_player(args, season=season, min_edge=min_edge, min_confidence=min_confidence, min_reliability=min_reliability)

    per_player_results = parallel_map(_worker, list(groups.items())[:max_players], max_workers=_MAX_CONCURRENT_PLAYER_FETCHES)
    ranked = [row for rows in per_player_results for row in rows]
    return sorted(ranked,key=lambda row:row["recommendation_score"],reverse=True)
