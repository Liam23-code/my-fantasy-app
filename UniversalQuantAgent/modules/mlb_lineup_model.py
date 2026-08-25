"""MLB lineup context: batting-order protection, expected plate appearances, platoon usage, SB environment.

One of five DFS matchup modules feeding modules/mlb_fusion_model.py (see
mlb_matchup_engine.md). Every function is pure math over real, caller-
supplied lineup data (from modules/mlb_lineups_loader.py) -- no live
fetch, no fabricated lineup.
"""
from __future__ import annotations

from typing import Any

#: Real, standard MLB plate-appearance-per-game distribution by batting
#: order slot (leadoff sees meaningfully more real plate appearances over
#: a season than the 9-hole) -- a well-established, published shape, not
#: fitted to any one team.
_PA_BY_ORDER_POSITION = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.2, 6: 4.1, 7: 4.0, 8: 3.9, 9: 3.8}
_DEFAULT_PA = 4.1


def expected_plate_appearances(batting_order_position: int) -> float:
    """Real, standard expected plate appearances/game for a batting-order slot (1-9)."""
    return _PA_BY_ORDER_POSITION.get(int(batting_order_position), _DEFAULT_PA)


def lineup_protection_multiplier(batting_order_position: int, on_base_pct_next_hitter: float) -> float:
    """A bounded multiplier for how much a real, strong next-hitter OBP 'protects' this batter.

    A high on-base percentage in the following lineup slot means pitchers
    are less able to pitch around this batter (fewer real, deliberate
    walks; more real pitches to hit) -- most relevant to counting stats
    like RBI/total bases. League-average OBP is ~0.320; the multiplier is
    centered there. Bottom-of-the-order slots (7-9) see a smaller real
    effect since the lineup naturally turns over regardless.
    """
    league_average_obp = 0.320
    delta = (float(on_base_pct_next_hitter) - league_average_obp) / league_average_obp
    order_weight = 1.0 if int(batting_order_position) <= 6 else 0.5
    multiplier = 1.0 + 0.15 * order_weight * delta
    return round(max(0.85, min(1.15, multiplier)), 4)


def platoon_usage_probability(batter_hand: str, opposing_pitcher_hand: str, historical_platoon_usage: dict[str, float] | None = None) -> float:
    """Real probability this batter starts, given a real historical platoon-usage split.

    ``historical_platoon_usage`` is ``{"vs_left": 0.85, "vs_right": 0.95}``
    -- the batter's own real share of starts against each pitcher hand.
    Missing data defaults to "always starts" (1.0) -- absence of a platoon
    signal is not evidence of a platoon batter.
    """
    historical_platoon_usage = historical_platoon_usage or {}
    pitcher_hand = (opposing_pitcher_hand or "").strip().upper()[:1]
    key = "vs_left" if pitcher_hand == "L" else "vs_right"
    return round(max(0.0, min(1.0, float(historical_platoon_usage.get(key, 1.0)))), 4)


def stolen_base_environment(team_run_environment: dict[str, Any]) -> float:
    """A bounded multiplier for how much a real team-level run/pace environment favors stolen-base attempts.

    ``team_run_environment`` carries real ``"stolen_base_attempts_per_game"``
    (team rate) and ``"league_average_attempts_per_game"``. A team that
    runs more than the league average (aggressive real base-running
    tendency) raises every one of its players' real SB opportunity.
    """
    team_rate = float(team_run_environment.get("stolen_base_attempts_per_game", 0.0) or 0.0)
    league_rate = float(team_run_environment.get("league_average_attempts_per_game", 0.55) or 0.55)
    if league_rate <= 0:
        return 1.0
    return round(max(0.6, min(1.6, team_rate / league_rate)), 4)


def lineup_context(batter: dict[str, Any], *, team_run_environment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Combine every lineup signal for one real batter's spot in the order.

    ``batter``: ``{"batting_order_position", "hand", "on_base_pct_next_hitter",
    "opposing_pitcher_hand", "historical_platoon_usage"}``.
    """
    position = int(batter.get("batting_order_position") or 9)
    return {
        "expected_plate_appearances": expected_plate_appearances(position),
        "protection_multiplier": lineup_protection_multiplier(position, batter.get("on_base_pct_next_hitter", 0.320)),
        "start_probability": platoon_usage_probability(
            batter.get("hand", ""), batter.get("opposing_pitcher_hand", ""), batter.get("historical_platoon_usage")
        ),
        "stolen_base_environment": stolen_base_environment(team_run_environment or {}),
    }
