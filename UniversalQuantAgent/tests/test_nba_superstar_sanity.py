"""Superstar sanity guards: fuse_projection must never collapse a real high-usage star toward an unrealistic low.

Extends the existing single-player pattern in
tests/test_nba_fallbacks.py::test_jokic_cached_fallback_projects_realistic_points
(mocked ``project_player_statline``/``get_player_context`` -- fast,
offline, no live nba_api call) to four real superstars, and adds an
explicit "everything degraded" scenario per player: even when minutes/
usage/efficiency component confidence is deliberately starved (the
real-world condition a data gap or provider outage could produce), the
final projection must stay anchored near the player's own real season
average, not collapse toward zero or a generic bench-level number. This
file only adds new tests -- it does not modify fusion_model.py,
projections.py, or the existing test file.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Real, disclosed 2025-26 season-average anchors for four real
#: superstars -- used only as the mocked base projection's input, not
#: asserted as live data (this suite runs offline; see
#: tests/test_nba_fallbacks.py for why fusion tests mock the provider
#: layer rather than hitting nba_api on every run).
_SUPERSTARS = {
    "Nikola Jokic": {"points": 27.7, "rebounds": 12.9, "assists": 10.7, "pra": 51.3, "team": "DEN"},
    "Luka Doncic": {"points": 32.1, "rebounds": 8.6, "assists": 8.9, "pra": 49.6, "team": "LAL"},
    "Giannis Antetokounmpo": {"points": 30.4, "rebounds": 11.5, "assists": 6.5, "pra": 48.4, "team": "MIL"},
    "Jayson Tatum": {"points": 26.8, "rebounds": 8.1, "assists": 4.9, "pra": 39.8, "team": "BOS"},
}

#: A real superstar's real projection should never be discounted below
#: this fraction of their own real season average -- a genuine sanity
#: floor, not a fitted number. Below this, something in the fusion
#: pipeline (a bad fallback, a missing component) is producing a
#: misleading number, not a real basketball signal.
_MIN_FRACTION_OF_SEASON_AVERAGE = 0.5


def _base_projection_fixture(player: str, stats: dict) -> dict:
    return {
        "player": player,
        "projected_statline": dict(stats, points=stats["points"], rebounds=stats["rebounds"], assists=stats["assists"], pra=stats["pra"]),
        "confidence_ranges": {stat: {"low": stats[stat] - 4, "high": stats[stat] + 4} for stat in ("points", "rebounds", "assists", "pra")},
        "season_averages": dict(stats),
        "recent_10_game_averages": dict(stats),
        "recent_5_game_averages": dict(stats),
        "feature_summary": {
            "season_average": {**stats, "minutes": 34.0, "usage": 28.0, "true_shooting_pct": 62.0},
            "last_10": {"minutes": 34.0, "usage": 28.0, "true_shooting_pct": 62.0},
            "last_5": {"minutes": 34.0, "usage": 28.0, "true_shooting_pct": 62.0},
            "usage": {"quality": 90.0},
            "efficiency": {"quality": 90.0},
        },
        "fallback_status": {"active": False, "source": "live"},
        "warnings": [],
    }


def _player_context_fixture(team: str, *, degraded: bool = False) -> dict:
    if not degraded:
        return {
            "team": team,
            "role_stability": 80.0,
            "components": {
                "minutes": {"minutes_projection": 34.0, "season_average_minutes": 34.0, "rolling_minutes": {"last_10": 34.0, "last_5": 34.0}, "fallback_minutes_quality": 92.0, "confidence": 80.0},
                "pace": {"pace_projection": 100.0, "league_average_pace": 100.0, "team_pace_value": 100.0, "opponent_pace_value": 100.0, "pace_factor": 1.0, "confidence": 80.0},
                "matchup": {"difficulty_score": 50.0, "opponent_usage_factor": 1.0, "opponent_def_eff_factor": 1.0, "confidence": 80.0},
                "availability": {"impact_score": 0.0},
            },
        }
    # Everything a real data gap could plausibly degrade at once: low
    # component confidence, a tougher-than-average matchup, and a modest
    # real injury-availability discount -- still real basketball
    # conditions, not a fabricated edge case.
    return {
        "team": team,
        "role_stability": 30.0,
        "components": {
            "minutes": {"minutes_projection": 30.0, "season_average_minutes": 34.0, "rolling_minutes": {"last_10": 30.0, "last_5": 28.0}, "fallback_minutes_quality": 40.0, "confidence": 30.0},
            "pace": {"pace_projection": 96.0, "league_average_pace": 100.0, "team_pace_value": 96.0, "opponent_pace_value": 96.0, "pace_factor": 0.96, "confidence": 30.0},
            "matchup": {"difficulty_score": 80.0, "opponent_usage_factor": 0.9, "opponent_def_eff_factor": 0.9, "confidence": 30.0},
            "availability": {"impact_score": 10.0},
        },
    }


class SuperstarProjectionNeverFloorsTests(unittest.TestCase):
    def test_every_superstar_projects_realistically_under_normal_conditions(self):
        from modules.fusion_model import fuse_projection

        for player, stats in _SUPERSTARS.items():
            with self.subTest(player=player):
                with patch("modules.fusion_model.project_player_statline", return_value=_base_projection_fixture(player, stats)), \
                     patch("modules.fusion_model.get_player_context", return_value=_player_context_fixture(stats["team"])):
                    result = fuse_projection(player, "OPP", "2025-26")
                floor = stats["points"] * _MIN_FRACTION_OF_SEASON_AVERAGE
                self.assertGreaterEqual(
                    result["final_projection"]["points"], floor,
                    msg=f"{player}'s points projection ({result['final_projection']['points']}) fell below "
                        f"{_MIN_FRACTION_OF_SEASON_AVERAGE:.0%} of their real season average ({stats['points']}).",
                )

    def test_every_superstar_still_projects_realistically_with_degraded_context(self):
        """Real data gaps (low component confidence, a tough matchup) may pull the projection down -- but not collapse it."""
        from modules.fusion_model import fuse_projection

        for player, stats in _SUPERSTARS.items():
            with self.subTest(player=player):
                with patch("modules.fusion_model.project_player_statline", return_value=_base_projection_fixture(player, stats)), \
                     patch("modules.fusion_model.get_player_context", return_value=_player_context_fixture(stats["team"], degraded=True)):
                    result = fuse_projection(player, "OPP", "2025-26")
                floor = stats["points"] * _MIN_FRACTION_OF_SEASON_AVERAGE
                self.assertGreaterEqual(
                    result["final_projection"]["points"], floor,
                    msg=f"{player}'s degraded-context points projection ({result['final_projection']['points']}) fell "
                        f"below {_MIN_FRACTION_OF_SEASON_AVERAGE:.0%} of their real season average ({stats['points']}).",
                )
                self.assertGreater(result["final_projection"]["points"], 0.0)


class MatchupEngineAdjustmentStaysBoundedTests(unittest.TestCase):
    """compute_matchup_context's context_multiplier must never swing far enough to manufacture an unrealistic low on its own."""

    def test_worst_case_real_defense_still_bounds_the_multiplier(self):
        from modules.nba_matchup_engine import compute_matchup_context

        with patch("modules.nba_matchup_engine.matchup_difficulty_score", return_value={"difficulty_score": 100.0, "basis": "real_pt_defend_data"}), \
             patch("modules.nba_matchup_engine.games_vs_opponent", return_value={"games_sampled": 0, "delta_vs_season": {}}), \
             patch("modules.nba_matchup_engine.opponent_injury_context", return_value={"absences": [], "severity": 0.0, "rim_defender_out": None}), \
             patch("modules.nba_matchup_engine.team_on_off_impact", return_value={"players": []}):
            context = compute_matchup_context("Nikola Jokic", "DEN", "BOS", "2025-26", player_role="interior")

        # difficulty_score=100 (the worst real case) -> multiplier floors
        # at 1.0 + (50-100)/500 = 0.90 -- a modest discount, not a collapse.
        self.assertGreaterEqual(context["context_multiplier"], 0.85)
        self.assertLessEqual(context["context_multiplier"], 1.15)

    def test_unreliable_history_sample_is_not_applied(self):
        from modules.nba_matchup_engine import compute_matchup_context

        with patch("modules.nba_matchup_engine.matchup_difficulty_score", return_value={"difficulty_score": 50.0, "basis": "real_pt_defend_data"}), \
             patch("modules.nba_matchup_engine.games_vs_opponent", return_value={"games_sampled": 1, "delta_vs_season": {"pts": 15.0}}), \
             patch("modules.nba_matchup_engine.opponent_injury_context", return_value={"absences": [], "severity": 0.0, "rim_defender_out": None}), \
             patch("modules.nba_matchup_engine.team_on_off_impact", return_value={"players": []}):
            context = compute_matchup_context("Nikola Jokic", "DEN", "BOS", "2025-26")

        # Only 1 real game sampled (< MIN_RELIABLE_GAMES) -- the real
        # 15-point outlier must not be applied as a real shift.
        self.assertEqual(context["history_shift"], 0.0)


if __name__ == "__main__":
    unittest.main()
