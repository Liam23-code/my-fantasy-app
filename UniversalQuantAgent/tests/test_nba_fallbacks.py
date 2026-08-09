"""Regression tests for NBA timeout, retry, and projection fallbacks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import modules.nba_cache as nba_cache
from modules.nba_cache import (
    ENHANCED_FALLBACK_WARNING,
    FALLBACK_WARNING,
    fetch_nba_frames,
    select_fallback_minutes,
    weighted_available_average,
)
from modules.projections import project_player_statline


class NbaFallbackContracts(unittest.TestCase):
    def setUp(self):
        self.temp_directory = TemporaryDirectory()
        self.original_cache_directory = nba_cache.CACHE_DIR
        nba_cache.CACHE_DIR = Path(self.temp_directory.name)
        nba_cache._MEMORY_CACHE.clear()
        nba_cache._PROVIDER_UNAVAILABLE_UNTIL = 0.0

    def tearDown(self):
        nba_cache.CACHE_DIR = self.original_cache_directory
        nba_cache._MEMORY_CACHE.clear()
        nba_cache._PROVIDER_UNAVAILABLE_UNTIL = 0.0
        self.temp_directory.cleanup()

    @patch("modules.nba_cache.time.sleep")
    def test_three_retries_then_explicit_fallback(self, sleep):
        calls = {"count": 0}

        def unavailable_provider():
            calls["count"] += 1
            raise TimeoutError("forced timeout")

        frames = fetch_nba_frames(
            "retry_contract",
            unavailable_provider,
            fallback_factory=lambda: pd.DataFrame([{"value": 42.0}]),
        )
        self.assertEqual(calls["count"], 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(frames[0].attrs["data_source"], "season_average_fallback")
        self.assertIn(FALLBACK_WARNING, frames[0].attrs["warnings"])

    @patch("modules.nba_cache.time.sleep")
    def test_projection_survives_complete_provider_outage(self, _sleep):
        outage = TimeoutError("forced NBA API outage")
        with patch(
            "modules.projections.playergamelog.PlayerGameLog", side_effect=outage
        ), patch(
            "modules.projections.leaguedashplayerstats.LeagueDashPlayerStats",
            side_effect=outage,
        ), patch(
            "modules.projections.leaguedashteamstats.LeagueDashTeamStats",
            side_effect=outage,
        ), patch(
            "modules.nba_advanced.leaguedashteamstats.LeagueDashTeamStats",
            side_effect=outage,
        ):
            result = project_player_statline(
                "Nikola Jokic", "BOS", "2025-26"
            )

        self.assertEqual(result["model"], "RollingAverageFallback")
        self.assertIn(FALLBACK_WARNING, result["warnings"])
        self.assertIn(ENHANCED_FALLBACK_WARNING, result["warnings"])
        self.assertGreater(result["projected_statline"]["points"], 0)


    def test_minutes_hierarchy_and_usage_reweighting(self):
        value, source = select_fallback_minutes(None, 35.0, 36.0)
        self.assertEqual((value, source), (35.0, "last_10"))
        value, source = select_fallback_minutes(34.5, 35.0, 36.0)
        self.assertEqual((value, source), (34.5, "season_average"))
        blend, coverage = weighted_available_average(
            [(30.0, .50), (None, .30), (40.0, .20)]
        )
        self.assertAlmostEqual(blend, (30 * .50 + 40 * .20) / .70)
        self.assertAlmostEqual(coverage, .70)

    @patch("modules.fusion_model.get_player_context")
    @patch("modules.fusion_model.project_player_statline")
    def test_jokic_cached_fallback_projects_realistic_points(
        self, base_projection, player_context
    ):
        base_projection.return_value = {
            "player": "Nikola Jokic",
            "projected_statline": {
                "points": 22.5,
                "rebounds": 12.3,
                "assists": 10.1,
                "pra": 44.9,
            },
            "confidence_ranges": {
                stat: {"low": value - 4, "high": value + 4}
                for stat, value in {
                    "points": 22.5,
                    "rebounds": 12.3,
                    "assists": 10.1,
                    "pra": 44.9,
                }.items()
            },
            "season_averages": {
                "points": 27.7,
                "rebounds": 12.9,
                "assists": 10.7,
                "pra": 51.3,
            },
            "recent_10_game_averages": {
                "points": 25.3,
                "rebounds": 14.5,
                "assists": 11.9,
                "pra": 51.7,
            },
            "recent_5_game_averages": {
                "points": 25.4,
                "rebounds": 12.6,
                "assists": 9.8,
                "pra": 47.8,
            },
            "feature_summary": {
                "season_average": {
                    "points": 27.7,
                    "rebounds": 12.9,
                    "assists": 10.7,
                    "pra": 51.3,
                    "minutes": 34.8,
                    "usage": 27.8,
                    "true_shooting_pct": 68.6,
                },
                "last_10": {
                    "minutes": 35.2,
                    "usage": 25.1,
                    "true_shooting_pct": 64.0,
                },
                "last_5": {
                    "minutes": 33.8,
                    "usage": 25.6,
                    "true_shooting_pct": 63.1,
                },
                "usage": {"quality": 90.0},
                "efficiency": {"quality": 92.0},
            },
            "fallback_status": {
                "active": True,
                "source": "historical_cache",
            },
            "warnings": [FALLBACK_WARNING],
        }
        player_context.return_value = {
            "team": "DEN",
            "role_stability": 80.0,
            "components": {
                "minutes": {
                    "minutes_projection": 34.4,
                    "season_average_minutes": 34.8,
                    "rolling_minutes": {"last_10": 35.2, "last_5": 33.8},
                    "fallback_minutes_quality": 92.0,
                    "confidence": 65.0,
                },
                "pace": {
                    "pace_projection": 97.8,
                    "league_average_pace": 100.0,
                    "team_pace_value": 99.0,
                    "opponent_pace_value": 96.0,
                    "pace_factor": .975,
                    "confidence": 72.0,
                },
                "matchup": {
                    "difficulty_score": 80.0,
                    "opponent_usage_factor": .99,
                    "opponent_def_eff_factor": .975,
                    "confidence": 78.0,
                },
                "availability": {"impact_score": 0.0},
            },
        }
        from modules.fusion_model import fuse_projection

        result = fuse_projection("Nikola Jokic", "BOS", "2025-26")
        self.assertGreaterEqual(result["final_projection"]["points"], 23.0)
        self.assertLessEqual(result["final_projection"]["points"], 27.0)
        self.assertAlmostEqual(sum(result["fusion_weights"].values()), 1.0, places=3)
        self.assertEqual(
            result["input_breakdown"]["minutes"]["selected_source"],
            "season_average",
        )

    @patch("modules.fusion_model.get_player_context")
    @patch("modules.fusion_model.project_player_statline")
    def test_fusion_does_not_floor_projection_output(
        self, base_projection, player_context
    ):
        base_projection.return_value = {
            "player": "Expressive Test",
            "projected_statline": {
                "points": -2.0,
                "rebounds": -1.0,
                "assists": -1.0,
                "pra": -4.0,
            },
            "confidence_ranges": {
                stat: {"low": value - 1, "high": value + 1}
                for stat, value in {
                    "points": -2.0,
                    "rebounds": -1.0,
                    "assists": -1.0,
                    "pra": -4.0,
                }.items()
            },
            "season_averages": {
                "points": -2.0,
                "rebounds": -1.0,
                "assists": -1.0,
            },
            "recent_10_game_averages": {
                "points": -2.0,
                "rebounds": -1.0,
                "assists": -1.0,
            },
            "recent_5_game_averages": {
                "points": -2.0,
                "rebounds": -1.0,
                "assists": -1.0,
            },
        }
        player_context.return_value = {
            "team": "DEN",
            "role_stability": 50.0,
            "components": {
                "minutes": {
                    "minutes_projection": 30.0,
                    "rolling_minutes": {"last_10": 30.0},
                    "confidence": 50.0,
                },
                "pace": {"pace_projection": 100.0, "confidence": 50.0},
                "matchup": {"difficulty_score": 50.0, "confidence": 50.0},
                "availability": {"impact_score": 0.0},
            },
        }
        from modules.fusion_model import fuse_projection

        result = fuse_projection("Expressive Test", "BOS", "2025-26")
        self.assertLess(result["final_projection"]["points"], 0.0)

    @patch(
        "modules.reliability._similarity_confidence",
        return_value=(80.0, "Similarity test."),
    )
    def test_reliability_includes_fallback_and_similarity_quality(self, _):
        from modules.reliability import get_reliability_score

        fusion = {
            "player": "Test Player",
            "final_projection": {"points": 24.0},
            "confidence_range": {"points": {"low": 20.0, "high": 28.0}},
            "input_breakdown": {
                "minutes": {"quality": 92.0},
                "usage": {"quality": 88.0},
            },
            "base_projection": {
                "data_quality": {"completeness_score": 90.0},
                "features_used": ["minutes", "usage"],
            },
            "context": {
                "role_stability": 85.0,
                "components": {
                    "minutes": {"coach_tendency_score": 82.0},
                    "pace": {"confidence": 76.0, "pace_projection": 100.0},
                    "matchup": {
                        "confidence": 79.0,
                        "difficulty_score": 55.0,
                    },
                    "availability": {"availability": "ACTIVE"},
                },
            },
        }
        result = get_reliability_score(
            "Test Player", "BOS", "2025-26", fusion_result=fusion
        )
        required = {
            "data_completeness",
            "fallback_usage_quality",
            "fallback_minutes_quality",
            "opponent_difficulty_confidence",
            "pace_confidence",
            "similarity_model_confidence",
        }
        self.assertTrue(required.issubset(result["components"]))
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 100.0)

if __name__ == "__main__":
    unittest.main()