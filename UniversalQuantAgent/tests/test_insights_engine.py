"""Offline contract tests for the short-form insights layer."""
import unittest

from modules.insights_engine import (
    generate_badge_insights,
    generate_correlation_insights,
    generate_edge_insights,
    generate_player_insights,
    generate_similarity_insights,
    generate_slate_insights,
)


class InsightsEngineContracts(unittest.TestCase):
    def test_player_insights_are_short_and_partial_data_safe(self):
        payload = {
            "season_avg": {"mpg": 34.0},
            "last5_avg": {"mpg": 36.0},
            "advanced": {
                "usage_pct": 29.0,
                "ts_pct": 63.0,
                "per_estimate": 25.0,
            },
        }
        insights = generate_player_insights(payload, {"difficulty_score": 68})
        self.assertTrue(insights)
        self.assertTrue(all(item.count(". ") <= 1 and len(item.split()) <= 42 for item in insights))

    def test_slate_and_edge_insights_accept_partial_rows(self):
        slate = {
            "games": [{"home_team": "DEN", "away_team": "BOS"}],
            "player_props": [],
            "projections": [{
                "player": "Example",
                "pace": 102,
                "matchup_difficulty": 60,
            }],
        }
        self.assertTrue(generate_slate_insights(slate))
        self.assertTrue(generate_edge_insights([{
            "player": "Example", "category": "points", "edge": 2.5,
        }]))

    def test_badge_identity_uses_visible_values(self):
        profile = {
            "player": "Test Player", "display_mode": "Adjusted",
            "attributes": [
                {"attribute": "Playmaking", "badge_value": 93, "sample_confidence": 1},
                {"attribute": "Defense", "badge_value": 72, "sample_confidence": 1},
                {"attribute": "Mid-range", "badge_value": 48, "sample_confidence": .4},
            ],
        }
        insights = generate_badge_insights(profile)
        self.assertIn("Playmaking (93)", insights[0])
        self.assertIn("Mid-range (48)", insights[0])
        self.assertIn("Low sample size", insights[1])
    def test_similarity_and_correlation_insights_are_explainable(self):
        similarity = {
            "target": {"player": "Target"},
            "similar_players": [{
                "player": "Match",
                "similarity_score": 91,
                "dimension_scores": {"usage": 95, "defense": 70},
            }],
        }
        correlation = {
            "correlation_matrix": {
                "points": {"points": 1, "minutes": .8},
                "minutes": {"points": .8, "minutes": 1},
            }
        }
        self.assertIn("Match", generate_similarity_insights(similarity)[0])
        self.assertIn("r=0.80", generate_correlation_insights(correlation)[0])


if __name__ == "__main__":
    unittest.main()