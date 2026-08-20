"""Contracts for fantasy rarity tier mapping and accessible badge markup."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.style import RARITY_TIERS, rarity_icon, rarity_metadata, rarity_tier


class RarityIconContracts(unittest.TestCase):
    def test_every_boundary_maps_to_the_documented_tier(self):
        expectations = {
            1: "Mythic",
            2: "Legendary",
            5: "Legendary",
            6: "Elite",
            10: "Elite",
            11: "Pro",
            20: "Pro",
            21: "Starter",
            40: "Starter",
            41: "Depth",
            500: "Depth",
        }
        for rank, expected in expectations.items():
            with self.subTest(rank=rank):
                self.assertEqual(rarity_tier(rank), expected)

    def test_badge_contains_tier_class_rank_and_accessible_label(self):
        badge = rarity_icon(7)
        self.assertIn('class="rarity-badge rarity-elite"', badge)
        self.assertIn('aria-label="Elite rarity, rank 7"', badge)
        self.assertIn("rarity-symbol", badge)
        self.assertIn("Elite", badge)

    def test_badge_can_hide_visual_label_without_losing_accessibility(self):
        badge = rarity_icon(1, include_label=False)
        self.assertNotIn("rarity-label", badge)
        self.assertIn('aria-label="Mythic rarity, rank 1"', badge)

    def test_invalid_rank_falls_back_to_depth(self):
        metadata = rarity_metadata("not-a-rank")
        self.assertEqual(metadata["name"], "Depth")
        self.assertEqual(metadata["rank"], 41)

    def test_tiers_are_contiguous_and_complete(self):
        self.assertEqual([tier["name"] for tier in RARITY_TIERS], [
            "Mythic",
            "Legendary",
            "Elite",
            "Pro",
            "Starter",
            "Depth",
        ])
        for rank in range(1, 250):
            self.assertIn(rarity_tier(rank), {tier["name"] for tier in RARITY_TIERS})


if __name__ == "__main__":
    unittest.main()
