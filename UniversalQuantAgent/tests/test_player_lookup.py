"""Offline contracts for modules.player_lookup -- the roster identity index."""

import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

from modules import player_lookup


_ROSTER = pd.DataFrame(
    [
        {"gsis_id": "00-0037070", "full_name": "Jaylen Waddle", "position": "WR", "team": "MIA",
         "birth_date": "1998-11-25", "years_exp": 4, "status": "ACT"},
        {"gsis_id": "00-0037012", "full_name": "Amon-Ra St. Brown", "position": "WR", "team": "DET",
         "birth_date": "1999-10-24", "years_exp": 4, "status": "ACT"},
        {"gsis_id": "00-0039338", "full_name": "Sam LaPorta", "position": "TE", "team": "DET",
         "birth_date": "2001-01-12", "years_exp": 2, "status": "ACT"},
        {"gsis_id": "00-0034857", "full_name": "Josh Allen", "position": "QB", "team": "BUF",
         "birth_date": "1996-05-21", "years_exp": 7, "status": "ACT"},
        {"gsis_id": "00-0038000", "full_name": "Josh Allen", "position": "LB", "team": "JAX",
         "birth_date": "1997-07-09", "years_exp": 5, "status": "ACT"},
        {"gsis_id": "00-0039491", "full_name": "Chase Brown", "position": "RB", "team": "CIN",
         "birth_date": "2000-03-15", "years_exp": 1, "status": "ACT"},
        {"gsis_id": "00-0036252", "full_name": "Michael Pittman", "position": "WR", "team": "IND",
         "birth_date": "1997-10-05", "years_exp": 5, "status": "ACT"},
    ]
    + [
        {"gsis_id": f"00-90{i:05d}", "full_name": f"Filler Player{i}", "position": "WR", "team": "FA",
         "birth_date": "1999-01-01", "years_exp": 3, "status": "ACT"}
        for i in range(220)
    ]
)


def _fake_nflreadpy():
    module = types.ModuleType("nflreadpy")
    module.load_rosters = lambda seasons: _ROSTER
    module.load_players = lambda: _ROSTER
    return module


class PlayerLookupContracts(unittest.TestCase):
    def setUp(self):
        player_lookup.clear_cache()
        self._patch = patch.dict(sys.modules, {"nflreadpy": _fake_nflreadpy()})
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        player_lookup.clear_cache()

    def test_resolves_a_non_star_full_name(self):
        hit = player_lookup.lookup_player("Chase Brown", season=2025)
        self.assertEqual(hit["player_id"], "00-0039491")
        self.assertEqual(hit["position"], "RB")
        self.assertGreaterEqual(hit["confidence"], 0.9)

    def test_resolves_the_f_last_abbreviation(self):
        hit = player_lookup.lookup_player("J.Waddle", season=2025)
        self.assertEqual(hit["player_id"], "00-0037070")
        self.assertEqual(hit["name"], "Jaylen Waddle")

    def test_resolves_a_hyphenated_apostrophe_name(self):
        hit = player_lookup.lookup_player("Amon Ra St Brown", season=2025)
        self.assertEqual(hit["player_id"], "00-0037012")

    def test_position_hint_disambiguates_a_shared_name(self):
        qb = player_lookup.lookup_player("Josh Allen", season=2025, position_hint="QB")
        self.assertEqual(qb["position"], "QB")
        lb = player_lookup.lookup_player("Josh Allen", season=2025, position_hint="LB")
        self.assertEqual(lb["position"], "LB")

    def test_computes_age_from_birth_date_as_of_september(self):
        hit = player_lookup.lookup_player("Josh Allen", season=2025, position_hint="QB")
        self.assertEqual(hit["age"], 29.0)  # born 1996-05-21, anchor 2025-09-01

    def test_a_raw_gsis_id_short_circuits(self):
        hit = player_lookup.lookup_player("00-0039338", season=2025)
        self.assertEqual(hit["name"], "Sam LaPorta")
        self.assertEqual(hit["source"], "id")

    def test_an_unknown_name_returns_an_unresolved_marker_with_candidates(self):
        hit = player_lookup.lookup_player("Zzyzx Notaplayer", season=2025)
        self.assertEqual(hit["source"], "unresolved")
        self.assertEqual(hit["player_id"], "")
        self.assertTrue(hit["candidates"])

    def test_a_shared_first_name_does_not_license_a_wrong_match(self):
        # "Michael Thomas" is absent from the index; it must NOT resolve to
        # "Michael Pittman" just because they share a first name.
        hit = player_lookup.lookup_player("Michael Thomas", season=2025)
        self.assertNotEqual(hit["source"], "roster")
        self.assertEqual(hit["player_id"], "")

    def test_a_misspelled_surname_still_resolves(self):
        # "Jalen Wadle" -> "Jaylen Waddle": typos in BOTH names, but the surname
        # is still ~0.9 similar. The exact-equality surname gate rejected this;
        # a near-match gate must let it through rather than fall to unresolved.
        hit = player_lookup.lookup_player("Jalen Wadle", season=2025)
        self.assertEqual(hit["player_id"], "00-0037070")
        self.assertGreaterEqual(hit["confidence"], 0.6)

    def test_a_shared_surname_still_resolves(self):
        # "Mike Pittman" -> "Michael Pittman": shared surname, so a fuzzy first
        # name is still allowed.
        hit = player_lookup.lookup_player("Mike Pittman", season=2025)
        self.assertEqual(hit["player_id"], "00-0036252")

    def test_a_dead_provider_returns_none_not_an_exception(self):
        broken = types.ModuleType("nflreadpy")

        def _boom(*_a, **_k):
            raise RuntimeError("provider down")

        broken.load_rosters = _boom
        broken.load_players = _boom
        with patch.dict(sys.modules, {"nflreadpy": broken}):
            player_lookup.clear_cache()
            self.assertIsNone(player_lookup.lookup_player("Chase Brown", season=2025))

    def test_a_failed_load_is_not_memoised(self):
        broken = types.ModuleType("nflreadpy")
        broken.load_rosters = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("down"))
        broken.load_players = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("down"))
        with patch.dict(sys.modules, {"nflreadpy": broken}):
            player_lookup.clear_cache()
            self.assertIsNone(player_lookup.lookup_player("Chase Brown", season=2025))
        # provider recovers -> next call succeeds (nothing was pinned)
        self.assertIsNotNone(player_lookup.lookup_player("Chase Brown", season=2025))


if __name__ == "__main__":
    unittest.main()
