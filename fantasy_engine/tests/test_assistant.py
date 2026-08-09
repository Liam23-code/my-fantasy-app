"""Tests for the draft assistant's recommendation signals."""

from __future__ import annotations

import pytest

from fantasy.assistant import (
    ADP_NEIGHBOR_WINDOW_PICKS,
    MAX_REPLACEMENTS,
    capped_positions,
    get_replacements,
    next_pick_number,
    picks_between,
    replacement_levels,
    suggest_draft_picks,
)

SETTINGS = {
    "n_teams": 2,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 3},
    "flex_eligible": [],
}


def _p(name, position, projection, **extra):
    return {
        "player_id": f"id-{name}",
        "name": name,
        "position": position,
        "team": "SF",
        "projection": projection,
        **extra,
    }


def _pool():
    return [
        _p("RB1", "RB", 300.0),
        _p("RB2", "RB", 250.0),
        _p("RB3", "RB", 100.0),
        _p("WR1", "WR", 280.0),
        _p("WR2", "WR", 270.0),
        _p("WR3", "WR", 260.0),
        _p("QB1", "QB", 320.0),
        _p("QB2", "QB", 310.0),
        _p("QB3", "QB", 300.0),
    ]


# --- pick arithmetic --------------------------------------------------------


def test_next_pick_number_snake_reverses_on_even_rounds():
    # 12-team snake, slot 5: 5, then 20 (round 2 runs 12..1), then 29.
    assert next_pick_number(12, 5, 1) == 5
    assert next_pick_number(12, 5, 2) == 20
    assert next_pick_number(12, 5, 3) == 29


def test_next_pick_number_linear_keeps_the_same_slot():
    assert next_pick_number(12, 5, 1, snake=False) == 5
    assert next_pick_number(12, 5, 2, snake=False) == 17
    assert next_pick_number(12, 5, 3, snake=False) == 29


def test_picks_between_is_shortest_at_the_turn():
    # The last slot picks back-to-back across the snake turn; slot 1 waits longest.
    assert picks_between(12, 12, 1) == 0
    assert picks_between(12, 1, 1) == 22
    assert picks_between(12, 5, 1, snake=False) == 11


@pytest.mark.parametrize(
    ("teams", "slot", "round_number"),
    [(12, 0, 1), (12, 13, 1), (12, 5, 0), (0, 1, 1)],
)
def test_pick_arithmetic_rejects_out_of_range_input(teams, slot, round_number):
    with pytest.raises(ValueError):
        next_pick_number(teams, slot, round_number)


# --- replacement level / VORP ----------------------------------------------


def test_replacement_level_is_the_first_player_past_the_starters():
    # 2 teams x 1 RB starter => RB3 (the 3rd best) is replacement level.
    levels = replacement_levels(_pool(), SETTINGS)
    assert levels["RB"] == 100.0
    assert levels["WR"] == 260.0


def test_vorp_reflects_positional_value_not_raw_points():
    """A lower-scoring RB can be worth more than a higher-scoring QB."""
    suggestions = suggest_draft_picks(_pool(), SETTINGS, limit=99)
    by_name = {entry["name"]: entry for entry in suggestions}
    # RB1 scores less than QB1 but has a far weaker replacement behind it.
    assert by_name["RB1"]["projection"] < by_name["QB1"]["projection"]
    assert by_name["RB1"]["vorp"] > by_name["QB1"]["vorp"]


def test_replacement_level_falls_back_when_a_position_is_too_thin():
    thin = [_p("TE1", "TE", 200.0)]
    levels = replacement_levels(thin, {**SETTINGS, "roster_requirements": {"TE": 1, "BENCH": 1}})
    assert levels["TE"] == 200.0  # everyone available is a starter


# --- suggestions ------------------------------------------------------------


def test_suggestions_are_ranked_and_sorted_by_score():
    suggestions = suggest_draft_picks(_pool(), SETTINGS, limit=5)
    assert [entry["rank"] for entry in suggestions] == [1, 2, 3, 4, 5]
    scores = [entry["score"] for entry in suggestions]
    assert scores == sorted(scores, reverse=True)


def test_synthetic_players_are_never_suggested():
    pool = _pool() + [
        {"player_id": "synthetic:rb:0", "name": "Synthetic RB 0", "position": "RB", "team": "XX", "projection": 9999.0}
    ]
    suggestions = suggest_draft_picks(pool, SETTINGS, limit=99)
    assert all(entry["name"] != "Synthetic RB 0" for entry in suggestions)


def test_empty_pool_returns_an_empty_list_not_an_error():
    assert suggest_draft_picks([], SETTINGS) == []
    assert suggest_draft_picks(None, SETTINGS) == []


def test_roster_need_demotes_a_position_that_is_already_filled():
    without_roster = suggest_draft_picks(_pool(), SETTINGS, limit=99)
    with_rbs = suggest_draft_picks(_pool(), SETTINGS, my_roster=[{"position": "RB"}] * 3, limit=99)

    rb_before = next(e for e in without_roster if e["name"] == "RB1")
    rb_after = next(e for e in with_rbs if e["name"] == "RB1")
    assert rb_after["need_multiplier"] < rb_before["need_multiplier"]
    assert rb_after["score"] < rb_before["score"]


def test_adp_value_is_positive_when_a_player_falls_past_their_adp():
    pool = [_p("Faller", "RB", 300.0, adp=5.0), _p("Reach", "RB", 300.0, adp=60.0)]
    suggestions = suggest_draft_picks(pool, SETTINGS, next_pick_overall=30, limit=99)
    by_name = {entry["name"]: entry for entry in suggestions}
    assert by_name["Faller"]["adp_value"] == 25.0
    assert by_name["Reach"]["adp_value"] == -30.0
    assert by_name["Faller"]["score"] > by_name["Reach"]["score"]


def test_adp_is_absent_rather_than_invented_when_the_source_lacks_it():
    suggestions = suggest_draft_picks(_pool(), SETTINGS, next_pick_overall=10, limit=1)
    assert suggestions[0]["adp"] is None
    assert suggestions[0]["adp_value"] is None


def test_position_filter_preserves_board_wide_scarcity_and_rank():
    everyone = suggest_draft_picks(_pool(), SETTINGS, picks_until_next=6, limit=99)
    qbs_only = suggest_draft_picks(_pool(), SETTINGS, picks_until_next=6, positions=["QB"], limit=99)

    assert {entry["position"] for entry in qbs_only} == {"QB"}
    board_qb = next(e for e in everyone if e["position"] == "QB")
    assert qbs_only[0]["scarcity"] == board_qb["scarcity"]
    assert qbs_only[0]["position_rank"] == board_qb["position_rank"]


def test_players_without_a_projection_are_scored_from_raw_stats():
    pool = [{"player_id": "x", "name": "Raw", "position": "RB", "team": "SF", "rushing_yards": 1000, "rushing_tds": 10}]
    suggestions = suggest_draft_picks(pool, SETTINGS, limit=1)
    assert suggestions[0]["projection"] > 0


def test_every_suggestion_carries_a_rationale():
    for entry in suggest_draft_picks(_pool(), SETTINGS, picks_until_next=6, limit=5):
        assert entry["rationale"]
        assert "over" in entry["rationale"]


# --- Hybrid Draft Intelligence Model: timing, scarcity bias, roster caps ----


def test_steal_bonus_rewards_a_player_who_fell_past_adp():
    """A star still on the board well after their ADP is a steal, not a penalty."""
    pool = [_p("Star", "WR", 300.0, adp=5.0)]
    suggestions = suggest_draft_picks(pool, SETTINGS, next_pick_overall=50, limit=1)
    entry = suggestions[0]
    assert entry["steal_bonus"] == 45.0
    assert entry["reach_penalty"] == 0.0
    assert entry["adp_value"] == 45.0


def test_reach_penalty_flags_drafting_well_before_adp():
    pool = [_p("Sleeper", "WR", 300.0, adp=80.0)]
    suggestions = suggest_draft_picks(pool, SETTINGS, next_pick_overall=10, limit=1)
    entry = suggestions[0]
    assert entry["reach_penalty"] == 70.0
    assert entry["steal_bonus"] == 0.0
    assert entry["adp_value"] == -70.0


def test_steal_and_reach_are_never_both_positive():
    for adp, current_pick in [(5.0, 50.0), (80.0, 10.0), (30.0, 30.0)]:
        pool = [_p("X", "WR", 300.0, adp=adp)]
        entry = suggest_draft_picks(pool, SETTINGS, next_pick_overall=current_pick, limit=1)[0]
        assert not (entry["steal_bonus"] > 0 and entry["reach_penalty"] > 0)


def test_a_falling_star_outscores_an_equivalent_reach():
    """The Jefferson-in-round-5 case: falling value must score higher than reaching for the same value."""
    falling = _p("Falling", "WR", 300.0, adp=5.0)
    reaching = _p("Reaching", "WR", 300.0, adp=95.0)
    suggestions = suggest_draft_picks([falling, reaching], SETTINGS, next_pick_overall=50, limit=2)
    by_name = {e["name"]: e for e in suggestions}
    assert by_name["Falling"]["score"] > by_name["Reaching"]["score"]


def test_must_draft_now_when_adp_falls_before_the_next_turn():
    """picks_until_next=8 at pick 10: a player with adp=15 will not survive to pick 19."""
    pool = [_p("WontLast", "WR", 300.0, adp=15.0), _p("WillLast", "WR", 280.0, adp=40.0)]
    suggestions = suggest_draft_picks(pool, SETTINGS, next_pick_overall=10, picks_until_next=8, limit=2)
    by_name = {e["name"]: e for e in suggestions}
    assert by_name["WontLast"]["must_draft_now"] is True
    assert by_name["WillLast"]["must_draft_now"] is False


def test_must_draft_now_boosts_score_above_a_similar_non_urgent_player():
    urgent = _p("Urgent", "WR", 250.0, adp=12.0)
    calm = _p("Calm", "WR", 250.0, adp=45.0)
    suggestions = suggest_draft_picks([urgent, calm], SETTINGS, next_pick_overall=10, picks_until_next=8, limit=2)
    by_name = {e["name"]: e for e in suggestions}
    assert by_name["Urgent"]["score"] > by_name["Calm"]["score"]


def test_must_draft_now_requires_both_pick_context_args():
    pool = [_p("X", "WR", 300.0, adp=5.0)]
    assert suggest_draft_picks(pool, SETTINGS, next_pick_overall=50, limit=1)[0]["must_draft_now"] is False


def test_scarcity_bias_boosts_te_and_discounts_wr():
    from fantasy.assistant import POSITION_SCARCITY_BIAS

    assert POSITION_SCARCITY_BIAS["TE"] > 1.0
    assert POSITION_SCARCITY_BIAS["RB"] > 1.0
    assert POSITION_SCARCITY_BIAS["WR"] < 1.0
    assert POSITION_SCARCITY_BIAS["QB"] < 1.0


def test_biased_scarcity_scales_the_raw_scarcity_number():
    from fantasy.assistant import POSITION_SCARCITY_BIAS

    pool = _pool()
    suggestions = suggest_draft_picks(pool, SETTINGS, picks_until_next=4, limit=99)
    for entry in suggestions:
        expected = round(entry["scarcity_raw"] * POSITION_SCARCITY_BIAS.get(entry["position"], 1.0), 2)
        assert entry["scarcity"] == expected


def test_roster_at_max_never_gets_suggested():
    """QB cap is (1, 2): a roster with 2 QBs should see zero QB suggestions."""
    pool = _pool()
    roster = [{"position": "QB"}, {"position": "QB"}]
    suggestions = suggest_draft_picks(pool, SETTINGS, my_roster=roster, limit=99)
    assert all(entry["position"] != "QB" for entry in suggestions)


def test_capped_positions_reports_positions_at_their_max():
    assert capped_positions([{"position": "QB"}, {"position": "QB"}]) == ["QB"]
    assert capped_positions([{"position": "QB"}]) == []
    assert capped_positions([]) == []
    assert capped_positions([{"position": "K"}]) == ["K"]  # K cap is (1, 1)


def test_need_label_distinguishes_need_from_depth():
    pool = _pool()
    empty_roster_entry = suggest_draft_picks(pool, SETTINGS, my_roster=[], limit=1)[0]
    assert empty_roster_entry["need_label"] == "Fills need"

    stacked_roster = [{"position": empty_roster_entry["position"]}] * 3
    stacked_entry = next(
        e for e in suggest_draft_picks(pool, SETTINGS, my_roster=stacked_roster, limit=99)
        if e["position"] == empty_roster_entry["position"]
    )
    assert stacked_entry["need_label"] in {"Depth", "Roster full"}


# --- market-implied value blend (holdout-driven) -----------------------------


def test_market_blend_weight_is_documented_and_in_range():
    from fantasy.assistant import MARKET_BLEND_WEIGHT

    assert 0.0 <= MARKET_BLEND_WEIGHT <= 1.0


def test_market_rank_drives_value_over_raw_vorp():
    """The holdout fix: a market-favoured player must outrank a stale-stats one.

    "Stale" here is the real failure mode measured on the 2024->2025 holdout:
    high last-season production the market has already priced down (Alvin
    Kamara 271 -> 101), versus a market favourite whose last season was poor
    (Christian McCaffrey, 54 points injured, ADP 7.8, 429 the next year).
    """
    pool = [
        _p("StaleStats", "RB", 300.0, adp=140.0),
        _p("MarketFavourite", "RB", 60.0, adp=8.0),
        _p("Filler1", "RB", 200.0, adp=60.0),
        _p("Filler2", "RB", 100.0, adp=90.0),
    ]
    suggestions = suggest_draft_picks(pool, SETTINGS, next_pick_overall=1, limit=4)
    assert suggestions[0]["name"] == "MarketFavourite"


def test_players_without_adp_fall_back_to_pure_vorp():
    pool = [_p("NoMarketView", "RB", 300.0), _p("AlsoNoView", "RB", 100.0)]
    suggestions = suggest_draft_picks(pool, SETTINGS, limit=2)
    assert suggestions[0]["name"] == "NoMarketView"
    assert suggestions[0]["market_vorp"] is None
    assert suggestions[0]["effective_vorp"] == suggestions[0]["vorp"]


def test_market_vorp_and_effective_vorp_are_exposed():
    pool = [_p("A", "RB", 300.0, adp=1.0), _p("B", "RB", 100.0, adp=50.0)]
    entry = suggest_draft_picks(pool, SETTINGS, limit=1)[0]
    assert entry["market_vorp"] is not None
    assert "effective_vorp" in entry
    assert "vorp" in entry  # raw VORP still reported for transparency


def test_raw_vorp_is_still_reported_unchanged_by_the_blend():
    """VORP moves from player-ranker to value-scale; it must stay visible as-is."""
    pool = [_p("A", "RB", 300.0, adp=90.0), _p("B", "RB", 100.0, adp=1.0)]
    by_name = {e["name"]: e for e in suggest_draft_picks(pool, SETTINGS, limit=2)}
    assert by_name["A"]["vorp"] > by_name["B"]["vorp"]  # raw VORP order unchanged


# --- replacements ("he just got taken") -------------------------------------

REPLACEMENT_SETTINGS = {
    "n_teams": 12,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1, "BENCH": 6},
    "flex_eligible": ["RB", "WR", "TE"],
}


# normalize_player_name() folds a name down to [a-z] only, so "RB1"/"RB2"
# collapse to the same key. Fixture names must therefore differ alphabetically
# or the name-matching paths cannot be tested at all.
_NATO = ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet", "Kilo")


def _rb(index):
    return f"Rusher {_NATO[index - 1]}"


def _wr(index):
    return f"Catcher {_NATO[index - 1]}"


def _deep_pool():
    """A board with real ADP spacing at two positions."""
    return [_p(_rb(i), "RB", 300.0 - i * 12, adp=float(i * 6 + 3)) for i in range(1, 12)] + [
        _p(_wr(i), "WR", 290.0 - i * 10, adp=float(i * 5 + 2)) for i in range(1, 12)
    ]


def _state(**overrides):
    state = {
        "league_settings": REPLACEMENT_SETTINGS,
        "my_roster": [],
        "available_players": _deep_pool(),
        "round_number": 2,
        "next_pick_overall": 20,
        "picks_until_next": 22,
    }
    state.update(overrides)
    return state


def test_get_replacements_returns_same_position_options():
    replacements = get_replacements("id-Rusher Delta", _state())
    assert 3 <= len(replacements) <= 5
    assert {entry["position"] for entry in replacements} == {"RB"}


def test_get_replacements_never_returns_the_drafted_player():
    replacements = get_replacements("id-Rusher Delta", _state())
    assert all(entry["name"] != "Rusher Delta" for entry in replacements)
    assert all(entry["replaces_name"] == "Rusher Delta" for entry in replacements)


def test_get_replacements_excludes_already_drafted_players():
    state = _state(drafted_player_ids=["id-Rusher Charlie", "id-Rusher Echo"])
    names = {entry["name"] for entry in get_replacements("id-Rusher Delta", state)}
    assert not names & {"Rusher Charlie", "Rusher Delta", "Rusher Echo"}


def test_get_replacements_accepts_pick_dicts_and_bare_names():
    """The UI holds 'who is gone' as simulate_draft pick dicts or plain names."""
    state = _state(drafted_players=[{"player_name": "Rusher Charlie"}, "Rusher Echo"])
    names = {entry["name"] for entry in get_replacements("id-Rusher Delta", state)}
    assert not names & {"Rusher Charlie", "Rusher Echo"}


def test_get_replacements_prefers_adp_neighbors_over_distant_players():
    """A replacement is someone available at a similar cost, not the best left."""
    replacements = get_replacements("id-Rusher Hotel", _state())
    gaps = [abs(entry["adp_gap"]) for entry in replacements if entry["adp_gap"] is not None]
    assert gaps and max(gaps) <= ADP_NEIGHBOR_WINDOW_PICKS


def test_get_replacements_backfills_when_the_adp_board_is_patchy():
    """No ADP anywhere must still yield a usable list rather than an empty one."""
    pool = [_p(_rb(i), "RB", 300.0 - i * 10) for i in range(1, 9)]
    replacements = get_replacements("id-Rusher Bravo", _state(available_players=pool))
    assert len(replacements) >= 3
    assert all(entry["adp_gap"] is None for entry in replacements)


def test_get_replacements_reports_pressure_and_rationale():
    entry = get_replacements("id-Rusher Delta", _state())[0]
    assert entry["rank"] == 1
    assert entry["pressure_multiplier"] > 0
    assert entry["position_pressure_now"] > 0
    # Both factors are rounded before the product is, so allow for that drift.
    assert entry["replacement_score"] == pytest.approx(
        entry["score"] * entry["pressure_multiplier"], rel=1e-2, abs=0.02
    )
    assert "Rusher Delta" in entry["replacement_rationale"]


def test_get_replacements_is_ranked_best_first():
    replacements = get_replacements("id-Rusher Delta", _state())
    scores = [entry["replacement_score"] for entry in replacements]
    assert scores == sorted(scores, reverse=True)
    assert [entry["rank"] for entry in replacements] == list(range(1, len(replacements) + 1))


def test_get_replacements_respects_the_limit_ceiling():
    assert len(get_replacements("id-Rusher Delta", _state(), limit=99)) <= MAX_REPLACEMENTS
    assert len(get_replacements("id-Rusher Delta", _state(), limit=3)) == 3


@pytest.mark.parametrize(
    ("player_id", "state"),
    [
        ("id-NOBODY", _state()),                    # unknown target
        ("id-Rusher Delta", {"league_settings": REPLACEMENT_SETTINGS}),  # no pool at all
        (None, _state()),                           # no target
    ],
)
def test_get_replacements_returns_empty_state_not_an_error(player_id, state):
    assert get_replacements(player_id, state) == []


def test_get_replacements_tolerates_a_missing_draft_state():
    assert get_replacements("id-Rusher Delta", None) == []


def test_get_replacements_drops_synthetic_players():
    """A fabricated player must never be offered as a pivot."""
    pool = _deep_pool() + [
        {"player_id": "synthetic:fake", "name": "Fake Guy", "position": "RB", "team": "SF", "projection": 999.0, "adp": 25.0}
    ]
    names = {entry["name"] for entry in get_replacements("id-Rusher Delta", _state(available_players=pool))}
    assert "Fake Guy" not in names
