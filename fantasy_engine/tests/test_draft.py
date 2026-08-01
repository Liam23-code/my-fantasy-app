"""Unit tests for fantasy.draft."""

from __future__ import annotations

import pytest

from fantasy.draft import (
    _replacement_levels,
    generate_cheatsheet,
    rank_players_for_draft,
    simulate_draft,
    suggest_picks,
)
from fantasy.models import LeagueSettings


def _flat_rb(name: str, rushing_yards: float) -> dict:
    return {"player_id": name, "name": name, "position": "RB", "rushing_yards": rushing_yards}


def _flat_wr(name: str, receiving_yards: float) -> dict:
    return {"player_id": name, "name": name, "position": "WR", "receiving_yards": receiving_yards}


@pytest.fixture
def flex_pool_settings() -> LeagueSettings:
    return LeagueSettings(
        n_teams=2,
        scoring_mode="ppr",
        roster_requirements={"QB": 0, "RB": 1, "WR": 1, "TE": 0, "FLEX": 1, "DST": 0, "K": 0, "BENCH": 0},
        flex_eligible=["RB", "WR"],
    )


@pytest.fixture
def flex_pool_projections() -> list[dict]:
    # RB yards/10 == points (no bonus, no receptions) so expected VOR is exactly computable by hand.
    rbs = [_flat_rb(f"RB{i}", yards) for i, yards in enumerate([90, 80, 70, 60, 50, 40], start=1)]
    wrs = [_flat_wr(f"WR{i}", yards) for i, yards in enumerate([95, 85, 75, 65, 55, 45], start=1)]
    return rbs + wrs


def test_replacement_levels_account_for_flex_pool(flex_pool_projections, flex_pool_settings):
    ranked = rank_players_for_draft(flex_pool_projections, flex_pool_settings)
    by_name = {p["name"]: p for p in ranked}

    # Dedicated starters: RB1/RB2, WR1/WR2. Flex pool (RB3-6, WR3-6) sorted
    # desc is WR3(7.5), RB3(7.0), WR4(6.5), RB4(6.0), ... -> 2 flex slots go
    # to WR3 and RB3, so replacement level is RB4's 6.0 and WR4's 6.5.
    assert by_name["RB1"]["vor"] == pytest.approx(3.0)
    assert by_name["RB2"]["vor"] == pytest.approx(2.0)
    assert by_name["RB3"]["vor"] == pytest.approx(1.0)
    assert by_name["RB4"]["vor"] == pytest.approx(0.0)
    assert by_name["RB6"]["vor"] == pytest.approx(-2.0)

    assert by_name["WR1"]["vor"] == pytest.approx(3.0)
    assert by_name["WR4"]["vor"] == pytest.approx(0.0)
    assert by_name["WR6"]["vor"] == pytest.approx(-2.0)


def test_replacement_levels_helper_matches_hand_computed_values(flex_pool_projections, flex_pool_settings):
    from fantasy.adapter import normalize_projection
    from fantasy.draft import _score_player

    scored = []
    for source in flex_pool_projections:
        canonical = normalize_projection(source)
        scored.append({**canonical, "points": _score_player(canonical, flex_pool_settings)})
    levels = _replacement_levels(scored, flex_pool_settings, flex_pool_settings.n_teams)
    assert levels["RB"] == pytest.approx(6.0)
    assert levels["WR"] == pytest.approx(6.5)


def test_ranked_players_are_sorted_descending_by_vor(named_player_projections, league_settings):
    ranked = rank_players_for_draft(named_player_projections, league_settings)
    vor_values = [p["vor"] for p in ranked]
    assert vor_values == sorted(vor_values, reverse=True)
    assert [p["overall_rank"] for p in ranked] == list(range(1, len(ranked) + 1))


def test_ranked_players_have_position_rank_and_rationale(named_player_projections, league_settings):
    ranked = rank_players_for_draft(named_player_projections, league_settings)
    qbs = [p for p in ranked if p["position"] == "QB"]
    assert {p["position_rank"] for p in qbs} == {1, 2}
    for player in ranked:
        assert player["rationale"]
        assert "vor" in player["rationale"] or "pts over positional replacement" in player["rationale"]


def test_n_teams_override_changes_replacement_levels(flex_pool_projections, flex_pool_settings):
    big_league = rank_players_for_draft(flex_pool_projections, flex_pool_settings, n_teams=2)
    tiny_league = rank_players_for_draft(flex_pool_projections, flex_pool_settings, n_teams=1)
    big_by_name = {p["name"]: p for p in big_league}
    tiny_by_name = {p["name"]: p for p in tiny_league}
    # A 1-team league only needs the single best RB as a dedicated starter, so
    # replacement level is RB2 -- a much closer comparison than the 2-team
    # league's replacement of RB4. RB1's surplus over replacement shrinks.
    assert tiny_by_name["RB1"]["vor"] < big_by_name["RB1"]["vor"]
    assert len(tiny_league) == len(big_league)


def test_roster_requirements_override_applies_without_mutating_league_settings(flex_pool_projections, flex_pool_settings):
    rank_players_for_draft(flex_pool_projections, flex_pool_settings, roster_requirements={"RB": 3, "WR": 0, "FLEX": 0})
    assert flex_pool_settings.roster_requirements.RB == 1  # original settings object must stay untouched


def test_generate_cheatsheet_truncates_to_top_n(named_player_projections, league_settings):
    cheatsheet = generate_cheatsheet(named_player_projections, league_settings, top_n=3)
    assert len(cheatsheet) == 3
    full = rank_players_for_draft(named_player_projections, league_settings)
    assert cheatsheet == full[:3]


def test_suggest_picks_prioritizes_position_of_need():
    # No FLEX slot, so the need multiplier is driven purely by dedicated
    # counts: RB is already saturated (2 rostered, needs 1) while WR is
    # empty (0 rostered, needs 1). RB has more raw points, but WR should
    # still win once the need penalty/boost is applied.
    settings = {
        "n_teams": 1,
        "scoring_mode": "ppr",
        "roster_requirements": {"QB": 0, "RB": 1, "WR": 1, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 0},
        "flex_eligible": [],
    }
    draft_state = {"league_settings": settings, "my_roster": [{"position": "RB"}, {"position": "RB"}]}
    available = [
        {"player_id": "rb", "name": "RB Guy", "position": "RB", "rushing_yards": 70},
        {"player_id": "wr", "name": "WR Guy", "position": "WR", "receiving_yards": 50},
    ]
    result = suggest_picks(draft_state, available, {"risk_tolerance": "balanced"})
    assert result["best_pick"]["position"] == "WR"
    assert len(result["alternatives"]) <= 4


def test_suggest_picks_empty_pool_returns_none():
    result = suggest_picks({"league_settings": {}, "my_roster": []}, [], {})
    assert result == {"best_pick": None, "alternatives": []}


def test_suggest_picks_invalid_risk_tolerance_raises():
    with pytest.raises(ValueError):
        suggest_picks({"my_roster": []}, [{"name": "X", "position": "RB"}], {"risk_tolerance": "reckless"})


def test_suggest_picks_boom_bust_favors_high_ceiling_over_safe_floor():
    safe_player = {
        "player_id": "safe",
        "name": "Safe Player",
        "position": "WR",
        "receiving_yards": 60,
        "floor": 8.0,
        "median": 9.0,
        "ceiling": 10.0,
    }
    volatile_player = {
        "player_id": "volatile",
        "name": "Volatile Player",
        "position": "WR",
        "receiving_yards": 60,
        "floor": 2.0,
        "median": 9.0,
        "ceiling": 25.0,
    }
    state = {"league_settings": {}, "my_roster": []}
    boom_bust = suggest_picks(state, [safe_player, volatile_player], {"risk_tolerance": "boom_bust"})
    safe = suggest_picks(state, [safe_player, volatile_player], {"risk_tolerance": "safe"})
    assert boom_bust["best_pick"]["name"] == "Volatile Player"
    assert safe["best_pick"]["name"] == "Safe Player"


def test_rationale_mentions_injury_risk_when_present():
    settings = {"n_teams": 1, "roster_requirements": {"RB": 1, "FLEX": 0}, "flex_eligible": []}
    hurt = {"player_id": "h", "name": "Hurt Guy", "position": "RB", "rushing_yards": 50, "status": "out"}
    questionable = {"player_id": "q", "name": "Iffy Guy", "position": "RB", "rushing_yards": 40, "status": "questionable"}
    ranked = rank_players_for_draft([hurt, questionable], settings)
    by_name = {p["name"]: p for p in ranked}
    assert "elevated injury risk" in by_name["Hurt Guy"]["rationale"]
    assert "mild injury risk" in by_name["Iffy Guy"]["rationale"]


def test_single_player_at_a_position_is_its_own_replacement():
    settings = {"n_teams": 1, "roster_requirements": {"RB": 1, "TE": 1, "FLEX": 0}, "flex_eligible": []}
    ranked = rank_players_for_draft([{"player_id": "rb", "name": "Only RB", "position": "RB", "rushing_yards": 40}], settings)
    assert ranked[0]["vor"] == 0.0
    assert ranked[0]["replacement_points"] == ranked[0]["points"]


def test_position_need_multiplier_neutral_when_exactly_at_requirement():
    settings = {"n_teams": 1, "roster_requirements": {"RB": 1, "WR": 1, "FLEX": 0}, "flex_eligible": []}
    draft_state = {"league_settings": settings, "my_roster": [{"position": "RB"}]}
    available = [{"player_id": "rb", "name": "RB Guy", "position": "RB", "rushing_yards": 50}]
    result = suggest_picks(draft_state, available, {})
    assert result["best_pick"]["need_multiplier"] == pytest.approx(1.0)


def test_suggest_picks_position_priority_boosts_score():
    rb = {"player_id": "rb", "name": "RB Guy", "position": "RB", "rushing_yards": 70}
    wr = {"player_id": "wr", "name": "WR Guy", "position": "WR", "receiving_yards": 70}
    state = {"league_settings": {}, "my_roster": []}
    without_priority = suggest_picks(state, [rb, wr], {})
    with_wr_priority = suggest_picks(state, [rb, wr], {"position_priority": ["WR"]})
    assert with_wr_priority["best_pick"]["name"] == "WR Guy"
    assert without_priority["best_pick"]["name"] in {"RB Guy", "WR Guy"}


def _draft_pool(n=20):
    players = []
    for i in range(n):
        players.append({"player_id": f"rb{i}", "name": f"RB{i}", "position": "RB", "rushing_yards": 100 - i * 4})
    return players


DRAFT_SETTINGS = {
    "n_teams": 4,
    "scoring_mode": "ppr",
    "roster_requirements": {"RB": 1, "WR": 0, "TE": 0, "QB": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 2},
    "flex_eligible": [],
}


def test_simulate_draft_same_seed_is_reproducible():
    first = simulate_draft(_draft_pool(), DRAFT_SETTINGS, rounds=3, seed=7)
    second = simulate_draft(_draft_pool(), DRAFT_SETTINGS, rounds=3, seed=7)
    assert first["picks"] == second["picks"]


def test_simulate_draft_snake_order_reverses_each_round():
    result = simulate_draft(_draft_pool(), DRAFT_SETTINGS, rounds=2, seed=1)
    round1_teams = [p["team"] for p in result["picks"] if p["round"] == 1]
    round2_teams = [p["team"] for p in result["picks"] if p["round"] == 2]
    assert round1_teams == ["Team 1", "Team 2", "Team 3", "Team 4"]
    assert round2_teams == ["Team 4", "Team 3", "Team 2", "Team 1"]


def test_simulate_draft_no_player_drafted_twice():
    result = simulate_draft(_draft_pool(), DRAFT_SETTINGS, rounds=3, seed=3)
    drafted_ids = [pick["player_id"] for pick in result["picks"]]
    assert len(drafted_ids) == len(set(drafted_ids))
    assert len(drafted_ids) == 4 * 3  # n_teams * rounds, pool is large enough


def test_simulate_draft_stops_gracefully_when_pool_runs_out():
    result = simulate_draft(_draft_pool(n=5), DRAFT_SETTINGS, rounds=3, seed=1)
    assert len(result["picks"]) == 5  # only 5 players existed, draft can't fill 12 slots


def test_simulate_draft_default_rounds_uses_total_roster_size():
    result = simulate_draft(_draft_pool(n=50), DRAFT_SETTINGS, seed=1)
    expected_rounds = sum(LeagueSettings(**DRAFT_SETTINGS).roster_requirements.model_dump().values())
    assert result["rounds"] == expected_rounds


def test_simulate_draft_rejects_invalid_rounds():
    with pytest.raises(ValueError):
        simulate_draft(_draft_pool(), DRAFT_SETTINGS, rounds=0)


def test_simulate_draft_rosters_match_picks():
    result = simulate_draft(_draft_pool(), DRAFT_SETTINGS, rounds=3, seed=5)
    total_rostered = sum(len(players) for players in result["rosters"].values())
    assert total_rostered == len(result["picks"])
