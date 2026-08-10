"""Unit tests for fantasy.draft."""

from __future__ import annotations

from collections import Counter

import pytest

from fantasy.draft import (
    active_run_position,
    _replacement_levels,
    build_draft_order,
    generate_cheatsheet,
    identify_adp_clusters,
    rank_players_for_draft,
    simulate_draft,
    suggest_picks,
)
from fantasy.models import ROSTER_POSITION_LIMITS, LeagueSettings


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


# --- draft order & round-by-round board -------------------------------------


def test_build_draft_order_snakes_on_even_rounds():
    order = build_draft_order(4, 3, snake=True)
    rounds = {r: [p["team_number"] for p in order if p["round"] == r] for r in (1, 2, 3)}
    assert rounds[1] == [1, 2, 3, 4]
    assert rounds[2] == [4, 3, 2, 1]
    assert rounds[3] == [1, 2, 3, 4]


def test_build_draft_order_linear_repeats_the_same_order():
    order = build_draft_order(4, 3, snake=False)
    for round_number in (1, 2, 3):
        assert [p["team_number"] for p in order if p["round"] == round_number] == [1, 2, 3, 4]


def test_build_draft_order_numbers_picks_consecutively():
    order = build_draft_order(4, 3)
    assert [p["overall_pick"] for p in order] == list(range(1, 13))


@pytest.mark.parametrize(("teams", "rounds"), [(0, 3), (4, 0), (-1, 2)])
def test_build_draft_order_rejects_invalid_shapes(teams, rounds):
    with pytest.raises(ValueError):
        build_draft_order(teams, rounds)


def test_simulate_draft_excludes_synthetic_players_and_warns():
    pool = _draft_pool(n=6) + [
        {"player_id": f"synthetic:rb:{i}", "name": f"Synthetic RB {i}", "position": "RB", "rushing_yards": 999}
        for i in range(10)
    ]
    result = simulate_draft(pool, DRAFT_SETTINGS, rounds=3, seed=1)
    assert all(not pick["player_id"].startswith("synthetic") for pick in result["picks"])
    assert any("synthetic" in warning for warning in result["warnings"])


def test_simulate_draft_by_round_matches_the_flat_pick_list():
    result = simulate_draft(_draft_pool(n=40), DRAFT_SETTINGS, rounds=3, seed=11)
    flattened = [pick for round_number in sorted(result["by_round"]) for pick in result["by_round"][round_number]]
    assert len(flattened) == len(result["picks"])
    assert [entry["pick"] for entry in flattened] == [pick["overall_pick"] for pick in result["picks"]]
    assert set(result["by_round"]) == {1, 2, 3}


def test_simulate_draft_by_round_entries_have_the_documented_shape():
    result = simulate_draft(_draft_pool(n=40), DRAFT_SETTINGS, rounds=2, seed=2)
    entry = result["by_round"][1][0]
    assert set(entry) >= {"pick", "team", "player_name", "position", "projection", "is_user_pick"}


def test_simulate_draft_flags_the_users_picks():
    result = simulate_draft(_draft_pool(n=40), DRAFT_SETTINGS, rounds=3, seed=4, user_draft_slot=2)
    assert result["user_team"] == "Team 2"
    assert len(result["user_picks"]) == 3
    assert all(pick["team"] == "Team 2" for pick in result["user_picks"])
    assert all(pick["is_user_pick"] for pick in result["user_picks"])


def test_simulate_draft_without_a_user_slot_flags_nobody():
    result = simulate_draft(_draft_pool(n=40), DRAFT_SETTINGS, rounds=2, seed=4)
    assert result["user_team"] is None
    assert result["user_picks"] == []


def test_simulate_draft_rejects_a_user_slot_outside_the_league():
    with pytest.raises(ValueError):
        simulate_draft(_draft_pool(), DRAFT_SETTINGS, rounds=2, user_draft_slot=99)


def test_simulate_draft_linear_keeps_team_order_every_round():
    result = simulate_draft(_draft_pool(n=40), DRAFT_SETTINGS, rounds=2, seed=6, snake=False)
    for round_number in (1, 2):
        teams = [pick["team"] for pick in result["by_round"][round_number]]
        assert teams == ["Team 1", "Team 2", "Team 3", "Team 4"]


def test_simulate_draft_num_teams_and_num_rounds_override_settings():
    result = simulate_draft(_draft_pool(n=60), DRAFT_SETTINGS, num_teams=6, num_rounds=4, seed=8)
    assert result["n_teams"] == 6
    assert result["rounds"] == 4
    assert len(result["picks"]) == 24


def test_simulate_draft_warns_when_the_pool_cannot_fill_the_draft():
    result = simulate_draft(_draft_pool(n=5), DRAFT_SETTINGS, rounds=3, seed=1)
    assert len(result["picks"]) == 5
    assert any("real players" in warning for warning in result["warnings"])


# --- Hybrid Draft Intelligence Model: ADP clusters & roster caps ------------


def _adp_pool():
    players = []
    for i in range(5):
        players.append({"player_id": f"rb{i}", "name": f"RB{i}", "position": "RB", "rushing_yards": 200 - i * 5, "adp": 1.0 + i})
    for i in range(5):
        players.append({"player_id": f"wr{i}", "name": f"WR{i}", "position": "WR", "receiving_yards": 180 - i * 5, "adp": 50.0 + i * 20})
    return players


def test_identify_adp_clusters_finds_a_tight_bunch():
    clusters = identify_adp_clusters(_adp_pool(), max_gap=6.0, min_size=3)
    positions = {c["position"] for c in clusters}
    assert "RB" in positions  # ADPs 1..5, all within max_gap of each other
    assert "WR" not in positions  # ADPs spaced 20 apart, no cluster


def test_identify_adp_clusters_ignores_players_without_adp():
    pool = [{"player_id": "x", "name": "NoADP", "position": "RB", "rushing_yards": 100}]
    assert identify_adp_clusters(pool) == []


def test_identify_adp_clusters_ignores_kicker_and_dst():
    pool = [
        {"player_id": f"k{i}", "name": f"K{i}", "position": "K", "adp": 100.0 + i}
        for i in range(5)
    ]
    assert identify_adp_clusters(pool) == []


def test_identify_adp_clusters_respects_min_size():
    pool = [{"player_id": f"rb{i}", "name": f"RB{i}", "position": "RB", "adp": 1.0 + i} for i in range(2)]
    assert identify_adp_clusters(pool, min_size=3) == []


def test_active_run_position_covers_the_lookahead_and_trailing_window():
    clusters = identify_adp_clusters(_adp_pool(), max_gap=6.0, min_size=3)
    rb_cluster = next(c for c in clusters if c["position"] == "RB")
    just_before = int(rb_cluster["start_adp"]) - 1  # inside the 3-pick lookahead
    assert active_run_position(just_before, clusters) == "RB"
    far_before = int(rb_cluster["start_adp"]) - 10
    assert active_run_position(far_before, clusters) is None


def test_simulate_draft_never_exceeds_a_positions_roster_cap():
    """QB cap is (1, 2): no simulated team should ever roster a 3rd QB.

    6 rounds so each team's cap budget (QB max 2 + RB max 5 = 7) comfortably
    covers the roster -- the point is to prove the cap holds when the draft
    *can* respect it, not to exercise the pool-exhausted fallback (see
    ``test_simulate_draft_relaxes_caps_rather_than_stalling`` for that).
    """
    pool = []
    for i in range(30):
        pool.append({"player_id": f"qb{i}", "name": f"QB{i}", "position": "QB", "passing_yards": 4000 - i * 50, "adp": 1.0 + i})
    for i in range(30):
        pool.append({"player_id": f"rb{i}", "name": f"RB{i}", "position": "RB", "rushing_yards": 1200 - i * 20, "adp": 40.0 + i})

    settings = {
        "n_teams": 2,
        "scoring_mode": "ppr",
        "roster_requirements": {"QB": 1, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 10},
        "flex_eligible": [],
    }
    result = simulate_draft(pool, settings, rounds=6, seed=3)
    for roster in result["rosters"].values():
        qb_count = sum(1 for p in roster if p["position"] == "QB")
        assert qb_count <= 2


def test_simulate_draft_relaxes_caps_rather_than_stalling():
    """When a team's only remaining options are all roster-capped, draft anyway.

    2 teams x 12 rounds = 12 picks/team, but QB(max 2) + RB(max 5) = 7 < 12
    and this pool has no other position -- caps cannot be satisfied for the
    full draft. The engine should keep drafting (fail open) instead of
    stalling once every candidate is capped out.
    """
    pool = []
    for i in range(30):
        pool.append({"player_id": f"qb{i}", "name": f"QB{i}", "position": "QB", "passing_yards": 4000 - i * 50, "adp": 1.0 + i})
    for i in range(30):
        pool.append({"player_id": f"rb{i}", "name": f"RB{i}", "position": "RB", "rushing_yards": 1200 - i * 20, "adp": 40.0 + i})

    settings = {
        "n_teams": 2,
        "scoring_mode": "ppr",
        "roster_requirements": {"QB": 1, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 10},
        "flex_eligible": [],
    }
    result = simulate_draft(pool, settings, rounds=12, seed=3)
    assert len(result["picks"]) == 24  # every slot still got filled
    total_qbs = sum(1 for roster in result["rosters"].values() for p in roster if p["position"] == "QB")
    assert total_qbs > 2 * 2  # caps had to be exceeded somewhere to fill 24 slots from 2 positions


def test_simulate_draft_returns_the_position_runs_it_detected():
    result = simulate_draft(_adp_pool() * 3, DRAFT_SETTINGS, rounds=2, seed=1)
    assert isinstance(result["position_runs"], list)


def test_simulate_draft_pick_entries_carry_a_position_run_field():
    result = simulate_draft(_draft_pool(n=40), DRAFT_SETTINGS, rounds=2, seed=2)
    assert all("position_run" in pick for pick in result["picks"])
    assert all("position_run" in pick for round_picks in result["by_round"].values() for pick in round_picks)


# --- two-stage roster-cap emergency fallback (Section 2/4) ------------------


def test_simulate_draft_never_lets_kicker_or_dst_exceed_cap_when_skill_depth_remains():
    """K cap is (1, 1): stage-1 (skill-position) relaxation must never touch it.

    Plenty of RB depth remains available (never exhausted), so once the team
    has its 1 kicker, every later pick should keep coming from RB rather than
    a 2nd K -- this isolates stage-1 relaxation from the true (stage-2)
    emergency exercised by the pool-exhaustion test above.
    """
    pool = [
        {"player_id": f"rb{i}", "name": f"RB{i}", "position": "RB", "rushing_yards": 200 - i, "adp": 1.0 + i}
        for i in range(20)
    ] + [{"player_id": "k1", "name": "K1", "position": "K", "adp": 50.0}]
    settings = {
        "n_teams": 1,
        "scoring_mode": "ppr",
        "roster_requirements": {"RB": 1, "K": 1, "QB": 0, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "BENCH": 3},
        "flex_eligible": [],
    }
    result = simulate_draft(pool, settings, rounds=5, seed=1)
    k_count = sum(1 for p in result["rosters"]["Team 1"] if p["position"] == "K")
    assert k_count == 1
    assert result["cap_emergencies"] == 0


def test_simulate_draft_allows_exceeding_kicker_cap_only_as_a_true_last_resort():
    """When literally nothing but a capped-out K remains anywhere in the pool,
    stage 2 must still draft *something* rather than stall -- and this really
    is the rare, tracked "true emergency", not routine cap enforcement.
    """
    pool = [
        {"player_id": "rb1", "name": "RB1", "position": "RB", "rushing_yards": 200, "adp": 1.0},
        {"player_id": "rb2", "name": "RB2", "position": "RB", "rushing_yards": 150, "adp": 2.0},
        {"player_id": "k1", "name": "K1", "position": "K", "adp": 50.0},
        {"player_id": "k2", "name": "K2", "position": "K", "adp": 51.0},
        {"player_id": "k3", "name": "K3", "position": "K", "adp": 52.0},
    ]
    settings = {
        "n_teams": 1,
        "scoring_mode": "ppr",
        "roster_requirements": {"RB": 1, "K": 1, "QB": 0, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "BENCH": 3},
        "flex_eligible": [],
    }
    result = simulate_draft(pool, settings, rounds=5, seed=1)
    assert len(result["picks"]) == 5  # every round still filled, pool had exactly 5 players
    assert result["cap_emergencies"] > 0


def test_simulate_draft_relaxes_skill_position_caps_before_a_true_emergency():
    """Running out of real positions (not K/DST) should not count as a cap_emergency."""
    pool = [
        {"player_id": f"rb{i}", "name": f"RB{i}", "position": "RB", "rushing_yards": 200 - i, "adp": 1.0 + i}
        for i in range(10)
    ]
    settings = {
        "n_teams": 1,
        "scoring_mode": "ppr",
        "roster_requirements": {"RB": 1, "QB": 0, "WR": 0, "TE": 0, "FLEX": 0, "DST": 0, "K": 0, "BENCH": 9},
        "flex_eligible": [],
    }
    # RB cap max is 4, but 10 rounds requested -- every pick past the 4th RB
    # must come from the stage-1 skill-position relaxation, not a true emergency.
    result = simulate_draft(pool, settings, rounds=10, seed=1)
    assert len(result["picks"]) == 10
    assert result["cap_emergencies"] == 0


def test_simulate_draft_reports_cap_emergencies_in_the_result():
    result = simulate_draft(_draft_pool(n=40), DRAFT_SETTINGS, rounds=2, seed=1)
    assert "cap_emergencies" in result
    assert isinstance(result["cap_emergencies"], int)


# --- roster-cap capacity warning ---------------------------------------------


def test_simulate_draft_warns_when_rounds_exceed_cap_capacity():
    """RB+WR+TE+QB+K caps sum to 14; a 15-round draft cannot fit inside them.

    DST carries a (1, 1) cap but nflverse has no DST data, so its slot never
    absorbs a pick -- every team is forced one over a cap on its last pick.
    """
    pool = []
    for position, count in (("RB", 40), ("WR", 40), ("TE", 20), ("QB", 20), ("K", 20)):
        for i in range(count):
            pool.append({"player_id": f"{position}{i}", "name": f"{position}{i}", "position": position, "adp": float(i + 1)})
    settings = {
        "n_teams": 2,
        "scoring_mode": "ppr",
        "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 0, "K": 1, "BENCH": 6},
        "flex_eligible": ["RB", "WR", "TE"],
    }
    result = simulate_draft(pool, settings, rounds=15, seed=1)
    assert any("exceeds the 14 roster spots" in warning for warning in result["warnings"])
    assert result["cap_relaxations"] > 0


def test_simulate_draft_stays_inside_caps_when_rounds_fit():
    pool = []
    for position, count in (("RB", 40), ("WR", 40), ("TE", 20), ("QB", 20), ("K", 20)):
        for i in range(count):
            pool.append({"player_id": f"{position}{i}", "name": f"{position}{i}", "position": position, "adp": float(i + 1)})
    settings = {
        "n_teams": 2,
        "scoring_mode": "ppr",
        "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 0, "K": 1, "BENCH": 5},
        "flex_eligible": ["RB", "WR", "TE"],
    }
    result = simulate_draft(pool, settings, rounds=14, seed=1)
    assert result["cap_relaxations"] == 0
    assert result["cap_emergencies"] == 0
    for roster in result["rosters"].values():
        counts = Counter(p["position"] for p in roster)
        for position, (_lo, hi) in ROSTER_POSITION_LIMITS.items():
            assert counts.get(position, 0) <= hi
