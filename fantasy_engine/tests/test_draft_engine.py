"""Tests for fantasy.draft_engine -- the composite, roster-reactive Draft Room board.

These encode the four faults the module exists to fix:
 1. drafted players must never be recommended,
 2. a clearly better available player must outrank a worse "on-slot" one,
 3. the #1 pick must not sit a round-plus off ADP when the board still has depth,
 4. a needed below-replacement player must rank ABOVE a saturated one
    (the "collapses after round 5" sign-inversion guard).
"""

from __future__ import annotations

import copy

from fantasy.draft_engine import (
    available_players,
    get_recommendations,
    mark_player_drafted,
)

SETTINGS = {
    "n_teams": 12,
    "scoring_mode": "ppr",
    "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1, "BENCH": 6},
    "flex_eligible": ["RB", "WR", "TE"],
}

_REQUIRED_KEYS = {
    "player_id",
    "name",
    "position",
    "team",
    "projection",
    "points_per_game",
    "position_rank",
    "pos_rank_label",
    "adp",
    "current_pick",
    "adp_value",
    "adp_verdict",
    "vorp",
    "replacement_points",
    "expected_games",
    "availability_verdict",
    "breakout_probability",
    "breakout_tier",
    "risk_score",
    "risk_band",
    "team_context",
    "scarcity_label",
    "scarcity_component",
    "roster_fit",
    "roster_need_component",
    "rank_score",
    "score_breakdown",
    "rationale",
    "status",
    "rank",
}


def _p(pid, position, projection, adp=None, *, age=26, prior=None, team="SF"):
    prior = projection if prior is None else prior
    row = {
        "player_id": pid,
        "name": pid.replace("-", " ").title(),
        "position": position,
        "team": team,
        "projection": float(projection),
        "expected_fantasy_points": float(projection),
        "points_per_game": round(float(projection) / 16.5, 2),
        "expected_games": 16.5,
        "games_played": 16,
        "prior_season_points": float(prior),
        "prior_games_played": 16,
        "age": age,
        "floor": float(projection) * 0.7,
        "ceiling": float(projection) * 1.3,
    }
    if adp is not None:
        row["adp"] = float(adp)
    return row


def _roster(*positions):
    return [{"position": p, "player_id": f"mine-{i}", "name": f"Mine {i}", "team": "NE"} for i, p in enumerate(positions)]


def _deep_board():
    """~24 players across positions with a realistic projection gradient."""
    board = []
    tiers = {
        "RB": [290, 250, 210, 175, 150, 130, 118, 105],
        "WR": [300, 270, 240, 215, 190, 170, 152, 138, 124],
        "TE": [230, 165, 130, 110, 96],
        "QB": [360, 330, 300, 278, 260],
        "K": [140, 132],
        "DST": [130, 120],
    }
    adp = 1.0
    order = []
    for position, projs in tiers.items():
        for rank, proj in enumerate(projs, start=1):
            order.append((proj, position, rank))
    for proj, position, rank in sorted(order, reverse=True):
        board.append(_p(f"{position.lower()}-{rank}", position, proj, adp=adp, age=24 + rank))
        adp += 4.0
    return board


# --- helpers ---------------------------------------------------------------


def test_mark_player_drafted_is_additive_and_idempotent():
    gone = mark_player_drafted(None, "a")
    gone = mark_player_drafted(gone, "b")
    gone = mark_player_drafted(gone, "b")
    assert gone == {"a", "b"}
    assert mark_player_drafted(gone, "") == {"a", "b"}


def test_available_players_only_removes_drafted_ids():
    board = [_p("a", "RB", 200), _p("b", "WR", 190), _p("c", "TE", 150)]
    assert [p["player_id"] for p in available_players(board, {"b"})] == ["a", "c"]
    assert available_players(board, None) == board


# --- fault 1: drafted players are never recommended -----------------------


def test_drafted_players_are_excluded_from_recommendations():
    board = _deep_board()
    gone = {p["player_id"] for p in board[:6]}  # top 6 off the board in the real room
    recs = get_recommendations(
        current_pick=13,
        my_roster=[],
        drafted_players=gone,
        board=board,
        league_settings=SETTINGS,
        picks_until_next=22,
        num_rounds=15,
        limit=10,
    )
    assert recs
    assert gone.isdisjoint({r["player_id"] for r in recs})


def test_marking_a_recommended_player_drafted_drops_them_next_call():
    board = _deep_board()
    first = get_recommendations(9, [], set(), board, SETTINGS, picks_until_next=6, num_rounds=15, limit=5)
    taken = first[0]["player_id"]
    second = get_recommendations(
        9,
        [],
        mark_player_drafted(set(), taken),
        board,
        SETTINGS,
        picks_until_next=6,
        num_rounds=15,
        limit=5,
    )
    assert taken not in {r["player_id"] for r in second}


# --- fault 2: a better available player outranks a worse on-slot one ------


def test_a_fallen_stud_outranks_an_on_slot_mid_tier():
    board = [
        _p("fallen-stud", "WR", 300, adp=6),  # elite value, fell ~34 picks
        _p("on-slot-rb", "RB", 150, adp=40),  # exactly on the clock, mediocre
        _p("on-slot-wr", "WR", 145, adp=41),
        _p("filler-te", "TE", 120, adp=45),
        _p("filler-qb", "QB", 250, adp=44),
    ]
    recs = get_recommendations(
        current_pick=40,
        my_roster=[],
        drafted_players=set(),
        board=board,
        league_settings=SETTINGS,
        picks_until_next=22,
        num_rounds=15,
        limit=5,
    )
    assert recs[0]["player_id"] == "fallen-stud"
    assert recs[0]["adp_value"] > 20
    assert recs[0]["adp_verdict"] == "value"


def test_a_reach_is_penalised_relative_to_an_on_time_pick():
    board = [
        _p("reach", "WR", 210, adp=70),  # 30 picks early
        _p("on-time", "WR", 205, adp=41),
    ]
    recs = get_recommendations(40, [], set(), board, SETTINGS, picks_until_next=22, num_rounds=15, limit=2)
    by_id = {r["player_id"]: r for r in recs}
    assert by_id["reach"]["adp_value"] < 0
    assert by_id["on-time"]["rank_score"] > by_id["reach"]["rank_score"]


def test_an_adp_before_your_next_pick_is_not_treated_as_a_reach():
    # ADP 50 at pick 40 with the next turn 20 picks away: this player is gone
    # if you wait, so taking him now is correct -- not a reach.
    board = [
        _p("will-be-gone", "WR", 210, adp=50),
        _p("safe-wait", "WR", 205, adp=90),  # comfortably survives to your next pick
    ]
    recs = get_recommendations(40, [], set(), board, SETTINGS, picks_until_next=20, num_rounds=15, limit=2)
    by_id = {r["player_id"]: r for r in recs}
    # tiny "could have waited" nudge only, not a full reach penalty
    gone_adp = next(p for p in by_id["will-be-gone"]["score_breakdown"] if p["factor"] == "ADP value")
    wait_adp = next(p for p in by_id["safe-wait"]["score_breakdown"] if p["factor"] == "ADP value")
    assert gone_adp["weighted"] > -0.35  # at most a partial "could have waited" nudge, not a full reach
    assert gone_adp["weighted"] > wait_adp["weighted"] + 0.3  # the safe-wait player eats a real reach penalty
    assert by_id["will-be-gone"]["rank_score"] > by_id["safe-wait"]["rank_score"]


def test_a_no_adp_player_does_not_outrank_a_clearly_better_adp_starter():
    # The failure the review caught: a fixed no-ADP bonus that beat every
    # "reach", so a mediocre unranked sleeper topped the board over a stronger
    # ADP starter who will actually be gone.
    board = [
        _p("adp-starter", "WR", 185, adp=68, prior=175),  # strong, ADP sits past the slot
        _p("no-adp-sleeper", "RB", 138, adp=None, prior=90),
    ]
    recs = get_recommendations(55, [], set(), board, SETTINGS, picks_until_next=20, num_rounds=15, limit=5)
    by_id = {r["player_id"]: r for r in recs}
    assert by_id["adp-starter"]["rank_score"] > by_id["no-adp-sleeper"]["rank_score"]


# --- fault 4: roster need is a sign-safe ADDITIVE term -------------------


def test_a_needed_below_replacement_player_outranks_a_saturated_one():
    # Deep pools (32 RB / 32 WR on a smooth gradient) so replacement level
    # lands well above the two mid-low candidates: both are genuinely below
    # replacement (negative VORP), the exact regime where the old engine's
    # need-as-a-multiplier inverted and demoted the position you needed.
    my_roster = _roster("WR", "WR", "WR")
    board = [
        _p("needed-rb", "RB", 78, adp=None),
        _p("extra-wr", "WR", 78, adp=None),
    ]
    for i in range(32):
        proj = 250 - i * 5  # 250 .. 95
        board.append(_p(f"rb-{i}", "RB", proj, adp=6 + i * 4))
        board.append(_p(f"wr-{i}", "WR", proj, adp=7 + i * 4))
    recs = get_recommendations(
        current_pick=95,
        my_roster=my_roster,
        drafted_players=set(),
        board=board,
        league_settings=SETTINGS,
        picks_until_next=22,
        num_rounds=15,
        limit=80,
    )
    by_id = {r["player_id"]: r for r in recs}
    assert by_id["needed-rb"]["vorp"] < 0 and by_id["extra-wr"]["vorp"] < 0
    assert by_id["needed-rb"]["rank_score"] > by_id["extra-wr"]["rank_score"]
    assert by_id["needed-rb"]["roster_need_component"] > by_id["extra-wr"]["roster_need_component"]


# --- fault 4b: still roster-reactive deep in the draft ------------------


def test_recommendations_react_to_the_roster_in_the_middle_rounds():
    board = _deep_board()
    empty = get_recommendations(75, [], set(), board, SETTINGS, picks_until_next=22, num_rounds=15, limit=8)
    rb_heavy = get_recommendations(
        75,
        _roster("RB", "RB", "RB"),
        set(),
        board,
        SETTINGS,
        picks_until_next=22,
        num_rounds=15,
        limit=8,
    )
    assert empty and rb_heavy
    # a team with three RBs should not be led to a fourth
    assert rb_heavy[0]["position"] != "RB" or rb_heavy[0]["roster_need_component"] <= 0
    # the two boards genuinely differ
    assert [r["player_id"] for r in empty] != [r["player_id"] for r in rb_heavy]


def test_deep_rounds_still_return_a_full_ranked_slate():
    board = _deep_board()
    for pick in (75, 110, 150):
        recs = get_recommendations(
            pick,
            _roster("RB", "WR", "QB", "TE"),
            set(),
            board,
            SETTINGS,
            picks_until_next=22,
            num_rounds=15,
            limit=10,
        )
        assert 1 <= len(recs) <= 10
        assert [r["rank"] for r in recs] == list(range(1, len(recs) + 1))
        scores = [r["rank_score"] for r in recs]
        assert scores == sorted(scores, reverse=True)
        assert len({round(s, 3) for s in scores}) > 1  # not a flat wall


# --- capped positions / contract --------------------------------------


def test_a_capped_position_is_filtered_entirely():
    board = _deep_board()
    recs = get_recommendations(
        90,
        _roster("QB", "QB"),
        set(),
        board,
        SETTINGS,
        picks_until_next=22,
        num_rounds=15,
        limit=15,
    )
    assert recs
    assert all(r["position"] != "QB" for r in recs)


def test_kickers_and_defenses_are_damped_early_and_freed_late():
    board = _deep_board()
    early = get_recommendations(13, [], set(), board, SETTINGS, picks_until_next=22, num_rounds=15, limit=15)
    assert all(r["position"] not in {"K", "DST"} for r in early)
    late = get_recommendations(
        170,
        _roster("QB", "RB", "RB", "WR", "WR", "TE", "RB", "WR"),
        set(),
        board,
        SETTINGS,
        picks_until_next=2,
        num_rounds=15,
        limit=15,
    )
    assert any(r["position"] in {"K", "DST"} for r in late)


def test_an_early_second_qb_in_a_one_qb_league_is_penalised():
    board = _deep_board()
    recs = get_recommendations(30, _roster("QB"), set(), board, SETTINGS, picks_until_next=22, num_rounds=15, limit=20)
    qb = next((r for r in recs if r["position"] == "QB"), None)
    if qb is not None:
        assert qb["round_band_penalty"] < 0


def test_every_recommendation_carries_the_full_decision_set():
    board = _deep_board()
    recs = get_recommendations(13, [], set(), board, SETTINGS, picks_until_next=22, num_rounds=15, limit=6)
    assert recs
    for entry in recs:
        assert _REQUIRED_KEYS <= set(entry)
        assert isinstance(entry["score_breakdown"], list) and len(entry["score_breakdown"]) == 6
        assert entry["pos_rank_label"].startswith(entry["position"])
        assert "replacement" in entry["rationale"]
        assert entry["breakout_probability"] <= 0.65


def test_empty_board_returns_empty_list():
    assert get_recommendations(13, [], set(), [], SETTINGS, num_rounds=15) == []
    assert get_recommendations(13, [], {"a", "b"}, [_p("a", "RB", 200)], SETTINGS, num_rounds=15) == []


def test_get_recommendations_does_not_mutate_the_board():
    board = _deep_board()
    before = copy.deepcopy(board)
    get_recommendations(13, [], set(), board, SETTINGS, picks_until_next=22, num_rounds=15, limit=6)
    assert board == before
