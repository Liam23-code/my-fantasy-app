"""Generate the bundled example data set: 200+ synthetic players plus the six
named fixture players used throughout the test suite.

Run once (already-generated output is committed under ``examples/``, so
running this again should be a no-op unless you want a larger/different
pool)::

    python examples/generate_sample_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).parent
SEED = 20260801

# Only QB/RB/WR/TE are generated: the engine's default scoring rules (see
# fantasy/scoring.py) only score passing/rushing/receiving stats, matching
# the spec's explicit default rule list -- there's no kicker or defense
# scoring to generate meaningful data for.
POSITION_PROFILES = {
    "QB": {"count": 40, "teams": 32},
    "RB": {"count": 70, "teams": 32},
    "WR": {"count": 80, "teams": 32},
    "TE": {"count": 40, "teams": 32},
}

NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

NAMED_PLAYERS = [
    {
        "player_id": "nfl:player:00-0034796", "name": "Lamar Jackson", "position": "QB", "team": "BAL", "opponent": "CLE",
        "season": 2026, "passing_yards": 245.0, "passing_tds": 1.8, "interceptions": 0.4, "rushing_yards": 65.0,
        "rushing_tds": 0.5, "receptions": 0.0, "receiving_yards": 0.0, "receiving_tds": 0.0, "fumbles_lost": 0.05,
        "floor": 8.2, "median": 15.6, "ceiling": 28.4, "drivers": ["pace", "red_zone_usage", "injury_risk"],
    },
    {
        "player_id": "nfl:player:00-0034857", "name": "Josh Allen", "position": "QB", "team": "BUF", "opponent": "KC",
        "season": 2026, "passing_yards": 219.5, "passing_tds": 1.65, "interceptions": 0.35, "rushing_yards": 31.2,
        "rushing_tds": 0.71, "receptions": 0.0, "receiving_yards": 0.0, "receiving_tds": 0.0, "fumbles_lost": 0.1,
        "floor": 14.0, "median": 22.5, "ceiling": 34.0, "drivers": ["pace", "goal_line_role"],
    },
    {
        "player_id": "nfl:player:00-0034844", "name": "Saquon Barkley", "position": "RB", "team": "PHI", "opponent": "DAL",
        "season": 2026, "passing_yards": 0.0, "passing_tds": 0.0, "interceptions": 0.0, "rushing_yards": 125.3,
        "rushing_tds": 0.81, "receptions": 2.1, "receiving_yards": 17.4, "receiving_tds": 0.12, "fumbles_lost": 0.05,
        "floor": 12.0, "median": 21.9, "ceiling": 34.5, "drivers": ["volume", "red_zone_share", "matchup"],
    },
    {
        "player_id": "nfl:player:00-0033280", "name": "Christian McCaffrey", "position": "RB", "team": "SF", "opponent": "SEA",
        "season": 2026, "passing_yards": 0.0, "passing_tds": 0.0, "interceptions": 0.0, "rushing_yards": 91.2,
        "rushing_tds": 0.85, "receptions": 3.9, "receiving_yards": 32.5, "receiving_tds": 0.42, "fumbles_lost": 0.05,
        "floor": 15.0, "median": 24.0, "ceiling": 38.0, "drivers": ["target_share", "goal_line_role"],
    },
    {
        "player_id": "nfl:player:00-0039075", "name": "Puka Nacua", "position": "WR", "team": "LA", "opponent": "SF",
        "season": 2026, "passing_yards": 0.0, "passing_tds": 0.0, "interceptions": 0.0, "rushing_yards": 0.0,
        "rushing_tds": 0.0, "receptions": 6.5, "receiving_yards": 92.0, "receiving_tds": 0.43, "fumbles_lost": 0.02,
        "floor": 9.0, "median": 17.5, "ceiling": 27.0, "drivers": ["target_share", "air_yards"],
    },
    {
        "player_id": "nfl:player:00-0030506", "name": "Travis Kelce", "position": "TE", "team": "KC", "opponent": "DEN",
        "season": 2026, "passing_yards": 0.0, "passing_tds": 0.0, "interceptions": 0.0, "rushing_yards": 0.0,
        "rushing_tds": 0.0, "receptions": 5.6, "receiving_yards": 47.4, "receiving_tds": 0.17, "fumbles_lost": 0.01,
        "floor": 4.0, "median": 10.5, "ceiling": 18.0, "drivers": ["red_zone_targets"],
    },
]


def _clip_normal(rng: np.random.Generator, mean: float, std: float, low: float = 0.0) -> float:
    return round(max(low, float(rng.normal(mean, std))), 2)


def _generate_qb(rng: np.random.Generator, index: int, team: str) -> dict:
    passing_yards = _clip_normal(rng, 210, 45)
    passing_tds = _clip_normal(rng, 1.4, 0.6)
    rushing_yards = _clip_normal(rng, 15, 18)
    median = round(passing_yards / 25 + passing_tds * 4 + rushing_yards / 10, 2)
    return {
        "player_id": f"synthetic:qb:{index}", "name": f"Synthetic QB {index}", "position": "QB", "team": team,
        "opponent": rng.choice(NFL_TEAMS), "season": 2026,
        "passing_yards": passing_yards, "passing_tds": passing_tds, "interceptions": _clip_normal(rng, 0.5, 0.3),
        "rushing_yards": rushing_yards, "rushing_tds": _clip_normal(rng, 0.15, 0.15),
        "receptions": 0.0, "receiving_yards": 0.0, "receiving_tds": 0.0, "fumbles_lost": _clip_normal(rng, 0.1, 0.08),
        "median": median, "floor": round(median * 0.55, 2), "ceiling": round(median * 1.65, 2),
        "drivers": ["pace", "matchup"],
    }


def _generate_rb(rng: np.random.Generator, index: int, team: str) -> dict:
    rushing_yards = _clip_normal(rng, 55, 28)
    receptions = _clip_normal(rng, 2.2, 1.6)
    receiving_yards = round(receptions * _clip_normal(rng, 7.5, 2.0, low=3.0), 2)
    median = round(rushing_yards / 10 + receptions * 1 + receiving_yards / 10, 2)
    return {
        "player_id": f"synthetic:rb:{index}", "name": f"Synthetic RB {index}", "position": "RB", "team": team,
        "opponent": rng.choice(NFL_TEAMS), "season": 2026,
        "passing_yards": 0.0, "passing_tds": 0.0, "interceptions": 0.0,
        "rushing_yards": rushing_yards, "rushing_tds": _clip_normal(rng, 0.35, 0.25),
        "receptions": receptions, "receiving_yards": receiving_yards, "receiving_tds": _clip_normal(rng, 0.08, 0.1),
        "fumbles_lost": _clip_normal(rng, 0.05, 0.05),
        "median": median, "floor": round(median * 0.5, 2), "ceiling": round(median * 1.7, 2),
        "drivers": ["volume", "red_zone_share"],
    }


def _generate_receiver(rng: np.random.Generator, index: int, team: str, position: str) -> dict:
    scale = 1.0 if position == "WR" else 0.75
    receptions = _clip_normal(rng, 4.0 * scale, 2.0)
    receiving_yards = round(receptions * _clip_normal(rng, 12.5, 3.0, low=4.0), 2)
    median = round(receptions * 1 + receiving_yards / 10, 2)
    return {
        "player_id": f"synthetic:{position.lower()}:{index}", "name": f"Synthetic {position} {index}", "position": position,
        "team": team, "opponent": rng.choice(NFL_TEAMS), "season": 2026,
        "passing_yards": 0.0, "passing_tds": 0.0, "interceptions": 0.0, "rushing_yards": 0.0, "rushing_tds": 0.0,
        "receptions": receptions, "receiving_yards": receiving_yards, "receiving_tds": _clip_normal(rng, 0.35 * scale, 0.2),
        "fumbles_lost": _clip_normal(rng, 0.03, 0.04),
        "median": median, "floor": round(median * 0.45, 2), "ceiling": round(median * 1.8, 2),
        "drivers": ["target_share", "air_yards"],
    }


def generate_players(seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    players = list(NAMED_PLAYERS)
    for position, profile in POSITION_PROFILES.items():
        for index in range(profile["count"]):
            team = NFL_TEAMS[index % len(NFL_TEAMS)]
            if position == "QB":
                players.append(_generate_qb(rng, index, team))
            elif position == "RB":
                players.append(_generate_rb(rng, index, team))
            else:
                players.append(_generate_receiver(rng, index, team, position))
    return players


def main() -> None:
    players = generate_players()
    (EXAMPLES_DIR / "sample_projections.json").write_text(json.dumps(players, indent=2), encoding="utf-8")

    league_settings = {
        "n_teams": 12,
        "scoring_mode": "ppr",
        "roster_requirements": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1, "BENCH": 6},
        "flex_eligible": ["RB", "WR", "TE"],
        "max_players_per_nfl_team": 8,
        "faab_budget": 100,
    }
    (EXAMPLES_DIR / "sample_league_settings.json").write_text(json.dumps(league_settings, indent=2), encoding="utf-8")

    roster_player_ids = [p["player_id"] for p in players[:9]]
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BENCH", "BENCH"]
    roster = [
        {"player_id": pid, "name": p["name"], "position": p["position"], "nfl_team": p["team"], "slot": slot}
        for pid, p, slot in zip(roster_player_ids, players[:9], slots)
    ]
    (EXAMPLES_DIR / "sample_roster.json").write_text(json.dumps(roster, indent=2), encoding="utf-8")

    rostered_ids = {p["player_id"] for p in roster}
    available = [p for p in players if p["player_id"] not in rostered_ids][:25]
    (EXAMPLES_DIR / "sample_available_players.json").write_text(json.dumps(available, indent=2), encoding="utf-8")

    print(f"Wrote {len(players)} players to sample_projections.json")
    print("Wrote sample_league_settings.json, sample_roster.json, sample_available_players.json")


if __name__ == "__main__":
    main()
