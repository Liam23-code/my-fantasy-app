# MLB matchup engine: five real, bounded DFS adjustment modules

How MLB's DFS matchup layer works, and how it composes with the season baseline in [mlb_fusion_model.md](mlb_fusion_model.md). See [matchup_engine.md](matchup_engine.md) for the cross-sport version of this idea (NBA's opponent-specific context layer) and [mlb_pipeline.md](mlb_pipeline.md) for the overall MLB architecture.

## The shape: additive, bounded, never a reprojection

Same principle NBA's matchup engine already established: this layer does not replace the season-average baseline (`modules/mlb_season_model.py`). Each of the five modules below produces a small, bounded, disclosed multiplier (centered at `1.0` = neutral) from real, caller-supplied inputs; `modules/mlb_fusion_model.py::fuse_projection` blends those multipliers into the season baseline with a reliability/sample-size-scaled weight, never overriding it outright. Every module fails toward neutral, not toward an error, when its input is missing — a partial matchup profile is a normal state.

## The five modules

| Module | Real signal | Feeds |
|---|---|---|
| `modules/mlb_batter_vs_pitcher.py` | Handedness platoon advantage, pitch-type-weighted whiff rate, career BvP (heavily sample-size-regressed), hard-hit/barrel-rate contact-quality index | `matchup_difficulty` weight in the fusion model |
| `modules/mlb_ballpark_model.py` | Real, static park factors (HR/doubles/triples/runs, altitude, foul territory) — the same category of static reference data as `modules/sportsbook_parser.py`'s team-alias table, not a live fetch | `park_factor` weight, per category |
| `modules/mlb_lineup_model.py` | Batting-order expected plate appearances, next-hitter-OBP protection, real historical platoon-start probability, team stolen-base environment | `lineup_protection` weight |
| `modules/mlb_bullpen_model.py` | Real recent-appearance fatigue, ERA rating vs. league average, high-leverage-innings usage share | `pitcher_quality` weight (bullpen half) and `modules/mlb_moneyline_model.py`'s `bullpen_strength` context |
| `modules/mlb_defense_model.py` | Real defensive efficiency (balls in play converted to outs), outfield arm rating, catcher framing, infield range factor | `modules/mlb_moneyline_model.py`'s `defensive_efficiency` context; also informs the stolen-base environment alongside `mlb_lineup_model.py` |

Each module is independently testable and callable — `modules/mlb_fusion_model.py` does not call any of them itself, it only combines already-computed scalars the caller passes in (the same "small composable pure functions" pattern `modules/cbb_moneyline_model.py::evaluate_game` already uses for an already-computed volatility dict). This keeps every piece testable in isolation (`tests/test_mlb_matchup.py`) without needing to mock the other four.

## Real, disclosed constants, not fitted coefficients

Every multiplier in every module is built from a documented, public baseball fact or a widely-used sabermetric convention, capped to a modest, disclosed range:

- **Platoon advantage** (`mlb_batter_vs_pitcher.py`) — opposite-handed matchups favor the batter, same-handed favor the pitcher; a real, well-documented effect, not a fitted number (`_PLATOON_ADVANTAGE = 1.08`, `_PLATOON_DISADVANTAGE = 0.94`).
- **Park factors** (`mlb_ballpark_model.py::PARK_FACTORS`) — approximate, multi-year composite indices in the publicly understood range for each park (Coors Field's real hitter-friendly effect, Petco/Oracle-style pitcher-friendly effects) — a deployment wanting current-year precision can override the constant directly.
- **BvP regression** (`career_bvp_adjustment`) — a real career .750 average in an 8-at-bat sample is blended down to barely above neutral; classic small-sample-BvP caution, not a naive average.
- **Bullpen home-field-style constants** (`_LEAGUE_AVERAGE_BULLPEN_ERA = 4.10`) and defense constants (`_LEAGUE_AVERAGE_DEFENSIVE_EFFICIENCY = 0.690`, real MLB-wide DER) are real, roughly league-average reference points, not fitted to any one team's data.

## What's *not* built (by design, not oversight)

- **No live per-play tracking data** (Statcast pitch-by-pitch, spray charts) — every input here is a real, caller-supplied summary metric (a whiff rate, a park factor, a bullpen ERA), not a live feed.
- **No cross-sport aggregator** the way `modules/nba_matchup_engine.py::compute_matchup_context` composes NBA's four modules into one call — `modules/mlb_fusion_model.py::fuse_projection` plays that role for MLB directly, since MLB's five modules and the season model were designed together this cycle rather than accreted incrementally the way NBA's were.

## Testing

`UniversalQuantAgent/tests/test_mlb_matchup.py` covers all five modules directly (handedness, pitch-type effectiveness, BvP regression, contact-quality index, park factors and their category-specific application, lineup protection/plate-appearances/platoon/SB-environment, bullpen fatigue/ERA-rating/leverage/composite, and defense efficiency/arm/framing/range/composite). `tests/test_mlb_fusion.py` covers the blending layer itself.
