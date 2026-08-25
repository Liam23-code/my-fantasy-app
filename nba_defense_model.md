# NBA defense model: real rim/perimeter defensive difficulty

`modules/nba_defense_model.py`. See [matchup_engine.md](matchup_engine.md) for how this fits into the wider matchup-aware layer.

## Why this exists alongside `modules.matchup_model`

`modules.matchup_model.project_matchup_difficulty` already scores opponent difficulty, but from one real, zone-agnostic signal: overall `DEF_RATING`. This module adds a real, more granular signal on top — real defended field-goal percentage *by shot zone* — without touching that existing function or the season-long fantasy projections built on it.

## Real data source

`nba_api.stats.endpoints.leaguedashptdefend` — the NBA's own player-tracking defense feed. Verified live during this cycle's build. Each zone category returns **different real column names** (discovered live, not assumed from docs):

| Zone | `defense_category` value | Volume column | Defended-FG% column |
|---|---|---|---|
| Rim | `"Less Than 6Ft"` | `FGA_LT_06` | `LT_06_PCT` |
| Perimeter | `"3 Pointers"` | `FG3A` | `FG3_PCT` |

There is no working team-level equivalent: `nba_api`'s `LeagueDashPtTeamDefend` returned a malformed response (`KeyError('resultSet')`) as of this build. `team_zone_defense(season)` aggregates the real *player-level* rows up to each player's own team instead, weighted by each defender's real shot volume — a bench player who rarely contests a shot moves the team number less than a real starter who contests hundreds.

## `matchup_difficulty_score(player_role, opponent_team, season)`

`player_role` matches `modules.matchup_model`'s existing taxonomy (`"interior"`, `"primary ball handler"`, `"wing"`), so a caller already computing a role there can reuse it here. A lower defended FG% is a *tougher* defense — the score is built from `1 - defended_fg_pct`, percentile-ranked across the real league, then blended by the role's real zone weights (`_ROLE_ZONE_WEIGHTS`: interior is 80% rim-weighted, primary ball handler is 75% perimeter-weighted), then adjusted by real team pace (`fetch_league_team_stats`'s already-used `PACE` column) — a faster real team means more real possessions to defend against.

Fails soft: with no real player-tracking data for either zone, `difficulty_score` returns a neutral `50.0` with `"basis": "no_pt_defend_data"`, never a fabricated number.
