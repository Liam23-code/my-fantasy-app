# CBB data pipeline

End-to-end map of every real data source the College Basketball side of this app touches, and which module owns each. See [offline_data_contract.md](offline_data_contract.md) for the rules this pipeline follows, [college_sports_betting.md](college_sports_betting.md) for why CBB needed a different data-source answer than CFB, and [betting_engine.md](betting_engine.md) for how the evaluation layer built on top of it works.

## Real data source: ESPN's public site API (`site.api.espn.com` / `site.web.api.espn.com`)

No keyless, non-scraping CBB stats package exists: `cbbpy` scrapes ESPN's HTML directly via `beautifulsoup4`+`requests`, and `sportsdataverse` pulls in `pandas-3.0.5` (breaking this project's `pandas>=2.2,<3.0` pin) plus matplotlib/xgboost/xarray/pyjanitor for a single stats feed. Instead, this pipeline calls ESPN's own public *JSON* API directly — the same legal category as `nba_api`'s live scoreboard endpoint already used elsewhere in this repo: real, structured data from the league's own public feed, not a sportsbook, and not HTML scraping. This required an explicit, narrow, documented exception to this project's usual "no `requests`" rule for market-pipeline files — see `offline_data_contract.md` and the allowlist in `tests/test_data_contract_hardening.py::_FILES_ALLOWED_TO_IMPORT_REQUESTS`.

Verified live during this cycle's build: real team lists (362 real Division I teams), real per-team schedules and scores, real team statistics, and real per-player season averages.

| Source | Endpoint | Used by | What it provides |
|---|---|---|---|
| Team list | `GET {ESPN_CBB_BASE}/teams` | `modules/cbb_team_model.py::list_teams` | Every real Division I team ESPN carries (id, abbreviation, display name). |
| Team statistics | `GET {ESPN_CBB_BASE}/teams/{id}/statistics` | `modules/cbb_team_model.py::team_pace_estimate` | Real box-score inputs (FGA, OREB, TOV, FTA) for a possession/pace estimate. |
| Team schedule | `GET {ESPN_CBB_BASE}/teams/{id}/schedule` | `modules/cbb_team_model.py::team_scoring_by_game` | Real completed-game scores for one team this season. |
| Roster | `GET {ESPN_CBB_BASE}/teams/{id}/roster` | `modules/cbb_props_generator.py` | Real roster (athlete ids) for one team. |
| Player season stats | `GET site.web.api.espn.com/apis/common/v3/.../athletes/{id}/stats` | `modules/cbb_props_generator.py` | Real per-player "Season Averages" (points, rebounds, assists, minutes, games played) — note this lives on a **different host** than the other endpoints above; a 404 on `site.api.espn.com` was how this was discovered. |
| Scoreboard | `GET {ESPN_CBB_BASE}/scoreboard` | `modules/cbb_schedule.py::fetch_todays_games` | Real home/away matchups for a given date (default: today). |

`ESPN_CBB_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"`, defined once in `modules/cbb_team_model.py` and imported by every other CBB module that needs it.

## Pipeline, by stage

### 1. Season-long, pre-generated (real content already seeded this cycle)

```
ESPN (roster + per-player season-averages, real season)
    -> modules/cbb_props_generator.py: generate_default_props()
        real per-game rate = ESPN's own "Season Averages" computation
        (no season-total/games division needed -- ESPN already reports
        the per-game rate), filtered to rotation players (minimum real
        games played and minutes/game)
    -> data/cbb_props.json  ("basis" on every row)
```
Regenerate with `python -m modules.cbb_props_generator`. Not on the request path. **Already run once this cycle** (`write_default_props_file(2026, team_pool_size=25)`, 6.8s, 484 real rows across 158 unique players) — `data/cbb_props.json` ships with real data, unlike CFB's empty default.

ESPN needs one request per team roster and one per player's stats — there is no single bulk "every player's real averages" call the way `nba_api`'s `LeagueDashPlayerStats` gives NBA — so the generator covers a bounded, real team pool (`team_pool_size`, default 20–25; ESPN's team-list order, not a quality ranking) fetched concurrently via `betting.parallel_utils.parallel_map` to keep generation time reasonable. A full-league run is possible by raising `team_pool_size`; empirically, wall time did not scale linearly with pool size (25 teams finished faster than 3 teams extrapolated to suggest, likely connection reuse).

### 2. Per-request, real-time (changes daily, like NBA)

```
ESPN (real scoreboard, live)
    -> modules/cbb_schedule.py: fetch_todays_games()
        real home/away matchups; TTL-cached 60s; fails closed to [] on
        any error (network down, off-season, no games scheduled)
```
Mirrors `modules/nba_schedule.py`'s shape and fail-closed convention exactly. Live-tested during this cycle: 52 real games returned for the next scheduled slate (college basketball is out of season as of this build).

### 3. Per-request, real, aggregated on demand

```
ESPN (per-team schedule, real completed games, bounded pool)
    -> modules/cbb_team_model.py: build_team_scoring_averages(), real_margin_and_total_volatility()
        real average points scored/allowed per team; real observed
        standard deviation of margins and totals from actual games in
        the fetched pool
    -> modules/cbb_moneyline_model.py: fair_moneyline(), evaluate_games()
        combined with modules/cbb_schedule.py's live matchups and (if
        loaded) modules/cbb_odds_loader.py's real game odds
```
`build_team_scoring_averages` returns `(averages, games_by_team)` — the raw per-game data is returned alongside the averages specifically so `real_margin_and_total_volatility` and the props generator can reuse it without a second real fetch. Live-tested at multiple pool sizes: 8 teams in 0.8s, 60 teams in 2.6s, producing real averages like Alabama 91.72 ppg / Air Force 61.53 ppg.

`team_pace_estimate` uses the standard basketball-analytics possession formula `Poss ≈ FGA - OREB + TOV + 0.44*FTA` (the same formula `nba_api`'s own Advanced measure type uses internally) against real box-score inputs — verified live against real Duke data (FGA=60, OREB=10, TOV=12, FTA=20 → 70.8 possessions, the expected magnitude).

`real_margin_and_total_volatility` computes CBB's own real standard deviation rather than reusing NBA's or CFB's constants — college basketball's real scoring/possession patterns differ from both. With fewer than 2 real games sampled, it falls back to a documented, disclosed estimate (`margin_stdev=12.0`, `total_stdev=11.5`, `"basis": "fallback_estimate"`).

## Prop and parlay modules

- **`modules/cbb_prop_model.py`** — a from-scratch Gaussian model, structurally identical to CFB's (no pre-existing rich per-player projection system to wrap). The one real signal CFB's equivalent lacks: ESPN's real per-player `minutes` average, which `minutes_volatility_multiplier` uses to widen the assumed coefficient of variation for a low-minutes player (`_BASE_CV = 0.35`, scaled by up to 2x below `_FULL_ROTATION_MINUTES = 24.0`) — a real, disclosed proxy rather than one flat constant for every player regardless of role.
- **`modules/cbb_parlay_engine.py`** — a thin re-export of `modules.nba_parlay_engine` (`make_leg`, `evaluate_parlay`, `nba_detect_correlations` as `detect_correlations`, …). Basketball correlation patterns (overlapping stat categories like `PRA`, teammate scoring stacks) are structurally the same real phenomenon in college basketball as in the NBA, so nothing new was written.
- **`modules/cbb_odds_loader.py`** / **`modules/cbb_injuries_loader.py`** — default file + user upload, offline, merged by key. `data/cbb_game_odds.json` and `data/cbb_injuries.json` ship empty by design (no fixed daily schedule to pre-populate against, matching NBA's equivalent files).

## Defensive difficulty (matchup-aware layer)

`modules/cbb_defense_model.py`, added alongside the NBA-first matchup-aware engine (see [matchup_engine.md](matchup_engine.md)). Same shape as CFB's equivalent: a real percentile-ranked difficulty score over real `points_allowed_avg`, rather than a second ESPN fetch. Takes an already-fetched `averages` dict as an argument (from `build_team_scoring_averages`) instead of fetching again, since the per-team schedule calls are the expensive part and a caller almost always already paid it — the same reuse rationale `real_margin_and_total_volatility` already uses.

## What's *not* built (by design, not oversight)

- **No live in-game tracking.** Same scope boundary as every other sport in this repo.
- **No bulk "every team" or "every player" real-time call.** ESPN's public API doesn't offer one; every aggregate here is built from a bounded, concurrently-fetched real pool, not the full ~360-team field, to keep request volume and wall time reasonable.
