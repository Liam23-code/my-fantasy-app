# College sports betting engine: CFB + CBB architecture overview

How College Football and College Basketball fit into the unified betting engine alongside NFL and NBA. See [cfb_pipeline.md](cfb_pipeline.md) / [cbb_pipeline.md](cbb_pipeline.md) for each sport's real-data source in full detail, [betting_engine.md](betting_engine.md) for the shared-vs-sport-specific evaluation layer, and [offline_data_contract.md](offline_data_contract.md) for the rules every sport in this repo follows.

## The architectural choice: shared contract, separate code

Same decision NFL and NBA already made (see [betting_engine.md](betting_engine.md)'s "Why NFL and NBA aren't one codebase"), applied again rather than revisited: CFB and CBB get their own loader/generator/team-model/prop-model/moneyline-model/parlay-engine modules, mirroring NFL/NBA's conventions file-for-file. **NFL and NBA's existing code was not touched or refactored to make room for this** — no sport-agnostic core was extracted, no existing function signature changed. What's genuinely shared (odds math, caching, parallelization, parlay combinatorics) was already sport-agnostic before this cycle and is imported directly, exactly as NBA already imported it from NFL's `betting` package; what's sport-specific (real scoring constants, real data sources, real category lists) is implemented once per sport, never merged.

The result: four sports, four independent implementations, one contract. Deleting CFB or CBB's modules entirely would not require touching a single NFL or NBA file.

## The real blocker this cycle: no clean keyless data source for either sport

Both college sports hit the same shape of problem NFL/NBA never had to solve — no equivalent of `nba_api` (a maintained, keyless, real-data Python package) exists for either sport — but the specific problem, and its resolution, differed:

| | CFB | CBB |
|---|---|---|
| The obvious package | `cfbd` (official CFBD client) | `cbbpy` |
| Why it doesn't work | Pins `pydantic<2`; conflicts with this project's pydantic 2.x (FastAPI dependency) | Scrapes ESPN's HTML directly via `beautifulsoup4`+`requests` — exactly what this project removed from the NBA/NFL side |
| The alternative considered | `requests` directly against CFBD's REST API | `sportsdataverse` |
| Why the alternative was also rejected (CBB only) | n/a | Requires `pandas-3.0.5`, breaking this project's `pandas>=2.2,<3.0` pin, plus matplotlib/xgboost/xarray/pyjanitor for one stats feed |
| The real blocker | Every CFBD endpoint requires a registered `Authorization: Bearer` key (confirmed live: unauthenticated request → `401`); this codebase cannot register one on its own | None, once ESPN's own public JSON API (not `cbbpy`'s scrape of it) was recognized as a legal, keyless source |
| Resolution, as chosen by the user | **"Build contract now, key later"** — full stack built today, gated on `CFBD_API_KEY`; real data activates the moment a key is set, no code changes needed | **"Allow ESPN's public JSON API"** — a narrow, documented exception to the "no `requests`" rule, scoped to five specific files |

Both resolutions follow the same underlying rule already established for NBA: a live *game-data* API is a legal, allowed source (`offline_data_contract.md` rule 2) as long as it's a real stats provider, not a sportsbook, and no price/odds field from that provider is ever read. CFBD and ESPN's public API both satisfy this; the only new work was recognizing that ESPN's public *JSON* endpoints are not the same thing as scraping ESPN's *HTML* (which remains banned, same as it always was for NBA/NFL).

## CFB: "build contract now, key later"

Every CFB module — loader, generator, team model, prop model, moneyline model — was built to the real-data contract today. With no `CFBD_API_KEY` set, every function fails closed to an empty/neutral result (never raises, never fabricates a line); `data/cfb_props.json` ships empty with a note explaining exactly why and how to activate real data. This is deliberately the *same* "missing source is normal" rule this codebase already applies to a single missing file (`offline_data_contract.md` rule 3), scaled up to cover an entire sport with zero special-casing. See [cfb_pipeline.md](cfb_pipeline.md) for the full endpoint map and the verification steps to run the first time a key becomes available.

## CBB: "allow ESPN's public JSON API"

A narrow, explicit exception was added to this project's usual "no `requests`" rule for market-pipeline files, scoped to exactly the files that need it:

```python
# tests/test_data_contract_hardening.py
_FILES_ALLOWED_TO_IMPORT_REQUESTS = frozenset({
    "cfb_team_model.py", "cfb_props_generator.py",
    "cbb_team_model.py", "cbb_props_generator.py", "cbb_schedule.py",
})
```

This is enforced, not just documented: `ComprehensiveBannedImportSweepTests.test_no_market_pipeline_file_imports_a_banned_library` (repo root `tests/test_data_contract_hardening.py`) walks every market-pipeline file's AST and fails if `requests` (or any other banned network/scraping library) appears anywhere outside this exact allowlist — including inside a nested function body, which an AST walk still catches. Two companion tests close the obvious loopholes: `test_every_allowlisted_file_actually_uses_requests` (an allowlist entry that doesn't need the exception should be removed) and `test_files_outside_the_allowlist_still_ban_requests` (the exception never silently widens).

Real data was already generated and verified this cycle: `data/cbb_props.json` carries 484 real rows across 158 unique players, and `modules/cbb_schedule.py` returned 52 real scheduled games live. CBB needed no "key later" gating — it's real today. See [cbb_pipeline.md](cbb_pipeline.md) for the full endpoint map.

## What's genuinely shared vs. sport-specific, for CFB/CBB specifically

Following the same rule NBA already established: anything with no sport in it is imported directly, never copy-pasted.

**Reused unmodified (thin re-export, no new logic):**
- `modules/cfb_parlay_engine.py` re-exports `betting.parlay_engine` in full — football correlation patterns (QB↔pass-catcher, RB volume↔total) are the same real phenomenon in college football as the NFL.
- `modules/cbb_parlay_engine.py` re-exports `modules.nba_parlay_engine` in full — basketball correlation patterns (overlapping `PRA`, teammate scoring stacks) are the same real phenomenon in college basketball as the NBA.
- `betting.odds_math`, `betting.prop_model._risk_tier`, `betting.team_model.project_game`, `betting.parallel_utils`, `betting.cache_utils` — imported directly by every CFB/CBB module that needs them, identically to how NFL/NBA already used them.
- `modules/sportsbook_parser.normalize_player_name` (via the new `modules/college_sports_common.py`) — name normalization has no sport in it.

**New, because the underlying facts genuinely differ:**
- `normalize_college_team_name` (`modules/college_sports_common.py`) — NBA's team-name normalizer is a fixed 30-team alias dict; college sports have hundreds of programs, so this is a real, different function (uppercase + strip punctuation), not a bigger dict.
- Every real scoring/volatility constant — CFB's `margin_stdev=21.0`/`total_stdev=17.0` fallback and CBB's `margin_stdev=12.0`/`total_stdev=11.5` fallback are real, sport-specific estimates; reusing NFL's or NBA's would be a modeling error, not a simplification.
- `cfb_prop_model._CFB_CV = 0.55` and `cbb_prop_model._BASE_CV = 0.35` — different disclosed coefficients of variation, reflecting each sport's real, different game-to-game variance.
- CFB's weekly `fetch_week_games(season, week)` vs. CBB's daily `fetch_todays_games()` — the two sports' real schedules behave differently (CFB is fixed and weekly, like the NFL; CBB changes daily, like the NBA), so each mirrors the sport it structurally resembles rather than sharing a schedule abstraction with each other.

## Unified integration: `modules/unified_betting_contract.py`

One dispatch layer over all four sports — `load_odds(sport)`, `compute_ev(sport, props, **context)`, `compute_confidence(sport, priced_rows)`, `build_parlays(sport, legs)` — added this cycle so the UI (and any future integration) has one call site and one output shape per function, regardless of sport. This is routing only, not new betting logic: calling `compute_ev("CFB", props)` does exactly what calling `modules.cfb_prop_model.evaluate_props(props)` directly already did.

The one place a uniform *output* doesn't mean a uniform *input*: NFL needs a real player pool (`players_by_id`), NBA needs already-computed matchup-adjusted rows (`comparison_rows`), while CFB needs no extra context at all (each row already carries its own real per-game rate) and CBB's `minutes_by_player` context is optional. `compute_ev`'s docstring states exactly what each sport needs, rather than hiding the real difference behind a fake one-size-fits-all signature.

CFB and CBB don't compute an explicit 0–1 confidence score the way NFL/NBA's richer models do — `compute_confidence` maps their real, already-computed `risk_tier` through a fixed, disclosed proxy table (`_RISK_TIER_CONFIDENCE_PROXY = {"low": 0.8, "medium": 0.5, "high": 0.25}`) instead of leaving the field blank or fabricating a number.

`modules/unified_parlay_engine.py` (built for NFL+NBA last cycle) was extended, not rebuilt, to route all four sports' legs through their own `make_leg`/correlation detector — `detect_cross_sport_correlations` partitions a mixed leg list by sport and only ever compares legs within the same sport, so a CFB leg and a CBB leg (or any other cross-sport pair) are never treated as correlated, matching the same real-world independence assumption already established for NFL/NBA.

## Unified UI: the same three tabs, once per sport

Each sport now has its own betting page (`32_CFB_Betting.py`, `33_CBB_Betting.py` — see [ui_betting_tabs.md](ui_betting_tabs.md) for why the original single sport-toggle page split into five). Because CFB and CBB's prop-model output schemas are structurally identical (both are the from-scratch Gaussian pattern, unlike NBA's richer wrapped model), `app/betting_shared.py` shares one rendering function (`render_college_props_tab`/`render_college_moneylines_tab`) between them rather than duplicating NBA's UI code a third and fourth time — the UI-level reuse mirrors the module-level reuse decision above. Sport-specific overlays surface each sport's own real signal: CFB shows real pace/plays-per-game where available; CBB shows real minutes-based volatility.

## Testing

- `UniversalQuantAgent/tests/test_cfb_engine.py` (29 tests) / `test_cbb_engine.py` (25 tests) — per-sport unit coverage, including mocked-live-response tests for the two sports with no `nba_api`-equivalent to test against directly.
- `UniversalQuantAgent/tests/test_unified_betting_contract.py` (13 tests) — asserts all four sports satisfy the same dispatch-layer contract.
- `UniversalQuantAgent/tests/test_unified_parlay_engine.py` — extended with four-sport cross-parlay and pairwise-non-correlation tests.
- `UniversalQuantAgent/tests/test_betting_engine_page.py` — extended with CFB/CBB render, parlay-tab, and cross-sport-option tests (via `AppTest`, no live browser needed).
- `tests/test_data_contract_hardening.py` (repo root) — extended with CFB/CBB coverage in every existing test class (fails-closed-on-malformed-fields, provenance disclosure, upload-overrides-by-key, the banned-import sweep and its new allowlist), so all four sports are asserted to follow the *same* offline data contract, not just four independently-trustworthy implementations.

All 1200 tests across `tests/`, `UniversalQuantAgent/tests/`, and `fantasy_engine/tests/` pass as of this cycle.
