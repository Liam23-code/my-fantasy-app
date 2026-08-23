# Developer onboarding

Practical get-started guide. For *why* things are built the way they are, see [architecture.md](architecture.md); for the legal/data-sourcing rules, [offline_data_contract.md](offline_data_contract.md); for the betting engine specifically, [betting_engine.md](betting_engine.md) and [betting_engine_advanced.md](betting_engine_advanced.md).

## Repo layout

```
UniversalQuantAgent/   Streamlit app: NBA + NFL analysis, fantasy tools, the Betting Engine page
  app/pages/30_Betting_Engine.py   the unified NFL+NBA betting UI
  modules/nba_*.py                 NBA-specific betting/data modules
  modules/parallel_utils.py        thin re-export of betting.parallel_utils
  data/                            real default data files (odds/props/injuries) -- gitignored: only data/nba_cache/*
  tests/
fantasy_engine/        standalone package: fantasy draft/season tools + the NFL betting engine
  betting/              odds_loader, odds_math, cache_utils, parallel_utils, prop_model, team_model,
                         moneyline_model, parlay_engine, odds_generator -- installed editable, importable
                         from anywhere in the venv as `betting.*`
  fantasy/, quant/, projections/   fantasy scoring/draft/projection logic
  tests/
tests/                  repo-root cross-package tests: NFL/NBA parity + data-contract hardening
```

## Setup

```bash
cd UniversalQuantAgent
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cd ../fantasy_engine
../UniversalQuantAgent/.venv/Scripts/pip install -e .
```

`fantasy_engine` is installed *editable* into `UniversalQuantAgent`'s venv, which is why `UniversalQuantAgent` code can `from betting.odds_math import ...` without any path manipulation, and why the repo-root `tests/` directory can import both `modules.*` (via an explicit `sys.path` insert of `UniversalQuantAgent/`) and `betting.*`/`fantasy.*` (already globally importable) in the same process.

## Running things

```bash
# All tests (from repo root, using UniversalQuantAgent's venv python):
UniversalQuantAgent/.venv/Scripts/python.exe -m pytest tests/ UniversalQuantAgent/tests fantasy_engine/tests

# Just the cross-package parity/hardening tests:
UniversalQuantAgent/.venv/Scripts/python.exe -m pytest tests/

# The Streamlit app:
cd UniversalQuantAgent && streamlit run app/app.py

# Regenerate real default data after a season completes:
UniversalQuantAgent/.venv/Scripts/python.exe -m modules.nba_props_generator   # from UniversalQuantAgent/
```

## Where to find things

- **A specific module's contract** (function signatures, what it's for): [module_docs.md](module_docs.md).
- **The betting engine specifically**: [betting_engine.md](betting_engine.md) (what's shared vs. sport-specific and why), [betting_engine_advanced.md](betting_engine_advanced.md) (correlation patterns, risk tiers, cross-sport parlays).
- **The NBA data pipeline end-to-end**: [nba_pipeline.md](nba_pipeline.md).
- **UI structure**: [ui_design.md](ui_design.md).
- **Caching/concurrency**: [performance_notes.md](performance_notes.md).
- **What's legal to fetch and what isn't**: [offline_data_contract.md](offline_data_contract.md) -- read this before adding any new data source.

## Conventions worth knowing before you touch code

- **Shared, sport-agnostic code lives in `fantasy_engine/betting/`** (`odds_math.py`, `cache_utils.py`, `parallel_utils.py`) and is imported directly by NBA-side modules, never duplicated. If you're about to write NBA-side math with zero NBA-specific content, check whether it already exists there first.
- **Every real number discloses its `"basis"`.** If you add a new generator or loader, give every row it produces a `"basis"` string naming the real source and sample size.
- **A missing default file, or a malformed field in an upload, returns an empty result -- it never raises.** Existing loaders all follow this; new ones should too (see offline_data_contract.md's enforcement tests).
- **TTL-cached functions need `cache_clear()` in test `setUp`.** This bit multiple tests during this cycle: `modules/nba_schedule.py::fetch_todays_games`, `modules/nba_props_generator.py::_fetch_base_player_stats`, `modules/nba_trend_signals.py::team_pace_trend`/`player_usage_trend`, and the NFL-side equivalents are all decorated with `@ttl_cache(...)` (see `betting.cache_utils`). If two tests call the same cached function with the same arguments but different mocks, the second test will silently see the first test's cached result unless its test class clears the cache in `setUp`. Grep for `@ttl_cache` before writing a new test against a cached function.
- **Don't build a "unified NFL+NBA" module unless the underlying content is genuinely sport-agnostic.** This project deliberately chose "shared contract, separate code" over a single merged codebase after weighing that NFL and NBA's real data and real modeling differ (different scoring environments, different existing projection depth) -- see betting_engine.md's opening section for the reasoning, before you're tempted to "simplify" by merging two sport-specific modules.
