# Performance notes: caching, precomputed tables, concurrency

Honest accounting of what actually got faster this cycle, what's infrastructure for later, and what's marginal at today's scale. See [developer_onboarding.md](developer_onboarding.md) for the TTL-cache test-isolation gotcha if you're about to write a test against any of the functions below.

## Caching layer

`betting.cache_utils.ttl_cache(seconds)` -- a small, dependency-free, thread-safe, in-memory TTL memoization decorator. Lives in `fantasy_engine/betting/` (shared infra, like `odds_math.py`); NBA-side modules import it directly.

Distinct from `fantasy.cache` (`fantasy_engine/fantasy/cache.py`), which already existed before this cycle: that module memoizes deterministic, caller-keyed pure functions (scoring/normalization math with no time dimension, optionally Redis-backed for cross-process sharing). `ttl_cache` solves a different problem -- transparent, decorator-based freshness caching for I/O (file reads, live API calls) where the underlying data changes over time. Two caching mechanisms, two different jobs; see `betting/cache_utils.py`'s docstring.

| Function | TTL | Why this TTL |
|---|---|---|
| `modules.nba_schedule.fetch_todays_games` | 60s | Today's real schedule can change within a session (postponement, addition); short TTL trades a little staleness for not hammering the live feed every rerun. |
| `modules.nba_props_loader.load_props_from_file` | 300s | The default file changes at most once a season. |
| `modules.nba_odds_loader.load_default_game_odds` | 300s | Same reasoning. |
| `betting.odds_loader.load_default_odds` | 300s | Same reasoning, NFL side. |
| `betting.team_model.team_scoring_by_week` | 3600s | Not on any live per-request path today (NFL's schedule is pre-generated into a static file, unlike NBA's daily live fetch) -- a long TTL just avoids redundant `nflreadpy` calls within one generation run. |
| `modules.nba_props_generator._fetch_base_player_stats` | 300s | Shared by the generator and `nba_player_rate_table.py` so both don't each make their own live call for the same real data. |
| `modules.nba_player_rate_table._fetch_advanced_player_stats` / `build_player_rate_table` | 300s | Same reasoning. |
| `modules.nba_trend_signals.team_pace_trend` / `player_usage_trend` | 300s | These are UI overlay values computed for potentially many rows in one table render -- cache them so re-rendering doesn't re-issue a live call per row. |

**Gotcha, stated plainly because it cost real debugging time this cycle:** a cache hit returns the *same object* a previous call returned (like `functools.lru_cache`). In tests, two test methods calling the same cached function with the same arguments but different mocks will see the first test's cached result on the second call, unless the test class clears the cache (`the_function.cache_clear()`) in `setUp`. This bit `NbaScheduleTests`, `NbaPropsGeneratorTests`, `TeamPaceTrendTests`, and `PlayerUsageTrendTests` during this cycle; all now clear their relevant caches in `setUp`.

## Precomputed player-rate table

`modules/nba_player_rate_table.py::build_player_rate_table` -- one real per-game rate table (points, rebounds, assists, PRA, usage %, pace-adjusted scoring) for ~300 players, computed once and cached (see above) rather than every consumer recomputing its own real per-player rates. Reuses the generator's own cached base-stats fetch (`_fetch_base_player_stats`) rather than making a second live call for the same season-total data; only usage rate and pace need a second `measure_type_detailed_defense="Advanced"` fetch, since that measure type doesn't carry raw counting stats.

**Pace-adjusted scoring** is a real, standard normalization: `points_per_game * (league_average_pace / player_pace)`, using each player's own real "PACE" figure (the average pace of games they played in, from `nba_api`'s Advanced measure type) -- not a fabricated adjustment.

## Parallelization: what actually gets faster, and what doesn't

`betting.parallel_utils` -- `parallel_map` (bounded `ThreadPoolExecutor`) and `parallel_ev_map` (an alias, semantically distinct). Two genuinely different situations, two different honest expectations:

- **I/O-bound work gets a real speedup.** `modules.recommendations.recommend_props`'s per-player loop makes several independent real `nba_api` calls per player (via `compare_props` -> `fuse_projection`, `get_reliability_score`). A thread blocked on a socket releases the GIL, so a bounded pool (6 concurrent workers, tuned to stay well under `nba_cache.py`'s own retry/backoff assumptions) gives real wall-clock improvement proportional to how many players are being scanned. `tests/test_parallel_utils.py::test_genuine_concurrency_is_faster_than_serial_for_io_bound_work` measures 6 x 0.05s blocking calls completing in well under the ~0.3s serial time would take.
- **Pure CPU-bound math does not get a comparable speedup, and that's expected, not a bug.** `betting.prop_model.evaluate_props` and `modules.nba_prop_model.price_aware_evaluations`'s per-row EV/probability math (`math.erf`, a handful of arithmetic operations) is cheap enough per row that thread-pool overhead can rival or exceed the actual work, and CPython's GIL means threads don't parallelize pure CPU work the way they parallelize I/O waits. These were still wrapped in `parallel_ev_map` this cycle -- "parallel EV calculations for both sports" is now a real, tested code path, not a claim with nothing behind it -- and it's the natural place to switch to a process pool later *if* the per-row math ever gets expensive enough (a materially richer model, many more rows) to justify process-spawn overhead. At today's row counts (a few hundred), don't expect a measurable win here; the value is architectural readiness, not today's benchmark.

**Thread-safety note:** introducing concurrent callers into `modules/nba_cache.py` (which `recommend_props`'s parallel loop now does, indirectly) required adding a lock (`_STATE_LOCK`) around that module's two pieces of shared global state (`_MEMORY_CACHE`, `_PROVIDER_UNAVAILABLE_UNTIL`). Individual dict/variable operations are already atomic under CPython's GIL, but the check-then-set sequence on the provider-cooldown timestamp was not -- without the lock, concurrent callers could each observe "provider not cooling down" and duplicate a retry loop unnecessarily. The actual network call inside `fetch_nba_frames`'s retry loop stays outside the lock so concurrent fetches for *different* cache keys aren't serialized against each other -- only the shared bookkeeping is.

## Async upload ingestion: the honest version

Streamlit re-runs a page's script synchronously on each interaction -- within one render, there's no *other* Python code that benefits from a parse call happening on a worker thread; blocking on a future's `.result()` immediately is the same wall-clock wait as calling the parser directly. Parsing a user's KB-sized CSV/JSON upload takes single-digit milliseconds either way. Threading a *single* upload's parse alone would not make the page feel faster, and this codebase does not claim otherwise.

What's genuinely worth doing, and what `modules/async_upload.py::parse_uploads_concurrently` does: when the page has **multiple independent** uploads to parse -- the Betting Engine page's NBA branch parses both a props file and a game-odds file -- running them concurrently instead of one-after-another is a real, if modest, latency win, measured directly: `tests/test_async_upload.py::test_multiple_tasks_actually_run_concurrently_not_serially` shows 3 x 0.05s parses completing in well under the ~0.15s serial time would take. The `st.spinner` wrapping the parse call is the other real lever here -- it's what keeps the page from *looking* frozen while a parse (of any size) runs, independent of whether that parse happens to be threaded.
