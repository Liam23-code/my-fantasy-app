# Offline data contract

The single set of rules every data source in this repo follows, for both NFL and NBA. See [architecture.md](architecture.md) for how the modules that implement this contract fit together, and [module_docs.md](module_docs.md) / [betting_engine.md](betting_engine.md) / [nba_pipeline.md](nba_pipeline.md) for per-module detail.

## The rules

1. **Sportsbook odds and injury data come from exactly two places: our own file, or a file the user uploads. Never a live sportsbook fetch.** No DraftKings, FanDuel, BetMGM, Caesars, ESPN BET, or any similar site, direct or scraped.
2. **Live *game* data ingestion is allowed and used.** Real schedules, real box-score stats, real team scoring, real completed-game results. This is not "odds" — the providers (`nba_api`, `nflreadpy`) are public sports-stats APIs, not sportsbooks, and none of the modules that call them read a price field even when one happens to be present in the response (see "What's read, what's ignored" below).
3. **A missing or unconfigured data source is a normal state, not an error.** Every loader returns an empty result (`[]`, `{}`, `{"games": {}, ...}`) rather than raising when its default file doesn't exist yet, and every live-data function returns an empty result rather than raising when the provider is unreachable.
4. **Every number presented as real discloses why it's real.** Anything derived from real historical data carries a `"basis"` string naming the season and sample size (e.g. `"2025-26 real per-game rate (79 games)"`). A number that can't yet be computed from real data (an early-season fallback constant) is also labeled as such, not silently blended in as if it were real.
5. **An approximation is disclosed as an approximation.** Where a model treats an existing field as if it had a specific statistical meaning it wasn't rigorously built to have (e.g. `modules/nba_prop_model.py` treating `confidence_low`/`confidence_high` as an approximate 90% interval), that assumption is stated in the docstring, not asserted as fact.

## What counts as a violation

- Any `import requests`, `httpx`, `aiohttp`, `urllib3`, `selenium`, `playwright`, or `bs4` outside `modules/sportsbook_scraper_disabled.py` / `modules/injury_scraper_disabled.py` (which exist specifically to *not* do this — every function in them is `raise RuntimeError(...)`).
- A generator or loader that fabricates a number instead of computing it from a real source, or presents a fabricated number without disclosing that it isn't real.
- A loader that raises on a missing/empty file instead of returning an empty result.
- Reading a price/odds field off a live game-data feed that happens to carry one (see below).

## What's read, what's ignored, from a live feed

Two live feeds used in this repo carry data beyond what this app reads from them:

- `nba_api.live.nba.endpoints.scoreboard.ScoreBoard` (used by `modules/nba_schedule.py`) includes a `pbOdds` field per game — a partner-sportsbook price. Only `homeTeam`/`awayTeam`/`gameTimeUTC` are read; `pbOdds` is never touched.
- `nba_api` also ships a dedicated `nba_api.live.nba.endpoints.odds` client. It is never imported anywhere in this repo — reusing it would reintroduce exactly the sportsbook-odds ingestion this project removed.

## Enforcement

- `UniversalQuantAgent/tests/test_new_pipeline.py::test_no_network_libraries_imported_by_sportsbook_or_injury_modules` and the equivalent AST-based check in `UniversalQuantAgent/tests/test_nba_offline_props_and_schedule.py` scan specific files for banned imports.
- `tests/test_nfl_nba_offline_market_parity.py` (repo root) asserts the *same* contract holds across every market-pipeline file in both projects at once — banned imports, missing-file → empty-not-error, upload-overrides-by-key, every default row carries `"basis"`, live ingestion fails closed. This is the test to extend if a new file joins the offline pipeline on either side.
- Before adding a new real-time or historical data source, run the same audit this project's build cycles have run manually: AST-scan for banned imports, check `requirements.txt`/`pyproject.toml` for dependencies nothing imports anymore, and verify the new source's provider is a stats API, not a sportsbook.
