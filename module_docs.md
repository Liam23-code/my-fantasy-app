# Module reference: offline props/odds + schedule pattern

Companion to [architecture.md](architecture.md), which explains how these modules fit together. This file is the per-module reference: what each one exports, its contract, and what it deliberately does not do.

---

## NBA (`UniversalQuantAgent/modules/`)

### `nba_props_loader.py`

Default-file-plus-upload loader for player prop lines. Never a sportsbook fetch.

- `load_props_from_file(path: str | Path | None = None) -> list[dict]`
  Reads `data/nba_props.json` (or `path`). Missing/unreadable file → `[]`, never raises.
- `load_props_from_user_upload(source: Any, *, file_format: str | None = None) -> list[dict]`
  `source` may be a path, a Streamlit `UploadedFile` (anything with `.read()`), or raw `str`/`bytes`. `file_format` forces `"json"`/`"csv"`/`"text"`; otherwise inferred from a file extension or content-sniffed.
- `unified_props(*, default_path=None, uploaded=None, uploaded_format=None) -> list[dict]`
  Default rows, with any uploaded row overriding the default row sharing its `(player_name, category)` key. This is the one function the three NBA consumer modules actually call.

Row shape: `{"player_name": str, "team": str, "category": "points"|"rebounds"|"assists"|"PRA"|"3PM", "line": float, "sportsbook": str, "timestamp": str, "basis": str (optional)}`.

Column aliasing accepts common variants on ingest (`market`/`stat`/`prop`/`prop_type` all map to `category`; `player`/`name`/`athlete` all map to `player_name`; etc.) — see `_CATEGORY_ALIASES` etc. at the top of the file. A JSON upload that isn't a flat list of rows falls back to `modules.sportsbook_parser.parse_market_json`, the recursive market-payload walker, so an arbitrarily-nested provider export still parses.

### `nba_props_generator.py`

One-off generation script — **not called by any Streamlit page, not on the request path.** Run it directly to refresh `data/nba_props.json` after a season completes:

```bash
python -m modules.nba_props_generator
```

- `generate_default_props(season: str | None = None, *, pool_size: int = 175) -> list[dict]`
  Makes one live call to `nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats(season)` (real season-total stats), keeps the highest-`GP` row per player (the "TOT" combined row for a traded player, selected automatically rather than double-counting each team stint), filters to rotation players (`_MIN_GAMES_PLAYED`, `_MIN_MINUTES_PER_GAME`), and for each of points/rebounds/assists/PRA/3PM computes `season_total / games_played`, rounds to the nearest sportsbook-style `x.5` (`_round_to_half`), and skips a category if the real rate is below `_MIN_RATE_BY_CATEGORY[category]` (a real but negligible rate, e.g. a center's 0.1 threes/game, isn't a real market).
- `write_default_props_file(season=None, *, pool_size=175, path=None) -> int`
  Calls the above and writes `data/nba_props.json`; returns the row count.

Every row carries `"basis": "<season> real per-game rate (<games> games)"`. A traded player's team is left `""` (not `"TOT"`) — consumers already have a live-lookup fallback for an unresolved team (`modules.props._player_team_name`).

### `nba_schedule.py`

Live "who's playing whom today" — legal live game-data ingestion, not odds.

- `fetch_todays_games(game_date: date | None = None) -> list[dict]`
  `game_date=None` (or today): `nba_api.live.nba.endpoints.scoreboard.ScoreBoard`. A specific other date: `nba_api.stats.endpoints.scoreboardv2.ScoreboardV2` (numeric team IDs resolved to abbreviations via `nba_api.stats.static.teams`). **Any exception (network, off-season, provider error) is caught and returns `[]`** — an empty slate is a normal state, not an error.

Row shape: `{"home_team": str, "away_team": str, "start_time": str}`. The live feed's `pbOdds` field (a partner-sportsbook price) is deliberately never read.

### `sportsbook_parser.py` *(pre-existing, reused)*

The "safe half" of the old scraper — pure, offline transforms only, no I/O:

- `normalize_player_name`, `normalize_team_name`, `normalize_category` — free-text → this app's vocabulary.
- `walk_market_records` / `parse_market_json` / `parse_market_json_text` — recursively recognize nested prop-market records in an already-loaded JSON payload. Used by `nba_props_loader`'s upload fallback path.

### `sportsbook_scraper_disabled.py` / `injury_scraper_disabled.py` *(pre-existing, permanently disabled)*

Every function is `raise RuntimeError(...)`. Kept only so the hard failure is loud and explicit rather than a silent empty return, and so the module names still exist for anything referencing them historically. **Never add `requests`/`httpx`/`aiohttp`/`urllib3`/`selenium`/`playwright`/`bs4` back into either file.**

### `injury_parser.py` *(pre-existing, sibling pattern)*

Same default-file-plus-upload shape as `nba_props_loader.py`, one file earlier: `load_injury_data_from_file()` (default `data/injuries.json`), `load_injury_data_from_user_upload()`, plus injury-severity impact modeling (`severity_impact_score`, `snap_share_impact`, `volatility_impact`) consumed by the projection/context engines.

### Consumers (unchanged logic, new data source)

- `daily_slate.py` — `get_daily_slate()` calls `nba_schedule.fetch_todays_games()` + `nba_props_loader.unified_props()`.
- `props.py` — `compare_props()` calls `nba_props_loader.unified_props()`.
- `recommendations.py` — `recommend_props()` calls `nba_props_loader.unified_props()` + `nba_schedule.fetch_todays_games()`.

---

## NFL (`fantasy_engine/betting/`)

### `odds_loader.py`

NBA's `nba_props_loader.py` is a direct port of this module's shape. `load_default_odds()` / `load_uploaded_odds()` / `merge_odds()` / `unified_odds()`, keyed on `game_id` (games) or `player_id:market` (props) instead of `(player_name, category)`. Reads `fantasy_engine/data/odds.json`.

### `odds_generator.py`

NBA's `nba_props_generator.py` counterpart. `generate_default_props()` uses `fantasy.projections.load_forward_projections()` (real per-game rates by position); `generate_default_games()` additionally prices a real upcoming schedule (`nflreadpy.load_schedules`) with `betting.moneyline_model.fair_moneyline`, built on real team scoring. Both are generation-time only, not on the request path.

### `odds_math.py`

Pure American-odds math shared by every consumer: `american_to_decimal`, `implied_probability`, `remove_vig_two_way`, `expected_value`, `edge_vs_fair`, `fair_price_from_probability`. No sport-specific logic.

### `prop_model.py` / `team_model.py` / `moneyline_model.py` / `parlay_engine.py`

The evaluation layer — turn a real player/team projection plus a loaded line into a probability, edge, EV, confidence, and risk tier. NBA does not yet have an equivalent evaluation layer; the NBA consumer modules (`props.py`, `recommendations.py`, `daily_slate.py`) have their own pre-existing fusion/reliability-based comparison logic and only needed their *data source* replaced, not their evaluation logic.

---

## Cross-cutting

### `tests/test_nfl_nba_offline_market_parity.py` *(repo root)*

Asserts the NFL and NBA pipelines above satisfy the same contract as a single executable spec, independent of either project's own unit tests: no banned imports, missing-default-file → empty not error, upload-overrides-by-key, every shipped default row carries a `"basis"` string, live schedule ingestion fails closed. Imports both `betting.*` (editable-installed from `fantasy_engine/`) and `modules.*` (via an explicit `sys.path` insert of `UniversalQuantAgent/`).
