# Architecture: offline sports-data & betting engine

This repo has two sub-projects that share one data-sourcing policy:

- **`UniversalQuantAgent/`** — a Streamlit app covering NBA and NFL player analysis, projections, fantasy tools, and a betting-adjacent props engine.
- **`fantasy_engine/`** — a standalone Python package (`fantasy`, `betting`, `quant`, `projections`) providing NFL fantasy draft/season tools and an offline NFL betting engine, installed editable into the same virtualenv and imported by `UniversalQuantAgent` as `fantasy.*` / `betting.*` / `projections.*`.

## The policy

1. **Sportsbook odds and injury data may only come from a file this repo ships, or a file the user uploads.** Never a live sportsbook fetch, never a scrape of DraftKings/FanDuel/BetMGM/Caesars/ESPN BET or any similar site.
2. **Live *game* data ingestion is allowed and used** — real schedules, real per-player box-score stats, real team scoring. This is not "odds," and the providers involved (`nba_api`, `nflreadpy`) are public stats APIs, not sportsbooks.
3. **Every number this app presents as real must disclose why it's real.** Any line derived from real historical data carries a `"basis"` string (e.g. `"2025-26 real per-game rate (79 games)"`); nothing is fabricated and silently presented as if it were.
4. **A missing or unconfigured data source is a normal state, not an error.** Every loader returns an empty result rather than raising when its file doesn't exist yet.

## The pattern, twice

Both sports independently arrived at the same three-part shape. NBA is the newer implementation and intentionally mirrors NFL's.

```
                        ┌─────────────────────────┐
   live stats API   →   │  generator (offline,    │  →  data/<sport>_props.json / odds.json
   (nba_api /            │  one-off script)         │      (committed, real per-game rates,
    nflreadpy)           └─────────────────────────┘       "basis" on every row)
                                                                     │
                                                                     ▼
                                                        ┌─────────────────────────┐
   user's own file   ─────────────────────────────────→│  loader (default + upload │→  unified rows
   (CSV / JSON /                                        │  merge, offline, never    │   consumed by
    upload widget)                                      │  network)                 │   models/pages
                                                        └─────────────────────────┘
```

### 1. Generator — turns real historical data into default lines

Not on the request path. Run manually (or on a schedule) to refresh the committed default file after a season/week completes.

| | NFL | NBA |
|---|---|---|
| File | `fantasy_engine/betting/odds_generator.py` | `UniversalQuantAgent/modules/nba_props_generator.py` |
| Real data source | `fantasy.projections.load_forward_projections()` (season-total stats, real games played) | `nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats` (season-total stats, real games played) |
| Rate → line | `season_total / games_played`, rounded to the nearest `x.5` (`_round_to_half`) | same |
| Minimum-sample guards | `_MIN_GAMES_PLAYED`, per-market `_MIN_RATE_BY_MARKET` | `_MIN_GAMES_PLAYED`, `_MIN_MINUTES_PER_GAME`, per-category `_MIN_RATE_BY_CATEGORY` |
| Output | `fantasy_engine/data/odds.json` (`games` + `player_props`) | `UniversalQuantAgent/data/nba_props.json` (`props`) |
| Schedule/game rows | `generate_default_games()` — real upcoming schedule via `nflreadpy.load_schedules`, priced with `betting.moneyline_model.fair_moneyline` from real team scoring | not applicable — NBA's "today's games" is live-fetched at request time instead (see below), since a season-long NBA schedule file would go stale daily the way a pre-set NFL week doesn't |

### 2. Loader — default file + optional user upload, merged, offline

This is what the live app actually calls. Zero network access.

| | NFL | NBA |
|---|---|---|
| File | `fantasy_engine/betting/odds_loader.py` | `UniversalQuantAgent/modules/nba_props_loader.py` |
| Default file | `load_default_odds()` reads `fantasy_engine/data/odds.json`; missing file → `{"games": {}, "player_props": {}}` | `load_props_from_file()` reads `UniversalQuantAgent/data/nba_props.json`; missing file → `[]` |
| Upload | `load_uploaded_odds()` — path / file-like / `str` / `bytes` / already-parsed `dict`/`list`; JSON or CSV/delimited text; alias-based column normalization | `load_props_from_user_upload()` — same source types; JSON list/`{"props": [...]}` (falls back to `modules.sportsbook_parser`'s recursive market-payload walker for arbitrarily nested JSON) or CSV/delimited text |
| Merge | `merge_odds(default, uploaded)` — an uploaded entry overrides the default entry sharing its key (`game_id`, or `player_id:market`); everything else passes through unchanged | `unified_props(default_path=, uploaded=, uploaded_format=)` — same override-by-key semantics, keyed on `(player_name, category)` |
| One-call convenience | `unified_odds()` | `unified_props()` |

### 3. Live game-data ingestion — legal, used, fails closed

"Live game data ingestion is allowed" is implemented differently per sport because the two schedules behave differently: NFL's week-by-week schedule is published months in advance and barely changes, so it's pre-generated into the same static `odds.json` as the props. NBA's daily slate changes every single day (including "no games today"), so it's fetched live, per request, from a legal non-sportsbook source:

- **`UniversalQuantAgent/modules/nba_schedule.py`** — `fetch_todays_games(game_date=None)`. Default (today): `nba_api.live.nba.endpoints.scoreboard.ScoreBoard`, the NBA's own live scoreboard feed. A specific past date: `nba_api.stats.endpoints.scoreboardv2.ScoreboardV2`. Only team-tricode and timing fields are read; the live feed's embedded partner-sportsbook odds field (`pbOdds`) is never touched. **Any failure (network down, off-season, provider error) returns `[]`, never raises** — an empty slate is a normal state for a page to render, not a crash.

### What's permanently off

`UniversalQuantAgent/modules/sportsbook_scraper_disabled.py` and `modules/injury_scraper_disabled.py` — every function is a one-line `raise RuntimeError(...)`. These used to make live HTTP requests to DraftKings/FanDuel/BetMGM/Caesars/ESPN BET and to ESPN's injury/scoreboard pages. The safe half of that old code (name/team/category normalization, JSON-payload parsing) survived and was extracted into `modules/sportsbook_parser.py` and `modules/injury_parser.py`, which have no network dependency and are reused by the new loaders above.

## Consumers

The three NBA pages that used to call the disabled scraper directly now call the loader/schedule modules instead — no other change to their logic:

- `modules/daily_slate.py` (`get_daily_slate`) → `nba_props_loader.unified_props()` + `nba_schedule.fetch_todays_games()`
- `modules/props.py` (`compare_props`) → `nba_props_loader.unified_props()`
- `modules/recommendations.py` (`recommend_props`) → `nba_props_loader.unified_props()` + `nba_schedule.fetch_todays_games()`

`app/pages/30_NFL_Betting.py` (one of five betting pages now — see [ui_betting_tabs.md](ui_betting_tabs.md)) is the NFL-side equivalent UI: three tabs (Money Lines, Props, Parlays) over `betting.odds_loader.unified_odds()`, with a `st.file_uploader` for overriding the default file — the one UI surface in this repo where a user can currently supply their own odds file. Uploading props/injuries on the NBA pages or the injury pages is backend-ready (`nba_props_loader.load_props_from_user_upload`, `injury_parser.load_injury_data_from_user_upload`) but has no upload widget wired up yet, matching where `injuries.json` already stood before this change.

## Testing

- `UniversalQuantAgent/tests/` and `fantasy_engine/tests/` — unit tests per project.
- `tests/` (repo root) — `test_nfl_nba_offline_market_parity.py` asserts the two independently-built pipelines satisfy the *same* contract (no banned imports, missing-file → empty not error, upload overrides by key, every default row carries real-data `"basis"`, live schedule ingestion fails closed). Run with `pytest tests/` from the repo root.
