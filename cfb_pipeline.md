# CFB data pipeline

End-to-end map of every real data source the College Football side of this app touches, and which module owns each. See [offline_data_contract.md](offline_data_contract.md) for the rules this pipeline follows, [college_sports_betting.md](college_sports_betting.md) for why CFB and CBB needed two different data-source answers, and [betting_engine.md](betting_engine.md) for how the evaluation layer built on top of it works.

## Real data source: College Football Data API (`api.collegefootballdata.com`)

CFBD is a free, public, real-stats API for college football — real schedules, real box scores, real per-player season stats. It is not a sportsbook and none of the modules below read a price field. The one real blocker: **every endpoint requires a registered API key** (`Authorization: Bearer <key>`), confirmed live this cycle (an unauthenticated request returns `401`). There is no way for this codebase to obtain one on its own, and the official `cfbd` PyPI client pins `pydantic<2`, which conflicts with this project's pydantic 2.x (required by FastAPI) — so every CFB module here calls the REST API directly with `requests`, never the official client.

**Decision this cycle ("build contract now, key later"):** every CFB loader, generator, team model, prop model, and moneyline model was built to the full real-data contract today, gated entirely on one environment variable — `CFBD_API_KEY`. With no key set, every function returns an empty/neutral result (never raises, never fabricates a number); the moment a real key is set, every one of these functions starts returning real data with no further code changes. This is the same "missing source is a normal state" rule (`offline_data_contract.md` rule 3) applied to an entire sport, not just one file.

| Source | Endpoint | Used by | What it provides |
|---|---|---|---|
| Games | `GET /games` | `modules/cfb_team_model.py::team_scoring_by_game`, `fetch_week_games` | Real per-game home/away teams and final scores, for one season or one specific week. |
| Player season stats | `GET /stats/player/season` | `modules/cfb_props_generator.py` | Real season-total per-(player, category, statType) rows — passing/rushing/receiving yards and touchdowns. |
| Advanced season stats | `GET /stats/season/advanced` | `modules/cfb_team_model.py::team_pace` | Real real offensive play counts, for a plays-per-game pace estimate. |

**Not yet verified against a live response** — every parser above is written defensively (`_first(row, "homeTeam", "home_team")`-style alias lookup, tolerant of either the documented camelCase or a snake_case variant) and covered by tests using hand-built mocked responses shaped to CFBD's published docs, but no real key has been available to confirm the exact live shape. Re-verify field names against a real response the first time a key is configured — see the "Enforcement" section below.

## Pipeline, by stage

### 1. Season-long, pre-generated (once a key exists)

```
CFBD (/stats/player/season, real season)
    -> modules/cfb_props_generator.py: generate_default_props()
        real per-game rate = season_total / team_games_played, rounded to x.5,
        filtered to a minimum games-played and minimum rate per category
    -> data/cfb_props.json  ("basis" on every row, or an empty file with a
       note explaining CFBD_API_KEY is unset)
```
Regenerate with `CFBD_API_KEY=... python -m modules.cfb_props_generator` once a key is available. Not on the request path.

Games-played uses the player's **team's** real games played this season (`team_scoring_by_game`) as a proxy for the player's own games played — CFBD's player-season endpoint doesn't directly carry individual games-played, and a player with a meaningful season stat line has almost always played close to the full team schedule. A disclosed simplification, not a fabricated number (see the row's `"basis"` string).

### 2. Per-request, real, aggregated on demand (once a key exists)

```
CFBD (/games, one real week or one full season)
    -> modules/cfb_team_model.py: team_scoring_averages(), real_margin_and_total_volatility()
        real average points scored/allowed per team; real observed standard
        deviation of margins and totals from this season's actual games
    -> modules/cfb_moneyline_model.py: fair_moneyline(), evaluate_games()
        combined with modules/cfb_team_model.py's fetch_week_games() (a
        specific real week's matchups) and (if loaded)
        modules/cfb_odds_loader.py's real game odds
```

`real_margin_and_total_volatility` computes CFB's own standard deviation directly from real completed games rather than reusing NFL's or NBA's published constants — real week-to-week variance in college football runs meaningfully higher than the NFL's (much wider real talent gaps between programs mean both blowouts and upsets are more common). With fewer than 2 real games sampled, it falls back to a documented, disclosed estimate (`margin_stdev=21.0`, `total_stdev=17.0`, `"basis": "fallback_estimate"`) — never silently presented as real.

### 3. Why CFB is weekly, not daily

Unlike NBA/CBB's daily-changing slate, CFB — like the NFL — plays a fixed weekly schedule known well in advance. `modules/cfb_team_model.py::fetch_week_games(season, week)` takes an explicit week number (surfaced as a `st.number_input` on the Betting Engine page) rather than "today's games," mirroring the NFL side's real-week-at-a-time model instead of NBA/CBB's live-scoreboard-per-day model.

## Prop, parlay, and injury modules (no CFBD dependency)

- **`modules/cfb_prop_model.py`** — a from-scratch Gaussian model over each prop's real per-game rate (CFB has no pre-existing rich per-player projection system to wrap, unlike NBA's `fuse_projection`). Uses a single disclosed coefficient of variation (`_CFB_CV = 0.55`, wider than the NFL's 0.28–0.65 range) rather than a per-player volatility source, since no such source has been verified live yet. Reuses `betting.odds_math` and `betting.prop_model._risk_tier` directly — both fully sport-agnostic.
- **`modules/cfb_parlay_engine.py`** — a thin re-export of `betting.parlay_engine` (`make_leg`, `evaluate_parlay`, `detect_correlations`, …). Football correlation patterns (QB↔pass-catcher stack, RB volume↔total) are structurally the same real phenomenon in college football as in the NFL, so nothing new was written.
- **`modules/cfb_odds_loader.py`** / **`modules/cfb_injuries_loader.py`** — default file + user upload, offline, merged by key — same shape as every other sport's loader (see [offline_data_contract.md](offline_data_contract.md)). `data/cfb_game_odds.json` and `data/cfb_injuries.json` ship empty by design (no fixed schedule to pre-populate against, matching NBA's `nba_game_odds.json`).

## What's *not* built (by design, not oversight)

- **No live in-game tracking.** Same scope boundary as every other sport in this repo.
- **No default `cfb_props.json` content yet.** Ships empty with a note, not a placeholder number — real content appears automatically the first time `CFBD_API_KEY` is set and the generator is re-run.

## Enforcement / next steps once a key exists

1. Set `CFBD_API_KEY` and run `python -m modules.cfb_props_generator` — confirm real rows are written and every row's `"basis"` string looks right.
2. Spot-check `modules/cfb_team_model.py`'s field-alias assumptions (`_first(row, "homeTeam", "home_team")` etc.) against the real response shape; CFBD's docs were followed but never confirmed live.
3. Re-run `tests/test_cfb_engine.py` — the mocked-response tests should still pass unmodified (they test the parsing/computation logic, not connectivity), and are the safety net for step 2.
