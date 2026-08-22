# NBA data pipeline

End-to-end map of every real data source the NBA side of this app touches, and which module owns each. See [offline_data_contract.md](offline_data_contract.md) for the rules this pipeline follows and [betting_engine.md](betting_engine.md) for how the evaluation layer built on top of it works.

## Real data sources (all `nba_api`, already a first-class dependency)

| Source | Endpoint | Used by | What it provides |
|---|---|---|---|
| Static roster | `nba_api.stats.static.players` / `.teams` | `modules/projections.py`, `modules/nba_team_model.py`, `modules/nba_schedule.py`, others | Player/team name resolution and ID lookup — no network call, bundled data. |
| Season player stats | `nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats` | `modules/nba_props_generator.py`, `modules/nba_trend_signals.py` | Real season-total and season/last-N-games per-player stats (points, rebounds, assists, 3PM, usage rate, minutes, games played). |
| Player game log | `nba_api.stats.endpoints.playergamelog` | `modules/projections.py` (via `_game_log_with_fuzzy_fallback`) | Real per-game history for one player, used by `fuse_projection`/`compare_props`'s richer projection. |
| Team stats | `nba_api.stats.endpoints.leaguedashteamstats` (via `modules/nba_advanced.py::fetch_league_team_stats`) | `modules/pace_model.py`, `modules/nba_trend_signals.py` | Real season and last-N-games team pace, ratings. |
| League game log | `nba_api.stats.endpoints.leaguegamelog` | `modules/nba_team_model.py`, `modules/nba_advanced.py` | Real per-team-per-game results (every completed game this season, two rows per game — one per team). |
| Live scoreboard | `nba_api.live.nba.endpoints.scoreboard.ScoreBoard` | `modules/nba_schedule.py` | Today's real matchups (home/away teams, game time). Only team/timing fields are read — see offline_data_contract.md on the `pbOdds` field this feed also carries. |
| Date-specific scoreboard | `nba_api.stats.endpoints.scoreboardv2.ScoreboardV2` | `modules/nba_schedule.py` | A specific non-today date's real matchups (numeric team IDs, resolved via the static roster). |

## Pipeline, by stage

### 1. Season-long, pre-generated (stable all season)

```
nba_api (LeagueDashPlayerStats, real season)
    -> modules/nba_props_generator.py: generate_default_props()
        real per-game rate = season_total / games_played, rounded to x.5,
        filtered to rotation players and markets clearing a minimum rate
    -> data/nba_props.json  (committed, "basis" on every row)
```
Regenerate with `python -m modules.nba_props_generator` after a season completes. Not on the request path.

### 2. Per-request, real-time (changes daily or faster)

```
nba_api (live ScoreBoard, or stats ScoreboardV2 for a specific date)
    -> modules/nba_schedule.py: fetch_todays_games()
        real home/away matchups; fails closed to [] on any error
```
Called directly by the Streamlit page and by `modules/daily_slate.py` / `modules/recommendations.py` on every request — no caching layer beyond Streamlit's own `@st.cache_data(ttl=...)` at the page level.

### 3. Per-request, real, aggregated on demand

```
nba_api (LeagueGameLog, every completed game this season)
    -> modules/nba_team_model.py: team_scoring_averages(), real_margin_and_total_volatility()
        real average points scored/allowed per team; real observed
        standard deviation of margins and totals from actual games
    -> modules/nba_moneyline_model.py: fair_moneyline(), evaluate_games()
        combined with modules/nba_schedule.py's live matchups and (if
        loaded) modules/nba_odds_loader.py's real game odds
```

```
nba_api (LeagueDashPlayerStats, season and last-10-games)
    -> modules/nba_trend_signals.py: team_pace_trend(), player_usage_trend()
        real recent-vs-season pace/usage deltas -- shown as extra context
        on the Betting Engine page, not fed into the pricing math
```

### 4. Existing, pre-built projection layer (reused, not touched)

```
nba_api (game logs, season stats, team stats)
    -> modules/projections.py, modules/minutes_model.py, modules/pace_model.py,
       modules/matchup_model.py, modules/fusion_model.py, modules/reliability.py
        real per-player projection with minutes/pace/matchup/injury context
        and a reliability score -- built before this betting work started,
        for the season-long fantasy/analysis pages
    -> modules/props.py: compare_props()
        compares that real projection to a loaded prop line, in raw stat
        units (no price concept originally)
    -> modules/nba_prop_model.py: price_prop_comparison(), price_aware_evaluations()
        adds probability/edge/EV on top, using the real over/under price
        from modules/nba_props_loader.py and betting.odds_math -- the one
        new piece this betting work added to an otherwise pre-existing,
        untouched pipeline
```

## Injury data (sibling pipeline, not scoring-specific)

`modules/injury_parser.py` (`load_injury_data_from_file`, default `data/injuries.json`, or `load_injury_data_from_user_upload`) feeds `modules/minutes_model.py` and `modules/pace_model.py`'s existing availability/blowout-risk adjustments. Same file-plus-upload, never-a-live-fetch convention as everything else here; not modified by this betting work.

## What's *not* built (by design, not oversight)

- **No live in-game (play-by-play/possession/substitution) tracking.** Every signal above is pre-game: real historical rates and real recent-vs-season trends, not a mid-game feed. See [architecture.md](architecture.md) and this project's build-cycle history for why that scope was deliberately excluded.
- **No pre-generated `nba_game_odds.json` content.** Unlike NFL's full-week pre-generated schedule, NBA's daily matchups aren't fixed far enough in advance to pre-generate — the file ships empty; fair lines are computed live from `team_scoring_averages` regardless of whether any game odds are loaded to compare against.
