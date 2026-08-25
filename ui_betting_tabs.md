# Betting UI: one page per sport, plus a shared cross-sport tools page

How the single `30_Betting_Engine.py` page (a sport toggle over five tabs) became seven separate pages across two build cycles. See [betting_engine.md](betting_engine.md) / [betting_engine_advanced.md](betting_engine_advanced.md) for the engine functions each page calls, [cross_sport_tools.md](cross_sport_tools.md) for the shared cross-sport page's own detail, and [ui_design.md](ui_design.md) for the general Streamlit patterns in play (widget-`session_state` seeding, cached loaders).

## The seven pages

| Page | Sport | Tabs |
|---|---|---|
| `app/pages/30_NFL_Betting.py` | NFL | Money Lines, Props, Parlays |
| `app/pages/31_NBA_Betting.py` | NBA | Money Lines, Props, Parlays |
| `app/pages/32_CFB_Betting.py` | CFB | Money Lines, Props, Parlays |
| `app/pages/33_CBB_Betting.py` | CBB | Money Lines, Props, Parlays |
| `app/pages/34_Cross_Sport_Tools.py` | all six | Player Comparison, Cross-Sport Parlay |
| `app/pages/35_MLB_Betting.py` | MLB | Money Lines, Props, Parlays |
| `app/pages/36_NHL_Betting.py` | NHL | Money Lines, Props, Parlays |

Each of the six sport pages is fixed to one sport (no toggle) and renders exactly the three tabs above, matching this cycle's spec (unchanged from the original four-sport spec). Player Comparison and Cross-Sport Parlay apply *across* sports by nature (comparing two props, or mixing legs from different sports in one parlay) — duplicating either across six pages would mean six copies of the same logic, so both stayed on one shared page as the roster grew from four sports to six, rather than being bolted onto each sport page as a fourth/fifth tab. MLB and NHL's page numbers continue the existing sequence (35/36) rather than reusing 34, which Cross-Sport Tools already occupies.

## `app/betting_shared.py`: one copy of every loader and renderer

The single old page had one copy of each sport's cached data loader (`_load_nfl_evaluations`, etc.) and each tab's rendering logic. Splitting into per-sport page files without an extraction step would have meant six copies of ~700 lines of near-identical code. Instead, `app/betting_shared.py` holds:

- Every loader (`load_nfl_evaluations`, `load_nba_evaluations`, `load_cfb_evaluations`, `load_cbb_evaluations`, `load_mlb_evaluations`, `load_nhl_evaluations`) — the first four `@st.cache_data`-wrapped with unchanged behavior from the original page; MLB/NHL's loaders need no such cache (no live fetch underneath to cache against — every call is a pure file/upload read).
- Every tab renderer (`render_nfl_props_tab`, `render_nba_props_tab`, `render_college_props_tab` for CFB/CBB's shared shape, `render_offline_props_tab`/`render_offline_moneylines_tab` for MLB/NHL's shared shape, and the equivalent Money Lines renderers) — pure functions that take already-loaded data and render, no page-specific state.
- `render_parlay_builder` — the shared quick-add/leg-picker/evaluate UX all six sports now use identically (its `else` branch, keyed on row shape rather than a hardcoded sport list, already handled CFB/CBB and needed no change at all when MLB/NHL joined).

Each of the six sport pages is genuinely thin: load this sport's data, call three render functions inside three `with tab:` blocks. `34_Cross_Sport_Tools.py` imports the same loaders directly (it needs fresh per-sport prop evaluations for comparison and for cross-sport parlay legs) but has its own render logic for the two tools, since neither has an equivalent in `betting_shared.py`.

### MLB/NHL: a new shared render path, not CFB/CBB's

MLB and NHL ship no default data and need no live schedule/team-averages fetch the way CFB (key-gated) or CBB/NBA (live ESPN/`nba_api` fetch) do — `render_college_moneylines_tab`'s volatility-caption logic (real margin/total standard deviation from live games) has no MLB/NHL equivalent, so both share a new, smaller `render_offline_props_tab`/`render_offline_moneylines_tab` pair instead of joining `COLLEGE_SPORTS`. This is an additive change: `render_college_props_tab`/`render_college_moneylines_tab` and their CFB/CBB callers are untouched.

## What's unchanged from the original page

- Odds still come from exactly two places per sport: our own default file, or a file you upload — nothing here fetches a sportsbook (see offline_data_contract.md). MLB/NHL ship no default file this cycle (see mlb_pipeline.md / nhl_pipeline.md) — both work fully from an uploaded file, the same experience CFB has without a `CFBD_API_KEY`.
- CFB's week selector, CFBD_API_KEY empty-state messaging, and CBB's ESPN-sourced disclosure text are identical to before, just now living in `32_CFB_Betting.py`/`33_CBB_Betting.py` directly instead of behind a sport-toggle `if` branch.
- The Cross-Sport Parlay tab's "default every sport's checkbox to `True`" widget-persistence workaround (see ui_design.md) is unchanged, since the underlying Streamlit gotcha it works around hasn't changed — now six checkboxes instead of four.

## One real behavior change (from the original single-page era)

`34_Cross_Sport_Tools.py`'s cross-sport loading no longer special-cases "reuse the currently-toggled sport's already-loaded data" the way the old single page did (it had a `sport == "NFL"` check per branch, since the page's own top-level `sport` toggle controlled what was already loaded). With no page-level sport toggle to reuse, every sport is loaded fresh when its checkbox is checked and the load button is clicked — slightly more real network/compute per click, but a simpler, more predictable page (no dependency on which page you visited most recently). The `st.cache_data` TTL layer underneath still avoids a true re-fetch within its window.

## Testing

`tests/test_betting_engine_page.py` covers the four original sport pages (`NflBettingPageTests`, `NbaBettingPageTests`, `CfbBettingPageTests`, `CbbBettingPageTests`) and Cross-Sport Tools (`CrossSportToolsPageTests`, extended this cycle with MLB/NHL cases) with one `AppTest`-per-page class each, asserting each page renders its real, correct tab set with no exceptions and, where a real default data source exists, real non-empty tables. `tests/test_mlb_nhl_betting_pages.py` covers the two new pages separately, since both need different empty-state assertions (no default data at all, rather than a key gate or a game-day dependency).
