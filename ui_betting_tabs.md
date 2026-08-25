# Betting UI: one page per sport, plus a shared cross-sport tools page

How the single `30_Betting_Engine.py` page (a sport toggle over five tabs) became five separate pages this cycle. See [betting_engine.md](betting_engine.md) / [betting_engine_advanced.md](betting_engine_advanced.md) for the engine functions each page calls, and [ui_design.md](ui_design.md) for the general Streamlit patterns in play (widget-`session_state` seeding, cached loaders).

## The five pages

| Page | Sport | Tabs |
|---|---|---|
| `app/pages/30_NFL_Betting.py` | NFL | Money Lines, Props, Parlays |
| `app/pages/31_NBA_Betting.py` | NBA | Money Lines, Props, Parlays |
| `app/pages/32_CFB_Betting.py` | CFB | Money Lines, Props, Parlays |
| `app/pages/33_CBB_Betting.py` | CBB | Money Lines, Props, Parlays |
| `app/pages/34_Cross_Sport_Tools.py` | all four | Player Comparison, Cross-Sport Parlay |

Each of the four sport pages is fixed to one sport (no toggle) and renders exactly the three tabs above, matching this cycle's spec. Player Comparison and Cross-Sport Parlay apply *across* sports by nature (comparing two props, or mixing legs from different sports in one parlay) — duplicating either across four pages would mean four copies of the same logic, so both moved to one shared fifth page instead of being dropped or bolted onto each sport page as a fourth/fifth tab.

## `app/betting_shared.py`: one copy of every loader and renderer

The single old page had one copy of each sport's cached data loader (`_load_nfl_evaluations`, etc.) and each tab's rendering logic. Splitting into five page files without an extraction step would have meant four to five copies of ~700 lines of near-identical code. Instead, `app/betting_shared.py` holds:

- Every `@st.cache_data`-wrapped loader (`load_nfl_evaluations`, `load_nba_evaluations`, `load_cfb_evaluations`, `load_cbb_evaluations`) — unchanged behavior from the original page, same TTLs, same odds-source disclosure text.
- Every tab renderer (`render_nfl_props_tab`, `render_nba_props_tab`, `render_college_props_tab` for CFB/CBB's shared shape, and the equivalent Money Lines renderers) — pure functions that take already-loaded data and render, no page-specific state.
- `render_parlay_builder` — the shared quick-add/leg-picker/evaluate UX all four sports already used identically.

Each of the four sport pages is now genuinely thin: load this sport's data, call three render functions inside three `with tab:` blocks. `34_Cross_Sport_Tools.py` imports the same loaders directly (it needs fresh per-sport prop evaluations for comparison and for cross-sport parlay legs) but has its own render logic for the two tools, since neither has an equivalent in `betting_shared.py`.

## What's unchanged from the original page

- Odds still come from exactly two places per sport: our own default file, or a file you upload — nothing here fetches a sportsbook (see offline_data_contract.md).
- CFB's week selector, CFBD_API_KEY empty-state messaging, and CBB's ESPN-sourced disclosure text are identical to before, just now living in `32_CFB_Betting.py`/`33_CBB_Betting.py` directly instead of behind a sport-toggle `if` branch.
- The Cross-Sport Parlay tab's "default every sport's checkbox to `True`" widget-persistence workaround (see ui_design.md) is unchanged, since the underlying Streamlit gotcha it works around hasn't changed.

## One real behavior change

`34_Cross_Sport_Tools.py`'s cross-sport loading no longer special-cases "reuse the currently-toggled sport's already-loaded data" the way the old page did (it had a `sport == "NFL"` check per branch, since the page's own top-level `sport` toggle controlled what was already loaded). With no page-level sport toggle to reuse, every sport is loaded fresh when its checkbox is checked and the load button is clicked — slightly more real network/compute per click, but a simpler, more predictable page (no dependency on which page you visited most recently). The `st.cache_data` TTL layer underneath still avoids a true re-fetch within its window.

## Testing

`tests/test_betting_engine_page.py` was rewritten from one `AppTest`-per-sport-toggle-state pattern to five `AppTest`-per-page classes (`NflBettingPageTests`, `NbaBettingPageTests`, `CfbBettingPageTests`, `CbbBettingPageTests`, `CrossSportToolsPageTests`), asserting each page renders its real, correct tab set with no exceptions and, where a real default data source exists, real non-empty tables.
