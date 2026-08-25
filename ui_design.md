# UI design: the unified Betting Engine page

Covers the Streamlit design decisions behind the betting pages (`UniversalQuantAgent/app/betting_shared.py` and `app/pages/30-34_*.py` -- see [ui_betting_tabs.md](ui_betting_tabs.md) for why one page became five). For the engine functions they call, see [betting_engine.md](betting_engine.md) / [betting_engine_advanced.md](betting_engine_advanced.md).

## Layout

One sport toggle (`st.radio`, "NFL" / "NBA"), five tabs beneath it:

1. **Player Props** -- fair-line comparison table, sortable, with an opt-in trend overlay (NBA only -- see below).
2. **Money Lines** -- fair spread/total/moneyline vs. loaded odds.
3. **Parlays** -- same-sport parlay builder.
4. **Player Comparison** -- two props from the current sport, side by side.
5. **Cross-Sport Parlay** -- NFL + NBA legs in one parlay, independent of the sport toggle.

Tabs 1-4 render whichever sport the toggle currently selects; tab 5 is the one place both sports can be active at once (see "Cross-sport is lazy," below).

## "Both NFL and NBA branches render identically under sport toggle"

Concretely: every tab that exists for one sport exists for the other, built from a shared code path where the underlying data allows it (`_parlay_builder` is one function serving both sports' Parlays tab, not two near-duplicate blocks; the Player Comparison tab is one code path with a sport-conditional column-metrics list, not two separate tabs). Where behavior genuinely differs, it differs for a real, disclosed reason:

- NFL's Player Props/Money Lines/Parlays always have real player/schedule data (a full season's real games). NBA's Money Lines and *matchup-adjusted* Props are empty outside a real game day -- there's no live NBA game to price. This isn't an inconsistency to "fix" by fabricating a matchup; it's the honest state, and the Props tab falls back to the raw, season-long default lines (no matchup adjustment) so the tab isn't empty even in the off-season.
- NFL parlay legs use `betting.make_leg`/`evaluate_parlay`; NBA legs use `modules.nba_parlay_engine`'s equivalents. `_parlay_builder(prop_evaluations, sport_key, empty_icon)` branches internally on `sport_key` only for the handful of genuinely different field names (`row["market"]` vs `row["category"]`, etc.) -- the UX around it (quick-add, leg summary table, metrics, correlation warning) is one shared code path.

## Consolidating duplicate evaluation calls

Before this cycle, the page called `evaluate_props`/`price_aware_evaluations` and `evaluate_games` **twice** each render -- once building the Props/Money Lines tab, again inside the Parlays tab. `_load_nfl_evaluations`/`_load_nba_evaluations` now compute each sport's prop and game evaluations **once**, near the top of the script, and every tab below reads from that single result. This is a real, measurable reduction in redundant computation, not just a style cleanup -- see [performance_notes.md](performance_notes.md).

## The trend overlay is opt-in, not always-on

`team_pace_trend`/`player_usage_trend` (`modules/nba_trend_signals.py`) are TTL-cached, but still each represent a real API call the first time they're requested for a given team/player within the cache window. Showing them as columns on every row of a props table by default would mean up to ~2x the shown row count in live calls on a cold cache. The **"Show pace / usage trend overlay"** checkbox on the Props tab makes this cost opt-in and visible (the checkbox label says so), rather than a hidden cost every page load pays regardless of whether anyone looks at the overlay.

## Cross-sport is lazy, not eager

The Cross-Sport Parlay tab needs *both* sports' priced props, but the sport toggle only loads one sport's data per render by default -- deliberately, so a user who only cares about NFL never pays NBA's live-data cost (and vice versa). Making the Cross-Sport tab eagerly load both sports on every render would defeat that. Instead, it shows a **"Load NFL + NBA legs for a cross-sport parlay"** button; clicking it loads whichever sport isn't already loaded (the already-loaded sport's data is reused via Streamlit's own `@st.cache_data`, so switching the toggle first and then opening this tab costs nothing extra) and stores both in `st.session_state` so subsequent reruns (e.g. picking legs) don't reload either sport.

## The Quick-add / session_state pattern

The Parlays tab's "Quick-add top N by edge" button writes directly to `st.session_state[picker_key]` so the leg `st.multiselect` reflects the button click on the next rerun. Streamlit forbids passing both a widget's `default=` argument *and* pre-setting its `session_state` value in the same run (it logs a policy warning and the behavior is fragile) -- so the multiselect here takes **no** `default=` at all; instead, `st.session_state[picker_key]` is seeded once (or reseeded if it contains a label no longer present in the current `options`, e.g. after a new upload changes what's available) directly, before the widget is instantiated. If you add another widget with a programmatic "set this value" button, follow this same pattern rather than fighting Streamlit's `default=`/session_state conflict.

## Player Comparison table: stringify before displaying

The comparison table mixes numeric fields (`"Line"`, `"Edge"`) and string fields (`"Recommended side"`, `"Risk"`) within the *same* display column (one column per compared prop, one row per metric). Left as native Python values, pandas infers an `object` dtype column with mixed types, which pyarrow (Streamlit's serialization layer) cannot convert cleanly -- it silently coerces with a logged warning rather than failing, but the result is fragile. Every value in this table is built with `str(...)` before going into the DataFrame for exactly this reason; if you add a new comparison metric, keep doing that rather than passing the raw value through.
