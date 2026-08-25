# Cross-Sport Tools: Player Comparison and Cross-Sport Parlay across six sports

How `app/pages/34_Cross_Sport_Tools.py`'s two tools generalize across NFL, NBA, CFB, CBB, MLB, and NHL. See [ui_betting_tabs.md](ui_betting_tabs.md) for where this page fits among the seven betting pages, [betting_engine.md](betting_engine.md) for the unified contract each sport satisfies, and [betting_engine_advanced.md](betting_engine_advanced.md) for the cross-sport parlay mechanism this page's second tab drives.

## Why one shared page, not per-sport tabs

Player Comparison and Cross-Sport Parlay both apply *across* sports by nature — comparing two priced props, or mixing legs from different sports in one parlay — duplicating either across six sport-specific pages would mean six copies of the same logic (see ui_betting_tabs.md's original five-page rationale, extended here to six). Both tools instead live on this one page, added to as each new sport joined the roster rather than redesigned.

## Player Comparison

A `st.radio` toggle (`["NFL", "NBA", "CFB", "CBB", "MLB", "NHL"]`) selects which sport's already-loaded, already-priced props to compare two of, side by side, in one small table. Each sport branch calls that sport's own `app.betting_shared.load_*_evaluations` loader and supplies a `label_fn` (how to name each option in the picker) and a `metrics` list (which fields to show as table rows) — CFB/CBB/MLB/NHL's branches are identical in shape (`player_name`/`category`/`line`/`recommended_side`/`recommended_edge`/`recommended_ev`/`risk_tier`/`basis`) since all four share the same "no pre-existing rich model" prop-evaluation row shape (see betting_engine.md); NFL and NBA each keep their own distinct field names.

MLB and NHL ship no default props this cycle (see mlb_pipeline.md / nhl_pipeline.md) — selecting either with nothing uploaded correctly renders the page's existing "not enough priced props to compare" empty state rather than raising; this is a normal state, not a bug, and is asserted directly by `tests/test_betting_engine_page.py::CrossSportToolsPageTests::test_mlb_and_nhl_comparison_shows_the_disclosed_empty_state_without_exceptions`.

## Cross-Sport Parlay

A row of six checkboxes (default-checked) selects which sports to load; clicking "Load selected sports for a cross-sport parlay" loads each checked sport fresh (this page doesn't reuse the sport pages' own cache keys) and stores the result in `st.session_state["cross_sport_loaded"]`. Every loaded sport's priced rows feed one combined `st.multiselect` leg picker (each option prefixed with its sport, e.g. `"MLB: ..."`), and `modules.unified_parlay_engine.make_unified_leg`/`evaluate_cross_sport_parlay` do the actual cross-sport combinatorics — this page's own leg-building loop is generic across CFB/CBB/MLB/NHL (all four share the same `recommended_side`/`category`/`player_name`/`team` row shape), needing no new branch per sport as the roster grew from four sports to six.

`modules.unified_parlay_engine`'s `_MAKE_LEG_BY_SPORT`/`_DETECT_CORRELATIONS_BY_SPORT` dispatch dicts (extended with `"MLB"`/`"NHL"` entries this cycle — see betting_engine.md) partition legs by sport before running each sport's own correlation detector, so an MLB leg and an NHL leg (or any other cross-sport pair) are never treated as correlated — this generalizes to any number of sports with no redesign, verified directly by `tests/test_unified_parlay_engine.py::DetectCrossSportCorrelationsTests`.

## Testing

`UniversalQuantAgent/tests/test_betting_engine_page.py::CrossSportToolsPageTests` — real `AppTest` runs of the page itself (both tabs render with no exceptions, the sport toggle offers all six sports, CFB/CBB/MLB/NHL each switch without raising, the cross-sport load button and leg picker work). `UniversalQuantAgent/tests/test_unified_parlay_engine.py` covers the underlying dispatch/correlation-partitioning math directly, independent of any one sport's live data availability.
