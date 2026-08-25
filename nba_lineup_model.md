# NBA lineup model: real team on/off net-rating swings

`modules/nba_lineup_model.py`. See [matchup_engine.md](matchup_engine.md) for how this fits into the wider matchup-aware layer.

## Reused, not duplicated

This module fetches nothing new. `modules.nba_advanced.fetch_on_off_splits` (wrapping `nba_api.stats.endpoints.teamplayeronoffsummary.TeamPlayerOnOffSummary`) and `summarize_on_off` already exist, built for the Correlation Lab / Model Drivers analysis pages before this cycle. `team_on_off_impact(team_name, season, *, limit=8)` imports and calls both directly — see betting_engine.md's "shared, not duplicated" rule, applied to an existing analysis-page capability instead of a new fetch.

## The one real bug this cycle found (and worked around locally)

`fetch_on_off_splits`'s real response is **three** tables, not two: `[0]` one team-wide overall row, `[1]` real per-player "On" splits, `[2]` real per-player "Off" splits (verified live — `nba_advanced.py`'s own docstring/type hints don't say this). `summarize_on_off` expects exactly `[on_table, off_table]`. `nba_advanced.py`'s *own* only other caller (inside `analyze_matchup`, building `snapshot["top_player_on_off_splits"]`) passes the raw 3-table list straight through, which silently produces a single bogus `{"player": "Unknown", ...all zeros}` row instead of real per-player data.

This module works around it correctly, locally: `team_on_off_impact` slices `tables[1:]` before calling `summarize_on_off`, which is a correct *call* to the existing function, not a modification of it. The upstream bug in `nba_advanced.py` itself was deliberately **not fixed** here — "do not refactor NFL/NBA internals" — and was flagged as a separate follow-up task instead (see this cycle's spawned background task).

## Real-name matching: "Last, First" and accented characters

`nba_api`'s on/off rows name real players as `"Last, First"` (e.g. `"Jokić, Nikola"`), not `"First Last"`. `teammate_on_off_swing(team_name, teammate_name, season)` normalizes both the caller's input and the real row's name — a comma-swap (`_as_first_last`) plus `modules.sportsbook_parser.normalize_player_name` (already used elsewhere in this codebase for accent/punctuation-insensitive matching) — so `"Nikola Jokic"` correctly matches the real row named `"Jokić, Nikola"`. Verified live this cycle: this exact accent mismatch silently returned `None` before the fix.

## What this is (and isn't)

A **team-level** signal: real net-rating swing while one named teammate is on vs. off the court, for the whole team. For one specific player's own real box-score rates conditional on a named teammate's presence (e.g. "Jokic's real assists with vs. without Murray"), see [nba_injury_impact.md](nba_injury_impact.md)'s `teammate_absence_split` instead — a different, individual-player-level real measurement.
