# NBA injury impact: real opponent absences and real teammate-absence splits

`modules/nba_injury_impact.py`. See [matchup_engine.md](matchup_engine.md) for how this fits into the wider matchup-aware layer, and [offline_data_contract.md](offline_data_contract.md) for the injury-data rule this module follows exactly.

Two genuinely different real signals live here — genuinely separate concerns, not two names for the same idea:

## `opponent_injury_context(opponent_team, season)` — is the opponent missing a real rim protector?

Loads the real, offline-only injury report (`modules.injury_parser.load_injury_data_from_file` — our own file or a user upload, never a live fetch, per `offline_data_contract.md` rule 1), filters to the opponent's real `OUT`/`QUESTIONABLE` rows (same `_SEVERITY_BY_STATUS` weighting `modules.matchup_model` already uses), then cross-references each real absent name against real per-player rim-defense volume from `modules.nba_defense_model`'s underlying `leaguedashptdefend` data. A name only counts as a real rim defender being out if they clear `_MIN_RIM_CONTESTS = 100` real contests this season — a real, disclosed volume floor, not a guess from roster position (the offline injury record has no position field to guess from in the first place).

Ships correctly empty by default: `data/injuries.json` is empty-by-design, so `absences` and `rim_defender_out` are both empty/`None` until a real report is loaded — a normal state, not an error.

## `teammate_absence_split(player_name, teammate_name, season, *, min_games=3)` — this player's own real stats, with vs. without one named teammate

A direct, real measurement, not an inference from a defense feed: pulls both players' own real game logs (`modules.projections._game_log_with_fuzzy_fallback` — the exact same fetch `fuse_projection` already uses), then partitions the focal player's real games by whether the named teammate's own log shows a matching real date. This is **game-level, not minute-level** — it does not distinguish "did not play" from "not on the roster that game" for any other reason, and that's disclosed in the docstring, not glossed over.

Every result carries `insufficient_sample: bool` (true when either split has fewer than `min_games` real games) so a two-game split is never presented with the same confidence as a twenty-game one. Live-tested this cycle: Jokic/Murray came back `insufficient_sample: True` (only 2 real games without Murray so far this season) — the module reported that honestly rather than confidently asserting a trend from 2 games.
