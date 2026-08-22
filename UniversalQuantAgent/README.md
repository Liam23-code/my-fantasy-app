# Universal Quant Agent

Universal Quant Agent is a modular, beginner-friendly research project for finance, NBA, NFL, player projections, and NBA player-prop comparison. The current models favor transparent rules and standard regression over opaque modeling. It is educational software, not financial or wagering advice.

## Project structure

```text
UniversalQuantAgent/
|-- app/
|   |-- app.py                    # Streamlit home and shared UI helpers
|   `-- pages/                    # Native Streamlit analysis pages
|-- data/                         # Optional local datasets and model results
|-- modules/
|   |-- finance.py, sports.py, nfl.py, nba_advanced.py
|   |-- projections.py, player_finder.py
|   |-- correlation_engine.py     # Player, team, and slate correlations
|   |-- similarity_engine.py      # Normalized cosine player matching
|   |-- edge_heatmap.py           # Slate-wide prop-edge matrices
|   |-- data_quality.py           # Shared normalization and fallback policy
|   |-- context_engine.py         # Role, trend, schedule, and injury context
|   |-- fusion_model.py           # Multi-model projection fusion
|   |-- reliability.py            # Quant Reliability Score
|   |-- recommendations.py        # Reliability-aware prop rankings
|   |-- sportsbook_parser.py      # Offline market/name normalization (no network)
|   |-- sportsbook_scraper_disabled.py  # Hard-disabled fetch stubs -- kept for reference only
|   |-- minutes_model.py          # Explainable minutes projection
|   |-- injury_parser.py          # Offline injury normalization + injuries.json/upload loaders
|   |-- injury_scraper_disabled.py      # Hard-disabled fetch stub -- kept for reference only
|   |-- pace_model.py             # Matchup pace projection
|   |-- matchup_model.py          # Player/opponent difficulty
|   |-- props.py                  # Prop comparison and recommendations
|   |-- parlay.py                 # Multi-leg risk and correlation review
|   |-- daily_slate.py            # Today's games, lines, and projections
|   `-- model_performance.py      # MAE, bias, and line hit-rate summaries
|-- main.py                       # Command-line application
|-- run_app.bat                   # Windows Streamlit launcher
`-- requirements.txt
```

## Install and launch

Python 3.10 or newer is recommended.

```powershell
cd UniversalQuantAgent
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\run_app.bat
```

The batch file activates the virtual environment and runs the Streamlit dashboard. You can also run the command manually. The sidebar exposes NBA, Player Analysis, Player Projections, Player Finder, NFL, Finance, Opportunities, prop tools, Daily Slate, Model Performance, and the Visual Analytics Lab.

## Prop comparison

```python
from modules.props import compare_props
from modules.recommendations import recommend_props

rows = compare_props(
    player_name="Nikola Jokic",
    opponent_team="BOS",
    categories=["points", "rebounds", "assists", "PRA"],
    season="2025-26",
)
recommendations = recommend_props(["points", "rebounds"], min_edge=1.5, min_confidence=55)
```

Each comparison row preserves the established prop schema while its projection now comes from the fusion model. The result contains player, team, category, projection, minutes-adjusted projection, line, edge, confidence range, best sportsbook, lean, drivers, sportsbook, and timestamp. Reliability is included in the drivers without changing the contract. `edge` is fused projection minus sportsbook line.

**Live sportsbook/injury fetching is permanently disabled.** The functions above
(`fetch_all_sportsbook_props`, `fetch_daily_games`, `fetch_injury_report`, ...) now live in
`modules/sportsbook_scraper_disabled.py` and `modules/injury_scraper_disabled.py` as hard-fail
stubs -- calling any of them raises `RuntimeError` immediately rather than making a network
request. This app does not scrape DraftKings, FanDuel, BetMGM, Caesars, ESPN BET, or any other
sportsbook or odds site, and does not call any live injury endpoint. The `UQA_*_PROPS_URL`
environment variables these functions used to read are no longer consulted.

Their safe parsing/normalization logic (player/team/category normalization, JSON-payload market
parsing) was preserved and now lives in `modules/sportsbook_parser.py` and
`modules/injury_parser.py`, both fully offline. Pages that depended on the old live fetch (Daily
Slate, Prop Analyzer, Prop Recommendations) will raise the same `RuntimeError` until they are
wired to an offline odds source, the same way the injury-consuming modules
(`context_engine.py`, `matchup_model.py`, `minutes_model.py`, `pace_model.py`) were rewired to
`modules.injury_parser.load_injury_data_from_file` in this pass.

## Upgraded Player Analysis

`fetch_nba_player_stats()` now returns a structured season and recent-form record:

```python
from modules.sports import fetch_nba_player_stats

player = fetch_nba_player_stats("Nikola Jokic", "2025-26")
print(player["season_avg"])
print(player["last5_avg"])
print(player["last10_avg"])
print(player["season_totals"])
print(player["advanced"])
print(player["league_ranks"])
```

The Player Analysis page displays PPG, RPG, APG, MPG, season totals, usage, true shooting, the transparent PER estimate, and a season-versus-last-5-versus-last-10 chart. It also shows each player's current NBA rank for PPG, RPG, APG, MPG, usage, true shooting, and estimated PER. Every rank uses the same small schema, for example `{"rank": 4, "out_of": 520}`. If game logs are temporarily unavailable, season-level averages remain available as an explicit fallback.

NBA team comparisons use the same ranking format for scoring, rebounding, assists, offensive rating, defensive rating, net rating, and pace. The dashboard presents these in a dedicated **League ranks** tab. Defensive rating is ranked with lower values treated as better; the other team categories use higher values as better.

## Dashboard organization and team colors

The Streamlit dashboard uses a consistent global theme, responsive cards, polished forms, compact metric groups, and tabbed detail views. NBA team colors accent player and team profiles. Team logos and player headshots load from the lightweight NBA CDN and disappear cleanly if an image is unavailable.

Player Analysis includes Overview, Recent Form, NBA Rankings, Season Totals, Similar Players, and Correlation Heatmaps. Recent Form uses grouped season-versus-last-5-versus-last-10 bars plus usage/minutes and TS%/PER trend lines. Player Projections keeps Projection, Model Drivers, Reliability, and Player Context, with a pace-versus-opponent-difficulty scatter and consistent Plotly styling.

Every native page under `app/pages/` calls `apply_global_theme()` on each rerun. This is important because Streamlit executes multipage scripts independently; importing the main page once is not enough to preserve CSS. The shared profile component displays player headshots at 60 pixels wide and team logos at 40 pixels wide everywhere it is used, including NBA Analysis, Player Analysis, Player Projections, Daily Slate, Similar Players, and the Visual Analytics Lab.

## NBA API retries and fallback projections

`modules/nba_cache.py` defines the shared NBA ingestion policy:

- Every projection-facing NBA endpoint uses a 10-second request timeout.
- Failed calls are attempted three times with exponential backoff.
- Successful responses are retained in memory and persisted under `data/nba_cache/` as last-known historical data.
- If the live provider remains unavailable, callers use memory, disk cache, or conservative season-average/league-average frames in that order.
- A short provider cooldown prevents minutes, pace, matchup, context, and fusion components from each repeating the same known outage.

Fallback frames use the same columns and model contracts as live frames, so `fusion_model`, `context_engine`, `minutes_model`, `pace_model`, and `matchup_model` continue normally. When any of these paths is used, Player Projections displays:

> NBA API unavailable &mdash; using enhanced fallback projections.

Fallback results are intentionally conservative and should be refreshed when NBA.com is available again. They are designed to keep the dashboard usable, not to masquerade as current live data.

### Enhanced fallback and fusion math

The minutes fallback never substitutes zero. It selects a positive value in this strict order: season average, last-10 average, then last-5 average. The returned payload identifies the selected source and its quality. If none of those windows exists, the model raises a clear data-quality error instead of silently collapsing every rate statistic.

Usage and true shooting use automatically reweighted 50/30/20 blends of season, last-10, and last-5 values. Usage is then multiplied by the opponent-usage factor and the matchup pace factor. True shooting is multiplied by the opponent defensive-efficiency factor.

Pace uses a 40% team, 40% opponent, and 20% league-average season blend before recent pace is incorporated. Its explicit matchup factor is `(team_pace + opponent_pace) / 2 / league_average_pace`. Opponent difficulty uses cached defensive rating when possible, league defensive rating otherwise, plus role-specific defense derived from rebounding, blocks, and steals percentiles. Small team/role references are used only when those columns are absent.

Fusion builds separate minutes, usage, efficiency, and context candidates. Their reliability scores are normalized into weights that sum to one. Each statistic is smoothed with its season average using `(weighted_sum + season_average) / 2`. Projected values and confidence ranges are not forced through output floors or ceilings; only metrics whose documented contract is 0-100, such as reliability and difficulty, remain bounded.

The Player Projections page shows final, season, last-10, and last-5 production together. Model Drivers exposes the selected minutes source, usage blend, TS% blend, context factors, normalized fusion weights, fallback source, and reliability components.

## Correlations, slate edges, and similar players

The Visual Analytics Lab brings the new visual engines together. The same views also appear where they are most useful: player correlations and similarity in Player Analysis, team correlations in NBA Analysis, and slate correlations plus edge maps in Daily Slate.

~~~python
from modules.correlation_engine import (
    compute_player_correlations,
    compute_team_correlations,
    compute_slate_correlations,
)
from modules.edge_heatmap import prepare_edge_heatmap
from modules.similarity_engine import compute_player_similarity

player_matrix = compute_player_correlations("Nikola Jokic", "2025-26")
team_matrix = compute_team_correlations("DEN", "2025-26")
slate_matrix = compute_slate_correlations(slate["projections"])
edge_matrix = prepare_edge_heatmap(prop_rows, ["points", "rebounds"], "edge")
similar = compute_player_similarity("Nikola Jokic", "2025-26")
~~~

Correlation results are plain dictionaries containing the subject, observation count, available statistics, a finite correlation matrix, source records, and warnings. Constant or missing columns are omitted instead of breaking the heatmap.

The slate edge map accepts points, rebounds, assists, PRA, fantasy, and minutes. Green cells indicate positive projection edges, yellow cells are near the market line, and red cells indicate negative edges. Rows can be sorted by edge, reliability, or confidence.

Player similarity uses min-max normalized season and recent-form vectors with cosine similarity. It returns the ten closest profiles, a 0–100 score, and auditable dimension scores for usage, minutes, efficiency, pace, role, scoring, rebounding, assisting, defense, recent form, and matchup context.
## Unified data quality and safe ingestion

**modules/data_quality.py** owns column normalization, duplicate-column merging, numeric coercion, missing-value policy, MM:SS minutes parsing, rolling windows, pace and defensive-rating fallbacks, availability conversion, fuzzy player matching, and completeness scoring. Existing helpers delegate to this layer so every domain uses the same numeric and missing-data rules.

Provider-facing code must use four defensive helpers before reading nested data:

- **safe_get(node, key, default)** reads dictionaries only and returns the default for lists, scalars, nulls, or malformed nodes.
- **safe_dict(node)** and **safe_list(node)** enforce the expected container without raising.
- **safe_scalar_to_dict(node)** converts a scalar provider response into a dictionary with a value field.

The injury adapter accepts mixed ESPN payloads, including integer or string type values, and always returns records with string team, player, and status fields plus a dictionary details field. Status is normalized to OUT, QUESTIONABLE, PROBABLE, or ACTIVE.

Minutes, pace, matchup, and context results preserve their original fields and also expose a common envelope:

~~~python
{
    "value": 32.4,
    "confidence": 76.0,
    "details": {...},
}
~~~

**normalize_model_output()** in **modules/fusion_model.py** applies the same contract to legacy dictionaries and scalar model results before projection fusion. This keeps existing callers compatible while preventing nested attribute errors.

## Context, fusion, and reliability

```python
from modules.context_engine import get_player_context
from modules.fusion_model import fuse_projection
from modules.reliability import get_reliability_score

context = get_player_context("Nikola Jokic", "BOS", "2025-26")
fused = fuse_projection("Nikola Jokic", "BOS", "2025-26")
reliability = get_reliability_score("Nikola Jokic", "BOS", "2025-26")
```

The context engine measures role and rotation stability, usage, minutes and efficiency trends, pace, opponent history, injury impact, blowout risk, home/away splits, back-to-back fatigue, and coach-rotation consistency. Fusion combines base regression, projected minutes, pace, matchup difficulty, availability, and recent form while returning per-stat contribution breakdowns. Reliability scores data completeness, minutes and role stability, opponent predictability, pace stability, injury certainty, interval confidence, and available historical MAE. Model boundaries normalize nested results before reading them, so a provider returning a scalar where a dictionary was expected no longer causes an `'int' object has no attribute 'get'` failure.
## Context models

```python
from modules.minutes_model import project_minutes
from modules.injury_parser import get_player_availability, load_injury_data_from_file
from modules.pace_model import project_pace
from modules.matchup_model import project_matchup_difficulty

minutes = project_minutes("Nikola Jokic", "BOS")
availability = get_player_availability("Nikola Jokic", load_injury_data_from_file())
pace = project_pace("DEN", "BOS")
matchup = project_matchup_difficulty("Nikola Jokic", "BOS")
```

Injury data is offline-only: `load_injury_data_from_file()` reads `data/injuries.json` (empty by
default -- populate it with real reported injuries, or use
`modules.injury_parser.load_injury_data_from_user_upload` for a user-supplied file). No module in
this app fetches injury or odds data from a live site; see
`modules/sportsbook_scraper_disabled.py` and `modules/injury_scraper_disabled.py`.

Minutes uses rolling minutes, role trend, schedule, rotation stability, injury status, and blowout risk. Pace blends both teams' season and recent rates. Matchup difficulty uses opponent defensive rating and available player matchup history. Missing injury records are clearly marked as an unconfirmed ACTIVE fallback.

## Parlay and daily slate

```python
from modules.parlay import build_parlay
from modules.daily_slate import get_daily_slate

parlay = build_parlay(rows[:2])
slate = get_daily_slate()
```

The parlay result reports combined edge, reliability-adjusted risk, correlation and injury warnings, pace correlation, blowout risk, simplified expected value, and a plain-language summary. Expected value is a heuristic placeholder, not a calibrated probability or recommendation. The daily slate joins ESPN's public schedule with verified book lines and builds projections for each distinct player with a resolvable team/opponent.

## Model performance

Place settled results in `data/model_performance.csv` with these columns:

```csv
category,projection,actual,sportsbook_line,date
points,25.4,27,24.5,2026-01-10
```

The Model Performance page displays MAE, model bias, hit rate versus the line, and rolling charts. Optional `matchup_difficulty`, `pace`, and `minutes` columns let reliability use MAE from similar historical contexts. With no settled file, it shows an honest empty-state message.

## Existing analysis examples

```python
from modules.finance import analyze_stock
from modules.sports import compare_teams
from modules.nfl import compare_nfl_teams
from modules.projections import project_player_statline

market = analyze_stock("AAPL")
nba = compare_teams("DEN", "BOS", include_advanced=True)
nfl = compare_nfl_teams("KC", "BUF")
player = project_player_statline("Nikola Jokic", "BOS")
```

## Graph Lab

Graph Lab adds four focused Streamlit pages under a collapsible sidebar section:

- **Shot-Chart Heatmap** draws an NBA half court with team-color density shading. Switch between makes, attempts, field-goal efficiency, and shot-frequency volume. Hovered bins show attempts, makes, FG%, and expected points per attempt.
- **Efficiency Radar** compares TS%, usage, assist rate, rebound rate, turnover rate, defensive impact, and pace fit on a league-relative 0-100 scale. Season, last-10, and last-5 windows use the same chart contract.
- **Badge Graph** defaults to a transparent radar/spider profile and keeps the circular identity wheel as an alternate view. Both views share the same eight skills, 60/40 league-position percentiles, red-to-green tiers, team ring, elite glow, sample weighting, and matchup context.
- **Trend Graphs** includes rolling momentum, a performance timeline, league usage-versus-efficiency context, position-specific opponent difficulty, and pace-impact curves.

The visual modules return ordinary Plotly figures, keeping them reusable outside Streamlit:

```python
from modules.shot_chart import render_shot_chart
from modules.radar_chart import render_efficiency_radar
from modules.badge_graph import render_badge_graph, render_spider_badge_graph
from modules.trend_graphs import (
    render_momentum_line,
    render_performance_timeline,
    render_usage_efficiency_scatter,
    render_opponent_difficulty_curve,
    render_pace_impact_curve,
)

shot_figure = render_shot_chart("Nikola Jokic", "2025-26", "Efficiency")
radar_figure = render_efficiency_radar("Nikola Jokic", "2025-26", "Last-10")
spider_figure = render_spider_badge_graph(
    "Nikola Jokic",
    "Adjusted",                      # or "Raw"
    "Same position",                 # or "Entire league"
    season="2025-26",
    filter_minimum_samples=True,
    opponent_team="BOS",
    gradient_fill=True,
)
badge_figure = render_badge_graph(
    "Nikola Jokic", "2025-26", "Adjusted", "Same position", True, "BOS"
)
```

Badge sample confidence uses dynamic thresholds: 100 three-point attempts, 75 mid-range attempts, 100 finishing attempts, 100 free throws, and 300 total shots. Dunks contribute only through Finishing and use its paint-scoring confidence. For shorter seasons, each threshold becomes the smaller of the default or 35% of the player's total shots. Low-sample skills are explicitly discounted in adjusted mode and surfaced with a warning. The minimum-sample toggle controls the peer pool used for percentile comparisons.

Close-shot dilution prevents overlapping zones from double-counting player identity. Finishing receives 60% of close-shot contribution and 90% of dunk contribution; mid-range retains true pull-up, fadeaway, elbow, floater, and short-jumper production plus only 55% of close-shot contribution. Dunking is not a separate axis or standalone score. Hover details show the resulting contribution breakdown alongside raw and adjusted ratings.

The spider view uses a 40-60% transparent polygon, a thin dark outline, optional layered depth, per-axis tier colors, position/league comparison geometry, and the same lightweight team ring and elite glow as the wheel. Its eight axes are Finishing, Mid-range, 3PT Shooting, Playmaking, Defense, Rebounding, Efficiency, and Pace Fit.

`generate_badge_insights()` reads the exact visible badge values to produce a one-to-two sentence identity summary and low-sample note. Hover details expose league percentile, position percentile, the canonical 60/40 rating, attempts, dynamic minimum, confidence, and context adjustment.
Player Analysis now includes Shot Profile, Efficiency, Badge Profile, and Trends tabs. These views load only when requested so normal player analysis remains lightweight.

The sidebar uses grouped, collapsible sections for NBA Tools, Graph Lab, Betting Tools, Advanced Models, NFL Tools, and Finance. Existing pages remain available through the same application entry point.

Graph data uses the shared NBA retry/cache policy. Cached data is preferred during provider outages; charts show an explicit data note when only a partial or neutral fallback is available and never invent shot locations.
## Premium insights and confidence UI

`modules/insights_engine.py` turns player, matchup, slate, similarity, correlation, and edge data into short plain-language observations. Each function returns a list of strings, so the same insights can be reused by Streamlit, an API, or a future report generator:

```python
from modules.insights_engine import (
    generate_player_insights,
    generate_slate_insights,
    generate_similarity_insights,
    generate_correlation_insights,
)

insights = generate_player_insights(player_result, matchup_result)
```

Player Projections, Player Analysis, Daily Slate, and the Visual Analytics Lab display these notes as compact bullet lists. Missing inputs are skipped instead of replaced with invented claims.

The shared Streamlit UI also provides `confidence_ring()`, `reliability_bar()`, and `confidence_card()`. Scores from 80–100 are green, 50–79 are yellow, and 0–49 are red. Player Projections, Model Drivers, Daily Slate, and slate edge views use the same scale and styling.

Player identity rows retain 60-pixel headshots and 40-pixel logos while adding a team-color border, soft glow, drop shadow, and lightweight hover highlight. The global theme adds consistent spacing, gradient section headings, typography, Plotly styling, and subtle native transitions to every page in `app/pages/`; it does not require an animation framework.
## Complete NFL analytics engine

The NFL workspace now mirrors the NBA architecture while keeping provider work centralized and cached:

- `modules/nfl_stats.py` loads canonical nflverse player data, applies shared normalization, and calculates the 60% league / 40% position percentile blend. It covers QB, RB, WR/TE, and defensive-unit metrics and exposes an explicit last-known fallback when nflverse is unavailable.
- `modules/nfl_player.py` provides clean lookup and profile contracts.
- `modules/nfl_analysis.py` adds role and volume trends, efficiency, explosive-play probability, red-zone involvement, opponent difficulty, pace, weather, offensive-line context, identity summaries, tier labels, and volatility.
- `modules/nfl_projections.py` projects carries, targets, touches, rushing and receiving yards, passing yards, touchdown probability, fantasy points, contextual adjustments, and confidence intervals.
- `modules/nfl_graph_lab.py` supplies the QB Passing Map, WR/TE Route Tree, RB Usage Funnel, Defensive Pressure Map, and Pace & Play Volume visual.
- `modules/nfl_slate.py` ranks weekly environments using pace, matchup difficulty, weather, trench mismatch, explosive plays, red-zone funnels, and fantasy scoring environment.

The **NFL Tools** navigation group contains Team Analysis, Player Analysis, Graph Lab, Projections, and Slate. NFL visuals use team colors, transparent 40-60% layers, hover details, restrained glow, and Raw/Adjusted plus League/Position controls without an animation dependency.

```python
from modules.nfl_analysis import analyze_nfl_player
from modules.nfl_projections import project_nfl_player
from modules.nfl_slate import analyze_nfl_slate

profile = analyze_nfl_player("Josh Allen", "KC", season=2025)
projection = project_nfl_player("Saquon Barkley", "DAL", season=2025)
slate = analyze_nfl_slate(season=2025)
```

Provider efficiency rules are deliberate: one season table is cached per process, all pages reuse the same canonical frame, and the fallback snapshot clearly identifies itself in `warnings` and `source`. The projections are explainable research estimates rather than betting or injury guarantees.
## Design and limitations

All domain functions return plain dictionaries, and presentation code stays under `app/`. This lets future services replace individual data adapters or models without rewriting the interface. Natural extensions include calibrated probability models, position-defense datasets, authenticated licensed feeds, databases, APIs, dashboards, sentiment analysis, and portfolio/risk tooling.

NBA.com and public sportsbook sources may rate-limit, block, change shape, or provide no lines in a location. Results should be checked against official live sources before any decision. The project deliberately contains no alerts.
