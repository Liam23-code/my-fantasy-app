"""NFL moneyline / spread analysis for one game, built on the existing betting engine.

This is the pure-logic layer behind the MoneyLines page (no Streamlit, no UI).
For a selected game it produces team strength, offensive / defensive
efficiency, a quarterback adjustment, a pace and scoring projection, win /
spread / moneyline probabilities, a written reasoning section, and a price
comparison against a sportsbook moneyline the user pastes.

How the numbers are built (every step is in points, then converted once)
------------------------------------------------------------------------
1. **Ratings** (:func:`load_team_ratings`).  Team efficiency and scoring come
   from nflverse play-by-play + the schedule for the *live* season and the
   *prior* season, blended per team by sample size: ``w = n / (n + K)`` with
   ``K = BLEND_K_GAMES`` (6).  The prior season is regressed toward the
   league average by ``PRIOR_CARRYOVER`` before it is blended (rosters turn
   over; year-over-year team EPA correlates only ~0.4).  A team with no data in
   either season gets exactly the league average -- never NaN.
2. **Base points.**  :func:`betting.team_model.project_game` (the engine's
   offense-vs-opposing-defense average) is run on the blended points-for /
   points-against averages.  Nothing here re-derives it.
3. **Adjustments, all in points.**
   * *Efficiency blend*: scoring is noisy (turnover and special-teams luck,
     garbage time), EPA per play is steadier.  A second expected-points
     estimate is built from EPA per play x projected plays, and the final
     expected points are ``(1 - EFFICIENCY_WEIGHT) * points-based +
     EFFICIENCY_WEIGHT * efficiency-based``.  This is also where **pace**
     enters the total: more projected plays means more scoring chances.
   * *Home field*: ``HOME_FIELD_POINTS`` (2.0) on the spread, except at
     neutral venues (nflverse ``location == "Neutral"``: London, Madrid,
     Melbourne, Mexico City...), where the designated "home" team gets 0.
   * *Quarterback*: the named starter's value over an average starter, *minus*
     the value already embedded in the team rating (so a team's usual starter
     is not double counted); a backup or unproven starter costs points.
   * *Situation*: a small rest-day differential, and a wind discount on the
     total for outdoor games.
4. **Probabilities.**  Win probability comes ONLY from
   :func:`betting.moneyline_model.win_probability_from_spread` (normal
   approximation) and fair prices ONLY from
   :func:`betting.odds_math.fair_price_from_probability`.  The spread-to-win
   conversion uses ``WIN_PROB_STDEV`` (11.5, backtested -- see the constant);
   spread-cover probability uses the engine's ``MARGIN_STDEV`` (13.5), the
   spread of the final margin around a line.

Sign conventions
----------------
* ``spread`` / ``model_spread_home`` / ``market_spread_home`` follow nflverse:
  **positive means the HOME team is favored** by that many points (a home
  favorite of -7 in sportsbook notation is ``+7.0`` here).
* ``edge_*_vs_market`` and ``value_gap`` are probability points: model minus
  market; positive means the model rates the side likelier than the price.

Honest limits (shown to the user, not hidden)
---------------------------------------------
* Only one player is modelled: the quarterback.  Other injuries, depth-chart
  changes, coaching, travel, and motivation are NOT in the numbers.
* EPA ratings are not opponent-adjusted; early in the season they lean on the
  prior year, which misses roster moves.
* Pace is the clock time between offensive snaps; it ignores game script.
* The margin model is a normal approximation with no push mass: whole-number
  spreads (3, 7) really carry extra push probability, and half-point lines
  (-3.5) are modelled exactly.  Real margins also cluster at 3 and 7.
* nflverse market columns are (typically closing) consensus lines, not the
  price at your sportsbook.  Compare your own number with
  :func:`evaluate_moneyline_price`.
* A model that disagrees with the market by a large amount is far more likely
  to be missing information than to have found an edge; the confidence score
  is cut when that happens.

Nothing here places a bet, and a network or data failure never raises into
the caller (it returns neutral numbers plus ``warnings``) -- except
:func:`load_schedule`, which raises :class:`ScheduleUnavailable` because there
is nothing to show without a schedule.
"""
from __future__ import annotations

import copy
import math
import re
import threading
import time
from datetime import date, datetime
from typing import Any, Callable, Mapping

import pandas as pd

from betting.cache_utils import ttl_cache
from betting.moneyline_model import NFL_MARGIN_STDEV, win_probability_from_spread
from betting.odds_math import fair_price_from_probability, hold as market_hold
from betting.team_model import project_game
from modules.nfl import NFL_TEAMS, _scrimmage_plays, build_team_season_stats, fetch_play_by_play
from modules.odds_input import evaluate_price, no_vig_probabilities

# --------------------------------------------------------------------------
# Constants (every one is documented; tests pin their meaning)
# --------------------------------------------------------------------------

#: Stdev of an NFL final margin, re-exported from the engine so the page and
#: the probabilities can never disagree.
MARGIN_STDEV: float = NFL_MARGIN_STDEV

#: Stdev used ONLY to turn a point spread into a WIN probability (moneyline).
#: The engine default (13.5) is the right spread for the margin around a line
#: (cover probability) but prices favorites too low when read as a win rate:
#: backtest on 3,649 regular-season games, 2012-2025, closing spread vs the
#: realized winner -- log-loss 0.6110 at 13.5 versus 0.6085 at 11.1 (the
#: minimum; 11-12 is flat), and favorites of 6-8.5 points actually won 74.5% of
#: the time versus 69.4% from the normal curve at 13.5.  A de-vigged moneyline
#: market sits right on the calibrated curve (log-loss 0.6094), so converting
#: our fair spread with the same curve keeps the model-vs-moneyline comparison
#: apples to apples instead of showing every big favorite as "overpriced".
#: Passed to the engine's own win_probability_from_spread(stdev=...).
WIN_PROB_STDEV: float = 11.5

#: Home-field advantage in points of spread (recent NFL seasons: ~1.5-2.5).
HOME_FIELD_POINTS: float = 2.0

#: ``K`` in ``w_current = n / (n + K)``: after 6 games the live season counts
#: as much as the (regressed) prior season.
BLEND_K_GAMES: float = 6.0

#: Fraction of last season's deviation-from-average that is carried into this
#: season's rating before blending.  Measured year-over-year team correlations
#: (nflverse, regular season, 32 teams, so about 0.17 standard error per
#: pair): offensive EPA/play 0.41 / 0.46 / 0.41 and points scored 0.46 / 0.39 /
#: 0.29 for 2022->23 / 2023->24 / 2024->25, but DEFENSIVE EPA/play only 0.08 /
#: -0.11 / 0.37 and points allowed 0.38 / -0.03 / 0.45.  0.4 is therefore the
#: right order of magnitude for offense and, if anything, generous for defense;
#: it is one shared constant, tuned on 2025 and checked out of sample on 2024.
#: Re-run on 2023-2025 the backtest correlation with the closing line is flat
#: from 0.3 to 0.5 (e.g. 0.909 / 0.850 / 0.910 at 0.4), so nothing hinges on it.
PRIOR_CARRYOVER: float = 0.4

#: Weight on the EPA-implied expected points versus the points-based
#: (project_game) expected points.  In the 2025 backtest the points-based side
#: predicted results slightly better and the EPA side matched the market more
#: closely, so the blend leans a little toward points.
EFFICIENCY_WEIGHT: float = 0.4

#: ``strength_rating = 50 + STRENGTH_POINTS_SCALE * (neutral-site margin vs an
#: average team, in points)``.  One rating point is therefore 0.2 points of
#: spread, and a +6 point team rates 80.
STRENGTH_POINTS_SCALE: float = 5.0

#: Rest-day edge: points of spread per extra day of rest, capped at +/- 4 days.
REST_POINTS_PER_DAY: float = 0.15
REST_DAYS_CAP: float = 4.0

#: Outdoor wind discount on the total: ``0.2`` points per mph above 12 mph,
#: capped at 3 points.  Roofed stadiums ("dome", "closed") are exempt.
WIND_THRESHOLD_MPH: float = 12.0
WIND_TOTAL_PER_MPH: float = 0.2
WIND_TOTAL_CAP: float = 3.0
_ROOFED = frozenset({"dome", "closed"})

#: A matchup projected this many plays above / below the league's typical
#: game is labelled "Fast" / "Slow" (about 0.7 standard deviations of the
#: matchup pace distribution).
PACE_LABEL_PLAYS: float = 2.5

#: Quarterback valuation.  ``QB_POINTS_PER_EPA`` converts a QB's EPA per
#: dropback edge into points per game: ~38 dropbacks x ~0.58 credit (the rest
#: belongs to the line, receivers, and play-calling).  Removing the QB term
#: entirely cost 0.06 of correlation with the market in the 2025 backtest, and
#: fit improved monotonically up to ~30, so 22 is the cautious middle.
QB_POINTS_PER_EPA: float = 22.0
#: Fallback scale when only passer rating is available (points per rating pt).
QB_POINTS_PER_RATING: float = 0.10
#: Largest QB value / adjustment in points, either direction.
QB_IMPACT_CLAMP: float = 6.0
#: Prior-season passing counts are weighted by this when pooled with the live
#: season's (recency).
QB_PRIOR_WEIGHT: float = 0.75
#: Dropbacks of evidence that count as much as the prior (league / replacement
#: level) when stabilising a QB's EPA per dropback.
QB_SHRINK_DROPBACKS: float = 400.0
QB_SHRINK_ATTEMPTS: float = 300.0
#: Pooled (weighted) attempts that make a QB a "starter" rather than a backup.
QB_STARTER_MIN_ATTEMPTS: float = 150.0
#: A team's top live-season passer with at least this many attempts counts as
#: its starter even without a long track record (rookies, new starters).
QB_LIVE_STARTER_ATTEMPTS: float = 60.0
#: Used only when too few backups exist in the data to measure replacement
#: level: EPA per dropback below the average starter (nflverse 2017-2025:
#: low-volume veterans -0.129 vs +0.078 for veteran starters, a 0.21 gap).
QB_REPLACEMENT_DELTA_DEFAULT: float = 0.20
#: A listed starter with no NFL passing history in the data is overwhelmingly a
#: rookie or new franchise quarterback, not a street backup.  Measured on
#: nflverse 2017-2025: 40 rookie starters (>= 200 dropbacks) averaged -0.061 EPA
#: per dropback against +0.078 for 272 veteran-starter seasons (about -0.14),
#: while low-volume veterans (backups) averaged -0.129.  Priced 0.12 below the
#: average starter, about -2.6 points: harsher than an average starter, milder
#: than a measured backup.
QB_UNPROVEN_DELTA_EPA: float = 0.12
#: A passer rating built on fewer career attempts than this is noise (0 for 0-of-3),
#: so the page shows the replacement-level rating instead and flags it as an estimate.
QB_MIN_RATING_ATTEMPTS: int = 30

#: Neutral league values used when no data is available at all.
_DEFAULT_LEAGUE = {
    "points_pg": 22.5,
    "off_epa": 0.0,
    "def_epa": 0.0,
    "pace_spp": 30.4,
    "total_plays_pg": 121.0,
    "offensive_efficiency": 50.0,
    "defensive_efficiency": 50.0,
}
_DEFAULT_QB_BASELINE = {
    "passer_rating": 90.0,
    "epa_per_dropback": 0.03,
    "replacement_epa": 0.03 - QB_REPLACEMENT_DELTA_DEFAULT,
    "replacement_passer_rating": 74.0,
    "unproven_epa": 0.03 - QB_UNPROVEN_DELTA_EPA,
    "unproven_passer_rating": 82.0,
    "starters_counted": 0,
    "backups_counted": 0,
    "source": "defaults",
}

_CACHE_SECONDS = 3600
_MEMO_OK_SECONDS = 900
_MEMO_DEGRADED_SECONDS = 60
_LEAGUE_AVG = "LEAGUE_AVG"  # synthetic opponent used for neutral-site team margins


class ScheduleUnavailable(RuntimeError):
    """Raised by :func:`load_schedule` when no regular-season schedule can be loaded."""


# --------------------------------------------------------------------------
# Small numeric / text helpers
# --------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    """A finite float, or ``None`` for None / NaN / inf / non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite(value: Any, default: float) -> float:
    number = _num(value)
    return default if number is None else number


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _truthy(value: Any) -> bool:
    """Strict flag reading: ``None`` / NaN / ``""`` / ``"false"`` / ``"0"`` are False.

    A bare ``bool(float("nan"))`` is True, which would turn a missing
    ``div_game`` / ``played`` / ``neutral_site`` value pulled out of a
    DataFrame row into a confident claim ("a divisional game").
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):  # array-likes: fall through to ordinary truthiness
        pass
    return bool(value)


def _signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}"


def _spread_text(home: str, away: str, spread_home: float) -> str:
    """``SEA -3.5`` style text from the nflverse positive-means-home convention."""
    if spread_home > 0:
        return f"{home} -{spread_home:g}"
    if spread_home < 0:
        return f"{away} -{abs(spread_home):g}"
    return "pick'em"


_ALIASES: dict[str, str] = {}
for _code, (_full, _nick, _alts) in NFL_TEAMS.items():
    _ALIASES[_code] = _code
    for _alt in _alts:
        _ALIASES[_alt.upper()] = _code
_ALIASES.update({"LAR": "LA", "STL": "LA", "SD": "LAC", "SDG": "LAC", "OAK": "LV", "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI"})


def normalize_team(code: Any) -> str:
    """Map any common team spelling (``LAR``, ``WSH``, ``JAC``...) to the nflverse code (``LA``, ``WAS``, ``JAX``).

    ``None`` / NaN / blank give ``""`` (never the string ``"NAN"``), so a row
    with a missing team is dropped instead of becoming a phantom team.
    """
    key = _text_or_empty(code).upper()
    return _ALIASES.get(key, key)


def team_name(code: str) -> str:
    """Full team name (``"Seattle Seahawks"``), or the code itself when unknown."""
    entry = NFL_TEAMS.get(normalize_team(code))
    return entry[0] if entry else str(code)


# --------------------------------------------------------------------------
# Season and schedule
# --------------------------------------------------------------------------

def current_season(today: date | datetime | None = None) -> int:
    """The NFL season whose schedule is live on ``today``.

    March-December -> that calendar year; January-February -> the prior year
    (the playoffs and Super Bowl still belong to last season).  Pure.
    """
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


SCHEDULE_COLUMNS = [
    "game_id", "season", "week", "gameday", "gametime", "weekday", "kickoff_label",
    "away_team", "home_team", "away_score", "home_score", "played",
    "market_spread_home", "market_total", "market_home_ml", "market_away_ml",
    "home_qb", "away_qb", "home_qb_id", "away_qb_id",
    "roof", "temp", "wind", "div_game", "home_rest", "away_rest",
    "neutral_site",
]


def _iso_day(value: Any) -> str:
    if value is None:
        return ""
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return ""
    return "" if pd.isna(stamp) else stamp.strftime("%Y-%m-%d")


def _hhmm(value: Any) -> str:
    """``"13:00"`` / ``"13:00:00"`` / ``datetime.time`` -> ``"13:00"``; anything else -> ``""``."""
    if value is None:
        return ""
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return f"{int(value.hour):02d}:{int(value.minute):02d}"
    match = re.match(r"^\s*(\d{1,2}):(\d{2})", str(value))
    return f"{int(match.group(1)):02d}:{match.group(2)}" if match else ""


def _kickoff_label(gameday: str, gametime: str) -> str:
    """``"Sun Oct 11 - 1:00 PM ET"`` (nflverse kickoff times are US Eastern)."""
    if not gameday:
        return "TBD"
    stamp = pd.Timestamp(gameday)
    day = f"{stamp:%a} {stamp:%b} {stamp.day}"
    if not gametime:
        return day
    hour, minute = int(gametime[:2]), gametime[3:5]
    suffix = "AM" if hour < 12 else "PM"
    return f"{day} - {hour % 12 or 12}:{minute} {suffix} ET"


def _text_or_empty(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>", "nat"} else text


def normalize_schedule(raw: pd.DataFrame | None, season: int | None = None) -> pd.DataFrame:
    """Turn a raw nflverse schedule frame into the stable ``SCHEDULE_COLUMNS`` frame.

    Pure and offline.  Keeps regular-season rows only, normalises team codes,
    and makes ``played`` true only when BOTH scores are real numbers --
    nflverse (via pandas) stores unplayed scores as NaN, which a naive
    ``is not None`` test would treat as played.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    frame = raw.copy()
    if "game_type" in frame.columns:
        frame = frame[frame["game_type"].astype(str).str.upper() == "REG"]
    if frame.empty:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)

    def col(name: str, default: Any = None) -> pd.Series:
        return frame[name] if name in frame.columns else pd.Series([default] * len(frame), index=frame.index)

    out = pd.DataFrame(index=frame.index)
    out["week"] = pd.to_numeric(col("week"), errors="coerce")
    out["away_team"] = col("away_team").map(normalize_team)
    out["home_team"] = col("home_team").map(normalize_team)
    valid = out["week"].notna() & (out["away_team"] != "") & (out["home_team"] != "")
    out, frame = out[valid].copy(), frame[valid]  # ``col`` reads the rebound ``frame``
    if out.empty:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)

    out["week"] = out["week"].astype(int)
    fallback_season = season if season is not None else _num(col("season").iloc[0]) if "season" in frame.columns else None
    out["season"] = pd.to_numeric(col("season"), errors="coerce").fillna(fallback_season if fallback_season is not None else 0).astype(int)
    ids = col("game_id").map(_text_or_empty)
    synthetic = out.apply(lambda r: f"{int(r['season'])}_{int(r['week']):02d}_{r['away_team']}_{r['home_team']}", axis=1)
    out["game_id"] = ids.where(ids != "", synthetic)
    out["gameday"] = col("gameday").map(_iso_day)
    out["gametime"] = col("gametime").map(_hhmm)
    out["weekday"] = col("weekday").map(_text_or_empty)
    out["kickoff_label"] = [_kickoff_label(d, t) for d, t in zip(out["gameday"], out["gametime"])]
    out["away_score"] = pd.to_numeric(col("away_score"), errors="coerce")
    out["home_score"] = pd.to_numeric(col("home_score"), errors="coerce")
    out["played"] = out["home_score"].notna() & out["away_score"].notna()
    # nflverse: spread_line > 0 means the HOME team is favored by that many points.
    out["market_spread_home"] = pd.to_numeric(col("spread_line"), errors="coerce")
    out["market_total"] = pd.to_numeric(col("total_line"), errors="coerce")
    out["market_home_ml"] = pd.to_numeric(col("home_moneyline"), errors="coerce")
    out["market_away_ml"] = pd.to_numeric(col("away_moneyline"), errors="coerce")
    out["home_qb"] = col("home_qb_name").map(_text_or_empty)
    out["away_qb"] = col("away_qb_name").map(_text_or_empty)
    out["home_qb_id"] = col("home_qb_id").map(_text_or_empty)
    out["away_qb_id"] = col("away_qb_id").map(_text_or_empty)
    out["roof"] = col("roof").map(_text_or_empty).str.lower()
    out["temp"] = pd.to_numeric(col("temp"), errors="coerce")
    out["wind"] = pd.to_numeric(col("wind"), errors="coerce")
    out["div_game"] = pd.to_numeric(col("div_game"), errors="coerce").fillna(0).astype(int).astype(bool)
    out["home_rest"] = pd.to_numeric(col("home_rest"), errors="coerce")
    out["away_rest"] = pd.to_numeric(col("away_rest"), errors="coerce")
    # nflverse ``location`` is "Home" or "Neutral" (London, Madrid, Melbourne, Mexico City...).  At a neutral
    # site the "home" team is only the designated home team, and the market line carries no home-field edge.
    out["neutral_site"] = col("location").map(_text_or_empty).str.lower() == "neutral"
    out = out[SCHEDULE_COLUMNS].sort_values(["gameday", "gametime", "week", "game_id"], kind="stable")
    return out.reset_index(drop=True)


def _download_schedule(season: int) -> pd.DataFrame:
    """Network seam (monkeypatched in tests): the raw nflverse schedule as pandas."""
    import nflreadpy

    return nflreadpy.load_schedules(seasons=[season]).to_pandas()


@ttl_cache(_CACHE_SECONDS)
def _cached_schedule(season: int) -> pd.DataFrame:
    try:
        raw = _download_schedule(season)
    except Exception as error:  # network, missing dependency, parse failure
        raise ScheduleUnavailable(
            f"The {season} NFL schedule could not be loaded ({type(error).__name__}: {error}). "
            "Check your connection and try again."
        ) from error
    try:
        frame = normalize_schedule(raw, season)
    except Exception as error:  # schema drift upstream must surface as the one documented exception
        raise ScheduleUnavailable(
            f"The {season} NFL schedule was downloaded but could not be read ({type(error).__name__}: {error})."
        ) from error
    if frame.empty:
        raise ScheduleUnavailable(f"No regular-season games were found in the {season} NFL schedule.")
    return frame


def load_schedule(season: int) -> pd.DataFrame:
    """Regular-season schedule for ``season`` with the stable ``SCHEDULE_COLUMNS``.

    Unplayed games have NaN scores and ``played == False``; market columns are
    NaN when no line is posted yet.  ``market_spread_home`` is positive when
    the home team is favored.  The result is cached for an hour (a copy is
    returned, so mutating it is safe).  Raises :class:`ScheduleUnavailable`
    with a user-facing message when it cannot be loaded.
    """
    return _cached_schedule(int(season)).copy()


def weeks_in_schedule(schedule: pd.DataFrame) -> list[int]:
    """Sorted distinct regular-season week numbers (empty list for an empty schedule)."""
    if schedule is None or len(schedule) == 0 or "week" not in schedule.columns:
        return []
    return sorted({int(week) for week in schedule["week"].dropna()})


def default_week(schedule: pd.DataFrame) -> int:
    """The week to open on: the first week that still has an unplayed game.

    All weeks played -> the last week; nothing played yet -> week 1 (which is
    also the first week with an unplayed game); empty schedule -> 1.
    """
    weeks = weeks_in_schedule(schedule)
    if not weeks:
        return 1
    unplayed = schedule[~schedule["played"].astype(bool)]
    if unplayed.empty:
        return weeks[-1]
    return int(unplayed["week"].min())


def _clean(value: Any) -> Any:
    """NaN / NaT / pandas NA -> ``None``; numpy scalars -> Python scalars."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _game_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    played = _truthy(row.get("played"))
    away_score, home_score = _num(row.get("away_score")), _num(row.get("home_score"))
    status = f"Final {int(away_score)}-{int(home_score)}" if played and away_score is not None and home_score is not None else "Upcoming"
    away, home = str(row["away_team"]), str(row["home_team"])
    return {
        "game_id": str(row["game_id"]),
        "week": int(row["week"]),
        "season": int(row["season"]) if _num(row.get("season")) is not None else None,
        "away_team": away,
        "home_team": home,
        "label": f"{away} @ {home}",
        "kickoff_label": str(row.get("kickoff_label") or "TBD"),
        "gameday": _clean(row.get("gameday")) or None,
        "gametime": _clean(row.get("gametime")) or None,
        "weekday": _clean(row.get("weekday")) or None,
        "played": played,
        "status": status,  # "Final AWAY-HOME" mirrors the "AWAY @ HOME" label order
        "away_score": int(away_score) if played and away_score is not None else None,
        "home_score": int(home_score) if played and home_score is not None else None,
        "market_spread_home": _num(row.get("market_spread_home")),
        "market_total": _num(row.get("market_total")),
        "market_home_ml": _num(row.get("market_home_ml")),
        "market_away_ml": _num(row.get("market_away_ml")),
        "home_qb": _clean(row.get("home_qb")) or None,
        "away_qb": _clean(row.get("away_qb")) or None,
        "home_qb_id": _clean(row.get("home_qb_id")) or None,
        "away_qb_id": _clean(row.get("away_qb_id")) or None,
        "roof": _clean(row.get("roof")) or None,
        "temp": _num(row.get("temp")),
        "wind": _num(row.get("wind")),
        "div_game": _truthy(row.get("div_game")),
        "neutral_site": _truthy(row.get("neutral_site")),
        "home_rest": _num(row.get("home_rest")),
        "away_rest": _num(row.get("away_rest")),
    }


def _qb_history(schedule: pd.DataFrame) -> dict[str, list[tuple[str, str, str, str]]]:
    """``{team: [(gameday, gametime, qb_name, qb_id), ...]}`` ascending, for every game that names a starter."""
    history: dict[str, list[tuple[str, str, str, str]]] = {}
    for row in schedule.to_dict("records"):
        for side in ("home", "away"):
            name = _clean(row.get(f"{side}_qb"))
            if name:
                history.setdefault(str(row[f"{side}_team"]), []).append(
                    (str(row.get("gameday") or ""), str(row.get("gametime") or ""), str(name), str(_clean(row.get(f"{side}_qb_id")) or ""))
                )
    for entries in history.values():
        entries.sort()
    return history


def games_for_week(schedule: pd.DataFrame, week: int) -> list[dict[str, Any]]:
    """Every game of ``week`` in kickoff order, as dicts ready for :func:`analyze_game`.

    Keys: ``game_id, week, season, away_team, home_team, label ("AWAY @ HOME"),
    kickoff_label, gameday, gametime, weekday, played, status ("Upcoming" or
    "Final 24-17" in away-home order), away_score, home_score (None until
    played), market_spread_home, market_total, market_home_ml, market_away_ml
    (None when no line), home_qb, away_qb, home_qb_id, away_qb_id,
    home_qb_source, away_qb_source, roof, temp, wind, div_game, neutral_site
    (True for international / neutral venues, where "home" is only the
    designated home team), home_rest, away_rest``.  Missing values are
    ``None``, never NaN.

    nflverse only names starters close to kickoff.  When a game lists none, the
    team's starter in its most recent earlier game is filled in and
    ``*_qb_source`` is ``"last_game"`` (``"listed"`` when the schedule names the
    QB, ``"none"`` when no earlier starter is known either).
    """
    if schedule is None or len(schedule) == 0:
        return []
    subset = schedule[schedule["week"] == int(week)]
    subset = subset.sort_values(["gameday", "gametime", "game_id"], kind="stable")
    history = _qb_history(schedule)
    games = [_game_dict(row) for row in subset.to_dict("records")]
    for game in games:
        kickoff = (game["gameday"] or "", game["gametime"] or "")
        for side in ("home", "away"):
            if game[f"{side}_qb"]:
                game[f"{side}_qb_source"] = "listed"
                continue
            earlier = [entry for entry in history.get(game[f"{side}_team"], []) if (entry[0], entry[1]) < kickoff]
            if earlier:
                _, _, name, pid = earlier[-1]
                game[f"{side}_qb"], game[f"{side}_qb_id"], game[f"{side}_qb_source"] = name, pid or None, "last_game"
            else:
                game[f"{side}_qb_source"] = "none"
    return games


# --------------------------------------------------------------------------
# Fail-soft loading plumbing
# --------------------------------------------------------------------------

def _download_pbp(season: int) -> pd.DataFrame:
    """Network seam (monkeypatched in tests): one season of play-by-play."""
    return fetch_play_by_play(season)


def _download_player_stats(season: int) -> pd.DataFrame:
    """Network seam (monkeypatched in tests): weekly player stats as pandas."""
    import nflreadpy

    return nflreadpy.load_player_stats(seasons=[season]).to_pandas()


@ttl_cache(_CACHE_SECONDS)
def _cached_pbp(season: int) -> pd.DataFrame:
    return _download_pbp(season)


@ttl_cache(_CACHE_SECONDS)
def _cached_player_stats(season: int) -> pd.DataFrame:
    return _download_player_stats(season)


_MEMO: dict[tuple[Any, ...], tuple[float, Any]] = {}
_MEMO_LOCK = threading.Lock()


def _memo_get(key: tuple[Any, ...]) -> Any | None:
    with _MEMO_LOCK:
        hit = _MEMO.get(key)
        if hit is not None and time.monotonic() < hit[0]:
            return copy.deepcopy(hit[1])
    return None


def _memo_put(key: tuple[Any, ...], value: Any, ttl: float) -> None:
    with _MEMO_LOCK:
        _MEMO[key] = (time.monotonic() + ttl, copy.deepcopy(value))


def clear_caches() -> None:
    """Drop every cache in this module (schedule, play-by-play, player stats, built ratings)."""
    for cached in (_cached_schedule, _cached_pbp, _cached_player_stats):
        cached.cache_clear()  # type: ignore[attr-defined]
    with _MEMO_LOCK:
        _MEMO.clear()


def _try(loader: Callable[..., Any], *args: Any) -> tuple[Any | None, str | None]:
    """Run a loader; return ``(result, None)`` or ``(None, short error text)``."""
    try:
        return loader(*args), None
    except Exception as error:  # fail-soft by design: a data failure must not reach the UI
        return None, f"{type(error).__name__}: {error}"


# --------------------------------------------------------------------------
# Team ratings
# --------------------------------------------------------------------------

def _points_from_schedule(schedule: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    """``{team: {pf, pa, n}}`` from played games only (NaN-safe via the ``played`` flag)."""
    if schedule is None or len(schedule) == 0:
        return {}
    played = schedule[schedule["played"].astype(bool)]
    if played.empty:
        return {}
    home = played[["home_team", "home_score", "away_score"]].set_axis(["team", "pf", "pa"], axis=1)
    away = played[["away_team", "away_score", "home_score"]].set_axis(["team", "pf", "pa"], axis=1)
    grouped = pd.concat([home, away]).groupby("team").agg(pf=("pf", "mean"), pa=("pa", "mean"), n=("pf", "size"))
    return {str(team): {"pf": float(r.pf), "pa": float(r.pa), "n": int(r.n)} for team, r in grouped.iterrows()}


def _points_from_pbp(pbp: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    """Fallback scoring from play-by-play when the schedule is unavailable.

    Final score per team = its largest ``posteam_score_post``; this can miss a
    defensive touchdown, so it is only used when the schedule failed.
    """
    needed = {"game_id", "posteam", "posteam_score_post"}
    if pbp is None or len(pbp) == 0 or not needed.issubset(pbp.columns):
        return {}
    plays = pbp[pbp["season_type"] == "REG"] if "season_type" in pbp.columns else pbp
    scores = plays.dropna(subset=["posteam", "posteam_score_post"]).groupby(["game_id", "posteam"])["posteam_score_post"].max().reset_index()
    paired = scores.merge(scores, on="game_id", suffixes=("", "_opp"))
    paired = paired[paired["posteam"] != paired["posteam_opp"]]
    if paired.empty:
        return {}
    grouped = paired.groupby("posteam").agg(pf=("posteam_score_post", "mean"), pa=("posteam_score_post_opp", "mean"), n=("game_id", "nunique"))
    return {str(team): {"pf": float(r.pf), "pa": float(r.pa), "n": int(r.n)} for team, r in grouped.iterrows()}


def _season_snapshot(pbp: pd.DataFrame | None, schedule: pd.DataFrame | None) -> dict[str, Any] | None:
    """One season's raw per-team numbers (no blending).  ``None`` when nothing usable.

    ``teams[T]`` may lack efficiency keys (no play-by-play) or scoring keys
    (no schedule); :func:`blend_ratings` treats missing keys as "no data".
    """
    teams: dict[str, dict[str, float]] = {}
    league_plays: float | None = None
    scoring_source = "schedule"
    if pbp is not None and len(pbp) > 0:
        try:
            table = build_team_season_stats(pbp)
        except (ValueError, KeyError):
            table = None
        if table is not None and not table.empty:
            for row in table.to_dict("records"):
                code = normalize_team(row["team"])
                teams[code] = {
                    "games": float(row["games"]),
                    "off_score": float(row["offensive_efficiency_score"]),
                    "def_score": float(row["defensive_efficiency_score"]),
                    "off_epa": float(row["offensive_epa_per_play"]),
                    "def_epa": float(row["defensive_epa_per_play"]),
                }
                spp = float(row["pace_seconds_per_play"])
                if 15.0 <= spp <= 60.0:  # build_team_season_stats reports 0.0 when the game clock is missing
                    teams[code]["spp"] = spp
            plays = _scrimmage_plays(pbp)
            per_game = plays.groupby("game_id").size()
            if len(per_game):
                league_plays = float(per_game.mean())
    points = _points_from_schedule(schedule)
    if not points:
        points = _points_from_pbp(pbp)
        scoring_source = "play-by-play"
    for team, scored in points.items():
        entry = teams.setdefault(normalize_team(team), {})
        entry.update({"pf": scored["pf"], "pa": scored["pa"], "n_points": float(scored["n"])})
    if not teams:
        return None

    def mean(key: str) -> float | None:
        values = [t[key] for t in teams.values() if key in t and _num(t[key]) is not None]
        return sum(values) / len(values) if values else None

    games = [t["games"] for t in teams.values() if "games" in t]
    league = {
        "off_score": mean("off_score"),
        "def_score": mean("def_score"),
        "off_epa": mean("off_epa"),
        "def_epa": mean("def_epa"),
        "spp": mean("spp"),
        "pf": mean("pf"),
        "total_plays_pg": league_plays,
        "games_avg": (sum(games) / len(games)) if games else None,
        "points_games_avg": mean("n_points"),
    }
    return {"teams": teams, "league": league, "scoring_source": scoring_source}


def _blend_value(
    cur: float | None, n_cur: float, prior: float | None, league_cur: float | None, league_prior: float | None,
    league_ref: float, *, k: float, carry: float,
) -> tuple[float, float]:
    """Sample-size blend of one metric.  Returns ``(value, weight_on_current)``.

    ``value = league_ref + w * (cur - league_cur) + (1 - w) * carry * (prior - league_prior)``
    with ``w = n / (n + k)``.  A missing side contributes zero deviation, so a
    team with no data at all lands exactly on ``league_ref``.
    """
    use_cur = cur is not None and league_cur is not None and n_cur > 0
    weight = n_cur / (n_cur + k) if use_cur else 0.0
    cur_dev = (cur - league_cur) if use_cur else 0.0  # type: ignore[operator]
    prior_dev = (prior - league_prior) if (prior is not None and league_prior is not None) else 0.0
    return league_ref + weight * cur_dev + (1.0 - weight) * carry * prior_dev, weight


def _blend_league(cur: float | None, prior: float | None, games_avg: float | None, default: float, k: float) -> float:
    """League-level average: same sample weighting, no carry-over regression."""
    if cur is None and prior is None:
        return default
    if cur is None:
        return prior  # type: ignore[return-value]
    if prior is None:
        return cur
    weight = (games_avg or 0.0) / ((games_avg or 0.0) + k)
    return weight * cur + (1.0 - weight) * prior


def _project(
    home: Mapping[str, float], away: Mapping[str, float], league: Mapping[str, float], *,
    home_team: str, away_team: str, home_field: float = 0.0, qb_home: float = 0.0, qb_away: float = 0.0,
    rest_spread: float = 0.0, weather_total: float = 0.0,
) -> dict[str, Any]:
    """Expected points for one matchup from two rating rows (the model core).

    Used both for real games and, with a synthetic league-average opponent and
    ``home_field=0``, for the neutral-site margin behind ``strength_rating``.
    Returns the final expected points plus every intermediate so the page and
    the reasoning can show where the number came from.
    """
    averages = {
        home_team: {"points_scored_avg": home["points_for_pg"], "points_allowed_avg": home["points_against_pg"], "games_played": 1},
        away_team: {"points_scored_avg": away["points_for_pg"], "points_allowed_avg": away["points_against_pg"], "games_played": 1},
    }
    base = project_game(home_team, away_team, averages=averages, league_average_points=league["points_pg"])

    matchup_spp = max((home["pace_spp"] + away["pace_spp"]) / 2.0, 1.0)
    plays = _clamp(league["total_plays_pg"] * league["pace_spp"] / matchup_spp, 90.0, 160.0)
    points_per_play = league["points_pg"] / (league["total_plays_pg"] / 2.0)
    half_plays = plays / 2.0
    eff_home = max(0.0, half_plays * (points_per_play + (home["off_epa"] - league["off_epa"]) + (away["def_epa"] - league["def_epa"])))
    eff_away = max(0.0, half_plays * (points_per_play + (away["off_epa"] - league["off_epa"]) + (home["def_epa"] - league["def_epa"])))

    delta_home = EFFICIENCY_WEIGHT * (eff_home - base["home_expected_points"])
    delta_away = EFFICIENCY_WEIGHT * (eff_away - base["away_expected_points"])
    home_pts = base["home_expected_points"] + delta_home + home_field / 2.0 + qb_home + rest_spread / 2.0 + weather_total / 2.0
    away_pts = base["away_expected_points"] + delta_away - home_field / 2.0 + qb_away - rest_spread / 2.0 + weather_total / 2.0
    home_pts, away_pts = round(max(3.0, home_pts), 2), round(max(3.0, away_pts), 2)
    return {
        "home_expected_points": home_pts,
        "away_expected_points": away_pts,
        "model_spread_home": round(home_pts - away_pts, 2),
        "model_total": round(home_pts + away_pts, 2),
        "base": base,
        "efficiency_home": round(eff_home, 2),
        "efficiency_away": round(eff_away, 2),
        "plays_projection": plays,
        "matchup_spp": matchup_spp,
        "adjustments": {
            "base_spread": base["spread"],
            "base_total": base["total"],
            "efficiency_spread": round(delta_home - delta_away, 2),
            "efficiency_total": round(delta_home + delta_away, 2),
            "home_field": round(home_field, 2),
            "qb_spread": round(qb_home - qb_away, 2),
            "qb_total": round(qb_home + qb_away, 2),
            "rest_spread": round(rest_spread, 2),
            "weather_total": round(weather_total, 2),
        },
    }


def blend_ratings(
    current: dict[str, Any] | None, prior: dict[str, Any] | None, *, season: int, prior_season: int,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Blend two season snapshots into the ratings dict (pure; see :func:`load_team_ratings`)."""
    warnings = list(warnings or [])
    k, carry = BLEND_K_GAMES, PRIOR_CARRYOVER
    cur_teams = (current or {}).get("teams", {})
    prior_teams = (prior or {}).get("teams", {})
    cur_league = (current or {}).get("league", {})
    prior_league = (prior or {}).get("league", {})

    def league_ref(key: str, default_key: str | None = None, *, games_key: str = "games_avg") -> tuple[float | None, float | None, float]:
        cur_value, prior_value = _num(cur_league.get(key)), _num(prior_league.get(key))
        games_avg = _num(cur_league.get(games_key))
        ref = _blend_league(cur_value, prior_value, games_avg, _DEFAULT_LEAGUE[default_key or key], k)
        return cur_value, prior_value, ref

    lg = {}
    spec = {  # public name -> (snapshot key, default key, games key)
        "offensive_efficiency": ("off_score", "offensive_efficiency", "games_avg"),
        "defensive_efficiency": ("def_score", "defensive_efficiency", "games_avg"),
        "off_epa": ("off_epa", "off_epa", "games_avg"),
        "def_epa": ("def_epa", "def_epa", "games_avg"),
        "pace_spp": ("spp", "pace_spp", "games_avg"),
        "points_pg": ("pf", "points_pg", "points_games_avg"),
    }
    refs: dict[str, tuple[float | None, float | None, float]] = {}
    for public, (snap_key, default_key, games_key) in spec.items():
        refs[public] = league_ref(snap_key, default_key, games_key=games_key)
        lg[public] = refs[public][2]
    plays_cur, plays_prior = _num(cur_league.get("total_plays_pg")), _num(prior_league.get("total_plays_pg"))
    lg["total_plays_pg"] = _blend_league(plays_cur, plays_prior, _num(cur_league.get("games_avg")), _DEFAULT_LEAGUE["total_plays_pg"], k)

    codes = sorted(set(NFL_TEAMS) | set(cur_teams) | set(prior_teams))
    teams: dict[str, dict[str, Any]] = {}
    games_current: dict[str, int] = {}
    blend_weight: dict[str, float] = {}
    no_data: list[str] = []
    for code in codes:
        c, p = cur_teams.get(code, {}), prior_teams.get(code, {})
        n_eff, n_pts = _finite(c.get("games"), 0.0), _finite(c.get("n_points"), 0.0)
        p_eff, p_pts = _finite(p.get("games"), 0.0), _finite(p.get("n_points"), 0.0)

        def blended(public: str, cur_key: str, n_cur: float) -> tuple[float, float]:
            cur_league_value, prior_league_value, ref = refs[public]
            cur_value = _num(c.get(cur_key))
            prior_value = _num(p.get(cur_key))
            return _blend_value(cur_value, n_cur, prior_value, cur_league_value, prior_league_value, ref, k=k, carry=carry)

        off, w_eff = blended("offensive_efficiency", "off_score", n_eff)
        dfn, _ = blended("defensive_efficiency", "def_score", n_eff)
        off_epa, _ = blended("off_epa", "off_epa", n_eff)
        def_epa, _ = blended("def_epa", "def_epa", n_eff)
        spp, _ = blended("pace_spp", "spp", n_eff)
        pf, _ = blended("points_pg", "pf", n_pts)
        pa, _ = _blend_value(
            _num(c.get("pa")), n_pts, _num(p.get("pa")), _num(cur_league.get("pf")), _num(prior_league.get("pf")),
            refs["points_pg"][2], k=k, carry=carry,
        )
        has_cur = n_eff > 0 or n_pts > 0
        has_prior = p_eff > 0 or p_pts > 0
        quality = "full" if has_cur and has_prior else "current_only" if has_cur else "prior_only" if has_prior else "none"
        if quality == "none":
            no_data.append(code)
        off, dfn = _clamp(off, 0.0, 100.0), _clamp(dfn, 0.0, 100.0)
        teams[code] = {
            "team": code,
            "offensive_efficiency": round(off, 2),
            "defensive_efficiency": round(dfn, 2),
            "net_efficiency": round(off + dfn - 100.0, 2),
            "off_epa": round(_clamp(off_epa, -0.5, 0.5), 4),
            "def_epa": round(_clamp(def_epa, -0.5, 0.5), 4),
            "pace_spp": round(_clamp(spp, 22.0, 40.0), 2),
            "points_for_pg": round(_clamp(pf, 5.0, 45.0), 2),
            "points_against_pg": round(_clamp(pa, 5.0, 45.0), 2),
            "strength_rating": 50.0,  # filled below once the league row exists
            "games_current": int(max(n_eff, n_pts)),
            "games_prior": int(max(p_eff, p_pts)),
            "blend_weight_current": round(w_eff if n_eff > 0 else (n_pts / (n_pts + k) if n_pts > 0 else 0.0), 4),
            "data_quality": quality,
        }
        games_current[code] = teams[code]["games_current"]
        blend_weight[code] = teams[code]["blend_weight_current"]

    league_out = {
        "points_pg": round(lg["points_pg"], 3),
        "off_epa": round(lg["off_epa"], 4),
        "def_epa": round(lg["def_epa"], 4),
        "epa_per_play": round((lg["off_epa"] + lg["def_epa"]) / 2.0, 4),
        "pace_spp": round(lg["pace_spp"], 3),
        "total_plays_pg": round(lg["total_plays_pg"], 2),
        "offensive_efficiency": round(lg["offensive_efficiency"], 2),
        "defensive_efficiency": round(lg["defensive_efficiency"], 2),
        "games_current_avg": round(_finite(cur_league.get("games_avg"), 0.0), 2),
        "margin_stdev": MARGIN_STDEV,
    }
    avg_team = {
        "points_for_pg": league_out["points_pg"], "points_against_pg": league_out["points_pg"],
        "off_epa": league_out["off_epa"], "def_epa": league_out["def_epa"], "pace_spp": league_out["pace_spp"],
    }
    for code, row in teams.items():
        margin = _project(row, avg_team, league_out, home_team=code, away_team=_LEAGUE_AVG)["model_spread_home"]
        row["neutral_margin"] = round(margin, 2)
        row["strength_rating"] = round(_clamp(50.0 + STRENGTH_POINTS_SCALE * margin, 0.0, 100.0), 1)

    if current is None:
        warnings.append(f"No {season} data loaded yet, so ratings are the {prior_season} season regressed {int((1 - carry) * 100)}% toward average.")
    elif _finite(cur_league.get("games_avg"), 0.0) < 6:
        warnings.append(
            f"Only ~{_finite(cur_league.get('games_avg'), 0.0):.0f} games of {season} data per team so far; "
            f"ratings lean on {prior_season} (current-season weight {_finite(cur_league.get('games_avg'), 0.0) / (_finite(cur_league.get('games_avg'), 0.0) + k):.0%})."
        )
    if prior is None:
        warnings.append(f"No {prior_season} data loaded, so only {season} games inform the ratings.")
    if no_data and len(no_data) == len(codes):
        warnings.append("No team data could be loaded for either season; every rating is the league average.")
    elif no_data:
        listed = ", ".join(no_data[:8]) + (f" and {len(no_data) - 8} more" if len(no_data) > 8 else "")
        warnings.append(f"No data in either season for {listed}; those teams use league-average ratings.")
    sources = []
    if current is not None:
        sources.append(f"{season} play-by-play + schedule")
    if prior is not None:
        sources.append(f"{prior_season} play-by-play + schedule")
    return {
        "teams": teams,
        "league": league_out,
        "meta": {
            "season": season,
            "prior_season": prior_season,
            "games_current": games_current,
            "blend_weight_current": blend_weight,
            "warnings": list(dict.fromkeys(warnings)),
            "source": "nflverse " + " blended with ".join(sources) if sources else "league-average defaults (no data)",
            "k_games": k,
            "prior_carryover": carry,
        },
    }


def load_team_ratings(season: int, prior_season: int | None = None) -> dict[str, Any]:
    """Blended team ratings for ``season`` (live) and ``prior_season`` (default ``season - 1``).

    Returns ``{"teams": {TEAM: {...}}, "league": {...}, "meta": {...}}``:

    * ``teams[T]`` -- ``team, offensive_efficiency (0-100), defensive_efficiency
      (0-100, higher = better defence), net_efficiency (off + def - 100),
      off_epa, def_epa (EPA/play; for ``def_epa`` LOWER is better), pace_spp
      (seconds per offensive play), points_for_pg, points_against_pg,
      strength_rating (0-100, 50 = average; see ``STRENGTH_POINTS_SCALE``),
      neutral_margin (model points vs an average team at a neutral site),
      games_current, games_prior, blend_weight_current, data_quality
      ("full" | "current_only" | "prior_only" | "none")``.
    * ``league`` -- ``points_pg, off_epa, def_epa, epa_per_play, pace_spp,
      total_plays_pg, offensive_efficiency, defensive_efficiency,
      games_current_avg, margin_stdev``.
    * ``meta`` -- ``season, prior_season, games_current{T:n},
      blend_weight_current{T:w}, warnings[], source, k_games, prior_carryover``.

    Every team in ``modules.nfl.NFL_TEAMS`` is present and no value is ever
    NaN: missing data degrades to the league average and adds a warning.
    Points for/against come from the schedule with ``pd.notna`` (NOT
    ``betting.team_model.team_scoring_averages``, which returns NaN for the
    in-progress season); the play-by-play score is the fallback.
    """
    season = int(season)
    prior_season = int(prior_season) if prior_season is not None else season - 1
    key = ("ratings", season, prior_season)
    cached = _memo_get(key)
    if cached is not None:
        return cached

    warnings: list[str] = []
    degraded = False
    snapshots: dict[int, dict[str, Any] | None] = {}
    for year in (season, prior_season):
        pbp, pbp_error = _try(_cached_pbp, year)
        schedule, schedule_error = _try(load_schedule, year)
        # Even a download that "succeeds" can return a frame the builders cannot digest (upstream schema drift);
        # that must degrade to a warning, not an exception in the UI.
        snapshots[year], snapshot_error = _try(_season_snapshot, pbp, schedule)
        if snapshot_error:
            degraded = True
            snapshots[year], _ = _try(_season_snapshot, None, schedule)  # keep the schedule's scoring if the play-by-play is the problem
            kept = f"only {year} scoring is used" if snapshots[year] is not None else f"{year} is left out of the ratings"
            warnings.append(f"{year} data could not be processed ({snapshot_error.split(':')[0]}); {kept}.")
        if pbp_error:
            degraded = True  # retried after the short degraded TTL (often just "season has not started")
            if snapshots[year] is not None:  # otherwise blend_ratings already says there is no data for that season
                warnings.append(f"{year} play-by-play unavailable ({pbp_error.split(':')[0]}); only {year} scoring is used.")
            elif year == prior_season:
                warnings.append(f"{year} play-by-play and schedule unavailable ({pbp_error.split(':')[0]}).")
        if schedule_error:
            degraded = True
            if snapshots[year] is not None and not pbp_error:
                warnings.append(f"{year} schedule unavailable; {year} scoring averages come from play-by-play.")
    ratings, blend_error = _try(
        lambda: blend_ratings(snapshots[season], snapshots[prior_season], season=season, prior_season=prior_season, warnings=warnings)
    )
    if blend_error:  # last line of defence: neutral league-average ratings beat a traceback
        degraded = True
        ratings = blend_ratings(
            None, None, season=season, prior_season=prior_season,
            warnings=warnings + [f"Team ratings could not be built ({blend_error.split(':')[0]}); every rating is the league average."],
        )
    _memo_put(key, ratings, _MEMO_DEGRADED_SECONDS if degraded else _MEMO_OK_SECONDS)
    return ratings


# --------------------------------------------------------------------------
# Quarterbacks
# --------------------------------------------------------------------------

def _name_key(name: Any) -> str:
    """Case / punctuation / suffix-insensitive name key (``"Michael Penix Jr."`` -> ``"michael penix"``)."""
    text = re.sub(r"[^a-z\s]", "", str(name or "").lower())
    words = [w for w in text.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(words)


def _passer_rating(completions: float, attempts: float, yards: float, touchdowns: float, interceptions: float) -> float:
    """The NFL passer-rating formula (0-158.3) from counting stats."""
    if attempts <= 0:
        return 0.0
    a = _clamp((completions / attempts - 0.3) * 5.0, 0.0, 2.375)
    b = _clamp((yards / attempts - 3.0) * 0.25, 0.0, 2.375)
    c = _clamp(touchdowns / attempts * 20.0, 0.0, 2.375)
    d = _clamp(2.375 - interceptions / attempts * 25.0, 0.0, 2.375)
    return (a + b + c + d) / 6.0 * 100.0


def _qb_rows(stats: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Aggregate weekly player stats to one row per (QB, team) for a season."""
    if stats is None or len(stats) == 0:
        return []
    frame = stats
    if "position" in frame.columns:
        frame = frame[frame["position"] == "QB"]
    if "season_type" in frame.columns:
        frame = frame[frame["season_type"] == "REG"]
    if frame.empty or "attempts" not in frame.columns:
        return []

    def series(*names: str) -> pd.Series:
        for name in names:
            if name in frame.columns:
                return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=frame.index)

    name_column = next((c for c in ("player_display_name", "player_name") if c in frame.columns), None)
    if name_column is None and "player_id" not in frame.columns:
        return []  # cannot tell quarterbacks apart without an id or a name
    names = frame[name_column].astype(str) if name_column else frame["player_id"].astype(str)
    work = pd.DataFrame({
        "player_id": frame["player_id"].astype(str) if "player_id" in frame.columns else names,
        "name": names,
        "team": frame["team"].map(normalize_team) if "team" in frame.columns else "",
        "week": frame["week"] if "week" in frame.columns else 0,
        "attempts": series("attempts"), "completions": series("completions"), "yards": series("passing_yards"),
        "tds": series("passing_tds"), "ints": series("passing_interceptions", "interceptions"),
        "sacks": series("sacks_suffered"), "epa": series("passing_epa"),
    })
    work = work[(work["attempts"] > 0) | (work["sacks"] > 0)]
    if work.empty:
        return []
    grouped = work.groupby(["player_id", "team"], sort=False).agg(
        name=("name", "first"), attempts=("attempts", "sum"), completions=("completions", "sum"), yards=("yards", "sum"),
        tds=("tds", "sum"), ints=("ints", "sum"), sacks=("sacks", "sum"), epa=("epa", "sum"), games=("week", "nunique"),
    ).reset_index()
    return grouped.to_dict("records")


def _neutral_qb_context(warnings: list[str] | None = None, season: int | None = None, prior_season: int | None = None) -> dict[str, Any]:
    return {
        "teams": {}, "players": {}, "names": {},
        "league": dict(_DEFAULT_QB_BASELINE),
        "meta": {"season": season, "prior_season": prior_season, "warnings": list(warnings or []), "source": "defaults (no quarterback data)"},
    }


def build_qb_context(
    current: pd.DataFrame | None, prior: pd.DataFrame | None, *, season: int | None = None,
    prior_season: int | None = None, warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Pure builder behind :func:`load_qb_context` (takes raw weekly player-stat frames)."""
    warnings = list(warnings or [])
    cur_rows, prior_rows = _qb_rows(current), _qb_rows(prior)
    if not cur_rows and not prior_rows:
        warnings.append("No quarterback passing data was available; every QB is priced at league-average value.")
        return _neutral_qb_context(warnings, season, prior_season)

    # Pool each player across seasons and teams (quality travels with the player).
    pooled: dict[str, dict[str, Any]] = {}
    for rows, weight, tag in ((cur_rows, 1.0, "current"), (prior_rows, QB_PRIOR_WEIGHT, "prior")):
        for row in rows:
            rec = pooled.setdefault(row["player_id"], {
                "player_id": row["player_id"], "name": row["name"], "team": None, "team_attempts": {},
                "w_att": 0.0, "w_cmp": 0.0, "w_yds": 0.0, "w_td": 0.0, "w_int": 0.0, "w_db": 0.0, "w_epa": 0.0,
                "attempts_current": 0.0, "attempts_prior": 0.0, "games_current": 0.0, "games_prior": 0.0,
            })
            rec["w_att"] += weight * row["attempts"]
            rec["w_cmp"] += weight * row["completions"]
            rec["w_yds"] += weight * row["yards"]
            rec["w_td"] += weight * row["tds"]
            rec["w_int"] += weight * row["ints"]
            rec["w_db"] += weight * (row["attempts"] + row["sacks"])
            rec["w_epa"] += weight * row["epa"]
            rec[f"attempts_{tag}"] += row["attempts"]
            rec[f"games_{tag}"] += row["games"]
            if tag == "current":
                rec["team_attempts"][row["team"]] = rec["team_attempts"].get(row["team"], 0.0) + row["attempts"]
            elif not rec["team_attempts"]:
                rec.setdefault("prior_team_attempts", {})
                rec["prior_team_attempts"][row["team"]] = rec["prior_team_attempts"].get(row["team"], 0.0) + row["attempts"]
    for rec in pooled.values():
        source = rec["team_attempts"] or rec.get("prior_team_attempts", {})
        rec["team"] = max(source, key=source.get) if source else ""  # latest-season team = current team

    # Baselines from the data itself: average starter and replacement (backup) level.
    starters = [r for r in pooled.values() if r["w_att"] >= QB_STARTER_MIN_ATTEMPTS]
    backups = [r for r in pooled.values() if 20.0 <= r["w_att"] < QB_STARTER_MIN_ATTEMPTS]

    def epa_per_db(group: list[dict[str, Any]]) -> float | None:
        dropbacks = sum(r["w_db"] for r in group)
        return sum(r["w_epa"] for r in group) / dropbacks if dropbacks > 0 else None

    def rating(group: list[dict[str, Any]]) -> float | None:
        attempts = sum(r["w_att"] for r in group)
        if attempts <= 0:
            return None
        total = {key: sum(r[key] for r in group) for key in ("w_cmp", "w_yds", "w_td", "w_int")}
        return _passer_rating(total["w_cmp"], attempts, total["w_yds"], total["w_td"], total["w_int"])

    base_epa = epa_per_db(starters)
    base_rating = rating(starters)
    replacement_epa = epa_per_db(backups) if sum(r["w_db"] for r in backups) >= 150 else None
    replacement_rating = rating(backups) if replacement_epa is not None else None
    baseline_epa = base_epa if base_epa is not None else _DEFAULT_QB_BASELINE["epa_per_dropback"]
    if replacement_epa is None:
        replacement_epa = baseline_epa - QB_REPLACEMENT_DELTA_DEFAULT
    replacement_epa = _clamp(replacement_epa, baseline_epa - 0.30, baseline_epa - 0.04)
    baseline_rating = base_rating if base_rating is not None else _DEFAULT_QB_BASELINE["passer_rating"]
    replacement_rating = replacement_rating if replacement_rating is not None else _DEFAULT_QB_BASELINE["replacement_passer_rating"]
    baseline = {
        "passer_rating": round(baseline_rating, 2),
        "epa_per_dropback": round(baseline_epa, 4),
        "replacement_epa": round(replacement_epa, 4),
        "replacement_passer_rating": round(replacement_rating, 2),
        "unproven_epa": round(baseline_epa - QB_UNPROVEN_DELTA_EPA, 4),
        "unproven_passer_rating": round((baseline_rating + replacement_rating) / 2.0, 2),
        "starters_counted": len(starters),
        "backups_counted": len(backups),
        "source": "pooled nflverse player stats" if base_epa is not None else "defaults",
    }

    # Team primary QB: most live-season attempts (plus a recency-weighted share of last season's).
    by_team: dict[str, list[dict[str, Any]]] = {}
    for rec in pooled.values():
        by_team.setdefault(rec["team"], []).append(rec)
    primary_by_team = {
        team: max(group, key=lambda r: (r["attempts_current"] + 0.35 * r["attempts_prior"], r["w_att"]))
        for team, group in by_team.items() if team
    }

    players: dict[str, dict[str, Any]] = {}
    names: dict[str, str] = {}
    for pid, rec in pooled.items():
        is_primary = primary_by_team.get(rec["team"]) is rec
        starter_like = rec["w_att"] >= QB_STARTER_MIN_ATTEMPTS or (is_primary and rec["attempts_current"] >= QB_LIVE_STARTER_ATTEMPTS)
        role = "starter" if starter_like else "backup"
        record = {
            "player_id": pid,
            "name": rec["name"],
            "team": rec["team"],
            "attempts": int(round(rec["attempts_current"] + rec["attempts_prior"])),
            "attempts_current": int(round(rec["attempts_current"])),
            "attempts_prior": int(round(rec["attempts_prior"])),
            "games_current": int(rec["games_current"]),
            "games_prior": int(rec["games_prior"]),
            "eff_attempts": round(rec["w_att"], 1),
            "eff_dropbacks": round(rec["w_db"], 1),
            "passer_rating": round(_passer_rating(rec["w_cmp"], rec["w_att"], rec["w_yds"], rec["w_td"], rec["w_int"]), 1),
            "epa_per_dropback": round(rec["w_epa"] / rec["w_db"], 4) if rec["w_db"] > 0 else None,
            "role": role,
            "is_team_primary": is_primary,
        }
        record["impact_points"] = qb_impact_points(record, baseline)
        players[pid] = record
        key = _name_key(rec["name"])
        if key:  # an all-symbol / non-Latin name reduces to "", which must never match unrelated names
            names.setdefault(key, pid)

    # Value already embedded in each team's rating: attempt-weighted impact of the QBs who actually played for it.
    embedded: dict[str, dict[str, float | None]] = {}
    for tag, rows in (("current", cur_rows), ("prior", prior_rows)):
        totals: dict[str, list[float]] = {}
        for row in rows:
            impact = players[row["player_id"]]["impact_points"]
            acc = totals.setdefault(row["team"], [0.0, 0.0])
            acc[0] += impact * row["attempts"]
            acc[1] += row["attempts"]
        for team, (weighted, attempts) in totals.items():
            entry = embedded.setdefault(team, {"current": None, "prior": None, "attempts_current": 0, "attempts_prior": 0})
            entry[tag] = round(weighted / attempts, 3) if attempts > 0 else None
            entry[f"attempts_{tag}"] = int(attempts)

    teams = {}
    for team, rec in primary_by_team.items():
        teams[team] = {**players[rec["player_id"]], "embedded_impact": embedded.get(team, {"current": None, "prior": None, "attempts_current": 0, "attempts_prior": 0})}
    for team, entry in embedded.items():  # teams whose primary QB now plays elsewhere still need an embedded value
        teams.setdefault(team, {"name": None, "embedded_impact": entry})

    meta_warnings = list(warnings)
    if season is not None and not cur_rows:
        meta_warnings.append(f"No {season} passing data yet; quarterbacks are valued on {prior_season} alone.")
    if prior_season is not None and not prior_rows:
        meta_warnings.append(f"No {prior_season} passing data; quarterbacks are valued on {season} alone.")
    return {
        "teams": teams,
        "players": players,
        "names": names,
        "league": baseline,
        "meta": {
            "season": season, "prior_season": prior_season, "warnings": list(dict.fromkeys(meta_warnings)),
            "source": "nflverse weekly player stats (pooled, recency-weighted)",
            "prior_weight": QB_PRIOR_WEIGHT, "shrink_dropbacks": QB_SHRINK_DROPBACKS,
        },
    }


def load_qb_context(season: int, prior_season: int | None = None) -> dict[str, Any]:
    """Quarterback values for ``season`` and ``prior_season`` (default ``season - 1``).

    Returns ``{"teams": {TEAM: qb_record}, "players": {player_id: qb_record},
    "names": {name_key: player_id}, "league": baseline, "meta": {...}}``.

    A ``qb_record`` has ``player_id, name, team (current team), attempts,
    attempts_current, attempts_prior, games_current, games_prior, eff_attempts,
    eff_dropbacks (recency-weighted), passer_rating (pooled NFL passer rating),
    epa_per_dropback (pooled, raw), role ("starter" | "backup"),
    is_team_primary, impact_points``; :func:`resolve_qb` can also return
    ``role == "unknown"`` for a name the data has never seen.  ``teams[T]``
    additionally carries
    ``embedded_impact {current, prior, attempts_current, attempts_prior}`` --
    the average value of the QBs that produced that team's rating.

    ``league`` is the baseline: ``passer_rating, epa_per_dropback`` of the
    average starter, ``replacement_epa, replacement_passer_rating`` measured
    from backups in the same data, ``unproven_epa, unproven_passer_rating``
    (the prior for a listed starter with no history), ``starters_counted,
    backups_counted, source``.  Current and prior seasons are pooled by
    attempts (the prior season at ``QB_PRIOR_WEIGHT``).  Fails soft to a neutral
    context with a warning when no passing data can be loaded.
    """
    season = int(season)
    prior_season = int(prior_season) if prior_season is not None else season - 1
    key = ("qb", season, prior_season)
    cached = _memo_get(key)
    if cached is not None:
        return cached
    warnings: list[str] = []
    frames: dict[int, pd.DataFrame | None] = {}
    degraded = False
    for year in (season, prior_season):
        frame, error = _try(_cached_player_stats, year)
        if error and year == prior_season:
            degraded = True
            warnings.append(f"{year} player stats unavailable ({error.split(':')[0]}).")
        elif error:
            degraded = True
        frames[year] = frame
    context, build_error = _try(
        lambda: build_qb_context(frames[season], frames[prior_season], season=season, prior_season=prior_season, warnings=warnings)
    )
    if build_error:  # a frame that downloaded but cannot be parsed degrades to a neutral context, never an exception
        degraded = True
        context = _neutral_qb_context(
            warnings + [f"Quarterback data could not be processed ({build_error.split(':')[0]}); every QB is priced at league-average value."],
            season, prior_season,
        )
    _memo_put(key, context, _MEMO_DEGRADED_SECONDS if degraded else _MEMO_OK_SECONDS)
    return context


def _resolve_baseline(baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill any missing baseline key from defaults, deriving replacement / unproven levels relative to the given average."""
    given = dict(baseline or {})
    base = {**_DEFAULT_QB_BASELINE, **given}
    if "replacement_epa" not in given:
        base["replacement_epa"] = base["epa_per_dropback"] - QB_REPLACEMENT_DELTA_DEFAULT
    if "unproven_epa" not in given:
        base["unproven_epa"] = base["epa_per_dropback"] - QB_UNPROVEN_DELTA_EPA
    if "unproven_passer_rating" not in given:
        base["unproven_passer_rating"] = (base["passer_rating"] + base["replacement_passer_rating"]) / 2.0
    return base


def qb_impact_points(qb: Mapping[str, Any] | None, baseline: Mapping[str, Any] | None = None) -> float:
    """Points of spread value of ``qb`` over an average starter (clamped to +/- ``QB_IMPACT_CLAMP``).

    ``points = QB_POINTS_PER_EPA * (stabilised EPA/dropback - league starter EPA/dropback)``.
    The QB's own pooled EPA/dropback is stabilised toward a prior with
    ``QB_SHRINK_DROPBACKS`` dropbacks of weight.  The prior depends on ``role``:
    the average starter for a "starter"; the measured *replacement level* (what
    backups actually produced in the same data) for a "backup"; and a slightly
    milder "unproven starter" level (``QB_UNPROVEN_DELTA_EPA`` below average)
    for "unknown" -- a listed starter with no NFL passing history, almost always
    a rookie.  Without EPA data the passer rating is used at
    ``QB_POINTS_PER_RATING`` points per rating point.  ``None`` or an empty
    record is an unknown QB (a small negative number).
    """
    base = _resolve_baseline(baseline)
    record = qb or {}
    role = record.get("role") or ("unknown" if not record else "starter")
    epa_prior = {"starter": base["epa_per_dropback"], "backup": base["replacement_epa"], "unknown": base["unproven_epa"]}.get(role, base["epa_per_dropback"])
    rating_prior = {"starter": base["passer_rating"], "backup": base["replacement_passer_rating"], "unknown": base["unproven_passer_rating"]}.get(role, base["passer_rating"])
    epa, dropbacks = _num(record.get("epa_per_dropback")), _finite(record.get("eff_dropbacks"), 0.0)
    if epa is not None and dropbacks > 0:
        stable = (dropbacks * epa + QB_SHRINK_DROPBACKS * epa_prior) / (dropbacks + QB_SHRINK_DROPBACKS)
        points = QB_POINTS_PER_EPA * (stable - base["epa_per_dropback"])
    elif _num(record.get("passer_rating")) is not None and _finite(record.get("eff_attempts"), 0.0) > 0:
        attempts = _finite(record.get("eff_attempts"), 0.0)
        stable = (attempts * record["passer_rating"] + QB_SHRINK_ATTEMPTS * rating_prior) / (attempts + QB_SHRINK_ATTEMPTS)
        points = QB_POINTS_PER_RATING * (stable - base["passer_rating"])
    else:
        points = QB_POINTS_PER_EPA * (epa_prior - base["epa_per_dropback"])
    return round(_clamp(points, -QB_IMPACT_CLAMP, QB_IMPACT_CLAMP), 2)


def resolve_qb(qb_context: Mapping[str, Any] | None, team: str, *, name: str | None = None, player_id: str | None = None) -> dict[str, Any]:
    """The QB record for one side of a game.

    Order: the schedule's QB id, then its QB name, then (when the schedule
    names nobody) the team's most-used QB.  A name the data has never seen
    returns an ``role == "unknown"`` record.  Adds ``source`` ("schedule_id" |
    "schedule_name" | "team_primary" | "schedule_unknown" | "none") and
    ``found`` (bool) to a copy of the record.
    """
    context = qb_context or {}
    team = normalize_team(team)
    players = context.get("players") or {}
    record, source = None, "none"
    if player_id and player_id in players:
        record, source = players[player_id], "schedule_id"
    elif name:
        key = _name_key(name)
        pid = (context.get("names") or {}).get(key) if key else None
        if pid in players:
            record, source = players[pid], "schedule_name"
    if record is None and not name:
        primary = (context.get("teams") or {}).get(team)
        if primary and primary.get("player_id"):
            record, source = primary, "team_primary"
    if record is None:
        label = name or "Unknown QB"
        return {
            "player_id": player_id or None, "name": label, "team": team, "attempts": 0, "attempts_current": 0, "attempts_prior": 0,
            "games_current": 0, "games_prior": 0, "eff_attempts": 0.0, "eff_dropbacks": 0.0, "passer_rating": None,
            "epa_per_dropback": None, "role": "unknown", "is_team_primary": False, "impact_points": None,
            "source": "schedule_unknown" if name else "none", "found": False,
        }
    return {**{k: v for k, v in record.items() if k != "embedded_impact"}, "source": source, "found": True}


def _embedded_impact(qb_context: Mapping[str, Any] | None, team: str, blend_weight_current: float) -> float:
    """QB value already inside the team's rating, regressed exactly like the rating is."""
    entry = ((qb_context or {}).get("teams") or {}).get(normalize_team(team), {}).get("embedded_impact") or {}
    cur, prior = _num(entry.get("current")), _num(entry.get("prior"))
    weight = blend_weight_current if cur is not None else 0.0
    return weight * (cur or 0.0) + (1.0 - weight) * PRIOR_CARRYOVER * (prior or 0.0)


# --------------------------------------------------------------------------
# Game analysis
# --------------------------------------------------------------------------

def _pace_label(vs_league_plays: float) -> str:
    """"Fast" at +``PACE_LABEL_PLAYS`` or more plays than the league's typical game, "Slow" at -``PACE_LABEL_PLAYS`` or fewer."""
    if vs_league_plays >= PACE_LABEL_PLAYS:
        return "Fast"
    if vs_league_plays <= -PACE_LABEL_PLAYS:
        return "Slow"
    return "Average"


def _neutral_ratings_row(code: str, league: Mapping[str, float]) -> dict[str, Any]:
    return {
        "team": code,
        "offensive_efficiency": league["offensive_efficiency"], "defensive_efficiency": league["defensive_efficiency"],
        "net_efficiency": league["offensive_efficiency"] + league["defensive_efficiency"] - 100.0,
        "off_epa": league["off_epa"], "def_epa": league["def_epa"], "pace_spp": league["pace_spp"],
        "points_for_pg": league["points_pg"], "points_against_pg": league["points_pg"],
        "strength_rating": 50.0, "neutral_margin": 0.0, "games_current": 0, "games_prior": 0,
        "blend_weight_current": 0.0, "data_quality": "none",
    }


def _neutral_ratings() -> dict[str, Any]:
    league = {**_DEFAULT_LEAGUE, "epa_per_play": 0.0, "games_current_avg": 0.0, "margin_stdev": MARGIN_STDEV}
    return {"teams": {}, "league": league, "meta": {"season": None, "prior_season": None, "games_current": {}, "blend_weight_current": {}, "warnings": [], "source": "defaults"}}


def _situation(game: Mapping[str, Any]) -> dict[str, Any]:
    """Rest / divisional / weather inputs, sanitised, with the points each is worth."""
    home_rest, away_rest = _num(game.get("home_rest")), _num(game.get("away_rest"))
    rest_diff = (home_rest - away_rest) if home_rest is not None and away_rest is not None else None
    rest_spread = REST_POINTS_PER_DAY * _clamp(rest_diff, -REST_DAYS_CAP, REST_DAYS_CAP) if rest_diff is not None else 0.0
    roof = _text_or_empty(game.get("roof")).lower() or None  # NaN must not become the string "nan"
    wind, temp = _num(game.get("wind")), _num(game.get("temp"))
    weather_total = 0.0
    if wind is not None and roof not in _ROOFED and wind > WIND_THRESHOLD_MPH:
        weather_total = -min(WIND_TOTAL_CAP, WIND_TOTAL_PER_MPH * (wind - WIND_THRESHOLD_MPH))
    neutral_site = _truthy(game.get("neutral_site"))
    return {
        "home_rest": home_rest, "away_rest": away_rest, "rest_diff": rest_diff, "rest_spread": rest_spread,
        "div_game": _truthy(game.get("div_game")), "neutral_site": neutral_site,
        # The market spread at a neutral venue carries no home-field edge, so neither may the model.
        "home_field": 0.0 if neutral_site else HOME_FIELD_POINTS,
        "roof": roof, "wind": wind, "temp": temp, "weather_total": weather_total,
    }


def _qb_side(qb_context: Mapping[str, Any] | None, ratings: Mapping[str, Any], team: str, name: str | None, pid: str | None, baseline: Mapping[str, Any]) -> dict[str, Any]:
    record = resolve_qb(qb_context, team, name=name, player_id=pid)
    impact = qb_impact_points(record, baseline)
    weight = _finite((ratings["teams"].get(team) or {}).get("blend_weight_current"), 0.0)
    embedded = _embedded_impact(qb_context, team, weight)
    adjustment = round(_clamp(impact - embedded, -QB_IMPACT_CLAMP, QB_IMPACT_CLAMP), 2)
    estimate = record["passer_rating"] is None or _finite(record.get("attempts"), 0.0) < QB_MIN_RATING_ATTEMPTS
    epa = _num(record.get("epa_per_dropback"))
    resolved = _resolve_baseline(baseline)
    proxy = "unproven" if record["role"] == "unknown" else "replacement"  # what an estimate stands in for
    return {
        "record": record,
        "name": record["name"],
        "rating": float(resolved[f"{proxy}_passer_rating"]) if estimate else float(record["passer_rating"]),
        "epa_per_dropback": float(resolved[f"{proxy}_epa"]) if (estimate or epa is None) else epa,
        "role": record["role"],
        "source": record["source"],
        "is_estimate": estimate,
        "impact": impact,
        "embedded": round(embedded, 2),
        "adjustment": adjustment,
    }


def _confidence(
    favored_prob: float, avg_games_current: float, missing_teams: int, qb_flags: int, market_gap: float | None, market_disagrees: bool,
) -> tuple[float, str, dict[str, float]]:
    """Confidence (0-1) that the model's favorite is the right side, plus its factors.

    ``margin``  : 2 * (favorite win prob - 0.5), i.e. |margin| measured on the calibrated ``WIN_PROB_STDEV`` win curve.
    ``sample``  : 0.85 -> 1.0 as live-season games per team go 0 -> 8; -0.15 for each team missing from the ratings.
    ``qb``      : -0.12 for each starter who is a backup / unproven / unknown.
    ``market``  : 1.0 within 1.5 points of the market, falling 0.06 per extra point (floor 0.55); x0.85 if the market
                  picks the other team; 0.92 when no line exists to cross-check.
    """
    margin = _clamp(2.0 * (favored_prob - 0.5), 0.0, 1.0)
    sample = _clamp(0.85 + 0.15 * min(1.0, avg_games_current / 8.0) - 0.15 * missing_teams, 0.5, 1.0)
    qb = _clamp(1.0 - 0.12 * qb_flags, 0.5, 1.0)
    if market_gap is None:
        market = 0.92
    else:
        market = _clamp(1.0 - max(0.0, market_gap - 1.5) * 0.06, 0.55, 1.0) * (0.85 if market_disagrees else 1.0)
    value = round(_clamp(margin * sample * qb * market, 0.0, 1.0), 3)
    label = "High" if value >= 0.40 else "Moderate" if value >= 0.20 else "Low"
    return value, label, {"margin": round(margin, 3), "sample": round(sample, 3), "qb": round(qb, 3), "market": round(market, 3)}


def analyze_game(
    game: Mapping[str, Any], ratings: Mapping[str, Any] | None, qb_context: Mapping[str, Any] | None, *,
    user_spread_home: float | None = None,
) -> dict[str, Any]:
    """Full moneyline / spread analysis of one game.

    ``game`` is one dict from :func:`games_for_week` (only ``home_team`` and
    ``away_team`` are required; everything else is optional).  ``ratings`` is
    :func:`load_team_ratings`, ``qb_context`` is :func:`load_qb_context`.  Either
    may be empty/``None`` -- the analysis then runs on league-average inputs
    and says so in ``warnings``.  ``user_spread_home`` is a spread the user
    typed, in the nflverse convention (positive = home favored).

    Returned keys (all numbers finite; ``None`` only where a market value does
    not exist):

    * ``game_id, week, away_team, home_team, kickoff_label, status, played, season``
    * ``teams`` -- ``{"home": {...}, "away": {...}}``, each with ``team, team_name,
      strength_rating, offensive_efficiency, defensive_efficiency, net_efficiency,
      off_epa, def_epa, points_for_pg, points_against_pg, pace_spp, qb_name,
      qb_rating, qb_epa_per_dropback (both replaced by replacement-level values
      and flagged ``qb_is_estimate`` when the QB has under ``QB_MIN_RATING_ATTEMPTS``
      attempts of history), qb_role ("starter" | "backup" | "unknown"), qb_source
      ("schedule_id" | "schedule_name" | "team_primary" | "schedule_unknown" | "none";
      "team_primary" and "none" mean the schedule named no starter and a warning says so),
      qb_is_estimate, qb_impact_points (vs an average starter), qb_embedded_points,
      qb_adjustment_points (what the spread actually uses), expected_points,
      win_probability, spread_cover_probability, fair_moneyline, games_current,
      games_prior, blend_weight_current, data_quality``.
    * ``pace`` -- ``plays_projection, seconds_per_play, label, vs_league, league_plays,
      league_seconds_per_play, home_spp, away_spp``.
    * ``scoring`` -- ``home_expected_points, away_expected_points, model_spread_home
      (positive = home favored), model_total, market_total, total_vs_market,
      base_spread_home, base_total, efficiency_home_points, efficiency_away_points,
      adjustments {base_spread, base_total, efficiency_spread, efficiency_total,
      home_field, qb_spread, qb_total, rest_spread, weather_total}``.
    * ``probabilities`` -- ``home_win, away_win, home_cover, away_cover,
      spread_used_home, spread_source ("user" | "market" | "model"),
      market_spread_home, model_spread_home, margin_stdev``.
    * ``moneyline`` -- ``home_fair_price, away_fair_price, market_home_ml,
      market_away_ml, market_home_prob_novig, market_away_prob_novig,
      market_hold, edge_home_vs_market, edge_away_vs_market``.
    * ``favored ("home" | "away"), favored_team, margin (abs model points),
      confidence (0-1), confidence_label ("Low" | "Moderate" | "High"),
      confidence_factors {margin, sample, qb, market}``.
    * ``context`` -- ``home_rest, away_rest, rest_diff, div_game, neutral_site, roof, wind,
      temp`` (``neutral_site`` True means no home-field points were applied).
    * ``reasoning`` -- list of ``{title, detail, tone, team}``; ``tone`` is "pro"
      when the factor supports the model's favorite, "con" when it works
      against that pick (or is a risk to it), "neutral" otherwise; ``team`` is
      the team the factor benefits (``None`` for game-level context).
    * ``warnings`` -- list of strings.
    """
    home = normalize_team(game.get("home_team"))
    away = normalize_team(game.get("away_team"))
    if not home or not away:
        raise ValueError("analyze_game needs both home_team and away_team")
    if home == away:
        raise ValueError("home_team and away_team must be different teams")

    ratings = ratings if ratings and ratings.get("teams") is not None and ratings.get("league") else _neutral_ratings()
    league = ratings["league"]
    warnings: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    missing = 0
    for side, code in (("home", home), ("away", away)):
        row = ratings["teams"].get(code)
        if row is None or row.get("data_quality") == "none":
            missing += 1
            warnings.append(f"No rating data for {code}; the model uses a league-average team.")
            row = _neutral_ratings_row(code, league)
        rows[side] = row

    baseline = (qb_context or {}).get("league") or _DEFAULT_QB_BASELINE
    no_qb_data = not qb_context or not qb_context.get("players")
    if no_qb_data:
        warnings.append("Quarterback data unavailable; no quarterback adjustment is applied.")
    qb = {  # _text_or_empty also turns a NaN that slipped in from a DataFrame row into "no name"
        side: _qb_side(
            qb_context, ratings, code, _text_or_empty(game.get(f"{side}_qb")) or None,
            _text_or_empty(game.get(f"{side}_qb_id")) or None, baseline,
        )
        for side, code in (("home", home), ("away", away))
    }
    if no_qb_data:  # nothing to value or adjust against: report zeros instead of invented numbers
        for side in qb.values():
            side["impact"] = side["embedded"] = side["adjustment"] = 0.0
    for side, code in (("home", home), ("away", away)):
        info = qb[side]
        if no_qb_data:
            continue
        if game.get(f"{side}_qb_source") == "last_game" and info["name"]:
            warnings.append(f"{code}: no starter is listed yet, so {info['name']} (the starter in {code}'s most recent game) is assumed.")
        elif info["source"] == "team_primary":  # nothing in the schedule named a starter: say so instead of presenting a guess as fact
            warnings.append(f"{code}: no starting quarterback is listed, so {info['name']} (the team's most-used passer in the data) is assumed.")
        if info["source"] == "none":
            warnings.append(f"{code}: no starting quarterback could be identified; priced as an unproven starter ({_signed(info['impact'])} pts vs an average starter).")
        elif info["is_estimate"] and info["role"] == "unknown":
            warnings.append(f"{code}: no NFL passing history found for {info['name']} (likely a rookie or new starter); priced as an unproven starter ({_signed(info['impact'])} pts vs an average starter).")
        elif info["role"] == "backup":
            warnings.append(f"{code}: {info['name']} is a backup or unproven starter ({info['record'].get('attempts', 0)} career attempts in the data); the model prices him at {_signed(info['impact'])} pts vs an average starter.")

    situation = _situation(game)
    projection = _project(
        rows["home"], rows["away"], league, home_team=home, away_team=away, home_field=situation["home_field"],
        qb_home=qb["home"]["adjustment"], qb_away=qb["away"]["adjustment"],
        rest_spread=situation["rest_spread"], weather_total=situation["weather_total"],
    )
    spread_home = projection["model_spread_home"]
    home_win = win_probability_from_spread(spread_home, stdev=WIN_PROB_STDEV)
    away_win = round(1.0 - home_win, 4)

    market_spread = _num(game.get("market_spread_home"))
    user_spread = _num(user_spread_home)
    if user_spread is not None:
        spread_used, spread_source = user_spread, "user"
    elif market_spread is not None:
        spread_used, spread_source = market_spread, "market"
    else:
        spread_used, spread_source = round(spread_home * 2.0) / 2.0, "model"
    # Home covers when (home margin) > S, so P = 1 - Phi((S - m) / sigma) = Phi((m - S) / sigma).
    home_cover = win_probability_from_spread(spread_home - spread_used)
    away_cover = round(1.0 - home_cover, 4)

    market_ml_home, market_ml_away = _num(game.get("market_home_ml")), _num(game.get("market_away_ml"))
    novig_home = novig_away = hold = None
    if market_ml_home is not None and market_ml_away is not None and abs(market_ml_home) >= 100 and abs(market_ml_away) >= 100:
        fair_home, fair_away = no_vig_probabilities(market_ml_home, market_ml_away)
        novig_home, novig_away = round(fair_home, 4), round(fair_away, 4)
        hold = round(market_hold(market_ml_home, market_ml_away), 4)
    else:
        market_ml_home = market_ml_away = None
    edge_home = round(home_win - novig_home, 4) if novig_home is not None else None
    edge_away = round(away_win - novig_away, 4) if novig_away is not None else None

    favored = "home" if spread_home >= 0 else "away"
    favored_team = home if favored == "home" else away
    margin = round(abs(spread_home), 2)
    favored_prob = home_win if favored == "home" else away_win

    market_gap = abs(spread_home - market_spread) if market_spread is not None else None
    market_disagrees = market_spread is not None and market_spread != 0 and (market_spread > 0) != (spread_home > 0)
    games_cur = [float(rows[s].get("games_current", 0)) for s in ("home", "away")]
    qb_flags = sum(1 for s in ("home", "away") if qb[s]["role"] in ("backup", "unknown")) if not no_qb_data else 1
    confidence, confidence_label, factors = _confidence(favored_prob, sum(games_cur) / 2.0, missing, qb_flags, market_gap, market_disagrees)

    lg_plays, lg_spp = league["total_plays_pg"], league["pace_spp"]
    plays = round(projection["plays_projection"], 1)
    vs_league = round(plays - lg_plays, 1)
    pace = {
        "plays_projection": plays,
        "seconds_per_play": round(projection["matchup_spp"], 2),
        "label": _pace_label(vs_league),
        "vs_league": vs_league,
        "league_plays": round(lg_plays, 1),
        "league_seconds_per_play": round(lg_spp, 2),
        "home_spp": round(rows["home"]["pace_spp"], 2),
        "away_spp": round(rows["away"]["pace_spp"], 2),
    }
    market_total = _num(game.get("market_total"))
    scoring = {
        "home_expected_points": projection["home_expected_points"],
        "away_expected_points": projection["away_expected_points"],
        "model_spread_home": spread_home,
        "model_total": projection["model_total"],
        "market_total": market_total,
        "total_vs_market": round(projection["model_total"] - market_total, 2) if market_total is not None else None,
        "base_spread_home": projection["base"]["spread"],
        "base_total": projection["base"]["total"],
        "efficiency_home_points": projection["efficiency_home"],
        "efficiency_away_points": projection["efficiency_away"],
        "adjustments": projection["adjustments"],
    }
    probabilities = {
        "home_win": home_win, "away_win": away_win, "home_cover": home_cover, "away_cover": away_cover,
        "spread_used_home": spread_used, "spread_source": spread_source,
        "market_spread_home": market_spread, "model_spread_home": spread_home, "margin_stdev": MARGIN_STDEV,
        "win_prob_stdev": WIN_PROB_STDEV,
    }
    moneyline = {
        "home_fair_price": fair_price_from_probability(home_win),
        "away_fair_price": fair_price_from_probability(away_win),
        "market_home_ml": market_ml_home, "market_away_ml": market_ml_away,
        "market_home_prob_novig": novig_home, "market_away_prob_novig": novig_away, "market_hold": hold,
        "edge_home_vs_market": edge_home, "edge_away_vs_market": edge_away,
    }

    teams: dict[str, dict[str, Any]] = {}
    for side, code, win, cover, expected in (
        ("home", home, home_win, home_cover, projection["home_expected_points"]),
        ("away", away, away_win, away_cover, projection["away_expected_points"]),
    ):
        row, info = rows[side], qb[side]
        teams[side] = {
            "team": code, "team_name": team_name(code),
            "strength_rating": row["strength_rating"], "offensive_efficiency": row["offensive_efficiency"],
            "defensive_efficiency": row["defensive_efficiency"], "net_efficiency": row["net_efficiency"],
            "off_epa": row["off_epa"], "def_epa": row["def_epa"],
            "points_for_pg": row["points_for_pg"], "points_against_pg": row["points_against_pg"], "pace_spp": row["pace_spp"],
            "qb_name": info["name"], "qb_rating": round(info["rating"], 1), "qb_epa_per_dropback": info["epa_per_dropback"],
            "qb_role": info["role"], "qb_source": info["source"], "qb_is_estimate": info["is_estimate"],
            "qb_impact_points": info["impact"], "qb_embedded_points": info["embedded"], "qb_adjustment_points": info["adjustment"],
            "expected_points": expected, "win_probability": win, "spread_cover_probability": cover,
            "fair_moneyline": moneyline["home_fair_price"] if side == "home" else moneyline["away_fair_price"],
            "games_current": int(row.get("games_current", 0)), "games_prior": int(row.get("games_prior", 0)),
            "blend_weight_current": float(row.get("blend_weight_current", 0.0)), "data_quality": row.get("data_quality", "none"),
        }

    played = _truthy(game.get("played"))
    if played:
        warnings.append("This game has already been played; the ratings already include its result.")
    meta_warnings = list((ratings.get("meta") or {}).get("warnings") or []) + list(((qb_context or {}).get("meta") or {}).get("warnings") or [])
    warnings.extend(meta_warnings)
    if market_ml_home is None and (_num(game.get("market_home_ml")) is not None or _num(game.get("market_away_ml")) is not None):
        warnings.append("The market moneyline is incomplete or invalid, so no market comparison is shown.")

    week_number, season_number = _num(game.get("week")), _num(game.get("season"))  # pass-through metadata is sanitised too: no NaN / inf / "nan" text
    result = {
        "game_id": _text_or_empty(game.get("game_id")) or f"{away}@{home}",
        "week": int(week_number) if week_number is not None else None,
        "season": int(season_number) if season_number is not None else None,
        "away_team": away, "home_team": home,
        "kickoff_label": _text_or_empty(game.get("kickoff_label")) or "TBD",
        "status": _text_or_empty(game.get("status")) or "Upcoming",
        "played": played,
        "teams": teams, "pace": pace, "scoring": scoring, "probabilities": probabilities, "moneyline": moneyline,
        "favored": favored, "favored_team": favored_team, "margin": margin,
        "confidence": confidence, "confidence_label": confidence_label, "confidence_factors": factors,
        "context": {k: situation[k] for k in ("home_rest", "away_rest", "rest_diff", "div_game", "neutral_site", "roof", "wind", "temp")},
        "warnings": [],
    }
    result["reasoning"] = _build_reasoning(result, rows, qb, situation, ratings, market_spread, market_gap)
    result["warnings"] = list(dict.fromkeys(warnings))
    return result


def _tone(team: str | None, favored_team: str, gain: float, threshold: float) -> str:
    """"pro" when ``team`` (who gains ``gain``) is the model favorite, "con" when it is the underdog, else neutral."""
    if team is None or abs(gain) < threshold:
        return "neutral"
    return "pro" if team == favored_team else "con"


def _build_reasoning(
    analysis: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]], qb: Mapping[str, Mapping[str, Any]],
    situation: Mapping[str, Any], ratings: Mapping[str, Any], market_spread: float | None, market_gap: float | None,
) -> list[dict[str, Any]]:
    """Why the model leans the way it does: every item cites the numbers behind it."""
    home, away = analysis["home_team"], analysis["away_team"]
    h, a = analysis["teams"]["home"], analysis["teams"]["away"]
    favored_team = analysis["favored_team"]
    league = ratings["league"]
    scoring, pace = analysis["scoring"], analysis["pace"]
    spread_home = scoring["model_spread_home"]
    items: list[dict[str, Any]] = []

    # 1. Overall power gap (neutral site, before QB / home field / situation).
    gap = h["strength_rating"] - a["strength_rating"]
    gap_points = gap / STRENGTH_POINTS_SCALE
    stronger = home if gap >= 0 else away
    weaker = away if gap >= 0 else home
    s_hi, s_lo = max(h["strength_rating"], a["strength_rating"]), min(h["strength_rating"], a["strength_rating"])
    items.append({
        "title": f"Team strength: {stronger} rates higher" if abs(gap_points) >= 0.3 else "Team strength is even",
        "detail": (
            f"{stronger} {s_hi:.0f} vs {weaker} {s_lo:.0f} on the 0-100 strength scale (50 = league average). "
            f"At {1 / STRENGTH_POINTS_SCALE:.1f} points of spread per rating point, that is worth about {abs(gap_points):.1f} points "
            f"at a neutral site, before home field, quarterback, and situation."
        ),
        "tone": _tone(stronger, favored_team, gap_points, 0.3), "team": stronger,
    })

    # 2-3. Offense vs defense, both directions, in EPA per play and points.
    half_plays = pace["plays_projection"] / 2.0
    for off_side, def_side, off_code, def_code in (("home", "away", home, away), ("away", "home", away, home)):
        off, dfn = analysis["teams"][off_side], analysis["teams"][def_side]
        edge = (off["off_epa"] - league["off_epa"]) + (dfn["def_epa"] - league["def_epa"])
        edge_points = edge * half_plays
        beneficiary = off_code if edge_points >= 0 else def_code
        items.append({
            "title": f"{off_code} offense vs {def_code} defense",
            "detail": (
                f"{off_code} offense scores {off['offensive_efficiency']:.0f}/100 ({off['off_epa']:+.3f} EPA/play, {off['points_for_pg']:.1f} pts/game); "
                f"{def_code} defense scores {dfn['defensive_efficiency']:.0f}/100 and allows {dfn['def_epa']:+.3f} EPA/play ({dfn['points_against_pg']:.1f} pts/game). "
                f"Combined, {off_code}'s offense should score about {abs(edge_points):.1f} points {'more' if edge_points >= 0 else 'fewer'} than a league-average matchup "
                f"({edge:+.3f} EPA/play over roughly {half_plays:.0f} plays)."
            ),
            "tone": _tone(beneficiary, favored_team, edge_points, 0.5), "team": beneficiary if abs(edge_points) >= 0.5 else None,
        })

    # 4. Quarterbacks: value over an average starter and what the spread actually uses.
    qb_gain = qb["home"]["adjustment"] - qb["away"]["adjustment"]
    qb_team = home if qb_gain >= 0 else away
    parts = []
    for side, code in (("home", home), ("away", away)):
        info = qb[side]
        rating_text = f"{info['rating']:.0f} passer rating" + (" est." if info["is_estimate"] else "")
        parts.append(f"{code}: {info['name']} ({rating_text}, {info['impact']:+.1f} pts vs an average starter, {info['adjustment']:+.1f} applied after the {info['embedded']:+.1f} already in the team rating)")
    items.append({
        "title": "Quarterback matchup" if abs(qb_gain) >= 0.3 else "Quarterbacks roughly cancel out",
        "detail": "; ".join(parts) + f". Net quarterback effect on the spread: {qb_gain:+.1f} points toward {qb_team}.",
        "tone": _tone(qb_team, favored_team, qb_gain, 0.3), "team": qb_team if abs(qb_gain) >= 0.3 else None,
    })

    # 5. Pace and total.
    market_total = scoring["market_total"]
    total_text = (
        f"The model total is {scoring['model_total']:.1f} against a market total of {market_total:.1f} ({scoring['total_vs_market']:+.1f})."
        if market_total is not None else f"The model total is {scoring['model_total']:.1f} points (no market total is posted)."
    )
    items.append({
        "title": f"{pace['label']} pace, {scoring['model_total']:.0f}-point scoring environment",
        "detail": (
            f"{home} runs {pace['home_spp']:.1f} sec/play and {away} {pace['away_spp']:.1f} (league {pace['league_seconds_per_play']:.1f}), "
            f"projecting {pace['plays_projection']:.0f} offensive plays ({pace['vs_league']:+.1f} vs the league's {pace['league_plays']:.0f}). "
            f"Expected score {home} {scoring['home_expected_points']:.1f} - {away} {scoring['away_expected_points']:.1f}. {total_text}"
        ),
        "tone": "neutral", "team": None,
    })

    # 6. Home field + situation.
    home_field = situation["home_field"]
    hfa_pct = (win_probability_from_spread(HOME_FIELD_POINTS, stdev=WIN_PROB_STDEV) - 0.5) * 100.0
    if situation["neutral_site"]:
        bits = [
            f"neutral-site game, so {home} (only the designated home team) gets {home_field:.1f} points of home-field edge instead of "
            f"the usual {HOME_FIELD_POINTS:.1f} points (about {hfa_pct:.1f} win-probability points in an otherwise even game)"
        ]
    else:
        bits = [f"home field is worth {home_field:.1f} points to {home} (about {hfa_pct:.1f} win-probability points in an otherwise even game)"]
    rest_note = ""
    if situation["rest_diff"] is not None:
        rest_note = f"rest: {home} {situation['home_rest']:.0f} days vs {away} {situation['away_rest']:.0f}"
        if situation["rest_spread"]:
            rest_note += f" ({situation['rest_spread']:+.1f} pts)"
        bits.append(rest_note)
    if situation["div_game"]:
        bits.append("a divisional game (familiar opponents; shown for context, not modelled)")
    if situation["wind"] is not None:
        if situation["roof"] in _ROOFED:
            bits.append(f"roofed stadium, so {situation['wind']:.0f} mph wind is ignored")
        elif situation["weather_total"]:
            bits.append(f"{situation['wind']:.0f} mph wind trims the total {abs(situation['weather_total']):.1f} points")
        else:
            bits.append(f"{situation['wind']:.0f} mph wind is below the {WIND_THRESHOLD_MPH:.0f} mph threshold")
    if situation["temp"] is not None:
        bits.append(f"{situation['temp']:.0f}F kickoff temperature (shown for context, not modelled)")
    situation_gain = home_field + situation["rest_spread"]  # points toward the designated home team (negative = toward the visitor)
    situation_team = home if situation_gain >= 0 else away
    items.append({
        "title": "Neutral site and situation" if situation["neutral_site"] else "Home field and situation",
        "detail": "; ".join(bits)[0].upper() + "; ".join(bits)[1:] + ".",
        "tone": _tone(situation_team, favored_team, situation_gain, 0.3),
        "team": situation_team if abs(situation_gain) >= 0.3 else None,
    })

    # 7. Market check.
    if market_spread is not None:
        agree = (market_spread > 0) == (spread_home > 0) if market_spread != 0 and spread_home != 0 else True
        novig = analysis["moneyline"]["market_home_prob_novig"]
        market_text = f"The market has {_spread_text(home, away, market_spread)}; the model has {_spread_text(home, away, spread_home)}, a {market_gap:.1f}-point gap"
        if novig is not None:
            market_text += f", and the de-vigged market win probability is {home} {novig:.1%} / {away} {1 - novig:.1%} vs the model's {analysis['probabilities']['home_win']:.1%} / {analysis['probabilities']['away_win']:.1%}"
        else:  # no moneyline posted: still cite the model's own probabilities so the item always carries real numbers
            market_text += f", and the model gives {home} {analysis['probabilities']['home_win']:.1%} / {away} {analysis['probabilities']['away_win']:.1%} to win"
        if not agree:
            market_text += f". The market picks {home if market_spread > 0 else away}, so treat this lean with caution"
        elif market_gap >= 3.0:
            market_text += ". A gap this large usually means the model is missing information (injuries, weather, line moves) rather than a true edge"
        else:
            market_text += ". The market and model broadly agree"
        items.append({
            "title": "Market check",
            "detail": market_text + ".",
            "tone": "con" if (not agree or market_gap >= 3.0) else "pro", "team": favored_team if agree else None,
        })
    else:
        items.append({
            "title": "No market line to check against",
            "detail": (
                f"No spread or moneyline is posted for this game yet, so the model's {_spread_text(home, away, spread_home)} "
                f"({analysis['probabilities']['home_win']:.1%} {home} / {analysis['probabilities']['away_win']:.1%} {away}) is unchecked. "
                f"Paste a sportsbook price to see the value gap, and expect a disagreement over 3 points to signal missing information."
            ),
            "tone": "neutral", "team": None,
        })

    # 8. Sample size and data quality.
    avg_games = (h["games_current"] + a["games_current"]) / 2.0
    w_home, w_away = h["blend_weight_current"], a["blend_weight_current"]
    meta = ratings.get("meta") or {}
    prior = meta.get("prior_season")
    season = meta.get("season")
    sample_detail = (
        f"{home} has {h['games_current']} and {away} {a['games_current']} {season if season else 'current-season'} games in the ratings "
        f"(current-season weight {w_home:.0%} and {w_away:.0%}); the rest comes from {prior if prior else 'last season'}, "
        f"regressed {int((1 - PRIOR_CARRYOVER) * 100)}% toward average. "
        f"Confidence {analysis['confidence']:.2f} ({analysis['confidence_label']}) = margin {analysis['confidence_factors']['margin']:.2f} x sample "
        f"{analysis['confidence_factors']['sample']:.2f} x quarterback {analysis['confidence_factors']['qb']:.2f} x market {analysis['confidence_factors']['market']:.2f}."
    )
    items.append({
        "title": "Sample size and data quality",
        "detail": sample_detail,
        "tone": "con" if avg_games < 4 else "neutral", "team": None,
    })

    # 9. Quarterback risk (only when a starter is a backup, unproven, or unknown).
    flagged = [(code, qb[side]) for side, code in (("home", home), ("away", away)) if qb[side]["role"] in ("backup", "unknown")]
    if flagged:
        detail = "; ".join(
            f"{code} lists {info['name']} ({info['record'].get('attempts', 0)} career attempts in the data), valued at {info['impact']:+.1f} points vs an average starter"
            for code, info in flagged
        )
        hurt_team = flagged[0][0]
        items.append({
            "title": "Quarterback risk",
            "detail": detail + f". A backup or unproven starter makes this projection less reliable than the model's {MARGIN_STDEV:g}-point margin stdev suggests.",
            "tone": "con", "team": hurt_team,
        })
    return items


def evaluate_moneyline_price(analysis: Mapping[str, Any], side: str, price: float, *, stake: float = 100.0) -> dict[str, Any]:
    """Compare the model's win probability for ``side`` with a pasted American ``price``.

    Delegates to :func:`modules.odds_input.evaluate_price` (so the numbers match
    the other betting pages) and adds ``side``, ``team``,
    ``market_probability_novig`` (the de-vigged market probability for that
    side when the analysis has both market moneylines, else ``None``) and
    ``edge_vs_market_novig`` (model probability minus that, or ``None``).
    Raises ``ValueError`` for an unknown ``side``, a price that is not a finite
    American price (|price| of at least 100; 0, NaN, inf and values like -50
    do not exist) or a negative / non-finite ``stake`` -- parse user text with
    :func:`modules.odds_input.parse_odds` first.  Without this guard a NaN or
    ``+50`` would flow through the price math and hand the page NaN or a
    fictional "implied probability".
    """
    if side not in ("home", "away"):
        raise ValueError(f"side must be 'home' or 'away', got {side!r}")
    clean_price, clean_stake = _num(price), _num(stake)
    if clean_price is None or abs(clean_price) < 100.0:
        raise ValueError(f"price must be a finite American price of at least 100 in size (e.g. -110, +150), got {price!r}")
    if clean_stake is None or clean_stake < 0.0:
        raise ValueError(f"stake must be a finite, non-negative amount, got {stake!r}")
    price, stake = clean_price, clean_stake
    probability = analysis["probabilities"]["home_win" if side == "home" else "away_win"]
    result = evaluate_price(probability, price, stake=stake)
    novig = analysis["moneyline"].get("market_home_prob_novig" if side == "home" else "market_away_prob_novig")
    result.update({
        "side": side,
        "team": analysis["home_team"] if side == "home" else analysis["away_team"],
        "market_probability_novig": novig,
        "edge_vs_market_novig": round(probability - novig, 4) if novig is not None else None,
    })
    return result


def load_moneyline_inputs(season: int | None = None, today: date | None = None) -> dict[str, Any]:
    """Convenience bundle for the page: ``{"season", "schedule", "ratings", "qb_context", "warnings"}``.

    ``season`` defaults to :func:`current_season`.  Raises
    :class:`ScheduleUnavailable` when the schedule cannot be loaded; ratings and
    quarterback context fail soft (see their docstrings).
    """
    season = int(season) if season is not None else current_season(today)
    schedule = load_schedule(season)
    ratings = load_team_ratings(season)
    qb_context = load_qb_context(season)
    warnings = list(dict.fromkeys(list(ratings["meta"]["warnings"]) + list(qb_context["meta"]["warnings"])))
    return {"season": season, "schedule": schedule, "ratings": ratings, "qb_context": qb_context, "warnings": warnings}


__all__ = [
    "BLEND_K_GAMES",
    "EFFICIENCY_WEIGHT",
    "HOME_FIELD_POINTS",
    "MARGIN_STDEV",
    "WIN_PROB_STDEV",
    "PRIOR_CARRYOVER",
    "SCHEDULE_COLUMNS",
    "ScheduleUnavailable",
    "analyze_game",
    "blend_ratings",
    "build_qb_context",
    "clear_caches",
    "current_season",
    "default_week",
    "evaluate_moneyline_price",
    "games_for_week",
    "load_moneyline_inputs",
    "load_qb_context",
    "load_schedule",
    "load_team_ratings",
    "normalize_schedule",
    "normalize_team",
    "qb_impact_points",
    "resolve_qb",
    "team_name",
    "weeks_in_schedule",
]
