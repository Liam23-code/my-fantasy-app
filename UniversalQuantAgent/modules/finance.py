"""Financial market data collection and beginner-friendly analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from modules.utils import clean_dataframe, safe_number


def fetch_stock_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download daily price history from Yahoo Finance via ``yfinance``.

    This function is deliberately separate from analysis. A future version can
    replace Yahoo with a database or paid feed without changing the indicators.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("A stock ticker is required.")

    # auto_adjust gives split/dividend-adjusted OHLC prices.
    history = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if history.empty:
        raise ValueError(f"No market data was returned for {symbol}.")

    # yfinance can return two-level column names; one ticker only needs level 1.
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    return clean_dataframe(history).sort_index()


def add_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """Add returns, volatility, volume change, and moving averages."""
    data = history.copy()

    # Daily percentage returns are the foundation for volatility calculations.
    data["Daily Return"] = data["Close"].pct_change()

    # Twenty trading days is approximately one market month. Multiplying by
    # sqrt(252) converts daily movement to familiar annualized volatility.
    data["Volatility 20D"] = data["Daily Return"].rolling(20).std() * (252**0.5)

    # Short and long moving averages reveal the direction of recent prices.
    data["SMA 20"] = data["Close"].rolling(20).mean()
    data["SMA 50"] = data["Close"].rolling(50).mean()
    data["Volume Change"] = data["Volume"].pct_change()
    data["Average Volume 20D"] = data["Volume"].rolling(20).mean()
    return clean_dataframe(data)


def calculate_correlations(data: pd.DataFrame) -> dict[str, float]:
    """Measure simple relationships between price, volume, and volatility."""
    columns = ["Close", "Volume", "Daily Return", "Volatility 20D", "SMA 20", "SMA 50"]
    available = [column for column in columns if column in data.columns]
    correlation = data[available].corr()

    # A small named result is easier for callers to use than a full matrix.
    pairs = {
        "price_volume": ("Close", "Volume"),
        "return_volume_change": ("Daily Return", "Volume Change"),
        "price_volatility": ("Close", "Volatility 20D"),
    }
    results: dict[str, float] = {}
    for name, (left, right) in pairs.items():
        if left in data.columns and right in data.columns:
            results[name] = round(safe_number(data[left].corr(data[right])), 4)
    return results


def detect_patterns(data: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect trend and volatility patterns using transparent rules."""
    if len(data) < 50:
        return [{"name": "insufficient_history", "strength": 0.0,
                 "description": "At least 50 trading days are needed for pattern detection."}]

    latest = data.iloc[-1]
    patterns: list[dict[str, Any]] = []
    close = safe_number(latest["Close"])
    sma20 = safe_number(latest["SMA 20"])
    sma50 = safe_number(latest["SMA 50"])

    # Moving-average alignment is a deliberately simple trend definition.
    if close > sma20 > sma50:
        separation = ((close - sma50) / sma50 * 100) if sma50 else 0
        patterns.append({"name": "uptrend", "strength": min(abs(separation), 20) / 20,
                         "description": "Price is above both rising-period moving averages."})
    elif close < sma20 < sma50:
        separation = ((sma50 - close) / sma50 * 100) if sma50 else 0
        patterns.append({"name": "downtrend", "strength": min(abs(separation), 20) / 20,
                         "description": "Price is below both moving averages."})
    else:
        patterns.append({"name": "mixed_trend", "strength": 0.25,
                         "description": "Price and moving averages are not clearly aligned."})

    # Flag volatility when today is far above its own recent baseline.
    current_volatility = safe_number(latest["Volatility 20D"])
    volatility_baseline = safe_number(data["Volatility 20D"].tail(60).median())
    if volatility_baseline and current_volatility > volatility_baseline * 1.5:
        ratio = current_volatility / volatility_baseline
        patterns.append({"name": "volatility_spike", "strength": min((ratio - 1) / 2, 1),
                         "description": "Recent volatility is well above its 60-day baseline."})
    return patterns


def analyze_stock(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Run the complete finance pipeline and return serializable insights."""
    symbol = ticker.strip().upper()
    history = fetch_stock_history(symbol, period)
    data = add_indicators(history)
    latest = data.iloc[-1]
    first_close = safe_number(data["Close"].iloc[0])
    latest_close = safe_number(latest["Close"])
    period_return = ((latest_close / first_close) - 1) * 100 if first_close else 0.0

    return {
        "domain": "finance",
        "subject": symbol,
        "summary": {
            "latest_price": round(latest_close, 2),
            "period_return_pct": round(period_return, 2),
            "average_volume_20d": round(safe_number(latest["Average Volume 20D"]), 0),
            "annualized_volatility_pct": round(safe_number(latest["Volatility 20D"]) * 100, 2),
            "sma_20": round(safe_number(latest["SMA 20"]), 2),
            "sma_50": round(safe_number(latest["SMA 50"]), 2),
        },
        "correlations": calculate_correlations(data),
        "patterns": detect_patterns(data),
        "data_points": len(data),
    }

