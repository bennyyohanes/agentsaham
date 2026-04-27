"""
Chart analyzer tool for AgentSaham.

Computes technical indicators using pandas-ta, detects candlestick patterns,
identifies support/resistance levels, and determines trend direction.
"""

from typing import Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger


def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Compute all technical indicators for the given OHLCV DataFrame.

    Calculates:
    - RSI (14)
    - MACD (12, 26, 9)
    - Bollinger Bands (20, 2)
    - Moving Averages: SMA20, SMA50, SMA200, EMA9, EMA21
    - Stochastic Oscillator (14, 3)
    - ATR (14) for volatility

    Args:
        df: DataFrame with columns Open, High, Low, Close, Volume.

    Returns:
        Dict with latest values for each indicator.
    """
    if df.empty or len(df) < 20:
        logger.warning("Insufficient data for indicator calculation (need at least 20 rows)")
        return {}

    df = df.copy()

    # RSI
    rsi = ta.rsi(df["Close"], length=14)
    latest_rsi = _safe_float(rsi.iloc[-1]) if rsi is not None and not rsi.empty else None

    # MACD
    macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        macd_val = _safe_float(macd_df.iloc[-1, 0])    # MACD line
        macd_signal = _safe_float(macd_df.iloc[-1, 1])  # Signal line
        macd_hist = _safe_float(macd_df.iloc[-1, 2])    # Histogram
    else:
        macd_val = macd_signal = macd_hist = None

    # Bollinger Bands
    bb_df = ta.bbands(df["Close"], length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        bb_lower = _safe_float(bb_df.iloc[-1, 0])   # BBL
        bb_mid = _safe_float(bb_df.iloc[-1, 1])     # BBM
        bb_upper = _safe_float(bb_df.iloc[-1, 2])   # BBU
        bb_bandwidth = _safe_float(bb_df.iloc[-1, 3]) if bb_df.shape[1] > 3 else None
        bb_percent = _safe_float(bb_df.iloc[-1, 4]) if bb_df.shape[1] > 4 else None
    else:
        bb_lower = bb_mid = bb_upper = bb_bandwidth = bb_percent = None

    # Moving Averages
    sma20 = ta.sma(df["Close"], length=20)
    sma50 = ta.sma(df["Close"], length=50) if len(df) >= 50 else None
    sma200 = ta.sma(df["Close"], length=200) if len(df) >= 200 else None
    ema9 = ta.ema(df["Close"], length=9)
    ema21 = ta.ema(df["Close"], length=21)

    # Stochastic
    stoch_df = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    if stoch_df is not None and not stoch_df.empty:
        stoch_k = _safe_float(stoch_df.iloc[-1, 0])
        stoch_d = _safe_float(stoch_df.iloc[-1, 1])
    else:
        stoch_k = stoch_d = None

    # ATR
    atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    latest_atr = _safe_float(atr.iloc[-1]) if atr is not None and not atr.empty else None

    current_price = _safe_float(df["Close"].iloc[-1])

    return {
        "current_price": current_price,
        "rsi": latest_rsi,
        "macd": macd_val,
        "macd_signal": macd_signal,
        "macd_histogram": macd_hist,
        "bb_lower": bb_lower,
        "bb_mid": bb_mid,
        "bb_upper": bb_upper,
        "bb_bandwidth": bb_bandwidth,
        "bb_percent": bb_percent,
        "sma20": _safe_float(sma20.iloc[-1]) if sma20 is not None and not sma20.empty else None,
        "sma50": _safe_float(sma50.iloc[-1]) if sma50 is not None and not sma50.empty else None,
        "sma200": _safe_float(sma200.iloc[-1]) if sma200 is not None and not sma200.empty else None,
        "ema9": _safe_float(ema9.iloc[-1]) if ema9 is not None and not ema9.empty else None,
        "ema21": _safe_float(ema21.iloc[-1]) if ema21 is not None and not ema21.empty else None,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "atr": latest_atr,
    }


def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """
    Detect common candlestick patterns in the last few candles.

    Checks for: Doji, Inside bar, and generic named patterns via
    pandas-ta's cdl_pattern() interface.

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        Dict mapping pattern names to int values (positive=bullish, negative=bearish).
    """
    if len(df) < 3:
        return {}

    patterns = {}

    # pandas-ta has cdl_doji and cdl_inside as standalone functions
    standalone = {
        "doji": ta.cdl_doji,
        "inside": ta.cdl_inside,
    }

    for name, func in standalone.items():
        try:
            result = func(df["Open"], df["High"], df["Low"], df["Close"])
            if result is not None and not result.empty:
                last_val = result.iloc[-1]
                if last_val != 0:
                    patterns[name] = int(last_val)
        except Exception as exc:
            logger.debug(f"Candlestick pattern {name} unavailable: {exc}")

    # Use cdl_pattern for named TA-Lib patterns (requires TA-Lib; skip gracefully if absent)
    named_patterns = ["hammer", "shootingstar", "engulfing", "morningstar", "eveningstar"]
    for pat_name in named_patterns:
        try:
            result = ta.cdl_pattern(
                df["Open"], df["High"], df["Low"], df["Close"], name=pat_name
            )
            if result is not None and not result.empty:
                last_val = result.iloc[-1]
                if last_val != 0:
                    patterns[pat_name] = int(last_val)
        except Exception as exc:
            logger.debug(f"cdl_pattern '{pat_name}' unavailable: {exc}")

    return patterns


def identify_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Identify support and resistance levels using pivot points.

    Uses the classic pivot point formula on the most recent candle
    and also finds swing highs/lows within the given window.

    Args:
        df: DataFrame with OHLCV columns.
        window: Lookback window for swing detection.

    Returns:
        Dict with keys: pivot, r1, r2, s1, s2, swing_highs, swing_lows.
    """
    if df.empty:
        return {}

    last = df.iloc[-1]
    high = float(last["High"])
    low = float(last["Low"])
    close = float(last["Close"])

    # Classic pivot points
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)

    # Swing highs/lows
    recent = df.tail(window)
    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())

    return {
        "pivot": round(pivot, 2),
        "resistance_1": round(r1, 2),
        "resistance_2": round(r2, 2),
        "support_1": round(s1, 2),
        "support_2": round(s2, 2),
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
    }


def detect_trend(df: pd.DataFrame) -> str:
    """
    Detect the overall price trend based on moving averages.

    Uses SMA20 vs SMA50 relationship and price slope to classify trend.

    Args:
        df: DataFrame with at least 50 rows of Close prices.

    Returns:
        One of "uptrend", "downtrend", or "sideways".
    """
    if len(df) < 20:
        return "sideways"

    close = df["Close"]
    sma20 = ta.sma(close, length=20)
    sma50 = ta.sma(close, length=50) if len(df) >= 50 else None

    if sma20 is None or sma20.empty:
        return "sideways"

    latest_sma20 = float(sma20.iloc[-1])
    prev_sma20 = float(sma20.iloc[-5]) if len(sma20) > 5 else float(sma20.iloc[0])
    current_price = float(close.iloc[-1])

    if sma50 is not None and not sma50.empty:
        latest_sma50 = float(sma50.iloc[-1])
        if current_price > latest_sma20 > latest_sma50:
            return "uptrend"
        if current_price < latest_sma20 < latest_sma50:
            return "downtrend"
    else:
        slope = (latest_sma20 - prev_sma20) / prev_sma20 if prev_sma20 else 0
        if slope > 0.02:
            return "uptrend"
        if slope < -0.02:
            return "downtrend"

    return "sideways"


class ChartAnalyzer:
    """
    High-level chart analysis interface.

    Combines indicator computation, pattern detection, support/resistance
    identification, and trend detection into a single convenient class.
    """

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Perform full chart analysis on an OHLCV DataFrame.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume.

        Returns:
            Dict with keys: indicators, patterns, support_resistance, trend.
        """
        logger.info("Running full chart analysis")

        indicators = compute_indicators(df)
        patterns = detect_candlestick_patterns(df)
        sr_levels = identify_support_resistance(df)
        trend = detect_trend(df)

        result = {
            "indicators": indicators,
            "patterns": patterns,
            "support_resistance": sr_levels,
            "trend": trend,
        }

        logger.debug(f"Chart analysis complete: trend={trend}, rsi={indicators.get('rsi')}")
        return result
