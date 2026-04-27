"""
Technical Analysis Agent for AgentSaham.

Analyzes stock price action using technical indicators and chart patterns
to produce a BUY/SELL/HOLD signal with confidence and reasoning.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from tools.chart_analyzer import ChartAnalyzer
from tools.data_fetcher import DataFetcher


class TechnicalAgent:
    """
    Technical analysis agent.

    Performs chart-based analysis using RSI, MACD, Bollinger Bands,
    Moving Averages, candlestick patterns, support/resistance, and trend
    detection. Optionally incorporates video transcript context.

    Attributes:
        data_fetcher: DataFetcher instance for OHLCV retrieval.
        chart_analyzer: ChartAnalyzer instance for indicator computation.
    """

    def __init__(
        self,
        data_fetcher: Optional[DataFetcher] = None,
        chart_analyzer: Optional[ChartAnalyzer] = None,
    ) -> None:
        """
        Initialize TechnicalAgent.

        Args:
            data_fetcher: Optional DataFetcher; creates a new one if not provided.
            chart_analyzer: Optional ChartAnalyzer; creates a new one if not provided.
        """
        self.data_fetcher = data_fetcher or DataFetcher()
        self.chart_analyzer = chart_analyzer or ChartAnalyzer()

    async def analyze(self, ticker: str, transcript: Optional[str] = None) -> dict:
        """
        Perform technical analysis on a stock ticker.

        Args:
            ticker: Stock ticker symbol (IDX tickers auto-suffixed with .JK).
            transcript: Optional video transcript text for additional context.

        Returns:
            Dict with keys:
                - signal: "BUY", "SELL", or "HOLD"
                - confidence: float 0.0–1.0
                - reasoning: str explanation
                - entry_point: suggested entry price
                - target: price target
                - stop_loss: stop-loss price
                - indicators: raw indicator values
                - trend: detected trend direction
        """
        logger.info(f"[TechnicalAgent] Analyzing {ticker}")

        # Fetch data
        df = self.data_fetcher.get_ohlcv(ticker, period="6mo", interval="1d")

        # Run chart analysis
        analysis = self.chart_analyzer.analyze(df)
        indicators = analysis["indicators"]
        patterns = analysis["patterns"]
        sr_levels = analysis["support_resistance"]
        trend = analysis["trend"]

        # Score-based decision
        score, reasoning_parts = self._score_indicators(indicators, patterns, trend)

        # Determine signal
        signal, confidence = self._score_to_signal(score)

        current_price = indicators.get("current_price") or float(df["Close"].iloc[-1])
        atr = indicators.get("atr") or (current_price * 0.02)

        entry_point = round(current_price, 2)
        if signal == "BUY":
            target = round(current_price + 2 * atr, 2)
            stop_loss = round(current_price - atr, 2)
        elif signal == "SELL":
            target = round(current_price - 2 * atr, 2)
            stop_loss = round(current_price + atr, 2)
        else:
            target = round(current_price + atr, 2)
            stop_loss = round(current_price - atr, 2)

        if transcript:
            reasoning_parts.append(f"Video transcript tersedia ({len(transcript)} karakter) untuk konteks tambahan.")

        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Sinyal teknikal campuran."

        result = {
            "agent": "technical",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "entry_point": entry_point,
            "target": target,
            "stop_loss": stop_loss,
            "indicators": indicators,
            "patterns": patterns,
            "support_resistance": sr_levels,
            "trend": trend,
        }

        logger.info(
            f"[TechnicalAgent] {ticker}: signal={signal}, confidence={confidence:.2f}, trend={trend}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_indicators(self, indicators: dict, patterns: dict, trend: str) -> Tuple[float, List[str]]:
        """
        Build a directional score (-1 bearish … +1 bullish) from indicators.

        Returns:
            Tuple of (score: float, reasoning_parts: List[str]).
        """
        score = 0.0
        parts: List[str] = []

        rsi = indicators.get("rsi")
        if rsi is not None:
            if rsi < 30:
                score += 0.3
                parts.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                score -= 0.3
                parts.append(f"RSI overbought ({rsi:.1f})")
            else:
                parts.append(f"RSI netral ({rsi:.1f})")

        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        macd_hist = indicators.get("macd_histogram")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal and (macd_hist or 0) > 0:
                score += 0.25
                parts.append("MACD bullish crossover")
            elif macd < macd_signal and (macd_hist or 0) < 0:
                score -= 0.25
                parts.append("MACD bearish crossover")

        sma20 = indicators.get("sma20")
        sma50 = indicators.get("sma50")
        price = indicators.get("current_price")
        if price and sma20 and sma50:
            if price > sma20 > sma50:
                score += 0.2
                parts.append("Harga di atas SMA20 > SMA50 (bullish)")
            elif price < sma20 < sma50:
                score -= 0.2
                parts.append("Harga di bawah SMA20 < SMA50 (bearish)")

        bb_lower = indicators.get("bb_lower")
        bb_upper = indicators.get("bb_upper")
        if price and bb_lower and bb_upper:
            if price <= bb_lower:
                score += 0.15
                parts.append("Harga menyentuh Bollinger Band bawah (potensi rebound)")
            elif price >= bb_upper:
                score -= 0.15
                parts.append("Harga menyentuh Bollinger Band atas (potensi reversal)")

        # Trend bonus
        if trend == "uptrend":
            score += 0.1
            parts.append("Tren keseluruhan: uptrend")
        elif trend == "downtrend":
            score -= 0.1
            parts.append("Tren keseluruhan: downtrend")

        # Candlestick patterns
        for pattern, value in patterns.items():
            if value > 0:
                score += 0.05
                parts.append(f"Pola candlestick bullish: {pattern}")
            elif value < 0:
                score -= 0.05
                parts.append(f"Pola candlestick bearish: {pattern}")

        return score, parts

    def _score_to_signal(self, score: float) -> Tuple[str, float]:
        """
        Convert a directional score to a BUY/SELL/HOLD signal and confidence.

        Args:
            score: Float in range ~[-1, +1].

        Returns:
            Tuple of (signal: str, confidence: float 0–1).
        """
        score = max(-1.0, min(1.0, score))
        abs_score = abs(score)
        confidence = min(0.95, 0.5 + abs_score * 0.5)

        if score >= 0.25:
            return "BUY", round(confidence, 2)
        if score <= -0.25:
            return "SELL", round(confidence, 2)
        return "HOLD", round(0.5 + abs_score * 0.3, 2)
