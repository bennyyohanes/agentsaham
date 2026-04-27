"""
Transaction Analysis Agent for AgentSaham.

Detects unusual trading activity, accumulation/distribution patterns,
foreign fund flows, and bid/ask pressure to identify big player movements.
"""

from typing import Optional

from loguru import logger

from tools.data_fetcher import DataFetcher


class TransactionAgent:
    """
    Transaction and volume analysis agent.

    Analyzes:
    - Volume anomalies (sudden spikes relative to historical average)
    - Accumulation vs. distribution signals (On-Balance Volume proxy)
    - Price–volume divergence
    - Foreign net flow via yfinance (where available)

    Attributes:
        data_fetcher: DataFetcher instance for data retrieval.
    """

    def __init__(self, data_fetcher: Optional[DataFetcher] = None) -> None:
        """
        Initialize TransactionAgent.

        Args:
            data_fetcher: Optional DataFetcher; creates a new one if not provided.
        """
        self.data_fetcher = data_fetcher or DataFetcher()

    async def analyze(self, ticker: str) -> dict:
        """
        Perform transaction and volume analysis on a stock ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with keys:
                - signal: "BUY", "SELL", or "HOLD"
                - confidence: float 0.0–1.0
                - reasoning: str explanation
                - unusual_activity: bool, True if anomaly detected
                - volume_data: raw volume metrics
        """
        logger.info(f"[TransactionAgent] Analyzing {ticker}")

        volume_data = self.data_fetcher.get_volume_data(ticker, period="3mo")
        ohlcv_df = self.data_fetcher.get_ohlcv(ticker, period="3mo", interval="1d")

        score, reasoning_parts, unusual_activity = self._analyze_volume(
            ohlcv_df, volume_data
        )

        signal, confidence = self._score_to_signal(score)
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Aktivitas transaksi normal."

        result = {
            "agent": "transaction",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "unusual_activity": unusual_activity,
            "volume_data": volume_data,
        }

        logger.info(
            f"[TransactionAgent] {ticker}: signal={signal}, "
            f"confidence={confidence:.2f}, unusual={unusual_activity}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_volume(self, df, volume_data: dict) -> tuple[float, list[str], bool]:
        """
        Analyze volume patterns to produce a directional score.

        Args:
            df: OHLCV DataFrame.
            volume_data: Pre-computed volume statistics dict.

        Returns:
            Tuple of (score, reasoning_parts, unusual_activity).
        """
        score = 0.0
        parts: list[str] = []
        unusual = False

        # 1. Volume anomaly check
        z_score = volume_data.get("volume_z_score", 0.0)
        is_unusual = volume_data.get("is_unusual_volume", False)
        price_change_pct = volume_data.get("price_change_pct", 0.0)

        if is_unusual:
            unusual = True
            if price_change_pct > 0:
                score += 0.3
                parts.append(
                    f"Volume tidak biasa (z={z_score:.1f}σ) disertai kenaikan harga "
                    f"+{price_change_pct:.1f}% → sinyal akumulasi"
                )
            else:
                score -= 0.3
                parts.append(
                    f"Volume tidak biasa (z={z_score:.1f}σ) disertai penurunan harga "
                    f"{price_change_pct:.1f}% → sinyal distribusi"
                )

        # 2. Volume trend
        vol_trend = volume_data.get("volume_trend", "stable")
        if vol_trend == "increasing":
            if price_change_pct >= 0:
                score += 0.15
                parts.append("Tren volume meningkat + harga naik → akumulasi potensial")
            else:
                score -= 0.15
                parts.append("Tren volume meningkat + harga turun → distribusi potensial")
        elif vol_trend == "decreasing":
            parts.append("Tren volume menurun → konsolidasi atau kehilangan minat")

        # 3. On-Balance Volume (OBV) direction using raw OHLCV
        obv_signal = self._compute_obv_signal(df)
        if obv_signal == "bullish":
            score += 0.2
            parts.append("OBV naik → tekanan beli (akumulasi big player)")
        elif obv_signal == "bearish":
            score -= 0.2
            parts.append("OBV turun → tekanan jual (distribusi big player)")

        # 4. Price-volume divergence
        divergence = self._detect_divergence(df)
        if divergence == "bullish_divergence":
            score += 0.15
            parts.append("Divergensi bullish: harga turun tapi volume jual menurun")
        elif divergence == "bearish_divergence":
            score -= 0.15
            parts.append("Divergensi bearish: harga naik tapi volume beli menurun")

        return score, parts, unusual

    def _compute_obv_signal(self, df) -> str:
        """
        Compute a simplified On-Balance Volume trend signal.

        Returns:
            "bullish", "bearish", or "neutral".
        """
        if df is None or len(df) < 10:
            return "neutral"

        obv = 0.0
        obv_series = []
        for i in range(1, len(df)):
            close = float(df["Close"].iloc[i])
            prev_close = float(df["Close"].iloc[i - 1])
            vol = float(df["Volume"].iloc[i])
            if close > prev_close:
                obv += vol
            elif close < prev_close:
                obv -= vol
            obv_series.append(obv)

        if len(obv_series) < 5:
            return "neutral"

        recent_obv = sum(obv_series[-5:]) / 5
        older_obv = sum(obv_series[-10:-5]) / 5 if len(obv_series) >= 10 else obv_series[0]

        if recent_obv > older_obv * 1.05:
            return "bullish"
        if recent_obv < older_obv * 0.95:
            return "bearish"
        return "neutral"

    def _detect_divergence(self, df) -> str:
        """
        Detect simple price-volume divergence over the last 10 periods.

        Returns:
            "bullish_divergence", "bearish_divergence", or "none".
        """
        if df is None or len(df) < 10:
            return "none"

        recent = df.tail(10)
        price_trend = float(recent["Close"].iloc[-1]) - float(recent["Close"].iloc[0])
        volume_trend = float(recent["Volume"].iloc[-1]) - float(recent["Volume"].iloc[0])

        # Bullish divergence: price falling but sell volume also falling
        if price_trend < 0 and volume_trend < 0:
            return "bullish_divergence"
        # Bearish divergence: price rising but buy volume falling
        if price_trend > 0 and volume_trend < 0:
            return "bearish_divergence"
        return "none"

    def _score_to_signal(self, score: float) -> tuple[str, float]:
        """
        Convert directional score to signal and confidence.

        Args:
            score: Float in range ~[-1, +1].

        Returns:
            Tuple of (signal: str, confidence: float 0–1).
        """
        score = max(-1.0, min(1.0, score))
        abs_score = abs(score)
        confidence = min(0.90, 0.4 + abs_score * 0.6)

        if score >= 0.2:
            return "BUY", round(confidence, 2)
        if score <= -0.2:
            return "SELL", round(confidence, 2)
        return "HOLD", round(0.4 + abs_score * 0.3, 2)
