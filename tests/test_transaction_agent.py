"""
Tests for TransactionAgent.

Tests validate volume analysis, OBV signal computation, divergence detection,
and signal generation without requiring live network access.
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from agents.transaction_agent import TransactionAgent


def _make_ohlcv_df(rows: int = 60, trend: str = "flat") -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    rng = pd.date_range("2024-01-01", periods=rows, freq="D")
    if trend == "up":
        close = 10000 + np.arange(rows) * 20 + np.random.randn(rows) * 5
    elif trend == "down":
        close = 10000 - np.arange(rows) * 20 + np.random.randn(rows) * 5
    else:
        close = 10000 + np.random.randn(rows) * 50

    volume = np.random.randint(1_000_000, 5_000_000, rows).astype(float)
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": volume,
        },
        index=rng,
    )
    return df


def _make_volume_data(unusual: bool = False, z_score: float = 0.5, price_change: float = 1.0) -> dict:
    """Create a minimal volume data dict."""
    return {
        "ticker": "BBCA.JK",
        "latest_volume": 8_000_000 if unusual else 2_000_000,
        "avg_volume_3mo": 2_000_000,
        "volume_z_score": z_score,
        "is_unusual_volume": unusual,
        "volume_trend": "stable",
        "latest_close": 8500.0,
        "price_change_pct": price_change,
    }


class TestTransactionAgent:
    """Unit tests for TransactionAgent."""

    def _make_agent(self, df: pd.DataFrame, volume_data: dict) -> TransactionAgent:
        """Create a TransactionAgent with mocked DataFetcher."""
        mock_fetcher = MagicMock()
        mock_fetcher.get_ohlcv.return_value = df
        mock_fetcher.get_volume_data.return_value = volume_data
        return TransactionAgent(data_fetcher=mock_fetcher)

    def test_analyze_returns_required_keys(self):
        """analyze() must return all required keys."""
        df = _make_ohlcv_df()
        vol_data = _make_volume_data()
        agent = self._make_agent(df, vol_data)
        result = asyncio.run(agent.analyze("BBCA"))

        required = {"agent", "signal", "confidence", "reasoning", "unusual_activity", "volume_data"}
        assert required.issubset(result.keys())

    def test_signal_is_valid(self):
        """Signal must be BUY, SELL, or HOLD."""
        df = _make_ohlcv_df()
        vol_data = _make_volume_data()
        agent = self._make_agent(df, vol_data)
        result = asyncio.run(agent.analyze("BBCA"))
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_range(self):
        """Confidence must be in [0, 1]."""
        df = _make_ohlcv_df()
        vol_data = _make_volume_data()
        agent = self._make_agent(df, vol_data)
        result = asyncio.run(agent.analyze("BBCA"))
        assert 0.0 <= result["confidence"] <= 1.0

    def test_unusual_volume_with_price_rise_gives_buy_tendency(self):
        """Unusual volume + price rise should lean toward BUY."""
        df = _make_ohlcv_df(trend="up")
        vol_data = _make_volume_data(unusual=True, z_score=2.5, price_change=3.0)
        agent = self._make_agent(df, vol_data)
        result = asyncio.run(agent.analyze("BBCA"))
        # With unusual volume + positive price change, signal should be BUY (not SELL)
        assert result["signal"] in ("BUY", "HOLD")
        assert result["unusual_activity"] is True

    def test_unusual_volume_with_price_drop_gives_sell_tendency(self):
        """Unusual volume + price drop should lean toward SELL."""
        df = _make_ohlcv_df(trend="down")
        vol_data = _make_volume_data(unusual=True, z_score=3.0, price_change=-3.0)
        agent = self._make_agent(df, vol_data)
        result = asyncio.run(agent.analyze("BBCA"))
        assert result["signal"] in ("SELL", "HOLD")
        assert result["unusual_activity"] is True

    def test_normal_volume_marks_unusual_false(self):
        """Normal volume should not trigger unusual_activity flag."""
        df = _make_ohlcv_df()
        vol_data = _make_volume_data(unusual=False, z_score=0.3)
        agent = self._make_agent(df, vol_data)
        result = asyncio.run(agent.analyze("BBCA"))
        assert result["unusual_activity"] is False

    def test_obv_signal_bullish(self):
        """OBV signal should be bullish for consistently rising close prices."""
        # Create a DataFrame where close always increases with high volume
        rows = 20
        rng = pd.date_range("2024-01-01", periods=rows, freq="D")
        close = np.arange(rows) * 10 + 10000
        volume = np.ones(rows) * 5_000_000
        df = pd.DataFrame(
            {"Open": close - 5, "High": close + 5, "Low": close - 10, "Close": close, "Volume": volume},
            index=rng,
        )
        agent = TransactionAgent()
        obv = agent._compute_obv_signal(df)
        assert obv == "bullish"

    def test_agent_name_is_transaction(self):
        """agent field must be 'transaction'."""
        df = _make_ohlcv_df()
        vol_data = _make_volume_data()
        agent = self._make_agent(df, vol_data)
        result = asyncio.run(agent.analyze("BBCA"))
        assert result["agent"] == "transaction"
