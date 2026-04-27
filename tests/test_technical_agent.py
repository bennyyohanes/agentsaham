"""
Tests for TechnicalAgent.

Tests validate signal generation, indicator scoring, and edge-case handling
without requiring live network access (uses mocked data fetcher and analyzer).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from agents.technical_agent import TechnicalAgent


def _make_ohlcv_df(rows: int = 60) -> pd.DataFrame:
    """Create a simple synthetic OHLCV DataFrame for testing."""
    import numpy as np

    rng = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = 10000 + np.cumsum(np.random.randn(rows) * 50)
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, rows).astype(float),
        },
        index=rng,
    )
    return df


class TestTechnicalAgent:
    """Unit tests for TechnicalAgent."""

    def _make_agent(self, df: pd.DataFrame) -> TechnicalAgent:
        """Create a TechnicalAgent with a mocked DataFetcher."""
        mock_fetcher = MagicMock()
        mock_fetcher.get_ohlcv.return_value = df
        return TechnicalAgent(data_fetcher=mock_fetcher)

    def test_analyze_returns_required_keys(self):
        """analyze() must return all required keys."""
        df = _make_ohlcv_df()
        agent = self._make_agent(df)
        result = asyncio.run(agent.analyze("BBCA"))

        required = {"agent", "signal", "confidence", "reasoning", "entry_point", "target", "stop_loss"}
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_signal_is_valid_value(self):
        """Signal must be one of BUY, SELL, HOLD."""
        df = _make_ohlcv_df()
        agent = self._make_agent(df)
        result = asyncio.run(agent.analyze("BBCA"))
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_is_between_0_and_1(self):
        """Confidence must be in [0, 1]."""
        df = _make_ohlcv_df()
        agent = self._make_agent(df)
        result = asyncio.run(agent.analyze("BBCA"))
        assert 0.0 <= result["confidence"] <= 1.0

    def test_entry_point_and_target_are_positive(self):
        """Entry point, target, and stop loss should be positive numbers."""
        df = _make_ohlcv_df()
        agent = self._make_agent(df)
        result = asyncio.run(agent.analyze("BBCA"))
        assert result["entry_point"] > 0
        assert result["target"] > 0
        assert result["stop_loss"] > 0

    def test_transcript_is_included_in_reasoning(self):
        """Reasoning should mention transcript when provided."""
        df = _make_ohlcv_df()
        agent = self._make_agent(df)
        result = asyncio.run(agent.analyze("BBCA", transcript="harga naik terus mantap"))
        assert "transkrip" in result["reasoning"].lower() or "transcript" in result["reasoning"].lower()

    def test_score_to_signal_buy(self):
        """Positive score >= 0.25 should produce BUY."""
        agent = TechnicalAgent()
        signal, confidence = agent._score_to_signal(0.5)
        assert signal == "BUY"
        assert confidence > 0.5

    def test_score_to_signal_sell(self):
        """Negative score <= -0.25 should produce SELL."""
        agent = TechnicalAgent()
        signal, confidence = agent._score_to_signal(-0.5)
        assert signal == "SELL"
        assert confidence > 0.5

    def test_score_to_signal_hold(self):
        """Score near zero should produce HOLD."""
        agent = TechnicalAgent()
        signal, _ = agent._score_to_signal(0.0)
        assert signal == "HOLD"

    def test_agent_name_is_technical(self):
        """agent field must be 'technical'."""
        df = _make_ohlcv_df()
        agent = self._make_agent(df)
        result = asyncio.run(agent.analyze("BBCA"))
        assert result["agent"] == "technical"
