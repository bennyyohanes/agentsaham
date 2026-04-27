"""
Tests for FundamentalAgent.

Tests validate signal generation, Gemini response parsing, and fallback
behavior without making real API calls.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.fundamental_agent import FundamentalAgent


def _make_fundamentals(per=15.0, pbv=2.0, roe=0.18, eps=500.0) -> dict:
    """Return a minimal fundamentals dict for testing."""
    return {
        "ticker": "BBCA.JK",
        "company_name": "Bank Central Asia Tbk.",
        "sector": "Financial Services",
        "per": per,
        "pbv": pbv,
        "roe": roe,
        "eps": eps,
        "dividend_yield": 0.025,
        "debt_to_equity": 50.0,
        "current_price": 8500,
        "target_mean_price": 9500,
        "analyst_recommendation": "buy",
    }


class TestFundamentalAgent:
    """Unit tests for FundamentalAgent."""

    def _make_agent(self, fundamentals: dict, gemini_text: str = "") -> FundamentalAgent:
        """Create a FundamentalAgent with mocked dependencies."""
        mock_fetcher = MagicMock()
        mock_fetcher.get_fundamentals.return_value = fundamentals
        mock_fetcher.get_news.return_value = []

        agent = FundamentalAgent(gemini_api_key="fake-key", data_fetcher=mock_fetcher)
        # Patch the Gemini call
        agent._call_gemini = AsyncMock(return_value=gemini_text)
        return agent

    def test_analyze_returns_required_keys(self):
        """analyze() should return all required output keys."""
        fundamentals = _make_fundamentals()
        gemini_resp = json.dumps({
            "signal": "BUY",
            "confidence": 0.75,
            "valuation": "undervalued",
            "reasoning": "Saham murah dengan ROE tinggi.",
        })
        agent = self._make_agent(fundamentals, gemini_resp)
        result = asyncio.run(agent.analyze("BBCA"))

        required = {"agent", "signal", "confidence", "reasoning", "valuation", "fundamentals"}
        assert required.issubset(result.keys())

    def test_signal_is_valid(self):
        """Signal must be BUY, SELL, or HOLD."""
        fundamentals = _make_fundamentals()
        gemini_resp = json.dumps({"signal": "HOLD", "confidence": 0.5, "valuation": "fairly_valued", "reasoning": "ok"})
        agent = self._make_agent(fundamentals, gemini_resp)
        result = asyncio.run(agent.analyze("BBCA"))
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_range(self):
        """Confidence must be between 0 and 1."""
        fundamentals = _make_fundamentals()
        gemini_resp = json.dumps({"signal": "BUY", "confidence": 0.8, "valuation": "undervalued", "reasoning": "ok"})
        agent = self._make_agent(fundamentals, gemini_resp)
        result = asyncio.run(agent.analyze("BBCA"))
        assert 0.0 <= result["confidence"] <= 1.0

    def test_gemini_json_parsing(self):
        """Should correctly parse valid JSON from Gemini response."""
        fundamentals = _make_fundamentals()
        payload = {"signal": "SELL", "confidence": 0.65, "valuation": "overvalued", "reasoning": "PER terlalu tinggi"}
        gemini_resp = "Berikut analisa:\n" + json.dumps(payload)
        agent = self._make_agent(fundamentals, gemini_resp)
        result = asyncio.run(agent.analyze("BBCA"))
        assert result["signal"] == "SELL"
        assert result["confidence"] == pytest.approx(0.65)

    def test_fallback_when_gemini_returns_garbage(self):
        """When Gemini response is unparseable, fallback analysis should be used."""
        fundamentals = _make_fundamentals(per=5.0, roe=0.25)  # cheap stock
        agent = self._make_agent(fundamentals, "Saya tidak bisa menganalisa ini.")
        result = asyncio.run(agent.analyze("BBCA"))
        # Fallback should still return valid output
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_fallback_high_per_suggests_sell(self):
        """Very high PER in fallback should lean toward SELL."""
        fundamentals = _make_fundamentals(per=60.0, roe=0.02)  # expensive + low ROE
        agent = self._make_agent(fundamentals, "{invalid json")
        result = asyncio.run(agent.analyze("BBCA"))
        # Should be SELL or HOLD due to expensive valuation
        assert result["signal"] in ("SELL", "HOLD")

    def test_agent_name_is_fundamental(self):
        """agent field must be 'fundamental'."""
        fundamentals = _make_fundamentals()
        gemini_resp = json.dumps({"signal": "BUY", "confidence": 0.6, "valuation": "undervalued", "reasoning": "ok"})
        agent = self._make_agent(fundamentals, gemini_resp)
        result = asyncio.run(agent.analyze("BBCA"))
        assert result["agent"] == "fundamental"
