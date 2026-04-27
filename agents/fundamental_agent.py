"""
Fundamental Analysis Agent for AgentSaham.

Analyzes a stock's fundamental financial health and sentiment via
yfinance data and news scraping, then uses Gemini to generate a
structured BUY/SELL/HOLD recommendation.
"""

import json
from typing import Optional

from loguru import logger

from tools.data_fetcher import DataFetcher


class FundamentalAgent:
    """
    Fundamental analysis agent.

    Retrieves key financial ratios (PER, PBV, ROE, EPS, debt ratio,
    dividend yield) from yfinance and scrapes news from RSS feeds.
    Uses Google Gemini to produce a structured analysis and signal.

    Attributes:
        data_fetcher: DataFetcher instance for data retrieval.
        gemini_api_key: Google Gemini API key.
    """

    def __init__(
        self,
        gemini_api_key: str,
        data_fetcher: Optional[DataFetcher] = None,
        rss_feeds: Optional[list] = None,
    ) -> None:
        """
        Initialize FundamentalAgent.

        Args:
            gemini_api_key: Google Gemini API key (required).
            data_fetcher: Optional DataFetcher; creates a new one if not provided.
            rss_feeds: Optional list of RSS feed URLs for news scraping.
        """
        if not gemini_api_key:
            raise ValueError("gemini_api_key is required for FundamentalAgent")
        self.gemini_api_key = gemini_api_key
        self.data_fetcher = data_fetcher or DataFetcher()
        self.rss_feeds = rss_feeds

    async def analyze(self, ticker: str, transcript: Optional[str] = None) -> dict:
        """
        Perform fundamental analysis on a stock ticker.

        Args:
            ticker: Stock ticker symbol.
            transcript: Optional video transcript for extra context.

        Returns:
            Dict with keys:
                - signal: "BUY", "SELL", or "HOLD"
                - confidence: float 0.0–1.0
                - reasoning: str explanation
                - valuation: "undervalued", "fairly_valued", or "overvalued"
                - fundamentals: raw fundamental metrics
        """
        logger.info(f"[FundamentalAgent] Analyzing {ticker}")

        fundamentals = self.data_fetcher.get_fundamentals(ticker)
        news = self.data_fetcher.get_news(ticker, rss_feeds=self.rss_feeds, max_articles=5)

        prompt = self._build_prompt(ticker, fundamentals, news, transcript)
        gemini_response = await self._call_gemini(prompt)
        parsed = self._parse_gemini_response(gemini_response, fundamentals)

        result = {
            "agent": "fundamental",
            "signal": parsed["signal"],
            "confidence": parsed["confidence"],
            "reasoning": parsed["reasoning"],
            "valuation": parsed["valuation"],
            "fundamentals": fundamentals,
            "news_count": len(news),
        }

        logger.info(
            f"[FundamentalAgent] {ticker}: signal={result['signal']}, "
            f"confidence={result['confidence']:.2f}, valuation={result['valuation']}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        ticker: str,
        fundamentals: dict,
        news: list[dict],
        transcript: Optional[str],
    ) -> str:
        """Build the Gemini prompt for fundamental analysis."""
        news_text = "\n".join(
            f"- {a['title']} ({a.get('published', 'n/a')})" for a in news
        ) or "Tidak ada berita terbaru."

        transcript_section = ""
        if transcript:
            transcript_section = f"""
## Transkrip Video Analisis:
{transcript[:2000]}
"""

        per = fundamentals.get("per")
        pbv = fundamentals.get("pbv")
        roe = fundamentals.get("roe")
        eps = fundamentals.get("eps")
        dividend_yield = fundamentals.get("dividend_yield")
        debt_to_equity = fundamentals.get("debt_to_equity")

        return f"""Kamu adalah analis saham fundamental yang berpengalaman. Analisa saham berikut:

## Data Fundamental {ticker}:
- Perusahaan: {fundamentals.get('company_name', 'N/A')}
- Sektor: {fundamentals.get('sector', 'N/A')}
- PER (Price to Earnings): {per if per else 'N/A'}
- PBV (Price to Book Value): {pbv if pbv else 'N/A'}
- ROE (Return on Equity): {f'{roe*100:.1f}%' if roe else 'N/A'}
- EPS: {eps if eps else 'N/A'}
- Dividend Yield: {f'{dividend_yield*100:.2f}%' if dividend_yield else 'N/A'}
- Debt to Equity: {debt_to_equity if debt_to_equity else 'N/A'}
- Rekomendasi Analis: {fundamentals.get('analyst_recommendation', 'N/A')}
- Harga Saat Ini: {fundamentals.get('current_price', 'N/A')}
- Target Harga Analis: {fundamentals.get('target_mean_price', 'N/A')}

## Berita Terkini:
{news_text}
{transcript_section}

Berikan analisa fundamental dalam format JSON berikut (hanya JSON, tanpa teks lain):
{{
  "signal": "BUY" atau "SELL" atau "HOLD",
  "confidence": <angka 0.0 sampai 1.0>,
  "valuation": "undervalued" atau "fairly_valued" atau "overvalued",
  "reasoning": "<penjelasan singkat dalam Bahasa Indonesia, maks 200 kata>"
}}"""

    async def _call_gemini(self, prompt: str) -> str:
        """
        Call the Gemini API with a prompt.

        Args:
            prompt: Text prompt to send.

        Returns:
            Response text from Gemini.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        logger.debug("Calling Gemini API for fundamental analysis")
        response = model.generate_content(prompt)
        return response.text

    def _parse_gemini_response(self, response_text: str, fundamentals: dict) -> dict:
        """
        Parse JSON response from Gemini.

        Falls back to rule-based analysis if parsing fails.

        Args:
            response_text: Raw text response from Gemini.
            fundamentals: Fundamental data dict (for fallback scoring).

        Returns:
            Dict with signal, confidence, valuation, reasoning.
        """
        # Try to extract JSON from response
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(response_text[start:end])
                signal = parsed.get("signal", "HOLD").upper()
                if signal not in ("BUY", "SELL", "HOLD"):
                    signal = "HOLD"
                confidence = float(parsed.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                valuation = parsed.get("valuation", "fairly_valued")
                reasoning = parsed.get("reasoning", "Analisa Gemini tersedia.")
                return {
                    "signal": signal,
                    "confidence": confidence,
                    "valuation": valuation,
                    "reasoning": reasoning,
                }
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning(f"Failed to parse Gemini JSON response: {exc}. Using fallback.")

        return self._fallback_analysis(fundamentals)

    def _fallback_analysis(self, fundamentals: dict) -> dict:
        """Simple rule-based fallback when Gemini parsing fails."""
        score = 0.0
        parts: list[str] = []

        per = fundamentals.get("per")
        if per:
            if per < 10:
                score += 0.3
                parts.append(f"PER rendah ({per:.1f}x) → undervalued")
            elif per > 30:
                score -= 0.2
                parts.append(f"PER tinggi ({per:.1f}x) → mahal")

        roe = fundamentals.get("roe")
        if roe:
            if roe > 0.15:
                score += 0.2
                parts.append(f"ROE tinggi ({roe*100:.1f}%)")
            elif roe < 0.05:
                score -= 0.1
                parts.append(f"ROE rendah ({roe*100:.1f}%)")

        debt_to_equity = fundamentals.get("debt_to_equity")
        if debt_to_equity:
            if debt_to_equity > 200:
                score -= 0.2
                parts.append(f"Utang tinggi (DER={debt_to_equity:.1f}%)")

        if score >= 0.25:
            signal, valuation = "BUY", "undervalued"
        elif score <= -0.25:
            signal, valuation = "SELL", "overvalued"
        else:
            signal, valuation = "HOLD", "fairly_valued"

        confidence = min(0.7, 0.4 + abs(score) * 0.5)
        reasoning = " | ".join(parts) if parts else "Data fundamental tidak mencukupi untuk analisa mendalam."

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "valuation": valuation,
            "reasoning": reasoning,
        }
