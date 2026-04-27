"""
Data fetcher tool for AgentSaham.

Provides functions to:
- Fetch OHLCV stock data from yfinance
- Scrape news from RSS feeds
- Support both IDX (.JK suffix) and US stocks
- Cache data to avoid redundant requests
"""

import time
from datetime import datetime
from functools import lru_cache
from typing import Optional

import feedparser
import pandas as pd
import yfinance as yf
from loguru import logger


def _normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol.

    Automatically appends .JK suffix for IDX stocks if not already present
    and no other exchange suffix is given.

    Args:
        ticker: Raw ticker symbol, e.g. "BBCA" or "AAPL".

    Returns:
        Normalized ticker string.
    """
    ticker = ticker.upper().strip()
    # If no dot suffix and not clearly a US ticker (short, all letters), assume IDX
    if "." not in ticker and len(ticker) <= 4 and ticker.isalpha():
        ticker = f"{ticker}.JK"
    return ticker


def fetch_ohlcv(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for a stock ticker.

    Args:
        ticker: Stock ticker symbol. IDX tickers without .JK will be auto-suffixed.
        period: Data period, e.g. "1mo", "3mo", "6mo", "1y", "2y".
        interval: Data interval, e.g. "1d", "1wk", "1mo".

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, index=Date.

    Raises:
        ValueError: If no data is found for the ticker.
    """
    normalized = _normalize_ticker(ticker)
    logger.info(f"Fetching OHLCV data for {normalized} (period={period}, interval={interval})")

    stock = yf.Ticker(normalized)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No OHLCV data found for ticker '{normalized}'")

    df.index = pd.to_datetime(df.index)
    logger.debug(f"Fetched {len(df)} rows for {normalized}")
    return df


def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental financial data for a stock ticker.

    Retrieves key metrics: PER, PBV, ROE, EPS, dividend yield, debt ratio,
    market cap, and company info.

    Args:
        ticker: Stock ticker symbol. IDX tickers without .JK will be auto-suffixed.

    Returns:
        Dictionary containing fundamental metrics. Missing values are None.
    """
    normalized = _normalize_ticker(ticker)
    logger.info(f"Fetching fundamental data for {normalized}")

    stock = yf.Ticker(normalized)
    info = stock.info or {}

    fundamentals = {
        "ticker": normalized,
        "company_name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency"),
        # Valuation
        "per": info.get("trailingPE") or info.get("forwardPE"),
        "pbv": info.get("priceToBook"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        # Profitability
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "eps": info.get("trailingEps") or info.get("forwardEps"),
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        # Dividend
        "dividend_yield": info.get("dividendYield"),
        "dividend_rate": info.get("dividendRate"),
        "payout_ratio": info.get("payoutRatio"),
        # Debt
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        # Growth
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        # Price
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "target_mean_price": info.get("targetMeanPrice"),
        "analyst_recommendation": info.get("recommendationKey"),
        "fetched_at": datetime.utcnow().isoformat(),
    }

    logger.debug(f"Fundamental data fetched for {normalized}: PER={fundamentals['per']}, PBV={fundamentals['pbv']}")
    return fundamentals


def fetch_volume_data(ticker: str, period: str = "3mo") -> dict:
    """
    Fetch volume and related transaction data for a stock.

    Args:
        ticker: Stock ticker symbol.
        period: Data period for volume history.

    Returns:
        Dictionary with volume statistics and anomaly indicators.
    """
    normalized = _normalize_ticker(ticker)
    logger.info(f"Fetching volume data for {normalized}")

    df = fetch_ohlcv(normalized, period=period, interval="1d")

    avg_volume = df["Volume"].mean()
    std_volume = df["Volume"].std()
    latest_volume = df["Volume"].iloc[-1]
    latest_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest_close

    # Volume anomaly: more than 2 std deviations above average
    volume_z_score = (latest_volume - avg_volume) / std_volume if std_volume > 0 else 0.0
    is_unusual = abs(volume_z_score) > 2.0

    # Price change
    price_change_pct = ((latest_close - prev_close) / prev_close * 100) if prev_close else 0.0

    # Trend: compare last 5 days volume vs previous 5 days
    recent_avg = df["Volume"].iloc[-5:].mean() if len(df) >= 5 else avg_volume
    prior_avg = df["Volume"].iloc[-10:-5].mean() if len(df) >= 10 else avg_volume
    volume_trend = "increasing" if recent_avg > prior_avg * 1.1 else (
        "decreasing" if recent_avg < prior_avg * 0.9 else "stable"
    )

    return {
        "ticker": normalized,
        "latest_volume": int(latest_volume),
        "avg_volume_3mo": float(avg_volume),
        "volume_z_score": float(volume_z_score),
        "is_unusual_volume": bool(is_unusual),
        "volume_trend": volume_trend,
        "latest_close": float(latest_close),
        "price_change_pct": float(price_change_pct),
        "fetched_at": datetime.utcnow().isoformat(),
    }


def scrape_news(
    ticker: str,
    rss_feeds: Optional[list] = None,
    max_articles: int = 10,
) -> list[dict]:
    """
    Scrape news articles related to a stock ticker from RSS feeds.

    Args:
        ticker: Stock ticker symbol (used as keyword filter).
        rss_feeds: List of RSS feed URLs to scrape. Uses defaults if None.
        max_articles: Maximum number of articles to return.

    Returns:
        List of dicts with keys: title, summary, link, published.
    """
    if rss_feeds is None:
        rss_feeds = [
            "https://www.cnbcindonesia.com/rss",
            "https://rss.kontan.co.id/category/investasi",
            "https://finance.yahoo.com/news/rssindex",
        ]

    # Strip exchange suffix for keyword matching
    keyword = ticker.replace(".JK", "").upper()
    articles: list[dict] = []

    for feed_url in rss_feeds:
        try:
            logger.debug(f"Parsing RSS feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")
                # Filter articles mentioning the ticker
                if keyword.lower() in (title + summary).lower():
                    articles.append({
                        "title": title,
                        "summary": summary[:500],
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": feed_url,
                    })
        except Exception as exc:
            logger.warning(f"Failed to parse RSS feed {feed_url}: {exc}")

    # Deduplicate by title and limit results
    seen_titles: set = set()
    unique_articles: list[dict] = []
    for article in articles:
        if article["title"] not in seen_titles:
            seen_titles.add(article["title"])
            unique_articles.append(article)
        if len(unique_articles) >= max_articles:
            break

    logger.info(f"Scraped {len(unique_articles)} news articles for {keyword}")
    return unique_articles


class DataFetcher:
    """
    Convenience class wrapping data-fetching utilities with optional caching.

    Attributes:
        cache_ttl: Cache time-to-live in seconds. Set to 0 to disable caching.
    """

    def __init__(self, cache_ttl: int = 300) -> None:
        """
        Initialize DataFetcher.

        Args:
            cache_ttl: Cache expiry time in seconds (default 5 minutes).
        """
        self.cache_ttl = cache_ttl
        self._ohlcv_cache: dict = {}
        self._fundamentals_cache: dict = {}

    def get_ohlcv(self, ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV data with in-memory caching."""
        cache_key = f"{ticker}_{period}_{interval}"
        cached = self._ohlcv_cache.get(cache_key)
        if cached and (time.time() - cached["ts"] < self.cache_ttl):
            logger.debug(f"Cache hit for OHLCV {cache_key}")
            return cached["data"]

        data = fetch_ohlcv(ticker, period=period, interval=interval)
        self._ohlcv_cache[cache_key] = {"data": data, "ts": time.time()}
        return data

    def get_fundamentals(self, ticker: str) -> dict:
        """Fetch fundamental data with in-memory caching."""
        cached = self._fundamentals_cache.get(ticker)
        if cached and (time.time() - cached["ts"] < self.cache_ttl):
            logger.debug(f"Cache hit for fundamentals {ticker}")
            return cached["data"]

        data = fetch_fundamentals(ticker)
        self._fundamentals_cache[ticker] = {"data": data, "ts": time.time()}
        return data

    def get_volume_data(self, ticker: str, period: str = "3mo") -> dict:
        """Fetch volume data (not cached due to real-time nature)."""
        return fetch_volume_data(ticker, period=period)

    def get_news(self, ticker: str, rss_feeds: Optional[list] = None, max_articles: int = 10) -> list[dict]:
        """Scrape news for ticker from RSS feeds."""
        return scrape_news(ticker, rss_feeds=rss_feeds, max_articles=max_articles)
