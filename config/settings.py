"""
Configuration settings for AgentSaham.

Loads all environment variables from .env file and provides
typed configuration values with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Global application settings loaded from environment variables."""

    # Gemini API
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

    # Telegram
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # Whisper
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    )

    # Data fetching
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "300"))
    )

    # Voting weights (must sum to 1.0)
    weight_technical: float = field(
        default_factory=lambda: float(os.getenv("WEIGHT_TECHNICAL", "0.35"))
    )
    weight_fundamental: float = field(
        default_factory=lambda: float(os.getenv("WEIGHT_FUNDAMENTAL", "0.40"))
    )
    weight_transaction: float = field(
        default_factory=lambda: float(os.getenv("WEIGHT_TRANSACTION", "0.25"))
    )

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/agentsaham.log"))

    # RSS feed URLs for news scraping
    rss_feeds: list = field(
        default_factory=lambda: [
            "https://www.cnbcindonesia.com/rss",
            "https://rss.kontan.co.id/category/investasi",
            "https://finance.yahoo.com/news/rssindex",
        ]
    )

    def validate(self) -> None:
        """Validate that required settings are present."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. "
                "Get a free key at https://makersuite.google.com/app/apikey"
            )

        total_weight = self.weight_technical + self.weight_fundamental + self.weight_transaction
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Voting weights must sum to 1.0, got {total_weight:.2f}"
            )


# Singleton settings instance
settings = Settings()
