"""
Telegram notifier for AgentSaham.

Sends formatted analysis results to a Telegram chat using
python-telegram-bot library.
"""

from typing import Optional

from loguru import logger


# Maximum Telegram message length (HTML parse mode)
_MAX_MESSAGE_LENGTH = 4096


def _truncate(text: str, max_len: int = _MAX_MESSAGE_LENGTH) -> str:
    """Truncate text to fit Telegram message limits."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _signal_to_emoji(signal: str) -> str:
    """Map final signal to emoji."""
    mapping = {
        "STRONG_BUY": "🚀🟢",
        "BUY": "🟢",
        "HOLD": "🟡",
        "SELL": "🔴",
        "STRONG_SELL": "💀🔴",
    }
    return mapping.get(signal, "⚪")


class TelegramNotifier:
    """
    Sends stock analysis notifications to a Telegram chat.

    Attributes:
        bot_token: Telegram bot token.
        chat_id: Target chat or group ID.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        """
        Initialize TelegramNotifier.

        Args:
            bot_token: Telegram bot API token.
            chat_id: Telegram chat ID or username.

        Raises:
            ValueError: If bot_token or chat_id is empty.
        """
        if not bot_token:
            raise ValueError("bot_token is required for TelegramNotifier")
        if not chat_id:
            raise ValueError("chat_id is required for TelegramNotifier")
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._bot = None

    def _get_bot(self):
        """Lazy-initialize the Telegram Bot instance."""
        if self._bot is None:
            try:
                from telegram import Bot
            except ImportError as exc:
                raise ImportError(
                    "python-telegram-bot is not installed. "
                    "Run: pip install python-telegram-bot"
                ) from exc
            self._bot = Bot(token=self.bot_token)
        return self._bot

    async def send_analysis(self, ticker: str, analysis_result: dict) -> bool:
        """
        Send a formatted analysis result to Telegram.

        Args:
            ticker: Stock ticker symbol.
            analysis_result: Output from Orchestrator.analyze().

        Returns:
            True if message was sent successfully, False otherwise.
        """
        message = self._format_message(ticker, analysis_result)
        return await self._send_message(message)

    async def send_message(self, text: str) -> bool:
        """
        Send a raw text message to Telegram.

        Args:
            text: Message text (Markdown formatting supported).

        Returns:
            True if sent successfully, False otherwise.
        """
        return await self._send_message(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_message(self, ticker: str, result: dict) -> str:
        """Build a nicely formatted Telegram message."""
        final_signal = result.get("final_signal", "HOLD")
        confidence = result.get("confidence", 0.0)
        emoji = _signal_to_emoji(final_signal)

        # Per-agent summaries
        tech = result.get("agent_results", {}).get("technical", {})
        fund = result.get("agent_results", {}).get("fundamental", {})
        trans = result.get("agent_results", {}).get("transaction", {})

        agent_lines = []
        for label, agent_data in [("📈 Teknikal", tech), ("📊 Fundamental", fund), ("💹 Transaksi", trans)]:
            s = agent_data.get("signal", "?")
            c = agent_data.get("confidence", 0.0)
            agent_lines.append(f"  {label}: *{s}* ({c:.0%})")

        entry = tech.get("entry_point")
        target = tech.get("target")
        stop_loss = tech.get("stop_loss")

        price_lines = []
        if entry:
            price_lines.append(f"🎯 Entry : Rp {entry:,.0f}")
        if target:
            price_lines.append(f"✅ Target: Rp {target:,.0f}")
        if stop_loss:
            price_lines.append(f"🛑 Stop  : Rp {stop_loss:,.0f}")

        conflict_warning = ""
        if result.get("voting_details", {}).get("has_conflict"):
            conflict_warning = "\n⚠️ *Peringatan: Konflik sinyal terdeteksi!*"

        message_parts = [
            f"{emoji} *SINYAL SAHAM: {ticker}*",
            f"━━━━━━━━━━━━━━━━",
            f"🏆 Keputusan: *{final_signal}*",
            f"📊 Confidence: *{confidence:.0%}*",
            "",
            "*Breakdown Agent:*",
            *agent_lines,
        ]

        if price_lines:
            message_parts += ["", *price_lines]

        if conflict_warning:
            message_parts.append(conflict_warning)

        message_parts += [
            "",
            "━━━━━━━━━━━━━━━━",
            "_⚠️ Bukan saran investasi. DYOR!_",
        ]

        return "\n".join(message_parts)

    async def _send_message(self, text: str) -> bool:
        """Send a message via Telegram Bot API."""
        try:
            bot = self._get_bot()
            truncated = _truncate(text)
            await bot.send_message(
                chat_id=self.chat_id,
                text=truncated,
                parse_mode="Markdown",
            )
            logger.info(f"[TelegramNotifier] Message sent to chat {self.chat_id}")
            return True
        except Exception as exc:
            logger.error(f"[TelegramNotifier] Failed to send message: {exc}")
            return False
