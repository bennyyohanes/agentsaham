"""
AgentSaham — Main CLI entry point.

Usage:
    python main.py --ticker BBCA
    python main.py --ticker BBCA --video path/to/video.mp4
    python main.py --ticker BBCA --youtube https://youtu.be/xxx
    python main.py --ticker BBCA --notify     # Send result to Telegram
"""

import argparse
import asyncio
import sys

from loguru import logger

from config.settings import settings


def _configure_logging(log_level: str) -> None:
    """Configure loguru logging."""
    import os
    logger.remove()
    logger.add(sys.stderr, level=log_level, colorize=True, format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ))
    log_dir = os.path.dirname(settings.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logger.add(settings.log_file, level="DEBUG", rotation="10 MB", retention="7 days")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="agentsaham",
        description="Multi-Agent AI Stock Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python main.py --ticker BBCA
  python main.py --ticker AAPL --video /path/to/video.mp4
  python main.py --ticker BBCA --youtube https://youtu.be/xxx
  python main.py --ticker TLKM --notify
        """,
    )
    parser.add_argument(
        "--ticker", "-t",
        required=True,
        help="Ticker saham (contoh: BBCA untuk IDX, AAPL untuk US)",
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        help="Path file video lokal (.mp4, .avi, .mkv) untuk ditranskripsi",
    )
    parser.add_argument(
        "--youtube", "-y",
        default=None,
        help="URL YouTube video untuk ditranskripsi",
    )
    parser.add_argument(
        "--notify", "-n",
        action="store_true",
        default=False,
        help="Kirim hasil ke Telegram (butuh TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID di .env)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args()


async def _run_analysis(args: argparse.Namespace) -> dict:
    """Run the full analysis pipeline."""
    from agents.orchestrator import Orchestrator
    from tools.transcriber import Transcriber

    transcript: Optional[str] = None
    video_source = args.video or args.youtube

    if video_source:
        logger.info(f"Transkripsi sumber: {video_source}")
        transcriber = Transcriber(
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        try:
            result = transcriber.transcribe(video_source)
            transcript = result["text"]
            logger.info(
                f"Transkripsi selesai: {len(transcript)} karakter "
                f"(bahasa: {result['language']})"
            )
        except Exception as exc:
            logger.warning(f"Transkripsi gagal, melanjutkan tanpa transkrip: {exc}")

    orchestrator = Orchestrator()
    analysis = await orchestrator.analyze(args.ticker.upper(), transcript=transcript)
    return analysis


async def _notify(analysis: dict) -> None:
    """Send analysis result to Telegram if configured."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning(
            "TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID tidak dikonfigurasi. "
            "Lewati pengiriman notifikasi."
        )
        return

    from output.notifier import TelegramNotifier
    notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
    success = await notifier.send_analysis(analysis["ticker"], analysis)
    if success:
        logger.info("Notifikasi Telegram berhasil dikirim.")
    else:
        logger.error("Gagal mengirim notifikasi Telegram.")


def _print_summary(analysis: dict) -> None:
    """Print a concise summary to stdout."""
    signal_art = {
        "STRONG_BUY": "🚀🟢",
        "BUY": "🟢",
        "HOLD": "🟡",
        "SELL": "🔴",
        "STRONG_SELL": "💀🔴",
    }
    emoji = signal_art.get(analysis["final_signal"], "⚪")

    print("\n" + "=" * 60)
    print(f"  ANALISIS SAHAM: {analysis['ticker']}")
    print("=" * 60)
    print(f"  {emoji}  Sinyal   : {analysis['final_signal']}")
    print(f"  📊  Confidence: {analysis['confidence']:.0%}")
    print()

    agent_results = analysis.get("agent_results", {})
    for agent_name, result in agent_results.items():
        s = result.get("signal", "?")
        c = result.get("confidence", 0.0)
        err = "(error)" if result.get("error") else ""
        print(f"  {agent_name.capitalize():15s}: {s} ({c:.0%}) {err}")

    tech = agent_results.get("technical", {})
    if tech.get("entry_point"):
        print()
        print(f"  🎯 Entry    : Rp {tech['entry_point']:>12,.0f}")
    if tech.get("target"):
        print(f"  ✅ Target   : Rp {tech['target']:>12,.0f}")
    if tech.get("stop_loss"):
        print(f"  🛑 Stop Loss: Rp {tech['stop_loss']:>12,.0f}")

    if analysis.get("voting_details", {}).get("has_conflict"):
        print()
        print("  ⚠️  KONFLIK sinyal terdeteksi — pertimbangkan wait & see")

    print("=" * 60)
    print()


def main() -> None:
    """Main entry point."""
    args = _parse_args()
    _configure_logging(args.log_level)

    logger.info(f"AgentSaham dimulai: ticker={args.ticker}")

    if not settings.gemini_api_key:
        logger.error(
            "GEMINI_API_KEY tidak ditemukan. "
            "Tambahkan ke file .env atau environment variable."
        )
        sys.exit(1)

    analysis = asyncio.run(_run_analysis(args))
    _print_summary(analysis)

    if args.notify:
        asyncio.run(_notify(analysis))

    # Print full report
    print(analysis.get("report", ""))


if __name__ == "__main__":
    main()
