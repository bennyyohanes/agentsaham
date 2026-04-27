"""
Report generator for AgentSaham.

Generates a human-readable Markdown analysis report from the results
of all agents and the voting system's final decision.
"""

from datetime import datetime

from loguru import logger


# Signal emoji mapping
_SIGNAL_EMOJI = {
    "STRONG_BUY": "🟢🟢",
    "BUY": "🟢",
    "HOLD": "🟡",
    "SELL": "🔴",
    "STRONG_SELL": "🔴🔴",
}

_VALUATION_EMOJI = {
    "undervalued": "💎",
    "fairly_valued": "⚖️",
    "overvalued": "💸",
}


class ReportGenerator:
    """
    Generates Markdown analysis reports suitable for Telegram or web display.

    The report includes:
    - Final voting signal with confidence
    - Individual agent analysis summaries
    - Key indicators (technical, fundamental, transaction)
    - Entry point, target, and stop-loss levels
    - Conflict warnings where applicable
    """

    def generate(self, ticker: str, agent_results: dict, voting_result: dict) -> str:
        """
        Generate a full Markdown analysis report.

        Args:
            ticker: Stock ticker symbol.
            agent_results: Dict of agent name → result dict.
            voting_result: Output from VotingSystem.vote().

        Returns:
            Formatted Markdown string.
        """
        logger.info(f"[ReportGenerator] Generating report for {ticker}")

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        final_signal = voting_result.get("final_signal", "HOLD")
        confidence = voting_result.get("confidence", 0.0)
        signal_emoji = _SIGNAL_EMOJI.get(final_signal, "⚪")

        lines = [
            f"# 📊 Laporan Analisis Saham: {ticker}",
            f"*Dibuat: {timestamp}*",
            "",
            "---",
            "",
            f"## {signal_emoji} Keputusan Akhir: **{final_signal}**",
            f"**Confidence:** {confidence:.0%}",
            "",
            "### Ringkasan Voting",
            voting_result.get("reasoning", ""),
            "",
        ]

        if voting_result.get("has_conflict"):
            lines += [
                "> ⚠️ **Peringatan:** Terdapat konflik sinyal antar agent. "
                "Disarankan untuk menunggu konfirmasi lebih lanjut.",
                "",
            ]

        # Technical Section
        tech = agent_results.get("technical", {})
        if tech and not tech.get("error"):
            lines += self._format_technical_section(tech)

        # Fundamental Section
        fund = agent_results.get("fundamental", {})
        if fund and not fund.get("error"):
            lines += self._format_fundamental_section(fund)

        # Transaction Section
        trans = agent_results.get("transaction", {})
        if trans and not trans.get("error"):
            lines += self._format_transaction_section(trans)

        lines += [
            "---",
            "",
            "> *Laporan ini bersifat informatif dan bukan saran investasi.*",
            "> *Selalu lakukan riset mandiri sebelum mengambil keputusan.*",
        ]

        report = "\n".join(lines)
        logger.debug(f"[ReportGenerator] Report generated ({len(report)} chars)")
        return report

    # ------------------------------------------------------------------
    # Section formatters
    # ------------------------------------------------------------------

    def _format_technical_section(self, tech: dict) -> list[str]:
        """Format the technical analysis section."""
        signal = tech.get("signal", "HOLD")
        confidence = tech.get("confidence", 0.0)
        trend = tech.get("trend", "sideways")
        reasoning = tech.get("reasoning", "")
        entry = tech.get("entry_point")
        target = tech.get("target")
        stop_loss = tech.get("stop_loss")
        indicators = tech.get("indicators", {})

        lines = [
            "---",
            "",
            f"## 📈 Analisa Teknikal — {_SIGNAL_EMOJI.get(signal, '')} {signal}",
            f"**Confidence:** {confidence:.0%} | **Trend:** {trend}",
            "",
            f"**Reasoning:** {reasoning}",
            "",
        ]

        if entry or target or stop_loss:
            lines += [
                "| Level | Harga |",
                "|---|---|",
            ]
            if entry:
                lines.append(f"| Entry Point | Rp {entry:,.0f} |")
            if target:
                lines.append(f"| Target | Rp {target:,.0f} |")
            if stop_loss:
                lines.append(f"| Stop Loss | Rp {stop_loss:,.0f} |")
            lines.append("")

        rsi = indicators.get("rsi")
        macd = indicators.get("macd")
        if rsi is not None or macd is not None:
            lines.append("**Indikator Utama:**")
            if rsi is not None:
                rsi_label = "🔴 Overbought" if rsi > 70 else ("🟢 Oversold" if rsi < 30 else "🟡 Netral")
                lines.append(f"- RSI(14): `{rsi:.1f}` {rsi_label}")
            if macd is not None:
                lines.append(f"- MACD: `{macd:.4f}`")
            bb_lower = indicators.get("bb_lower")
            bb_upper = indicators.get("bb_upper")
            if bb_lower and bb_upper:
                lines.append(f"- Bollinger Bands: `{bb_lower:.2f}` — `{bb_upper:.2f}`")
            lines.append("")

        patterns = tech.get("patterns", {})
        if patterns:
            pattern_str = ", ".join(
                f"{k} ({'bullish' if v > 0 else 'bearish'})" for k, v in patterns.items()
            )
            lines += [f"**Candlestick Patterns:** {pattern_str}", ""]

        return lines

    def _format_fundamental_section(self, fund: dict) -> list[str]:
        """Format the fundamental analysis section."""
        signal = fund.get("signal", "HOLD")
        confidence = fund.get("confidence", 0.0)
        valuation = fund.get("valuation", "fairly_valued")
        reasoning = fund.get("reasoning", "")
        fundamentals = fund.get("fundamentals", {})
        val_emoji = _VALUATION_EMOJI.get(valuation, "")

        lines = [
            "---",
            "",
            f"## 📊 Analisa Fundamental — {_SIGNAL_EMOJI.get(signal, '')} {signal}",
            f"**Confidence:** {confidence:.0%} | **Valuasi:** {val_emoji} {valuation.replace('_', ' ').title()}",
            "",
            f"**Reasoning:** {reasoning}",
            "",
        ]

        per = fundamentals.get("per")
        pbv = fundamentals.get("pbv")
        roe = fundamentals.get("roe")
        eps = fundamentals.get("eps")
        div_yield = fundamentals.get("dividend_yield")
        der = fundamentals.get("debt_to_equity")

        if any(v is not None for v in [per, pbv, roe, eps, div_yield, der]):
            lines += ["**Data Keuangan:**", ""]
            if per is not None:
                lines.append(f"- PER: `{per:.1f}x`")
            if pbv is not None:
                lines.append(f"- PBV: `{pbv:.2f}x`")
            if roe is not None:
                lines.append(f"- ROE: `{roe*100:.1f}%`")
            if eps is not None:
                lines.append(f"- EPS: `{eps:.2f}`")
            if div_yield is not None:
                lines.append(f"- Dividend Yield: `{div_yield*100:.2f}%`")
            if der is not None:
                lines.append(f"- Debt to Equity: `{der:.1f}%`")
            lines.append("")

        return lines

    def _format_transaction_section(self, trans: dict) -> list[str]:
        """Format the transaction analysis section."""
        signal = trans.get("signal", "HOLD")
        confidence = trans.get("confidence", 0.0)
        reasoning = trans.get("reasoning", "")
        unusual = trans.get("unusual_activity", False)
        volume_data = trans.get("volume_data", {})

        lines = [
            "---",
            "",
            f"## 💹 Analisa Transaksi — {_SIGNAL_EMOJI.get(signal, '')} {signal}",
            f"**Confidence:** {confidence:.0%} | **Unusual Activity:** {'⚠️ YA' if unusual else '✅ Tidak'}",
            "",
            f"**Reasoning:** {reasoning}",
            "",
        ]

        latest_vol = volume_data.get("latest_volume")
        avg_vol = volume_data.get("avg_volume_3mo")
        z_score = volume_data.get("volume_z_score")
        vol_trend = volume_data.get("volume_trend")
        price_chg = volume_data.get("price_change_pct")

        if latest_vol or avg_vol:
            lines += ["**Data Volume:**", ""]
            if latest_vol:
                lines.append(f"- Volume Terkini: `{latest_vol:,}`")
            if avg_vol:
                lines.append(f"- Rata-rata Volume (3 Bulan): `{avg_vol:,.0f}`")
            if z_score is not None:
                lines.append(f"- Volume Z-Score: `{z_score:.2f}σ`")
            if vol_trend:
                lines.append(f"- Tren Volume: `{vol_trend}`")
            if price_chg is not None:
                lines.append(f"- Perubahan Harga: `{price_chg:+.2f}%`")
            lines.append("")

        return lines
