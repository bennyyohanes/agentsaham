"""
Streamlit Dashboard for AgentSaham.

Provides a web-based UI where users can enter a stock ticker,
optionally upload a video for transcription, run the multi-agent
analysis, and view the results interactively.

Run with:
    streamlit run output/dashboard.py
"""

import asyncio
import sys
import os

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.settings import settings
from agents.orchestrator import Orchestrator
from tools.transcriber import Transcriber
from tools.data_fetcher import DataFetcher


# -----------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="AgentSaham 🤖",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Konfigurasi")
    ticker_input = st.text_input(
        "Ticker Saham",
        value="BBCA",
        help="Contoh: BBCA (IDX) atau AAPL (US). Suffix .JK otomatis ditambahkan untuk IDX.",
    ).upper()

    uploaded_video = st.file_uploader(
        "Upload Video (Opsional)",
        type=["mp4", "avi", "mkv", "mov"],
        help="Upload video analisis saham untuk ditranskripsi.",
    )

    youtube_url = st.text_input(
        "YouTube URL (Opsional)",
        placeholder="https://www.youtube.com/watch?v=...",
        help="URL video YouTube untuk ditranskripsi.",
    )

    run_analysis = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ Tentang")
    st.markdown(
        "**AgentSaham** adalah sistem multi-agent AI yang menganalisa saham dari perspektif:\n"
        "- 📈 Teknikal (chart & indikator)\n"
        "- 📊 Fundamental (laporan keuangan)\n"
        "- 💹 Transaksi (volume & big player)"
    )


# -----------------------------------------------------------------------
# Main Content
# -----------------------------------------------------------------------
st.title("🤖 AgentSaham — Multi-Agent AI Stock Analysis")
st.markdown("Analisa saham komprehensif menggunakan 3 agent AI spesialis.")

if not run_analysis:
    st.info("👈 Masukkan ticker saham di sidebar dan klik **Jalankan Analisis**.")
    st.stop()

# Validate settings
if not settings.gemini_api_key:
    st.error(
        "❌ **GEMINI_API_KEY** belum dikonfigurasi. "
        "Buat file `.env` dengan `GEMINI_API_KEY=...` lalu restart Streamlit."
    )
    st.stop()

# -----------------------------------------------------------------------
# Progress tracking
# -----------------------------------------------------------------------
progress_bar = st.progress(0, text="Memulai analisis...")
status_placeholder = st.empty()


def update_progress(pct: int, msg: str) -> None:
    progress_bar.progress(pct, text=msg)
    status_placeholder.info(msg)


# -----------------------------------------------------------------------
# Run analysis
# -----------------------------------------------------------------------
with st.spinner(f"Menganalisa {ticker_input}..."):
    transcript: str | None = None

    # Step 1: Transcription (if source provided)
    video_source: str | None = None
    if uploaded_video is not None:
        import tempfile
        update_progress(10, "📁 Menyimpan video yang di-upload...")
        suffix = os.path.splitext(uploaded_video.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_video.read())
            video_source = tmp.name
    elif youtube_url.strip():
        video_source = youtube_url.strip()

    if video_source:
        update_progress(20, "🎙️ Mentranskrip video...")
        try:
            transcriber = Transcriber(
                model_size=settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            transcription = transcriber.transcribe(video_source)
            transcript = transcription["text"]
            st.success(
                f"✅ Transkripsi selesai: {len(transcript)} karakter "
                f"(bahasa: {transcription['language']})"
            )
        except Exception as exc:
            st.warning(f"⚠️ Transkripsi gagal: {exc}. Melanjutkan tanpa transkrip.")

    # Step 2: Multi-agent analysis
    update_progress(40, f"🤖 Menjalankan 3 agent analisis untuk {ticker_input}...")
    try:
        orchestrator = Orchestrator()
        result = asyncio.run(orchestrator.analyze(ticker_input, transcript=transcript))
    except Exception as exc:
        st.error(f"❌ Analisis gagal: {exc}")
        st.stop()

    update_progress(90, "📄 Memformat laporan...")

progress_bar.progress(100, text="✅ Analisis selesai!")
status_placeholder.empty()


# -----------------------------------------------------------------------
# Results Display
# -----------------------------------------------------------------------
final_signal = result["final_signal"]
confidence = result["confidence"]

signal_colors = {
    "STRONG_BUY": "green",
    "BUY": "lightgreen",
    "HOLD": "orange",
    "SELL": "salmon",
    "STRONG_SELL": "red",
}
color = signal_colors.get(final_signal, "gray")

# Summary cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🏆 Sinyal Akhir", value=final_signal)
with col2:
    st.metric(label="📊 Confidence", value=f"{confidence:.0%}")
with col3:
    tech_signal = result["agent_results"].get("technical", {}).get("signal", "?")
    fund_signal = result["agent_results"].get("fundamental", {}).get("signal", "?")
    trans_signal = result["agent_results"].get("transaction", {}).get("signal", "?")
    st.metric(label="🤖 Agents", value=f"T:{tech_signal} F:{fund_signal} V:{trans_signal}")

if result["voting_details"].get("has_conflict"):
    st.warning("⚠️ **Konflik sinyal terdeteksi.** Pertimbangkan untuk menunggu konfirmasi lebih lanjut.")


# -----------------------------------------------------------------------
# Price Chart
# -----------------------------------------------------------------------
st.markdown("---")
st.subheader(f"📈 Chart Harga — {ticker_input}")

try:
    fetcher = DataFetcher()
    df = fetcher.get_ohlcv(ticker_input, period="6mo", interval="1d")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLCV",
    ))

    # Add SMA20
    sma20 = df["Close"].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20", line=dict(color="blue", width=1)))

    # Add SMA50
    if len(df) >= 50:
        sma50 = df["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA50", line=dict(color="orange", width=1)))

    fig.update_layout(
        title=f"{ticker_input} — 6 Bulan",
        xaxis_rangeslider_visible=False,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as exc:
    st.warning(f"Tidak bisa menampilkan chart: {exc}")


# -----------------------------------------------------------------------
# Agent Details
# -----------------------------------------------------------------------
st.markdown("---")
st.subheader("🤖 Detail Setiap Agent")

tab_tech, tab_fund, tab_trans = st.tabs(["📈 Teknikal", "📊 Fundamental", "💹 Transaksi"])

with tab_tech:
    tech = result["agent_results"].get("technical", {})
    if tech.get("error"):
        st.error(f"Agent error: {tech['error']}")
    else:
        col_a, col_b = st.columns(2)
        col_a.metric("Sinyal", tech.get("signal", "?"))
        col_b.metric("Confidence", f"{tech.get('confidence', 0):.0%}")
        st.markdown(f"**Trend:** {tech.get('trend', '?')}")
        st.markdown(f"**Reasoning:** {tech.get('reasoning', '')}")

        indicators = tech.get("indicators", {})
        if indicators:
            ind_data = {k: v for k, v in indicators.items() if v is not None}
            st.json(ind_data)

with tab_fund:
    fund = result["agent_results"].get("fundamental", {})
    if fund.get("error"):
        st.error(f"Agent error: {fund['error']}")
    else:
        col_a, col_b = st.columns(2)
        col_a.metric("Sinyal", fund.get("signal", "?"))
        col_b.metric("Confidence", f"{fund.get('confidence', 0):.0%}")
        st.markdown(f"**Valuasi:** {fund.get('valuation', '?')}")
        st.markdown(f"**Reasoning:** {fund.get('reasoning', '')}")

        fundamentals = fund.get("fundamentals", {})
        if fundamentals:
            display_keys = ["per", "pbv", "roe", "eps", "dividend_yield", "debt_to_equity"]
            fund_display = {k: fundamentals.get(k) for k in display_keys if fundamentals.get(k) is not None}
            if fund_display:
                st.table(pd.DataFrame.from_dict(fund_display, orient="index", columns=["Value"]))

with tab_trans:
    trans = result["agent_results"].get("transaction", {})
    if trans.get("error"):
        st.error(f"Agent error: {trans['error']}")
    else:
        col_a, col_b = st.columns(2)
        col_a.metric("Sinyal", trans.get("signal", "?"))
        col_b.metric("Confidence", f"{trans.get('confidence', 0):.0%}")
        unusual = trans.get("unusual_activity", False)
        st.markdown(f"**Unusual Activity:** {'⚠️ YA' if unusual else '✅ Tidak'}")
        st.markdown(f"**Reasoning:** {trans.get('reasoning', '')}")

        vol_data = trans.get("volume_data", {})
        if vol_data:
            st.json({k: v for k, v in vol_data.items() if v is not None})


# -----------------------------------------------------------------------
# Full Report
# -----------------------------------------------------------------------
st.markdown("---")
with st.expander("📄 Laporan Lengkap (Markdown)", expanded=False):
    st.markdown(result.get("report", "Laporan tidak tersedia."))
