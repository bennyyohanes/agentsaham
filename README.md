# 🤖 AgentSaham — Multi-Agent AI Stock Analysis

> Sistem analisa saham berbasis AI dengan **3 agent spesialis** yang bekerja secara paralel dan menghasilkan rekomendasi konsensus.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Apa Itu AgentSaham?

AgentSaham adalah sistem **Multi-Agent AI** yang menganalisa saham dari tiga perspektif berbeda secara bersamaan:

| Agent | Fokus | Tools |
|---|---|---|
| 📈 **Technical Agent** | Chart, indikator, pola candlestick | pandas-ta, pivot points |
| 📊 **Fundamental Agent** | Laporan keuangan, valuasi, berita | yfinance, feedparser, Gemini AI |
| 💹 **Transaction Agent** | Volume, big player, akumulasi/distribusi | OBV, analisa anomali volume |

Ketiga agent bekerja **paralel** menggunakan `asyncio`, kemudian hasilnya dikumpulkan oleh **Voting System** berbobot untuk menghasilkan sinyal final:

```
STRONG_BUY 🚀 | BUY 🟢 | HOLD 🟡 | SELL 🔴 | STRONG_SELL 💀
```

---

## 📁 Struktur Project

```
agentsaham/
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py          # Koordinator utama semua agent
│   ├── technical_agent.py       # Agent analisa teknikal
│   ├── fundamental_agent.py     # Agent analisa fundamental
│   └── transaction_agent.py     # Agent analisa transaksi/volume
├── tools/
│   ├── __init__.py
│   ├── transcriber.py           # Faster-Whisper untuk transkripsi video
│   ├── data_fetcher.py          # yfinance + RSS feed scraping
│   └── chart_analyzer.py        # Analisa chart dengan pandas-ta
├── core/
│   ├── __init__.py
│   ├── voting_system.py         # Sistem voting & consensus antar agent
│   └── report_generator.py      # Generate laporan analisis Markdown
├── output/
│   ├── __init__.py
│   ├── dashboard.py             # Streamlit dashboard
│   └── notifier.py              # Telegram notifikasi
├── config/
│   ├── __init__.py
│   └── settings.py              # Konfigurasi global
├── tests/
│   ├── __init__.py
│   ├── test_technical_agent.py
│   ├── test_fundamental_agent.py
│   └── test_transaction_agent.py
├── main.py                      # Entry point CLI
├── .env.example                 # Contoh environment variables
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/bennyyohanes/agentsaham.git
cd agentsaham

# Buat virtual environment (disarankan)
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Pastikan `ffmpeg` sudah terinstall di sistem untuk fitur transkripsi video.
> - Ubuntu/Debian: `sudo apt install ffmpeg`
> - Mac: `brew install ffmpeg`
> - Windows: Download dari [ffmpeg.org](https://ffmpeg.org/download.html)

### 2. Setup Environment Variables

```bash
cp .env.example .env
# Edit .env dengan text editor favorit kamu
```

Isi minimal yang dibutuhkan:
```
GEMINI_API_KEY=your_key_here
```

### 3. Jalankan Analisis

```bash
# Analisa saham IDX (suffix .JK otomatis ditambahkan)
python main.py --ticker BBCA

# Analisa saham US
python main.py --ticker AAPL

# Dengan video lokal
python main.py --ticker BBCA --video /path/to/video.mp4

# Dengan YouTube URL
python main.py --ticker BBCA --youtube https://youtu.be/xxx

# Kirim ke Telegram
python main.py --ticker BBCA --notify
```

---

## 🔑 Cara Mendapatkan Gemini API Key (Gratis)

1. Buka [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Login dengan akun Google
3. Klik **"Create API Key"**
4. Copy key yang dihasilkan
5. Paste ke `.env`: `GEMINI_API_KEY=AIza...`

> ✅ **Gratis!** Tier gratis Gemini sudah lebih dari cukup untuk penggunaan normal.

---

## 🤖 Cara Setup Telegram Bot

1. Buka Telegram, cari **@BotFather**
2. Kirim `/newbot` dan ikuti instruksi
3. Copy token yang diberikan (format: `123456789:ABCdef...`)
4. Dapatkan chat ID kamu:
   - Kirim pesan ke bot kamu
   - Buka: `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Cari `"chat":{"id":<NUMBER>}`
5. Isi `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456789:ABCdef...
   TELEGRAM_CHAT_ID=123456789
   ```

---

## 🌐 Streamlit Dashboard

```bash
streamlit run output/dashboard.py
```

Buka browser di `http://localhost:8501`

Fitur dashboard:
- Input ticker saham
- Upload video atau masukkan YouTube URL
- Chart harga candlestick dengan SMA20/SMA50
- Detail analisa setiap agent (tab terpisah)
- Laporan lengkap dalam format Markdown

---

## 🐳 Docker

```bash
# Build image
docker-compose build

# Analisa saham (ganti BBCA dengan ticker yang diinginkan)
TICKER=BBCA docker-compose run agentsaham

# Jalankan dashboard
docker-compose up dashboard
# Buka: http://localhost:8501
```

---

## ⚖️ Sistem Voting

Ketiga agent memberikan sinyal BUY/SELL/HOLD dengan confidence level, kemudian digabungkan menggunakan **weighted voting**:

| Agent | Bobot Default |
|---|---|
| 📈 Technical Agent | 35% |
| 📊 Fundamental Agent | 40% |
| 💹 Transaction Agent | 25% |

Formula:
```
score = Σ (direction × confidence × weight)
```

Dimana `direction`: BUY=+1, HOLD=0, SELL=-1

| Weighted Score | Sinyal Final |
|---|---|
| ≥ 0.60 | 🚀 STRONG_BUY |
| 0.20 – 0.60 | 🟢 BUY |
| -0.20 – 0.20 | 🟡 HOLD |
| -0.60 – -0.20 | 🔴 SELL |
| ≤ -0.60 | 💀 STRONG_SELL |

> **Konflik**: Jika Technical mengatakan BUY tapi Fundamental mengatakan SELL (atau sebaliknya), sistem akan menandai ⚠️ "Konflik Terdeteksi" dan merekomendasikan untuk menunggu konfirmasi.

---

## 📊 Contoh Output

```
============================================================
  ANALISIS SAHAM: BBCA.JK
============================================================
  🟢  Sinyal   : BUY
  📊  Confidence: 72%

  Technical      : BUY (78%)
  Fundamental    : BUY (65%)
  Transaction    : HOLD (55%)

  🎯 Entry    :    Rp   8,500
  ✅ Target   :    Rp   9,200
  🛑 Stop Loss:    Rp   8,100
============================================================
```

**Laporan Markdown yang dikirim ke Telegram:**
```
📊 Laporan Analisis Saham: BBCA.JK

🟢 Keputusan Akhir: BUY
Confidence: 72%

📈 Analisa Teknikal — 🟢 BUY
RSI(14): 42.3 🟡 Netral
MACD: bullish crossover
Trend: uptrend

📊 Analisa Fundamental — 🟢 BUY
Valuasi: ⚖️ Fairly Valued
PER: 14.2x | PBV: 2.1x | ROE: 19.3%

💹 Analisa Transaksi — 🟡 HOLD
Volume stabil, tidak ada aktivitas unusual
OBV: netral
```

---

## 🧪 Testing

```bash
# Jalankan semua test
pytest tests/ -v

# Test spesifik
pytest tests/test_technical_agent.py -v
pytest tests/test_fundamental_agent.py -v
pytest tests/test_transaction_agent.py -v
```

---

## 🗺️ Roadmap

### ✅ Phase 1 (Current) — Core Multi-Agent
- [x] TechnicalAgent dengan pandas-ta
- [x] FundamentalAgent dengan Gemini AI
- [x] TransactionAgent dengan analisa OBV
- [x] Voting System berbobot
- [x] Telegram notifikasi
- [x] Streamlit dashboard
- [x] CLI entry point
- [x] Docker support

### 🔜 Phase 2 — Enhanced Analysis
- [ ] Deteksi chart pattern visual dengan YOLO/OpenCV
- [ ] Integrasi data IDX real-time (JATS feed)
- [ ] Foreign flow dari data BEI
- [ ] Backtest framework

### 🔮 Phase 3 — Production Ready
- [ ] Database untuk menyimpan historis analisis
- [ ] Scheduler untuk analisis otomatis (cron/celery)
- [ ] API endpoint (FastAPI)
- [ ] Multi-ticker watchlist
- [ ] Portfolio tracking

---

## 🤝 Kontribusi

Pull request sangat disambut! Untuk perubahan besar, buka issue terlebih dahulu.

---

## ⚠️ Disclaimer

> Sistem ini dibuat untuk tujuan edukasi dan riset. **Bukan saran investasi.** Selalu lakukan riset mandiri (DYOR — Do Your Own Research) sebelum membuat keputusan investasi. Past performance does not guarantee future results.

---

## 📄 Lisensi

[MIT License](LICENSE) — bebas digunakan untuk keperluan pribadi dan komersial.
