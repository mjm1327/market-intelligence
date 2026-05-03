# Market Intelligence Dashboard

A personal investment research tool that ingests content from YouTube, Substack, and pasted text, extracts company and theme mentions using Claude AI, and tracks stocks against their 200-week moving average.

## Features

- **Content Ingestion** — Paste a YouTube URL or Substack link and the app fetches the transcript/article automatically. Or paste any raw text (tweet threads, newsletter excerpts).
- **AI-Powered Extraction** — Claude analyzes each source and extracts every meaningfully discussed company, tagging it with an AI sub-theme (infrastructure, edge inference, biotech/healthtech, applications) and a sentiment signal (bullish/bearish/neutral).
- **200 WMA Scanner** — Scans the full S&P 500 and Russell 2000 (~2,500 companies) and shows which are trading below their 200-week moving average. Results are cached; refresh on demand.
- **Unified Watchlist** — One list fed by both discovery paths — content pipeline and the WMA screen. Each entry shows a live price chart with the 200-week MA overlay and a notes field.

## Tech Stack

| Layer | Library |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| AI extraction | [Anthropic Claude](https://docs.anthropic.com) |
| YouTube transcripts | `youtube-transcript-api` |
| Substack content | `requests` + `beautifulsoup4` |
| Price data | `yfinance` |
| Charts | `plotly` |
| Storage | SQLite (local) |

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/market-intelligence.git
cd market-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Anthropic API key

```bash
cp .env.example .env
# Edit .env and add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

Get an API key at [console.anthropic.com](https://console.anthropic.com).

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Usage

### Adding Sources

1. Go to **Sources** → **Add Source**
2. Choose YouTube Video, Substack Article, or Paste Text
3. Click **Extract** — Claude will identify companies, sub-themes, and sentiment
4. Results appear immediately and persist in the local SQLite database

### Browsing Intelligence

The **Intelligence** page groups all extracted companies by AI sub-theme. Expand any company to see which sources mentioned it and why. Click **＋ Watch** to add it to your watchlist.

### Running the WMA Scanner

1. Go to **200 WMA Scanner**
2. Click **Run Full Scan** (takes ~5–10 min for ~2,500 tickers)
3. Filter by index, % below WMA, and sort order
4. Select rows and click **Add to Watchlist**

Results are cached — you only need to re-scan when you want fresh data (weekly is typical).

### Watchlist

Shows all tracked companies with live price vs. 200 WMA. Click any ticker to see a 5-year weekly price chart with the 200-week MA overlay and add personal notes.

## Project Structure

```
├── app.py                  # Streamlit UI — all four pages
├── db.py                   # SQLite schema and queries
├── extractors/
│   ├── youtube.py          # YouTube transcript fetching
│   ├── substack.py         # Substack article fetching
│   └── llm.py              # Claude extraction prompt + parsing
├── scanner/
│   └── wma_scanner.py      # S&P 500 + Russell 2000 200-week MA scan
├── requirements.txt
└── .env.example
```

## Notes

- `portfolio.db` (SQLite) is created locally on first run and is excluded from git.
- X/Twitter ingestion is not included due to API costs. Use **Paste Text** to add tweet threads manually.
- The WMA scanner batches yfinance requests to avoid rate limits; a full scan takes roughly 5–10 minutes.
