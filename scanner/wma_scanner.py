"""
200-week moving average scanner for S&P 500 and Russell 2000.

Results are cached in SQLite — call run_scan() to refresh, then
get_cached_results() to read them.
"""

from __future__ import annotations

import time
from typing import Generator

import pandas as pd
import yfinance as yf

import db

_BATCH_SIZE = 50  # tickers per yfinance batch download
_SLEEP_BETWEEN_BATCHES = 1.0  # seconds — be polite to the API


# ── Index constituent fetching ────────────────────────────────────────────────

def get_sp500_tickers() -> list[tuple[str, str]]:
    """Return list of (ticker, company_name) for S&P 500."""
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        attrs={"id": "constituents"},
    )
    df = tables[0]
    # Wikipedia uses "Symbol" and "Security"
    return list(zip(df["Symbol"].str.replace(".", "-", regex=False), df["Security"]))


def get_russell2000_tickers() -> list[tuple[str, str]]:
    """Return list of (ticker, company_name) for Russell 2000 via iShares IWM CSV."""
    url = (
        "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
        "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
    )
    try:
        # iShares CSV has 2 header rows; data starts at row 3
        df = pd.read_csv(url, skiprows=2, on_bad_lines="skip")
        # Column names vary; we want ticker and name
        ticker_col = next((c for c in df.columns if "ticker" in c.lower()), None)
        name_col = next((c for c in df.columns if "name" in c.lower()), None)
        if ticker_col and name_col:
            pairs = (
                df[[ticker_col, name_col]]
                .dropna()
                .apply(lambda r: (str(r[ticker_col]).strip(), str(r[name_col]).strip()), axis=1)
                .tolist()
            )
            # Filter out cash/other non-equity rows
            return [(t, n) for t, n in pairs if t and t != "-" and len(t) <= 5]
    except Exception:
        pass
    return []


# ── WMA calculation ───────────────────────────────────────────────────────────

def _batches(items: list, size: int) -> Generator[list, None, None]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _compute_wma_batch(
    batch: list[tuple[str, str]],
    index_name: str,
) -> list[dict]:
    tickers = [t for t, _ in batch]
    names = {t: n for t, n in batch}

    try:
        raw = yf.download(
            tickers,
            period="5y",
            interval="1wk",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return []

    if raw.empty:
        return []

    # yfinance returns MultiIndex columns when >1 ticker
    close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
    if isinstance(close, pd.Series):
        # Single ticker edge case
        close = close.to_frame(tickers[0])

    results = []
    for ticker in tickers:
        if ticker not in close.columns:
            continue
        series = close[ticker].dropna()
        if len(series) < 20:  # not enough history
            continue

        wma = series.rolling(200, min_periods=20).mean().iloc[-1]
        current = series.iloc[-1]

        if pd.isna(wma) or pd.isna(current) or wma == 0:
            continue

        pct = ((current - wma) / wma) * 100

        results.append(
            {
                "ticker": ticker,
                "company_name": names.get(ticker, ticker),
                "index_name": index_name,
                "current_price": round(float(current), 2),
                "wma_200": round(float(wma), 2),
                "pct_from_wma": round(float(pct), 2),
            }
        )
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def run_scan(progress_callback=None) -> int:
    """
    Fetch constituents, compute 200-week MA, cache in DB.
    progress_callback(pct: float, message: str) called periodically.
    Returns total number of tickers processed.
    """
    sp500 = get_sp500_tickers()
    russell = get_russell2000_tickers()

    combined = [("sp500", t, n) for t, n in sp500] + [
        ("russell2000", t, n) for t, n in russell
    ]

    # Deduplicate by ticker (S&P 500 takes precedence)
    seen: dict[str, tuple[str, str, str]] = {}
    for index_name, ticker, name in combined:
        if ticker not in seen:
            seen[ticker] = (index_name, ticker, name)
    unique = list(seen.values())

    total = len(unique)
    processed = 0
    all_rows: list[dict] = []

    for batch_items in _batches([(t, n) for _, t, n in unique], _BATCH_SIZE):
        # Determine index name for batch (use first item's index)
        batch_tickers = {t for t, _ in batch_items}
        # Map each ticker to its index
        index_map = {t: idx for idx, t, _ in unique if t in batch_tickers}

        rows = _compute_wma_batch(batch_items, "sp500")  # index resolved per-ticker below
        for row in rows:
            row["index_name"] = index_map.get(row["ticker"], "sp500")
        all_rows.extend(rows)

        processed += len(batch_items)
        if progress_callback:
            progress_callback(processed / total, f"Scanned {processed}/{total} tickers…")

        time.sleep(_SLEEP_BETWEEN_BATCHES)

    if all_rows:
        db.upsert_wma_cache(all_rows)

    return processed


def get_cached_results(below_only: bool = True) -> list[dict]:
    return db.get_wma_cache(below_only=below_only)
