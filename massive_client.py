"""
Massive Market Data API client.

Free (Basic) plan — available endpoints:
  /v3/reference/tickers                               → ticker metadata search
  /v3/reference/tickers/{ticker}                      → single ticker detail
  /v3/reference/dividends                             → dividend history
  /v3/reference/options/contracts                     → option contract metadata

Paid-plan endpoints (Starter $29/mo or Advanced $199/mo) are stubbed out
and raise ``MassiveUpgradeRequired`` so callers can fall back gracefully.

Auth: Bearer token via MASSIVE_API_KEY env var.
"""

from __future__ import annotations

import os
from typing import Any

import requests

BASE_URL = "https://api.massive.com/v3"
_DEFAULT_TIMEOUT = 10


class MassiveUpgradeRequired(Exception):
    """Raised when the endpoint requires a paid Massive plan."""


def _headers() -> dict[str, str]:
    key = os.environ.get("MASSIVE_API_KEY", "")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def _get(path: str, params: dict | None = None) -> Any:
    """Low-level GET against the Massive API. Returns parsed JSON."""
    url = f"{BASE_URL}/{path.lstrip('/')}"
    resp = requests.get(url, headers=_headers(), params=params, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ── Free-tier reference endpoints ─────────────────────────────────────────────

def search_tickers(query: str, limit: int = 20) -> list[dict]:
    """Search for tickers by keyword. Returns list of ticker metadata dicts."""
    try:
        data = _get("/reference/tickers", params={"search": query, "limit": limit})
        return data.get("results", data) if isinstance(data, dict) else data
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise MassiveUpgradeRequired("search_tickers requires a paid Massive plan.") from e
        raise


def get_ticker_detail(ticker: str) -> dict:
    """Return reference metadata for a single ticker symbol."""
    try:
        return _get(f"/reference/tickers/{ticker.upper()}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise MassiveUpgradeRequired(f"get_ticker_detail({ticker}) requires a paid Massive plan.") from e
        raise


def get_dividends(ticker: str, limit: int = 10) -> list[dict]:
    """Return recent dividend history for a ticker."""
    try:
        data = _get("/reference/dividends", params={"ticker": ticker.upper(), "limit": limit})
        return data.get("results", data) if isinstance(data, dict) else data
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise MassiveUpgradeRequired(f"get_dividends({ticker}) requires a paid Massive plan.") from e
        raise


def get_option_contracts(underlying: str, limit: int = 50) -> list[dict]:
    """
    Return option contract reference data for an underlying ticker.
    This is metadata (contract symbol, expiry, strike, type) — NOT live greeks.
    """
    try:
        data = _get(
            "/reference/options/contracts",
            params={"underlying_ticker": underlying.upper(), "limit": limit},
        )
        return data.get("results", data) if isinstance(data, dict) else data
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise MassiveUpgradeRequired(
                f"get_option_contracts({underlying}) requires a paid Massive plan."
            ) from e
        raise


# ── Paid-tier stubs (Starter / Advanced plan) ─────────────────────────────────

def get_snapshot(ticker: str) -> dict:
    """
    [PAID — Starter $29/mo] Real-time quote snapshot for a ticker.
    Raises MassiveUpgradeRequired on the free plan.
    """
    try:
        return _get(f"/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise MassiveUpgradeRequired(
                "get_snapshot() requires Massive Starter plan ($29/mo) or higher."
            ) from e
        raise


def get_options_snapshot(underlying: str) -> dict:
    """
    [PAID — Advanced $199/mo] Live IV / greeks snapshot for all option contracts.
    Raises MassiveUpgradeRequired on free or Starter plans.
    """
    try:
        return _get(f"/snapshot/options/{underlying.upper()}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise MassiveUpgradeRequired(
                "get_options_snapshot() requires Massive Advanced plan ($199/mo)."
            ) from e
        raise


def get_quotes(ticker: str) -> list[dict]:
    """
    [PAID — Starter $29/mo] NBBO quote stream for a ticker.
    Raises MassiveUpgradeRequired on the free plan.
    """
    try:
        return _get(f"/quotes/{ticker.upper()}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise MassiveUpgradeRequired(
                "get_quotes() requires Massive Starter plan ($29/mo) or higher."
            ) from e
        raise
