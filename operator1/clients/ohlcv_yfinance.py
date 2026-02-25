"""Global OHLCV provider using yfinance.

Serves as both the primary provider for markets without region-specific
wrappers and the fallback for all markets.

No API key required. Uses Yahoo Finance public API.
Research: .roo/research/ohlcv-yfinance-2026-02-24.md
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Market ID -> yfinance ticker suffix mapping (from per-region research)
_MARKET_SUFFIX: dict[str, str] = {
    "us_sec_edgar": "",
    "uk_companies_house": ".L",
    "eu_esef": ".PA",
    "fr_esef": ".PA",
    "de_esef": ".DE",
    "jp_jquants": ".T",
    "kr_dart": ".KS",
    "tw_mops": ".TW",
    "br_cvm": ".SA",
    "cl_cmf": ".SN",
    "ca_sedar": ".TO",
    "au_asx": ".AX",
    "in_bse": ".NS",
    "cn_sse": ".SS",
    "hk_hkex": ".HK",
    "sg_sgx": ".SI",
    "mx_bmv": ".MX",
    "za_jse": ".JO",
    "ch_six": ".SW",
    "sa_tadawul": ".SR",
    "ae_dfm": ".AE",
    "nl_esef": ".AS",   # Euronext Amsterdam
    "es_esef": ".MC",   # BME Madrid
    "it_esef": ".MI",   # Borsa Italiana Milan
    "se_esef": ".ST",   # Nasdaq Stockholm
}


def get_yfinance_suffix(market_id: str) -> str:
    """Return the yfinance ticker suffix for a market."""
    return _MARKET_SUFFIX.get(market_id, "")


def fetch_ohlcv_yfinance(
    ticker: str,
    market_id: str = "",
    years: int = 5,
) -> pd.DataFrame:
    """Fetch OHLCV data via yfinance.

    Parameters
    ----------
    ticker:
        Raw ticker symbol (e.g. "AAPL", "7203", "005930").
    market_id:
        Market identifier to determine the suffix.
    years:
        Number of years of history.

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume.
    Empty DataFrame on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed. pip install yfinance>=1.2.0")
        return pd.DataFrame()

    suffix = get_yfinance_suffix(market_id)
    yf_ticker = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker

    try:
        # yfinance handles rate limiting and retries internally via sessions.
        # Set timeout to 60s to handle slow connections gracefully.
        df = yf.download(
            yf_ticker,
            period=f"{years}y",
            auto_adjust=False,
            progress=False,
            timeout=60,
        )

        if df is None or df.empty:
            logger.debug("yfinance returned empty for %s", yf_ticker)
            return pd.DataFrame()

        # yfinance >= 1.2.0 returns MultiIndex columns like ('Open', 'AAPL')
        # for single-ticker downloads. Flatten to simple column names.
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)

        # Normalize columns to lowercase
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]

        # Ensure standard OHLCV columns
        rename_map = {
            "adj close": "adj_close",
        }
        df = df.rename(columns=rename_map)

        # Keep only OHLCV columns
        keep = ["date", "open", "high", "low", "close", "volume"]
        available = [c for c in keep if c in df.columns]
        df = df[available]

        logger.info("yfinance fetched %d rows for %s", len(df), yf_ticker)
        return df

    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", yf_ticker, exc)
        return pd.DataFrame()
