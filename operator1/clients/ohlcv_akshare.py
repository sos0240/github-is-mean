"""China OHLCV provider using akshare.

Primary OHLCV source for Chinese market (SSE/SZSE).
Fallback: yfinance with .SS/.SZ suffix.

No API key required.
Research: .roo/research/ohlcv-per-region-2026-02-25.md
Package: akshare 1.18.29
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_ohlcv_akshare(
    ticker: str,
    years: int = 5,
) -> pd.DataFrame:
    """Fetch Chinese A-share OHLCV via akshare.

    Parameters
    ----------
    ticker:
        Chinese stock code (e.g. "600519" for Kweichow Moutai).
    years:
        Number of years of history.

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.debug("akshare not installed; falling back to yfinance")
        return pd.DataFrame()

    try:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365 * years)

        # Verbatim from akshare docs
        df = ak.stock_zh_a_hist(
            symbol=ticker,
            period="daily",
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # akshare returns Chinese column names
        col_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
        df = df.rename(columns=col_map)
        df.columns = [c.lower() for c in df.columns]

        keep = ["date", "open", "high", "low", "close", "volume"]
        available = [c for c in keep if c in df.columns]
        df = df[available]

        logger.info("akshare fetched %d rows for %s", len(df), ticker)
        return df

    except Exception as exc:
        logger.warning("akshare failed for %s: %s", ticker, exc)
        return pd.DataFrame()
