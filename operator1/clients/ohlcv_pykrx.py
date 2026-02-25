"""Korea OHLCV provider using pykrx.

Primary OHLCV source for Korean market (KOSPI/KOSDAQ).
Fallback: yfinance with .KS/.KQ suffix.

No API key required. Direct KRX scraping.
Research: .roo/research/ohlcv-per-region-2026-02-25.md
Package: pykrx 1.2.4
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_ohlcv_pykrx(
    ticker: str,
    years: int = 5,
) -> pd.DataFrame:
    """Fetch Korean OHLCV via pykrx.

    Parameters
    ----------
    ticker:
        Korean stock code (e.g. "005930" for Samsung).
    years:
        Number of years of history.

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume.
    """
    try:
        from pykrx import stock
    except ImportError:
        logger.debug("pykrx not installed; falling back to yfinance")
        return pd.DataFrame()

    try:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365 * years)

        # Verbatim from pykrx README
        df = stock.get_market_ohlcv(
            start_dt.strftime("%Y%m%d"),
            end_dt.strftime("%Y%m%d"),
            ticker,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        # pykrx returns Korean column names: 날짜, 시가, 고가, 저가, 종가, 거래량
        col_map = {
            "날짜": "date",
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
        }
        df = df.rename(columns=col_map)

        # Also handle English column names if present
        df.columns = [c.lower() for c in df.columns]

        keep = ["date", "open", "high", "low", "close", "volume"]
        available = [c for c in keep if c in df.columns]
        df = df[available]

        logger.info("pykrx fetched %d rows for %s", len(df), ticker)
        return df

    except Exception as exc:
        logger.warning("pykrx failed for %s: %s", ticker, exc)
        return pd.DataFrame()
