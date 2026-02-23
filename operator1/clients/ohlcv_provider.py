"""OHLCV price data provider -- free-tier sources for all markets.

Fetches raw (unadjusted) OHLCV data from free-tier APIs.  Raw exchange
OHLCV is inherently point-in-time: a trade at a given price on a given
date is an immutable fact that never changes retroactively.

Primary source: **Alpha Vantage** (free tier, 25 requests/day, global)
  - Register at https://www.alphavantage.co/support/#api-key (free, instant)
  - Returns raw OHLCV via TIME_SERIES_DAILY endpoint
  - Global coverage: US, UK, EU, Japan, Korea, Taiwan, Brazil, Chile

Supplementary exchange-specific sources:
  - **TWSE** (Taiwan): direct from Taiwan Stock Exchange, no key needed
  - Additional exchange APIs can be added per region

Usage:
    from operator1.clients.ohlcv_provider import fetch_ohlcv
    df = fetch_ohlcv("AAPL", market_id="us_sec_edgar")
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alpha Vantage (global, free tier: 25 requests/day)
# ---------------------------------------------------------------------------

_AV_BASE = "https://www.alphavantage.co/query"

# Daily call counter for Alpha Vantage free tier (25 requests/day)
_av_daily_calls: int = 0
_av_daily_reset: float = 0.0  # timestamp of last reset
_AV_FREE_TIER_LIMIT = 25

# Maps market_id -> Alpha Vantage ticker suffix
_AV_SUFFIX: dict[str, str] = {
    "us_sec_edgar": "",          # AAPL
    "uk_companies_house": ".LON",  # BP.LON
    "eu_esef": "",                 # varied
    "fr_esef": ".PAR",            # AIR.PAR
    "de_esef": ".FRK",            # SIE.FRK  (or .DEX)
    "jp_edinet": ".TYO",          # 7203.TYO
    "kr_dart": ".KRX",            # 005930.KRX
    "tw_mops": ".TPE",            # 2330.TPE
    "br_cvm": ".SAO",             # PETR4.SAO
    "cl_cmf": ".SNX",             # SQM.SNX
}

# Standard OHLCV column names
_OHLCV_COLS = ["date", "open", "high", "low", "close", "volume"]


def _to_av_ticker(ticker: str, market_id: str) -> str:
    """Convert a PIT-client ticker to Alpha Vantage format."""
    suffix = _AV_SUFFIX.get(market_id, "")
    if suffix and not ticker.upper().endswith(suffix.upper()):
        return f"{ticker}{suffix}"
    return ticker


def _check_av_daily_limit() -> bool:
    """Check if we're within Alpha Vantage free tier daily limit.

    Resets the counter every 24 hours.  Returns True if a call is allowed.
    """
    global _av_daily_calls, _av_daily_reset
    import time as _time
    now = _time.time()
    # Reset counter every 24 hours
    if now - _av_daily_reset > 86400:
        _av_daily_calls = 0
        _av_daily_reset = now
    if _av_daily_calls >= _AV_FREE_TIER_LIMIT:
        logger.warning(
            "Alpha Vantage daily limit reached (%d/%d). "
            "Upgrade to premium or wait 24h.",
            _av_daily_calls, _AV_FREE_TIER_LIMIT,
        )
        return False
    return True


def _increment_av_counter() -> None:
    """Increment the Alpha Vantage daily call counter."""
    global _av_daily_calls
    _av_daily_calls += 1
    if _av_daily_calls >= _AV_FREE_TIER_LIMIT - 5:
        logger.info(
            "Alpha Vantage daily calls: %d/%d (approaching limit)",
            _av_daily_calls, _AV_FREE_TIER_LIMIT,
        )


def _fetch_alpha_vantage(
    ticker: str,
    market_id: str,
    api_key: str,
    outputsize: str = "full",
) -> pd.DataFrame:
    """Fetch daily OHLCV from Alpha Vantage TIME_SERIES_DAILY.

    Returns raw (unadjusted) OHLCV data.  The free tier allows 25
    requests per day, which is sufficient for target + a few linked entities.
    Tracks daily usage to avoid silent exhaustion of the free tier.
    """
    if not _check_av_daily_limit():
        return pd.DataFrame(columns=_OHLCV_COLS)

    av_ticker = _to_av_ticker(ticker, market_id)

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": av_ticker,
        "outputsize": outputsize,
        "apikey": api_key,
    }

    try:
        data = cached_get(_AV_BASE, params=params)
        _increment_av_counter()
    except HTTPError as exc:
        logger.warning("Alpha Vantage request failed: %s", exc)
        _increment_av_counter()
        return pd.DataFrame(columns=_OHLCV_COLS)

    if not isinstance(data, dict):
        return pd.DataFrame(columns=_OHLCV_COLS)

    # Check for error messages
    if "Error Message" in data:
        logger.warning("Alpha Vantage error for %s: %s", av_ticker, data["Error Message"])
        return pd.DataFrame(columns=_OHLCV_COLS)

    if "Note" in data:
        logger.warning("Alpha Vantage rate limit: %s", data["Note"])
        return pd.DataFrame(columns=_OHLCV_COLS)

    time_series = data.get("Time Series (Daily)", {})
    if not time_series:
        # Try alternate ticker without suffix
        if _AV_SUFFIX.get(market_id, ""):
            logger.info("Retrying Alpha Vantage with raw ticker: %s", ticker)
            params["symbol"] = ticker
            try:
                data = cached_get(_AV_BASE, params=params)
                time_series = data.get("Time Series (Daily)", {})
            except Exception:
                pass

    if not time_series:
        logger.warning("Alpha Vantage returned no data for %s", av_ticker)
        return pd.DataFrame(columns=_OHLCV_COLS)

    rows: list[dict] = []
    for date_str, values in time_series.items():
        try:
            rows.append({
                "date": date_str,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": int(float(values.get("5. volume", 0))),
            })
        except (ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame(columns=_OHLCV_COLS)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "Alpha Vantage OHLCV: %d days for %s (%s to %s)",
        len(df), av_ticker,
        df["date"].iloc[0].date(),
        df["date"].iloc[-1].date(),
    )

    return df


# ---------------------------------------------------------------------------
# TWSE (Taiwan Stock Exchange -- free, no key, exchange-operated)
# ---------------------------------------------------------------------------

_TWSE_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"


def _fetch_twse(ticker: str, months: int = 24) -> pd.DataFrame:
    """Fetch daily OHLCV from TWSE (Taiwan Stock Exchange).

    Direct from the exchange, no API key needed.
    """
    rows: list[dict] = []
    today = date.today()

    for month_offset in range(months):
        d = today.replace(day=1) - timedelta(days=month_offset * 30)
        date_str = f"{d.year}{d.month:02d}01"

        try:
            data = cached_get(
                _TWSE_URL,
                params={
                    "response": "json",
                    "date": date_str,
                    "stockNo": ticker,
                },
            )
            if not isinstance(data, dict):
                continue

            fields = data.get("data", [])
            for row in fields:
                if len(row) < 9:
                    continue
                try:
                    # TWSE columns: date, volume, turnover, open, high, low, close, delta, tx_count
                    roc_date = row[0]  # ROC date format: 113/01/02
                    parts = roc_date.split("/")
                    if len(parts) == 3:
                        ce_year = int(parts[0]) + 1911
                        dt = f"{ce_year}-{parts[1]}-{parts[2]}"
                    else:
                        continue

                    # Parse numeric values (may have commas)
                    vol = row[1].replace(",", "")
                    rows.append({
                        "date": dt,
                        "open": float(row[3].replace(",", "")),
                        "high": float(row[4].replace(",", "")),
                        "low": float(row[5].replace(",", "")),
                        "close": float(row[6].replace(",", "")),
                        "volume": int(float(vol)),
                    })
                except (ValueError, TypeError, IndexError):
                    continue
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=_OHLCV_COLS)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    logger.info(
        "TWSE OHLCV: %d days for %s (%s to %s)",
        len(df), ticker,
        df["date"].iloc[0].date(),
        df["date"].iloc[-1].date(),
    )

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_ohlcv(
    ticker: str,
    market_id: str = "us_sec_edgar",
    api_key: str = "",
    years: float = 2.0,
) -> pd.DataFrame:
    """Fetch OHLCV price data from the best available free source.

    Routing:
    - Taiwan (tw_mops): TWSE exchange API (free, no key)
    - All others: Alpha Vantage (free tier, key required)

    Parameters
    ----------
    ticker:
        Company ticker symbol.
    market_id:
        PIT market identifier for suffix mapping.
    api_key:
        Alpha Vantage API key (free registration).
    years:
        Lookback window in years.

    Returns
    -------
    pd.DataFrame with columns: date, open, high, low, close, volume.
    Empty DataFrame if no source available or fetch fails.
    """
    if not ticker:
        return pd.DataFrame(columns=_OHLCV_COLS)

    # Taiwan: use TWSE directly (exchange-operated, no key)
    if market_id == "tw_mops":
        result = _fetch_twse(ticker, months=int(years * 12))
        if not result.empty:
            return result

    # All regions: Alpha Vantage (free tier, key required)
    if api_key:
        result = _fetch_alpha_vantage(
            ticker, market_id, api_key,
            outputsize="full" if years > 0.5 else "compact",
        )
        if not result.empty:
            # Trim to requested years
            cutoff = pd.Timestamp(date.today() - timedelta(days=int(years * 365)))
            result = result[result["date"] >= cutoff].reset_index(drop=True)
            return result

    if not api_key:
        logger.info(
            "ALPHA_VANTAGE_API_KEY not set -- cannot fetch OHLCV for %s. "
            "Register for free at https://www.alphavantage.co/support/#api-key",
            ticker,
        )

    return pd.DataFrame(columns=_OHLCV_COLS)
