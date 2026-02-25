"""US macro provider using fredapi (Federal Reserve FRED).

Primary macro source for US market.
Fallback: wbgapi (World Bank).

Requires FRED_API_KEY (free registration).
Research: .roo/research/macro-per-region-2026-02-25.md
Package: fredapi 0.5.2
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# FRED series IDs for canonical macro variables
_FRED_SERIES: dict[str, str] = {
    "gdp_growth": "A191RL1Q225SBEA",  # Real GDP growth (quarterly, annualized)
    "inflation_rate_yoy": "CPIAUCSL",  # CPI for All Urban Consumers
    "unemployment_rate": "UNRATE",  # Unemployment Rate
    "interest_rate": "FEDFUNDS",  # Federal Funds Rate
    "exchange_rate": "DTWEXBGS",  # Trade Weighted US Dollar Index
}


def fetch_macro_fred(
    api_key: str = "",
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch US macro indicators from FRED.

    Parameters
    ----------
    api_key:
        FRED API key. Falls back to FRED_API_KEY env var.
    years:
        Number of years of history.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        logger.debug("FRED_API_KEY not set; falling back to wbgapi")
        return {}

    try:
        from fredapi import Fred
    except ImportError:
        logger.debug("fredapi not installed; falling back to wbgapi")
        return {}

    try:
        fred = Fred(api_key=key)
    except Exception as exc:
        logger.warning("fredapi init failed: %s", exc)
        return {}

    from datetime import date, timedelta
    start = date.today() - timedelta(days=365 * years)
    results: dict[str, pd.Series] = {}

    for canonical_name, series_id in _FRED_SERIES.items():
        try:
            series = fred.get_series(series_id, observation_start=start)
            if series is not None and not series.empty:
                series.name = canonical_name
                results[canonical_name] = series
                logger.debug("FRED %s: %d observations", series_id, len(series))
        except Exception as exc:
            logger.debug("FRED failed for %s: %s", series_id, exc)

    logger.info("FRED fetched %d/%d indicators", len(results), len(_FRED_SERIES))
    return results
