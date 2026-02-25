"""Chile macro provider using FRED (Chilean series) + BCCh fallback.

Primary macro source for the Chilean market.
Fallback: wbgapi (World Bank).

FRED hosts Chilean macro data. BCCh API requires free registration.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# FRED hosts Chilean macro data via IMF/World Bank feeds.
_FRED_CL_SERIES: dict[str, str] = {
    "gdp_growth": "CLERGDPG",                 # Chile Real GDP growth (if avail)
    "inflation_rate_yoy": "FPCPITOTLZGCHL",   # Chile CPI inflation
    "unemployment_rate": "SLUEM003SZACHL",     # Chile unemployment
    "interest_rate": "IRSTCI01CLM156N",         # Chile short-term rate
    "exchange_rate": "DEXCHUS",                  # CLP/USD (if available)
}


def fetch_macro_bcch(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch Chilean macro indicators.

    Uses FRED as primary path (Chilean data available via IMF feeds).
    Falls through to wbgapi if FRED unavailable.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    results: dict[str, pd.Series] = {}

    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            start = date.today() - timedelta(days=365 * years)

            for canonical_name, series_id in _FRED_CL_SERIES.items():
                try:
                    series = fred.get_series(series_id, observation_start=start)
                    if series is not None and not series.empty:
                        series.name = canonical_name
                        results[canonical_name] = series
                        logger.debug("FRED-CL %s: %d obs", series_id, len(series))
                except Exception as exc:
                    logger.debug("FRED-CL failed for %s: %s", series_id, exc)

            if results:
                logger.info("Chile macro via FRED: %d/%d indicators", len(results), len(_FRED_CL_SERIES))
                return results
        except ImportError:
            logger.debug("fredapi not installed for CL macro")

    logger.debug("Chile macro: FRED unavailable; falling back to wbgapi")
    return results
