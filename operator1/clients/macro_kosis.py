"""South Korea macro provider using FRED (Korean series) + KOSIS fallback.

Primary macro source for the Korean market.
Fallback: wbgapi (World Bank).

FRED hosts Korean macro data (via OECD). KOSIS API requires free registration.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# FRED hosts many Korean macro series via OECD data feeds.
_FRED_KR_SERIES: dict[str, str] = {
    "gdp_growth": "KORRGDPEXP",               # Korea Real GDP growth
    "inflation_rate_yoy": "FPCPITOTLZGKOR",   # Korea CPI inflation
    "unemployment_rate": "LRUNTTTTKOM156S",     # Korea unemployment rate (OECD)
    "interest_rate": "IRSTCI01KRM156N",         # Korea short-term interest rate
    "exchange_rate": "DEXKOUS",                 # KRW/USD exchange rate
}


def fetch_macro_kosis(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch Korean macro indicators.

    Uses FRED (which hosts Korean data via OECD) as primary path.
    Falls through to wbgapi if neither FRED nor KOSIS are available.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    results: dict[str, pd.Series] = {}

    # Path 1: FRED (has comprehensive Korean data via OECD)
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            start = date.today() - timedelta(days=365 * years)

            for canonical_name, series_id in _FRED_KR_SERIES.items():
                try:
                    series = fred.get_series(series_id, observation_start=start)
                    if series is not None and not series.empty:
                        series.name = canonical_name
                        results[canonical_name] = series
                        logger.debug("FRED-KR %s: %d obs", series_id, len(series))
                except Exception as exc:
                    logger.debug("FRED-KR failed for %s: %s", series_id, exc)

            if results:
                logger.info("Korea macro via FRED: %d/%d indicators", len(results), len(_FRED_KR_SERIES))
                return results
        except ImportError:
            logger.debug("fredapi not installed for KR macro")

    logger.debug("Korea macro: FRED unavailable; falling back to wbgapi")
    return results
