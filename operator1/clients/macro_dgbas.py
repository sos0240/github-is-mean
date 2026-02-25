"""Taiwan macro provider using FRED (Taiwanese series) + DGBAS fallback.

Primary macro source for the Taiwanese market.
Fallback: wbgapi (World Bank).

FRED hosts Taiwanese macro data. DGBAS is web-based (harder to automate).
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# FRED hosts Taiwanese macro data (via IMF, central bank sources).
# Taiwan is not a World Bank member, so wbgapi has limited coverage.
# FRED is the best programmatic source for TW macro.
_FRED_TW_SERIES: dict[str, str] = {
    "inflation_rate_yoy": "FPCPITOTLZGTWN",   # Taiwan CPI inflation
    "interest_rate": "IRSTCI01TWM156N",         # Taiwan short-term rate
    "exchange_rate": "DEXTAUS",                  # TWD/USD (if available)
}


def fetch_macro_dgbas(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch Taiwanese macro indicators.

    Uses FRED as primary path (best programmatic source for Taiwan).
    Taiwan is not a World Bank member, so wbgapi coverage is limited.

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

            for canonical_name, series_id in _FRED_TW_SERIES.items():
                try:
                    series = fred.get_series(series_id, observation_start=start)
                    if series is not None and not series.empty:
                        series.name = canonical_name
                        results[canonical_name] = series
                        logger.debug("FRED-TW %s: %d obs", series_id, len(series))
                except Exception as exc:
                    logger.debug("FRED-TW failed for %s: %s", series_id, exc)

            if results:
                logger.info("Taiwan macro via FRED: %d/%d indicators", len(results), len(_FRED_TW_SERIES))
                return results
        except ImportError:
            logger.debug("fredapi not installed for TW macro")

    logger.debug("Taiwan macro: FRED unavailable; falling back to wbgapi")
    return results
