"""UK macro provider using FRED (UK series via OECD) + wbgapi fallback.

Primary macro source for the UK market.
Fallback: wbgapi (World Bank).

NOTE: The original ONS time series API (api.ons.gov.uk) was decommissioned
on 25/11/2024. We use FRED instead, which hosts comprehensive UK data
via OECD and Bank of England feeds.

Requires FRED_API_KEY (free registration).
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# FRED series IDs for UK macro indicators (sourced from OECD/BoE feeds)
_FRED_GB_SERIES: dict[str, str] = {
    "gdp_growth": "CLVMNACSCAB1GQUK",       # UK Real GDP (quarterly)
    "inflation_rate_yoy": "FPCPITOTLZGGBR",  # UK CPI inflation (WB via FRED)
    "unemployment_rate": "LRUNTTTTGBM156S",   # UK unemployment rate (OECD)
    "interest_rate": "IRSTCI01GBM156N",       # UK short-term interest rate
    "exchange_rate": "DEXUSUK",               # USD/GBP exchange rate
}


def fetch_macro_ons(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch UK macro indicators via FRED (UK/OECD series).

    The ONS API was decommissioned Nov 2024. FRED hosts comprehensive
    UK macro data via OECD and Bank of England data feeds.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    results: dict[str, pd.Series] = {}

    fred_key = os.environ.get("FRED_API_KEY", "")
    if not fred_key:
        logger.debug("FRED_API_KEY not set; UK macro falling back to wbgapi")
        return {}

    try:
        from fredapi import Fred
    except ImportError:
        logger.debug("fredapi not installed for UK macro")
        return {}

    try:
        fred = Fred(api_key=fred_key)
    except Exception as exc:
        logger.warning("fredapi init failed for UK macro: %s", exc)
        return {}

    start = date.today() - timedelta(days=365 * years)

    for canonical_name, series_id in _FRED_GB_SERIES.items():
        try:
            series = fred.get_series(series_id, observation_start=start)
            if series is not None and not series.empty:
                series.name = canonical_name
                results[canonical_name] = series
                logger.debug("FRED-GB %s: %d obs", series_id, len(series))
        except Exception as exc:
            logger.debug("FRED-GB failed for %s: %s", series_id, exc)

    logger.info("UK macro via FRED: %d/%d indicators", len(results), len(_FRED_GB_SERIES))
    return results
