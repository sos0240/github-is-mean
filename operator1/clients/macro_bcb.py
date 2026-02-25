"""Brazil macro provider using python-bcb (BCB SGS).

Primary macro source for Brazilian market.
Fallback: wbgapi (World Bank).

No API key required. Brazilian Central Bank open data.
Research: .roo/research/macro-per-region-2026-02-25.md
Package: python-bcb 0.3.3
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# BCB SGS series codes for canonical macro variables
_BCB_SERIES: dict[str, int] = {
    "interest_rate": 432,      # SELIC rate
    "inflation_rate_yoy": 433, # IPCA inflation
    "gdp_growth": 4380,        # GDP monthly proxy (IBC-Br)
    "unemployment_rate": 24369,  # Unemployment (PNAD)
    "exchange_rate": 1,         # USD/BRL exchange rate
}


def fetch_macro_bcb(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch Brazilian macro indicators from BCB SGS.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    try:
        from bcb import sgs
    except ImportError:
        logger.debug("python-bcb not installed; falling back to wbgapi")
        return {}

    start = (date.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    results: dict[str, pd.Series] = {}

    for canonical_name, series_code in _BCB_SERIES.items():
        try:
            # Verbatim from python-bcb README
            df = sgs.get({canonical_name: series_code}, start=start)
            if df is not None and not df.empty:
                series = df.iloc[:, 0]
                series.name = canonical_name
                results[canonical_name] = series
                logger.debug("BCB %s (code %d): %d observations", canonical_name, series_code, len(series))
        except Exception as exc:
            logger.debug("BCB failed for %s: %s", canonical_name, exc)

    logger.info("BCB fetched %d/%d indicators", len(results), len(_BCB_SERIES))
    return results
