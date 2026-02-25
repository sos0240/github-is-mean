"""Macro provider dispatcher -- routes to per-region primary or wbgapi fallback.

Each market has a primary macro source (government central bank API).
If the primary fails or its library is not installed, falls back to
wbgapi (World Bank, global coverage, no key needed).

Research: .roo/research/macro-per-region-2026-02-25.md
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Country code -> primary macro fetcher
_PRIMARY_FETCHERS: dict[str, str] = {
    "US": "fred",
    "EU": "ecb",
    "DE": "ecb",
    "FR": "ecb",
    "BR": "bcb",
    "MX": "banxico",
    # UK (ONS), JP (e-Stat), KR (KOSIS), CL (BCC) -- use wbgapi for now
    # These need API keys and direct HTTP implementation
}


def fetch_macro(
    country_iso2: str,
    market_id: str = "",
    secrets: dict | None = None,
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch macro data for a country, using per-region primary + wbgapi fallback.

    Parameters
    ----------
    country_iso2:
        ISO-2 country code (e.g. "US", "JP", "BR").
    market_id:
        Market identifier (for logging).
    secrets:
        API key dictionary.
    years:
        Number of years of history.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    if secrets is None:
        secrets = {}

    results: dict[str, pd.Series] = {}
    cc = country_iso2.upper()

    # Try per-region primary first
    primary = _PRIMARY_FETCHERS.get(cc)

    if primary == "fred":
        try:
            from operator1.clients.macro_fredapi import fetch_macro_fred
            results = fetch_macro_fred(
                api_key=secrets.get("FRED_API_KEY", ""),
                years=years,
            )
        except Exception as exc:
            logger.debug("FRED primary failed: %s", exc)

    elif primary == "ecb":
        try:
            from operator1.clients.macro_sdmx import fetch_macro_ecb
            results = fetch_macro_ecb(years=years)
        except Exception as exc:
            logger.debug("ECB primary failed: %s", exc)

    elif primary == "bcb":
        try:
            from operator1.clients.macro_bcb import fetch_macro_bcb
            results = fetch_macro_bcb(years=years)
        except Exception as exc:
            logger.debug("BCB primary failed: %s", exc)

    elif primary == "banxico":
        try:
            from operator1.clients.macro_banxico import fetch_macro_banxico
            results = fetch_macro_banxico(
                api_token=secrets.get("BANXICO_TOKEN", ""),
                years=years,
            )
        except Exception as exc:
            logger.debug("Banxico primary failed: %s", exc)

    # Fallback to wbgapi if primary returned nothing or insufficient data
    if len(results) < 3:  # Need at least GDP, inflation, unemployment
        if primary and results:
            logger.info(
                "Per-region macro (%s) returned only %d indicators; "
                "supplementing with wbgapi",
                primary, len(results),
            )
        elif primary:
            logger.info(
                "Per-region macro (%s) returned empty; falling back to wbgapi",
                primary,
            )

        try:
            from operator1.clients.macro_wbgapi import fetch_macro_wbgapi
            wb_results = fetch_macro_wbgapi(cc, years=years)

            # Merge: only add indicators not already fetched from primary
            for key, series in wb_results.items():
                if key not in results:
                    results[key] = series

        except Exception as exc:
            logger.warning("wbgapi fallback also failed for %s: %s", cc, exc)

    if not results:
        logger.warning("No macro data available for %s (market: %s)", cc, market_id)

    return results
