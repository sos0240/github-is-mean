"""Japan macro provider using e-Stat (Statistics Bureau of Japan) API.

Primary macro source for the Japanese market.
Fallback: wbgapi (World Bank).

Requires ESTAT_API_KEY (free registration at https://api.e-stat.go.jp/).
Alternative: uses FRED for some Japanese series if FRED_API_KEY is available.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# FRED has several Japanese macro series available without needing
# a separate e-Stat key. Use FRED as primary path with e-Stat HTTP
# fallback, since fredapi is already a project dependency.
_FRED_JP_SERIES: dict[str, str] = {
    "gdp_growth": "JPNRGDPEXP",       # Japan Real GDP growth
    "inflation_rate_yoy": "FPCPITOTLZGJPN",  # Japan CPI inflation (World Bank via FRED)
    "unemployment_rate": "LRUNTTTTJPM156S",    # Japan unemployment rate (OECD via FRED)
    "interest_rate": "IRSTCI01JPM156N",        # Japan short-term interest rate
    "exchange_rate": "DEXJPUS",                 # JPY/USD exchange rate
}


def fetch_macro_estat(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch Japanese macro indicators.

    Uses FRED (which hosts many Japanese economic series) as the
    primary path, since fredapi is already installed. Falls back
    to direct e-Stat HTTP API if FRED is unavailable.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    results: dict[str, pd.Series] = {}

    # Path 1: FRED (has comprehensive Japanese data)
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            start = date.today() - timedelta(days=365 * years)

            for canonical_name, series_id in _FRED_JP_SERIES.items():
                try:
                    series = fred.get_series(series_id, observation_start=start)
                    if series is not None and not series.empty:
                        series.name = canonical_name
                        results[canonical_name] = series
                        logger.debug("FRED-JP %s: %d obs", series_id, len(series))
                except Exception as exc:
                    logger.debug("FRED-JP failed for %s: %s", series_id, exc)

            if results:
                logger.info("Japan macro via FRED: %d/%d indicators", len(results), len(_FRED_JP_SERIES))
                return results
        except ImportError:
            logger.debug("fredapi not installed for JP macro")

    # Path 2: Direct e-Stat HTTP API (needs ESTAT_API_KEY)
    estat_key = os.environ.get("ESTAT_API_KEY", "")
    if estat_key:
        results = _fetch_from_estat(estat_key, years)

    if not results:
        logger.debug("Japan macro: no data from FRED or e-Stat; falling back to wbgapi")

    return results


def _fetch_from_estat(api_key: str, years: int) -> dict[str, pd.Series]:
    """Fetch from e-Stat JSON API."""
    import requests

    # e-Stat statsDataId for key indicators
    _ESTAT_IDS: dict[str, str] = {
        "gdp_growth": "0003109741",
        "inflation_rate_yoy": "0003143513",
        "unemployment_rate": "0003009482",
    }

    results: dict[str, pd.Series] = {}
    base = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"

    for canonical_name, stats_id in _ESTAT_IDS.items():
        try:
            params = {
                "appId": api_key,
                "statsDataId": stats_id,
                "limit": 200,
            }
            resp = requests.get(base, params=params, timeout=30)
            if resp.status_code != 200:
                continue

            data = resp.json()
            stat_data = (
                data.get("GET_STATS_DATA", {})
                .get("STATISTICAL_DATA", {})
                .get("DATA_INF", {})
                .get("VALUE", [])
            )
            if not stat_data:
                continue

            cutoff = date.today() - timedelta(days=365 * years)
            dates = []
            values = []
            for item in stat_data:
                time_str = item.get("@time", "")
                val = item.get("$", "")
                if not time_str or not val:
                    continue
                try:
                    # e-Stat time format: "2024000101" or "2024Q1" etc
                    dt = pd.to_datetime(time_str[:4] + "-01-01")
                    if dt.date() < cutoff:
                        continue
                    dates.append(dt)
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue

            if dates:
                series = pd.Series(values, index=dates).sort_index()
                series.name = canonical_name
                results[canonical_name] = series

        except Exception as exc:
            logger.debug("e-Stat failed for %s: %s", canonical_name, exc)

    logger.info("e-Stat fetched %d indicators", len(results))
    return results
