"""UK macro provider using ONS (Office for National Statistics) API.

Primary macro source for the UK market.
Fallback: wbgapi (World Bank).

No API key required. ONS open data.
API docs: https://developer.ons.gov.uk/
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# ONS time series IDs for canonical macro variables
# These are the 4-character CDID codes used by the ONS time series API.
_ONS_SERIES: dict[str, str] = {
    "gdp_growth": "IHYQ",         # GDP quarter on quarter growth rate
    "inflation_rate_yoy": "D7G7",  # CPI annual rate (all items)
    "unemployment_rate": "MGSX",   # Unemployment rate (aged 16+)
    "interest_rate": "IUMABEDR",   # Bank of England base rate
    "exchange_rate": "XUMAUSS",    # USD/GBP spot exchange rate
}

_ONS_BASE = "https://api.ons.gov.uk/timeseries"


def fetch_macro_ons(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch UK macro indicators from ONS.

    Uses the ONS time series API (free, no key required).

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series with DatetimeIndex.
    """
    import requests

    results: dict[str, pd.Series] = {}

    for canonical_name, cdid in _ONS_SERIES.items():
        try:
            url = f"{_ONS_BASE}/{cdid}/dataset/months/data"
            resp = requests.get(url, timeout=30, headers={"Accept": "application/json"})

            if resp.status_code != 200:
                # Try quarters endpoint for GDP
                if canonical_name == "gdp_growth":
                    url = f"{_ONS_BASE}/{cdid}/dataset/quarters/data"
                    resp = requests.get(url, timeout=30, headers={"Accept": "application/json"})

            if resp.status_code != 200:
                logger.debug("ONS %s returned %d", cdid, resp.status_code)
                continue

            data = resp.json()

            # ONS returns {"months": [{"date": "...", "value": "...", ...}, ...]}
            # or {"quarters": [...]}
            observations = data.get("months") or data.get("quarters") or data.get("years") or []

            if not observations:
                continue

            cutoff = date.today() - timedelta(days=365 * years)
            dates = []
            values = []
            for obs in observations:
                obs_date = obs.get("date", "")
                obs_val = obs.get("value", "")
                if not obs_date or not obs_val:
                    continue
                try:
                    dt = pd.to_datetime(obs_date)
                    if dt.date() < cutoff:
                        continue
                    dates.append(dt)
                    values.append(float(obs_val))
                except (ValueError, TypeError):
                    continue

            if dates and values:
                series = pd.Series(values, index=dates).sort_index()
                series.name = canonical_name
                results[canonical_name] = series
                logger.debug("ONS %s (%s): %d observations", canonical_name, cdid, len(series))

        except Exception as exc:
            logger.debug("ONS failed for %s: %s", canonical_name, exc)

    logger.info("ONS fetched %d/%d indicators", len(results), len(_ONS_SERIES))
    return results
