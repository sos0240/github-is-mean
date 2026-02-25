"""EU/DE/FR macro provider using sdmx1 (ECB / Bundesbank / INSEE).

Primary macro source for European markets.
Fallback: wbgapi (World Bank).

No API key for ECB and Bundesbank. INSEE may need key.
Research: .roo/research/macro-per-region-2026-02-25.md
Package: sdmx1 2.25.1
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ECB series keys for canonical macro variables
_ECB_SERIES: dict[str, tuple[str, dict]] = {
    "gdp_growth": ("MNA", {"FREQ": "A", "REF_AREA": "I8", "STS_INDICATOR": "OVGD", "STS_SUFFIX": "PCH"}),
    "inflation_rate_yoy": ("ICP", {"FREQ": "A", "REF_AREA": "U2", "ICP_ITEM": "000000"}),
    "interest_rate": ("FM", {"FREQ": "M", "REF_AREA": "U2", "FM_TYPE": "IR"}),
}


def fetch_macro_ecb(
    years: int = 10,
) -> dict[str, pd.Series]:
    """Fetch EU macro indicators from ECB via sdmx1.

    Returns
    -------
    Dict mapping canonical indicator name -> pd.Series.
    """
    try:
        import sdmx
    except ImportError:
        logger.debug("sdmx1 not installed; falling back to wbgapi")
        return {}

    results: dict[str, pd.Series] = {}

    try:
        ecb = sdmx.Client("ECB")

        for canonical_name, (flow, key) in _ECB_SERIES.items():
            try:
                data = ecb.data(flow, key=key)
                if data and data.data:
                    for ds in data.data:
                        for series_key, obs in ds.series.items():
                            values = []
                            dates = []
                            for time_period, value in obs.items():
                                dates.append(str(time_period))
                                values.append(float(value[0].value) if value else None)
                            if values:
                                series = pd.Series(values, index=pd.to_datetime(dates))
                                series.name = canonical_name
                                results[canonical_name] = series
                                break  # Take first matching series
            except Exception as exc:
                logger.debug("ECB %s failed: %s", canonical_name, exc)

    except Exception as exc:
        logger.warning("ECB client failed: %s", exc)

    logger.info("ECB fetched %d/%d indicators", len(results), len(_ECB_SERIES))
    return results
