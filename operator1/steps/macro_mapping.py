"""Step 0.2 -- Macro data mapping.

Fetches macroeconomic indicators (GDP, inflation, interest rates,
unemployment, currency) for the selected market's region using the
per-region macro APIs registered in ``pit_registry.py``.

The fetched time-series data is packaged into a ``MacroDataset`` that
downstream modules (``macro_alignment.py``, ``macro_quadrant.py``,
``survival_mode.py``) consume.

Supported macro sources (all free):
  - FRED (US)            - ECB SDW (EU)
  - ONS (UK)             - Bundesbank (DE)
  - INSEE (FR)           - e-Stat (JP)
  - KOSIS (KR)           - DGBAS (TW)
  - BCB (BR)             - BCCh (CL)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical indicator names expected by downstream modules
# (macro_alignment.py, macro_quadrant.py, survival_mode.py).
_MACRO_TO_CANONICAL = {
    "gdp": "gdp_growth",
    "inflation": "inflation_rate_yoy",
    "interest_rate": "real_interest_rate",
    "unemployment": "unemployment_rate",
    "currency": "official_exchange_rate_lcu_per_usd",
}


@dataclass
class MacroDataset:
    """Container for macro data aligned to canonical variable names.

    ``indicators`` maps canonical names (e.g. ``gdp_growth``) to
    DataFrames with ``year`` and ``value`` columns so that
    ``macro_alignment.py`` can convert them to daily frequency.
    """

    country_iso3: str = ""
    indicators: dict[str, pd.DataFrame] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    gemini_suggestions: dict[str, str] = field(default_factory=dict)


def _series_to_yearly_df(series: pd.Series | None) -> pd.DataFrame:
    """Convert a macro pd.Series (DatetimeIndex) into a yearly DataFrame.

    Downstream consumers (``macro_alignment``, ``macro_quadrant``) expect
    DataFrames with ``year`` and ``value`` columns.  If the input series
    has sub-yearly frequency we resample to annual (last observation).
    """
    if series is None or series.empty:
        return pd.DataFrame(columns=["year", "value"])

    # Ensure DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception:
            return pd.DataFrame(columns=["year", "value"])

    # Resample to annual if the data is monthly/quarterly
    try:
        yearly = series.resample("YE").last().dropna()
    except Exception:
        yearly = series.groupby(series.index.year).last()
        yearly.index = pd.to_datetime(yearly.index.astype(str))

    df = pd.DataFrame({
        "year": yearly.index.year,
        "value": yearly.values,
    })
    return df.dropna(subset=["value"]).reset_index(drop=True)


def fetch_macro_data(
    country_iso2: str,
    market_id: str = "",
    macro_api_info: Any = None,
    macro_raw: dict[str, "pd.Series | None"] | None = None,
    secrets: dict[str, str] | None = None,
    **_legacy_kwargs: Any,
) -> MacroDataset:
    """Fetch macro indicators for a region and return a ``MacroDataset``.

    This function supports two modes:

    1. **Pre-fetched** -- pass ``macro_raw`` (output of
       ``macro_client.fetch_macro_indicators()``) to convert it into
       a ``MacroDataset`` without re-fetching.
    2. **Auto-fetch** -- pass ``market_id`` (or ``macro_api_info``) to
       have this function resolve the macro API and fetch the data.

    Parameters
    ----------
    country_iso2:
        2-letter ISO country code for the target market.
    market_id:
        PIT market identifier (e.g. ``"us_sec_edgar"``).  Used to
        resolve the macro API when ``macro_api_info`` is not provided.
    macro_api_info:
        A ``MacroAPIInfo`` instance from the registry.  Takes priority
        over ``market_id`` lookup.
    macro_raw:
        Pre-fetched dict from ``macro_client.fetch_macro_indicators()``.
        When provided, no HTTP calls are made.
    secrets:
        API key dictionary (for macro sources that require registration).

    Returns
    -------
    MacroDataset
        Contains yearly DataFrames for each available indicator and a
        list of indicators that could not be fetched.
    """
    if secrets is None:
        secrets = {}

    # ------------------------------------------------------------------
    # Resolve macro API if not provided
    # ------------------------------------------------------------------
    if macro_api_info is None and market_id:
        try:
            from operator1.clients.pit_registry import get_macro_api_for_market
            macro_api_info = get_macro_api_for_market(market_id)
        except Exception as exc:
            logger.warning("Could not resolve macro API for market %s: %s", market_id, exc)

    if macro_api_info is None and macro_raw is None:
        logger.info(
            "No macro API available for country %s -- returning empty MacroDataset",
            country_iso2,
        )
        return MacroDataset(
            country_iso3=country_iso2,
            missing=list(_MACRO_TO_CANONICAL.values()),
        )

    # ------------------------------------------------------------------
    # Fetch indicators if not pre-fetched
    # ------------------------------------------------------------------
    if macro_raw is None:
        logger.info(
            "Macro data fetching skipped (government macro APIs removed)."
        )
        return MacroDataset(
            country_iso3=country_iso2,
            missing=list(_MACRO_TO_CANONICAL.values()),
        )

    # ------------------------------------------------------------------
    # Convert raw series -> yearly DataFrames for downstream consumers
    # ------------------------------------------------------------------
    indicators: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for raw_name, canonical_name in _MACRO_TO_CANONICAL.items():
        series = macro_raw.get(raw_name) if macro_raw else None
        if series is not None and not series.empty:
            yearly_df = _series_to_yearly_df(series)
            if not yearly_df.empty:
                indicators[canonical_name] = yearly_df
                logger.info(
                    "  %s -> %s: %d yearly observations",
                    raw_name, canonical_name, len(yearly_df),
                )
            else:
                missing.append(canonical_name)
                logger.debug("  %s: converted to empty yearly DF", raw_name)
        else:
            missing.append(canonical_name)
            logger.debug("  %s: no data available", raw_name)

    n_ok = len(indicators)
    n_miss = len(missing)
    logger.info(
        "MacroDataset built: %d indicators available, %d missing (country=%s)",
        n_ok, n_miss, country_iso2,
    )

    return MacroDataset(
        country_iso3=country_iso2,
        indicators=indicators,
        missing=missing,
    )


def save_macro_metadata(
    dataset: MacroDataset,
    output_path: str = "cache/macro_metadata.json",
) -> None:
    """Persist a summary of the MacroDataset to disk for auditing.

    Parameters
    ----------
    dataset:
        The ``MacroDataset`` returned by ``fetch_macro_data()``.
    output_path:
        File path for the JSON metadata file.
    """
    import json
    import os

    meta = {
        "country": dataset.country_iso3,
        "indicators_available": list(dataset.indicators.keys()),
        "indicators_missing": dataset.missing,
        "indicator_stats": {},
    }

    for name, df in dataset.indicators.items():
        if not df.empty and "value" in df.columns:
            meta["indicator_stats"][name] = {
                "n_years": len(df),
                "year_min": int(df["year"].min()) if "year" in df.columns else None,
                "year_max": int(df["year"].max()) if "year" in df.columns else None,
                "value_latest": float(df["value"].iloc[-1]),
            }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    logger.info("Macro metadata saved: %s", output_path)
