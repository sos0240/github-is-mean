"""Unified macro economic data client.

Fetches key macroeconomic indicators (GDP, inflation, interest rates,
unemployment, currency) from free government APIs for each supported
region.  Used by the survival mode analysis to assess country-level
economic conditions.

Supported sources:
  - FRED (US)            -- Federal Reserve Economic Data
  - ECB SDW (EU)         -- European Central Bank
  - ONS (UK)             -- Office for National Statistics
  - Bundesbank (DE)      -- Deutsche Bundesbank
  - INSEE (FR)           -- French statistics
  - e-Stat (JP)          -- Statistics Bureau of Japan
  - KOSIS (KR)           -- Korean Statistical Information
  - DGBAS (TW)           -- Taiwan statistics
  - BCB (BR)             -- Banco Central do Brasil
  - BCCh (CL)            -- Banco Central de Chile

All sources are free.  Some require a (free) API key registration.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

# Standard lookback for macro data (5 years of context)
_MACRO_LOOKBACK_DAYS = 365 * 5

# FRED demo key rate limit: 5 requests/minute (12s between requests)
_FRED_DEMO_RATE_LIMIT_SECONDS = 12.0
_fred_last_request_time: float = 0.0
_fred_demo_key_warned: bool = False


# ---------------------------------------------------------------------------
# FRED (US Federal Reserve Economic Data)
# ---------------------------------------------------------------------------

def _fetch_fred(api_url: str, series_id: str, api_key: str = "") -> pd.Series:
    """Fetch a single FRED series using fredapi library.

    Primary: fredapi (https://github.com/mortada/fredapi)
    Fallback: Direct FRED REST API

    FRED API: https://fred.stlouisfed.org/docs/api/fred/
    Free API key required (register at https://fred.stlouisfed.org/docs/api/api_key.html)

    When using the DEMO_KEY fallback, rate limiting is enforced at
    5 requests/minute to respect FRED's demo tier limits.
    """
    import time as _time

    global _fred_last_request_time, _fred_demo_key_warned

    is_demo = False
    if not api_key:
        api_key = "DEMO_KEY"
        is_demo = True

    if is_demo:
        if not _fred_demo_key_warned:
            logger.warning(
                "No FRED_API_KEY configured -- using DEMO_KEY with strict "
                "rate limiting (5 req/min). Register a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html for "
                "faster macro data fetching."
            )
            _fred_demo_key_warned = True

        # Enforce rate limit for demo key
        elapsed = _time.time() - _fred_last_request_time
        if elapsed < _FRED_DEMO_RATE_LIMIT_SECONDS:
            sleep_time = _FRED_DEMO_RATE_LIMIT_SECONDS - elapsed
            logger.debug("FRED demo rate limit: sleeping %.1fs", sleep_time)
            _time.sleep(sleep_time)

    start = (date.today() - timedelta(days=_MACRO_LOOKBACK_DAYS)).isoformat()

    # Try fredapi library first
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        series = fred.get_series(series_id, observation_start=start)
        _fred_last_request_time = _time.time()
        if series is not None and not series.empty:
            series.name = series_id
            logger.debug("fredapi: fetched %d observations for %s", len(series), series_id)
            return series.dropna()
    except ImportError:
        logger.debug("fredapi not installed; falling back to direct API")
    except Exception as exc:
        _fred_last_request_time = _time.time()
        logger.debug("fredapi failed for %s: %s; falling back to direct API", series_id, exc)

    # Fallback: direct REST API
    url = f"{api_url}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }

    data = cached_get(url, params=params)
    _fred_last_request_time = _time.time()
    observations = data.get("observations", [])

    if not observations:
        return pd.Series(dtype=float)

    dates = [obs["date"] for obs in observations]
    values = []
    for obs in observations:
        try:
            values.append(float(obs["value"]))
        except (ValueError, TypeError):
            values.append(None)

    series = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
    return series.dropna()


# ---------------------------------------------------------------------------
# ECB SDW (European Central Bank Statistical Data Warehouse)
# ---------------------------------------------------------------------------

def _fetch_ecb(api_url: str, series_key: str) -> pd.Series:
    """Fetch a single ECB SDW series using sdmx1 or direct CSV.

    Primary: sdmx1 (https://github.com/khaeru/sdmx) -- successor to pandasdmx
    Fallback: Direct ECB SDW REST API (CSV)

    ECB SDW REST API: https://sdw-wsrest.ecb.europa.eu/help/
    No API key required.
    """
    start_period = (date.today() - timedelta(days=_MACRO_LOOKBACK_DAYS)).strftime("%Y-%m")

    # Try sdmx1 first (successor to pandasdmx, supports pydantic v2)
    try:
        import sdmx
        ecb = sdmx.Client("ECB")
        # Parse the series key: e.g. "ICP.M.U2.N.000000.4.ANR"
        # The flow_id is the first part before the first dot
        parts = series_key.split(".")
        if len(parts) >= 2:
            flow_id = parts[0]
            key = ".".join(parts[1:])
            resp = ecb.data(flow_id, key=key, params={"startPeriod": start_period})
            df = sdmx.to_pandas(resp)
            if isinstance(df, pd.Series) and not df.empty:
                df.name = series_key
                logger.debug("sdmx1: fetched %d ECB observations for %s", len(df), series_key)
                return df.dropna()
            elif isinstance(df, pd.DataFrame) and not df.empty:
                # Take the first column
                series = df.iloc[:, 0]
                series.name = series_key
                return series.dropna()
    except ImportError:
        logger.debug("sdmx1 not installed; falling back to direct ECB API")
    except Exception as exc:
        logger.debug("sdmx1 ECB failed for %s: %s; falling back to direct API", series_key, exc)

    # Fallback: direct CSV
    url = f"{api_url}/data/{series_key}"
    params = {
        "startPeriod": start_period,
        "format": "csvdata",
    }

    try:
        import io
        import requests as _requests
        # ECB SDW returns CSV text, not JSON -- use requests directly
        # since cached_get only handles JSON responses.
        resp = _requests.get(url, params=params, timeout=30, headers={
            "Accept": "text/csv",
            "User-Agent": "Operator1/1.0",
        })
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            if "TIME_PERIOD" in df.columns and "OBS_VALUE" in df.columns:
                df["date"] = pd.to_datetime(df["TIME_PERIOD"])
                return pd.Series(
                    df["OBS_VALUE"].values,
                    index=df["date"],
                    name=series_key,
                ).sort_index()
    except Exception as exc:
        logger.debug("ECB fetch failed for %s: %s", series_key, exc)

    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Bundesbank API (Germany)
# ---------------------------------------------------------------------------

def _fetch_bundesbank(api_url: str, series_key: str) -> pd.Series:
    """Fetch a single Bundesbank time series.

    Bundesbank REST API: https://api.statistiken.bundesbank.de/doc/
    No API key required.
    """
    url = f"{api_url}/data/BBSIS/{series_key}"
    params = {"detail": "dataonly", "format": "json"}

    try:
        data = cached_get(url, params=params)
        observations = (
            data.get("dataSets", [{}])[0]
            .get("series", {})
        )
        # Extract time periods from structure
        time_periods = (
            data.get("structure", {})
            .get("dimensions", {})
            .get("observation", [{}])[0]
            .get("values", [])
        )

        if observations and time_periods:
            first_series = list(observations.values())[0]
            obs = first_series.get("observations", {})
            dates = []
            values = []
            for idx_str, val_list in obs.items():
                idx = int(idx_str)
                if idx < len(time_periods):
                    dates.append(time_periods[idx].get("id", ""))
                    try:
                        values.append(float(val_list[0]))
                    except (ValueError, TypeError, IndexError):
                        values.append(None)

            series = pd.Series(
                values,
                index=pd.to_datetime(dates),
                name=series_key,
            )
            return series.dropna().sort_index()
    except Exception as exc:
        logger.debug("Bundesbank fetch failed for %s: %s", series_key, exc)

    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# BCB (Banco Central do Brasil)
# ---------------------------------------------------------------------------

def _fetch_bcb(api_url: str, series_id: str) -> pd.Series:
    """Fetch a single BCB SGS series.

    BCB API: https://dadosabertos.bcb.gov.br/
    No API key required.
    """
    start = (date.today() - timedelta(days=_MACRO_LOOKBACK_DAYS)).strftime("%d/%m/%Y")
    end = date.today().strftime("%d/%m/%Y")
    url = f"{api_url}/{series_id}/dados"
    params = {"formato": "json", "dataInicial": start, "dataFinal": end}

    try:
        data = cached_get(url, params=params)
        if isinstance(data, list):
            dates = [d.get("data", "") for d in data]
            values = []
            for d in data:
                try:
                    values.append(float(d.get("valor", "")))
                except (ValueError, TypeError):
                    values.append(None)
            series = pd.Series(
                values,
                index=pd.to_datetime(dates, format="%d/%m/%Y"),
                name=f"bcb_{series_id}",
            )
            return series.dropna().sort_index()
    except Exception as exc:
        logger.debug("BCB fetch failed for series %s: %s", series_id, exc)

    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Generic JSON fetcher (for APIs with simple JSON responses)
# ---------------------------------------------------------------------------

def _fetch_generic_json(url: str, params: dict | None = None) -> pd.Series:
    """Generic fallback fetcher for simple JSON APIs."""
    try:
        data = cached_get(url, params=params)
        if isinstance(data, list) and data:
            # Try to auto-detect date and value columns
            first = data[0]
            date_key = next(
                (k for k in first if "date" in k.lower() or "period" in k.lower()),
                None,
            )
            value_key = next(
                (k for k in first if "value" in k.lower() or "obs" in k.lower()),
                None,
            )
            if date_key and value_key:
                dates = [d[date_key] for d in data]
                values = []
                for d in data:
                    try:
                        values.append(float(d[value_key]))
                    except (ValueError, TypeError):
                        values.append(None)
                return pd.Series(
                    values,
                    index=pd.to_datetime(dates),
                ).dropna().sort_index()
    except Exception as exc:
        logger.debug("Generic JSON fetch failed: %s", exc)

    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# World Bank (wbgapi) -- global fallback for any country
# ---------------------------------------------------------------------------

# Map indicator names to World Bank indicator codes
_WB_INDICATOR_MAP = {
    "gdp": "NY.GDP.MKTP.CD",              # GDP (current US$)
    "inflation": "FP.CPI.TOTL.ZG",        # Inflation, consumer prices (annual %)
    "interest_rate": "FR.INR.RINR",        # Real interest rate (%)
    "unemployment": "SL.UEM.TOTL.ZS",     # Unemployment, total (% of labor force)
    "currency": "PA.NUS.FCRF",             # Official exchange rate (LCU per US$)
}


# ---------------------------------------------------------------------------
# IMF (International Monetary Fund) -- global fallback via imf-reader
# Research: .roo/research/macro-imf-reader-2026-02-24.md
# ---------------------------------------------------------------------------

# IMF indicator codes for the World Economic Outlook (WEO) dataset
_IMF_INDICATOR_MAP: dict[str, str] = {
    "gdp": "NGDP_RPCH",          # Real GDP growth (annual percent change)
    "inflation": "PCPIPCH",        # Inflation, average consumer prices (percent change)
    "interest_rate": "NGDP_RPCH",  # No direct rate; GDP growth as proxy
    "unemployment": "LUR",         # Unemployment rate
    "currency": "NGDP_RPCH",      # No direct FX; GDP growth as proxy
}


def _fetch_imf(country_code: str, indicator_name: str) -> pd.Series:
    """Fetch a macro indicator from IMF using imf-reader.

    imf-reader: https://pypi.org/project/imf-reader/
    No API key required. Covers 190+ countries.
    Research: .roo/research/macro-imf-reader-2026-02-24.md
    """
    imf_code = _IMF_INDICATOR_MAP.get(indicator_name)
    if not imf_code:
        return pd.Series(dtype=float)

    # Only fetch meaningful indicators (skip proxies)
    if indicator_name in ("interest_rate", "currency") and imf_code == "NGDP_RPCH":
        return pd.Series(dtype=float)

    iso3 = _iso2_to_iso3(country_code)
    if not iso3:
        return pd.Series(dtype=float)

    try:
        from imf_reader import weo

        # imf-reader: weo.fetch() returns WEO dataset
        df = weo.fetch()
        if df is not None and not df.empty:
            # Filter for country and indicator
            mask = (
                (df["ISO"].str.upper() == iso3.upper())
                & (df["WEO Subject Code"].str.upper() == imf_code.upper())
            )
            filtered = df[mask]
            if filtered.empty:
                return pd.Series(dtype=float)

            # Extract year columns (numeric columns are years)
            row = filtered.iloc[0]
            dates = []
            values = []
            for col in filtered.columns:
                try:
                    year = int(col)
                    val = float(row[col])
                    dates.append(pd.Timestamp(f"{year}-12-31"))
                    values.append(val)
                except (ValueError, TypeError):
                    continue

            if dates:
                series = pd.Series(values, index=dates, name=f"imf_{imf_code}")
                return series.dropna().sort_index()

    except ImportError:
        logger.debug("imf-reader not installed; IMF fallback unavailable")
    except Exception as exc:
        logger.debug("IMF fetch failed for %s/%s: %s", country_code, indicator_name, exc)

    return pd.Series(dtype=float)


def _fetch_worldbank(country_code: str, indicator_name: str) -> pd.Series:
    """Fetch a macro indicator from World Bank using wbgapi.

    wbgapi: https://github.com/tjbarreto/wbgapi
    No API key required. Covers 200+ countries.
    """
    wb_code = _WB_INDICATOR_MAP.get(indicator_name)
    if not wb_code:
        return pd.Series(dtype=float)

    try:
        import wbgapi as wb
        # wbgapi uses ISO-3 codes; convert ISO-2 -> ISO-3
        iso3 = _iso2_to_iso3(country_code)
        if not iso3:
            return pd.Series(dtype=float)

        start_year = date.today().year - 5
        data = wb.data.DataFrame(wb_code, economy=iso3, time=range(start_year, date.today().year + 1))
        if data is not None and not data.empty:
            # wbgapi returns a DataFrame with year columns
            if isinstance(data, pd.DataFrame):
                # Flatten: years as index, values as series
                series_data = data.iloc[0] if len(data) == 1 else data.mean()
                dates = []
                values = []
                for col in series_data.index:
                    try:
                        year = int(str(col).replace("YR", ""))
                        dates.append(pd.Timestamp(f"{year}-12-31"))
                        values.append(float(series_data[col]))
                    except (ValueError, TypeError):
                        continue
                if dates:
                    series = pd.Series(values, index=dates, name=f"wb_{wb_code}")
                    logger.debug("wbgapi: fetched %d observations for %s/%s", len(series), country_code, indicator_name)
                    return series.dropna().sort_index()
    except ImportError:
        logger.debug("wbgapi not installed")
    except Exception as exc:
        logger.debug("wbgapi failed for %s/%s: %s", country_code, indicator_name, exc)

    return pd.Series(dtype=float)


def _iso2_to_iso3(iso2: str) -> str:
    """Convert ISO-2 country code to ISO-3 for World Bank API."""
    _MAP = {
        "US": "USA", "GB": "GBR", "EU": "EMU", "FR": "FRA", "DE": "DEU",
        "JP": "JPN", "KR": "KOR", "TW": "TWN", "BR": "BRA", "CL": "CHL",
        "CA": "CAN", "AU": "AUS", "IN": "IND", "CN": "CHN", "HK": "HKG",
        "SG": "SGP", "MX": "MEX", "ZA": "ZAF", "CH": "CHE", "NL": "NLD",
        "ES": "ESP", "IT": "ITA", "SE": "SWE", "SA": "SAU", "AE": "ARE",
    }
    return _MAP.get(iso2.upper(), "")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Map macro_id prefix to fetcher function
_FETCHERS = {
    "us_fred": "fred",
    "eu_ecb": "ecb",
    "uk_ons": "generic",
    "de_bundesbank": "bundesbank",
    "fr_insee": "generic",
    "jp_estat": "generic",
    "kr_kosis": "generic",
    "tw_dgbas": "generic",
    "br_bcb": "bcb",
    "cl_bcch": "generic",
}


def fetch_macro_indicators(
    macro_info: Any,
    api_key: str = "",
) -> dict[str, pd.Series | None]:
    """Fetch all key macro indicators for a given macro API source.

    Parameters
    ----------
    macro_info:
        A ``MacroAPIInfo`` instance from the registry.
    api_key:
        Optional API key for sources that require registration.

    Returns
    -------
    Dict of ``{indicator_name: pd.Series}`` with keys:
    ``gdp``, ``inflation``, ``interest_rate``, ``unemployment``, ``currency``.
    Values are None if the fetch failed or series ID is empty.
    """
    macro_id = macro_info.macro_id
    api_url = macro_info.api_url
    fetcher_type = _FETCHERS.get(macro_id, "generic")

    indicators = {
        "gdp": macro_info.series_gdp,
        "inflation": macro_info.series_inflation,
        "interest_rate": macro_info.series_interest_rate,
        "unemployment": macro_info.series_unemployment,
        "currency": macro_info.series_currency,
    }

    results: dict[str, pd.Series | None] = {}

    for name, series_id in indicators.items():
        if not series_id:
            results[name] = None
            continue

        try:
            if fetcher_type == "fred":
                results[name] = _fetch_fred(api_url, series_id, api_key=api_key)
            elif fetcher_type == "ecb":
                results[name] = _fetch_ecb(api_url, series_id)
            elif fetcher_type == "bundesbank":
                results[name] = _fetch_bundesbank(api_url, series_id)
            elif fetcher_type == "bcb":
                results[name] = _fetch_bcb(api_url, series_id)
            else:
                # Generic: try JSON fetch
                results[name] = _fetch_generic_json(
                    f"{api_url}/{series_id}",
                )

            if results[name] is not None and not results[name].empty:
                logger.info(
                    "  %s: %d observations (%s to %s)",
                    name,
                    len(results[name]),
                    results[name].index[0].date(),
                    results[name].index[-1].date(),
                )
            else:
                results[name] = None
                logger.debug("  %s: no data returned", name)

        except Exception as exc:
            logger.warning("  %s: fetch failed: %s", name, exc)
            results[name] = None

    # Fallback: try World Bank (wbgapi) for any missing indicators
    country_code = macro_info.country_code
    for name, series in results.items():
        if series is None and country_code:
            try:
                wb_series = _fetch_worldbank(country_code, name)
                if wb_series is not None and not wb_series.empty:
                    results[name] = wb_series
                    logger.info(
                        "  %s: World Bank fallback: %d observations",
                        name, len(wb_series),
                    )
            except Exception as exc:
                logger.debug("  %s: World Bank fallback failed: %s", name, exc)

    # Second fallback: try IMF (imf-reader) for any still-missing indicators
    # Research: .roo/research/macro-imf-reader-2026-02-24.md
    # IMF covers 190+ countries with GDP, CPI, exchange rates, fiscal data
    for name, series in results.items():
        if series is None and country_code:
            try:
                imf_series = _fetch_imf(country_code, name)
                if imf_series is not None and not imf_series.empty:
                    results[name] = imf_series
                    logger.info(
                        "  %s: IMF fallback: %d observations",
                        name, len(imf_series),
                    )
            except Exception as exc:
                logger.debug("  %s: IMF fallback failed: %s", name, exc)

    return results
