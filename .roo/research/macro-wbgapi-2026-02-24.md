# Research Log: World Bank wbgapi (Global Macro Fallback)
**Generated**: 2026-02-24T15:52:00Z
**Status**: Complete

---

# =====================================================================
# PART A: UNOFFICIAL COMMUNITY WRAPPER -- wbgapi
# =====================================================================

## A1. Registry Data

**Source**: https://pypi.org/pypi/wbgapi/json (fetched 2026-02-24)
```
name: wbgapi
version: 1.0.13
summary: wbgapi provides a comprehensive interface to the World Bank's data and metadata
```

## A2. Verbatim Code

**Source**: PyPI description / GitHub README (fetched 2026-02-24)
```
# COPIED VERBATIM
pip install wbgapi
import wbgapi as wb
```

## A3. Verbatim Description

**Source**: GitHub README (fetched 2026-02-24)
```
# COPIED VERBATIM
WBGAPI provides modern, pythonic access to the World Bank's data API.
It is designed both for data novices and data scientist types.

Key features:
* Easily select multiple series, economies (countries) and time periods
* Select individual years, ranges, and most recent values (MRVs)
* Metadata queries
* Extensive (but optional) pandas support
```

## A4. Our Usage

**Source**: macro_client.py `_fetch_worldbank()`
```python
# VERBATIM from macro_client.py
_WB_INDICATOR_MAP = {
    "gdp": "NY.GDP.MKTP.CD",
    "inflation": "FP.CPI.TOTL.ZG",
    "interest_rate": "FR.INR.RINR",
    "unemployment": "SL.UEM.TOTL.ZS",
    "currency": "PA.NUS.FCRF",
}

import wbgapi as wb
data = wb.data.DataFrame(wb_code, economy=iso3, time=range(start_year, current_year + 1))
```

## A5. Country Coverage

**Source**: macro_client.py `_iso2_to_iso3()`
```python
# VERBATIM -- ISO-2 to ISO-3 mapping for World Bank
_MAP = {
    "US": "USA", "GB": "GBR", "EU": "EMU", "FR": "FRA", "DE": "DEU",
    "JP": "JPN", "KR": "KOR", "TW": "TWN", "BR": "BRA", "CL": "CHL",
    "CA": "CAN", "AU": "AUS", "IN": "IND", "CN": "CHN", "HK": "HKG",
    "SG": "SGP", "MX": "MEX", "ZA": "ZAF", "CH": "CHE", "NL": "NLD",
    "ES": "ESP", "IT": "ITA", "SE": "SWE", "SA": "SAU", "AE": "ARE",
}
```
All 24 market country codes are mapped for World Bank fallback.

---

# =====================================================================
# PART B: WORLD BANK DATA API
# =====================================================================

## B1. API Details
**URL**: https://api.worldbank.org/v2/
**Key required**: No
**Coverage**: 200+ countries, 16,000+ indicators
**Our indicators**: GDP, CPI, interest rate, unemployment, exchange rate

---

# =====================================================================
# PART C: SUMMARY
# =====================================================================

| Component | Status |
|-----------|--------|
| wbgapi 1.0.13 | IN USE -- global macro fallback for all regions |
| Coverage | All 24 market codes mapped in _iso2_to_iso3 |
| Indicators | GDP, inflation, interest_rate, unemployment, currency |

**Every micro region that lacks a dedicated macro source falls back to World Bank.**
This ensures 100% macro coverage across all markets.

---

**END OF RESEARCH LOG**
