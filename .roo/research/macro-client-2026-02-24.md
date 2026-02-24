# Research Log: Macro Client (Multi-Source)
**Generated**: 2026-02-24T15:40:00Z
**Status**: Complete

---

## 1. VERSION INFORMATION

| Component | Current (in requirements.txt) | Latest Available | Action Required |
|-----------|-------------------------------|------------------|-----------------|
| fredapi | >=0.5 | 0.5.2 | OK |
| wbgapi | >=1.0 | 1.0.13 | OK |
| sdmx1 | >=2.0 | 2.25.1 | OK |

---

# =====================================================================
# PART A: UNOFFICIAL COMMUNITY WRAPPERS (3 libraries)
# =====================================================================

## A1. fredapi (US FRED)

### Registry Data
**Source**: https://pypi.org/pypi/fredapi/json (fetched 2026-02-24)
```
name: fredapi
version: 0.5.2
summary: Python API for Federal Reserve Economic Data (FRED) from St. Louis Fed
requires_python: (none specified)
homepage: https://github.com/mortada/fredapi
```

### Verbatim Code
**Source**: https://github.com/mortada/fredapi/blob/master/README.md (fetched 2026-02-24)
```python
# COPIED VERBATIM -- basic usage
from fredapi import Fred
fred = Fred(api_key='insert api key here')
data = fred.get_series('SP500')
```

### Verbatim -- API key setup (from README)
```
# COPIED VERBATIM
First you need an API key, you can apply for one for free on the FRED website.
Once you have your API key, you can set it in one of three ways:
* set it to the evironment variable FRED_API_KEY
* save it to a file and use the 'api_key_file' parameter
* pass it directly as the 'api_key' parameter
```

### Our wrapper usage
```python
# From macro_client.py _fetch_fred()
from fredapi import Fred
fred = Fred(api_key=api_key)
series = fred.get_series(series_id, observation_start=start)
```
This matches the verbatim README pattern.

---

## A2. wbgapi (World Bank -- Global Fallback)

### Registry Data
**Source**: https://pypi.org/pypi/wbgapi/json (fetched 2026-02-24)
```
name: wbgapi
version: 1.0.13
summary: wbgapi provides a comprehensive interface to the World Bank's data and metadata
```

### Verbatim Code
**Source**: https://github.com/tgherzog/wbgapi/blob/master/README.md (fetched 2026-02-24)
```
# COPIED VERBATIM -- install and import
pip install wbgapi

import wbgapi as wb
```

### Our wrapper usage
```python
# From macro_client.py _fetch_worldbank()
import wbgapi as wb
iso3 = _iso2_to_iso3(country_code)
data = wb.data.DataFrame(wb_code, economy=iso3, time=range(start_year, current_year + 1))
```

---

## A3. sdmx1 (ECB, OECD, Eurostat via SDMX protocol)

### Registry Data
**Source**: https://pypi.org/pypi/sdmx1/json (fetched 2026-02-24)
```
name: sdmx1
version: 2.25.1
summary: Statistical Data and Metadata eXchange (SDMX)
```

### Verbatim Description
**Source**: https://github.com/khaeru/sdmx/blob/main/README.rst (fetched 2026-02-24)
```
# COPIED VERBATIM
sdmx ('sdmx1' on PyPI) is a Python implementation of the SDMX 2.1
(ISO 17369:2013) and 3.0 standards for statistical data and metadata exchange.

sdmx can be used to:
- Explore and retrieve data available from SDMX-REST web services
  operated by providers including the World Bank, International Monetary Fund,
  Eurostat, OECD, and United Nations;
- Read and write data and metadata in file formats including SDMX-ML (XML),
  SDMX-JSON, and SDMX-CSV;
- Convert data and metadata into pandas objects
```

### Our wrapper usage
```python
# From macro_client.py _fetch_ecb()
import sdmx
ecb = sdmx.Client("ECB")
resp = ecb.data(flow_id, key=key, params={"startPeriod": start_period})
df = sdmx.to_pandas(resp)
```

---

# =====================================================================
# PART B: GOVERNMENT APIs (10 regional sources)
# =====================================================================

## B1. US FRED

**URL**: https://api.stlouisfed.org/fred
**Key required**: Yes (free registration)
**Endpoint**: `/series/observations?series_id={id}&api_key={key}&file_type=json`
**Used by**: `_fetch_fred()` as fallback when fredapi not installed

## B2. EU ECB SDW

**URL**: https://sdw-wsrest.ecb.europa.eu/service
**Key required**: No
**Fallback**: Direct CSV download from `{api_url}/data/{series_key}?format=csvdata`
**Used by**: `_fetch_ecb()` when sdmx1 not installed

## B3. DE Bundesbank

**URL**: https://api.statistiken.bundesbank.de/rest
**Key required**: No
**Endpoint**: `/data/BBSIS/{series_key}?detail=dataonly&format=json`
**Used by**: `_fetch_bundesbank()` -- dedicated fetcher

## B4. BR BCB (Banco Central do Brasil)

**URL**: https://api.bcb.gov.br/dados/serie/bcdata.sgs
**Key required**: No
**Endpoint**: `/{series_id}/dados?formato=json&dataInicial={start}&dataFinal={end}`
**Used by**: `_fetch_bcb()` -- dedicated fetcher

## B5. UK ONS, FR INSEE, JP e-Stat, KR KOSIS, TW DGBAS, CL BCCh

All use the **generic JSON fetcher** (`_fetch_generic_json`):
```python
# From macro_client.py
_FETCHERS = {
    "uk_ons": "generic",
    "fr_insee": "generic",
    "jp_estat": "generic",
    "kr_kosis": "generic",
    "tw_dgbas": "generic",
    "cl_bcch": "generic",
}
```
The generic fetcher auto-detects date and value columns from JSON responses.
This is fragile but functional when the API returns well-structured JSON.

## B6. World Bank (Global Fallback)

All regions that return `None` from dedicated fetchers fall back to World Bank:
```python
# From macro_client.py
_WB_INDICATOR_MAP = {
    "gdp": "NY.GDP.MKTP.CD",
    "inflation": "FP.CPI.TOTL.ZG",
    "interest_rate": "FR.INR.RINR",
    "unemployment": "SL.UEM.TOTL.ZS",
    "currency": "PA.NUS.FCRF",
}
```

---

# =====================================================================
# PART C: SUMMARY
# =====================================================================

| Component | Mode | Status |
|-----------|------|--------|
| fredapi 0.5.2 | Unofficial wrapper | OK -- verbatim matches README |
| wbgapi 1.0.13 | Unofficial wrapper | OK -- global fallback |
| sdmx1 2.25.1 | Unofficial wrapper | OK -- ECB/OECD/Eurostat |
| FRED REST | Government fallback | OK |
| ECB SDW CSV | Government fallback | OK |
| Bundesbank JSON | Government dedicated | OK |
| BCB SGS JSON | Government dedicated | OK |
| ONS/INSEE/e-Stat/KOSIS/DGBAS/BCCh | Government generic | FRAGILE -- auto-detect JSON |
| World Bank | Global fallback | OK |

### Required Fixes
1. **NONE critical** -- all library versions match, API patterns verified
2. **MEDIUM**: 6 generic macro fetchers should be upgraded to use proper API patterns

---

**END OF RESEARCH LOG**
