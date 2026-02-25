# Research Log: Per-Region Macro Wrappers
**Generated**: 2026-02-25T12:30:00Z
**Status**: Complete
**Strategy**: Per-region primary wrappers + wbgapi as global fallback

---

## Architecture

```
Macro Request for {country_code}
    |
    v
[Per-region primary wrapper] -- try first (government central bank API)
    |
    | (if fails, not installed, or no key)
    v
[wbgapi fallback] -- World Bank, global coverage, no key needed
```

---

## Per-Region Macro Wrappers

### 1. US -- fredapi (FRED)
| Field | Value |
|-------|-------|
| Package | fredapi 0.5.2 |
| PyPI | https://pypi.org/project/fredapi/ |
| GitHub | mortada/fredapi |
| Key Required | **YES** (FRED_API_KEY, free registration) |
| Verified | 2026-02-25 (PyPI fetch) |

**Verbatim usage**:
```python
from fredapi import Fred
fred = Fred(api_key='your_key')
# GDP growth
gdp = fred.get_series('GDP')
# CPI
cpi = fred.get_series('CPIAUCSL')
# Unemployment
unemp = fred.get_series('UNRATE')
# Federal funds rate
ffr = fred.get_series('FEDFUNDS')
```

**Key indicators**: GDP, CPIAUCSL, UNRATE, FEDFUNDS, DGS10
**Fallback**: wbgapi economy="USA"

---

### 2. EU/DE/FR -- sdmx1 (ECB / Bundesbank / INSEE)
| Field | Value |
|-------|-------|
| Package | sdmx1 2.25.1 |
| PyPI | https://pypi.org/project/sdmx1/ |
| GitHub | khaeru/sdmx |
| Key Required | **NO** (ECB, Bundesbank), **YES** (INSEE needs key) |
| Python | >=3.10 |
| Verified | 2026-02-25 (PyPI fetch) |

**Verbatim usage**:
```python
import sdmx

# ECB: Euro area GDP
ecb = sdmx.Client('ECB')
data = ecb.data('MNA', key={'FREQ': 'A', 'REF_AREA': 'I8'})

# Bundesbank: German rates
bbk = sdmx.Client('BBK')

# INSEE: French CPI (needs key)
insee = sdmx.Client('INSEE')
```

**Fallback**: wbgapi economy="EMU"/"DEU"/"FRA"

---

### 3. UK -- ONS API (direct HTTP)
| Field | Value |
|-------|-------|
| API | https://api.ons.gov.uk/dataset |
| Key Required | **NO** |
| Verified | 2026-02-25 (pit_registry.py source) |

**Usage**: Direct HTTP requests to ONS REST API.
```python
import requests
resp = requests.get("https://api.ons.gov.uk/dataset/gdp/timeseries/IHYQ/data")
```

**Fallback**: wbgapi economy="GBR"

---

### 4. Brazil -- python-bcb (BCB SGS)
| Field | Value |
|-------|-------|
| Package | python-bcb 0.3.3 |
| PyPI | https://pypi.org/project/python-bcb/ |
| GitHub | wilsonfreitas/python-bcb |
| Key Required | **NO** |
| Python | >=3.10 |
| Verified | 2026-02-25 (PyPI + README fetch) |

**Verbatim usage** (from README):
```python
from bcb import sgs
# SELIC rate (series 432)
selic = sgs.get({'selic': 432}, start='2020-01-01')
# IPCA inflation (series 433)
ipca = sgs.get({'ipca': 433}, start='2020-01-01')
# GDP monthly proxy (series 4380)
gdp = sgs.get({'gdp': 4380}, start='2020-01-01')
```

**Key indicators**: 432 (SELIC), 433 (IPCA), 4380 (GDP proxy), 3698 (unemployment)
**Fallback**: wbgapi economy="BRA"

---

### 5. Mexico -- banxicoapi (Banxico)
| Field | Value |
|-------|-------|
| Package | banxicoapi 1.0.2 |
| PyPI | https://pypi.org/project/banxicoapi/ |
| GitHub | EliasManj/banxico-api |
| Key Required | **YES** (BANXICO_TOKEN, free registration) |
| Python | >=3.6 |
| Verified | 2026-02-25 (PyPI + README fetch) |

**Verbatim usage** (from README):
```python
import banxicoapi
api = banxicoapi.BanxicoApi('YOUR_API_TOKEN')
# Interest rate
rate = api.get_series_data('SF61745', '2020-01-01', '2025-12-31')
```

**Fallback**: wbgapi economy="MEX"

---

### 6. Japan -- e-Stat API (direct HTTP)
| Field | Value |
|-------|-------|
| API | https://api.e-stat.go.jp/rest/3.0/app |
| Key Required | **YES** (ESTAT_API_KEY, free registration) |
| Verified | pit_registry.py source |

**Fallback**: wbgapi economy="JPN"

---

### 7. Korea -- KOSIS API (direct HTTP)
| Field | Value |
|-------|-------|
| API | https://kosis.kr/openapi |
| Key Required | **YES** (KOSIS_API_KEY, free registration) |
| Verified | pit_registry.py source |

**Fallback**: wbgapi economy="KOR"

---

### 8. Chile -- BCC API (direct HTTP)
| Field | Value |
|-------|-------|
| API | https://si3.bcentral.cl/SieteRestWS |
| Key Required | **YES** (BCC_API_KEY, free registration) |
| Verified | pit_registry.py source |

**Fallback**: wbgapi economy="CHL"

---

### 9-20. All Other Markets -- wbgapi (primary AND only)

| Country | WB Code | Macro Available via wbgapi |
|---------|---------|---------------------------|
| Canada | CAN | GDP, CPI, unemployment |
| Australia | AUS | GDP, CPI, unemployment |
| India | IND | GDP, CPI, unemployment |
| China | CHN | GDP, CPI, unemployment |
| Hong Kong | HKG | GDP, CPI, unemployment |
| Singapore | SGP | GDP, CPI, unemployment |
| South Africa | ZAF | GDP, CPI, unemployment |
| Switzerland | CHE | GDP, CPI, unemployment |
| Saudi Arabia | SAU | GDP, CPI, unemployment |
| UAE | ARE | GDP, CPI, unemployment |
| Taiwan | (skip) | Not in World Bank, use CHN |

---

## Global Fallback: wbgapi

| Field | Value |
|-------|-------|
| Package | wbgapi 1.0.13 |
| Key Required | **NO** |
| Research | .roo/research/macro-wbgapi-2026-02-24.md |

```python
import wbgapi as wb
df = wb.data.DataFrame(
    ["NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS"],
    economy="USA",
    mrv=10,
    numericTimeKeys=True,
)
```

**Key World Bank indicators**:
- NY.GDP.MKTP.KD.ZG -- GDP growth (annual %)
- FP.CPI.TOTL.ZG -- CPI inflation (annual %)
- SL.UEM.TOTL.ZS -- Unemployment (% labor force)
- FR.INR.RINR -- Real interest rate (%)
- PA.NUS.FCRF -- Exchange rate (LCU per USD)

**END OF RESEARCH LOG**
