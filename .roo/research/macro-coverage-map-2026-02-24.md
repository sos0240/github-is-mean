# Research Log: Macro Coverage Map -- Every Micro Region Needs a Macro Source
**Generated**: 2026-02-24T15:51:00Z
**Status**: Complete

---

## Current Macro Coverage (from pit_registry.py MACRO_APIS)

### Tier 1 Markets -- All Have Dedicated Macro Sources

| Micro Region | Micro PIT Source | Macro Source | Macro ID | Fetcher Type | Unofficial Wrapper |
|-------------|-----------------|-------------|----------|-------------|-------------------|
| US | SEC EDGAR | FRED | us_fred | **dedicated** (fredapi) | fredapi 0.5.2 |
| EU | ESEF | ECB SDW | eu_ecb | **dedicated** (sdmx1) | sdmx1 2.25.1 |
| UK | Companies House | ONS | uk_ons | generic | NONE |
| DE | ESEF (DE) | Bundesbank | de_bundesbank | **dedicated** | NONE |
| FR | ESEF (FR) | INSEE | fr_insee | generic | api-insee 1.6 (Sirene only) |
| JP | EDINET | e-Stat | jp_estat | generic | NONE |
| KR | DART | KOSIS | kr_kosis | generic | NONE |
| TW | MOPS | DGBAS | tw_dgbas | generic | NONE |
| BR | CVM | BCB | br_bcb | **dedicated** | NONE |
| CL | CMF | BCCh | cl_bcch | generic | NONE |

### Tier 2 Markets -- World Bank Fallback Only

| Micro Region | Micro Stub | Macro Source | Found Wrapper? |
|-------------|-----------|-------------|----------------|
| CA | ca_sedar.py | **World Bank only** | NONE |
| AU | au_asx.py | **World Bank only** | NONE |
| IN | in_bse.py | **World Bank only** | jugaad-data 0.29 includes RBI |
| CN | cn_sse.py | **World Bank only** | akshare 1.18.27 includes NBS macro |
| HK | hk_hkex.py | **World Bank only** | NONE |
| SG | sg_sgx.py | **World Bank only** | **singstat 2.0.2** |
| MX | mx_bmv.py | **World Bank only** | **banxicoapi 1.0.2** |
| ZA | za_jse.py | **World Bank only** | NONE |
| CH | ch_six.py | **World Bank only** | NONE |
| SA | sa_tadawul.py | **World Bank only** | NONE |
| AE | ae_dfm.py | **World Bank only** | NONE |

---

## Global Fallback Macro Sources

### Already in Use

| Package | Version | Purpose | Research |
|---------|---------|---------|----------|
| wbgapi | 1.0.13 | World Bank global fallback (200+ countries) | .roo/research/macro-client-2026-02-24.md |
| fredapi | 0.5.2 | US FRED (primary for US) | .roo/research/macro-client-2026-02-24.md |
| sdmx1 | 2.25.1 | SDMX: ECB, OECD, Eurostat, IMF | .roo/research/macro-client-2026-02-24.md |

### Found on PyPI (Not Yet Used)

| Package | Version | PyPI Summary (verbatim) | Use Case |
|---------|---------|------------------------|----------|
| imf-reader | 1.4.1 | "A package to access imf data" | Global macro fallback (IMF data covers all countries) |
| pandas-datareader | 0.10.0 | "Data readers extracted from the pandas codebase" | Can access FRED, Eurostat, World Bank, OECD |
| world-bank-data | 0.1.4 | "World Bank Data API in Python" | Alternative to wbgapi |
| singstat | 2.0.2 | "Python package for interacting with APIs available at SingStat.gov.sg" | Singapore macro |
| banxicoapi | 1.0.2 | "Client for Banxico API" | Mexico macro |
| datacommons | 1.4.4 | "A library to access Data Commons Python API." | DEPRECATED -- use datacommons-client |

---

## Gap Analysis: Regions Without Dedicated Macro Wrappers

### Tier 1 Gaps (using generic fetcher)

| Region | Generic Fetcher | What Would Fix It |
|--------|----------------|-------------------|
| UK (ONS) | auto-detect JSON | ONS has proper REST API -- build dedicated fetcher |
| FR (INSEE) | auto-detect JSON | sdmx1 supports INSEE -- could use SDMX query |
| JP (e-Stat) | auto-detect JSON | e-Stat has JSON API with app ID -- build dedicated fetcher |
| KR (KOSIS) | auto-detect JSON | KOSIS has JSON API with app key -- build dedicated fetcher |
| TW (DGBAS) | auto-detect JSON | DGBAS may not return clean JSON -- fragile |
| CL (BCCh) | auto-detect JSON | BCCh has specific API format -- build dedicated fetcher |

### Tier 2 Gaps (World Bank only)

| Region | What Could Help |
|--------|----------------|
| CA | pandas-datareader can access Bank of Canada via FRED |
| AU | RBA tables are free but no wrapper found |
| IN | jugaad-data includes RBI rates |
| CN | akshare includes NBS macro data |
| HK | HKMA free data but no wrapper |
| SG | **singstat 2.0.2** found |
| MX | **banxicoapi 1.0.2** found |
| ZA | SARB free data but no wrapper |
| CH | SNB free data but no wrapper |
| SA | SAMA free data but no wrapper |
| AE | CBUAE free data but no wrapper |

---

## Recommended Macro Wrapper Additions

### High Priority (good packages found)
1. **imf-reader 1.4.1** -- global macro fallback via IMF (covers ALL countries)
2. **singstat 2.0.2** -- dedicated Singapore macro
3. **banxicoapi 1.0.2** -- dedicated Mexico macro

### Medium Priority (already used libraries could do more)
4. **sdmx1** already supports INSEE, OECD, IMF -- wire it for FR/JP/KR
5. **pandas-datareader** supports FRED, Eurostat, World Bank, OECD

### Low Priority (already covered by World Bank fallback)
6. Dedicated fetchers for UK ONS, JP e-Stat, KR KOSIS, TW DGBAS, CL BCCh
   would be more reliable than generic fetcher but World Bank covers them

---

**END OF RESEARCH LOG**
