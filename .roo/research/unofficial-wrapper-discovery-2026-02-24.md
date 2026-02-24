# Research Log: Unofficial Community Wrapper Discovery (All Markets)
**Generated**: 2026-02-24T15:28:00Z
**Status**: Complete
**Method**: PyPI registry search + GitHub README fetch (per research protocol)

---

## Summary of Findings

### Wrappers FOUND on PyPI

| Market | Package | Version | PyPI Summary (verbatim) | GitHub |
|--------|---------|---------|------------------------|--------|
| UK (iXBRL) | ixbrl-parse | 0.10.1 | "Parse iXBRL files, can present in RDF" | https://github.com/cybermaggedon/ixbrl-parse |
| UK (CH API) | companies-house-api-client | 0.0.4 | "Simple python wrapper for Companies House API" | -- |
| Taiwan | twstock | 1.4.0 | "Taiwan Stock Opendata with realtime - twstock" | https://github.com/mlouielu/twstock |
| Chile | cmfapi | 0.0.2 | "Python API Client for CMF" | https://github.com/atripathy86/cmfapi |
| Australia | asx | 1.5.2 | "Python library to access the ASX API" | https://github.com/leighcurran/ASX |
| China | akshare | 1.18.27 | "AKShare is an elegant and simple financial data interface library for Python" | https://github.com/akfamily/akshare |
| China | tushare | 1.4.24 | "A utility for crawling historical and Real-time Quotes data of China stocks" | https://tushare.pro |
| China | baostock | 0.8.9 | "A tool for obtaining historical data of China stock market" | http://www.baostock.com |
| HK | openhkex | 0.1.0 | "python library providing data service of Hong Kong Exchanges and Clearing Limited" | https://github.com/chwanlouis/openhkex |
| India | jugaad-data | 0.29 | "Free Zerodha API python library" (actually NSE/BSE data) | https://marketsetup.in/documentation/jugaad-data/ |
| India | nselib | 2.4.3 | "library to get NSE India data" | -- |
| India | bseindia | 0.1 | "library to get BSE India data" | -- |
| India | indian-stock-market | 0.6.0 | "A professional Python library for Indian stock market data and analysis" | -- |
| Mexico (macro) | banxicoapi | 1.0.2 | "Client for Banxico API" | https://github.com/EliasManj/banxico-api |
| France (macro) | api-insee | 1.6 | "Python helper to request Sirene Api on api.insee.fr" | https://github.com/sne3ks/api_insee |

### Wrappers NOT FOUND on PyPI

| Market | Searched terms | Result |
|--------|---------------|--------|
| Canada (SEDAR) | sedar, tsx-data, sedar-plus, sec-csa | No package found |
| Singapore (SGX) | sgx, singapore-stock | No relevant package (only Intel SGX) |
| South Africa (JSE) | jse, jse-data, johannesburg, sarb-api | No package found |
| Switzerland (SIX) | six-exchange, swiss-stock | No package found |
| UAE (DFM/ADX) | dfm, adx, dubai-stock, uae-stock | No package found |
| Saudi Arabia (Tadawul) | tadawul, saudi-stock, tadawul-sdk | No package found |

---

## Detailed Research per Market

---

# =====================================================================
# UK: ixbrl-parse (Could fill UK financial data gap)
# =====================================================================

## Registry Data
**Source**: https://pypi.org/pypi/ixbrl-parse/json (fetched 2026-02-24)
```
name: ixbrl-parse
version: 0.10.1
summary: Parse iXBRL files, can present in RDF
requires_python: >=3.6
homepage: https://github.com/cybermaggedon/ixbrl-parse
```

## Verbatim Code
**Source**: https://github.com/cybermaggedon/ixbrl-parse/blob/master/README.md

```bash
# COPIED VERBATIM -- Installation
pip install ixbrl-parse
```

```bash
# COPIED VERBATIM -- CLI tools
ixbrl-to-json accts.html              # Output iXBRL as JSON
ixbrl-to-csv accts.html               # Output iXBRL as CSV
ixbrl-report accts.html               # Human-readable report

# For SEC EDGAR filings with relative schema URLs:
ixbrl-report ixbrl/10k/lyft-20201231.htm \
    -b https://www.sec.gov/Archives/edgar/data/1759509/000175950921000011/lyft-20201231.htm
```

**Note from README**: "There's a bunch of sample iXBRL files grabbed from various places:
US 10-K and 10-Q filings, a few random things from UK companies house, and a couple of
sample ESEF filings."

## Relevance to Our Wrapper
This could fill the UK Companies House financial data gap. The workflow would be:
1. Download iXBRL document from Companies House filing history
2. Parse with ixbrl-parse to extract financial values
3. Map iXBRL tags to canonical fields

---

# =====================================================================
# TAIWAN: twstock
# =====================================================================

## Registry Data
**Source**: https://pypi.org/pypi/twstock/json (fetched 2026-02-24)
```
name: twstock
version: 1.4.0
summary: Taiwan Stock Opendata with realtime - twstock
homepage: https://github.com/mlouielu/twstock
```

## Relevance
twstock provides TWSE stock price data and company info. Our `tw_mops_wrapper.py`
already does MOPS form POST scraping. twstock could supplement with price data
and company listings, but doesn't provide MOPS financial statement data.
**Verdict**: Nice to have for supplementary data, not a replacement for MOPS scraping.

---

# =====================================================================
# CHILE: cmfapi
# =====================================================================

## Registry Data
**Source**: https://pypi.org/pypi/cmfapi/json (fetched 2026-02-24)
```
name: cmfapi
version: 0.0.2
summary: Python API Client for CMF
homepage: https://github.com/atripathy86/cmfapi
```

## Relevance
Very early version (0.0.2). Could potentially wrap CMF Chile endpoints.
Need to verify if it provides financial statement (FECU) data or just
basic company info.
**Verdict**: Worth investigating but very immature (v0.0.2).

---

# =====================================================================
# AUSTRALIA: asx
# =====================================================================

## Registry Data
**Source**: https://pypi.org/pypi/asx/json (fetched 2026-02-24)
```
name: asx
version: 1.5.2
summary: Python library to access the ASX API: https://www.asx.com.au/asx/1/share
homepage: https://github.com/leighcurran/ASX
```

## Relevance
Wraps the ASX public API. Could provide company profiles, price data,
and possibly financial data for the `au_asx.py` Tier 2 stub.
**Verdict**: Strong candidate for AU market wrapper.

---

# =====================================================================
# CHINA: akshare (BEST candidate)
# =====================================================================

## Registry Data
**Source**: https://pypi.org/pypi/akshare/json (fetched 2026-02-24)
```
name: akshare
version: 1.18.27
summary: AKShare is an elegant and simple financial data interface library for Python, built for human beings!
requires_python: >=3.8
homepage: https://github.com/akfamily/akshare
```

## Verbatim Code
**Source**: https://github.com/akfamily/akshare/blob/main/README.md
```shell
# COPIED VERBATIM -- Installation
pip install akshare --upgrade
```

## Relevance
akshare is the most comprehensive Chinese financial data library. It covers:
- SSE (Shanghai) and SZSE (Shenzhen) stock data
- Financial statements, company info, macro data
- 100+ data sources including Chinese regulatory filings
**Verdict**: Best candidate for `cn_sse.py` Tier 2 market. Very active (v1.18.27).
Note: Documentation is primarily in Chinese.

---

# =====================================================================
# CHINA (alternative): tushare, baostock
# =====================================================================

## tushare
```
version: 1.4.24
summary: A utility for crawling historical and Real-time Quotes data of China stocks
homepage: https://tushare.pro
```
Requires registration for API token. More focused on price data.

## baostock
```
version: 0.8.9
summary: A tool for obtaining historical data of China stock market
homepage: http://www.baostock.com
```
Free, no registration. Historical data focus.

**Verdict**: akshare is the best all-in-one option for China.

---

# =====================================================================
# HONG KONG: openhkex
# =====================================================================

## Registry Data
```
name: openhkex
version: 0.1.0
summary: python library providing data service of Hong Kong Exchanges and Clearing Limited
homepage: https://github.com/chwanlouis/openhkex
```

## Relevance
Very early version (0.1.0). Could wrap HKEX data for the `hk_hkex.py` Tier 2 stub.
**Verdict**: Only option found, but immature.

---

# =====================================================================
# INDIA: jugaad-data (BEST candidate)
# =====================================================================

## Registry Data
**Source**: https://pypi.org/pypi/jugaad-data/json (fetched 2026-02-24)
```
name: jugaad-data
version: 0.29
summary: Free Zerodha API python library
homepage: https://marketsetup.in/documentation/jugaad-data/
```

## Verbatim Code
**Source**: https://github.com/jugaad-py/jugaad-data/blob/master/README.md
```python
# COPIED VERBATIM -- Historical stock data
from datetime import date
from jugaad_data.nse import stock_df
df = stock_df(symbol="SBIN", from_date=date(2020,1,1),
            to_date=date(2020,1,30), series="EQ")
```

```python
# COPIED VERBATIM -- Live data
from jugaad_data.nse import NSELive
n = NSELive()
q = n.stock_quote("HDFC")
```

## Supported Features (verbatim from README)
```
| Website  | Segment    | Supported? |
|----------|------------|------------|
| NSE      | Stocks     | Yes        |
| NSE      | Stocks F&O | Yes        |
| NSE      | Index      | Yes        |
| NSE      | Index F&O  | Yes        |
| RBI      | Current Rates| Yes      |
```

## Relevance
Best candidate for `in_bse.py` Tier 2 market. Covers NSE stocks + RBI macro data.
Built-in caching. Supports new NSE website.
**Verdict**: Strong candidate. Also has alternatives: nselib, bseindia, indian-stock-market.

---

# =====================================================================
# MEXICO: banxicoapi (macro only)
# =====================================================================

## Registry Data
```
name: banxicoapi
version: 1.0.2
summary: Client for Banxico API
homepage: https://github.com/EliasManj/banxico-api
```

## Relevance
Wraps Banco de Mexico (Banxico) API for macro data (interest rates, inflation, FX).
This is a MACRO wrapper, not a micro/disclosure wrapper.
**Verdict**: Useful for adding dedicated Mexico macro source to macro_client.py.
No unofficial wrapper found for BMV stock disclosure data.

---

# =====================================================================
# MARKETS WITH NO WRAPPERS FOUND
# =====================================================================

| Market | Conclusion |
|--------|-----------|
| Canada (SEDAR+) | No PyPI package. SEDAR+ is relatively new. Gov API only. |
| Singapore (SGX) | No PyPI package. SGX doesn't expose a free API. |
| South Africa (JSE) | No PyPI package. JSE data behind paywall. |
| Switzerland (SIX) | No PyPI package. SIX data behind paywall. |
| UAE (DFM) | No PyPI package. DFM has limited free data. |
| Saudi Arabia (Tadawul) | No PyPI package. Tadawul has limited free data. |

---

## Recommendations

### High Priority (should add as dependencies)
1. **ixbrl-parse** -- Fills the UK Companies House financial data gap
2. **akshare** -- Comprehensive China market data for cn_sse.py
3. **jugaad-data** -- India NSE/BSE data for in_bse.py

### Medium Priority (worth adding)
4. **asx** -- Australia ASX data for au_asx.py
5. **twstock** -- Taiwan supplementary data for tw_mops_wrapper.py
6. **banxicoapi** -- Mexico macro data for macro_client.py

### Low Priority (too immature or limited)
7. **cmfapi** -- Chile CMF (v0.0.2, very early)
8. **openhkex** -- Hong Kong HKEX (v0.1.0, very early)
9. **tushare/baostock** -- China alternatives (akshare is better)

---

**END OF RESEARCH LOG**
