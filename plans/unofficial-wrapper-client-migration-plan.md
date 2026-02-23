# Unofficial Wrapper Client Migration Plan

## Overview

Replace the current direct-API PIT clients in `operator1/clients/` with unofficial Python wrappers/scrapers. Expand from 10 core markets to 25 markets total. Add multi-source macro data clients using community libraries. **OHLCV provider stays unchanged** unless a new wrapper natively provides price data.

## Guiding Principles

1. **Preserve the PITClient protocol** -- every new client implements the same interface defined in `pit_base.py`
2. **Normalize via canonical_translator.py** -- all raw data flows through the existing translation layer
3. **Cache only what OP1 needs** -- profile, normalized financials, filing metadata (~50-550KB per company)
4. **One market at a time** -- implement, test, debug sequentially
5. **OHLCV untouched** -- keep `ohlcv_provider.py` and Alpha Vantage as-is; only override if a wrapper provides native OHLCV
6. **Point-in-time integrity** -- every financial row must have `filing_date` + `report_date`

---

## Architecture Diagram

```mermaid
graph TD
    subgraph User Input
        A[Select Region + Market + Company]
    end

    subgraph PIT Registry
        B[pit_registry.py -- 25 markets]
    end

    subgraph Wrapper Clients -- NEW
        C1[us_edgar.py -- edgartools]
        C2[uk_companies_house.py -- scraper]
        C3[eu_esef.py -- pyesef/openesef]
        C4[fr_esef.py -- py-xbrl]
        C5[de_esef.py -- deutschland]
        C6[jp_edinet.py -- edinet-tools]
        C7[kr_dart.py -- dart-fss]
        C8[tw_mops.py -- TWSE scraper]
        C9[br_cvm.py -- pycvm]
        C10[cl_cmf.py -- cmf-scraper]
        C11[ca_sedar.py -- NEW]
        C12[au_asx.py -- pyasx]
        C13[in_bse.py -- BseIndiaApi]
        C14[cn_sse.py -- py-xbrl]
        C15[hk_hkex.py -- hkex-scraper]
        C16[sg_sgx.py -- SGX scraper]
        C17[mx_bmv.py -- bmv-info-api]
        C18[za_jse.py -- jse scraper]
        C19[ch_six.py -- six-scraper]
        C20[nl_afm.py -- pyesef]
        C21[es_cnmv.py -- pyesef]
        C22[it_consob.py -- py-xbrl]
        C23[se_fi.py -- insynsregistret]
        C24[sa_tadawul.py -- TadawulStocks]
        C25[ae_dfm.py -- Financial scraper]
    end

    subgraph Existing - Unchanged
        D[canonical_translator.py]
        E[ohlcv_provider.py -- Alpha Vantage]
        F[supplement.py -- OpenFIGI]
        G[llm_factory.py -- Gemini/Claude]
    end

    subgraph Macro Clients -- NEW
        M1[fredapi -- US FRED]
        M2[wbgapi -- World Bank global]
        M3[pandasdmx -- ECB/OECD/Eurostat]
        M4[global-macro-data -- offline panel]
    end

    subgraph Cache Layer
        H[cache/market/code/profile.json]
        I[cache/market/code/filings/period_end.json]
        J[cache/macro/country/indicator.parquet]
    end

    A --> B --> C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8 & C9 & C10
    C1 --> D --> H & I
    E --> H
    M1 & M2 & M3 & M4 --> J
```

---

## What Each Client Must Return

### Profile -- `get_profile(identifier)` returns dict

| Field | Type | Required | Source |
|-------|------|----------|--------|
| name | str | YES | wrapper |
| ticker | str | YES | wrapper |
| isin | str | optional | wrapper or OpenFIGI |
| country | str | YES | hardcoded per client |
| sector | str | YES | wrapper or OpenFIGI fallback |
| industry | str | optional | wrapper or OpenFIGI fallback |
| exchange | str | YES | wrapper |
| currency | str | YES | hardcoded per market |
| cik | str | optional | market-specific ID |
| market_cap | float | optional | wrapper if available |

### Financial Statements -- `get_income_statement()`, `get_balance_sheet()`, `get_cashflow_statement()` return pd.DataFrame

**Required columns in every DataFrame:**

| Column | Description |
|--------|-------------|
| `filing_date` | Date filing became publicly available -- CRITICAL for PIT |
| `report_date` | Fiscal period end date |
| `period_type` | annual / quarterly / semi-annual |

**Income statement fields to extract:**

| Canonical Name | Description |
|----------------|-------------|
| revenue | Net sales / total revenue |
| cost_of_revenue | Cost of goods sold |
| gross_profit | Revenue minus COGS |
| operating_income | EBIT proxy |
| net_income | Bottom line |
| ebit | Earnings before interest and tax |
| ebitda | EBIT + depreciation + amortization |
| taxes | Income tax expense |
| interest_expense | Finance costs |
| sga_expense | Selling, general, admin |
| research_and_development | R and D expense |

**Balance sheet fields:**

| Canonical Name | Description |
|----------------|-------------|
| total_assets | Sum of all assets |
| total_liabilities | Sum of all liabilities |
| total_equity | Shareholders equity |
| current_assets | Short-term assets |
| current_liabilities | Short-term liabilities |
| cash_and_equivalents | Cash + near-cash |
| short_term_debt | Current portion of debt |
| long_term_debt | Non-current debt |
| total_debt | Short + long term |
| retained_earnings | Accumulated earnings |
| receivables | Trade receivables |
| inventory | Inventories |
| payables | Trade payables |

**Cash flow fields:**

| Canonical Name | Description |
|----------------|-------------|
| operating_cash_flow | Cash from operations |
| investing_cf | Cash from investing |
| financing_cf | Cash from financing |
| capex | Capital expenditures |
| free_cash_flow | OCF minus capex |
| dividends_paid | Cash dividends |

### Filing Metadata -- embedded in financial DataFrames

| Field | Description |
|-------|-------------|
| filing_date | Submission/acceptance date |
| report_date | Period end |
| period_start | Period begin |
| form_code | Document type e.g. 10-K, Yuho |
| status | Active / withdrawn / restated |

---

## Cache Structure

```
cache/
  {market_id}/
    {ticker_or_code}/
      profile.json              # ~3 KB, refreshed weekly
      filings/
        {period_end}.json       # ~5-15 KB each, one per filing period
  macro/
    {country_code}/
      {indicator}_{start}_{end}.parquet   # macro time series
```

**Cache rules:**
- 2-year rolling window: delete filings older than 2 years from period_end
- Refresh only new data since last cached filing_date
- Profile refreshed if older than 7 days
- If wrapper fails, fall back to last cached version

---

## Market Implementation Schedule

### Phase 1: Core 10 Markets -- Replace Existing Clients

| # | Market | Wrapper 1 -- Primary | Wrapper 2 -- Fallback | Existing Client to Replace |
|---|--------|----------------------|----------------------|---------------------------|
| 1 | US -- SEC EDGAR | edgartools | sec-edgar-api | sec_edgar.py |
| 2 | UK -- Companies House | companies_house_scraper | companies-house-uk-scraper | companies_house.py |
| 3 | EU -- ESEF | pyesef | openesef | esef.py |
| 4 | France -- AMF/ESEF | py-xbrl | amf_scrapping | esef.py -- fr variant |
| 5 | Germany -- BaFin/ESEF | deutschland | bafin | esef.py -- de variant |
| 6 | Japan -- EDINET | edinet-tools | edinet-python | edinet.py |
| 7 | South Korea -- DART | dart-fss | gurukorea | dart.py |
| 8 | Taiwan -- MOPS | TWSE-Data-Crawling | MOPS revenue scraper | mops.py |
| 9 | Brazil -- CVM | pycvm | fundspy | cvm.py |
| 10 | Chile -- CMF | cmf-scraper | cmf_links_scraper | cmf.py |

### Phase 2: New 15 Markets -- Add New Clients

| # | Market | Wrapper 1 | Wrapper 2 | New Client File |
|---|--------|-----------|-----------|-----------------|
| 11 | Canada -- SEDAR | SEDAR-Data | pudo/sedar | ca_sedar.py |
| 12 | Australia -- ASX | pyasx | asx PyPI | au_asx.py |
| 13 | India -- BSE/SEBI | BseIndiaApi | bsedata | in_bse.py |
| 14 | China -- SSE/CSRC | py-xbrl | sec-xbrl-scraper adapted | cn_sse.py |
| 15 | Hong Kong -- HKEX | HKEX-web-scraping | hkex-filing-scraper | hk_hkex.py |
| 16 | Singapore -- SGX | SGX-data-project | SGX-Derivatives-Scraper | sg_sgx.py |
| 17 | Mexico -- BMV | bmv-info-api | pycvm adapted | mx_bmv.py |
| 18 | South Africa -- JSE | jse-stock-visualiser | jse_stock_analysis | za_jse.py |
| 19 | Switzerland -- SIX | six-scraper | manual adapter | ch_six.py |
| 20 | Netherlands -- AFM | pyesef -- NL filter | py-xbrl | nl_afm.py |
| 21 | Spain -- CNMV | pyesef -- ES filter | investpy | es_cnmv.py |
| 22 | Italy -- CONSOB | py-xbrl -- IT filter | sec-xbrl-scraper adapted | it_consob.py |
| 23 | Sweden -- FI | insynsregistret | pyscbwrapper | se_fi.py |
| 24 | Saudi Arabia -- Tadawul | TadawulStocks | Saudi_Stock_Web_Scrape | sa_tadawul.py |
| 25 | UAE -- DFM/ADX | Financial-Data-Scraper | investpy | ae_dfm.py |

### Phase 3: Macro Client Upgrade

| Library | Coverage | Replaces |
|---------|----------|----------|
| fredapi | US FRED -- 800k+ series | Current `_fetch_fred()` in macro_client.py |
| wbgapi | World Bank -- 200+ countries | Current manual World Bank calls |
| pandasdmx | ECB, OECD, Eurostat, INSEE, Bundesbank | Current `_fetch_ecb()` etc. |
| global-macro-data | 46 variables, 243 countries, offline | New fallback for all |

---

## Per-Client Implementation Template

Each new client file follows this pattern:

```python
# operator1/clients/{market}.py

class {Market}Client:
    # Implements PITClient protocol

    @property
    def market_id(self) -> str: ...

    @property
    def market_name(self) -> str: ...

    def list_companies(self, query='') -> list[dict]: ...
    def search_company(self, name) -> list[dict]: ...
    def get_profile(self, identifier) -> dict: ...
    def get_income_statement(self, identifier) -> pd.DataFrame: ...
    def get_balance_sheet(self, identifier) -> pd.DataFrame: ...
    def get_cashflow_statement(self, identifier) -> pd.DataFrame: ...
    def get_quotes(self, identifier) -> pd.DataFrame:
        # Return empty -- OHLCV handled by ohlcv_provider.py
        return pd.DataFrame()
    def get_peers(self, identifier) -> list[str]: ...
    def get_executives(self, identifier) -> list[dict]: ...
```

---

## Files Modified vs Created

### Modified Files

| File | Changes |
|------|---------|
| `operator1/clients/pit_registry.py` | Add 15 new MarketInfo entries for Phase 2 markets |
| `operator1/clients/canonical_translator.py` | Add field mappings for new market GAAP variants |
| `operator1/clients/equity_provider.py` | Update factory to instantiate new client classes |
| `operator1/clients/__init__.py` | Update docstring with new markets |
| `operator1/clients/macro_client.py` | Replace manual HTTP calls with fredapi/wbgapi/pandasdmx |
| `config/global_config.yml` | Add wrapper-specific config if needed |
| `.env.example` | Add any new API key placeholders |
| `requirements.txt` | Add new wrapper dependencies |

### New Files -- Phase 1 (replace existing)

| File | Replaces |
|------|----------|
| `operator1/clients/us_edgar.py` | `sec_edgar.py` -- uses edgartools |
| `operator1/clients/uk_ch.py` | `companies_house.py` -- uses scraper |
| `operator1/clients/eu_esef_wrapper.py` | `esef.py` -- uses pyesef |
| `operator1/clients/jp_edinet_wrapper.py` | `edinet.py` -- uses edinet-tools |
| `operator1/clients/kr_dart_wrapper.py` | `dart.py` -- uses dart-fss |
| `operator1/clients/tw_mops_wrapper.py` | `mops.py` -- uses TWSE scraper |
| `operator1/clients/br_cvm_wrapper.py` | `cvm.py` -- uses pycvm |
| `operator1/clients/cl_cmf_wrapper.py` | `cmf.py` -- uses cmf-scraper |

### New Files -- Phase 2 (new markets)

| File | Market |
|------|--------|
| `operator1/clients/ca_sedar.py` | Canada |
| `operator1/clients/au_asx.py` | Australia |
| `operator1/clients/in_bse.py` | India |
| `operator1/clients/cn_sse.py` | China |
| `operator1/clients/hk_hkex.py` | Hong Kong |
| `operator1/clients/sg_sgx.py` | Singapore |
| `operator1/clients/mx_bmv.py` | Mexico |
| `operator1/clients/za_jse.py` | South Africa |
| `operator1/clients/ch_six.py` | Switzerland |
| `operator1/clients/nl_afm.py` | Netherlands |
| `operator1/clients/es_cnmv.py` | Spain |
| `operator1/clients/it_consob.py` | Italy |
| `operator1/clients/se_fi.py` | Sweden |
| `operator1/clients/sa_tadawul.py` | Saudi Arabia |
| `operator1/clients/ae_dfm.py` | UAE |

### New Test Files

| File | Tests |
|------|-------|
| `tests/test_wrapper_us.py` | US edgartools client -- AAPL + 2 peers |
| `tests/test_wrapper_uk.py` | UK CH client |
| `tests/test_wrapper_{market}.py` | One per market |
| `tests/test_macro_wrappers.py` | fredapi + wbgapi + pandasdmx integration |

---

## OHLCV Policy

**The existing `ohlcv_provider.py` is NOT modified.** It continues using Alpha Vantage for all markets. The only exception: if a new wrapper natively returns OHLCV data in its `get_quotes()` method, that client can return it instead of an empty DataFrame. The pipeline already handles both cases -- if a PIT client returns OHLCV, it uses it; otherwise it falls back to `ohlcv_provider.fetch_ohlcv()`.

---

## Dependency Additions to requirements.txt

```
# Phase 1: Unofficial wrappers for core markets
edgartools>=0.30          # US SEC EDGAR
pyesef>=0.3               # EU ESEF XBRL
py-xbrl>=2.0              # France/Germany/Italy XBRL
dart-fss>=0.4             # South Korea DART
pycvm>=0.5                # Brazil CVM

# Phase 2: New market wrappers
pyasx>=0.1                # Australia ASX
bsedata>=0.6              # India BSE

# Phase 3: Macro data libraries
fredapi>=0.5              # US FRED (replaces manual calls)
wbgapi>=1.0               # World Bank (global)
pandasdmx>=1.0            # SDMX: ECB, OECD, Eurostat, INSEE
global-macro-data>=0.1    # Offline 243-country macro panel
```

Note: Some wrappers are scrapers without PyPI packages -- those will be vendored or installed via git URL.

---

## Implementation Order -- Detailed Steps

For EACH market, the workflow is:

1. Install wrapper dependency
2. Create client file implementing PITClient
3. Add canonical_translator mappings for that markets GAAP
4. Register in pit_registry.py
5. Update equity_provider.py factory
6. Write test for sample company + 2 peers
7. Run test and debug
8. Confirm with user before moving to next market

**Start with US market using edgartools, then proceed sequentially.**
