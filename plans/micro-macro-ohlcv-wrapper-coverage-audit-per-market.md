# Micro / Macro / OHLCV Wrapper Coverage Audit

## Coverage Matrix (25 Markets)

| # | Market ID | Country | Micro (PIT) | OHLCV Primary | OHLCV Fallback | Macro Primary | Macro Fallback | Translator |
|---|-----------|---------|-------------|---------------|----------------|---------------|----------------|------------|
| | **Tier 1 (10 markets)** | | | | | | | |
| 1 | us_sec_edgar | US | us_edgar.py | -- | yfinance | fred | wbgapi | US-GAAP |
| 2 | uk_companies_house | UK | uk_ch_wrapper.py | -- | yfinance (.L) | -- | wbgapi | UK-GAAP+IFRS |
| 3 | eu_esef | EU | eu_esef_wrapper.py | -- | yfinance (.PA) | ecb | wbgapi | IFRS |
| 4 | fr_esef | France | eu_esef_wrapper.py | -- | yfinance (.PA) | ecb | wbgapi | IFRS |
| 5 | de_esef | Germany | eu_esef_wrapper.py | -- | yfinance (.DE) | ecb | wbgapi | IFRS |
| 6 | jp_jquants | Japan | jp_jquants_wrapper.py | -- | yfinance (.T) | -- | wbgapi | JPPFS+IFRS |
| 7 | kr_dart | Korea | kr_dart_wrapper.py | pykrx | yfinance (.KS) | -- | wbgapi | K-IFRS |
| 8 | tw_mops | Taiwan | tw_mops_wrapper.py | twstock | yfinance (.TW) | -- | wbgapi | TIFRS+IFRS |
| 9 | br_cvm | Brazil | br_cvm_wrapper.py | -- | yfinance (.SA) | bcb | wbgapi | CVM codes |
| 10 | cl_cmf | Chile | cl_cmf_wrapper.py | -- | yfinance (.SN) | -- | wbgapi | CMF+IFRS |
| | **Tier 2 (15 markets)** | | | | | | | |
| 11 | ca_sedar | Canada | ca_sedar.py | -- | yfinance (.TO) | -- | wbgapi | IFRS+US-GAAP |
| 12 | au_asx | Australia | au_asx.py | -- | yfinance (.AX) | -- | wbgapi | IFRS |
| 13 | in_bse | India | in_bse.py | jugaad | yfinance (.NS) | -- | wbgapi | IFRS |
| 14 | cn_sse | China | cn_sse.py | akshare | yfinance (.SS) | -- | wbgapi | CAS+IFRS |
| 15 | hk_hkex | Hong Kong | hk_hkex.py | -- | yfinance (.HK) | -- | wbgapi | IFRS |
| 16 | sg_sgx | Singapore | sg_sgx.py | -- | yfinance (.SI) | -- | wbgapi | IFRS |
| 17 | mx_bmv | Mexico | mx_bmv.py | -- | yfinance (.MX) | banxico | wbgapi | IFRS |
| 18 | za_jse | South Africa | za_jse.py | -- | yfinance (.JO) | -- | wbgapi | IFRS |
| 19 | ch_six | Switzerland | ch_six.py | -- | yfinance (.SW) | -- | wbgapi | IFRS |
| 20 | nl_esef | Netherlands | eu_esef_wrapper.py | -- | **MISSING** | -- | wbgapi | IFRS |
| 21 | es_esef | Spain | eu_esef_wrapper.py | -- | **MISSING** | -- | wbgapi | IFRS |
| 22 | it_esef | Italy | eu_esef_wrapper.py | -- | **MISSING** | -- | wbgapi | IFRS |
| 23 | se_esef | Sweden | eu_esef_wrapper.py | -- | **MISSING** | -- | wbgapi | IFRS |
| 24 | sa_tadawul | Saudi Arabia | sa_tadawul.py | -- | yfinance (.SR) | -- | wbgapi | IFRS |
| 25 | ae_dfm | UAE | ae_dfm.py | -- | yfinance (.AE) | -- | wbgapi | IFRS |

## Gaps Found

### OHLCV Gaps (4 markets)

**Missing yfinance suffix mappings** in `ohlcv_yfinance.py`:
- `nl_esef` -- needs `.AS` (Euronext Amsterdam)
- `es_esef` -- needs `.MC` (BME Madrid)
- `it_esef` -- needs `.MI` (Borsa Italiana)
- `se_esef` -- needs `.ST` (Nasdaq Stockholm)

Without these suffixes, yfinance cannot resolve tickers for Netherlands, Spain, Italy, and Sweden. These markets will have zero OHLCV data.

### Macro Gaps (19 markets relying only on wbgapi fallback)

Markets with **no primary macro fetcher** -- they all fall back to wbgapi (World Bank), which provides annual data only (GDP, inflation, unemployment). This means:
- No high-frequency (monthly/quarterly) macro indicators
- No interest rates, yield curves, or credit spreads from primary sources

| Country Code | Market | Missing Primary | Registered in pit_registry MACRO_APIS |
|---|---|---|---|
| GB | UK | ONS (planned) | Yes -- uk_ons |
| JP | Japan | e-Stat (planned) | Yes -- jp_estat |
| KR | Korea | KOSIS (planned) | Yes -- kr_kosis |
| TW | Taiwan | DGBAS | Yes -- tw_dgbas |
| CL | Chile | BCCh | Yes -- cl_bcch |
| CA | Canada | -- | No |
| AU | Australia | -- | No |
| IN | India | -- | No |
| CN | China | -- | No |
| HK | Hong Kong | -- | No |
| SG | Singapore | -- | No |
| ZA | South Africa | -- | No |
| CH | Switzerland | -- | No |
| NL | Netherlands | -- | No (uses EU/ECB) |
| ES | Spain | -- | No (uses EU/ECB) |
| IT | Italy | -- | No (uses EU/ECB) |
| SE | Sweden | -- | No |
| SA | Saudi Arabia | -- | No |
| AE | UAE | -- | No |

Note: UK, JP, KR, TW, CL have their macro APIs defined in `pit_registry.MACRO_APIS` but their fetchers are NOT implemented in `macro_provider.py`. The registry metadata exists but the actual data fetching code is missing.

### Micro (PIT) Protocol Gap

**J-Quants wrapper** (`jp_jquants_wrapper.py`) is missing the `PITClient` protocol methods:
- `get_income_statement()` -- missing (has `get_financials()` instead)
- `get_balance_sheet()` -- missing
- `get_cashflow_statement()` -- missing
- `get_quotes()` -- missing
- `search_company()` -- missing

This means `data_extraction.py` will fail when calling these methods on the J-Quants client. The wrapper needs adapter methods that delegate to `get_financials()`.

### Translator Coverage

All 25 markets have concept maps in `canonical_translator.py`. No gaps here.

## Recommended Fix Priority

1. **CRITICAL**: Add missing PITClient methods to J-Quants wrapper (or add adapter in equity_provider)
2. **HIGH**: Add yfinance suffix mappings for NL, ES, IT, SE
3. **MEDIUM**: Wire the 5 planned macro primary fetchers (UK/JP/KR/TW/CL) that have registry entries but no implementation
4. **LOW**: Add macro primary fetchers for remaining Tier 2 markets (wbgapi provides adequate fallback)
