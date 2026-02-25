# Research Log: Alternative Wrapper Search (Zero-Key PIT Alternatives)
**Generated**: 2026-02-24T22:05:00Z
**Status**: Complete
**Purpose**: Search for community wrappers that can provide PIT financial statement data WITHOUT API keys

---

## Search Results

### Korea -- Alternatives to dart-fss (which needs DART_API_KEY)

| Library | Version | Stars | Key Required? | Financial Statements? | Last Push |
|---------|---------|-------|---------------|----------------------|-----------|
| dart-fss | 0.4.15 | N/A | YES (free) | YES -- full income, balance, cashflow | 2026-02-20 |
| pykrx | 1.2.4 | 905 | NO | NO -- OHLCV + basic ratios only (PER, PBR, EPS) | 2026-02-20 |
| finance-datareader | 0.9.102 | 1,418 | NO | NO -- OHLCV + listings only | 2026-02-01 |
| krxmarket | 0.1.0 | 0 | NO | NO -- basic market info only | 2023-03-07 |

**Conclusion**: No zero-key alternative provides Korean financial statements. dart-fss with free DART_API_KEY is the only option.

### Japan -- Alternatives to edinet-tools (which needs EDINET_API_KEY)

| Library | Version | Stars | Key Required? | Financial Statements? | Last Push |
|---------|---------|-------|---------------|----------------------|-----------|
| edinet-tools | 0.3.0 | N/A | YES (free) | YES -- securities reports, quarterly reports | 2026-01-19 |
| jquants-api-client | 2.0.0 | 183 | YES (free J-Quants key) | YES -- JPX official data | 2026-01-19 |
| japandas | 0.5.1 | N/A | NO | NO -- pandas Japanese locale extensions only | N/A |
| kabu | 0.0.4 | N/A | NO | NO -- basic stock investment library | N/A |

**Conclusion**: No zero-key alternative provides Japanese financial statements. Both edinet-tools and jquants-api-client need free registration keys.

### UK -- Alternatives to uk_ch_wrapper (which needs COMPANIES_HOUSE_API_KEY)

| Library | Version | Stars | Key Required? | Financial Statements? | Last Push |
|---------|---------|-------|---------------|----------------------|-----------|
| uk_ch_wrapper | N/A | N/A | YES (free) | NO (documented gap -- returns metadata only, not numbers) | N/A |
| ixbrl-parse | 0.10.1 | 9 | NO | YES (can parse iXBRL files into numbers) | 2025-12-16 |

**Conclusion**: ixbrl-parse can extract financial numbers from iXBRL documents, but you still need Companies House API to DOWNLOAD those documents. Without the CH API key, you can't get the documents to parse.

### Global -- SimFin

| Library | Version | Stars | Key Required? | Financial Statements? | Coverage |
|---------|---------|-------|---------------|----------------------|----------|
| simfin | 1.0.1 | 330 | YES (free) | YES -- income, balance, cashflow | US primarily, some global |

**Conclusion**: SimFin needs a free API key and is mostly US-focused.

---

## Why Zero-Key PIT Financial Data Is Impossible

Financial statement data (income statement, balance sheet, cash flow) with filing dates is **regulatory disclosure data**. It originates from government filing systems:

- **US**: SEC EDGAR (the filing system IS the data source)
- **Korea**: DART (operated by Korea's Financial Supervisory Service)
- **Japan**: EDINET (operated by Japan's Financial Services Agency)
- **UK**: Companies House (UK government executive agency)
- **EU**: ESEF (EU regulation, filings via XBRL registries)
- **Taiwan**: MOPS (operated by TWSE/TPEX)
- **Brazil**: CVM (operated by Brazilian Securities Commission)

Community wrappers (edgartools, dart-fss, edinet-tools) are just Python interfaces to these systems. The government systems control access. When they require a key, there's no way around it -- you can't scrape the data without going through their system.

The ONLY exceptions are:
- **US SEC EDGAR**: Truly free, no key -- just needs User-Agent email
- **EU ESEF (filings.xbrl.org)**: Truly free, no key
- **Taiwan MOPS**: Truly free, no key (form POST scraping)
- **Brazil CVM**: Truly free, no key (CSV downloads)
- **Chile CMF**: Truly free, no key

---

## Final Assessment

| Market | Wrapper | Key Required? | Can Avoid Key? |
|--------|---------|---------------|----------------|
| US | edgartools | Email only (not a real key) | YES -- email is just a User-Agent |
| EU | eu_esef_wrapper | NO | Already key-free |
| UK | uk_ch_wrapper | YES (free) | NO -- no alternative exists |
| Japan | edinet-tools | YES (free) | NO -- jquants also needs key |
| Korea | dart-fss | YES (free) | NO -- pykrx only has OHLCV, not financials |
| Taiwan | tw_mops_wrapper | NO | Already key-free |
| Brazil | br_cvm_wrapper | NO | Already key-free |
| Chile | cl_cmf_wrapper | NO | Already key-free |

**Bottom line**: 5 of 8 Tier 1 markets are already zero-key. The 3 that need keys (UK, Japan, Korea) have no zero-key alternative for financial statement data. The keys are all free registration -- no cost involved.

---

**END OF RESEARCH LOG**
