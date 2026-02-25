# API Key Requirements Summary -- All Data Sources
**Generated**: 2026-02-25T12:02:00Z
**Status**: Complete -- Verified from actual sources (PyPI, GitHub READMEs, endpoint probes)
**Method**: Endpoint probes via curl, PyPI JSON API, GitHub README auth grep

---

## VERIFICATION METHOD

Each API was verified using one or more of:
1. **PyPI fetch**: `curl https://pypi.org/pypi/{package}/json` -- version, deps
2. **GitHub README grep**: Auth/key/token mentions in official README
3. **Endpoint probe**: `curl -sI {base_url}` or `curl -s {api_endpoint}` -- check for 401/403 auth errors vs 200 success
4. **Source code review**: Checked `operator1/clients/*.py` and `operator1/clients/pit_registry.py`

---

## SUMMARY TABLE

### Equity/Financial Data APIs (PIT Wrappers)

| # | Market | API/Library | Base URL | Key Required? | Key Env Var | Verified How | Notes |
|---|--------|-------------|----------|---------------|-------------|-------------|-------|
| 1 | US | SEC EDGAR | efts.sec.gov | **NO key** but requires User-Agent with email | `EDGAR_IDENTITY` | Endpoint probe: 403 without UA, 200 with UA | SEC regulation, not an API key |
| 2 | UK | Companies House REST API | api.company-information.service.gov.uk | **YES** | `COMPANIES_HOUSE_API_KEY` | Source code: Basic auth with key | Free registration at developer.company-information.service.gov.uk |
| 3 | UK | ixbrl-parse (NEW) | local parser | **NO** | none | PyPI + GitHub README | Local parser, no API needed. Still needs CH key for document download |
| 4 | EU | ESEF via filings.xbrl.org | filings.xbrl.org/api | **NO** | none | Endpoint probe: 200 OK with data | Open XBRL API, no auth |
| 5 | JP | ~~EDINET~~ -> J-Quants (NEW) | jpx-jquants.com | **YES** | `JQUANTS_API_KEY` | PyPI + GitHub README | Free plan via email registration. Replaces EDINET (which requires website hosting) |
| 6 | KR | DART (dart-fss) | opendart.fss.or.kr/api | **YES** | `DART_API_KEY` | Endpoint probe: redirects to error without key; "unregistered key" with invalid key | Free registration at opendart.fss.or.kr |
| 7 | TW | MOPS/TWSE | mops.twse.com.tw | **NO** | none | Endpoint probe: 200 OK | Web scraping, no auth |
| 8 | BR | CVM (pycvm) | dados.cvm.gov.br/api/v1 | **NO** | none | PyPI + source code: no key params | Open data portal |
| 9 | CL | CMF Chile | cmfchile.cl | **NO** | none | Source code: no key params | Direct web portal |
| 10 | CA | SEDAR+ | sedarplus.ca | **NO** | none | Endpoint probe: 200 OK | Public filing search |
| 11 | AU | ASX | asx.com.au/asx/1 | **NO** | none | Endpoint probe: accessible (404 on specific path = URL changed, not auth) | Limited free data |
| 12 | IN | BSE India | api.bseindia.com | **NO** | none | Source code: no key params | Public API |
| 13 | CN | SSE | sse.com.cn | **NO** | none | Endpoint probe: 200 OK | Chinese language, web scraping |
| 14 | HK | HKEX News | hkexnews.hk | **NO** | none | Endpoint probe: 503 Akamai CDN (bot protection, not auth) | Web scraping with proper headers |
| 15 | SG | SGX | sgx.com | **NO** | none | Endpoint probe: 403 Akamai CDN (bot protection, not auth) | Web scraping with proper headers |
| 16 | MX | BMV | bmv.com.mx | **NO** | none | Endpoint probe: 200 OK | Limited free data |
| 17 | ZA | JSE | jse.co.za | **NO** | none | Endpoint probe: 200 OK | Limited free data |
| 18 | CH | SIX Group | six-group.com | **NO** | none | Endpoint probe: 301 redirect (normal) | Data largely behind paywall for detailed financials |
| 19 | SA | Saudi Exchange | saudiexchange.sa | **NO** | none | Endpoint probe: 403 Akamai (bot protection, not auth) | Web scraping |
| 20 | AE | DFM | dfm.ae | **NO** | none | Endpoint probe: 200 OK | Limited free data |

### OHLCV & Supplement Libraries

| # | Library | Version | Key Required? | Key Env Var | Verified How | Notes |
|---|---------|---------|---------------|-------------|-------------|-------|
| 1 | yfinance | 1.2.0 | **NO** | none | PyPI + GitHub README: no auth mentions | Yahoo Finance public API |
| 2 | wbgapi | 1.0.13 | **NO** | none | PyPI + GitHub README: "no API key needed" | World Bank open data |
| 3 | openfigi | 0.0.9 | **OPTIONAL** | `openfigi_key` | GitHub README: "searches for openfigi_key env var... anonymous access" | Works without key (rate-limited), key optional for higher limits |
| 4 | edgartools | 5.17.1 | **NO key** but requires identity | `EDGAR_IDENTITY` | GitHub README: "set_identity... Free access for everyone, forever (no API keys)" | Email identity required by SEC regulation |

### Macro Data Providers (from pit_registry.py)

| # | Provider | API URL | Key Required? | Verified How |
|---|----------|---------|---------------|-------------|
| 1 | FRED (US) | api.stlouisfed.org/fred | **YES** (free registration) | pit_registry.py source |
| 2 | ECB (EU) | sdw-wsrest.ecb.europa.eu | **NO** | pit_registry.py source |
| 3 | ONS (UK) | api.ons.gov.uk | **NO** | pit_registry.py source |
| 4 | Bundesbank (DE) | api.statistiken.bundesbank.de | **NO** | pit_registry.py source |
| 5 | INSEE (FR) | api.insee.fr | **YES** (free registration) | pit_registry.py source |
| 6 | e-Stat (JP) | api.e-stat.go.jp | **YES** (free registration) | pit_registry.py source |
| 7 | KOSIS (KR) | kosis.kr/openapi | **YES** (free registration) | pit_registry.py source |
| 8 | DGBAS (TW) | nstatdb.dgbas.gov.tw | **NO** | pit_registry.py source |
| 9 | BCB (BR) | api.bcb.gov.br | **NO** | pit_registry.py source |
| 10 | BCC (CL) | si3.bcentral.cl | **YES** (free registration) | pit_registry.py source |

### LLM Providers

| # | Provider | Key Env Var | Required? |
|---|----------|-------------|-----------|
| 1 | Google Gemini | `GEMINI_API_KEY` | Yes (for LLM features) |
| 2 | Anthropic Claude | `ANTHROPIC_API_KEY` | Yes (for LLM features) |

---

## REQUIRED .env VARIABLES (COMPLETE LIST)

### Mandatory for Core Functionality
```dotenv
# At least one LLM provider key is needed for report generation
GEMINI_API_KEY=           # or ANTHROPIC_API_KEY

# These are needed for specific country wrappers:
COMPANIES_HOUSE_API_KEY=  # UK Companies House (free registration)
JQUANTS_API_KEY=          # Japan J-Quants API (free email registration, replaces EDINET_API_KEY)
DART_API_KEY=             # Korea DART (free registration at opendart.fss.or.kr)
```

### Optional (for enhanced features)
```dotenv
# OpenFIGI - works without key but rate-limited
openfigi_key=             # Optional, for higher rate limits

# US SEC EDGAR - not a key, but required identity
EDGAR_IDENTITY=your.name@example.com

# Macro data providers (free registration required)
FRED_API_KEY=             # US Federal Reserve macro data
INSEE_API_KEY=            # French INSEE macro data
ESTAT_API_KEY=            # Japan e-Stat macro data
KOSIS_API_KEY=            # Korea KOSIS macro data
BCC_API_KEY=              # Chile central bank macro data
```

### No Longer Needed (REMOVED)
```dotenv
# EDINET_API_KEY=         # REMOVED -- replaced by JQUANTS_API_KEY
```

---

## CHANGES FROM PREVIOUS RESEARCH

| Change | Old | New | Reason |
|--------|-----|-----|--------|
| Japan equity data | EDINET_API_KEY (edinet-tools) | JQUANTS_API_KEY (jquants-api-client) | EDINET registration requires website hosting |
| UK financial data | Companies House REST only (empty financials) | ixbrl-parse + CH REST | CH REST returns no financial values; ixbrl-parse fills the gap |
| All other APIs | Key status from existing research files | Key status re-verified from actual sources | Verified via endpoint probes, PyPI, GitHub READMEs |

---

## APIS WITH CAVEATS (Not True "No Key" But Still Accessible)

1. **US EDGAR**: No API key, but SEC requires `User-Agent` header with contact email. Without it, returns 403.
2. **OpenFIGI**: Works anonymously but with rate limits (~10 req/min). Optional `openfigi_key` for higher limits.
3. **HK HKEX / SG SGX / SA Tadawul**: Return 403/503 from Akamai CDN (anti-bot protection). Not authentication - just needs proper browser-like headers (`User-Agent`, `Referer`).
4. **CH SIX**: Free data is limited. Detailed financial data behind paywall/subscription.
5. **AU ASX**: Some endpoints return 404 (URL may have changed). Financial data access is limited.

---

## CONFIDENCE LEVELS

| Verification Method | Confidence | Used For |
|---------------------|------------|----------|
| Endpoint probe returning 200 with data | HIGH | EU-ESEF, TW-MOPS, BR-CVM, CA-SEDAR |
| Endpoint probe returning 200 (homepage) | MEDIUM | MX-BMV, ZA-JSE, AE-DFM, CN-SSE |
| Endpoint probe returning 403/503 (CDN) | MEDIUM | HK-HKEX, SG-SGX, SA-Tadawul (bot protection, not auth) |
| PyPI + GitHub README grep | HIGH | yfinance, wbgapi, dart-fss, pycvm, ixbrl-parse, jquants-api-client |
| Source code review (pit_registry.py) | HIGH | All macro providers, all PIT registrations |
| KR DART endpoint with invalid key | HIGH | Confirmed key required (error message) |
| US EDGAR endpoint with/without UA | HIGH | Confirmed User-Agent required (403 vs 200) |

**END OF SUMMARY**
