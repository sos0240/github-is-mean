# API Endpoint, Rate Limit, and Flow Audit

## Audit Summary

Reviewed all 10 PIT clients, supplement APIs, macro clients, OHLCV provider, `run.py`, and `main.py` for endpoint correctness, rate limit handling, menu completeness, and logical flow.

---

## Critical Bugs Found

### BUG-1: `main.py` uses wrong constructor parameter for ESEF

**File**: `main.py` line 291  
**Issue**: `ESEFClient(country_filter=country_filter)` -- but the actual constructor in `esef.py` uses `country_code`, not `country_filter`. Also passes `None` instead of `""` for pan-EU.  
**Impact**: ESEFClient will get `country_filter` as an unexpected kwarg and raise `TypeError`.  
**Fix**: Change to `ESEFClient(country_code=country_filter or "", market_id=market_id)`.  
**Note**: The duplicate `_create_pit_client()` in `main.py` (lines 262-320) is redundant with the one in `equity_provider.py` (which is correct). Ideally remove the `main.py` duplicate and use `create_pit_client()` from `equity_provider.py`.

### BUG-2: `http_utils.py` `cached_get` does not support `raw_text` parameter

**File**: `macro_client.py` line 98  
**Issue**: `cached_get(url, params=params, raw_text=True)` -- but `cached_get()` has no `raw_text` parameter. This will cause a `TypeError` when ECB macro data is fetched.  
**Impact**: ECB SDW macro fetch will fail for EU/DE/FR markets.  
**Fix**: Either add `raw_text` support to `cached_get()`, or use `requests.get()` directly in the ECB fetcher.

### BUG-3: Global rate limiter is per-process, not per-host

**File**: `http_utils.py` lines 99-118  
**Issue**: The rate limiter uses a single `_last_request_time` global. This means ALL API calls share one rate limit, not per-host. A burst of SEC EDGAR calls will throttle FRED calls too.  
**Impact**: Unnecessarily slow when hitting multiple APIs in sequence. Also, the `rate_limit_calls_per_second: 2.0` in config is calibrated for FMP (legacy) not for the actual APIs being used.  
**Recommendation**: Implement per-host rate limiting with a dict `{host: last_time}`. Different APIs have very different limits.

---

## API Endpoint Verification

### Correct Endpoints

| Client | Base URL | Status |
|--------|----------|--------|
| SEC EDGAR | `https://data.sec.gov/submissions`, `https://data.sec.gov/api/xbrl/companyfacts` | Correct, well-documented |
| Companies House | `https://api.company-information.service.gov.uk` | Correct |
| ESEF | `https://filings.xbrl.org/api` | Correct |
| EDINET | `https://api.edinet-fsa.go.jp/api/v2` | Correct |
| DART | `https://opendart.fss.or.kr/api` | Correct |
| CVM | `https://dados.cvm.gov.br/api/v1` | Correct |
| Alpha Vantage | `https://www.alphavantage.co/query` | Correct |
| TWSE OHLCV | `https://www.twse.com.tw/exchangeReport/STOCK_DAY` | Correct |
| FRED | `https://api.stlouisfed.org/fred` | Correct |
| ECB SDW | `https://sdw-wsrest.ecb.europa.eu/service` | Correct |
| BCB | `https://api.bcb.gov.br/dados/serie/bcdata.sgs` | Correct |

### Endpoints Needing Verification

| Client | Base URL | Concern |
|--------|----------|---------|
| MOPS | `https://mops.twse.com.tw` | MOPS endpoints (`/mops/web/ajax_t51sb01`, etc.) use form POST not GET. `cached_get()` only does GET. Financial data fetches may silently fail and return empty DataFrames. |
| CMF | `https://www.cmfchile.cl` | The `/portal/informacion/entidades/busqueda` and `/portal/estadisticas/fecu` endpoints may not return JSON. CMF portal is primarily HTML-based. |
| OpenFIGI | `https://api.openfigi.com/v3/mapping` | Uses POST (correctly implemented with `requests.post`), not routed through `cached_get`. No caching for FIGI lookups. |
| Euronext | `https://live.euronext.com/en/search_instruments` | This endpoint may not return JSON. Euronext live site is HTML-based. May need their official API. |
| JPX | `https://quote.jpx.co.jp/jpx/template/quote.cgi` | Returns HTML, not JSON. The `cached_get` will fail to parse it. |
| TWSE Company Info | `https://www.twse.com.tw/en/api/codeQuery` | May or may not return JSON depending on headers. |

---

## Rate Limits by API

| API | Rate Limit | Current Handling | Issue |
|-----|-----------|-----------------|-------|
| SEC EDGAR | 10 requests/sec (with User-Agent) | Global 2/sec rate limit | Over-throttled. SEC allows 10/s. |
| Companies House | 600/5min (no key), 600/5min (with key) | Global 2/sec | Adequate |
| ESEF (filings.xbrl.org) | No documented limit | Global 2/sec | Adequate |
| EDINET | No documented limit (generous) | Global 2/sec | Adequate |
| DART | 1000/day (free key) | Global 2/sec | Adequate but no daily counter |
| MOPS (TWSE) | No documented limit, but aggressive scraping blocked | Global 2/sec | Should be slower (1/sec recommended) |
| CVM | No documented limit | Global 2/sec | Adequate |
| CMF | No documented limit | Global 2/sec | Adequate |
| Alpha Vantage | 25/day (free), 500/min (paid) | No daily counter | **Missing daily call counter**. Will silently exhaust free tier. |
| OpenFIGI | 25/min (no key), 250/min (with key) | No rate limiting | Uses `requests.post` directly, bypasses `cached_get` rate limiter |
| FRED | 120/min (with key) | Global 2/sec | Adequate |
| ECB SDW | No limit | Global 2/sec | Adequate |

### Key Tier Recognition

**Not implemented.** The app treats all keys the same regardless of tier. Specifically:
- Alpha Vantage: free (25/day) vs premium (500/min) -- no differentiation
- DART: no daily call counter
- FRED: no per-minute counter

**Recommendation**: Add a `RateLimitProfile` config that maps `{api_name: {calls_per_day, calls_per_min}}` and track usage per-host.

---

## Menu Completeness

### Company Listing Issues

| Client | `list_companies()` Behavior | Issue |
|--------|---------------------------|-------|
| SEC EDGAR | Fetches `company_tickers.json` (~13K entries) | Complete, correct |
| Companies House | Requires a search query (no browse-all) | Correct behavior (CH API requires search term) |
| ESEF | Fetches 500 recent filings, deduplicates entities | Incomplete. Only shows companies with recent filings, not all EU-listed companies. `limit: 500` may miss many. |
| EDINET | Scans last 7 days of filings | Very incomplete. Only shows companies that filed in the last 7 days. Most companies file annually. Should expand to 90+ days or use the EDINET code list CSV. |
| DART | Searches recent disclosures via `/list.json` | Incomplete. Only shows companies with recent disclosures. For full list, needs the `corpCode.xml` download. |
| MOPS | Tries `/mops/web/ajax_t51sb01` | Likely fails (POST endpoint called via GET). Returns empty. |
| CVM | Fetches `/cia_aberta` with `$top=500` | Incomplete. CVM has 400+ companies, 500 should cover most but pagination may be needed. |
| CMF | Tries `/portal/informacion/entidades/busqueda` | Likely fails (HTML endpoint). Returns empty. |

### Recommendation for EDINET
EDINET provides a downloadable EDINET code list (CSV) that maps all ~5800 edinetCodes to company names/tickers. This should be fetched once and cached, rather than scanning 7 days of filings.

---

## run.py vs main.py Flow

### Logical Flow Assessment

`run.py` is the interactive launcher that:
1. Checks Python version, internet, dependencies
2. Lets user pick region, market, company
3. Checks optional API keys (prompts for GEMINI_API_KEY if missing)
4. Asks about pipeline options (linked entities, models, PDF)
5. Shows confirmation summary
6. Spawns `main.py` as a subprocess

`main.py` is the actual pipeline that:
1. Loads secrets from .env
2. Selects region/market/company (interactive or CLI)
3. Creates PIT client
4. Fetches profile, financials, OHLCV
5. Fetches macro data
6. Builds daily cache
7. Runs estimation, features, models
8. Generates report

### Issues

1. **Duplicate client creation**: `main.py` has its own `_create_pit_client()` (lines 262-320) that duplicates `equity_provider.create_pit_client()`. The `main.py` version has the ESEF bug (BUG-1 above). Should use the `equity_provider` version.

2. **ALPHA_VANTAGE_API_KEY missing from run.py key check**: `run.py` line 309 lists optional keys but does NOT include `ALPHA_VANTAGE_API_KEY`. Users won't see its status. It IS in `secrets_loader.py` though, so it's loaded -- just not displayed.

3. **Company search in run.py is too simple**: `run.py`'s `choose_company()` just takes a text input. It doesn't search the PIT API to show matches. The user types a name blind, and the actual search happens later in `main.py`. This means the user gets no feedback on whether their company exists. `main.py` has the proper `_select_company()` with search/browse.

4. **No validation that company exists before spawning pipeline**: `run.py` passes the raw text to `main.py --company`. If the company doesn't exist, `main.py` falls back to using it as a raw identifier, which may produce empty results.

---

## Action Items (Prioritized)

### Must Fix (Bugs)

- [ ] **BUG-1**: Fix ESEF constructor call in `main.py` (`country_filter` -> `country_code`), or better yet, delete `_create_pit_client()` from `main.py` and use `equity_provider.create_pit_client()`
- [ ] **BUG-2**: Fix `raw_text` parameter in ECB macro fetcher (either add support to `cached_get` or use `requests.get` directly)
- [ ] **BUG-3**: Make rate limiter per-host instead of global

### Should Fix (Data Completeness)

- [ ] EDINET: Expand company listing from 7-day scan to EDINET code list CSV download
- [ ] MOPS: Fix endpoints to use POST instead of GET (or switch to a working TWSE API endpoint)
- [ ] CMF: Fix company listing to use a working endpoint or web scraping
- [ ] ESEF: Increase filing scan limit from 500 to cover more companies
- [ ] DART: Add `corpCode.xml` download for full company listing

### Nice to Have (Rate Limits and Key Tiers)

- [ ] Implement per-host rate limiting in `http_utils.py`
- [ ] Add Alpha Vantage daily call counter (25/day free tier)
- [ ] Add key tier detection for Alpha Vantage, DART
- [ ] Add OpenFIGI rate limiting (25/min without key)
- [ ] Cache OpenFIGI lookups (currently not cached)
- [ ] Add `ALPHA_VANTAGE_API_KEY` to `run.py` key display

### Code Quality

- [ ] Remove duplicate `_create_pit_client()` from `main.py`
- [ ] Add company search preview to `run.py` before spawning pipeline
