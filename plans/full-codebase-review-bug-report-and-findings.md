# Full Codebase Review -- Bug Report and Findings

## Review Scope

Complete review of all source files in the Operator 1 pipeline:
- 10 PIT clients, 3 supplementary modules, macro/OHLCV providers
- Core infrastructure: config, secrets, HTTP, constants
- 25+ model/analysis/feature modules
- main.py orchestrator, run.py launcher
- All 16 test files

## Test Results Summary

**Current: 567 passed, 7 failed, 32 skipped**

The 7 new failures in `test_phase6_prediction_aggregator.py` are caused by `pyarrow` not being installed in the test environment. The `save_predictions()` function at `operator1/models/prediction_aggregator.py:572` calls `df.to_parquet()` which requires pyarrow. This is NOT a code bug -- it is an environment dependency issue.

The 32 skipped tests are caused by 6 missing ML libraries (torch, statsmodels, scikit-learn, arch, ruptures, hmmlearn). All 6 are listed in `requirements.txt` as real dependencies. The tests use `try/except ImportError: self.skipTest(...)` patterns to gracefully skip when these are not installed.

---

## Confirmed Bugs

### BUG 1: load_config called with .yml extension -- will look for .yml.yml [RUNTIME]

**Severity: Medium** -- causes FileNotFoundError or silent fallback to defaults

**File:** `main.py:904`
```python
_global_cfg = load_config("global_config.yml")  # BUG: double extension
```

**Root cause:** `load_config()` in `operator1/config_loader.py:43` appends `.yml` to the name parameter:
```python
path = _CONFIG_DIR / f"{name}.yml"
```

So `load_config("global_config.yml")` tries to open `config/global_config.yml.yml` which does not exist.

**Impact:** The estimation step in main.py catches this in a `try/except` block at line 928, so it silently falls back. The `estimation_imputer` config value is never loaded, so it always defaults to `"bayesian_ridge"` regardless of what the user configured.

**Fix:** Change to `load_config("global_config")`

---

### BUG 2: Same load_config .yml extension bug in economic_planes.py [RUNTIME]

**Severity: Medium** -- causes silent fallback to empty plane config

**File:** `operator1/analysis/economic_planes.py:33`
```python
_PLANES_CONFIG = load_config("economic_planes.yml")  # BUG: double extension
```

**Impact:** The `except Exception` block on line 34 catches the `FileNotFoundError` and falls back to `{"planes": {}, "default_plane": "manufacturing"}`. This means the entire 5-plane economic classification system NEVER reads the actual `config/economic_planes.yml` config file. Every company is classified using hardcoded defaults instead of the real plane definitions.

**Fix:** Change to `load_config("economic_planes")`

---

### BUG 3: MOPS POST requests bypass caching and rate limiting [DESIGN]

**Severity: Low** -- potential for rate-limit violations

**File:** `operator1/clients/mops.py:66-91`

The `_post()` method uses raw `requests.post()` directly, bypassing the centralized `cached_get()` system. This means:
1. MOPS POST responses are never cached to disk
2. MOPS requests are never rate-limited through the per-host rate limiter
3. MOPS requests are never logged to the request audit log

The same issue applies to ECB macro fetcher at `operator1/clients/macro_client.py:96-117` which uses raw `requests.get()` for CSV responses.

---

### BUG 4: CVM client uses raw requests.get bypassing cache/rate-limiter [DESIGN]

**Severity: Low** -- same pattern as BUG 3

**File:** `operator1/clients/cvm.py:173-174`

The CVM client's `_fetch_financials()` downloads CSV files using raw `requests.get()`, bypassing the `cached_get()` system. Large CSV downloads from CVM are not cached, causing unnecessary re-downloads on subsequent runs.

---

### BUG 5: Companies House iXBRL downloader bypasses cache/rate-limiter [DESIGN]

**Severity: Low** -- same pattern

**File:** `operator1/clients/companies_house.py:328`

The `_download_ixbrl_values()` method uses raw `requests.get()` to download HTML documents from the Companies House document API.

---

### BUG 6: OpenFIGI and Gemini POST calls bypass rate limiter [DESIGN]

**Severity: Low** -- OpenFIGI has 25/min limit

**Files:**
- `operator1/clients/supplement.py:75` -- OpenFIGI POST
- `operator1/clients/supplement.py:126` -- OpenFIGI search POST
- `operator1/clients/gemini.py:62,469` -- Gemini API POST

These use raw `requests.post()` without any rate limiting. OpenFIGI allows only 25 requests/minute without a key, and these could easily exceed that when enriching profiles for multiple companies.

---

### BUG 7: EDINET list_companies scans 90 days with 13 API calls per invocation [PERFORMANCE]

**Severity: Low-Medium** -- slow and wasteful

**File:** `operator1/clients/edinet.py:104`

`list_companies()` makes 13 separate API calls (one per week for 90 days) every time it is called. There is no caching of the resulting company list. This means each call to `list_companies()` or `search_company()` triggers 13 HTTP requests.

Even worse, `_fetch_filings()` at line 262 scans 365 days (52 API calls) for each financial statement type. A single company analysis would trigger 52 x 3 = 156 EDINET API calls just for financial data.

---

### BUG 8: Duplicate create_pit_client functions [CODE SMELL]

**Severity: Low** -- maintenance risk, not runtime

**Files:**
- `main.py:262` -- `_create_pit_client()`
- `operator1/clients/equity_provider.py:62` -- `create_pit_client()`

Both functions do the exact same thing. `main.py` uses its own copy, while `entity_discovery.py:358` imports from `equity_provider`. If one is updated but not the other, client instantiation could diverge.

---

### BUG 9: SEC EDGAR module-level global cache is never invalidated [DESIGN]

**Severity: Low** -- only affects long-running processes

**File:** `operator1/clients/sec_edgar.py:40-41`

```python
_company_list_cache: list | None = None
_companyfacts_cache: dict = {}
```

These module-level caches persist for the lifetime of the Python process. In a notebook or long-running service, stale SEC data would never be refreshed. Same issue in `operator1/clients/esef.py:31` with `_filing_list_cache`.

---

### BUG 10: NaN checks use x != x idiom instead of math.isnan [CODE SMELL]

**Severity: Informational** -- works correctly but unreadable

**File:** `main.py:871,1341`

```python
if not (macro_quadrant_result.growth_trend != macro_quadrant_result.growth_trend)
if not (hp.lower_ci != hp.lower_ci)
```

These rely on the IEEE 754 property that NaN != NaN. While technically correct, `math.isnan()` or `pd.isna()` would be clearer.

---

## Architecture Observations

### Strengths
1. **Well-structured PIT client abstraction**: The `PITClient` protocol with per-market implementations is clean and extensible
2. **Canonical translator is comprehensive**: 8 GAAP/IFRS standards mapped to canonical names
3. **Per-host rate limiting**: The `http_utils.py` rate limiter is well designed with per-API limits
4. **Defensive error handling**: Nearly every step in main.py is wrapped in try/except with fallbacks
5. **OHLCV routing**: Smart fallback from PIT source to Alpha Vantage with daily call tracking

### Areas for Improvement
1. **No cached_post function**: 6 modules bypass the centralized HTTP layer for POST requests. A `cached_post()` function would bring caching, rate limiting, and audit logging to these endpoints.
2. **EDINET scanning is extremely chatty**: The weekly-sampling approach for company/filing discovery generates hundreds of API calls. A bulk download approach would be far more efficient.
3. **Missing integration tests**: There are no tests that exercise the main pipeline end-to-end with mocked API responses.
4. **Config validation**: No schema validation for YAML configs. Misspelled keys silently fall back to defaults.
5. **Module-level caches lack TTL**: SEC, ESEF, and EDINET caches persist indefinitely in memory with no refresh mechanism.

---

## Priority Fix List

| Priority | Bug | Files | Impact |
|----------|-----|-------|--------|
| P1 | BUG 1: load_config .yml extension in main.py | main.py:904 | Estimation config never loaded |
| P1 | BUG 2: load_config .yml extension in economic_planes | economic_planes.py:33 | 5-plane system always uses defaults |
| P2 | BUG 8: Duplicate create_pit_client | main.py:262, equity_provider.py:62 | Maintenance risk |
| P2 | BUG 3-6: POST requests bypass http_utils | mops.py, cvm.py, companies_house.py, supplement.py, gemini.py, macro_client.py | No caching/rate-limiting for POSTs |
| P3 | BUG 7: EDINET chatty scanning | edinet.py:104,262 | Performance: 150+ API calls per company |
| P3 | BUG 9: Module-level caches | sec_edgar.py, esef.py | Stale data in long sessions |
| P3 | BUG 10: NaN idiom | main.py:871,1341 | Readability |
