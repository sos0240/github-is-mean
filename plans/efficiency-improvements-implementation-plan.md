# Efficiency Improvements Implementation Plan

## Scope

5 changes ordered by impact and safety. Each change is independent and testable.

---

## Change 1: EDINET Company List Caching + Reduced Scanning

**File:** `operator1/clients/edinet.py`

**Problem:** `list_companies()` makes 13 API calls (90 days / 7-day steps). `_fetch_filings()` makes 52 API calls (365 days / 7-day steps) PER statement type. Total: ~170 calls per company.

**Fix:**
1. Add a module-level cache for company list results (like SEC already does at line 40):
   ```python
   _company_list_cache: list[dict] | None = None
   ```
2. In `list_companies()`, check cache first. Only scan if cache is empty.
3. In `_fetch_filings()`, reduce scanning window from 365 to 90 days (financial reports are filed within 90 days of period end). Add early termination: stop scanning once we find 4+ filings for the target company.
4. Increase step from 7 to 14 days (biweekly sampling is sufficient since filings persist for days).

**Expected result:** 170 API calls -> ~20 API calls (88% reduction)

---

## Change 2: Parallel Financial Data Fetching in main.py

**File:** `main.py` (lines 592-619)

**Problem:** 4 API fetch calls run sequentially. For EDINET, each takes ~5-10 seconds, totaling 20-40 seconds.

**Fix:** Use `concurrent.futures.ThreadPoolExecutor` to run all 4 fetches in parallel:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(pit_client.get_income_statement, identifier): "income",
        executor.submit(pit_client.get_balance_sheet, identifier): "balance",
        executor.submit(pit_client.get_cashflow_statement, identifier): "cashflow",
        executor.submit(pit_client.get_quotes, identifier): "quotes",
    }
    for future in as_completed(futures):
        label = futures[future]
        try:
            result = future.result()
            if label == "income": income_df = result
            elif label == "balance": balance_df = result
            elif label == "cashflow": cashflow_df = result
            elif label == "quotes": quotes_df = result
            logger.info("%s: %d rows", label, len(result))
        except Exception as exc:
            logger.warning("%s fetch failed: %s", label, exc)
```

**Note:** Thread safety is fine because each call uses its own HTTP session via `cached_get()`, and the per-host rate limiter already uses thread-safe module-level dicts. Each call hits different endpoints.

**Expected result:** ~75% faster data fetching (wall clock time = max of the 4 calls, not sum)

---

## Change 3: Deduplicate create_pit_client

**Files:** `main.py` (lines 262-320), `operator1/clients/equity_provider.py` (lines 62-180)

**Problem:** Two identical functions. `main.py` uses its own, `entity_discovery.py` uses the one in `equity_provider.py`.

**Fix:** Delete `_create_pit_client()` from `main.py` and import from `equity_provider.py`:
```python
from operator1.clients.equity_provider import create_pit_client
```
Then replace `_create_pit_client(market_id, secrets)` with `create_pit_client(market_id, secrets)` at line 502.

---

## Change 4: Remove Redundant Client-Level Concept Maps

**Files:** `operator1/clients/esef.py`, `operator1/clients/edinet.py`, `operator1/clients/companies_house.py`

**Problem:** Each client has its own `_CONCEPT_MAP` dict that duplicates a subset of the canonical translator's maps. After our naming fix, both levels now agree, but maintaining two copies is fragile.

**Fix:** Remove `_CONCEPT_MAP` from each client and delegate concept extraction to the canonical translator:
- In `_fetch_statements()` / `_fetch_filings()` / `_fetch_accounts()`, extract raw facts without client-level mapping
- Pass raw facts through `translate_financials()` which handles all concept mapping centrally

**However**, this refactor changes how raw data is extracted within each client's internal methods. The client-level maps serve a different purpose -- they control WHICH facts to extract from the XBRL response, not just name translation. We should keep client-level maps but make them reference the translator's maps.

**Revised approach:** Replace client-level maps with imports from the translator:
```python
from operator1.clients.canonical_translator import get_concept_map
# In _fetch_statements():
concept_map = get_concept_map(self._market_id)
```

Wait -- the client-level maps are used during XBRL fact EXTRACTION (deciding which facts to pull from the filing response), while the translator maps are for concept NAME normalization. These are different operations. The client maps should stay but be generated FROM the translator maps.

**Final approach:** Leave client maps in place but verify they are subsets of the translator maps. Add a test to assert consistency. This is a code quality task, not an efficiency one. Defer to a later PR.

---

## Change 5: Add cached_post() to http_utils.py

**File:** `operator1/http_utils.py`

**Problem:** 6 modules make POST requests via raw `requests.post()`, bypassing disk caching, rate limiting, and audit logging.

**Fix:** Add a `cached_post()` function alongside `cached_get()`:
```python
def cached_post(
    url: str,
    json_data: dict | list | None = None,
    form_data: dict | None = None,
    headers: dict[str, str] | None = None,
    cache_dir: str | None = None,
    ttl_hours: float | None = None,
) -> Any:
```

Key differences from `cached_get()`:
- Cache key includes the POST body hash (not just URL + params)
- Supports both JSON and form-encoded payloads
- Same retry logic, rate limiting, and audit logging

Then update callers:
- `operator1/clients/mops.py:_post()` -- use `cached_post()` with form data
- `operator1/clients/supplement.py:openfigi_enrich()` -- use `cached_post()` with JSON
- `operator1/clients/supplement.py:openfigi_search()` -- use `cached_post()` with JSON

Do NOT update Gemini API calls -- those should not be cached (each prompt is unique).

---

## Execution Order

1. Change 1: EDINET scanning reduction (highest API savings)
2. Change 2: Parallel financial data fetching (highest wall-clock savings)
3. Change 3: Deduplicate create_pit_client (simplest, no risk)
4. Change 5: Add cached_post() (medium effort, good infrastructure)
5. Change 4: Defer client concept map deduplication (needs more design)

## Files Modified

| File | Changes |
|------|---------|
| `operator1/clients/edinet.py` | Add company cache, reduce scan windows, early termination |
| `main.py` | Parallel fetching, replace _create_pit_client with import |
| `operator1/http_utils.py` | Add cached_post() function |
| `operator1/clients/mops.py` | Use cached_post() for form submissions |
| `operator1/clients/supplement.py` | Use cached_post() for OpenFIGI calls |

## Risk Assessment

All changes are **backward compatible**:
- EDINET: same data output, fewer API calls
- Parallel fetch: same data, faster
- Dedup client creation: same behavior, single source of truth
- cached_post: same behavior + caching/rate-limiting
