# Research Log: US SEC EDGAR
**Generated**: 2026-02-24T14:20:00Z
**Status**: Complete

---

## 1. VERSION INFORMATION

| Component | Current (in requirements.txt) | Latest Available | Action Required |
|-----------|-------------------------------|------------------|-----------------|
| edgartools | >=4.0 | 5.17.1 | UPGRADE -- pin to >=5.15 |
| sec-edgar-api | >=1.0 | 1.1.0 | OK (minor) |

**edgartools Last Published**: 2026-02-21 (v5.16.3)
**sec-edgar-api Last Published**: stable at 1.1.0

---

## 2. DOCUMENTATION SOURCES

### Primary Source -- edgartools (Unofficial Wrapper)
- **URL**: https://github.com/dgunning/edgartools
- **Verified**: 2026-02-24
- **Version Covered**: 5.17.1

### Secondary Source -- SEC EDGAR (Government API)
- **URL**: https://www.sec.gov/edgar/sec-api-documentation
- **Verified**: 2026-02-24 (via repo doc `SEC EDGAR Api doc`)
- **Endpoints**: data.sec.gov (XBRL), efts.sec.gov (search)

### Tertiary Source -- sec-edgar-api (Unofficial Wrapper)
- **URL**: https://github.com/jadchaar/sec-edgar-api
- **Verified**: 2026-02-24
- **Version**: 1.1.0

---

## 3. VERBATIM CODE SNIPPETS

### 3a. Authentication Pattern (edgartools)
**Source**: edgartools README, Quick Start section
```python
# COPIED VERBATIM
from edgar import *
set_identity("your.name@example.com")
```
- No API key needed
- SEC requires User-Agent with contact info

### 3b. Primary Request Schema (edgartools)
**Source**: edgartools README + CHANGELOG v5.15.0-5.16.3
```python
# COPIED VERBATIM -- current pattern (v5.15+)
company = Company("MSFT")
financials = company.get_financials()
financials.balance_sheet()
financials.income_statement()

# Also available directly on Company (v5.15+):
company.income_statement(periods=8, annual=True, as_dataframe=True)
company.balance_sheet(periods=8, annual=True, as_dataframe=True)
company.cashflow_statement(periods=8, annual=True, as_dataframe=True)  # RENAMED from cash_flow()

# Company search
company = Company("AAPL")
filings = company.get_filings(form="10-K")
```

### 3c. Authentication Pattern (sec-edgar-api fallback)
**Source**: sec-edgar-api README
```python
# COPIED VERBATIM
from sec_edgar_api import EdgarClient
edgar = EdgarClient(user_agent="<Sample Company Name> <Admin Contact>@<Sample Company Domain>")
edgar.get_submissions(cik="320193")
edgar.get_company_facts(cik="320193")
```

### 3d. Government API Endpoints (direct fallback)
**Source**: SEC EDGAR Api doc in repo + sec.gov docs
```
# All free, no key, User-Agent required
https://data.sec.gov/submissions/CIK{cik10}.json          -- submissions history
https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json -- all XBRL facts
https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json
https://www.sec.gov/files/company_tickers.json              -- full ticker list
https://www.sec.gov/cgi-bin/browse-edgar                    -- SIC-based search
```

---

## 4. BREAKING CHANGES ANALYSIS

### edgartools Changes in Last 12 Months (CRITICAL)

- **v5.15.0 (2026-02-07)**: `cash_flow()` RENAMED to `cashflow_statement()`
  - Our wrapper calls `company.cash_flow()` -- WILL BREAK on v5.15+
  - Migration: replace `cash_flow` with `cashflow_statement`

- **v5.16.0 (2026-02-14)**: `Company("INVALID")` now raises `CompanyNotFoundError`
  - Previously returned placeholder with `company.not_found = True`
  - Our wrapper checks `company.not_found` -- WILL BREAK on v5.16+
  - Migration: catch `CompanyNotFoundError` instead

- **v5.16.1 (2026-02-18)**: Fixed standard_concept propagation in stitched statements
  - May affect DataFrame column names returned by financial statements

- **v5.16.3 (2026-02-21)**: TTM periods parameter now correctly forwarded

### sec-edgar-api -- No breaking changes
- Stable at v1.1.0, API unchanged

### Government API -- No breaking changes
- SEC EDGAR REST API endpoints remain unchanged
- Rate limit: 10 requests/second (enforced by sec-edgar-api automatically)

---

## 5. WRAPPER CODE AUDIT

### Mode 1: edgartools (Unofficial Wrapper) -- PRIMARY

| Method | Status | Issue |
|--------|--------|-------|
| `_init_edgartools()` | OK | `edgar.set_identity()` is correct |
| `list_companies()` | VERIFY | `edgar.get_company_tickers()` -- need to confirm still exists |
| `search_company()` | OK | Uses direct SEC tickers JSON (gov API fallback) |
| `get_profile()` | BROKEN | `company.not_found` check will fail on v5.16+ |
| `get_profile()` | VERIFY | `company.get_ticker()`, `company.get_exchanges()` -- verify exist |
| `_fetch_statement_edgartools()` | BROKEN | `company.cash_flow()` renamed to `cashflow_statement()` in v5.15+ |
| `_build_filing_date_map()` | OK | `company.get_filings(form=...)` still valid |
| `_map_edgartools_concept()` | OK | Label mapping is comprehensive |

### Mode 2: sec-edgar-api + Direct Gov API -- FALLBACK

| Method | Status | Issue |
|--------|--------|-------|
| `_get_sec_api_client()` | OK | `EdgarClient(user_agent=...)` correct |
| `get_profile()` fallback | OK | `client.get_submissions(cik)` correct |
| `_fetch_statement_fallback()` | OK | `client.get_company_facts(cik)` correct |
| US-GAAP concept mapping | OK | `_USGAAP_INCOME_CONCEPTS` etc. are comprehensive |
| Rate limiting | OK | sec-edgar-api auto-limits to 10 req/s |

### Canonical Field Coverage

**Income**: revenue, cost_of_revenue, gross_profit, operating_income, net_income, ebit, ebitda, taxes, interest_expense, sga_expense, research_and_development -- ALL COVERED

**Balance**: total_assets, total_liabilities, total_equity, current_assets, current_liabilities, cash_and_equivalents, short_term_debt, long_term_debt, total_debt, retained_earnings, goodwill, intangible_assets, receivables, inventory, payables -- ALL COVERED

**Cashflow**: operating_cash_flow, capex, investing_cf, financing_cf, dividends_paid, free_cash_flow, buybacks -- ALL COVERED

**PIT columns**: filing_date, report_date -- BOTH PRESENT

---

## 6. REQUIRED FIXES

1. **CRITICAL**: Replace `company.cash_flow()` with `company.cashflow_statement()` (edgartools v5.15+ rename)
2. **CRITICAL**: Replace `company.not_found` check with `try/except CompanyNotFoundError` (edgartools v5.16+)
3. **MINOR**: Update requirements.txt to pin `edgartools>=5.15` to avoid using old broken API
4. **MINOR**: Verify `company.get_ticker()` and `company.get_exchanges()` attribute names

---

## 7. IMPLEMENTATION READINESS

### Pre-Flight Checklist
- [x] Official documentation found and verified
- [x] Latest version identified (edgartools 5.17.1, sec-edgar-api 1.1.0)
- [x] Verbatim code snippets extracted
- [x] Breaking changes reviewed (2 CRITICAL found)
- [x] Dependencies checked
- [x] Migration path understood

### Recommendation
- **UPGRADE REQUIRED** -- Must fix `cash_flow()` -> `cashflow_statement()` rename and `not_found` -> exception handling

---

**END OF RESEARCH LOG**
