# Research Log: US SEC EDGAR
**Generated**: 2026-02-24T14:35:00Z
**Status**: Complete

---

## 1. VERSION INFORMATION

| Component | Current (in requirements.txt) | Latest Available | Action Required |
|-----------|-------------------------------|------------------|-----------------|
| edgartools | >=5.15 (updated) | 5.17.1 | OK |
| sec-edgar-api | >=1.0 | 1.1.0 | OK |

**edgartools Published**: 2026-02-24T11:57:56 (from PyPI dist-tags)
**sec-edgar-api Published**: stable at 1.1.0

---

# =====================================================================
# PART A: UNOFFICIAL COMMUNITY WRAPPER -- edgartools
# =====================================================================

## A1. Registry Data

**Source**: https://pypi.org/pypi/edgartools/json (fetched 2026-02-24)
```
name: edgartools
version: 5.17.1
summary: Python library to access and analyze SEC Edgar filings, XBRL financial statements, 10-K, 10-Q, and 8-K reports
requires_python: >=3.10
homepage: https://github.com/dgunning/edgartools
documentation: https://dgunning.github.io/edgartools/
```

## A2. Verbatim Code -- Authentication

**Source**: https://github.com/dgunning/edgartools/blob/main/README.md (Quick Start section)
```python
# COPIED VERBATIM FROM README
from edgar import *
set_identity("your.name@example.com")
```

## A3. Verbatim Code -- Company Lookup

**Source**: https://github.com/dgunning/edgartools/blob/main/docs/complete-guide.md
```python
# COPIED VERBATIM
apple = Company("AAPL")        # By ticker
microsoft = Company("MSFT")    # Another ticker
berkshire = Company(1067983)    # By CIK number
```

**Company attributes** (verbatim from docs/complete-guide.md):
```python
company = Company("AAPL")
company.name                # 'APPLE INC'
company.cik                 # 320193
company.sic_code            # '3571'
company.sic_description     # 'Electronic Computers'
company.shares_outstanding  # 15115785000.0
```

**get_ticker()** (verbatim from docs/api/company.md):
```python
def get_ticker(self) -> Optional[str]
```

## A4. Verbatim Code -- Financial Statements

**Source**: https://github.com/dgunning/edgartools/blob/main/docs/api/company.md (Financial Data section)
```python
# COPIED VERBATIM -- get_financials() API
financials = company.get_financials()
if financials:
    balance_sheet = financials.balance_sheet
    income_statement = financials.income
    cash_flow = financials.cash_flow
    
    # Access specific metrics
    revenue = income_statement.loc['Revenue'].iloc[0]
    total_assets = balance_sheet.loc['Total Assets'].iloc[0]
```

**Source**: https://github.com/dgunning/edgartools/blob/main/docs/complete-guide.md (Financial Analysis section)
```python
# COPIED VERBATIM -- per-filing approach
tenk = Company("AAPL").get_filings(form="10-K")[0].obj()
income = tenk.financials.income_statement()
balance = tenk.financials.balance_sheet()
cashflow = tenk.financials.cash_flow_statement()
```

```python
# COPIED VERBATIM -- multi-year approach
company = Company("MSFT")
financials = company.get_financials()
income = financials.income_statement()
balance = financials.balance_sheet()
```

**Source**: https://github.com/dgunning/edgartools/blob/main/docs/getting-xbrl.md (XBRL statements)
```python
# COPIED VERBATIM -- XBRL statements API
balance_sheet = statements.balance_sheet()
income_statement = statements.income_statement()
cash_flow = statements.cashflow_statement()
```

**NOTE**: The CHANGELOG v5.16.3 also references convenience methods directly on Company:
```
# From CHANGELOG v5.16.3 (2026-02-21):
# "Company.income_statement() and Company.cashflow_statement() now correctly
#  forward the periods parameter through to the TTM statement builder."
```
These convenience methods exist but are NOT in the official API docs page.

## A5. Verbatim Code -- Filings

**Source**: docs/api/company.md
```python
# COPIED VERBATIM
all_filings = company.get_filings()
annual_reports = company.get_filings(form="10-K")
quarterly_reports = company.get_filings(form=["10-K", "10-Q"])
```

## A6. Breaking Changes

**Source**: https://github.com/dgunning/edgartools/blob/main/CHANGELOG.md

**v5.15.0 (2026-02-07)** -- VERBATIM from CHANGELOG:
```
### Changed
- **API Consistency** — Renamed `cash_flow()` to `cashflow_statement()` for consistency
  with other statement methods
```

**v5.16.0 (2026-02-14)** -- VERBATIM from CHANGELOG:
```
### Fixed
- **CompanyNotFoundError** — `Company("INVALID")` now raises `CompanyNotFoundError` with
  fuzzy-match suggestions instead of silently returning a placeholder entity with CIK
  -999999999
```

**v5.16.3 (2026-02-21)** -- VERBATIM from CHANGELOG:
```
### Fixed
- **TTM `max_periods` threading** — `Company.income_statement()` and
  `Company.cashflow_statement()` now correctly forward the `periods` parameter through
  to the TTM statement builder.
```

## A7. Fixes Applied to Our Wrapper

1. `company.cash_flow()` -> compat shim: tries `cashflow_statement()` first, falls back to `cash_flow()`
2. `company.not_found` check -> wrapped in try/except for `CompanyNotFoundError`
3. requirements.txt pinned to `edgartools>=5.15`

---

# =====================================================================
# PART A2: UNOFFICIAL COMMUNITY WRAPPER -- sec-edgar-api
# =====================================================================

## A2.1. Registry Data

**Source**: https://pypi.org/pypi/sec-edgar-api/json (fetched 2026-02-24)
```
name: sec-edgar-api
version: 1.1.0
summary: Unofficial SEC EDGAR API wrapper for Python
requires_python: >=3.8
```

## A2.2. Verbatim Code -- Authentication

**Source**: https://github.com/jadchaar/sec-edgar-api/blob/main/README.md
```python
# COPIED VERBATIM
from sec_edgar_api import EdgarClient
edgar = EdgarClient(user_agent="<Sample Company Name> <Admin Contact>@<Sample Company Domain>")
```

## A2.3. Verbatim Code -- Primary Methods

**Source**: https://github.com/jadchaar/sec-edgar-api/blob/main/README.md
```python
# COPIED VERBATIM
# Get submissions for Apple
edgar.get_submissions(cik="320193")
# Returns: {"cik": "320193", "name": "Apple Inc.", "tickers": ["AAPL"],
#           "exchanges": ["Nasdaq"], "sic": "3571", "sicDescription": "Electronic Computers", ...}

# Get company facts
edgar.get_company_facts(cik="320193")
# Returns: {"facts": {"us-gaap": {...}, "dei": {...}}}
```

## A2.4. Breaking Changes

None -- stable at v1.1.0. Auto rate-limiting at 10 req/s (from README).

---

# =====================================================================
# PART B: GOVERNMENT API -- SEC EDGAR REST
# =====================================================================

## B1. Documentation Source

**Source**: `SEC EDGAR Api doc` in repo root (fetched from sec.gov)
```
"data.sec.gov" was created to host RESTful data APIs delivering JSON-formatted
data to external customers and to web pages on SEC.gov. These APIs do not
require any authentication or API keys to access.
```

## B2. Verbatim Endpoint Specifications

**Source**: SEC EDGAR Api doc, lines 12-50

### Submissions endpoint:
```
https://data.sec.gov/submissions/CIK##########.json
Where the ########## is the entity's 10-digit central index key (CIK),
including leading zeros.
```

### XBRL companyfacts endpoint:
```
https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
This API returns all the company concepts data for a company into a
single API call.
```

### XBRL companyconcept endpoint:
```
https://data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/AccountsPayableCurrent.json
The company-concept API returns all the XBRL disclosures from a single
company (CIK) and concept (a taxonomy and tag) into a single JSON file.
```

### Company tickers list:
```
https://www.sec.gov/files/company_tickers.json
```

### SIC-based browse:
```
https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC=XXXX&owner=include&count=100&output=atom
```

## B3. Authentication

**Source**: SEC EDGAR Api doc
```
These APIs do not require any authentication or API keys to access.
```

SEC fair access policy requires User-Agent header with contact info.

## B4. Rate Limiting

**Source**: SEC EDGAR Api doc + sec.gov developer FAQ
- 10 requests per second per User-Agent
- sec-edgar-api library enforces this automatically

## B5. Breaking Changes

None -- SEC EDGAR REST API endpoints have been stable since 2020.

---

# =====================================================================
# PART C: AUDIT RESULTS SUMMARY
# =====================================================================

| Mode | Component | Status | Fixes Applied |
|------|-----------|--------|---------------|
| Unofficial (primary) | edgartools 5.17.1 | FIXED | cash_flow()->cashflow_statement(), CompanyNotFoundError |
| Unofficial (fallback) | sec-edgar-api 1.1.0 | OK | None needed |
| Government (direct) | data.sec.gov REST | OK | None needed |

### Canonical Field Coverage
All canonical fields covered in both modes:
- Income: revenue, cost_of_revenue, gross_profit, operating_income, net_income, ebit, ebitda, taxes, interest_expense, sga_expense, research_and_development
- Balance: total_assets, total_liabilities, total_equity, current_assets, current_liabilities, cash_and_equivalents, short_term_debt, long_term_debt, retained_earnings, goodwill, intangible_assets, receivables, inventory, payables
- Cashflow: operating_cash_flow, capex, investing_cf, financing_cf, dividends_paid, free_cash_flow, buybacks
- PIT columns: filing_date (SEC acceptance date), report_date (fiscal period end)

---

**END OF RESEARCH LOG**
