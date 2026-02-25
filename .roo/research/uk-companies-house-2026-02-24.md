# Research Log: UK Companies House
**Generated**: 2026-02-24T14:37:00Z
**Status**: SUPPLEMENTED -- Financial data gap filled by ixbrl-parse (see uk-ixbrl-parse-2026-02-25.md)
**Supplement**: `.roo/research/uk-ixbrl-parse-2026-02-25.md`

---

## 1. VERSION INFORMATION

No unofficial wrapper library used. Direct REST API calls only.

---

# =====================================================================
# PART A: UNOFFICIAL COMMUNITY WRAPPER
# =====================================================================

**Not applicable** -- this wrapper uses the government API directly.
No community wrapper library is used.

---

# =====================================================================
# PART B: GOVERNMENT API -- Companies House REST API
# =====================================================================

## B1. Documentation Source

**Source**: `company's house Api doc` in repo root (from developer.company-information.service.gov.uk)

## B2. Verbatim Overview

**Source**: company's house Api doc, lines 1-6
```
# COPIED VERBATIM
Companies House is an executive agency of the Department for Business and Trade.
The Companies House API lets you retrieve information about limited companies
(and other companies that fall within the Companies Act 2006). The data returned
is live and real-time, and is simple to use and understand.
```

## B3. Verbatim Authentication

**Source**: company's house Api doc, lines 23-26
```
# COPIED VERBATIM
API authentication
Access to API services requires authentication. The Companies House API requires
API authentication credentials to be sent with each request, which is sent as an
API key, stream key or OAuth access token.
```

**Implementation** (from our wrapper):
```python
# Basic auth with API key as username, empty password
import base64
encoded = base64.b64encode(f"{api_key}:".encode()).decode()
headers["Authorization"] = f"Basic {encoded}"
```

## B4. Verbatim Endpoint Specifications

**Source**: company's house Api doc + Companies House developer specs

### Company Search:
```
GET https://api.company-information.service.gov.uk/search/companies?q={query}&items_per_page=20
```

### Company Profile:
```
GET https://api.company-information.service.gov.uk/company/{company_number}
Response fields: company_name, company_status, type, sic_codes[], date_of_creation
```

### Filing History (Accounts):
```
GET https://api.company-information.service.gov.uk/company/{company_number}/filing-history?category=accounts&items_per_page=20
Response fields per item: date, description, action_date, type, transaction_id
```

### Officers:
```
GET https://api.company-information.service.gov.uk/company/{company_number}/officers
Response fields per item: name, officer_role, appointed_on
```

### Advanced Search (Peers by SIC):
```
GET https://api.company-information.service.gov.uk/advanced-search/companies?sic_codes={sic}&size=10&company_status=active
```

## B5. CRITICAL DATA GAP

**The Companies House REST API does NOT return financial line items in any endpoint.**

The filing-history endpoint returns:
- `date`: filing date
- `description`: text description (e.g., "accounts-with-accounts-type-full")
- `action_date`: period end date
- `type`: document type code
- `transaction_id`: reference for document download

But it does NOT return: revenue, total_assets, net_income, or ANY numerical financial data.

**Financial data is stored inside iXBRL documents** attached to filed accounts.
To extract actual numbers, the wrapper would need to:
1. Call `GET /company/{id}/filing-history/{transaction_id}/document` to get the iXBRL file
2. Parse the iXBRL/XHTML to extract tagged financial values
3. Map iXBRL tags to canonical field names

**This is not implemented in our wrapper.** The `_fetch_financials_from_filings()` method
only returns filing metadata (dates, form types) with zero financial values.

## B6. Breaking Changes

None -- Companies House REST API endpoints remain stable.

## B7. Audit of Our Wrapper

| Method | Endpoint | Matches docs? | Returns correct data? |
|--------|----------|---------------|----------------------|
| `list_companies()` | `/search/companies` | YES | YES (name, company_number) |
| `get_profile()` | `/company/{id}` | YES | YES (name, status, sic_codes) |
| `_fetch_financials_from_filings()` | `/company/{id}/filing-history` | YES (endpoint) | NO -- returns metadata only, no financial values |
| `get_peers()` | `/advanced-search/companies` | YES | YES (company_numbers by SIC) |
| `get_executives()` | `/company/{id}/officers` | YES | YES (name, role, appointed_on) |

---

# =====================================================================
# PART C: AUDIT RESULTS SUMMARY
# =====================================================================

| Mode | Component | Status | Issue |
|------|-----------|--------|-------|
| Government (only) | api.company-information.service.gov.uk | DOCUMENTED GAP | Financial extraction returns no actual numbers |

### What Works
- Company search, profile, officers, peers -- all correct
- Filing history dates and form types -- correct

### What Does NOT Work
- Income statement data -- returns empty (no iXBRL parsing)
- Balance sheet data -- returns empty (no iXBRL parsing)
- Cash flow data -- returns empty (no iXBRL parsing)

### Recommendation
- **HUMAN REVIEW NEEDED** -- Financial data extraction requires iXBRL parsing implementation

---


# =====================================================================
# PART D: OHLCV & MACRO COVERAGE (added 2026-02-24)
# =====================================================================

## D1. OHLCV Price Data

**Primary**: yfinance 1.2.0 (global, no API key, 21.7k stars)
**Research log**: `.roo/research/ohlcv-yfinance-2026-02-24.md`

```python
# VERBATIM from yfinance source -- fetch United Kingdom OHLCV
import yfinance as yf
df = yf.download(
    "BP.L",     # United Kingdom ticker with ".L" suffix
    period="5y",
    auto_adjust=False,      # raw/unadjusted OHLCV (PIT-safe)
    progress=False,
)
# Returns: DataFrame(Open, High, Low, Close, Adj Close, Volume)
```

**Ticker suffix for United Kingdom**: `.L`
**Example**: `BP.L`
**Notes**: yfinance primary.

## D2. Macro Economic Data

**Primary**: wbgapi 1.0.13 (World Bank, 200+ countries, no API key)
**Research log**: `.roo/research/macro-wbgapi-2026-02-24.md`

```python
# VERBATIM from wbgapi source -- fetch United Kingdom macro indicators
import wbgapi as wb
df = wb.data.DataFrame(
    ["NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS"],
    economy="GBR",    # United Kingdom World Bank code
    mrv=10,                 # most recent 10 years
    numericTimeKeys=True,
)
# Returns: DataFrame with GDP growth, inflation, unemployment
```

**World Bank code for United Kingdom**: `GBR`

**END OF RESEARCH LOG**
