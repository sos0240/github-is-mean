# Research Log: UK Companies House
**Generated**: 2026-02-24T14:24:00Z
**Status**: Complete

---

## 1. VERSION INFORMATION

| Component | Current (in requirements.txt) | Latest Available | Action Required |
|-----------|-------------------------------|------------------|-----------------|
| (no library) | N/A -- direct HTTP | N/A | N/A |

No unofficial wrapper library used. Direct REST API calls only.

---

## 2. DOCUMENTATION SOURCES

### Primary Source -- Companies House REST API (Government API)
- **URL**: https://developer-specs.company-information.service.gov.uk/
- **Verified**: 2026-02-24 (via repo doc `company's house Api doc`)
- **Key required**: Yes (free registration)

---

## 3. VERBATIM CODE SNIPPETS

### 3a. Authentication Pattern
**Source**: company's house Api doc in repo
```
# Basic auth with API key (key as username, empty password)
Authorization: Basic base64encode("{api_key}:")
```

### 3b. Key Endpoints
```
GET /search/companies?q={query}&items_per_page=20
GET /company/{company_number}
GET /company/{company_number}/filing-history?category=accounts
GET /company/{company_number}/officers
GET /advanced-search/companies?sic_codes={sic}
```

---

## 4. BREAKING CHANGES ANALYSIS

### Government API -- No breaking changes detected
- Companies House REST API endpoints remain stable
- Authentication still Basic auth with API key

---

## 5. WRAPPER CODE AUDIT

### Mode 1: No unofficial wrapper -- this wrapper uses gov API directly

### Mode 2: Direct Gov API -- PRIMARY AND ONLY MODE

| Method | Status | Issue |
|--------|--------|-------|
| `list_companies()` | OK | `/search/companies` endpoint correct |
| `get_profile()` | OK | `/company/{id}` endpoint correct |
| `_fetch_financials_from_filings()` | **MAJOR GAP** | Only extracts filing METADATA, no actual financial numbers |
| `get_peers()` | OK | `/advanced-search/companies` correct |
| `get_executives()` | OK | `/company/{id}/officers` correct |

### CRITICAL ISSUE: Financial Data Extraction

The `_fetch_financials_from_filings()` method only returns filing dates, descriptions, and form types.
It does NOT extract any actual financial values (revenue, assets, etc.).

The DataFrame it returns has columns: `filing_date`, `report_date`, `form`, `description`,
`transaction_id`, `period_type` -- but NO financial line items.

This means the canonical translator receives empty financial data for UK companies.

**Root cause**: Companies House REST API stores financial data inside iXBRL documents
(attachments to filings). The wrapper would need to:
1. Download the iXBRL document via `/company/{id}/filing-history/{transaction_id}/document`
2. Parse the iXBRL/XHTML to extract financial values
3. Map to canonical fields

This is a known limitation noted in the wrapper docstring ("iXBRL document parsing for
financial data") but is not actually implemented.

### Canonical Field Coverage
- **Profile fields**: name, ticker, country, sector, industry, exchange, currency -- COVERED
- **Income statement**: NONE (all missing)
- **Balance sheet**: NONE (all missing)
- **Cash flow**: NONE (all missing)
- **PIT columns**: filing_date, report_date -- present but no financial data alongside

---

## 6. REQUIRED FIXES

1. **MAJOR**: The financial extraction returns only metadata, no numbers.
   This is an architectural limitation requiring iXBRL parsing.
   Recommended: Add a comment documenting this limitation explicitly, and ensure
   the fallback to the original `companies_house.py` direct client works.
2. **MINOR**: Verify the original `companies_house.py` direct client also has this issue.

---

## 7. IMPLEMENTATION READINESS

### Recommendation
- **HUMAN REVIEW NEEDED** -- Financial data extraction is fundamentally incomplete.
  UK companies will have profile data but zero financial statements.
  This requires iXBRL parsing implementation or an alternative data source.

---

**END OF RESEARCH LOG**
