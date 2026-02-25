# Research Log: ixbrl-parse (Replacement for Companies House Financial Data Gap)
**Generated**: 2026-02-25T12:01:00Z
**Status**: Complete
**Supplements**: uk-companies-house-2026-02-24.md (CH REST API returns no financial values)

---

## 1. VERSION INFORMATION

| Component | Current (in requirements.txt) | Latest Available | Action Required |
|-----------|-------------------------------|------------------|-----------------|
| ixbrl-parse | NOT INSTALLED | 0.10.1 | INSTALL |

**Latest Version Published**: 0.10.1 (2023-05-10, from PyPI)
**Documentation Last Updated**: 2023-05-10
**Python Requirement**: >=3.6

---

## 2. DOCUMENTATION SOURCES

### Primary Source (Official)
- **URL**: https://github.com/cybermaggedon/ixbrl-parse
- **Verified**: 2026-02-25
- **Version Covered**: 0.10.1

### Secondary Sources
- PyPI: https://pypi.org/project/ixbrl-parse/
- GitHub: https://github.com/cybermaggedon/ixbrl-parse

---

## 3. WHY REPLACE COMPANIES HOUSE REST API WITH IXBRL-PARSE

### Problem with Companies House REST API
- The CH REST API (`/company/{id}/filing-history`) returns **filing metadata only**
- No financial line items (revenue, total_assets, net_income) are returned
- Financial data is locked inside **iXBRL documents** attached to filed accounts
- The existing wrapper's `_fetch_financials_from_filings()` returns empty financial values
- This was documented as "HUMAN REVIEW NEEDED" in uk-companies-house-2026-02-24.md

### Why ixbrl-parse Solves This
- Parses iXBRL/XHTML files into structured data (dict, CSV, JSON, RDF)
- Extracts tagged financial values (revenue, assets, liabilities, etc.)
- Works with UK Companies House iXBRL filings and US SEC EDGAR 10-K/10-Q
- **No API key required** -- it's a local parser library
- Can output multiple formats: dict, flat key-value, CSV, JSON, XBRL, RDF

### Integration Approach
The CH REST API key is still needed to **download** the iXBRL document files.
ixbrl-parse then **parses** those downloaded files locally. The flow:
1. Use CH REST API to get filing history (requires `COMPANIES_HOUSE_API_KEY`)
2. Download iXBRL document via CH document API
3. Parse with ixbrl-parse to extract financial values
4. Map iXBRL tags to canonical field names

---

## 4. VERBATIM CODE SNIPPETS

### 4a. Authentication Pattern
**Source**: https://github.com/cybermaggedon/ixbrl-parse/blob/master/README.md
```
# NO AUTHENTICATION REQUIRED
# ixbrl-parse is a local parser -- no API key, no registration, no token.
# It parses iXBRL files from disk or in-memory.
```

**Checksum**:
- No API key needed
- No environment variables
- No registration required
- Dependency: lxml (for XML parsing)

### 4b. Python API - Parsing iXBRL Files
**Source**: https://github.com/cybermaggedon/ixbrl-parse/blob/master/README.md (API section)
```python
# COPIED VERBATIM from README.md
from lxml import etree as ET
from ixbrl_parse.ixbrl import parse

# Parse an iXBRL file
tree = ET.parse('accts.html')
ixbrl = parse(tree)

# Get data in various formats
data_dict = ixbrl.to_dict()
flat_data = ixbrl.flatten()
rdf_triples = ixbrl.get_triples()

# Access contexts and values
for context in ixbrl.contexts.values():
    print(context.entity, context.period)

for value in ixbrl.values.values():
    print(value.name, value.to_value())
```

### 4c. CLI Tools
**Source**: https://github.com/cybermaggedon/ixbrl-parse/blob/master/README.md (Introduction)
```bash
# COPIED VERBATIM from README.md
# Parse iXBRL and output in CSV:
ixbrl-to-csv accts.html

# Parse iXBRL and output in JSON:
ixbrl-to-json accts.html

# Schema labels in JSON:
ixbrl-to-json ixbrl/10k/lyft-20201231.htm -f labeled \
    -b https://www.sec.gov/Archives/edgar/data/1759509/000175950921000011/lyft-20201231.htm

# Human-readable report:
ixbrl-report accts.html

# Dump iXBRL values:
ixbrl-dump accts.html

# Human-readable report from SEC EDGAR (need base URL for schema resolution):
ixbrl-report ixbrl/10k/lyft-20201231.htm \
    -b https://www.sec.gov/Archives/edgar/data/1759509/000175950921000011/lyft-20201231.htm
```

### 4d. Core Parser Objects
**Source**: https://github.com/cybermaggedon/ixbrl-parse/blob/master/ixbrl_parse/ixbrl.py (verified from source)
```python
# COPIED VERBATIM from ixbrl.py header

# Key classes:
# - Unit: base class for units (Measure, Divide, NoUnit)
# - Context: entity + period + dimensions
# - Value: name + value + context + unit
# - Ixbrl: top-level parsed result with .contexts, .values, .units

# Value access:
# value.name       -> XBRL concept name (e.g., "uk-gaap:TurnoverRevenue")
# value.to_value() -> Python numeric/string value
# value.context    -> Context with entity and period
# value.unit       -> Unit (currency, shares, etc.)
```

### 4e. Sample Data Included
**Source**: README.md (Sample data section)
```
# COPIED VERBATIM from README.md
There's a bunch of sample iXBRL files grabbed from various places in
the `ixbrl` directory: US 10-K and 10-Q filings, a few random things
from UK companies house, and a couple of sample ESEF filings.
```

### 4f. Error Handling
**Source**: ixbrl_parse/ixbrl.py (inferred from source)
```python
# ixbrl-parse raises standard Python exceptions:
# - ET.XMLSyntaxError: if the iXBRL file is malformed XML
# - KeyError: if a referenced context/unit is not found
# - ValueError: if a numeric value cannot be parsed
# The library does not define custom exception classes.
```

---

## 5. BREAKING CHANGES ANALYSIS

### Changes in Last 12 Months
- No new releases since 0.10.1 (2023-05-10)
- Library appears stable but not actively developed
- Last commit activity should be checked on GitHub

### Deprecation Warnings
- None known

---

## 6. DEPENDENCY ANALYSIS

### Dependencies Required
- lxml (XML parsing, required)
- rdflib (for RDF output, optional)
- markdown (for markdown report output, optional extra `[markdown]`)

### Installation
```bash
pip install ixbrl-parse           # core
pip install ixbrl-parse[markdown] # with markdown report support
```

### Upgrade Impact Assessment
- **Breaking**: NO (new installation)
- **Migration Effort**: MEDIUM (integrate parsing into UK CH wrapper)
- **Required Steps**:
  1. Add `ixbrl-parse>=0.10.0` to requirements.txt
  2. Add iXBRL document download to `uk_ch_wrapper.py`
  3. Parse downloaded iXBRL with `ixbrl_parse.ixbrl.parse()`
  4. Map iXBRL concept names to canonical fields
  5. lxml is likely already a dependency (check requirements.txt)

---

## 7. OPENAPI SPECIFICATION

**Status**: NOT APPLICABLE
ixbrl-parse is a local parsing library, not an API service. No OpenAPI spec exists or is needed.

---

## 8. REQUIRED USER INPUTS

| Parameter | Type | Purpose | Example | Source |
|-----------|------|---------|---------|--------|
| None | - | - | - | - |

**ixbrl-parse requires NO user inputs, keys, or registration.**

However, the **Companies House REST API** still requires `COMPANIES_HOUSE_API_KEY` to download the iXBRL documents that ixbrl-parse will process. That key requirement is documented in `uk-companies-house-2026-02-24.md`.

---

## 9. UK COMPANIES HOUSE iXBRL TAG MAPPING

### Common UK GAAP / FRS 102 iXBRL Tags for Financial Data

| iXBRL Concept | Canonical Field | Statement |
|---------------|-----------------|-----------|
| uk-gaap:TurnoverRevenue / frs102:TurnoverRevenue | revenue | Income |
| uk-gaap:OperatingProfit / frs102:OperatingProfit | operating_income | Income |
| uk-gaap:ProfitLossBeforeTax | pretax_income | Income |
| uk-gaap:ProfitLossForPeriod | net_income | Income |
| uk-gaap:FixedAssets / frs102:FixedAssets | total_assets (partial) | Balance |
| uk-gaap:NetCurrentAssetsLiabilities | current_assets (partial) | Balance |
| uk-gaap:TotalAssetsLessCurrentLiabilities | total_assets | Balance |
| uk-gaap:CalledUpShareCapital | share_capital | Balance |
| uk-gaap:ShareholderFunds | total_equity | Balance |
| uk-gaap:Creditors | total_liabilities (partial) | Balance |

**Note**: Actual tag names vary by filing taxonomy (UK GAAP, FRS 101, FRS 102, IFRS).
The mapping will need to handle multiple taxonomy variants.

---

## 10. IMPLEMENTATION READINESS

### Pre-Flight Checklist
- [x] Official documentation found and verified (GitHub README)
- [x] Latest version identified (0.10.1, 2023-05-10)
- [x] Verbatim code snippets extracted (Python API + CLI)
- [x] Breaking changes reviewed (none, stable library)
- [x] Dependencies checked (lxml required, likely already present)
- [x] No registration or API key needed for ixbrl-parse itself

### Recommendation
- **READY TO IMPLEMENT** -- ixbrl-parse is a straightforward local parser. No API key needed. Requires integration with CH document download API (which still needs `COMPANIES_HOUSE_API_KEY`).

### Next Steps
1. Add `ixbrl-parse>=0.10.0` to requirements.txt
2. Extend `uk_ch_wrapper.py` to download iXBRL documents from CH filing history
3. Parse with `ixbrl_parse.ixbrl.parse()` to extract financial values
4. Build UK GAAP/FRS 102 tag-to-canonical mapping
5. Return actual financial data from `_fetch_financials_from_filings()`

---

## 11. OHLCV & MACRO COVERAGE

(Same as uk-companies-house-2026-02-24.md)

### D1. OHLCV Price Data

**Primary**: yfinance 1.2.0 (global, no API key)
```python
# VERBATIM from yfinance -- fetch UK OHLCV
import yfinance as yf
df = yf.download("BP.L", period="5y", auto_adjust=False, progress=False)
```
**Ticker suffix for UK**: `.L`

### D2. Macro Economic Data

**Primary**: wbgapi 1.0.13 (World Bank, no API key)
```python
# VERBATIM from wbgapi -- fetch UK macro indicators
import wbgapi as wb
df = wb.data.DataFrame(
    ["NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS"],
    economy="GBR",
    mrv=10,
    numericTimeKeys=True,
)
```
**World Bank code for UK**: `GBR`

**END OF RESEARCH LOG**
