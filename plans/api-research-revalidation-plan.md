# API Research Revalidation Plan

## Overview

Replace EDINET research with J-Quants API, replace Companies House with ixbrl-parse, and re-verify ALL other API sources from their actual endpoints to confirm key requirements and completeness.

## Current State from Source Code Analysis

### APIs Requiring Keys (from `pit_registry.py` + wrapper source)
| API | Key Env Var | Registration |
|-----|-------------|-------------|
| UK Companies House | `COMPANIES_HOUSE_API_KEY` | developer.company-information.service.gov.uk |
| JP EDINET | `EDINET_API_KEY` | Requires project on website - BLOCKED |
| KR DART | `DART_API_KEY` | opendart.fss.or.kr - free |
| FRED macro | requires key | free registration |
| INSEE macro | requires key | free registration |
| Japan e-Stat macro | requires key | free registration |
| Korea KOSIS macro | requires key | free registration |
| Chile BCC macro | requires key | free registration |

### APIs Marked No Key in Registry
AE-DFM, AU-ASX, BR-CVM, CA-SEDAR, CH-SIX, CL-CMF, CN-SSE, EU-ESEF, HK-HKEX, IN-BSE, MX-BMV, SA-Tadawul, SG-SGX, TW-MOPS, US-EDGAR, ZA-JSE

### Community Wrappers Used
| Wrapper | Package | Used By |
|---------|---------|---------|
| edinet-tools | edinet-tools 0.3.0 | JP EDINET - TO BE REPLACED |
| dart-fss | dart_fss | KR DART |
| pycvm | pycvm | BR CVM |
| pyesef | pyesef | EU ESEF |

---

## Phase 1: Replacements

### 1A. Replace EDINET with J-Quants API
- **Why**: EDINET registration requires project hosted on website, user cannot register
- **Replacement**: jquants-api-client 2.0.0 from JPX
- **Key required**: Yes - JQUANTS_API_KEY, but registration is email-only at jpx-jquants.com
- **Already researched**: README fetched, PyPI data fetched, financial summary columns documented
- **Research file created**: `.roo/research/jp-jquants-2026-02-25.md` - partially written

### 1B. Replace Companies House REST API with ixbrl-parse
- **Why**: CH REST API returns filing metadata only, no actual financial values
- **Replacement**: ixbrl-parse 0.10.1 - local parser, no API key
- **Still needs**: CH API key for downloading documents, but ixbrl-parse itself is key-free
- **Research needed**: Fetch README, verify parsing API, document integration approach

---

## Phase 2: Re-verify All Other APIs

For each API below, the research protocol requires:
1. Fetch from PyPI/npm if a package is used
2. Fetch official API documentation
3. Verify authentication requirements
4. Check for breaking changes
5. Document in research file

### Batch 1: Government APIs with Direct REST Access - no community wrapper
These use direct HTTP calls. Verify the actual endpoint responds and check auth headers.

| # | API | Base URL | Verify Auth |
|---|-----|----------|-------------|
| 1 | AE-DFM | https://www.dfm.ae | curl endpoint |
| 2 | AU-ASX | https://www.asx.com.au/asx/1 | curl endpoint |
| 3 | CA-SEDAR | https://www.sedarplus.ca/csa-party | curl endpoint |
| 4 | CH-SIX | https://www.six-group.com | curl endpoint |
| 5 | CN-SSE | http://www.sse.com.cn | curl endpoint |
| 6 | HK-HKEX | https://www.hkexnews.hk | curl endpoint |
| 7 | IN-BSE | https://api.bseindia.com | curl endpoint |
| 8 | MX-BMV | https://www.bmv.com.mx | curl endpoint |
| 9 | SA-Tadawul | https://www.saudiexchange.sa | curl endpoint |
| 10 | SG-SGX | https://www.sgx.com | curl endpoint |
| 11 | ZA-JSE | https://www.jse.co.za | curl endpoint |

### Batch 2: Government APIs with Community Wrappers
These use Python packages. Verify from PyPI + GitHub README.

| # | API | Package | PyPI URL |
|---|-----|---------|----------|
| 1 | BR-CVM | pycvm | https://pypi.org/pypi/pycvm/json |
| 2 | CL-CMF | direct REST | https://www.cmfchile.cl |
| 3 | EU-ESEF | pyesef | https://pypi.org/pypi/pyesef/json |
| 4 | KR-DART | dart-fss | https://pypi.org/pypi/dart-fss/json |
| 5 | TW-MOPS | direct scraping | https://mops.twse.com.tw |

### Batch 3: Known Key-Free Global Libraries
| # | Library | PyPI URL |
|---|---------|----------|
| 1 | US-SEC-EDGAR | sec-edgar-downloader or edgartools |
| 2 | yfinance | https://pypi.org/pypi/yfinance/json |
| 3 | wbgapi | https://pypi.org/pypi/wbgapi/json |
| 4 | OpenFIGI | https://api.openfigi.com |

---

## Phase 3: Compile Results

Create/update research files:
1. New: `.roo/research/jp-jquants-2026-02-25.md` - DONE partially
2. New: `.roo/research/uk-ixbrl-parse-2026-02-25.md`
3. Update/annotate all other research files with verification timestamp
4. Create summary: `.roo/research/api-key-requirements-summary-2026-02-25.md`

---

## Phase 4: Commit and PR

1. Commit all research files
2. Push to feature branch
3. Create draft PR

---

## Execution Strategy

Since the research protocol requires curl/web_fetch for each API, this work must be done in Code mode. The approach:

1. **Batch PyPI checks** - fetch JSON from PyPI for all Python packages in one go
2. **Batch endpoint probes** - curl each government API base URL to check for auth requirements
3. **GitHub README fetches** - get READMEs for packages with wrappers
4. **Write research files** - create/update based on findings
5. **Summary compilation** - one file listing all API key requirements
