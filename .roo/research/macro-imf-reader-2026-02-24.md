# Research Log: IMF Reader (Global Macro Fallback)
**Generated**: 2026-02-24T15:52:00Z
**Status**: Complete

---

# =====================================================================
# PART A: UNOFFICIAL COMMUNITY WRAPPER -- imf-reader
# =====================================================================

## A1. Registry Data

**Source**: https://pypi.org/pypi/imf-reader/json (fetched 2026-02-24)
```
name: imf-reader
version: 1.4.1
summary: A package to access imf data
```

## A2. Assessment
- Wraps IMF (International Monetary Fund) data API
- IMF covers macro indicators for ALL countries (GDP, inflation, rates, etc.)
- Could serve as a global macro fallback alongside wbgapi
- Version 1.4.1 -- reasonably mature
- Could fill macro gaps for regions without dedicated sources (CA, AU, HK, ZA, CH, SA, AE)

---

# =====================================================================
# PART B: IMF DATA API (Government/International Organization)
# =====================================================================

## B1. IMF Data API
**URL**: https://www.imf.org/external/datamapper/api/v1/
**Key required**: No
**Coverage**: 190+ countries, GDP, CPI, exchange rates, fiscal data

---

**END OF RESEARCH LOG**
