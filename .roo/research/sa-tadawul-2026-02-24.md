# Research Log: Saudi Arabia Tadawul
**Generated**: 2026-02-24T15:46:00Z
**Status**: Complete

---

# =====================================================================
# PART A: UNOFFICIAL COMMUNITY WRAPPER
# =====================================================================

## A1. PyPI Search

**Source**: PyPI simple index search (fetched 2026-02-24)
Searched: tadawul, saudi-stock, tadawul-sdk, saudi-exchange
```
Results: NONE FOUND on PyPI
```

## A2. GitHub Search

**Source**: GitHub API search (fetched 2026-02-24)
```
Results:
- Logic-gate/TadawulStocks (8 stars) -- "A minimalistic Python based portfolio manager.
  This version only supports (Tadaw"
- zaakki-ahamed/Saudi_Stock_Web_Scrape (3 stars) -- web scraper
- Others: <2 stars
```
No proper API wrapper library. Only basic scrapers and portfolio tools.

---

# =====================================================================
# PART B: GOVERNMENT API -- Tadawul
# =====================================================================

## B1. API Used by Stub Client

**Source**: pit_registry.py
```
pit_api_url: https://www.saudiexchange.sa
```
No key required. Tadawul provides limited free data.

## B2. Macro Source
**Current**: World Bank fallback only (wbgapi)
**Potential**: SAMA (Saudi Arabian Monetary Authority) has free data but no PyPI wrapper found

---

# =====================================================================
# PART C: SUMMARY
# =====================================================================

| Component | Status |
|-----------|--------|
| Micro wrapper | NONE FOUND on PyPI -- only basic GitHub scrapers |
| Macro wrapper | NONE FOUND -- World Bank fallback |

---

**END OF RESEARCH LOG**
