# Research Log: wbgapi (Global Macro Provider)
**Generated**: 2026-02-24T21:41:00Z
**Status**: Complete

---

## 1. VERSION INFORMATION

| Component | Current (in requirements.txt) | Latest Available | Action Required |
|-----------|-------------------------------|------------------|-----------------|
| wbgapi | commented out (was 1.0) | 1.0.13 | INSTALL |

**Latest Version Published**: 2026-02-21 (pushed to GitHub)
**Documentation**: GitHub README + extensive docstrings

---

## 2. DOCUMENTATION SOURCES

### Primary Source (Official)
- **URL**: https://github.com/tgherzog/wbgapi
- **Verified**: 2026-02-24
- **Version Covered**: 1.0.13

### Secondary Sources
- PyPI: https://pypi.org/project/wbgapi/
- GitHub: https://github.com/tgherzog/wbgapi (172 stars, MIT license)
- Last push: 2026-02-21T20:38:27Z (actively maintained)

---

## 3. VERBATIM CODE SNIPPETS

### 3a. Authentication Pattern
**No authentication required** -- World Bank API is free and open, no API key needed.

### 3b. Primary Usage: Fetch Macro Indicators
**Source**: GitHub README
```python
# COPIED VERBATIM from README
import wbgapi as wb

# Fetch GDP for a country, most recent 5 years
wb.data.DataFrame('NY.GDP.PCAP.CD', 'USA', mrv=5)

# Fetch multiple indicators for multiple countries
wb.data.DataFrame(
    ['NY.GDP.PCAP.CD', 'SP.POP.TOTL'],
    'CAN',
    mrv=5,
)

# Fetch for specific years
wb.data.DataFrame('SP.POP.TOTL', wb.region.members('AFR'), range(2010, 2020, 2))

# Fetch as row iterator (dict objects)
for row in wb.data.fetch('SP.POP.TOTL', 'USA'):
    print(row)
```

### 3c. Key Indicator IDs for Survival Mode

| Indicator | Series ID | Description |
|-----------|-----------|-------------|
| GDP (current USD) | NY.GDP.MKTP.CD | GDP at current prices |
| GDP per capita | NY.GDP.PCAP.CD | GDP per capita, current USD |
| GDP growth | NY.GDP.MKTP.KD.ZG | GDP growth (annual %) |
| Inflation (CPI) | FP.CPI.TOTL.ZG | Consumer price inflation (annual %) |
| Interest rate (real) | FR.INR.RINR | Real interest rate (%) |
| Unemployment | SL.UEM.TOTL.ZS | Unemployment (% of total labor force) |
| Exchange rate | PA.NUS.FCRF | Official exchange rate (LCU per USD) |
| Current account | BN.CAB.XOKA.GD.ZS | Current account balance (% of GDP) |
| Government debt | GC.DOD.TOTL.GD.ZS | Central gov debt (% of GDP) |
| Population | SP.POP.TOTL | Total population |

### 3d. Country Codes (ISO-2 to World Bank ISO-3)

| Country | ISO-2 | WB Code (ISO-3) |
|---------|-------|-----------------|
| United States | US | USA |
| United Kingdom | GB | GBR |
| European Union | EU | EMU (Euro area) |
| France | FR | FRA |
| Germany | DE | DEU |
| Japan | JP | JPN |
| South Korea | KR | KOR |
| Taiwan | TW | (not in WB -- use CHN or skip) |
| Brazil | BR | BRA |
| Chile | CL | CHL |
| Canada | CA | CAN |
| Australia | AU | AUS |
| India | IN | IND |
| China | CN | CHN |
| Hong Kong | HK | HKG |
| Singapore | SG | SGP |
| Mexico | MX | MEX |
| South Africa | ZA | ZAF |
| Switzerland | CH | CHE |
| Saudi Arabia | SA | SAU |
| UAE | AE | ARE |
| Netherlands | NL | NLD |
| Spain | ES | ESP |
| Italy | IT | ITA |
| Sweden | SE | SWE |

### 3e. DataFrame Response Structure
```python
# wb.data.DataFrame returns a pandas DataFrame
# Index: economy (country code)
# Columns: time periods (YR2020, YR2021, etc.)
# Values: indicator values (float)

# Example:
df = wb.data.DataFrame('NY.GDP.MKTP.CD', ['USA', 'GBR', 'JPN'], mrv=5)
#           YR2021          YR2022          YR2023          YR2024          YR2025
# USA  23315080560000  25462700000000  27360935000000  ...
# GBR  3131225973700   3070668348000   3332059267350   ...
# JPN  4940877780000   4231141199520   4212945260000   ...
```

### 3f. Error Handling
```python
# wbgapi raises exceptions for invalid indicators or countries
try:
    df = wb.data.DataFrame('INVALID_ID', 'USA', mrv=5)
except Exception as e:
    # Typically raises requests-related errors or empty results
    print(f"Error: {e}")
```

---

## 4. COVERAGE ANALYSIS

### Countries covered: 200+
World Bank covers virtually all countries except Taiwan (not a WB member).

### Taiwan workaround
Taiwan is not in the World Bank database. Options:
1. Use IMF data via pandas-datareader
2. Use DGBAS (Taiwan statistics) as a fallback
3. Skip macro for Taiwan

### Data freshness
World Bank data has a ~6-12 month lag (annual data published next year).
For survival mode analysis, this is acceptable (macro trends are slow-moving).

---

## 5. DEPENDENCY ANALYSIS

### Dependencies (3 packages -- very lightweight)
- requests (already in requirements.txt)
- PyYAML (already in requirements.txt as pyyaml)
- tabulate (new, lightweight)

### Upgrade Impact Assessment
- **Breaking**: NO (new install)
- **Migration Effort**: LOW
- **No heavy dependencies** -- very clean

---

## 6. IMPLEMENTATION READINESS

### Pre-Flight Checklist
- [x] Official documentation found and verified
- [x] Latest version identified (1.0.13)
- [x] Verbatim code snippets extracted
- [x] Indicator IDs mapped for all needed macro variables
- [x] Country code mapping documented
- [x] Dependencies checked (only tabulate is new)
- [x] Coverage gap identified (Taiwan)

### Recommendation
- **READY TO IMPLEMENT** -- All checks passed

### Implementation Notes
1. Use `wb.data.DataFrame(indicator, country, mrv=10)` for recent data
2. Map ISO-2 country codes to ISO-3 for WB API
3. Fetch 5 core indicators: GDP growth, inflation, interest rate, unemployment, exchange rate
4. Handle Taiwan separately (WB does not cover Taiwan)
5. Data is annual -- forward-fill to daily for cache alignment
6. No API key needed

---

**END OF RESEARCH LOG**
